"""Offline paired source/platform analysis for FVM source-to-repair boundaries."""

from __future__ import annotations

import bisect
import csv
import json
import math
import statistics
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2

from fov import (INTERLEAVE_WINDOW, MetaPacket, PacketError, SymbolPacket, parse_packet,
                 validate_metadata)
from fvm_diagnostics import FailureTracker, PacketDiagnostic
from fvm_file_common import (FPS, HEIGHT, PHYSICAL_CONFIG, WIDTH, decode_transport_physical,
                             matrix_to_physical)
from scripts.fvm0_common import decode_frame
from video2file import _validate_meta


@dataclass
class CodecFrame:
    key_frame: bool | None = None
    timestamp_time: float | None = None
    pkt_size: int | None = None
    pict_type: str | None = None
    coded_picture_number: int | None = None


@dataclass
class SourceFrame:
    observed_frame_index: int
    transport_index: int
    packet_type: str
    block_id: int | None = None
    symbol_id: int | None = None
    source_symbols: int | None = None
    encoded_symbols: int | None = None
    classification: str | None = None
    boundary_offset: int | None = None
    is_first_source: bool = False
    is_last_source: bool = False
    is_first_repair: bool = False
    is_last_repair: bool = False
    frames_since_previous_meta: int | None = None
    frames_until_next_meta: int | None = None


@dataclass
class PlatformFrame:
    observed_frame_index: int
    transport_index: int | None
    inference_confident: bool
    rs_failed: bool
    codeword_successes: int
    codeword_failures: int
    failed_codeword_indices: tuple[int, ...]
    corrected_symbols: int
    max_corrections_per_codeword: int
    corrections_per_codeword: tuple[int | None, ...]
    transport_failed: bool = False
    packet_failed: bool = False
    packet_descriptor: PacketDiagnostic | None = None


@dataclass
class PairedFrameRecord:
    transport_index: int | None
    source_observed_frame_index: int | None
    platform_observed_frame_index: int
    packet_type: str | None
    block_id: int | None
    symbol_id: int | None
    source_symbols: int | None
    encoded_symbols: int | None
    classification: str
    boundary_offset: int | None
    is_last_source: bool
    is_first_repair: bool
    platform_rs_failed: bool
    codeword_failures: int
    failed_codeword_indices: tuple[int, ...]
    corrected_symbols: int
    max_corrections_per_codeword: int
    corrections_per_codeword: tuple[int | None, ...]
    transport_failed: bool
    packet_failed: bool
    mapping_confident: bool
    observed_alignment_match: bool | None
    frames_since_previous_meta: int | None
    frames_until_next_meta: int | None


def classify_symbol(symbol_id: int, source_symbols: int, encoded_symbols: int) -> dict[str, Any]:
    if not 0 <= symbol_id < encoded_symbols:
        raise ValueError("symbol_id is outside the block layout")
    if not 1 <= source_symbols <= encoded_symbols:
        raise ValueError("invalid source/encoded symbol counts")
    offset = symbol_id - (source_symbols - 1)
    return {
        "classification": "SOURCE" if symbol_id < source_symbols else "REPAIR",
        "boundary_offset": offset,
        "is_first_source": symbol_id == 0,
        "is_last_source": symbol_id == source_symbols - 1,
        "is_first_repair": symbol_id == source_symbols,
        "is_last_repair": symbol_id == encoded_symbols - 1,
    }


def parse_ffprobe_frames(payload: dict[str, Any]) -> list[CodecFrame]:
    frames = []
    for raw in payload.get("frames", []):
        frames.append(CodecFrame(
            key_frame=_optional_bool(raw.get("key_frame")),
            timestamp_time=_optional_float(raw.get("best_effort_timestamp_time")),
            pkt_size=_optional_int(raw.get("pkt_size")),
            pict_type=raw.get("pict_type") if isinstance(raw.get("pict_type"), str) else None,
            coded_picture_number=_optional_int(raw.get("coded_picture_number")),
        ))
    return frames


def probe_codec_frames(video: Path) -> tuple[list[CodecFrame], str | None]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "frame=key_frame,best_effort_timestamp_time,pkt_size,pict_type,coded_picture_number",
        "-of", "json", str(video),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return parse_ffprobe_frames(json.loads(completed.stdout)), None
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return [], f"ffprobe unavailable or failed: {exc}"


def decode_source_video(video: Path) -> tuple[list[SourceFrame], dict[int, SourceFrame], dict[str, Any]]:
    capture, width, height, fps, reported_frames = _open_video(video)
    records: list[SourceFrame] = []
    by_transport: dict[int, SourceFrame] = {}
    layouts = None
    metadata = None
    meta_indices: list[int] = []
    previous_transport: int | None = None
    alignment_matches = 0
    try:
        observed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded = _decode_frame(frame)
            if decoded.rs.logical is None:
                raise RuntimeError(f"source RS failure at observed frame {observed}")
            if decoded.transport is None:
                raise RuntimeError(
                    f"source transport failure at observed frame {observed}: {decoded.transport_error}"
                )
            transport_index = decoded.transport.frame_index
            if transport_index in by_transport:
                raise RuntimeError(f"duplicate source transport index {transport_index}")
            if previous_transport is not None and transport_index <= previous_transport:
                raise RuntimeError("source transport indices are not strictly increasing")
            try:
                packet = parse_packet(decoded.transport.packet)
            except PacketError as exc:
                raise RuntimeError(f"source packet failure at observed frame {observed}: {exc}") from exc
            if isinstance(packet, MetaPacket):
                _validate_meta(packet)
                packet_layouts = validate_metadata(packet.metadata)
                if metadata is None:
                    metadata = packet.metadata
                    layouts = {layout.block_id: layout for layout in packet_layouts}
                elif packet.metadata != metadata:
                    raise RuntimeError("conflicting META packets in source video")
                record = SourceFrame(observed, transport_index, "META")
                meta_indices.append(transport_index)
            else:
                if layouts is None:
                    raise RuntimeError("source SYMBOL appeared before valid META")
                layout = layouts.get(packet.block_id)
                if layout is None:
                    raise RuntimeError(f"source SYMBOL has invalid block_id {packet.block_id}")
                fields = classify_symbol(packet.symbol_id, layout.source_symbols, layout.encoded_symbols)
                record = SourceFrame(
                    observed, transport_index, "SYMBOL", packet.block_id, packet.symbol_id,
                    layout.source_symbols, layout.encoded_symbols, **fields,
                )
            records.append(record)
            by_transport[transport_index] = record
            alignment_matches += int(transport_index == observed)
            previous_transport = transport_index
            observed += 1
            if observed % 1000 == 0:
                print(f"source decoded {observed}/{reported_frames or '?'} frames", flush=True)
    finally:
        capture.release()
    if not records or metadata is None or layouts is None:
        raise RuntimeError("source video contains no valid FVM metadata")
    _assign_meta_distances(records, meta_indices)
    summary = {
        "video": str(video), "resolution": f"{width}x{height}", "fps": fps,
        "observed_frames": len(records), "reported_frames": reported_frames,
        "rs_failures": 0, "transport_errors": 0, "packet_errors": 0,
        "unique_transport_indices": len(by_transport),
        "strictly_increasing_transport_indices": True,
        "observed_alignment_matches": alignment_matches,
        "observed_alignment_all": alignment_matches == len(records),
        "metadata": metadata,
        "block_layout": [asdict(layouts[index]) for index in sorted(layouts)],
        "meta_frames": len(meta_indices),
    }
    return records, by_transport, summary


def decode_platform_video(video: Path) -> tuple[list[PlatformFrame], dict[str, Any], dict[str, Any]]:
    capture, width, height, fps, reported_frames = _open_video(video)
    records: list[PlatformFrame] = []
    failures = FailureTracker()
    totals = Counter()
    try:
        observed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded = _decode_frame(frame)
            rs = decoded.rs
            totals["codewords_success"] += rs.codeword_successes
            totals["codewords_failed"] += rs.codeword_failures
            totals["corrected_symbols"] += rs.corrected_symbols
            if rs.logical is None:
                totals["rs_failures"] += 1
                failures.observe_failure(observed, rs)
                records.append(PlatformFrame(
                    observed, None, False, True, rs.codeword_successes, rs.codeword_failures,
                    rs.failed_codeword_indices, rs.corrected_symbols,
                    rs.max_corrections_per_codeword, rs.corrections_per_codeword,
                ))
            elif decoded.transport is None:
                totals["rs_successes"] += 1
                totals["transport_errors"] += 1
                failures.observe_non_failure_without_transport()
                records.append(PlatformFrame(
                    observed, None, False, False, rs.codeword_successes, rs.codeword_failures,
                    rs.failed_codeword_indices, rs.corrected_symbols,
                    rs.max_corrections_per_codeword, rs.corrections_per_codeword,
                    transport_failed=True,
                ))
            else:
                packet_failed = False
                descriptor = None
                try:
                    packet = parse_packet(decoded.transport.packet)
                    descriptor = PacketDiagnostic.from_packet(packet)
                except PacketError:
                    packet_failed = True
                    totals["packet_errors"] += 1
                failures.observe_success(decoded.transport.frame_index, descriptor)
                records.append(PlatformFrame(
                    observed, decoded.transport.frame_index, True, False,
                    rs.codeword_successes, rs.codeword_failures, rs.failed_codeword_indices,
                    rs.corrected_symbols, rs.max_corrections_per_codeword,
                    rs.corrections_per_codeword, packet_failed=packet_failed,
                    packet_descriptor=descriptor,
                ))
                totals["rs_successes"] += 1
            observed += 1
            if observed % 1000 == 0:
                print(f"platform decoded {observed}/{reported_frames or '?'} frames", flush=True)
    finally:
        capture.release()
    failure_summary = failures.summary()
    for event in failure_summary["events"]:
        if event["inference_confident"]:
            record = records[event["observed_frame_index"]]
            record.transport_index = event["inferred_transport_index"]
            record.inference_confident = True
    summary = {
        "video": str(video), "resolution": f"{width}x{height}", "fps": fps,
        "observed_frames": len(records), "reported_frames": reported_frames,
        "rs_success_frames": totals["rs_successes"], "rs_failed_frames": totals["rs_failures"],
        "codewords_success": totals["codewords_success"],
        "codewords_failed": totals["codewords_failed"],
        "corrected_symbols": totals["corrected_symbols"],
        "transport_errors": totals["transport_errors"], "packet_errors": totals["packet_errors"],
    }
    return records, summary, failure_summary


def pair_frames(
    source_by_transport: dict[int, SourceFrame], platform: list[PlatformFrame]
) -> tuple[list[PairedFrameRecord], dict[str, int]]:
    paired = []
    stats = Counter()
    for frame in platform:
        source = source_by_transport.get(frame.transport_index) if frame.transport_index is not None else None
        mapping_confident = source is not None and (not frame.rs_failed or frame.inference_confident)
        packet_mismatch = False
        if source is not None and frame.packet_descriptor is not None:
            expected = {"type": source.packet_type}
            if source.packet_type == "SYMBOL":
                expected.update({"block_id": source.block_id, "symbol_id": source.symbol_id})
            packet_mismatch = frame.packet_descriptor.to_dict() != expected
        packet_failed = frame.packet_failed or packet_mismatch
        if source is None:
            stats["unmapped"] += 1
        elif frame.rs_failed and mapping_confident:
            stats["failures_mapped"] += 1
        else:
            stats["success_mapped"] += 1
        if packet_mismatch:
            stats["packet_mapping_mismatches"] += 1
        alignment = source.observed_frame_index == frame.observed_frame_index if source else None
        stats["observed_alignment_matches"] += int(alignment is True)
        paired.append(PairedFrameRecord(
            transport_index=frame.transport_index,
            source_observed_frame_index=source.observed_frame_index if source else None,
            platform_observed_frame_index=frame.observed_frame_index,
            packet_type=source.packet_type if source else None,
            block_id=source.block_id if source else None,
            symbol_id=source.symbol_id if source else None,
            source_symbols=source.source_symbols if source else None,
            encoded_symbols=source.encoded_symbols if source else None,
            classification=source.classification if source and source.classification else "UNKNOWN",
            boundary_offset=source.boundary_offset if source else None,
            is_last_source=source.is_last_source if source else False,
            is_first_repair=source.is_first_repair if source else False,
            platform_rs_failed=frame.rs_failed,
            codeword_failures=frame.codeword_failures,
            failed_codeword_indices=frame.failed_codeword_indices,
            corrected_symbols=frame.corrected_symbols,
            max_corrections_per_codeword=frame.max_corrections_per_codeword,
            corrections_per_codeword=frame.corrections_per_codeword,
            transport_failed=frame.transport_failed,
            packet_failed=packet_failed,
            mapping_confident=mapping_confident,
            observed_alignment_match=alignment,
            frames_since_previous_meta=source.frames_since_previous_meta if source else None,
            frames_until_next_meta=source.frames_until_next_meta if source else None,
        ))
    stats["platform_frames"] = len(platform)
    stats["failures_total"] = sum(frame.rs_failed for frame in platform)
    stats["failures_ambiguous"] = stats["failures_total"] - stats["failures_mapped"]
    return paired, dict(stats)


def aggregate_frame_metrics(records: Iterable[PairedFrameRecord]) -> dict[str, Any]:
    eligible = [record for record in records if record.mapping_confident and not record.packet_failed]
    corrected = [record.corrected_symbols for record in eligible]
    successful_corrected = [
        record.corrected_symbols for record in eligible if not record.platform_rs_failed
    ]
    maxima = [record.max_corrections_per_codeword for record in eligible]
    successful_maxima = [
        record.max_corrections_per_codeword for record in eligible if not record.platform_rs_failed
    ]
    failed = sum(record.platform_rs_failed for record in eligible)
    count = len(eligible)
    return {
        "frame_count": count,
        "rs_failed_count": failed,
        "rs_failure_rate": failed / count if count else None,
        "mean_corrected_symbols": _mean(corrected),
        "median_corrected_symbols": _percentile(corrected, 50),
        "p90_corrected_symbols": _percentile(corrected, 90),
        "p95_corrected_symbols": _percentile(corrected, 95),
        "max_corrected_symbols": max(corrected, default=None),
        "rs_success_frame_count": len(successful_corrected),
        "mean_corrected_symbols_rs_success_frames": _mean(successful_corrected),
        "median_corrected_symbols_rs_success_frames": _percentile(successful_corrected, 50),
        "p95_corrected_symbols_rs_success_frames": _percentile(successful_corrected, 95),
        "max_corrected_symbols_rs_success_frames": max(successful_corrected, default=None),
        "mean_max_corrections_per_codeword": _mean(maxima),
        "mean_max_corrections_per_codeword_rs_success_frames": _mean(successful_maxima),
        "frames_with_max_corrections_ge_7": sum(value >= 7 for value in maxima),
        "frames_with_max_corrections_eq_8": sum(value == 8 for value in maxima),
        "transport_failure_count": sum(record.transport_failed for record in records),
        "corrected_symbols_note": "For RS-failed frames this counts corrections only in successful codewords.",
    }


def aggregate_codewords(records: Iterable[PairedFrameRecord]) -> list[dict[str, Any]]:
    eligible = [record for record in records if record.mapping_confident and not record.packet_failed]
    output = []
    for index in range(28):
        values = [record.corrections_per_codeword[index] for record in eligible]
        corrections = [value for value in values if value is not None]
        output.append({
            "codeword_index": index,
            "frame_count": len(values),
            "failed_count": sum(value is None for value in values),
            "failure_rate": sum(value is None for value in values) / len(values) if values else None,
            "mean_corrections": _mean(corrections),
            "p95_corrections": _percentile(corrections, 95),
            "max_corrections": max(corrections, default=None),
            "count_corrections_ge_7": sum(value >= 7 for value in corrections),
            "count_corrections_eq_8": sum(value == 8 for value in corrections),
        })
    return output


def spacing_analysis(boundaries: list[SourceFrame], source_records: list[SourceFrame]) -> dict[str, Any]:
    chronological = sorted(boundaries, key=lambda record: record.transport_index)
    chronological_rows = _spacing_rows(chronological, source_records)
    by_block = sorted(boundaries, key=lambda record: record.block_id if record.block_id is not None else -1)
    block_rows = _spacing_rows(by_block, source_records)
    lane_rows = []
    for lane in range(INTERLEAVE_WINDOW):
        lane_boundaries = [
            record for record in by_block if record.block_id is not None and record.block_id % INTERLEAVE_WINDOW == lane
        ]
        lane_rows.extend(_spacing_rows(lane_boundaries, source_records))
    return {
        "spacing_histogram": _spacing_histogram(chronological_rows),
        "chronological_spacing_histogram": _spacing_histogram(chronological_rows),
        "same_interleave_lane_spacing_histogram": _spacing_histogram(lane_rows),
        "chronological_pairs": chronological_rows,
        "same_interleave_lane_pairs": lane_rows,
        "block_order_pairs": block_rows,
    }


def _spacing_rows(ordered: list[SourceFrame], source_records: list[SourceFrame]) -> list[dict[str, Any]]:
    rows = []
    for previous, current in zip(ordered, ordered[1:]):
        lower, upper = sorted((previous.transport_index, current.transport_index))
        between = [record for record in source_records if lower < record.transport_index < upper]
        delta = current.transport_index - previous.transport_index
        rows.append({
            "from_block_id": previous.block_id,
            "to_block_id": current.block_id,
            "from_transport_index": previous.transport_index,
            "to_transport_index": current.transport_index,
            "transport_delta": delta,
            "absolute_transport_delta": abs(delta),
            "meta_frames_between": sum(record.packet_type == "META" for record in between),
            "symbol_frames_between": sum(record.packet_type == "SYMBOL" for record in between),
            "from_block_mod_interleave": previous.block_id % INTERLEAVE_WINDOW,
            "to_block_mod_interleave": current.block_id % INTERLEAVE_WINDOW,
        })
    return rows


def _spacing_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(Counter(row["transport_delta"] for row in rows).items())
    }


def analyze(
    source_video: Path,
    platform_video: Path,
    output_dir: Path,
    *,
    window_symbols: int = 8,
    full_frame_csv: bool = False,
) -> dict[str, Any]:
    if window_symbols <= 0:
        raise ValueError("window_symbols must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_records, source_map, source_summary = decode_source_video(source_video)
    platform_records, platform_summary, failure_diagnostics = decode_platform_video(platform_video)
    paired, mapping = pair_frames(source_map, platform_records)
    source_codec, source_codec_warning = probe_codec_frames(source_video)
    platform_codec, platform_codec_warning = probe_codec_frames(platform_video)
    codec_available = bool(source_codec and platform_codec)
    codec_warnings = [warning for warning in (source_codec_warning, platform_codec_warning) if warning]
    if source_codec and len(source_codec) != len(source_records):
        codec_warnings.append("source ffprobe/OpenCV frame counts differ; unavailable rows are not joined")
    if platform_codec and len(platform_codec) != len(platform_records):
        codec_warnings.append("platform ffprobe/OpenCV frame counts differ; unavailable rows are not joined")

    offsets = {
        offset: [record for record in paired if record.boundary_offset == offset]
        for offset in range(-window_symbols, window_symbols + 1)
    }
    offset_summary = {str(offset): aggregate_frame_metrics(records) for offset, records in offsets.items()}
    non_boundary_source = [
        record for record in paired
        if record.classification == "SOURCE" and record.boundary_offset is not None
        and abs(record.boundary_offset) > 16
    ]
    offset_zero = offsets[0]
    baseline = aggregate_frame_metrics(non_boundary_source)
    boundary_zero = aggregate_frame_metrics(offset_zero)
    boundary_effect = {
        "mean_corrected_symbols_difference": _difference(
            boundary_zero["mean_corrected_symbols_rs_success_frames"],
            baseline["mean_corrected_symbols_rs_success_frames"],
        ),
        "mean_corrected_symbols_ratio": _ratio(
            boundary_zero["mean_corrected_symbols_rs_success_frames"],
            baseline["mean_corrected_symbols_rs_success_frames"],
        ),
        "p95_corrected_symbols_difference": _difference(
            boundary_zero["p95_corrected_symbols_rs_success_frames"],
            baseline["p95_corrected_symbols_rs_success_frames"],
        ),
        "comparison_population": "RS-success frames only",
    }
    codeword_groups = {
        "all_source": [record for record in paired if record.classification == "SOURCE"],
        "boundary_offset_0": offset_zero,
        "boundary_offset_0_rs_success_frames": [
            record for record in offset_zero if not record.platform_rs_failed
        ],
        "near_boundary_-2_to_2": [
            record for record in paired if record.boundary_offset is not None and abs(record.boundary_offset) <= 2
        ],
        "non_boundary_source": non_boundary_source,
    }
    codeword_summary = {name: aggregate_codewords(records) for name, records in codeword_groups.items()}
    boundaries = [record for record in source_records if record.boundary_offset == 0]
    periodicity = spacing_analysis(boundaries, source_records)
    block_lanes = {
        str(lane): aggregate_frame_metrics([
            record for record in offset_zero if record.block_id is not None and record.block_id % 4 == lane
        ])
        for lane in range(4)
    }
    failures = [record for record in paired if record.platform_rs_failed]
    failure_events = {
        event["observed_frame_index"]: event for event in failure_diagnostics["events"]
    }
    failure_rows = [
        _failure_row(record, failure_events.get(record.platform_observed_frame_index),
                     source_codec, platform_codec)
        for record in failures
    ]
    boundary_rows = [
        _paired_row(record, source_codec, platform_codec)
        for record in paired
        if record.boundary_offset is not None and abs(record.boundary_offset) <= window_symbols
    ]
    failure_windows = _failure_observed_windows(failures, platform_records, platform_codec)
    failure_window_summary = {
        str(relative): {
            "frame_count": len(rows),
            "rs_failed_count": sum(row["rs_failed"] for row in rows),
            "mean_corrected_symbols": _mean([row["corrected_symbols"] for row in rows]),
            "mean_max_corrections_per_codeword": _mean([
                row["max_corrections_per_codeword"] for row in rows
            ]),
            "mean_pkt_size": _mean([
                row["pkt_size"] for row in rows if row["pkt_size"] is not None
            ]),
        }
        for relative in range(-3, 4)
        if (rows := [row for row in failure_windows if row["relative_observed_offset"] == relative])
    }
    success_boundary_cw = codeword_summary["boundary_offset_0_rs_success_frames"]
    cw_7_10_means = [success_boundary_cw[index]["mean_corrections"] for index in range(7, 11)]
    other_cw_means = [
        row["mean_corrections"] for row in success_boundary_cw if row["codeword_index"] not in range(7, 11)
    ]
    codec_summary = {
        "available": codec_available,
        "warnings": codec_warnings,
        "source_ffprobe_frames": len(source_codec),
        "platform_ffprobe_frames": len(platform_codec),
        "groups": {
            "offset_-1": _codec_group(offsets.get(-1, []), source_codec, platform_codec),
            "offset_0": _codec_group(offset_zero, source_codec, platform_codec),
            "offset_1": _codec_group(offsets.get(1, []), source_codec, platform_codec),
            "non_boundary_source": _codec_group(non_boundary_source, source_codec, platform_codec),
        },
    }
    failure_offsets = Counter(
        "UNKNOWN" if record.boundary_offset is None else str(record.boundary_offset)
        for record in failures
    )
    meta_distance_failures = [
        {
            "transport_index": record.transport_index,
            "boundary_offset": record.boundary_offset,
            "frames_since_previous_meta": record.frames_since_previous_meta,
            "frames_until_next_meta": record.frames_until_next_meta,
        }
        for record in failures
    ]
    result = {
        "format": "FVM_OFFLINE_BOUNDARY_DIAGNOSTICS_V1",
        "source": source_summary,
        "platform": platform_summary,
        "mapping": mapping,
        "failure_diagnostics": failure_diagnostics,
        "boundary": {
            "window_symbols": window_symbols,
            "last_source_frames": len(offset_zero),
            "last_source_rs_failures": boundary_zero["rs_failed_count"],
            "last_source_failure_rate": boundary_zero["rs_failure_rate"],
            "first_repair": offset_summary.get("1"),
            "non_boundary_source": baseline,
            "offset_0_effect_vs_non_boundary_source": boundary_effect,
            "failure_offset_counts": dict(failure_offsets),
            "block_mod_4": block_lanes,
            "failure_meta_distances": meta_distance_failures,
        },
        "offset_summary": offset_summary,
        "codewords": codeword_summary,
        "periodicity": periodicity,
        "codec": codec_summary,
        "failure_observed_window_summary": failure_window_summary,
        "conclusion_inputs": {
            "mapped_failure_count": mapping.get("failures_mapped", 0),
            "ambiguous_failure_count": mapping.get("failures_ambiguous", 0),
            "offset_0_failure_count": failure_offsets.get("0", 0),
            "failure_offsets": dict(failure_offsets),
            "boundary_offset_0_success_frame_cw_7_10_mean": _mean(cw_7_10_means),
            "boundary_offset_0_success_frame_other_cw_mean": _mean(other_cw_means),
        },
    }
    _write_json(output_dir / "boundary_summary.json", result)
    _write_csv(output_dir / "boundary_offsets.csv", [
        {"boundary_offset": offset, **offset_summary[str(offset)]}
        for offset in range(-window_symbols, window_symbols + 1)
    ])
    _write_csv(output_dir / "boundary_frames.csv", boundary_rows)
    _write_csv(output_dir / "rs_failure_attribution.csv", failure_rows)
    codeword_rows = [
        {"group": group, **row} for group, rows in codeword_summary.items() for row in rows
    ]
    _write_csv(output_dir / "codeword_boundary_summary.csv", codeword_rows)
    _write_csv(output_dir / "failure_observed_windows.csv", failure_windows)
    if full_frame_csv:
        _write_csv(output_dir / "all_paired_frames.csv", [
            _paired_row(record, source_codec, platform_codec) for record in paired
        ])
    return result


def _decode_frame(frame):
    bits, _ = decode_frame(frame, PHYSICAL_CONFIG)
    return decode_transport_physical(matrix_to_physical(bits))


def _open_video(video: Path):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width, height) != (WIDTH, HEIGHT) or not math.isclose(fps, FPS, abs_tol=0.01):
        capture.release()
        raise RuntimeError(f"video does not match fixed FVM profile: {width}x{height} {fps}fps")
    return capture, width, height, fps, reported_frames


def _assign_meta_distances(records: list[SourceFrame], meta_indices: list[int]) -> None:
    for record in records:
        position = bisect.bisect_left(meta_indices, record.transport_index)
        previous = meta_indices[position - 1] if position else None
        next_index = meta_indices[position] if position < len(meta_indices) else None
        if next_index == record.transport_index:
            previous = next_index
        record.frames_since_previous_meta = (
            record.transport_index - previous if previous is not None else None
        )
        record.frames_until_next_meta = (
            next_index - record.transport_index
            if next_index is not None and next_index >= record.transport_index else None
        )


def _paired_row(
    record: PairedFrameRecord,
    source_codec: list[CodecFrame],
    platform_codec: list[CodecFrame],
) -> dict[str, Any]:
    source_meta = _codec_at(source_codec, record.source_observed_frame_index)
    platform_meta = _codec_at(platform_codec, record.platform_observed_frame_index)
    source_size = source_meta.pkt_size if source_meta else None
    platform_size = platform_meta.pkt_size if platform_meta else None
    row = asdict(record)
    row["failed_codeword_indices"] = json.dumps(list(record.failed_codeword_indices))
    row["corrections_per_codeword"] = json.dumps(list(record.corrections_per_codeword))
    row.update({
        "source_pkt_size": source_size,
        "source_pict_type": source_meta.pict_type if source_meta else None,
        "source_key_frame": source_meta.key_frame if source_meta else None,
        "platform_pkt_size": platform_size,
        "platform_pict_type": platform_meta.pict_type if platform_meta else None,
        "platform_key_frame": platform_meta.key_frame if platform_meta else None,
        "pkt_size_ratio": platform_size / source_size if source_size and platform_size is not None else None,
    })
    return row


def _failure_row(
    record: PairedFrameRecord,
    event: dict[str, Any] | None,
    source_codec: list[CodecFrame],
    platform_codec: list[CodecFrame],
) -> dict[str, Any]:
    row = _paired_row(record, source_codec, platform_codec)
    previous = event.get("previous_successful") if event else None
    following = event.get("next_successful") if event else None
    row.update({
        "previous_successful_transport_index": previous.get("transport_index") if previous else None,
        "previous_successful_packet": json.dumps(previous.get("packet")) if previous else None,
        "next_successful_transport_index": following.get("transport_index") if following else None,
        "next_successful_packet": json.dumps(following.get("packet")) if following else None,
        "inference_confident": event.get("inference_confident", False) if event else False,
    })
    return row


def _failure_observed_windows(
    failures: list[PairedFrameRecord],
    platform_records: list[PlatformFrame],
    platform_codec: list[CodecFrame],
) -> list[dict[str, Any]]:
    rows = []
    for failure in failures:
        for observed in range(max(0, failure.platform_observed_frame_index - 3),
                              min(len(platform_records), failure.platform_observed_frame_index + 4)):
            record = platform_records[observed]
            codec = _codec_at(platform_codec, observed)
            rows.append({
                "failure_observed_frame_index": failure.platform_observed_frame_index,
                "neighbor_observed_frame_index": observed,
                "relative_observed_offset": observed - failure.platform_observed_frame_index,
                "rs_failed": record.rs_failed,
                "corrected_symbols": record.corrected_symbols,
                "max_corrections_per_codeword": record.max_corrections_per_codeword,
                "pkt_size": codec.pkt_size if codec else None,
                "pict_type": codec.pict_type if codec else None,
                "key_frame": codec.key_frame if codec else None,
            })
    return rows


def _codec_group(
    records: list[PairedFrameRecord], source_codec: list[CodecFrame], platform_codec: list[CodecFrame]
) -> dict[str, Any]:
    source_rows = [_codec_at(source_codec, record.source_observed_frame_index) for record in records]
    platform_rows = [_codec_at(platform_codec, record.platform_observed_frame_index) for record in records]
    return {
        "frame_count": len(records),
        "source": _codec_stats([row for row in source_rows if row]),
        "platform": _codec_stats([row for row in platform_rows if row]),
    }


def _codec_stats(rows: list[CodecFrame]) -> dict[str, Any]:
    sizes = [row.pkt_size for row in rows if row.pkt_size is not None]
    return {
        "pkt_size_mean": _mean(sizes), "pkt_size_median": _percentile(sizes, 50),
        "pkt_size_p10": _percentile(sizes, 10), "pkt_size_p90": _percentile(sizes, 90),
        "pict_type_counts": dict(Counter(row.pict_type for row in rows if row.pict_type)),
        "key_frame_count": sum(row.key_frame is True for row in rows),
    }


def _codec_at(rows: list[CodecFrame], index: int | None) -> CodecFrame | None:
    return rows[index] if index is not None and 0 <= index < len(rows) else None


def _mean(values: list[int | float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[int | float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator not in (None, 0) else None


def _difference(left: float | None, right: float | None) -> float | None:
    return left - right if left is not None and right is not None else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    integer = _optional_int(value)
    return bool(integer) if integer is not None else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
