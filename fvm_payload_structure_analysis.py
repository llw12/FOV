"""Offline payload, physical-structure, and raw-channel diagnostics for FVM FILE MODE."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

import fov
from fov import (INTERLEAVE_WINDOW, MetaPacket, PacketError, SymbolPacket, encode_symbol,
                 parse_packet, sha256_file, validate_metadata)
from fvm_boundary_analysis import classify_symbol
from fvm_diagnostics import PacketDiagnostic
from fvm_file_common import (FPS, HEIGHT, PHYSICAL_CONFIG, TRANSPORT_HEADER, WIDTH,
                             decode_transport_physical, encode_transport_physical,
                             matrix_to_physical, wrap_packet)
from scripts.fvm0_common import decode_frame
from fvm_physical_analysis import (TILE_COLS, TILE_ROWS, compare_error_masks,
                                   lane_physical_summary, physical_metrics,
                                   raw_physical_errors, tile_structure_rows)
from scripts.fvm0_rs_common import (BIT_ORDER, CODED_RS_BYTES, LOGICAL_BYTES, RS_CODEWORDS,
                                    RS_K, RS_N, encode_logical, interleave)
from video2file import _validate_meta

FOV_HEADER_BYTES = fov._SYMBOL_HEADER.size
TRANSPORT_CRC_BYTES = 4


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class SourceIndex:
    observed_frame_index: int
    transport_index: int
    packet_type: str
    block_id: int | None = None
    symbol_id: int | None = None
    source_symbols: int | None = None
    encoded_symbols: int | None = None
    data_size: int | None = None
    boundary_offset: int | None = None
    classification: str | None = None
    selected: bool = False
    selection: str | None = None


@dataclass
class AnalyzedFrame:
    source: SourceIndex
    packet: SymbolPacket
    packet_bytes: bytes
    payload_metrics: dict[str, Any]
    packet_metrics: dict[str, Any]
    logical_metrics: dict[str, Any]
    padding_metrics: dict[str, Any]
    reserved_metrics: dict[str, Any]
    physical_metrics: dict[str, Any]
    physical_tiles: list[dict[str, Any]]
    source_raw: dict[str, Any]
    source_codeword_errors: list[dict[str, Any]]
    source_tile_errors: list[dict[str, Any]]
    codeword_structure: list[dict[str, Any]]
    canonical_physical: bytes = field(repr=False)
    source_error_mask: np.ndarray = field(repr=False)
    expected_original_length: int | None = None
    prefix_match_bytes: int | None = None
    prefix_match_ratio: float | None = None
    prefix_matches_original: bool | None = None
    systematic_identity: bool | None = None
    suffix_metrics: dict[str, Any] | None = None
    suffix_sha256: str | None = None
    platform_observed_frame_index: int | None = None
    platform_rs_failed: bool | None = None
    platform_failed_codeword_indices: tuple[int, ...] = ()
    platform_raw: dict[str, Any] | None = None
    platform_codeword_errors: list[dict[str, Any]] | None = None
    platform_tile_errors: list[dict[str, Any]] | None = None
    error_overlap_count: int | None = None
    platform_only_error_count: int | None = None
    source_only_error_count: int | None = None


@dataclass
class PendingFailure:
    observed_frame_index: int
    observed_physical: bytes
    failed_codeword_indices: tuple[int, ...]
    codeword_failures: int


def byte_metrics(data: bytes) -> dict[str, Any]:
    """Return deterministic byte/bit structure metrics without retaining the input."""
    if not data:
        return {
            "length": 0, "byte_zero_fraction": None, "byte_ff_fraction": None,
            "unique_byte_count": 0, "entropy_bits_per_byte": None,
            "bit_one_fraction": None, "bit_zero_fraction": None,
            "bit_transition_density_1d": None, "longest_zero_byte_run": 0,
            "longest_ff_byte_run": 0, "longest_identical_byte_run": 0,
            "longest_zero_bit_run": 0, "longest_one_bit_run": 0,
        }
    values = np.frombuffer(data, dtype=np.uint8)
    counts = np.bincount(values, minlength=256)
    probabilities = counts[counts > 0] / len(values)
    bits = np.unpackbits(values, bitorder=BIT_ORDER)
    ones = int(bits.sum())
    return {
        "length": len(data),
        "byte_zero_fraction": float(counts[0] / len(values)),
        "byte_ff_fraction": float(counts[255] / len(values)),
        "unique_byte_count": int(np.count_nonzero(counts)),
        "entropy_bits_per_byte": float(-(probabilities * np.log2(probabilities)).sum()),
        "bit_one_fraction": ones / len(bits),
        "bit_zero_fraction": (len(bits) - ones) / len(bits),
        "bit_transition_density_1d": float(np.count_nonzero(bits[1:] != bits[:-1]) / (len(bits) - 1))
        if len(bits) > 1 else 0.0,
        "longest_zero_byte_run": _longest_value_run(values, 0),
        "longest_ff_byte_run": _longest_value_run(values, 255),
        "longest_identical_byte_run": _longest_identical_run(values),
        "longest_zero_bit_run": _longest_value_run(bits, 0),
        "longest_one_bit_run": _longest_value_run(bits, 1),
    }


def symbol_logical_segments(packet: SymbolPacket, transport_index: int) -> tuple[bytes, bytes, list[Segment]]:
    """Rebuild the canonical logical frame and its contiguous production-defined segments."""
    packet_bytes = encode_symbol(packet.file_id, packet.block_id, packet.symbol_id, packet.payload)
    logical = wrap_packet(packet_bytes, transport_index)
    packet_start = TRANSPORT_HEADER.size
    payload_start = packet_start + FOV_HEADER_BYTES
    payload_end = payload_start + len(packet.payload)
    packet_end = packet_start + len(packet_bytes)
    segments = [
        Segment("transport_header", 0, packet_start),
        Segment("fov_symbol_header", packet_start, payload_start),
        Segment("raptorq_payload", payload_start, payload_end),
        Segment("fov_packet_crc", payload_end, packet_end),
        Segment("transport_shake_padding", packet_end, LOGICAL_BYTES - TRANSPORT_CRC_BYTES),
        Segment("transport_crc", LOGICAL_BYTES - TRANSPORT_CRC_BYTES, LOGICAL_BYTES),
    ]
    validate_segment_cover(segments, LOGICAL_BYTES)
    return packet_bytes, logical, segments


def validate_segment_cover(segments: list[Segment], total_length: int) -> None:
    if not segments or segments[0].start != 0 or segments[-1].end != total_length:
        raise ValueError("segments do not cover the logical frame")
    if any(segment.start < 0 or segment.end < segment.start for segment in segments):
        raise ValueError("invalid segment range")
    if any(left.end != right.start for left, right in zip(segments, segments[1:])):
        raise ValueError("segments overlap or contain gaps")


def range_overlap(start: int, end: int, other_start: int, other_end: int) -> int:
    return max(0, min(end, other_end) - max(start, other_start))


def codeword_segment_rows(
    segments: list[Segment], *, actual_source_length: int | None = None
) -> list[dict[str, Any]]:
    """Map each RS data codeword to logical segments and optional tail/suffix regions."""
    payload = next(segment for segment in segments if segment.name == "raptorq_payload")
    regions = list(segments)
    if actual_source_length is not None:
        boundary = payload.start + actual_source_length
        if not payload.start <= boundary <= payload.end:
            raise ValueError("actual source length falls outside payload")
        regions.extend([
            Segment("real_source", payload.start, boundary),
            Segment("observed_payload_suffix", boundary, payload.end),
        ])
    rows = []
    for codeword in range(RS_CODEWORDS):
        start, end = codeword * RS_K, (codeword + 1) * RS_K
        row: dict[str, Any] = {
            "codeword_index": codeword, "logical_start": start, "logical_end": end,
        }
        for region in regions:
            row[f"{region.name}_bytes"] = range_overlap(start, end, region.start, region.end)
        if range_overlap(start, end, payload.start, payload.end):
            row["payload_byte_start"] = max(start, payload.start) - payload.start
            row["payload_byte_end"] = min(end, payload.end) - payload.start
        else:
            row["payload_byte_start"] = row["payload_byte_end"] = None
        rows.append(row)
    return rows


def payload_boundary_location(segments: list[Segment], actual_source_length: int) -> dict[str, int]:
    payload = next(segment for segment in segments if segment.name == "raptorq_payload")
    logical_offset = payload.start + actual_source_length
    return {
        "logical_offset": logical_offset,
        "codeword_index": logical_offset // RS_K,
        "offset_within_codeword": logical_offset % RS_K,
    }


def validate_original_tail(payload: bytes, original_chunk: bytes) -> dict[str, Any]:
    if len(original_chunk) > len(payload):
        raise ValueError("original chunk is longer than payload")
    matching = 0
    for expected, actual in zip(original_chunk, payload):
        if expected != actual:
            break
        matching += 1
    suffix = payload[len(original_chunk):]
    return {
        "expected_original_length": len(original_chunk),
        "payload_prefix_match_bytes": matching,
        "payload_prefix_match_ratio": matching / len(original_chunk) if original_chunk else 1.0,
        "payload_prefix_matches_original": payload[:len(original_chunk)] == original_chunk,
        "systematic_identity": payload[:len(original_chunk)] == original_chunk,
        "suffix": suffix,
        "suffix_metrics": byte_metrics(suffix),
        "suffix_sha256": hashlib.sha256(suffix).hexdigest(),
    }


def infer_pending_transport_indices(
    previous_transport: int | None,
    next_transport: int,
    pending_count: int,
    *,
    first_observed_index: int | None = None,
) -> list[int] | None:
    """Conservatively infer a contiguous failed run without observed-index fallback."""
    if pending_count <= 0:
        return []
    if previous_transport is None:
        if first_observed_index != 0 or next_transport != pending_count:
            return None
        first = 0
    else:
        if next_transport <= previous_transport or next_transport - previous_transport - 1 != pending_count:
            return None
        first = previous_transport + 1
    return list(range(first, first + pending_count))


def analyze(
    source_video: Path,
    platform_video: Path,
    output_dir: Path,
    *,
    original_file: Path | None = None,
    boundary_summary: Path | None = None,
    window_symbols: int = 8,
    dump_target_bytes: bool = False,
) -> dict[str, Any]:
    if window_symbols <= 0:
        raise ValueError("window_symbols must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    original_handle = original_file.open("rb") if original_file else None
    try:
        source_map, analyzed, source_gate, metadata = _decode_source(
            source_video, original_handle, window_symbols
        )
    finally:
        if original_handle:
            original_handle.close()
    platform_summary, mapping_summary, failures = _decode_platform(
        platform_video, source_map, analyzed
    )
    original_validation = _original_validation(original_file, metadata, analyzed)
    boundary_crosscheck = _boundary_crosscheck(boundary_summary, failures)
    if boundary_summary and not boundary_crosscheck["matches"]:
        raise RuntimeError("boundary-summary failure attribution does not match fresh analysis")

    rows = [_target_row(frame) for frame in analyzed.values()]
    offset_summary = _build_group_summary(analyzed.values())
    suffix_rows = [_suffix_row(frame) for frame in analyzed.values() if frame.source.boundary_offset == 0]
    codeword_structure_rows = _codeword_structure_rows(analyzed.values())
    segment_rows = _representative_segment_rows(analyzed.values())
    lane_rows = lane_physical_summary()
    raw_frame_rows = _raw_frame_rows(analyzed.values())
    raw_codeword_rows = _raw_codeword_rows(analyzed.values())
    raw_tile_rows = _aggregate_raw_tiles(analyzed.values())
    structure_tile_rows = _aggregate_structure_tiles(analyzed.values())
    failure_rows = [_failure_row(frame) for frame in analyzed.values() if frame.platform_rs_failed]
    suffix_summary = _suffix_summary(analyzed.values())
    conclusion_inputs = _conclusion_inputs(analyzed.values(), suffix_summary)

    result = {
        "format": "FVM_PAYLOAD_STRUCTURE_DIAGNOSTICS_V1",
        "source_gate": source_gate,
        "platform": platform_summary,
        "mapping": mapping_summary,
        "boundary_crosscheck": boundary_crosscheck,
        "original_file_validation": original_validation,
        "payload": {
            "baseline_sampling": "per block: source symbols 0, K//4, K//2, 3K//4 outside abs(offset)<=16",
            "systematic_identity": _systematic_summary(analyzed.values()),
            "offsets": offset_summary,
            "last_source_suffix": suffix_summary,
        },
        "transport_logical": _transport_summary(analyzed.values()),
        "rs_codewords": {
            "rs_k": RS_K, "rs_n": RS_N, "codewords": RS_CODEWORDS,
            "cw7_to_cw10_segment_attribution": [
                row for row in segment_rows if row["codeword_index"] in range(7, 11)
            ],
        },
        "physical_matrix": {
            "rows": PHYSICAL_CONFIG.rows, "columns": PHYSICAL_CONFIG.cols,
            "tile_rows": TILE_ROWS, "tile_columns": TILE_COLS,
            "lane_mapping_rows": len(lane_rows),
        },
        "raw_channel": _raw_channel_summary(analyzed.values()),
        "block_lanes": _block_lane_summary(analyzed.values()),
        "failures": {"count": len(failure_rows), "rows": failure_rows},
        "conclusion_inputs": conclusion_inputs,
    }
    _write_json(output_dir / "payload_structure_summary.json", result)
    _write_csv(output_dir / "target_frame_structure.csv", rows)
    _write_csv(output_dir / "offset_structure_summary.csv", _flatten_group_summary(offset_summary))
    _write_csv(output_dir / "last_source_suffix_summary.csv", suffix_rows)
    _write_csv(output_dir / "rs_codeword_structure.csv", codeword_structure_rows)
    _write_csv(output_dir / "rs_codeword_segment_map.csv", segment_rows)
    _write_csv(output_dir / "lane_physical_mapping.csv", lane_rows)
    _write_csv(output_dir / "raw_physical_error_frames.csv", raw_frame_rows)
    _write_csv(output_dir / "raw_physical_error_codewords.csv", raw_codeword_rows)
    _write_csv(output_dir / "raw_physical_error_tiles.csv", raw_tile_rows)
    _write_csv(output_dir / "physical_tile_structure.csv", structure_tile_rows)
    _write_csv(output_dir / "failure_structure_attribution.csv", failure_rows)
    if dump_target_bytes:
        dump_dir = output_dir / "target-bytes"
        dump_dir.mkdir(exist_ok=True)
        for frame in analyzed.values():
            if frame.source.boundary_offset is not None and abs(frame.source.boundary_offset) <= window_symbols:
                (dump_dir / f"transport-{frame.source.transport_index}-payload.bin").write_bytes(frame.packet.payload)
    return result


def _decode_source(
    video: Path, original_handle, window_symbols: int
) -> tuple[dict[int, SourceIndex], dict[int, AnalyzedFrame], dict[str, Any], dict[str, Any]]:
    capture, width, height, fps, reported = _open_video(video)
    source_map: dict[int, SourceIndex] = {}
    analyzed: dict[int, AnalyzedFrame] = {}
    layouts = None
    metadata = None
    previous_transport = None
    alignment = 0
    meta_frames = 0
    try:
        observed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            bits, _ = decode_frame(frame, PHYSICAL_CONFIG)
            observed_physical = matrix_to_physical(bits)
            decoded = decode_transport_physical(observed_physical)
            if decoded.rs.logical is None:
                raise RuntimeError(f"source RS failure at observed frame {observed}")
            if decoded.transport is None:
                raise RuntimeError(f"source transport failure at observed frame {observed}: {decoded.transport_error}")
            transport = decoded.transport.frame_index
            if transport in source_map:
                raise RuntimeError(f"duplicate source transport index {transport}")
            if previous_transport is not None and transport <= previous_transport:
                raise RuntimeError("source transport indices are not strictly increasing")
            try:
                packet = parse_packet(decoded.transport.packet)
            except PacketError as exc:
                raise RuntimeError(f"source packet failure at observed frame {observed}: {exc}") from exc
            if isinstance(packet, MetaPacket):
                _validate_meta(packet)
                candidate = validate_metadata(packet.metadata)
                if metadata is None:
                    metadata = packet.metadata
                    layouts = {item.block_id: item for item in candidate}
                elif packet.metadata != metadata:
                    raise RuntimeError("conflicting source META packets")
                index = SourceIndex(observed, transport, "META")
                meta_frames += 1
            else:
                if layouts is None:
                    raise RuntimeError("source SYMBOL appeared before META")
                layout = layouts.get(packet.block_id)
                if layout is None:
                    raise RuntimeError(f"invalid source block {packet.block_id}")
                classification = classify_symbol(packet.symbol_id, layout.source_symbols, layout.encoded_symbols)
                offset = classification["boundary_offset"]
                sample = packet.symbol_id in {
                    0, layout.source_symbols // 4, layout.source_symbols // 2,
                    3 * layout.source_symbols // 4,
                } and abs(offset) > 16
                selected = abs(offset) <= window_symbols or sample
                selection = "BOUNDARY_WINDOW" if abs(offset) <= window_symbols else (
                    "NON_BOUNDARY_SAMPLE" if sample else None
                )
                index = SourceIndex(
                    observed, transport, "SYMBOL", packet.block_id, packet.symbol_id,
                    layout.source_symbols, layout.encoded_symbols, layout.data_size,
                    offset, classification["classification"], selected, selection,
                )
                if selected:
                    original_chunk = _read_original_chunk(
                        original_handle, metadata, layout.data_size, packet.block_id, packet.symbol_id
                    ) if original_handle and packet.symbol_id < layout.source_symbols else None
                    analyzed[transport] = _analyze_source_frame(
                        index, packet, decoded.transport.packet, observed_physical, original_chunk
                    )
            source_map[transport] = index
            alignment += int(transport == observed)
            previous_transport = transport
            observed += 1
            if observed % 1000 == 0:
                print(f"source decoded {observed}/{reported or '?'} frames", flush=True)
    finally:
        capture.release()
    if not source_map or metadata is None:
        raise RuntimeError("source video has no valid FVM metadata")
    gate = {
        "video": str(video), "resolution": f"{width}x{height}", "fps": fps,
        "observed_frames": len(source_map), "reported_frames": reported,
        "rs_failures": 0, "transport_failures": 0, "packet_failures": 0,
        "unique_transport_indices": len(source_map),
        "strictly_increasing_transport_indices": True,
        "observed_alignment_matches": alignment,
        "observed_alignment_all": alignment == len(source_map),
        "meta_frames": meta_frames, "selected_symbol_frames": len(analyzed),
        "metadata": metadata,
    }
    return source_map, analyzed, gate, metadata


def _analyze_source_frame(
    source: SourceIndex,
    packet: SymbolPacket,
    raw_packet: bytes,
    observed_physical: bytes,
    original_chunk: bytes | None,
) -> AnalyzedFrame:
    canonical_packet, logical, segments = symbol_logical_segments(packet, source.transport_index)
    if canonical_packet != raw_packet:
        raise RuntimeError("source packet re-encoding did not reproduce source-video packet truth")
    codewords = encode_logical(logical)
    canonical_physical = interleave(codewords) + encode_transport_physical(
        canonical_packet, source.transport_index
    )[CODED_RS_BYTES:]
    if canonical_physical != encode_transport_physical(canonical_packet, source.transport_index):
        raise RuntimeError("canonical physical helper disagreement")
    raw, raw_cw, raw_tiles, error_mask = raw_physical_errors(canonical_physical, observed_physical)
    padding = next(item for item in segments if item.name == "transport_shake_padding")
    reserved = canonical_physical[CODED_RS_BYTES:]
    validation = validate_original_tail(packet.payload, original_chunk) if original_chunk is not None else None
    frame = AnalyzedFrame(
        source=source, packet=packet, packet_bytes=canonical_packet,
        payload_metrics=byte_metrics(packet.payload), packet_metrics=byte_metrics(canonical_packet),
        logical_metrics=byte_metrics(logical), padding_metrics=byte_metrics(logical[padding.start:padding.end]),
        reserved_metrics=byte_metrics(reserved), physical_metrics=physical_metrics(canonical_physical),
        physical_tiles=tile_structure_rows(canonical_physical),
        source_raw=raw, source_codeword_errors=raw_cw, source_tile_errors=raw_tiles,
        codeword_structure=_codeword_metrics(codewords), canonical_physical=canonical_physical,
        source_error_mask=error_mask,
    )
    if validation:
        frame.expected_original_length = validation["expected_original_length"]
        frame.prefix_match_bytes = validation["payload_prefix_match_bytes"]
        frame.prefix_match_ratio = validation["payload_prefix_match_ratio"]
        frame.prefix_matches_original = validation["payload_prefix_matches_original"]
        frame.systematic_identity = validation["systematic_identity"]
        frame.suffix_metrics = validation["suffix_metrics"]
        frame.suffix_sha256 = validation["suffix_sha256"]
    return frame


def _decode_platform(
    video: Path, source_map: dict[int, SourceIndex], analyzed: dict[int, AnalyzedFrame]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    capture, width, height, fps, reported = _open_video(video)
    totals = Counter()
    previous_success: int | None = None
    pending: list[PendingFailure] = []
    failure_rows: list[dict[str, Any]] = []
    observed_alignment = 0

    def close_pending(next_transport: int) -> None:
        nonlocal pending
        inferred = infer_pending_transport_indices(
            previous_success, next_transport, len(pending),
            first_observed_index=pending[0].observed_frame_index if pending else None,
        )
        for offset, failure in enumerate(pending):
            transport = inferred[offset] if inferred is not None else None
            source = source_map.get(transport) if transport is not None else None
            if source is not None:
                totals["failures_mapped"] += 1
                observed_alignment_nonlocal[0] += int(source.observed_frame_index == failure.observed_frame_index)
                if transport in analyzed:
                    _attach_platform(
                        analyzed[transport], failure.observed_frame_index, failure.observed_physical,
                        True, failure.failed_codeword_indices,
                    )
            else:
                totals["failures_ambiguous"] += 1
            failure_rows.append({
                "observed_frame_index": failure.observed_frame_index,
                "transport_index": transport,
                "mapping_confident": source is not None,
                "block_id": source.block_id if source else None,
                "symbol_id": source.symbol_id if source else None,
                "boundary_offset": source.boundary_offset if source else None,
                "failed_codeword_indices": list(failure.failed_codeword_indices),
            })
        pending = []

    observed_alignment_nonlocal = [0]
    try:
        observed = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            bits, _ = decode_frame(frame, PHYSICAL_CONFIG)
            physical = matrix_to_physical(bits)
            decoded = decode_transport_physical(physical)
            totals["codewords_failed"] += decoded.rs.codeword_failures
            if decoded.rs.logical is None:
                totals["rs_failures"] += 1
                pending.append(PendingFailure(
                    observed, physical, decoded.rs.failed_codeword_indices,
                    decoded.rs.codeword_failures,
                ))
            elif decoded.transport is None:
                totals["transport_failures"] += 1
                if pending:
                    totals["failures_ambiguous"] += len(pending)
                    pending = []
                previous_success = None
            else:
                transport = decoded.transport.frame_index
                if pending:
                    close_pending(transport)
                source = source_map.get(transport)
                if source is None:
                    totals["success_unmapped"] += 1
                else:
                    totals["success_mapped"] += 1
                    observed_alignment_nonlocal[0] += int(source.observed_frame_index == observed)
                    try:
                        packet = parse_packet(decoded.transport.packet)
                        descriptor = PacketDiagnostic.from_packet(packet)
                    except PacketError:
                        descriptor = None
                        totals["packet_failures"] += 1
                    if descriptor is not None and not _descriptor_matches(source, descriptor):
                        totals["packet_mapping_mismatches"] += 1
                    if transport in analyzed:
                        _attach_platform(analyzed[transport], observed, physical, False, ())
                previous_success = transport
            observed += 1
            if observed % 1000 == 0:
                print(f"platform decoded {observed}/{reported or '?'} frames", flush=True)
    finally:
        capture.release()
    if pending:
        totals["failures_ambiguous"] += len(pending)
        for failure in pending:
            failure_rows.append({
                "observed_frame_index": failure.observed_frame_index,
                "transport_index": None, "mapping_confident": False,
                "block_id": None, "symbol_id": None, "boundary_offset": None,
                "failed_codeword_indices": list(failure.failed_codeword_indices),
            })
    observed_alignment = observed_alignment_nonlocal[0]
    summary = {
        "video": str(video), "resolution": f"{width}x{height}", "fps": fps,
        "observed_frames": observed, "reported_frames": reported,
        "rs_failures": totals["rs_failures"], "failed_codewords": totals["codewords_failed"],
        "transport_failures": totals["transport_failures"], "packet_failures": totals["packet_failures"],
    }
    mapping = {
        "success_mapped": totals["success_mapped"],
        "success_unmapped": totals["success_unmapped"],
        "failures_total": totals["rs_failures"],
        "failures_confidently_mapped": totals["failures_mapped"],
        "ambiguous_failures": totals["failures_ambiguous"],
        "observed_alignment_matches": observed_alignment,
        "packet_mapping_mismatches": totals["packet_mapping_mismatches"],
    }
    return summary, mapping, failure_rows


def _attach_platform(
    frame: AnalyzedFrame,
    observed_index: int,
    observed_physical: bytes,
    rs_failed: bool,
    failed_codewords: tuple[int, ...],
) -> None:
    raw, codewords, tiles, error_mask = raw_physical_errors(frame.canonical_physical, observed_physical)
    overlap = compare_error_masks(frame.source_error_mask, error_mask)
    frame.platform_observed_frame_index = observed_index
    frame.platform_rs_failed = rs_failed
    frame.platform_failed_codeword_indices = failed_codewords
    frame.platform_raw = raw
    frame.platform_codeword_errors = codewords
    frame.platform_tile_errors = tiles
    frame.error_overlap_count = overlap["error_overlap_count"]
    frame.platform_only_error_count = overlap["platform_only_error_count"]
    frame.source_only_error_count = overlap["source_only_error_count"]


def _read_original_chunk(handle, metadata, data_size: int, block_id: int, symbol_id: int) -> bytes:
    symbol_size = metadata["symbol_size"]
    within_block = symbol_id * symbol_size
    length = max(0, min(symbol_size, data_size - within_block))
    handle.seek(block_id * metadata["block_size"] + within_block)
    data = handle.read(length)
    if len(data) != length:
        raise RuntimeError("original file ended before metadata-declared data")
    return data


def _codeword_metrics(codewords: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for index, codeword in enumerate(codewords):
        data = codeword[:RS_K].tobytes()
        parity = codeword[RS_K:].tobytes()
        full = codeword.tobytes()
        row = {"codeword_index": index}
        for prefix, values in (("data", data), ("parity", parity), ("full", full)):
            metrics = byte_metrics(values)
            for name in ("entropy_bits_per_byte", "bit_one_fraction", "bit_transition_density_1d", "unique_byte_count"):
                row[f"{prefix}_{name}"] = metrics[name]
        rows.append(row)
    return rows


def _target_row(frame: AnalyzedFrame) -> dict[str, Any]:
    row = _frame_identity(frame)
    for prefix, metrics in (
        ("payload", frame.payload_metrics), ("packet", frame.packet_metrics),
        ("logical", frame.logical_metrics), ("transport_padding", frame.padding_metrics),
        ("reserved", frame.reserved_metrics), ("physical", frame.physical_metrics),
        ("source_raw", frame.source_raw),
    ):
        row.update({f"{prefix}_{key}": value for key, value in metrics.items()})
    if frame.platform_raw:
        row.update({f"platform_raw_{key}": value for key, value in frame.platform_raw.items()})
    row.update({
        "expected_original_length": frame.expected_original_length,
        "payload_prefix_match_bytes": frame.prefix_match_bytes,
        "payload_prefix_match_ratio": frame.prefix_match_ratio,
        "payload_prefix_matches_original": frame.prefix_matches_original,
        "systematic_identity": frame.systematic_identity,
        "suffix_sha256": frame.suffix_sha256,
        "platform_observed_frame_index": frame.platform_observed_frame_index,
        "platform_rs_failed": frame.platform_rs_failed,
        "platform_failed_codeword_indices": json.dumps(list(frame.platform_failed_codeword_indices)),
        "error_overlap_count": frame.error_overlap_count,
        "platform_only_error_count": frame.platform_only_error_count,
        "source_only_error_count": frame.source_only_error_count,
    })
    return row


def _frame_identity(frame: AnalyzedFrame) -> dict[str, Any]:
    source = frame.source
    return {
        "transport_index": source.transport_index,
        "source_observed_frame_index": source.observed_frame_index,
        "block_id": source.block_id, "symbol_id": source.symbol_id,
        "K": source.source_symbols, "N": source.encoded_symbols,
        "data_size": source.data_size, "classification": source.classification,
        "boundary_offset": source.boundary_offset,
        "block_lane": source.block_id % INTERLEAVE_WINDOW if source.block_id is not None else None,
        "selection": source.selection,
    }


def _suffix_row(frame: AnalyzedFrame) -> dict[str, Any]:
    row = _frame_identity(frame)
    row.update({
        "actual_tail_length": frame.expected_original_length,
        "payload_prefix_matches_original": frame.prefix_matches_original,
        "payload_prefix_match_bytes": frame.prefix_match_bytes,
        "suffix_length": frame.suffix_metrics["length"] if frame.suffix_metrics else None,
        "suffix_all_zero": frame.suffix_metrics["byte_zero_fraction"] == 1.0 if frame.suffix_metrics else None,
        "suffix_all_ff": frame.suffix_metrics["byte_ff_fraction"] == 1.0 if frame.suffix_metrics else None,
        "suffix_sha256": frame.suffix_sha256,
    })
    if frame.suffix_metrics:
        row.update({f"suffix_{key}": value for key, value in frame.suffix_metrics.items()})
    return row


def _codeword_structure_rows(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for frame in frames:
        for group in _groups(frame):
            for codeword in frame.codeword_structure:
                buckets[(group, codeword["codeword_index"])].append(codeword)
    rows = []
    metric_names = (
        "data_entropy_bits_per_byte", "data_bit_one_fraction",
        "data_bit_transition_density_1d", "parity_entropy_bits_per_byte",
        "parity_bit_one_fraction", "parity_bit_transition_density_1d",
        "full_entropy_bits_per_byte", "full_bit_one_fraction",
        "full_bit_transition_density_1d",
    )
    for (group, codeword), values in sorted(buckets.items()):
        row: dict[str, Any] = {
            "group": group, "codeword_index": codeword, "frame_count": len(values),
        }
        for metric in metric_names:
            row.update({
                f"{metric}_{name}": value
                for name, value in _distribution([item[metric] for item in values]).items()
            })
        rows.append(row)
    return rows


def _representative_segment_rows(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    offset_zero = [frame for frame in frames if frame.source.boundary_offset == 0]
    representatives = []
    full = next((frame for frame in offset_zero if frame.source.data_size == 8 * 1024 * 1024), None)
    short = next((frame for frame in offset_zero if frame.source.data_size != 8 * 1024 * 1024), None)
    if full:
        representatives.append(("FULL_BLOCK_LAST_SOURCE", full))
    if short:
        representatives.append(("SHORT_BLOCK_LAST_SOURCE", short))
    rows = []
    for label, frame in representatives:
        _, _, segments = symbol_logical_segments(frame.packet, frame.source.transport_index)
        for row in codeword_segment_rows(segments, actual_source_length=frame.expected_original_length):
            rows.append({"representative": label, **row})
    return rows


def _raw_frame_rows(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        identity = _frame_identity(frame)
        rows.append({"channel": "source", **identity, **frame.source_raw})
        if frame.platform_raw:
            rows.append({
                "channel": "platform", **identity, **frame.platform_raw,
                "platform_rs_failed": frame.platform_rs_failed,
                "error_overlap_count": frame.error_overlap_count,
                "platform_only_error_count": frame.platform_only_error_count,
                "source_only_error_count": frame.source_only_error_count,
            })
    return rows


def _raw_codeword_rows(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    rows = []
    for frame in frames:
        identity = _frame_identity(frame)
        for row in frame.source_codeword_errors:
            rows.append({"channel": "source", **identity, **row, "rs_failed": False})
        if frame.platform_codeword_errors:
            failed = set(frame.platform_failed_codeword_indices)
            for row in frame.platform_codeword_errors:
                rows.append({
                    "channel": "platform", **identity, **row,
                    "rs_failed": row["codeword_index"] in failed,
                    "frame_rs_failed": frame.platform_rs_failed,
                })
    return rows


def _aggregate_raw_tiles(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int, int], Counter] = defaultdict(Counter)
    for frame in frames:
        for group in _groups(frame):
            for channel, tiles in (("source", frame.source_tile_errors), ("platform", frame.platform_tile_errors or [])):
                for tile in tiles:
                    bucket = buckets[(group, channel, tile["tile_row"], tile["tile_col"])]
                    bucket["frame_count"] += 1
                    bucket["total_bits"] += tile["total_bits"]
                    bucket["raw_bit_errors"] += tile["raw_bit_errors"]
    rows = []
    for (group, channel, tile_row, tile_col), values in sorted(buckets.items()):
        rows.append({
            "group": group, "channel": channel, "tile_row": tile_row, "tile_col": tile_col,
            **dict(values),
            "raw_bit_error_rate": values["raw_bit_errors"] / values["total_bits"] if values["total_bits"] else None,
        })
    return rows


def _aggregate_structure_tiles(frames: Iterable[AnalyzedFrame]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for frame in frames:
        for group in _groups(frame):
            for tile in frame.physical_tiles:
                bucket = buckets[(group, tile["tile_row"], tile["tile_col"])]
                for metric in ("one_fraction", "horizontal_transition_density", "vertical_transition_density"):
                    bucket[metric].append(tile[metric])
    rows = []
    for (group, tile_row, tile_col), metrics in sorted(buckets.items()):
        row: dict[str, Any] = {"group": group, "tile_row": tile_row, "tile_col": tile_col}
        for name, values in metrics.items():
            row.update({f"{name}_{key}": value for key, value in _distribution(values).items()})
        rows.append(row)
    return rows


def _failure_row(frame: AnalyzedFrame) -> dict[str, Any]:
    row = _frame_identity(frame)
    row.update({
        "platform_observed_frame_index": frame.platform_observed_frame_index,
        "failed_codeword_indices": json.dumps(list(frame.platform_failed_codeword_indices)),
        "platform_raw_bit_errors": frame.platform_raw["raw_bit_errors"] if frame.platform_raw else None,
        "platform_raw_bit_error_rate": frame.platform_raw["raw_bit_error_rate"] if frame.platform_raw else None,
        "platform_raw_byte_errors": frame.platform_raw["raw_byte_errors"] if frame.platform_raw else None,
        "per_codeword_raw_byte_errors": json.dumps([
            item["raw_byte_errors"] for item in (frame.platform_codeword_errors or [])
        ]),
        "per_codeword_raw_bit_errors": json.dumps([
            item["raw_bit_errors"] for item in (frame.platform_codeword_errors or [])
        ]),
        "payload_entropy": frame.payload_metrics["entropy_bits_per_byte"],
        "payload_bit_one_fraction": frame.payload_metrics["bit_one_fraction"],
        "payload_transition_density": frame.payload_metrics["bit_transition_density_1d"],
        "suffix_length": frame.suffix_metrics["length"] if frame.suffix_metrics else None,
        "suffix_sha256": frame.suffix_sha256,
    })
    return row


def _groups(frame: AnalyzedFrame) -> list[str]:
    groups = []
    offset = frame.source.boundary_offset
    if offset is not None and frame.source.selection == "BOUNDARY_WINDOW":
        groups.append("offset_0" if offset == 0 else f"offset_{offset:+d}")
    elif frame.source.classification == "SOURCE":
        groups.append("NON_BOUNDARY_SOURCE")
    if offset == 0:
        groups.extend([
            "OFFSET0_FAILURE" if frame.platform_rs_failed else "OFFSET0_SUCCESS",
            f"BLOCK_LANE_{frame.source.block_id % INTERLEAVE_WINDOW}_OFFSET0",
            "SHORT_BLOCK_OFFSET0" if frame.source.data_size != 8 * 1024 * 1024 else "FULL_BLOCK_OFFSET0",
        ])
    return groups


def _build_group_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    buckets: dict[str, list[AnalyzedFrame]] = defaultdict(list)
    for frame in frames:
        for group in _groups(frame):
            buckets[group].append(frame)
    return {group: _summarize_frames(rows) for group, rows in sorted(buckets.items())}


def _summarize_frames(frames: list[AnalyzedFrame]) -> dict[str, Any]:
    metrics = {
        "payload_entropy": [frame.payload_metrics["entropy_bits_per_byte"] for frame in frames],
        "payload_bit_one_fraction": [frame.payload_metrics["bit_one_fraction"] for frame in frames],
        "payload_transition_density": [frame.payload_metrics["bit_transition_density_1d"] for frame in frames],
        "payload_longest_zero_byte_run": [frame.payload_metrics["longest_zero_byte_run"] for frame in frames],
        "logical_entropy": [frame.logical_metrics["entropy_bits_per_byte"] for frame in frames],
        "physical_bit_one_fraction": [frame.physical_metrics["bit_one_fraction"] for frame in frames],
        "physical_horizontal_transition_density": [frame.physical_metrics["horizontal_transition_density"] for frame in frames],
        "physical_vertical_transition_density": [frame.physical_metrics["vertical_transition_density"] for frame in frames],
        "source_raw_bit_errors": [frame.source_raw["raw_bit_errors"] for frame in frames],
        "platform_raw_bit_errors": [frame.platform_raw["raw_bit_errors"] for frame in frames if frame.platform_raw],
    }
    return {
        "frame_count": len(frames),
        "platform_rs_failures": sum(frame.platform_rs_failed is True for frame in frames),
        **{name: _distribution(values) for name, values in metrics.items()},
    }


def _flatten_group_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for group, values in summary.items():
        row = {"group": group, "frame_count": values["frame_count"],
               "platform_rs_failures": values["platform_rs_failures"]}
        for metric, distribution in values.items():
            if isinstance(distribution, dict):
                row.update({f"{metric}_{name}": value for name, value in distribution.items()})
        rows.append(row)
    return rows


def _systematic_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    checked = [frame for frame in frames if frame.systematic_identity is not None]
    matched = sum(frame.systematic_identity is True for frame in checked)
    return {
        "checked_source_symbols": len(checked), "matching_source_symbols": matched,
        "match_rate": matched / len(checked) if checked else None,
        "systematic_identity_observed": bool(checked) and matched == len(checked),
    }


def _suffix_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    full = [frame for frame in frames if frame.source.boundary_offset == 0 and frame.source.data_size == 8 * 1024 * 1024]
    short = [frame for frame in frames if frame.source.boundary_offset == 0 and frame.source.data_size != 8 * 1024 * 1024]
    return {
        "full_blocks": _suffix_group(full),
        "short_blocks": _suffix_group(short),
    }


def _suffix_group(frames: list[AnalyzedFrame]) -> dict[str, Any]:
    fingerprints = Counter(frame.suffix_sha256 for frame in frames if frame.suffix_sha256)
    return {
        "count": len(frames),
        "actual_tail_lengths": dict(Counter(str(frame.expected_original_length) for frame in frames)),
        "suffix_lengths": dict(Counter(str(frame.suffix_metrics["length"]) for frame in frames if frame.suffix_metrics)),
        "all_zero_count": sum(frame.suffix_metrics and frame.suffix_metrics["byte_zero_fraction"] == 1.0 for frame in frames),
        "mean_entropy_bits_per_byte": _mean([
            frame.suffix_metrics["entropy_bits_per_byte"] for frame in frames if frame.suffix_metrics
        ]),
        "unique_suffix_fingerprints": len(fingerprints),
        "fingerprint_counts": dict(fingerprints),
    }


def _transport_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    first = next(iter(frames), None)
    if first is None:
        return {}
    _, logical, segments = symbol_logical_segments(first.packet, first.source.transport_index)
    return {
        "logical_bytes": len(logical),
        "segments": [{"name": item.name, "start": item.start, "end": item.end, "length": item.length}
                     for item in segments],
        "transport_padding": _distribution([
            frame.padding_metrics["entropy_bits_per_byte"] for frame in frames
        ]),
        "transport_padding_bit_one_fraction": _distribution([
            frame.padding_metrics["bit_one_fraction"] for frame in frames
        ]),
        "reserved_bit_one_fraction": _distribution([
            frame.reserved_metrics["bit_one_fraction"] for frame in frames
        ]),
    }


def _raw_channel_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    frame_list = list(frames)
    return {
        "source": _raw_summary(frame_list, "source"),
        "platform": _raw_summary([frame for frame in frame_list if frame.platform_raw], "platform"),
        "groups": {
            group: {
                "source": _raw_summary(rows, "source"),
                "platform": _raw_summary([row for row in rows if row.platform_raw], "platform"),
            }
            for group, rows in _group_frames(frame_list).items()
        },
        "codewords": _aggregate_codeword_errors(frame_list),
        "codeword_groups": _aggregate_codeword_error_groups(frame_list),
    }


def _raw_summary(frames: list[AnalyzedFrame], channel: str) -> dict[str, Any]:
    values = [
        (frame.source_raw if channel == "source" else frame.platform_raw)["raw_bit_errors"]
        for frame in frames
    ]
    total_bits = len(frames) * PHYSICAL_CONFIG.cells_per_frame
    return {
        "frame_count": len(frames), "raw_bit_errors": sum(values),
        "raw_bit_error_rate": sum(values) / total_bits if total_bits else None,
        "frames_with_errors": sum(value > 0 for value in values),
        "distribution": _distribution(values),
    }


def _aggregate_codeword_errors(frames: list[AnalyzedFrame]) -> dict[str, Any]:
    result = {}
    for channel in ("source", "platform"):
        rows = []
        for frame in frames:
            current = frame.source_codeword_errors if channel == "source" else frame.platform_codeword_errors
            if current:
                rows.extend(current)
        output = []
        for cw in range(RS_CODEWORDS):
            selected = [row for row in rows if row["codeword_index"] == cw]
            output.append({
                "codeword_index": cw, "frame_count": len(selected),
                "raw_byte_errors": sum(row["raw_byte_errors"] for row in selected),
                "raw_bit_errors": sum(row["raw_bit_errors"] for row in selected),
            })
        result[channel] = output
    return result


def _aggregate_codeword_error_groups(frames: list[AnalyzedFrame]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group, selected_frames in _group_frames(frames).items():
        output[group] = {}
        for channel in ("source", "platform"):
            rows = []
            for frame in selected_frames:
                current = frame.source_codeword_errors if channel == "source" else frame.platform_codeword_errors
                if current:
                    rows.extend(current)
            output[group][channel] = [
                {
                    "codeword_index": codeword,
                    "frame_count": len(selected),
                    "raw_byte_errors": sum(row["raw_byte_errors"] for row in selected),
                    "raw_bit_errors": sum(row["raw_bit_errors"] for row in selected),
                    "raw_bit_error_rate": (
                        sum(row["raw_bit_errors"] for row in selected) / (len(selected) * RS_N * 8)
                        if selected else None
                    ),
                }
                for codeword in range(RS_CODEWORDS)
                if (selected := [row for row in rows if row["codeword_index"] == codeword])
            ]
    return output


def _block_lane_summary(frames: Iterable[AnalyzedFrame]) -> dict[str, Any]:
    offset_zero = [frame for frame in frames if frame.source.boundary_offset == 0]
    return {
        str(lane): _summarize_frames([
            frame for frame in offset_zero if frame.source.block_id % INTERLEAVE_WINDOW == lane
        ])
        for lane in range(INTERLEAVE_WINDOW)
    }


def _group_frames(frames: list[AnalyzedFrame]) -> dict[str, list[AnalyzedFrame]]:
    result: dict[str, list[AnalyzedFrame]] = defaultdict(list)
    for frame in frames:
        for group in _groups(frame):
            result[group].append(frame)
    return dict(result)


def _original_validation(original_file: Path | None, metadata: dict[str, Any], frames: dict[int, AnalyzedFrame]) -> dict[str, Any]:
    if original_file is None:
        return {"available": False}
    size = original_file.stat().st_size
    digest = sha256_file(original_file)
    systematic = _systematic_summary(frames.values())
    return {
        "available": True, "path": str(original_file), "size": size,
        "size_matches_metadata": size == metadata["original_size"],
        "sha256": digest, "sha256_matches_metadata": digest == metadata["sha256"],
        **systematic,
    }


def _boundary_crosscheck(path: Path | None, failures: list[dict[str, Any]]) -> dict[str, Any]:
    actual = sorted(row["transport_index"] for row in failures if row["mapping_confident"])
    if path is None:
        return {"available": False, "actual_failure_transport_indices": actual}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = sorted(
        event["inferred_transport_index"] for event in payload.get("failure_diagnostics", {}).get("events", [])
        if event.get("inference_confident") and event.get("inferred_transport_index") is not None
    )
    return {
        "available": True, "path": str(path), "expected_failure_transport_indices": expected,
        "actual_failure_transport_indices": actual, "matches": expected == actual,
        "matching_count": len(set(expected) & set(actual)),
    }


def _conclusion_inputs(frames: Iterable[AnalyzedFrame], suffix: dict[str, Any]) -> dict[str, Any]:
    frame_list = list(frames)
    groups = _group_frames(frame_list)
    lane0 = groups.get("BLOCK_LANE_0_OFFSET0", [])
    other_lanes = [frame for lane in range(1, INTERLEAVE_WINDOW)
                   for frame in groups.get(f"BLOCK_LANE_{lane}_OFFSET0", [])]
    return {
        "full_suffix_all_zero": suffix["full_blocks"]["all_zero_count"] == suffix["full_blocks"]["count"],
        "full_suffix_identical": suffix["full_blocks"]["unique_suffix_fingerprints"] == 1,
        "short_suffix_all_zero": suffix["short_blocks"]["all_zero_count"] == suffix["short_blocks"]["count"],
        "offset0_payload_entropy_mean": _mean([frame.payload_metrics["entropy_bits_per_byte"] for frame in groups.get("offset_0", [])]),
        "non_boundary_payload_entropy_mean": _mean([frame.payload_metrics["entropy_bits_per_byte"] for frame in groups.get("NON_BOUNDARY_SOURCE", [])]),
        "lane0_payload_entropy_mean": _mean([frame.payload_metrics["entropy_bits_per_byte"] for frame in lane0]),
        "other_lane_payload_entropy_mean": _mean([frame.payload_metrics["entropy_bits_per_byte"] for frame in other_lanes]),
        "lane0_platform_raw_ber": _raw_summary([frame for frame in lane0 if frame.platform_raw], "platform")["raw_bit_error_rate"],
        "other_lane_platform_raw_ber": _raw_summary([frame for frame in other_lanes if frame.platform_raw], "platform")["raw_bit_error_rate"],
    }


def _descriptor_matches(source: SourceIndex, descriptor: PacketDiagnostic) -> bool:
    if descriptor.packet_type != source.packet_type:
        return False
    return source.packet_type != "SYMBOL" or (
        descriptor.block_id == source.block_id and descriptor.symbol_id == source.symbol_id
    )


def _open_video(video: Path):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if (width, height) != (WIDTH, HEIGHT) or not math.isclose(fps, FPS, abs_tol=0.01):
        capture.release()
        raise RuntimeError(f"video does not match fixed FVM profile: {width}x{height} {fps}fps")
    return capture, width, height, fps, frames


def _distribution(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": _percentile(values, 50), "p90": _percentile(values, 90),
        "p95": _percentile(values, 95), "max": max(values),
    }


def _percentile(values: list[int | float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: list[int | float]) -> float | None:
    return statistics.fmean(values) if values else None


def _longest_value_run(values: np.ndarray, target: int) -> int:
    longest = current = 0
    for value in values:
        if int(value) == target:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _longest_identical_run(values: np.ndarray) -> int:
    if not len(values):
        return 0
    longest = current = 1
    previous = int(values[0])
    for value in values[1:]:
        current = current + 1 if int(value) == previous else 1
        longest = max(longest, current)
        previous = int(value)
    return longest


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
