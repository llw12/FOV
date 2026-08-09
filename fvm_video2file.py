"""Decode fixed-profile FVM FILE MODE video without sidecar or source truth."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from fov import MetaPacket, MetadataError, PacketError, SymbolPacket, make_raptorq_engine, parse_packet, sha256_file
from fvm_diagnostics import FailureTracker, PacketDiagnostic
from fvm_file_common import (FPS, HEIGHT, PHYSICAL_CONFIG, WIDTH, decode_transport_physical,
                             matrix_to_physical)
from scripts.fvm0_common import decode_frame
from video2file import PreMetaBuffer, StreamingDecodeSession, _validate_meta

DIAGNOSTIC_DIRNAME = ".fvm"
RESULT_FILENAME = "fvm_decode_results.json"


@dataclass
class TransportIndexTracker:
    seen: set[int] = field(default_factory=set)
    previous: int | None = None
    duplicates: int = 0
    out_of_order: int = 0

    def observe(self, frame_index: int) -> None:
        if frame_index in self.seen:
            self.duplicates += 1
        else:
            self.seen.add(frame_index)
        if self.previous is not None and frame_index < self.previous:
            self.out_of_order += 1
        self.previous = frame_index

    @property
    def gaps(self) -> int:
        """Count observable leading and internal gaps; trailing loss is unknowable."""
        ordered = sorted(self.seen)
        if not ordered:
            return 0
        return ordered[0] + sum(
            current - previous - 1 for previous, current in zip(ordered, ordered[1:])
        )


class PacketRecoveryCoordinator:
    """Thin FVM adapter around the existing bounded FOV1 recovery classes."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        diagnostic_dir(output_dir).mkdir(parents=True, exist_ok=True)
        self.premeta = PreMetaBuffer()
        self.session: StreamingDecodeSession | None = None
        self.stats: Counter[str] = Counter()

    def feed(
        self, packet_bytes: bytes, frame_index: int | None = None
    ) -> MetaPacket | SymbolPacket | None:
        try:
            packet = parse_packet(packet_bytes)
        except PacketError as exc:
            self.stats["packet_crc_failed" if "CRC32" in str(exc) else "invalid_packets"] += 1
            return None
        if isinstance(packet, SymbolPacket):
            self.stats["valid_symbols"] += 1
            if self.session is None:
                if self.premeta.add(packet, frame_index=frame_index):
                    self.stats["premeta_cached"] += 1
                    self.stats["symbols_before_meta"] += 1
                else:
                    self.stats["duplicate_symbols"] += 1
            else:
                self.session.feed(packet, frame_index=frame_index)
            return packet
        assert isinstance(packet, MetaPacket)
        try:
            _validate_meta(packet)
        except (KeyError, TypeError, ValueError, MetadataError):
            self.stats["invalid_meta"] += 1
            return None
        if self.session is None:
            self.session = StreamingDecodeSession(
                packet.file_id,
                packet.metadata,
                self.output_dir,
                make_raptorq_engine(),
            )
            self.stats["valid_meta"] += 1
            for cached, cached_frame_index in self.premeta.take_for_file_with_frames(packet.file_id):
                self.session.feed(cached, frame_index=cached_frame_index)
        elif packet.file_id != self.session.file_id:
            raise RuntimeError("multiple FOV files detected")
        elif packet.metadata != self.session.metadata:
            raise RuntimeError("conflicting META detected")
        else:
            self.stats["valid_meta"] += 1
        return packet

    def finalize(self) -> Path:
        if self.session is None:
            raise RuntimeError("no valid META packet found")
        return self.session.finalize()

    def cleanup(self) -> None:
        if self.session is not None:
            self.session.cleanup()


def _transport_error_key(message: str) -> str:
    if "CRC32" in message:
        return "transport_crc_failures"
    if "packet length" in message:
        return "invalid_packet_length"
    return "invalid_header"


def _base_result(video_path: Path, width: int, height: int, fps: float) -> dict[str, Any]:
    return {
        "format": "FVM_FILE_FRAME_V1",
        "video": {
            "path": str(video_path),
            "resolution": f"{width}x{height}",
            "fps": fps,
            "observed_frames": 0,
        },
        "transport": {
            "rs_frames_success": 0,
            "rs_frames_failed": 0,
            "transport_crc_failures": 0,
            "invalid_header": 0,
            "invalid_packet_length": 0,
            "embedded_frame_gaps": 0,
            "duplicate_embedded_indices": 0,
            "out_of_order_count": 0,
        },
        "rs": {
            "codewords_success": 0,
            "codewords_failed": 0,
            "corrected_symbols": 0,
            "max_corrections_per_codeword": 0,
            "correction_histogram": {str(count): 0 for count in range(9)},
        },
        "packets": {},
        "raptorq": {},
        "file": {"exact": False},
    }


def diagnostic_dir(output_dir: Path) -> Path:
    return output_dir / DIAGNOSTIC_DIRNAME


def result_path(output_dir: Path) -> Path:
    return diagnostic_dir(output_dir) / RESULT_FILENAME


def _write_result(output_dir: Path, result: dict[str, Any]) -> None:
    path = result_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _record_index_diagnostics(result: dict[str, Any], indices: TransportIndexTracker) -> None:
    result["transport"]["embedded_frame_gaps"] = indices.gaps
    result["transport"]["duplicate_embedded_indices"] = indices.duplicates
    result["transport"]["out_of_order_count"] = indices.out_of_order


def _record_failure_diagnostics(result: dict[str, Any], failures: FailureTracker) -> None:
    result["rs_failures"] = failures.summary()


def _print_failure_summary(summary: dict[str, Any]) -> None:
    indices = summary["observed_indices"]
    if len(indices) <= 20 and not summary["events_truncated"]:
        index_text = ",".join(str(index) for index in indices) if indices else "-"
    else:
        index_text = ",".join(str(index) for index in indices[:20]) + "..."
    distances = summary["inter_failure_distance_stats"]
    distance_text = (
        "-"
        if distances["count"] == 0
        else f"{distances['min']}/{distances['mean']:.2f}/{distances['max']}"
    )
    print(
        f"RS failed frame indices (observed): {index_text}\n"
        f"Longest consecutive RS failure burst: {summary['longest_consecutive_burst']}\n"
        f"RS failure distance min/mean/max (observed frames): {distance_text}"
    )


def _packet_stats(coordinator: PacketRecoveryCoordinator) -> dict[str, int]:
    adapter = coordinator.stats
    session = coordinator.session.stats if coordinator.session is not None else Counter()
    return {
        "valid_meta": int(adapter["valid_meta"]),
        "invalid_meta": int(adapter["invalid_meta"]),
        "valid_symbols": int(adapter["valid_symbols"]),
        "packet_crc_failed": int(adapter["packet_crc_failed"]),
        "invalid_packets": int(adapter["invalid_packets"]),
        "duplicate_symbols": int(adapter["duplicate_symbols"] + session["duplicate_symbols"]),
        "symbols_before_meta": int(adapter["symbols_before_meta"]),
        "premeta_cached": int(adapter["premeta_cached"]),
        "foreign_symbols": int(session["foreign_symbols"]),
        "post_decode_symbols": int(session["post_decode_symbols"]),
    }


def decode(video_path: Path, output_dir: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    result = _base_result(video_path, width, height, actual_fps)
    coordinator = PacketRecoveryCoordinator(output_dir)
    indices = TransportIndexTracker()
    failures = FailureTracker()
    if (width, height) != (WIDTH, HEIGHT):
        capture.release()
        result["failure"] = "video resolution does not match fixed FVM profile"
        _record_failure_diagnostics(result, failures)
        _write_result(output_dir, result)
        raise RuntimeError(result["failure"])
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            observed_frame_index = result["video"]["observed_frames"]
            result["video"]["observed_frames"] += 1
            bits, _ = decode_frame(frame, PHYSICAL_CONFIG)
            physical_result = decode_transport_physical(matrix_to_physical(bits))
            rs = physical_result.rs
            result["rs"]["codewords_success"] += rs.codeword_successes
            result["rs"]["codewords_failed"] += rs.codeword_failures
            result["rs"]["corrected_symbols"] += rs.corrected_symbols
            result["rs"]["max_corrections_per_codeword"] = max(
                result["rs"]["max_corrections_per_codeword"], rs.max_corrections_per_codeword
            )
            for count, occurrences in rs.correction_histogram.items():
                result["rs"]["correction_histogram"][str(count)] += occurrences
            if rs.logical is None:
                result["transport"]["rs_frames_failed"] += 1
                failures.observe_failure(observed_frame_index, rs)
                continue
            result["transport"]["rs_frames_success"] += 1
            if physical_result.transport is None:
                failures.observe_non_failure_without_transport()
                key = _transport_error_key(physical_result.transport_error or "invalid header")
                result["transport"][key] += 1
                continue
            embedded_index = physical_result.transport.frame_index
            indices.observe(embedded_index)
            packet = coordinator.feed(physical_result.transport.packet, frame_index=embedded_index)
            failures.observe_success(embedded_index, PacketDiagnostic.from_packet(packet))
        _record_index_diagnostics(result, indices)
        _record_failure_diagnostics(result, failures)
        recovered = coordinator.finalize()
        assert coordinator.session is not None
        session = coordinator.session
        recovered_sha = sha256_file(recovered)
        result["packets"] = _packet_stats(coordinator)
        result["raptorq"] = {
            "blocks_total": len(session.layout),
            "blocks_decoded": session.decoded_block_count,
            "source_symbols_received": session.stats["source_symbols_received"],
            "repair_symbols_received": session.stats["repair_symbols_received"],
        }
        result["file"] = {
            "filename": recovered.name,
            "original_sha256": session.metadata["sha256"],
            "recovered_sha256": recovered_sha,
            "exact": recovered_sha == session.metadata["sha256"],
        }
        _write_result(output_dir, result)
        failure_summary = result["rs_failures"]
        print(
            f"FVM decode complete: {recovered}\nRS failed frames: {result['transport']['rs_frames_failed']}\n"
            f"Transport CRC failures: {result['transport']['transport_crc_failures']}\n"
            f"Packet CRC failures: {result['packets'].get('packet_crc_failed', 0)}"
        )
        _print_failure_summary(failure_summary)
        return recovered
    except BaseException as exc:
        coordinator.cleanup()
        _record_index_diagnostics(result, indices)
        _record_failure_diagnostics(result, failures)
        result["packets"] = _packet_stats(coordinator)
        if coordinator.session is not None:
            session = coordinator.session
            result["raptorq"] = {
                "blocks_total": len(session.layout),
                "blocks_decoded": session.decoded_block_count,
                "source_symbols_received": session.stats["source_symbols_received"],
                "repair_symbols_received": session.stats["repair_symbols_received"],
            }
        result["failure"] = str(exc)
        _write_result(output_dir, result)
        raise
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode an FVM 6 px H.264 video to a file")
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    decode(args.video, args.output_dir)


if __name__ == "__main__":
    main()
