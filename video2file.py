"""Bounded-memory FOV v1 QR video decoder."""

from __future__ import annotations

import argparse
import base64
import os
import tempfile
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import zxingcpp
from pyraptorq import Decoder

from fov import (BlockLayout, MetaPacket, MetadataError, PacketError, SymbolPacket, file_id_from_sha256,
                 make_raptorq_engine, parse_packet, safe_filename, sha256_file, validate_metadata)

PREMETA_MAX_SYMBOLS = 4_096
PREMETA_MAX_BYTES = 16 * 1024 * 1024
PREMETA_MAX_FILE_IDS = 4
FAILED_FRAME_INDEX_SAMPLE_LIMIT = 512


def bit_is_set(bitmap: bytearray, index: int) -> bool:
    return bool(bitmap[index // 8] & (1 << (index % 8)))


def set_bit(bitmap: bytearray, index: int) -> None:
    bitmap[index // 8] |= 1 << (index % 8)


@dataclass
class PreMetaBuffer:
    max_symbols: int = PREMETA_MAX_SYMBOLS
    max_bytes: int = PREMETA_MAX_BYTES
    max_file_ids: int = PREMETA_MAX_FILE_IDS
    entries: OrderedDict[tuple[bytes, int, int], bytes] = field(default_factory=OrderedDict)
    frame_indices: dict[tuple[bytes, int, int], int | None] = field(default_factory=dict)
    total_bytes: int = 0
    evicted_symbols: int = 0
    evicted_bytes: int = 0

    def add(self, packet: SymbolPacket, frame_index: int | None = None) -> bool:
        key = (packet.file_id, packet.block_id, packet.symbol_id)
        if key in self.entries:
            return False
        if len(packet.payload) > self.max_bytes:
            self.evicted_symbols += 1
            self.evicted_bytes += len(packet.payload)
            return False
        self.entries[key] = packet.payload
        self.frame_indices[key] = frame_index
        self.total_bytes += len(packet.payload)
        self._evict_to_limits()
        return key in self.entries

    def take_for_file(self, file_id: bytes) -> list[SymbolPacket]:
        return [packet for packet, _ in self.take_for_file_with_frames(file_id)]

    def take_for_file_with_frames(self, file_id: bytes) -> list[tuple[SymbolPacket, int | None]]:
        selected: list[tuple[SymbolPacket, int | None]] = []
        for key, payload in self.entries.items():
            entry_file_id, block_id, symbol_id = key
            if entry_file_id == file_id:
                selected.append((
                    SymbolPacket(entry_file_id, block_id, symbol_id, payload),
                    self.frame_indices.get(key),
                ))
        self.clear()
        return selected

    def clear(self) -> None:
        self.entries.clear()
        self.frame_indices.clear()
        self.total_bytes = 0

    def _evict_to_limits(self) -> None:
        while (len(self.entries) > self.max_symbols or self.total_bytes > self.max_bytes
               or len({file_id for file_id, _, _ in self.entries}) > self.max_file_ids):
            key, payload = self.entries.popitem(last=False)
            self.frame_indices.pop(key, None)
            self.total_bytes -= len(payload)
            self.evicted_symbols += 1
            self.evicted_bytes += len(payload)


@dataclass
class BlockDecodeState:
    layout: BlockLayout
    decoder: Any
    seen_bitmap: bytearray
    received_unique: int = 0
    received_source: int = 0
    received_repair: int = 0


class StreamingDecodeSession:
    """Feeds symbols directly into native decoders and writes completed blocks immediately."""

    def __init__(self, file_id: bytes, metadata: dict[str, Any], output_dir: Path, engine: Any) -> None:
        self.file_id = file_id
        self.metadata = metadata
        self.layout = validate_metadata(metadata)
        self.engine = engine
        self.block_states: dict[int, BlockDecodeState] = {}
        self.decoded_bitmap = bytearray((len(self.layout) + 7) // 8)
        self.decoded_block_count = 0
        self.decoded_at_frames: dict[int, int | None] = {}
        self.stats: Counter[str] = Counter()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.target = output_path(output_dir, metadata["filename"])
        descriptor, temporary_name = tempfile.mkstemp(prefix="fov-recover-", suffix=".tmp", dir=output_dir)
        os.close(descriptor)
        self.temporary = Path(temporary_name)
        self._file = self.temporary.open("w+b")
        self._file.truncate(metadata["original_size"])

    def feed(self, packet: SymbolPacket, frame_index: int | None = None) -> None:
        if packet.file_id != self.file_id:
            self.stats["foreign_symbols"] += 1
            return
        if packet.block_id >= len(self.layout):
            self.stats["invalid_symbols"] += 1
            return
        layout = self.layout[packet.block_id]
        if packet.symbol_id >= layout.encoded_symbols or len(packet.payload) != self.metadata["symbol_size"]:
            self.stats["invalid_symbols"] += 1
            return
        if bit_is_set(self.decoded_bitmap, packet.block_id):
            self.stats["post_decode_symbols"] += 1
            return
        state = self.block_states.get(packet.block_id)
        if state is None:
            state = BlockDecodeState(
                layout,
                Decoder(layout.source_symbols, self.metadata["symbol_size"], layout.data_size, self.engine),
                bytearray((layout.encoded_symbols + 7) // 8),
            )
            self.block_states[packet.block_id] = state
        if bit_is_set(state.seen_bitmap, packet.symbol_id):
            self.stats["duplicate_symbols"] += 1
            return
        set_bit(state.seen_bitmap, packet.symbol_id)
        state.received_unique += 1
        if packet.symbol_id < layout.source_symbols:
            state.received_source += 1
            self.stats["source_symbols_received"] += 1
        else:
            state.received_repair += 1
            self.stats["repair_symbols_received"] += 1
        if not state.decoder.add_symbol(packet.symbol_id, packet.payload):
            raise RuntimeError(f"RaptorQ native add_symbol failed: block_id={packet.block_id}, symbol_id={packet.symbol_id}")
        self.stats["valid_symbols"] += 1
        if state.decoder.may_try_decode():
            result = state.decoder.try_decode()
            if result is not None:
                self._write_completed_block(state, result, frame_index)

    def failure_summary(self) -> str:
        failures = []
        for block in self.layout:
            if bit_is_set(self.decoded_bitmap, block.block_id):
                continue
            state = self.block_states.get(block.block_id)
            received = state.received_unique if state else 0
            failures.append(
                f"block_id={block.block_id}, K={block.source_symbols}, N={block.encoded_symbols}, "
                f"received_unique={received}, received/K={received / block.source_symbols:.2f}"
            )
        return "; ".join(failures)

    def finalize(self) -> Path:
        if self.decoded_block_count != len(self.layout):
            raise RuntimeError(f"RaptorQ decode failed: {self.failure_summary()}")
        self._file.flush()
        self._file.close()
        actual_sha256 = sha256_file(self.temporary)
        print(f"\nOriginal SHA256: {self.metadata['sha256']}\nRecovered SHA256: {actual_sha256}")
        if actual_sha256 != self.metadata["sha256"]:
            raise RuntimeError("SHA256 mismatch")
        self.temporary.replace(self.target)
        print(f"[OK] File fully recovered\nOutput: {self.target}")
        return self.target

    def cleanup(self) -> None:
        if not self._file.closed:
            self._file.close()
        self.temporary.unlink(missing_ok=True)

    def _write_completed_block(self, state: BlockDecodeState, result: bytes, frame_index: int | None) -> None:
        block = state.layout
        if len(result) < block.data_size:
            raise RuntimeError(f"RaptorQ returned a short block: block_id={block.block_id}")
        self._file.seek(block.block_id * self.metadata["block_size"])
        self._file.write(result[:block.data_size])
        self._file.flush()
        set_bit(self.decoded_bitmap, block.block_id)
        self.decoded_block_count += 1
        self.decoded_at_frames[block.block_id] = frame_index
        self.stats["decoded_blocks"] += 1
        decoded_at = str(frame_index) if frame_index is not None else "unknown"
        print(
            f"Block {block.block_id}:\n"
            f"  K: {block.source_symbols}\n"
            f"  N: {block.encoded_symbols}\n"
            f"  received unique: {state.received_unique}\n"
            f"  received source: {state.received_source}\n"
            f"  received repair: {state.received_repair}\n"
            f"  received/K: {state.received_unique / block.source_symbols:.2f}\n"
            f"  decoded at frame: {decoded_at}\n"
            f"  [OK]"
        )
        del self.block_states[block.block_id]


def decode_qr(frame) -> str | None:
    height, width = frame.shape[:2]
    # FOV centers every QR in the frame. Use the full short edge so decoding
    # automatically follows the video resolution instead of assuming 700 px.
    size = min(height, width)
    roi = frame[(height - size) // 2:(height + size) // 2, (width - size) // 2:(width + size) // 2]
    candidates = [roi, cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)]
    gray = candidates[-1]
    candidates.append(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    candidates.append(cv2.resize(candidates[-1], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    for candidate in candidates:
        try:
            results = zxingcpp.read_barcodes(candidate)
            if results:
                return results[0].text
        except Exception:
            continue
    return None


def output_path(output_dir: Path, filename: str) -> Path:
    safe_name = safe_filename(filename)
    candidate = output_dir / safe_name
    if not candidate.exists():
        return candidate
    candidate = output_dir / f"recovered_{safe_name}"
    attempt = 2
    while candidate.exists():
        candidate = output_dir / f"recovered_{attempt}_{safe_name}"
        attempt += 1
    return candidate


def _validate_meta(packet: MetaPacket) -> None:
    validate_metadata(packet.metadata)
    if file_id_from_sha256(packet.metadata["sha256"]) != packet.file_id:
        raise MetadataError("META file_id does not match SHA256")


def _format_frame_indices(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices) if indices else "-"


def _print_stats(
    stats: Counter[str],
    premeta: PreMetaBuffer,
    session: StreamingDecodeSession,
    failed_frame_indices: list[int],
    failed_frame_indices_truncated: bool,
) -> None:
    print(
        f"\nQR:\n"
        f"  decoded: {stats['qr_decoded']}\n"
        f"  failed: {stats['qr_failed']}\n"
        f"  failed frame indices (0-based): {_format_frame_indices(failed_frame_indices)}\n"
        f"  failed frame indices truncated: {'yes' if failed_frame_indices_truncated else 'no'}\n\n"
        f"Packets:\n"
        f"  valid META: {stats['valid_meta']}\n"
        f"  invalid META: {stats['invalid_meta']}\n"
        f"  CRC failed: {stats['crc_failed']}\n"
        f"  duplicate symbol: {session.stats['duplicate_symbols']}\n"
        f"  post-decode symbols: {session.stats['post_decode_symbols']}\n"
        f"  symbols before META: {stats['symbols_before_meta']}\n"
        f"  pre-META cached: {stats['premeta_cached']}\n"
        f"  pre-META evicted: {premeta.evicted_symbols} ({premeta.evicted_bytes} bytes)\n"
        f"  foreign symbols: {session.stats['foreign_symbols']}\n"
        f"  active block decoders: {len(session.block_states)}\n"
        f"  decoded blocks: {session.decoded_block_count}\n"
        f"  selected file_id: {session.file_id.hex()}"
    )


def decode(video_path: Path, output_dir: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    stats: Counter[str] = Counter()
    premeta = PreMetaBuffer()
    session: StreamingDecodeSession | None = None
    failed_frame_indices: list[int] = []
    failed_frame_indices_truncated = False
    next_frame_index = 0
    print(f"FOV v1 Decoder\n\nVideo: {video_path}\nTotal frames: {int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index = next_frame_index
            next_frame_index += 1
            text = decode_qr(frame)
            if not text:
                stats["qr_failed"] += 1
                if len(failed_frame_indices) < FAILED_FRAME_INDEX_SAMPLE_LIMIT:
                    failed_frame_indices.append(frame_index)
                else:
                    failed_frame_indices_truncated = True
                continue
            stats["qr_decoded"] += 1
            try:
                packet = parse_packet(base64.b64decode(text, validate=True))
            except PacketError as exc:
                stats["crc_failed" if "CRC32" in str(exc) else "invalid_packets"] += 1
                continue
            except (ValueError, UnicodeEncodeError):
                stats["invalid_packets"] += 1
                continue
            if isinstance(packet, SymbolPacket):
                if session is None:
                    if premeta.add(packet, frame_index=frame_index):
                        stats["premeta_cached"] += 1
                        stats["symbols_before_meta"] += 1
                    else:
                        stats["duplicate_symbols"] += 1
                else:
                    session.feed(packet, frame_index=frame_index)
                continue
            try:
                _validate_meta(packet)
            except (KeyError, TypeError, ValueError, MetadataError):
                stats["invalid_meta"] += 1
                continue
            if session is None:
                session = StreamingDecodeSession(packet.file_id, packet.metadata, output_dir, make_raptorq_engine())
                stats["valid_meta"] += 1
                for cached, cached_frame_index in premeta.take_for_file_with_frames(packet.file_id):
                    session.feed(cached, frame_index=cached_frame_index)
            elif packet.file_id != session.file_id:
                raise RuntimeError("multiple FOV files detected")
            elif packet.metadata != session.metadata:
                raise RuntimeError("conflicting META detected")
            else:
                stats["valid_meta"] += 1
    except BaseException:
        if session is not None:
            session.cleanup()
        raise
    finally:
        capture.release()
    if session is None:
        raise RuntimeError("no valid META packet found")
    try:
        _print_stats(stats, premeta, session, failed_frame_indices, failed_frame_indices_truncated)
        return session.finalize()
    except BaseException:
        session.cleanup()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="FOV v1 QR video decoder")
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    decode(args.video, args.output_dir)


if __name__ == "__main__":
    main()
