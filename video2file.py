"""Decode an FOV v1 QR-frame video back to its original file."""

from __future__ import annotations

import argparse
import base64
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import zxingcpp

from fov import (MetaPacket, MetadataError, PacketError, SymbolPacket, decode_block,
                 derive_block_layout, file_id_from_sha256, make_raptorq_engine, parse_packet, sha256_file,
                 safe_filename, validate_metadata)

ROI_SIZE = 700


@dataclass
class PacketCollection:
    metadata_by_file_id: dict[bytes, dict[str, Any]] = field(default_factory=dict)
    conflicting_file_ids: set[bytes] = field(default_factory=set)
    symbols_by_file: dict[bytes, dict[int, dict[int, bytes]]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(dict)))
    stats: Counter[str] = field(default_factory=Counter)


def decode_qr(frame) -> str | None:
    height, width = frame.shape[:2]
    size = min(ROI_SIZE, height, width)
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


def collect_packet(collection: PacketCollection, packet: MetaPacket | SymbolPacket) -> None:
    """Collect CRC-valid packets without assuming META arrives first."""
    if isinstance(packet, MetaPacket):
        try:
            validate_metadata(packet.metadata)
            if file_id_from_sha256(packet.metadata["sha256"]) != packet.file_id:
                raise MetadataError("META file_id does not match SHA256")
        except (KeyError, TypeError, ValueError, MetadataError):
            collection.stats["invalid_meta"] += 1
            return
        previous = collection.metadata_by_file_id.get(packet.file_id)
        if previous is None:
            collection.metadata_by_file_id[packet.file_id] = packet.metadata
            collection.stats["valid_meta"] += 1
        elif previous != packet.metadata:
            collection.conflicting_file_ids.add(packet.file_id)
            collection.stats["conflicting_meta"] += 1
        else:
            collection.stats["valid_meta"] += 1
        return
    file_symbols = collection.symbols_by_file[packet.file_id][packet.block_id]
    if packet.symbol_id in file_symbols:
        collection.stats["duplicate_symbols"] += 1
        return
    file_symbols[packet.symbol_id] = packet.payload
    collection.stats["valid_symbols"] += 1
    if packet.file_id not in collection.metadata_by_file_id:
        collection.stats["symbols_before_meta"] += 1


def select_metadata(collection: PacketCollection) -> tuple[bytes, dict[str, Any]]:
    candidates = [(file_id, metadata) for file_id, metadata in collection.metadata_by_file_id.items()
                  if file_id not in collection.conflicting_file_ids]
    if not candidates:
        if collection.conflicting_file_ids:
            raise RuntimeError("conflicting META detected")
        raise RuntimeError("no valid META packet found")
    if len(candidates) != 1:
        raise RuntimeError("multiple FOV files detected")
    return candidates[0]


def output_path(output_dir: Path, filename: str) -> Path:
    target = output_dir / safe_filename(filename)
    return target if not target.exists() else output_dir / f"recovered_{safe_name}"


def decode(video_path: Path, output_dir: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    collection = PacketCollection()
    collection.stats["total_frames"] = 0
    print(f"FOV v1 Decoder\n\nVideo: {video_path}\nTotal frames: {int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        collection.stats["total_frames"] += 1
        text = decode_qr(frame)
        if not text:
            collection.stats["qr_failed"] += 1
            continue
        collection.stats["qr_decoded"] += 1
        try:
            packet = parse_packet(base64.b64decode(text, validate=True))
        except PacketError as exc:
            collection.stats["crc_failed" if "CRC32" in str(exc) else "invalid_packets"] += 1
            continue
        except (ValueError, UnicodeEncodeError):
            collection.stats["invalid_packets"] += 1
            continue
        collect_packet(collection, packet)
    capture.release()
    file_id, metadata = select_metadata(collection)
    layout = validate_metadata(metadata)
    selected_symbols = collection.symbols_by_file.get(file_id, {})
    print(f"\nQR:\n  decoded: {collection.stats['qr_decoded']}\n  failed: {collection.stats['qr_failed']}\n\nPackets:\n  valid META: {collection.stats['valid_meta']}\n  invalid META: {collection.stats['invalid_meta']}\n  conflicting META: {collection.stats['conflicting_meta']}\n  valid SYMBOL: {collection.stats['valid_symbols']}\n  CRC failed: {collection.stats['crc_failed']}\n  duplicate symbol: {collection.stats['duplicate_symbols']}\n  symbol before META: {collection.stats['symbols_before_meta']}\n  selected file_id: {file_id.hex()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_path(output_dir, metadata["filename"])
    descriptor, temporary_name = tempfile.mkstemp(prefix="fov-recover-", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        engine = make_raptorq_engine()
        with temporary.open("wb") as recovered:
            for block in layout:
                received = selected_symbols.get(block.block_id, {})
                usable = {symbol_id: payload for symbol_id, payload in received.items()
                          if symbol_id < block.encoded_symbols and len(payload) == metadata["symbol_size"]}
                ratio = len(usable) / block.source_symbols
                print(f"\nBlock {block.block_id}:\n  K: {block.source_symbols}\n  N: {block.encoded_symbols}\n  received unique: {len(usable)}\n  received / K: {ratio:.2f}\n  decoding...")
                result = decode_block(block.data_size, metadata["symbol_size"], usable, engine)
                if result is None:
                    raise RuntimeError(f"RaptorQ decode failed: block_id={block.block_id}, K={block.source_symbols}, received={len(usable)}, received/K={ratio:.2f}")
                recovered.write(result)
                print(f"Block {block.block_id}: [OK]")
        actual_sha256 = sha256_file(temporary)
        print(f"\nOriginal SHA256: {metadata['sha256']}\nRecovered SHA256: {actual_sha256}")
        if actual_sha256 != metadata["sha256"]:
            raise RuntimeError("SHA256 mismatch")
        temporary.replace(target)
        print(f"[OK] File fully recovered\nOutput: {target}")
        return target
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="FOV v1 QR video decoder")
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    decode(args.video, args.output_dir)


if __name__ == "__main__":
    main()
