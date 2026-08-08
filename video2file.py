"""Decode an FOV v1 QR-frame video back to its original file."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import zxingcpp

from fov import MetaPacket, PacketError, SymbolPacket, decode_block, file_id_from_sha256, parse_packet, sha256_bytes

ROI_SIZE = 700


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


def validate_metadata(metadata: dict[str, Any]) -> None:
    required = {"version", "filename", "original_size", "sha256", "encoded_size", "compression", "symbol_size", "repair_ratio", "block_size", "block_count", "blocks"}
    if not required <= metadata.keys() or metadata["version"] != 1 or metadata["compression"] != "none":
        raise RuntimeError("unsupported or incomplete FOV metadata")
    if (not isinstance(metadata["blocks"], list) or len(metadata["blocks"]) != metadata["block_count"]
            or any(not isinstance(entry, list) or len(entry) != 3 or any(not isinstance(value, int) or value <= 0 for value in entry)
                   for entry in metadata["blocks"])):
        raise RuntimeError("invalid metadata blocks")
    if metadata["original_size"] <= 0 or metadata["symbol_size"] <= 0:
        raise RuntimeError("invalid metadata sizes")


def output_path(output_dir: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    target = output_dir / safe_name
    return target if not target.exists() else output_dir / f"recovered_{safe_name}"


def decode(video_path: Path, output_dir: Path) -> Path:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    stats = defaultdict(int)
    symbols: dict[int, dict[int, bytes]] = defaultdict(dict)
    metadata: dict[str, Any] | None = None
    meta_file_id: bytes | None = None
    print(f"FOV v1 Decoder\n\n视频: {video_path}\n总帧数: {int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        stats["total_frames"] += 1
        text = decode_qr(frame)
        if not text:
            stats["qr_failed_frames"] += 1
            continue
        stats["qr_decoded_frames"] += 1
        try:
            packet = parse_packet(base64.b64decode(text, validate=True))
        except PacketError as exc:
            if "CRC32" in str(exc):
                stats["crc_failed_packets"] += 1
            continue
        except (ValueError, UnicodeEncodeError):
            continue
        if isinstance(packet, MetaPacket):
            try:
                validate_metadata(packet.metadata)
                if file_id_from_sha256(packet.metadata["sha256"]) != packet.file_id:
                    raise RuntimeError("META file_id does not match SHA256")
            except (KeyError, TypeError, ValueError, RuntimeError):
                continue
            if metadata is None:
                metadata, meta_file_id = packet.metadata, packet.file_id
            elif packet.metadata != metadata or packet.file_id != meta_file_id:
                continue
            stats["meta_packets"] += 1
            continue
        if metadata is None or packet.file_id != meta_file_id:
            continue
        expected_blocks = metadata["block_count"]
        if packet.block_id >= expected_blocks:
            continue
        _, _, encoded_symbols = metadata["blocks"][packet.block_id]
        if packet.symbol_id >= encoded_symbols or len(packet.payload) != metadata["symbol_size"]:
            continue
        block_symbols = symbols[packet.block_id]
        if packet.symbol_id in block_symbols:
            stats["duplicate_symbols"] += 1
        else:
            block_symbols[packet.symbol_id] = packet.payload
            stats["symbol_packets"] += 1
    capture.release()
    if metadata is None:
        raise RuntimeError("no valid META packet found")
    print(f"\nQR:\n  成功识别: {stats['qr_decoded_frames']}\n  失败: {stats['qr_failed_frames']}\n\nPackets:\n  META: {stats['meta_packets']}\n  SYMBOL: {stats['symbol_packets']}\n  CRC错误: {stats['crc_failed_packets']}\n  重复symbol: {stats['duplicate_symbols']}")
    recovered = []
    for block_id, entry in enumerate(metadata["blocks"]):
        data_size, source_symbols, _ = entry
        block_symbols = symbols[block_id]
        print(f"\nBlock {block_id}:\n  K: {source_symbols}\n  received unique: {len(block_symbols)}\n\n尝试 RaptorQ 恢复...")
        result = decode_block(data_size, metadata["symbol_size"], block_symbols)
        if result is None:
            ratio = len(block_symbols) / source_symbols
            raise RuntimeError(f"RaptorQ decode failed: block_id={block_id}, K={source_symbols}, received={len(block_symbols)}, received/K={ratio:.2f}")
        print(f"Block {block_id}: [OK]")
        recovered.append(result)
    data = b"".join(recovered)[:metadata["original_size"]]
    actual_sha256 = sha256_bytes(data)
    print(f"\n原始 SHA256: {metadata['sha256']}\n恢复 SHA256: {actual_sha256}")
    if actual_sha256 != metadata["sha256"]:
        raise RuntimeError("SHA256 mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_path(output_dir, metadata["filename"])
    target.write_bytes(data)
    print(f"[OK] 文件完整恢复\n输出: {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="FOV v1 QR video decoder")
    parser.add_argument("video", type=Path)
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path("."))
    args = parser.parse_args()
    decode(args.video, args.output_dir)


if __name__ == "__main__":
    main()
