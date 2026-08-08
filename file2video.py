"""Encode a file into an FOV v1 QR-frame MP4."""

from __future__ import annotations

import argparse
import base64
import subprocess
from pathlib import Path

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from fov import (FPS, HEIGHT, META_INTERVAL, META_REPEAT, REPAIR_RATIO, SYMBOL_SIZE, WIDTH, build_metadata,
                 encode_meta, encode_symbol, encoded_symbol_count, encode_block, file_id_from_sha256,
                 source_symbol_count, split_blocks)


def qr_frame(packet: bytes, width: int, height: int) -> np.ndarray:
    qr = qrcode.QRCode(version=None, error_correction=ERROR_CORRECT_M, box_size=8, border=4)
    qr.add_data(base64.b64encode(packet).decode("ascii"))
    qr.make(fit=True)
    image = cv2.cvtColor(np.asarray(qr.make_image(fill_color="black", back_color="white").convert("RGB")), cv2.COLOR_RGB2BGR)
    if image.shape[0] > height - 40 or image.shape[1] > width - 40:
        raise RuntimeError(f"QR code is {image.shape[1]}x{image.shape[0]}; reduce --symbol-size")
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    y, x = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
    frame[y:y + image.shape[0], x:x + image.shape[1]] = image
    return frame


def packet_stream(data: bytes, metadata: dict, file_id: bytes):
    meta = encode_meta(file_id, metadata)
    yield from (meta for _ in range(META_REPEAT))
    blocks = split_blocks(data, metadata["block_size"])
    symbols_by_block = []
    for entry, block in zip(metadata["blocks"], blocks, strict=True):
        symbols_by_block.append(encode_block(block, metadata["symbol_size"], entry[2]))
    max_symbols = max(map(len, symbols_by_block))
    sent = 0
    for symbol_id in range(max_symbols):
        for block_id, symbols in enumerate(symbols_by_block):
            if symbol_id < len(symbols):
                yield encode_symbol(file_id, block_id, symbol_id, symbols[symbol_id])
                sent += 1
                if sent % META_INTERVAL == 0:
                    yield meta
    yield from (meta for _ in range(META_REPEAT))


def encode(input_path: Path, output_path: Path, *, symbol_size: int, repair: float, fps: int, width: int, height: int) -> None:
    if symbol_size <= 0 or not 0 <= repair <= 5 or min(fps, width, height) <= 0:
        raise ValueError("invalid video or FEC parameters")
    data = input_path.read_bytes()
    metadata = build_metadata(input_path.name, data, symbol_size, repair)
    file_id = file_id_from_sha256(metadata["sha256"])
    total_symbols = sum(entry[2] for entry in metadata["blocks"])
    total_frames = total_symbols + META_REPEAT * 2 + total_symbols // META_INTERVAL
    print("FOV v1 Encoder\n\n文件:", input_path.name, "\n大小:", len(data), "bytes\nSHA256:", metadata["sha256"])
    print(f"\nBlock count: {metadata['block_count']}\nSymbol size: {symbol_size}\nRepair ratio: {repair:.0%}")
    for block_id, entry in enumerate(metadata["blocks"]):
        print(f"\nBlock {block_id}:\n  data size: {entry[0]}\n  source symbols K: {entry[1]}\n  encoded symbols N: {entry[2]}")
    print(f"\nVideo:\n  {width}x{height}\n  {fps} fps\n  total frames: {total_frames}\n  duration: {total_frames / fps:.2f}s")
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", str(output_path)]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    assert process.stdin is not None
    try:
        for index, packet in enumerate(packet_stream(data, metadata, file_id), 1):
            process.stdin.write(qr_frame(packet, width, height).tobytes())
            if index % 100 == 0:
                print(f"\r{index}/{total_frames} frames", end="", flush=True)
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr}")
    except BrokenPipeError as exc:
        raise RuntimeError("ffmpeg stopped accepting frames") from exc
    print(f"\n编码完成: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FOV v1 file to QR video encoder")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--symbol-size", type=int, default=SYMBOL_SIZE)
    parser.add_argument("--repair", type=float, default=REPAIR_RATIO)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    encode(args.input, args.output, symbol_size=args.symbol_size, repair=args.repair, fps=args.fps, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
