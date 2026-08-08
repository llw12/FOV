"""Encode a file into an FOV v1 QR-frame MP4 without loading it all at once."""

from __future__ import annotations

import argparse
import base64
import itertools
import subprocess
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from fov import (FPS, HEIGHT, INTERLEAVE_WINDOW, META_INTERVAL, META_REPEAT, REPAIR_RATIO, SYMBOL_SIZE, WIDTH,
                 BlockLayout, build_metadata, create_raptor_encoder, derive_block_layout, encode_meta, encode_symbol,
                 file_id_from_sha256, iter_file_blocks, make_raptorq_engine, sha256_file)


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


def packet_stream(input_path: Path, metadata: dict, file_id: bytes, engine) -> Iterator[bytes]:
    """Yield packets in bounded temporal-interleave windows, never all symbols."""
    layout = derive_block_layout(metadata["original_size"], metadata["block_size"], metadata["symbol_size"], metadata["repair_ratio"])
    meta = encode_meta(file_id, metadata)
    yield from itertools.repeat(meta, META_REPEAT)
    blocks = iter_file_blocks(input_path, metadata["block_size"])
    sent_symbols = 0
    while window := list(itertools.islice(blocks, INTERLEAVE_WINDOW)):
        active = []
        for block_id, data in window:
            expected = layout[block_id]
            if len(data) != expected.data_size:
                raise RuntimeError("input file changed while encoding")
            active.append((expected, create_raptor_encoder(data, metadata["symbol_size"], expected, engine)))
        max_symbols = max(item[0].encoded_symbols for item in active)
        for symbol_id in range(max_symbols):
            for block, encoder in active:
                if symbol_id < block.encoded_symbols:
                    yield encode_symbol(file_id, block.block_id, symbol_id, encoder.gen_symbol(symbol_id))
                    sent_symbols += 1
                    if sent_symbols % META_INTERVAL == 0:
                        yield meta
        # Dropping this window releases its source data and native encoders.
        del active
    yield from itertools.repeat(meta, META_REPEAT)


def estimate_packet_count(layout: list[BlockLayout]) -> tuple[int, int, int]:
    """Return (total packets, symbols, META packets) using packet_stream's insertion rule."""
    symbols = sum(block.encoded_symbols for block in layout)
    inserted_meta = sum(1 for symbol_number in range(1, symbols + 1) if symbol_number % META_INTERVAL == 0)
    meta_packets = META_REPEAT * 2 + inserted_meta
    return symbols + meta_packets, symbols, meta_packets


def terminate_ffmpeg(process: subprocess.Popen[bytes]) -> None:
    """Close and stop FFmpeg on any encoder exception without leaving a child behind."""
    if process.stdin and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def encode(input_path: Path, output_path: Path, *, symbol_size: int, repair: float, fps: int, width: int, height: int) -> None:
    if min(fps, width, height) <= 0:
        raise ValueError("video dimensions and fps must be positive")
    original_size = input_path.stat().st_size
    if original_size <= 0:
        raise ValueError("empty files are not supported by pyraptorq")
    digest = sha256_file(input_path)
    metadata = build_metadata(input_path.name, original_size, digest, symbol_size, repair)
    layout = derive_block_layout(original_size, metadata["block_size"], symbol_size, repair)
    total_packets, total_symbols, meta_packets = estimate_packet_count(layout)
    file_id = file_id_from_sha256(digest)
    print(f"FOV v1 Encoder\n\nFile: {input_path.name}\nSize: {original_size} bytes\nSHA256: {digest}")
    print(f"\nBlock count: {len(layout)}\nBlock size: {metadata['block_size']}\nSymbol size: {symbol_size}\nRepair ratio: {repair:.0%}\nInterleave window: {INTERLEAVE_WINDOW}")
    for block in layout:
        print(f"\nBlock {block.block_id}:\n  data size: {block.data_size}\n  source symbols K: {block.source_symbols}\n  encoded symbols N: {block.encoded_symbols}")
    print(f"\nSymbols: {total_symbols}\nMETA packets: {meta_packets}\nVideo:\n  {width}x{height}\n  {fps} fps\n  total frames: {total_packets}\n  duration: {total_packets / fps:.2f}s")
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", str(output_path)]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    try:
        assert process.stdin is not None
        engine = make_raptorq_engine()
        for index, packet in enumerate(packet_stream(input_path, metadata, file_id, engine), 1):
            process.stdin.write(qr_frame(packet, width, height).tobytes())
            if index % 100 == 0:
                print(f"\r{index}/{total_packets} frames", end="", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    except BaseException:
        terminate_ffmpeg(process)
        raise
    print(f"\nEncoding complete: {output_path}")


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
