"""Encode an FOV1 packet stream into fixed-profile FVM FILE MODE video."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from file2video import estimate_packet_count, packet_stream, terminate_ffmpeg
from fov import (INTERLEAVE_WINDOW, build_metadata, derive_block_layout, file_id_from_sha256,
                 make_raptorq_engine, sha256_file, symbol_packet_size)
from fvm_file_common import (FPS, HEIGHT, MAX_PACKET_BYTES, REPAIR_RATIO, SYMBOL_SIZE, WIDTH,
                             PHYSICAL_CONFIG, encode_transport_physical, physical_to_matrix)
from scripts.fvm0_common import render_bits


def encode(
    input_path: Path,
    output_path: Path,
    *,
    symbol_size: int = SYMBOL_SIZE,
    repair: float = REPAIR_RATIO,
    crf: int = 15,
    preset: str = "medium",
) -> dict[str, Any]:
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")
    original_size = input_path.stat().st_size
    if original_size <= 0:
        raise ValueError("empty files are not supported by pyraptorq")
    max_symbol_packet_bytes = symbol_packet_size(symbol_size)
    if max_symbol_packet_bytes > MAX_PACKET_BYTES:
        raise ValueError(
            f"FVM symbol_size {symbol_size} produces a {max_symbol_packet_bytes}-byte FOV SYMBOL packet, "
            f"exceeding transport capacity {MAX_PACKET_BYTES} bytes"
        )
    digest = sha256_file(input_path)
    metadata = build_metadata(input_path.name, original_size, digest, symbol_size, repair)
    layout = derive_block_layout(original_size, metadata["block_size"], symbol_size, repair)
    total_frames, total_symbols, meta_frames = estimate_packet_count(layout)
    source_symbols = sum(block.source_symbols for block in layout)
    repair_symbols = total_symbols - source_symbols
    duration = total_frames / FPS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", str(crf),
        "-preset", preset, "-pix_fmt", "yuv420p", str(output_path),
    ]
    print(
        f"FVM FILE MODE v1 Encoder\n\nFile: {input_path.name}\nSize: {original_size} bytes\n"
        f"SHA256: {digest}\nSymbol size: {symbol_size}\nRepair ratio: {repair:.0%}\n"
        f"Interleave window: {INTERLEAVE_WINDOW}\nSource symbols: {source_symbols}\n"
        f"Repair symbols: {repair_symbols}\nMETA frames: {meta_frames}\nTotal frames: {total_frames}\n"
        f"Video: {WIDTH}x{HEIGHT} {FPS}fps, {duration:.3f}s"
    )
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    try:
        assert process.stdin is not None
        stream = packet_stream(input_path, metadata, file_id_from_sha256(digest), make_raptorq_engine())
        emitted_frames = 0
        for frame_index, packet in enumerate(stream):
            if len(packet) > MAX_PACKET_BYTES:
                raise ValueError(
                    f"FOV packet at frame {frame_index} is {len(packet)} bytes; maximum is {MAX_PACKET_BYTES}"
                )
            physical = encode_transport_physical(packet, frame_index)
            process.stdin.write(render_bits(physical_to_matrix(physical), PHYSICAL_CONFIG).tobytes())
            emitted_frames += 1
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == total_frames:
                print(f"\rEncoded {frame_index + 1}/{total_frames} frames", end="", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
        if emitted_frames != total_frames:
            raise RuntimeError(f"packet count changed during encoding: expected {total_frames}, got {emitted_frames}")
    except BaseException:
        terminate_ffmpeg(process)
        raise
    result = {
        "input_size": original_size,
        "sha256": digest,
        "blocks": len(layout),
        "source_symbols": source_symbols,
        "repair_symbols": repair_symbols,
        "symbol_frames": total_symbols,
        "meta_frames": meta_frames,
        "total_frames": total_frames,
        "duration_seconds": duration,
        "effective_source_bytes_per_second": original_size / duration,
    }
    print(
        f"\nEncoding complete: {output_path}\n"
        f"Effective source-file throughput: {result['effective_source_bytes_per_second']:.2f} bytes/s"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a file as an FVM 6 px H.264 video")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--symbol-size", type=int, default=SYMBOL_SIZE)
    parser.add_argument("--repair", type=float, default=REPAIR_RATIO)
    parser.add_argument("--crf", type=int, default=15)
    parser.add_argument("--preset", default="medium")
    args = parser.parse_args()
    encode(
        args.input,
        args.output,
        symbol_size=args.symbol_size,
        repair=args.repair,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
