"""Encode deterministic FVM0_RS_PROBE frames into H.264 video."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from fvm0_common import render_bits
    from fvm0_encode import terminate_ffmpeg
    from fvm0_rs_common import FVM0RSConfig, bytes_to_matrix, physical_frame
except ImportError:
    from scripts.fvm0_common import render_bits
    from scripts.fvm0_encode import terminate_ffmpeg
    from scripts.fvm0_rs_common import FVM0RSConfig, bytes_to_matrix, physical_frame


def encode(output: Path, config: FVM0RSConfig, crf: int, preset: str, manifest_path: Path) -> None:
    if not 0 <= crf <= 51: raise ValueError("crf must be between 0 and 51")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
               f"{config.width}x{config.height}", "-r", str(config.fps), "-i", "-", "-an",
               "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", str(output)]
    try: process = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc: raise RuntimeError("ffmpeg not found on PATH") from exc
    try:
        assert process.stdin is not None
        for frame_index in range(config.frames):
            bits = bytes_to_matrix(physical_frame(config, frame_index), config)
            process.stdin.write(render_bits(bits, config).tobytes())
            if (frame_index + 1) % 100 == 0 or frame_index + 1 == config.frames:
                print(f"\rFVM0-RS encoded {frame_index + 1}/{config.frames} frames", end="", flush=True)
        process.stdin.close()
        if process.wait() != 0: raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    except BaseException:
        terminate_ffmpeg(process); raise
    manifest_path.write_text(json.dumps(config.manifest(crf, preset, output.name), indent=2), encoding="utf-8")
    print(f"\nVideo: {output}\nManifest: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode an FVM0 6px RS probe")
    parser.add_argument("output", type=Path); parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080); parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cell-size", type=int, default=6); parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260809); parser.add_argument("--crf", type=int, default=15)
    parser.add_argument("--preset", default="medium"); parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(); config = FVM0RSConfig(args.width, args.height, args.fps, args.cell_size, args.frames, args.seed)
    encode(args.output, config, args.crf, args.preset, args.manifest or args.output.with_suffix(".manifest.json"))


if __name__ == "__main__": main()
