"""Encode a deterministic full-frame black/white FVM0 raw matrix probe."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

try:
    from fvm0_common import FVM0Config, bit_matrices, render_bits
except ImportError:  # Supports ``python -m scripts.fvm0_encode`` in tests.
    from scripts.fvm0_common import FVM0Config, bit_matrices, render_bits


def terminate_ffmpeg(process: subprocess.Popen[bytes]) -> None:
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


def default_manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def encode(output: Path, config: FVM0Config, crf: int, preset: str, manifest_path: Path) -> None:
    if crf < 0 or crf > 51:
        raise ValueError("crf must be between 0 and 51")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{config.width}x{config.height}",
               "-r", str(config.fps), "-i", "-", "-an", "-c:v", "libx264", "-crf", str(crf),
               "-preset", preset, "-pix_fmt", "yuv420p", str(output)]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found on PATH") from exc
    try:
        assert process.stdin is not None
        for index, bits in enumerate(bit_matrices(config), 1):
            process.stdin.write(render_bits(bits, config).tobytes())
            if index % 100 == 0 or index == config.frames:
                print(f"\rFVM0 encoded {index}/{config.frames} frames", end="", flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    except BaseException:
        terminate_ffmpeg(process)
        raise
    manifest = config.manifest()
    manifest["source_video"] = output.name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFVM0 video: {output}\nManifest: {manifest_path}\nRaw probe rate: {config.raw_bytes_per_second:g} equivalent byte/s ({config.raw_bits_per_second} bit/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="FVM0 full-frame raw matrix probe encoder")
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cell-size", type=int, default=8)
    parser.add_argument("--frames", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--crf", type=int, default=15)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    config = FVM0Config(args.width, args.height, args.fps, args.cell_size, args.frames, args.seed)
    encode(args.output, config, args.crf, args.preset, args.manifest or default_manifest_path(args.output))


if __name__ == "__main__":
    main()
