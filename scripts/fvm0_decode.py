"""Measure FVM0 raw matrix BER from a video using its local sidecar manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from fvm0_common import FVM0Config, Measurements, bit_matrices, decode_frame, top_error_cells
except ImportError:
    from scripts.fvm0_common import FVM0Config, Measurements, bit_matrices, decode_frame, top_error_cells


def render_heatmap(error_map: np.ndarray, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(error_map, aspect="auto", origin="upper", cmap="magma")
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    axis.set_title("FVM0 cell error count")
    figure.colorbar(image, ax=axis, label="error count")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render_luma_histogram(measurements: Measurements, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    values = np.arange(256)
    axis.step(values, measurements.zero_luma.histogram, where="mid", label="expected 0")
    axis.step(values, measurements.one_luma.histogram, where="mid", label="expected 1")
    axis.set_xlabel("mean cell luminance")
    axis.set_ylabel("cell count")
    axis.set_title("FVM0 observed cell luminance")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def decode(video_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = FVM0Config.from_manifest(manifest)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (actual_width, actual_height) != (config.width, config.height):
        capture.release()
        raise RuntimeError(f"video resolution {actual_width}x{actual_height} does not match manifest {config.width}x{config.height}")
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements = Measurements(config)
    expected = bit_matrices(config)
    actual_frames = 0
    csv_path = output_dir / "fvm0_frames.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["frame_index", "bit_errors", "zero_to_one", "one_to_zero", "ber"])
            writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = actual_frames
                actual_frames += 1
                if frame_index >= config.frames:
                    continue
                actual, luma = decode_frame(frame, config)
                writer.writerow(measurements.add_frame(frame_index, next(expected), actual, luma))
                if actual_frames % 100 == 0:
                    print(f"\rFVM0 compared {measurements.compared_frames}/{config.frames} frames", end="", flush=True)
    finally:
        capture.release()
    result = measurements.result(actual_frames)
    result["parameters"] = config.manifest()
    result["top_error_cells"] = top_error_cells(measurements.spatial_errors, measurements.compared_frames)
    result["warning"] = None if actual_frames == config.frames else (
        "FVM0 has no synchronization metadata; frame insertion/deletion makes index-based BER after the synchronization loss unreliable."
    )
    result_path = output_dir / "fvm0_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    render_heatmap(measurements.spatial_errors, output_dir / "fvm0_error_heatmap.png")
    render_luma_histogram(measurements, output_dir / "fvm0_luma_histogram.png")
    bits = result["bits"]
    frames = result["frames"]
    print(f"\nVideo: {actual_width}x{actual_height}, {actual_fps:.3f} fps\nExpected frames: {config.frames}\nActual frames: {actual_frames}\nCompared frames: {measurements.compared_frames}")
    print(f"Matrix: {config.rows}x{config.cols}, cell {config.cell_size}, {config.bits_per_frame} bits/frame, raw {config.raw_bytes_per_second} byte/s")
    print(f"Bits: total {bits['total_compared_bits']}, correct {bits['correct_bits']}, errors {bits['bit_errors']}, BER {bits['ber']}")
    if bits["bit_errors"] == 0:
        print("observed BER: 0")
    print(f"Frames: errors {frames['frames_with_errors']}, FER {frames['fer']}, worst {frames['worst_frame_index']} ({frames['worst_frame_bit_errors']}), longest burst {frames['longest_consecutive_error_frame_burst']}")
    print(f"0 -> 1: {bits['zero_to_one']}; 1 -> 0: {bits['one_to_zero']}")
    print(f"Luma expected 0: {result['luminance']['expected_zero']}")
    print(f"Luma expected 1: {result['luminance']['expected_one']}")
    if result["top_error_cells"]:
        print(f"Top error cells: {result['top_error_cells']}")
    if result["warning"]:
        print(f"WARNING: {result['warning']}")
    print(f"Results: {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="FVM0 raw matrix BER decoder")
    parser.add_argument("video", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("fvm0-result"))
    args = parser.parse_args()
    decode(args.video, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
