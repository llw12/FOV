from __future__ import annotations

import numpy as np
import pytest
import io
import csv
import json

from scripts.fvm0_common import LEGACY_PRNG, FVM0Config, Measurements, bit_matrices, decode_frame, render_bits
from scripts.fvm0_encode import encode
from scripts.fvm0_common import aggregate_gop_phase
from scripts.fvm0_gop_analyze import write_analysis


def test_default_geometry() -> None:
    config = FVM0Config()
    assert (config.rows, config.cols, config.bits_per_frame, config.equivalent_bytes_per_frame) == (135, 240, 32_400, 4_050.0)


@pytest.mark.parametrize("width,height", [(1919, 1080), (1920, 1079)])
def test_invalid_geometry(width: int, height: int) -> None:
    with pytest.raises(ValueError, match="divisible"):
        FVM0Config(width=width, height=height, cell_size=8)


def test_deterministic_prbs() -> None:
    config = FVM0Config(width=16, height=8, cell_size=4, frames=3, seed=7)
    first = list(bit_matrices(config))
    second = list(bit_matrices(config))
    different = list(bit_matrices(FVM0Config(width=16, height=8, cell_size=4, frames=3, seed=8)))
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert any(not np.array_equal(left, right) for left, right in zip(first, different, strict=True))


def test_raw_lsb_golden_vector_and_frame_boundaries() -> None:
    config = FVM0Config(width=16, height=8, cell_size=4, frames=2, seed=7)
    frames = list(bit_matrices(config))
    raw = np.random.PCG64(7).random_raw(2)
    expected = [((int(word) >> bit) & 1) for word in raw for bit in range(config.bits_per_frame)]
    assert np.array_equal(np.concatenate([frame.ravel() for frame in frames]), np.array(expected, dtype=np.uint8))


def test_legacy_prng_and_unknown_prng_rejection() -> None:
    config = FVM0Config(width=8, height=4, cell_size=2, frames=1, seed=7)
    legacy = next(bit_matrices(config, LEGACY_PRNG))
    expected = np.random.Generator(np.random.PCG64(7)).integers(0, 2, size=(2, 4), dtype=np.uint8)
    assert np.array_equal(legacy, expected)
    with pytest.raises(ValueError, match="unsupported PRNG"):
        next(bit_matrices(config, "unknown"))
    manifest = config.manifest()
    manifest["prng"] = "unknown"
    with pytest.raises(ValueError):
        FVM0Config.from_manifest(manifest)


def test_non_byte_aligned_raw_rate_and_actual_video_metadata() -> None:
    config = FVM0Config(width=6, height=2, cell_size=2, fps=3, frames=1, seed=1)
    assert config.bits_per_frame == 3
    assert config.equivalent_bytes_per_frame == 0.375
    assert config.raw_bits_per_second == 9
    assert config.raw_bytes_per_second == 1.125
    result = Measurements(config).result(1, actual_width=6, actual_height=2, actual_fps=2.5)
    assert result["video"]["expected_fps"] == 3
    assert result["video"]["actual_fps"] == 2.5
    assert result["matrix"]["raw_bitrate_bps"] == 9


def test_encoder_creates_output_parents_before_launch(tmp_path, monkeypatch) -> None:
    class Process:
        def __init__(self): self.stdin = io.BytesIO(); self.returncode = 0
        def wait(self): return 0
        def poll(self): return 0
    def fake_popen(command, stdin):
        assert (tmp_path / "nested").is_dir()
        return Process()
    monkeypatch.setattr("scripts.fvm0_encode.subprocess.Popen", fake_popen)
    config = FVM0Config(width=4, height=2, cell_size=2, frames=1, seed=1)
    encode(tmp_path / "nested" / "video.mp4", config, 15, "medium", tmp_path / "nested" / "video.manifest.json")


def test_gop_phase_aggregation_and_standalone_analyzer(tmp_path) -> None:
    records = [{"frame_index": i, "bit_errors": value, "zero_to_one": value - 1 if value else 0, "one_to_zero": 1 if value else 0, "ber": 0} for i, value in enumerate([10, 1, 0, 20, 2, 3, 4, 5, 6, 7])]
    analysis = aggregate_gop_phase(records, 100, 3)
    phase0 = analysis["phases"][0]
    assert (phase0["frame_count"], phase0["total_bits"], phase0["bit_errors"], phase0["ber"], phase0["fer"]) == (4, 400, 41, 0.1025, 1.0)
    assert (phase0["min_bit_errors_per_frame"], phase0["max_bit_errors_per_frame"], phase0["median_bit_errors_per_frame"]) == (4, 20, 8.5)
    with pytest.raises(ValueError): aggregate_gop_phase([], 100, 0)
    with (tmp_path / "fvm0_frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)
    (tmp_path / "fvm0_results.json").write_text(json.dumps({"matrix": {"bits_per_frame": 100}}), encoding="utf-8")
    assert write_analysis(tmp_path, 4)["phases"][0]["frame_count"] == 3
    assert all((tmp_path / name).exists() for name in ("fvm0_gop_phase.csv", "fvm0_gop_phase.json", "fvm0_gop_phase_ber.png"))


def test_frame_rendering_and_direct_decode_are_bit_exact() -> None:
    config = FVM0Config(width=8, height=4, cell_size=2, frames=1, seed=1)
    bits = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    frame = render_bits(bits, config)
    assert np.all(frame[0:2, 0:2] == 0)
    assert np.all(frame[0:2, 2:4] == 255)
    decoded, luma = decode_frame(frame, config)
    assert np.array_equal(decoded, bits)
    assert np.array_equal(luma, bits * 255)


def test_threshold_behavior() -> None:
    config = FVM0Config(width=4, height=2, cell_size=2, frames=1, seed=1)
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, :2] = 127
    frame[:, 2:] = 128
    decoded, _ = decode_frame(frame, config)
    assert np.array_equal(decoded, np.array([[0, 1]], dtype=np.uint8))


def test_ber_metrics_and_spatial_map() -> None:
    config = FVM0Config(width=4, height=4, cell_size=2, frames=2, seed=1)
    expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    actual = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    measurements = Measurements(config)
    record = measurements.add_frame(0, expected, actual, expected.astype(float) * 255)
    result = measurements.result(actual_frames=1)
    assert record["bit_errors"] == 2
    assert result["bits"]["bit_errors"] == 2
    assert result["bits"]["zero_to_one"] == 1
    assert result["bits"]["one_to_zero"] == 1
    assert result["frames"]["frames_with_errors"] == 1
    assert result["frames"]["fer"] == 1.0
    assert measurements.spatial_errors.tolist() == [[1, 0], [0, 1]]
