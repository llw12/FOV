from __future__ import annotations

import numpy as np
import pytest

from scripts.fvm0_common import FVM0Config, Measurements, bit_matrices, decode_frame, render_bits


def test_default_geometry() -> None:
    config = FVM0Config()
    assert (config.rows, config.cols, config.bits_per_frame, config.bytes_per_frame) == (135, 240, 32_400, 4_050)


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
