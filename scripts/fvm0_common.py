"""Shared deterministic matrix generation and measurements for the FVM0 probe."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterator

import cv2
import numpy as np


FORMAT = "FVM0_RAW"
LEGACY_PRNG = "PCG64"
PRNG = "PCG64_RAW_LSB_V1"
SUPPORTED_PRNGS = {LEGACY_PRNG, PRNG}


@dataclass(frozen=True)
class FVM0Config:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    cell_size: int = 8
    frames: int = 12_000
    seed: int = 20_260_809

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.cell_size or self.height % self.cell_size:
            raise ValueError("width and height must be divisible by cell_size")

    @property
    def rows(self) -> int:
        return self.height // self.cell_size

    @property
    def cols(self) -> int:
        return self.width // self.cell_size

    @property
    def bits_per_frame(self) -> int:
        return self.rows * self.cols

    @property
    def equivalent_bytes_per_frame(self) -> float:
        return self.bits_per_frame / 8.0

    @property
    def raw_bits_per_second(self) -> int:
        return self.bits_per_frame * self.fps

    @property
    def raw_bytes_per_second(self) -> float:
        return self.raw_bits_per_second / 8.0

    def manifest(self) -> dict[str, Any]:
        return {"format": FORMAT, "width": self.width, "height": self.height, "fps": self.fps,
                "cell_size": self.cell_size, "rows": self.rows, "cols": self.cols,
                "bits_per_cell": 1, "bits_per_frame": self.bits_per_frame,
                "equivalent_bytes_per_frame": self.equivalent_bytes_per_frame, "frames": self.frames,
                "seed": self.seed, "prng": PRNG}

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "FVM0Config":
        required = {"format", "width", "height", "fps", "cell_size", "rows", "cols", "bits_per_cell",
                    "bits_per_frame", "frames", "seed", "prng"}
        if not isinstance(manifest, dict) or not required <= manifest.keys():
            raise ValueError("manifest is missing FVM0 fields")
        if manifest["format"] != FORMAT or manifest["prng"] not in SUPPORTED_PRNGS or manifest["bits_per_cell"] != 1:
            raise ValueError("unsupported FVM0 manifest")
        config = cls(*(manifest[name] for name in ("width", "height", "fps", "cell_size", "frames", "seed")))
        if any(manifest[name] != getattr(config, name) for name in ("rows", "cols", "bits_per_frame")):
            raise ValueError("manifest geometry does not match its parameters")
        equivalent = manifest.get("equivalent_bytes_per_frame", manifest.get("bytes_per_frame"))
        if equivalent is None or float(equivalent) != config.equivalent_bytes_per_frame:
            raise ValueError("manifest equivalent bytes per frame does not match geometry")
        return config


def bit_matrices(config: FVM0Config, prng: str = PRNG) -> Iterator[np.ndarray]:
    """Yield the protocol PRBS one frame at a time, without caching the stream."""
    if prng == LEGACY_PRNG:
        rng = np.random.Generator(np.random.PCG64(config.seed))
        for _ in range(config.frames):
            yield rng.integers(0, 2, size=(config.rows, config.cols), dtype=np.uint8)
        return
    if prng != PRNG:
        raise ValueError(f"unsupported PRNG: {prng}")
    generator = np.random.PCG64(config.seed)
    shifts = np.arange(64, dtype=np.uint64)
    words = math.ceil(config.bits_per_frame / 64)
    for _ in range(config.frames):
        raw = generator.random_raw(words).astype(np.uint64, copy=False)
        bits = ((raw[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
        yield bits[:config.bits_per_frame].reshape(config.rows, config.cols)


def render_bits(bits: np.ndarray, config: FVM0Config) -> np.ndarray:
    if bits.shape != (config.rows, config.cols) or bits.dtype != np.uint8:
        raise ValueError("bits must be a uint8 matrix matching FVM0 geometry")
    pixels = np.repeat(np.repeat(bits, config.cell_size, axis=0), config.cell_size, axis=1) * 255
    return np.repeat(pixels[:, :, None], 3, axis=2)


def decode_frame(frame: np.ndarray, config: FVM0Config) -> tuple[np.ndarray, np.ndarray]:
    if frame.shape[:2] != (config.height, config.width):
        raise ValueError("video resolution does not match manifest")
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luma = grayscale.reshape(config.rows, config.cell_size, config.cols, config.cell_size).mean(axis=(1, 3))
    return (luma >= 128).astype(np.uint8), luma


@dataclass
class LumaStats:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = 255.0
    maximum: float = 0.0
    histogram: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.histogram is None:
            self.histogram = np.zeros(256, dtype=np.int64)

    def add(self, values: np.ndarray) -> None:
        if not values.size:
            return
        flattened = values.astype(np.float64, copy=False).ravel()
        self.count += flattened.size
        self.total += float(flattened.sum())
        self.total_squared += float(np.square(flattened).sum())
        self.minimum = min(self.minimum, float(flattened.min()))
        self.maximum = max(self.maximum, float(flattened.max()))
        bins = np.clip(np.rint(flattened), 0, 255).astype(np.intp)
        self.histogram += np.bincount(bins, minlength=256)

    def summary(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None,
                    "p1": None, "p5": None, "p50": None, "p95": None, "p99": None}
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        return {"count": self.count, "mean": mean, "std": math.sqrt(variance), "min": self.minimum,
                "max": self.maximum, **{f"p{percent}": self._percentile(percent) for percent in (1, 5, 50, 95, 99)}}

    def _percentile(self, percent: int) -> float:
        target = math.ceil(self.count * percent / 100)
        return float(np.searchsorted(np.cumsum(self.histogram), target, side="left"))


@dataclass
class Measurements:
    config: FVM0Config
    compared_frames: int = 0
    compared_bits: int = 0
    correct_bits: int = 0
    bit_errors: int = 0
    zero_to_one: int = 0
    one_to_zero: int = 0
    frames_with_errors: int = 0
    worst_frame_index: int | None = None
    worst_frame_bit_errors: int = 0
    current_error_burst: int = 0
    longest_error_burst: int = 0
    spatial_errors: np.ndarray | None = None
    zero_luma: LumaStats | None = None
    one_luma: LumaStats | None = None

    def __post_init__(self) -> None:
        if self.spatial_errors is None:
            self.spatial_errors = np.zeros((self.config.rows, self.config.cols), dtype=np.uint64)
        self.zero_luma = self.zero_luma or LumaStats()
        self.one_luma = self.one_luma or LumaStats()

    def add_frame(self, frame_index: int, expected: np.ndarray, actual: np.ndarray, luma: np.ndarray) -> dict[str, float | int]:
        errors = expected != actual
        error_count = int(errors.sum())
        zero_to_one = int(((expected == 0) & (actual == 1)).sum())
        one_to_zero = int(((expected == 1) & (actual == 0)).sum())
        self.compared_frames += 1
        self.compared_bits += expected.size
        self.bit_errors += error_count
        self.correct_bits += expected.size - error_count
        self.zero_to_one += zero_to_one
        self.one_to_zero += one_to_zero
        self.spatial_errors += errors
        self.zero_luma.add(luma[expected == 0])
        self.one_luma.add(luma[expected == 1])
        if error_count:
            self.frames_with_errors += 1
            self.current_error_burst += 1
            self.longest_error_burst = max(self.longest_error_burst, self.current_error_burst)
            if error_count > self.worst_frame_bit_errors:
                self.worst_frame_index, self.worst_frame_bit_errors = frame_index, error_count
        else:
            self.current_error_burst = 0
        return {"frame_index": frame_index, "bit_errors": error_count, "zero_to_one": zero_to_one,
                "one_to_zero": one_to_zero, "ber": error_count / expected.size}

    def result(self, actual_frames: int, *, actual_width: int | None = None, actual_height: int | None = None, actual_fps: float | None = None) -> dict[str, Any]:
        return {"video": {"expected_resolution": f"{self.config.width}x{self.config.height}", "actual_resolution": f"{actual_width or self.config.width}x{actual_height or self.config.height}", "expected_fps": self.config.fps, "actual_fps": float(self.config.fps if actual_fps is None else actual_fps),
                           "expected_frames": self.config.frames, "actual_frames": actual_frames,
                           "compared_frames": self.compared_frames},
                "matrix": {"cell_size": self.config.cell_size, "rows": self.config.rows, "columns": self.config.cols,
                           "bits_per_frame": self.config.bits_per_frame, "equivalent_bytes_per_frame": self.config.equivalent_bytes_per_frame,
                           "raw_bitrate_bps": self.config.raw_bits_per_second, "raw_equivalent_bytes_per_second": self.config.raw_bytes_per_second},
                "bits": {"total_compared_bits": self.compared_bits, "correct_bits": self.correct_bits,
                         "bit_errors": self.bit_errors, "ber": self.bit_errors / self.compared_bits if self.compared_bits else None,
                         "zero_to_one": self.zero_to_one, "one_to_zero": self.one_to_zero},
                "frames": {"frames_with_errors": self.frames_with_errors,
                           "fer": self.frames_with_errors / self.compared_frames if self.compared_frames else None,
                           "worst_frame_index": self.worst_frame_index, "worst_frame_bit_errors": self.worst_frame_bit_errors,
                           "longest_consecutive_error_frame_burst": self.longest_error_burst},
                "luminance": {"expected_zero": self.zero_luma.summary(), "expected_one": self.one_luma.summary()}}


def top_error_cells(error_map: np.ndarray, frame_count: int, limit: int = 20) -> list[dict[str, float | int]]:
    flat_indices = np.argsort(error_map.ravel())[::-1]
    cells = []
    for index in flat_indices[:limit]:
        count = int(error_map.ravel()[index])
        if not count:
            break
        row, col = np.unravel_index(index, error_map.shape)
        cells.append({"row": int(row), "col": int(col), "error_count": count, "error_rate": count / frame_count})
    return cells
