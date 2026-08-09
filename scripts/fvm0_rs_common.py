"""Protocol and Reed-Solomon helpers for the independent FVM0_RS_PROBE."""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

FORMAT = "FVM0_RS_PROBE"
VERSION = 1
MAGIC = b"F6RS"
PAYLOAD_GENERATOR = "SHAKE256_FVM0_RS_V1"
BIT_ORDER = "big"
INTERLEAVE = "byte_column_v1"
THRESHOLD = 128
RS_N = 255
RS_K = 239
RS_PARITY = 16
RS_CODEWORDS = 28
LOGICAL_BYTES = RS_K * RS_CODEWORDS
CODED_RS_BYTES = RS_N * RS_CODEWORDS
HEADER_BYTES = 9
CRC_BYTES = 4


@dataclass(frozen=True)
class FVM0RSConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    cell_size: int = 6
    frames: int = 1200
    seed: int = 20260809

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps", "cell_size", "frames", "seed"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.cell_size or self.height % self.cell_size:
            raise ValueError("width and height must be divisible by cell_size")
        if self.cells_per_frame % 8:
            raise ValueError("cells_per_frame must be divisible by 8")
        if self.physical_bytes_per_frame < CODED_RS_BYTES:
            raise ValueError("geometry cannot hold 28 RS(255,239) codewords")
        if self.seed >= 1 << 64 or self.frames > 1 << 32:
            raise ValueError("seed/frame count exceed the fixed unsigned protocol encoding")

    @property
    def rows(self) -> int: return self.height // self.cell_size
    @property
    def cols(self) -> int: return self.width // self.cell_size
    @property
    def cells_per_frame(self) -> int: return self.rows * self.cols
    @property
    def physical_bytes_per_frame(self) -> int: return self.cells_per_frame // 8
    @property
    def reserved_bytes_per_frame(self) -> int: return self.physical_bytes_per_frame - CODED_RS_BYTES

    def manifest(self, crf: int, preset: str, source_video: str) -> dict[str, Any]:
        return {"format": FORMAT, "version": VERSION, "width": self.width, "height": self.height,
                "fps": self.fps, "cell_size": self.cell_size, "rows": self.rows, "cols": self.cols,
                "cells_per_frame": self.cells_per_frame, "physical_bytes_per_frame": self.physical_bytes_per_frame,
                "frames": self.frames, "seed": self.seed, "rs_n": RS_N, "rs_k": RS_K,
                "rs_parity": RS_PARITY, "rs_codewords_per_frame": RS_CODEWORDS,
                "logical_bytes_per_frame": LOGICAL_BYTES, "coded_rs_bytes_per_frame": CODED_RS_BYTES,
                "reserved_bytes_per_frame": self.reserved_bytes_per_frame, "bit_order": BIT_ORDER,
                "interleave": INTERLEAVE, "threshold": THRESHOLD, "payload_generator": PAYLOAD_GENERATOR,
                "crf": crf, "preset": preset, "source_video": source_video}

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "FVM0RSConfig":
        try:
            if manifest["format"] != FORMAT or manifest["version"] != VERSION:
                raise ValueError("unsupported FVM0 RS manifest")
            config = cls(manifest["width"], manifest["height"], manifest["fps"],
                         manifest["cell_size"], manifest["frames"], manifest["seed"])
            expected = {"rows": config.rows, "cols": config.cols, "cells_per_frame": config.cells_per_frame,
                        "physical_bytes_per_frame": config.physical_bytes_per_frame, "rs_n": RS_N,
                        "rs_k": RS_K, "rs_parity": RS_PARITY, "rs_codewords_per_frame": RS_CODEWORDS,
                        "logical_bytes_per_frame": LOGICAL_BYTES, "coded_rs_bytes_per_frame": CODED_RS_BYTES,
                        "reserved_bytes_per_frame": config.reserved_bytes_per_frame,
                        "bit_order": BIT_ORDER, "interleave": INTERLEAVE, "threshold": THRESHOLD,
                        "payload_generator": PAYLOAD_GENERATOR}
            if any(manifest[key] != value for key, value in expected.items()):
                raise ValueError("manifest parameters are inconsistent with FVM0_RS_PROBE")
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete FVM0 RS manifest") from exc
        return config


def _domain(prefix: bytes, seed: int, frame_index: int) -> bytes:
    return prefix + seed.to_bytes(8, "big") + frame_index.to_bytes(4, "big")


def logical_frame(seed: int, frame_index: int) -> bytes:
    if not 0 <= seed < 1 << 64 or not 0 <= frame_index < 1 << 32:
        raise ValueError("seed and frame_index must fit uint64/uint32")
    prefix = MAGIC + bytes([VERSION]) + frame_index.to_bytes(4, "big")
    payload = hashlib.shake_256(_domain(b"FVM0_RS_PROBE|", seed, frame_index)).digest(LOGICAL_BYTES - HEADER_BYTES - CRC_BYTES)
    body = prefix + payload
    return body + zlib.crc32(body).to_bytes(4, "big")


def reserved_bytes(seed: int, frame_index: int, length: int) -> bytes:
    if length < 0: raise ValueError("reserved length must be nonnegative")
    return hashlib.shake_256(_domain(b"FVM0_RS_RESERVED|", seed, frame_index)).digest(length)


_RS = RSCodec(RS_PARITY, nsize=RS_N)


def encode_logical(logical: bytes) -> np.ndarray:
    if len(logical) != LOGICAL_BYTES: raise ValueError(f"logical frame must be {LOGICAL_BYTES} bytes")
    chunks = []
    for offset in range(0, LOGICAL_BYTES, RS_K):
        encoded = bytes(_RS.encode(logical[offset:offset + RS_K]))
        if len(encoded) != RS_N: raise RuntimeError("RS encoder returned an unexpected codeword length")
        chunks.append(np.frombuffer(encoded, dtype=np.uint8))
    return np.stack(chunks)


def interleave(codewords: np.ndarray) -> bytes:
    if codewords.shape != (RS_CODEWORDS, RS_N): raise ValueError("codewords must have shape (28,255)")
    return codewords.T.reshape(-1).tobytes()


def deinterleave(data: bytes) -> np.ndarray:
    if len(data) != CODED_RS_BYTES: raise ValueError(f"interleaved RS data must be {CODED_RS_BYTES} bytes")
    return np.frombuffer(data, dtype=np.uint8).reshape(RS_N, RS_CODEWORDS).T.copy()


def physical_frame(config: FVM0RSConfig, frame_index: int) -> bytes:
    coded = interleave(encode_logical(logical_frame(config.seed, frame_index)))
    return coded + reserved_bytes(config.seed, frame_index, config.reserved_bytes_per_frame)


def bytes_to_matrix(data: bytes, config: FVM0RSConfig) -> np.ndarray:
    if len(data) != config.physical_bytes_per_frame: raise ValueError("physical frame byte length is invalid")
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder=BIT_ORDER).reshape(config.rows, config.cols)


def matrix_to_bytes(bits: np.ndarray, config: FVM0RSConfig) -> bytes:
    if bits.shape != (config.rows, config.cols): raise ValueError("matrix geometry is invalid")
    return np.packbits(bits.astype(np.uint8, copy=False).reshape(-1), bitorder=BIT_ORDER).tobytes()


@dataclass
class RSFrameRecovery:
    logical: bytes | None
    codeword_successes: int
    codeword_failures: int
    decoder_reported_corrections: int | None
    header_valid: bool
    crc_valid: bool
    payload_exact: bool
    embedded_frame_index: int | None


@dataclass
class RSDecodeResult:
    """Production-safe RS result that does not depend on expected payload bytes."""

    logical: bytes | None
    codeword_successes: int
    codeword_failures: int
    corrected_symbols: int
    max_corrections_per_codeword: int
    correction_histogram: dict[int, int]


def decode_rs_codewords(codewords: np.ndarray) -> RSDecodeResult:
    if codewords.shape != (RS_CODEWORDS, RS_N):
        raise ValueError("codewords must have shape (28,255)")
    chunks: list[bytes] = []
    correction_counts: list[int] = []
    failures = 0
    for codeword in codewords:
        try:
            decoded, _, errata = _RS.decode(codeword.tobytes())
            decoded_bytes = bytes(decoded)
            if len(decoded_bytes) != RS_K:
                raise ReedSolomonError("RS decoder returned an unexpected message length")
            chunks.append(decoded_bytes)
            correction_counts.append(len(errata))
        except ReedSolomonError:
            failures += 1
    histogram = {count: correction_counts.count(count) for count in range(9)}
    return RSDecodeResult(
        logical=b"".join(chunks) if not failures else None,
        codeword_successes=RS_CODEWORDS - failures,
        codeword_failures=failures,
        corrected_symbols=sum(correction_counts),
        max_corrections_per_codeword=max(correction_counts, default=0),
        correction_histogram=histogram,
    )


def decode_codewords(codewords: np.ndarray, expected: bytes, expected_frame_index: int) -> RSFrameRecovery:
    result = decode_rs_codewords(codewords)
    if result.logical is None:
        return RSFrameRecovery(
            None,
            result.codeword_successes,
            result.codeword_failures,
            result.corrected_symbols,
            False,
            False,
            False,
            None,
        )
    logical = result.logical
    embedded = int.from_bytes(logical[5:9], "big") if len(logical) >= HEADER_BYTES else None
    header = logical[:4] == MAGIC and logical[4:5] == bytes([VERSION]) and embedded == expected_frame_index
    crc = len(logical) == LOGICAL_BYTES and zlib.crc32(logical[:-4]) == int.from_bytes(logical[-4:], "big")
    return RSFrameRecovery(
        logical,
        result.codeword_successes,
        result.codeword_failures,
        result.corrected_symbols,
        header,
        crc,
        logical == expected,
        embedded,
    )
