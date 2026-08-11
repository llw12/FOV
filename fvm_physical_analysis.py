"""Physical-matrix mapping and raw BER helpers for offline FVM diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from fvm_file_common import PHYSICAL_CONFIG, physical_to_matrix
from scripts.fvm0_rs_common import (BIT_ORDER, CODED_RS_BYTES, RS_CODEWORDS, RS_N,
                                    deinterleave)

TILE_ROWS = 20
TILE_COLS = 20
_BIT_COUNTS = np.array([int(index).bit_count() for index in range(256)], dtype=np.uint8)


def physical_metrics(physical: bytes) -> dict[str, Any]:
    matrix = physical_to_matrix(physical)
    horizontal = matrix[:, 1:] != matrix[:, :-1]
    vertical = matrix[1:, :] != matrix[:-1, :]
    return {
        "bit_one_fraction": float(matrix.mean()),
        "horizontal_transition_density": float(horizontal.mean()),
        "vertical_transition_density": float(vertical.mean()),
        "longest_equal_bit_horizontal_run": _longest_matrix_run(matrix, axis=1),
        "longest_equal_bit_vertical_run": _longest_matrix_run(matrix, axis=0),
    }


def tile_structure_rows(physical: bytes) -> list[dict[str, Any]]:
    matrix = physical_to_matrix(physical)
    rows = []
    for tile_row, row_start in enumerate(range(0, matrix.shape[0], TILE_ROWS)):
        for tile_col, col_start in enumerate(range(0, matrix.shape[1], TILE_COLS)):
            tile = matrix[row_start:row_start + TILE_ROWS, col_start:col_start + TILE_COLS]
            rows.append({
                "tile_row": tile_row, "tile_col": tile_col, "cell_count": int(tile.size),
                "one_fraction": float(tile.mean()),
                "horizontal_transition_density": float((tile[:, 1:] != tile[:, :-1]).mean()),
                "vertical_transition_density": float((tile[1:, :] != tile[:-1, :]).mean()),
            })
    return rows


def codeword_physical_mapping(codeword_index: int) -> list[tuple[int, int]]:
    if not 0 <= codeword_index < RS_CODEWORDS:
        raise ValueError("codeword index out of range")
    coordinates = []
    for byte_index in range(RS_N):
        physical_byte = byte_index * RS_CODEWORDS + codeword_index
        for bit_index in range(8):
            flat = physical_byte * 8 + bit_index
            coordinates.append((flat // PHYSICAL_CONFIG.cols, flat % PHYSICAL_CONFIG.cols))
    return coordinates


def lane_physical_summary() -> list[dict[str, Any]]:
    output = []
    for codeword in range(RS_CODEWORDS):
        coordinates = codeword_physical_mapping(codeword)
        row_counts = Counter(row for row, _ in coordinates)
        col_counts = Counter(col for _, col in coordinates)
        tile_counts = Counter((row // TILE_ROWS, col // TILE_COLS) for row, col in coordinates)
        for (tile_row, tile_col), count in sorted(tile_counts.items()):
            output.append({
                "codeword_index": codeword,
                "row_min": min(row_counts), "row_max": max(row_counts),
                "occupied_row_count": len(row_counts),
                "col_min": min(col_counts), "col_max": max(col_counts),
                "occupied_column_count": len(col_counts),
                "tile_row": tile_row, "tile_col": tile_col,
                "cell_count": count,
                "lane_cell_fraction_in_tile": count / (TILE_ROWS * TILE_COLS),
                "sample_coordinates": json.dumps(coordinates[:8]),
            })
    return output


def raw_physical_errors(canonical: bytes, observed: bytes) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], np.ndarray
]:
    if len(canonical) != len(observed) or len(canonical) != PHYSICAL_CONFIG.physical_bytes_per_frame:
        raise ValueError("raw physical frames have incompatible lengths")
    expected_values = np.frombuffer(canonical, dtype=np.uint8)
    observed_values = np.frombuffer(observed, dtype=np.uint8)
    xor = np.bitwise_xor(expected_values, observed_values)
    bit_errors = int(_BIT_COUNTS[xor].sum())
    error_mask = np.unpackbits(xor, bitorder=BIT_ORDER).astype(bool).reshape(
        PHYSICAL_CONFIG.rows, PHYSICAL_CONFIG.cols
    )
    expected_cw = deinterleave(canonical[:CODED_RS_BYTES])
    observed_cw = deinterleave(observed[:CODED_RS_BYTES])
    codeword_rows = []
    for index in range(RS_CODEWORDS):
        difference = np.bitwise_xor(expected_cw[index], observed_cw[index])
        byte_positions = np.flatnonzero(difference)
        cw_bit_errors = int(_BIT_COUNTS[difference].sum())
        codeword_rows.append({
            "codeword_index": index,
            "raw_byte_errors": int(len(byte_positions)),
            "raw_bit_errors": cw_bit_errors,
            "raw_bit_error_rate": cw_bit_errors / (RS_N * 8),
            "error_byte_positions": json.dumps(byte_positions.tolist()),
        })
    tile_rows = []
    for tile_row, row_start in enumerate(range(0, PHYSICAL_CONFIG.rows, TILE_ROWS)):
        for tile_col, col_start in enumerate(range(0, PHYSICAL_CONFIG.cols, TILE_COLS)):
            tile = error_mask[row_start:row_start + TILE_ROWS, col_start:col_start + TILE_COLS]
            errors = int(tile.sum())
            tile_rows.append({
                "tile_row": tile_row, "tile_col": tile_col,
                "total_bits": int(tile.size), "raw_bit_errors": errors,
                "raw_bit_error_rate": errors / tile.size,
            })
    summary = {
        "raw_bit_errors": bit_errors,
        "raw_bit_error_rate": bit_errors / (len(canonical) * 8),
        "raw_byte_errors": int(np.count_nonzero(xor)),
        "reserved_raw_bit_errors": int(_BIT_COUNTS[xor[CODED_RS_BYTES:]].sum()),
    }
    return summary, codeword_rows, tile_rows, error_mask


def compare_error_masks(source: np.ndarray, platform: np.ndarray) -> dict[str, int]:
    if source.shape != platform.shape:
        raise ValueError("error masks have incompatible shapes")
    source = source.astype(bool, copy=False)
    platform = platform.astype(bool, copy=False)
    return {
        "error_overlap_count": int(np.logical_and(source, platform).sum()),
        "platform_only_error_count": int(np.logical_and(platform, ~source).sum()),
        "source_only_error_count": int(np.logical_and(source, ~platform).sum()),
    }


def _longest_matrix_run(matrix: np.ndarray, axis: int) -> int:
    arrays = matrix if axis == 1 else matrix.T
    return max((_longest_identical_run(row) for row in arrays), default=0)


def _longest_identical_run(values: np.ndarray) -> int:
    if not len(values):
        return 0
    longest = current = 1
    previous = int(values[0])
    for value in values[1:]:
        current = current + 1 if int(value) == previous else 1
        longest = max(longest, current)
        previous = int(value)
    return longest
