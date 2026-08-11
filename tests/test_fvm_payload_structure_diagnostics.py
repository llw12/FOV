from __future__ import annotations

import hashlib

import numpy as np
import pytest

from fov import SymbolPacket
from fvm_file_common import PHYSICAL_CONFIG, matrix_to_physical, physical_to_matrix
from fvm_payload_structure_analysis import (
    byte_metrics,
    codeword_segment_rows,
    infer_pending_transport_indices,
    payload_boundary_location,
    range_overlap,
    symbol_logical_segments,
    validate_original_tail,
    validate_segment_cover,
)
from fvm_physical_analysis import (codeword_physical_mapping, compare_error_masks,
                                   raw_physical_errors)
from scripts.fvm0_rs_common import (BIT_ORDER, CODED_RS_BYTES, RS_CODEWORDS, RS_K, RS_N,
                                    encode_logical, interleave)


def _packet(payload: bytes = bytes(range(256)) * 25) -> SymbolPacket:
    return SymbolPacket(b"12345678", 2, 1310, payload)


def test_zero_payload_metrics_are_exact():
    metrics = byte_metrics(bytes(6400))
    assert metrics["byte_zero_fraction"] == 1
    assert metrics["entropy_bits_per_byte"] == 0
    assert metrics["bit_one_fraction"] == 0
    assert metrics["bit_transition_density_1d"] == 0
    assert metrics["longest_zero_byte_run"] == 6400


def test_alternating_bit_transition_density_is_one():
    metrics = byte_metrics(bytes([0xAA]) * 100)
    assert metrics["bit_one_fraction"] == 0.5
    assert metrics["bit_transition_density_1d"] == 1.0
    assert metrics["longest_identical_byte_run"] == 100


def test_symbol_segments_are_contiguous_and_cover_logical_frame():
    _, logical, segments = symbol_logical_segments(_packet(), 101)
    validate_segment_cover(segments, len(logical))
    assert len(logical) == 6692
    assert [segment.name for segment in segments] == [
        "transport_header", "fov_symbol_header", "raptorq_payload",
        "fov_packet_crc", "transport_shake_padding", "transport_crc",
    ]
    assert sum(segment.length for segment in segments) == 6692
    assert next(segment for segment in segments if segment.name == "raptorq_payload").length == 6400


def test_invalid_segment_cover_is_rejected():
    _, _, segments = symbol_logical_segments(_packet(), 101)
    segments[1] = type(segments[1])(segments[1].name, segments[1].start + 1, segments[1].end)
    with pytest.raises(ValueError, match="overlap or contain gaps"):
        validate_segment_cover(segments, 6692)


def test_codeword_ranges_cover_all_logical_bytes_and_payload_overlap():
    _, _, segments = symbol_logical_segments(_packet(), 101)
    rows = codeword_segment_rows(segments, actual_source_length=4608)
    assert len(rows) == RS_CODEWORDS
    assert rows[0]["logical_start"] == 0
    assert rows[-1]["logical_end"] == 6692
    assert sum(row["raptorq_payload_bytes"] for row in rows) == 6400
    assert sum(row["real_source_bytes"] for row in rows) == 4608
    assert sum(row["observed_payload_suffix_bytes"] for row in rows) == 1792
    assert range_overlap(0, 10, 5, 12) == 5


def test_full_and_short_tail_boundaries_map_to_exact_codewords():
    _, _, segments = symbol_logical_segments(_packet(), 101)
    full = payload_boundary_location(segments, 4608)
    short = payload_boundary_location(segments, 2304)
    assert full == {"logical_offset": 4640, "codeword_index": 19, "offset_within_codeword": 99}
    assert short == {"logical_offset": 2336, "codeword_index": 9, "offset_within_codeword": 185}


def test_tail_validation_reports_observed_nonzero_suffix_without_assumption():
    original = b"tail-data"
    suffix = bytes([0xAA]) * 20
    result = validate_original_tail(original + suffix, original)
    assert result["payload_prefix_matches_original"]
    assert result["suffix"] == suffix
    assert result["suffix_metrics"]["byte_zero_fraction"] == 0
    assert result["suffix_sha256"] == hashlib.sha256(suffix).hexdigest()


def test_tail_validation_reports_prefix_mismatch():
    result = validate_original_tail(b"abcXYZ", b"abd")
    assert not result["systematic_identity"]
    assert result["payload_prefix_match_bytes"] == 2


@pytest.mark.parametrize(
    "previous,next_index,count,first_observed,expected",
    [
        (100, 102, 1, 101, [101]),
        (100, 104, 2, 101, None),
        (None, 1, 1, 0, [0]),
        (None, 101, 1, 100, None),
    ],
)
def test_failure_inference_never_uses_observed_index_fallback(
    previous, next_index, count, first_observed, expected
):
    assert infer_pending_transport_indices(
        previous, next_index, count, first_observed_index=first_observed
    ) == expected


def test_codeword_physical_mapping_matches_real_interleave():
    markers = np.stack([
        np.full(RS_N, codeword, dtype=np.uint8) for codeword in range(RS_CODEWORDS)
    ])
    interleaved = np.frombuffer(interleave(markers), dtype=np.uint8)
    for codeword in (0, 8, 10, 27):
        coordinates = codeword_physical_mapping(codeword)
        byte_indices = sorted({(row * PHYSICAL_CONFIG.cols + col) // 8 for row, col in coordinates})
        assert len(coordinates) == RS_N * 8
        assert len(byte_indices) == RS_N
        assert all(interleaved[index] == codeword for index in byte_indices)


def test_matrix_mapping_is_big_endian_row_major():
    physical = bytes([0x80]) + bytes(PHYSICAL_CONFIG.physical_bytes_per_frame - 1)
    matrix = physical_to_matrix(physical)
    assert matrix.shape == (180, 320)
    assert matrix[0, 0] == 1
    assert int(matrix.sum()) == 1
    assert matrix_to_physical(matrix) == physical


def test_raw_physical_errors_report_exact_codeword_bytes_and_bits():
    logical = bytes(6692)
    canonical_codewords = encode_logical(logical)
    observed_codewords = canonical_codewords.copy()
    for byte_index in (1, 4, 9):
        observed_codewords[8, byte_index] ^= 0x01
    for byte_index in range(9):
        observed_codewords[10, byte_index] ^= 0x80
    reserved = bytes(PHYSICAL_CONFIG.reserved_bytes_per_frame)
    canonical = interleave(canonical_codewords) + reserved
    observed = interleave(observed_codewords) + reserved
    summary, codewords, _, _ = raw_physical_errors(canonical, observed)
    assert summary["raw_bit_errors"] == 12
    assert summary["raw_byte_errors"] == 12
    assert codewords[8]["raw_byte_errors"] == 3
    assert codewords[8]["raw_bit_errors"] == 3
    assert codewords[10]["raw_byte_errors"] == 9
    assert codewords[10]["raw_bit_errors"] == 9
    assert sum(row["raw_bit_errors"] for row in codewords if row["codeword_index"] not in (8, 10)) == 0


def test_raw_oracle_identical_frame_has_zero_ber():
    physical = bytes(PHYSICAL_CONFIG.physical_bytes_per_frame)
    summary, codewords, tiles, mask = raw_physical_errors(physical, physical)
    assert summary["raw_bit_errors"] == 0
    assert all(row["raw_byte_errors"] == 0 for row in codewords)
    assert all(row["raw_bit_errors"] == 0 for row in tiles)
    assert not mask.any()


def test_source_platform_error_overlap_is_set_based():
    source = np.array([[True, False, True], [False, False, False]])
    platform = np.array([[True, True, False], [False, False, True]])
    assert compare_error_masks(source, platform) == {
        "error_overlap_count": 1,
        "platform_only_error_count": 2,
        "source_only_error_count": 1,
    }
