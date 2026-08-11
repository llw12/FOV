from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from fvm_codec_sweep import (CRFS, LEVELS, SweepCase, actual_bitrate, cases,
                             generate_deterministic_file, parse_bitrate, render_levels,
                             resume_valid, run_cases, run_sweep, stable_hash, validate_levels)
from fvm_codec_sweep_analysis import expansion_ratio, mark_pareto
from fvm_file_common import PHYSICAL_CONFIG


def test_level_renderer_maps_grayscale_cells() -> None:
    bits = np.zeros((PHYSICAL_CONFIG.rows, PHYSICAL_CONFIG.cols), dtype=np.uint8)
    bits[0, 1] = bits[1, 0] = 1
    frame = render_levels(bits, 64, 192)
    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8
    assert np.all(frame[0:6, 0:6] == 64)
    assert np.all(frame[0:6, 6:12] == 192)
    assert np.array_equal(frame[:, :, 0], frame[:, :, 1])
    assert np.array_equal(frame[:, :, 1], frame[:, :, 2])


@pytest.mark.parametrize("low,high", LEVELS)
def test_level_validation_accepts_matrix(low: int, high: int) -> None:
    validate_levels(low, high)


@pytest.mark.parametrize("low,high", [(128, 192), (64, 128), (192, 64), (-1, 255), (0, 256)])
def test_level_validation_rejects_invalid(low: int, high: int) -> None:
    with pytest.raises(ValueError): validate_levels(low, high)


def test_deterministic_small_input(tmp_path: Path) -> None:
    first, second = tmp_path / "first.bin", tmp_path / "second.bin"
    first_sha = generate_deterministic_file(first, 8193, b"fixture", 1024)
    second_sha = generate_deterministic_file(second, 8193, b"fixture", 1024)
    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha
    assert generate_deterministic_file(first, 8193, b"fixture", 1024) == first_sha


def test_case_matrix_is_complete_unique_and_ordered() -> None:
    matrix = cases()
    assert len(matrix) == 20
    assert len({case.case_id for case in matrix}) == 20
    assert [(case.low_level, case.high_level, case.crf) for case in matrix] == [
        (low, high, crf) for low, high in LEVELS for crf in CRFS
    ]


def test_expansion_ratio_and_actual_bitrate() -> None:
    assert expansion_ratio(350, 100) == 3.5
    assert actual_bitrate(1000, 2) == 4000
    assert actual_bitrate(1000, 0) is None


def test_pareto_recoverable_frontier() -> None:
    rows = [
        {"case_id": "A", "source_expansion_ratio": 2, "proxy_rs_frame_failure_rate": .01, "source_sha_exact": True, "proxy_sha_exact": True},
        {"case_id": "B", "source_expansion_ratio": 3, "proxy_rs_frame_failure_rate": .02, "source_sha_exact": True, "proxy_sha_exact": True},
        {"case_id": "C", "source_expansion_ratio": 4, "proxy_rs_frame_failure_rate": 0, "source_sha_exact": True, "proxy_sha_exact": True},
        {"case_id": "D", "source_expansion_ratio": 2.5, "proxy_rs_frame_failure_rate": .005, "source_sha_exact": True, "proxy_sha_exact": True},
        {"case_id": "E", "source_expansion_ratio": 1, "proxy_rs_frame_failure_rate": 0, "source_sha_exact": False, "proxy_sha_exact": True},
    ]
    marked = mark_pareto(rows)
    assert {row["case_id"] for row in marked if row["is_pareto"]} == {"A", "C", "D"}
    assert next(row for row in marked if row["case_id"] == "B")["pareto_rank"] == 2
    assert next(row for row in marked if row["case_id"] == "E")["pareto_rank"] is None


@pytest.mark.parametrize("value,expected", [("1400k", 1_400_000), ("2M", 2_000_000), ("123456", 123456), (123456, 123456)])
def test_proxy_bitrate_parser(value, expected: int) -> None:
    assert parse_bitrate(value) == expected


def test_resume_requires_matching_complete_files(tmp_path: Path) -> None:
    result_path = tmp_path / "case_result.json"; source = tmp_path / "source.mp4"; proxy = tmp_path / "proxy.mp4"
    source.touch(); proxy.touch(); config_hash = stable_hash({"a": 1})
    result_path.write_text(json.dumps({"status": "COMPLETE", "case_config_hash": config_hash, "input_sha256": "input"}))
    assert resume_valid(result_path, config_hash, "input", source, proxy)
    assert not resume_valid(result_path, stable_hash({"a": 2}), "input", source, proxy)
    assert not resume_valid(result_path, config_hash, "changed", source, proxy)
    proxy.unlink()
    assert not resume_valid(result_path, config_hash, "input", source, proxy)


def test_case_failure_isolation() -> None:
    visited = []
    selected = [SweepCase(0, 255, crf) for crf in (18, 21, 24)]
    def runner(case: SweepCase):
        visited.append(case.crf)
        if case.crf == 21: raise RuntimeError("ffmpeg failed")
        return {"case_id": case.case_id, "status": "COMPLETE"}
    results = run_cases(selected, runner)
    assert visited == [18, 21, 24]
    assert [result["status"] for result in results] == ["COMPLETE", "FAILED", "COMPLETE"]


@pytest.mark.slow
def test_tiny_full_chain_smoke_is_explicit_only(tmp_path: Path) -> None:
    if os.environ.get("FVM_RUN_SLOW_TESTS") != "1":
        pytest.skip("set FVM_RUN_SLOW_TESTS=1 to run external codec smoke")
    input_path = tmp_path / "smoke-256KiB.bin"
    generate_deterministic_file(input_path, 256 * 1024, b"FVM_CODEC_SWEEP_SMOKE_V1")
    selected = {SweepCase(low, high, crf).case_id for low, high in ((0, 255), (64, 192)) for crf in (18, 24)}
    results = run_sweep(tmp_path / "run", input_path=input_path, proxy_reference=None, proxy_bitrate="12M",
                        resume=False, keep_proxy=False, keep_recovered=False, skip_proxy=False,
                        selected_ids=selected, preflight=False)
    assert len(results) == 4
    assert all(result["status"] == "COMPLETE" for result in results)
    assert all(result["source"]["sha_exact"] and result["proxy"]["sha_exact"] for result in results)
    assert (tmp_path / "run" / "summary" / "sweep_results.csv").is_file()
