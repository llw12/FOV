from __future__ import annotations

import json
from pathlib import Path

from fvm_platform_cliff import (atomic_write_json, can_start_upload, choose_candidate, crf_midpoint, has_resumable_case,
                                level_midpoint, level_non_monotonic)


def row(low: int, crf: int, verdict: str, ratio: float = 8.0) -> dict:
    return {"case_id": f"level-{low:03d}-{256-low:03d}_crf{crf}", "low_level": low,
            "high_level": 256-low, "crf": crf, "verdict": verdict, "source_ratio": ratio}


def cases(*rows: dict) -> dict:
    return {item["case_id"]: item for item in rows}


def test_first_candidate_is_fixed() -> None:
    selected = choose_candidate(cases(row(80, 30, "FAIL")))
    assert (selected.low_level, selected.high_level, selected.crf) == (64, 192, 30)


def test_level_midpoint_and_stop() -> None:
    assert level_midpoint(64, 80) == 72
    assert level_midpoint(76, 78) is None


def test_crf_midpoint_and_stop() -> None:
    assert crf_midpoint(27, 30) == 29
    assert crf_midpoint(27, 29) == 28
    assert crf_midpoint(28, 29) is None


def test_pass_path_bisects_level_axis() -> None:
    selected = choose_candidate(cases(row(64, 30, "PASS"), row(80, 30, "FAIL")))
    assert (selected.low_level, selected.high_level) == (72, 184)


def test_fail_path_runs_orthogonal_diagnostics_in_order() -> None:
    first = choose_candidate(cases(row(64, 30, "FAIL"), row(80, 30, "FAIL")))
    assert (first.low_level, first.crf) == (48, 30)
    second = choose_candidate(cases(row(48, 30, "FAIL"), row(64, 30, "FAIL"), row(80, 30, "FAIL")))
    assert (second.low_level, second.crf) == (80, 27)


def test_coarse_pass_after_diagnostics_opens_level_bracket() -> None:
    selected = choose_candidate(cases(row(0, 30, "PASS"), row(48, 30, "FAIL"),
                                      row(64, 30, "FAIL"), row(80, 27, "FAIL"), row(80, 30, "FAIL")))
    assert (selected.low_level, selected.high_level, selected.crf) == (24, 232, 30)


def test_non_monotonic_axis_stops_bisection() -> None:
    observations = {64: "FAIL", 72: "PASS", 80: "FAIL"}
    assert level_non_monotonic(observations)
    assert choose_candidate(cases(*(row(low, 30, result) for low, result in observations.items()))) is None


def test_upload_budget() -> None:
    assert can_start_upload({"new_upload_count": 7, "max_new_uploads": 8})
    assert not can_start_upload({"new_upload_count": 8, "max_new_uploads": 8})


def test_budget_does_not_block_finishing_uploaded_case(tmp_path: Path) -> None:
    run = tmp_path / "cases" / "case-a"; run.mkdir(parents=True)
    (run / "validation_result.json").write_text("{}")
    state = {"new_upload_count": 8, "max_new_uploads": 8,
             "cases": {"case-a": {"kind": "NEW PLATFORM", "verdict": None, "upload_attempt_count": 1}}}
    assert not can_start_upload(state) and has_resumable_case(state, tmp_path)


def test_existing_bvid_is_a_resume_identity() -> None:
    stored = {"case_id": "x", "bvid": "BV123", "upload_attempt_count": 1}
    assert stored["bvid"] and stored["upload_attempt_count"] == 1


def test_atomic_state_replaces_tmp(tmp_path: Path) -> None:
    target = tmp_path / "search_state.json"
    atomic_write_json(target, {"status": "RUNNING"})
    assert json.loads(target.read_text()) == {"status": "RUNNING"}
    assert not target.with_suffix(".json.tmp").exists()
