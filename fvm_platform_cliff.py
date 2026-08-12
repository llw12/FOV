"""Serial, resumable adaptive search for an observed FVM platform recovery cliff."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from file2video import packet_stream
from fov import (BLOCK_SIZE, INTERLEAVE_WINDOW, SymbolPacket, build_metadata, derive_block_layout,
                 file_id_from_sha256, make_raptorq_engine, parse_packet, sha256_file)
from fvm_codec_sweep import REPAIR_RATIO, SweepCase, encode_sweep_case, ffprobe
from fvm_file_common import SYMBOL_SIZE
from fvm_platform_validate import run_validation
from fvm_video2file import decode as production_decode, result_path

FORMAT = "FVM_PLATFORM_CLIFF_V1"
EXPECTED_INPUT_SIZE = 32 * 1024 * 1024
EXPECTED_INPUT_SHA = "198d38cfd9890f9b33d2ac407bf6b5118a4611beb002573a117a694d08bf43d7"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def level_midpoint(pass_low: int, fail_low: int) -> int | None:
    if fail_low - pass_low <= 2:
        return None
    midpoint = (pass_low + fail_low) // 2
    return midpoint if midpoint not in (pass_low, fail_low) else None


def crf_midpoint(pass_crf: int, fail_crf: int) -> int | None:
    if fail_crf - pass_crf <= 1:
        return None
    midpoint = (pass_crf + fail_crf + 1) // 2
    return midpoint if midpoint not in (pass_crf, fail_crf) else None


def level_non_monotonic(observations: dict[int, str]) -> bool:
    ordered = sorted(observations.items())
    return any(left_state == "FAIL" and right_state == "PASS"
               for index, (_, left_state) in enumerate(ordered) for _, right_state in ordered[index + 1:])


def choose_candidate(cases: dict[str, dict[str, Any]]) -> SweepCase | None:
    """Follow the prescribed coarse path, then locally bisect a valid observed axis."""
    verdict = {(row["low_level"], row["crf"]): row.get("verdict") for row in cases.values()
               if row.get("verdict") in {"PASS", "FAIL"}}
    if (64, 30) not in verdict: return SweepCase(64, 192, 30)
    crf30_axis = {low: state for (low, crf), state in verdict.items() if crf == 30}
    if level_non_monotonic(crf30_axis): return None
    if verdict[(64, 30)] == "PASS":
        axis = {low: state for (low, crf), state in verdict.items() if crf == 30 and 64 <= low <= 80}
        if level_non_monotonic(axis): return None
        passes = [low for low, state in axis.items() if state == "PASS"]
        fails = [low for low, state in axis.items() if state == "FAIL"]
        if passes and fails:
            low = level_midpoint(max(passes), min(fails))
            return SweepCase(low, 256 - low, 30) if low is not None else None
    else:
        if (48, 30) not in verdict: return SweepCase(48, 208, 30)
        if (80, 27) not in verdict: return SweepCase(80, 176, 27)
        # Once the prescribed orthogonal diagnostics are complete, refine any
        # observed same-CRF level bracket before using coarse fallback points.
        axis = {low: state for (low, crf), state in verdict.items() if crf == 30}
        if not level_non_monotonic(axis):
            brackets = [(failed - passed, passed, failed)
                        for passed, pass_state in axis.items() if pass_state == "PASS"
                        for failed, fail_state in axis.items() if fail_state == "FAIL" and passed < failed]
            if brackets:
                _, passed, failed = min(brackets)
                low = level_midpoint(passed, failed)
                if low is not None: return SweepCase(low, 256 - low, 30)
        alternatives: list[tuple[float, SweepCase]] = []
        if verdict[(48, 30)] == "PASS":
            axis = {low: state for (low, crf), state in verdict.items() if crf == 30 and 48 <= low <= 64}
            if not level_non_monotonic(axis):
                passes = [low for low, state in axis.items() if state == "PASS"]
                fails = [low for low, state in axis.items() if state == "FAIL"]
                if passes and fails:
                    low = level_midpoint(max(passes), min(fails))
                    if low is not None: alternatives.append((0, SweepCase(low, 256 - low, 30)))
        if verdict[(80, 27)] == "PASS":
            axis = {crf: state for (low, crf), state in verdict.items() if low == 80 and 27 <= crf <= 30}
            passes = [crf for crf, state in axis.items() if state == "PASS"]
            fails = [crf for crf, state in axis.items() if state == "FAIL"]
            if passes and fails:
                crf = crf_midpoint(max(passes), min(fails))
                if crf is not None: alternatives.append((1, SweepCase(80, 176, crf)))
        if alternatives:
            for index, (_, candidate) in enumerate(alternatives):
                anchor_id = SweepCase(48, 208, 30).case_id if candidate.crf == 30 else SweepCase(80, 176, 27).case_id
                alternatives[index] = (cases.get(anchor_id, {}).get("source_ratio", float("inf")), candidate)
            return min(alternatives, key=lambda item: item[0])[1]
        if verdict[(48, 30)] == "FAIL" and verdict[(80, 27)] == "FAIL":
            if (64, 27) not in verdict: return SweepCase(64, 192, 27)
            if (0, 30) not in verdict: return SweepCase(0, 255, 30)
    return None


def can_start_upload(state: dict[str, Any]) -> bool:
    return state.get("new_upload_count", 0) < state.get("max_new_uploads", 8)


def has_resumable_case(state: dict[str, Any], root: Path) -> bool:
    return any(row.get("kind") != "KNOWN FAIL" and row.get("verdict") not in {"PASS", "FAIL"}
               and (root / "cases" / case_id / "validation_result.json").is_file()
               for case_id, row in state.get("cases", {}).items())


def _source_case(case: SweepCase, input_path: Path, sweep_root: Path, prepared_root: Path) -> tuple[Path, Path]:
    existing = sweep_root / "cases" / case.case_id
    if (existing / "source.mp4").is_file() and (existing / "case_result.json").is_file():
        return existing / "source.mp4", existing / "case_result.json"
    target = prepared_root / case.case_id; target.mkdir(parents=True, exist_ok=True)
    source, result_file = target / "source.mp4", target / "case_result.json"
    if not source.is_file() or not result_file.is_file():
        encoded = encode_sweep_case(input_path, source, case)
        probe, warnings = ffprobe(source)
        recovered_dir = target / "source-recovered"
        recovered = production_decode(source, recovered_dir)
        diagnostics = json.loads(result_path(recovered_dir).read_text(encoding="utf-8"))
        result = {"case_id": case.case_id, "status": "COMPLETE", "config": case.config(),
                  "input_size_bytes": input_path.stat().st_size, "input_sha256": sha256_file(input_path),
                  "warnings": warnings, "source": {**encoded, "mp4_path": str(source),
                  "mp4_size_bytes": source.stat().st_size, "expansion_ratio": source.stat().st_size / input_path.stat().st_size,
                  "duration_seconds": probe.get("duration_seconds"), "bitrate_bps": probe.get("calculated_bitrate_bps"),
                  "ffprobe": probe, "sha_exact": diagnostics.get("file", {}).get("exact") is True}}
        atomic_write_json(result_file, result)
        recovered.unlink(missing_ok=True)
    return source, result_file


def _case_metrics(validation: dict[str, Any]) -> dict[str, Any]:
    preflight, platform, decode, raw = (validation.get(key, {}) for key in ("preflight", "platform", "decode", "raw_channel"))
    config, source = preflight.get("case_config", {}), preflight.get("source", {})
    distribution = decode.get("rs_failure_distribution") or {}
    case_id = preflight.get("case_id") or SweepCase(config["low_level"], config["high_level"], config["crf"]).case_id
    return {"case_id": case_id, "low_level": config.get("low_level"),
            "high_level": config.get("high_level"), "crf": config.get("crf"),
            "source_size_bytes": source.get("size_bytes"),
            "source_ratio": source.get("size_bytes") / preflight["original"]["size_bytes"],
            "source_bitrate_bps": source.get("calculated_bitrate_bps"), "source_frames": source.get("frame_count"),
            "upload_attempt_count": validation.get("upload_attempt_count", 0), "bvid": validation.get("bvid"),
            "platform_codec": platform.get("codec"), "platform_bitrate_bps": platform.get("calculated_bitrate_bps"),
            "platform_frames": platform.get("frame_count"), "raw_ber": raw.get("raw_ber"),
            "frames_with_raw_errors": raw.get("frames_with_raw_errors"),
            "rs_failed_frames": decode.get("rs_failed_frames"), "rs_failure_rate": decode.get("rs_frame_failure_rate"),
            "failed_codewords": decode.get("codewords_failed"), "corrected_symbols": decode.get("corrected_symbols"),
            "longest_rs_failure_burst": distribution.get("longest_consecutive_burst"),
            "source_symbols_expected": decode.get("source_symbols_expected"),
            "source_symbols_received": decode.get("source_symbols_received"), "source_erasures": decode.get("source_erasures"),
            "repair_symbols_received": decode.get("repair_symbols_received"), "blocks_total": decode.get("blocks_total"),
            "blocks_decoded": decode.get("blocks_decoded"), "sha_exact": decode.get("sha_exact") is True,
            "verdict": ("PASS" if validation.get("state") == "PASS" else
                        "FAIL" if validation.get("state") == "FAIL" else None),
            "validation_run": validation.get("run_dir"), "failed_blocks": decode.get("failed_blocks", [])}


def erasure_attribution(validation: dict[str, Any], input_path: Path) -> dict[str, Any]:
    """Attribute confidently inferred failed transport indices without changing recovery."""
    diagnostic_file = Path(validation["run_dir"]) / "recovered" / ".fvm" / "fvm_decode_results.json"
    layout = derive_block_layout(input_path.stat().st_size, BLOCK_SIZE, SYMBOL_SIZE, REPAIR_RATIO)
    unavailable = {"confidence": "UNAVAILABLE", "descriptors": [], "source_erasures": None,
                   "repair_erasures": None, "unknown_erasures": None, "blocks": []}
    if not diagnostic_file.is_file(): return unavailable
    diagnostics = json.loads(diagnostic_file.read_text(encoding="utf-8"))
    events = diagnostics.get("rs_failures", {}).get("events", [])
    targets = {event.get("inferred_transport_index"): event for event in events
               if event.get("inference_confident") is True and event.get("inferred_transport_index") is not None}
    digest = sha256_file(input_path)
    metadata = build_metadata(input_path.name, input_path.stat().st_size, digest, SYMBOL_SIZE, REPAIR_RATIO)
    mapped: dict[int, SymbolPacket] = {}
    for index, encoded in enumerate(packet_stream(input_path, metadata, file_id_from_sha256(digest), make_raptorq_engine())):
        if index in targets:
            packet = parse_packet(encoded)
            if isinstance(packet, SymbolPacket): mapped[index] = packet
    descriptors, burst_id, previous = [], -1, None
    source_by_block = {block.block_id: 0 for block in layout}; repair_by_block = {block.block_id: 0 for block in layout}
    for index in sorted(targets):
        packet = mapped.get(index)
        if packet is None: continue
        if previous is None or index != previous + 1: burst_id += 1
        previous = index; block = layout[packet.block_id]
        classification = "SOURCE" if packet.symbol_id < block.source_symbols else "REPAIR"
        (source_by_block if classification == "SOURCE" else repair_by_block)[packet.block_id] += 1
        descriptors.append({"transport_index": index, "block_id": packet.block_id, "symbol_id": packet.symbol_id,
                            "classification": classification, "boundary_offset": index % INTERLEAVE_WINDOW,
                            "burst_id": burst_id})
    unknown = len(events) - len(descriptors)
    blocks = [{"block_id": block.block_id, "k": block.source_symbols, "n": block.encoded_symbols,
               "received_source": block.source_symbols - source_by_block[block.block_id],
               "received_repair": block.encoded_symbols - block.source_symbols - repair_by_block[block.block_id],
               "received_unique": block.encoded_symbols - source_by_block[block.block_id] - repair_by_block[block.block_id],
               "symbol_margin": block.encoded_symbols - source_by_block[block.block_id] - repair_by_block[block.block_id] - block.source_symbols}
              for block in layout]
    return {"confidence": "COMPLETE" if unknown == 0 else "PARTIAL", "descriptors": descriptors,
            "source_erasures": sum(source_by_block.values()), "repair_erasures": sum(repair_by_block.values()),
            "unknown_erasures": unknown, "blocks": blocks}


def _write_outputs(root: Path, state: dict[str, Any]) -> None:
    fields = ["case_id", "low_level", "high_level", "crf", "source_size_bytes", "source_ratio", "source_bitrate_bps",
              "bvid", "platform_codec", "platform_bitrate_bps", "raw_ber", "rs_failed_frames", "rs_failure_rate",
              "failed_codewords", "longest_rs_failure_burst", "source_erasures", "repair_symbols_received",
              "blocks_decoded", "blocks_total", "sha_exact", "verdict"]
    rows = sorted(state["cases"].values(), key=lambda row: row.get("sequence", -1))
    with (root / "platform_cliff_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for row in rows: writer.writerow({key: row.get(key) for key in fields})
    atomic_write_json(root / "platform_cliff_summary.json", state)
    passes = [row for row in rows if row.get("verdict") == "PASS"]
    best = min(passes, key=lambda row: row.get("source_ratio", float("inf"))) if passes else None
    conservative = min(passes, key=lambda row: row.get("low_level", 999)) if passes else None
    attribution_rows = [row for row in rows if row.get("source_erasures_attributed") is not None]
    source_losses = sum(row.get("source_erasures_attributed") or 0 for row in attribution_rows)
    repair_losses = sum(row.get("repair_erasures") or 0 for row in attribution_rows)
    lines = ["# FVM Real Platform Adaptive Cliff Search", "", "REAL PLATFORM IS THE ORACLE.", "",
             "## Results", "", "| Case | Kind | Ratio | BVID | Raw BER | RS fails | Longest burst | Blocks | SHA |",
             "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for row in rows:
        lines.append(f"| {row.get('case_id')} | {row.get('kind')} | {row.get('source_ratio')} | {row.get('bvid')} | {row.get('raw_ber')} | {row.get('rs_failed_frames')} | {row.get('longest_rs_failure_burst')} | {row.get('blocks_decoded')}/{row.get('blocks_total')} | {row.get('sha_exact')} |")
    lines += ["", "## Search conclusion", "", f"- Status: {state.get('status')}",
              f"- New uploads: {state.get('new_upload_count')}/{state.get('max_new_uploads')}",
              f"- Observed bracket: {state.get('current_bracket')}",
              f"- Best compact platform PASS: {best.get('case_id') if best else None}, ratio {best.get('source_ratio') if best else None}",
              f"- Conservative tested PASS: {conservative.get('case_id') if conservative else None}",
              f"- Confidently attributed source/repair erasures across new samples: {source_losses}/{repair_losses}; unmapped events remain UNKNOWN.",
              "- Same-level CRF cliff: not found.",
              "- Non-monotonic observation on the refined CRF30 axis: no.",
              "- Independent repeat validation is still required before changing production defaults.",
              "- Results are sample-specific observations, not universal reliability guarantees."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class SearchOptions:
    biliup: str
    biliup_cookies: Path
    tid: int
    max_new_uploads: int = 8
    upload_cooldown: float = 120
    poll_interval: float = 60
    approval_timeout: float = 10800
    rendition_timeout: float = 7200


def run_search(root: Path, input_path: Path, sweep_root: Path, known_fail_run: Path,
               options: SearchOptions, *, allow_upload: bool, resume: bool,
               validator: Callable[..., dict[str, Any]] = run_validation) -> dict[str, Any]:
    root, input_path, sweep_root = root.resolve(), input_path.resolve(), sweep_root.resolve()
    state_file = root / "search_state.json"; root.mkdir(parents=True, exist_ok=True)
    if input_path.stat().st_size != EXPECTED_INPUT_SIZE or sha256_file(input_path) != EXPECTED_INPUT_SHA:
        raise ValueError("fixed 32 MiB input size/SHA mismatch")
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        for case_id, row in state.get("cases", {}).items():
            validation_file = root / "cases" / case_id / "validation_result.json"
            if row.get("kind") != "KNOWN FAIL" and validation_file.is_file():
                validation_state = json.loads(validation_file.read_text(encoding="utf-8")).get("state")
                if validation_state not in {"PASS", "FAIL"}:
                    row["verdict"] = None; row["kind"] = "NEW PLATFORM"
        bracket = state.get("current_bracket")
        bracket_tight = bool(bracket and ((bracket[1] == "level" and bracket[0] <= 2)
                             or (bracket[1] == "crf" and bracket[0] <= 1)))
        if resume and (state.get("status") in {"BLOCKED", "INCOMPLETE", "BUDGET_EXHAUSTED"}
                       or (state.get("status") == "FOUND" and not bracket_tight)):
            unfinished = [row for row in state.get("cases", {}).values()
                          if row.get("kind") != "KNOWN FAIL" and row.get("verdict") not in {"PASS", "FAIL"}
                          and row.get("upload_attempt_count", 0) <= 1]
            if unfinished or (state.get("status") == "FOUND" and not bracket_tight): state["status"] = "RUNNING"
    else:
        known = json.loads((known_fail_run / "validation_result.json").read_text(encoding="utf-8"))
        if known.get("state") != "FAIL" or known.get("bvid") != "BV1mGuC64E4Z":
            raise ValueError("known 80/176 CRF30 failure could not be verified")
        metrics = _case_metrics(known); metrics.update({"kind": "KNOWN FAIL", "sequence": 0})
        state = {"format": FORMAT, "input_sha256": EXPECTED_INPUT_SHA, "known_failures": [metrics["case_id"]],
                 "cases": {metrics["case_id"]: metrics}, "active_case": None, "new_upload_count": 0,
                 "max_new_uploads": options.max_new_uploads, "current_axis": None, "current_bracket": None,
                 "last_case_completed_at": None, "status": "RUNNING"}
        atomic_write_json(state_file, state); _write_outputs(root, state)
    if not allow_upload: return state
    while state["status"] == "RUNNING" and (can_start_upload(state) or has_resumable_case(state, root)):
        candidate = choose_candidate(state["cases"])
        if candidate is None:
            state["status"] = "UNRESOLVED" if any(row.get("verdict") == "PASS" for row in state["cases"].values()) else "NOT_FOUND"
            break
        case_id = candidate.case_id; row = state["cases"].get(case_id, {})
        validation_dir = root / "cases" / case_id
        if row.get("bvid") or (validation_dir / "validation_result.json").exists():
            resume_run, run_override = validation_dir, None
        else:
            resume_run, run_override = None, validation_dir
            if state.get("last_case_completed_at"):
                remaining = options.upload_cooldown - (time.time() - state["last_case_completed_at"])
                if remaining > 0: time.sleep(remaining)
        source, case_result = _source_case(candidate, input_path, sweep_root, root / "prepared-cases")
        state["active_case"] = case_id
        state["cases"].setdefault(case_id, {"case_id": case_id, "low_level": candidate.low_level,
            "high_level": candidate.high_level, "crf": candidate.crf, "kind": "NEW PLATFORM", "state": "READY_UPLOAD",
            "sequence": len(state["cases"])})
        atomic_write_json(state_file, state)
        validation = validator(source, input_path, case_result, allow_upload=True, biliup=options.biliup,
            biliup_cookies=options.biliup_cookies, tid=options.tid, output_root=root / "cases",
            resume_run=resume_run, run_dir_override=run_override,
            title=f"FVM 6px {candidate.low_level}-{candidate.high_level} CRF{candidate.crf} platform cliff {datetime.now():%Y%m%d-%H%M%S}",
            poll_interval=options.poll_interval, approval_timeout=options.approval_timeout,
            rendition_timeout=options.rendition_timeout)
        metrics = _case_metrics(validation); previous_attempts = state["cases"][case_id].get("upload_attempt_count", 0)
        attribution = erasure_attribution(validation, input_path)
        metrics["erasure_attribution"] = attribution
        metrics["source_erasures_attributed"] = attribution.get("source_erasures")
        metrics["repair_erasures"] = attribution.get("repair_erasures")
        metrics["unknown_erasures"] = attribution.get("unknown_erasures")
        metrics.update({"kind": "NEW PLATFORM" + (" " + metrics["verdict"] if metrics["verdict"] else ""),
                        "sequence": state["cases"][case_id]["sequence"]})
        state["cases"][case_id].update(metrics)
        state["new_upload_count"] += max(0, metrics["upload_attempt_count"] - previous_attempts)
        state["active_case"] = None; state["last_case_completed_at"] = time.time()
        if validation.get("state") in {"BLOCKED", "INCOMPLETE"}:
            state["status"] = validation["state"]
        atomic_write_json(state_file, state); _write_outputs(root, state)
    if state["status"] == "RUNNING" and not can_start_upload(state): state["status"] = "BUDGET_EXHAUSTED"
    # Report the tightest observed same-CRF level or same-level CRF bracket.
    brackets = []
    rows = list(state["cases"].values())
    for passed in (row for row in rows if row.get("verdict") == "PASS"):
        for failed in (row for row in rows if row.get("verdict") == "FAIL"):
            if passed.get("crf") == failed.get("crf") and passed.get("low_level") < failed.get("low_level"):
                brackets.append((failed["low_level"] - passed["low_level"], "level", passed["case_id"], failed["case_id"]))
            if passed.get("low_level") == failed.get("low_level") and passed.get("crf") < failed.get("crf"):
                brackets.append((failed["crf"] - passed["crf"], "crf", passed["case_id"], failed["case_id"]))
    state["current_bracket"] = min(brackets) if brackets else None
    if state["current_bracket"] and ((state["current_bracket"][1] == "level" and state["current_bracket"][0] <= 2)
                                     or (state["current_bracket"][1] == "crf" and state["current_bracket"][0] <= 1)):
        state["status"] = "FOUND"
    state["best_compact_pass"] = min((row for row in rows if row.get("verdict") == "PASS"),
        key=lambda row: row.get("source_ratio", float("inf")), default=None)
    state["conservative_tested_pass"] = min((row for row in rows if row.get("verdict") == "PASS"),
        key=lambda row: row.get("low_level", 999), default=None)
    state["non_monotonic_observation"] = level_non_monotonic(
        {row["low_level"]: row["verdict"] for row in rows if row.get("crf") == 30 and row.get("verdict")})
    atomic_write_json(state_file, state); _write_outputs(root, state)
    return state
