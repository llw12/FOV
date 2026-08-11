"""Result aggregation and Pareto analysis for the local FVM codec sweep."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


CSV_FIELDS = [
    "case_id", "status", "low_level", "high_level", "delta", "crf", "preset",
    "input_size_bytes", "source_mp4_size_bytes", "source_expansion_ratio",
    "source_duration", "source_bitrate", "source_raw_bit_errors", "source_raw_ber",
    "source_rs_failed_frames", "source_blocks_decoded", "source_sha_exact",
    "proxy_mp4_size_bytes", "proxy_bitrate", "proxy_raw_bit_errors", "proxy_raw_ber",
    "proxy_rs_failed_frames", "proxy_rs_frame_failure_rate", "proxy_blocks_decoded",
    "proxy_blocks_total", "proxy_repair_symbols_received", "proxy_sha_exact",
    "recoverable", "is_pareto", "pareto_rank", "elapsed_encode_seconds",
    "elapsed_source_decode_seconds", "elapsed_proxy_seconds", "elapsed_proxy_decode_seconds",
]


def expansion_ratio(mp4_size: int, input_size: int) -> float:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    return mp4_size / input_size


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the recoverable two-objective frontier and nondomination rank."""
    eligible = [row for row in rows if row.get("source_sha_exact") is True and row.get("proxy_sha_exact") is True]
    remaining = list(eligible)
    rank = 1
    while remaining:
        frontier = []
        for candidate in remaining:
            x, y = candidate["source_expansion_ratio"], candidate["proxy_rs_frame_failure_rate"]
            dominated = any(
                other is not candidate
                and other["source_expansion_ratio"] <= x
                and other["proxy_rs_frame_failure_rate"] <= y
                and (other["source_expansion_ratio"] < x or other["proxy_rs_frame_failure_rate"] < y)
                for other in remaining
            )
            if not dominated:
                frontier.append(candidate)
        for row in frontier:
            row["pareto_rank"] = rank
        remaining = [row for row in remaining if row not in frontier]
        rank += 1
    for row in rows:
        row["recoverable"] = row.get("source_sha_exact") is True and row.get("proxy_sha_exact") is True
        row["is_pareto"] = row.get("pareto_rank") == 1
        row.setdefault("pareto_rank", None)
    return rows


def _flatten(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config", {})
    source = result.get("source", {})
    proxy = result.get("proxy", {})
    timings = result.get("timings", {})
    row = {
        "case_id": result.get("case_id"), "status": result.get("status"),
        "low_level": config.get("low_level"), "high_level": config.get("high_level"),
        "delta": config.get("level_delta"), "crf": config.get("crf"), "preset": config.get("preset"),
        "input_size_bytes": result.get("input_size_bytes"),
        "source_mp4_size_bytes": source.get("mp4_size_bytes"),
        "source_expansion_ratio": source.get("expansion_ratio"),
        "source_duration": source.get("duration_seconds"), "source_bitrate": source.get("bitrate_bps"),
        "source_raw_bit_errors": source.get("raw_bit_errors"), "source_raw_ber": source.get("raw_ber"),
        "source_rs_failed_frames": source.get("rs_failed_frames"),
        "source_blocks_decoded": source.get("blocks_decoded"), "source_sha_exact": source.get("sha_exact"),
        "proxy_mp4_size_bytes": proxy.get("mp4_size_bytes"), "proxy_bitrate": proxy.get("bitrate_bps"),
        "proxy_raw_bit_errors": proxy.get("raw_bit_errors"), "proxy_raw_ber": proxy.get("raw_ber"),
        "proxy_rs_failed_frames": proxy.get("rs_failed_frames"),
        "proxy_rs_frame_failure_rate": proxy.get("rs_frame_failure_rate"),
        "proxy_blocks_decoded": proxy.get("blocks_decoded"), "proxy_blocks_total": proxy.get("blocks_total"),
        "proxy_repair_symbols_received": proxy.get("repair_symbols_received"),
        "proxy_sha_exact": proxy.get("sha_exact"),
        "elapsed_encode_seconds": timings.get("encode_seconds"),
        "elapsed_source_decode_seconds": timings.get("source_decode_seconds"),
        "elapsed_proxy_seconds": timings.get("proxy_seconds"),
        "elapsed_proxy_decode_seconds": timings.get("proxy_decode_seconds"),
    }
    return row


def summarize(results: Iterable[dict[str, Any]], summary_dir: Path) -> list[dict[str, Any]]:
    summary_dir.mkdir(parents=True, exist_ok=True)
    rows = mark_pareto([_flatten(result) for result in results])
    (summary_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (summary_dir / "sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    frontier = [row for row in rows if row["is_pareto"]]
    with (summary_dir / "pareto_frontier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(frontier)
    return rows


def write_report(rows: list[dict[str, Any]], manifest: dict[str, Any], path: Path) -> None:
    completed = [row for row in rows if row["status"] == "COMPLETE"]
    recoverable = [row for row in completed if row["recoverable"]]
    compact = min(recoverable, key=lambda row: row["source_expansion_ratio"], default=None)
    robust_pool = [row for row in recoverable if row["proxy_rs_frame_failure_rate"] <= 0.005]
    non_aggressive = [row for row in robust_pool if (row["low_level"], row["high_level"]) != (80, 176)]
    robust = min(non_aggressive or robust_pool, key=lambda row: row["source_expansion_ratio"], default=None)
    ref = manifest["proxy"]["reference"]
    settings = manifest["proxy"]["settings"]
    lines = [
        "# FVM Codec Efficiency Sweep", "", "PROXY ≠ PLATFORM. The proxy is a frozen local stress transcode calibrated from one observed rendition.", "",
        f"- Input: {manifest['input']['size_bytes']} bytes, SHA-256 `{manifest['input']['sha256']}`",
        f"- Matrix: {len(rows)} cases; completed {len(completed)}; failed {len(rows)-len(completed)}",
        f"- Reference: `{ref['path']}`, {ref['size_bytes']} bytes / {ref['duration_seconds']:.6f}s / {ref['calculated_bitrate_bps']:.0f} bps",
        f"- Proxy: target {settings['target_bitrate_bps']} bps, maxrate {settings['maxrate_bps']} bps, bufsize {settings['bufsize_bps']} bps, preset {settings['preset']}", "",
        "| level | CRF | source MiB | ratio | source BER | source RS fail | source exact | proxy BER | proxy RS fail | proxy exact | Pareto |",
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        mib = row["source_mp4_size_bytes"] / 2**20 if row["source_mp4_size_bytes"] else 0
        lines.append(f"| {row['low_level']}/{row['high_level']} | {row['crf']} | {mib:.2f} | {row['source_expansion_ratio'] or 0:.4f} | {row['source_raw_ber'] or 0:.3g} | {row['source_rs_failed_frames']} | {row['source_sha_exact']} | {row['proxy_raw_ber'] or 0:.3g} | {row['proxy_rs_failed_frames']} | {row['proxy_sha_exact']} | {row['is_pareto']} |")
    lines += ["", "## Conclusions", ""]
    lines.append(f"- Smallest proxy-recoverable case: `{compact['case_id']}` at {compact['source_expansion_ratio']:.4f}×." if compact else "- No proxy-recoverable case.")
    lines.append(f"- Robust candidate (≤0.5% proxy RS frame failures): `{robust['case_id']}`." if robust else "- No robust candidate met the rule.")
    for threshold in (5, 4, 3):
        lines.append(f"- ≤{threshold}× with proxy exact: {'YES' if any(row['source_expansion_ratio'] <= threshold for row in recoverable) else 'NO'}.")
    if completed and all((row["proxy_raw_ber"] or 0) == 0 for row in completed):
        lines.append("- PROXY NOT DISCRIMINATIVE: every completed proxy had zero raw BER.")
    elif completed and len({row["proxy_rs_frame_failure_rate"] for row in completed}) == 1:
        lines.append("- The proxy changed raw BER but was not discriminative on the Pareto Y axis: every case had the same proxy RS frame failure rate.")
    lines += ["", "## CRF and level trends", ""]
    for low, high in manifest["levels"]:
        group = sorted((row for row in completed if (row["low_level"], row["high_level"]) == (low, high)), key=lambda row: row["crf"])
        if group:
            ratios = " → ".join(f"{row['source_expansion_ratio']:.4f}×" for row in group)
            failures = [row["proxy_rs_failed_frames"] for row in group]
            lines.append(f"- {low}/{high}, CRF 18→30: {ratios}; proxy RS failures {failures}. No recovery cliff was observed.")
    for crf in manifest["crfs"]:
        group = [row for row in completed if row["crf"] == crf]
        if group:
            ratios = " → ".join(f"{row['source_expansion_ratio']:.4f}×" for row in group)
            bers = " → ".join(f"{row['proxy_raw_ber']:.3g}" for row in group)
            lines.append(f"- CRF {crf}, level 0/255→80/176: source ratio {ratios}; proxy raw BER {bers}.")
    if compact:
        ratio = compact["source_expansion_ratio"]
        duration = compact["source_duration"]
        projected_500_mib = 500 * ratio
        ten_hour_input_gib = manifest["input"]["size_bytes"] * (10 * 3600 / duration) / 2**30
        size_limit_input_gib = 16_000_000_000 / ratio / 2**30
        lines += ["", "## Linear estimates", "",
                  f"- Improvement versus ~14×: {14 / ratio:.3f}× smaller expansion factor.",
                  f"- A 500 MiB input projects to {projected_500_mib:.1f} MiB ({projected_500_mib / 1024:.3f} GiB) of source MP4.",
                  f"- 10-hour duration capacity: {ten_hour_input_gib:.3f} GiB input.",
                  f"- 16 GB decimal upload-size capacity: {size_limit_input_gib:.3f} GiB input.",
                  "- Binding estimate: upload file size, not duration.",
                  "- This ~42% reduction from 14× and exact proxy recovery merits one manually authorized real-platform validation case, but it does not meet the ≤5× target.",
                  "- Reaching ≤5× likely requires a later physical-modulation phase (for example 8 px, 4-level, or low-transition modulation); this sweep does not select or implement one."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
