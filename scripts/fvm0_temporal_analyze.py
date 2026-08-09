"""Rebuild FVM0-T aggregate diagnostics without decoding video again."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    from fvm0_temporal_common import SOFT_CORE_FIELDS, SOFT_PREFIXES, TRANSITION_FIELDS, aggregate_ratios
except ImportError:
    from scripts.fvm0_temporal_common import SOFT_CORE_FIELDS, SOFT_PREFIXES, TRANSITION_FIELDS, aggregate_ratios

REQUIRED_COLUMNS = {"frame_index", "block_index", "phase", "is_anchor", "is_warmup",
                    "transition_ratio", "flip_count", "bit_errors", "ber", "zero_to_one",
                    "one_to_zero", "changed_bit_errors", "unchanged_bit_errors"}


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists(): raise RuntimeError(f"missing frames CSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and not set(TRANSITION_FIELDS) <= set(reader.fieldnames):
            raise RuntimeError("transition-mask metrics are unavailable; re-run fvm0_temporal_decode with the updated decoder")
        soft_fields={prefix+field for prefix in SOFT_PREFIXES for field in SOFT_CORE_FIELDS}
        if reader.fieldnames and not soft_fields <= set(reader.fieldnames):
            raise RuntimeError("soft fixed-weight metrics are unavailable; re-run fvm0_temporal_decode with the updated decoder")
        if not reader.fieldnames or not REQUIRED_COLUMNS <= set(reader.fieldnames):
            raise RuntimeError("frames CSV is empty or missing required columns")
        rows = list(reader)
    if not rows: raise RuntimeError("frames CSV contains no records")
    for row in rows:
        for key in ("frame_index", "block_index", "phase", "flip_count", "bit_errors", "zero_to_one", "one_to_zero", "changed_bit_errors", "unchanged_bit_errors"):
            row[key] = int(row[key] or 0)
        for key in TRANSITION_FIELDS:
            if key not in ("transition_mask_ber","transition_recall","transition_precision","transition_f1","missed_flip_rate","false_flip_rate","zero_to_one_direction_recall","one_to_zero_direction_recall","transition_direction_accuracy"):
                row[key]=int(row[key] or 0)
        for prefix in SOFT_PREFIXES:
            for key in SOFT_CORE_FIELDS:
                if key not in ("transition_mask_ber","transition_recall","transition_precision","transition_f1","missed_flip_rate","false_flip_rate","swap_error_rate"):
                    row[prefix+key]=int(row[prefix+key] or 0)
        for key in ("is_anchor", "is_warmup"): row[key] = row[key] == "True"
        row["transition_ratio"] = float(row["transition_ratio"]) if row["transition_ratio"] else None
        row["ber"] = float(row["ber"])
    return rows


def probe_frames(video: Path) -> list[dict[str, Any]]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=key_frame,best_effort_timestamp_time,pict_type,pkt_size",
               "-of", "json", str(video)]
    try: completed = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc: raise RuntimeError("ffprobe not found on PATH") from exc
    except subprocess.CalledProcessError as exc: raise RuntimeError(f"ffprobe failed: {exc.stderr.strip()}") from exc
    try: payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc: raise RuntimeError("ffprobe returned malformed JSON") from exc
    if not isinstance(payload.get("frames"), list): raise RuntimeError("ffprobe JSON does not contain a frames list")
    return payload["frames"]


def _plot_basic(directory: Path, ratios: dict[str, Any], luminance: dict[str, Any], score_stats: dict[str, Any]) -> None:
    entries = list(ratios.values()); x = [e["ratio"]*100 for e in entries]
    figures = [
        ("fvm0_temporal_ber_vs_ratio.png", [e["ber"] for e in entries], None, "aggregate BER", "FVM0-T BER by transition ratio"),
        ("fvm0_temporal_changed_vs_unchanged_ber.png", [e["changed_ber"] for e in entries], [e["unchanged_ber"] for e in entries], "changed / unchanged BER", "FVM0-T changed vs unchanged BER"),
        ("fvm0_temporal_theoretical_rate_vs_ratio.png", [e["theoretical_effective_bytes_per_second_including_anchor"]/1000 for e in entries], [e["theoretical_raw_bytes_per_second"]/1000 for e in entries], "KB/s", "Constant-weight theoretical capacity ceiling"),
    ]
    for name, first, second, ylabel, title in figures:
        fig, ax = plt.subplots(); ax.plot(x, first, marker="o", label="primary")
        if second is not None: ax.plot(x, second, marker="o", label="secondary"); ax.legend()
        ax.set(xlabel="transition ratio (%)", ylabel=ylabel, title=title); fig.tight_layout(); fig.savefig(directory/name); plt.close(fig)
    transition_figures=[
        ("fvm0_temporal_transition_mask_ber_vs_ratio.png",[e["ber"] for e in entries],[e["transition_mask_ber"] for e in entries],"absolute BER","transition-mask BER"),
        ("fvm0_temporal_transition_error_types_vs_ratio.png",[e["missed_flip_rate"] for e in entries],[e["false_flip_rate"] for e in entries],"missed flip rate","false flip rate"),
        ("fvm0_temporal_transition_precision_recall.png",[e["transition_precision"] for e in entries],[e["transition_recall"] for e in entries],"precision","recall"),
        ("fvm0_temporal_transition_direction.png",[e["zero_to_one_direction_recall"] for e in entries],[e["one_to_zero_direction_recall"] for e in entries],"0->1 direction recall","1->0 direction recall")]
    for name,first,second,label1,label2 in transition_figures:
        fig,ax=plt.subplots();ax.plot(x,first,marker="o",label=label1);ax.plot(x,second,marker="o",label=label2);ax.set(xlabel="transition ratio (%)",ylabel="rate");ax.legend();fig.tight_layout();fig.savefig(directory/name);plt.close(fig)
    soft_figures=[
        ("fvm0_temporal_soft_mask_ber_vs_ratio.png","transition_mask_ber","mask BER"),
        ("fvm0_temporal_soft_recall_vs_ratio.png","transition_recall","recall"),
        ("fvm0_temporal_soft_precision_vs_ratio.png","transition_precision","precision")]
    for name,metric,ylabel in soft_figures:
        fig,ax=plt.subplots();ax.plot(x,[e[metric] for e in entries],marker="o",label="HARD_XOR");ax.plot(x,[e["soft_abs_"+metric] for e in entries],marker="o",label="ABS_DELTA_TOP_M");ax.plot(x,[e["soft_state_"+metric] for e in entries],marker="o",label="STATE_AWARE_TOP_M");ax.set(xlabel="transition ratio (%)",ylabel=ylabel);ax.legend();fig.tight_layout();fig.savefig(directory/name);plt.close(fig)
    fig,ax=plt.subplots();ax.plot(x,[e["soft_abs_relative_ber_reduction"] for e in entries],marker="o",label="hard -> abs");ax.plot(x,[e["soft_state_relative_ber_reduction"] for e in entries],marker="o",label="hard -> state");ax.set(xlabel="transition ratio (%)",ylabel="relative BER reduction");ax.legend();fig.tight_layout();fig.savefig(directory/"fvm0_temporal_soft_gain_vs_ratio.png");plt.close(fig)
    if score_stats:
        fig,ax=plt.subplots()
        for mode,label in (("abs_delta","ABS transition"),("state_aware","STATE transition")):
            ax.plot(x,[score_stats[str(e["ratio"])][mode]["expected_transition"]["p50"] for e in entries],marker="o",label=label)
            ax.plot(x,[score_stats[str(e["ratio"])][mode]["expected_unchanged"]["p50"] for e in entries],marker="o",linestyle="--",label=label.replace("transition","unchanged"))
        ax.set(xlabel="transition ratio (%)",ylabel="median score",title="Soft score separation diagnostic");ax.legend();fig.tight_layout();fig.savefig(directory/"fvm0_temporal_soft_score_separation.png");plt.close(fig)
    if luminance:
        zero = [luminance[str(e["ratio"])]["expected_zero"]["p99"] for e in entries]
        one = [luminance[str(e["ratio"])]["expected_one"]["p1"] for e in entries]
        fig, ax = plt.subplots(); ax.plot(x, zero, marker="o", label="expected-zero p99"); ax.plot(x, one, marker="o", label="expected-one p1")
        ax.set(xlabel="transition ratio (%)", ylabel="mean cell luminance", title="FVM0-T luminance distribution diagnostic"); ax.legend(); fig.tight_layout(); fig.savefig(directory/"fvm0_temporal_luma_margin_vs_ratio.png"); plt.close(fig)


def _codec_analysis(directory: Path, records: list[dict[str, Any]], frames: list[dict[str, Any]], cells: int) -> dict[str, Any]:
    joined = min(len(records), len(frames)); fields = list(records[0]) + ["key_frame", "pts_time", "pict_type", "pkt_size"]
    enriched = []
    with (directory/"fvm0_temporal_frames_with_codec.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for record, frame in zip(records, frames):
            row = {**record, "key_frame": frame.get("key_frame"), "pts_time": frame.get("best_effort_timestamp_time"), "pict_type": frame.get("pict_type"), "pkt_size": frame.get("pkt_size")}
            writer.writerow(row); enriched.append(row)
    groups = {}
    for row in enriched:
        if row["is_warmup"] or row["is_anchor"]: continue
        groups.setdefault(float(row["transition_ratio"]), []).append(row)
    output = {}
    for ratio in sorted(groups):
        rows = groups[ratio]; sizes = sorted(int(r["pkt_size"]) for r in rows if r["pkt_size"] not in (None, "")); n=len(sizes)
        types = [r["pict_type"] if r["pict_type"] in ("I","P","B") else "other" for r in rows]
        output[str(ratio)] = {"codec_frames":len(rows),"pkt_size_count":n,"mean_pkt_size":sum(sizes)/n if n else None,"median_pkt_size":float(median(sizes)) if n else None,"p10_pkt_size":sizes[max(0,math.ceil(.1*n)-1)] if n else None,"p90_pkt_size":sizes[math.ceil(.9*n)-1] if n else None,"keyframe_count":sum(int(r["key_frame"] or 0)==1 for r in rows),"I_frame_count":types.count("I"),"P_frame_count":types.count("P"),"B_frame_count":types.count("B"),"other_frame_count":types.count("other")}
    x=[float(k)*100 for k in output]; fig,ax=plt.subplots(); ax.plot(x,[v["mean_pkt_size"] for v in output.values()],label="mean"); ax.plot(x,[v["median_pkt_size"] for v in output.values()],label="median"); ax.legend(); ax.set(xlabel="transition ratio (%)",ylabel="packet size bytes"); fig.savefig(directory/"fvm0_temporal_pkt_size_vs_ratio.png"); plt.close(fig)
    fig,ax=plt.subplots()
    for ratio,rows in groups.items():
        points=[(int(r["pkt_size"]),float(r["ber"])) for r in rows if r["pkt_size"] not in (None,"")]; ax.scatter([p[0] for p in points],[p[1] for p in points],label=str(ratio))
    ax.set(xlabel="packet size bytes",ylabel="frame BER",title="Observed diagnostic relationship"); ax.legend(); fig.savefig(directory/"fvm0_temporal_ber_vs_pkt_size.png"); plt.close(fig)
    keyframes=[r for r in enriched if int(r["key_frame"] or 0)==1]
    non_keyframes=[r for r in enriched if int(r["key_frame"] or 0)!=1]
    def ber_summary(rows):
        errors=sum(int(r["bit_errors"]) for r in rows); total=len(rows)*cells
        return {"frames":len(rows),"total_bits":total,"bit_errors":errors,"ber":errors/total if total else None}
    return {"frame_alignment":{"decoded_records":len(records),"ffprobe_frames":len(frames),"joined_frames":joined,"exact_count_match":len(records)==len(frames)},"ratios":output,"keyframes":ber_summary(keyframes),"non_keyframes":ber_summary(non_keyframes)}


def analyze(directory: Path, ffprobe_video: Path | None = None) -> dict[str, Any]:
    records = load_records(directory/"fvm0_temporal_frames.csv")
    try: results = json.loads((directory/"fvm0_temporal_results.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise RuntimeError("missing or malformed FVM0-T results JSON") from exc
    ratios = aggregate_ratios(records, results["matrix"]["cells_per_frame"], results["video"]["expected_fps"], results["experiment"]["block_size"])
    results["ratios"] = ratios; _plot_basic(directory, ratios, results.get("luminance_by_ratio", {}), results.get("soft_score_by_ratio", {}))
    if ffprobe_video:
        frames = probe_frames(ffprobe_video); results["codec_analysis"] = _codec_analysis(directory, records, frames, results["matrix"]["cells_per_frame"])
        if len(frames) != len(records):
            warning = "ffprobe frame count differs from decoded records; index-based codec join may be unreliable after frame insertion/deletion"
            results.setdefault("warnings", []).append(warning); results["warnings"] = list(dict.fromkeys(results["warnings"]))
    entries=list(ratios.values())
    (directory/"fvm0_temporal_ratios.json").write_text(json.dumps({"ratios":entries},indent=2),encoding="utf-8")
    with (directory/"fvm0_temporal_ratios.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(entries[0]));writer.writeheader();writer.writerows(entries)
    temp=directory/"fvm0_temporal_results.json.tmp"; temp.write_text(json.dumps(results,indent=2),encoding="utf-8"); temp.replace(directory/"fvm0_temporal_results.json")
    return results


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("result_dir",type=Path); parser.add_argument("--ffprobe-video",type=Path); args=parser.parse_args(); results=analyze(args.result_dir,args.ffprobe_video)
    def display(value): return "null" if value is None else f"{value:.6g}"
    for entry in results["ratios"].values():
        print(f"ratio {entry['ratio']:.3g}: absolute BER={display(entry['ber'])}, changed BER={display(entry['changed_ber'])}, unchanged BER={display(entry['unchanged_ber'])}, transition-mask BER={display(entry['transition_mask_ber'])}, recall={display(entry['transition_recall'])}, precision={display(entry['transition_precision'])}, F1={display(entry['transition_f1'])}, missed={display(entry['missed_flip_rate'])}, false={display(entry['false_flip_rate'])}, 0->1 recall={display(entry['zero_to_one_direction_recall'])}, 1->0 recall={display(entry['one_to_zero_direction_recall'])}, expected/observed={entry['expected_transition_cells']}/{entry['observed_transition_cells']}")
        print(f"  ABS_DELTA_TOP_M: BER={display(entry['soft_abs_transition_mask_ber'])}, recall={display(entry['soft_abs_transition_recall'])}, precision={display(entry['soft_abs_transition_precision'])}, swap rate={display(entry['soft_abs_swap_error_rate'])}, observed={entry['soft_abs_observed_transition_cells']}, gain={display(entry['soft_abs_relative_ber_reduction'])}")
        print(f"  STATE_AWARE_TOP_M: BER={display(entry['soft_state_transition_mask_ber'])}, recall={display(entry['soft_state_transition_recall'])}, precision={display(entry['soft_state_transition_precision'])}, swap rate={display(entry['soft_state_swap_error_rate'])}, observed={entry['soft_state_observed_transition_cells']}, gain={display(entry['soft_state_relative_ber_reduction'])}")

if __name__ == "__main__": main()
