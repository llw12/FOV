"""Run FVM0 modulo-period BER diagnostics from an existing result directory."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import matplotlib.pyplot as plt
try:
    from fvm0_common import aggregate_gop_phase
except ImportError:
    from scripts.fvm0_common import aggregate_gop_phase


def write_analysis(result_dir: Path, gop_size: int) -> dict:
    with (result_dir / "fvm0_frames.csv").open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    results = json.loads((result_dir / "fvm0_results.json").read_text(encoding="utf-8"))
    analysis = aggregate_gop_phase(records, int(results["matrix"]["bits_per_frame"]), gop_size)
    phases = analysis["phases"]
    with (result_dir / "fvm0_gop_phase.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(phases[0]) if phases else [])
        writer.writeheader(); writer.writerows(phases)
    (result_dir / "fvm0_gop_phase.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    figure, axis = plt.subplots(figsize=(12, 5)); axis.plot([p["phase"] for p in phases], [p["ber"] for p in phases])
    axis.axvline(0, color="gray", linestyle="--", label="phase 0")
    for label, key in (("best", "best_phase_by_ber"), ("worst", "worst_phase_by_ber")):
        phase = analysis[key]; axis.scatter([phase], [phases[phase]["ber"]], label=f"{label} phase {phase}")
    axis.set(xlabel="phase", ylabel="BER", title=f"FVM0 BER by frame modulo {gop_size}"); axis.legend(); figure.tight_layout()
    figure.savefig(result_dir / "fvm0_gop_phase_ber.png", dpi=160); plt.close(figure)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("result_dir", type=Path); parser.add_argument("--gop-size", type=int, required=True)
    args = parser.parse_args()
    if args.gop_size <= 0: parser.error("--gop-size must be positive")
    analysis = write_analysis(args.result_dir, args.gop_size)
    print(f"GOP phase diagnostic complete: best={analysis['best_phase_by_ber']}, worst={analysis['worst_phase_by_ber']}")
    print("IMPORTANT: modulo-GOP analysis is diagnostic only; phase 0 is not proven to be an I-frame.")

if __name__ == "__main__": main()
