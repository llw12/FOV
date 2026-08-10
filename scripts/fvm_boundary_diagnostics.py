"""CLI for offline paired FVM source/platform boundary diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from fvm_boundary_analysis import analyze
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fvm_boundary_analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FVM source/repair boundaries offline")
    parser.add_argument("source", type=Path)
    parser.add_argument("platform", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-symbols", type=int, default=8)
    parser.add_argument("--full-frame-csv", action="store_true")
    args = parser.parse_args()
    result = analyze(
        args.source,
        args.platform,
        args.output_dir,
        window_symbols=args.window_symbols,
        full_frame_csv=args.full_frame_csv,
    )
    mapping = result["mapping"]
    boundary = result["boundary"]
    print(
        f"Boundary analysis complete: {args.output_dir}\n"
        f"Platform RS failures: {result['platform']['rs_failed_frames']}\n"
        f"Confidently mapped failures: {mapping.get('failures_mapped', 0)}\n"
        f"Ambiguous failures: {mapping.get('failures_ambiguous', 0)}\n"
        f"Last-source failures: {boundary['last_source_rs_failures']}/{boundary['last_source_frames']}"
    )


if __name__ == "__main__":
    main()
