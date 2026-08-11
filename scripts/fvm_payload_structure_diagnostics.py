"""CLI for offline FVM payload and physical-structure diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fvm_payload_structure_analysis import analyze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pair source/platform FVM videos for payload, structure, and raw-BER diagnostics"
    )
    parser.add_argument("source_mp4", type=Path)
    parser.add_argument("platform_mp4", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--original-file", type=Path)
    parser.add_argument("--boundary-summary", type=Path)
    parser.add_argument("--window-symbols", type=int, default=8)
    parser.add_argument("--dump-target-bytes", action="store_true")
    args = parser.parse_args()
    result = analyze(
        args.source_mp4,
        args.platform_mp4,
        args.output_dir,
        original_file=args.original_file,
        boundary_summary=args.boundary_summary,
        window_symbols=args.window_symbols,
        dump_target_bytes=args.dump_target_bytes,
    )
    print(f"Payload structure analysis complete: {args.output_dir}")
    print(f"Source frames: {result['source_gate']['observed_frames']}")
    print(f"Platform RS failures: {result['platform']['rs_failures']}")
    print(f"Confidently mapped failures: {result['mapping']['failures_confidently_mapped']}")
    print(f"Boundary cross-check: {result['boundary_crosscheck'].get('matches')}")


if __name__ == "__main__":
    main()
