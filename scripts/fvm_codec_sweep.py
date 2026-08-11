"""CLI for the completely local FVM codec efficiency sweep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvm_codec_sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local FVM codec efficiency sweep")
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--proxy-reference", type=Path)
    parser.add_argument("--proxy-bitrate")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-proxy", action="store_true")
    parser.add_argument("--keep-recovered", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--skip-preflight", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_sweep(args.output, input_path=args.input, proxy_reference=args.proxy_reference, proxy_bitrate=args.proxy_bitrate,
              resume=args.resume, keep_proxy=args.keep_proxy, keep_recovered=args.keep_recovered, skip_proxy=args.skip_proxy,
              selected_ids=set(args.cases) if args.cases else None, preflight=not args.skip_preflight)


if __name__ == "__main__": main()
