"""CLI for the bounded FVM real-platform adaptive cliff search."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvm_platform_cliff import SearchOptions, run_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--known-fail-run", type=Path, required=True)
    parser.add_argument("--biliup", default=os.environ.get("BILIUP_EXE"), required=not os.environ.get("BILIUP_EXE"))
    parser.add_argument("--biliup-cookies", type=Path, default=os.environ.get("BILIUP_COOKIES"), required=not os.environ.get("BILIUP_COOKIES"))
    parser.add_argument("--tid", type=int, default=os.environ.get("BILI_TID"), required=not os.environ.get("BILI_TID"))
    parser.add_argument("--allow-upload", action="store_true")
    parser.add_argument("--max-new-uploads", type=int, default=8)
    parser.add_argument("--upload-cooldown", type=float, default=120)
    parser.add_argument("--poll-interval", type=float, default=60)
    parser.add_argument("--approval-timeout", type=float, default=10800)
    parser.add_argument("--rendition-timeout", type=float, default=7200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_uploads < 1 or args.max_new_uploads > 8: parser.error("max-new-uploads must be 1..8")
    if args.upload_cooldown < 120: parser.error("upload-cooldown must be at least 120 seconds")
    options = SearchOptions(args.biliup, args.biliup_cookies, args.tid, args.max_new_uploads,
                            args.upload_cooldown, args.poll_interval, args.approval_timeout, args.rendition_timeout)
    state = run_search(args.root, args.input, args.sweep_root, args.known_fail_run, options,
                       allow_upload=args.allow_upload, resume=args.resume)
    print(f"FVM platform cliff search: {state['status']}")


if __name__ == "__main__": main()
