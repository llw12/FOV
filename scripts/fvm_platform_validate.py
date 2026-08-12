"""CLI for one authorized FVM real-platform validation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvm_platform_validate import run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate one existing FVM source through one authorized upload")
    parser.add_argument("source", type=Path)
    parser.add_argument("--original-file", type=Path, required=True)
    parser.add_argument("--case-result", type=Path, required=True)
    parser.add_argument("--allow-upload", action="store_true")
    parser.add_argument("--biliup", default=os.environ.get("BILIUP_EXE"))
    parser.add_argument("--biliup-cookies", type=Path, default=Path(os.environ["BILIUP_COOKIES"]) if os.environ.get("BILIUP_COOKIES") else None)
    parser.add_argument("--tid", type=int, default=int(os.environ["BILI_TID"]) if os.environ.get("BILI_TID") else None)
    parser.add_argument("--title")
    parser.add_argument("--tag", default="FVM,视频信道,测试")
    parser.add_argument("--desc", default="FVM 6px binary codec robustness validation")
    parser.add_argument("--poll-interval", type=float, default=60)
    parser.add_argument("--approval-timeout", type=float, default=10800)
    parser.add_argument("--rendition-timeout", type=float, default=7200)
    parser.add_argument("--status-max-pages", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    result = run_validation(args.source, args.original_file, args.case_result, allow_upload=args.allow_upload,
        biliup=args.biliup, biliup_cookies=args.biliup_cookies, tid=args.tid, output_root=args.output_root,
        resume_run=args.resume_run, title=args.title, tag=args.tag, desc=args.desc,
        poll_interval=args.poll_interval, approval_timeout=args.approval_timeout,
        rendition_timeout=args.rendition_timeout, status_max_pages=args.status_max_pages)
    print(f"Validation state: {result['state']}")


if __name__ == "__main__": main()
