"""Scan FOV parameter combinations through the real Bilibili video channel.

Each locally valid case is uploaded with biliup, waited until publication, downloaded
with yt-dlp, and decoded with video2file.py. Every uploaded case creates a real
Bilibili submission; use small targeted scans first and only operate accounts/videos
you are authorized to use.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bilibili_roundtrip import (  # noqa: E402
    DEFAULT_DOWNLOAD_FORMAT,
    biliup_base,
    build_upload_command,
    download_with_ytdlp,
    export_biliup_cookies_to_netscape,
    ffprobe_video as platform_ffprobe_video,
    parse_decoder_output,
    parse_upload_identity,
    recover_bvid_by_title,
    resolve_executable,
    run_logged,
    wait_for_published,
)
from scripts.scan_params import (  # noqa: E402
    FULL_RESOLUTIONS,
    FULL_SYMBOL_SIZES,
    TARGETED_BASE_CASES,
    ScanCase,
    ffprobe_video as source_ffprobe_video,
    parse_case,
    sha256_file,
)

PLATFORM_TARGETED_BASE_CASES = [
    (1920, 1080, 500),
    (1920, 1080, 520),
]


@dataclass
class PlatformScanResult:
    case_id: str
    width: int
    height: int
    fps: int
    symbol_size: int
    repair: float
    input_bytes: int

    source_video: str = ""
    encode_ok: bool = False
    encode_seconds: float | None = None
    encode_error: str = ""
    source_video_bytes: int | None = None
    source_frames: int | None = None
    source_duration_s: float | None = None
    source_bitrate_bps: int | None = None
    effective_file_Bps: float | None = None

    local_ok: bool = False
    local_total_frames: int | None = None
    local_qr_decoded: int | None = None
    local_qr_failed: int | None = None
    local_crc_failed: int | None = None
    local_error: str = ""

    upload_ok: bool = False
    upload_seconds: float | None = None
    aid: int | None = None
    bvid: str = ""
    share_url: str = ""
    review_history: list[dict[str, Any]] = field(default_factory=list)
    published_at: str = ""
    review_error: str = ""

    platform_video: str = ""
    platform_video_bytes: int | None = None
    platform_ffprobe: dict[str, Any] = field(default_factory=dict)
    platform_ok: bool = False
    platform_total_frames: int | None = None
    platform_qr_decoded: int | None = None
    platform_qr_failed: int | None = None
    platform_crc_failed: int | None = None
    platform_post_decode_symbols: int | None = None
    platform_failed_frame_indices: list[int] = field(default_factory=list)
    platform_failed_frame_indices_truncated: bool = False
    platform_blocks: list[dict[str, Any]] = field(default_factory=list)
    platform_original_sha256: str = ""
    platform_recovered_sha256: str = ""
    platform_sha256_match: bool = False
    platform_error: str = ""

    @property
    def platform_erasure_rate(self) -> float | None:
        if self.platform_qr_decoded is None or self.platform_qr_failed is None:
            return None
        total = self.platform_qr_decoded + self.platform_qr_failed
        return self.platform_qr_failed / total if total else None

    @property
    def repair_symbols_received(self) -> int | None:
        values = [
            block.get("received_repair")
            for block in self.platform_blocks
            if isinstance(block.get("received_repair"), int)
        ]
        return sum(values) if values else None


def build_cases(args: argparse.Namespace) -> list[ScanCase]:
    if args.case:
        cases = list(dict.fromkeys(args.case))
    else:
        bases: Iterable[tuple[int, int, int]]
        if args.preset == "platform":
            bases = PLATFORM_TARGETED_BASE_CASES
        elif args.preset == "targeted":
            bases = TARGETED_BASE_CASES
        else:
            bases = [
                (width, height, symbol)
                for width, height in FULL_RESOLUTIONS
                for symbol in FULL_SYMBOL_SIZES
            ]
        cases = [
            ScanCase(width, height, symbol, repair, args.fps)
            for width, height, symbol in bases
            for repair in args.repairs
        ]
    return cases[: args.max_cases] if args.max_cases is not None else cases


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def error_tail(text: str, lines: int = 8) -> str:
    compact = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(compact[-lines:])[-1600:]


def make_title(prefix: str, case: ScanCase, scan_stamp: str, index: int) -> str:
    repair_pct = round(case.repair * 100)
    return (
        f"{prefix} {case.width}x{case.height} s{case.symbol_size} "
        f"r{repair_pct} f{case.fps} {scan_stamp}-{index:02d}"
    )[:80]


def decode_video(
    video: Path,
    run_dir: Path,
    expected_sha256: str,
) -> tuple[bool, dict[str, Any], str]:
    recovered = run_dir / "recovered"
    shutil.rmtree(recovered, ignore_errors=True)
    recovered.mkdir(parents=True, exist_ok=True)
    completed = run_logged(
        [sys.executable, "video2file.py", str(video), str(recovered)],
        run_dir / "decode.log",
    )
    stats = parse_decoder_output(completed.output)
    stats["returncode"] = completed.returncode

    errors: list[str] = []
    if completed.returncode != 0:
        errors.append(f"decoder exited {completed.returncode}")
    if not stats.get("file_fully_recovered"):
        errors.append("decoder did not report full recovery")
    if not stats.get("sha256_match"):
        errors.append("decoder SHA256 mismatch")
    if stats.get("original_sha256") != expected_sha256:
        errors.append("metadata SHA256 differs from scan input")
    if stats.get("recovered_sha256") != expected_sha256:
        errors.append("recovered SHA256 differs from scan input")
    if errors:
        tail = error_tail(completed.output)
        if tail:
            errors.append(tail)
        return False, stats, " | ".join(errors)
    return True, stats, ""


def write_case_result(case_dir: Path, result: PlatformScanResult) -> None:
    (case_dir / "result.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def platform_cell(item: PlatformScanResult) -> str:
    if item.platform_ok:
        if item.platform_qr_failed is not None and item.platform_qr_decoded is not None:
            total = item.platform_qr_failed + item.platform_qr_decoded
            return f"PASS {item.platform_qr_failed}/{total}"
        return "PASS"
    if item.review_error:
        return "REVIEW FAIL"
    if item.encode_ok and not item.local_ok:
        return "LOCAL FAIL"
    if item.upload_ok:
        return "PLATFORM FAIL"
    return "FAIL"


def save_results(results: list[PlatformScanResult], run_dir: Path) -> None:
    (run_dir / "results.json").write_text(
        json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for item in results:
        row = asdict(item)
        row["platform_erasure_rate"] = item.platform_erasure_rate
        row["repair_symbols_received"] = item.repair_symbols_received
        for key in ("review_history", "platform_ffprobe", "platform_blocks"):
            row[key] = json.dumps(row[key], ensure_ascii=False)
        row["platform_failed_frame_indices"] = ",".join(
            map(str, item.platform_failed_frame_indices)
        )
        rows.append(row)

    if rows:
        with (run_dir / "results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# FOV Bilibili real-channel parameter scan",
        "",
        "| case | local | throughput | BVID | platform | QR erasure | repair received | decoded frame |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        throughput = (
            f"{item.effective_file_Bps / 1000:.2f} KB/s"
            if item.effective_file_Bps
            else "-"
        )
        erasure = (
            f"{item.platform_erasure_rate * 100:.2f}%"
            if item.platform_erasure_rate is not None
            else "-"
        )
        decoded_frames = ",".join(
            str(block["decoded_at_frame"])
            for block in item.platform_blocks
            if block.get("decoded_at_frame") is not None
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.case_id}`",
                    "PASS" if item.local_ok else "FAIL",
                    throughput,
                    item.bvid or "-",
                    platform_cell(item),
                    erasure,
                    (
                        str(item.repair_symbols_received)
                        if item.repair_symbols_received is not None
                        else "-"
                    ),
                    decoded_frames or "-",
                ]
            )
            + " |"
        )

    passed = sorted(
        (item for item in results if item.platform_ok),
        key=lambda item: -(item.effective_file_Bps or 0),
    )
    lines.extend(["", "## Passed cases", ""])
    if not passed:
        lines.append("No case has passed the real Bilibili round trip yet.")
    for item in passed:
        erasure = (
            f"{item.platform_erasure_rate * 100:.2f}%"
            if item.platform_erasure_rate is not None
            else "-"
        )
        lines.append(
            f"- `{item.case_id}` — "
            f"{(item.effective_file_Bps or 0) / 1000:.2f} KB/s — "
            f"QR erasure `{erasure}` — `{item.bvid}`"
        )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def setup_download_cookie(
    *,
    biliup_cookies: Path,
    browser: str | None,
    explicit_cookie: Path | None,
    anonymous: bool,
) -> tuple[Path | None, Path | None, str]:
    if browser:
        return None, None, f"browser:{browser}"
    if explicit_cookie:
        return explicit_cookie, None, "explicit-cookie-file"
    if anonymous:
        return None, None, "anonymous"

    temp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        prefix="fov-biliup-scan-",
        delete=False,
    )
    temp.close()
    path = Path(temp.name)
    try:
        count = export_biliup_cookies_to_netscape(biliup_cookies, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path, path, f"biliup-cookies:{count}"


def run_case(
    case: ScanCase,
    result: PlatformScanResult,
    *,
    index: int,
    total: int,
    input_path: Path,
    expected_sha256: str,
    run_dir: Path,
    scan_stamp: str,
    title_prefix: str,
    executable: str,
    biliup_cookies: Path,
    tid: int,
    tag: str,
    desc: str,
    copyright_value: int,
    submit: str,
    limit: int,
    cover: Path | None,
    poll_interval: float,
    approval_timeout: float,
    status_max_pages: int,
    download_format: str,
    browser: str | None,
    download_cookie: Path | None,
    download_retries: int,
    verbose: bool,
) -> None:
    case_dir = run_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_video = case_dir / "source.mp4"
    result.source_video = relative(source_video)

    print(f"\n[{index}/{total}] {case.case_id}")
    print("[1/5] encode source")
    encoded = run_logged(
        [
            sys.executable,
            "file2video.py",
            str(input_path),
            str(source_video),
            "--symbol-size",
            str(case.symbol_size),
            "--repair",
            str(case.repair),
            "--fps",
            str(case.fps),
            "--width",
            str(case.width),
            "--height",
            str(case.height),
        ],
        case_dir / "encode.log",
        echo=verbose,
    )
    result.encode_seconds = round(encoded.elapsed_s, 3)
    if encoded.returncode or not source_video.is_file():
        result.encode_error = error_tail(encoded.output)
        write_case_result(case_dir, result)
        raise RuntimeError(f"encode failed: {result.encode_error}")
    result.encode_ok = True

    try:
        probe = source_ffprobe_video(source_video)
        result.source_video_bytes = probe["size_bytes"]
        result.source_frames = probe["frames"]
        result.source_duration_s = probe["duration_s"]
        result.source_bitrate_bps = probe["bitrate_bps"]
        if result.source_duration_s:
            result.effective_file_Bps = result.input_bytes / result.source_duration_s
    except Exception as exc:
        result.encode_error = f"ffprobe warning: {type(exc).__name__}: {exc}"
    write_case_result(case_dir, result)

    print("[2/5] local decode verification")
    try:
        local_ok, local_stats, local_error = decode_video(
            source_video, case_dir / "local", expected_sha256
        )
        result.local_ok = local_ok
        result.local_total_frames = local_stats.get("total_frames")
        result.local_qr_decoded = local_stats.get("qr_decoded")
        result.local_qr_failed = local_stats.get("qr_failed")
        result.local_crc_failed = local_stats.get("crc_failed")
        if not local_ok:
            raise RuntimeError(local_error)
    except Exception as exc:
        result.local_error = f"{type(exc).__name__}: {exc}"
        write_case_result(case_dir, result)
        raise RuntimeError(f"local validation failed: {result.local_error}") from exc
    write_case_result(case_dir, result)

    print("[3/5] biliup upload + wait for publication")
    title = make_title(title_prefix, case, scan_stamp, index)
    uploaded = run_logged(
        build_upload_command(
            executable,
            biliup_cookies,
            source_video,
            title=title,
            tid=tid,
            tag=tag,
            desc=desc,
            copyright_value=copyright_value,
            submit=submit,
            limit=limit,
            cover=cover,
        ),
        case_dir / "upload.log",
        echo=verbose,
    )
    result.upload_seconds = round(uploaded.elapsed_s, 3)
    if uploaded.returncode:
        write_case_result(case_dir, result)
        raise RuntimeError(
            f"biliup upload failed: {error_tail(uploaded.output)}"
        )

    identity = parse_upload_identity(uploaded.output)
    if identity is None:
        bvid = recover_bvid_by_title(
            executable,
            biliup_cookies,
            title,
            status_max_pages,
            case_dir / "review.log",
        )
        if not bvid:
            write_case_result(case_dir, result)
            raise RuntimeError("upload succeeded but BVID could not be discovered")
        result.bvid = bvid
    else:
        result.bvid = identity.bvid
        result.aid = identity.aid

    result.upload_ok = True
    result.share_url = f"https://www.bilibili.com/video/{result.bvid}"
    write_case_result(case_dir, result)
    print(f"  bvid: {result.bvid}")

    def persist_review(history: list[dict[str, Any]]) -> None:
        result.review_history = history
        write_case_result(case_dir, result)

    try:
        history = wait_for_published(
            executable,
            biliup_cookies,
            result.bvid,
            poll_interval=poll_interval,
            timeout=approval_timeout,
            max_pages=status_max_pages,
            log_path=case_dir / "review.log",
            on_poll=persist_review,
        )
        result.review_history = history
        result.published_at = history[-1]["checked_at"]
    except Exception as exc:
        result.review_error = f"{type(exc).__name__}: {exc}"
        write_case_result(case_dir, result)
        raise
    write_case_result(case_dir, result)

    show = run_logged(
        biliup_base(executable, biliup_cookies) + ["show", result.bvid],
        case_dir / "show.log",
        echo=verbose,
    )
    if show.returncode:
        print("  warning: biliup show failed; continuing")

    print("[4/5] download Bilibili rendition")
    platform_dir = case_dir / "platform"
    platform_video = platform_dir / "platform.mp4"
    try:
        download_with_ytdlp(
            result.share_url,
            platform_video,
            format_selector=download_format,
            browser=browser,
            cookies=download_cookie,
            retries=download_retries,
        )
        result.platform_video = relative(platform_video)
        result.platform_video_bytes = platform_video.stat().st_size
        try:
            result.platform_ffprobe = platform_ffprobe_video(platform_video)
        except Exception as exc:
            result.platform_error = f"ffprobe warning: {type(exc).__name__}: {exc}"
    except Exception as exc:
        result.platform_error = f"{type(exc).__name__}: {exc}"
        write_case_result(case_dir, result)
        raise
    write_case_result(case_dir, result)

    print("[5/5] decode Bilibili rendition")
    try:
        platform_ok, stats, decode_error = decode_video(
            platform_video, platform_dir, expected_sha256
        )
        result.platform_total_frames = stats.get("total_frames")
        result.platform_qr_decoded = stats.get("qr_decoded")
        result.platform_qr_failed = stats.get("qr_failed")
        result.platform_crc_failed = stats.get("crc_failed")
        result.platform_post_decode_symbols = stats.get("post_decode_symbols")
        result.platform_failed_frame_indices = stats.get(
            "failed_frame_indices", []
        )
        result.platform_failed_frame_indices_truncated = bool(
            stats.get("failed_frame_indices_truncated")
        )
        result.platform_blocks = stats.get("blocks", [])
        result.platform_original_sha256 = stats.get("original_sha256") or ""
        result.platform_recovered_sha256 = stats.get("recovered_sha256") or ""
        result.platform_sha256_match = bool(stats.get("sha256_match"))
        result.platform_ok = platform_ok
        if not platform_ok:
            raise RuntimeError(decode_error)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        result.platform_error = (
            f"{result.platform_error} | {message}"
            if result.platform_error
            else message
        )
        write_case_result(case_dir, result)
        raise
    write_case_result(case_dir, result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode FOV cases, upload each with biliup, download the published "
            "rendition, and verify recovery"
        )
    )
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "test.zip")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--preset",
        choices=("platform", "targeted", "full"),
        default="platform",
        help=(
            "platform=1080p 500/520B; targeted/full reuse scan_params.py bases"
        ),
    )
    parser.add_argument(
        "--repairs",
        type=float,
        nargs="+",
        default=[0.20],
        help="repair ratios crossed with preset cases",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--case",
        type=parse_case,
        action="append",
        help="WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]; repeatable",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print planned real submissions without encoding/uploading",
    )

    parser.add_argument(
        "--biliup",
        default=os.environ.get("BILIUP_EXE", "biliup"),
        help="biliup executable path/name; or BILIUP_EXE",
    )
    parser.add_argument(
        "--biliup-cookies",
        type=Path,
        default=(
            Path(os.environ["BILIUP_COOKIES"])
            if os.environ.get("BILIUP_COOKIES")
            else None
        ),
        help="biliup cookies.json; or BILIUP_COOKIES",
    )
    parser.add_argument(
        "--tid",
        type=int,
        default=(
            int(os.environ["BILI_TID"])
            if os.environ.get("BILI_TID")
            else None
        ),
        help="Bilibili category id; or BILI_TID",
    )
    parser.add_argument("--title-prefix", default="FOV 参数扫描")
    parser.add_argument("--tag", default="FOV,二维码,测试")
    parser.add_argument("--desc", default="FOV 视频信道自动化参数扫描实验")
    parser.add_argument("--copyright", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--submit",
        choices=("app", "web", "b-cut-android"),
        default="app",
    )
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--cover", type=Path, default=None)

    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--approval-timeout", type=float, default=7200.0)
    parser.add_argument("--status-max-pages", type=int, default=3)
    parser.add_argument("--download-format", default=DEFAULT_DOWNLOAD_FORMAT)
    parser.add_argument("--cookies-from-browser", default=None)
    parser.add_argument("--yt-dlp-cookies", type=Path, default=None)
    parser.add_argument("--anonymous-download", action="store_true")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument(
        "--case-delay",
        type=float,
        default=5.0,
        help="seconds between completed cases",
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file() or input_path.stat().st_size <= 0:
        raise SystemExit(f"input file missing/empty: {input_path}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise SystemExit("ffmpeg and ffprobe are required on PATH")
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "yt-dlp is required; run: python -m pip install -r requirements.txt"
        ) from exc

    if any(not 0 <= repair <= 5 for repair in args.repairs):
        raise SystemExit("repair ratios must be within 0..5")
    if args.fps <= 0 or args.limit <= 0 or args.status_max_pages <= 0:
        raise SystemExit("fps/limit/status-max-pages must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        raise SystemExit("--max-cases must be positive")
    if args.poll_interval < 10:
        raise SystemExit("--poll-interval must be >= 10 seconds")
    if (
        args.approval_timeout <= 0
        or args.download_retries <= 0
        or args.case_delay < 0
    ):
        raise SystemExit("timeout/retry/delay values are invalid")

    cases = build_cases(args)
    if not cases:
        raise SystemExit("no scan cases selected")
    print("Planned Bilibili submissions:")
    for index, case in enumerate(cases, 1):
        print(f"  {index}. {case.case_id}")
    if args.plan_only:
        return

    executable = resolve_executable(args.biliup)
    if args.biliup_cookies is None:
        raise SystemExit("missing --biliup-cookies (or BILIUP_COOKIES)")
    biliup_cookies = args.biliup_cookies.expanduser().resolve()
    if not biliup_cookies.is_file():
        raise SystemExit(f"biliup cookie file not found: {biliup_cookies}")
    if args.tid is None:
        raise SystemExit("missing --tid (or BILI_TID)")

    cover = args.cover.expanduser().resolve() if args.cover else None
    if cover is not None and not cover.is_file():
        raise SystemExit(f"cover not found: {cover}")
    explicit_cookie = (
        args.yt_dlp_cookies.expanduser().resolve()
        if args.yt_dlp_cookies
        else None
    )
    if explicit_cookie is not None and not explicit_cookie.is_file():
        raise SystemExit(f"yt-dlp cookie file not found: {explicit_cookie}")

    scan_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_root.expanduser().resolve()
        / f"bilibili-scan-{scan_stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    expected_sha256 = sha256_file(input_path)

    download_cookie: Path | None = None
    temporary_cookie: Path | None = None
    try:
        download_cookie, temporary_cookie, download_auth = setup_download_cookie(
            biliup_cookies=biliup_cookies,
            browser=args.cookies_from_browser,
            explicit_cookie=explicit_cookie,
            anonymous=args.anonymous_download,
        )
        config = {
            "input": str(input_path),
            "input_bytes": input_path.stat().st_size,
            "input_sha256": expected_sha256,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "preset": args.preset,
            "cases": [asdict(case) for case in cases],
            "biliup_executable": executable,
            "tid": args.tid,
            "title_prefix": args.title_prefix,
            "submit": args.submit,
            "download_format": args.download_format,
            "download_auth": download_auth,
            "poll_interval": args.poll_interval,
            "approval_timeout": args.approval_timeout,
        }
        (run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(
            f"\nFOV Bilibili real-channel scan\n"
            f"Input: {input_path}\n"
            f"SHA256: {expected_sha256}\n"
            f"Cases: {len(cases)}\n"
            f"Run: {run_dir}"
        )

        results: list[PlatformScanResult] = []
        for index, case in enumerate(cases, 1):
            item = PlatformScanResult(
                case_id=case.case_id,
                width=case.width,
                height=case.height,
                fps=case.fps,
                symbol_size=case.symbol_size,
                repair=case.repair,
                input_bytes=input_path.stat().st_size,
            )
            try:
                run_case(
                    case,
                    item,
                    index=index,
                    total=len(cases),
                    input_path=input_path,
                    expected_sha256=expected_sha256,
                    run_dir=run_dir,
                    scan_stamp=scan_stamp,
                    title_prefix=args.title_prefix,
                    executable=executable,
                    biliup_cookies=biliup_cookies,
                    tid=args.tid,
                    tag=args.tag,
                    desc=args.desc,
                    copyright_value=args.copyright,
                    submit=args.submit,
                    limit=args.limit,
                    cover=cover,
                    poll_interval=args.poll_interval,
                    approval_timeout=args.approval_timeout,
                    status_max_pages=args.status_max_pages,
                    download_format=args.download_format,
                    browser=args.cookies_from_browser,
                    download_cookie=download_cookie,
                    download_retries=args.download_retries,
                    verbose=args.verbose,
                )
                print(
                    f"  PASS: {item.bvid} "
                    f"QR failed={item.platform_qr_failed} "
                    f"repair received={item.repair_symbols_received}"
                )
            except KeyboardInterrupt:
                results.append(item)
                save_results(results, run_dir)
                print("\nInterrupted; partial results have been preserved.")
                raise SystemExit(130)
            except Exception as exc:
                print(f"  FAIL: {type(exc).__name__}: {exc}")
                if args.stop_on_error:
                    results.append(item)
                    save_results(results, run_dir)
                    raise

            results.append(item)
            save_results(results, run_dir)
            if index < len(cases) and args.case_delay:
                print(f"  next case in {args.case_delay:g}s")
                time.sleep(args.case_delay)

        passed = sum(item.platform_ok for item in results)
        print(
            f"\nDone: {passed}/{len(results)} cases passed "
            f"the real Bilibili round trip"
        )
        print(f"Results: {run_dir / 'results.csv'}")
        print(f"Summary: {run_dir / 'summary.md'}")
    finally:
        if temporary_cookie is not None:
            temporary_cookie.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
