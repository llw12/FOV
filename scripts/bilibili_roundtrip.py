"""Automate an FOV round trip through a normal Bilibili account via biliup.

Use only accounts/videos you are authorized to operate. biliup relies on non-public
client/web APIs, so this integration may need updates when Bilibili changes them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOWNLOAD_FORMAT = (
    "bestvideo[height<=720][vcodec^=avc1]/"
    "bestvideo[height<=720]/best[height<=720]/best"
)
BVID_RE = re.compile(r"\b(BV[0-9A-Za-z]{10,})\b")
UPLOAD_BVID_RE = re.compile(r'"bvid"\s*:\s*String\("(?P<bvid>BV[0-9A-Za-z]+)"\)')
UPLOAD_AID_RE = re.compile(r'"aid"\s*:\s*Number\((?P<aid>\d+)\)')


@dataclass(frozen=True)
class UploadIdentity:
    bvid: str
    aid: int | None = None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    elapsed_s: float


def parse_upload_identity(text: str) -> UploadIdentity | None:
    match = UPLOAD_BVID_RE.search(text) or BVID_RE.search(text)
    if not match:
        return None
    bvid = match.groupdict().get("bvid") or match.group(1)
    aid_match = UPLOAD_AID_RE.search(text)
    aid = int(aid_match.group("aid")) if aid_match else None
    return UploadIdentity(bvid, aid)


def parse_list_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(BV[0-9A-Za-z]{10,})\s+(.*\S)\s*$", line)
        if match:
            entries.append((match.group(1), match.group(2)))
    return entries


def bvid_in_list(text: str, bvid: str) -> bool:
    return any(value == bvid for value, _ in parse_list_entries(text))


def find_bvid_by_title(text: str, title: str) -> str | None:
    for bvid, remainder in parse_list_entries(text):
        if title in remainder:
            return bvid
    return None


def export_biliup_cookies_to_netscape(source: Path, target: Path) -> int:
    """Convert biliup LoginInfo.cookie_info.cookies to a temporary yt-dlp cookie jar."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    cookies = (payload.get("cookie_info") or {}).get("cookies") or []
    if not isinstance(cookies, list) or not cookies:
        raise ValueError("biliup cookie file has no cookie_info.cookies entries")

    lines = [
        "# Netscape HTTP Cookie File",
        "# Temporary FOV export from biliup cookies.json",
    ]
    count = 0
    for item in cookies:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        value = str(item.get("value") or "")
        if not name or any(ch in name + value for ch in "\t\r\n"):
            continue

        domain = str(item.get("domain") or ".bilibili.com")
        if domain == "bilibili.com":
            domain = ".bilibili.com"
        path = str(item.get("path") or "/")
        secure = "TRUE" if bool(item.get("secure", True)) else "FALSE"
        try:
            expires = max(0, int(float(item.get("expires", 0))))
        except (TypeError, ValueError):
            expires = 0

        lines.append(
            "\t".join(
                [
                    domain,
                    "TRUE" if domain.startswith(".") else "FALSE",
                    path,
                    secure,
                    str(expires),
                    name,
                    value,
                ]
            )
        )
        count += 1

    if not count:
        raise ValueError("biliup cookie file has no usable cookies")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def parse_decoder_output(text: str) -> dict[str, Any]:
    def metric(label: str) -> int | None:
        match = re.search(
            rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", text, re.MULTILINE
        )
        return int(match.group(1)) if match else None

    def text_metric(label: str) -> str | None:
        match = re.search(
            rf"^\s*{re.escape(label)}:\s*(.*?)\s*$", text, re.MULTILINE
        )
        return match.group(1) if match else None

    failed = text_metric("failed frame indices (0-based)")
    failed_indices = (
        []
        if not failed or failed == "-"
        else [int(value.strip()) for value in failed.split(",") if value.strip()]
    )

    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    labels = {
        "K": "k",
        "N": "n",
        "received unique": "received_unique",
        "received source": "received_source",
        "received repair": "received_repair",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip()
        block_match = re.fullmatch(r"Block\s+(\d+):", line)
        if block_match:
            if current is not None:
                blocks.append(current)
            current = {"block_id": int(block_match.group(1))}
            continue
        if current is None:
            continue

        matched = False
        for label, key in labels.items():
            value_match = re.fullmatch(rf"{re.escape(label)}:\s*(\d+)", line)
            if value_match:
                current[key] = int(value_match.group(1))
                matched = True
                break
        if not matched:
            decoded_match = re.fullmatch(r"decoded at frame:\s*(\d+|unknown)", line)
            if decoded_match:
                value = decoded_match.group(1)
                current["decoded_at_frame"] = (
                    None if value == "unknown" else int(value)
                )
    if current is not None:
        blocks.append(current)

    original_sha = text_metric("Original SHA256")
    recovered_sha = text_metric("Recovered SHA256")
    return {
        "total_frames": metric("Total frames"),
        "qr_decoded": metric("decoded"),
        "qr_failed": metric("failed"),
        "failed_frame_indices": failed_indices,
        "failed_frame_indices_truncated": (
            text_metric("failed frame indices truncated") == "yes"
        ),
        "valid_meta": metric("valid META"),
        "crc_failed": metric("CRC failed"),
        "post_decode_symbols": metric("post-decode symbols"),
        "blocks": blocks,
        "original_sha256": original_sha,
        "recovered_sha256": recovered_sha,
        "sha256_match": bool(
            original_sha and recovered_sha and original_sha == recovered_sha
        ),
        "file_fully_recovered": "[OK] File fully recovered" in text,
    }


def run_logged(
    command: list[str],
    log_path: Path,
    *,
    append: bool = False,
    echo: bool = True,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines: list[str] = []
    with log_path.open(
        "a" if append else "w", encoding="utf-8", errors="replace"
    ) as log:
        if append and log.tell():
            log.write("\n")
        log.write("$ " + subprocess.list2cmdline(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line)
            log.write(line)
            log.flush()
            if echo:
                print(line, end="")
        returncode = process.wait()
    return CommandResult(
        returncode, "".join(lines), time.perf_counter() - started
    )


def resolve_executable(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    found = shutil.which(value)
    if found:
        return found
    raise SystemExit(f"executable not found: {value}")


def biliup_base(executable: str, cookies: Path) -> list[str]:
    return [executable, "-u", str(cookies)]


def build_upload_command(
    executable: str,
    cookies: Path,
    video: Path,
    *,
    title: str,
    tid: int,
    tag: str,
    desc: str,
    copyright_value: int,
    submit: str,
    limit: int,
    cover: Path | None,
) -> list[str]:
    command = biliup_base(executable, cookies) + [
        "upload",
        str(video),
        "--title",
        title,
        "--tid",
        str(tid),
        "--tag",
        tag,
        "--desc",
        desc,
        "--copyright",
        str(copyright_value),
        "--submit",
        submit,
        "--limit",
        str(limit),
    ]
    if cover:
        command += ["--cover", str(cover)]
    return command


def list_status(
    executable: str,
    cookies: Path,
    flag: str,
    max_pages: int,
    log_path: Path,
) -> CommandResult:
    return run_logged(
        biliup_base(executable, cookies)
        + ["list", flag, "--max-pages", str(max_pages)],
        log_path,
        append=True,
        echo=False,
    )


def recover_bvid_by_title(
    executable: str,
    cookies: Path,
    title: str,
    max_pages: int,
    log_path: Path,
) -> str | None:
    for flag in ("--is-pubing", "--pubed", "--not-pubed"):
        result = list_status(executable, cookies, flag, max_pages, log_path)
        if result.returncode == 0:
            found = find_bvid_by_title(result.output, title)
            if found:
                return found
    return None


def wait_for_published(
    executable: str,
    cookies: Path,
    bvid: str,
    *,
    poll_interval: float,
    timeout: float,
    max_pages: int,
    log_path: Path,
    on_poll: Any | None = None,
) -> list[dict[str, Any]]:
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    poll_number = 0

    while True:
        poll_number += 1
        checked_at = datetime.now().isoformat(timespec="seconds")
        published = list_status(
            executable, cookies, "--pubed", max_pages, log_path
        )
        if published.returncode == 0 and bvid_in_list(published.output, bvid):
            history.append(
                {"checked_at": checked_at, "poll": poll_number, "state": "published"}
            )
            if on_poll:
                on_poll(history)
            print(f"  review: published ({bvid})")
            return history

        rejected = list_status(
            executable, cookies, "--not-pubed", max_pages, log_path
        )
        if rejected.returncode == 0 and bvid_in_list(rejected.output, bvid):
            history.append(
                {"checked_at": checked_at, "poll": poll_number, "state": "not_pubed"}
            )
            if on_poll:
                on_poll(history)
            raise RuntimeError(f"Bilibili archive {bvid} entered --not-pubed")

        state = (
            "poll_error"
            if published.returncode or rejected.returncode
            else "waiting"
        )
        history.append(
            {
                "checked_at": checked_at,
                "poll": poll_number,
                "state": state,
                "pubed_command_ok": published.returncode == 0,
                "not_pubed_command_ok": rejected.returncode == 0,
            }
        )
        if on_poll:
            on_poll(history)

        if time.monotonic() - started >= timeout:
            raise TimeoutError(
                f"Bilibili review did not finish within {timeout:.0f}s for {bvid}"
            )
        print(f"  review: {state}; retry in {poll_interval:g}s")
        time.sleep(poll_interval)


def download_with_ytdlp(
    url: str,
    target: Path,
    *,
    format_selector: str,
    browser: str | None,
    cookies: Path | None,
    retries: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    output_template = target.with_name(target.stem + ".%(ext)s")
    for attempt in range(1, retries + 1):
        target.unlink(missing_ok=True)
        command = [
            sys.executable,
            "-m",
            "yt_dlp",
            "--no-playlist",
            "--no-part",
            "--force-overwrites",
            "-f",
            format_selector,
            "--remux-video",
            "mp4",
            "-o",
            str(output_template),
        ]
        if browser:
            command += ["--cookies-from-browser", browser]
        if cookies:
            command += ["--cookies", str(cookies)]
        command.append(url)

        result = run_logged(
            command, target.parent / f"download_attempt_{attempt}.log"
        )
        if result.returncode == 0 and target.is_file() and target.stat().st_size:
            return
        if attempt < retries:
            wait_seconds = min(60, 10 * attempt)
            print(
                f"  download attempt {attempt} failed; retry in {wait_seconds}s"
            )
            time.sleep(wait_seconds)
    raise RuntimeError(f"yt-dlp failed after {retries} attempts")


def ffprobe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,width,height,pix_fmt,bit_rate,nb_frames,avg_frame_rate:"
            "format=duration,size,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return json.loads(completed.stdout)


def decode_downloaded(video: Path, run_dir: Path) -> dict[str, Any]:
    recovered = run_dir / "recovered"
    shutil.rmtree(recovered, ignore_errors=True)
    recovered.mkdir(parents=True)
    result = run_logged(
        [sys.executable, "video2file.py", str(video), str(recovered)],
        run_dir / "decode.log",
    )
    stats = parse_decoder_output(result.output)
    stats["returncode"] = result.returncode
    if (
        result.returncode != 0
        or not stats["sha256_match"]
        or not stats["file_fully_recovered"]
    ):
        raise RuntimeError(
            f"FOV decode verification failed; see {run_dir / 'decode.log'}"
        )
    return stats


def write_result(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Upload FOV with biliup, wait for publication, download the platform "
            "rendition, and decode it"
        )
    )
    parser.add_argument("video", type=Path)
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
    parser.add_argument("--title", default=None)
    parser.add_argument("--tag", default="FOV,二维码,测试")
    parser.add_argument("--desc", default="FOV 视频信道自动化实验")
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
        "--output-root", type=Path, default=REPO_ROOT / "runs"
    )
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required on PATH")
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "yt-dlp is required; run: python -m pip install -r requirements.txt"
        ) from exc

    executable = resolve_executable(args.biliup)
    if args.biliup_cookies is None:
        raise SystemExit("missing --biliup-cookies (or BILIUP_COOKIES)")
    biliup_cookies = args.biliup_cookies.expanduser().resolve()
    if not biliup_cookies.is_file():
        raise SystemExit(f"biliup cookie file not found: {biliup_cookies}")

    video = args.video.expanduser().resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    if video.stat().st_size <= 0:
        raise SystemExit("video is empty")
    if args.tid is None:
        raise SystemExit("missing --tid (or BILI_TID)")
    if args.limit <= 0 or args.status_max_pages <= 0:
        raise SystemExit("--limit and --status-max-pages must be positive")
    if args.poll_interval < 10:
        raise SystemExit("--poll-interval must be >= 10 seconds")
    if args.approval_timeout <= 0 or args.download_retries <= 0:
        raise SystemExit("timeout/retry values must be positive")

    cover = args.cover.expanduser().resolve() if args.cover else None
    if cover is not None and not cover.is_file():
        raise SystemExit(f"cover not found: {cover}")
    yt_dlp_cookies = (
        args.yt_dlp_cookies.expanduser().resolve()
        if args.yt_dlp_cookies
        else None
    )
    if yt_dlp_cookies is not None and not yt_dlp_cookies.is_file():
        raise SystemExit(
            f"yt-dlp cookie file not found: {yt_dlp_cookies}"
        )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        args.output_root.expanduser().resolve() / f"bilibili-{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    review_log = run_dir / "review.log"
    title = (args.title or f"FOV auto {timestamp} {video.stem}")[:80]

    result: dict[str, Any] = {
        "source_video": str(video),
        "source_video_bytes": video.stat().st_size,
        "title": title,
        "tid": args.tid,
        "submit": args.submit,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "biliup_executable": executable,
    }
    write_result(result_path, result)

    bvid: str | None = None
    temporary_cookie: Path | None = None
    try:
        print("[1/6] biliup upload + submit")
        upload = run_logged(
            build_upload_command(
                executable,
                biliup_cookies,
                video,
                title=title,
                tid=args.tid,
                tag=args.tag,
                desc=args.desc,
                copyright_value=args.copyright,
                submit=args.submit,
                limit=args.limit,
                cover=cover,
            ),
            run_dir / "upload.log",
        )
        result["upload_seconds"] = round(upload.elapsed_s, 3)
        result["upload_returncode"] = upload.returncode
        if upload.returncode:
            raise RuntimeError(
                f"biliup upload failed; see {run_dir / 'upload.log'}"
            )

        identity = parse_upload_identity(upload.output)
        if identity is None:
            print("  upload output had no BVID; falling back to title lookup")
            fallback = recover_bvid_by_title(
                executable,
                biliup_cookies,
                title,
                args.status_max_pages,
                review_log,
            )
            if not fallback:
                raise RuntimeError(
                    "upload succeeded but BVID could not be discovered"
                )
            identity = UploadIdentity(fallback)

        bvid = identity.bvid
        result.update(
            {
                "bvid": bvid,
                "aid": identity.aid,
                "share_url": f"https://www.bilibili.com/video/{bvid}",
            }
        )
        write_result(result_path, result)
        print(f"  bvid: {bvid}")

        print("[2/6] wait for review/publication")

        def persist(history: list[dict[str, Any]]) -> None:
            result["review_history"] = history
            write_result(result_path, result)

        history = wait_for_published(
            executable,
            biliup_cookies,
            bvid,
            poll_interval=args.poll_interval,
            timeout=args.approval_timeout,
            max_pages=args.status_max_pages,
            log_path=review_log,
            on_poll=persist,
        )
        result["published_at"] = history[-1]["checked_at"]
        write_result(result_path, result)

        print("[3/6] fetch published archive details")
        show = run_logged(
            biliup_base(executable, biliup_cookies) + ["show", bvid],
            run_dir / "show.log",
        )
        result["show_returncode"] = show.returncode
        write_result(result_path, result)

        print("[4/6] download published rendition")
        download_cookie = yt_dlp_cookies
        download_auth = "anonymous"
        if args.cookies_from_browser:
            download_auth = f"browser:{args.cookies_from_browser}"
        elif yt_dlp_cookies:
            download_auth = "explicit-cookie-file"
        elif not args.anonymous_download:
            temp = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix="fov-biliup-",
                delete=False,
            )
            temp.close()
            temporary_cookie = Path(temp.name)
            try:
                cookie_count = export_biliup_cookies_to_netscape(
                    biliup_cookies, temporary_cookie
                )
                download_cookie = temporary_cookie
                download_auth = f"biliup-cookies:{cookie_count}"
            except Exception as exc:
                temporary_cookie.unlink(missing_ok=True)
                temporary_cookie = None
                print(
                    "  warning: cookie conversion failed "
                    f"({type(exc).__name__}: {exc}); trying anonymous"
                )
                download_auth = "anonymous-fallback"

        result["download_auth"] = download_auth
        downloaded = run_dir / "platform.mp4"
        download_with_ytdlp(
            result["share_url"],
            downloaded,
            format_selector=args.download_format,
            browser=args.cookies_from_browser,
            cookies=download_cookie,
            retries=args.download_retries,
        )
        result["downloaded_video"] = str(downloaded)
        result["downloaded_video_bytes"] = downloaded.stat().st_size
        try:
            result["downloaded_ffprobe"] = ffprobe_video(downloaded)
        except Exception as exc:
            result["ffprobe_warning"] = f"{type(exc).__name__}: {exc}"
        write_result(result_path, result)

        print("[5/6] FOV decode")
        result["decode"] = decode_downloaded(downloaded, run_dir)
        result["decode_ok"] = True
        write_result(result_path, result)

        print("[6/6] finish")
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_result(result_path, result)
        print(f"[OK] Bilibili roundtrip passed\nResult: {result_path}")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failed_at"] = datetime.now().isoformat(timespec="seconds")
        if bvid:
            result["bvid"] = bvid
        write_result(result_path, result)
        raise
    finally:
        if temporary_cookie is not None:
            temporary_cookie.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
