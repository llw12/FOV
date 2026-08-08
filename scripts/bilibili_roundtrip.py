"""Upload an FOV source video through Bilibili Open Platform, wait for review,
download the published rendition, and run the FOV decoder.

Upload/review polling uses Bilibili's documented Open Platform APIs. Download is
performed with yt-dlp because the Open Platform does not expose a documented video
rendition download API. Use this only for videos/accounts you are authorized to use
and follow Bilibili's current platform rules.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

import cv2
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER_BASE = "https://member.bilibili.com"
UPOS_BASE = "https://openupos.bilivideo.com"
SMALL_UPLOAD_LIMIT = 100 * 1024 * 1024
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
PART_SIZE = 8 * 1024 * 1024
DEFAULT_DOWNLOAD_FORMAT = (
    "bestvideo[height<=720][vcodec^=avc1]/"
    "bestvideo[height<=720]/best[height<=720]/best"
)


class BilibiliOpenApiError(RuntimeError):
    def __init__(self, operation: str, payload: Any) -> None:
        if isinstance(payload, dict):
            code = payload.get("code")
            message = payload.get("message")
            request_id = payload.get("request_id")
            detail = f"code={code}, message={message}, request_id={request_id}"
        else:
            detail = repr(payload)
        super().__init__(f"Bilibili Open API {operation} failed: {detail}")


@dataclass(frozen=True)
class Credentials:
    client_id: str
    app_secret: str
    access_token: str


@dataclass(frozen=True)
class ArchiveState:
    outcome: str  # open | failed | waiting
    state: int | None
    state_desc: str
    reject_reason: str
    share_url: str


def compact_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def build_signed_headers(
    credentials: Credentials,
    *,
    body_bytes: bytes = b"",
    content_type: str | None = "application/json",
    timestamp: int | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """Build Bilibili Open Platform v2 HMAC-SHA256 headers.

    For binary video uploads and cover multipart uploads, callers pass an empty
    body_bytes because the documented content MD5 excludes file content.
    """
    timestamp_value = str(timestamp if timestamp is not None else int(time.time()))
    nonce_value = nonce or uuid.uuid4().hex
    x_headers = {
        "x-bili-accesskeyid": credentials.client_id,
        "x-bili-content-md5": hashlib.md5(body_bytes).hexdigest(),
        "x-bili-signature-method": "HMAC-SHA256",
        "x-bili-signature-nonce": nonce_value,
        "x-bili-signature-version": "2.0",
        "x-bili-timestamp": timestamp_value,
    }
    canonical = "\n".join(f"{key}:{x_headers[key]}" for key in sorted(x_headers))
    signature = hmac.new(
        credentials.app_secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    headers = {
        "Accept": "application/json",
        "Access-Token": credentials.access_token,
        "Authorization": signature,
        **x_headers,
    }
    if content_type is not None:
        headers["Content-Type"] = content_type
    return headers


def classify_archive(payload: dict[str, Any]) -> ArchiveState:
    data = payload.get("data") or {}
    addit = data.get("addit_info") or {}
    video_info = data.get("video_info") or {}
    state = addit.get("state")
    try:
        state_int = int(state) if state is not None else None
    except (TypeError, ValueError):
        state_int = None
    state_desc = str(addit.get("state_desc") or "")
    reject_reason = str(addit.get("reject_reason") or "")
    share_url = str(video_info.get("share_url") or "")

    if state_int == 0 or "开放浏览" in state_desc or "已发布" in state_desc:
        return ArchiveState("open", state_int, state_desc, reject_reason, share_url)

    failed_words = ("退回", "失败", "删除", "封禁", "锁定", "不通过", "拒绝")
    if reject_reason or any(word in state_desc for word in failed_words):
        return ArchiveState("failed", state_int, state_desc, reject_reason, share_url)

    return ArchiveState("waiting", state_int, state_desc, reject_reason, share_url)


class BilibiliClient:
    def __init__(self, credentials: Credentials) -> None:
        self.credentials = credentials
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "FOV-Bilibili-Roundtrip/1.0"})

    def _json_request(
        self,
        method: str,
        url: str,
        *,
        operation: str,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: tuple[int, int] = (20, 120),
    ) -> dict[str, Any]:
        body_bytes = compact_json(body) if body is not None else b""
        response = self.session.request(
            method,
            url,
            params=params,
            data=body_bytes if body is not None else None,
            headers=build_signed_headers(self.credentials, body_bytes=body_bytes),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise BilibiliOpenApiError(operation, payload)
        return payload

    def video_init(self, filename: str, *, small: bool) -> str:
        payload = self._json_request(
            "POST",
            f"{MEMBER_BASE}/arcopen/fn/archive/video/init",
            operation="video init",
            body={"name": filename, "utype": "1" if small else "0"},
        )
        token = (payload.get("data") or {}).get("upload_token")
        if not token:
            raise BilibiliOpenApiError("video init missing upload_token", payload)
        return str(token)

    def _upload_binary(
        self,
        url: str,
        source: bytes | BinaryIO,
        *,
        operation: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        response = self.session.post(
            url,
            params=params,
            data=source,
            headers=build_signed_headers(self.credentials, body_bytes=b""),
            timeout=(20, 300),
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise BilibiliOpenApiError(operation, payload)
        return payload

    def upload_small_video(self, video_path: Path, upload_token: str) -> None:
        with video_path.open("rb") as stream:
            self._upload_binary(
                f"{UPOS_BASE}/video/v2/upload",
                stream,
                operation="small video upload",
                params={"upload_token": upload_token},
            )

    def upload_parts(self, video_path: Path, upload_token: str) -> int:
        part_number = 0
        with video_path.open("rb") as stream:
            while True:
                chunk = stream.read(PART_SIZE)
                if not chunk:
                    break
                part_number += 1
                print(f"  upload part {part_number} ({len(chunk) / 1024 / 1024:.2f} MiB)")
                self._upload_binary(
                    f"{UPOS_BASE}/video/v2/part/upload",
                    chunk,
                    operation=f"video part {part_number} upload",
                    params={"upload_token": upload_token, "part_number": part_number},
                )
        return part_number

    def complete_parts(self, upload_token: str) -> None:
        self._json_request(
            "POST",
            f"{MEMBER_BASE}/arcopen/fn/archive/video/complete",
            operation="video complete",
            params={"upload_token": upload_token},
        )

    def upload_cover(self, cover_path: Path) -> str:
        headers = build_signed_headers(
            self.credentials,
            body_bytes=b"",
            content_type=None,  # requests adds multipart/form-data with its boundary.
        )
        with cover_path.open("rb") as stream:
            response = self.session.post(
                f"{MEMBER_BASE}/arcopen/fn/archive/cover/upload",
                files={"file": (cover_path.name, stream, "image/jpeg")},
                headers=headers,
                timeout=(20, 120),
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise BilibiliOpenApiError("cover upload", payload)
        url = (payload.get("data") or {}).get("url")
        if not url:
            raise BilibiliOpenApiError("cover upload missing url", payload)
        return str(url)

    def submit_archive(
        self,
        upload_token: str,
        *,
        title: str,
        cover_url: str,
        tid: int,
        tag: str,
        desc: str,
    ) -> str:
        body = {
            "title": title,
            "cover": cover_url,
            "tid": tid,
            "tag": tag,
            "desc": desc,
            "copyright": 1,
            "no_reprint": 1,
        }
        payload = self._json_request(
            "POST",
            f"{MEMBER_BASE}/arcopen/fn/archive/add-by-utoken",
            operation="archive submit",
            body=body,
            params={"upload_token": upload_token},
        )
        resource_id = (payload.get("data") or {}).get("resource_id")
        if not resource_id:
            raise BilibiliOpenApiError("archive submit missing resource_id", payload)
        return str(resource_id)

    def archive_view(self, resource_id: str) -> dict[str, Any]:
        return self._json_request(
            "GET",
            f"{MEMBER_BASE}/arcopen/fn/archive/view",
            operation="archive view",
            params={"resource_id": resource_id},
        )

    def delete_archive(self, resource_id: str) -> None:
        self._json_request(
            "POST",
            f"{MEMBER_BASE}/arcopen/fn/archive/delete",
            operation="archive delete",
            body={"resource_id": resource_id},
        )


def make_cover(video_path: Path, target: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"cannot read first frame for cover: {video_path}")

    target_ratio = 1146 / 717
    height, width = frame.shape[:2]
    ratio = width / height
    if ratio > target_ratio:
        crop_width = round(height * target_ratio)
        left = (width - crop_width) // 2
        frame = frame[:, left:left + crop_width]
    elif ratio < target_ratio:
        crop_height = round(width / target_ratio)
        top = (height - crop_height) // 2
        frame = frame[top:top + crop_height, :]
    resized = cv2.resize(frame, (1146, 717), interpolation=cv2.INTER_AREA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"cannot write cover: {target}")


def wait_for_open(
    client: BilibiliClient,
    resource_id: str,
    *,
    poll_interval: float,
    timeout: float,
) -> tuple[ArchiveState, list[dict[str, Any]]]:
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    last_label = ""
    while True:
        payload = client.archive_view(resource_id)
        state = classify_archive(payload)
        entry = {
            "checked_at": datetime.now().isoformat(timespec="seconds"),
            "state": state.state,
            "state_desc": state.state_desc,
            "reject_reason": state.reject_reason,
        }
        history.append(entry)
        label = f"state={state.state} {state.state_desc}".strip()
        if label != last_label:
            print(f"  review: {label or 'unknown'}")
            last_label = label
        if state.outcome == "open":
            return state, history
        if state.outcome == "failed":
            reason = state.reject_reason or state.state_desc or str(state.state)
            raise RuntimeError(f"Bilibili review failed: {reason}")
        if time.monotonic() - started >= timeout:
            raise TimeoutError(f"Bilibili review did not finish within {timeout:.0f}s; last={label}")
        time.sleep(poll_interval)


def run_logged(command: list[str], log_path: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return completed


def download_with_ytdlp(
    share_url: str,
    target: Path,
    *,
    format_selector: str,
    cookies_from_browser: str | None,
    cookies: Path | None,
    retries: int,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
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
            str(target),
        ]
        if cookies_from_browser:
            command += ["--cookies-from-browser", cookies_from_browser]
        if cookies:
            command += ["--cookies", str(cookies)]
        command.append(share_url)
        completed = run_logged(command, target.parent / f"download_attempt_{attempt}.log")
        if completed.returncode == 0 and target.is_file() and target.stat().st_size > 0:
            return
        if attempt < retries:
            wait_seconds = min(60, 10 * attempt)
            print(f"  download attempt {attempt} failed; retry in {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"yt-dlp failed after {retries} attempts")


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,width,height,pix_fmt,bit_rate,nb_frames,avg_frame_rate:format=duration,size,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return json.loads(completed.stdout)


def decode_downloaded(video_path: Path, run_dir: Path) -> None:
    recovered = run_dir / "recovered"
    shutil.rmtree(recovered, ignore_errors=True)
    recovered.mkdir(parents=True, exist_ok=True)
    completed = run_logged(
        [sys.executable, "video2file.py", str(video_path), str(recovered)],
        run_dir / "decode.log",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"FOV decode failed; see {run_dir / 'decode.log'}")


def env_or_arg(value: str | None, env_name: str) -> str:
    resolved = value or os.environ.get(env_name, "")
    if not resolved:
        raise SystemExit(f"missing credential: pass the option or set {env_name}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload an FOV video to Bilibili, wait for review, download the published rendition, and decode it"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--client-id", default=None, help="or BILI_CLIENT_ID")
    parser.add_argument("--app-secret", default=None, help="or BILI_APP_SECRET")
    parser.add_argument("--access-token", default=None, help="or BILI_ACCESS_TOKEN")
    parser.add_argument("--tid", type=int, default=None, help="Bilibili archive category id; or BILI_TID")
    parser.add_argument("--title", default=None)
    parser.add_argument("--tag", default="FOV,实验")
    parser.add_argument(
        "--desc",
        default="FOV 数据经二维码视频传输的鲁棒性实验；本稿件仅用于本人自动回环测试。",
    )
    parser.add_argument("--cover", type=Path, default=None, help="optional jpeg/png cover; otherwise first frame is used")
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--approval-timeout", type=float, default=2 * 60 * 60)
    parser.add_argument("--download-format", default=DEFAULT_DOWNLOAD_FORMAT)
    parser.add_argument("--cookies-from-browser", default=None, help="yt-dlp browser name, e.g. chrome or edge")
    parser.add_argument("--cookies", type=Path, default=None, help="Netscape cookies.txt for yt-dlp")
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--delete-after", action="store_true", help="delete the Bilibili archive after successful decode")
    args = parser.parse_args()

    if shutil.which("ffprobe") is None:
        raise SystemExit("ffprobe is required on PATH")
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise SystemExit("yt-dlp is required; run: python -m pip install -r requirements.txt") from exc

    video_path = args.video.resolve()
    if not video_path.is_file():
        raise SystemExit(f"video not found: {video_path}")
    size = video_path.stat().st_size
    if size <= 0 or size > MAX_UPLOAD_BYTES:
        raise SystemExit(f"video size must be within 1..{MAX_UPLOAD_BYTES} bytes")
    if args.poll_interval < 10:
        raise SystemExit("--poll-interval must be >= 10 seconds")
    if args.approval_timeout <= 0 or args.download_retries <= 0:
        raise SystemExit("timeout/retry values must be positive")

    tid_value = args.tid if args.tid is not None else os.environ.get("BILI_TID")
    if tid_value in (None, ""):
        raise SystemExit("missing --tid (or BILI_TID); use a category id granted/available to your Open Platform app")
    tid = int(tid_value)

    credentials = Credentials(
        client_id=env_or_arg(args.client_id, "BILI_CLIENT_ID"),
        app_secret=env_or_arg(args.app_secret, "BILI_APP_SECRET"),
        access_token=env_or_arg(args.access_token, "BILI_ACCESS_TOKEN"),
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root.resolve() / f"bilibili-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    title = args.title or f"FOV roundtrip {timestamp}"
    cover_path = args.cover.resolve() if args.cover else run_dir / "cover.jpg"
    if not args.cover:
        make_cover(video_path, cover_path)
    elif not cover_path.is_file():
        raise SystemExit(f"cover not found: {cover_path}")

    result: dict[str, Any] = {
        "source_video": str(video_path),
        "source_video_bytes": size,
        "title": title,
        "tid": tid,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    result_path = run_dir / "result.json"
    client = BilibiliClient(credentials)
    resource_id: str | None = None

    try:
        small = size <= SMALL_UPLOAD_LIMIT
        print(f"[1/6] init upload ({'small' if small else 'multipart'})")
        upload_token = client.video_init(video_path.name, small=small)
        result["upload_mode"] = "small" if small else "multipart"

        print("[2/6] upload video")
        if small:
            client.upload_small_video(video_path, upload_token)
        else:
            result["uploaded_parts"] = client.upload_parts(video_path, upload_token)
            client.complete_parts(upload_token)

        print("[3/6] upload cover + submit archive")
        cover_url = client.upload_cover(cover_path)
        resource_id = client.submit_archive(
            upload_token,
            title=title,
            cover_url=cover_url,
            tid=tid,
            tag=args.tag,
            desc=args.desc,
        )
        result["resource_id"] = resource_id
        result["cover_url"] = cover_url
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  resource_id: {resource_id}")

        print("[4/6] wait for review")
        archive_state, history = wait_for_open(
            client,
            resource_id,
            poll_interval=args.poll_interval,
            timeout=args.approval_timeout,
        )
        result["review_history"] = history
        share_url = archive_state.share_url or f"https://www.bilibili.com/video/{resource_id}"
        result["share_url"] = share_url
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  open: {share_url}")

        print("[5/6] download published rendition")
        downloaded = run_dir / "platform.mp4"
        download_with_ytdlp(
            share_url,
            downloaded,
            format_selector=args.download_format,
            cookies_from_browser=args.cookies_from_browser,
            cookies=args.cookies.resolve() if args.cookies else None,
            retries=args.download_retries,
        )
        result["downloaded_video"] = str(downloaded)
        try:
            result["downloaded_ffprobe"] = ffprobe_video(downloaded)
        except Exception as exc:
            result["ffprobe_warning"] = f"{type(exc).__name__}: {exc}"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print("[6/6] FOV decode")
        decode_downloaded(downloaded, run_dir)
        result["decode_ok"] = True

        if args.delete_after:
            print("[cleanup] delete Bilibili archive")
            client.delete_archive(resource_id)
            result["archive_deleted"] = True

        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Bilibili roundtrip passed\nResult: {result_path}")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["failed_at"] = datetime.now().isoformat(timespec="seconds")
        if resource_id:
            result["resource_id"] = resource_id
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
