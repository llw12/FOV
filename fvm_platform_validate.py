"""One-shot real-platform validation for an existing FVM source video."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fov import BLOCK_SIZE, derive_block_layout, sha256_file
from fvm_codec_sweep import ffprobe as sweep_ffprobe, raw_oracle
from fvm_video2file import decode as production_decode, result_path
from scripts.bilibili_roundtrip import (
    UploadIdentity, biliup_base, build_upload_command, export_biliup_cookies_to_netscape,
    parse_upload_identity, recover_bvid_by_title, resolve_executable, run_logged,
    wait_for_published,
    list_status,
)

EXPECTED_WIDTH = 1920
EXPECTED_HEIGHT = 1080
EXPECTED_FPS = 30.0
EXPECTED_SYMBOL_SIZE = 6400
EXPECTED_BLOCK_SIZE = 8 * 1024 * 1024
EXPECTED_REPAIR = 0.03
STATES = {"PRECHECK", "UPLOADED", "WAITING_REVIEW", "PUBLISHED", "WAITING_1080P",
          "DOWNLOADED", "DECODED", "PASS", "FAIL", "BLOCKED", "INCOMPLETE"}


@dataclass
class UploadGuard:
    attempted: bool = False
    attempt_count: int = 0

    def run(self, uploader: Callable[[], Any]) -> Any:
        if self.attempted:
            raise RuntimeError("upload already attempted in this run")
        self.attempted = True
        self.attempt_count += 1
        return uploader()


def _fraction(value: str | None) -> float | None:
    if not value:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(path: Path) -> dict[str, Any]:
    probe, warnings = sweep_ffprobe(path)
    if warnings:
        raise RuntimeError(warnings[0])
    stream = probe["stream"]
    probe["fps"] = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
    probe["frame_count"] = int(stream["nb_frames"]) if stream.get("nb_frames") else None
    return probe


def validate_1080p_probe(probe: dict[str, Any], label: str) -> None:
    stream = probe.get("stream", {})
    if (stream.get("width"), stream.get("height")) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        raise ValueError(f"{label} is not 1920x1080")
    if probe.get("fps") is None or not math.isclose(probe["fps"], EXPECTED_FPS, abs_tol=0.05):
        raise ValueError(f"{label} is not approximately 30 fps")


def select_1080p_format(formats: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [item for item in formats if item.get("width") == EXPECTED_WIDTH
                  and item.get("height") == EXPECTED_HEIGHT and item.get("vcodec") not in (None, "none")]
    if not candidates:
        return None
    avc = [item for item in candidates if str(item.get("vcodec", "")).lower().startswith(("avc1", "h264"))]
    pool = avc or candidates
    return max(pool, key=lambda item: (float(item.get("tbr") or 0), float(item.get("filesize") or item.get("filesize_approx") or 0)))


def format_state(formats: list[dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    selected = select_1080p_format(formats)
    return ("AVAILABLE", selected) if selected else ("WAITING_1080P", None)


def decode_summary(payload: dict[str, Any]) -> dict[str, Any]:
    video = payload.get("video", {}); transport = payload.get("transport", {})
    rs = payload.get("rs", {}); packets = payload.get("packets", {}); raptorq = payload.get("raptorq", {})
    file_result = payload.get("file", {})
    return {
        "observed_frames": video.get("observed_frames"),
        "rs_success_frames": transport.get("rs_frames_success"),
        "rs_failed_frames": transport.get("rs_frames_failed"),
        "codewords_success": rs.get("codewords_success"), "codewords_failed": rs.get("codewords_failed"),
        "corrected_symbols": rs.get("corrected_symbols"),
        "correction_histogram": rs.get("correction_histogram"),
        "transport_crc_failures": transport.get("transport_crc_failures", 0),
        "invalid_header": transport.get("invalid_header", 0),
        "invalid_packet_length": transport.get("invalid_packet_length", 0),
        "embedded_frame_gaps": transport.get("embedded_frame_gaps", 0),
        "duplicate_embedded_indices": transport.get("duplicate_embedded_indices", 0),
        "out_of_order_count": transport.get("out_of_order_count", 0),
        "packet_crc_failures": packets.get("packet_crc_failed", 0), "valid_meta": packets.get("valid_meta"),
        "blocks_total": raptorq.get("blocks_total"), "blocks_decoded": raptorq.get("blocks_decoded"),
        "source_symbols_received": raptorq.get("source_symbols_received"),
        "repair_symbols_received": raptorq.get("repair_symbols_received"),
        "post_decode_symbols": packets.get("post_decode_symbols"),
        "valid_symbols": packets.get("valid_symbols"),
        "rs_failure_distribution": summarize_rs_failures(payload.get("rs_failures", {})),
        "original_sha256": file_result.get("original_sha256"), "recovered_sha256": file_result.get("recovered_sha256"),
        "sha_exact": file_result.get("exact") is True,
    }


def summarize_rs_failures(failures: dict[str, Any]) -> dict[str, Any]:
    return {key: failures.get(key) for key in (
        "event_count_total", "failed_codewords_total", "failed_codewords_per_failed_frame",
        "failed_codeword_index_histogram", "inter_failure_distance_stats", "burst_count",
        "longest_consecutive_burst")}


def parse_failed_blocks(message: str | None) -> list[dict[str, int]]:
    """Extract block-local symbol deficits from the production decoder error."""
    pattern = re.compile(r"block_id=(\d+), K=(\d+), N=(\d+), received_unique=(\d+)")
    return [{"block_id": int(block_id), "required_k": int(k), "encoded_n": int(n),
             "received_unique": int(received), "symbol_deficit": max(0, int(k) - int(received))}
            for block_id, k, n, received in pattern.findall(message or "")]


def recovery_pass(summary: dict[str, Any], expected_sha: str) -> bool:
    return (summary.get("sha_exact") is True and summary.get("recovered_sha256") == expected_sha
            and summary.get("blocks_total") == summary.get("blocks_decoded"))


def alignment_allows_positional(source_probe: dict[str, Any], platform_probe: dict[str, Any], diagnostics: dict[str, Any]) -> bool:
    missing_transport = ((diagnostics.get("rs_failed_frames") or 0)
                         + (diagnostics.get("transport_crc_failures") or 0)
                         + (diagnostics.get("invalid_header") or 0)
                         + (diagnostics.get("invalid_packet_length") or 0))
    return (source_probe.get("frame_count") is not None
            and source_probe.get("frame_count") == platform_probe.get("frame_count")
            and diagnostics.get("duplicate_embedded_indices", 0) == 0
            and diagnostics.get("out_of_order_count", 0) == 0
            and (diagnostics.get("embedded_frame_gaps") or 0) <= missing_transport)


def compare_proxy_platform(proxy: dict[str, Any], platform: dict[str, Any]) -> dict[str, Any]:
    proxy_ber, platform_ber = proxy.get("raw_ber"), platform.get("raw_ber")
    proxy_rs, platform_rs = proxy.get("rs_failed_frames"), platform.get("rs_failed_frames")
    ratio = None if proxy_ber in (None, 0) or platform_ber is None else platform_ber / proxy_ber
    if proxy_rs is not None and platform_rs is not None and platform_rs > max(2, proxy_rs * 2):
        verdict = "UNDER-STRESS"
    elif ratio is not None and ratio > 2:
        verdict = "UNDER-STRESS"
    elif ratio is not None and ratio < 0.5 and (proxy_rs or 0) >= (platform_rs or 0):
        verdict = "OVER-STRESS"
    else:
        verdict = "ROUGHLY COMPARABLE"
    return {"platform_to_proxy_ber_ratio": ratio, "verdict": verdict,
            "classification": "observational classification from one platform sample"}


def redacted(payload: Any) -> Any:
    """Remove credential-bearing keys before persistence."""
    if isinstance(payload, dict):
        return {key: redacted(value) for key, value in payload.items()
                if "cookie" not in key.lower() and key.lower() not in {"authorization", "token"}}
    if isinstance(payload, list):
        return [redacted(value) for value in payload]
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(redacted(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def set_state(result: dict[str, Any], path: Path, state: str, **values: Any) -> None:
    if state not in STATES:
        raise ValueError(f"invalid validation state: {state}")
    result.update(values); result["state"] = state
    result.setdefault("state_history", []).append({"state": state, "at": datetime.now().isoformat(timespec="seconds")})
    write_json(path, result)


def fetch_formats(url: str, cookie_path: Path | None, log_path: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--skip-download", "--dump-single-json"]
    if cookie_path:
        command += ["--cookies", str(cookie_path)]
    command.append(url)
    completed = run_logged(command, log_path, append=True, echo=False)
    if completed.returncode:
        raise RuntimeError("yt-dlp format discovery failed")
    lines = [line for line in completed.output.splitlines() if line.strip().startswith("{")]
    if not lines:
        raise RuntimeError("yt-dlp returned no metadata JSON")
    return json.loads(lines[-1])


def wait_for_1080p(url: str, cookie_path: Path | None, *, poll_interval: float, timeout: float,
                   formats_log: Path, on_poll: Callable[[list[dict[str, Any]]], None] | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic(); history = []
    while True:
        metadata = fetch_formats(url, cookie_path, formats_log)
        state, selected = format_state(metadata.get("formats") or [])
        history.append({"checked_at": datetime.now().isoformat(timespec="seconds"), "state": state,
                        "format_id": selected.get("format_id") if selected else None})
        if on_poll: on_poll(history)
        if selected: return metadata, selected, history
        if time.monotonic() - started >= timeout: raise TimeoutError("1080p rendition did not appear before timeout")
        time.sleep(poll_interval)


def download_format(url: str, format_id: str, target: Path, cookie_path: Path | None, log: Path) -> None:
    template = target.with_name(target.stem + ".%(ext)s")
    command = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-part", "--force-overwrites",
               "--retries", "3", "-f", format_id, "--remux-video", "mp4", "-o", str(template)]
    if cookie_path: command += ["--cookies", str(cookie_path)]
    command.append(url)
    completed = run_logged(command, log)
    if completed.returncode or not target.is_file() or target.stat().st_size <= 0:
        raise RuntimeError("yt-dlp 1080p download failed")


def production_decode_summary(video: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir.parent / "decode.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        recovered = production_decode(video, output_dir)
    payload = json.loads(result_path(output_dir).read_text(encoding="utf-8"))
    summary = decode_summary(payload); summary["recovered_path"] = str(recovered)
    return summary


def preflight(source: Path, original: Path, case_result: Path, run_dir: Path) -> dict[str, Any]:
    expected_sha = sha256_file(original); case = json.loads(case_result.read_text(encoding="utf-8")); config = case.get("config", {})
    if original.stat().st_size != 32 * 1024 * 1024:
        raise ValueError("original input size mismatch")
    if case.get("input_sha256") != expected_sha:
        raise ValueError("case input SHA mismatch")
    expected = {"low_level": 80, "high_level": 176, "crf": 30, "preset": "slow", "repair_ratio": EXPECTED_REPAIR,
                "symbol_size": EXPECTED_SYMBOL_SIZE, "cell_size": 6}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("case configuration is not the fixed 80/176 CRF30 candidate")
    if case.get("source", {}).get("sha_exact") is not True or case.get("proxy", {}).get("sha_exact") is not True:
        raise ValueError("sweep source/proxy baseline is not exact")
    source_probe = probe_video(source); validate_1080p_probe(source_probe, "source")
    local = production_decode_summary(source, run_dir / "local-recovered")
    if not recovery_pass(local, expected_sha):
        raise RuntimeError("local production decode gate failed")
    layout = derive_block_layout(original.stat().st_size, BLOCK_SIZE, EXPECTED_SYMBOL_SIZE, EXPECTED_REPAIR)
    return {"original": {"path": str(original), "size_bytes": original.stat().st_size, "sha256": expected_sha},
            "source": source_probe, "case_config": config, "local_decode": local,
            "expected_source_symbols": sum(block.source_symbols for block in layout),
            "local_proxy": case["proxy"]}


def write_report(result: dict[str, Any], path: Path) -> None:
    source = result.get("preflight", {}).get("source", {}); platform = result.get("platform", {})
    decode = result.get("decode", {}); raw = result.get("raw_channel", {}); proxy = result.get("preflight", {}).get("local_proxy", {})
    comparison = result.get("proxy_comparison", {})
    lines = ["# FVM Real Platform Validation", "", "## Source case", "",
             "- Candidate: 80/176, CRF30, x264 slow, 6 px binary, RS(255,239), RaptorQ 3%.",
             f"- Source: {source.get('size_bytes')} bytes, {source.get('frame_count')} frames.", "", "## Upload", "",
             f"- Attempts: {result.get('upload_attempt_count', 0)}", f"- BVID: {result.get('bvid')}", "", "## Platform rendition", "",
             f"- Format: {platform.get('format_id')} / {platform.get('codec')}",
             f"- Resolution: {platform.get('width')}x{platform.get('height')}",
             f"- Size / duration / calculated bitrate: {platform.get('size_bytes')} bytes / {platform.get('duration_seconds')} s / {platform.get('calculated_bitrate_bps')} bps",
             f"- Frames: {platform.get('frame_count')} (source delta {platform.get('frame_count_delta')})", "", "## Decode / recovery", "",
             f"- Blocks: {decode.get('blocks_decoded')}/{decode.get('blocks_total')}",
             f"- RS failed frames: {decode.get('rs_failed_frames')}",
             f"- Repair symbols received: {decode.get('repair_symbols_received')}",
             f"- Source symbols: expected {decode.get('source_symbols_expected')}, received {decode.get('source_symbols_received')}, erasures {decode.get('source_erasures')}",
             f"- SHA exact: {decode.get('sha_exact')}", "", "## Raw channel", "",
             f"- Mode: {raw.get('mode')}", f"- Raw BER: {raw.get('raw_ber')}",
             f"- Frames with raw errors: {raw.get('frames_with_raw_errors')}", "", "## Local proxy vs real platform", "",
             "| Metric | Local proxy | Real platform |", "|---|---:|---:|",
             f"| Raw BER | {proxy.get('raw_ber')} | {raw.get('raw_ber')} |",
             f"| RS failed frames | {proxy.get('rs_failed_frames')} | {decode.get('rs_failed_frames')} |",
             f"| RS failure rate | {proxy.get('rs_frame_failure_rate')} | {decode.get('rs_frame_failure_rate')} |",
             f"| Failed codewords | {proxy.get('codewords_failed', 'N/A')} | {decode.get('codewords_failed')} |",
             f"| Corrected symbols | {proxy.get('corrected_symbols', 'N/A')} | {decode.get('corrected_symbols')} |",
             f"| Source symbols lost | {proxy.get('symbol_erasure_count', 'N/A')} | {decode.get('source_erasures')} |",
             f"| Repair symbols used | {proxy.get('repair_symbols_received')} | {decode.get('repair_symbols_received')} |",
             f"| Blocks decoded | {proxy.get('blocks_decoded')}/{proxy.get('blocks_total')} | {decode.get('blocks_decoded')}/{decode.get('blocks_total')} |",
             f"| SHA exact | {proxy.get('sha_exact')} | {decode.get('sha_exact')} |", "",
             f"- BER ratio (platform/proxy): {comparison.get('platform_to_proxy_ber_ratio')}",
             f"- Calibration: PROXY: {comparison.get('verdict')} ({comparison.get('classification')})", "", "## Verdict", ""]
    if result.get("state") == "PASS":
        lines += ["80/176 CRF30 REAL PLATFORM SAMPLE: PASS", "", "One authorized 1080p platform rendition. Not a universal reliability guarantee."]
    elif result.get("state") == "FAIL":
        lines.append("80/176 CRF30 REAL PLATFORM SAMPLE: FAIL")
        lines += ["", "### Failed blocks", "", "| Block | Required K | Received unique | Symbol deficit |",
                  "|---:|---:|---:|---:|"]
        for block in decode.get("failed_blocks", []):
            lines.append(f"| {block['block_id']} | {block['required_k']} | {block['received_unique']} | {block['symbol_deficit']} |")
        lines += ["", f"- Aggregate received source / repair: {decode.get('source_symbols_received')} / {decode.get('repair_symbols_received')}",
                  f"- RS failure distribution: `{json.dumps(decode.get('rs_failure_distribution'), ensure_ascii=False)}`",
                  "- Root cause established at the recovery layer: platform-induced RS frame failures produced block-local symbol deficits beyond the available 3% repair sample. The underlying spatial/temporal error mechanism is not established by this single sample."]
    else: lines.append(f"Validation state: {result.get('state')}")
    lines += ["", "## Next decision", "", f"- {result.get('next_decision', 'Pending')}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_formats(metadata: dict[str, Any]) -> dict[str, Any]:
    fields = ("format_id", "format", "ext", "vcodec", "acodec", "width", "height", "fps", "tbr", "filesize", "filesize_approx")
    return {"id": metadata.get("id"), "title": metadata.get("title"),
            "formats": [{key: item.get(key) for key in fields} for item in metadata.get("formats", [])]}


def _temporary_cookie(biliup_cookies: Path) -> tuple[Path, int]:
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", prefix="fvm-biliup-", delete=False)
    handle.close(); path = Path(handle.name)
    try:
        count = export_biliup_cookies_to_netscape(biliup_cookies, path)
        return path, count
    except BaseException:
        path.unlink(missing_ok=True); raise


def run_validation(source: Path, original: Path, case_result: Path, *, allow_upload: bool,
                   biliup: str | None, biliup_cookies: Path | None, tid: int | None,
                   output_root: Path, resume_run: Path | None = None, title: str | None = None,
                   tag: str = "FVM,视频信道,测试", desc: str = "FVM 6px binary codec robustness validation",
                   poll_interval: float = 60, approval_timeout: float = 10800,
                   rendition_timeout: float = 7200, status_max_pages: int = 3) -> dict[str, Any]:
    if poll_interval < 60:
        raise ValueError("poll interval must be at least 60 seconds")
    source, original, case_result = source.resolve(), original.resolve(), case_result.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = resume_run.resolve() if resume_run else output_root.resolve() / f"fvm-platform-validation-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=bool(resume_run))
    result_file = run_dir / "validation_result.json"
    if resume_run:
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if result.get("upload_attempt_count", 0) > 1:
            raise RuntimeError("stored run violates one-upload invariant")
    else:
        result = {"format": "FVM_REAL_PLATFORM_VALIDATION_V1", "run_dir": str(run_dir),
                  "source_video": str(source), "original_file": str(original),
                  "case_result": str(case_result), "upload_attempt_count": 0}
        set_state(result, result_file, "PRECHECK")
        result["preflight"] = preflight(source, original, case_result, run_dir)
        write_json(run_dir / "preflight.json", result["preflight"])
        write_json(result_file, result)
    if not allow_upload:
        write_report(result, run_dir / "REPORT.md")
        return result
    if not biliup or not biliup_cookies or tid is None:
        raise ValueError("--allow-upload requires biliup executable, cookies, and tid")
    executable = resolve_executable(biliup); biliup_cookies = biliup_cookies.resolve()
    if not biliup_cookies.is_file(): raise ValueError("biliup cookies file not found")
    guard = UploadGuard(attempted=result.get("upload_attempt_count", 0) > 0,
                        attempt_count=result.get("upload_attempt_count", 0))
    bvid = result.get("bvid")
    if not bvid:
        if guard.attempted:
            # Resume may perform a read-only identity lookup for the one completed upload.
            fallback = recover_bvid_by_title(executable, biliup_cookies, result.get("title", ""),
                                             status_max_pages, run_dir / "review.log")
            if fallback:
                bvid = fallback
                set_state(result, result_file, "UPLOADED", bvid=bvid, aid=result.get("aid"),
                          share_url=f"https://www.bilibili.com/video/{bvid}", error=None)
            else:
                set_state(result, result_file, "BLOCKED", error="upload was attempted but no BVID is stored; refusing another upload")
                write_report(result, run_dir / "REPORT.md"); return result
    if not bvid:
        auth_check = list_status(executable, biliup_cookies, "--pubed", 1, run_dir / "auth-check.log")
        if auth_check.returncode:
            set_state(result, result_file, "BLOCKED", error="existing biliup login state failed read-only validation; no upload attempted")
            write_report(result, run_dir / "REPORT.md"); return result
        unique_title = (title or f"FVM 6px L80-176 CRF30 validation {timestamp}")[:80]
        command = build_upload_command(executable, biliup_cookies, source, title=unique_title, tid=tid, tag=tag, desc=desc,
                                       copyright_value=1, submit="app", limit=3, cover=None)
        result.update({"title": unique_title, "tid": tid, "upload_attempt_count": 1})
        write_json(result_file, result)  # Persist the guard before external mutation.
        upload_started = time.monotonic()
        upload = guard.run(lambda: run_logged(command, run_dir / "upload.log"))
        result.update({"upload_returncode": upload.returncode, "upload_elapsed_seconds": time.monotonic() - upload_started})
        if upload.returncode:
            set_state(result, result_file, "BLOCKED", error="biliup upload failed; no retry was attempted")
            write_report(result, run_dir / "REPORT.md"); return result
        identity = parse_upload_identity(upload.output)
        if identity is None:
            fallback = recover_bvid_by_title(executable, biliup_cookies, unique_title, status_max_pages, run_dir / "review.log")
            identity = UploadIdentity(fallback) if fallback else None
        if identity is None:
            set_state(result, result_file, "BLOCKED", error="upload succeeded but BVID could not be discovered; refusing another upload")
            write_report(result, run_dir / "REPORT.md"); return result
        bvid = identity.bvid
        set_state(result, result_file, "UPLOADED", bvid=bvid, aid=identity.aid,
                  share_url=f"https://www.bilibili.com/video/{bvid}")
    review_started = time.monotonic()
    if result.get("state") not in {"PUBLISHED", "WAITING_1080P", "DOWNLOADED", "DECODED", "PASS", "FAIL"}:
        set_state(result, result_file, "WAITING_REVIEW")
        try:
            history = wait_for_published(executable, biliup_cookies, bvid, poll_interval=poll_interval,
                                         timeout=approval_timeout, max_pages=status_max_pages,
                                         log_path=run_dir / "review.log",
                                         on_poll=lambda rows: (result.update(review_history=rows), write_json(result_file, result)))
        except TimeoutError as exc:
            set_state(result, result_file, "INCOMPLETE", error=str(exc)); write_report(result, run_dir / "REPORT.md"); return result
        except RuntimeError as exc:
            set_state(result, result_file, "BLOCKED", error=str(exc)); write_report(result, run_dir / "REPORT.md"); return result
        set_state(result, result_file, "PUBLISHED", review_history=history,
                  review_elapsed_seconds=time.monotonic() - review_started, published=True)
        run_logged(biliup_base(executable, biliup_cookies) + ["show", bvid], run_dir / "show.log", echo=False)
    temporary_cookie = None
    try:
        temporary_cookie, cookie_count = _temporary_cookie(biliup_cookies)
        if result.get("state") not in {"DOWNLOADED", "DECODED", "PASS", "FAIL"}:
            set_state(result, result_file, "WAITING_1080P")
            rendition_started = time.monotonic()
            try:
                metadata, selected, history = wait_for_1080p(result["share_url"], temporary_cookie,
                    poll_interval=poll_interval, timeout=rendition_timeout, formats_log=run_dir / "formats.log",
                    on_poll=lambda rows: (result.update(format_history=rows), write_json(result_file, result)))
            except TimeoutError as exc:
                set_state(result, result_file, "INCOMPLETE", error=str(exc)); write_report(result, run_dir / "REPORT.md"); return result
            write_json(run_dir / "formats.json", _safe_formats(metadata))
            platform_path = run_dir / "platform-1080p.mp4"
            download_format(result["share_url"], str(selected["format_id"]), platform_path, temporary_cookie, run_dir / "download.log")
            set_state(result, result_file, "DOWNLOADED", rendition_detection_elapsed_seconds=time.monotonic() - rendition_started,
                      platform={"path": str(platform_path), "format_id": selected.get("format_id"), "codec": selected.get("vcodec"),
                                "width": selected.get("width"), "height": selected.get("height"), "cookie_count_used": cookie_count})
        platform_path = Path(result["platform"]["path"])
        platform_probe = probe_video(platform_path); validate_1080p_probe(platform_probe, "platform")
        write_json(run_dir / "platform_ffprobe.json", platform_probe)
        result["platform"].update({key: platform_probe.get(key) for key in ("size_bytes", "duration_seconds", "calculated_bitrate_bps", "frame_count", "fps")})
        result["platform"]["stream"] = platform_probe.get("stream")
        if result.get("state") not in {"DECODED", "PASS", "FAIL"}:
            try:
                decoded = production_decode_summary(platform_path, run_dir / "recovered")
            except Exception as exc:
                diagnostic_file = result_path(run_dir / "recovered")
                decoded = decode_summary(json.loads(diagnostic_file.read_text(encoding="utf-8"))) if diagnostic_file.exists() else {}
                expected_k = result["preflight"]["expected_source_symbols"]
                decoded["source_symbols_expected"] = expected_k
                decoded["source_erasures"] = max(0, expected_k - (decoded.get("source_symbols_received") or 0))
                decoded["failed_blocks"] = parse_failed_blocks(str(exc))
                observed = decoded.get("observed_frames")
                if platform_probe.get("frame_count") is None and observed is not None:
                    platform_probe["frame_count"] = observed
                    platform_probe["frame_count_source"] = "production decoder observed_frames (ffprobe nb_frames unavailable)"
                    result["platform"]["frame_count"] = observed
                    result["platform"]["frame_count_source"] = platform_probe["frame_count_source"]
                    write_json(run_dir / "platform_ffprobe.json", platform_probe)
                result["decode"] = decoded
                result["next_decision"] = "Analyze RS, transport, and block-local erasures before changing parameters."
                set_state(result, result_file, "FAIL", error=f"production decode failed: {type(exc).__name__}: {exc}")
            expected_k = result["preflight"]["expected_source_symbols"]
            decoded["source_symbols_expected"] = expected_k
            decoded["source_erasures"] = max(0, expected_k - (decoded.get("source_symbols_received") or 0))
            result["decode"] = decoded
            set_state(result, result_file, "DECODED")
        if not result["decode"].get("failed_blocks"):
            result["decode"]["failed_blocks"] = parse_failed_blocks(result.get("error"))
        if not result["decode"].get("rs_failure_distribution"):
            diagnostic_file = result_path(run_dir / "recovered")
            if diagnostic_file.exists():
                diagnostics = json.loads(diagnostic_file.read_text(encoding="utf-8"))
                result["decode"]["rs_failure_distribution"] = summarize_rs_failures(diagnostics.get("rs_failures", {}))
        observed = result["decode"].get("observed_frames")
        if result["platform"].get("frame_count") is None and observed is not None:
            platform_probe["frame_count"] = observed
            platform_probe["frame_count_source"] = "production decoder observed_frames (ffprobe nb_frames unavailable)"
            result["platform"]["frame_count"] = observed
            result["platform"]["frame_count_source"] = platform_probe["frame_count_source"]
            write_json(run_dir / "platform_ffprobe.json", platform_probe)
        result["decode"]["rs_frame_failure_rate"] = ((result["decode"].get("rs_failed_frames") or 0)
            / result["decode"]["observed_frames"] if result["decode"].get("observed_frames") else None)
        result["platform"]["frame_count_delta"] = (result["platform"].get("frame_count") - result["preflight"]["source"].get("frame_count")
            if result["platform"].get("frame_count") is not None and result["preflight"]["source"].get("frame_count") is not None else None)
        source_probe = result["preflight"]["source"]
        raw_result_file = run_dir / "raw_channel.json"
        if raw_result_file.exists() and result.get("raw_channel"):
            channel = json.loads(raw_result_file.read_text(encoding="utf-8"))
        elif alignment_allows_positional(source_probe, platform_probe, result["decode"]):
            channel = raw_oracle(platform_path, original); channel["mode"] = "positional"
        else:
            channel = {"mode": "unavailable", "reason": "frame/transport alignment is not trustworthy",
                       "raw_ber": None, "frames_with_raw_errors": None}
        result["raw_channel"] = channel
        write_json(raw_result_file, channel)
        proxy = result["preflight"]["local_proxy"]
        result["proxy_comparison"] = compare_proxy_platform(proxy, {**channel, **result["decode"]})
        exact = recovery_pass(result["decode"], result["preflight"]["original"]["sha256"])
        if exact:
            if result["decode"].get("rs_failed_frames", 0) <= 2 and result["decode"].get("repair_symbols_received", 0) < 40:
                result["next_decision"] = "6PX BINARY CLIFF SWEEP: RECOMMENDED"
            elif result["proxy_comparison"]["verdict"] == "UNDER-STRESS":
                result["next_decision"] = "PROXY RECALIBRATION RECOMMENDED"
            else: result["next_decision"] = "Further decision requires review of the single sample."
            set_state(result, result_file, "PASS")
        else:
            result["next_decision"] = "Analyze RS, transport, and block-local erasures before changing parameters."
            set_state(result, result_file, "FAIL")
        write_report(result, run_dir / "REPORT.md")
        return result
    finally:
        if temporary_cookie: temporary_cookie.unlink(missing_ok=True)
