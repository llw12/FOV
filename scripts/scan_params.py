"""Automate FOV parameter scans with local and platform-like FFmpeg validation.

This script intentionally treats the FFmpeg transcode as a reproducible approximation
of a video-platform transcode, not an exact model of any platform's private pipeline.
The generated source MP4 files are kept so the best candidates can be uploaded for
manual platform validation afterwards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "test.zip"

# Targeted defaults follow the currently useful search region. The 720p cases map
# the known box_size=8 range; the 1080p cases probe the larger QR payload range.
TARGETED_BASE_CASES = [
    (1280, 720, 200),
    (1280, 720, 240),
    (1280, 720, 280),
    (1920, 1080, 400),
    (1920, 1080, 500),
    (1920, 1080, 600),
    (1920, 1080, 700),
]
FULL_SYMBOL_SIZES = [200, 240, 280, 400, 500, 600, 700]
FULL_RESOLUTIONS = [(1280, 720), (1920, 1080)]

# Approximate H.264 stress profiles. 720p ~= 900 kbps is based on an observed
# platform rendition from the current FOV experiments. 1080p=1500 kbps is a
# deliberately conservative approximation until more measured platform samples
# are collected. Both are CLI-overridable.
DEFAULT_SIM_BITRATE_720_KBPS = 900
DEFAULT_SIM_BITRATE_1080_KBPS = 1500


@dataclass(frozen=True)
class ScanCase:
    width: int
    height: int
    symbol_size: int
    repair: float
    fps: int = 30

    @property
    def case_id(self) -> str:
        repair_pct = round(self.repair * 100)
        return f"{self.width}x{self.height}_s{self.symbol_size}_r{repair_pct:02d}"


@dataclass
class CaseResult:
    case_id: str
    width: int
    height: int
    fps: int
    symbol_size: int
    repair: float
    input_bytes: int
    encode_ok: bool = False
    encode_seconds: float | None = None
    encode_error: str = ""
    source_video: str = ""
    source_video_bytes: int | None = None
    source_frames: int | None = None
    source_duration_s: float | None = None
    source_bitrate_bps: int | None = None
    effective_file_Bps: float | None = None
    local_ok: bool = False
    local_qr_decoded: int | None = None
    local_qr_failed: int | None = None
    local_crc_failed: int | None = None
    local_post_decode_symbols: int | None = None
    local_error: str = ""
    sim_profile: str = ""
    sim_target_bitrate_kbps: int | None = None
    sim_video: str = ""
    sim_video_bytes: int | None = None
    sim_bitrate_bps: int | None = None
    sim_ok: bool = False
    sim_qr_decoded: int | None = None
    sim_qr_failed: int | None = None
    sim_crc_failed: int | None = None
    sim_post_decode_symbols: int | None = None
    sim_error: str = ""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str
    elapsed_s: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_case(text: str) -> ScanCase:
    """Parse WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]."""
    match = re.fullmatch(r"(\d+)x(\d+):(\d+)(?::([0-9.]+))?(?::(\d+))?", text.strip())
    if not match:
        raise argparse.ArgumentTypeError("case must be WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]")
    width, height, symbol_size = map(int, match.group(1, 2, 3))
    repair = float(match.group(4) or 0.20)
    fps = int(match.group(5) or 30)
    if min(width, height, symbol_size, fps) <= 0 or not 0 <= repair <= 5:
        raise argparse.ArgumentTypeError("case contains an invalid numeric value")
    return ScanCase(width, height, symbol_size, repair, fps)


def metric(text: str, label: str) -> int | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(\d+)\s*$", text, re.MULTILINE)
    return int(match.group(1)) if match else None


def parse_decoder_output(text: str) -> dict[str, int | None]:
    return {
        "qr_decoded": metric(text, "decoded"),
        "qr_failed": metric(text, "failed"),
        "crc_failed": metric(text, "CRC failed"),
        "post_decode_symbols": metric(text, "post-decode symbols"),
    }


def run_logged(command: list[str], log_path: Path, *, verbose: bool = False) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
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
            if verbose:
                print(line, end="")
        returncode = process.wait()
    return CommandResult(returncode, "".join(lines), time.perf_counter() - started)


def ffprobe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,bit_rate,nb_frames,avg_frame_rate:format=duration,size,bit_rate",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
    )
    payload = json.loads(completed.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}

    def integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def number(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    return {
        "width": integer(stream.get("width")),
        "height": integer(stream.get("height")),
        "frames": integer(stream.get("nb_frames")),
        "duration_s": number(fmt.get("duration")),
        "size_bytes": integer(fmt.get("size")) or path.stat().st_size,
        "bitrate_bps": integer(stream.get("bit_rate")) or integer(fmt.get("bit_rate")),
    }


def error_tail(text: str, lines: int = 6) -> str:
    compact = [line.strip() for line in text.splitlines() if line.strip()]
    return " | ".join(compact[-lines:])[-1200:]


def independent_decode_check(output_dir: Path, expected_sha256: str) -> bool:
    for candidate in output_dir.iterdir():
        if candidate.is_file() and not candidate.name.startswith("fov-recover-"):
            try:
                if sha256_file(candidate) == expected_sha256:
                    return True
            except OSError:
                continue
    return False


def validate_video(
    video_path: Path,
    output_dir: Path,
    expected_sha256: str,
    log_path: Path,
    *,
    verbose: bool,
) -> tuple[bool, dict[str, int | None], str]:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_logged([sys.executable, "video2file.py", str(video_path), str(output_dir)], log_path, verbose=verbose)
    stats = parse_decoder_output(result.output)
    if result.returncode != 0:
        return False, stats, error_tail(result.output)
    if not independent_decode_check(output_dir, expected_sha256):
        return False, stats, "decoder exited successfully but independent SHA256 verification failed"
    return True, stats, ""


def simulation_bitrate_kbps(case: ScanCase, bitrate_720: int, bitrate_1080: int) -> int:
    if case.height <= 720:
        return bitrate_720
    return bitrate_1080


def transcode_platform_like(
    source: Path,
    target: Path,
    case: ScanCase,
    bitrate_kbps: int,
    log_path: Path,
    *,
    verbose: bool,
) -> CommandResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-i", str(source), "-an",
        "-vf", f"scale={case.width}:{case.height}:flags=lanczos",
        "-r", str(case.fps), "-fps_mode", "cfr",
        "-c:v", "libx264", "-profile:v", "high", "-preset", "medium",
        "-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k",
        "-bufsize", f"{bitrate_kbps * 2}k",
        "-g", "250", "-bf", "3", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(target),
    ]
    return run_logged(command, log_path, verbose=verbose)


def build_cases(args: argparse.Namespace) -> list[ScanCase]:
    if args.case:
        # Preserve order but remove exact duplicates.
        return list(dict.fromkeys(args.case))

    repairs = args.repairs
    bases: Iterable[tuple[int, int, int]]
    if args.preset == "targeted":
        bases = TARGETED_BASE_CASES
    else:
        bases = [(width, height, symbol) for width, height in FULL_RESOLUTIONS for symbol in FULL_SYMBOL_SIZES]
    return [ScanCase(width, height, symbol, repair, args.fps) for width, height, symbol in bases for repair in repairs]


def save_results(results: list[CaseResult], run_dir: Path) -> None:
    rows = [asdict(item) for item in results]
    json_path = run_dir / "results.json"
    csv_path = run_dir / "results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_summary(results, run_dir / "summary.md")
    write_manual_queue(results, run_dir / "manual_queue.csv")


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def write_summary(results: list[CaseResult], path: Path) -> None:
    lines = [
        "# FOV parameter scan\n",
        "> FFmpeg simulation is a reproducible platform-like approximation, not an exact model of Bilibili's private encoder.\n",
        "| case | encode | local | simulated | QR fail(sim) | source bitrate | throughput | source video |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        qr_fail = "-"
        if item.sim_qr_failed is not None and item.sim_qr_decoded is not None:
            total = item.sim_qr_failed + item.sim_qr_decoded
            qr_fail = f"{item.sim_qr_failed}/{total} ({item.sim_qr_failed / total:.2%})" if total else "-"
        bitrate = f"{item.source_bitrate_bps / 1000:.0f} kbps" if item.source_bitrate_bps else "-"
        throughput = f"{item.effective_file_Bps / 1000:.2f} KB/s" if item.effective_file_Bps else "-"
        lines.append(
            f"| {item.case_id} | {status(item.encode_ok)} | {status(item.local_ok)} | {status(item.sim_ok)} | "
            f"{qr_fail} | {bitrate} | {throughput} | `{item.source_video}` |"
        )
    passed = sorted(
        (item for item in results if item.local_ok and item.sim_ok),
        key=lambda item: item.effective_file_Bps or 0,
        reverse=True,
    )
    lines.extend(["", "## Manual validation candidates", ""])
    if not passed:
        lines.append("No case passed both local and simulated validation.")
    else:
        lines.append("Upload the original source MP4, not the simulated MP4. Highest-throughput candidates are listed first.\n")
        for index, item in enumerate(passed, 1):
            lines.append(f"{index}. `{item.case_id}` — `{item.source_video}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_queue(results: list[CaseResult], path: Path) -> None:
    candidates = sorted(
        (item for item in results if item.local_ok and item.sim_ok),
        key=lambda item: item.effective_file_Bps or 0,
        reverse=True,
    )
    fields = [
        "case_id", "width", "height", "symbol_size", "repair", "source_video",
        "effective_file_Bps", "sim_qr_failed", "sim_qr_decoded",
        "platform_video_path", "platform_qr_decoded", "platform_qr_failed",
        "platform_crc_failed", "platform_sha256_match", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({
                "case_id": item.case_id,
                "width": item.width,
                "height": item.height,
                "symbol_size": item.symbol_size,
                "repair": item.repair,
                "source_video": item.source_video,
                "effective_file_Bps": item.effective_file_Bps,
                "sim_qr_failed": item.sim_qr_failed,
                "sim_qr_decoded": item.sim_qr_decoded,
                "platform_video_path": "",
                "platform_qr_decoded": "",
                "platform_qr_failed": "",
                "platform_crc_failed": "",
                "platform_sha256_match": "",
                "notes": "",
            })


def ensure_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("missing required executable(s) on PATH: " + ", ".join(missing))


def run_case(
    case: ScanCase,
    input_path: Path,
    expected_sha256: str,
    run_dir: Path,
    *,
    bitrate_720: int,
    bitrate_1080: int,
    verbose: bool,
    keep_sim_video: bool,
) -> CaseResult:
    case_dir = run_dir / "cases" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    source_video = case_dir / "source.mp4"
    result = CaseResult(
        case_id=case.case_id,
        width=case.width,
        height=case.height,
        fps=case.fps,
        symbol_size=case.symbol_size,
        repair=case.repair,
        input_bytes=input_path.stat().st_size,
        source_video=str(source_video.relative_to(REPO_ROOT) if source_video.is_relative_to(REPO_ROOT) else source_video),
    )

    print(f"\n=== {case.case_id} ===")
    print("[1/3] encode")
    encode_command = [
        sys.executable, "file2video.py", str(input_path), str(source_video),
        "--symbol-size", str(case.symbol_size), "--repair", str(case.repair),
        "--fps", str(case.fps), "--width", str(case.width), "--height", str(case.height),
    ]
    encoded = run_logged(encode_command, case_dir / "encode.log", verbose=verbose)
    result.encode_seconds = round(encoded.elapsed_s, 3)
    if encoded.returncode != 0 or not source_video.exists():
        result.encode_error = error_tail(encoded.output)
        print(f"  FAIL: {result.encode_error}")
        return result
    result.encode_ok = True
    try:
        probe = ffprobe_video(source_video)
        result.source_video_bytes = probe["size_bytes"]
        result.source_frames = probe["frames"]
        result.source_duration_s = probe["duration_s"]
        result.source_bitrate_bps = probe["bitrate_bps"]
        if result.source_duration_s:
            result.effective_file_Bps = result.input_bytes / result.source_duration_s
    except Exception as exc:  # ffprobe metadata is useful but not required for decode validation.
        result.encode_error = f"ffprobe warning: {exc}"

    print("[2/3] local decode")
    local_ok, local_stats, local_error = validate_video(
        source_video, case_dir / "local_recovered", expected_sha256, case_dir / "local_decode.log", verbose=verbose
    )
    result.local_ok = local_ok
    result.local_qr_decoded = local_stats["qr_decoded"]
    result.local_qr_failed = local_stats["qr_failed"]
    result.local_crc_failed = local_stats["crc_failed"]
    result.local_post_decode_symbols = local_stats["post_decode_symbols"]
    result.local_error = local_error
    if not local_ok:
        print(f"  FAIL: {local_error}")
        return result

    bitrate_kbps = simulation_bitrate_kbps(case, bitrate_720, bitrate_1080)
    result.sim_profile = f"platform_like_h264_{bitrate_kbps}k"
    result.sim_target_bitrate_kbps = bitrate_kbps
    sim_video = case_dir / "sim_platform.mp4"
    result.sim_video = str(sim_video.relative_to(REPO_ROOT) if sim_video.is_relative_to(REPO_ROOT) else sim_video)
    print(f"[3/3] simulated platform transcode @ {bitrate_kbps} kbps")
    transcoded = transcode_platform_like(
        source_video, sim_video, case, bitrate_kbps, case_dir / "sim_ffmpeg.log", verbose=verbose
    )
    if transcoded.returncode != 0 or not sim_video.exists():
        result.sim_error = error_tail(transcoded.output)
        print(f"  FAIL: {result.sim_error}")
        return result
    try:
        sim_probe = ffprobe_video(sim_video)
        result.sim_video_bytes = sim_probe["size_bytes"]
        result.sim_bitrate_bps = sim_probe["bitrate_bps"]
    except Exception as exc:
        result.sim_error = f"ffprobe warning: {exc}"

    sim_ok, sim_stats, sim_error = validate_video(
        sim_video, case_dir / "sim_recovered", expected_sha256, case_dir / "sim_decode.log", verbose=verbose
    )
    result.sim_ok = sim_ok
    result.sim_qr_decoded = sim_stats["qr_decoded"]
    result.sim_qr_failed = sim_stats["qr_failed"]
    result.sim_crc_failed = sim_stats["crc_failed"]
    result.sim_post_decode_symbols = sim_stats["post_decode_symbols"]
    if sim_error:
        result.sim_error = sim_error if not result.sim_error else result.sim_error + " | " + sim_error
    print("  PASS" if sim_ok else f"  FAIL: {result.sim_error}")

    if not keep_sim_video:
        sim_video.unlink(missing_ok=True)
        result.sim_video += " (deleted after validation)"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan FOV video parameters and validate local/platform-like roundtrips")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--preset", choices=("targeted", "full"), default="targeted")
    parser.add_argument("--repairs", type=float, nargs="+", default=[0.20], help="repair ratios used with preset cases")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--case", type=parse_case, action="append", help="custom WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]; repeatable")
    parser.add_argument("--sim-bitrate-720", type=int, default=DEFAULT_SIM_BITRATE_720_KBPS)
    parser.add_argument("--sim-bitrate-1080", type=int, default=DEFAULT_SIM_BITRATE_1080_KBPS)
    parser.add_argument("--keep-sim-videos", action="store_true", help="keep simulated transcodes; source videos are always kept")
    parser.add_argument("--verbose", action="store_true", help="echo child process output while also saving logs")
    args = parser.parse_args()

    ensure_tools()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input file not found: {input_path}")
    if input_path.stat().st_size <= 0:
        raise SystemExit("input file is empty")
    for repair in args.repairs:
        if not 0 <= repair <= 5:
            raise SystemExit(f"invalid repair ratio: {repair}")

    cases = build_cases(args)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.output_root.resolve() / f"scan-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    expected_sha256 = sha256_file(input_path)

    config = {
        "input": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "input_sha256": expected_sha256,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "cases": [asdict(case) for case in cases],
        "simulation": {
            "description": "reproducible Bilibili-like H.264 approximation, not an exact platform model",
            "bitrate_720_kbps": args.sim_bitrate_720,
            "bitrate_1080_kbps": args.sim_bitrate_1080,
            "codec": "libx264 High / yuv420p / CFR / GOP 250 / 3 B-frames",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"FOV parameter scan\nInput: {input_path}\nSHA256: {expected_sha256}\nCases: {len(cases)}\nRun: {run_dir}")
    results: list[CaseResult] = []
    try:
        for index, case in enumerate(cases, 1):
            print(f"\n[{index}/{len(cases)}] {case.case_id}")
            try:
                item = run_case(
                    case, input_path, expected_sha256, run_dir,
                    bitrate_720=args.sim_bitrate_720,
                    bitrate_1080=args.sim_bitrate_1080,
                    verbose=args.verbose,
                    keep_sim_video=args.keep_sim_videos,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                item = CaseResult(
                    case_id=case.case_id, width=case.width, height=case.height, fps=case.fps,
                    symbol_size=case.symbol_size, repair=case.repair, input_bytes=input_path.stat().st_size,
                    encode_error=f"unexpected scan error: {type(exc).__name__}: {exc}",
                )
                print(f"  ERROR: {item.encode_error}")
            results.append(item)
            save_results(results, run_dir)  # Persist after every case so long scans are crash-tolerant.
    except KeyboardInterrupt:
        print("\nInterrupted; partial results have been preserved.")
        save_results(results, run_dir)
        raise SystemExit(130)

    passed = sum(item.local_ok and item.sim_ok for item in results)
    print(f"\nDone: {passed}/{len(results)} cases passed local + simulated validation")
    print(f"Results: {run_dir / 'results.csv'}")
    print(f"Summary: {run_dir / 'summary.md'}")
    print(f"Manual queue: {run_dir / 'manual_queue.csv'}")


if __name__ == "__main__":
    main()
