"""Automate FOV parameter scans with local and repeatable FFmpeg stress validation.

The FFmpeg stress transcodes are intentionally empirical screening profiles, not an
exact model of any platform's private pipeline. Source MP4 files are retained so the
best candidates can be uploaded for manual platform validation afterwards.
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
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "test.zip"

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

# The real 1080p->platform sample observed in this project was delivered as 720p,
# while simple 720p/2.1 Mbps FFmpeg transcodes were still milder than the platform.
# Scan a bitrate ladder instead of pretending one bitrate is an exact platform model.
DEFAULT_STRESS_BITRATES_KBPS = [2100, 1400, 1000, 700]
DEFAULT_STRESS_WIDTH = 1280
DEFAULT_STRESS_HEIGHT = 720
DEFAULT_STRESS_SCALE = "bicubic"


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
class StressResult:
    target_bitrate_kbps: int
    target_width: int
    target_height: int
    scale_flags: str
    video: str = ""
    video_bytes: int | None = None
    bitrate_bps: int | None = None
    ok: bool = False
    qr_decoded: int | None = None
    qr_failed: int | None = None
    crc_failed: int | None = None
    post_decode_symbols: int | None = None
    failed_frame_indices: list[int] = field(default_factory=list)
    failed_frame_indices_truncated: bool = False
    decoded_at_frames: list[int] = field(default_factory=list)
    error: str = ""

    @property
    def profile_id(self) -> str:
        return f"{self.target_width}x{self.target_height}_{self.scale_flags}_{self.target_bitrate_kbps}k"


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
    local_failed_frame_indices: list[int] = field(default_factory=list)
    local_failed_frame_indices_truncated: bool = False
    local_decoded_at_frames: list[int] = field(default_factory=list)
    local_error: str = ""
    stress_results: list[StressResult] = field(default_factory=list)


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


def text_metric(text: str, label: str) -> str | None:
    match = re.search(rf"^\s*{re.escape(label)}:\s*(.*?)\s*$", text, re.MULTILINE)
    return match.group(1) if match else None


def parse_int_list(value: str | None) -> list[int]:
    if value is None or value.strip() in {"", "-"}:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_decoder_output(text: str) -> dict[str, Any]:
    failed_indices = parse_int_list(text_metric(text, "failed frame indices (0-based)"))
    truncated = text_metric(text, "failed frame indices truncated")
    decoded_at_frames = [
        int(value) for value in re.findall(r"^\s*decoded at frame:\s*(\d+)\s*$", text, re.MULTILINE)
    ]
    return {
        "qr_decoded": metric(text, "decoded"),
        "qr_failed": metric(text, "failed"),
        "crc_failed": metric(text, "CRC failed"),
        "post_decode_symbols": metric(text, "post-decode symbols"),
        "failed_frame_indices": failed_indices,
        "failed_frame_indices_truncated": truncated == "yes",
        "decoded_at_frames": decoded_at_frames,
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
) -> tuple[bool, dict[str, Any], str]:
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = run_logged([sys.executable, "video2file.py", str(video_path), str(output_dir)], log_path, verbose=verbose)
    stats = parse_decoder_output(result.output)
    if result.returncode != 0:
        return False, stats, error_tail(result.output)
    if not independent_decode_check(output_dir, expected_sha256):
        return False, stats, "decoder exited successfully but independent SHA256 verification failed"
    return True, stats, ""


def transcode_stress(
    source: Path,
    target: Path,
    case: ScanCase,
    bitrate_kbps: int,
    target_width: int,
    target_height: int,
    scale_flags: str,
    log_path: Path,
    *,
    verbose: bool,
) -> CommandResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-i", str(source), "-an",
        "-vf", f"scale={target_width}:{target_height}:flags={scale_flags}",
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
        return list(dict.fromkeys(args.case))

    repairs = args.repairs
    bases: Iterable[tuple[int, int, int]]
    if args.preset == "targeted":
        bases = TARGETED_BASE_CASES
    else:
        bases = [(width, height, symbol) for width, height in FULL_RESOLUTIONS for symbol in FULL_SYMBOL_SIZES]
    return [ScanCase(width, height, symbol, repair, args.fps) for width, height, symbol in bases for repair in repairs]


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)


def _stress_pass_floor(item: CaseResult) -> int | None:
    passed = [stress.target_bitrate_kbps for stress in item.stress_results if stress.ok]
    return min(passed) if passed else None


def _stress_failure_cell(stress: StressResult | None) -> str:
    if stress is None:
        return "-"
    status = "PASS" if stress.ok else "FAIL"
    if stress.qr_failed is None or stress.qr_decoded is None:
        return status
    total = stress.qr_failed + stress.qr_decoded
    return f"{status} {stress.qr_failed}/{total}" if total else status


def save_results(results: list[CaseResult], run_dir: Path) -> None:
    json_payload = [asdict(item) for item in results]
    (run_dir / "results.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flat_rows: list[dict[str, Any]] = []
    for item in results:
        base = {
            "case_id": item.case_id,
            "width": item.width,
            "height": item.height,
            "fps": item.fps,
            "symbol_size": item.symbol_size,
            "repair": item.repair,
            "input_bytes": item.input_bytes,
            "encode_ok": item.encode_ok,
            "encode_seconds": item.encode_seconds,
            "source_video": item.source_video,
            "source_video_bytes": item.source_video_bytes,
            "source_frames": item.source_frames,
            "source_duration_s": item.source_duration_s,
            "source_bitrate_bps": item.source_bitrate_bps,
            "effective_file_Bps": item.effective_file_Bps,
            "local_ok": item.local_ok,
            "local_qr_decoded": item.local_qr_decoded,
            "local_qr_failed": item.local_qr_failed,
            "local_crc_failed": item.local_crc_failed,
            "local_decoded_at_frames": ",".join(map(str, item.local_decoded_at_frames)),
            "local_error": item.local_error,
        }
        if not item.stress_results:
            flat_rows.append(base)
        for stress in item.stress_results:
            row = dict(base)
            row.update({
                "stress_profile": stress.profile_id,
                "stress_target_bitrate_kbps": stress.target_bitrate_kbps,
                "stress_target_width": stress.target_width,
                "stress_target_height": stress.target_height,
                "stress_scale_flags": stress.scale_flags,
                "stress_video": stress.video,
                "stress_video_bytes": stress.video_bytes,
                "stress_bitrate_bps": stress.bitrate_bps,
                "stress_ok": stress.ok,
                "stress_qr_decoded": stress.qr_decoded,
                "stress_qr_failed": stress.qr_failed,
                "stress_crc_failed": stress.crc_failed,
                "stress_post_decode_symbols": stress.post_decode_symbols,
                "stress_failed_frame_indices": ",".join(map(str, stress.failed_frame_indices)),
                "stress_failed_frame_indices_truncated": stress.failed_frame_indices_truncated,
                "stress_decoded_at_frames": ",".join(map(str, stress.decoded_at_frames)),
                "stress_error": stress.error,
            })
            flat_rows.append(row)

    csv_path = run_dir / "results.csv"
    if flat_rows:
        fieldnames: list[str] = []
        for row in flat_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    write_summary(results, run_dir / "summary.md")
    write_manual_queue(results, run_dir / "manual_queue.csv")


def write_summary(results: list[CaseResult], path: Path) -> None:
    bitrates = sorted(
        {stress.target_bitrate_kbps for item in results for stress in item.stress_results},
        reverse=True,
    )
    header = ["case", "local", "throughput"] + [f"{bitrate}k" for bitrate in bitrates] + ["source video"]
    lines = [
        "# FOV parameter × stress bitrate scan",
        "",
        "> Stress profiles are reproducible screening transcodes, not an exact model of Bilibili's private encoder.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for item in results:
        throughput = f"{item.effective_file_Bps / 1000:.2f} KB/s" if item.effective_file_Bps else "-"
        by_bitrate = {stress.target_bitrate_kbps: stress for stress in item.stress_results}
        cells = [
            item.case_id,
            "PASS" if item.local_ok else "FAIL",
            throughput,
            *[_stress_failure_cell(by_bitrate.get(bitrate)) for bitrate in bitrates],
            f"`{item.source_video}`" if item.source_video else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    candidates = sorted(
        (item for item in results if item.local_ok and any(stress.ok for stress in item.stress_results)),
        key=lambda item: (-(item.effective_file_Bps or 0), _stress_pass_floor(item) or 10**9),
    )
    lines.extend(["", "## Manual validation candidates", ""])
    if not candidates:
        lines.append("No case passed local validation plus any stress profile.")
    else:
        lines.append(
            "Upload the original source MP4, not a stress MP4. "
            "`pass floor` is the lowest tested bitrate that still recovered the file."
        )
        lines.append("")
        for index, item in enumerate(candidates, 1):
            lines.append(
                f"{index}. `{item.case_id}` — pass floor `{_stress_pass_floor(item)}k` — `{item.source_video}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manual_queue(results: list[CaseResult], path: Path) -> None:
    candidates = sorted(
        (item for item in results if item.local_ok and any(stress.ok for stress in item.stress_results)),
        key=lambda item: (-(item.effective_file_Bps or 0), _stress_pass_floor(item) or 10**9),
    )
    fields = [
        "case_id", "width", "height", "symbol_size", "repair", "source_video",
        "effective_file_Bps", "stress_pass_floor_kbps", "stress_pass_bitrates_kbps",
        "platform_video_path", "platform_qr_decoded", "platform_qr_failed",
        "platform_crc_failed", "platform_sha256_match", "platform_failed_frame_indices", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            pass_bitrates = sorted(
                (stress.target_bitrate_kbps for stress in item.stress_results if stress.ok),
                reverse=True,
            )
            writer.writerow({
                "case_id": item.case_id,
                "width": item.width,
                "height": item.height,
                "symbol_size": item.symbol_size,
                "repair": item.repair,
                "source_video": item.source_video,
                "effective_file_Bps": item.effective_file_Bps,
                "stress_pass_floor_kbps": min(pass_bitrates) if pass_bitrates else "",
                "stress_pass_bitrates_kbps": ",".join(map(str, pass_bitrates)),
                "platform_video_path": "",
                "platform_qr_decoded": "",
                "platform_qr_failed": "",
                "platform_crc_failed": "",
                "platform_sha256_match": "",
                "platform_failed_frame_indices": "",
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
    stress_bitrates: list[int],
    stress_width: int,
    stress_height: int,
    stress_scale: str,
    verbose: bool,
    keep_stress_videos: bool,
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
        source_video=_relative(source_video),
    )

    print(f"\n=== {case.case_id} ===")
    print("[1/2] encode")
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
    except Exception as exc:
        result.encode_error = f"ffprobe warning: {exc}"

    print("[2/2] local decode")
    local_ok, local_stats, local_error = validate_video(
        source_video, case_dir / "local_recovered", expected_sha256, case_dir / "local_decode.log", verbose=verbose
    )
    result.local_ok = local_ok
    result.local_qr_decoded = local_stats["qr_decoded"]
    result.local_qr_failed = local_stats["qr_failed"]
    result.local_crc_failed = local_stats["crc_failed"]
    result.local_post_decode_symbols = local_stats["post_decode_symbols"]
    result.local_failed_frame_indices = local_stats["failed_frame_indices"]
    result.local_failed_frame_indices_truncated = local_stats["failed_frame_indices_truncated"]
    result.local_decoded_at_frames = local_stats["decoded_at_frames"]
    result.local_error = local_error
    if not local_ok:
        print(f"  FAIL: {local_error}")
        return result

    for stress_index, bitrate_kbps in enumerate(stress_bitrates, 1):
        stress = StressResult(
            target_bitrate_kbps=bitrate_kbps,
            target_width=stress_width,
            target_height=stress_height,
            scale_flags=stress_scale,
        )
        stress_dir = case_dir / "stress" / stress.profile_id
        sim_video = stress_dir / "video.mp4"
        stress.video = _relative(sim_video)
        print(
            f"[stress {stress_index}/{len(stress_bitrates)}] "
            f"{stress_width}x{stress_height} {stress_scale} @ {bitrate_kbps} kbps"
        )
        transcoded = transcode_stress(
            source_video, sim_video, case, bitrate_kbps, stress_width, stress_height, stress_scale,
            stress_dir / "ffmpeg.log", verbose=verbose,
        )
        if transcoded.returncode != 0 or not sim_video.exists():
            stress.error = error_tail(transcoded.output)
            result.stress_results.append(stress)
            print(f"  FAIL: {stress.error}")
            continue

        try:
            sim_probe = ffprobe_video(sim_video)
            stress.video_bytes = sim_probe["size_bytes"]
            stress.bitrate_bps = sim_probe["bitrate_bps"]
        except Exception as exc:
            stress.error = f"ffprobe warning: {exc}"

        sim_ok, sim_stats, sim_error = validate_video(
            sim_video, stress_dir / "recovered", expected_sha256, stress_dir / "decode.log", verbose=verbose
        )
        stress.ok = sim_ok
        stress.qr_decoded = sim_stats["qr_decoded"]
        stress.qr_failed = sim_stats["qr_failed"]
        stress.crc_failed = sim_stats["crc_failed"]
        stress.post_decode_symbols = sim_stats["post_decode_symbols"]
        stress.failed_frame_indices = sim_stats["failed_frame_indices"]
        stress.failed_frame_indices_truncated = sim_stats["failed_frame_indices_truncated"]
        stress.decoded_at_frames = sim_stats["decoded_at_frames"]
        if sim_error:
            stress.error = sim_error if not stress.error else stress.error + " | " + sim_error
        result.stress_results.append(stress)
        print("  PASS" if stress.ok else f"  FAIL: {stress.error}")

        if not keep_stress_videos:
            sim_video.unlink(missing_ok=True)
            stress.video += " (deleted after validation)"

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan FOV symbol parameters against a repeatable FFmpeg stress bitrate ladder"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--preset", choices=("targeted", "full"), default="targeted")
    parser.add_argument("--repairs", type=float, nargs="+", default=[0.20], help="repair ratios used with preset cases")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--case", type=parse_case, action="append",
        help="custom WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]; repeatable",
    )
    parser.add_argument(
        "--stress-bitrates", type=int, nargs="+", default=DEFAULT_STRESS_BITRATES_KBPS,
        help="bitrate ladder in kbps; each case is encoded once then tested at every bitrate",
    )
    parser.add_argument("--stress-width", type=int, default=DEFAULT_STRESS_WIDTH)
    parser.add_argument("--stress-height", type=int, default=DEFAULT_STRESS_HEIGHT)
    parser.add_argument(
        "--stress-scale",
        choices=("fast_bilinear", "bilinear", "bicubic", "lanczos"),
        default=DEFAULT_STRESS_SCALE,
    )
    parser.add_argument(
        "--keep-stress-videos", "--keep-sim-videos",
        dest="keep_stress_videos", action="store_true",
        help="keep stress transcodes; source videos are always kept",
    )
    parser.add_argument("--verbose", action="store_true", help="echo child output while also saving logs")
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
    if min(args.fps, args.stress_width, args.stress_height) <= 0:
        raise SystemExit("fps and stress dimensions must be positive")
    if not args.stress_bitrates or any(bitrate <= 0 for bitrate in args.stress_bitrates):
        raise SystemExit("stress bitrates must be positive")

    cases = build_cases(args)
    stress_bitrates = list(dict.fromkeys(args.stress_bitrates))
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
        "stress": {
            "description": "repeatable screening ladder, not an exact platform model",
            "target_width": args.stress_width,
            "target_height": args.stress_height,
            "scale_flags": args.stress_scale,
            "bitrates_kbps": stress_bitrates,
            "codec": "libx264 High / yuv420p / CFR / GOP 250 / 3 B-frames",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"FOV parameter × stress scan\n"
        f"Input: {input_path}\n"
        f"SHA256: {expected_sha256}\n"
        f"Cases: {len(cases)}\n"
        f"Stress bitrates: {stress_bitrates} kbps\n"
        f"Stress output: {args.stress_width}x{args.stress_height} ({args.stress_scale})\n"
        f"Run: {run_dir}"
    )

    results: list[CaseResult] = []
    try:
        for index, case in enumerate(cases, 1):
            print(f"\n[{index}/{len(cases)}] {case.case_id}")
            try:
                item = run_case(
                    case, input_path, expected_sha256, run_dir,
                    stress_bitrates=stress_bitrates,
                    stress_width=args.stress_width,
                    stress_height=args.stress_height,
                    stress_scale=args.stress_scale,
                    verbose=args.verbose,
                    keep_stress_videos=args.keep_stress_videos,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                item = CaseResult(
                    case_id=case.case_id,
                    width=case.width,
                    height=case.height,
                    fps=case.fps,
                    symbol_size=case.symbol_size,
                    repair=case.repair,
                    input_bytes=input_path.stat().st_size,
                    encode_error=f"unexpected scan error: {type(exc).__name__}: {exc}",
                )
                print(f"  ERROR: {item.encode_error}")
            results.append(item)
            save_results(results, run_dir)
    except KeyboardInterrupt:
        print("\nInterrupted; partial results have been preserved.")
        save_results(results, run_dir)
        raise SystemExit(130)

    local_passed = sum(item.local_ok for item in results)
    stress_passed = sum(stress.ok for item in results for stress in item.stress_results)
    stress_total = sum(len(item.stress_results) for item in results)
    print(
        f"\nDone: {local_passed}/{len(results)} cases passed local validation; "
        f"{stress_passed}/{stress_total} stress cells passed"
    )
    print(f"Results: {run_dir / 'results.csv'}")
    print(f"Summary: {run_dir / 'summary.md'}")
    print(f"Manual queue: {run_dir / 'manual_queue.csv'}")


if __name__ == "__main__":
    main()
