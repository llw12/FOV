"""Completely local FVM codec-efficiency experiment; production paths remain unchanged."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from file2video import estimate_packet_count, packet_stream, terminate_ffmpeg
from fov import BLOCK_SIZE, INTERLEAVE_WINDOW, build_metadata, derive_block_layout, file_id_from_sha256, make_raptorq_engine, sha256_file, symbol_packet_size
from fvm_codec_sweep_analysis import expansion_ratio, summarize, write_report
from fvm_file_common import FPS, HEIGHT, MAX_PACKET_BYTES, PHYSICAL_CONFIG, SYMBOL_SIZE, WIDTH, encode_transport_physical, physical_to_matrix
from fvm_video2file import decode as production_decode, result_path
from scripts.fvm0_common import decode_frame

FORMAT = "FVM_CODEC_SWEEP_V1"
TEST_PAYLOAD_DOMAIN = b"FVM_CODEC_SWEEP_32M_V1"
INPUT_SIZE = 32 * 1024 * 1024
REPAIR_RATIO = 0.03
PRESET = "slow"
LEVELS = ((0, 255), (48, 208), (64, 192), (80, 176))
CRFS = (18, 21, 24, 27, 30)


@dataclass(frozen=True)
class SweepCase:
    low_level: int
    high_level: int
    crf: int
    preset: str = PRESET

    def __post_init__(self) -> None:
        validate_levels(self.low_level, self.high_level)
        if not 0 <= self.crf <= 51:
            raise ValueError("crf must be between 0 and 51")

    @property
    def case_id(self) -> str:
        return f"level-{self.low_level:03d}-{self.high_level:03d}_crf{self.crf}"

    def config(self) -> dict[str, Any]:
        return {**asdict(self), "level_delta": self.high_level - self.low_level,
                "level_margin_low": 128 - self.low_level, "level_margin_high": self.high_level - 128,
                "width": WIDTH, "height": HEIGHT, "fps": FPS, "cell_size": PHYSICAL_CONFIG.cell_size,
                "symbol_size": SYMBOL_SIZE, "block_size": BLOCK_SIZE, "repair_ratio": REPAIR_RATIO,
                "interleave_window": INTERLEAVE_WINDOW}


def validate_levels(low: int, high: int) -> None:
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (low, high)) or not 0 <= low < 128 < high <= 255:
        raise ValueError("levels must satisfy 0 <= low < 128 < high <= 255")


def render_levels(bits: np.ndarray, low: int, high: int) -> np.ndarray:
    validate_levels(low, high)
    if bits.shape != (PHYSICAL_CONFIG.rows, PHYSICAL_CONFIG.cols) or bits.dtype != np.uint8:
        raise ValueError("bits must be a uint8 matrix matching fixed FVM geometry")
    cells = np.where(bits == 0, low, high).astype(np.uint8)
    pixels = np.repeat(np.repeat(cells, PHYSICAL_CONFIG.cell_size, axis=0), PHYSICAL_CONFIG.cell_size, axis=1)
    return np.repeat(pixels[:, :, None], 3, axis=2)


def cases() -> list[SweepCase]:
    return [SweepCase(low, high, crf) for low, high in LEVELS for crf in CRFS]


def parse_bitrate(value: str | int) -> int:
    if isinstance(value, int):
        result = value
    else:
        text = value.strip()
        multiplier = 1
        if text[-1:].lower() == "k": multiplier, text = 1000, text[:-1]
        elif text[-1:].lower() == "m": multiplier, text = 1_000_000, text[:-1]
        try: result = int(float(text) * multiplier)
        except ValueError as exc: raise ValueError(f"invalid bitrate: {value}") from exc
    if result <= 0: raise ValueError("bitrate must be positive")
    return result


def actual_bitrate(size_bytes: int, duration_seconds: float) -> float | None:
    return size_bytes * 8 / duration_seconds if duration_seconds > 0 else None


def _rate(value: str | None) -> float | None:
    if not value: return None
    try:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        return None


def validate_video_probe(probe: dict[str, Any], label: str) -> None:
    stream = probe.get("stream", {})
    if stream.get("width") not in (None, WIDTH) or stream.get("height") not in (None, HEIGHT):
        raise RuntimeError(f"{label} resolution is invalid")
    fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
    if fps is not None and abs(fps - FPS) > 0.05:
        raise RuntimeError(f"{label} fps is invalid: {fps}")


def stable_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_deterministic_file(path: Path, size: int = INPUT_SIZE, domain: bytes = TEST_PAYLOAD_DOMAIN, chunk_size: int = 1024 * 1024) -> str:
    if size <= 0 or chunk_size <= 0: raise ValueError("size and chunk_size must be positive")
    expected_hash = hashlib.sha256()
    offset = 0
    while offset < size:
        length = min(chunk_size, size - offset)
        expected_hash.update(hashlib.shake_256(domain + offset.to_bytes(8, "big")).digest(length))
        offset += length
    expected = expected_hash.hexdigest()
    if path.exists() and path.stat().st_size == size and sha256_file(path) == expected:
        return expected
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # hashlib has no streaming XOF read cursor, so generate independent, indexed chunks.
    with temporary.open("wb") as handle:
        offset = 0
        while offset < size:
            length = min(chunk_size, size - offset)
            handle.write(hashlib.shake_256(domain + offset.to_bytes(8, "big")).digest(length))
            offset += length
    temporary.replace(path)
    return sha256_file(path)


def ffprobe(path: Path) -> tuple[dict[str, Any], list[str]]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_frames,bit_rate:format=duration,size,bit_rate", "-of", "json", str(path)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        raw = json.loads(completed.stdout); stream = (raw.get("streams") or [{}])[0]; fmt = raw.get("format", {})
        duration = float(fmt.get("duration") or stream.get("duration") or 0)
        size = path.stat().st_size
        return {"path": str(path), "size_bytes": size, "duration_seconds": duration,
                "calculated_bitrate_bps": actual_bitrate(size, duration), "stream": stream, "format": fmt}, []
    except Exception as exc:
        return {"path": str(path), "size_bytes": path.stat().st_size if path.exists() else None}, [f"ffprobe failed: {exc}"]


def _layout(input_path: Path) -> tuple[dict[str, Any], list[Any], str]:
    digest = sha256_file(input_path)
    metadata = build_metadata(input_path.name, input_path.stat().st_size, digest, SYMBOL_SIZE, REPAIR_RATIO)
    layout = derive_block_layout(input_path.stat().st_size, metadata["block_size"], SYMBOL_SIZE, REPAIR_RATIO)
    return metadata, layout, digest


def encode_sweep_case(input_path: Path, output_path: Path, case: SweepCase) -> dict[str, Any]:
    if symbol_packet_size(SYMBOL_SIZE) > MAX_PACKET_BYTES: raise ValueError("symbol packet exceeds transport")
    metadata, layout, digest = _layout(input_path)
    total_frames, total_symbols, meta_frames = estimate_packet_count(layout)
    source_symbols = sum(block.source_symbols for block in layout)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", str(case.crf), "-preset", case.preset, "-pix_fmt", "yuv420p", str(output_path)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    emitted = 0
    try:
        assert process.stdin is not None
        stream = packet_stream(input_path, metadata, file_id_from_sha256(digest), make_raptorq_engine())
        for index, packet in enumerate(stream):
            physical = encode_transport_physical(packet, index)
            process.stdin.write(render_levels(physical_to_matrix(physical), case.low_level, case.high_level).tobytes())
            emitted += 1
            if emitted % 250 == 0 or emitted == total_frames: print(f"\r  encode {emitted}/{total_frames}", end="", flush=True)
        process.stdin.close()
        if process.wait() != 0: raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    except BaseException:
        terminate_ffmpeg(process); raise
    if emitted != total_frames: raise RuntimeError(f"expected {total_frames} frames, emitted {emitted}")
    print()
    return {"command": command, "input_sha256": digest, "blocks": len(layout), "source_symbols": source_symbols,
            "repair_symbols": total_symbols - source_symbols, "meta_frames": meta_frames, "total_frames": total_frames,
            "duration_seconds": total_frames / FPS}


def raw_oracle(video_path: Path, input_path: Path) -> dict[str, Any]:
    metadata, _, digest = _layout(input_path)
    expected_stream = packet_stream(input_path, metadata, file_id_from_sha256(digest), make_raptorq_engine())
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened(): raise RuntimeError(f"cannot open video: {video_path}")
    bit_errors = byte_errors = frames_with_errors = observed = compared_bits = 0
    try:
        for index, packet in enumerate(expected_stream):
            ok, frame = capture.read()
            if not ok: break
            actual, _ = decode_frame(frame, PHYSICAL_CONFIG)
            expected = physical_to_matrix(encode_transport_physical(packet, index))
            difference = expected != actual
            errors = int(difference.sum())
            bit_errors += errors; frames_with_errors += int(errors > 0); compared_bits += expected.size
            byte_errors += sum(a != b for a, b in zip(np.packbits(actual).tobytes(), np.packbits(expected).tobytes()))
            observed += 1
            if observed % 500 == 0: print(f"\r  oracle {observed}", end="", flush=True)
        while capture.read()[0]: observed += 1
    finally: capture.release()
    print()
    return {"observed_frames": observed, "compared_bits": compared_bits, "raw_bit_errors": bit_errors,
            "raw_ber": bit_errors / compared_bits if compared_bits else None, "raw_byte_errors": byte_errors,
            "frames_with_raw_errors": frames_with_errors}


def decode_diagnostics(video: Path, output_dir: Path, input_sha: str, keep_recovered: bool) -> dict[str, Any]:
    recovered = production_decode(video, output_dir)
    report = json.loads(result_path(output_dir).read_text(encoding="utf-8"))
    transport = report.get("transport", {}); packets = report.get("packets", {}); raptorq = report.get("raptorq", {})
    result = {"rs_failed_frames": transport.get("rs_frames_failed", 0),
              "transport_failures": sum(transport.get(key, 0) for key in ("transport_crc_failures", "invalid_header", "invalid_packet_length")),
              "packet_failures": packets.get("packet_crc_failed", 0) + packets.get("invalid_packets", 0),
              "blocks_total": raptorq.get("blocks_total"), "blocks_decoded": raptorq.get("blocks_decoded"),
              "source_symbols_received": raptorq.get("source_symbols_received"), "repair_symbols_received": raptorq.get("repair_symbols_received"),
              "sha_exact": report.get("file", {}).get("exact") is True and sha256_file(recovered) == input_sha}
    if not keep_recovered: recovered.unlink(missing_ok=True)
    return result


def proxy_transcode(source: Path, output: Path, settings: dict[str, Any]) -> list[str]:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-an", "-c:v", "libx264", "-b:v", str(settings["target_bitrate_bps"]),
               "-maxrate", str(settings["maxrate_bps"]), "-bufsize", str(settings["bufsize_bps"]), "-preset", settings["preset"],
               "-pix_fmt", "yuv420p", "-r", str(FPS), "-s", f"{WIDTH}x{HEIGHT}", str(output)]
    subprocess.run(command, check=True)
    return command


def resume_valid(path: Path, config_hash: str, input_sha: str, source: Path, proxy: Path, proxy_required: bool = True) -> bool:
    try: result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError): return False
    return result.get("status") == "COMPLETE" and result.get("case_config_hash") == config_hash and result.get("input_sha256") == input_sha and source.exists() and (not proxy_required or proxy.exists())


def run_case(case: SweepCase, input_path: Path, case_dir: Path, proxy_settings: dict[str, Any], *, keep_proxy: bool, keep_recovered: bool, skip_proxy: bool) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True); source = case_dir / "source.mp4"; proxy = case_dir / "proxy.mp4"
    hashed_config = {"case": case.config(), "proxy": None if skip_proxy else proxy_settings}
    result: dict[str, Any] = {"case_id": case.case_id, "status": "RUNNING", "config": case.config(), "case_config_hash": stable_hash(hashed_config),
                              "input_size_bytes": input_path.stat().st_size, "input_sha256": sha256_file(input_path), "timings": {}, "warnings": [], "exceptions": []}
    started = time.monotonic()
    try:
        then = time.monotonic(); encoded = encode_sweep_case(input_path, source, case); result["timings"]["encode_seconds"] = time.monotonic() - then
        probe, warnings = ffprobe(source); result["warnings"] += warnings
        validate_video_probe(probe, "source")
        source_result = {**encoded, "mp4_path": str(source), "mp4_size_bytes": source.stat().st_size, "expansion_ratio": expansion_ratio(source.stat().st_size, input_path.stat().st_size),
                         "duration_seconds": probe.get("duration_seconds") or encoded["duration_seconds"], "ffprobe": probe}
        source_result["bitrate_bps"] = actual_bitrate(source.stat().st_size, source_result["duration_seconds"])
        then = time.monotonic(); source_result.update(raw_oracle(source, input_path)); source_result.update(decode_diagnostics(source, case_dir / "source-recovered", result["input_sha256"], keep_recovered)); result["timings"]["source_decode_seconds"] = time.monotonic() - then
        result["source"] = source_result
        if skip_proxy:
            result["proxy"] = {}; result["status"] = "COMPLETE"
        else:
            then = time.monotonic(); proxy_command = proxy_transcode(source, proxy, proxy_settings); result["timings"]["proxy_seconds"] = time.monotonic() - then
            probe, warnings = ffprobe(proxy); result["warnings"] += warnings
            validate_video_probe(probe, "proxy")
            proxy_result = {"command": proxy_command, "mp4_path": str(proxy), "mp4_size_bytes": proxy.stat().st_size, "expansion_ratio": expansion_ratio(proxy.stat().st_size, input_path.stat().st_size),
                            "duration_seconds": probe.get("duration_seconds"), "ffprobe": probe}
            proxy_result["bitrate_bps"] = actual_bitrate(proxy.stat().st_size, proxy_result["duration_seconds"] or 0)
            then = time.monotonic(); proxy_result.update(raw_oracle(proxy, input_path)); proxy_result.update(decode_diagnostics(proxy, case_dir / "proxy-recovered", result["input_sha256"], keep_recovered)); result["timings"]["proxy_decode_seconds"] = time.monotonic() - then
            proxy_result["rs_frame_failure_rate"] = proxy_result["rs_failed_frames"] / proxy_result["observed_frames"] if proxy_result["observed_frames"] else None
            proxy_result["symbol_erasure_count"] = 0; proxy_result["unknown_erasure_count"] = proxy_result["rs_failed_frames"]
            result["proxy"] = proxy_result; result["status"] = "COMPLETE"
    except Exception as exc:
        result["status"] = "FAILED"; result["exceptions"].append({"type": type(exc).__name__, "message": str(exc)})
    result["timings"]["total_seconds"] = time.monotonic() - started
    (case_dir / "case_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_cases(selected: Iterable[SweepCase], runner: Callable[[SweepCase], dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for case in selected:
        try: results.append(runner(case))
        except Exception as exc: results.append({"case_id": case.case_id, "status": "FAILED", "exceptions": [{"type": type(exc).__name__, "message": str(exc)}]})
    return results


def reference_settings(reference: Path | None, bitrate: str | int | None, summary_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if reference and reference.exists():
        observed, warnings = ffprobe(reference)
        if not observed.get("calculated_bitrate_bps"): raise RuntimeError("reference duration is unavailable")
        target = round(observed["calculated_bitrate_bps"])
    elif bitrate is not None:
        target = parse_bitrate(bitrate); observed = {"path": None, "manual_bitrate_bps": target}; warnings = []
    else: raise RuntimeError("proxy reference missing; provide --proxy-bitrate")
    settings = {"codec": "libx264", "target_bitrate_bps": target, "maxrate_bps": round(target * 1.25), "bufsize_bps": target * 2, "preset": "medium", "pix_fmt": "yuv420p", "width": WIDTH, "height": HEIGHT, "fps": FPS}
    source_observed = None
    if reference:
        source_candidate = reference.with_name(reference.name.replace("-platform-1080p", "-source"))
        if source_candidate != reference and source_candidate.exists():
            source_observed, source_warnings = ffprobe(source_candidate); warnings += source_warnings
    payload = {"reference": observed, "source_reference": source_observed, "settings": settings, "warnings": warnings}
    summary_dir.mkdir(parents=True, exist_ok=True); (summary_dir / "proxy_reference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return observed, settings


def software_versions() -> dict[str, Any]:
    def version(command: list[str]) -> str | None:
        try: return subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()[0]
        except Exception: return None
    return {"python": platform.python_version(), "ffmpeg": version(["ffmpeg", "-version"]), "opencv": cv2.__version__,
            "pyraptorq": version([sys.executable, "-c", "import importlib.metadata as m; print(m.version('pyraptorq'))"]),
            "reedsolo": version([sys.executable, "-c", "import importlib.metadata as m; print(m.version('reedsolo'))"])}


def run_sweep(root: Path, *, input_path: Path | None, proxy_reference: Path | None, proxy_bitrate: str | int | None, resume: bool, keep_proxy: bool, keep_recovered: bool, skip_proxy: bool, selected_ids: set[str] | None = None, preflight: bool = True) -> list[dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True); summary_dir = root / "summary"
    if input_path is None:
        input_path = root / "input" / "codec-sweep-32MiB.bin"
        input_sha = generate_deterministic_file(input_path)
    else:
        if not input_path.is_file() or input_path.stat().st_size <= 0:
            raise ValueError("--input must name a non-empty file")
        input_sha = sha256_file(input_path)
    reference, settings = reference_settings(proxy_reference, proxy_bitrate, summary_dir)
    selected = [case for case in cases() if not selected_ids or case.case_id in selected_ids]
    started = datetime.now(timezone.utc).isoformat()
    baseline = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    manifest = {"format": FORMAT, "baseline_git_sha": baseline, "input": {"path": str(input_path), "size_bytes": input_path.stat().st_size, "sha256": input_sha},
                "levels": LEVELS, "crfs": CRFS, "preset": PRESET, "repair": REPAIR_RATIO, "symbol_size": SYMBOL_SIZE, "block_size": BLOCK_SIZE,
                "cell_size": PHYSICAL_CONFIG.cell_size, "fps": FPS, "resolution": f"{WIDTH}x{HEIGHT}", "interleave_window": INTERLEAVE_WINDOW,
                "proxy": {"reference": reference, "settings": settings}, "start_time": started, "finish_time": None, "software": software_versions()}
    summary_dir.mkdir(parents=True, exist_ok=True); manifest_path = summary_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if preflight and not skip_proxy:
        preflight_input = root / "preflight" / "preflight-1MiB.bin"
        generate_deterministic_file(preflight_input, 1024 * 1024, b"FVM_CODEC_SWEEP_PREFLIGHT_V1")
        pre = run_case(SweepCase(64, 192, 24), preflight_input, root / "preflight" / "case", settings, keep_proxy=False, keep_recovered=False, skip_proxy=False)
        if pre["status"] != "COMPLETE" or not pre["source"]["sha_exact"] or not pre["proxy"]["sha_exact"]: raise RuntimeError(f"preflight failed: {pre.get('exceptions')}")
        (root / "preflight" / "case" / "proxy.mp4").unlink(missing_ok=True)
    results = []
    overall = time.monotonic()
    for index, case in enumerate(selected, 1):
        case_dir = root / "cases" / case.case_id; result_file = case_dir / "case_result.json"; source = case_dir / "source.mp4"; proxy = case_dir / "proxy.mp4"
        config_hash = stable_hash({"case": case.config(), "proxy": None if skip_proxy else settings})
        if resume and resume_valid(result_file, config_hash, input_sha, source, proxy, proxy_required=not skip_proxy):
            print(f"[{index}/{len(selected)}] {case.case_id}: resume skip"); results.append(json.loads(result_file.read_text(encoding="utf-8"))); continue
        print(f"[{index}/{len(selected)}] {case.case_id}; overall {time.monotonic()-overall:.1f}s")
        results.append(run_case(case, input_path, case_dir, settings, keep_proxy=keep_proxy, keep_recovered=keep_recovered, skip_proxy=skip_proxy))
    rows = summarize(results, summary_dir)
    manifest["finish_time"] = datetime.now(timezone.utc).isoformat(); manifest["completed_cases"] = sum(row["status"] == "COMPLETE" for row in rows); manifest["failed_cases"] = sum(row["status"] != "COMPLETE" for row in rows)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8"); write_report(rows, manifest, summary_dir / "REPORT.md")
    if not keep_proxy and all(row["status"] == "COMPLETE" for row in rows):
        for case in selected: (root / "cases" / case.case_id / "proxy.mp4").unlink(missing_ok=True)
    return results
