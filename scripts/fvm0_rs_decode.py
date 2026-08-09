"""Decode and measure an FVM0_RS_PROBE video."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from fvm0_common import LumaStats, decode_frame
    from fvm0_rs_common import (CODED_RS_BYTES, RS_CODEWORDS, FVM0RSConfig, decode_codewords,
                                deinterleave, encode_logical, interleave, logical_frame,
                                matrix_to_bytes, physical_frame)
except ImportError:
    from scripts.fvm0_common import LumaStats, decode_frame
    from scripts.fvm0_rs_common import (CODED_RS_BYTES, RS_CODEWORDS, FVM0RSConfig, decode_codewords,
                                        deinterleave, encode_logical, interleave, logical_frame,
                                        matrix_to_bytes, physical_frame)


def decode(video: Path, manifest_path: Path, output_dir: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = FVM0RSConfig.from_manifest(manifest)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError(f"cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if (width, height) != (config.width, config.height):
        capture.release(); raise RuntimeError("video resolution does not match FVM0 RS manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []; spatial = np.zeros((config.rows, config.cols), dtype=np.uint64)
    zero_luma = LumaStats(); one_luma = LumaStats(); codeword_hist = np.zeros(10, dtype=np.int64)
    totals = {key: 0 for key in ("raw_bits", "raw_bit_errors", "raw_byte_errors", "zero_to_one", "one_to_zero",
                                  "reserved_bits", "reserved_bit_errors", "reserved_byte_errors", "codewords",
                                  "codewords_with_raw_errors", "decode_successes", "decode_failures",
                                  "decoder_reported_corrections", "crc_failures", "miscorrections", "exact_frames")}
    for frame_index in range(config.frames):
        ok, frame = capture.read()
        if not ok: break
        actual_bits, luma = decode_frame(frame, config)
        expected_bytes = physical_frame(config, frame_index)
        expected_bits = np.unpackbits(np.frombuffer(expected_bytes, dtype=np.uint8), bitorder="big").reshape(config.rows, config.cols)
        errors = actual_bits != expected_bits; spatial += errors
        zero_luma.add(luma[expected_bits == 0]); one_luma.add(luma[expected_bits == 1])
        received_bytes = matrix_to_bytes(actual_bits, config)
        raw_byte_errors = np.frombuffer(received_bytes, dtype=np.uint8) != np.frombuffer(expected_bytes, dtype=np.uint8)
        received_codewords = deinterleave(received_bytes[:CODED_RS_BYTES])
        expected_codewords = encode_logical(logical_frame(config.seed, frame_index))
        codeword_errors = (received_codewords != expected_codewords).sum(axis=1)
        for count in codeword_errors: codeword_hist[min(int(count), 9)] += 1
        recovery = decode_codewords(received_codewords, logical_frame(config.seed, frame_index), frame_index)
        reserved_expected = np.unpackbits(np.frombuffer(expected_bytes[CODED_RS_BYTES:], dtype=np.uint8), bitorder="big")
        reserved_actual = np.unpackbits(np.frombuffer(received_bytes[CODED_RS_BYTES:], dtype=np.uint8), bitorder="big")
        reserved_errors = reserved_expected != reserved_actual
        bit_errors = int(errors.sum()); zero_to_one = int(((expected_bits == 0) & (actual_bits == 1)).sum())
        one_to_zero = int(((expected_bits == 1) & (actual_bits == 0)).sum())
        record = {"frame_index": frame_index, "raw_bit_errors": bit_errors,
                  "raw_ber": bit_errors / config.cells_per_frame, "raw_byte_errors": int(raw_byte_errors.sum()),
                  "zero_to_one": zero_to_one, "one_to_zero": one_to_zero,
                  "reserved_bit_errors": int(reserved_errors.sum()),
                  "reserved_byte_errors": int(raw_byte_errors[CODED_RS_BYTES:].sum()),
                  "codewords_with_raw_errors": int((codeword_errors > 0).sum()),
                  "max_codeword_raw_byte_errors": int(codeword_errors.max()),
                  "mean_codeword_raw_byte_errors": float(codeword_errors.mean()),
                  "rs_codewords_success": recovery.codeword_successes,
                  "rs_codewords_failed": recovery.codeword_failures,
                  "decoder_reported_corrected_symbols": recovery.decoder_reported_corrections,
                  "rs_all_codewords_decoded": recovery.codeword_failures == 0,
                  "header_valid": recovery.header_valid, "crc_valid": recovery.crc_valid,
                  "payload_exact": recovery.payload_exact,
                  "embedded_frame_index": recovery.embedded_frame_index,
                  "frame_index_mismatch": recovery.embedded_frame_index is not None and recovery.embedded_frame_index != frame_index}
        records.append(record)
        totals["raw_bits"] += config.cells_per_frame; totals["raw_bit_errors"] += bit_errors
        totals["raw_byte_errors"] += int(raw_byte_errors.sum()); totals["zero_to_one"] += zero_to_one; totals["one_to_zero"] += one_to_zero
        totals["reserved_bits"] += config.reserved_bytes_per_frame * 8; totals["reserved_bit_errors"] += int(reserved_errors.sum())
        totals["reserved_byte_errors"] += int(raw_byte_errors[CODED_RS_BYTES:].sum()); totals["codewords"] += RS_CODEWORDS
        totals["codewords_with_raw_errors"] += int((codeword_errors > 0).sum()); totals["decode_successes"] += recovery.codeword_successes
        totals["decode_failures"] += recovery.codeword_failures; totals["decoder_reported_corrections"] += recovery.decoder_reported_corrections or 0
        totals["crc_failures"] += int(recovery.logical is not None and not recovery.crc_valid)
        totals["miscorrections"] += int(recovery.logical is not None and not recovery.payload_exact)
        totals["exact_frames"] += int(recovery.payload_exact)
        if (frame_index + 1) % 100 == 0: print(f"\rFVM0-RS compared {frame_index + 1}/{config.frames} frames", end="", flush=True)
    actual_frames = len(records)
    while capture.read()[0]: actual_frames += 1
    capture.release()
    if not records: raise RuntimeError("no video frames available for FVM0 RS comparison")
    with (output_dir / "fvm0_rs_frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    compared = len(records); warnings = []
    if actual_frames != config.frames: warnings.append("FVM0_RS_PROBE currently uses index-based expected-frame comparison; frame insertion/deletion can invalidate later measurements.")
    if abs(actual_fps - config.fps) > .01: warnings.append("decoded FPS differs from manifest")
    if any(record["frame_index_mismatch"] for record in records): warnings.append("recovered embedded frame_index differs from index-based expected frame")
    result = {"format": manifest["format"], "video": {"expected_resolution": f"{config.width}x{config.height}",
              "actual_resolution": f"{width}x{height}", "expected_fps": config.fps, "actual_fps": actual_fps,
              "expected_frames": config.frames, "actual_frames": actual_frames, "compared_frames": compared},
              "matrix": {"cell_size": config.cell_size, "rows": config.rows, "cols": config.cols,
              "cells": config.cells_per_frame, "physical_bytes_per_frame": config.physical_bytes_per_frame,
              "raw_bytes_per_second": config.physical_bytes_per_frame * config.fps},
              "fec": {"rs_n": 255, "rs_k": 239, "rs_parity": 16, "codewords_per_frame": 28,
              "logical_bytes_per_frame": 6692, "coded_bytes_per_frame": 7140,
              "reserved_bytes_per_frame": config.reserved_bytes_per_frame, "inner_code_efficiency": 239/255,
              "frame_logical_efficiency": 6692/config.physical_bytes_per_frame,
              "logical_probe_bytes_per_second": 6692*config.fps},
              "raw": {"total_bits": totals["raw_bits"], "bit_errors": totals["raw_bit_errors"],
              "ber": totals["raw_bit_errors"]/totals["raw_bits"], "byte_errors": totals["raw_byte_errors"],
              "zero_to_one": totals["zero_to_one"], "one_to_zero": totals["one_to_zero"],
              "reserved_bits": totals["reserved_bits"], "reserved_bit_errors": totals["reserved_bit_errors"],
              "reserved_ber": totals["reserved_bit_errors"]/totals["reserved_bits"] if totals["reserved_bits"] else None,
              "reserved_byte_errors": totals["reserved_byte_errors"]},
              "rs": {"total_codewords": totals["codewords"], "codewords_with_raw_errors": totals["codewords_with_raw_errors"],
              "decode_successes": totals["decode_successes"], "decode_failures": totals["decode_failures"],
              "decoder_reported_corrected_symbols": totals["decoder_reported_corrections"],
              "crc_failures": totals["crc_failures"], "miscorrection_or_residual_count": totals["miscorrections"],
              "max_raw_byte_errors_in_codeword": max(record["max_codeword_raw_byte_errors"] for record in records),
              "codeword_raw_error_histogram": {**{str(i): int(codeword_hist[i]) for i in range(9)}, "9+": int(codeword_hist[9])}},
              "frames": {"exact": totals["exact_frames"], "failed": compared-totals["exact_frames"],
              "exact_recovery_rate": totals["exact_frames"]/compared, "failure_rate": (compared-totals["exact_frames"])/compared},
              "luminance": {"expected_zero": zero_luma.summary(), "expected_one": one_luma.summary()}, "warnings": warnings}
    (output_dir / "fvm0_rs_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plots(output_dir, spatial, codeword_hist, records, zero_luma, one_luma)
    print(f"\nRaw BER: {result['raw']['ber']:.8g}; exact frames: {totals['exact_frames']}/{compared}; RS failures: {totals['decode_failures']}")
    return result


def _plots(output: Path, spatial: np.ndarray, histogram: np.ndarray, records: list[dict],
           zero_luma: LumaStats, one_luma: LumaStats) -> None:
    fig, ax = plt.subplots(figsize=(12, 6)); image = ax.imshow(spatial, aspect="auto", cmap="magma")
    ax.set_title("FVM0-RS raw cell error count"); fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output / "fvm0_rs_error_heatmap.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.bar([str(i) for i in range(9)] + ["9+"], histogram); ax.set(xlabel="raw byte errors / RS codeword", ylabel="codeword count"); fig.tight_layout(); fig.savefig(output / "fvm0_rs_codeword_byte_errors.png"); plt.close(fig)
    fig, ax = plt.subplots(); ax.plot([r["frame_index"] for r in records], [r["raw_bit_errors"] for r in records]); failed=[r for r in records if not r["payload_exact"]]; ax.scatter([r["frame_index"] for r in failed], [r["raw_bit_errors"] for r in failed], color="red", label="RS frame failure"); ax.set(xlabel="frame index", ylabel="raw bit errors"); fig.tight_layout(); fig.savefig(output / "fvm0_rs_frame_raw_errors.png"); plt.close(fig)
    fig, ax = plt.subplots(); values=np.arange(256); ax.step(values,zero_luma.histogram,where="mid",label="expected 0"); ax.step(values,one_luma.histogram,where="mid",label="expected 1"); ax.set(xlabel="mean cell luminance",ylabel="cell count"); ax.legend(); fig.tight_layout(); fig.savefig(output / "fvm0_rs_luma_histogram.png"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode an FVM0 6px RS probe")
    parser.add_argument("video", type=Path); parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    decode(args.video, args.manifest, args.output_dir)


if __name__ == "__main__": main()
