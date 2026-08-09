"""Deterministic generation and statistics for the FVM0-T probe."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterator

import numpy as np

FORMAT = "FVM0_TEMPORAL_PROBE"
PRNG = "PCG64_TEMPORAL_V1"


def flip_count(cells: int, ratio: float) -> int:
    """Return floor(cells * ratio + 0.5), the protocol round-half-up rule."""
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not math.isfinite(ratio):
        raise ValueError("ratio must be finite")
    if not 0 < ratio <= 0.5:
        raise ValueError("ratio must be in (0, 0.5]")
    return math.floor(cells * ratio + 0.5)


@dataclass(frozen=True)
class TemporalConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    cell_size: int = 5
    block_size: int = 150
    warmup_blocks: int = 1
    repeats: int = 5
    seed: int = 20260809
    ratios: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)

    def __post_init__(self) -> None:
        positive = (self.width, self.height, self.fps, self.cell_size, self.seed)
        nonnegative = (self.warmup_blocks,)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in positive):
            raise ValueError("width, height, fps, cell_size, and seed must be positive integers")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in nonnegative):
            raise ValueError("warmup_blocks must be a nonnegative integer")
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size < 2:
            raise ValueError("block_size must be an integer >= 2")
        if not isinstance(self.repeats, int) or isinstance(self.repeats, bool) or self.repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if self.width % self.cell_size or self.height % self.cell_size:
            raise ValueError("width and height must be divisible by cell_size")
        if not isinstance(self.ratios, (tuple, list)) or not self.ratios:
            raise ValueError("ratios must be non-empty")
        if len(set(self.ratios)) != len(self.ratios):
            raise ValueError("ratios must be unique")
        for ratio in self.ratios:
            flips = flip_count(self.cells_per_frame, ratio)
            if flips <= 0 or flips >= self.cells_per_frame:
                raise ValueError("ratio is not meaningful for this geometry")

    @property
    def rows(self) -> int: return self.height // self.cell_size
    @property
    def cols(self) -> int: return self.width // self.cell_size
    @property
    def cells_per_frame(self) -> int: return self.rows * self.cols

    def schedule(self) -> list[dict[str, Any]]:
        schedule = []
        for _ in range(self.warmup_blocks):
            schedule.append({"block_index": len(schedule), "warmup": True, "ratio": 0.5,
                             "flip_count": flip_count(self.cells_per_frame, 0.5)})
        rng = np.random.Generator(np.random.PCG64(self.seed))
        for repeat in range(self.repeats):
            for value in rng.permutation(self.ratios):
                ratio = float(value)
                schedule.append({"block_index": len(schedule), "warmup": False, "repeat": repeat,
                                 "ratio": ratio, "flip_count": flip_count(self.cells_per_frame, ratio)})
        return schedule

    def manifest(self) -> dict[str, Any]:
        return {"format": FORMAT, "prng": PRNG, "generator_implementation": "numpy",
                "numpy_version": np.__version__, "width": self.width, "height": self.height,
                "fps": self.fps, "cell_size": self.cell_size, "rows": self.rows, "cols": self.cols,
                "cells_per_frame": self.cells_per_frame, "block_size": self.block_size,
                "warmup_blocks": self.warmup_blocks, "repeats": self.repeats,
                "ratios": list(self.ratios), "seed": self.seed, "schedule": self.schedule()}

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "TemporalConfig":
        try:
            if manifest["format"] != FORMAT or manifest["prng"] != PRNG:
                raise ValueError("unsupported FVM0-T format or PRNG")
            config = cls(width=manifest["width"], height=manifest["height"], fps=manifest["fps"],
                         cell_size=manifest["cell_size"], block_size=manifest["block_size"],
                         warmup_blocks=manifest["warmup_blocks"], repeats=manifest["repeats"],
                         seed=manifest["seed"], ratios=tuple(manifest["ratios"]))
            if (manifest["rows"], manifest["cols"], manifest["cells_per_frame"]) != (config.rows, config.cols, config.cells_per_frame):
                raise ValueError("manifest geometry metadata is inconsistent")
            schedule = manifest["schedule"]
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete FVM0-T manifest") from exc
        expected_length = config.warmup_blocks + config.repeats * len(config.ratios)
        if not isinstance(schedule, list) or len(schedule) != expected_length:
            raise ValueError("manifest schedule has invalid length")
        for index, item in enumerate(schedule):
            is_warmup = index < config.warmup_blocks
            expected_repeat = None if is_warmup else (index - config.warmup_blocks) // len(config.ratios)
            allowed = item.get("ratio") == 0.5 if is_warmup else item.get("ratio") in config.ratios
            if (item.get("block_index") != index or item.get("warmup") is not is_warmup or not allowed or
                    item.get("repeat") != expected_repeat or
                    item.get("flip_count") != flip_count(config.cells_per_frame, item.get("ratio"))):
                raise ValueError("manifest schedule entry is invalid")
        for repeat in range(config.repeats):
            start = config.warmup_blocks + repeat * len(config.ratios)
            if sorted(item["ratio"] for item in schedule[start:start + len(config.ratios)]) != sorted(config.ratios):
                raise ValueError("manifest schedule repeat is not balanced")
        return config


def temporal_frames(config: TemporalConfig, schedule: list[dict[str, Any]] | None = None) -> Iterator[tuple[dict[str, Any], np.ndarray, np.ndarray | None]]:
    absolute_rng = np.random.Generator(np.random.PCG64(config.seed + 1))
    mask_rng = np.random.Generator(np.random.PCG64(config.seed + 2))
    for block in schedule or config.schedule():
        previous = None
        for phase in range(config.block_size):
            anchor = phase == 0
            mask = None
            if anchor:
                bits = absolute_rng.integers(0, 2, (config.rows, config.cols), dtype=np.uint8)
            else:
                chosen = mask_rng.choice(config.cells_per_frame, block["flip_count"], replace=False)
                mask = np.zeros(config.cells_per_frame, dtype=bool)
                mask[chosen] = True
                mask = mask.reshape(config.rows, config.cols)
                bits = previous.copy()
                bits[mask] ^= 1
            record = {"frame_index": block["block_index"] * config.block_size + phase,
                      "block_index": block["block_index"], "phase": phase, "is_anchor": anchor,
                      "is_warmup": block["warmup"], "repeat_index": block.get("repeat"),
                      "transition_ratio": None if anchor else block["ratio"],
                      "flip_count": 0 if anchor else block["flip_count"]}
            yield record, bits, mask
            previous = bits


def theoretical_bits(cells: int, flips: int) -> float:
    return (math.lgamma(cells + 1) - math.lgamma(flips + 1) - math.lgamma(cells - flips + 1)) / math.log(2)


TRANSITION_FIELDS = (
    "expected_transition_cells", "observed_transition_cells", "transition_true_positive",
    "transition_missed", "transition_false_positive", "transition_true_negative",
    "transition_mask_errors", "transition_mask_ber", "transition_recall",
    "transition_precision", "transition_f1", "missed_flip_rate", "false_flip_rate",
    "expected_zero_to_one_transitions", "correct_zero_to_one_transitions",
    "opposite_zero_to_one_transitions", "missed_zero_to_one_transitions",
    "expected_one_to_zero_transitions", "correct_one_to_zero_transitions",
    "opposite_one_to_zero_transitions", "missed_one_to_zero_transitions",
    "zero_to_one_direction_recall", "one_to_zero_direction_recall",
    "transition_direction_accuracy",
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def measure_transition(expected_previous: np.ndarray, expected_current: np.ndarray,
                       actual_previous: np.ndarray, actual_current: np.ndarray) -> dict[str, Any]:
    """Measure transition-mask and direction recovery between adjacent frames."""
    shapes = {array.shape for array in (expected_previous, expected_current, actual_previous, actual_current)}
    if len(shapes) != 1:
        raise ValueError("transition arrays must have identical shapes")
    expected = expected_previous != expected_current
    observed = actual_previous != actual_current
    tp = int((expected & observed).sum()); fn = int((expected & ~observed).sum())
    fp = int((~expected & observed).sum()); tn = int((~expected & ~observed).sum())
    expected_01 = expected & (expected_previous == 0); expected_10 = expected & (expected_previous == 1)
    actual_01 = observed & (actual_previous == 0); actual_10 = observed & (actual_previous == 1)
    e01=int(expected_01.sum()); e10=int(expected_10.sum())
    c01=int((expected_01 & actual_01).sum()); o01=int((expected_01 & actual_10).sum())
    c10=int((expected_10 & actual_10).sum()); o10=int((expected_10 & actual_01).sum())
    precision=_ratio(tp,tp+fp); recall=_ratio(tp,tp+fn)
    f1=None if precision is None or recall is None or precision+recall==0 else 2*precision*recall/(precision+recall)
    return {"expected_transition_cells":tp+fn,"observed_transition_cells":tp+fp,
            "transition_true_positive":tp,"transition_missed":fn,"transition_false_positive":fp,
            "transition_true_negative":tn,"transition_mask_errors":fn+fp,
            "transition_mask_ber":(fn+fp)/expected.size,"transition_recall":recall,
            "transition_precision":precision,"transition_f1":f1,"missed_flip_rate":_ratio(fn,tp+fn),
            "false_flip_rate":_ratio(fp,fp+tn),"expected_zero_to_one_transitions":e01,
            "correct_zero_to_one_transitions":c01,"opposite_zero_to_one_transitions":o01,
            "missed_zero_to_one_transitions":e01-c01-o01,"expected_one_to_zero_transitions":e10,
            "correct_one_to_zero_transitions":c10,"opposite_one_to_zero_transitions":o10,
            "missed_one_to_zero_transitions":e10-c10-o10,
            "zero_to_one_direction_recall":_ratio(c01,e01),"one_to_zero_direction_recall":_ratio(c10,e10),
            "transition_direction_accuracy":_ratio(c01+c10,e01+e10)}


def aggregate_ratios(records: list[dict[str, Any]], cells: int, fps: int, block_size: int) -> dict[str, Any]:
    """Aggregate non-warmup transition records using weighted bit denominators."""
    groups: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        if not record["is_warmup"] and not record["is_anchor"]:
            groups.setdefault(float(record["transition_ratio"]), []).append(record)
    output = {}
    for ratio in sorted(groups):
        rows = groups[ratio]
        values = sorted(int(row["bit_errors"]) for row in rows)
        count = len(rows); flips = int(rows[0]["flip_count"])
        total_errors = sum(values); changed_errors = sum(int(r["changed_bit_errors"]) for r in rows)
        unchanged_errors = sum(int(r["unchanged_bit_errors"]) for r in rows)
        capacity = theoretical_bits(cells, flips); raw_bps = capacity * fps
        output[str(ratio)] = {"ratio": ratio, "flip_count": flips,
            "blocks": len({r["block_index"] for r in rows}), "transition_frames": count,
            "total_bits": count*cells, "bit_errors": total_errors, "correct_bits": count*cells-total_errors,
            "ber": total_errors/(count*cells), "changed_bits": count*flips,
            "changed_bit_errors": changed_errors, "changed_ber": changed_errors/(count*flips),
            "unchanged_bits": count*(cells-flips), "unchanged_bit_errors": unchanged_errors,
            "unchanged_ber": unchanged_errors/(count*(cells-flips)),
            "zero_to_one": sum(int(r["zero_to_one"]) for r in rows),
            "one_to_zero": sum(int(r["one_to_zero"]) for r in rows),
            "frames_with_errors": sum(v > 0 for v in values), "fer": sum(v > 0 for v in values)/count,
            "mean_bit_errors_per_frame": sum(values)/count, "median_bit_errors_per_frame": float(median(values)),
            "p90_bit_errors_per_frame": values[math.ceil(.9*count)-1], "min_bit_errors_per_frame": values[0],
            "max_bit_errors_per_frame": values[-1], "mean_ber_per_frame": sum(values)/count/cells,
            "median_ber_per_frame": float(median(values))/cells,
            "theoretical_bits_per_transition_frame": capacity, "theoretical_raw_bps": raw_bps,
            "theoretical_raw_bytes_per_second": raw_bps/8,
            "theoretical_effective_bps_including_anchor": raw_bps*(block_size-1)/block_size,
            "theoretical_effective_bytes_per_second_including_anchor": raw_bps*(block_size-1)/block_size/8}
        result=output[str(ratio)]
        for field in ("expected_transition_cells","observed_transition_cells","transition_true_positive",
                      "transition_missed","transition_false_positive","transition_true_negative",
                      "transition_mask_errors","expected_zero_to_one_transitions",
                      "correct_zero_to_one_transitions","opposite_zero_to_one_transitions",
                      "missed_zero_to_one_transitions","expected_one_to_zero_transitions",
                      "correct_one_to_zero_transitions","opposite_one_to_zero_transitions",
                      "missed_one_to_zero_transitions"):
            result[field]=sum(int(row[field]) for row in rows)
        tp=result["transition_true_positive"]; fn=result["transition_missed"]; fp=result["transition_false_positive"]; tn=result["transition_true_negative"]
        result.update(transition_total_cells=count*cells,transition_mask_ber=(fn+fp)/(count*cells),
                      transition_recall=_ratio(tp,tp+fn),transition_precision=_ratio(tp,tp+fp),
                      missed_flip_rate=_ratio(fn,tp+fn),false_flip_rate=_ratio(fp,fp+tn),
                      transition_count_bias=(result["observed_transition_cells"]-result["expected_transition_cells"])/result["expected_transition_cells"],
                      mean_observed_transition_cells_per_frame=result["observed_transition_cells"]/count)
        precision=result["transition_precision"]; recall=result["transition_recall"]
        result["transition_f1"]=None if precision is None or recall is None or precision+recall==0 else 2*precision*recall/(precision+recall)
        result["zero_to_one_direction_recall"]=_ratio(result["correct_zero_to_one_transitions"],result["expected_zero_to_one_transitions"])
        result["one_to_zero_direction_recall"]=_ratio(result["correct_one_to_zero_transitions"],result["expected_one_to_zero_transitions"])
        result["transition_direction_accuracy"]=_ratio(result["correct_zero_to_one_transitions"]+result["correct_one_to_zero_transitions"],result["expected_transition_cells"])
    return output
