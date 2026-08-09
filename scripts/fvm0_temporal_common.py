"""Deterministic FVM0 temporal-transition probe primitives."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterator, Any
import numpy as np

FORMAT = "FVM0_TEMPORAL_PROBE"
PRNG = "PCG64_TEMPORAL_V1"

def flip_count(cells: int, ratio: float) -> int:
    if not 0 < ratio <= .5: raise ValueError("ratio must be in (0, 0.5]")
    return math.floor(cells * ratio + .5)

@dataclass(frozen=True)
class TemporalConfig:
    width: int = 1920; height: int = 1080; fps: int = 30; cell_size: int = 5
    block_size: int = 150; warmup_blocks: int = 1; repeats: int = 5; seed: int = 20260809
    ratios: tuple[float, ...] = (.1, .2, .3, .4, .5)
    def __post_init__(self):
        if self.width % self.cell_size or self.height % self.cell_size: raise ValueError("geometry must divide cell_size")
        if self.block_size < 2 or self.warmup_blocks < 0 or self.repeats < 1: raise ValueError("invalid block parameters")
        for ratio in self.ratios: flip_count(self.cells_per_frame, ratio)
    @property
    def rows(self): return self.height // self.cell_size
    @property
    def cols(self): return self.width // self.cell_size
    @property
    def cells_per_frame(self): return self.rows * self.cols
    def schedule(self) -> list[dict[str, Any]]:
        result = [{"block_index": i, "warmup": True, "ratio": .5, "flip_count": flip_count(self.cells_per_frame,.5)} for i in range(self.warmup_blocks)]
        rng = np.random.Generator(np.random.PCG64(self.seed))
        for repeat in range(self.repeats):
            for ratio in rng.permutation(self.ratios):
                ratio=float(ratio); result.append({"block_index": len(result), "warmup": False, "repeat": repeat, "ratio": ratio, "flip_count": flip_count(self.cells_per_frame,ratio)})
        return result
    def manifest(self):
        return {"format":FORMAT,"prng":PRNG,"width":self.width,"height":self.height,"fps":self.fps,"cell_size":self.cell_size,"rows":self.rows,"cols":self.cols,"cells_per_frame":self.cells_per_frame,"block_size":self.block_size,"warmup_blocks":self.warmup_blocks,"repeats":self.repeats,"ratios":list(self.ratios),"seed":self.seed,"schedule":self.schedule()}
    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "TemporalConfig":
        required = {"format", "prng", "width", "height", "fps", "cell_size", "rows", "cols", "cells_per_frame", "block_size", "warmup_blocks", "repeats", "ratios", "seed", "schedule"}
        if not isinstance(manifest, dict) or not required <= manifest.keys() or manifest["format"] != FORMAT or manifest["prng"] != PRNG:
            raise ValueError("unsupported or incomplete FVM0-T manifest")
        config = cls(**{key: manifest[key] for key in ("width", "height", "fps", "cell_size", "block_size", "warmup_blocks", "repeats", "seed")}, ratios=tuple(manifest["ratios"]))
        if (manifest["rows"], manifest["cols"], manifest["cells_per_frame"]) != (config.rows, config.cols, config.cells_per_frame):
            raise ValueError("manifest geometry does not match parameters")
        schedule = manifest["schedule"]
        if not isinstance(schedule, list) or len(schedule) != config.warmup_blocks + config.repeats * len(config.ratios):
            raise ValueError("manifest schedule has invalid length")
        for index, item in enumerate(schedule):
            if item.get("block_index") != index or item.get("ratio") not in config.ratios or item.get("flip_count") != flip_count(config.cells_per_frame, item["ratio"]):
                raise ValueError("manifest schedule entry is invalid")
            if index < config.warmup_blocks and (not item.get("warmup") or item["ratio"] != .5): raise ValueError("invalid warmup schedule")
        for repeat in range(config.repeats):
            chunk = schedule[config.warmup_blocks + repeat * len(config.ratios):config.warmup_blocks + (repeat + 1) * len(config.ratios)]
            if sorted(item.get("ratio") for item in chunk) != sorted(config.ratios): raise ValueError("schedule repeat is not balanced")
        return config

def temporal_frames(config: TemporalConfig, schedule: list[dict[str,Any]] | None=None) -> Iterator[tuple[dict[str,Any],np.ndarray,np.ndarray|None]]:
    schedule=schedule or config.schedule(); absolute=np.random.Generator(np.random.PCG64(config.seed+1)); masks=np.random.Generator(np.random.PCG64(config.seed+2))
    for block in schedule:
        previous=None
        for phase in range(config.block_size):
            anchor=phase==0
            if anchor:
                bits=absolute.integers(0,2,(config.rows,config.cols),dtype=np.uint8); mask=None
            else:
                chosen=masks.choice(config.cells_per_frame, block["flip_count"], replace=False)
                mask=np.zeros(config.cells_per_frame,dtype=bool); mask[chosen]=True; mask=mask.reshape(config.rows,config.cols)
                bits=previous.copy(); bits[mask]^=1
            record={"frame_index":block["block_index"]*config.block_size+phase,"block_index":block["block_index"],"phase":phase,"is_anchor":anchor,"is_warmup":block["warmup"],"repeat_index":block.get("repeat"),"transition_ratio":None if anchor else block["ratio"],"flip_count":0 if anchor else block["flip_count"]}
            yield record,bits,mask; previous=bits

def theoretical_bits(cells:int, flips:int)->float: return (math.lgamma(cells+1)-math.lgamma(flips+1)-math.lgamma(cells-flips+1))/math.log(2)

def aggregate_ratios(records:list[dict[str,Any]], cells:int)->dict[str,Any]:
    groups={}
    for r in records:
        if r["is_warmup"] or r["is_anchor"]: continue
        groups.setdefault(str(r["transition_ratio"]),[]).append(r)
    out={}
    for ratio in sorted(groups, key=float):
        rows = groups[ratio]
        sums=lambda k:sum(int(x[k]) for x in rows); errors=sums("bit_errors"); changed=sums("changed_bit_errors"); flip=int(rows[0]["flip_count"]); n=len(rows)
        values=sorted(int(x["bit_errors"]) for x in rows); unchanged=sums("unchanged_bit_errors"); p90=values[math.ceil(.9*n)-1]
        bits=theoretical_bits(cells,flip); raw=bits*30
        out[ratio]={"ratio":float(ratio),"flip_count":flip,"blocks":len(set(x["block_index"] for x in rows)),"transition_frames":n,"total_bits":n*cells,"bit_errors":errors,"correct_bits":n*cells-errors,"ber":errors/(n*cells),"changed_bits":n*flip,"changed_bit_errors":changed,"changed_ber":changed/(n*flip),"unchanged_bits":n*(cells-flip),"unchanged_bit_errors":unchanged,"unchanged_ber":unchanged/(n*(cells-flip)),"zero_to_one":sums("zero_to_one"),"one_to_zero":sums("one_to_zero"),"frames_with_errors":sum(v>0 for v in values),"fer":sum(v>0 for v in values)/n,"mean_bit_errors_per_frame":sum(values)/n,"median_bit_errors_per_frame":float(np.median(values)),"p90_bit_errors_per_frame":p90,"min_bit_errors_per_frame":values[0],"max_bit_errors_per_frame":values[-1],"mean_ber_per_frame":sum(values)/n/cells,"median_ber_per_frame":float(np.median(values))/cells,"theoretical_bits_per_transition_frame":bits,"theoretical_raw_bps":raw,"theoretical_raw_bytes_per_second":raw/8,"theoretical_effective_bps_including_anchor":raw*(149/150),"theoretical_effective_bytes_per_second_including_anchor":raw*(149/150)/8}
    return out
