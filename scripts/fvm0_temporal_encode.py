"""Encode an FVM0-T temporal-transition probe."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
try:
    from fvm0_common import render_bits
    from fvm0_encode import terminate_ffmpeg
    from fvm0_temporal_common import TemporalConfig, temporal_frames
except ImportError:
    from scripts.fvm0_common import render_bits
    from scripts.fvm0_encode import terminate_ffmpeg
    from scripts.fvm0_temporal_common import TemporalConfig, temporal_frames


def encode(output: Path, config: TemporalConfig, crf: int, preset: str) -> Path:
    if not 0 <= crf <= 51: raise ValueError("crf must be between 0 and 51")
    output.parent.mkdir(parents=True,exist_ok=True)
    command=["ffmpeg","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{config.width}x{config.height}","-r",str(config.fps),"-i","-","-an","-c:v","libx264","-crf",str(crf),"-preset",preset,"-pix_fmt","yuv420p",str(output)]
    try: process=subprocess.Popen(command,stdin=subprocess.PIPE)
    except FileNotFoundError as exc: raise RuntimeError("ffmpeg not found on PATH") from exc
    try:
        assert process.stdin is not None
        total=len(config.schedule())*config.block_size
        for index,(_,bits,_) in enumerate(temporal_frames(config),1):
            process.stdin.write(render_bits(bits,config).tobytes())
            if index%100==0 or index==total: print(f"\rFVM0-T encoded {index}/{total} frames",end="",flush=True)
        process.stdin.close()
        if process.wait()!=0: raise RuntimeError(f"ffmpeg exited with code {process.returncode}")
    except BaseException:
        terminate_ffmpeg(process); raise
    manifest=config.manifest();manifest.update({"source_video":output.name,"crf":crf,"preset":preset});path=output.with_suffix(".manifest.json");path.write_text(json.dumps(manifest,indent=2),encoding="utf-8");return path


def main():
    p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--width",type=int,default=1920);p.add_argument("--height",type=int,default=1080);p.add_argument("--fps",type=int,default=30);p.add_argument("--cell-size",type=int,default=5);p.add_argument("--block-size",type=int,default=150);p.add_argument("--ratios",default=".1,.2,.3,.4,.5");p.add_argument("--repeats",type=int,default=5);p.add_argument("--warmup-blocks",type=int,default=1);p.add_argument("--seed",type=int,default=20260809);p.add_argument("--crf",type=int,default=15);p.add_argument("--preset",default="medium");a=p.parse_args();config=TemporalConfig(a.width,a.height,a.fps,a.cell_size,a.block_size,a.warmup_blocks,a.repeats,a.seed,tuple(map(float,a.ratios.split(','))));print(encode(a.output,config,a.crf,a.preset))
if __name__=="__main__":main()
