from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
from scripts.fvm0_common import render_bits
from scripts.fvm0_temporal_common import TemporalConfig,temporal_frames
def main():
 p=argparse.ArgumentParser();p.add_argument("output",type=Path);p.add_argument("--width",type=int,default=1920);p.add_argument("--height",type=int,default=1080);p.add_argument("--fps",type=int,default=30);p.add_argument("--cell-size",type=int,default=5);p.add_argument("--block-size",type=int,default=150);p.add_argument("--ratios",default=".1,.2,.3,.4,.5");p.add_argument("--repeats",type=int,default=5);p.add_argument("--warmup-blocks",type=int,default=1);p.add_argument("--seed",type=int,default=20260809);p.add_argument("--crf",type=int,default=15);p.add_argument("--preset",default="medium");a=p.parse_args()
 c=TemporalConfig(width=a.width,height=a.height,fps=a.fps,cell_size=a.cell_size,block_size=a.block_size,ratios=tuple(map(float,a.ratios.split(','))),repeats=a.repeats,warmup_blocks=a.warmup_blocks,seed=a.seed);a.output.parent.mkdir(parents=True,exist_ok=True)
 cmd=["ffmpeg","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{c.width}x{c.height}","-r",str(c.fps),"-i","-","-an","-c:v","libx264","-crf",str(a.crf),"-preset",a.preset,"-pix_fmt","yuv420p",str(a.output)]
 try: proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
 except FileNotFoundError as e: raise RuntimeError("ffmpeg not found") from e
 try:
  for _,bits,_ in temporal_frames(c): proc.stdin.write(render_bits(bits,c).tobytes())
  proc.stdin.close()
  if proc.wait()!=0: raise RuntimeError("ffmpeg failed")
 except BaseException:
  if proc.poll() is None: proc.kill()
  raise
 manifest=c.manifest();manifest.update({"source_video":a.output.name,"crf":a.crf,"preset":a.preset});path=a.output.with_suffix('.manifest.json');path.write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(path)
if __name__=='__main__': main()
