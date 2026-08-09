"""Decode and measure an FVM0-T video."""
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
    from fvm0_temporal_common import TemporalConfig, aggregate_ratios, temporal_frames
    from fvm0_temporal_analyze import analyze
except ImportError:
    from scripts.fvm0_common import LumaStats, decode_frame
    from scripts.fvm0_temporal_common import TemporalConfig, aggregate_ratios, temporal_frames
    from scripts.fvm0_temporal_analyze import analyze


def _summary(records, cells):
    frames=len(records); errors=sum(r["bit_errors"] for r in records); total=frames*cells
    return {"frames":frames,"compared_frames":frames,"total_bits":total,"correct_bits":total-errors,"bit_errors":errors,"ber":errors/total if total else None,"zero_to_one":sum(r["zero_to_one"] for r in records),"one_to_zero":sum(r["one_to_zero"] for r in records),"frames_with_errors":sum(r["bit_errors"]>0 for r in records),"fer":sum(r["bit_errors"]>0 for r in records)/frames if frames else None}


def frame_warnings(actual_frames: int, expected_frames: int, actual_fps: float, expected_fps: float) -> list[str]:
    warnings=[]
    if actual_frames!=expected_frames: warnings.append("FVM0-T has no synchronization metadata; frame insertion/deletion makes index-based comparison unreliable after synchronization loss.")
    if abs(actual_fps-expected_fps)>.01: warnings.append("decoded FPS differs from manifest")
    return warnings


def decode(video: Path, manifest_path: Path, output: Path) -> dict:
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")); config=TemporalConfig.from_manifest(manifest)
    capture=cv2.VideoCapture(str(video))
    if not capture.isOpened(): raise RuntimeError("cannot open video")
    width,height=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=float(capture.get(cv2.CAP_PROP_FPS))
    if (width,height)!=(config.width,config.height): capture.release(); raise RuntimeError("video resolution does not match manifest")
    records=[]; spatial=np.zeros((config.rows,config.cols),dtype=np.uint64); luma_stats={}
    for metadata,bits,mask in temporal_frames(config,manifest["schedule"]):
        ok,frame=capture.read()
        if not ok: break
        actual,luma=decode_frame(frame,config); errors=actual!=bits; spatial+=errors
        row={**metadata,"bit_errors":int(errors.sum()),"ber":float(errors.mean()),"zero_to_one":int(((bits==0)&(actual==1)).sum()),"one_to_zero":int(((bits==1)&(actual==0)).sum()),"changed_cells":None,"changed_bit_errors":None,"changed_ber":None,"unchanged_cells":None,"unchanged_bit_errors":None,"unchanged_ber":None}
        if mask is not None:
            row.update(changed_cells=int(mask.sum()),changed_bit_errors=int((errors&mask).sum()),changed_ber=float((errors&mask).sum()/mask.sum()),unchanged_cells=int((~mask).sum()),unchanged_bit_errors=int((errors&~mask).sum()),unchanged_ber=float((errors&~mask).sum()/(~mask).sum()))
        if not metadata["is_warmup"] and not metadata["is_anchor"]:
            key=str(float(metadata["transition_ratio"])); pair=luma_stats.setdefault(key,(LumaStats(),LumaStats())); pair[0].add(luma[bits==0]); pair[1].add(luma[bits==1])
        records.append(row)
    actual_frames=len(records)
    while capture.read()[0]: actual_frames+=1
    capture.release()
    if not records: raise RuntimeError("no video frames available for FVM0-T comparison")
    output.mkdir(parents=True,exist_ok=True)
    with (output/"fvm0_temporal_frames.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    expected_frames=len(manifest["schedule"])*config.block_size
    warnings=frame_warnings(actual_frames,expected_frames,fps,config.fps)
    luminance={}
    for ratio in sorted(luma_stats, key=float):
        zero,one=luma_stats[ratio]
        zs,os=zero.summary(),one.summary(); luminance[ratio]={"expected_zero":zs,"expected_one":os,"expected_zero_histogram":zero.histogram.tolist(),"expected_one_histogram":one.histogram.tolist(),"p1_p99_margin":os["p1"]-zs["p99"]}
    result={"format":manifest["format"],"video":{"expected_resolution":f"{config.width}x{config.height}","actual_resolution":f"{width}x{height}","expected_fps":config.fps,"actual_fps":fps,"expected_frames":expected_frames,"actual_frames":actual_frames,"compared_frames":len(records)},"matrix":{"cell_size":config.cell_size,"rows":config.rows,"columns":config.cols,"cells_per_frame":config.cells_per_frame},"experiment":manifest,"overall":_summary(records,config.cells_per_frame),"warmup":_summary([r for r in records if r["is_warmup"]],config.cells_per_frame),"anchors":_summary([r for r in records if r["is_anchor"]],config.cells_per_frame),"ratios":aggregate_ratios(records,config.cells_per_frame,config.fps,config.block_size),"luminance_by_ratio":luminance,"warnings":warnings}
    (output/"fvm0_temporal_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(12,6)); image=ax.imshow(spatial,aspect="auto",cmap="magma");ax.set_title("FVM0-T cell error count - all compared frames");fig.colorbar(image,ax=ax);fig.tight_layout();fig.savefig(output/"fvm0_temporal_error_heatmap.png");plt.close(fig)
    return analyze(output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("video",type=Path);parser.add_argument("manifest",type=Path);parser.add_argument("--output-dir",type=Path,required=True);args=parser.parse_args();decode(args.video,args.manifest,args.output_dir)
if __name__=="__main__":main()
