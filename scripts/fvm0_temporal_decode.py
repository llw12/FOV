from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2
from scripts.fvm0_common import decode_frame
from scripts.fvm0_temporal_common import TemporalConfig,temporal_frames,aggregate_ratios
def decode(video:Path, manifest_path:Path, output:Path):
 m=json.loads(manifest_path.read_text(encoding='utf8')); c=TemporalConfig(width=m['width'],height=m['height'],fps=m['fps'],cell_size=m['cell_size'],block_size=m['block_size'],warmup_blocks=m['warmup_blocks'],repeats=m['repeats'],seed=m['seed'],ratios=tuple(m['ratios']))
 cap=cv2.VideoCapture(str(video)); output.mkdir(parents=True,exist_ok=True); records=[]
 if (int(cap.get(3)),int(cap.get(4))) != (c.width,c.height): raise RuntimeError('video resolution does not match manifest')
 for expected_record,bits,mask in temporal_frames(c,m['schedule']):
  ok,frame=cap.read()
  if not ok: break
  actual,luma=decode_frame(frame,c); errors=actual!=bits; row=dict(expected_record); row.update(bit_errors=int(errors.sum()),ber=float(errors.mean()),zero_to_one=int(((bits==0)&(actual==1)).sum()),one_to_zero=int(((bits==1)&(actual==0)).sum()),changed_cells=None if mask is None else int(mask.sum()),changed_bit_errors=None if mask is None else int((errors&mask).sum()),changed_ber=None if mask is None else float((errors&mask).sum()/mask.sum()),unchanged_cells=None if mask is None else int((~mask).sum()),unchanged_bit_errors=None if mask is None else int((errors&~mask).sum()),unchanged_ber=None if mask is None else float((errors&~mask).sum()/(~mask).sum())) ;records.append(row)
 cap.release(); fields=list(records[0]);
 with (output/'fvm0_temporal_frames.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 result={'format':m['format'],'matrix':{'cells_per_frame':c.cells_per_frame},'experiment':m,'overall':{'bit_errors':sum(x['bit_errors'] for x in records)},'ratios':aggregate_ratios(records,c.cells_per_frame),'warnings':([] if len(records)==len(m['schedule'])*c.block_size else ['frame count differs; no synchronization recovery'])}
 (output/'fvm0_temporal_results.json').write_text(json.dumps(result,indent=2),encoding='utf8'); return result
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();decode(a.video,a.manifest,a.output_dir)
