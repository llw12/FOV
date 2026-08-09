from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
from scripts.fvm0_temporal_common import aggregate_ratios
def analyze(directory:Path):
 records=list(csv.DictReader((directory/'fvm0_temporal_frames.csv').open(encoding='utf8')))
 for r in records:
  for k in ('frame_index','block_index','phase','flip_count','bit_errors','zero_to_one','one_to_zero','changed_bit_errors','unchanged_bit_errors'): r[k]=int(r[k])
  for k in ('is_anchor','is_warmup'): r[k]=r[k]=='True'
  r['transition_ratio']=None if not r['transition_ratio'] else float(r['transition_ratio'])
 results=json.loads((directory/'fvm0_temporal_results.json').read_text(encoding='utf8')); ratios=aggregate_ratios(records,results['matrix']['cells_per_frame'])
 (directory/'fvm0_temporal_ratios.json').write_text(json.dumps(ratios,indent=2),encoding='utf8')
 with (directory/'fvm0_temporal_ratios.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=next(iter(ratios.values())).keys());w.writeheader();w.writerows(ratios.values())
 x=[float(k)*100 for k in ratios];fig,ax=plt.subplots();ax.plot(x,[v['ber'] for v in ratios.values()]);ax.set(xlabel='transition ratio (%)',ylabel='BER');fig.savefig(directory/'fvm0_temporal_ber_vs_ratio.png');plt.close(fig)
 return ratios
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);p.add_argument('--ffprobe-video');a=p.parse_args();analyze(a.result_dir)
