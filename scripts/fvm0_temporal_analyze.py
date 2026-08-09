from __future__ import annotations
import argparse,csv,json,subprocess
from pathlib import Path
import matplotlib.pyplot as plt
try:
 from fvm0_temporal_common import aggregate_ratios
except ImportError:
 from scripts.fvm0_temporal_common import aggregate_ratios
def probe_frames(video: Path):
 command=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=key_frame,best_effort_timestamp_time,pict_type,pkt_size','-of','json',str(video)]
 try: data=json.loads(subprocess.run(command,capture_output=True,text=True,check=True).stdout)
 except FileNotFoundError as error: raise RuntimeError('ffprobe not found on PATH') from error
 except subprocess.CalledProcessError as error: raise RuntimeError(f'ffprobe failed: {error.stderr}') from error
 except json.JSONDecodeError as error: raise RuntimeError('ffprobe returned malformed JSON') from error
 if not isinstance(data.get('frames'),list): raise RuntimeError('ffprobe JSON lacks frames')
 return data['frames']
def analyze(directory:Path, ffprobe_video:Path|None=None):
 records=list(csv.DictReader((directory/'fvm0_temporal_frames.csv').open(encoding='utf8')))
 for r in records:
  for k in ('frame_index','block_index','phase','flip_count','bit_errors','zero_to_one','one_to_zero','changed_bit_errors','unchanged_bit_errors'): r[k]=int(r[k] or 0)
  for k in ('is_anchor','is_warmup'): r[k]=r[k]=='True'
  r['transition_ratio']=None if not r['transition_ratio'] else float(r['transition_ratio'])
 results=json.loads((directory/'fvm0_temporal_results.json').read_text(encoding='utf8')); ratios=aggregate_ratios(records,results['matrix']['cells_per_frame'])
 (directory/'fvm0_temporal_ratios.json').write_text(json.dumps(ratios,indent=2),encoding='utf8')
 with (directory/'fvm0_temporal_ratios.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=next(iter(ratios.values())).keys());w.writeheader();w.writerows(ratios.values())
 x=[float(k)*100 for k in ratios];fig,ax=plt.subplots();ax.plot(x,[v['ber'] for v in ratios.values()]);ax.set(xlabel='transition ratio (%)',ylabel='BER');fig.savefig(directory/'fvm0_temporal_ber_vs_ratio.png');plt.close(fig)
 if ffprobe_video:
  frames=probe_frames(ffprobe_video); warning=[]
  if len(frames)!=len(records): warning=['ffprobe frame count differs from FVM0-T decoded frame count; index-based codec join may be unreliable after insertion/deletion']
  fields=list(records[0])+['key_frame','pts_time','pict_type','pkt_size']
  with (directory/'fvm0_temporal_frames_with_codec.csv').open('w',newline='',encoding='utf8') as handle:
   writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
   for record,meta in zip(records,frames): writer.writerow({**record,'key_frame':meta.get('key_frame'),'pts_time':meta.get('best_effort_timestamp_time'),'pict_type':meta.get('pict_type'),'pkt_size':meta.get('pkt_size')})
  results.setdefault('warnings',[]).extend(warning);(directory/'fvm0_temporal_results.json').write_text(json.dumps(results,indent=2),encoding='utf8')
 return ratios
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('result_dir',type=Path);p.add_argument('--ffprobe-video',type=Path);a=p.parse_args();analyze(a.result_dir,a.ffprobe_video)
