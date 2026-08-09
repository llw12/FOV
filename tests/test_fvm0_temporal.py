import json
import math
import numpy as np
import pytest
import csv
import subprocess
from pathlib import Path
from scripts.fvm0_temporal_common import TemporalConfig, aggregate_ratios, flip_count, temporal_frames, theoretical_bits
from scripts.fvm0_common import decode_frame, render_bits
from scripts.fvm0_temporal_analyze import load_records, probe_frames
from scripts.fvm0_temporal_decode import frame_warnings

def test_default_geometry():
    c=TemporalConfig(); assert (c.cols,c.rows,c.cells_per_frame)==(384,216,82944)

@pytest.mark.parametrize("kwargs",[{"width":0},{"height":0},{"fps":0},{"cell_size":0},{"width":True},{"ratios":()},{"ratios":(.1,.1)},{"ratios":(float('nan'),)},{"ratios":(float('inf'),)},{"ratios":(-.1,)},{"ratios":(.6,)}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError): TemporalConfig(**kwargs)

def test_flip_count_and_small_geometry_rejection():
    assert flip_count(5,.5)==3
    with pytest.raises(ValueError,match="not meaningful"): TemporalConfig(width=2,height=2,cell_size=2,ratios=(.5,))

def test_schedule_deterministic_balanced_and_manifest_validation():
    c=TemporalConfig(width=8,height=4,cell_size=2,repeats=2,ratios=(.25,.5),seed=7)
    assert c.schedule()==c.schedule()
    for repeat in range(2): assert sorted(x['ratio'] for x in c.schedule()[1+2*repeat:3+2*repeat])==[.25,.5]
    m=c.manifest(); assert TemporalConfig.from_manifest(m)==c
    for key,value in (("format","bad"),("prng","bad"),("rows",99)):
        bad=json.loads(json.dumps(m));bad[key]=value
        with pytest.raises(ValueError): TemporalConfig.from_manifest(bad)
    for key,value in (("warmup",True),("repeat",9),("block_index",9),("flip_count",9)):
        bad=json.loads(json.dumps(m));bad['schedule'][1][key]=value
        with pytest.raises(ValueError): TemporalConfig.from_manifest(bad)

def test_hard_coded_v1_golden_vector():
    c=TemporalConfig(8,4,30,2,3,1,1,7,(.25,.5)); frames=list(temporal_frames(c))
    assert c.schedule()==[{'block_index':0,'warmup':True,'ratio':.5,'flip_count':4},{'block_index':1,'warmup':False,'repeat':0,'ratio':.25,'flip_count':2},{'block_index':2,'warmup':False,'repeat':0,'ratio':.5,'flip_count':4}]
    assert np.array_equal(frames[0][1],np.array([[1,0,0,1],[1,0,1,0]],dtype=np.uint8))
    assert np.array_equal(frames[1][2],np.array([[0,0,1,0],[0,1,1,1]],dtype=bool))
    assert np.array_equal(frames[1][1],np.array([[1,0,1,1],[1,1,0,1]],dtype=np.uint8))
    assert np.array_equal(frames[3][1],np.array([[0,1,0,0],[1,0,1,1]],dtype=np.uint8))

def test_exact_transitions_and_anchor_reset():
    c=TemporalConfig(8,4,30,2,3,1,1,7,(.25,.5)); frames=list(temporal_frames(c))
    for index in (1,2,4,5,7,8): assert np.count_nonzero(frames[index][1]!=frames[index-1][1])==frames[index][0]['flip_count']
    assert not np.array_equal(frames[3][1],frames[2][1])

def _records():
    return [{'is_warmup':True,'is_anchor':False,'transition_ratio':.3,'bit_errors':99,'changed_bit_errors':99,'unchanged_bit_errors':0,'flip_count':3,'block_index':0,'zero_to_one':0,'one_to_zero':0},{'is_warmup':False,'is_anchor':True,'transition_ratio':None,'bit_errors':99,'changed_bit_errors':0,'unchanged_bit_errors':0,'flip_count':0,'block_index':1,'zero_to_one':0,'one_to_zero':0},{'is_warmup':False,'is_anchor':False,'transition_ratio':.3,'bit_errors':3,'changed_bit_errors':2,'unchanged_bit_errors':1,'flip_count':3,'block_index':1,'zero_to_one':1,'one_to_zero':2},{'is_warmup':False,'is_anchor':False,'transition_ratio':.1,'bit_errors':1,'changed_bit_errors':1,'unchanged_bit_errors':0,'flip_count':1,'block_index':2,'zero_to_one':1,'one_to_zero':0}]

def test_aggregate_sort_exclusion_denominators_and_rate():
    out=aggregate_ratios(_records(),10,10,5); assert list(out)==['0.1','0.3']; row=out['0.3']
    assert (row['ber'],row['changed_ber'],row['unchanged_ber'],row['fer'])==(.3,2/3,1/7,1)
    assert row['theoretical_raw_bps']==pytest.approx(theoretical_bits(10,3)*10)
    assert row['theoretical_effective_bps_including_anchor']==pytest.approx(theoretical_bits(10,3)*10*.8)

def test_theoretical_and_direct_render_decode():
    assert theoretical_bits(8,2)==pytest.approx(math.log2(28)); c=TemporalConfig(8,4,30,2,3,1,1,7,(.25,.5)); bits=next(temporal_frames(c))[1]; actual,_=decode_frame(render_bits(bits,c),c); assert np.array_equal(bits,actual)

def test_empty_or_malformed_csv_has_clear_error(tmp_path):
    with pytest.raises(RuntimeError,match="missing"): load_records(tmp_path/"missing.csv")
    (tmp_path/"empty.csv").write_text("",encoding="utf-8")
    with pytest.raises(RuntimeError,match="empty"): load_records(tmp_path/"empty.csv")

def test_ffprobe_json_and_errors(monkeypatch):
    class Completed:
        stdout='{"frames":[{"key_frame":1,"pict_type":"I","pkt_size":"42"}]}'
    monkeypatch.setattr(subprocess,"run",lambda *args,**kwargs:Completed())
    assert probe_frames(Path("video.mp4"))[0]["key_frame"]==1
    Completed.stdout="not json"
    with pytest.raises(RuntimeError,match="malformed"): probe_frames(Path("video.mp4"))
    def missing(*args,**kwargs): raise FileNotFoundError
    monkeypatch.setattr(subprocess,"run",missing)
    with pytest.raises(RuntimeError,match="not found"): probe_frames(Path("video.mp4"))

def test_missing_extra_frame_and_fps_warnings():
    assert "insertion/deletion" in frame_warnings(9,10,30,30)[0]
    assert "insertion/deletion" in frame_warnings(11,10,30,30)[0]
    assert any("FPS" in warning for warning in frame_warnings(10,10,29.9,30))
    assert frame_warnings(10,10,30,30)==[]
