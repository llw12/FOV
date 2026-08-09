import math
import numpy as np
import pytest
from scripts.fvm0_temporal_common import TemporalConfig,flip_count,temporal_frames,theoretical_bits,aggregate_ratios
def test_default_geometry_schedule_and_density():
 c=TemporalConfig(repeats=1,warmup_blocks=1,block_size=3,seed=7)
 assert (c.rows,c.cols,c.cells_per_frame)==(216,384,82944)
 s=c.schedule();assert len(s)==6 and sorted(x['ratio'] for x in s[1:])==[.1,.2,.3,.4,.5]
 frames=list(temporal_frames(c));assert len(frames)==18
 for (_,a,_),(_,b,mask) in zip(frames,frames[1:]):
  if mask is not None: assert np.count_nonzero(a!=b)==mask.sum()
def test_validation_and_math():
 for r in (0,-.1,.6):
  with pytest.raises(ValueError): flip_count(10,r)
 assert flip_count(5,.5)==3
 assert theoretical_bits(8,2)==pytest.approx(math.log2(28))
def test_ratio_aggregate_excludes_anchor_warmup():
 rows=[{'is_warmup':True,'is_anchor':False,'transition_ratio':.1,'bit_errors':9,'changed_bit_errors':9,'unchanged_bit_errors':0,'flip_count':2,'block_index':0},{'is_warmup':False,'is_anchor':False,'transition_ratio':.1,'bit_errors':3,'changed_bit_errors':2,'unchanged_bit_errors':1,'flip_count':2,'block_index':1,'zero_to_one':1,'one_to_zero':2}]
 out=aggregate_ratios(rows,10)['0.1'];assert out['bit_errors']==3 and out['ber']==.3 and out['changed_ber']==1
