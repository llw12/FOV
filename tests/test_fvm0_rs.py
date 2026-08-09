import json
import zlib
import numpy as np
import pytest

from scripts.fvm0_common import decode_frame, render_bits
from scripts.fvm0_rs_common import (CODED_RS_BYTES, LOGICAL_BYTES, MAGIC, RS_CODEWORDS, RS_K, RS_N,
    FVM0RSConfig, bytes_to_matrix, decode_codewords, deinterleave, encode_logical, interleave,
    logical_frame, matrix_to_bytes, physical_frame)


def test_default_geometry_and_rates():
    config=FVM0RSConfig(); assert (config.cols,config.rows,config.cells_per_frame)==(320,180,57600)
    assert config.physical_bytes_per_frame==7200 and config.reserved_bytes_per_frame==60
    assert LOGICAL_BYTES==6692 and CODED_RS_BYTES==7140


def test_logical_frame_hard_coded_golden():
    frame=logical_frame(7,0)
    assert frame[:25].hex()=="463652530100000000bbbd2a3e43754cb81c90b441a48a7568"
    assert frame[-4:].hex()=="9481a248"
    assert int.from_bytes(frame[-4:],"big")==zlib.crc32(frame[:-4])


def test_rs_layout_and_interleave_round_trip():
    logical=logical_frame(7,0); coded=encode_logical(logical)
    assert coded.shape==(28,255) and np.array_equal(deinterleave(interleave(coded)),coded)
    assert coded[0,:RS_K].tobytes()==logical[:RS_K]


@pytest.mark.parametrize("errors",[0,1,8])
def test_rs_corrects_up_to_eight_symbol_errors(errors):
    expected=logical_frame(7,0); coded=encode_logical(expected); coded[0,:errors]^=np.arange(1,errors+1,dtype=np.uint8)
    recovered=decode_codewords(coded,expected,0)
    assert recovered.payload_exact and recovered.crc_valid and recovered.codeword_failures==0


def test_nine_errors_are_never_accepted_as_exact():
    expected=logical_frame(7,0); coded=encode_logical(expected); coded[0,:9]^=np.arange(1,10,dtype=np.uint8)
    recovered=decode_codewords(coded,expected,0)
    assert not (recovered.crc_valid and recovered.payload_exact)


def test_interleave_spreads_aligned_bursts():
    coded=np.arange(RS_CODEWORDS*RS_N,dtype=np.uint32).astype(np.uint8).reshape(RS_CODEWORDS,RS_N)
    physical=np.frombuffer(interleave(coded),dtype=np.uint8).copy(); physical[:28]^=1
    assert np.all((deinterleave(physical.tobytes())!=coded).sum(axis=1)==1)
    physical=np.frombuffer(interleave(coded),dtype=np.uint8).copy(); physical[:28*8]^=1
    assert np.all((deinterleave(physical.tobytes())!=coded).sum(axis=1)==8)


def test_big_endian_bit_pack_render_decode_exact():
    config=FVM0RSConfig(frames=1); expected=physical_frame(config,0); bits=bytes_to_matrix(expected,config)
    actual,_=decode_frame(render_bits(bits,config),config)
    assert matrix_to_bytes(actual,config)==expected


def test_artificial_corruption_recovers_and_over_limit_rejects():
    expected=logical_frame(7,0); coded=encode_logical(expected); coded[:,:8]^=np.arange(1,9,dtype=np.uint8)
    assert decode_codewords(coded,expected,0).payload_exact
    coded=encode_logical(expected); coded[0,:9]^=np.arange(1,10,dtype=np.uint8)
    assert not decode_codewords(coded,expected,0).payload_exact


@pytest.mark.parametrize("key,value",[("format","bad"),("cell_size",5),("rows",1),("cols",1),("cells_per_frame",1),("physical_bytes_per_frame",1),("rs_n",1),("rs_k",1),("rs_codewords_per_frame",1),("logical_bytes_per_frame",1),("reserved_bytes_per_frame",1),("bit_order","little"),("interleave","bad")])
def test_manifest_validation(key,value):
    config=FVM0RSConfig(); manifest=config.manifest(15,"medium","probe.mp4"); manifest[key]=value
    with pytest.raises(ValueError): FVM0RSConfig.from_manifest(manifest)


def test_manifest_round_trip():
    config=FVM0RSConfig(); assert FVM0RSConfig.from_manifest(config.manifest(15,"medium","probe.mp4"))==config
