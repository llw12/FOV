from __future__ import annotations

import json
import math
import os
import random

import pytest

import fov
import file2video
from fov import (MAX_BLOCK_COUNT, MAX_SOURCE_SYMBOLS, BlockLayout, MetaPacket, MetadataError, PacketError,
                 SymbolPacket, build_metadata, decode_block, derive_block_layout, encode_block, encode_meta,
                 encode_symbol, encoded_symbol_count, file_id_from_sha256, iter_encoded_symbols,
                 make_raptorq_engine, parse_packet, sha256_bytes, source_symbol_count, validate_metadata)
from video2file import PacketCollection, collect_packet, select_metadata


def metadata_for(data: bytes, *, block_size: int = 1_000) -> dict:
    return build_metadata("input.bin", len(data), sha256_bytes(data), 200, 0.20, block_size)


def test_symbol_packet_roundtrip() -> None:
    packet = encode_symbol(b"12345678", 7, 123, b"payload" * 20)
    assert parse_packet(packet) == SymbolPacket(b"12345678", 7, 123, b"payload" * 20)


def test_meta_packet_roundtrip() -> None:
    metadata = metadata_for(b"test")
    assert parse_packet(encode_meta(file_id_from_sha256(metadata["sha256"]), metadata)) == MetaPacket(file_id_from_sha256(metadata["sha256"]), metadata)


@pytest.mark.parametrize("packet", [encode_meta(b"12345678", {"version": 1}), encode_symbol(b"12345678", 0, 0, b"x")])
def test_crc_rejects_any_modified_byte(packet: bytes) -> None:
    for index in range(len(packet)):
        corrupted = bytearray(packet)
        corrupted[index] ^= 1
        with pytest.raises(PacketError):
            parse_packet(bytes(corrupted))


def test_raptorq_recovers_shuffled_symbols_after_loss() -> None:
    data = os.urandom(10_000)
    size = 200
    k = source_symbol_count(len(data), size)
    engine = make_raptorq_engine()
    symbols = list(enumerate(encode_block(data, size, encoded_symbol_count(k, 0.20), engine)))
    random.Random(9).shuffle(symbols)
    kept = dict(symbols)
    for symbol_id in list(kept)[::10]:
        del kept[symbol_id]
    assert decode_block(len(data), size, kept, engine) == data


def test_duplicate_symbols_do_not_affect_decode() -> None:
    data = os.urandom(1_000)
    size = 200
    engine = make_raptorq_engine()
    symbols = encode_block(data, size, encoded_symbol_count(source_symbol_count(len(data), size), 0.20), engine)
    collected = {symbol_id: payload for symbol_id, payload in enumerate(symbols)}
    collected[0] = symbols[0]
    assert decode_block(len(data), size, collected, engine) == data


def test_multi_block_roundtrip_and_sha256() -> None:
    block_size = 1_000
    data = os.urandom(block_size * 2 + 123)
    metadata = metadata_for(data, block_size=block_size)
    layout = validate_metadata(metadata)
    engine = make_raptorq_engine()
    restored = []
    for block in layout:
        source = data[block.block_id * block_size:block.block_id * block_size + block.data_size]
        restored.append(decode_block(block.data_size, 200, dict(enumerate(encode_block(source, 200, block.encoded_symbols, engine))), engine))
    assert b"".join(restored) == data
    assert sha256_bytes(b"".join(restored)) == metadata["sha256"]


def test_meta_is_fixed_size_and_has_no_blocks() -> None:
    small = build_metadata("a.bin", 1_000, "0" * 64, 200, 0.20, 1_000)
    large = build_metadata("a.bin", 1_000 * 50_000, "0" * 64, 200, 0.20, 1_000)
    assert "blocks" not in small and "blocks" not in large
    assert len(json.dumps(large)) - len(json.dumps(small)) < 16


def test_derive_block_layout() -> None:
    layout = derive_block_layout(2_123, 1_000, 200, 0.20)
    assert [block.data_size for block in layout] == [1_000, 1_000, 123]
    assert [block.source_symbols for block in layout] == [5, 5, 1]
    assert [block.encoded_symbols for block in layout] == [6, 6, 2]


@pytest.mark.parametrize("mutator", [
    lambda meta: meta.update(block_count=2),
    lambda meta: meta.update(sha256="z" * 64),
    lambda meta: meta.update(original_size=-1),
    lambda meta: meta.update(symbol_size=0),
    lambda meta: meta.update(repair_ratio=-0.1),
    lambda meta: meta.update(repair_ratio=math.nan),
    lambda meta: meta.update(original_size=True),
    lambda meta: meta.update(filename=".."),
    lambda meta: meta.update(original_size=(MAX_SOURCE_SYMBOLS + 1) * 200, block_size=(MAX_SOURCE_SYMBOLS + 1) * 200),
    lambda meta: meta.update(original_size=MAX_BLOCK_COUNT + 1, block_size=1, block_count=MAX_BLOCK_COUNT + 1),
])
def test_invalid_metadata_fails_safely(mutator) -> None:
    metadata = metadata_for(b"data")
    mutator(metadata)
    with pytest.raises(MetadataError):
        validate_metadata(metadata)


def test_symbols_before_meta_are_collected_and_decoded() -> None:
    data = os.urandom(500)
    metadata = metadata_for(data)
    file_id = file_id_from_sha256(metadata["sha256"])
    block = validate_metadata(metadata)[0]
    engine = make_raptorq_engine()
    symbols = encode_block(data, 200, block.encoded_symbols, engine)
    collection = PacketCollection()
    for symbol_id, payload in enumerate(symbols[:3]):
        collect_packet(collection, SymbolPacket(file_id, 0, symbol_id, payload))
    collect_packet(collection, MetaPacket(file_id, metadata))
    for symbol_id, payload in enumerate(symbols[3:], 3):
        collect_packet(collection, SymbolPacket(file_id, 0, symbol_id, payload))
    selected_id, selected_meta = select_metadata(collection)
    assert selected_id == file_id and selected_meta == metadata
    assert collection.stats["symbols_before_meta"] == 3
    assert decode_block(len(data), 200, collection.symbols_by_file[file_id][0], engine) == data


def test_conflicting_meta_is_rejected() -> None:
    data = b"data"
    metadata = metadata_for(data)
    file_id = file_id_from_sha256(metadata["sha256"])
    collection = PacketCollection()
    collect_packet(collection, MetaPacket(file_id, metadata))
    conflicting = dict(metadata)
    conflicting["filename"] = "other.bin"
    collect_packet(collection, MetaPacket(file_id, conflicting))
    with pytest.raises(RuntimeError, match="conflicting META"):
        select_metadata(collection)


def test_multiple_file_ids_are_rejected() -> None:
    collection = PacketCollection()
    for data in (b"first", b"second"):
        metadata = metadata_for(data)
        collect_packet(collection, MetaPacket(file_id_from_sha256(metadata["sha256"]), metadata))
    with pytest.raises(RuntimeError, match="multiple FOV files"):
        select_metadata(collection)


def test_lazy_symbol_generator_generates_on_demand(monkeypatch) -> None:
    calls: list[int] = []

    class FakeEncoder:
        def __init__(self, data, symbol_size, engine):
            assert data == b"abc" and symbol_size == 1

        def gen_symbol(self, symbol_id: int) -> bytes:
            calls.append(symbol_id)
            return bytes([symbol_id])

    monkeypatch.setattr(fov, "Encoder", FakeEncoder)
    iterator = iter_encoded_symbols(b"abc", 1, BlockLayout(0, 3, 3, 3), object())
    assert calls == []
    assert next(iterator) == (0, b"\x00")
    assert calls == [0]


def test_packet_stream_uses_bounded_interleave_windows(monkeypatch, tmp_path) -> None:
    path = tmp_path / "five.bin"
    path.write_bytes(b"abcdefghi")
    metadata = build_metadata(path.name, 9, sha256_bytes(b"abcdefghi"), 1, 0.0, block_size=2)
    calls: list[tuple[bytes, int]] = []

    class FakeEncoder:
        def __init__(self, data: bytes):
            self.data = data

        def gen_symbol(self, symbol_id: int) -> bytes:
            calls.append((self.data, symbol_id))
            return b"x"

    monkeypatch.setattr(file2video, "create_raptor_encoder", lambda data, symbol_size, layout, engine: FakeEncoder(data))
    packets = file2video.packet_stream(path, metadata, file_id_from_sha256(metadata["sha256"]), object())
    symbols = []
    for packet in packets:
        parsed = parse_packet(packet)
        if isinstance(parsed, SymbolPacket):
            symbols.append(parsed)
    assert [(packet.block_id, packet.symbol_id) for packet in symbols] == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 0)]
    assert calls[0] == (b"ab", 0)


def test_estimated_packet_count_handles_meta_interval_boundary() -> None:
    layout = [BlockLayout(0, 1, 1, 300)]
    assert file2video.estimate_packet_count(layout) == (321, 300, 21)


def test_windows_dll_directory_handle_is_retained(monkeypatch, tmp_path) -> None:
    sentinel = object()
    fov._DLL_DIRECTORY_HANDLES.clear()
    monkeypatch.setattr(fov.os, "name", "nt")
    monkeypatch.setattr(fov.os, "add_dll_directory", lambda path: sentinel, raising=False)
    fov._add_windows_dll_directory(tmp_path)
    assert fov._DLL_DIRECTORY_HANDLES == [sentinel]
