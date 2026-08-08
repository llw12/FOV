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
import video2file
from video2file import (PreMetaBuffer, StreamingDecodeSession, bit_is_set, output_path, set_bit)


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


def test_output_path_uses_original_when_available(tmp_path) -> None:
    assert output_path(tmp_path, "../input.bin") == tmp_path / "input.bin"


def test_output_path_avoids_existing_file(tmp_path) -> None:
    (tmp_path / "input.bin").write_bytes(b"old")
    assert output_path(tmp_path, "input.bin") == tmp_path / "recovered_input.bin"


def test_output_path_handles_multiple_existing_recovered_files(tmp_path) -> None:
    for name in ("input.bin", "recovered_input.bin", "recovered_2_input.bin"):
        (tmp_path / name).write_bytes(b"old")
    assert output_path(tmp_path, "input.bin") == tmp_path / "recovered_3_input.bin"


def test_premeta_buffer_is_bounded_and_evicts() -> None:
    buffer = PreMetaBuffer(max_symbols=3, max_bytes=5, max_file_ids=2)
    for symbol_id in range(5):
        buffer.add(SymbolPacket(bytes([symbol_id % 3]) * 8, 0, symbol_id, b"xx"))
    assert len(buffer.entries) <= 3
    assert buffer.total_bytes <= 5
    assert len({file_id for file_id, _, _ in buffer.entries}) <= 2
    assert buffer.evicted_symbols > 0
    assert buffer.evicted_bytes > 0


def _feed_all_symbols(session: StreamingDecodeSession, data: bytes, metadata: dict, engine, *, reverse_blocks: bool = False) -> None:
    layout = validate_metadata(metadata)
    blocks = list(layout)
    if reverse_blocks:
        blocks.reverse()
    for block in blocks:
        source = data[block.block_id * metadata["block_size"]:block.block_id * metadata["block_size"] + block.data_size]
        for symbol_id, payload in enumerate(encode_block(source, metadata["symbol_size"], block.encoded_symbols, engine)):
            session.feed(SymbolPacket(session.file_id, block.block_id, symbol_id, payload))


def test_symbols_before_meta_flush_into_streaming_session(tmp_path) -> None:
    data = os.urandom(500)
    metadata = metadata_for(data)
    file_id = file_id_from_sha256(metadata["sha256"])
    engine = make_raptorq_engine()
    block = validate_metadata(metadata)[0]
    buffer = PreMetaBuffer()
    symbols = encode_block(data, 200, block.encoded_symbols, engine)
    for symbol_id, payload in enumerate(symbols[:2]):
        buffer.add(SymbolPacket(file_id, 0, symbol_id, payload))
    session = StreamingDecodeSession(file_id, metadata, tmp_path, engine)
    for packet in buffer.take_for_file(file_id):
        session.feed(packet)
    for symbol_id, payload in enumerate(symbols[2:], 2):
        session.feed(SymbolPacket(file_id, 0, symbol_id, payload))
    assert session.finalize().read_bytes() == data


def test_streaming_decoder_writes_blocks_immediately_and_handles_out_of_order_blocks(tmp_path) -> None:
    data = os.urandom(700)
    metadata = metadata_for(data, block_size=200)
    engine = make_raptorq_engine()
    session = StreamingDecodeSession(file_id_from_sha256(metadata["sha256"]), metadata, tmp_path, engine)
    first = validate_metadata(metadata)[0]
    first_data = data[:first.data_size]
    for symbol_id, payload in enumerate(encode_block(first_data, 200, first.encoded_symbols, engine)):
        session.feed(SymbolPacket(session.file_id, 0, symbol_id, payload))
    assert bit_is_set(session.decoded_bitmap, 0)
    assert 0 not in session.block_states
    with session.temporary.open("rb") as temporary:
        assert temporary.read(first.data_size) == first_data
    _feed_all_symbols(session, data, metadata, engine, reverse_blocks=True)
    assert session.finalize().read_bytes() == data


def test_duplicate_bitmap_only_calls_native_decoder_once(monkeypatch, tmp_path) -> None:
    calls = []

    class FakeDecoder:
        def __init__(self, *args):
            pass

        def add_symbol(self, symbol_id, payload):
            calls.append(symbol_id)
            return True

        def may_try_decode(self):
            return False

    monkeypatch.setattr(video2file, "Decoder", FakeDecoder)
    data = b"x"
    metadata = build_metadata("input.bin", 1, sha256_bytes(data), 1, 0.0, 1)
    session = StreamingDecodeSession(file_id_from_sha256(metadata["sha256"]), metadata, tmp_path, object())
    packet = SymbolPacket(session.file_id, 0, 0, b"x")
    session.feed(packet)
    session.feed(packet)
    assert calls == [0]
    assert session.stats["duplicate_symbols"] == 1
    session.cleanup()


def test_streaming_session_does_not_store_all_symbol_payloads(monkeypatch, tmp_path) -> None:
    class FakeDecoder:
        def __init__(self, *args):
            pass

        def add_symbol(self, symbol_id, payload):
            return True

        def may_try_decode(self):
            return False

    monkeypatch.setattr(video2file, "Decoder", FakeDecoder)
    data_size = 10_000
    metadata = build_metadata("large.bin", data_size, sha256_bytes(b"x" * data_size), 1, 0.0, data_size)
    session = StreamingDecodeSession(file_id_from_sha256(metadata["sha256"]), metadata, tmp_path, object())
    for symbol_id in range(data_size):
        session.feed(SymbolPacket(session.file_id, 0, symbol_id, b"x"))
    state = session.block_states[0]
    assert len(state.seen_bitmap) == (data_size + 7) // 8
    assert not hasattr(session, "symbols_by_file")
    assert state.received_unique == data_size
    session.cleanup()


def test_decode_failures_and_sha_mismatch_clean_temp(monkeypatch, tmp_path) -> None:
    class NeverDecoder:
        def __init__(self, *args):
            pass

        def add_symbol(self, symbol_id, payload):
            return True

        def may_try_decode(self):
            return False

    monkeypatch.setattr(video2file, "Decoder", NeverDecoder)
    metadata = build_metadata("input.bin", 1, sha256_bytes(b"x"), 1, 0.0, 1)
    failed = StreamingDecodeSession(file_id_from_sha256(metadata["sha256"]), metadata, tmp_path, object())
    with pytest.raises(RuntimeError, match="RaptorQ decode failed"):
        failed.finalize()
    failed.cleanup()
    assert not failed.temporary.exists()

    class WrongDecoder(NeverDecoder):
        def may_try_decode(self):
            return True

        def try_decode(self):
            return b"y"

    monkeypatch.setattr(video2file, "Decoder", WrongDecoder)
    mismatch = StreamingDecodeSession(file_id_from_sha256(metadata["sha256"]), metadata, tmp_path, object())
    mismatch.feed(SymbolPacket(mismatch.file_id, 0, 0, b"x"))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        mismatch.finalize()
    mismatch.cleanup()
    assert not mismatch.temporary.exists()
    assert not (tmp_path / "input.bin").exists()


@pytest.mark.parametrize("second_metadata, message", [
    (lambda metadata: dict(metadata, filename="other.bin"), "conflicting META"),
    (lambda metadata: build_metadata("second.bin", 1, sha256_bytes(b"z"), 1, 0.0, 1), "multiple FOV files"),
])
def test_decode_rejects_conflicting_or_multiple_meta_and_releases_capture(monkeypatch, tmp_path, second_metadata, message) -> None:
    metadata = build_metadata("input.bin", 1, sha256_bytes(b"x"), 1, 0.0, 1)
    second = second_metadata(metadata)
    packets = [MetaPacket(file_id_from_sha256(metadata["sha256"]), metadata),
               MetaPacket(file_id_from_sha256(second["sha256"]), second)]

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, _):
            return len(packets)

        def read(self):
            return (True, object()) if packets else (False, None)

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(video2file.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(video2file, "decode_qr", lambda frame: "eA==")
    monkeypatch.setattr(video2file, "parse_packet", lambda data: packets.pop(0))
    monkeypatch.setattr(video2file, "make_raptorq_engine", lambda: object())
    with pytest.raises(RuntimeError, match=message):
        video2file.decode(tmp_path / "video.mp4", tmp_path)
    assert capture.released


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
