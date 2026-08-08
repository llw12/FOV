from __future__ import annotations

import os
import random

import pytest

from fov import (MetaPacket, PacketError, SymbolPacket, build_metadata, decode_block, encode_block, encode_meta,
                 encode_symbol, encoded_symbol_count, file_id_from_sha256, parse_packet, sha256_bytes,
                 source_symbol_count, split_blocks)


def test_symbol_packet_roundtrip() -> None:
    file_id = b"12345678"
    parsed = encode_symbol(file_id, 7, 123, b"payload" * 20)
    result = parse_packet(parsed)
    assert result == SymbolPacket(file_id, 7, 123, b"payload" * 20)


def test_meta_packet_roundtrip() -> None:
    metadata = {"version": 1, "filename": "测试.bin", "original_size": 4}
    result = parse_packet(encode_meta(b"12345678", metadata))
    assert result == MetaPacket(b"12345678", metadata)


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
    symbols = list(enumerate(encode_block(data, size, encoded_symbol_count(k, 0.20))))
    random.Random(9).shuffle(symbols)
    kept = dict(symbols[::1])
    for symbol_id in list(kept)[::10]:
        del kept[symbol_id]
    assert decode_block(len(data), size, kept) == data


def test_duplicate_symbols_do_not_affect_decode() -> None:
    data = os.urandom(1_000)
    size = 200
    symbols = encode_block(data, size, encoded_symbol_count(source_symbol_count(len(data), size), 0.20))
    collected = {symbol_id: payload for symbol_id, payload in enumerate(symbols)}
    collected[0] = symbols[0]  # duplicate would be ignored by video2file's symbol map
    assert decode_block(len(data), size, collected) == data


def test_multi_block_roundtrip_and_sha256() -> None:
    block_size = 1_000
    data = os.urandom(block_size * 2 + 123)
    metadata = build_metadata("input.bin", data, 200, 0.20, block_size)
    restored = []
    for entry, block in zip(metadata["blocks"], split_blocks(data, block_size), strict=True):
        encoded = encode_block(block, 200, entry[2])
        restored.append(decode_block(len(block), 200, dict(enumerate(encoded))))
    result = b"".join(restored)
    assert result == data
    assert sha256_bytes(result) == metadata["sha256"]
    assert file_id_from_sha256(metadata["sha256"]) == bytes.fromhex(metadata["sha256"])[:8]
