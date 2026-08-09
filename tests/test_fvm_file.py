import hashlib
import zlib
from pathlib import Path

import numpy as np
import pytest

from file2video import packet_stream
from fov import (MetaPacket, SymbolPacket, build_metadata, encode_meta, encode_symbol,
                 file_id_from_sha256, make_raptorq_engine, parse_packet, sha256_bytes)
from fvm_file_common import (CODED_RS_BYTES, LOGICAL_BYTES, MAGIC, MAX_PACKET_BYTES, PHYSICAL_CONFIG,
                             VERSION, TRANSPORT_HEADER, TransportError, decode_transport_physical,
                             encode_transport_physical, matrix_to_physical, physical_to_matrix,
                             unwrap_packet, wrap_packet)
from fvm_video2file import PacketRecoveryCoordinator, TransportIndexTracker
from scripts.fvm0_rs_common import deinterleave, interleave


def _rewrite_crc(logical: bytearray) -> bytes:
    logical[-4:] = (zlib.crc32(logical[:-4]) & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(logical)


def _metadata(filename: str, data: bytes) -> tuple[dict, bytes]:
    digest = sha256_bytes(data)
    return build_metadata(filename, len(data), digest, 6400, 0.10), file_id_from_sha256(digest)


def test_transport_max_packet_and_validation():
    packet = bytes(MAX_PACKET_BYTES)
    assert unwrap_packet(wrap_packet(packet, 0)).packet == packet
    with pytest.raises(ValueError, match="exceeds"):
        wrap_packet(packet + b"x", 0)
    with pytest.raises(ValueError, match="non-empty"):
        wrap_packet(b"", 0)
    for index in (-1, 1 << 32):
        with pytest.raises(ValueError, match="uint32"):
            wrap_packet(b"x", index)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data.__setitem__(slice(0, 4), b"bad!"), "magic"),
        (lambda data: data.__setitem__(4, VERSION + 1), "version"),
        (lambda data: data.__setitem__(slice(9, 11), b"\x00\x00"), "packet length"),
    ],
)
def test_transport_malformed_headers_rejected(mutation, match):
    logical = bytearray(wrap_packet(b"packet", 7))
    mutation(logical)
    with pytest.raises(TransportError, match=match):
        unwrap_packet(_rewrite_crc(logical))


def test_transport_bad_crc_rejected():
    logical = bytearray(wrap_packet(b"packet", 7))
    logical[100] ^= 1
    with pytest.raises(TransportError, match="CRC32"):
        unwrap_packet(bytes(logical))


def test_symbol_6400_fits_full_physical_round_trip():
    payload = bytes((index * 29 + 7) & 0xFF for index in range(6400))
    packet = encode_symbol(b"12345678", 3, 9, payload)
    assert len(packet) == 6425 and len(packet) <= MAX_PACKET_BYTES
    physical = encode_transport_physical(packet, 11)
    assert len(physical) == 7200
    result = decode_transport_physical(matrix_to_physical(physical_to_matrix(physical)))
    assert result.transport is not None and result.transport.frame_index == 11
    parsed = parse_packet(result.transport.packet)
    assert isinstance(parsed, SymbolPacket) and parsed.payload == payload


def test_real_meta_packet_fits():
    data = b"meta-test" * 100
    metadata, file_id = _metadata("sample.bin", data)
    packet = encode_meta(file_id, metadata)
    assert len(packet) <= MAX_PACKET_BYTES
    parsed = parse_packet(unwrap_packet(wrap_packet(packet, 2)).packet)
    assert isinstance(parsed, MetaPacket) and parsed.metadata == metadata


def test_rs_corrects_eight_errors_per_codeword_then_all_crcs_pass():
    payload = bytes((index * 17) & 0xFF for index in range(6400))
    packet = encode_symbol(b"abcdefgh", 0, 1, payload)
    physical = bytearray(encode_transport_physical(packet, 4))
    codewords = deinterleave(bytes(physical[:CODED_RS_BYTES]))
    codewords[:, :8] ^= np.arange(1, 9, dtype=np.uint8)
    physical[:CODED_RS_BYTES] = interleave(codewords)
    result = decode_transport_physical(bytes(physical))
    assert result.transport is not None
    assert result.rs.corrected_symbols == 28 * 8
    parsed = parse_packet(result.transport.packet)
    assert isinstance(parsed, SymbolPacket) and parsed.payload == payload


def test_rs_over_limit_is_not_accepted():
    packet = encode_symbol(b"abcdefgh", 0, 1, bytes(6400))
    physical = bytearray(encode_transport_physical(packet, 4))
    codewords = deinterleave(bytes(physical[:CODED_RS_BYTES]))
    codewords[0, :12] ^= np.arange(1, 13, dtype=np.uint8)
    physical[:CODED_RS_BYTES] = interleave(codewords)
    result = decode_transport_physical(bytes(physical))
    assert result.transport is None


def test_fov_packet_crc_failure_is_dropped(tmp_path: Path):
    coordinator = PacketRecoveryCoordinator(tmp_path)
    damaged = bytearray(encode_symbol(b"abcdefgh", 0, 1, bytes(6400)))
    damaged[-1] ^= 1
    coordinator.feed(bytes(damaged), 0)
    assert coordinator.stats["packet_crc_failed"] == 1


def test_transport_indices_are_diagnostic_and_accept_gap_duplicate_reorder():
    tracker = TransportIndexTracker()
    for frame_index in (0, 2, 1, 1, 5):
        tracker.observe(frame_index)
    assert tracker.gaps == 2
    assert tracker.duplicates == 1
    assert tracker.out_of_order == 1


def test_memory_file_roundtrip_with_erasure_late_meta_reorder_and_duplicate(tmp_path: Path):
    data = bytes((index * 73 + 19) & 0xFF for index in range(128 * 1024))
    source = tmp_path / "original.bin"
    source.write_bytes(data)
    metadata, file_id = _metadata(source.name, data)
    packets = list(packet_stream(source, metadata, file_id, make_raptorq_engine()))
    parsed = [parse_packet(packet) for packet in packets]
    source_positions = [
        index for index, packet in enumerate(parsed)
        if isinstance(packet, SymbolPacket) and packet.symbol_id < 21
    ]
    assert len(source_positions) == 21
    dropped_position = source_positions[0]
    kept = [packet for index, packet in enumerate(packets) if index != dropped_position]
    meta_packets = [packet for packet in kept if isinstance(parse_packet(packet), MetaPacket)]
    symbol_packets = [packet for packet in kept if isinstance(parse_packet(packet), SymbolPacket)]
    assert len(symbol_packets) == 23  # K=21, N=24, with one source-symbol erasure.
    reordered = symbol_packets[:4] + [meta_packets[0]] + list(reversed(symbol_packets[4:12]))
    reordered += [symbol_packets[5]] + symbol_packets[12:] + meta_packets[1:]
    output_dir = tmp_path / "decoded"
    coordinator = PacketRecoveryCoordinator(output_dir)
    for observed_index, packet in enumerate(reordered):
        embedded_index = (observed_index * 7) % len(reordered)
        physical = encode_transport_physical(packet, embedded_index)
        result = decode_transport_physical(physical)
        assert result.transport is not None
        coordinator.feed(result.transport.packet, result.transport.frame_index)
    recovered = coordinator.finalize()
    assert recovered.read_bytes() == data
    assert hashlib.sha256(recovered.read_bytes()).hexdigest() == metadata["sha256"]
    assert coordinator.stats["symbols_before_meta"] == 4
    assert coordinator.session is not None
    assert coordinator.session.stats["duplicate_symbols"] >= 1
