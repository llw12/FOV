import hashlib
import zlib
from pathlib import Path

import numpy as np
import pytest

from file2video import packet_stream
from fov import (SYMBOL_PACKET_OVERHEAD, MetaPacket, SymbolPacket, build_metadata, encode_meta,
                 encode_symbol, file_id_from_sha256, make_raptorq_engine, parse_packet, sha256_bytes,
                 symbol_packet_size)
from fvm_file2video import encode as encode_video
from fvm_file_common import (CODED_RS_BYTES, LOGICAL_BYTES, MAGIC, MAX_PACKET_BYTES, PHYSICAL_CONFIG,
                             VERSION, TRANSPORT_HEADER, TransportError, decode_transport_physical,
                             encode_transport_physical, matrix_to_physical, physical_to_matrix,
                             unwrap_packet, wrap_packet)
from fvm_diagnostics import FailureTracker, PacketDiagnostic
from fvm_video2file import (PacketRecoveryCoordinator, TransportIndexTracker, _write_result,
                            result_path)
from scripts.fvm0_rs_common import (RS_CODEWORDS, RSDecodeResult, decode_rs_codewords, deinterleave,
                                    encode_logical, interleave)


def _rewrite_crc(logical: bytearray) -> bytes:
    logical[-4:] = (zlib.crc32(logical[:-4]) & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(logical)


def _metadata(filename: str, data: bytes) -> tuple[dict, bytes]:
    digest = sha256_bytes(data)
    return build_metadata(filename, len(data), digest, 6400, 0.10), file_id_from_sha256(digest)


def _failed_rs(*failed_indices: int) -> RSDecodeResult:
    corrections = tuple(None if index in failed_indices else 0 for index in range(RS_CODEWORDS))
    histogram = {index: 0 for index in range(9)}
    histogram[0] = RS_CODEWORDS - len(failed_indices)
    return RSDecodeResult(
        logical=None,
        codeword_successes=RS_CODEWORDS - len(failed_indices),
        codeword_failures=len(failed_indices),
        corrected_symbols=0,
        max_corrections_per_codeword=0,
        correction_histogram=histogram,
        failed_codeword_indices=tuple(failed_indices),
        corrections_per_codeword=corrections,
    )


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


def test_symbol_packet_size_is_derived_and_validated():
    for payload_size in (1, 6400, 65535):
        packet = encode_symbol(b"12345678", 0, 0, bytes(payload_size))
        assert symbol_packet_size(payload_size) == len(packet) == payload_size + SYMBOL_PACKET_OVERHEAD
    for payload_size in (0, -1, 65536, True, 1.5):
        with pytest.raises(ValueError, match="payload size"):
            symbol_packet_size(payload_size)


def test_encoder_rejects_oversize_symbol_before_ffmpeg(tmp_path: Path, monkeypatch):
    maximum = MAX_PACKET_BYTES - SYMBOL_PACKET_OVERHEAD
    assert symbol_packet_size(maximum) == MAX_PACKET_BYTES
    source = tmp_path / "input.bin"
    source.write_bytes(b"fail-fast")

    def unexpected_popen(*args, **kwargs):
        pytest.fail("FFmpeg was started before FVM packet-capacity validation")

    monkeypatch.setattr("fvm_file2video.subprocess.Popen", unexpected_popen)
    with pytest.raises(ValueError, match=rf"symbol_size {maximum + 1}.*exceeding transport capacity"):
        encode_video(source, tmp_path / "invalid.mp4", symbol_size=maximum + 1)


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


def test_rs_reports_failed_codeword_indices_without_expected_truth():
    logical = wrap_packet(encode_symbol(b"abcdefgh", 0, 1, bytes(6400)), 4)
    codewords = encode_logical(logical)
    errors = np.arange(1, 13, dtype=np.uint8)
    codewords[3, :12] ^= errors
    codewords[17, :12] ^= errors
    result = decode_rs_codewords(codewords)
    assert result.logical is None
    assert result.failed_codeword_indices == (3, 17)
    assert result.corrections_per_codeword[3] is None
    assert result.corrections_per_codeword[17] is None


def test_fov_packet_crc_failure_is_dropped(tmp_path: Path):
    coordinator = PacketRecoveryCoordinator(tmp_path)
    damaged = bytearray(encode_symbol(b"abcdefgh", 0, 1, bytes(6400)))
    damaged[-1] ^= 1
    coordinator.feed(bytes(damaged), 0)
    assert coordinator.stats["packet_crc_failed"] == 1


def test_packet_diagnostic_uses_coordinator_parse_result(tmp_path: Path):
    data = b"descriptor"
    metadata, file_id = _metadata("descriptor.bin", data)
    coordinator = PacketRecoveryCoordinator(tmp_path)
    meta = coordinator.feed(encode_meta(file_id, metadata), 0)
    symbol = coordinator.feed(encode_symbol(file_id, 0, 0, bytes(6400)), 1)
    assert PacketDiagnostic.from_packet(meta).to_dict() == {"type": "META"}
    assert PacketDiagnostic.from_packet(symbol).to_dict() == {
        "type": "SYMBOL",
        "block_id": 0,
        "symbol_id": 0,
    }
    assert "payload" not in PacketDiagnostic.from_packet(symbol).to_dict()
    coordinator.cleanup()


def test_single_failure_gets_unambiguous_neighbor_context_and_inference():
    tracker = FailureTracker()
    previous = PacketDiagnostic("SYMBOL", 5, 721)
    following = PacketDiagnostic("SYMBOL", 5, 722)
    tracker.observe_success(10, previous)
    tracker.observe_failure(20, _failed_rs(3, 17))
    tracker.observe_success(12, following)
    summary = tracker.summary()
    event = summary["events"][0]
    assert event["observed_frame_index"] == 20
    assert event["failed_codewords"] == 2
    assert event["previous_successful"]["transport_index"] == 10
    assert event["next_successful"]["transport_index"] == 12
    assert event["inferred_transport_index"] == 11
    assert event["inference_confident"] is True
    assert event["neighbor_block_consistent"] is True
    assert event["neighbor_block_id"] == 5
    assert summary["longest_consecutive_burst"] == 1


def test_two_consecutive_failures_are_one_burst_and_inferred_in_order():
    tracker = FailureTracker()
    tracker.observe_success(100, PacketDiagnostic("META"))
    tracker.observe_failure(500, _failed_rs(3))
    tracker.observe_failure(501, _failed_rs(4, 8))
    tracker.observe_success(103, PacketDiagnostic("SYMBOL", 0, 1))
    summary = tracker.summary()
    assert [event["inferred_transport_index"] for event in summary["events"]] == [101, 102]
    assert [event["failed_codewords"] for event in summary["events"]] == [1, 2]
    assert summary["burst_count"] == 1
    assert summary["longest_consecutive_burst"] == 2


def test_transport_inference_is_rejected_when_gap_is_ambiguous():
    tracker = FailureTracker()
    tracker.observe_success(100, PacketDiagnostic("META"))
    tracker.observe_failure(200, _failed_rs(3))
    tracker.observe_success(105, PacketDiagnostic("SYMBOL", 0, 1))
    event = tracker.summary()["events"][0]
    assert event["inferred_transport_index"] is None
    assert event["inference_confident"] is False


def test_leading_failures_can_be_inferred_from_protocol_origin():
    tracker = FailureTracker()
    tracker.observe_failure(0, _failed_rs(3))
    tracker.observe_failure(1, _failed_rs(4))
    tracker.observe_success(2, PacketDiagnostic("META"))
    events = tracker.summary()["events"]
    assert [event["inferred_transport_index"] for event in events] == [0, 1]
    assert all(event["inference_confident"] for event in events)


def test_leading_inference_is_rejected_if_failure_does_not_start_at_observed_zero():
    tracker = FailureTracker()
    tracker.observe_failure(1, _failed_rs(3))
    tracker.observe_success(1, PacketDiagnostic("META"))
    event = tracker.summary()["events"][0]
    assert event["inferred_transport_index"] is None
    assert event["inference_confident"] is False


def test_trailing_failure_is_not_inferred_without_next_transport():
    tracker = FailureTracker()
    tracker.observe_success(100, PacketDiagnostic("META"))
    tracker.observe_failure(300, _failed_rs(7))
    event = tracker.summary()["events"][0]
    assert event["previous_successful"]["transport_index"] == 100
    assert event["next_successful"] is None
    assert event["inferred_transport_index"] is None
    assert event["inference_confident"] is False


def test_failure_burst_and_distance_summary():
    tracker = FailureTracker()
    for observed_index in (10, 11, 50, 100, 101, 102):
        tracker.observe_failure(observed_index, _failed_rs(observed_index % RS_CODEWORDS))
    summary = tracker.summary()
    assert summary["inter_failure_distances"] == [1, 39, 50, 1, 1]
    assert [burst["frame_count"] for burst in summary["consecutive_bursts"]] == [2, 1, 3]
    assert summary["burst_count"] == 3
    assert summary["longest_consecutive_burst"] == 3


def test_failure_event_details_are_bounded_but_aggregates_are_complete():
    tracker = FailureTracker(event_limit=2)
    for observed_index in range(5):
        tracker.observe_failure(observed_index, _failed_rs(3))
    summary = tracker.summary()
    assert summary["event_count_total"] == 5
    assert summary["events_recorded"] == 2
    assert summary["events_truncated"] is True
    assert summary["failed_codewords_total"] == 5
    assert summary["longest_consecutive_burst"] == 5


@pytest.mark.parametrize(
    "indices,gaps,duplicates,out_of_order",
    [
        ([4, 5, 6], 4, 0, 0),
        ([0, 2, 5], 3, 0, 0),
        ([0, 1, 1, 3], 1, 1, 0),
        ([3, 1, 2], 1, 0, 1),
    ],
)
def test_transport_indices_are_diagnostic_and_accept_gap_duplicate_reorder(
    indices, gaps, duplicates, out_of_order
):
    tracker = TransportIndexTracker()
    for frame_index in indices:
        tracker.observe(frame_index)
    assert tracker.gaps == gaps
    assert tracker.duplicates == duplicates
    assert tracker.out_of_order == out_of_order


def test_result_path_is_isolated_and_atomic(tmp_path: Path):
    expected = tmp_path / ".fvm" / "fvm_decode_results.json"
    assert result_path(tmp_path) == expected
    _write_result(tmp_path, {"failure": "no valid META packet found"})
    assert expected.read_text(encoding="utf-8")
    assert not expected.with_name("fvm_decode_results.json.tmp").exists()
    assert not (tmp_path / "fvm_decode_results.json").exists()


def test_diagnostic_namespace_file_collision_fails_explicitly(tmp_path: Path):
    (tmp_path / ".fvm").write_bytes(b"ordinary file")
    with pytest.raises(OSError):
        PacketRecoveryCoordinator(tmp_path)


@pytest.mark.parametrize(
    "filename,expected_recovered_name",
    [
        ("fvm_decode_results.json", "fvm_decode_results.json"),
        (".fvm", "recovered_.fvm"),
    ],
)
def test_recovered_filename_cannot_collide_with_diagnostics(
    tmp_path: Path, filename: str, expected_recovered_name: str
):
    data = (b"user-payload-not-diagnostic-json\x00\xff" * 1024) + bytes(range(251))
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / filename
    source.write_bytes(data)
    metadata, file_id = _metadata(filename, data)
    output_dir = tmp_path / "decoded"
    coordinator = PacketRecoveryCoordinator(output_dir)
    packets = packet_stream(source, metadata, file_id, make_raptorq_engine())
    for frame_index, packet in enumerate(packets):
        physical = encode_transport_physical(packet, frame_index)
        decoded = decode_transport_physical(physical)
        assert decoded.transport is not None
        coordinator.feed(decoded.transport.packet, decoded.transport.frame_index)
    recovered = coordinator.finalize()
    assert recovered.name == expected_recovered_name
    assert recovered.read_bytes() == data
    assert hashlib.sha256(recovered.read_bytes()).hexdigest() == metadata["sha256"]
    diagnostic = result_path(output_dir)
    _write_result(output_dir, {"file": {"exact": True}})
    assert diagnostic.exists()
    assert diagnostic != recovered
    assert recovered.read_bytes() == data


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
