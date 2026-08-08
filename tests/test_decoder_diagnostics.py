from __future__ import annotations

import video2file
from fov import SymbolPacket, build_metadata, file_id_from_sha256, sha256_bytes
from video2file import PreMetaBuffer, StreamingDecodeSession


def test_premeta_buffer_preserves_frame_index() -> None:
    packet = SymbolPacket(b"12345678", 0, 7, b"xx")
    buffer = PreMetaBuffer()
    assert buffer.add(packet, frame_index=42)
    assert buffer.take_for_file_with_frames(packet.file_id) == [(packet, 42)]


def test_streaming_session_records_decode_frame(monkeypatch, tmp_path) -> None:
    class FakeDecoder:
        def __init__(self, *args):
            pass

        def add_symbol(self, symbol_id, payload):
            return True

        def may_try_decode(self):
            return True

        def try_decode(self):
            return b"x"

    monkeypatch.setattr(video2file, "Decoder", FakeDecoder)
    metadata = build_metadata("input.bin", 1, sha256_bytes(b"x"), 1, 1.0, 1)
    file_id = file_id_from_sha256(metadata["sha256"])
    session = StreamingDecodeSession(file_id, metadata, tmp_path, object())

    session.feed(SymbolPacket(file_id, 0, 1, b"x"), frame_index=37)

    assert session.decoded_at_frames == {0: 37}
    assert session.decoded_block_count == 1
    session.cleanup()
