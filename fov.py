"""FOV v1 binary protocol and RaptorQ helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyraptorq import Decoder, Encoder, RaptorQCppEngine

MAGIC = b"FOV1"
VERSION = 1
TYPE_META = 0
TYPE_SYMBOL = 1
WIDTH, HEIGHT, FPS = 1280, 720, 30
SYMBOL_SIZE = 200
REPAIR_RATIO = 0.20
BLOCK_SIZE = 8 * 1024 * 1024
META_REPEAT = 10
META_INTERVAL = 300
_META_HEADER = struct.Struct(">4sB8sH")
_SYMBOL_HEADER = struct.Struct(">4sB8sHIH")
_CRC = struct.Struct(">I")


class PacketError(ValueError):
    """An untrusted FOV packet failed structural or CRC validation."""


@dataclass(frozen=True)
class MetaPacket:
    file_id: bytes
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SymbolPacket:
    file_id: bytes
    block_id: int
    symbol_id: int
    payload: bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_id_from_sha256(sha256_hex: str) -> bytes:
    try:
        digest = bytes.fromhex(sha256_hex)
    except ValueError as exc:
        raise ValueError("sha256 must be hexadecimal") from exc
    if len(digest) != 32:
        raise ValueError("sha256 must be 32 bytes")
    return digest[:8]


def source_symbol_count(data_size: int, symbol_size: int) -> int:
    if data_size <= 0 or symbol_size <= 0:
        raise ValueError("data_size and symbol_size must be positive")
    return math.ceil(data_size / symbol_size)


def encoded_symbol_count(k: int, repair_ratio: float) -> int:
    if k <= 0 or repair_ratio < 0:
        raise ValueError("invalid symbol count or repair ratio")
    return math.ceil(k * (1 + repair_ratio))


def encode_meta(file_id: bytes, metadata: dict[str, Any]) -> bytes:
    _validate_file_id(file_id)
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > 0xFFFF:
        raise ValueError("metadata is too large")
    body = _META_HEADER.pack(MAGIC, TYPE_META, file_id, len(payload)) + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def encode_symbol(file_id: bytes, block_id: int, symbol_id: int, payload: bytes) -> bytes:
    _validate_file_id(file_id)
    if not 0 <= block_id <= 0xFFFF or not 0 <= symbol_id <= 0xFFFFFFFF:
        raise ValueError("block_id or symbol_id out of range")
    if not payload or len(payload) > 0xFFFF:
        raise ValueError("symbol payload must contain 1..65535 bytes")
    body = _SYMBOL_HEADER.pack(MAGIC, TYPE_SYMBOL, file_id, block_id, symbol_id, len(payload)) + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def parse_packet(packet: bytes) -> MetaPacket | SymbolPacket:
    if len(packet) < _META_HEADER.size + _CRC.size:
        raise PacketError("packet is too short")
    magic, packet_type = packet[:4], packet[4]
    if magic != MAGIC:
        raise PacketError("invalid magic")
    if packet_type == TYPE_META:
        header = _META_HEADER
        if len(packet) < header.size + _CRC.size:
            raise PacketError("META packet is too short")
        _, _, file_id, payload_len = header.unpack_from(packet)
        expected_size = header.size + payload_len + _CRC.size
        if len(packet) != expected_size:
            raise PacketError("META payload length mismatch")
        payload = packet[header.size:-_CRC.size]
        _validate_crc(packet)
        try:
            metadata = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PacketError("invalid META JSON") from exc
        if not isinstance(metadata, dict):
            raise PacketError("META JSON must be an object")
        return MetaPacket(file_id, metadata)
    if packet_type == TYPE_SYMBOL:
        header = _SYMBOL_HEADER
        if len(packet) < header.size + _CRC.size:
            raise PacketError("SYMBOL packet is too short")
        _, _, file_id, block_id, symbol_id, payload_len = header.unpack_from(packet)
        expected_size = header.size + payload_len + _CRC.size
        if len(packet) != expected_size:
            raise PacketError("SYMBOL payload length mismatch")
        _validate_crc(packet)
        return SymbolPacket(file_id, block_id, symbol_id, packet[header.size:-_CRC.size])
    raise PacketError("unknown packet type")


def make_raptorq_engine() -> RaptorQCppEngine:
    """Load pyraptorq's bundled engine, including its Windows AMD64 naming workaround."""
    try:
        return RaptorQCppEngine.default()
    except Exception as default_error:
        import pyraptorq

        dll = Path(pyraptorq.__file__).parent / "distlib" / "windows" / "libpyraptorq.x86_64.dll"
        if dll.is_file():
            # The wheel's MinGW-built DLL has external runtime dependencies on
            # many Windows hosts. Git for Windows is a common source of them.
            for runtime_dir in (dll.parent, Path(r"C:\Program Files\Git\mingw64\bin")):
                if runtime_dir.is_dir() and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(runtime_dir))
            return RaptorQCppEngine(str(dll))
        raise RuntimeError("Unable to load pyraptorq native engine") from default_error


def encode_block(data: bytes, symbol_size: int, symbol_count: int) -> list[bytes]:
    encoder = Encoder(data, symbol_size, make_raptorq_engine())
    return [encoder.gen_symbol(symbol_id) for symbol_id in range(symbol_count)]


def decode_block(data_size: int, symbol_size: int, symbols: dict[int, bytes]) -> bytes | None:
    decoder = Decoder(source_symbol_count(data_size, symbol_size), symbol_size, data_size, make_raptorq_engine())
    for symbol_id, payload in symbols.items():
        decoder.add_symbol(symbol_id, payload)
        if decoder.may_try_decode():
            result = decoder.try_decode()
            if result is not None:
                return result[:data_size]
    return None


def split_blocks(data: bytes, block_size: int = BLOCK_SIZE) -> list[bytes]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not data:
        raise ValueError("empty files are not supported by pyraptorq")
    return [data[offset:offset + block_size] for offset in range(0, len(data), block_size)]


def build_metadata(filename: str, data: bytes, symbol_size: int, repair_ratio: float, block_size: int = BLOCK_SIZE) -> dict[str, Any]:
    blocks = split_blocks(data, block_size)
    # Compact tuples keep the repeated META QR below the 720p limit at
    # box_size=8: [data_size, source_symbol_count, encoded_symbol_count].
    block_entries: list[list[int]] = []
    for block_id, block in enumerate(blocks):
        k = source_symbol_count(len(block), symbol_size)
        block_entries.append([len(block), k, encoded_symbol_count(k, repair_ratio)])
    digest = sha256_bytes(data)
    return {"version": VERSION, "filename": Path(filename).name, "original_size": len(data),
            "sha256": digest, "encoded_size": len(data), "compression": "none", "symbol_size": symbol_size,
            "repair_ratio": repair_ratio, "block_size": block_size, "block_count": len(blocks), "blocks": block_entries}


def _validate_crc(packet: bytes) -> None:
    body, checksum = packet[:-_CRC.size], packet[-_CRC.size:]
    if _CRC.unpack(checksum)[0] != (zlib.crc32(body) & 0xFFFFFFFF):
        raise PacketError("CRC32 mismatch")


def _validate_file_id(file_id: bytes) -> None:
    if len(file_id) != 8:
        raise ValueError("file_id must be exactly 8 bytes")
