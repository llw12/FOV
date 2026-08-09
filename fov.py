"""FOV v1 packet protocol, block layout, and pyraptorq adapter."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import platform
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from pyraptorq import Decoder, Encoder, RaptorQCppEngine

MAGIC = b"FOV1"
VERSION = 1
TYPE_META = 0
TYPE_SYMBOL = 1
WIDTH, HEIGHT, FPS = 1280, 720, 30
SYMBOL_SIZE = 200
REPAIR_RATIO = 0.20
MAX_REPAIR_RATIO = 5.0
BLOCK_SIZE = 8 * 1024 * 1024
INTERLEAVE_WINDOW = 4
MAX_BLOCK_COUNT = 1 << 16
MAX_SYMBOL_ID = (1 << 32) - 1
# cpp-raptorq's RFC6330 parameter table ends at K_padded=56403; Rfc::get_parameters
# returns "K is too big" after this value (tdfec/td/fec/raptorq/Rfc.cpp).
MAX_SOURCE_SYMBOLS = 56_403
META_REPEAT = 10
META_INTERVAL = 300
_META_HEADER = struct.Struct(">4sB8sH")
_SYMBOL_HEADER = struct.Struct(">4sB8sHIH")
_CRC = struct.Struct(">I")
SYMBOL_PACKET_OVERHEAD = _SYMBOL_HEADER.size + _CRC.size
_DLL_DIRECTORY_HANDLES: list[Any] = []


class PacketError(ValueError):
    """An untrusted FOV packet failed structural or CRC validation."""


class MetadataError(ValueError):
    """Untrusted META JSON failed FOV v1 validation."""


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


@dataclass(frozen=True)
class BlockLayout:
    block_id: int
    data_size: int
    source_symbols: int
    encoded_symbols: int


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
    _require_positive_int("data_size", data_size)
    _require_positive_int("symbol_size", symbol_size)
    count = math.ceil(data_size / symbol_size)
    if count > MAX_SOURCE_SYMBOLS:
        raise ValueError(f"RaptorQ source symbol count {count} exceeds library limit {MAX_SOURCE_SYMBOLS}")
    return count


def encoded_symbol_count(source_symbols: int, repair_ratio: float) -> int:
    _require_positive_int("source_symbols", source_symbols)
    _validate_repair_ratio(repair_ratio)
    count = math.ceil(source_symbols * (1.0 + repair_ratio))
    if count - 1 > MAX_SYMBOL_ID:
        raise ValueError("RaptorQ symbol_id exceeds uint32 range")
    return count


def derive_block_layout(original_size: int, block_size: int, symbol_size: int, repair_ratio: float) -> list[BlockLayout]:
    """Derive all FOV block properties from fixed-size META fields."""
    _require_positive_int("original_size", original_size)
    _require_positive_int("block_size", block_size)
    _require_positive_int("symbol_size", symbol_size)
    _validate_repair_ratio(repair_ratio)
    block_count = math.ceil(original_size / block_size)
    if block_count > MAX_BLOCK_COUNT:
        raise ValueError(f"block_count {block_count} exceeds uint16 limit {MAX_BLOCK_COUNT}")
    layout = []
    for block_id in range(block_count):
        data_size = min(block_size, original_size - block_id * block_size)
        k = source_symbol_count(data_size, symbol_size)
        layout.append(BlockLayout(block_id, data_size, k, encoded_symbol_count(k, repair_ratio)))
    return layout


def build_metadata(filename: str, original_size: int, sha256_hex: str, symbol_size: int, repair_ratio: float,
                   block_size: int = BLOCK_SIZE) -> dict[str, Any]:
    safe_name = safe_filename(filename)
    file_id_from_sha256(sha256_hex)
    layout = derive_block_layout(original_size, block_size, symbol_size, repair_ratio)
    return {"version": VERSION, "filename": safe_name, "original_size": original_size,
            "sha256": sha256_hex, "encoded_size": original_size, "compression": "none",
            "symbol_size": symbol_size, "repair_ratio": repair_ratio, "block_size": block_size,
            "block_count": len(layout)}


def validate_metadata(metadata: dict[str, Any]) -> list[BlockLayout]:
    """Validate untrusted META and return its deterministic block layout."""
    required = {"version", "filename", "original_size", "sha256", "encoded_size", "compression",
                "symbol_size", "repair_ratio", "block_size", "block_count"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise MetadataError("META fields do not match FOV v1")
    if not _is_int(metadata["version"]) or metadata["version"] != VERSION:
        raise MetadataError("unsupported META version")
    try:
        safe_filename(metadata["filename"])
    except ValueError as exc:
        raise MetadataError(str(exc)) from exc
    for name in ("original_size", "encoded_size", "symbol_size", "block_size", "block_count"):
        try:
            _require_positive_int(name, metadata[name])
        except ValueError as exc:
            raise MetadataError(str(exc)) from exc
    if metadata["block_count"] > MAX_BLOCK_COUNT:
        raise MetadataError("block_count exceeds uint16 limit")
    if not isinstance(metadata["compression"], str) or metadata["compression"] != "none":
        raise MetadataError("unsupported compression")
    if metadata["encoded_size"] != metadata["original_size"]:
        raise MetadataError("encoded_size must equal original_size for compression=none")
    if not isinstance(metadata["sha256"], str) or len(metadata["sha256"]) != 64:
        raise MetadataError("invalid sha256")
    try:
        file_id_from_sha256(metadata["sha256"])
        _validate_repair_ratio(metadata["repair_ratio"])
        layout = derive_block_layout(metadata["original_size"], metadata["block_size"], metadata["symbol_size"], metadata["repair_ratio"])
    except ValueError as exc:
        raise MetadataError(str(exc)) from exc
    if len(layout) != metadata["block_count"]:
        raise MetadataError("block_count does not match derived layout")
    return layout


def encode_meta(file_id: bytes, metadata: dict[str, Any]) -> bytes:
    _validate_file_id(file_id)
    payload = json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > 0xFFFF:
        raise ValueError("metadata is too large")
    body = _META_HEADER.pack(MAGIC, TYPE_META, file_id, len(payload)) + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def encode_symbol(file_id: bytes, block_id: int, symbol_id: int, payload: bytes) -> bytes:
    _validate_file_id(file_id)
    if not _is_int(block_id) or not 0 <= block_id < MAX_BLOCK_COUNT:
        raise ValueError("block_id out of range")
    if not _is_int(symbol_id) or not 0 <= symbol_id <= MAX_SYMBOL_ID:
        raise ValueError("symbol_id out of range")
    if not payload or len(payload) > 0xFFFF:
        raise ValueError("symbol payload must contain 1..65535 bytes")
    body = _SYMBOL_HEADER.pack(MAGIC, TYPE_SYMBOL, file_id, block_id, symbol_id, len(payload)) + payload
    return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)


def symbol_packet_size(payload_size: int) -> int:
    """Return the encoded FOV1 SYMBOL packet size for a valid payload length."""
    if not _is_int(payload_size) or not 1 <= payload_size <= 0xFFFF:
        raise ValueError("symbol payload size must be an integer between 1 and 65535")
    return SYMBOL_PACKET_OVERHEAD + payload_size


def parse_packet(packet: bytes) -> MetaPacket | SymbolPacket:
    if len(packet) < _META_HEADER.size + _CRC.size:
        raise PacketError("packet is too short")
    magic, packet_type = packet[:4], packet[4]
    if magic != MAGIC:
        raise PacketError("invalid magic")
    if packet_type == TYPE_META:
        if len(packet) < _META_HEADER.size + _CRC.size:
            raise PacketError("META packet is too short")
        _, _, file_id, payload_len = _META_HEADER.unpack_from(packet)
        if len(packet) != _META_HEADER.size + payload_len + _CRC.size:
            raise PacketError("META payload length mismatch")
        _validate_crc(packet)
        try:
            metadata = json.loads(packet[_META_HEADER.size:-_CRC.size].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PacketError("invalid META JSON") from exc
        if not isinstance(metadata, dict):
            raise PacketError("META JSON must be an object")
        return MetaPacket(file_id, metadata)
    if packet_type == TYPE_SYMBOL:
        if len(packet) < _SYMBOL_HEADER.size + _CRC.size:
            raise PacketError("SYMBOL packet is too short")
        _, _, file_id, block_id, symbol_id, payload_len = _SYMBOL_HEADER.unpack_from(packet)
        if len(packet) != _SYMBOL_HEADER.size + payload_len + _CRC.size:
            raise PacketError("SYMBOL payload length mismatch")
        _validate_crc(packet)
        return SymbolPacket(file_id, block_id, symbol_id, packet[_SYMBOL_HEADER.size:-_CRC.size])
    raise PacketError("unknown packet type")


def iter_file_blocks(path: Path, block_size: int) -> Iterator[tuple[int, bytes]]:
    _require_positive_int("block_size", block_size)
    with path.open("rb") as source:
        for block_id in itertools.count():
            data = source.read(block_size)
            if not data:
                return
            yield block_id, data


def iter_encoded_symbols(data: bytes, symbol_size: int, layout: BlockLayout,
                         engine: RaptorQCppEngine) -> Iterator[tuple[int, bytes]]:
    """Lazily generate one RaptorQ symbol at a time for a source block."""
    encoder = create_raptor_encoder(data, symbol_size, layout, engine)
    for symbol_id in range(layout.encoded_symbols):
        yield symbol_id, encoder.gen_symbol(symbol_id)


def create_raptor_encoder(data: bytes, symbol_size: int, layout: BlockLayout, engine: RaptorQCppEngine) -> Encoder:
    if len(data) != layout.data_size:
        raise ValueError("block data length does not match layout")
    if source_symbol_count(len(data), symbol_size) != layout.source_symbols:
        raise ValueError("layout source symbol count does not match symbol_size")
    return Encoder(data, symbol_size, engine)


def encode_block(data: bytes, symbol_size: int, symbol_count: int, engine: RaptorQCppEngine | None = None) -> list[bytes]:
    layout = BlockLayout(0, len(data), source_symbol_count(len(data), symbol_size), symbol_count)
    engine = engine or make_raptorq_engine()
    encoder = create_raptor_encoder(data, symbol_size, layout, engine)
    return [encoder.gen_symbol(symbol_id) for symbol_id in range(symbol_count)]


def decode_block(data_size: int, symbol_size: int, symbols: dict[int, bytes], engine: RaptorQCppEngine | None = None) -> bytes | None:
    k = source_symbol_count(data_size, symbol_size)
    engine = engine or make_raptorq_engine()
    decoder = Decoder(k, symbol_size, data_size, engine)
    for symbol_id, payload in symbols.items():
        if not 0 <= symbol_id <= MAX_SYMBOL_ID or len(payload) != symbol_size:
            continue
        decoder.add_symbol(symbol_id, payload)
        if decoder.may_try_decode():
            result = decoder.try_decode()
            if result is not None:
                return result[:data_size]
    return None


def make_raptorq_engine() -> RaptorQCppEngine:
    """Load pyraptorq's bundled engine and retain Windows DLL search handles."""
    try:
        return RaptorQCppEngine.default()
    except Exception as default_error:
        import pyraptorq

        package_dir = Path(pyraptorq.__file__).parent
        windows_dir = package_dir / "distlib" / "windows"
        machine = platform.machine().lower()
        aliases = {"amd64", "x86_64"} if machine in {"amd64", "x86_64"} else {machine}
        candidates = [path for path in windows_dir.glob("libpyraptorq.*.dll")
                      if path.stem.rsplit(".", 1)[-1].lower() in aliases]
        if not candidates:
            raise RuntimeError(f"Unable to find pyraptorq bundled DLL for {machine}; default loader failed: {default_error}") from default_error
        _add_windows_dll_directory(windows_dir)
        git_runtime = Path(r"C:\Program Files\Git\mingw64\bin")
        if git_runtime.is_dir():
            _add_windows_dll_directory(git_runtime)
        try:
            return RaptorQCppEngine(str(candidates[0]))
        except OSError as exc:
            raise RuntimeError(f"Unable to load pyraptorq bundled DLL {candidates[0]}: {exc}") from exc


def _add_windows_dll_directory(directory: Path) -> None:
    if os.name == "nt" and directory.is_dir() and hasattr(os, "add_dll_directory"):
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))


def _validate_crc(packet: bytes) -> None:
    body, checksum = packet[:-_CRC.size], packet[-_CRC.size:]
    if _CRC.unpack(checksum)[0] != (zlib.crc32(body) & 0xFFFFFFFF):
        raise PacketError("CRC32 mismatch")


def _validate_file_id(file_id: bytes) -> None:
    if not isinstance(file_id, bytes) or len(file_id) != 8:
        raise ValueError("file_id must be exactly 8 bytes")


def safe_filename(filename: Any) -> str:
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")
    name = Path(filename).name
    if not name or name in {".", ".."}:
        raise ValueError("filename must be a safe basename")
    return name


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_positive_int(name: str, value: Any) -> None:
    if not _is_int(value) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_repair_ratio(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= MAX_REPAIR_RATIO:
        raise ValueError(f"repair_ratio must be finite and between 0 and {MAX_REPAIR_RATIO}")
