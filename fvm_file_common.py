"""FVM FILE MODE v1 transport and fixed 6 px physical-frame helpers."""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass

import numpy as np

from scripts.fvm0_rs_common import (
    CODED_RS_BYTES,
    LOGICAL_BYTES,
    FVM0RSConfig,
    RSDecodeResult,
    bytes_to_matrix,
    decode_rs_codewords,
    deinterleave,
    encode_logical,
    interleave,
    matrix_to_bytes,
)

FORMAT = "FVM_FILE_FRAME_V1"
MAGIC = b"F6F1"
VERSION = 1
WIDTH = 1920
HEIGHT = 1080
FPS = 30
CELL_SIZE = 6
SYMBOL_SIZE = 6400
REPAIR_RATIO = 0.10
TRANSPORT_HEADER = struct.Struct(">4sBIH")
TRANSPORT_BODY_BYTES = LOGICAL_BYTES - 4
MAX_PACKET_BYTES = TRANSPORT_BODY_BYTES - TRANSPORT_HEADER.size
PADDING_DOMAIN = b"FVM_FILE_PADDING_V1|"
RESERVED_DOMAIN = b"FVM_FILE_RESERVED_V1|"
PHYSICAL_CONFIG = FVM0RSConfig(
    width=WIDTH,
    height=HEIGHT,
    fps=FPS,
    cell_size=CELL_SIZE,
    frames=1,
    seed=1,
)


class TransportError(ValueError):
    """An untrusted FVM transport frame failed validation."""


@dataclass(frozen=True)
class TransportFrame:
    frame_index: int
    packet: bytes


@dataclass(frozen=True)
class PhysicalDecodeResult:
    transport: TransportFrame | None
    rs: RSDecodeResult
    transport_error: str | None


def _frame_domain(prefix: bytes, frame_index: int) -> bytes:
    if not isinstance(frame_index, int) or isinstance(frame_index, bool) or not 0 <= frame_index < 1 << 32:
        raise ValueError("frame_index must fit uint32")
    return prefix + frame_index.to_bytes(4, "big")


def wrap_packet(packet: bytes, frame_index: int) -> bytes:
    if not isinstance(packet, bytes) or not packet:
        raise ValueError("packet must be non-empty bytes")
    if len(packet) > MAX_PACKET_BYTES:
        raise ValueError(f"packet exceeds {MAX_PACKET_BYTES} bytes")
    domain = _frame_domain(PADDING_DOMAIN, frame_index)
    header = TRANSPORT_HEADER.pack(MAGIC, VERSION, frame_index, len(packet))
    padding = hashlib.shake_256(domain).digest(TRANSPORT_BODY_BYTES - len(header) - len(packet))
    body = header + packet + padding
    return body + (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "big")


def unwrap_packet(logical: bytes) -> TransportFrame:
    if len(logical) != LOGICAL_BYTES:
        raise TransportError(f"transport logical frame must be {LOGICAL_BYTES} bytes")
    body, checksum = logical[:-4], logical[-4:]
    if (zlib.crc32(body) & 0xFFFFFFFF) != int.from_bytes(checksum, "big"):
        raise TransportError("transport CRC32 mismatch")
    magic, version, frame_index, packet_length = TRANSPORT_HEADER.unpack_from(body)
    if magic != MAGIC:
        raise TransportError("invalid transport magic")
    if version != VERSION:
        raise TransportError("unsupported transport version")
    if not 1 <= packet_length <= MAX_PACKET_BYTES:
        raise TransportError("invalid transport packet length")
    packet_start = TRANSPORT_HEADER.size
    return TransportFrame(frame_index, body[packet_start:packet_start + packet_length])


def reserved_bytes(frame_index: int) -> bytes:
    domain = _frame_domain(RESERVED_DOMAIN, frame_index)
    return hashlib.shake_256(domain).digest(PHYSICAL_CONFIG.reserved_bytes_per_frame)


def encode_transport_physical(packet: bytes, frame_index: int) -> bytes:
    coded = interleave(encode_logical(wrap_packet(packet, frame_index)))
    return coded + reserved_bytes(frame_index)


def decode_transport_physical(physical: bytes) -> PhysicalDecodeResult:
    if len(physical) != PHYSICAL_CONFIG.physical_bytes_per_frame:
        raise ValueError("physical frame byte length is invalid")
    rs = decode_rs_codewords(deinterleave(physical[:CODED_RS_BYTES]))
    if rs.logical is None:
        return PhysicalDecodeResult(None, rs, "RS decode failed")
    try:
        transport = unwrap_packet(rs.logical)
    except TransportError as exc:
        return PhysicalDecodeResult(None, rs, str(exc))
    return PhysicalDecodeResult(transport, rs, None)


def physical_to_matrix(physical: bytes) -> np.ndarray:
    return bytes_to_matrix(physical, PHYSICAL_CONFIG)


def matrix_to_physical(bits: np.ndarray) -> bytes:
    return matrix_to_bytes(bits, PHYSICAL_CONFIG)
