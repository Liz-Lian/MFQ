"""Unified MXFP4-SQ tensor container.

SQ2 and SQ3 are payload profiles of one public format.  Their symbol width is
self-described by the wire magic and is not part of the MFQ dtype name.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass


_HEADER = struct.Struct("<4sBBHQQ")
_CURRENT_MAGIC = {2: b"SQ2\0", 3: b"SQ3\0"}
_LEGACY_SQ3_MAGIC = b"SQ31"


@dataclass(frozen=True)
class Mxfp4SqTensor:
    """Validated, self-describing MXFP4-SQ payload."""

    payload: bytes
    bits: int
    output_size: int
    input_size: int
    matrix_scale_base: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_size, self.input_size

    @property
    def profile(self) -> str:
        return f"SQ{self.bits}"


def _payload_nbytes(bits: int, output_size: int, input_size: int) -> int:
    weights = output_size * input_size
    symbols = (weights * bits + 7) // 8
    selectors = (weights // 32 + 7) // 8
    return _HEADER.size + symbols + selectors + output_size * 2 + output_size * 5


def unpack_mxfp4_sq(blob: bytes | memoryview) -> Mxfp4SqTensor:
    """Validate and retain an SQ2/SQ3 payload under one tensor type."""

    payload = bytes(blob)
    if len(payload) < _HEADER.size:
        raise ValueError("truncated MXFP4-SQ header")
    magic, version, base, reserved, output_size, input_size = _HEADER.unpack_from(
        payload
    )
    if magic == b"SQ2\0":
        bits = 2
    elif magic in {b"SQ3\0", _LEGACY_SQ3_MAGIC}:
        bits = 3
    else:
        raise ValueError(f"invalid MXFP4-SQ payload magic: {magic!r}")
    if version != 1 or reserved != 0:
        raise ValueError("invalid MXFP4-SQ payload version or reserved field")
    if not 0 <= base <= 251:
        raise ValueError("invalid MXFP4-SQ E8M0 scale base")
    if not 0 < output_size <= 0x7FFFFFFF:
        raise ValueError("invalid MXFP4-SQ output size")
    if not 0 < input_size <= 0x7FFFFFFF or input_size % 32:
        raise ValueError("MXFP4-SQ input size must be a positive multiple of 32")
    expected = _payload_nbytes(bits, output_size, input_size)
    if len(payload) != expected:
        raise ValueError(
            f"MXFP4-SQ payload length mismatch: expected {expected}, got {len(payload)}"
        )
    return Mxfp4SqTensor(
        payload=payload,
        bits=bits,
        output_size=output_size,
        input_size=input_size,
        matrix_scale_base=base,
    )


def pack_mxfp4_sq(tensor: Mxfp4SqTensor) -> bytes:
    """Serialize one validated tensor using the backend-neutral profile magic."""

    parsed = unpack_mxfp4_sq(tensor.payload)
    if (
        parsed.bits != tensor.bits
        or parsed.output_size != tensor.output_size
        or parsed.input_size != tensor.input_size
        or parsed.matrix_scale_base != tensor.matrix_scale_base
    ):
        raise ValueError("MXFP4-SQ tensor metadata does not match its payload")
    if parsed.payload[:4] == _LEGACY_SQ3_MAGIC:
        return _CURRENT_MAGIC[3] + parsed.payload[4:]
    return parsed.payload


__all__ = ["Mxfp4SqTensor", "pack_mxfp4_sq", "unpack_mxfp4_sq"]
