from __future__ import annotations

import struct

import numpy as np
import pytest

from mfq.formats.compat import MXFP4_SQ_DTYPE, canonical_dtype
from mfq.formats.io import pack_tensor_payload, unpack_tensor_payload
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mxfp4_sq import pack_mxfp4_sq, unpack_mxfp4_sq


def _pack_bits(values: list[int], bits: int) -> bytes:
    packed = bytearray((len(values) * bits + 7) // 8)
    for index, value in enumerate(values):
        bit_offset = index * bits
        packed[bit_offset // 8] |= value << (bit_offset % 8) & 0xFF
        if bit_offset % 8 + bits > 8:
            packed[bit_offset // 8 + 1] |= value >> (8 - bit_offset % 8)
    return bytes(packed)


def _payload(bits: int, *, legacy_sq3: bool = False) -> bytes:
    outputs, width, base = 3, 64, 120
    magic = b"SQ31" if legacy_sq3 else f"SQ{bits}\0".encode()
    symbols = _pack_bits(
        [index % (1 << bits) for index in range(outputs * width)],
        bits,
    )
    selectors = _pack_bits(
        [index & 1 for index in range(outputs * width // 32)],
        1,
    )
    scales = _pack_bits([index & 3 for index in range(outputs * 8)], 2)
    palettes = _pack_bits([index & 31 for index in range(outputs * 8)], 5)
    return (
        struct.pack("<4sBBHQQ", magic, 1, base, 0, outputs, width)
        + symbols
        + selectors
        + scales
        + palettes
    )


@pytest.mark.parametrize("bits", [2, 3])
def test_mxfp4_sq_profiles_share_one_public_dtype(bits: int) -> None:
    source = _payload(bits)
    tensor = unpack_tensor_payload(f"MXFP4-SQ{bits}", source)
    assert tensor.bits == bits
    assert tensor.shape == (3, 64)
    assert tensor.matrix_scale_base == 120

    dtype, encoded = pack_tensor_payload(tensor)
    assert dtype == MXFP4_SQ_DTYPE
    assert encoded == source


def test_legacy_metal_sq3_magic_is_read_only_compatibility() -> None:
    tensor = unpack_mxfp4_sq(_payload(3, legacy_sq3=True))
    encoded = pack_mxfp4_sq(tensor)
    assert tensor.bits == 3
    assert encoded[:4] == b"SQ3\0"


@pytest.mark.parametrize("stored", ["MXFP4-SQ", "MXFP4-SQ2", "MXFP4-SQ3"])
def test_mxfp4_sq_dtype_compatibility(stored: str) -> None:
    assert canonical_dtype(stored) == MXFP4_SQ_DTYPE


@pytest.mark.parametrize(
    "mutate",
    [
        lambda blob: b"SQ4\0" + blob[4:],
        lambda blob: blob[:4] + b"\x02" + blob[5:],
        lambda blob: blob[:6] + b"\x01\x00" + blob[8:],
        lambda blob: blob[:5] + b"\xfc" + blob[6:],
        lambda blob: blob[:8] + (0).to_bytes(8, "little") + blob[16:],
        lambda blob: blob[:16] + (33).to_bytes(8, "little") + blob[24:],
        lambda blob: blob[:8]
        + (0x80000000).to_bytes(8, "little")
        + blob[16:],
        lambda blob: blob[:-1],
        lambda blob: blob + b"\x00",
    ],
)
def test_mxfp4_sq_rejects_malformed_payloads(mutate) -> None:
    with pytest.raises(ValueError):
        unpack_mxfp4_sq(mutate(_payload(2)))


def test_mxfp4_sq_rejects_detached_metadata_that_disagrees_with_payload() -> None:
    tensor = unpack_mxfp4_sq(_payload(2))
    with pytest.raises(ValueError, match="metadata does not match"):
        pack_mxfp4_sq(
            type(tensor)(
                payload=tensor.payload,
                bits=3,
                output_size=tensor.output_size,
                input_size=tensor.input_size,
                matrix_scale_base=tensor.matrix_scale_base,
            )
        )


def test_mfe_serializes_sq_profiles_as_one_nested_dtype() -> None:
    sq2 = unpack_mxfp4_sq(_payload(2))
    sq3 = unpack_mxfp4_sq(_payload(3))
    tensor = MfeTensor(
        shape=(2, 3, 64),
        pools=(
            MfePool(np.asarray([0], dtype=np.int32), sq2),
            MfePool(np.asarray([1], dtype=np.int32), sq3),
        ),
    )
    dtype, payload = pack_tensor_payload(tensor)
    assert dtype == "MFE"
    assert payload.count(MXFP4_SQ_DTYPE.encode()) == 2
    restored = unpack_tensor_payload(dtype, payload)
    assert restored.expert_profiles == ("MXFP4-SQ2", "MXFP4-SQ3")
