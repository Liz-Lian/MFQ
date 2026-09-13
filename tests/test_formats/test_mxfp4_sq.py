from __future__ import annotations

import struct

import numpy as np
import pytest

from mfq.formats.compat import MXFP4_SQ_DTYPE, canonical_dtype
from mfq.formats.io import pack_tensor_payload, unpack_tensor_payload
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.formats.mxfp4_sq import (
    build_mxfp4_sq,
    concatenate_mxfp4_sq_rows,
    pack_mxfp4_sq,
    unpack_mxfp4_sq,
)


def _pack_bits(values: list[int], bits: int) -> bytes:
    packed = bytearray((len(values) * bits + 7) // 8)
    for index, value in enumerate(values):
        bit_offset = index * bits
        packed[bit_offset // 8] |= value << (bit_offset % 8) & 0xFF
        if bit_offset % 8 + bits > 8:
            packed[bit_offset // 8 + 1] |= value >> (8 - bit_offset % 8)
    return bytes(packed)


def _payload(
    bits: int,
    *,
    legacy_sq3: bool = False,
    base: int = 120,
) -> bytes:
    outputs, width = 3, 64
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


def _adaptive_payload() -> bytes:
    tensor = build_mxfp4_sq(
        row_q_bits=(1, 2, 3, 4),
        input_size=64,
        matrix_scale_base=120,
        row_symbols=(
            bytes(8),
            bytes(16),
            bytes(24),
            bytes(range(32)),
        ),
        sq_block_selectors=np.asarray(
            [[0, 1], [1, 0], [1, 1]], dtype=np.uint8
        ),
        sq_state_scales=np.arange(24, dtype=np.uint8).reshape(3, 8) & 3,
        sq_state_palettes=np.arange(24, dtype=np.uint8).reshape(3, 8),
        sq4_native_scales=np.asarray([[119, 121]], dtype=np.uint8),
    )
    return tensor.payload


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


def test_adaptive_mxfp4_sq_uses_two_bit_q_descriptors() -> None:
    tensor = unpack_mxfp4_sq(_adaptive_payload())
    assert tensor.format_version == 2
    assert tensor.bits == 0
    assert tensor.row_q_bits == (1, 2, 3, 4)
    assert tensor.profile == "SQ1/SQ2/SQ3/SQ4"
    assert tensor.descriptor.format_version == 2
    assert tensor.descriptor.distribution_entropy == pytest.approx(2.0)
    assert pack_mxfp4_sq(tensor) == tensor.payload


def test_adaptive_mxfp4_sq_rejects_nonzero_q_padding() -> None:
    tensor = build_mxfp4_sq(
        row_q_bits=(1, 4, 2),
        input_size=32,
        matrix_scale_base=120,
        row_symbols=(bytes(4), bytes(16), bytes(8)),
        sq_block_selectors=np.zeros((2, 1), dtype=np.uint8),
        sq_state_scales=np.zeros((2, 8), dtype=np.uint8),
        sq_state_palettes=np.zeros((2, 8), dtype=np.uint8),
        sq4_native_scales=np.asarray([[120]], dtype=np.uint8),
    )
    damaged = bytearray(tensor.payload)
    damaged[24] |= 0xC0
    with pytest.raises(ValueError, match="padding"):
        unpack_mxfp4_sq(damaged)


def test_adaptive_mxfp4_sq_validates_profile_specific_rows() -> None:
    with pytest.raises(ValueError, match="row 1"):
        build_mxfp4_sq(
            row_q_bits=(1, 4),
            input_size=32,
            matrix_scale_base=120,
            row_symbols=(bytes(4), bytes(15)),
            sq_block_selectors=np.zeros((1, 1), dtype=np.uint8),
            sq_state_scales=np.zeros((1, 8), dtype=np.uint8),
            sq_state_palettes=np.zeros((1, 8), dtype=np.uint8),
            sq4_native_scales=np.asarray([[120]], dtype=np.uint8),
        )


def test_adaptive_mxfp4_sq_rejects_e8m0_nan_native_scale() -> None:
    with pytest.raises(ValueError, match="E8M0"):
        build_mxfp4_sq(
            row_q_bits=(4,),
            input_size=32,
            matrix_scale_base=120,
            row_symbols=(bytes(16),),
            sq_block_selectors=np.empty((0, 1), dtype=np.uint8),
            sq_state_scales=np.empty((0, 8), dtype=np.uint8),
            sq_state_palettes=np.empty((0, 8), dtype=np.uint8),
            sq4_native_scales=np.asarray([[255]], dtype=np.uint8),
        )


def test_mxfp4_sq_row_concatenation_promotes_legacy_and_preserves_q4() -> None:
    adaptive = unpack_mxfp4_sq(_adaptive_payload())
    legacy = unpack_mxfp4_sq(_payload(2))

    merged = concatenate_mxfp4_sq_rows(
        (legacy, adaptive),
        row_indices=(6, 0, 4, 5, 3),
    )

    assert merged.format_version == 2
    assert merged.row_q_bits == (4, 2, 2, 3, 1)
    assert merged.shape == (5, 64)


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


def test_mfe_coalesces_sq_profiles_into_one_adaptive_pool() -> None:
    sq2 = unpack_mxfp4_sq(_payload(2))
    sq3 = unpack_mxfp4_sq(_payload(3))
    tensor = MfeTensor(
        shape=(2, 3, 64),
        pools=(
            MfePool(np.asarray([1], dtype=np.int32), sq2),
            MfePool(np.asarray([0], dtype=np.int32), sq3),
        ),
    )
    dtype, payload = pack_tensor_payload(tensor)
    assert dtype == "MFE"
    assert payload.count(MXFP4_SQ_DTYPE.encode()) == 1
    restored = unpack_tensor_payload(dtype, payload)
    assert restored.expert_profiles == ("MXFP4-SQ", "MXFP4-SQ")
    assert len(restored.pools) == 1
    np.testing.assert_array_equal(restored.pools[0].expert_ids, [0, 1])
    assert restored.pools[0].tensor.row_q_bits == (3, 3, 3, 2, 2, 2)
    assert pack_tensor_payload(restored) == (dtype, payload)


def test_mfe_keeps_distinct_mxfp4_sq_matrix_scale_bases_separate() -> None:
    tensor = MfeTensor(
        shape=(2, 3, 64),
        pools=(
            MfePool(
                np.asarray([0], dtype=np.int32),
                unpack_mxfp4_sq(_payload(2, base=119)),
            ),
            MfePool(
                np.asarray([1], dtype=np.int32),
                unpack_mxfp4_sq(_payload(3, base=120)),
            ),
        ),
    )

    restored = unpack_tensor_payload(*pack_tensor_payload(tensor))

    assert len(restored.pools) == 2
    assert tuple(
        pool.tensor.matrix_scale_base for pool in restored.pools
    ) == (119, 120)


def test_cuda_mfe_upload_retains_adaptive_mxfp4_sq_row_metadata() -> None:
    pytest.importorskip("torch")
    from mfq.kernels.cuda.moe import to_gpu

    tensor = unpack_mxfp4_sq(_adaptive_payload())
    weight = to_gpu(
        MfeTensor(
            shape=(2, 2, 64),
            pools=(MfePool(np.asarray([0, 1], dtype=np.int32), tensor),),
        ),
        device="cpu",
    )
    pool = weight.pools[0]
    assert pool.family == "mxfp4_sq"
    assert pool.weight["row_q"].tolist() == [1, 2, 3, 4]
    assert pool.weight["row_symbol_byte_offsets"].tolist() == [0, 8, 24, 48]
    assert pool.weight["row_auxiliary"].tolist() == [0, 1, 2, 0]
