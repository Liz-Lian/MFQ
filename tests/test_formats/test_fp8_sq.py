from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.fp8_sq import (
    FP8_128SQ_DTYPE,
    MXFP8_SQ_DTYPE,
    Fp8_128SqTensor,
    Mxfp8SqTensor,
    decode_fp8_sq_codes,
    fp8_sq_blob_nbytes,
    fp8_sq_palettes,
    fp8_sq_scale_bytes,
    pack_fp8_sq,
    parse_fp8_sq_layout,
    unpack_fp8_sq,
)
from mfq.formats.io import pack_tensor_payload, unpack_tensor_payload
from mfq.formats.mfe import MfePool, MfeTensor
from mfq.quantize.fp8_sq import (
    E4M3_LEGAL_CODES,
    quantize_fp8_128_sq,
    quantize_mxfp8_sq,
)


def _source(rows: int = 8, columns: int = 256) -> np.ndarray:
    rng = np.random.default_rng(20260911 + rows + columns)
    return rng.choice(E4M3_LEGAL_CODES, size=(rows, columns)).astype(np.uint8)


def test_mxfp8_sq_mixed_q_uses_three_bit_descriptors_and_exact_q8() -> None:
    source = _source()
    source[-1, :4] = (0x00, 0x80, 0x7E, 0xFE)
    scales = np.asarray([[121, 122, 123, 124, 125, 126, 127, 128]], dtype=np.uint8)
    q = np.arange(1, 9, dtype=np.uint8)

    tensor = quantize_mxfp8_sq(
        source,
        scales,
        block_shape=(32, 32),
        row_q_bits=q,
    )

    assert isinstance(tensor, Mxfp8SqTensor)
    assert tensor.dtype == MXFP8_SQ_DTYPE
    assert tensor.row_q_bits == tuple(range(1, 9))
    assert tensor.profile == "SQ1/SQ2/SQ3/SQ4/SQ5/SQ6/SQ7/SQ8"
    assert tensor.descriptor.distribution_entropy == pytest.approx(3.0)
    np.testing.assert_array_equal(decode_fp8_sq_codes(tensor)[-1], source[-1])
    assert fp8_sq_scale_bytes(tensor) == scales.tobytes()
    assert pack_fp8_sq(tensor) == tensor.payload
    assert len(tensor.payload) == fp8_sq_blob_nbytes(
        MXFP8_SQ_DTYPE,
        8,
        256,
        q,
        scale_dtype="F8_E8M0",
        scale_shape=(1, 8),
        block_shape=(32, 32),
    )


@pytest.mark.parametrize("scale_dtype", ("BF16", "F16", "F32"))
def test_fp8_128sq_preserves_its_high_precision_scale_payload(scale_dtype: str) -> None:
    source = _source(rows=3, columns=257)
    if scale_dtype == "BF16":
        scales = np.asarray([[0x3F80, 0x4000, 0x4040]], dtype=np.uint16)
    elif scale_dtype == "F16":
        scales = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float16)
    else:
        scales = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

    tensor = quantize_fp8_128_sq(
        source,
        scales,
        row_q_bits=(1, 7, 8),
        scale_dtype=scale_dtype,
    )

    assert isinstance(tensor, Fp8_128SqTensor)
    assert tensor.dtype == FP8_128SQ_DTYPE
    assert tensor.block_shape == (128, 128)
    assert tensor.scale_dtype == scale_dtype
    assert fp8_sq_scale_bytes(tensor) == scales.tobytes()
    np.testing.assert_array_equal(decode_fp8_sq_codes(tensor)[2], source[2])


def test_two_public_formats_cannot_be_reinterpreted_as_each_other() -> None:
    source = _source(rows=2, columns=128)
    mxfp8 = quantize_mxfp8_sq(
        source,
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
        row_q_bits=4,
    )
    fp8 = quantize_fp8_128_sq(
        source,
        np.asarray([[1.0]], dtype=np.float16),
        row_q_bits=4,
    )

    with pytest.raises(ValueError, match="header"):
        unpack_fp8_sq(FP8_128SQ_DTYPE, mxfp8.payload)
    with pytest.raises(ValueError, match="header"):
        unpack_fp8_sq(MXFP8_SQ_DTYPE, fp8.payload)


def test_public_mfq_io_retains_both_distinct_dtype_names() -> None:
    source = _source(rows=2, columns=128)
    tensors = (
        quantize_mxfp8_sq(
            source,
            np.asarray([[127]], dtype=np.uint8),
            block_shape=(128, 128),
        ),
        quantize_fp8_128_sq(
            source,
            np.asarray([[1.0]], dtype=np.float32),
        ),
    )

    for tensor in tensors:
        dtype, payload = pack_tensor_payload(tensor)
        assert dtype == tensor.dtype
        restored = unpack_tensor_payload(dtype, payload)
        assert type(restored) is type(tensor)
        assert pack_tensor_payload(restored) == (dtype, payload)


def test_mfe_accepts_native_output_fp8_sq_expert_pools() -> None:
    source = _source(rows=2, columns=128)
    mx = quantize_mxfp8_sq(
        source,
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
        row_q_bits=(3, 8),
    )
    block = quantize_fp8_128_sq(
        source,
        np.asarray([[1.0]], dtype=np.float16),
        row_q_bits=(2, 7),
    )
    tensor = MfeTensor(
        shape=(2, 2, 128),
        pools=(
            MfePool(np.asarray([0], dtype=np.int32), mx),
            MfePool(np.asarray([1], dtype=np.int32), block),
        ),
    )

    restored = unpack_tensor_payload(*pack_tensor_payload(tensor))

    assert isinstance(restored, MfeTensor)
    assert restored.expert_profiles == (MXFP8_SQ_DTYPE, FP8_128SQ_DTYPE)
    assert isinstance(restored.pools[0].tensor, Mxfp8SqTensor)
    assert isinstance(restored.pools[1].tensor, Fp8_128SqTensor)


def test_cuda_mfe_upload_keeps_both_fp8_sq_families_distinct() -> None:
    pytest.importorskip("torch")
    from mfq.kernels.cuda.moe import to_gpu

    source = _source(rows=2, columns=128)
    mx = quantize_mxfp8_sq(
        source,
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
        row_q_bits=(3, 8),
    )
    block = quantize_fp8_128_sq(
        source,
        np.asarray([[1.0]], dtype=np.float16),
        row_q_bits=(2, 7),
    )
    weight = to_gpu(
        MfeTensor(
            shape=(2, 2, 128),
            pools=(
                MfePool(np.asarray([0], dtype=np.int32), mx),
                MfePool(np.asarray([1], dtype=np.int32), block),
            ),
        ),
        device="cpu",
    )
    assert tuple(pool.family for pool in weight.pools) == (
        "mxfp8_sq",
        "fp8_128_sq",
    )
    assert weight.pools[0].weight["row_q"].tolist() == [3, 8]
    assert weight.pools[1].weight["row_q"].tolist() == [2, 7]


def test_all_stored_palette_entries_and_reconstructed_codes_are_legal() -> None:
    source = _source()
    tensor = quantize_mxfp8_sq(
        source,
        np.asarray([[127, 128]], dtype=np.uint8),
        block_shape=(128, 128),
        row_q_bits=np.arange(1, 9),
    )

    for bits, palette in fp8_sq_palettes(tensor).items():
        assert palette.size == 1 << bits
        assert np.unique(palette).size == palette.size
        assert not np.any((palette & 0x7F) == 0x7F)
    assert not np.any((decode_fp8_sq_codes(tensor) & 0x7F) == 0x7F)


def test_parser_rejects_nonzero_descriptor_padding() -> None:
    source = _source(rows=1, columns=128)
    tensor = quantize_mxfp8_sq(
        source,
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
        row_q_bits=8,
    )
    damaged = bytearray(tensor.payload)
    damaged[44] |= 0xF8

    with pytest.raises(ValueError, match="padding"):
        unpack_fp8_sq(MXFP8_SQ_DTYPE, damaged)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload[:-1],
        lambda payload: payload + b"\0",
        lambda payload: b"FAIL" + payload[4:],
        lambda payload: payload[:4] + b"\2" + payload[5:],
    ),
)
def test_parser_rejects_malformed_payloads(mutate) -> None:
    tensor = quantize_mxfp8_sq(
        _source(rows=1, columns=128),
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
    )
    with pytest.raises(ValueError):
        parse_fp8_sq_layout(MXFP8_SQ_DTYPE, mutate(tensor.payload))


def test_pack_rejects_detached_metadata_disagreement() -> None:
    tensor = quantize_mxfp8_sq(
        _source(rows=1, columns=128),
        np.asarray([[127]], dtype=np.uint8),
        block_shape=(128, 128),
    )
    damaged = Mxfp8SqTensor(
        payload=tensor.payload,
        output_size=tensor.output_size,
        input_size=tensor.input_size,
        scale_shape=tensor.scale_shape,
        block_shape=tensor.block_shape,
        row_q_bits=(5,),
    )
    with pytest.raises(ValueError, match="metadata does not match"):
        pack_fp8_sq(damaged)
