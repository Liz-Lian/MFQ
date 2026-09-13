from __future__ import annotations

import numpy as np
import pytest

from mfq.formats.fp8_sq import decode_fp8_sq_codes, fp8_sq_scale_bytes
from mfq.quantize.fp8_sq import (
    E4M3_LEGAL_CODES,
    E4M3_VALUES,
    fit_e4m3_sq_palette,
    quantize_fp8_128_sq,
    quantize_mxfp8_sq,
)


def _source(rows: int, columns: int) -> np.ndarray:
    rng = np.random.default_rng(20260912 + rows * 17 + columns)
    return rng.choice(E4M3_LEGAL_CODES, size=(rows, columns)).astype(np.uint8)


@pytest.mark.parametrize("bits", range(1, 8))
def test_palette_solver_is_deterministic_and_has_exact_cardinality(bits: int) -> None:
    source = _source(5, 257)
    weights = np.linspace(0.1, 1.0, source.size).reshape(source.shape)

    first = fit_e4m3_sq_palette(source, weights, bits)
    second = fit_e4m3_sq_palette(source, weights, bits)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (1 << bits,)
    assert np.unique(first).size == 1 << bits
    assert not np.any((first & 0x7F) == 0x7F)


def test_more_scalar_bits_do_not_increase_optimal_reconstruction_sse() -> None:
    source = _source(8, 253)
    scales = np.ones((1, 2), dtype=np.float32)
    source_values = E4M3_VALUES[source]
    errors = []
    for bits in range(1, 9):
        tensor = quantize_fp8_128_sq(
            source,
            scales,
            row_q_bits=bits,
        )
        reconstruction = E4M3_VALUES[decode_fp8_sq_codes(tensor)]
        errors.append(float(np.square(source_values - reconstruction).sum()))

    assert all(right <= left + 1e-12 for left, right in zip(errors, errors[1:]))
    assert errors[-1] == 0.0


def test_q8_is_byte_exact_for_both_scale_contracts() -> None:
    source = _source(2, 256)
    source[0, :2] = (0x00, 0x80)
    mx_scales = np.asarray([[119, 137]], dtype=np.uint8)
    fp_scales = np.asarray([[0x3F80, 0x4000]], dtype=np.uint16)

    mx = quantize_mxfp8_sq(
        source,
        mx_scales,
        block_shape=(128, 128),
        row_q_bits=8,
    )
    fp = quantize_fp8_128_sq(
        source,
        fp_scales,
        scale_dtype="BF16",
        row_q_bits=8,
    )

    np.testing.assert_array_equal(decode_fp8_sq_codes(mx), source)
    np.testing.assert_array_equal(decode_fp8_sq_codes(fp), source)
    assert fp8_sq_scale_bytes(mx) == mx_scales.tobytes()
    assert fp8_sq_scale_bytes(fp) == fp_scales.tobytes()


@pytest.mark.parametrize(
    "row_q_bits",
    (
        (1, 2),
        (0, 2, 3),
        (1, 2, 9),
        (1, 2, 257),
        (1.0, 2.0, 3.0),
    ),
)
def test_quantizer_rejects_invalid_q_maps(row_q_bits) -> None:
    with pytest.raises(ValueError, match="q"):
        quantize_mxfp8_sq(
            _source(3, 128),
            np.asarray([[127]], dtype=np.uint8),
            block_shape=(128, 128),
            row_q_bits=row_q_bits,
        )


def test_quantizer_rejects_e4m3_nan_and_invalid_scales() -> None:
    source = _source(2, 128)
    source[0, 0] = 0x7F
    with pytest.raises(ValueError, match="NaN"):
        quantize_mxfp8_sq(
            source,
            np.asarray([[127]], dtype=np.uint8),
            block_shape=(128, 128),
        )

    source[0, 0] = 0
    with pytest.raises(ValueError, match="E8M0"):
        quantize_mxfp8_sq(
            source,
            np.asarray([[255]], dtype=np.uint8),
            block_shape=(128, 128),
        )
    with pytest.raises(ValueError, match="positive and finite"):
        quantize_fp8_128_sq(
            source,
            np.asarray([[0.0]], dtype=np.float32),
        )


def test_neuron_importance_changes_the_shared_palette_objective() -> None:
    source = np.vstack(
        (
            np.full(128, 0x30, dtype=np.uint8),
            np.full(128, 0x38, dtype=np.uint8),
            np.full(128, 0x40, dtype=np.uint8),
        )
    )
    scales = np.asarray([[1.0]], dtype=np.float32)
    ordinary = quantize_fp8_128_sq(source, scales, row_q_bits=1)
    weighted = quantize_fp8_128_sq(
        source,
        scales,
        row_q_bits=1,
        neuron_importance=np.asarray([1.0, 1000.0, 1.0]),
    )

    assert ordinary.payload != weighted.payload
