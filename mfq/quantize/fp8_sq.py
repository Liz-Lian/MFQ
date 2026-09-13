"""Production SQ1--SQ8 quantizers for native E4M3 source weights.

The scalar solver is shared by MXFP8-SQ and FP8-128SQ, while their scale
contracts remain separate.  Palettes contain raw, legal E4M3 codes.  SQ8 is
the byte-exact identity endpoint and therefore bypasses palette lookup.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from mfq.formats.fp8_sq import (
    FP8_128SQ_DTYPE,
    MXFP8_SQ_DTYPE,
    Fp8SqTensor,
    build_fp8_sq,
)


def _decode_e4m3_code(raw: int) -> float:
    magnitude = raw & 0x7F
    if magnitude == 0x7F:
        return float("nan")
    exponent = (magnitude >> 3) & 0x0F
    mantissa = magnitude & 0x07
    if exponent == 0:
        value = mantissa * 2.0**-9
    else:
        value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return -value if raw & 0x80 else value


E4M3_VALUES = np.asarray(
    [_decode_e4m3_code(raw) for raw in range(256)],
    dtype=np.float64,
)
E4M3_LEGAL_CODES = np.asarray(
    [raw for raw in range(256) if (raw & 0x7F) != 0x7F],
    dtype=np.uint8,
)

# Collapse the two signed-zero codes for SQ1--SQ7.  SQ8 retains the original
# raw byte and can therefore preserve negative zero if it exists in a source.
_value_to_code: dict[float, int] = {}
for _raw in E4M3_LEGAL_CODES.tolist():
    _value_to_code.setdefault(float(E4M3_VALUES[_raw]), int(_raw))
_ordered = sorted(_value_to_code.items())
E4M3_CANONICAL_VALUES = np.asarray([item[0] for item in _ordered], dtype=np.float64)
E4M3_CANONICAL_CODES = np.asarray([item[1] for item in _ordered], dtype=np.uint8)
_RAW_TO_LEVEL = np.full(256, -1, dtype=np.int16)
for _raw in E4M3_LEGAL_CODES.tolist():
    _RAW_TO_LEVEL[_raw] = int(
        np.searchsorted(E4M3_CANONICAL_VALUES, E4M3_VALUES[_raw])
    )


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    source = np.asarray(values, dtype=np.uint8).reshape(-1)
    output = bytearray((source.size * bits + 7) // 8)
    for index, raw_value in enumerate(source.tolist()):
        value = int(raw_value)
        bit = index * bits
        output[bit // 8] |= (value << (bit % 8)) & 0xFF
        if bit % 8 + bits > 8:
            output[bit // 8 + 1] |= value >> (8 - bit % 8)
    return bytes(output)


def _normalize_row_q_bits(
    row_q_bits: int | Sequence[int] | np.ndarray,
    rows: int,
) -> np.ndarray:
    if isinstance(row_q_bits, (int, np.integer)):
        q = np.full(rows, int(row_q_bits), dtype=np.uint8)
    else:
        source = np.asarray(row_q_bits)
        if source.ndim != 1 or source.shape[0] != rows:
            raise ValueError("FP8-SQ q map must contain one value per neuron")
        if not np.issubdtype(source.dtype, np.integer):
            raise ValueError("FP8-SQ q descriptors must be integers")
        if np.any(source < 1) or np.any(source > 8):
            raise ValueError("FP8-SQ q descriptors must be in [1,8]")
        q = source.astype(np.uint8, copy=False)
    if np.any(q < 1) or np.any(q > 8):
        raise ValueError("FP8-SQ q descriptors must be in [1,8]")
    return np.ascontiguousarray(q)


def _default_palette(bits: int) -> np.ndarray:
    count = 1 << bits
    indices = np.rint(
        np.linspace(0, E4M3_CANONICAL_CODES.size - 1, count)
    ).astype(np.int64)
    if np.unique(indices).size != count:  # Defensive; current E4M3 lattice cannot hit this.
        indices = np.arange(count, dtype=np.int64)
    return E4M3_CANONICAL_CODES[indices]


def _segment_tables(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact contiguous-cluster SSE and source-medoid indices."""

    count = values.size
    prefix_w = np.concatenate((np.zeros(1), np.cumsum(weights)))
    prefix_x = np.concatenate((np.zeros(1), np.cumsum(weights * values)))
    prefix_x2 = np.concatenate((np.zeros(1), np.cumsum(weights * values * values)))
    costs = np.full((count, count), np.inf, dtype=np.float64)
    representatives = np.zeros((count, count), dtype=np.int16)
    for begin in range(count):
        ends = np.arange(begin + 1, count + 1)
        total_w = prefix_w[ends] - prefix_w[begin]
        total_x = prefix_x[ends] - prefix_x[begin]
        mean = total_x / total_w
        insertion = np.searchsorted(values, mean)
        insertion = np.clip(insertion, begin, ends - 1)
        left = np.maximum(begin, insertion - 1)
        right = insertion
        left_error = np.abs(values[left] - mean)
        right_error = np.abs(values[right] - mean)
        chosen = np.where(left_error <= right_error, left, right)
        reconstruction = values[chosen]
        costs[begin, begin:] = (
            prefix_x2[ends]
            - prefix_x2[begin]
            - 2.0 * reconstruction * total_x
            + reconstruction * reconstruction * total_w
        )
        representatives[begin, begin:] = chosen.astype(np.int16)
    return costs, representatives


def fit_e4m3_sq_palette(
    encoded: np.ndarray,
    element_weight: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Fit an exact one-dimensional weighted E4M3 scalar palette.

    Dynamic programming partitions the ordered source levels into contiguous
    clusters.  Every representative is itself a legal source E4M3 level, so
    the resulting palette is native by construction.
    """

    if bits < 1 or bits > 7:
        raise ValueError("E4M3 palette fitting supports SQ1 through SQ7")
    raw = np.asarray(encoded, dtype=np.uint8)
    weight = np.asarray(element_weight, dtype=np.float64)
    if raw.shape != weight.shape or raw.size == 0:
        raise ValueError("E4M3 palette fitting needs equally shaped non-empty arrays")
    if np.any((raw & 0x7F) == 0x7F):
        raise ValueError("source contains an E4M3 NaN code")
    if np.any(weight < 0) or not np.isfinite(weight).all() or not np.any(weight > 0):
        raise ValueError("E4M3 palette weights must be finite, non-negative, and nonzero")

    level_weight = np.bincount(
        _RAW_TO_LEVEL[raw].reshape(-1),
        weights=weight.reshape(-1),
        minlength=E4M3_CANONICAL_CODES.size,
    )
    active = np.flatnonzero(level_weight > 0)
    palette_size = 1 << bits
    if active.size <= palette_size:
        selected = active.tolist()
    else:
        values = E4M3_CANONICAL_VALUES[active]
        weights = level_weight[active]
        costs, representatives = _segment_tables(values, weights)
        count = active.size
        previous = np.full(count + 1, np.inf, dtype=np.float64)
        previous[0] = 0.0
        backtrack = np.full((palette_size + 1, count + 1), -1, dtype=np.int16)
        for groups in range(1, palette_size + 1):
            current = np.full(count + 1, np.inf, dtype=np.float64)
            for end in range(groups, count + 1):
                begins = np.arange(groups - 1, end)
                candidates = previous[begins] + costs[begins, end - 1]
                choice = int(np.argmin(candidates))
                current[end] = candidates[choice]
                backtrack[groups, end] = int(begins[choice])
            previous = current
        selected = []
        end = count
        for groups in range(palette_size, 0, -1):
            begin = int(backtrack[groups, end])
            representative = int(representatives[begin, end - 1])
            selected.append(int(active[representative]))
            end = begin
        selected.reverse()

    selected_set = set(selected)
    if len(selected) < palette_size:
        remaining = [
            index
            for index in range(E4M3_CANONICAL_CODES.size)
            if index not in selected_set
        ]
        needed = palette_size - len(selected)
        fill_indices = np.rint(
            np.linspace(0, len(remaining) - 1, needed)
        ).astype(np.int64)
        selected.extend(remaining[int(index)] for index in fill_indices)
    selected = sorted(selected)
    result = E4M3_CANONICAL_CODES[np.asarray(selected, dtype=np.int64)]
    if result.size != palette_size or np.unique(result).size != palette_size:
        raise RuntimeError("internal E4M3 palette cardinality error")
    return np.ascontiguousarray(result)


def _palette_region() -> np.ndarray:
    region = np.zeros(256, dtype=np.uint8)
    for bits in range(1, 8):
        start = (1 << bits) - 2
        region[start : start + (1 << bits)] = _default_palette(bits)
    return region


def _scale_multipliers(
    scale_values: np.ndarray,
    shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> np.ndarray:
    rows, columns = shape
    block_rows, block_columns = block_shape
    expanded = np.repeat(
        np.repeat(scale_values, block_rows, axis=0),
        block_columns,
        axis=1,
    )[:rows, :columns]
    maximum = float(np.max(expanded))
    if not maximum > 0 or not np.isfinite(maximum):
        raise ValueError("FP8-SQ scales must be positive and finite")
    return np.square(expanded / maximum, dtype=np.float64)


def _normalize_mxfp8_scales(
    scales: np.ndarray,
    expected_shape: tuple[int, int],
) -> tuple[bytes, np.ndarray]:
    source = np.asarray(scales)
    if source.shape != expected_shape or not np.issubdtype(source.dtype, np.integer):
        raise ValueError(f"MXFP8-SQ E8M0 scales must have shape {expected_shape}")
    if np.any(source < 0) or np.any(source > 254):
        raise ValueError("MXFP8-SQ contains an E8M0 NaN or out-of-range scale")
    raw = np.ascontiguousarray(source, dtype=np.uint8)
    values = np.ldexp(np.ones(raw.shape, dtype=np.float64), raw.astype(np.int16) - 127)
    return raw.tobytes(), values


def _bf16_to_float32(raw: np.ndarray) -> np.ndarray:
    words = np.ascontiguousarray(raw, dtype="<u2")
    return (words.astype(np.uint32) << 16).view(np.float32)


def _normalize_fp8_128_scales(
    scales: np.ndarray,
    expected_shape: tuple[int, int],
    scale_dtype: str | None,
) -> tuple[str, bytes, np.ndarray]:
    source = np.asarray(scales)
    inferred = scale_dtype
    if inferred is None:
        if type(scales).__name__ == "BFloat16Array":
            inferred = "BF16"
        elif source.dtype == np.dtype(np.float16):
            inferred = "F16"
        elif source.dtype == np.dtype(np.float32):
            inferred = "F32"
    if inferred not in {"BF16", "F16", "F32"}:
        raise ValueError("FP8-128SQ scale_dtype must be BF16, F16, or F32")
    if source.shape != expected_shape:
        raise ValueError(f"FP8-128SQ scales must have shape {expected_shape}")
    if inferred == "BF16":
        if source.dtype.itemsize != 2:
            raise ValueError("FP8-128SQ BF16 scales require raw 16-bit storage")
        raw = np.ascontiguousarray(source, dtype="<u2")
        values = _bf16_to_float32(raw).astype(np.float64)
    elif inferred == "F16":
        raw = np.ascontiguousarray(source, dtype="<f2")
        values = raw.astype(np.float64)
    else:
        raw = np.ascontiguousarray(source, dtype="<f4")
        values = raw.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("FP8-128SQ scale multipliers must be positive and finite")
    return inferred, raw.tobytes(), values


def _quantize_fp8_sq(
    *,
    dtype: str,
    encoded: np.ndarray,
    scale_bytes: bytes,
    scale_values: np.ndarray,
    scale_dtype: str,
    block_shape: tuple[int, int],
    row_q_bits: int | Sequence[int] | np.ndarray,
    neuron_importance: np.ndarray | None,
) -> Fp8SqTensor:
    raw = np.asarray(encoded)
    if raw.ndim != 2 or raw.dtype.itemsize != 1:
        raise ValueError("FP8-SQ source values must be a rank-2 byte matrix")
    raw = np.ascontiguousarray(raw).view(np.uint8)
    if np.any((raw & 0x7F) == 0x7F):
        raise ValueError("FP8-SQ source contains an E4M3 NaN code")
    rows, columns = raw.shape
    q = _normalize_row_q_bits(row_q_bits, rows)
    scale_shape = tuple(int(value) for value in scale_values.shape)
    element_weight = _scale_multipliers(
        np.asarray(scale_values, dtype=np.float64),
        (rows, columns),
        block_shape,
    )
    if neuron_importance is not None:
        importance = np.asarray(neuron_importance, dtype=np.float64)
        if importance.shape != (rows,):
            raise ValueError("FP8-SQ neuron importance must have one value per row")
        if np.any(importance < 0) or not np.isfinite(importance).all():
            raise ValueError("FP8-SQ neuron importance must be finite and non-negative")
        if not np.any(importance > 0):
            raise ValueError("FP8-SQ neuron importance cannot be all zero")
        element_weight *= importance[:, None] / float(np.max(importance))

    palettes = _palette_region()
    for bits in range(1, 8):
        rows_for_q = q == bits
        if not np.any(rows_for_q):
            continue
        cohort_weight = element_weight[rows_for_q]
        # A valid NAQ map may deliberately assign zero importance to every
        # row in one q cohort.  The codec still needs a deterministic palette;
        # fall back to the source-scale objective for that cohort instead of
        # rejecting an otherwise valid tensor.
        if not np.any(cohort_weight > 0):
            cohort_weight = _scale_multipliers(
                np.asarray(scale_values, dtype=np.float64),
                (rows, columns),
                block_shape,
            )[rows_for_q]
        fitted = fit_e4m3_sq_palette(
            raw[rows_for_q],
            cohort_weight,
            bits,
        )
        start = (1 << bits) - 2
        palettes[start : start + (1 << bits)] = fitted

    row_streams: list[bytes] = []
    for row, raw_bits in enumerate(q.tolist()):
        bits = int(raw_bits)
        if bits == 8:
            row_streams.append(raw[row].tobytes())
            continue
        start = (1 << bits) - 2
        palette = palettes[start : start + (1 << bits)]
        distances = np.square(
            E4M3_VALUES[:, None] - E4M3_VALUES[palette][None, :]
        )
        nearest = np.argmin(distances, axis=1).astype(np.uint8)
        row_streams.append(_pack_bits(nearest[raw[row]], bits))

    return build_fp8_sq(
        dtype=dtype,
        row_q_bits=q,
        input_size=columns,
        block_shape=block_shape,
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
        palettes=palettes,
        row_symbols=row_streams,
        scale_bytes=scale_bytes,
    )


def quantize_mxfp8_sq(
    encoded: np.ndarray,
    scales: np.ndarray,
    *,
    block_shape: tuple[int, int],
    row_q_bits: int | Sequence[int] | np.ndarray = 4,
    neuron_importance: np.ndarray | None = None,
) -> Fp8SqTensor:
    """Quantize native E4M3/E8M0 storage into MXFP8-SQ."""

    raw = np.asarray(encoded)
    if raw.ndim != 2:
        raise ValueError("MXFP8-SQ source values must be a matrix")
    rows, columns = raw.shape
    block_rows, block_columns = block_shape
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("MXFP8-SQ block dimensions must be positive")
    expected = (
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    )
    scale_bytes, scale_values = _normalize_mxfp8_scales(scales, expected)
    return _quantize_fp8_sq(
        dtype=MXFP8_SQ_DTYPE,
        encoded=raw,
        scale_bytes=scale_bytes,
        scale_values=scale_values,
        scale_dtype="F8_E8M0",
        block_shape=block_shape,
        row_q_bits=row_q_bits,
        neuron_importance=neuron_importance,
    )


def quantize_fp8_128_sq(
    encoded: np.ndarray,
    scales: np.ndarray,
    *,
    row_q_bits: int | Sequence[int] | np.ndarray = 4,
    scale_dtype: str | None = None,
    neuron_importance: np.ndarray | None = None,
) -> Fp8SqTensor:
    """Quantize E4M3 plus 128x128 multiplier storage into FP8-128SQ."""

    raw = np.asarray(encoded)
    if raw.ndim != 2:
        raise ValueError("FP8-128SQ source values must be a matrix")
    expected = (
        (raw.shape[0] + 127) // 128,
        (raw.shape[1] + 127) // 128,
    )
    resolved_dtype, scale_bytes, scale_values = _normalize_fp8_128_scales(
        scales,
        expected,
        scale_dtype,
    )
    return _quantize_fp8_sq(
        dtype=FP8_128SQ_DTYPE,
        encoded=raw,
        scale_bytes=scale_bytes,
        scale_values=scale_values,
        scale_dtype=resolved_dtype,
        block_shape=(128, 128),
        row_q_bits=row_q_bits,
        neuron_importance=neuron_importance,
    )


def write_mxfp8_sq_blob(
    output_path: str | Path,
    encoded: np.ndarray,
    scales: np.ndarray,
    *,
    block_shape: tuple[int, int],
    row_q_bits: int | Sequence[int] | np.ndarray = 4,
    neuron_importance: np.ndarray | None = None,
) -> int:
    """Quantize exact native MXFP8 storage and write one payload."""

    tensor = quantize_mxfp8_sq(
        encoded,
        scales,
        block_shape=block_shape,
        row_q_bits=row_q_bits,
        neuron_importance=neuron_importance,
    )
    target = Path(output_path)
    target.write_bytes(tensor.payload)
    return target.stat().st_size


def write_fp8_128_sq_blob(
    output_path: str | Path,
    encoded: np.ndarray,
    scales: np.ndarray,
    *,
    row_q_bits: int | Sequence[int] | np.ndarray = 4,
    scale_dtype: str | None = None,
    neuron_importance: np.ndarray | None = None,
) -> int:
    """Quantize exact block-128 E4M3 storage and write one payload."""

    tensor = quantize_fp8_128_sq(
        encoded,
        scales,
        row_q_bits=row_q_bits,
        scale_dtype=scale_dtype,
        neuron_importance=neuron_importance,
    )
    target = Path(output_path)
    target.write_bytes(tensor.payload)
    return target.stat().st_size


__all__ = [
    "E4M3_CANONICAL_CODES",
    "E4M3_CANONICAL_VALUES",
    "E4M3_LEGAL_CODES",
    "E4M3_VALUES",
    "fit_e4m3_sq_palette",
    "quantize_fp8_128_sq",
    "quantize_mxfp8_sq",
    "write_fp8_128_sq_blob",
    "write_mxfp8_sq_blob",
]
