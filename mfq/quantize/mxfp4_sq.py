"""Production quantizer for the unified native-output MXFP4-SQ format.

Every output neuron independently selects SQ1, SQ2, SQ3, or lossless SQ4 via
the format's two-bit row descriptor.  SQ1--SQ3 share one matrix-local E8M0
scale base and eight row-local states; SQ4 copies the source E2M1 nibbles and
block-32 E8M0 scales exactly.  The caller may provide a complete per-neuron
q map or use a uniform q preset.  No calibration framework or Core-v2 solver
abstraction is required.

The historical fixed-width SQ2/SQ3 encoder remains available for compatibility
with old payload writers, while :func:`quantize_mxfp4_sq` always emits the
canonical adaptive container.
"""

from __future__ import annotations

import itertools
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mfq.formats.mxfp4_sq import Mxfp4SqTensor, build_mxfp4_sq

try:  # Optional acceleration; the scalar fallback remains format-correct.
    from numba import njit, prange
except ImportError:  # pragma: no cover - exercised on minimal installations.
    njit = None
    prange = range


_HEADER = struct.Struct("<4sBBHQQ")
_NIBBLE_VALUES = np.asarray(
    (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
    dtype=np.float64,
)
_ORDERED_VALUES = np.asarray(
    (-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    dtype=np.float64,
)
_ORDERED_NIBBLES = np.asarray(
    (0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7),
    dtype=np.uint8,
)

# Frozen full-catalog IDs used by the Metal/CUDA decoders.  These are format
# constants selected during the original expert-0 design screen, not a tensor
# codebook and not model-specific metadata.
_SQ1_PALETTE_IDS = (
    12,
    26,
    38,
    11,
    13,
    58,
    22,
    48,
    25,
    23,
    67,
    21,
    37,
    24,
    75,
    20,
    8,
    59,
    68,
    9,
    76,
    7,
    19,
    36,
    46,
    55,
    63,
    70,
    35,
    47,
    56,
    45,
)
_SQ2_PALETTE_IDS = (
    120,
    127,
    187,
    504,
    512,
    518,
    547,
    548,
    558,
    562,
    592,
    767,
    806,
    833,
    966,
    967,
    192,
    1112,
    971,
    802,
    559,
    240,
    112,
    226,
    232,
    182,
    188,
    121,
    505,
    848,
    0,
    1,
)
_SQ3_PALETTE_IDS = (
    499,
    2407,
    523,
    5738,
    1196,
    2001,
    5497,
    4705,
    1087,
    2400,
    2402,
    1077,
    1078,
    4116,
    5503,
    2002,
    3726,
    2010,
    5498,
    4117,
    5552,
    4706,
    500,
    1483,
    3717,
    3727,
    1478,
    4818,
    5811,
    2003,
    2989,
    2035,
)


def _selected_palettes(levels: int, ids: tuple[int, ...]) -> np.ndarray:
    wanted = {value: index for index, value in enumerate(ids)}
    result = np.empty((len(ids), levels), dtype=np.uint8)
    remaining = len(ids)
    for index, combination in enumerate(
        itertools.combinations(range(len(_ORDERED_VALUES)), levels)
    ):
        target = wanted.get(index)
        if target is None:
            continue
        result[target] = _ORDERED_NIBBLES[np.asarray(combination, dtype=np.int64)]
        remaining -= 1
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("frozen MXFP4-SQ palette ID is outside its scalar catalog")
    return result


SQ1_PALETTE_NIBBLES = _selected_palettes(2, _SQ1_PALETTE_IDS)
SQ2_PALETTE_NIBBLES = _selected_palettes(4, _SQ2_PALETTE_IDS)
SQ3_PALETTE_NIBBLES = _selected_palettes(8, _SQ3_PALETTE_IDS)


@dataclass(frozen=True)
class Mxfp4SqEncoding:
    bits: int
    rows: int
    columns: int
    matrix_scale_base: int
    packed_symbols: np.ndarray
    packed_block_selectors: np.ndarray
    packed_state_scales: np.ndarray
    packed_state_palettes: np.ndarray

    @property
    def payload_nbytes(self) -> int:
        return (
            _HEADER.size
            + self.packed_symbols.nbytes
            + self.packed_block_selectors.nbytes
            + self.packed_state_scales.nbytes
            + self.packed_state_palettes.nbytes
        )

    def to_bytes(self) -> bytes:
        magic = b"SQ2\0" if self.bits == 2 else b"SQ31"
        return b"".join(
            (
                _HEADER.pack(
                    magic,
                    1,
                    self.matrix_scale_base,
                    0,
                    self.rows,
                    self.columns,
                ),
                np.ascontiguousarray(self.packed_symbols).tobytes(),
                np.ascontiguousarray(self.packed_block_selectors).tobytes(),
                np.ascontiguousarray(self.packed_state_scales).tobytes(),
                np.ascontiguousarray(self.packed_state_palettes).tobytes(),
            )
        )


def _legacy_mxfp4_sq_blob_nbytes(bits: int, rows: int, columns: int) -> int:
    if bits not in {2, 3}:
        raise ValueError("legacy MXFP4-SQ supports two- or three-bit symbols")
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("MXFP4-SQ requires a positive block-32 matrix")
    weights = rows * columns
    blocks = weights // 32
    return _HEADER.size + weights * bits // 8 + (blocks + 7) // 8 + rows * 2 + rows * 5


def _normalize_row_q_bits(
    row_q_bits: int | Sequence[int] | np.ndarray,
    rows: int,
) -> np.ndarray:
    values = np.asarray(row_q_bits)
    if values.ndim == 0:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("MXFP4-SQ q must be an integer in [1,4]")
        scalar = int(values)
        if scalar < 1 or scalar > 4:
            raise ValueError("MXFP4-SQ q values must lie in [1,4]")
        values = np.full(rows, scalar, dtype=np.uint8)
    else:
        if values.ndim != 1 or values.shape != (rows,):
            raise ValueError(f"MXFP4-SQ q map must have shape [{rows}]")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("MXFP4-SQ q map must use integer values")
        if np.any(values < 1) or np.any(values > 4):
            raise ValueError("MXFP4-SQ q values must lie in [1,4]")
        values = np.ascontiguousarray(values, dtype=np.uint8)
    return values


def mxfp4_sq_blob_nbytes(
    row_q_bits: int | Sequence[int] | np.ndarray,
    rows: int,
    columns: int,
) -> int:
    """Return the exact canonical payload size for a q preset or row q map."""

    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("MXFP4-SQ requires a positive block-32 matrix")
    q = _normalize_row_q_bits(row_q_bits, rows)
    sq_rows = int(np.count_nonzero(q < 4))
    sq4_rows = rows - sq_rows
    blocks = columns // 32
    q_nbytes = ((rows * 2 + 7) // 8 + 3) & ~3
    return (
        _HEADER.size
        + q_nbytes
        + int(q.astype(np.int64).sum()) * columns // 8
        + (sq_rows * blocks + 7) // 8
        + sq_rows * 2
        + sq_rows * 5
        + sq4_rows * blocks
    )


def _pack_fixed_width(values: np.ndarray, bits: int) -> np.ndarray:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    if flat.size and int(flat.max()) >= 1 << bits:
        raise ValueError(f"value exceeds {bits}-bit stream")
    bit_planes = ((flat[:, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(-1)
    return np.packbits(bit_planes, bitorder="little")


def _select_scale_base(packed: np.ndarray, scales: np.ndarray) -> int:
    """Choose the lowest scale-only-SSE four-exponent window."""

    source_min = int(scales.min())
    source_max = int(scales.max())
    low = np.asarray(packed, dtype=np.uint8) & 0x0F
    high = np.asarray(packed, dtype=np.uint8) >> 4
    energy = np.square(_NIBBLE_VALUES[low]).reshape(scales.shape[0], scales.shape[1], 16).sum(
        2
    ) + np.square(_NIBBLE_VALUES[high]).reshape(scales.shape[0], scales.shape[1], 16).sum(2)
    source = scales.astype(np.int16)
    best: tuple[float, int, int] | None = None
    for base in range(max(0, source_min - 3), min(251, source_max) + 1):
        reconstructed = np.clip(source, base, base + 3)
        delta = np.exp2(source - 127) - np.exp2(reconstructed - 127)
        error = float((energy * np.square(delta)).sum())
        candidate = (error, abs(base - source_min), base)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("MXFP4-SQ could not select a scale base")
    return int(best[2])


def _quantization_tables(
    palette_nibbles: np.ndarray,
    minimum_delta: int,
    maximum_delta: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    levels = _NIBBLE_VALUES[palette_nibbles]
    delta_count = maximum_delta - minimum_delta + 1
    symbols = np.empty((delta_count, 32, 16), dtype=np.uint8)
    errors = np.empty((delta_count, 32, 16), dtype=np.float32)
    penalties = np.empty((delta_count, 32, 16, 4), dtype=np.float32)
    for delta_index, delta in enumerate(range(minimum_delta, maximum_delta + 1)):
        target = _NIBBLE_VALUES * math.ldexp(1.0, delta)
        costs = np.square(target[None, :, None] - levels[:, None, :])
        best = costs.argmin(axis=2).astype(np.uint8)
        base = np.take_along_axis(costs, best[:, :, None], axis=2)[:, :, 0]
        symbols[delta_index] = best
        errors[delta_index] = base
        penalties[delta_index, :, :, 0] = 0.0
        for repair in range(1, 4):
            if repair >= palette_nibbles.shape[1]:
                penalties[delta_index, :, :, repair] = np.inf
            else:
                alternative = best ^ np.uint8(repair)
                penalties[delta_index, :, :, repair] = (
                    np.take_along_axis(costs, alternative[:, :, None], axis=2)[:, :, 0]
                    - base
                )
    return symbols, errors, penalties


def _block_option(
    packed: np.ndarray,
    row: int,
    block: int,
    palette: int,
    delta_index: int,
    desired_tag: int,
    nearest: np.ndarray,
    base_errors: np.ndarray,
    penalties: np.ndarray,
) -> tuple[float, int, int, int, int]:
    error = 0.0
    xor_value = 0
    first_value = np.empty(4, dtype=np.float32)
    second_value = np.empty(4, dtype=np.float32)
    first_index = np.empty(4, dtype=np.int32)
    second_index = np.empty(4, dtype=np.int32)
    for repair in range(4):
        first_value[repair] = np.float32(np.inf)
        second_value[repair] = np.float32(np.inf)
        first_index[repair] = -1
        second_index[repair] = -1
    byte_base = block * 16
    for lane in range(32):
        byte = int(packed[row, byte_base + (lane >> 1)])
        nibble = (byte >> (4 * (lane & 1))) & 15
        symbol = int(nearest[delta_index, palette, nibble])
        xor_value ^= symbol & 3
        error += float(base_errors[delta_index, palette, nibble])
        for repair in range(1, 4):
            value = penalties[delta_index, palette, nibble, repair]
            if value < first_value[repair]:
                second_value[repair] = first_value[repair]
                second_index[repair] = first_index[repair]
                first_value[repair] = value
                first_index[repair] = lane
            elif value < second_value[repair]:
                second_value[repair] = value
                second_index[repair] = lane

    required = xor_value ^ desired_tag
    if required == 0:
        return error, -1, 0, -1, 0
    repair_a = 2 if required == 1 else 1
    repair_b = 3 if required in {1, 2} else 2
    if required == 2:
        repair_a = 1
    pair_error = np.float32(np.inf)
    pair_a = -1
    pair_b = -1
    if first_index[repair_a] != first_index[repair_b]:
        pair_error = first_value[repair_a] + first_value[repair_b]
        pair_a = first_index[repair_a]
        pair_b = first_index[repair_b]
    else:
        left = first_value[repair_a] + second_value[repair_b]
        right = second_value[repair_a] + first_value[repair_b]
        if left <= right:
            pair_error = left
            pair_a = first_index[repair_a]
            pair_b = second_index[repair_b]
        else:
            pair_error = right
            pair_a = second_index[repair_a]
            pair_b = first_index[repair_b]
    if first_value[required] <= pair_error:
        return error + float(first_value[required]), first_index[required], required, -1, 0
    return error + float(pair_error), pair_a, repair_a, pair_b, repair_b


_compiled_block_option = (
    njit(inline="always")(_block_option) if njit is not None else _block_option  # pragma: no cover
)


def _sq1_block_option(
    packed: np.ndarray,
    row: int,
    block: int,
    palette: int,
    delta_index: int,
    desired_tag: int,
    nearest: np.ndarray,
    base_errors: np.ndarray,
    penalties: np.ndarray,
) -> tuple[float, int, int, int, int]:
    """Return the best SQ1 block while embedding even/odd parity bits."""

    error = 0.0
    even_parity = 0
    odd_parity = 0
    even_penalty = np.float32(np.inf)
    odd_penalty = np.float32(np.inf)
    even_index = -1
    odd_index = -1
    byte_base = block * 16
    for lane in range(32):
        byte = int(packed[row, byte_base + (lane >> 1)])
        nibble = (byte >> (4 * (lane & 1))) & 15
        symbol = int(nearest[delta_index, palette, nibble])
        error += float(base_errors[delta_index, palette, nibble])
        penalty = penalties[delta_index, palette, nibble, 1]
        if lane & 1:
            odd_parity ^= symbol
            if penalty < odd_penalty:
                odd_penalty = penalty
                odd_index = lane
        else:
            even_parity ^= symbol
            if penalty < even_penalty:
                even_penalty = penalty
                even_index = lane

    repair_even = even_parity ^ (desired_tag & 1)
    repair_odd = odd_parity ^ ((desired_tag >> 1) & 1)
    if repair_even:
        error += float(even_penalty)
    if repair_odd:
        error += float(odd_penalty)
    return (
        error,
        even_index if repair_even else -1,
        repair_even,
        odd_index if repair_odd else -1,
        repair_odd,
    )


_compiled_sq1_block_option = (
    njit(inline="always")(_sq1_block_option)
    if njit is not None
    else _sq1_block_option  # pragma: no cover
)


def _refine_row_states(
    packed: np.ndarray,
    scales: np.ndarray,
    row: int,
    scale_base: int,
    symbol_bits: int,
    delta_minimum: int,
    nearest: np.ndarray,
    base_errors: np.ndarray,
    penalties: np.ndarray,
    source_minimum: int,
    source_count: int,
    best_scales: np.ndarray,
    best_palettes: np.ndarray,
) -> None:
    """Run one block-assignment/state-refit step in place."""

    blocks = scales.shape[1]
    assigned_counts = np.zeros((8, source_count, 16), dtype=np.int32)
    for block in range(blocks):
        source_scale = int(scales[row, block])
        selected_state = 0
        selected_error = np.inf
        for state in range(8):
            delta_index = source_scale - (scale_base + best_scales[state]) - delta_minimum
            if symbol_bits == 1:
                option = _compiled_sq1_block_option(
                    packed,
                    row,
                    block,
                    best_palettes[state],
                    delta_index,
                    state & 3,
                    nearest,
                    base_errors,
                    penalties,
                )
            else:
                option = _compiled_block_option(
                    packed,
                    row,
                    block,
                    best_palettes[state],
                    delta_index,
                    state & 3,
                    nearest,
                    base_errors,
                    penalties,
                )
            scaled_error = option[0] * float(1 << (2 * best_scales[state]))
            if scaled_error < selected_error:
                selected_error = scaled_error
                selected_state = state
        source_index = source_scale - source_minimum
        byte_base = block * 16
        for lane in range(32):
            byte = int(packed[row, byte_base + (lane >> 1)])
            nibble = (byte >> (4 * (lane & 1))) & 15
            assigned_counts[selected_state, source_index, nibble] += 1

    for state in range(8):
        population = 0
        for source_index in range(source_count):
            for nibble in range(16):
                population += assigned_counts[state, source_index, nibble]
        if population == 0:
            continue
        best_error = np.inf
        chosen_scale = best_scales[state]
        chosen_palette = best_palettes[state]
        for target_offset in range(4):
            for palette in range(32):
                candidate = 0.0
                for source_index in range(source_count):
                    source_scale = source_minimum + source_index
                    delta_index = source_scale - (scale_base + target_offset) - delta_minimum
                    for nibble in range(16):
                        candidate += (
                            assigned_counts[state, source_index, nibble]
                            * base_errors[delta_index, palette, nibble]
                            * float(1 << (2 * target_offset))
                        )
                if candidate < best_error:
                    best_error = candidate
                    chosen_scale = target_offset
                    chosen_palette = palette
        best_scales[state] = chosen_scale
        best_palettes[state] = chosen_palette


_compiled_refine_row_states = (
    njit(inline="always")(_refine_row_states)
    if njit is not None
    else _refine_row_states  # pragma: no cover
)


def _encode_rows(
    packed: np.ndarray,
    scales: np.ndarray,
    scale_base: int,
    symbol_bits: int,
    delta_minimum: int,
    nearest: np.ndarray,
    base_errors: np.ndarray,
    penalties: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = packed.shape[0]
    columns = packed.shape[1] * 2
    blocks = columns // 32
    symbol_bytes = columns * symbol_bits // 8
    packed_symbols = np.zeros((rows, symbol_bytes), dtype=np.uint8)
    selectors = np.zeros((rows, blocks), dtype=np.uint8)
    state_scales = np.zeros((rows, 8), dtype=np.uint8)
    state_palettes = np.zeros((rows, 8), dtype=np.uint8)

    for row in prange(rows):
        source_minimum = 255
        source_maximum = 0
        for block in range(blocks):
            source_scale = int(scales[row, block])
            source_minimum = min(source_minimum, source_scale)
            source_maximum = max(source_maximum, source_scale)
        source_count = source_maximum - source_minimum + 1
        block_counts = np.zeros((blocks, 16), dtype=np.int32)
        for block in range(blocks):
            byte_base = block * 16
            for lane in range(32):
                byte = int(packed[row, byte_base + (lane >> 1)])
                nibble = (byte >> (4 * (lane & 1))) & 15
                block_counts[block, nibble] += 1

        candidate_errors = np.zeros((4, 32, blocks), dtype=np.float32)
        for target_offset in range(4):
            scale_factor = float(1 << (2 * target_offset))
            for palette in range(32):
                for block in range(blocks):
                    source_scale = int(scales[row, block])
                    delta_index = source_scale - (scale_base + target_offset) - delta_minimum
                    error = 0.0
                    for nibble in range(16):
                        error += (
                            block_counts[block, nibble]
                            * base_errors[delta_index, palette, nibble]
                            * scale_factor
                        )
                    candidate_errors[target_offset, palette, block] = error

        # Greedily cover the row's (source exponent, E2M1 value) histogram
        # with eight distinct states.  This retains the full search's useful
        # concentration around central exponents without paying its
        # state-by-block hard-EM cost for every model row.
        best_scales = np.zeros(8, dtype=np.int32)
        best_palettes = np.zeros(8, dtype=np.int32)
        covered = np.full(blocks, np.inf, dtype=np.float32)
        selected = np.zeros((4, 32), dtype=np.uint8)
        for state in range(8):
            best_error = np.inf
            chosen_scale = 0
            chosen_palette = 0
            for target_offset in range(4):
                for palette in range(32):
                    if selected[target_offset, palette] != 0:
                        continue
                    candidate = 0.0
                    for block in range(blocks):
                        candidate += min(
                            covered[block],
                            candidate_errors[target_offset, palette, block],
                        )
                    if candidate < best_error:
                        best_error = candidate
                        chosen_scale = target_offset
                        chosen_palette = palette
            best_scales[state] = chosen_scale
            best_palettes[state] = chosen_palette
            selected[chosen_scale, chosen_palette] = 1
            for block in range(blocks):
                covered[block] = min(
                    covered[block],
                    candidate_errors[chosen_scale, chosen_palette, block],
                )
        # Repeated hard assignment/refit captures the useful row-local state
        # structure without materializing the research solver's large error
        # tensor.  The count is fixed, deterministic, and shared by q1--q3.
        for _ in range(3):
            _compiled_refine_row_states(
                packed,
                scales,
                row,
                scale_base,
                symbol_bits,
                delta_minimum,
                nearest,
                base_errors,
                penalties,
                source_minimum,
                source_count,
                best_scales,
                best_palettes,
            )
        for state in range(8):
            state_scales[row, state] = best_scales[state]
            state_palettes[row, state] = best_palettes[state]

        for block in range(blocks):
            source_scale = int(scales[row, block])
            selected_state = 0
            selected_delta_index = 0
            selected_option = (np.inf, -1, 0, -1, 0)
            for state in range(8):
                delta_index = source_scale - (scale_base + best_scales[state]) - delta_minimum
                if symbol_bits == 1:
                    option = _compiled_sq1_block_option(
                        packed,
                        row,
                        block,
                        best_palettes[state],
                        delta_index,
                        state & 3,
                        nearest,
                        base_errors,
                        penalties,
                    )
                else:
                    option = _compiled_block_option(
                        packed,
                        row,
                        block,
                        best_palettes[state],
                        delta_index,
                        state & 3,
                        nearest,
                        base_errors,
                        penalties,
                    )
                scaled_error = option[0] * float(1 << (2 * best_scales[state]))
                if scaled_error < selected_option[0]:
                    selected_state = state
                    selected_delta_index = delta_index
                    selected_option = (
                        scaled_error,
                        option[1],
                        option[2],
                        option[3],
                        option[4],
                    )
            palette = best_palettes[selected_state]
            selectors[row, block] = selected_state >> 2
            repair_index_a, repair_a = selected_option[1], selected_option[2]
            repair_index_b, repair_b = selected_option[3], selected_option[4]
            byte_base = block * 16
            for lane in range(32):
                byte = int(packed[row, byte_base + (lane >> 1)])
                nibble = (byte >> (4 * (lane & 1))) & 15
                symbol = int(nearest[selected_delta_index, palette, nibble])
                if lane == repair_index_a:
                    symbol ^= repair_a
                if lane == repair_index_b:
                    symbol ^= repair_b
                linear = block * 32 + lane
                bit = linear * symbol_bits
                output_byte = bit >> 3
                shift = bit & 7
                packed_symbols[row, output_byte] |= np.uint8((symbol << shift) & 255)
                if shift + symbol_bits > 8:
                    packed_symbols[row, output_byte + 1] |= np.uint8(symbol >> (8 - shift))
    return packed_symbols, selectors, state_scales, state_palettes


if njit is not None:
    _compiled_encode_rows = njit(parallel=True, nogil=True)(_encode_rows)
else:  # pragma: no cover
    _compiled_encode_rows = _encode_rows


def _normalize_source(
    packed: np.ndarray,
    scale_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    source = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scale_raw, dtype=np.uint8)
    if source.ndim != 2 or scales.ndim != 2:
        raise ValueError("MXFP4-SQ source values and scales must be matrices")
    rows, packed_columns = source.shape
    columns = packed_columns * 2
    if rows <= 0 or columns <= 0 or columns % 32:
        raise ValueError("MXFP4-SQ source must be a non-empty block-32 matrix")
    if scales.shape != (rows, columns // 32):
        raise ValueError("MXFP4-SQ source geometry is inconsistent")
    if np.any(scales == 255):
        raise ValueError("MXFP4-SQ source contains an E8M0 NaN scale")
    return source, scales, rows, columns


def _encode_profile_rows(
    source: np.ndarray,
    scales: np.ndarray,
    *,
    bits: int,
    matrix_scale_base: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if bits not in {1, 2, 3}:
        raise ValueError("MXFP4-SQ compressed rows require q in [1,3]")
    delta_minimum = int(scales.min()) - (matrix_scale_base + 3)
    delta_maximum = int(scales.max()) - matrix_scale_base
    palettes = {
        1: SQ1_PALETTE_NIBBLES,
        2: SQ2_PALETTE_NIBBLES,
        3: SQ3_PALETTE_NIBBLES,
    }[bits]
    nearest, errors, penalties = _quantization_tables(
        palettes,
        delta_minimum,
        delta_maximum,
    )
    return _compiled_encode_rows(
        source,
        scales,
        matrix_scale_base,
        bits,
        delta_minimum,
        nearest,
        errors,
        penalties,
    )


def encode_mxfp4_sq(
    packed: np.ndarray,
    scale_raw: np.ndarray,
    *,
    bits: int,
    matrix_scale_base: int | None = None,
) -> Mxfp4SqEncoding:
    """Encode a historical fixed-width SQ2/SQ3 payload.

    New callers should use :func:`quantize_mxfp4_sq`, which always writes the
    unified q1--q4 container.
    """

    if bits not in {2, 3}:
        raise ValueError("MXFP4-SQ supports two- or three-bit symbols")
    source, scales, rows, columns = _normalize_source(packed, scale_raw)
    scale_base = (
        _select_scale_base(source, scales) if matrix_scale_base is None else int(matrix_scale_base)
    )
    if not 0 <= scale_base <= 251:
        raise ValueError("MXFP4-SQ scale base must leave four E8M0 values")
    packed_symbols, selectors, state_scales, state_palettes = _encode_profile_rows(
        source,
        scales,
        bits=bits,
        matrix_scale_base=scale_base,
    )
    encoding = Mxfp4SqEncoding(
        bits=bits,
        rows=rows,
        columns=columns,
        matrix_scale_base=scale_base,
        packed_symbols=np.ascontiguousarray(packed_symbols).reshape(-1),
        packed_block_selectors=np.packbits(selectors.reshape(-1), bitorder="little"),
        packed_state_scales=_pack_fixed_width(state_scales, 2),
        packed_state_palettes=_pack_fixed_width(state_palettes, 5),
    )
    expected = _legacy_mxfp4_sq_blob_nbytes(bits, rows, columns)
    if encoding.payload_nbytes != expected:
        raise RuntimeError(
            f"MXFP4-SQ payload size mismatch: {encoding.payload_nbytes} != {expected}"
        )
    return encoding


def quantize_mxfp4_sq(
    packed: np.ndarray,
    scale_raw: np.ndarray,
    *,
    row_q_bits: int | Sequence[int] | np.ndarray = 2,
    matrix_scale_base: int | None = None,
) -> Mxfp4SqTensor:
    """Quantize native packed MXFP4 into one q1--q4 logical tensor.

    ``row_q_bits`` may be a scalar uniform preset or one integer per output
    neuron.  q=4 is an exact storage endpoint; its packed values and E8M0
    scales are copied bit-for-bit.  q=1/2/3 rows use the established fixed
    palette SQ solver and share one matrix-level four-exponent window.
    """

    source, scales, rows, columns = _normalize_source(packed, scale_raw)
    q = _normalize_row_q_bits(row_q_bits, rows)
    compressed = q < 4
    if matrix_scale_base is None:
        scale_base = (
            _select_scale_base(source[compressed], scales[compressed])
            if bool(compressed.any())
            else min(251, int(scales.min()))
        )
    else:
        scale_base = int(matrix_scale_base)
    if not 0 <= scale_base <= 251:
        raise ValueError("MXFP4-SQ scale base must leave four E8M0 values")

    blocks = columns // 32
    sq_rows = int(np.count_nonzero(compressed))
    sq4_rows = rows - sq_rows
    sq_position = np.full(rows, -1, dtype=np.int64)
    sq_position[compressed] = np.arange(sq_rows, dtype=np.int64)
    sq4_position = np.full(rows, -1, dtype=np.int64)
    sq4_position[~compressed] = np.arange(sq4_rows, dtype=np.int64)
    row_symbols: list[bytes | None] = [None] * rows
    sq_selectors = np.empty((sq_rows, blocks), dtype=np.uint8)
    sq_state_scales = np.empty((sq_rows, 8), dtype=np.uint8)
    sq_state_palettes = np.empty((sq_rows, 8), dtype=np.uint8)
    sq4_native_scales = np.empty((sq4_rows, blocks), dtype=np.uint8)

    for bits in (1, 2, 3):
        row_ids = np.flatnonzero(q == bits)
        if not row_ids.size:
            continue
        symbols, selectors, state_scales, state_palettes = _encode_profile_rows(
            source[row_ids],
            scales[row_ids],
            bits=bits,
            matrix_scale_base=scale_base,
        )
        positions = sq_position[row_ids]
        sq_selectors[positions] = selectors
        sq_state_scales[positions] = state_scales
        sq_state_palettes[positions] = state_palettes
        for local_row, global_row in enumerate(row_ids.tolist()):
            row_symbols[global_row] = np.ascontiguousarray(symbols[local_row]).tobytes()

    for row in np.flatnonzero(q == 4).tolist():
        row_symbols[row] = source[row].tobytes()
        sq4_native_scales[sq4_position[row]] = scales[row]
    if any(value is None for value in row_symbols):
        raise RuntimeError("MXFP4-SQ failed to materialize every output row")

    tensor = build_mxfp4_sq(
        row_q_bits=q,
        input_size=columns,
        matrix_scale_base=scale_base,
        row_symbols=tuple(value for value in row_symbols if value is not None),
        sq_block_selectors=sq_selectors,
        sq_state_scales=sq_state_scales,
        sq_state_palettes=sq_state_palettes,
        sq4_native_scales=sq4_native_scales,
    )
    expected = mxfp4_sq_blob_nbytes(q, rows, columns)
    if len(tensor.payload) != expected:
        raise RuntimeError(
            f"MXFP4-SQ payload size mismatch: {len(tensor.payload)} != {expected}"
        )
    return tensor


def write_mxfp4_sq_blob(
    path: str | Path,
    packed: np.ndarray,
    scale_raw: np.ndarray,
    *,
    bits: int | None = None,
    row_q_bits: int | Sequence[int] | np.ndarray | None = None,
    matrix_scale_base: int | None = None,
) -> int:
    """Quantize and write one canonical MXFP4-SQ payload."""

    if bits is not None and row_q_bits is not None:
        raise ValueError("specify either bits or row_q_bits, not both")
    if bits is None and row_q_bits is None:
        q: int | Sequence[int] | np.ndarray = 2
    elif row_q_bits is None:
        q = bits
    else:
        q = row_q_bits
    tensor = quantize_mxfp4_sq(
        packed,
        scale_raw,
        row_q_bits=q,
        matrix_scale_base=matrix_scale_base,
    )
    target = Path(path)
    target.write_bytes(tensor.payload)
    return target.stat().st_size


__all__ = [
    "Mxfp4SqEncoding",
    "SQ1_PALETTE_NIBBLES",
    "SQ2_PALETTE_NIBBLES",
    "SQ3_PALETTE_NIBBLES",
    "encode_mxfp4_sq",
    "mxfp4_sq_blob_nbytes",
    "quantize_mxfp4_sq",
    "write_mxfp4_sq_blob",
]
