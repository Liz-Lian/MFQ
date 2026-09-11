"""Unified metadata-driven NINT kernels for Apple silicon.

Every NINT tensor, including a uniform preset, carries per-neuron ``q`` and
``k`` metadata.  The runtime therefore has one packed matmul kernel for every
primary-code-width mixture.  Uniform presets are constant metadata, not
separate execution formats or kernel families.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "MFQ's Metal backend requires MLX; install with `pip install -e '.[metal]'`"
    ) from exc

from mfq.formats.nint import (
    NINT_ADAPTIVE_FLAG,
    NINT_K_SELECTOR_BITS,
    NINT_Q_SELECTOR_BITS,
    NintTensor,
)

_NINT_HEADER = struct.Struct("<BBiii")

_NINT_HEADER_SOURCE = r"""
template <typename Stream>
inline uint mfq_nint_read_row_value(
    Stream stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    if (bits == 8u && shift == 0u) {
        return uint(stream[byte_index]);
    }
    if (bits == 4u && row_bit_shift == 0u) {
        uint packed = uint(stream[row_byte_offset + (value_index >> 1u)]);
        return (packed >> ((value_index & 1u) * 4u)) & 15u;
    }
    if (bits == 2u && row_bit_shift == 0u) {
        uint packed = uint(stream[row_byte_offset + (value_index >> 2u)]);
        return (packed >> ((value_index & 3u) * 2u)) & 3u;
    }
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1ul]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

template <typename Stream>
inline uint4 mfq_nint_read_row_value4(
    Stream stream,
    uint row_byte_offset,
    uint row_bit_shift,
    uint value_index,
    uint bits
) {
    uint row_relative_bits = row_bit_shift + value_index * bits;
    uint byte_index = row_byte_offset + (row_relative_bits >> 3u);
    uint shift = row_relative_bits & 7u;
    if (shift == 0u) {
        if (bits == 8u) {
            return uint4(
                uint(stream[byte_index]),
                uint(stream[byte_index + 1u]),
                uint(stream[byte_index + 2u]),
                uint(stream[byte_index + 3u]));
        }
        if (bits == 4u) {
            uint lo = uint(stream[byte_index]);
            uint hi = uint(stream[byte_index + 1u]);
            return uint4(lo & 15u, lo >> 4u, hi & 15u, hi >> 4u);
        }
        if (bits == 2u) {
            uint packed = uint(stream[byte_index]);
            return uint4(
                packed & 3u,
                (packed >> 2u) & 3u,
                (packed >> 4u) & 3u,
                packed >> 6u);
        }
    }
    uint required_bits = shift + 4u * bits;
    uint packed = uint(stream[byte_index]);
    if (required_bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    if (required_bits > 16u) {
        packed |= uint(stream[byte_index + 2u]) << 16u;
    }
    if (required_bits > 24u) {
        packed |= uint(stream[byte_index + 3u]) << 24u;
    }
    if (shift != 0u) {
        packed = (packed >> shift)
            | (required_bits > 32u
                ? uint(stream[byte_index + 4u]) << (32u - shift)
                : 0u);
    }
    uint mask = (1u << bits) - 1u;
    return uint4(
        packed & mask,
        (packed >> bits) & mask,
        (packed >> (2u * bits)) & mask,
        (packed >> (3u * bits)) & mask);
}
"""

_NINT_MATMUL_SOURCE = r"""
    constexpr uint SIMD_GROUPS = 8u;
    constexpr uint OUTPUTS_PER_SIMD = 2u;
    constexpr uint OUTPUTS_PER_TG = SIMD_GROUPS * OUTPUTS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint output_base =
        threadgroup_position_in_grid.x * OUTPUTS_PER_TG
        + simd_group * OUTPUTS_PER_SIMD;
    uint first_row = threadgroup_position_in_grid.y * uint(TILE_M);
    if (output_base >= uint(LOGICAL_OUT) || first_row >= uint(M)) {
        return;
    }

    int local_expert = 0;
    bool route_valid = true;
    if (uint(ROUTED) != 0u) {
        int expert = expert_ids[first_row];
        route_valid = expert >= 0 && expert < int(EXPERT_MAP_SIZE);
        local_expert = route_valid ? expert_map[expert] : -1;
        route_valid = route_valid && local_expert >= 0
            && local_expert < int(LOCAL_EXPERTS);
    }

    uint outputs[OUTPUTS_PER_SIMD];
    uint metadata_bases[OUTPUTS_PER_SIMD];
    uint q_widths[OUTPUTS_PER_SIMD];
    uint q_row_byte_offsets[OUTPUTS_PER_SIMD];
    uint q_row_bit_shifts[OUTPUTS_PER_SIMD];
    float neuron_scales[OUTPUTS_PER_SIMD];
    float neuron_minimums[OUTPUTS_PER_SIMD];
    float accumulators[OUTPUTS_PER_SIMD][TILE_M];
#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        uint logical_output = min(
            output_base + output_row,
            uint(LOGICAL_OUT) - 1u);
        uint output = uint(ROUTED) != 0u
            ? (route_valid
                ? uint(local_expert) * uint(OUT_PER_EXPERT)
                    + logical_output
                : 0u)
            : logical_output;
        output = min(output, uint(OUT) - 1u);
        outputs[output_row] = output;
        metadata_bases[output_row] = output * uint(NG);
        uint row_layout = uint(row_q_layout[output]);
        q_widths[output_row] = row_layout & 15u;
        q_row_byte_offsets[output_row] = row_q_byte_offsets[output];
        q_row_bit_shifts[output_row] = row_layout >> 4u;
        neuron_scales[output_row] = neuron_scale[output];
        neuron_minimums[output_row] = neuron_min[output];
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            accumulators[output_row][local_row] = 0.0f;
        }
    }

    constexpr uint CHUNKS = (uint(GS) + 3u) / 4u;
    constexpr uint Q4_WORDS = (uint(GS) + 7u) / 8u;
    if (route_valid) {
    for (uint group = lane; group < uint(NG); group += 32u) {
        uint q4_words[OUTPUTS_PER_SIMD][Q4_WORDS];
#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            if (q_widths[output_row] == 4u &&
                q_row_bit_shifts[output_row] == 0u &&
                (uint(GS) & 7u) == 0u) {
                uint byte_offset = q_row_byte_offsets[output_row]
                    + group * (uint(GS) >> 1u);
#pragma unroll
                for (uint word = 0u; word < Q4_WORDS; ++word) {
                    uint base = byte_offset + word * 4u;
                    q4_words[output_row][word] =
                        uint(q_packed[base])
                        | (uint(q_packed[base + 1u]) << 8u)
                        | (uint(q_packed[base + 2u]) << 16u)
                        | (uint(q_packed[base + 3u]) << 24u);
                }
            }
        }
        float activation_sums[TILE_M];
        float quantized_dots[OUTPUTS_PER_SIMD][TILE_M];
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            activation_sums[local_row] = 0.0f;
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                quantized_dots[output_row][local_row] = 0.0f;
            }
        }

#pragma unroll
        for (uint chunk = 0u; chunk < CHUNKS; ++chunk) {
            uint group_element = chunk * 4u;
            uint column = group * uint(GS) + group_element;
            uint4 codes[OUTPUTS_PER_SIMD];
#pragma unroll
            for (uint output_row = 0u;
                 output_row < OUTPUTS_PER_SIMD;
                 ++output_row) {
                if (q_widths[output_row] == 4u &&
                    q_row_bit_shifts[output_row] == 0u &&
                    (uint(GS) & 7u) == 0u) {
                    uint packed = q4_words[output_row][chunk >> 1u]
                        >> ((chunk & 1u) * 16u);
                    codes[output_row] = uint4(
                        packed & 15u,
                        (packed >> 4u) & 15u,
                        (packed >> 8u) & 15u,
                        (packed >> 12u) & 15u);
                } else if (group_element + 3u < uint(GS)) {
                    codes[output_row] = mfq_nint_read_row_value4(
                        q_packed,
                        q_row_byte_offsets[output_row],
                        q_row_bit_shifts[output_row],
                        column,
                        q_widths[output_row]);
                } else {
                    codes[output_row] = uint4(0u);
#pragma unroll
                    for (uint element = 0u; element < 4u; ++element) {
                        if (group_element + element < uint(GS)) {
                            codes[output_row][element] =
                                mfq_nint_read_row_value(
                                    q_packed,
                                    q_row_byte_offsets[output_row],
                                    q_row_bit_shifts[output_row],
                                    column + element,
                                    q_widths[output_row]);
                        }
                    }
                }
            }
#pragma unroll
            for (uint local_row = 0u;
                 local_row < uint(TILE_M);
                 ++local_row) {
                uint row = first_row + local_row;
                uint input_row = uint(ROUTED) != 0u
                    && uint(SHARED_INPUT) != 0u
                    ? row / uint(ROUTES)
                    : row;
                float4 activation = float4(0.0f);
                if (row < uint(M)) {
                    uint input_base = input_row * uint(K) + column;
                    if (group_element + 3u < uint(GS) &&
                        column + 3u < uint(K)) {
                        activation = float4(
                            x[input_base],
                            x[input_base + 1u],
                            x[input_base + 2u],
                            x[input_base + 3u]);
                    } else {
                        activation.x = column < uint(K)
                            ? float(x[input_base]) : 0.0f;
                        activation.y =
                            group_element + 1u < uint(GS) && column + 1u < uint(K)
                            ? float(x[input_base + 1u]) : 0.0f;
                        activation.z =
                            group_element + 2u < uint(GS) && column + 2u < uint(K)
                            ? float(x[input_base + 2u]) : 0.0f;
                        activation.w =
                            group_element + 3u < uint(GS) && column + 3u < uint(K)
                            ? float(x[input_base + 3u]) : 0.0f;
                    }
                }
                activation_sums[local_row] +=
                    activation.x + activation.y + activation.z + activation.w;
#pragma unroll
                for (uint output_row = 0u;
                     output_row < OUTPUTS_PER_SIMD;
                     ++output_row) {
                    quantized_dots[output_row][local_row] +=
                        dot(activation, float4(codes[output_row]));
                }
            }
        }

#pragma unroll
        for (uint output_row = 0u;
             output_row < OUTPUTS_PER_SIMD;
             ++output_row) {
            uint metadata_index = metadata_bases[output_row] + group;
            float scale =
                neuron_scales[output_row] * float(sub_scale[metadata_index]);
            float minimum =
                neuron_minimums[output_row] * float(sub_min[metadata_index]);
#pragma unroll
            for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
                accumulators[output_row][local_row] = fma(
                    scale,
                    quantized_dots[output_row][local_row],
                    fma(
                        -minimum,
                        activation_sums[local_row],
                        accumulators[output_row][local_row]));
            }
        }
    }
    }

#pragma unroll
    for (uint output_row = 0u;
         output_row < OUTPUTS_PER_SIMD;
         ++output_row) {
        uint output = output_base + output_row;
#pragma unroll
        for (uint local_row = 0u; local_row < uint(TILE_M); ++local_row) {
            float total = simd_sum(accumulators[output_row][local_row]);
            uint row = first_row + local_row;
            if (lane == 0u && output < uint(LOGICAL_OUT) && row < uint(M)) {
                y[row * uint(LOGICAL_OUT) + output] =
                    T(route_valid ? total : 0.0f);
            }
        }
    }
"""

_NINT_ROW_DECODE_SOURCE = r"""
    uint output_index = thread_position_in_grid.x;
    if (output_index >= uint(COUNT) * uint(K)) {
        return;
    }
    uint token_position = output_index / uint(K);
    uint input_index = output_index - token_position * uint(K);
    uint output = uint(row_ids[token_position]);
    uint group = input_index / uint(GS);
    uint element = input_index - group * uint(GS);
    uint metadata_index = output * uint(NG) + group;
    uint quantized = mfq_nint_read_row_value(
        q_packed,
        row_q_byte_offsets[output],
        uint(row_q_layout[output]) >> 4u,
        group * uint(GS) + element,
        uint(row_q_layout[output]) & 15u);
    float scale = neuron_scale[output] * float(sub_scale[metadata_index]);
    float minimum = neuron_min[output] * float(sub_min[metadata_index]);
    y[output_index] = T(scale * float(quantized) - minimum);
"""


def _make_kernel(name: str, input_names: list[str], source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=input_names,
        output_names=["y"],
        header=_NINT_HEADER_SOURCE,
        source=source,
        compile_options={"math_mode": "fast"},
    )


_NINT_MATMUL_KERNEL = _make_kernel(
    "mfq_nint_matmul",
    [
        "q_packed",
        "row_q_layout",
        "row_q_byte_offsets",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "x",
        "expert_ids",
        "expert_map",
    ],
    _NINT_MATMUL_SOURCE,
)

_NINT_UNROUTED_SENTINEL = mx.zeros((1,), dtype=mx.int32)

_NINT_ROW_DECODE_KERNEL = _make_kernel(
    "mfq_nint_row_decode",
    [
        "q_packed",
        "row_q_layout",
        "row_q_byte_offsets",
        "sub_scale",
        "sub_min",
        "neuron_scale",
        "neuron_min",
        "row_ids",
    ],
    _NINT_ROW_DECODE_SOURCE,
)


def _packed_nbytes(count: int, bits: int) -> int:
    if count < 0 or not 1 <= int(bits) <= 8:
        raise ValueError("invalid packed NINT bit count")
    return (int(count) * int(bits) + 7) // 8


def _pack_values(values: np.ndarray, bits: int) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.uint8).reshape(-1)
    if np.any(source > (1 << int(bits)) - 1):
        raise ValueError(f"NINT values exceed the selected {bits}-bit width")
    if int(bits) == 8:
        return source
    if source.size % 8:
        source = np.concatenate(
            (source, np.zeros((-source.size) % 8, dtype=np.uint8))
        )
    blocks = source.reshape(-1, 8)
    packed = np.zeros((blocks.shape[0], int(bits)), dtype=np.uint8)
    for value_index in range(8):
        bit_position = value_index * int(bits)
        byte_index, shift = divmod(bit_position, 8)
        packed[:, byte_index] |= blocks[:, value_index] << shift
        if shift + int(bits) > 8:
            packed[:, byte_index + 1] |= blocks[:, value_index] >> (8 - shift)
    return np.ascontiguousarray(
        packed.reshape(-1)[: _packed_nbytes(values.size, bits)]
    )


def _unpack_values(
    blob: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> tuple[np.ndarray, int]:
    nbytes = _packed_nbytes(count, bits)
    end = int(offset) + nbytes
    if end > len(blob):
        raise ValueError("truncated packed NINT values")
    packed = np.frombuffer(blob, dtype=np.uint8, count=nbytes, offset=offset)
    if int(bits) == 8:
        return packed.copy(), end
    values = np.empty(int(count), dtype=np.uint8)
    full_blocks = int(count) // 8
    blocks_per_chunk = 1 << 20
    mask = (1 << int(bits)) - 1
    for block_start in range(0, full_blocks, blocks_per_chunk):
        block_end = min(full_blocks, block_start + blocks_per_chunk)
        blocks = packed[
            block_start * int(bits) : block_end * int(bits)
        ].reshape(-1, int(bits))
        value_start = block_start * 8
        value_end = block_end * 8
        for value_index in range(8):
            bit_position = value_index * int(bits)
            byte_index, shift = divmod(bit_position, 8)
            decoded = blocks[:, byte_index].astype(np.uint16) >> shift
            if shift + int(bits) > 8:
                decoded |= blocks[:, byte_index + 1].astype(np.uint16) << (
                    8 - shift
                )
            values[value_start + value_index : value_end : 8] = decoded & mask
    tail = int(count) - full_blocks * 8
    if tail:
        byte_offset = full_blocks * int(bits)
        accumulator = int.from_bytes(packed[byte_offset:].tobytes(), "little")
        for value_index in range(tail):
            values[full_blocks * 8 + value_index] = (
                accumulator >> (value_index * int(bits))
            ) & mask
    return values, end


def _pack_q_rows(
    q: np.ndarray,
    row_q_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, groups, groupsize = (int(value) for value in q.shape)
    values_per_row = groups * groupsize
    if bool(np.all(row_q_bits == row_q_bits[0])):
        bits = int(row_q_bits[0])
        row_bit_offsets = np.arange(rows, dtype=np.uint64) * values_per_row * bits
        if int(row_bit_offsets[-1] >> np.uint64(3)) > np.iinfo(np.uint32).max:
            raise OverflowError("NINT packed stream exceeds Metal uint range")
        byte_offsets = np.ascontiguousarray(row_bit_offsets >> 3, dtype=np.uint32)
        row_layout = np.ascontiguousarray(
            bits | ((row_bit_offsets & 7).astype(np.uint8) << 4),
            dtype=np.uint8,
        )
        packed = np.concatenate(
            (_pack_values(np.ascontiguousarray(q), bits), np.zeros(4, np.uint8))
        )
        return packed, row_layout, byte_offsets

    byte_offsets = np.empty(rows, dtype=np.uint32)
    row_layout = np.empty(rows, dtype=np.uint8)
    cursor = 0
    for row, bits in enumerate(row_q_bits):
        if cursor > np.iinfo(np.uint32).max:
            raise OverflowError("NINT packed stream exceeds Metal uint range")
        byte_offsets[row] = cursor
        row_layout[row] = int(bits)
        cursor += (values_per_row * int(bits) + 7) // 8
    packed = np.zeros(cursor + 4, dtype=np.uint8)
    for bits in range(1, 9):
        selected = np.flatnonzero(row_q_bits == bits)
        if not selected.size:
            continue
        values = np.ascontiguousarray(q[selected])
        stream = _pack_values(values, bits)
        for local_row, row in enumerate(selected):
            _copy_packed_bit_range(
                stream,
                local_row * values_per_row * bits,
                values_per_row * bits,
                packed,
                int(byte_offsets[int(row)]),
            )
    return np.ascontiguousarray(packed), row_layout, byte_offsets


def _copy_packed_bit_range(
    source: np.ndarray,
    start_bit: int,
    bit_count: int,
    destination: np.ndarray,
    destination_byte: int,
) -> None:
    """Copy one packed row to a byte-aligned destination without decoding it."""

    count = (int(bit_count) + 7) // 8
    source_byte, shift = divmod(int(start_bit), 8)
    if shift == 0:
        destination[destination_byte : destination_byte + count] = source[
            source_byte : source_byte + count
        ]
    else:
        combined = source[source_byte : source_byte + count].astype(np.uint16)
        combined >>= shift
        high = source[source_byte + 1 : source_byte + count + 1]
        combined[: high.size] |= high.astype(np.uint16) << (8 - shift)
        destination[destination_byte : destination_byte + count] = combined.astype(
            np.uint8
        )
    tail = int(bit_count) & 7
    if tail:
        destination[destination_byte + count - 1] &= (1 << tail) - 1


def _metadata_u8(values: np.ndarray, name: str) -> np.ndarray:
    source = np.asarray(values)
    if not np.issubdtype(source.dtype, np.integer):
        raise TypeError(f"{name} must contain integers")
    if np.any(source < 0) or np.any(source > 255):
        raise ValueError(f"{name} values must fit in uint8")
    return np.ascontiguousarray(source, dtype=np.uint8)


def _mlx_buffer(values: np.ndarray) -> mx.array:
    source = np.ascontiguousarray(values)
    if all(int(value) <= (1 << 31) - 1 for value in source.shape):
        return mx.array(source)
    size = int(source.size)
    minimum_rows = (size + (1 << 31) - 2) // ((1 << 31) - 1)
    for rows in range(max(2, minimum_rows), max(2, minimum_rows) + 1024):
        if size % rows == 0 and size // rows <= (1 << 31) - 1:
            return mx.array(source.reshape(rows, size // rows))
    raise OverflowError("packed NINT buffer cannot be represented by MLX dimensions")


@dataclass(frozen=True)
class MetalNintWeight:
    """Execution-ready NINT weight using the single metadata-driven layout."""

    q_packed: mx.array
    row_q_layout: mx.array
    row_q_byte_offsets: mx.array
    sub_scale: mx.array
    sub_min: mx.array
    neuron_scale: mx.array
    neuron_min: mx.array
    bits: int
    groupsize: int
    out: int
    groups: int
    neuron_len: int

    @classmethod
    def from_tensor(cls, tensor: NintTensor) -> MetalNintWeight:
        if len(tensor.shape) != 2 or int(tensor.axis) != 0:
            raise ValueError("Metal NINT requires a 2D weight quantized on axis 0")
        out, groups, groupsize = (int(value) for value in tensor.q.shape)
        if groupsize != int(tensor.spec.groupsize):
            raise ValueError("NINT groupsize metadata does not match q values")
        if int(tensor.neuron_len) > groups * groupsize:
            raise ValueError("NINT neuron length exceeds packed group capacity")
        if tensor.sub_scale.shape != (out, groups) or tensor.sub_min.shape != (
            out,
            groups,
        ):
            raise ValueError("NINT subgroup metadata shape mismatch")
        row_q_bits = np.ascontiguousarray(tensor.row_q_bits, dtype=np.uint8)
        q_packed, row_q_layout, row_byte_offsets = _pack_q_rows(
            np.asarray(tensor.q), row_q_bits
        )
        return cls(
            q_packed=_mlx_buffer(q_packed),
            row_q_layout=mx.array(row_q_layout),
            row_q_byte_offsets=mx.array(row_byte_offsets),
            sub_scale=mx.array(_metadata_u8(tensor.sub_scale, "sub_scale")),
            sub_min=mx.array(_metadata_u8(tensor.sub_min, "sub_min")),
            neuron_scale=mx.array(
                np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32)
            ),
            neuron_min=mx.array(
                np.ascontiguousarray(tensor.neuron_min, dtype=np.float32)
            ),
            bits=int(tensor.spec.bits),
            groupsize=groupsize,
            out=out,
            groups=groups,
            neuron_len=int(tensor.neuron_len),
        )

    @classmethod
    def from_blob(cls, blob: bytes | memoryview) -> MetalNintWeight:
        if len(blob) < _NINT_HEADER.size:
            raise ValueError("truncated NINT blob")
        raw_bits, sub_bits, groupsize, axis, neuron_len = _NINT_HEADER.unpack_from(
            blob, 0
        )
        adaptive = bool(raw_bits & NINT_ADAPTIVE_FLAG)
        bits = int(raw_bits & ~NINT_ADAPTIVE_FLAG)
        if not 1 <= bits <= 8 or not 1 <= int(sub_bits) <= 8:
            raise ValueError("invalid NINT bit widths")
        offset = _NINT_HEADER.size
        if offset + 4 > len(blob):
            raise ValueError("truncated NINT shape header")
        ndim = struct.unpack_from("<I", blob, offset)[0]
        offset += 4
        shape_nbytes = int(ndim) * 8
        if offset + shape_nbytes + 8 > len(blob):
            raise ValueError("truncated NINT shape")
        shape = struct.unpack_from(f"<{ndim}q", blob, offset)
        offset += shape_nbytes
        out, groups = struct.unpack_from("<II", blob, offset)
        offset += 8
        if len(shape) != 2 or int(axis) != 0 or int(shape[0]) != int(out):
            raise ValueError("unsupported NINT Metal geometry")
        if int(neuron_len) > int(groups) * int(groupsize):
            raise ValueError("inconsistent NINT neuron length")

        anchor_bytes = int(out) * 2
        if offset + 2 * anchor_bytes > len(blob):
            raise ValueError("truncated NINT neuron metadata")
        neuron_scale = np.frombuffer(
            blob, dtype="<f2", count=out, offset=offset
        ).astype(np.float32)
        offset += anchor_bytes
        neuron_min = np.frombuffer(
            blob, dtype="<f2", count=out, offset=offset
        ).astype(np.float32)
        offset += anchor_bytes
        if not np.isfinite(neuron_scale).all() or not np.isfinite(neuron_min).all():
            raise ValueError("NINT neuron metadata must be finite")

        metadata_count = int(out) * int(groups)
        values_per_row = int(groups) * int(groupsize)
        q_count = int(out) * values_per_row
        if adaptive:
            selectors, offset = _unpack_values(
                blob, offset, int(out), NINT_K_SELECTOR_BITS
            )
            row_sub_bits = selectors.astype(np.int16) + int(sub_bits) - 1
            if np.any(row_sub_bits < 1) or np.any(row_sub_bits > 8):
                raise ValueError("invalid NINT subgroup selector")
            sub_scale = np.empty((out, groups), dtype=np.uint8)
            sub_min = np.empty((out, groups), dtype=np.uint8)
            for selector in range(1 << NINT_K_SELECTOR_BITS):
                selected = np.flatnonzero(selectors == selector)
                if not selected.size:
                    continue
                row_bits = int(sub_bits) - 1 + selector
                count = int(selected.size) * int(groups)
                scales, offset = _unpack_values(blob, offset, count, row_bits)
                minima, offset = _unpack_values(blob, offset, count, row_bits)
                sub_scale[selected] = scales.reshape(selected.size, groups)
                sub_min[selected] = minima.reshape(selected.size, groups)

            q_selectors, offset = _unpack_values(
                blob, offset, int(out), NINT_Q_SELECTOR_BITS
            )
            row_q_bits = np.ascontiguousarray(q_selectors + 1, dtype=np.uint8)
            mixed_q = not bool(np.all(row_q_bits == row_q_bits[0]))
            if mixed_q:
                row_bytes = (
                    values_per_row * row_q_bits.astype(np.uint64) + 7
                ) // 8
                row_starts = np.empty(out, dtype=np.uint64)
                row_starts[0] = 0
                if int(out) > 1:
                    np.cumsum(row_bytes[:-1], out=row_starts[1:])
                if int(row_starts[-1]) > np.iinfo(np.uint32).max:
                    raise OverflowError("NINT packed stream exceeds Metal uint range")
                row_byte_offsets = np.ascontiguousarray(row_starts, dtype=np.uint32)
                row_bit_shifts = np.zeros(out, dtype=np.uint8)
                q_packed = np.zeros(
                    int(row_starts[-1] + row_bytes[-1]), dtype=np.uint8
                )
            else:
                row_bit_offsets = (
                    np.arange(out, dtype=np.uint64)
                    * values_per_row
                    * int(row_q_bits[0])
                )
                if int(row_bit_offsets[-1] >> np.uint64(3)) > np.iinfo(
                    np.uint32
                ).max:
                    raise OverflowError("NINT packed stream exceeds Metal uint range")
                row_byte_offsets = np.ascontiguousarray(
                    row_bit_offsets >> 3, dtype=np.uint32
                )
                row_bit_shifts = np.ascontiguousarray(
                    row_bit_offsets & 7, dtype=np.uint8
                )
                q_packed = np.empty(0, dtype=np.uint8)
            for selector in range(1 << NINT_Q_SELECTOR_BITS):
                selected = np.flatnonzero(q_selectors == selector)
                row_bits = selector + 1
                stream_nbytes = _packed_nbytes(
                    int(selected.size) * values_per_row, row_bits
                )
                if offset + stream_nbytes > len(blob):
                    raise ValueError("truncated NINT q stream")
                stream = np.frombuffer(
                    blob,
                    dtype=np.uint8,
                    count=stream_nbytes,
                    offset=offset,
                )
                if mixed_q:
                    for local_row, row in enumerate(selected):
                        _copy_packed_bit_range(
                            stream,
                            local_row * values_per_row * row_bits,
                            values_per_row * row_bits,
                            q_packed,
                            int(row_byte_offsets[int(row)]),
                        )
                elif selected.size:
                    q_packed = stream.copy()
                offset += stream_nbytes
        else:
            packed_metadata = _packed_nbytes(metadata_count, int(sub_bits))
            packed_q = _packed_nbytes(q_count, bits)
            old_metadata_dtype = (
                np.uint8 if (1 << int(sub_bits)) - 1 <= 255 else np.uint16
            )
            old_q_dtype = np.uint8 if (1 << bits) - 1 <= 255 else np.uint16
            old_tail = (
                2 * metadata_count * np.dtype(old_metadata_dtype).itemsize
                + q_count * np.dtype(old_q_dtype).itemsize
            )
            packed_tail = 2 * packed_metadata + packed_q
            remaining = len(blob) - offset
            if remaining == packed_tail:
                scales, offset = _unpack_values(
                    blob, offset, metadata_count, int(sub_bits)
                )
                minima, offset = _unpack_values(
                    blob, offset, metadata_count, int(sub_bits)
                )
                q_packed = np.frombuffer(
                    blob, dtype=np.uint8, count=packed_q, offset=offset
                ).copy()
                offset += packed_q
            elif remaining == old_tail:
                scales = np.frombuffer(
                    blob,
                    dtype=old_metadata_dtype,
                    count=metadata_count,
                    offset=offset,
                ).astype(np.uint8)
                offset += metadata_count * np.dtype(old_metadata_dtype).itemsize
                minima = np.frombuffer(
                    blob,
                    dtype=old_metadata_dtype,
                    count=metadata_count,
                    offset=offset,
                ).astype(np.uint8)
                offset += metadata_count * np.dtype(old_metadata_dtype).itemsize
                q = np.frombuffer(
                    blob, dtype=old_q_dtype, count=q_count, offset=offset
                )
                q_packed = _pack_values(q, bits)
                offset += q_count * np.dtype(old_q_dtype).itemsize
            else:
                raise ValueError("invalid legacy NINT payload length")
            sub_scale = scales.reshape(out, groups)
            sub_min = minima.reshape(out, groups)
            row_q_bits = np.full(out, bits, dtype=np.uint8)
            row_bit_offsets = (
                np.arange(out, dtype=np.uint64) * values_per_row * bits
            )
            if (
                row_bit_offsets.size
                and int(row_bit_offsets[-1] >> np.uint64(3))
                > np.iinfo(np.uint32).max
            ):
                raise OverflowError("NINT packed stream exceeds Metal uint range")
            row_byte_offsets = np.ascontiguousarray(
                row_bit_offsets >> 3, dtype=np.uint32
            )
            row_bit_shifts = np.ascontiguousarray(
                row_bit_offsets & 7, dtype=np.uint8
            )
        if offset != len(blob):
            raise ValueError(f"invalid NINT trailing bytes: {len(blob) - offset}")
        q_packed = np.concatenate(
            (np.ascontiguousarray(q_packed, dtype=np.uint8), np.zeros(4, np.uint8))
        )
        row_q_layout = np.ascontiguousarray(
            row_q_bits | (row_bit_shifts << 4), dtype=np.uint8
        )
        return cls(
            q_packed=_mlx_buffer(q_packed),
            row_q_layout=mx.array(row_q_layout),
            row_q_byte_offsets=mx.array(row_byte_offsets),
            sub_scale=mx.array(_metadata_u8(sub_scale, "sub_scale")),
            sub_min=mx.array(_metadata_u8(sub_min, "sub_min")),
            neuron_scale=mx.array(neuron_scale),
            neuron_min=mx.array(neuron_min),
            bits=bits,
            groupsize=int(groupsize),
            out=int(out),
            groups=int(groups),
            neuron_len=int(neuron_len),
        )

    @property
    def packed_nbytes(self) -> int:
        arrays = (
            self.q_packed,
            self.row_q_layout,
            self.row_q_byte_offsets,
            self.sub_scale,
            self.sub_min,
            self.neuron_scale,
            self.neuron_min,
        )
        return sum(int(value.size) * int(value.itemsize) for value in arrays)

    @property
    def row_q_bits(self) -> mx.array:
        return self.row_q_layout & mx.array(15, dtype=mx.uint8)

    @property
    def has_uniform_q_bits(self) -> bool:
        values = np.asarray(self.row_q_layout) & 15
        return bool(values.size == 0 or np.all(values == values[0]))


def _floating_input(value: mx.array | np.ndarray) -> mx.array:
    result = value if isinstance(value, mx.array) else mx.array(value)
    if result.dtype not in (mx.float16, mx.float32):
        result = result.astype(mx.float16)
    return result


def _prepare_matmul_input(
    weight: MetalNintWeight,
    value: mx.array | np.ndarray,
) -> tuple[mx.array, tuple[int, ...], int]:
    source = _floating_input(value)
    if source.ndim < 1 or int(source.shape[-1]) != weight.neuron_len:
        raise ValueError("NINT matmul input width does not match packed weight")
    prefix = tuple(int(size) for size in source.shape[:-1])
    rows = int(source.size) // weight.neuron_len
    return mx.contiguous(source.reshape(rows, weight.neuron_len)), prefix, rows


def _packed_matmul(
    weight: MetalNintWeight,
    value: mx.array | np.ndarray,
) -> mx.array:
    source, prefix, rows = _prepare_matmul_input(weight, value)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source.dtype)
    tile_rows = 1 if rows == 1 else (rows if rows <= 16 else 8)
    row_tiles = (rows + tile_rows - 1) // tile_rows
    outputs_per_threadgroup = 16
    threads_per_threadgroup = 256
    output_threadgroups = (
        weight.out + outputs_per_threadgroup - 1
    ) // outputs_per_threadgroup
    grid_x = output_threadgroups * threads_per_threadgroup
    if grid_x > (1 << 31) - 1:
        raise OverflowError("NINT Metal launch grid exceeds MLX limits")
    result = _NINT_MATMUL_KERNEL(
        inputs=[
            weight.q_packed,
            weight.row_q_layout,
            weight.row_q_byte_offsets,
            weight.sub_scale,
            weight.sub_min,
            weight.neuron_scale,
            weight.neuron_min,
            source,
            _NINT_UNROUTED_SENTINEL,
            _NINT_UNROUTED_SENTINEL,
        ],
        template=[
            ("T", source.dtype),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("K", weight.neuron_len),
            ("OUT", weight.out),
            ("M", rows),
            ("TILE_M", tile_rows),
            ("ROUTED", 0),
            ("ROUTES", 1),
            ("SHARED_INPUT", 0),
            ("EXPERT_MAP_SIZE", 1),
            ("LOCAL_EXPERTS", 1),
            ("OUT_PER_EXPERT", weight.out),
            ("LOGICAL_OUT", weight.out),
        ],
        grid=(grid_x, row_tiles, 1),
        threadgroup=(threads_per_threadgroup, 1, 1),
        output_shapes=[(rows, weight.out)],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((*prefix, weight.out))


def nint_routed_matmul(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
    expert_ids: mx.array | np.ndarray,
    expert_map: mx.array | np.ndarray,
    out_per_expert: int,
) -> mx.array:
    """Apply one packed NINT cohort through the ordinary NINT matmul kernel."""

    if out_per_expert <= 0 or weight.out % int(out_per_expert):
        raise ValueError("NINT routed output width is inconsistent")
    ids = expert_ids if isinstance(expert_ids, mx.array) else mx.array(expert_ids)
    mapping = expert_map if isinstance(expert_map, mx.array) else mx.array(expert_map)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    if mapping.dtype not in (mx.int32, mx.uint32):
        mapping = mapping.astype(mx.int32)
    if ids.ndim != 2 or mapping.ndim != 1:
        raise ValueError("NINT routing metadata must contain IDs and a one-dimensional map")
    ids = mx.contiguous(ids.astype(mx.int32))
    mapping = mx.contiguous(mapping.astype(mx.int32))
    tokens, routes = (int(value) for value in ids.shape)

    source = _floating_input(x)
    shared_input = source.ndim == 2 and tuple(int(value) for value in source.shape) == (
        tokens,
        weight.neuron_len,
    )
    if not shared_input and (
        source.ndim != 3
        or tuple(int(value) for value in source.shape)
        != (tokens, routes, weight.neuron_len)
    ):
        raise ValueError("NINT routed input must have [tokens,K] or [tokens,routes,K]")
    source = mx.contiguous(source)
    route_count = tokens * routes
    if route_count > (1 << 31) - 1:
        raise OverflowError("NINT routed Metal route count exceeds MLX limits")
    if int(mapping.size) > (1 << 31) - 1:
        raise OverflowError("NINT routed Metal expert map exceeds MLX limits")
    if route_count == 0:
        return mx.zeros((tokens, routes, out_per_expert), dtype=source.dtype)

    threads_per_threadgroup = 256
    outputs_per_threadgroup = 16
    output_threadgroups = (
        int(out_per_expert) + outputs_per_threadgroup - 1
    ) // outputs_per_threadgroup
    grid_x = output_threadgroups * threads_per_threadgroup
    if grid_x > (1 << 31) - 1:
        raise OverflowError("NINT routed Metal launch grid exceeds MLX limits")
    result = _NINT_MATMUL_KERNEL(
        inputs=[
            weight.q_packed,
            weight.row_q_layout,
            weight.row_q_byte_offsets,
            weight.sub_scale,
            weight.sub_min,
            weight.neuron_scale,
            weight.neuron_min,
            source,
            ids,
            mapping,
        ],
        template=[
            ("T", source.dtype),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("K", weight.neuron_len),
            ("OUT", weight.out),
            ("M", route_count),
            ("TILE_M", 1),
            ("ROUTED", 1),
            ("ROUTES", routes),
            ("SHARED_INPUT", int(shared_input)),
            ("EXPERT_MAP_SIZE", int(mapping.size)),
            ("LOCAL_EXPERTS", weight.out // int(out_per_expert)),
            ("OUT_PER_EXPERT", int(out_per_expert)),
            ("LOGICAL_OUT", int(out_per_expert)),
        ],
        grid=(grid_x, route_count, 1),
        threadgroup=(threads_per_threadgroup, 1, 1),
        output_shapes=[(route_count, int(out_per_expert))],
        output_dtypes=[source.dtype],
    )[0]
    return result.reshape((tokens, routes, int(out_per_expert)))


def nint_gemv(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    _, _, rows = _prepare_matmul_input(weight, x)
    if rows != 1:
        raise ValueError("NINT GEMV requires exactly one input row")
    return _packed_matmul(weight, x)


def nint_mmq(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    _, _, rows = _prepare_matmul_input(weight, x)
    if not 2 <= rows <= 16:
        raise ValueError("NINT MMQ requires 2 to 16 input rows")
    return _packed_matmul(weight, x)


def nint_gemm(weight: MetalNintWeight, x: mx.array | np.ndarray) -> mx.array:
    return _packed_matmul(weight, x)


def nint_embedding(
    weight: MetalNintWeight,
    token_ids: mx.array | np.ndarray,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    if dtype not in (mx.float16, mx.float32):
        raise TypeError("NINT row decode output must be float16 or float32")
    ids = token_ids if isinstance(token_ids, mx.array) else mx.array(token_ids)
    if ids.dtype not in (mx.int32, mx.uint32):
        ids = ids.astype(mx.int32)
    shape = tuple(int(value) for value in ids.shape)
    count = int(ids.size)
    if count == 0:
        return mx.zeros((*shape, weight.neuron_len), dtype=dtype)
    ids = mx.contiguous(ids.reshape(count))
    size = count * weight.neuron_len
    result = _NINT_ROW_DECODE_KERNEL(
        inputs=[
            weight.q_packed,
            weight.row_q_layout,
            weight.row_q_byte_offsets,
            weight.sub_scale,
            weight.sub_min,
            weight.neuron_scale,
            weight.neuron_min,
            ids,
        ],
        template=[
            ("T", dtype),
            ("GS", weight.groupsize),
            ("NG", weight.groups),
            ("K", weight.neuron_len),
            ("COUNT", count),
        ],
        grid=(size, 1, 1),
        threadgroup=(min(256, max(1, size)), 1, 1),
        output_shapes=[(count, weight.neuron_len)],
        output_dtypes=[dtype],
    )[0]
    return result.reshape((*shape, weight.neuron_len))


def nint_dequantize(
    weight: MetalNintWeight,
    *,
    dtype: mx.Dtype = mx.float16,
) -> mx.array:
    return nint_embedding(weight, mx.arange(weight.out, dtype=mx.int32), dtype=dtype)


def nint_dequantize_matmul(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    source, prefix, rows = _prepare_matmul_input(weight, x)
    if rows == 0:
        return mx.zeros((*prefix, weight.out), dtype=source.dtype)
    dtype = mx.float16 if source.dtype == mx.float16 else mx.float32
    result = source.astype(dtype) @ nint_dequantize(weight, dtype=dtype).T
    return result.reshape((*prefix, weight.out))


def nint_backward_input(
    weight: MetalNintWeight,
    output_gradient: mx.array | np.ndarray,
) -> mx.array:
    gradient = _floating_input(output_gradient)
    if gradient.ndim < 1 or int(gradient.shape[-1]) != weight.out:
        raise ValueError("NINT output-gradient width does not match packed weight")
    prefix = tuple(int(value) for value in gradient.shape[:-1])
    rows = int(gradient.size) // weight.out
    gradient = mx.contiguous(gradient.reshape(rows, weight.out))
    dense = nint_dequantize(weight, dtype=gradient.dtype)
    return (gradient @ dense).reshape((*prefix, weight.neuron_len))


def _nint_matmul_with_vjp(
    weight: MetalNintWeight,
    source: mx.array,
    *,
    dequantize_threshold: int | None,
) -> mx.array:
    @mx.custom_function
    def operation(value: mx.array) -> mx.array:
        rows = int(value.size) // int(value.shape[-1])
        if dequantize_threshold is not None and rows >= int(dequantize_threshold):
            return nint_dequantize_matmul(weight, value)
        return _packed_matmul(weight, value)

    @operation.vjp
    def operation_vjp(primals, cotangent, output):
        del primals, output
        return nint_backward_input(weight, cotangent)

    return operation(source)


def nint_matmul(
    weight: MetalNintWeight,
    x: mx.array | np.ndarray,
    *,
    dequantize_threshold: int | None = 64,
) -> mx.array:
    source = x if isinstance(x, mx.array) else mx.array(x)
    if source.ndim < 1:
        raise ValueError("NINT matmul input must have at least one dimension")
    return _nint_matmul_with_vjp(
        weight,
        source,
        dequantize_threshold=dequantize_threshold,
    )


def nint_swiglu(
    gate: MetalNintWeight,
    up: MetalNintWeight,
    x: mx.array | np.ndarray,
) -> mx.array:
    gate_value = nint_matmul(gate, x)
    up_value = nint_matmul(up, x)
    if gate_value.shape != up_value.shape:
        raise ValueError("NINT SwiGLU gate/up projections must have matching shapes")
    return mx.sigmoid(gate_value) * gate_value * up_value


__all__ = [
    "MetalNintWeight",
    "nint_backward_input",
    "nint_dequantize",
    "nint_dequantize_matmul",
    "nint_embedding",
    "nint_gemm",
    "nint_gemv",
    "nint_matmul",
    "nint_mmq",
    "nint_routed_matmul",
    "nint_swiglu",
]
