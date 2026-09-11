#include "mlx_nint.h"

#include "mlx_staging_allocator.h"

#include <mlx/primitives.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;
using mlx::core::array;

constexpr std::uint8_t kAdaptiveStorageFlag = 0x80;
constexpr int kSubSelectorBits = 2;
constexpr int kQSelectorBits = 3;

constexpr const char* kNintHeader = R"METAL(
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
)METAL";

constexpr const char* kNintMatmul = R"METAL(
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
)METAL";

constexpr const char* kNintDequantize = R"METAL(
    uint output_index = thread_position_in_grid.x;
    if (output_index >= uint(OUT) * uint(K)) {
        return;
    }
    uint output = output_index / uint(K);
    uint input_index = output_index - output * uint(K);
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
)METAL";

constexpr const char* kNintEmbedding = R"METAL(
    uint output_index = thread_position_in_grid.x;
    if (output_index >= uint(COUNT) * uint(K)) {
        return;
    }
    uint token_position = output_index / uint(K);
    uint input_index = output_index - token_position * uint(K);
    uint output = uint(token_ids[token_position]);
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
)METAL";

class BlobCursor {
public:
    explicit BlobCursor(std::span<const std::uint8_t> blob)
        : blob_(blob) {}

    template <typename T>
    T scalar(const char* name) {
        require(sizeof(T), name);
        T value{};
        std::memcpy(&value, blob_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return value;
    }

    detail::StagingVector<std::uint8_t> bytes(
        std::size_t count,
        const char* name) {
        require(count, name);
        detail::StagingVector<std::uint8_t> result(
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_),
            blob_.begin() + static_cast<std::ptrdiff_t>(offset_ + count));
        offset_ += count;
        return result;
    }

    std::size_t remaining() const noexcept {
        return blob_.size() - offset_;
    }

private:
    void require(std::size_t count, const char* name) {
        if (count > blob_.size() - offset_) {
            throw std::runtime_error(std::string("truncated NINT ") + name);
        }
    }

    std::span<const std::uint8_t> blob_;
    std::size_t offset_ = 0;
};

std::size_t packed_size(std::size_t count, int bits) {
    if (bits <= 0 || bits > 8 ||
        count > (std::numeric_limits<std::size_t>::max() - 7) /
            static_cast<std::size_t>(bits)) {
        throw std::runtime_error("invalid NINT packed bit count");
    }
    return (count * static_cast<std::size_t>(bits) + 7) / 8;
}

std::uint8_t packed_value(
    std::span<const std::uint8_t> data,
    std::size_t index,
    int bits) {
    const auto bit_index = index * static_cast<std::size_t>(bits);
    const auto byte_index = bit_index / 8;
    const auto shift = static_cast<unsigned>(bit_index & 7);
    std::uint32_t value = data[byte_index];
    if (shift + static_cast<unsigned>(bits) > 8) {
        value |= static_cast<std::uint32_t>(data[byte_index + 1]) << 8;
    }
    return static_cast<std::uint8_t>(
        (value >> shift) & ((1u << bits) - 1u));
}

detail::StagingVector<std::uint8_t> unpack_values(
    std::span<const std::uint8_t> data,
    std::size_t count,
    int bits) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        result[index] = packed_value(data, index, bits);
    }
    return result;
}

detail::StagingVector<std::uint8_t> pack_values(
    std::span<const std::uint8_t> values,
    int bits) {
    detail::StagingVector<std::uint8_t> result(
        packed_size(values.size(), bits), 0);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto bit_index = index * static_cast<std::size_t>(bits);
        for (int bit = 0; bit < bits; ++bit) {
            if ((values[index] >> bit) & 1u) {
                const auto target = bit_index + static_cast<std::size_t>(bit);
                result[target / 8] |=
                    static_cast<std::uint8_t>(1u << (target & 7));
            }
        }
    }
    return result;
}

void copy_packed_bit_range(
    std::span<const std::uint8_t> source,
    std::size_t source_bit,
    std::size_t bit_count,
    std::span<std::uint8_t> destination) {
    const auto byte_count = packed_size(bit_count, 1);
    if (destination.size() != byte_count) {
        throw std::logic_error("NINT aligned row destination size mismatch");
    }
    const auto source_byte = source_bit / 8;
    const auto shift = static_cast<unsigned>(source_bit & 7u);
    for (std::size_t byte = 0; byte < byte_count; ++byte) {
        const auto index = source_byte + byte;
        std::uint16_t packed = source[index];
        if (shift != 0 && index + 1 < source.size()) {
            packed |= static_cast<std::uint16_t>(source[index + 1]) << 8u;
        }
        destination[byte] = static_cast<std::uint8_t>(packed >> shift);
    }
    if ((bit_count & 7u) != 0) {
        destination.back() &= static_cast<std::uint8_t>(
            (1u << (bit_count & 7u)) - 1u);
    }
}

detail::StagingVector<std::uint8_t> read_old_values(
    BlobCursor& cursor,
    std::size_t count,
    int storage_bytes,
    const char* name) {
    detail::StagingVector<std::uint8_t> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t value = storage_bytes == 1
            ? cursor.scalar<std::uint8_t>(name)
            : cursor.scalar<std::uint16_t>(name);
        if (value > 255) {
            throw std::runtime_error(
                std::string("NINT ") + name + " exceeds uint8 storage");
        }
        result[index] = static_cast<std::uint8_t>(value);
    }
    return result;
}

float half_to_float(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const auto exponent = static_cast<unsigned>((bits >> 10) & 0x1fu);
    const auto mantissa = static_cast<unsigned>(bits & 0x03ffu);
    float value = 0.0f;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exponent == 31) {
        value = mantissa == 0
            ? std::numeric_limits<float>::infinity()
            : std::numeric_limits<float>::quiet_NaN();
    } else {
        value = std::ldexp(
            1.0f + static_cast<float>(mantissa) / 1024.0f,
            static_cast<int>(exponent) - 15);
    }
    return negative ? -value : value;
}

template <typename T, typename Allocator>
array make_array(
    const std::vector<T, Allocator>& values,
    Shape shape) {
    return array(values.begin(), std::move(shape));
}

std::int32_t checked_shape(std::int64_t value, const char* name) {
    if (value <= 0 ||
        value > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error(std::string("invalid NINT ") + name);
    }
    return static_cast<std::int32_t>(value);
}

mlx::core::fast::CustomKernelFunction make_nint_matmul_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_matmul",
        {
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
        },
        {"y"},
        kNintMatmul,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_matmul_kernel() {
    static const auto kernel = make_nint_matmul_kernel();
    return kernel;
}

const array& nint_unrouted_sentinel() {
    static const auto sentinel = mlx::core::zeros(
        Shape{1}, mlx::core::int32);
    return sentinel;
}

mlx::core::fast::CustomKernelFunction make_nint_dequantize_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_dequantize",
        {
            "q_packed",
            "row_q_layout",
            "row_q_byte_offsets",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
        },
        {"y"},
        kNintDequantize,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_dequantize_kernel() {
    static const auto kernel = make_nint_dequantize_kernel();
    return kernel;
}

mlx::core::fast::CustomKernelFunction make_nint_embedding_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    return mlx::core::fast::metal_kernel(
        "mfq_cpp_nint_embedding",
        {
            "q_packed",
            "row_q_layout",
            "row_q_byte_offsets",
            "sub_scale",
            "sub_min",
            "neuron_scale",
            "neuron_min",
            "token_ids",
        },
        {"y"},
        kNintEmbedding,
        kNintHeader,
        true,
        false,
        options);
}

const mlx::core::fast::CustomKernelFunction& nint_embedding_kernel() {
    static const auto kernel = make_nint_embedding_kernel();
    return kernel;
}

} // namespace

bool is_nint_dtype(std::string_view dtype) noexcept {
    return dtype == "NINT";
}

MlxNintWeight::MlxNintWeight(
    array q_packed,
    array sub_scale,
    array sub_min,
    array neuron_scale,
    array neuron_min,
    array row_q_layout,
    array row_q_byte_offsets,
    int bits,
    int group_size,
    int groups,
    int input_size,
    int output_size,
    bool uniform_q_bits)
    : q_packed_(std::move(q_packed)),
      sub_scale_(std::move(sub_scale)),
      sub_min_(std::move(sub_min)),
      neuron_scale_(std::move(neuron_scale)),
      neuron_min_(std::move(neuron_min)),
      row_q_layout_(std::move(row_q_layout)),
      row_q_byte_offsets_(std::move(row_q_byte_offsets)),
      bits_(bits),
      group_size_(group_size),
      groups_(groups),
      input_size_(input_size),
      output_size_(output_size),
      uniform_q_bits_(uniform_q_bits) {}

MlxNintWeight MlxNintWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    BlobCursor cursor(blob);
    const auto raw_bits = cursor.scalar<std::uint8_t>("bits");
    const bool adaptive_storage = (raw_bits & kAdaptiveStorageFlag) != 0;
    const int bits = raw_bits & ~kAdaptiveStorageFlag;
    const int sub_bits = cursor.scalar<std::uint8_t>("sub bits");
    const int group_size = cursor.scalar<std::int32_t>("group size");
    const int axis = cursor.scalar<std::int32_t>("axis");
    const int input_size = cursor.scalar<std::int32_t>("input size");
    const auto dimensions = cursor.scalar<std::uint32_t>("dimension count");
    if (bits <= 0 || bits > 8 || sub_bits <= 0 || sub_bits > 8 ||
        group_size <= 0 || input_size <= 0 || dimensions != 2 || axis != 0) {
        throw std::runtime_error("unsupported NINT Metal dimensions");
    }

    std::vector<std::int64_t> shape(dimensions);
    for (auto& value : shape) {
        value = cursor.scalar<std::int64_t>("shape");
    }
    const auto output_size = cursor.scalar<std::uint32_t>("output size");
    const auto groups = cursor.scalar<std::uint32_t>("group count");
    if (shape[0] != output_size || shape[1] != input_size || groups == 0 ||
        static_cast<std::uint64_t>(input_size) >
            static_cast<std::uint64_t>(groups) * group_size) {
        throw std::runtime_error("inconsistent NINT Metal dimensions");
    }

    detail::StagingVector<float> neuron_scale(output_size);
    detail::StagingVector<float> neuron_min(output_size);
    for (auto& value : neuron_scale) {
        value = half_to_float(cursor.scalar<std::uint16_t>("neuron scale"));
    }
    for (auto& value : neuron_min) {
        value = half_to_float(cursor.scalar<std::uint16_t>("neuron minimum"));
    }
    if (!std::all_of(
            neuron_scale.begin(), neuron_scale.end(),
            [](float value) { return std::isfinite(value); }) ||
        !std::all_of(
            neuron_min.begin(), neuron_min.end(),
            [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("NINT neuron metadata must be finite");
    }

    const auto metadata_count =
        static_cast<std::size_t>(output_size) * groups;
    const auto values_per_row =
        static_cast<std::size_t>(groups) * group_size;
    const auto q_count =
        static_cast<std::size_t>(output_size) * values_per_row;
    const auto packed_metadata_bytes = packed_size(metadata_count, sub_bits);
    const auto packed_q_bytes = packed_size(q_count, bits);
    const auto packed_tail = 2 * packed_metadata_bytes + packed_q_bytes;
    const auto old_tail = 2 * metadata_count + q_count;
    const bool old_unpacked_storage =
        !adaptive_storage && cursor.remaining() == old_tail;

    detail::StagingVector<std::uint8_t> sub_scale;
    detail::StagingVector<std::uint8_t> sub_min;
    if (adaptive_storage) {
        const auto selector_bytes = packed_size(output_size, kSubSelectorBits);
        const auto selectors = unpack_values(
            cursor.bytes(selector_bytes, "sub-bit selectors"),
            output_size,
            kSubSelectorBits);
        sub_scale.resize(metadata_count);
        sub_min.resize(metadata_count);
        for (int selector = 0; selector < (1 << kSubSelectorBits); ++selector) {
            const int row_bits = sub_bits - 1 + selector;
            const auto selected_rows = static_cast<std::size_t>(std::count(
                selectors.begin(), selectors.end(),
                static_cast<std::uint8_t>(selector)));
            if (selected_rows == 0) {
                continue;
            }
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT subgroup width");
            }
            const auto selected_values = selected_rows * groups;
            const auto stream_bytes = packed_size(selected_values, row_bits);
            const auto scales = unpack_values(
                cursor.bytes(stream_bytes, "sub scale"),
                selected_values,
                row_bits);
            const auto minima = unpack_values(
                cursor.bytes(stream_bytes, "sub minimum"),
                selected_values,
                row_bits);
            std::size_t local_row = 0;
            for (std::size_t row = 0; row < output_size; ++row) {
                if (selectors[row] != selector) {
                    continue;
                }
                const auto source = local_row * groups;
                const auto destination = row * groups;
                std::copy_n(scales.begin() + source, groups, sub_scale.begin() + destination);
                std::copy_n(minima.begin() + source, groups, sub_min.begin() + destination);
                ++local_row;
            }
        }
    } else if (old_unpacked_storage) {
        sub_scale = read_old_values(cursor, metadata_count, 1, "sub scale");
        sub_min = read_old_values(cursor, metadata_count, 1, "sub minimum");
    } else {
        if (cursor.remaining() != packed_tail) {
            throw std::runtime_error("invalid NINT packed payload length");
        }
        sub_scale = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub scale"),
            metadata_count,
            sub_bits);
        sub_min = unpack_values(
            cursor.bytes(packed_metadata_bytes, "sub minimum"),
            metadata_count,
            sub_bits);
    }

    detail::StagingVector<std::uint8_t> q_packed;
    detail::StagingVector<std::uint8_t> row_q_layout(output_size);
    detail::StagingVector<std::uint32_t> row_q_byte_offsets(output_size);
    if (adaptive_storage) {
        const auto selector_bytes = packed_size(output_size, kQSelectorBits);
        const auto selectors = unpack_values(
            cursor.bytes(selector_bytes, "q-bit selectors"),
            output_size,
            kQSelectorBits);
        std::size_t aligned_bytes = 0;
        for (std::size_t row = 0; row < output_size; ++row) {
            const int row_bits = static_cast<int>(selectors[row]) + 1;
            if (aligned_bytes > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "NINT packed stream exceeds Metal uint range");
            }
            row_q_layout[row] = static_cast<std::uint8_t>(row_bits);
            row_q_byte_offsets[row] =
                static_cast<std::uint32_t>(aligned_bytes);
            const auto row_bytes = packed_size(values_per_row, row_bits);
            if (row_bytes > std::numeric_limits<std::size_t>::max() - aligned_bytes) {
                throw std::runtime_error("NINT aligned q stream is too large");
            }
            aligned_bytes += row_bytes;
        }
        q_packed.resize(aligned_bytes, 0);
        for (int selector = 0; selector < (1 << kQSelectorBits); ++selector) {
            const int row_bits = selector + 1;
            const auto selected_rows = static_cast<std::size_t>(std::count(
                selectors.begin(), selectors.end(),
                static_cast<std::uint8_t>(selector)));
            const auto row_bit_count = values_per_row * row_bits;
            const auto stream_bytes = packed_size(
                selected_rows * values_per_row,
                row_bits);
            const auto stream = cursor.bytes(stream_bytes, "q values");
            std::size_t local_row = 0;
            for (std::size_t row = 0; row < output_size; ++row) {
                if (selectors[row] != selector) {
                    continue;
                }
                const auto destination = static_cast<std::size_t>(
                    row_q_byte_offsets[row]);
                const auto row_bytes = packed_size(values_per_row, row_bits);
                copy_packed_bit_range(
                    stream,
                    local_row * row_bit_count,
                    row_bit_count,
                    std::span<std::uint8_t>(
                        q_packed.data() + destination,
                        row_bytes));
                ++local_row;
            }
        }
    } else {
        q_packed = old_unpacked_storage
            ? pack_values(
                  read_old_values(cursor, q_count, 1, "quantized value"),
                  bits)
            : cursor.bytes(packed_q_bytes, "q values");
        const auto row_bit_count = values_per_row * bits;
        for (std::size_t row = 0; row < output_size; ++row) {
            const auto bit_offset = row * row_bit_count;
            const auto byte_offset = bit_offset / 8;
            if (byte_offset > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "NINT packed stream exceeds Metal uint range");
            }
            row_q_layout[row] = static_cast<std::uint8_t>(
                bits | ((bit_offset & 7u) << 4u));
            row_q_byte_offsets[row] = static_cast<std::uint32_t>(byte_offset);
        }
    }
    if (cursor.remaining() != 0) {
        throw std::runtime_error("trailing bytes in NINT tensor");
    }
    q_packed.insert(q_packed.end(), 4, 0);

    const bool uniform_q_bits = std::all_of(
        row_q_layout.begin(), row_q_layout.end(),
        [bits](std::uint8_t value) { return (value & 15u) == bits; });
    return MlxNintWeight(
        make_array(q_packed, Shape{checked_shape(q_packed.size(), "q bytes")}),
        make_array(
            sub_scale,
            Shape{checked_shape(output_size, "output size"), checked_shape(groups, "groups")}),
        make_array(
            sub_min,
            Shape{checked_shape(output_size, "output size"), checked_shape(groups, "groups")}),
        make_array(neuron_scale, Shape{checked_shape(output_size, "output size")}),
        make_array(neuron_min, Shape{checked_shape(output_size, "output size")}),
        make_array(row_q_layout, Shape{checked_shape(output_size, "output size")}),
        make_array(row_q_byte_offsets, Shape{checked_shape(output_size, "output size")}),
        bits,
        group_size,
        static_cast<int>(groups),
        input_size,
        static_cast<int>(output_size),
        uniform_q_bits);
}

array MlxNintWeight::matmul(const array& input) const {
    return matmul_impl(input, nullptr);
}

array MlxNintWeight::matmul_add(
    const array& input,
    const array& residual) const {
    return matmul_impl(input, &residual);
}

array MlxNintWeight::matmul_impl(
    const array& input,
    const array* residual) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("NINT input width does not match packed weight");
    }
    std::int64_t rows = 1;
    Shape output_shape = input.shape();
    for (std::size_t index = 0; index + 1 < input.ndim(); ++index) {
        rows *= input.shape(static_cast<int>(index));
    }
    if (rows <= 0 || rows > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("unsupported NINT input row count");
    }
    output_shape.back() = output_size_;
    if (residual != nullptr && residual->shape() != output_shape) {
        throw std::runtime_error("NINT residual shape does not match output");
    }

    auto source = input;
    if (rows >= 64 && source.dtype() == mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    } else if (source.dtype() != mlx::core::float16 &&
               source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<std::int32_t>(rows), input_size_});

    if (rows >= 64) {
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dequantize(mlx::core::float16)));
        result = mlx::core::reshape(std::move(result), std::move(output_shape));
        return residual != nullptr ? result + *residual : result;
    }

    const int tile_rows = rows == 1
        ? 1
        : (rows <= 16 ? static_cast<int>(rows) : 8);
    const auto row_tiles = (rows + tile_rows - 1) / tile_rows;
    constexpr std::int64_t outputs_per_threadgroup = 16;
    constexpr std::int64_t threads_per_threadgroup = 256;
    const auto output_threadgroups =
        (static_cast<std::int64_t>(output_size_) + outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NINT Metal grid exceeds MLX limits");
    }
    auto outputs = nint_matmul_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            source,
            nint_unrouted_sentinel(),
            nint_unrouted_sentinel(),
        },
        {Shape{static_cast<std::int32_t>(rows), output_size_}},
        {source.dtype()},
        {static_cast<int>(grid_x), static_cast<int>(row_tiles), 1},
        {static_cast<int>(threads_per_threadgroup), 1, 1},
        {
            {"T", source.dtype()},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"OUT", output_size_},
            {"M", static_cast<int>(rows)},
            {"TILE_M", tile_rows},
            {"ROUTED", 0},
            {"ROUTES", 1},
            {"SHARED_INPUT", 0},
            {"EXPERT_MAP_SIZE", 1},
            {"LOCAL_EXPERTS", 1},
            {"OUT_PER_EXPERT", output_size_},
            {"LOGICAL_OUT", output_size_},
        },
        std::nullopt,
        false,
        {});
    auto result = mlx::core::reshape(outputs.front(), std::move(output_shape));
    return residual != nullptr ? result + *residual : result;
}

array MlxNintWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    int out_per_expert) const {
    if (out_per_expert <= 0 || output_size_ % out_per_expert != 0) {
        throw std::invalid_argument(
            "NINT routed output width is inconsistent");
    }
    if (expert_ids.ndim() != 2 || expert_map.ndim() != 1 ||
        expert_ids.dtype() != mlx::core::int32 ||
        expert_map.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "NINT routing metadata must be contiguous int32 IDs");
    }
    const int tokens = expert_ids.shape(0);
    const int routes = expert_ids.shape(1);
    const bool shared_input = input.ndim() == 2 &&
        input.shape(0) == tokens && input.shape(1) == input_size_;
    if (!shared_input && (
        input.ndim() != 3 || input.shape(0) != tokens ||
        input.shape(1) != routes || input.shape(2) != input_size_)) {
        throw std::invalid_argument(
            "NINT routed input must be [tokens,K] or [tokens,routes,K]");
    }
    const auto route_count =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(routes);
    if (route_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("NINT route count exceeds MLX limits");
    }
    const Shape output_shape{tokens, routes, out_per_expert};
    if (route_count == 0) {
        const auto output_dtype = input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16;
        return mlx::core::zeros(output_shape, output_dtype);
    }

    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::contiguous(std::move(source));
    auto ids = mlx::core::contiguous(expert_ids);
    auto map = mlx::core::contiguous(expert_map);

    constexpr int threads_per_threadgroup = 256;
    constexpr int outputs_per_threadgroup = 16;
    const auto output_threadgroups =
        (static_cast<std::int64_t>(out_per_expert)
            + outputs_per_threadgroup - 1) / outputs_per_threadgroup;
    const auto grid_x = output_threadgroups * threads_per_threadgroup;
    if (grid_x > std::numeric_limits<int>::max()) {
        throw std::runtime_error("NINT routed Metal grid exceeds MLX limits");
    }
    auto outputs = nint_matmul_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            source,
            ids,
            map,
        },
        {Shape{static_cast<int>(route_count), out_per_expert}},
        {source.dtype()},
        {
            static_cast<int>(grid_x),
            static_cast<int>(route_count),
            1,
        },
        {threads_per_threadgroup, 1, 1},
        {
            {"T", source.dtype()},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"OUT", output_size_},
            {"M", static_cast<int>(route_count)},
            {"TILE_M", 1},
            {"ROUTED", 1},
            {"ROUTES", routes},
            {"SHARED_INPUT", static_cast<int>(shared_input)},
            {"EXPERT_MAP_SIZE", expert_map.shape(0)},
            {"LOCAL_EXPERTS", output_size_ / out_per_expert},
            {"OUT_PER_EXPERT", out_per_expert},
            {"LOGICAL_OUT", out_per_expert},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        output_shape);
}

std::optional<array> MlxNintWeight::grouped_row_matmul(
    const array& input,
    int group_count) const {
    if (group_count <= 0 || input.ndim() < 2 ||
        input.shape(-2) != group_count || input.shape(-1) != input_size_ ||
        output_size_ % group_count != 0) {
        throw std::runtime_error("NINT grouped-row input geometry mismatch");
    }

    std::int64_t rows = 1;
    for (std::size_t dimension = 0;
         dimension + 2 < input.ndim();
         ++dimension) {
        const auto dimension_size =
            input.shape(static_cast<int>(dimension));
        if (dimension_size != 0 &&
            rows > std::numeric_limits<int>::max() / dimension_size) {
            throw std::runtime_error(
                "NINT grouped-row prefix exceeds MLX limits");
        }
        rows *= dimension_size;
    }

    Shape output_shape = input.shape();
    output_shape.back() = output_size_ / group_count;
    if (rows == 0) {
        const auto output_dtype = input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16;
        return mlx::core::zeros(output_shape, output_dtype);
    }

    std::vector<std::int32_t> ids(
        static_cast<std::size_t>(rows) * group_count);
    for (std::int64_t row = 0; row < rows; ++row) {
        for (int group = 0; group < group_count; ++group) {
            ids[static_cast<std::size_t>(row) * group_count + group] =
                group;
        }
    }
    std::vector<std::int32_t> identity_map(
        static_cast<std::size_t>(group_count));
    for (int group = 0; group < group_count; ++group) {
        identity_map[static_cast<std::size_t>(group)] = group;
    }

    const auto source = mlx::core::reshape(
        input,
        Shape{
            static_cast<int>(rows),
            group_count,
            input_size_,
        });
    const array expert_ids(
        ids.begin(),
        Shape{static_cast<int>(rows), group_count});
    const array expert_map(
        identity_map.begin(),
        Shape{group_count});
    auto output = routed_matmul(
        source,
        expert_ids,
        expert_map,
        output_size_ / group_count);
    return mlx::core::reshape(std::move(output), std::move(output_shape));
}

std::optional<array> MlxNintWeight::greedy_argmax(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error("NINT greedy input width mismatch");
    }
    return std::nullopt;
}

bool MlxNintWeight::can_fuse_swiglu(const MlxNintWeight&) const noexcept {
    return false;
}

array MlxNintWeight::swiglu(
    const MlxNintWeight&,
    const array&) const {
    throw std::runtime_error(
        "NINT SwiGLU uses the shared matmul path and elementwise fallback");
}

array MlxNintWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT dequantization output must be float16 or float32");
    }
    const auto element_count =
        static_cast<std::uint64_t>(output_size_) * input_size_;
    if (element_count > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
        throw std::runtime_error("NINT dequantization grid exceeds MLX limits");
    }
    const int grid = static_cast<int>(element_count);
    auto outputs = nint_dequantize_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
        },
        {Shape{output_size_, input_size_}},
        {dtype},
        {grid, 1, 1},
        {std::min(256, std::max(1, grid)), 1, 1},
        {
            {"T", dtype},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"OUT", output_size_},
        },
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxNintWeight::embedding(
    const array& token_ids,
    Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "NINT embedding output must be float16 or float32");
    }
    auto ids = token_ids;
    if (ids.dtype() != mlx::core::int32 && ids.dtype() != mlx::core::uint32) {
        ids = mlx::core::astype(ids, mlx::core::int32);
    }
    const auto count = ids.size();
    Shape output_shape = ids.shape();
    output_shape.push_back(input_size_);
    if (count == 0) {
        return mlx::core::zeros(output_shape, dtype);
    }
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count > static_cast<std::size_t>(
                    std::numeric_limits<int>::max() / input_size_)) {
        throw std::runtime_error("NINT embedding input is too large");
    }
    ids = mlx::core::reshape(ids, Shape{static_cast<std::int32_t>(count)});
    const int grid = static_cast<int>(count) * input_size_;
    auto outputs = nint_embedding_kernel()(
        {
            q_packed_,
            row_q_layout_,
            row_q_byte_offsets_,
            sub_scale_,
            sub_min_,
            neuron_scale_,
            neuron_min_,
            ids,
        },
        {Shape{static_cast<std::int32_t>(count), input_size_}},
        {dtype},
        {grid, 1, 1},
        {std::min(256, std::max(1, grid)), 1, 1},
        {
            {"T", dtype},
            {"GS", group_size_},
            {"NG", groups_},
            {"K", input_size_},
            {"COUNT", static_cast<int>(count)},
        },
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(outputs.front(), std::move(output_shape));
}

std::size_t MlxNintWeight::packed_nbytes() const noexcept {
    return q_packed_.nbytes() +
        sub_scale_.nbytes() +
        sub_min_.nbytes() +
        neuron_scale_.nbytes() +
        neuron_min_.nbytes() +
        row_q_layout_.nbytes() +
        row_q_byte_offsets_.nbytes();
}

} // namespace mfq::metal
