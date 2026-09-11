#include "mlx_mxfp4_sq.h"

#include "mlx_mxfp4_sq2.h"
#include "mlx_mxfp4_sq3.h"
#include "mfq/mxfp4_sq_blob.h"
#include "mfq_format_compat.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using mlx::core::array;
using mlx::core::CompileOptions;
using mlx::core::Dtype;
using mlx::core::MathMode;
using mlx::core::Shape;

// Both profiles share one decoder.  The symbol width is read from the payload
// header at execution time, so SQ2/SQ3 do not create separate pipeline
// specializations.  The immutable palette catalog is one small shared input.
constexpr const char* kSqHeader = R"METAL(
inline uint mfq_sq_read_bits(
    device const uchar* stream,
    uint value_index,
    uint bits
) {
    uint bit_offset = value_index * bits;
    uint byte_index = bit_offset >> 3u;
    uint shift = bit_offset & 7u;
    uint packed = uint(stream[byte_index]);
    if (shift + bits > 8u) {
        packed |= uint(stream[byte_index + 1u]) << 8u;
    }
    return (packed >> shift) & ((1u << bits) - 1u);
}

inline uint mfq_sq_block_tag(
    device const uchar* symbols,
    device const uchar* selectors,
    uint block_index,
    uint bits
) {
    device const uint* words = (device const uint*)(
        symbols + block_index * bits * 4u);
    uint low = 0u;
    if (bits == 2u) {
        uint folded = words[0] ^ words[1];
        folded ^= folded >> 16u;
        folded ^= folded >> 8u;
        folded ^= folded >> 4u;
        folded ^= folded >> 2u;
        low = folded & 3u;
    } else {
        constexpr uint MASK0 = 0x49249249u;
        constexpr uint MASK1 = 0x92492492u;
        constexpr uint MASK2 = 0x24924924u;
        uint word0 = words[0];
        uint word1 = words[1];
        uint word2 = words[2];
        uint low_count =
            popcount(word0 & MASK0)
            + popcount(word1 & MASK1)
            + popcount(word2 & MASK2);
        uint high_count =
            popcount(word0 & MASK1)
            + popcount(word1 & MASK2)
            + popcount(word2 & MASK0);
        low = (low_count & 1u) | ((high_count & 1u) << 1u);
    }
    uint explicit_high =
        (uint(selectors[block_index >> 3u])
            >> (block_index & 7u)) & 1u;
    return low | (explicit_high << 2u);
}

inline float mfq_sq_scale(uint exponent) {
    uint raw = exponent == 0u ? 0x00400000u : exponent << 23u;
    return as_type<float>(raw);
}

inline float mfq_sq_decode(
    device const uchar* palette_catalog,
    uint palette,
    uint symbol,
    uint exponent,
    uint bits
) {
    constexpr float MAGNITUDES[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    uint catalog_offset = bits == 2u ? 0u : 128u;
    uint nibble = uint(palette_catalog[
        catalog_offset + palette * (1u << bits) + symbol]);
    float magnitude = MAGNITUDES[nibble & 7u];
    if ((nibble & 8u) != 0u) {
        magnitude = -magnitude;
    }
    return magnitude * mfq_sq_scale(exponent);
}
)METAL";

constexpr const char* kSqDequantize = R"METAL(
    uint lane = thread_index_in_simdgroup;
    uint block_index = thread_position_in_grid.x >> 5u;
    if (block_index >= uint(TOTAL_BLOCKS)) {
        return;
    }
    device const uchar* symbols = blob + uint(SYMBOLS_OFFSET);
    device const uchar* selectors = blob + uint(SELECTORS_OFFSET);
    device const uchar* state_scales = blob + uint(SCALES_OFFSET);
    device const uchar* state_palettes = blob + uint(PALETTES_OFFSET);
    uint bits = uint(blob[2]) - 48u;
    uint output = block_index / uint(BLOCKS);
    uint state_value = 0u;
    if (lane == 0u) {
        uint state = output * 8u
            + mfq_sq_block_tag(symbols, selectors, block_index, bits);
        state_value =
            (mfq_sq_read_bits(state_palettes, state, 5u) << 8u)
            | (uint(blob[5])
                + mfq_sq_read_bits(state_scales, state, 2u));
    }
    state_value = simd_broadcast_first(state_value);
    uint index = block_index * 32u + lane;
    uint symbol = mfq_sq_read_bits(symbols, index, bits);
    y[index] = T(mfq_sq_decode(
        palette, state_value >> 8u, symbol, state_value & 255u, bits));
)METAL";

// One metadata-driven packed matrix kernel for every SQ profile and small M.
// A lane subgroup owns one output and whole block-32 weight vectors.  Decoded
// weights are reused across TILE_M input rows; the eight row states are cached
// once per output rather than reread for every block.
constexpr const char* kSqMatmul = R"METAL(
    constexpr uint K_LANES_VALUE = uint(K_LANES);
    constexpr uint SIMD_GROUPS_VALUE = uint(SIMD_GROUPS);
    constexpr uint THREADS = SIMD_GROUPS_VALUE * 32u;
    constexpr uint OUTPUTS_PER_SIMD = 32u / K_LANES_VALUE;
    constexpr uint OUTPUTS_PER_TG =
        SIMD_GROUPS_VALUE * OUTPUTS_PER_SIMD;

    threadgroup float row_scales[OUTPUTS_PER_TG * 8u];
    threadgroup uchar row_palettes[OUTPUTS_PER_TG * 8u];

    uint local_thread = thread_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES_VALUE - 1u);
    uint simd_output = lane / K_LANES_VALUE;
    uint output_slot =
        simd_group * OUTPUTS_PER_SIMD + simd_output;
    uint first_row =
        threadgroup_position_in_grid.x * uint(TILE_M);
    uint output_index =
        threadgroup_position_in_grid.y * OUTPUTS_PER_TG + output_slot;
    int local_expert = 0;
    bool route_valid = true;
    if (uint(ROUTED) != 0u) {
        int expert = expert_ids[first_row];
        route_valid = expert >= 0 && expert < int(EXPERT_MAP_SIZE);
        local_expert = route_valid ? expert_map[expert] : -1;
        route_valid = route_valid && local_expert >= 0;
    }
    uint output = uint(ROUTED) != 0u
        ? (route_valid
            ? uint(local_expert) * uint(OUT_PER_EXPERT) + output_index
            : 0u)
        : output_index;
    output = min(output, uint(OUT) - 1u);

    device const uchar* symbols = blob + uint(SYMBOLS_OFFSET);
    device const uchar* selectors = blob + uint(SELECTORS_OFFSET);
    device const uchar* state_scales = blob + uint(SCALES_OFFSET);
    device const uchar* state_palettes = blob + uint(PALETTES_OFFSET);
    uint bits = uint(blob[2]) - 48u;

    for (uint metadata = local_thread;
         metadata < OUTPUTS_PER_TG * 8u;
         metadata += THREADS) {
        uint local_output = metadata >> 3u;
        uint state = metadata & 7u;
        uint row_index =
            threadgroup_position_in_grid.y * OUTPUTS_PER_TG
            + local_output;
        uint row = uint(ROUTED) != 0u
            ? (route_valid
                ? uint(local_expert) * uint(OUT_PER_EXPERT) + row_index
                : 0u)
            : row_index;
        row = min(row, uint(OUT) - 1u);
        uint state_index = row * 8u + state;
        row_scales[metadata] = mfq_sq_scale(
            uint(blob[5])
            + mfq_sq_read_bits(state_scales, state_index, 2u));
        row_palettes[metadata] = uchar(
            mfq_sq_read_bits(state_palettes, state_index, 5u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulators[TILE_M];
    for (uint local_row = 0u;
         local_row < uint(TILE_M);
         ++local_row) {
        accumulators[local_row] = 0.0f;
    }

    for (uint block = k_lane;
         block < uint(BLOCKS);
         block += K_LANES_VALUE) {
        uint block_index = output * uint(BLOCKS) + block;
        uint tag = mfq_sq_block_tag(
            symbols, selectors, block_index, bits);
        uint cached_state = output_slot * 8u + tag;
        float scale = row_scales[cached_state];
        uint palette_index = uint(row_palettes[cached_state]);
        uint column_base = block * 32u;

        if (bits == 2u) {
            for (uint packed_index = 0u;
                 packed_index < 8u;
                 ++packed_index) {
                uint column = column_base + packed_index * 4u;
                uint packed = uint(
                    symbols[block_index * 8u + packed_index]);
                float4 weight = float4(
                    mfq_sq_decode(
                        palette, palette_index, packed & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 2u) & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 4u) & 3u, 127u, 2u),
                    mfq_sq_decode(
                        palette, palette_index, packed >> 6u, 127u, 2u)) * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(
                        first_row + local_row,
                        uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    accumulators[local_row] += dot(
                        float4(activation), weight);
                }
            }
        } else {
            for (uint group = 0u; group < 4u; ++group) {
                uint column = column_base + group * 8u;
                uint byte_offset = block_index * 12u + group * 3u;
                uint packed = uint(symbols[byte_offset])
                    | (uint(symbols[byte_offset + 1u]) << 8u)
                    | (uint(symbols[byte_offset + 2u]) << 16u);
                float4 weight0 = float4(
                    mfq_sq_decode(
                        palette, palette_index, packed & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 3u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 6u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 9u) & 7u, 127u, 3u))
                    * scale;
                float4 weight1 = float4(
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 12u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 15u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 18u) & 7u, 127u, 3u),
                    mfq_sq_decode(
                        palette, palette_index, (packed >> 21u) & 7u, 127u, 3u))
                    * scale;
                for (uint local_row = 0u;
                     local_row < uint(TILE_M);
                     ++local_row) {
                    uint row = min(
                        first_row + local_row,
                        uint(M) - 1u);
                    uint input_row = uint(ROUTED) != 0u
                        && uint(SHARED_INPUT) != 0u
                        ? row / uint(ROUTES)
                        : row;
                    half4 activation0 = *(device const half4*)(
                        x + input_row * uint(K) + column);
                    half4 activation1 = *(device const half4*)(
                        x + input_row * uint(K) + column + 4u);
                    accumulators[local_row] +=
                        dot(float4(activation0), weight0)
                        + dot(float4(activation1), weight1);
                }
            }
        }
    }

    for (uint local_row = 0u;
         local_row < uint(TILE_M);
         ++local_row) {
        if (K_LANES_VALUE >= 16u) {
            accumulators[local_row] += simd_shuffle_down(
                accumulators[local_row], 8u);
        }
        if (K_LANES_VALUE >= 8u) {
            accumulators[local_row] += simd_shuffle_down(
                accumulators[local_row], 4u);
        }
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 2u);
        accumulators[local_row] += simd_shuffle_down(
            accumulators[local_row], 1u);
        uint row = first_row + local_row;
        if (k_lane == 0u && row < uint(M)
            && output_index < uint(LOGICAL_OUT)) {
            y[row * uint(LOGICAL_OUT) + output_index] = T(
                route_valid ? accumulators[local_row] : 0.0f);
        }
    }
)METAL";

const array& palette_catalog() {
    static const auto catalog = [] {
        std::vector<std::uint8_t> values;
        values.reserve(
            kMxfp4Sq2PaletteNibbles.size() +
            kMxfp4Sq3PaletteNibbles.size());
        values.insert(
            values.end(),
            kMxfp4Sq2PaletteNibbles.begin(),
            kMxfp4Sq2PaletteNibbles.end());
        values.insert(
            values.end(),
            kMxfp4Sq3PaletteNibbles.begin(),
            kMxfp4Sq3PaletteNibbles.end());
        return array(values.begin(), Shape{static_cast<int>(values.size())});
    }();
    return catalog;
}

const mlx::core::fast::CustomKernelFunction& dequantize_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp4_sq_dequantize",
            {"blob", "palette"},
            {"y"},
            kSqDequantize,
            kSqHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

const mlx::core::fast::CustomKernelFunction& matmul_kernel() {
    static const auto kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_mxfp4_sq_matmul",
            {"blob", "palette", "x", "expert_ids", "expert_map"},
            {"y"},
            kSqMatmul,
            kSqHeader,
            true,
            false,
            options);
    }();
    return kernel;
}

std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
templates(const MlxMxfp4SqWeight& weight, Dtype dtype) {
    const auto layout = mfq::sq::layout(
        weight.bits(),
        weight.output_size(),
        weight.input_size(),
        weight.matrix_scale_base());
    const auto checked_int = [](std::size_t value, const char* name) {
        if (value > static_cast<std::size_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                std::string("MXFP4-SQ ") + name + " exceeds MLX limits");
        }
        return static_cast<int>(value);
    };
    return {
        {"T", dtype},
        {"K", weight.input_size()},
        {"OUT", weight.output_size()},
        {"BLOCKS", weight.input_size() / 32},
        {"TOTAL_BLOCKS", checked_int(
            static_cast<std::size_t>(weight.input_size() / 32) *
                static_cast<std::size_t>(weight.output_size()),
            "block count")},
        {"WEIGHTS", checked_int(
            static_cast<std::size_t>(weight.input_size()) *
                static_cast<std::size_t>(weight.output_size()),
            "weight count")},
        {"SYMBOLS_OFFSET", checked_int(layout.symbols, "symbol offset")},
        {"SELECTORS_OFFSET", checked_int(
            layout.selectors, "selector offset")},
        {"SCALES_OFFSET", checked_int(layout.scales, "scale offset")},
        {"PALETTES_OFFSET", checked_int(
            layout.palettes, "palette offset")},
    };
}

} // namespace

bool is_mxfp4_sq_dtype(std::string_view dtype) noexcept {
    return mfq::canonical_format_dtype(dtype) == mfq::kMxfp4SqDtype;
}

MlxMxfp4SqWeight::MlxMxfp4SqWeight(
    array blob,
    int bits,
    int input_size,
    int output_size,
    std::uint8_t matrix_scale_base)
    : blob_(std::move(blob)),
      bits_(bits),
      input_size_(input_size),
      output_size_(output_size),
      matrix_scale_base_(matrix_scale_base) {}

MlxMxfp4SqWeight MlxMxfp4SqWeight::from_blob(
    const std::vector<std::uint8_t>& blob) {
    return from_blob(std::span<const std::uint8_t>(blob));
}

MlxMxfp4SqWeight MlxMxfp4SqWeight::from_blob(
    std::span<const std::uint8_t> blob) {
    const auto layout = mfq::sq::parse(blob.data(), blob.size());
    if (blob.size() > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MXFP4-SQ payload exceeds MLX limits");
    }
    return MlxMxfp4SqWeight(
        array(blob.begin(), Shape{static_cast<int>(blob.size())}),
        layout.bits,
        layout.width,
        layout.outputs,
        static_cast<std::uint8_t>(layout.base));
}

array MlxMxfp4SqWeight::dequantize(Dtype dtype) const {
    if (dtype != mlx::core::float16 && dtype != mlx::core::float32) {
        throw std::runtime_error(
            "MXFP4-SQ dequantization requires float16 or float32");
    }
    const auto count =
        static_cast<std::size_t>(input_size_) *
        static_cast<std::size_t>(output_size_);
    if (count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "MXFP4-SQ dequantization grid exceeds MLX limits");
    }
    constexpr std::size_t threads = 256;
    const auto grid = (count + threads - 1) / threads * threads;
    auto outputs = dequantize_kernel()(
        {blob_, palette_catalog()},
        {Shape{output_size_, input_size_}},
        {dtype},
        {static_cast<int>(grid), 1, 1},
        {static_cast<int>(threads), 1, 1},
        templates(*this, dtype),
        std::nullopt,
        false,
        {});
    return std::move(outputs.front());
}

array MlxMxfp4SqWeight::embedding(
    const array& rows,
    Dtype dtype) const {
    return mlx::core::take(dequantize(dtype), rows, 0);
}

array MlxMxfp4SqWeight::matmul(const array& input) const {
    if (input.ndim() == 0 || input.shape(-1) != input_size_) {
        throw std::runtime_error(
            "MXFP4-SQ input width does not match packed weight");
    }
    const auto rows =
        input.size() / static_cast<std::size_t>(input_size_);
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("unsupported MXFP4-SQ input row count");
    }
    Shape output_shape = input.shape();
    output_shape.back() = output_size_;
    auto source = input;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), input_size_});

    if (rows > 48 || source.dtype() == mlx::core::float32) {
        auto dense = dequantize(source.dtype());
        auto result = mlx::core::matmul(
            source,
            mlx::core::transpose(dense));
        return mlx::core::reshape(
            std::move(result),
            std::move(output_shape));
    }

    const bool first_bucket = rows <= 6;
    const int k_lanes = first_bucket && rows <= 3 ? 16 : 8;
    const int simd_groups = first_bucket && rows <= 3 ? 4 : 2;
    const int threads = simd_groups * 32;
    const int row_tiles = first_bucket
        ? 1
        : (static_cast<int>(rows) + 7) / 8;
    const int tile_rows =
        (static_cast<int>(rows) + row_tiles - 1) / row_tiles;
    const int outputs_per_threadgroup =
        simd_groups * 32 / k_lanes;
    const int output_tiles =
        (output_size_ + outputs_per_threadgroup - 1) /
        outputs_per_threadgroup;
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("M", static_cast<int>(rows));
    arguments.emplace_back("TILE_M", tile_rows);
    arguments.emplace_back("K_LANES", k_lanes);
    arguments.emplace_back("SIMD_GROUPS", simd_groups);
    arguments.emplace_back("ROUTED", 0);
    arguments.emplace_back("ROUTES", 1);
    arguments.emplace_back("SHARED_INPUT", 0);
    arguments.emplace_back("EXPERT_MAP_SIZE", 1);
    arguments.emplace_back("OUT_PER_EXPERT", output_size_);
    arguments.emplace_back("LOGICAL_OUT", output_size_);
    const auto unused_ids = mlx::core::zeros(
        Shape{1}, mlx::core::int32);
    auto outputs = matmul_kernel()(
        {
            blob_,
            palette_catalog(),
            std::move(source),
            unused_ids,
            unused_ids,
        },
        {Shape{static_cast<int>(rows), output_size_}},
        {input.dtype() == mlx::core::float32
            ? mlx::core::float32
            : mlx::core::float16},
        {row_tiles * threads, output_tiles, 1},
        {threads, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()),
        std::move(output_shape));
}

array MlxMxfp4SqWeight::routed_matmul(
    const array& input,
    const array& expert_ids,
    const array& expert_map,
    int out_per_expert) const {
    if (out_per_expert <= 0 || output_size_ % out_per_expert != 0) {
        throw std::invalid_argument(
            "MXFP4-SQ routed output width is inconsistent");
    }
    if (expert_ids.ndim() != 2 || expert_map.ndim() != 1 ||
        expert_ids.dtype() != mlx::core::int32 ||
        expert_map.dtype() != mlx::core::int32) {
        throw std::invalid_argument(
            "MXFP4-SQ routing metadata must be contiguous int32 IDs");
    }
    const int tokens = expert_ids.shape(0);
    const int routes = expert_ids.shape(1);
    const bool shared_input = input.ndim() == 2 &&
        input.shape(0) == tokens && input.shape(1) == input_size_;
    if (!shared_input && (
        input.ndim() != 3 || input.shape(0) != tokens ||
        input.shape(1) != routes || input.shape(2) != input_size_)) {
        throw std::invalid_argument(
            "MXFP4-SQ routed input must be [tokens,K] or "
            "[tokens,routes,K]");
    }
    const auto route_count =
        static_cast<std::size_t>(tokens) * static_cast<std::size_t>(routes);
    if (route_count > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error("MXFP4-SQ route count exceeds MLX limits");
    }
    const Shape output_shape{tokens, routes, out_per_expert};
    if (route_count == 0) {
        return mlx::core::zeros(output_shape, mlx::core::float16);
    }
    auto source = input.dtype() == mlx::core::float16
        ? input
        : mlx::core::astype(input, mlx::core::float16);
    source = mlx::core::contiguous(source);
    auto ids = mlx::core::contiguous(expert_ids);
    auto map = mlx::core::contiguous(expert_map);

    constexpr int kLanes = 8;
    constexpr int kSimdGroups = 2;
    constexpr int kThreads = kSimdGroups * 32;
    constexpr int kOutputsPerThreadgroup =
        kSimdGroups * 32 / kLanes;
    const int output_tiles =
        (out_per_expert + kOutputsPerThreadgroup - 1) /
        kOutputsPerThreadgroup;
    auto arguments = templates(*this, source.dtype());
    arguments.emplace_back("M", static_cast<int>(route_count));
    arguments.emplace_back("TILE_M", 1);
    arguments.emplace_back("K_LANES", kLanes);
    arguments.emplace_back("SIMD_GROUPS", kSimdGroups);
    arguments.emplace_back("ROUTED", 1);
    arguments.emplace_back("ROUTES", routes);
    arguments.emplace_back("SHARED_INPUT", static_cast<int>(shared_input));
    arguments.emplace_back("EXPERT_MAP_SIZE", expert_map.shape(0));
    arguments.emplace_back("OUT_PER_EXPERT", out_per_expert);
    arguments.emplace_back("LOGICAL_OUT", out_per_expert);
    auto outputs = matmul_kernel()(
        {blob_, palette_catalog(), source, ids, map},
        {Shape{static_cast<int>(route_count), out_per_expert}},
        {mlx::core::float16},
        {static_cast<int>(route_count) * kThreads, output_tiles, 1},
        {kThreads, 1, 1},
        std::move(arguments),
        std::nullopt,
        false,
        {});
    return mlx::core::reshape(
        std::move(outputs.front()), output_shape);
}

array MlxMxfp4SqWeight::backward_input(
    const array& output_gradient) const {
    if (output_gradient.ndim() == 0 ||
        output_gradient.shape(-1) != output_size_) {
        throw std::runtime_error(
            "MXFP4-SQ output-gradient width does not match packed weight");
    }
    Shape output_shape = output_gradient.shape();
    output_shape.back() = input_size_;
    auto source = output_gradient;
    if (source.dtype() != mlx::core::float16 &&
        source.dtype() != mlx::core::float32) {
        source = mlx::core::astype(source, mlx::core::float16);
    }
    const auto rows =
        source.size() / static_cast<std::size_t>(output_size_);
    if (rows == 0 || rows > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "unsupported MXFP4-SQ backward row count");
    }
    source = mlx::core::reshape(
        source,
        Shape{static_cast<int>(rows), output_size_});
    auto result = mlx::core::matmul(
        source,
        dequantize(source.dtype()));
    return mlx::core::reshape(
        std::move(result),
        std::move(output_shape));
}

} // namespace mfq::metal
