// Canonical CUDA execution for NINT.
//
// Every ordinary NINT tensor is normalized at the loader boundary to one
// row-major bitstream plus per-neuron q metadata.  Uniform presets and
// adaptive q/k allocations therefore execute through exactly the same
// kernels.  Large-M execution uses the same row decoder followed by GEMM.

// NINT8-0 is a separate symmetric format.  It retains one small-M packed
// kernel; larger batches use its row decoder followed by GEMM.


#include <algorithm>
#include <cstdint>
#include <vector>

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "mfq_tensor_backend.h"
#include "packed_backward.cuh"


#define MFQ_CUBLAS_CHECK(expression) \
    MFQ_RUNTIME_CHECK( \
        (expression) == CUBLAS_STATUS_SUCCESS, \
        "cuBLAS call failed: ", #expression)


namespace {


__device__ __forceinline__ uint8_t unpack_nint_code(
        const uint8_t * stream,
        uint64_t row_bit_offset,
        int element,
        int bits) {
    const uint64_t bit = row_bit_offset +
        static_cast<uint64_t>(element) * static_cast<uint64_t>(bits);
    const uint64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7u);
    uint32_t word = static_cast<uint32_t>(stream[byte]);
    if (shift + bits > 8) {
        word |= static_cast<uint32_t>(stream[byte + 1]) << 8;
    }
    return static_cast<uint8_t>(
        (word >> shift) & ((1u << bits) - 1u));
}


__device__ __forceinline__ int unpack_nint_codes4(
        const uint8_t * stream,
        uint64_t bit_offset,
        int bits) {
    const uint64_t byte = bit_offset >> 3;
    const int shift = static_cast<int>(bit_offset & 7u);
    const int required_bits = shift + 4 * bits;
    uint64_t packed = static_cast<uint64_t>(stream[byte]);
    if (required_bits > 8) {
        packed |= static_cast<uint64_t>(stream[byte + 1]) << 8;
    }
    if (required_bits > 16) {
        packed |= static_cast<uint64_t>(stream[byte + 2]) << 16;
    }
    if (required_bits > 24) {
        packed |= static_cast<uint64_t>(stream[byte + 3]) << 24;
    }
    if (required_bits > 32) {
        packed |= static_cast<uint64_t>(stream[byte + 4]) << 32;
    }
    packed >>= shift;
    const uint32_t mask = (1u << bits) - 1u;
    return static_cast<int>(packed & mask) |
        (static_cast<int>((packed >> bits) & mask) << 8) |
        (static_cast<int>((packed >> (2 * bits)) & mask) << 16) |
        (static_cast<int>((packed >> (3 * bits)) & mask) << 24);
}


__device__ __forceinline__ int load_i8x4(const int8_t * source) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(source);
    return static_cast<int>(bytes[0]) |
        (static_cast<int>(bytes[1]) << 8) |
        (static_cast<int>(bytes[2]) << 16) |
        (static_cast<int>(bytes[3]) << 24);
}


__global__ void nint8_one_quantize_reconstruct_kernel(
        const __half * __restrict__ input,
        int8_t * __restrict__ quantized,
        __half * __restrict__ scale_output,
        __half * __restrict__ sum_output,
        __half * __restrict__ reconstructed,
        int rows,
        int width,
        int groups) {
    const int group_index = static_cast<int>(blockIdx.x);
    if (group_index >= rows * groups) {
        return;
    }
    const int lane = static_cast<int>(threadIdx.x);
    const int row = group_index / groups;
    const int group = group_index - row * groups;
    const int column = group * 32 + lane;
    const float value = column < width
        ? __half2float(input[static_cast<size_t>(row) * width + column])
        : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_xor_sync(0xffffffffu, maximum, offset));
    }
    const float scale = maximum / 127.0f;
    const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    int code = static_cast<int>(roundf(value * inverse));
    code = max(-127, min(127, code));
    quantized[static_cast<size_t>(group_index) * 32 + lane] =
        static_cast<int8_t>(code);

    int sum = code;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    const __half stored_scale = __float2half_rn(scale);
    if (lane == 0) {
        scale_output[group_index] = stored_scale;
        sum_output[group_index] = __float2half_rn(
            static_cast<float>(sum) * scale);
    }
    if (column < width) {
        reconstructed[static_cast<size_t>(row) * width + column] =
            __float2half_rn(
                static_cast<float>(code) * __half2float(stored_scale));
    }
}


__global__ void nint_decode_rows_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        __half * __restrict__ output,
        int rows,
        int groups,
        int group_size,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int column = static_cast<int>(index % width);
        const int row = static_cast<int>(index / width);
        const int group = column / group_size;
        const size_t metadata = static_cast<size_t>(row) * groups + group;
        const int bits = static_cast<int>(row_q_bits[row]);
        const uint8_t code = unpack_nint_code(
            bitstream,
            static_cast<uint64_t>(row_q_bit_offsets[row]),
            column,
            bits);
        const float scale = neuron_scale[row] *
            static_cast<float>(subgroup_scale[metadata]);
        const float minimum = neuron_minimum[row] *
            static_cast<float>(subgroup_minimum[metadata]);
        output[index] = __float2half_rn(
            scale * static_cast<float>(code) - minimum);
    }
}


// Runtime group size deliberately avoids a kernel family per quantizer preset.
__global__ void __launch_bounds__(64) nint_quantize_activation_kernel(
        const __half * __restrict__ input,
        int8_t * __restrict__ quantized,
        float * __restrict__ scale_output,
        int rows,
        int real_width,
        int padded_width,
        int groups,
        int group_size) {
    const int row = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int column = group * group_size + lane;
    const bool valid = lane < group_size && column < real_width;
    const float value = valid
        ? __half2float(input[static_cast<size_t>(row) * real_width + column])
        : 0.0f;
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    __shared__ float warp_maxima[2];
    if ((lane & 31) == 0) {
        warp_maxima[lane >> 5] = maximum;
    }
    __syncthreads();
    if (lane < 32) {
        const int active_warps = (blockDim.x + 31) / 32;
        maximum = lane < active_warps ? warp_maxima[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffu, maximum, offset));
        }
    }
    __shared__ float group_scale;
    if (lane == 0) {
        group_scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        scale_output[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    __syncthreads();
    if (lane < group_size) {
        int code = 0;
        if (valid) {
            code = static_cast<int>(roundf(value / group_scale));
            code = max(-127, min(127, code));
        }
        quantized[static_cast<size_t>(row) * padded_width + column] =
            static_cast<int8_t>(code);
    }
}


// One NINT compute kernel covers every q, k, group size, M<=8, and routed MFE
// projection. q is row metadata; k has already been baked into the subgroup
// metadata values. Routed execution changes only the indexing contract, not
// the packed-weight compute kernel.
__global__ void __launch_bounds__(128) nint_matmul_kernel(
        const uint8_t * __restrict__ bitstream,
        const uint8_t * __restrict__ row_q_bits,
        const int64_t * __restrict__ row_q_bit_offsets,
        const uint8_t * __restrict__ subgroup_scale,
        const uint8_t * __restrict__ subgroup_minimum,
        const float * __restrict__ neuron_scale,
        const float * __restrict__ neuron_minimum,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        const int32_t * __restrict__ route_ids,
        const int32_t * __restrict__ expert_local,
        int routes,
        int experts,
        int q_expert_stride,
        bool routed_input) {
    constexpr int warps_per_block = 4;
    constexpr int maximum_activation_rows = 8;
    constexpr int routed_rows_per_warp = 2;
    const int chunks = (group_size + 3) / 4;
    const int groups_per_warp = 32 / chunks;
    const int lane = static_cast<int>(threadIdx.x);

    if (route_ids != nullptr) {
        const int token = static_cast<int>(blockIdx.z) * warps_per_block +
            static_cast<int>(threadIdx.y);
        const int route = static_cast<int>(blockIdx.y);
        const int output_row0 =
            static_cast<int>(blockIdx.x) * routed_rows_per_warp;
        if (token >= activation_rows || route >= routes) {
            return;
        }
        const int pair = token * routes + route;
        const int expert = route_ids[pair];
        if (static_cast<unsigned int>(expert) >=
                static_cast<unsigned int>(experts)) {
            return;
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            return;
        }
        const uint8_t * expert_stream = bitstream +
            static_cast<size_t>(local_expert) *
                static_cast<size_t>(q_expert_stride);
        const int source_row = routed_input ? pair : token;

        float accumulators[routed_rows_per_warp] = {0.0f, 0.0f};
        float outer_scales[routed_rows_per_warp] = {};
        float outer_minima[routed_rows_per_warp] = {};
        int row_bits[routed_rows_per_warp] = {};
        uint64_t row_bit_offsets[routed_rows_per_warp] = {};
#pragma unroll
        for (int row = 0; row < routed_rows_per_warp; ++row) {
            const int output_row = output_row0 + row;
            if (output_row < output_rows) {
                const int weight_row =
                    local_expert * output_rows + output_row;
                outer_scales[row] = neuron_scale[weight_row];
                outer_minima[row] = neuron_minimum[weight_row];
                row_bits[row] = static_cast<int>(row_q_bits[weight_row]);
                row_bit_offsets[row] = static_cast<uint64_t>(
                    row_q_bit_offsets[weight_row]);
            }
        }

        const int relative_group = lane / chunks;
        const int chunk = lane - relative_group * chunks;
        const int element = chunk * 4;
        const bool active_lane = relative_group < groups_per_warp;
        for (int group_base = 0;
             group_base < groups;
             group_base += groups_per_warp) {
            const int group = group_base + relative_group;
            if (!active_lane || group >= groups || element >= group_size) {
                continue;
            }
            const int width = min(4, group_size - element);
            const int column = group * group_size + element;
            const int8_t * activation_ptr = activation +
                static_cast<size_t>(source_row) * padded_width + column;
            int activation_codes = 0;
            int activation_sum = 0;
            if (width == 4) {
                activation_codes = load_i8x4(activation_ptr);
                activation_sum = __dp4a(
                    0x01010101, activation_codes, 0);
            } else {
#pragma unroll
                for (int component = 0; component < 4; ++component) {
                    if (component < width) {
                        const int code = static_cast<int>(
                            activation_ptr[component]);
                        activation_codes |= (code & 255) << (8 * component);
                        activation_sum += code;
                    }
                }
            }
            const float input_scale = activation_scale[
                static_cast<size_t>(source_row) * groups + group];
#pragma unroll
            for (int row = 0; row < routed_rows_per_warp; ++row) {
                const int output_row = output_row0 + row;
                if (output_row >= output_rows) {
                    continue;
                }
                const int weight_row =
                    local_expert * output_rows + output_row;
                const size_t metadata_index =
                    static_cast<size_t>(weight_row) * groups + group;
                const int bits = row_bits[row];
                int weight_codes = 0;
                if (width == 4) {
                    const uint64_t bit_offset = row_bit_offsets[row] +
                        static_cast<uint64_t>(column) *
                            static_cast<uint64_t>(bits);
                    weight_codes = unpack_nint_codes4(
                        expert_stream, bit_offset, bits);
                } else {
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (component < width) {
                            weight_codes |= static_cast<int>(
                                unpack_nint_code(
                                    expert_stream,
                                    row_bit_offsets[row],
                                    column + component,
                                    bits)) << (8 * component);
                        }
                    }
                }
                const int dot = bits == 8
                    ? __dp4a(
                          weight_codes ^ static_cast<int>(0x80808080u),
                          activation_codes,
                          0) + 128 * activation_sum
                    : __dp4a(weight_codes, activation_codes, 0);
                accumulators[row] += input_scale * (
                    outer_scales[row] *
                        static_cast<float>(subgroup_scale[metadata_index]) *
                        static_cast<float>(dot) -
                    outer_minima[row] *
                        static_cast<float>(subgroup_minimum[metadata_index]) *
                        static_cast<float>(activation_sum));
            }
        }

#pragma unroll
        for (int row = 0; row < routed_rows_per_warp; ++row) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                accumulators[row] += __shfl_xor_sync(
                    0xffffffffu, accumulators[row], offset);
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int row = 0; row < routed_rows_per_warp; ++row) {
                const int output_row = output_row0 + row;
                if (output_row < output_rows) {
                    output[
                        static_cast<size_t>(pair) * output_rows +
                        output_row] = __float2half(accumulators[row]);
                }
            }
        }
        return;
    }

    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    if (output_row >= output_rows) {
        return;
    }

    const int bits = static_cast<int>(row_q_bits[output_row]);
    const uint64_t row_bit_offset =
        static_cast<uint64_t>(row_q_bit_offsets[output_row]);
    const uint8_t * scale_row = subgroup_scale +
        static_cast<size_t>(output_row) * groups;
    const uint8_t * minimum_row = subgroup_minimum +
        static_cast<size_t>(output_row) * groups;
    const float outer_scale = neuron_scale[output_row];
    const float outer_minimum = neuron_minimum[output_row];
    float accumulators[maximum_activation_rows];
#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
        accumulators[row] = 0.0f;
    }

    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int element = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const bool full_chunk = active_lane && element + 3 < group_size;
    const bool tail_chunk =
        active_lane && element < group_size && !full_chunk;
    for (int group_base = 0;
         group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const float inner_scale = static_cast<float>(scale_row[group]);
        const float inner_minimum = static_cast<float>(minimum_row[group]);
        const int column = group * group_size + element;
        if (full_chunk) {
            const uint64_t bit_offset = row_bit_offset +
                static_cast<uint64_t>(column) * static_cast<uint64_t>(bits);
            const int weight_codes = unpack_nint_codes4(
                bitstream, bit_offset, bits);
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    const int activation_codes = load_i8x4(
                        activation + static_cast<size_t>(row) * padded_width +
                        column);
                    const int activation_sum = __dp4a(
                        0x01010101, activation_codes, 0);
                    const int dot = bits == 8
                        ? __dp4a(
                              weight_codes ^ static_cast<int>(0x80808080u),
                              activation_codes,
                              0) + 128 * activation_sum
                        : __dp4a(weight_codes, activation_codes, 0);
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        } else if (tail_chunk) {
#pragma unroll
            for (int row = 0; row < maximum_activation_rows; ++row) {
                if (row < activation_rows) {
                    int dot = 0;
                    int activation_sum = 0;
#pragma unroll
                    for (int component = 0; component < 4; ++component) {
                        if (element + component < group_size) {
                            const int activation_code = static_cast<int>(
                                activation[
                                    static_cast<size_t>(row) * padded_width +
                                    column + component]);
                            const int weight_code = static_cast<int>(
                                unpack_nint_code(
                                    bitstream,
                                    row_bit_offset,
                                    column + component,
                                    bits));
                            dot += weight_code * activation_code;
                            activation_sum += activation_code;
                        }
                    }
                    const float input_scale = activation_scale[
                        static_cast<size_t>(row) * groups + group];
                    accumulators[row] += input_scale * (
                        outer_scale * inner_scale * static_cast<float>(dot) -
                        outer_minimum * inner_minimum *
                            static_cast<float>(activation_sum));
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < maximum_activation_rows; ++row) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            accumulators[row] += __shfl_xor_sync(
                0xffffffffu, accumulators[row], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < maximum_activation_rows; ++row) {
            if (row < activation_rows) {
                output[static_cast<size_t>(row) * output_rows + output_row] =
                    __float2half_rn(accumulators[row]);
            }
        }
    }
}


__global__ void nint8_zero_decode_rows_kernel(
        const int8_t * __restrict__ quantized,
        const __half * __restrict__ scale,
        __half * __restrict__ output,
        int rows,
        int groups,
        int width) {
    const size_t total = static_cast<size_t>(rows) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         index < total;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(index / width);
        const int column = static_cast<int>(index % width);
        const int group = column / 32;
        const int lane = column & 31;
        const size_t block = static_cast<size_t>(row) * groups + group;
        output[index] = __float2half_rn(
            __half2float(scale[block]) * static_cast<float>(
                quantized[block * 32 + lane]));
    }
}


// NINT8-0 has one packed small-M compute kernel; M is a grid dimension rather
// than a template parameter, so it does not create one binary per M.
__global__ void __launch_bounds__(128) nint8_zero_matmul_kernel(
        const int8_t * __restrict__ weight,
        const __half * __restrict__ weight_scale,
        const int8_t * __restrict__ activation,
        const float * __restrict__ activation_scale,
        __half * __restrict__ output,
        int activation_rows,
        int output_rows,
        int groups,
        int padded_width) {
    constexpr int warps_per_block = 4;
    const int output_row = static_cast<int>(blockIdx.x) * warps_per_block +
        static_cast<int>(threadIdx.y);
    const int activation_row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (output_row >= output_rows || activation_row >= activation_rows) {
        return;
    }
    float accumulator = 0.0f;
    for (int base = lane * 4; base < padded_width; base += 32 * 4) {
        const int group = base / 32;
        const int offset = base & 31;
        const size_t block = static_cast<size_t>(output_row) * groups + group;
        const int weight_codes = *reinterpret_cast<const int *>(
            weight + block * 32 + offset);
        const int activation_codes = *reinterpret_cast<const int *>(
            activation + static_cast<size_t>(activation_row) * padded_width +
            base);
        accumulator += __half2float(weight_scale[block]) *
            activation_scale[
                static_cast<size_t>(activation_row) * groups + group] *
            static_cast<float>(__dp4a(weight_codes, activation_codes, 0));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        accumulator += __shfl_xor_sync(
            0xffffffffu, accumulator, offset);
    }
    if (lane == 0) {
        output[static_cast<size_t>(activation_row) * output_rows +
            output_row] = __float2half_rn(accumulator);
    }
}


void validate_nint8_zero(
        const mfq_tensor_backend::Tensor & quantized,
        const mfq_tensor_backend::Tensor & scale) {
    MFQ_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == mfq_tensor_backend::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    MFQ_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == mfq_tensor_backend::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    MFQ_RUNTIME_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}


mfq_tensor_backend::Tensor cublas_gemm_nt_f32_output(
        const mfq_tensor_backend::Tensor & input,
        const mfq_tensor_backend::Tensor & weight) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        input.options().dtype(mfq_tensor_backend::kFloat32));
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<float>(),
        CUDA_R_32F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


}  // namespace


void launch_nint_matmul_routed_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale,
        mfq_tensor_backend::Tensor route_ids,
        mfq_tensor_backend::Tensor expert_local,
        mfq_tensor_backend::Tensor output,
        int tokens,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int padded_width,
        int group_size,
        int q_expert_stride,
        bool routed_input,
        cudaStream_t stream) {
    constexpr int warps_per_block = 4;
    constexpr int rows_per_warp = 2;
    const int token_blocks =
        (tokens + warps_per_block - 1) / warps_per_block;
    const int row_blocks =
        (output_rows + rows_per_warp - 1) / rows_per_warp;
    nint_matmul_kernel<<<
        dim3(row_blocks, routes, token_blocks),
        dim3(32, warps_per_block), 0, stream>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            tokens,
            output_rows,
            groups,
            padded_width,
            group_size,
            route_ids.data_ptr<int32_t>(),
            expert_local.data_ptr<int32_t>(),
            routes,
            experts,
            q_expert_stride,
            routed_input);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
}


std::vector<mfq_tensor_backend::Tensor> nint8_one_quantize_reconstruct_cuda(
        mfq_tensor_backend::Tensor input) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT8-1 input must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    MFQ_RUNTIME_CHECK(rows > 0 && width > 0, "NINT8-1 input must be non-empty");
    const int groups = (width + 31) / 32;
    auto quantized = mfq_tensor_backend::empty(
        {rows, groups, 32},
        input.options().dtype(mfq_tensor_backend::kInt8));
    auto scale = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto sum = mfq_tensor_backend::empty({rows, groups}, input.options());
    auto reconstructed = mfq_tensor_backend::empty_like(input);
    nint8_one_quantize_reconstruct_kernel<<<
        rows * groups, 32, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            quantized.data_ptr<int8_t>(),
            reinterpret_cast<__half *>(scale.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(sum.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(reconstructed.data_ptr<mfq_half>()),
            rows,
            width,
            groups);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scale, sum, reconstructed};
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f16acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_16F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


mfq_tensor_backend::Tensor nint_cublas_gemm_nt_f32acc_cuda(
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == mfq_tensor_backend::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    MFQ_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    cublasHandle_t handle = mfq_current_cublas_handle();
    MFQ_CUBLAS_CHECK(cublasSetStream(handle, mfq_current_cuda_stream()));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    MFQ_CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        outputs,
        rows,
        width,
        &alpha,
        weight.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        input.data_ptr<mfq_half>(),
        CUDA_R_16F,
        width,
        &beta,
        result.data_ptr<mfq_half>(),
        CUDA_R_16F,
        outputs,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    return result;
}


mfq_tensor_backend::Tensor nint_decode_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1,
        "NINT bitstream must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1,
        "NINT q metadata must be contiguous CUDA uint8 rank-1");
    MFQ_RUNTIME_CHECK(
        row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT row offsets must be contiguous CUDA int64 rank-1");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    MFQ_RUNTIME_CHECK(
        rows > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT decode dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == rows && row_q_bit_offsets.numel() == rows &&
        neuron_scale.numel() == rows && neuron_minimum.numel() == rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device(),
        "NINT tensors must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        neuron_scale.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    nint_decode_rows_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            rows,
            groups,
            static_cast<int>(group_size),
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint_matmul_ws_cuda(
        mfq_tensor_backend::Tensor bitstream,
        mfq_tensor_backend::Tensor row_q_bits,
        mfq_tensor_backend::Tensor row_q_bit_offsets,
        mfq_tensor_backend::Tensor subgroup_scale,
        mfq_tensor_backend::Tensor subgroup_minimum,
        mfq_tensor_backend::Tensor neuron_scale,
        mfq_tensor_backend::Tensor neuron_minimum,
        mfq_tensor_backend::Tensor input,
        int64_t group_size,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT input must be contiguous CUDA fp16 rank-2");
    MFQ_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == mfq_tensor_backend::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == mfq_tensor_backend::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == mfq_tensor_backend::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == mfq_tensor_backend::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    MFQ_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(input.size(0));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * static_cast<int>(group_size);
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8,
        "NINT packed matmul supports M in [1,8]");
    MFQ_RUNTIME_CHECK(
        group_size >= 4 && group_size <= 64 && real_width <= padded_width,
        "NINT packed matmul dimensions are invalid");
    MFQ_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT row metadata shape mismatch");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        bitstream.device() == input.device() &&
        row_q_bits.device() == input.device() &&
        row_q_bit_offsets.device() == input.device() &&
        subgroup_scale.device() == input.device() &&
        subgroup_minimum.device() == input.device() &&
        neuron_scale.device() == input.device() &&
        neuron_minimum.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
    const int quantize_threads = group_size <= 32 ? 32 : 64;
    nint_quantize_activation_kernel<<<
        dim3(activation_rows, groups), quantize_threads, 0, stream>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            static_cast<int>(group_size));
    nint_matmul_kernel<<<
        dim3((output_rows + 3) / 4), dim3(32, 4), 0, stream>>>(
            bitstream.data_ptr<uint8_t>(),
            row_q_bits.data_ptr<uint8_t>(),
            row_q_bit_offsets.data_ptr<int64_t>(),
            subgroup_scale.data_ptr<uint8_t>(),
            subgroup_minimum.data_ptr<uint8_t>(),
            neuron_scale.data_ptr<float>(),
            neuron_minimum.data_ptr<float>(),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            padded_width,
            static_cast<int>(group_size),
            nullptr,
            nullptr,
            0,
            0,
            0,
            false);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_dequant_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 width is invalid");
    const int rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    auto output = mfq_tensor_backend::empty(
        {rows, width},
        quantized.options().dtype(mfq_tensor_backend::kFloat16));
    constexpr int threads = 256;
    const size_t total = static_cast<size_t>(rows) * width;
    const int blocks = static_cast<int>(std::min<size_t>(
        (total + threads - 1) / threads, 65535));
    nint8_zero_decode_rows_kernel<<<
        blocks, threads, 0, mfq_current_cuda_stream()>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            rows,
            groups,
            static_cast<int>(width));
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_gemv_ws_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor quantized_input,
        mfq_tensor_backend::Tensor input_scale) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2,
        "NINT8-0 input must be contiguous CUDA fp16 rank-2");
    const int activation_rows = static_cast<int>(input.size(0));
    const int output_rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * 32;
    MFQ_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        real_width <= padded_width,
        "NINT8-0 packed matmul expects M in [1,8] and K <= packed K");
    MFQ_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == mfq_tensor_backend::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == mfq_tensor_backend::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT8-0 activation workspace is invalid");
    MFQ_RUNTIME_CHECK(
        quantized.device() == input.device() && scale.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT8-0 tensors and workspace must share one CUDA device");
    auto output = mfq_tensor_backend::empty(
        {activation_rows, output_rows}, input.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
    nint_quantize_activation_kernel<<<
        dim3(activation_rows, groups), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(input.data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            activation_rows,
            real_width,
            padded_width,
            groups,
            32);
    nint8_zero_matmul_kernel<<<
        dim3((output_rows + 3) / 4, activation_rows),
        dim3(32, 4),
        0,
        stream>>>(
            reinterpret_cast<const int8_t *>(
                quantized.data_ptr<uint8_t>()),
            reinterpret_cast<const __half *>(scale.data_ptr<mfq_half>()),
            quantized_input.data_ptr<int8_t>(),
            input_scale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),
            activation_rows,
            output_rows,
            groups,
            padded_width);
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f16_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return nint_cublas_gemm_nt_f16acc_cuda(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_mmq_f32_packed_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == mfq_tensor_backend::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 FP32 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return cublas_gemm_nt_f32_output(input, weight);
}


mfq_tensor_backend::Tensor nint8_zero_backward_input_cuda(
        mfq_tensor_backend::Tensor quantized,
        mfq_tensor_backend::Tensor scale,
        mfq_tensor_backend::Tensor output_gradient,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    MFQ_RUNTIME_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == quantized.size(0),
        "NINT8-0 output-gradient geometry is invalid");
    MFQ_RUNTIME_CHECK(
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kBFloat16 ||
        output_gradient.scalar_type() == mfq_tensor_backend::kFloat32,
        "NINT8-0 output gradient must be fp16, bf16, or fp32");
    MFQ_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32 &&
        quantized.device() == output_gradient.device(),
        "NINT8-0 backward dimensions or device are invalid");
    const int rows = static_cast<int>(output_gradient.size(0));
    const int outputs = static_cast<int>(quantized.size(0));
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    auto result = mfq_tensor_backend::empty(
        {rows, width}, output_gradient.options());
    mfq_packed_backward::launch_dense_half_weight(
        output_gradient,
        weight,
        result,
        rows,
        outputs,
        static_cast<int>(width),
        mfq_current_cuda_stream());
    MFQ_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
