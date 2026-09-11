#pragma once
#include "cpp_runtime/backends/cuda/include/mfq_tensor_backend.h"
#include "cpp_runtime/backends/cuda/include/mfq_mxfp4_sq_blob.h"

// Validate the CPU v1 header with mfq::sq::parse before upload. These
// capture-safe entry points check the descriptor without GPU-to-CPU reads.
mfq_tensor_backend::Tensor mxfp4_sq_dequant_cuda(
    mfq_tensor_backend::Tensor blob, std::int64_t bits, std::int64_t outputs,
    std::int64_t width, std::int64_t base, bool fp32);
mfq_tensor_backend::Tensor mxfp4_sq_matmul_cuda(
    mfq_tensor_backend::Tensor blob, mfq_tensor_backend::Tensor input,
    std::int64_t bits, std::int64_t outputs, std::int64_t width, std::int64_t base);
void mxfp4_sq_moe_matmul_cuda(
    mfq_tensor_backend::Tensor blob,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor expert_ids,
    mfq_tensor_backend::Tensor expert_local,
    std::int64_t bits,
    std::int64_t n_experts,
    std::int64_t local_experts,
    std::int64_t out_per_expert,
    std::int64_t width,
    std::int64_t base,
    mfq_tensor_backend::Tensor output);
mfq_tensor_backend::Tensor mxfp4_sq_backward_input_cuda(
    mfq_tensor_backend::Tensor blob, mfq_tensor_backend::Tensor output_gradient,
    std::int64_t bits, std::int64_t outputs, std::int64_t width,
    std::int64_t base);
