#include "common.h"

#include "mfq_tensor_backend.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

std::vector<mfq_tensor_backend::Tensor> kv_cache_write_cuda(
    mfq_tensor_backend::Tensor k_cache,
    mfq_tensor_backend::Tensor v_cache,
    mfq_tensor_backend::Tensor k,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor positions);

namespace mfq::bench {

void run_kv_cache(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    constexpr std::int64_t batch = 1;
    constexpr std::int64_t heads = 8;
    const std::int64_t tokens = options.rows;
    const std::int64_t capacity = std::max<std::int64_t>(4096, tokens);
    auto value_options = mfq::cuda::TensorOptions()
        .device(gpu).dtype(scalar_type(options.dtype));
    auto id_options = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kInt64);
    mfq::cuda::manual_seed(options.seed);
    auto k = mfq::cuda::randn(
        {batch, heads, tokens, options.width}, value_options).contiguous();
    auto v = mfq::cuda::randn(
        {batch, heads, tokens, options.width}, value_options).contiguous();
    auto k_cache = mfq::cuda::zeros(
        {batch, heads, capacity, options.width}, value_options);
    auto v_cache = mfq::cuda::zeros(
        {batch, heads, capacity, options.width}, value_options);
    std::vector<std::int64_t> host_positions(
        static_cast<std::size_t>(tokens));
    for (std::int64_t token = 0; token < tokens; ++token) {
        host_positions[static_cast<std::size_t>(token)] = token;
    }
    auto positions = mfq::cuda::tensor(host_positions, id_options).contiguous();

    auto call = [&] {
        return kv_cache_write_cuda(k_cache, v_cache, k, v, positions);
    };
    auto output = call();
    Correctness correctness;
    const auto written_k = k_cache.narrow(2, 0, tokens);
    const auto written_v = v_cache.narrow(2, 0, tokens);
    correctness.maximum_output_error = std::max(
        maximum_error(written_k, k), maximum_error(written_v, v));
    if (correctness.maximum_output_error != 0.0) {
        throw std::runtime_error("kv-cache output differs from input rows");
    }
    auto invoke = [&] { output = call(); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
