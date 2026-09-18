#include "common.h"

#include "mfq_tensor_backend.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

mfq_tensor_backend::Tensor embedding_lookup_cuda(
    mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor token_ids);

namespace mfq::bench {

void run_embedding(
    const Options& options,
    const mfq::cuda::Device& gpu,
    const cudaDeviceProp& properties) {
    if (options.dtype == "bf16") {
        usage_error("embedding supports --dtype f16 or f32");
    }
    constexpr std::int64_t vocabulary = 32768;
    auto value_options = mfq::cuda::TensorOptions()
        .device(gpu).dtype(scalar_type(options.dtype));
    auto id_options = mfq::cuda::TensorOptions()
        .device(gpu).dtype(mfq::cuda::kInt64);
    mfq::cuda::manual_seed(options.seed);
    auto weight = mfq::cuda::randn(
        {vocabulary, options.width}, value_options).contiguous();
    std::vector<std::int64_t> host_ids(
        static_cast<std::size_t>(options.rows));
    for (std::int64_t row = 0; row < options.rows; ++row) {
        host_ids[static_cast<std::size_t>(row)] =
            (row * 7919 + 17) % vocabulary;
    }
    auto ids = mfq::cuda::tensor(host_ids, id_options).contiguous();
    auto reference = weight.index_select(0, ids);
    auto call = [&] { return embedding_lookup_cuda(weight, ids); };
    auto output = call();

    Correctness correctness;
    correctness.maximum_output_error = maximum_error(output, reference);
    if (correctness.maximum_output_error != 0.0) {
        throw std::runtime_error("embedding output differs from index_select");
    }
    auto invoke = [&] { output = call(); };
    const auto samples = measure(
        invoke, options.warmup, options.iterations, options.samples);
    print_result(options, properties, correctness, samples);
}

}  // namespace mfq::bench
