#pragma once

#include <memory>
#include <string>

namespace mfq { class ModelSource; }
struct Block;
struct CudaRuntimeParameters;

namespace mfq::cuda::qwen35 {

struct Config;

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const ::CudaRuntimeParameters& runtime,
    const Config& config,
    int layer,
    const std::string& type);

} // namespace mfq::cuda::qwen35
