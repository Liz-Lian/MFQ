#pragma once

#include "cuda_model_config.h"

#include "mfq/model_source.h"

#include <string>

namespace mfq::cuda {

inline constexpr const char* kTokenizerGgufAsset =
    "__mfq_asset__/tokenizer.gguf";

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    const std::string& external_path = {});

} // namespace mfq::cuda
