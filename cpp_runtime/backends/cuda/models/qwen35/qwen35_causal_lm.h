#pragma once

#include "models/qwen35.h"

#include <memory>
#include <string>
#include <string_view>

namespace mfq { class ModelSource; }
struct Block;

namespace mfq::cuda::qwen35 {

using Config = mfq::models::qwen35::Config;

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const Config& config,
    int layer,
    const std::string& type,
    std::string_view tensor_root = "model");

} // namespace mfq::cuda::qwen35
