#pragma once

#include <memory>
#include <string>

namespace mfq { class ModelSource; }
struct Block;
struct CudaRuntimeParameters;

std::unique_ptr<Block> load_transformer_block(
    const mfq::ModelSource& source,
    const CudaRuntimeParameters& config,
    int layer,
    const std::string& type);
