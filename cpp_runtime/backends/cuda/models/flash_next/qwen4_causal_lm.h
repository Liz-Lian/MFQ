#pragma once

#include "../../runtime/cuda_transformer.h"
#include "flash_next_model.h"
#include "state.h"
#include "qwen4.h"
#include "flash_next_layers.h"

#include <memory>

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::flash_next::QwenConfig& config,
    int layer);
std::unique_ptr<::Block> load_block(
    const mfq::ModelSource& source,
    const mfq::flash_next::GlmConfig& config,
    int layer);
void validate_load_options(const ::CudaRuntimeParameters& config);
std::unique_ptr<::flash_runtime::Gr> load_final_mixer(
    const mfq::ModelSource& source,
    const ::CudaRuntimeParameters& config);

} // namespace mfq::cuda::flash_next
