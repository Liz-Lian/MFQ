#include "qwen4_causal_lm.h"

#include "../../runtime/cuda_transformer.h"
#include "flash_next_model.h"
#include "flash_next_layers.h"

namespace mfq::cuda::flash_next {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::flash_next::QwenConfig& config,
        int layer) {
    return std::make_unique<::Qwen4Block>(source, config, layer);
}

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const mfq::flash_next::GlmConfig& config,
        int layer) {
    return std::make_unique<::Glm5NextBlock>(source, config, layer);
}

void validate_load_options(const ::CudaRuntimeParameters& config) {
    if (!config.is_flash_next()) return;
    if (g_tensor_parallel.enabled() || g_layer_placement.enabled() ||
            g_n_gpu_layers >= 0 || g_moe_expert_cache) {
        throw std::runtime_error(
            "Flash-Next native adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload still require a different placement path");
    }
}

std::unique_ptr<::flash_runtime::Gr> load_final_mixer(
        const mfq::ModelSource& source,
        const ::CudaRuntimeParameters& runtime) {
    auto config = mfq::flash_next::QwenConfig::from_json(
        runtime.resolved_config_json);
    config.maximum = runtime.max_position_embeddings;
    return std::make_unique<::flash_runtime::Gr>(
        source, config, "model.mhc.pre", false);
}

} // namespace mfq::cuda::flash_next
