#include "server_components.h"

#include "../models/qwen35/qwen35_model.h"

#include <iostream>
#include <stdexcept>

CudaRuntimeComponents load_cuda_runtime_components(
        CudaModel& model,
        bool load_optional_components) {
    CudaRuntimeComponents result;
    result.graph = model.c.model_graph;
    result.plan = model.c.runtime_plan;
    if (!load_optional_components) return result;

    switch (result.plan.vision) {
        case mfq::cuda::CudaVisionAdapter::none:
            break;
        case mfq::cuda::CudaVisionAdapter::grid_vit: {
            const auto* component = result.graph.component("vision");
            const auto config = mfq::cuda::qwen35::Config::from_json(
                model.c.resolved_config_json, result.graph);
            if (component == nullptr || !config.grid_vision ||
                    !config.image_token_id || !config.video_token_id) {
                throw std::runtime_error(
                    "CUDA grid-Vision configuration is incomplete");
            }
            result.grid_vision.emplace(
                mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent::load(
                    *model.source, *config.grid_vision,
                    *config.image_token_id, *config.video_token_id,
                    component->input_contract, component->position_policy));
            result.vision_available = true;
            break;
        }
        case mfq::cuda::CudaVisionAdapter::minicpmo45:
            result.minicpmo.emplace(
                MiniCPMO45Runtime::load_with_language(
                    std::move(model)));
            result.vision_available = true;
            break;
    }
    if (result.plan.predictor == mfq::cuda::CudaPredictorAdapter::qwen35) {
        const bool supported_placement = !g_layer_placement.enabled() &&
            g_dense_cpu_layer_count == 0 && g_dsv4_cpu_offload_layers.empty() && !g_moe_expert_cache;
        if (supported_placement && model.c.num_experts == 0 && model.supports_qwen_speculation()) {
            auto predictor = CudaQwen35Mtp::load_if_present(
                *model.source, model.c);
            if (predictor) {
                result.mtp = std::make_unique<CudaQwen35Mtp>(std::move(*predictor));
            }
            result.mtp_available = static_cast<bool>(result.mtp);
        } else {
            std::cerr << "qwen_mtp unavailable: CUDA adapter requires dense GPU-resident Qwen blocks\n";
        }
    }
    if (result.plan.predictor == mfq::cuda::CudaPredictorAdapter::flash_next) {
        MFQ_RUNTIME_CHECK(model.c.is_flash_next() && model.supports_qwen_speculation(),
            "invalid Flash-Next predictor backbone");
        auto predictor = CudaFlashNextMtp::load_if_present(
            *model.source, model.c);
        if (predictor) {
            result.mtp = std::make_unique<CudaFlashNextMtp>(std::move(*predictor));
        }
        result.mtp_available = static_cast<bool>(result.mtp);
    }
    if (result.plan.predictor ==
            mfq::cuda::CudaPredictorAdapter::deepseek_v41_dspark) {
        MFQ_RUNTIME_CHECK(
            model.c.is_deepseek_v41() &&
                model.supports_deepseek_v41_speculation(),
            "invalid DeepSeek-V4.1 DSpark backbone");
        MFQ_RUNTIME_CHECK(
            model.deepseek_v41_state,
            "DeepSeek-V4.1 shared state is unavailable");
        result.mtp =
            mfq::cuda::deepseek_v41_runtime::load_dspark_if_present(
                *model.source, model.c,
                model.deepseek_v41_state->config);
        result.mtp_available = static_cast<bool>(result.mtp);
    }
    return result;
}
