#include "flash_next_model.h"

#include "../cuda_model_config.h"
#include "mfq/model_source.h"

namespace mfq::flash_next {

QwenConfig QwenConfig::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

QwenConfig QwenConfig::from_source(const mfq::ModelSource& source) {
    const auto graph = source.resolved_model_graph();
    if (graph.backbone != "qwen4_exp") {
        throw std::runtime_error(
            "Qwen4-Exp CUDA loading requires a qwen4_exp model graph");
    }
    return from_json(source.model_config_json());
}

GlmConfig GlmConfig::from_json(std::string_view payload) {
    return parse(nlohmann::json::parse(payload));
}

GlmConfig GlmConfig::from_source(const mfq::ModelSource& source) {
    const auto graph = source.resolved_model_graph();
    if (graph.backbone != "glm5_next") {
        throw std::runtime_error(
            "GLM5-Next CUDA loading requires a glm5_next model graph");
    }
    return from_json(source.model_config_json());
}

} // namespace mfq::flash_next

namespace mfq::cuda::flash_next {

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    CudaRuntimeParameters c;
    c.model_graph = model_graph;
    c.runtime_plan = runtime_plan;
    if (runtime_plan.backbone == mfq::cuda::CudaBackbone::glm5_next) {
        const auto parsed = embedded_config
            ? mfq::flash_next::GlmConfig::from_source(source)
            : mfq::flash_next::GlmConfig::from_json(payload);
        c.model_type="glm5_next";
        c.vocab_size=parsed.vocab; c.hidden_size=parsed.hidden;
        c.intermediate_size=parsed.intermediate; c.num_hidden_layers=parsed.layers;
        c.max_position_embeddings=parsed.maximum; c.num_attention_heads=parsed.heads;
        c.num_key_value_heads=1; c.head_dim=parsed.nope;
        c.hc_mult=parsed.streams; c.rms_norm_eps=parsed.eps; c.norm_weight_offset=0;
        c.tie_word_embeddings=parsed.tied_embeddings;
        c.layer_types=parsed.layer_types;
        c.num_experts=parsed.experts; c.num_experts_per_tok=parsed.topk;
        c.moe_intermediate_size=parsed.moe_intermediate;
        c.mtp_num_hidden_layers=parsed.predictor_layers;
        return c;
    }
    if (runtime_plan.backbone == mfq::cuda::CudaBackbone::qwen4_exp) {
        const auto parsed = embedded_config
            ? mfq::flash_next::QwenConfig::from_source(source)
            : mfq::flash_next::QwenConfig::from_json(payload);
        c.model_type="qwen4_exp";
        c.vocab_size=parsed.vocab;c.hidden_size=parsed.hidden;c.num_hidden_layers=parsed.layers;
        c.max_position_embeddings=parsed.maximum;c.num_attention_heads=parsed.heads;c.num_key_value_heads=parsed.kv_heads;
        c.head_dim=parsed.width;c.hc_mult=parsed.streams;c.rms_norm_eps=parsed.eps;c.norm_weight_offset=0;
        c.tie_word_embeddings=parsed.tied_embeddings;c.layer_types=parsed.layer_types;
        c.num_experts=parsed.experts;c.num_experts_per_tok=parsed.topk;c.moe_intermediate_size=parsed.moe_width;
        c.mtp_num_hidden_layers=parsed.predictor_layers;
        return c;
    }
    throw std::runtime_error("invalid Flash-Next config dispatch");
}

} // namespace mfq::cuda::flash_next
