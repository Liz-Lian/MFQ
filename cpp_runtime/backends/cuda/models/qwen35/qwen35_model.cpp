#include "qwen35_model.h"

#include "../config.h"
#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace mfq::cuda::qwen35 {

Config Config::from_source(const mfq::ModelSource& source) {
    const auto graph = source.resolved_model_graph();
    if (cuda_backbone(graph.backbone) != CudaBackbone::generic_qwen) {
        throw std::runtime_error(
            "Qwen3.5 CUDA loading requires a Qwen-compatible model graph");
    }
    return from_json(source.model_config_json(), graph);
}

Config Config::from_json(
        std::string_view payload,
        const mfq::ModelGraph& graph) {
    const auto document = nlohmann::json::parse(
        payload.begin(), payload.end(), nullptr, false);
    if (!document.is_object()) {
        throw std::runtime_error("Qwen model config must be an object");
    }
    const auto& text = document.contains("text_config")
        ? document.at("text_config") : document;
    if (!text.is_object()) {
        throw std::runtime_error("Qwen text_config must be an object");
    }

    Config c;
    c.model_type = document.value(
        "model_type", text.value("model_type", std::string{}));
    c.vocab_size = text.at("vocab_size").get<std::int64_t>();
    c.hidden_size = text.at("hidden_size").get<std::int64_t>();
    c.intermediate_size = text.at("intermediate_size").get<std::int64_t>();
    c.num_hidden_layers = text.at("num_hidden_layers").get<std::int64_t>();
    c.num_attention_heads =
        text.at("num_attention_heads").get<std::int64_t>();
    c.num_key_value_heads =
        text.at("num_key_value_heads").get<std::int64_t>();
    c.max_position_embeddings =
        text.at("max_position_embeddings").get<std::int64_t>();
    c.head_dim = text.value(
        "head_dim", c.num_attention_heads > 0
            ? c.hidden_size / c.num_attention_heads : 0);
    if (c.vocab_size <= 0 || c.hidden_size <= 0 ||
            c.intermediate_size <= 0 || c.num_hidden_layers <= 0 ||
            c.num_attention_heads <= 0 || c.num_key_value_heads <= 0 ||
            c.head_dim <= 0 || c.max_position_embeddings <= 0) {
        throw std::runtime_error("invalid Qwen text configuration");
    }
    c.rms_norm_eps = text.value("rms_norm_eps", 1e-6);
    c.tie_word_embeddings = text.value("tie_word_embeddings", false);
    c.attention_output_gate = text.value("attn_output_gate", false);
    c.mtp_num_hidden_layers =
        text.value("mtp_num_hidden_layers", std::int64_t{0});
    c.mtp_use_dedicated_embeddings =
        text.value("mtp_use_dedicated_embeddings", false);
    c.linear_conv_kernel_dim =
        text.value("linear_conv_kernel_dim", std::int64_t{4});
    c.linear_key_head_dim =
        text.value("linear_key_head_dim", std::int64_t{128});
    c.linear_value_head_dim =
        text.value("linear_value_head_dim", std::int64_t{128});
    c.linear_num_key_heads = text.value(
        "linear_num_key_heads", c.num_key_value_heads);
    c.linear_num_value_heads = text.value(
        "linear_num_value_heads", c.num_attention_heads);
    c.layer_types = mfq::cuda::config_json::layer_types(
        text.dump(), c.num_hidden_layers);

    const nlohmann::json empty = nlohmann::json::object();
    const auto& rope_parameters = text.contains("rope_parameters")
        ? text.at("rope_parameters") : empty;
    if (!rope_parameters.is_object()) {
        throw std::runtime_error("Qwen rope_parameters must be an object");
    }
    const auto& full_rope = rope_parameters.contains("full_attention")
        ? rope_parameters.at("full_attention") : empty;
    if (!full_rope.is_object()) {
        throw std::runtime_error(
            "Qwen rope_parameters.full_attention must be an object");
    }
    c.rope_base = full_rope.value(
        "rope_theta", rope_parameters.value(
            "rope_theta", text.value("rope_theta", 1'000'000.0)));
    c.full_rotary_factor = full_rope.value(
        "partial_rotary_factor", rope_parameters.value(
            "partial_rotary_factor",
            text.value("partial_rotary_factor", 1.0)));
    c.rotary_dim = static_cast<std::int64_t>(std::llround(
        c.full_rotary_factor * static_cast<double>(c.head_dim)));

    if (graph.component("vision") != nullptr) {
        if (!document.contains("vision_config") ||
                !document.at("vision_config").is_object()) {
            throw std::runtime_error(
                "Qwen grid-ViT requires an object vision_config");
        }
        const auto& vision = document.at("vision_config");
        if (vision.value(
                "hidden_act", std::string("gelu_pytorch_tanh")) !=
                "gelu_pytorch_tanh") {
            throw std::runtime_error(
                "Qwen grid-ViT requires gelu_pytorch_tanh hidden_act");
        }
        mfq::GridVisionConfig parsed;
        parsed.hidden_size = vision.at("hidden_size").get<std::int64_t>();
        parsed.intermediate_size =
            vision.at("intermediate_size").get<std::int64_t>();
        parsed.depth = vision.at("depth").get<std::int64_t>();
        parsed.num_heads = vision.at("num_heads").get<std::int64_t>();
        parsed.in_channels = vision.value("in_channels", std::int64_t{3});
        parsed.patch_size = vision.at("patch_size").get<std::int64_t>();
        parsed.temporal_patch_size =
            vision.at("temporal_patch_size").get<std::int64_t>();
        parsed.spatial_merge_size =
            vision.at("spatial_merge_size").get<std::int64_t>();
        parsed.out_hidden_size =
            vision.at("out_hidden_size").get<std::int64_t>();
        parsed.num_position_embeddings =
            vision.at("num_position_embeddings").get<std::int64_t>();
        parsed.rope_theta = vision.value("rope_theta", 10'000.0);
        if (vision.contains("rope_parameters")) {
            if (!vision.at("rope_parameters").is_object()) {
                throw std::runtime_error(
                    "vision_config.rope_parameters must be an object");
            }
            parsed.rope_theta = vision.at("rope_parameters").value(
                "rope_theta", parsed.rope_theta);
        }
        parsed.layer_norm_eps = vision.value("layer_norm_eps", 1e-6);
        parsed.validate();
        if (parsed.out_hidden_size != c.hidden_size) {
            throw std::runtime_error(
                "Qwen grid-ViT output width must equal text hidden_size");
        }
        const auto token_id = [&](const char* name) {
            if (!document.contains(name) ||
                    !document.at(name).is_number_integer()) {
                throw std::runtime_error(
                    std::string("Qwen grid-ViT requires ") + name);
            }
            const auto value = document.at(name).get<std::int64_t>();
            if (value < 0 || value >= c.vocab_size) {
                throw std::runtime_error(
                    std::string("invalid Qwen grid-ViT ") + name);
            }
            return value;
        };
        c.image_token_id = token_id("image_token_id");
        c.video_token_id = token_id("video_token_id");
        if (*c.image_token_id == *c.video_token_id) {
            throw std::runtime_error(
                "Qwen grid-ViT placeholder token IDs must differ");
        }
        const auto* rope = full_rope.contains("mrope_section")
            ? &full_rope : &rope_parameters;
        if (!rope->contains("mrope_section") ||
                !rope->at("mrope_section").is_array()) {
            throw std::runtime_error(
                "Qwen grid-MRoPE requires rope mrope_section");
        }
        c.mrope_sections =
            rope->at("mrope_section").get<std::vector<std::int64_t>>();
        c.mrope_interleaved =
            rope_parameters.value("mrope_interleaved", false);
        if (c.mrope_sections.size() != 3 ||
                std::any_of(
                    c.mrope_sections.begin(), c.mrope_sections.end(),
                    [](std::int64_t value) { return value < 0; }) ||
                std::accumulate(
                    c.mrope_sections.begin(), c.mrope_sections.end(),
                    std::int64_t{0}) != c.rotary_dim / 2) {
            throw std::runtime_error(
                "Qwen grid-MRoPE sections must be three nonnegative values summing to rotary_dim / 2");
        }
        c.grid_vision = parsed;
    }
    return c;
}

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    const auto parsed = embedded_config
        ? Config::from_source(source)
        : Config::from_json(payload, model_graph);
    if (model_graph.component("vision") != nullptr &&
            runtime_plan.vision != mfq::cuda::CudaVisionAdapter::grid_vit) {
        throw std::runtime_error(
            "Qwen CUDA vision requires grid_vit/grid_vision.v1/grid_mrope");
    }

    CudaRuntimeParameters c;
    c.model_graph = model_graph;
    c.runtime_plan = runtime_plan;
    c.model_type = parsed.model_type;
    c.vocab_size = parsed.vocab_size;
    c.hidden_size = parsed.hidden_size;
    c.intermediate_size = parsed.intermediate_size;
    c.num_hidden_layers = parsed.num_hidden_layers;
    c.num_attention_heads = parsed.num_attention_heads;
    c.num_key_value_heads = parsed.num_key_value_heads;
    c.max_position_embeddings = parsed.max_position_embeddings;
    c.head_dim = parsed.head_dim;
    c.rms_norm_eps = parsed.rms_norm_eps;
    c.tie_word_embeddings = parsed.tie_word_embeddings;
    c.mtp_num_hidden_layers = parsed.mtp_num_hidden_layers;
    c.layer_types = parsed.layer_types;
    c.rope_base = parsed.rope_base;
    c.full_rotary_factor = parsed.full_rotary_factor;
    c.rotary_dim = parsed.rotary_dim;
    return c;
}

} // namespace mfq::cuda::qwen35
