#include "minicpmo45_model.h"

#include "../config.h"
#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfq::cuda::minicpmo45 {

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
        const auto config = embedded_config
            ? source.model_config_json() : std::string(payload);
        const auto document = nlohmann::json::parse(config, nullptr, false);
        if (!document.is_object()) {
            throw std::runtime_error("MiniCPM-o model config must be an object");
        }
        if (document.value("version", std::string{}) != "4.5") {
            throw std::runtime_error(
                "only MiniCPM-o version 4.5 is supported");
        }
        CudaRuntimeParameters c;
        c.model_graph = model_graph;
        c.runtime_plan = runtime_plan;
        c.model_type = document.value("model_type", model_graph.architecture);
        c.vocab_size = document.at("vocab_size").get<int64_t>();
        c.hidden_size = document.at("hidden_size").get<int64_t>();
        c.intermediate_size = document.at("intermediate_size").get<int64_t>();
        c.num_hidden_layers = document.at("num_hidden_layers").get<int64_t>();
        c.num_attention_heads = document.at("num_attention_heads").get<int64_t>();
        c.num_key_value_heads = document.at("num_key_value_heads").get<int64_t>();
        c.max_position_embeddings =
            document.at("max_position_embeddings").get<int64_t>();
        c.head_dim = document.value(
            "head_dim", c.hidden_size / c.num_attention_heads);
        c.rope_base = document.value("rope_theta", 1000000.0);
        c.rotary_dim = c.head_dim;
        c.rms_norm_eps = document.value("rms_norm_eps", 1e-6);
        c.tie_word_embeddings =
            document.value("tie_word_embeddings", false);
        c.norm_weight_offset = 0.0;
        c.layer_types.assign(
            static_cast<size_t>(c.num_hidden_layers), "full_attention");
        if (c.hidden_size != 4096 || c.intermediate_size != 12288 ||
            c.num_hidden_layers != 36 || c.num_attention_heads != 32 ||
            c.num_key_value_heads != 8 || c.head_dim != 128 ||
            document.value("hidden_act", std::string{}) != "silu" ||
            document.value("attention_bias", true) ||
            document.value("use_sliding_window", true)) {
            throw std::runtime_error(
                "unsupported MiniCPM-o 4.5 Qwen3 configuration");
        }
        return c;
    }

CudaRuntimeParameters load_tts_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    const auto config = embedded_config
        ? source.model_config_json() : std::string(payload);
    return make_common_runtime_parameters(config, model_graph, runtime_plan);
}

} // namespace mfq::cuda::minicpmo45
