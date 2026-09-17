#include "glm_dsa_model.h"

#include "../config.h"
#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfq::cuda::glm_dsa {

Config Config::from_json(
        std::string_view payload,
        const CudaRuntimeParameters& common) {
    using namespace mfq::cuda::config_json;
    const std::string source(payload);
    Config config;
    config.indexer_types = string_array(source, "indexer_types");
    config.mlp_layer_types = string_array(source, "mlp_layer_types");
    const auto index_topk_frequency = integer(source, "index_topk_freq", 1);
    const auto index_skip_topk_offset = integer(
        source, "index_skip_topk_offset", 0);
    const auto first_dense_layers = integer(source, "first_k_dense_replace", 0);
    const auto moe_layer_frequency = integer(source, "moe_layer_freq", 1);
    if (config.indexer_types.empty()) {
        config.indexer_types.reserve(
            static_cast<std::size_t>(common.num_hidden_layers));
        const auto frequency = std::max<std::int64_t>(
            1, index_topk_frequency);
        for (std::int64_t layer = 0;
             layer < common.num_hidden_layers; ++layer) {
            const auto phase = std::max<std::int64_t>(
                layer - index_skip_topk_offset + 1, 0);
            config.indexer_types.push_back(
                phase % frequency == 0 ? "full" : "shared");
        }
    }
    if (config.mlp_layer_types.empty()) {
        config.mlp_layer_types.reserve(
            static_cast<std::size_t>(common.num_hidden_layers));
        for (std::int64_t layer = 0;
             layer < common.num_hidden_layers; ++layer) {
            const bool sparse = layer >= first_dense_layers &&
                layer % std::max<std::int64_t>(1, moe_layer_frequency) == 0;
            config.mlp_layer_types.push_back(sparse ? "sparse" : "dense");
        }
    }
    if (config.indexer_types.size() !=
            static_cast<std::size_t>(common.num_hidden_layers) ||
            config.mlp_layer_types.size() !=
            static_cast<std::size_t>(common.num_hidden_layers)) {
        throw std::runtime_error(
            "GLM DSA layer schedules do not match num_hidden_layers");
    }
    bool have_full_indexer = false;
    for (std::size_t layer = 0; layer < config.indexer_types.size(); ++layer) {
        const auto& indexer = config.indexer_types[layer];
        if (indexer == "full") {
            have_full_indexer = true;
        } else if (indexer != "shared" || !have_full_indexer) {
            throw std::runtime_error(
                "invalid GLM DSA indexer schedule at layer " +
                std::to_string(layer));
        }
        const auto& mlp = config.mlp_layer_types[layer];
        if (mlp != "dense" && mlp != "sparse") {
            throw std::runtime_error(
                "invalid GLM DSA MLP schedule at layer " +
                std::to_string(layer));
        }
    }
    return config;
}

Config Config::from_source(
        const mfq::ModelSource& source,
        const CudaRuntimeParameters& common) {
    return from_json(source.model_config_json(), common);
}

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    using namespace mfq::cuda::config_json;
    const auto s = embedded_config
        ? source.model_config_json() : std::string(payload);
    auto c = make_common_runtime_parameters(s, model_graph, runtime_plan);
    c.num_experts = integer(
        s, "num_experts", integer(s, "n_routed_experts", 0));
    c.num_experts_per_tok = integer(
        s, "num_experts_per_tok", integer(s, "top_k_experts", 0));
    c.moe_intermediate_size = integer(s, "moe_intermediate_size", 0);
    const auto shared_expert_count = integer(s, "n_shared_experts", 0);
    c.shared_expert_intermediate_size =
        integer(s, "shared_expert_intermediate_size", 0);
    c.q_lora_rank = integer(s, "q_lora_rank", 0);
    c.kv_lora_rank = integer(s, "kv_lora_rank", 0);
    c.qk_nope_head_dim = integer(s, "qk_nope_head_dim", 0);
    c.qk_rope_head_dim = integer(s, "qk_rope_head_dim", 0);
    c.v_head_dim = integer(s, "v_head_dim", 0);
    c.index_head_dim = integer(s, "index_head_dim", 0);
    c.index_n_heads = integer(s, "index_n_heads", 0);
    c.index_topk = integer(s, "index_topk", 0);
    const auto expert_group_count = integer(s, "n_group", 1);
    const auto selected_group_count = integer(s, "topk_group", 1);
    c.routed_scaling_factor = number(s, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = boolean(s, "norm_topk_prob", false);
    c.swiglu_limit = number(s, "swiglu_limit", 0.0);
    c.expert_gating_func = string(s, "scoring_func", "softmax");
        c.layer_types.assign(static_cast<size_t>(c.num_hidden_layers), "glm_dsa");
        c.rotary_dim = c.qk_rope_head_dim;
        if (c.shared_expert_intermediate_size <= 0) {
            c.shared_expert_intermediate_size =
                shared_expert_count * c.moe_intermediate_size;
        }
        static_cast<void>(embedded_config
            ? Config::from_source(source, c)
            : Config::from_json(payload, c));
        if (c.q_lora_rank <= 0 || c.kv_lora_rank <= 0 ||
            c.qk_nope_head_dim <= 0 || c.qk_rope_head_dim <= 0 ||
            c.v_head_dim <= 0 || c.index_head_dim <= 0 ||
            c.index_n_heads <= 0 || c.index_topk <= 0 ||
            c.num_experts <= 0 || c.num_experts_per_tok <= 0 ||
            c.num_attention_heads != 64 || c.num_key_value_heads != 64 ||
            c.kv_lora_rank != 512 ||
            c.qk_nope_head_dim != 192 || c.qk_rope_head_dim != 64 ||
            c.v_head_dim != 256 || c.index_head_dim != 128 ||
            c.index_n_heads != 32 || c.index_topk != 2048 ||
            mfq::cuda::config_json::integer(s, "qk_head_dim", 0) !=
                c.qk_nope_head_dim + c.qk_rope_head_dim ||
            mfq::cuda::config_json::boolean(s, "attention_bias", false) ||
            !mfq::cuda::config_json::boolean(s, "rope_interleave", true) ||
            !mfq::cuda::config_json::boolean(s, "indexer_rope_interleave", true) ||
            mfq::cuda::config_json::string(s, "hidden_act") != "silu" ||
            expert_group_count != 1 || selected_group_count != 1 ||
            shared_expert_count != 1 ||
            mfq::cuda::config_json::string(s, "scoring_func") != "sigmoid" ||
            mfq::cuda::config_json::string(s, "topk_method") != "noaux_tc") {
            throw std::runtime_error("unsupported GLM DSA configuration");
        }
        return c;
}

} // namespace mfq::cuda::glm_dsa
