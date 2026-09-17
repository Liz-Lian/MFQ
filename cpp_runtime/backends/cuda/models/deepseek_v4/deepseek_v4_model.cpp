#include "deepseek_v4_model.h"

#include "../config.h"
#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfq::cuda::deepseek_v4 {

Config Config::from_json(
        std::string_view payload,
        std::int64_t num_hidden_layers) {
    using namespace mfq::cuda::config_json;
    const std::string source(payload);
    Config config;
    config.compress_ratios = integer_array(source, "compress_ratios");
    config.hash_layer_count = integer(
        source, "num_hash_layers", integer(source, "n_hash_layers", 0));
    config.compress_rope_base = number(source, "compress_rope_theta", 0.0);
    config.rope_original_positions = static_cast<std::int64_t>(object_number(
        source, "rope_scaling", "original_max_position_embeddings", 0.0));
    config.rope_factor = object_number(source, "rope_scaling", "factor", 1.0);
    config.rope_beta_fast = object_number(
        source, "rope_scaling", "beta_fast", 32.0);
    config.rope_beta_slow = object_number(
        source, "rope_scaling", "beta_slow", 1.0);
    if (config.compress_ratios.size() <
            static_cast<std::size_t>(num_hidden_layers)) {
        throw std::runtime_error(
            "DeepSeek V4 compress_ratios is shorter than num_hidden_layers");
    }
    config.compress_ratios.resize(
        static_cast<std::size_t>(num_hidden_layers));
    return config;
}

Config Config::from_source(
        const mfq::ModelSource& source,
        std::int64_t num_hidden_layers) {
    return from_json(source.model_config_json(), num_hidden_layers);
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
    c.hc_mult = integer(s, "hc_mult", 1);
    c.hc_sinkhorn_iters = integer(s, "hc_sinkhorn_iters", 0);
    c.hc_eps = number(s, "hc_eps", 1e-6);
    c.o_groups = integer(s, "o_groups", 1);
    c.o_lora_rank = integer(s, "o_lora_rank", 0);
    c.swiglu_limit = number(s, "swiglu_limit", 0.0);
    static_cast<void>(embedded_config
        ? Config::from_source(source, c.num_hidden_layers)
        : Config::from_json(payload, c.num_hidden_layers));
    c.routed_scaling_factor = number(s, "routed_scaling_factor", 1.0);
    c.norm_topk_prob = boolean(s, "norm_topk_prob", false);
    c.expert_gating_func = string(s, "scoring_func", "softmax");

        c.norm_weight_offset = 0.0;
        c.layer_types.assign(
            static_cast<size_t>(c.num_hidden_layers), "deepseek_v4");
        if (c.shared_expert_intermediate_size <= 0) {
            c.shared_expert_intermediate_size =
                shared_expert_count * c.moe_intermediate_size;
        }
        if (c.hidden_size != 4096 || c.num_attention_heads != 64 ||
            c.head_dim != 512 || c.q_lora_rank != 1024 ||
            c.qk_rope_head_dim != 64 || c.index_head_dim != 128 ||
            c.index_n_heads != 64 || c.index_topk != 512 ||
            c.o_groups != 8 || c.o_lora_rank != 1024 ||
            c.hc_mult != 4 || c.hc_sinkhorn_iters != 20 ||
            c.num_experts != 256 || c.num_experts_per_tok != 6 ||
            c.moe_intermediate_size != 2048 ||
            shared_expert_count != 1 ||
            c.expert_gating_func != "sqrtsoftplus") {
            throw std::runtime_error("unsupported DeepSeek V4 configuration");
        }
        return c;
}

} // namespace mfq::cuda::deepseek_v4
