#include "deepseek_v41_model.h"

#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfq::cuda::deepseek_v41 {

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        std::string_view payload,
        bool embedded_config,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    const auto config = embedded_config
        ? source.model_config_json() : std::string(payload);
    const auto document = nlohmann::json::parse(config, nullptr, false);
    CudaRuntimeParameters c;
    c.model_graph = model_graph;
    c.runtime_plan = runtime_plan;

        const auto parsed = embedded_config
            ? mfq::models::deepseek_v41::Config::from_source(source)
            : mfq::models::deepseek_v41::Config::from_json(payload);
        c.model_type = parsed.text_model_type;
        c.vocab_size = parsed.vocab;
        c.hidden_size = parsed.hidden;
        c.num_hidden_layers = parsed.n_layers;
        c.num_attention_heads = parsed.n_heads;
        c.num_key_value_heads = parsed.n_kv_heads;
        c.max_position_embeddings = parsed.max_position_embeddings;
        c.head_dim = parsed.head_dim;
        c.rope_base = parsed.rope_theta;
        c.rotary_dim = parsed.rope_head_dim;
        c.rms_norm_eps = parsed.rms_eps;
        c.norm_weight_offset = 0.0;
        c.tie_word_embeddings = document.value("tie_word_embeddings", false);
        c.mtp_num_hidden_layers = parsed.n_mtp_layers;
        c.num_experts = parsed.n_experts;
        c.num_experts_per_tok = parsed.top_k;
        c.moe_intermediate_size = parsed.moe_inter;
        c.shared_expert_intermediate_size = parsed.n_shared * parsed.moe_inter;
        c.q_lora_rank = parsed.q_lora_rank;
        c.qk_nope_head_dim = parsed.head_dim - parsed.rope_head_dim;
        c.qk_rope_head_dim = parsed.rope_head_dim;
        c.v_head_dim = parsed.head_dim;
        c.index_head_dim = parsed.index_head_dim;
        c.index_n_heads = parsed.index_n_heads;
        c.index_topk = parsed.index_topk;
        c.hc_mult = parsed.hc_mult;
        c.hc_sinkhorn_iters = parsed.hc_sinkhorn_iters;
        c.hc_eps = parsed.hc_eps;
        c.o_groups = parsed.o_groups;
        c.o_lora_rank = parsed.o_lora_rank;
        c.swiglu_limit = parsed.swiglu_limit;
        c.routed_scaling_factor = parsed.routed_scaling;
        c.norm_topk_prob = parsed.norm_topk_prob;
        c.expert_gating_func = "sqrtsoftplus";
        c.layer_types.assign(
            static_cast<std::size_t>(parsed.n_layers), "deepseek_v41");
        return c;
    }

} // namespace mfq::cuda::deepseek_v41
