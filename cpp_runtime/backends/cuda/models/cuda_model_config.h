#pragma once

#include "cuda_model_plan.h"
#include "mfq/model_graph.h"
#include <cstdint>
#include <string>
#include <vector>

// Common CUDA execution parameters produced by architecture-owned loaders.
// Architecture Config parsing stays in each *_model.cpp.

struct CudaRuntimeParameters {
    std::string model_type;
    std::string resolved_config_json;
    std::string tensor_root = "model";
    mfq::ModelGraph model_graph;
    mfq::cuda::CudaModelPlan runtime_plan;
    int64_t vocab_size = 0, hidden_size = 0, intermediate_size = 0, num_hidden_layers = 0;
    int64_t num_attention_heads = 0, num_key_value_heads = 0, max_position_embeddings = 0;
    int64_t head_dim = 0;
    double rope_base = 1000000.0;
    int64_t rotary_dim = 0;
    double full_rotary_factor = 1.0;
    double final_logit_softcapping = 0.0;
    double embed_scale = 1.0;
    double rms_norm_eps = 1e-6;
    bool tie_word_embeddings = false;
    int64_t mtp_num_hidden_layers = 0;
    int64_t num_experts = 0, num_experts_per_tok = 0;
    int64_t moe_intermediate_size = 0, shared_expert_intermediate_size = 0;
    int64_t q_lora_rank = 0, kv_lora_rank = 0;
    int64_t qk_nope_head_dim = 0, qk_rope_head_dim = 0, v_head_dim = 0;
    int64_t index_head_dim = 0, index_n_heads = 0, index_topk = 0;
    int64_t hc_mult = 1, hc_sinkhorn_iters = 0;
    int64_t o_groups = 1, o_lora_rank = 0;
    double routed_scaling_factor = 1.0;
    double hc_eps = 1e-6, swiglu_limit = 0.0;
    bool norm_topk_prob = false;
    std::string expert_gating_func = "softmax";
    double norm_weight_offset = 1.0;
    std::vector<std::string> layer_types;
    bool is_qwen4() const { return runtime_plan.backbone == mfq::cuda::CudaBackbone::qwen4_exp; }
    bool is_flash_next() const { return is_qwen4() || is_glm5_next(); }
    bool is_glm5_next() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::glm5_next;
    }
    bool is_gemma4() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::gemma4;
    }
    bool is_glm_dsa() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::glm_dsa;
    }
    bool is_dsv4() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::deepseek_v4;
    }
    bool is_deepseek_v41() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::deepseek_v41;
    }
    bool is_minicpmo45() const {
        return runtime_plan.backbone == mfq::cuda::CudaBackbone::minicpmo45;
    }
    bool uses_minicpmo45_bf16_graph() const {
        return is_minicpmo45() ||
            runtime_plan.backbone == mfq::cuda::CudaBackbone::minicpmo_tts;
    }
    int64_t attention_size() const { return num_attention_heads * head_dim; }
    int64_t kv_size() const { return num_key_value_heads * head_dim; }
};

CudaRuntimeParameters make_common_runtime_parameters(
    const std::string& source,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

