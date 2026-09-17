#pragma once

#include "cuda_model_plan.h"
#include "grid_vision.h"
#include "mfq/model_graph.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct CudaRuntimeParameters;

namespace mfq { class ModelSource; }

namespace mfq::cuda::qwen35 {

struct Config {
    std::string model_type;
    std::int64_t vocab_size = 0;
    std::int64_t hidden_size = 0;
    std::int64_t intermediate_size = 0;
    std::int64_t num_hidden_layers = 0;
    std::int64_t num_attention_heads = 0;
    std::int64_t num_key_value_heads = 0;
    std::int64_t max_position_embeddings = 0;
    std::int64_t head_dim = 0;
    double rms_norm_eps = 1e-6;
    bool tie_word_embeddings = false;
    bool attention_output_gate = false;
    std::int64_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;
    std::int64_t linear_conv_kernel_dim = 4;
    std::int64_t linear_key_head_dim = 128;
    std::int64_t linear_value_head_dim = 128;
    std::int64_t linear_num_key_heads = 0;
    std::int64_t linear_num_value_heads = 0;
    std::vector<std::string> layer_types;
    double rope_base = 1'000'000.0;
    double full_rotary_factor = 1.0;
    std::int64_t rotary_dim = 0;
    std::vector<std::int64_t> mrope_sections;
    bool mrope_interleaved = false;
    std::optional<mfq::GridVisionConfig> grid_vision;
    std::optional<std::int64_t> image_token_id;
    std::optional<std::int64_t> video_token_id;

    std::int64_t linear_k_size() const {
        return linear_num_key_heads * linear_key_head_dim;
    }
    std::int64_t linear_v_size() const {
        return linear_num_value_heads * linear_value_head_dim;
    }

    static Config from_json(
        std::string_view payload,
        const mfq::ModelGraph& graph);
    static Config from_source(const mfq::ModelSource& source);
};

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

} // namespace mfq::cuda::qwen35
