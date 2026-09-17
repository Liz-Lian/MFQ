#pragma once

#include "cuda_model_plan.h"
#include "mfq/model_graph.h"

#include <cstdint>
#include <string_view>
#include <vector>

struct CudaRuntimeParameters;
namespace mfq { class ModelSource; }

namespace mfq::cuda::deepseek_v4 {

struct Config {
    std::vector<std::int64_t> compress_ratios;
    std::int64_t hash_layer_count = 0;
    double compress_rope_base = 0.0;
    std::int64_t rope_original_positions = 0;
    double rope_factor = 1.0;
    double rope_beta_fast = 32.0;
    double rope_beta_slow = 1.0;

    static Config from_json(
        std::string_view payload,
        std::int64_t num_hidden_layers);
    static Config from_source(
        const mfq::ModelSource& source,
        std::int64_t num_hidden_layers);
};

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

} // namespace mfq::cuda::deepseek_v4
