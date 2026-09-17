#pragma once

#include "cuda_model_plan.h"
#include "mfq/model_graph.h"

#include <cstdint>
#include <string_view>

struct CudaRuntimeParameters;
namespace mfq { class ModelSource; }

namespace mfq::cuda::gemma4 {

struct Config {
    std::int64_t global_head_dim = 0;
    std::int64_t num_global_key_value_heads = 0;
    std::int64_t sliding_window = 0;
    double sliding_rope_base = 10'000.0;
    bool attention_key_equals_value = false;

    static Config from_json(
        std::string_view payload,
        std::int64_t head_dim,
        std::int64_t num_key_value_heads);
    static Config from_source(
        const mfq::ModelSource& source,
        std::int64_t head_dim,
        std::int64_t num_key_value_heads);
};

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

} // namespace mfq::cuda::gemma4
