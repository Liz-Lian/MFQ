#pragma once

#include "cuda_model_plan.h"
#include "mfq/model_graph.h"

#include <string>
#include <string_view>
#include <vector>

struct CudaRuntimeParameters;
namespace mfq { class ModelSource; }

namespace mfq::cuda::glm_dsa {

struct Config {
    std::vector<std::string> indexer_types;
    std::vector<std::string> mlp_layer_types;

    static Config from_json(
        std::string_view payload,
        const CudaRuntimeParameters& common);
    static Config from_source(
        const mfq::ModelSource& source,
        const CudaRuntimeParameters& common);
};

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

} // namespace mfq::cuda::glm_dsa
