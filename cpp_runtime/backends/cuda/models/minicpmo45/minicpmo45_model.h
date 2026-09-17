#pragma once

#include "cuda_model_plan.h"
#include "mfq/model_graph.h"

#include <string_view>

struct CudaRuntimeParameters;
namespace mfq { class ModelSource; }

namespace mfq::cuda::minicpmo45 {

CudaRuntimeParameters load_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

CudaRuntimeParameters load_tts_runtime_parameters(
    const mfq::ModelSource& source,
    std::string_view payload,
    bool embedded_config,
    const mfq::ModelGraph& model_graph,
    const mfq::cuda::CudaModelPlan& runtime_plan);

} // namespace mfq::cuda::minicpmo45
