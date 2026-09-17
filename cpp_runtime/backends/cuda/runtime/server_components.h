#pragma once

#include "cuda_model.h"
#include "grid_vision_runtime.h"
#include "mtp.h"
#include "../models/deepseek_v41/deepseek_v41_dspark.h"
#include "../models/flash_next/flash_next_mtp.h"
#include "../models/minicpmo45/minicpmo45_runtime.h"
#include "../models/qwen35/mtp.h"

#include <memory>
#include <optional>

struct CudaRuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<MiniCPMO45Runtime> minicpmo;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::unique_ptr<CudaMtpModule> mtp;
    bool vision_available = false;
    bool mtp_available = false;

    CudaModel& language(CudaModel& fallback) {
        return minicpmo ? minicpmo->language : fallback;
    }

    mfq::cuda::CudaComponentState state() const noexcept {
        return mfq::cuda::cuda_component_state(
            graph, plan, vision_available, mtp_available);
    }
};

CudaRuntimeComponents load_cuda_runtime_components(
    CudaModel& model,
    bool load_optional_components);
