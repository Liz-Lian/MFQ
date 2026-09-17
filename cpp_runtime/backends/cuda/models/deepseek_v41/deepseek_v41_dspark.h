#pragma once

#include <memory>

namespace mfq {
class ModelSource;
namespace models::deepseek_v41 { struct Config; }
}
struct CudaMtpModule;
struct CudaRuntimeParameters;

namespace mfq::cuda::deepseek_v41_runtime {

std::unique_ptr<::CudaMtpModule> load_dspark_if_present(
    const mfq::ModelSource& source,
    const ::CudaRuntimeParameters& runtime,
    const mfq::models::deepseek_v41::Config& config);
void run_dspark_self_check();

} // namespace mfq::cuda::deepseek_v41_runtime
