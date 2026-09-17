#include "gemma4_model.h"

#include "../config.h"
#include "../cuda_model_config.h"
#include "mfq/model_source.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

namespace mfq::cuda::gemma4 {
namespace {

float round_to_bfloat16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

} // namespace

Config Config::from_json(
        std::string_view payload,
        std::int64_t head_dim,
        std::int64_t num_key_value_heads) {
    using namespace mfq::cuda::config_json;
    const std::string source(payload);
    Config config;
    config.global_head_dim = integer(source, "global_head_dim", head_dim);
    config.num_global_key_value_heads = integer(
        source, "num_global_key_value_heads", num_key_value_heads);
    config.sliding_window = integer(source, "sliding_window", 0);
    config.sliding_rope_base = object_number(
        source, "sliding_attention", "rope_theta", 10'000.0);
    config.attention_key_equals_value =
        boolean(source, "attention_k_eq_v", false);
    return config;
}

Config Config::from_source(
        const mfq::ModelSource& source,
        std::int64_t head_dim,
        std::int64_t num_key_value_heads) {
    return from_json(
        source.model_config_json(), head_dim, num_key_value_heads);
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
    static_cast<void>(embedded_config
        ? Config::from_source(source, c.head_dim, c.num_key_value_heads)
        : Config::from_json(payload, c.head_dim, c.num_key_value_heads));
    c.final_logit_softcapping =
        number(s, "final_logit_softcapping", 0.0);

        c.norm_weight_offset = 0.0;
        c.embed_scale = round_to_bfloat16(
            static_cast<float>(std::sqrt(static_cast<double>(c.hidden_size))));
        return c;
}

} // namespace mfq::cuda::gemma4
