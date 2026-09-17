#include "cuda_model_config.h"

#include "config.h"

#include <cmath>
#include <stdexcept>

CudaRuntimeParameters make_common_runtime_parameters(
        const std::string& source,
        const mfq::ModelGraph& model_graph,
        const mfq::cuda::CudaModelPlan& runtime_plan) {
    using namespace mfq::cuda::config_json;
    CudaRuntimeParameters config;
    config.model_graph = model_graph;
    config.runtime_plan = runtime_plan;
    if (runtime_plan.backbone == mfq::cuda::CudaBackbone::unsupported) {
        throw std::runtime_error(
            "CUDA runtime does not implement backbone architecture: " +
            model_graph.backbone);
    }
    config.model_type = string(source, "model_type");
    config.vocab_size = integer(source, "vocab_size");
    config.hidden_size = integer(source, "hidden_size");
    config.intermediate_size = integer(source, "intermediate_size", 0);
    config.num_hidden_layers = integer(source, "num_hidden_layers");
    config.num_attention_heads = integer(source, "num_attention_heads");
    config.num_key_value_heads = integer(source, "num_key_value_heads");
    config.max_position_embeddings =
        integer(source, "max_position_embeddings");
    config.head_dim = integer(
        source,
        "head_dim",
        config.hidden_size / config.num_attention_heads);
    config.rope_base = object_number(
        source,
        "full_attention",
        "rope_theta",
        number(source, "rope_theta", 1'000'000.0));
    config.full_rotary_factor = object_number(
        source,
        "full_attention",
        "partial_rotary_factor",
        number(source, "partial_rotary_factor", 1.0));
    config.rotary_dim = static_cast<std::int64_t>(std::llround(
        config.full_rotary_factor * static_cast<double>(config.head_dim)));
    config.rms_norm_eps = number(source, "rms_norm_eps", 1e-6);
    config.tie_word_embeddings =
        boolean(source, "tie_word_embeddings", false);
    config.layer_types = layer_types(source, config.num_hidden_layers);
    return config;
}
