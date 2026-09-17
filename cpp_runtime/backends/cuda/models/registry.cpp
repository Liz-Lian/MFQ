#include "registry.h"

#include "deepseek_v4/deepseek_v4_model.h"
#include "deepseek_v41/deepseek_v41_model.h"
#include "flash_next/flash_next_model.h"
#include "gemma4/gemma4_model.h"
#include "glm_dsa/glm_dsa_model.h"
#include "minicpmo45/minicpmo45_model.h"
#include "qwen35/qwen35_model.h"

#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mfq::cuda {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open config: " + path);
    }
    return {std::istreambuf_iterator<char>(input), {}};
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
                (tensor.name.size() > prefix.size() &&
                 tensor.name.compare(0, prefix.size(), prefix) == 0 &&
                 tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

CudaRuntimeParameters make_runtime_parameters(
        const mfq::ModelSource& source,
        const std::string& payload,
        bool embedded_config,
        const mfq::ModelGraph& graph) {
    const auto plan = cuda_model_plan(graph);
    using Backbone = CudaBackbone;
    switch (plan.backbone) {
        case Backbone::minicpmo45:
            return minicpmo45::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::glm5_next:
        case Backbone::qwen4_exp:
            return flash_next::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::deepseek_v41:
            return deepseek_v41::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::generic_qwen:
            return qwen35::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::glm_dsa:
            return glm_dsa::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::gemma4:
            return gemma4::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::deepseek_v4:
            return deepseek_v4::load_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::minicpmo_tts:
            return minicpmo45::load_tts_runtime_parameters(
                source, payload, embedded_config, graph, plan);
        case Backbone::unsupported:
            throw std::runtime_error(
                "CUDA runtime does not implement backbone architecture: " +
                graph.backbone);
    }
    throw std::runtime_error("invalid CUDA config dispatch");
}

} // namespace

CudaRuntimeParameters load_runtime_parameters(
        const mfq::ModelSource& source,
        const std::string& external_path) {
    std::string payload;
    if (!external_path.empty()) {
        payload = read_file(external_path);
    }

    const auto graph = source.resolved_model_graph();
    auto parameters = make_runtime_parameters(
        source, payload, external_path.empty(), graph);
    parameters.resolved_config_json = external_path.empty()
        ? source.model_config_json() : std::move(payload);
    if (parameters.num_hidden_layers != graph.topology.text_layers) {
        throw std::runtime_error(
            "model graph/config text-layer topology mismatch");
    }
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : graph.components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
    return parameters;
}

} // namespace mfq::cuda
