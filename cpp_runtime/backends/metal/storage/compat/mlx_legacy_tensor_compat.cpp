#include "mlx_legacy_tensor_compat.h"

#include "mfq_container.h"
#include "mfq_legacy_model_graph.h"
#include "mfq_legacy_tensor_names.h"
#include "mfq_model_graph.h"

#include "nlohmann/json.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mfq::metal {
namespace {

using json = nlohmann::json;

inline constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";

std::vector<std::string> stored_tensor_names(const MfqContainer& model) {
    std::vector<std::string> result;
    result.reserve(model.records().size());
    for (const auto& [name, _record] : model.records()) {
        if (!name.starts_with("__mfq_asset__/")) result.push_back(name);
    }
    return result;
}

std::string legacy_model_config(const MfqContainer& model) {
    const std::string asset(kModelConfigAsset);
    if (!model.contains(asset)) {
        throw std::runtime_error(
            "pre-schema MFQ artifact has no embedded model_config.json");
    }
    return model.read_text(asset);
}

std::unordered_map<std::string, std::string> hf_source_map(
        const MfqContainer& model) {
    const std::string asset(mfq::kMfqHfSourceMapAsset);
    json payload;
    try {
        payload = json::parse(model.read_text(asset));
    } catch (const json::exception& error) {
        throw std::runtime_error(
            std::string("invalid native HF source map: ") + error.what());
    }
    if (!payload.is_object() ||
        payload.value("schema", std::string{}) != "mfq.hf-source-map" ||
        payload.value("version", 0) != 1) {
        throw std::runtime_error("unsupported native HF source map contract");
    }
    const auto found = payload.find("canonical_to_source");
    if (found == payload.end() || !found->is_object()) {
        throw std::runtime_error(
            "native HF source map has no canonical_to_source object");
    }
    std::unordered_map<std::string, std::string> result;
    result.reserve(found->size());
    for (const auto& [canonical, value] : found->items()) {
        if (canonical.empty() || !value.is_string() ||
            value.get_ref<const std::string&>().empty()) {
            throw std::runtime_error("native HF source map has an invalid entry");
        }
        result.emplace(canonical, value.get<std::string>());
    }
    return result;
}

} // namespace

void install_hf_source_compatibility(MfqContainer& model) {
    if (!model.contains(std::string(kModelConfigAsset))) {
        throw std::runtime_error(
            "HF source checkpoint has no model_config.json view");
    }
    if (model.contains(std::string(mfq::kMfqHfSourceMapAsset))) {
        auto source_map = hf_source_map(model);
        model.install_hf_mfe_views(source_map);
        for (auto iterator = source_map.begin(); iterator != source_map.end();) {
            if (iterator->first == iterator->second ||
                model.records().find(iterator->second) == model.records().end()) {
                iterator = source_map.erase(iterator);
            } else {
                ++iterator;
            }
        }
        model.install_source_aliases(std::move(source_map));
        return;
    }

    // Compatibility for invoking the native binary directly on an HF folder
    // created before source-map sidecars existed. Normal server launches use
    // the schema-generated map above and never dispatch an HF loader by model.
    auto compatibility = mfq::make_legacy_tensor_aliases(
        model.header().architecture,
        legacy_model_config(model),
        stored_tensor_names(model));
    model.install_hf_mfe_views(compatibility.canonical_to_stored);
    model.install_source_aliases(
        std::move(compatibility.canonical_to_stored));
}

void install_legacy_tensor_compatibility(MfqContainer& model) {
    if (model.model_graph()) return;
    if (!model.contains(std::string(kModelConfigAsset))) return;
    auto compatibility = mfq::make_legacy_tensor_aliases(
        model.header().architecture,
        legacy_model_config(model),
        stored_tensor_names(model));
    model.install_hf_mfe_views(
        compatibility.canonical_to_stored);
    model.install_legacy_aliases(
        std::move(compatibility.canonical_to_stored),
        compatibility.layout);
}

mfq::MfqModelGraph effective_model_graph(const MfqContainer& model) {
    if (auto graph = model.model_graph()) return *graph;
    return mfq::synthesize_legacy_model_graph(
        model.header().architecture,
        legacy_model_config(model),
        [&model](std::string_view name) {
            return model.contains(std::string(name));
        });
}

} // namespace mfq::metal
