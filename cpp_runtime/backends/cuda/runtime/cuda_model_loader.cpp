#include "cuda_model.h"

#include "cuda_transformer_loader.h"
#include "../models/registry.h"
#include "../models/deepseek_v4/deepseek_v4_causal_lm.h"
#include "../models/deepseek_v4/deepseek_v4_model.h"
#include "../models/deepseek_v41/deepseek_v41_causal_lm.h"
#include "../models/flash_next/qwen4_causal_lm.h"
#include "../models/gemma4/gemma4_causal_lm.h"
#include "../models/gemma4/gemma4_model.h"
#include "../models/glm_dsa/glm_dsa_causal_lm.h"
#include "../models/glm_dsa/glm_dsa_model.h"
#include "../models/qwen35/qwen35_causal_lm.h"
#include "../models/qwen35/qwen35_model.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace {

template <typename Loader>
void load_model_blocks(CudaModel& model, Loader&& load) {
    model.blocks.reserve(static_cast<std::size_t>(model.c.num_hidden_layers));
    for (int layer = 0; layer < model.c.num_hidden_layers; ++layer) {
        const int device = g_layer_placement.device_for_layer(layer);
        const bool cpu_offloaded = layer < g_dense_cpu_layer_count;
        g_loading_cpu_layer = cpu_offloaded;
        g_layer_placement.load_device = device;
        MfqCudaGuard layer_guard(device);
        const auto& type = model.c.layer_types[static_cast<std::size_t>(layer)];
        std::cerr << "loading layer " << layer << ' ' << type << ' '
                  << (cpu_offloaded ? "CPU" : "CUDA") << std::endl;
        auto block = load(layer, device, type);
        block->cuda_device = device;
        block->cpu_offloaded = cpu_offloaded;
        model.blocks.push_back(std::move(block));
    }
}

} // namespace

CudaModel load_model(const std::string & model_path, const std::string & config_path,
                 int64_t context_size_override,
                 bool load_blocks,
                 bool defer_moe_cache_finalize) {
    CudaModel m;
    m.source = mfq::open_model_source(model_path);
    const auto& source = *m.source;
    m.c = mfq::cuda::load_runtime_parameters(source, config_path);
    mfq::cuda::flash_next::validate_load_options(m.c);
    mfq::cuda::deepseek_v41_runtime::validate_load_options(m.c);
    if (g_expert_parallel.enabled() && m.c.num_experts <= 0) {
        throw std::runtime_error(
            "--expert-parallel requires a model with routed experts");
    }
    g_layer_placement.prepare(m.c.num_hidden_layers);
    g_dense_cpu_layer_count = 0;
    if (g_n_gpu_layers >= 0) {
        g_dense_cpu_layer_count = static_cast<int>(std::max<int64_t>(
            m.c.num_hidden_layers - g_n_gpu_layers, 0));
        if (g_dense_cpu_layer_count > 0) {
            if (model_parallel_enabled() || g_layer_placement.enabled()) {
                throw std::runtime_error(
                    "--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            }
            const bool supported_architecture =
                !m.c.is_dsv4() && !m.c.is_glm_dsa() && !m.c.is_gemma4() &&
                m.c.num_experts <= 0 &&
                std::all_of(
                    m.c.layer_types.begin(), m.c.layer_types.end(),
                    [](const std::string & type) {
                        return type == "full_attention" ||
                            type == "linear_attention";
                    });
            if (!supported_architecture) {
                throw std::runtime_error(
                    "--n-gpu-layers currently supports dense Qwen-style blocks only");
            }
            std::cerr << "dense_layer_placement cpu=0-"
                      << (g_dense_cpu_layer_count - 1)
                      << " gpu=" << g_dense_cpu_layer_count << '-'
                      << (m.c.num_hidden_layers - 1)
                      << " cpu_threads=" << mfq_get_num_threads()
                      << std::endl;
        }
    }
    g_layer_placement.load_device =
        g_layer_placement.primary_device();
    MfqCudaGuard model_guard(
        g_layer_placement.primary_device());
    if (m.c.runtime_plan.vision == mfq::cuda::CudaVisionAdapter::grid_vit &&
            g_dense_cpu_layer_count > 0) {
        throw std::runtime_error(
            "CUDA grid-Vision currently requires GPU-resident text layers");
    }
    mfq::cuda::deepseek_v4::validate_load_options(m.c);
    if (context_size_override > 0) {
        if (context_size_override > m.c.max_position_embeddings) {
            throw std::runtime_error("--ctx-size exceeds max_position_embeddings");
        }
        m.c.max_position_embeddings = context_size_override;
    }
    std::optional<mfq::cuda::qwen35::Config> qwen35_config;
    if (m.c.runtime_plan.backbone == mfq::cuda::CudaBackbone::generic_qwen) {
        qwen35_config = mfq::cuda::qwen35::Config::from_json(
            m.c.resolved_config_json, m.c.model_graph);
        qwen35_config->max_position_embeddings = m.c.max_position_embeddings;
    }
    if (!m.c.is_gemma4() && !m.c.is_dsv4() && !m.c.is_flash_next() &&
            !m.c.is_deepseek_v41()) {
        auto configure_mrope = [&](RopeCache& rope,
                                   mfq_tensor_backend::Device device) {
            if (qwen35_config) {
                rope.configure_mrope(
                    qwen35_config->mrope_sections,
                    qwen35_config->mrope_interleaved,
                    qwen35_config->rotary_dim,
                    device);
            }
        };
        const auto primary = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, g_layer_placement.primary_device());
        m.rope = RopeCache(m.c, primary);
        configure_mrope(m.rope, primary);
        if (g_dense_cpu_layer_count > 0) {
            const auto cpu = mfq_tensor_backend::Device(
                mfq_tensor_backend::kCPU);
            m.cpu_rope = RopeCache(m.c, cpu);
            configure_mrope(m.cpu_rope, cpu);
        }
        if (g_layer_placement.enabled()) {
            for (int device : g_layer_placement.devices) {
                MfqCudaGuard rope_guard(device);
                const auto target = mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA, device);
                auto [found, inserted] = m.device_ropes.emplace(
                    device, RopeCache(m.c, target));
                static_cast<void>(inserted);
                configure_mrope(found->second, target);
            }
        }
    }
    const std::string embed_name = "model.token_embedding.weight";
    const std::string norm_name = "model.output_norm.weight";
    const std::string output_name = "model.output.weight";
    m.embed = load_quant_linear(source, embed_name);
    if (m.c.is_qwen4()) {
        m.qwen4_final_mixer =
            mfq::cuda::flash_next::load_final_mixer(source, m.c);
    } else {
        m.output_norm = load_dense_gpu(source, norm_name);
    }
    if (m.c.is_dsv4()) {
        auto head = mfq::cuda::deepseek_v4::load_output_head(source, m.c);
        m.dsv4_hc_head_fn = std::move(head.function);
        m.dsv4_hc_head_scale = std::move(head.scale);
        m.dsv4_hc_head_base = std::move(head.base);
    }
    if (m.c.tie_word_embeddings || !has_tensor(source, output_name)) {
        m.lm_head = g_tensor_parallel.enabled()
            ? load_quant_linear(
                source, embed_name,
                TensorParallelAxis::Output)
            : m.embed;
    } else {
        m.lm_head = load_quant_linear(
            source, output_name,
            TensorParallelAxis::Output);
    }
    if (load_blocks) {
        using Backbone = mfq::cuda::CudaBackbone;
        switch (m.c.runtime_plan.backbone) {
            case Backbone::qwen4_exp: {
                auto config = mfq::flash_next::QwenConfig::from_json(
                    m.c.resolved_config_json);
                config.maximum = m.c.max_position_embeddings;
                load_model_blocks(m, [&](int layer, int, const std::string&) {
                    return mfq::cuda::flash_next::load_block(
                        source, config, layer);
                });
                break;
            }
            case Backbone::glm5_next: {
                auto config = mfq::flash_next::GlmConfig::from_json(
                    m.c.resolved_config_json);
                config.maximum = m.c.max_position_embeddings;
                load_model_blocks(m, [&](int layer, int, const std::string&) {
                    return mfq::cuda::flash_next::load_block(
                        source, config, layer);
                });
                break;
            }
            case Backbone::deepseek_v41:
                m.deepseek_v41_state =
                    mfq::cuda::deepseek_v41_runtime::load_shared_state(
                        source, m.c);
                load_model_blocks(m, [&](int layer, int, const std::string& type) {
                    MFQ_RUNTIME_CHECK(
                        type == "deepseek_v41" && m.deepseek_v41_state,
                        "invalid DeepSeek-V4.1 block loader state");
                    return mfq::cuda::deepseek_v41_runtime::load_block(
                        source, m.c, layer, m.deepseek_v41_state);
                });
                break;
            case Backbone::deepseek_v4: {
                const auto config =
                    mfq::cuda::deepseek_v4::Config::from_json(
                        m.c.resolved_config_json, m.c.num_hidden_layers);
                std::unordered_map<int, std::shared_ptr<Dsv4SharedState>> states;
                load_model_blocks(m, [&](int layer, int device, const std::string& type) {
                    auto& state = states[device];
                    if (!state) state = std::make_shared<Dsv4SharedState>();
                    return mfq::cuda::deepseek_v4::load_block(
                        source, m.c, config, layer, type, state);
                });
                break;
            }
            case Backbone::glm_dsa: {
                const auto config = mfq::cuda::glm_dsa::Config::from_json(
                    m.c.resolved_config_json, m.c);
                std::unordered_map<int, std::shared_ptr<GlmDsaSharedState>> states;
                load_model_blocks(m, [&](int layer, int device, const std::string& type) {
                    auto& state = states[device];
                    if (!state) state = std::make_shared<GlmDsaSharedState>();
                    return mfq::cuda::glm_dsa::load_block(
                        source, m.c, config, layer, type, state);
                });
                break;
            }
            case Backbone::gemma4: {
                const auto config = mfq::cuda::gemma4::Config::from_json(
                    m.c.resolved_config_json,
                    m.c.head_dim, m.c.num_key_value_heads);
                load_model_blocks(m, [&](int layer, int, const std::string& type) {
                    return mfq::cuda::gemma4::load_block(
                        source, m.c, config, layer, type);
                });
                break;
            }
            case Backbone::generic_qwen:
                MFQ_RUNTIME_CHECK(
                    qwen35_config,
                    "missing Qwen3.5 typed configuration");
                load_model_blocks(m, [&](int layer, int, const std::string& type) {
                    return mfq::cuda::qwen35::load_block(
                        source, m.c, *qwen35_config, layer, type);
                });
                break;
            case Backbone::minicpmo45:
            case Backbone::minicpmo_tts:
                load_model_blocks(m, [&](int layer, int, const std::string& type) {
                    return load_transformer_block(source, m.c, layer, type);
                });
                break;
            case Backbone::unsupported:
                throw std::runtime_error("unsupported CUDA backbone");
        }
    }
    g_loading_cpu_layer = false;
    g_layer_placement.load_device =
        g_layer_placement.primary_device();
    if (moe_expert_cache_has_sources() &&
            !moe_expert_cache_finalized() &&
            !defer_moe_cache_finalize) {
        finalize_moe_expert_cache();
    }
    return m;
}
