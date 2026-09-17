#include "qwen35_causal_lm.h"
#include "qwen35_linear_attention.h"

#include "../../runtime/cuda_transformer.h"
#include "../../runtime/cuda_transformer_loader.h"
#include "../cuda_model_config.h"

namespace mfq::cuda::qwen35 {

std::unique_ptr<::Block> load_block(
        const mfq::ModelSource& source,
        const ::CudaRuntimeParameters& runtime,
        const Config& config,
        int layer,
        const std::string& type) {
    if (runtime.runtime_plan.backbone !=
            mfq::cuda::CudaBackbone::generic_qwen) {
        return nullptr;
    }
    const auto& c = runtime;
    const int i = layer;
    const std::string lp =
        c.tensor_root + ".block." + std::to_string(i) + ".";
    if (type == "full_attention") {
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->attention_output_gate = config.attention_output_gate;
        b->attn_norm = load_dense_gpu(
            source, lp + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(source, lp + "mlp.norm.weight");
        const std::string ap = lp + "attention.";
        const std::string query_name = ap + "query.weight";
        const std::string key_name = ap + "key.weight";
        const std::string value_name = ap + "value.weight";
        const bool mirror_kv =
            config.attention_output_gate &&
            tensor_parallel_mirror_qwen35_attention_kv_enabled() &&
            is_quant_dtype(require_tensor(source, key_name).dtype) &&
            is_quant_dtype(require_tensor(source, value_name).dtype);
        if (mirror_kv) {
            b->split_q_kv_projections = true;
            b->q_projection = load_quant_linear(source, query_name);
            const auto mirrored = std::optional<TensorParallelAxis>(
                TensorParallelAxis::Mirrored);
            b->k_projection = load_quant_linear(
                source, key_name, mirrored);
            b->v_projection = load_quant_linear(
                source, value_name, mirrored);
        } else {
            b->qkv = load_quant_group(
                source, {query_name, key_name, value_name}, 2);
        }
        b->o = load_quant_linear(source, ap + "output.weight");
        if (has_tensor(source, ap + "query_norm.weight")) {
            b->q_norm = load_dense_gpu(
                source, ap + "query_norm.weight");
        }
        if (has_tensor(source, ap + "key_norm.weight")) {
            b->k_norm = load_dense_gpu(
                source, ap + "key_norm.weight");
        }
        b->ffn = load_ffn(source, runtime, layer);
        return b;
    }
    if (type == "linear_attention") {
        auto b = std::make_unique<LinearAttentionBlock>();
        b->qwen_config = config;
        b->attn_norm = load_dense_gpu(source, lp + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(source, lp + "mlp.norm.weight");
        const std::string sp = lp + "linear_attention.";
        const std::string alpha_name = sp + "alpha.weight";
        const std::string beta_name = sp + "beta.weight";
        if (has_tensor(source, sp + "qk.weight") &&
                has_tensor(source, sp + "value.weight")) {
            b->split_in_proj = true;
            b->qkv_proj = load_quant_group(
                source, {sp + "qk.weight", sp + "value.weight"});
            if (is_quant_dtype(require_tensor(source, sp + "gate.weight").dtype)) {
                b->z_proj = load_quant_linear(source, sp + "gate.weight");
                const bool a_nint = is_quant_dtype(require_tensor(source, alpha_name).dtype);
                const bool b_nint = is_quant_dtype(require_tensor(source, beta_name).dtype);
                if (a_nint != b_nint) throw std::runtime_error("linear_attn a/b must use the same storage kind");
                b->ab_is_nint = a_nint;
                if (b->ab_is_nint) {
                    const auto scalar_axis =
                        tensor_parallel_mirror_linear_attention_scalars_enabled()
                            ? std::optional<TensorParallelAxis>(
                                TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    b->ab_nint_proj = make_quant_group({
                        load_quant_linear(source, alpha_name, scalar_axis),
                        load_quant_linear(source, beta_name, scalar_axis),
                    });
                } else {
                    b->ab_proj = make_dense_group({
                        load_dense_gpu(source, alpha_name),
                        load_dense_gpu(source, beta_name),
                    });
                }
            } else {
                b->split_dense_zab = true;
                b->zab_proj = make_dense_group({
                    load_dense_gpu(source, sp + "gate.weight"),
                    load_dense_gpu(source, alpha_name),
                    load_dense_gpu(source, beta_name),
                });
            }
        } else {
            const bool alpha_quant =
                is_quant_dtype(require_tensor(source, alpha_name).dtype);
            const bool beta_quant =
                is_quant_dtype(require_tensor(source, beta_name).dtype);
            if (alpha_quant != beta_quant) {
                throw std::runtime_error(
                    "linear_attention alpha/beta must use the same storage kind");
            }
            if (alpha_quant) {
                b->in_proj = load_quant_group(source, {
                    sp + "qkv.weight", sp + "gate.weight",
                    alpha_name, beta_name});
            } else {
                b->dense_ab_tail = true;
                b->in_proj = load_quant_group(
                    source, {sp + "qkv.weight", sp + "gate.weight"});
                b->ab_proj = make_dense_group({
                    load_dense_gpu(source, alpha_name),
                    load_dense_gpu(source, beta_name),
                });
            }
        }
        b->conv_weight = load_dense_gpu(source, sp + "conv.weight");
        if (has_tensor(source, sp + "conv.bias")) {
            b->conv_bias = load_dense_gpu(source, sp + "conv.bias");
        }
        b->dt_bias = load_dense_gpu(source, sp + "dt_bias");
        b->a_log = load_dense_gpu(source, sp + "a");
        b->linear_norm = load_dense_gpu(source, sp + "norm.weight");
        const std::string out_name = sp + "output.weight";
        if (is_quant_dtype(require_tensor(source, out_name).dtype)) {
            b->out_proj = load_quant_linear(source, out_name);
        } else {
            b->dense_out_proj = true;
            b->out_proj_dense = load_dense_gpu(source, out_name);
            if (require_tensor(source, out_name).dtype == "F16") {
                b->out_proj_dense = b->out_proj_dense
                    .to(mfq_tensor_backend::kFloat16).contiguous();
            }
            if (b->out_proj_dense.dim() != 2) {
                throw std::runtime_error(
                    "dense linear_attention output projection must be 2D");
            }
        }
        b->ffn = load_ffn(source, runtime, layer);
        return b;
    }
    return load_transformer_block(source, runtime, layer, type);
}

void LinearAttentionBlock::clear_speculative() noexcept {
    speculative_recurrent = {};
    speculative_config = nullptr;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_tokens = 0;
    speculative_pending = false;
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward_context(
        mfq_tensor_backend::Tensor input,
        const Block::Context& context,
        const CudaRuntimeParameters& config,
        const RopeCache& rope) {
    if (context.confirmed_prefix == 0) {
        return Block::forward_context(
            std::move(input), context, config, rope);
    }
    MFQ_RUNTIME_CHECK(
        input.is_cuda() && !speculative_pending &&
            context.confirmed_prefix > 0 &&
            context.confirmed_prefix < input.size(1) &&
            conv_state.defined() && gdn_state.defined(),
        "invalid Qwen3.5 speculative linear-attention transaction");

    if (!speculative_conv.defined() ||
            speculative_conv.sizes() != conv_state.sizes() ||
            !speculative_gdn.defined() ||
            speculative_gdn.sizes() != gdn_state.sizes()) {
        speculative_conv = conv_state.clone();
        speculative_gdn = gdn_state.clone();
    } else {
        speculative_conv.copy_(conv_state);
        speculative_gdn.copy_(gdn_state);
    }
    speculative_config = &config;
    speculative_start = context.cache_position;
    speculative_confirmed = context.confirmed_prefix;
    speculative_tokens = input.size(1);
    speculative_pending = true;

    try {
        // Match Metal: evaluate the complete [confirmed, drafts...] window
        // once so projections and FFN stay batched. Rollback replays only the
        // recurrent state prefix from this layer's retained projections.
        auto attention = forward_attention_cuda(
            std::move(input), config, &speculative_recurrent);
        auto result = forward_ffn_cuda(
            std::move(attention[0]), std::move(attention[1]));
        ++speculative_projection_batches;
        ++speculative_ffn_batches;
        return result;
    } catch (...) {
        conv_state.copy_(speculative_conv);
        gdn_state.copy_(speculative_gdn);
        clear_speculative();
        throw;
    }
}

void LinearAttentionBlock::commit_speculative() noexcept {
    clear_speculative();
}

void LinearAttentionBlock::rollback_speculative(int64_t keep_position) {
    MFQ_RUNTIME_CHECK(
        speculative_pending && speculative_config != nullptr &&
            keep_position >= speculative_start + speculative_confirmed &&
            keep_position < speculative_start + speculative_tokens,
        "invalid Qwen3.5 speculative rollback position");
    const int64_t retained = keep_position - speculative_start;
    conv_state.copy_(speculative_conv);
    gdn_state.copy_(speculative_gdn);
    try {
        // Restore only convolution/GDN state from the retained projected
        // rows; target projections, output projection and FFN are not rerun.
        replay_recurrent_cuda(
            speculative_recurrent, retained, *speculative_config);
        clear_speculative();
    } catch (...) {
        clear_speculative();
        throw;
    }
}

} // namespace mfq::cuda::qwen35
