#include "cuda_transformer_loader.h"

#include "cuda_transformer.h"

std::unique_ptr<Block> load_transformer_block(
        const mfq::ModelSource& mfq,
        const CudaRuntimeParameters& c,
        int i,
        const std::string& type) {
    const std::string lp =
        c.tensor_root + ".block." + std::to_string(i) + ".";
    if (type == "full_attention") {
        auto b = std::make_unique<FullBlock>();
        b->layer = i;
        b->attn_norm = load_dense_gpu(mfq, lp + "attention.norm.weight");
        b->ffn_norm = load_dense_gpu(mfq, lp + "mlp.norm.weight");
        const std::string ap = lp + "attention.";
        b->qkv = load_quant_group(mfq, {
            ap + "query.weight", ap + "key.weight", ap + "value.weight"},
            2, nullptr, c.is_minicpmo45());
        b->o = load_quant_linear(mfq, ap + "output.weight");
        if (has_tensor(mfq, ap + "query_norm.weight")) {
            b->q_norm = load_dense_gpu(mfq, ap + "query_norm.weight");
        }
        if (has_tensor(mfq, ap + "key_norm.weight")) {
            b->k_norm = load_dense_gpu(mfq, ap + "key_norm.weight");
        }
        b->ffn = load_ffn(mfq, c, i);
        return b;
    }

    throw std::runtime_error("unsupported layer type: " + type);
}

FFN load_ffn(const mfq::ModelSource & mfq, const CudaRuntimeParameters & c, int i) {
    FFN f;
    const std::string p =
        c.tensor_root + ".block." + std::to_string(i) + ".mlp.";
    const std::string expert_gate_up = p + "experts.gate_up.weight";
    const std::string expert_gate = p + "experts.gate.weight";
    const std::string expert_up = p + "experts.up.weight";
    const std::string expert_down = p + "experts.down.weight";
    const bool has_expert_gate_up = has_tensor(mfq, expert_gate_up);
    const bool has_expert_gate = has_tensor(mfq, expert_gate);
    const bool has_expert_up = has_tensor(mfq, expert_up);
    const bool has_expert_down = has_tensor(mfq, expert_down);
    if (has_expert_gate_up || has_expert_gate ||
            has_expert_up || has_expert_down) {
        if (has_expert_gate != has_expert_up) {
            throw std::runtime_error(
                "MoE split Gate/Up records are incomplete: " + p);
        }
        if (has_expert_gate_up == has_expert_gate || !has_expert_down) {
            throw std::runtime_error(
                "MoE layer requires exactly one fused or split Gate/Up representation: " + p);
        }
        if (c.num_experts <= 0 || c.num_experts_per_tok <= 0 ||
            c.moe_intermediate_size <= 0 || c.shared_expert_intermediate_size <= 0) {
            throw std::runtime_error("MoE config fields are missing");
        }
        f.is_moe = true;
        f.moe_split_gate_up = has_expert_gate;
        if (f.moe_split_gate_up) {
            f.moe_gate = load_mfe_gpu(
                mfq, expert_gate, true, i, "gate");
            f.moe_up = load_mfe_gpu(
                mfq, expert_up, true, i, "up");
        } else {
            f.moe_gate_up = load_mfe_gpu(
                mfq, expert_gate_up, true, i, "gate_up");
        }
        f.moe_down = load_mfe_gpu(
            mfq, expert_down, true, i, "down");
        f.moe_router = load_dense_gpu(
            mfq, p + "router.weight").to(mfq_tensor_backend::kFloat32).contiguous();
        f.moe_shared_gate = load_dense_gpu(
            mfq, p + "shared_expert.router.weight")
            .to(mfq_tensor_backend::kFloat32).contiguous();
        f.moe_top_k = static_cast<int>(c.num_experts_per_tok);
        f.moe_use_sqrt_softplus = c.expert_gating_func == "sqrtsoftplus";
        if (f.moe_use_sqrt_softplus) {
            f.moe_normalize = c.norm_topk_prob;
            f.moe_delayed_softmax = false;
            f.moe_router_scale = c.routed_scaling_factor;
        }
        f.shared = std::make_unique<FFN>();
        f.shared->down = load_quant_linear(
            mfq, p + "shared_expert.down.weight");
        f.shared->gate_up = load_paired_gate_up(mfq, {
            p + "shared_expert.gate.weight",
            p + "shared_expert.up.weight"},
            f.shared->down);
        prepare_ffn_workspaces(*f.shared);
        const bool routed_gate_shapes = f.moe_split_gate_up
            ? f.moe_gate.n_experts == c.num_experts &&
                f.moe_up.n_experts == c.num_experts &&
                f.moe_gate.neuron_len == c.hidden_size &&
                f.moe_up.neuron_len == c.hidden_size &&
                f.moe_gate.out_per_expert == c.moe_intermediate_size &&
                f.moe_up.out_per_expert == c.moe_intermediate_size
            : f.moe_gate_up.n_experts == c.num_experts &&
                f.moe_gate_up.neuron_len == c.hidden_size &&
                f.moe_gate_up.out_per_expert == 2 * c.moe_intermediate_size;
        if (!routed_gate_shapes ||
            f.moe_down.n_experts != c.num_experts ||
            f.moe_down.neuron_len != c.moe_intermediate_size ||
            f.moe_down.out_per_expert != c.hidden_size ||
            f.moe_router.dim() != 2 || f.moe_router.size(0) != c.num_experts ||
            f.moe_router.size(1) != c.hidden_size ||
            f.moe_shared_gate.dim() != 2 || f.moe_shared_gate.size(0) != 1 ||
            f.moe_shared_gate.size(1) != c.hidden_size) {
            throw std::runtime_error(
                "MoE tensor shapes disagree with config at layer " +
                std::to_string(i));
        }
        return f;
    }
    const std::string down_name = p + "down.weight";
    const std::string gate_name = p + "gate.weight";
    const std::string up_name = p + "up.weight";
    f.down = load_quant_linear(mfq, down_name);
    f.gate_up = load_paired_gate_up(
        mfq, {gate_name, up_name}, f.down, 2, c.is_minicpmo45());
    load_important_neuron_branch(
        mfq, c, f, down_name, gate_name, up_name);
    prepare_ffn_workspaces(f);
    return f;
}
