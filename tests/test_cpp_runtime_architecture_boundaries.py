"""Executable ownership rules for shared native-runtime behavior."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METAL = ROOT / "cpp_runtime" / "backends" / "metal"
MODELS = METAL / "models"
MTP_HEADER = (METAL / "runtime" / "mlx_mtp.h").read_text(encoding="utf-8")
MTP_SOURCE = (METAL / "runtime" / "mlx_mtp.cpp").read_text(encoding="utf-8")
SAMPLING_HEADER = (METAL / "runtime" / "mlx_sampling.h").read_text(
    encoding="utf-8"
)
TRANSFORMER_HEADER = (METAL / "runtime" / "mlx_transformer.h").read_text(
    encoding="utf-8"
)
QWEN = (MODELS / "qwen35" / "mlx_qwen35_causal_lm.cpp").read_text(
    encoding="utf-8"
)
QWEN4 = (MODELS / "flash_next" / "mlx_qwen4_causal_lm.cpp").read_text(
    encoding="utf-8"
)
QWEN4_HEADER = (
    MODELS / "flash_next" / "mlx_qwen4_causal_lm.h"
).read_text(encoding="utf-8")
DSV = (MODELS / "deepseek_v4" / "mlx_deepseek_v4_causal_lm.cpp").read_text(
    encoding="utf-8"
)
DSV_MOE = (MODELS / "deepseek_v4" / "mlx_deepseek_v4_moe.cpp").read_text(
    encoding="utf-8"
)
DSV41 = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_causal_lm.cpp"
).read_text(encoding="utf-8")
DSV_DSPARK = (
    MODELS / "deepseek_v4" / "mlx_deepseek_v4_dspark.cpp"
).read_text(encoding="utf-8")
DSV41_DSPARK = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_dspark.cpp"
).read_text(encoding="utf-8")
DSV41_MOE = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_moe.cpp"
).read_text(encoding="utf-8")
DSV41_ENGRAM = (
    MODELS / "deepseek_v41" / "mlx_deepseek_v41_engram.cpp"
).read_text(encoding="utf-8")
MXFP8_ROW_STORE = (
    METAL / "storage" / "mlx_mxfp8_row_store.cpp"
).read_text(encoding="utf-8")
SERVER_COMPONENTS = (
    METAL / "runtime" / "mlx_server_components.cpp"
).read_text(encoding="utf-8")
CONTRIBUTING = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
CUDA_MTP_HEADER = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "include" / "mfq_cuda_mtp.h"
).read_text(encoding="utf-8")
CUDA_DECODE = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp"
).read_text(encoding="utf-8")
SPARSE_OPERATOR = (METAL / "ops" / "mlx_sparse_attention.cpp").read_text(
    encoding="utf-8"
)
SPARSE_HEADER = (METAL / "ops" / "mlx_sparse_attention.h").read_text(
    encoding="utf-8"
)
DSA_OPERATOR = (METAL / "ops" / "mlx_dsa.cpp").read_text(encoding="utf-8")
DSA_HEADER = (METAL / "ops" / "mlx_dsa.h").read_text(encoding="utf-8")
MX_OPERATOR = (METAL / "ops" / "mlx_mx.cpp").read_text(encoding="utf-8")
MX_HEADER = (METAL / "ops" / "mlx_mx.h").read_text(encoding="utf-8")
MOE_OPERATOR = (METAL / "ops" / "mlx_moe.cpp").read_text(encoding="utf-8")
DECODE_APP = (METAL / "apps" / "mfq_decode_mlx.cpp").read_text(
    encoding="utf-8"
)
PLATFORM = (METAL / "runtime" / "mlx_platform.h").read_text(encoding="utf-8")


def model_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in MODELS.rglob("*")
        if path.suffix in {".h", ".cpp"}
    )


def test_development_rules_forbid_architecture_bound_reuse() -> None:
    normalized = " ".join(CONTRIBUTING.split())
    assert "reusable code must not be architecture-bound" in normalized
    assert "mandatory extraction point" in normalized
    runtime_readme = " ".join(
        (ROOT / "cpp_runtime" / "README.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert "must never live in a model-architecture directory" in runtime_readme


def test_sparse_attention_and_deepselect_are_runtime_owned() -> None:
    assert "../models/" not in SPARSE_OPERATOR
    assert "../models/" not in DSA_OPERATOR
    assert "mlx_deepselect_topk512(" in SPARSE_HEADER
    assert "mlx_deepselect_topk512_preferred(" in SPARSE_HEADER
    assert "kDeepSelectTopkSource" in SPARSE_OPERATOR
    assert "mlx_dsa_indexer_scores(" in DSA_HEADER
    assert "mlx_cache_write_inplace(" in DSA_HEADER
    assert not (MODELS / "deepseek_v4" / "mlx_deepseek_v4_sparse.cpp").exists()
    assert not (MODELS / "deepseek_v4" / "mlx_deepseek_v4_sparse.h").exists()
    assert not (
        MODELS
        / "deepseek_v4"
        / "mlx_deepseek_v4_sparse_kernels.inc"
    ).exists()


def test_v41_never_imports_v4_runtime_implementation() -> None:
    for source in (MODELS / "deepseek_v41").glob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        assert '"mlx_deepseek_v4_' not in text
    assert "mlx_yarn_tables(" in TRANSFORMER_HEADER
    assert "mlx_rope_adjacent(" in TRANSFORMER_HEADER


def test_native_mx_activation_boundaries_are_operator_owned() -> None:
    for symbol in (
        "mlx_mxfp8_sim(",
        "mlx_mxfp4_e4m3_scale_sim(",
        "mlx_weighted_rms_rope_mxfp8_sim(",
    ):
        assert symbol in MX_HEADER
    v41_attention = (
        MODELS / "deepseek_v41" / "mlx_deepseek_v41_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "kMxfpSimHeader" not in v41_attention
    assert "deepseek_v41_mxfp8_e4m3_sim" not in model_sources()
    assert "mlx_apple_chip_is(\"Apple M3 Ultra\")" in MX_OPERATOR


def test_engram_streaming_reuses_the_generic_mxfp8_row_store() -> None:
    assert "MlxMxfp8RowStore" in DSV41_ENGRAM
    assert "WorkerPool" in MXFP8_ROW_STORE
    assert "DeepSeek" not in MXFP8_ROW_STORE
    assert "prefetched_rows" in DSV41_ENGRAM
    assert not (METAL / "storage" / "deepseek_v41_engram_store.cpp").exists()
    assert not (METAL / "storage" / "deepseek_v41_engram_store.h").exists()


def test_metal_device_capability_detection_is_runtime_owned() -> None:
    assert "mlx_apple_chip_name" in PLATFORM
    assert "sysctlbyname" in PLATFORM
    for source in (
        (METAL / "ops" / "mlx_nint.cpp"),
        (METAL / "ops" / "mlx_moe.cpp"),
        (METAL / "ops" / "mlx_sparse_attention.cpp"),
        (METAL / "ops" / "mlx_mx.cpp"),
        (MODELS / "minicpmo45" / "mlx_minicpmo45.cpp"),
    ):
        assert "sysctlbyname" not in source.read_text(encoding="utf-8")


def test_m3_ultra_small_m_mxfp4_dispatch_is_geometry_based() -> None:
    assert "const bool m3_ultra_native_geometry" in MOE_OPERATOR
    assert "apple_m3_ultra() && logical_experts == 256" in MOE_OPERATOR
    assert "ids[index] >= addressable_experts" in MOE_OPERATOR
    assert "input_width == 4096" in MOE_OPERATOR
    assert "output_width == 2048 || output_width == 4096" in MOE_OPERATOR
    assert "input_width == 2048 && output_width == 4096" in MOE_OPERATOR


def test_deepseek_v4_split_resident_moe_keeps_fused_pair_path() -> None:
    small_m = DSV_MOE[DSV_MOE.index("const bool smallm_gather_qmm =") :]
    small_m = small_m[: small_m.index("} else if (grouped_prefill)")]
    assert "!split_resident" in small_m
    assert "gate->forward_sorted(" not in small_m

    direct = DSV_MOE[DSV_MOE.index('"moe.dispatch.resident_direct"') :]
    direct = direct[: direct.index("if (detail::component_profile_active())")]
    assert "gate->swiglu_pair(" in direct


def test_qwen4_uses_the_shared_ssd_expert_cache() -> None:
    assert "std::optional<std::size_t> expert_cache_bytes" in QWEN4_HEADER
    assert "std::shared_ptr<MlxMoeSsdExpertCache>" in QWEN4
    assert "ssd_expert_cache_->prepare_routes(" in QWEN4
    assert "ssd_expert_cache_->prefetch_layer(" in QWEN4
    assert '"predictor.block." + std::to_string(index)' in QWEN4
    qwen4_server = DECODE_APP[DECODE_APP.index('backbone == "qwen4_exp"') :]
    assert "requested_cache_bytes(" in qwen4_server
    assert "container, context, expert_cache_bytes" in qwen4_server
    assert "Qwen4ExpertCache" not in model_sources()


def test_qwen4_small_m_down_reduce_is_format_neutral() -> None:
    moe = QWEN4[QWEN4.index("class Qwen4Moe") :]
    assert moe.count("const bool combine_routes = tokens <= 6;") == 2
    assert moe.count("routed_matmul_reduce(") >= 2
    assert "supports_mxfp4_blocks" not in moe


def test_qwen4_qsa_caches_completed_index_blocks_incrementally() -> None:
    assert "MlxSequenceCache pooled_index_cache_;" in QWEN4
    assert "const int cached = pooled_index_cache_.position();" in QWEN4
    assert "pooled_index_cache_.append(pool_index_keys(" in QWEN4
    assert "return pooled_index_cache_.view();" in QWEN4
    assert "trim_pooled_index_cache();" in QWEN4


def test_native_server_prewarms_shared_ssd_arenas_on_load_and_reload() -> None:
    serving = DECODE_APP[
        DECODE_APP.index("int serve_loaded_runtime(") :
        DECODE_APP.index("int run_native_server(")
    ]
    assert "runtime.prewarm_ssd_expert_arena();" in serving
    # Three capability checks and their matching calls cover initial load,
    # successful context reload, and restoration after a failed reload.
    assert serving.count(".prewarm_ssd_expert_arena();") == 6


def test_dspark_moe_is_explicitly_text_only() -> None:
    dspark = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_dspark.cpp"
    ).read_text()
    block = dspark[dspark.index("auto branches = stage.components.moe.forward_branches(") :]
    block = block[: block.index("return deepseek_v4_hc_post_sum(")]
    assert "token_ids,\n            nullptr,\n            false);" in block


def test_deepseek_v41_moe_uses_fused_down_reduce_for_every_backing() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_moe.cpp"
    ).read_text()
    forward = source[source.index("MlxDeepseekV41Moe::forward(") :]
    assert "prepared.weights().down.combine(" in forward
    assert "down.routed_matmul_reduce(" in forward
    assert "routed_down_->combine(" in forward
    assert "moe_weighted_reduce(routed_pairs" not in forward


def test_deepseek_v41_resident_split_gate_up_keeps_two_projection_weight() -> None:
    header = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_moe.h"
    ).read_text()
    source = DSV41_MOE
    assert "std::optional<MlxMoeWeight> routed_gate_up" in header
    assert "std::optional<MlxMoeWeight> gate_up;" in source
    assert "routed_gate_up_->routed_swiglu(" in source
    assert "std::optional<MlxRoutedLinear> routed_gate_up" not in header


def test_dspark_reuses_model_neutral_inverse_rope_output_projection() -> None:
    tensor_header = (
        METAL / "runtime" / "mlx_tensor.h"
    ).read_text(encoding="utf-8")
    tensor_source = (
        METAL / "runtime" / "mlx_tensor.cpp"
    ).read_text(encoding="utf-8")
    assert "MlxLinear::grouped_row_matmul_inverse_rope(" in tensor_source
    assert "grouped_row_matmul_inverse_rope(" in tensor_header
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert ".grouped_row_matmul_inverse_rope(" in source


def test_dspark_reads_circular_context_in_chronological_order() -> None:
    transformer_header = (
        METAL / "runtime" / "mlx_transformer.h"
    ).read_text(encoding="utf-8")
    transformer_source = (
        METAL / "runtime" / "mlx_transformer.cpp"
    ).read_text(encoding="utf-8")
    assert "mlx_circular_cache_history(" in transformer_header
    assert "array mlx_circular_cache_history(" in transformer_source
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "mlx_circular_cache_history(ring, position)" in source
        assert "slice_axis(ring, 1, 0, active)" not in source


def test_deepseek_attention_input_projection_grouping_is_runtime_owned() -> None:
    tensor_header = (
        METAL / "runtime" / "mlx_tensor.h"
    ).read_text(encoding="utf-8")
    tensor_source = (
        METAL / "runtime" / "mlx_tensor.cpp"
    ).read_text(encoding="utf-8")
    assert "mlx_group_linears(" in tensor_header
    assert "std::optional<MlxGroupedLinear> mlx_group_linears(" in tensor_source
    assert "class MlxProjectionBatch" in tensor_header
    assert "MlxProjectionBatch::MlxProjectionBatch(" in tensor_source

    v4_attention = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "std::optional<MlxProjectionBatch> projections;" in v4_attention
    assert "class ProjectionGroup" not in v4_attention

    v41_header = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.h"
    ).read_text(encoding="utf-8")
    v41_attention = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.cpp"
    ).read_text(encoding="utf-8")
    assert "std::optional<MlxProjectionBatch> input_projections_;" in v41_header
    assert "input_projections_.emplace(std::move(input_projections));" in v41_attention
    assert "(*input_projections_)(input)" in v41_attention

    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "MlxProjectionBatch input_projections;" in source
        assert "input_projections(std::vector<const MlxLinear*>" in source
        assert "stage.input_projections(input)" in source


def test_deepseek_v4_mfe_streaming_uses_fused_down_reduce() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_moe.cpp"
    ).read_text()
    streamed = source[source.index("} else if (!expert_offload_) {") :]
    streamed = streamed[: streamed.index("if (!shared.has_value())")]
    assert "down_weight.routed_matmul_reduce(" in streamed
    # Legacy TPQ has no matching fused primitive and retains its fallback.
    assert "return moe_weighted_reduce(" in streamed


def test_deepseek_v4_only_tracks_token_counts_for_active_penalties() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_causal_lm.cpp"
    ).read_text()
    generation = source[source.index("MlxDeepseekV4CausalLm::generate_impl(") :]
    assert "std::optional<array> counts;" in generation
    assert "if (sampling.has_penalties())" in generation
    assert "? sampler.sample(logits, *counts)" in generation
    assert "if (counts) {\n            *counts = sample_token_counts_add(" in generation


def test_deepseek_v4_small_m_reuses_ssd_route_transactions() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v4/"
        "mlx_deepseek_v4_causal_lm.cpp"
    ).read_text()
    forward = source[
        source.index("MlxDeepseekV4CausalLm::forward_streaming_layers(") :
        source.index("MlxDeepseekV4CausalLm::begin_speculative_target(")
    ]
    transaction_begin = forward.index("const bool route_transaction")
    transaction = forward[transaction_begin:]
    assert "&& routed_rows >= 1" in forward
    assert "&& routed_rows <= 6" in forward
    assert "has_full_residency_capacity" not in transaction
    assert "targets == nullptr" not in transaction
    assert "capture_dspark_target(group_begin, hidden, targets);" in transaction
    assert "auto trial_targets = targets != nullptr" in transaction
    assert "*targets = std::move(trial_targets);" in transaction


def test_dspark_small_m_reuses_the_shared_ssd_route_transaction() -> None:
    draft = DSV_DSPARK[DSV_DSPARK.index("MlxDeepseekV4DSpark::draft_impl(") :]
    assert "routed_rows <= 6" in draft
    assert "mlx_ssd_route_transactions_enabled()" in draft
    assert "begin_route_transaction();" in draft
    assert "resolve_route_transaction();" in draft
    assert "route_layer_likely_hit(" in draft


def test_deepseek_v41_multitoken_attention_orders_circular_cache_write() -> None:
    source = (
        ROOT
        / "cpp_runtime/backends/metal/models/deepseek_v41/"
        "mlx_deepseek_v41_attention.cpp"
    ).read_text()
    forward = source[source.index("MlxDeepseekV41Attention::forward(") :]
    assert "std::optional<array> pending_local_values;" in forward
    assert "std::optional<array> pending_local_rows;" in forward
    assert "mlx::core::depends(\n            std::vector<array>{state.local_kv},\n            std::vector<array>{attended})" in forward
    assert "ordered_local.front(),\n            *pending_local_values" in forward


def test_deepseek_v41_route_transactions_keep_wide_global_expert_ids() -> None:
    assert "snapshot.defer_transaction_global(routes.ids);" in DSV41_MOE
    assert "snapshot.defer_transaction(routes.ids);" not in DSV41_MOE


def test_direct_native_hf_server_uses_size_aware_expert_residency() -> None:
    assert "native_hf_source_bytes(" in DECODE_APP
    assert "*source_bytes <= automatic_limit" in DECODE_APP
    assert (
        "requested_cache_bytes(\n"
        "                arguments.expert_cache_gb, native_hf, container"
    ) in DECODE_APP


def test_mixed_mfe_offload_uses_the_shared_pager() -> None:
    for source in (QWEN4, DSV41_MOE):
        assert "can_group_mfe(" in source
        assert "grouped_mfe(" in source
    assert "MlxMfeOffloadCache" in DSV41
    assert "class MlxMfeOffloadCache" not in model_sources()


def test_mtp_policy_and_lifecycle_are_runtime_owned() -> None:
    for symbol in (
        "struct MlxMtpGenerationStats",
        "class MlxMtpDepthController",
        "struct MlxMtpEngineCallbacks",
        "run_mlx_mtp_generation",
        "verify_stochastic_mtp_top_k_chain_device",
    ):
        assert symbol in MTP_HEADER or symbol in MTP_SOURCE

    sources = model_sources()
    assert "struct MlxMtpGenerationStats" not in sources
    assert "class MlxMtpDepthController" not in sources
    assert "verify_greedy_mtp(" not in sources
    assert "verify_stochastic_mtp(" not in sources
    assert "host_sampling_distribution(" not in sources


def test_every_mtp_architecture_is_a_thin_client_of_one_engine() -> None:
    clients = (QWEN, QWEN4, DSV, DSV41)
    for source in clients:
        assert source.count("run_mlx_mtp_generation(") == 1
        assert source.count("mlx_mtp_verification_ids(") == 1
        assert source.count("mlx_mtp_committed_hidden(") == 1

    all_sources = model_sources()
    assert all_sources.count("run_mlx_mtp_generation(") == len(clients)
    assert "MlxMtpDepthPolicy" not in all_sources
    assert "AcceptanceOnly" not in all_sources
    assert "plain_decode" not in MTP_HEADER
    assert "should_exit" not in MTP_HEADER
    assert "should_exit" not in MTP_SOURCE
    assert "draft_greedy(" not in DSV
    assert "MlxMtpGenerationStats last_mtp_stats_" in (
        MODELS / "deepseek_v4" / "mlx_deepseek_v4_causal_lm.h"
    ).read_text(encoding="utf-8")
    assert "begin_speculative_target(1, draft_count + 1)" in DSV
    assert "rollback_speculative_target(" in DSV
    assert "forward_chunk(\n                    committed_ids" not in DSV


def test_dspark_adapters_keep_only_predictor_math_and_cache_state() -> None:
    for source in (DSV_DSPARK, DSV41_DSPARK):
        assert "mlx_sparse_selected_mla_attention(" in source
        assert "void MlxDeepseek" in source and "::propose(" in source
        assert "draft_impl(" in source
        assert "const int physical_width = available_width;" in source
        assert "evaluate the complete physical block" in source
    assert "stable_dspark_state_" in DSV
    assert "stable_dspark_state_" in DSV41
    assert "MlxMtpDepthPolicy" not in DSV
    assert "MlxMtpDepthPolicy" not in DSV41


def test_v41_keeps_shared_moe_hc_fusion_in_target_and_predictor() -> None:
    assert "ffn_mhc_.expand_sum(" in DSV41
    assert "stage.ffn_mhc.expand_sum(" in DSV41_DSPARK
    assert "return mlx::core::reshape(routed + shared" not in DSV41_MOE


def test_server_exposes_predictors_by_role_not_architecture_name() -> None:
    assert SERVER_COMPONENTS.count(
        'component_declared(graph, "predictor")'
    ) == 4
    assert re.search(
        r'component_with_implementation\([^;]*"predictor"',
        SERVER_COMPONENTS,
        re.DOTALL,
    ) is None


def test_cuda_mtp_generation_loop_is_architecture_independent_and_reversible() -> None:
    begin = CUDA_DECODE.index("static int32_t generate_mtp_tokens(")
    end = CUDA_DECODE.index("\nstatic int32_t generate_server_tokens(", begin)
    generation = CUDA_DECODE[begin:end]
    for architecture_name in (
        "Qwen",
        "DeepSeek",
        "GLM",
        "Flash",
        "MiniCPM",
    ):
        assert architecture_name not in generation
    assert "should_exit" not in generation
    assert "should_exit" not in CUDA_MTP_HEADER
    assert "exit_streak" not in CUDA_MTP_HEADER
    assert "DepthController depth_controller" in generation
    assert "bounded_depth(depth_controller.depth())" in generation


def test_generic_generation_and_sequence_cache_helpers_are_not_redeclared() -> None:
    assert "mlx_last_token_logits" in SAMPLING_HEADER
    assert "class MlxSequenceCache" in TRANSFORMER_HEADER
    sources = model_sources()
    forbidden_definitions = (
        r"\b(?:array|mlx::core::array)\s+last_(?:token_)?logits\s*\(",
        r"\bclass\s+SequenceCache\b",
        r"\bstd::optional<[^>]*array[^>]*>\s+\w*generation_token_counts\s*\(",
    )
    for pattern in forbidden_definitions:
        assert re.search(pattern, sources) is None, pattern
