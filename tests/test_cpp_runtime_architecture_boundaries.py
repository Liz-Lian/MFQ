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
    assert "apple_m3_ultra() && experts == 256" in MOE_OPERATOR
    assert "input_width == 4096" in MOE_OPERATOR
    assert "output_width == 2048 || output_width == 4096" in MOE_OPERATOR
    assert "input_width == 2048 && output_width == 4096" in MOE_OPERATOR


def test_qwen4_uses_the_shared_ssd_expert_cache() -> None:
    assert "std::optional<std::size_t> expert_cache_bytes" in QWEN4_HEADER
    assert "std::shared_ptr<MlxMoeSsdExpertCache>" in QWEN4
    assert "ssd_expert_cache_->prepare_routes(" in QWEN4
    assert "ssd_expert_cache_->prefetch_layer(" in QWEN4
    assert '"predictor.block." + std::to_string(index)' in QWEN4
    qwen4_server = DECODE_APP[DECODE_APP.index('backbone == "qwen4_exp"') :]
    assert "requested_cache_bytes(" in qwen4_server
    assert "container, context, expert_cache_bytes" in qwen4_server
    assert "runtime.prewarm_ssd_expert_arena();" in qwen4_server
    assert "Qwen4ExpertCache" not in model_sources()


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
