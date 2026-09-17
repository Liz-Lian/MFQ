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
CUDA_APP = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "apps" / "mfq_decode.cpp"
).read_text(encoding="utf-8")
CUDA_MODELS = ROOT / "cpp_runtime" / "backends" / "cuda" / "models"
CUDA_RUNTIME = ROOT / "cpp_runtime" / "backends" / "cuda" / "runtime"
CUDA_RUNTIME_SOURCE = (CUDA_RUNTIME / "cuda_decode_runtime.cpp").read_text(
    encoding="utf-8"
)
CUDA_MTP_SOURCE = (CUDA_RUNTIME / "mtp.cpp").read_text(encoding="utf-8")
CUDA_DECODE = CUDA_APP + "\n" + CUDA_RUNTIME_SOURCE
CUDA_BACKEND_SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (ROOT / "cpp_runtime" / "backends" / "cuda").rglob("*")
    if path.suffix in {".h", ".cpp"}
)
CUDA_REGISTRY = (CUDA_MODELS / "registry.cpp").read_text(encoding="utf-8")
CUDA_TRANSFORMER_LOADER = (
    CUDA_RUNTIME / "cuda_transformer_loader.cpp"
).read_text(encoding="utf-8")
CUDA_TRANSFORMER_HEADER = (
    CUDA_RUNTIME / "cuda_transformer.h"
).read_text(encoding="utf-8")
CUDA_QWEN_LINEAR = (
    CUDA_MODELS / "qwen35" / "qwen35_linear_attention.h"
).read_text(encoding="utf-8") + (\
    CUDA_MODELS / "qwen35" / "qwen35_causal_lm.cpp"
).read_text(encoding="utf-8")
CUDA_RUNTIME_PARAMETERS = (CUDA_MODELS / "cuda_model_config.h").read_text(
    encoding="utf-8"
)
CUDA_RUNTIME_PARAMETERS_SOURCE = "\n".join(
    (CUDA_MODELS / name).read_text(encoding="utf-8")
    for name in ("cuda_model_config.h", "cuda_model_config.cpp")
)
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
CORE = ROOT / "cpp_runtime" / "core"
MODEL_SOURCE_HEADER = (
    CORE / "include" / "mfq" / "model_source.h"
).read_text(encoding="utf-8")
MFQ_SOURCE = (CORE / "mfq_model_source.cpp").read_text(encoding="utf-8")
HF_SOURCE = (CORE / "hf_safetensors_source.cpp").read_text(encoding="utf-8")
HF_MODEL_SOURCE = (CORE / "hf_model_source.cpp").read_text(encoding="utf-8")
MODEL_SOURCE_FACTORY = (CORE / "model_source.cpp").read_text(encoding="utf-8")
METAL_CONTAINER = (
    METAL / "storage" / "mfq_container.cpp"
).read_text(encoding="utf-8")
CUDA_MFE_STORE = (
    ROOT / "cpp_runtime" / "backends" / "cuda" / "storage" / "mfe_expert_store.cpp"
).read_text(encoding="utf-8")


def model_sources() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in MODELS.rglob("*")
        if path.suffix in {".h", ".cpp"}
    )


def test_model_sources_are_backend_neutral_and_shared() -> None:
    assert "class ModelSource" in MODEL_SOURCE_HEADER
    assert "class MfqModelSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "mfq_model_source.h"
    ).read_text(encoding="utf-8")
    assert "class HfSafetensorsSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "hf_safetensors_source.h"
    ).read_text(encoding="utf-8")
    assert "class HfModelSource final : public ModelSource" in (
        CORE / "include" / "mfq" / "hf_model_source.h"
    ).read_text(encoding="utf-8")
    for source in (
        MODEL_SOURCE_HEADER,
        MFQ_SOURCE,
        HF_SOURCE,
        HF_MODEL_SOURCE,
    ):
        assert "backends/" not in source
        assert "mlx::" not in source
        assert "#include <cuda" not in source.lower()
        assert "#include <metal" not in source.lower()
    assert '#include "mfq/mfq_model_source.h"' not in CUDA_BACKEND_SOURCE
    assert "HfModelSource" not in CUDA_BACKEND_SOURCE
    assert '#include "mfq/mfq_model_source.h"' in MODEL_SOURCE_FACTORY
    assert "HfModelSource" in MODEL_SOURCE_FACTORY
    assert '#include "mfq/mfq_model_source.h"' in METAL_CONTAINER
    assert "HfModelSource" in METAL_CONTAINER
    assert "bad MFQ magic" not in CUDA_BACKEND_SOURCE
    assert "MfqContainer::load_records" not in METAL_CONTAINER
    assert "HfSafetensorStore" not in METAL_CONTAINER
    assert "MXT1" not in METAL_CONTAINER
    assert "struct MfqFile" not in CUDA_BACKEND_SOURCE
    assert "open_model_source" in CUDA_BACKEND_SOURCE
    assert "const mfq::ModelSource" in CUDA_BACKEND_SOURCE
    assert "direct_source" not in CUDA_BACKEND_SOURCE
    assert 'record.metadata.dtype = "FP8-128SQ"' in HF_MODEL_SOURCE
    assert "record_.read_range(offset, destination)" in CUDA_MFE_STORE


def test_native_cli_uses_backend_neutral_model_and_tokenizer_options() -> None:
    sources = (
        CUDA_DECODE,
        DECODE_APP,
        (METAL / "apps" / "mfq_perplexity_mlx.cpp").read_text(encoding="utf-8"),
    )
    for source in sources:
        assert '"--model"' in source
        assert '"--tokenizer"' in source
        assert '"--mfq"' not in source
        assert '"--tokenizer-model"' not in source
        assert '"--tokenizer-gguf"' not in source


def test_cuda_cli_is_a_thin_client_of_the_runtime_library() -> None:
    assert "mfq::cuda::run_decode(argc, argv)" in CUDA_APP
    assert len(CUDA_APP.splitlines()) <= 10
    assert "struct Model" not in CUDA_APP
    assert "run_linear_check" not in CUDA_APP

    cmake = (ROOT / "cpp_runtime" / "cmake" / "CudaRuntime.cmake").read_text(
        encoding="utf-8"
    )
    assert "add_library(mfq-cuda-runtime STATIC" in cmake
    assert "runtime/cuda_decode_runtime.cpp" in cmake
    assert "target_link_libraries(mfq-decode PRIVATE mfq-cuda-runtime)" in cmake


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


def test_cuda_config_loading_lives_with_each_model() -> None:
    models = {
        "minicpmo45": "minicpmo45_model",
        "flash_next": "flash_next_model",
        "deepseek_v41": "deepseek_v41_model",
        "qwen35": "qwen35_model",
        "glm_dsa": "glm_dsa_model",
        "gemma4": "gemma4_model",
        "deepseek_v4": "deepseek_v4_model",
    }

    assert (CUDA_MODELS / "config.h").is_file()
    assert not (CUDA_MODELS / "config_json.h").exists()
    assert not list(CUDA_MODELS.rglob("config_loader.h"))
    assert not (CUDA_RUNTIME / "model_config.h").exists()
    assert "struct CudaRuntimeParameters" in CUDA_RUNTIME_PARAMETERS
    assert "struct ModelStructure" not in CUDA_RUNTIME_PARAMETERS_SOURCE
    assert "architecture_config" not in CUDA_RUNTIME_PARAMETERS
    assert "#include <any>" not in CUDA_RUNTIME_PARAMETERS
    assert "resolved_config_json" in CUDA_RUNTIME_PARAMETERS
    assert "resolved_config_json" in CUDA_REGISTRY
    assert "flash_next::" not in CUDA_RUNTIME_PARAMETERS
    assert "deepseek_v41::" not in CUDA_RUNTIME_PARAMETERS
    qwen_config = (CUDA_MODELS / "qwen35" / "qwen35_model.h").read_text(
        encoding="utf-8"
    )
    for field in (
        "attention_output_gate",
        "mtp_use_dedicated_embeddings",
        "linear_conv_kernel_dim",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "mrope_sections",
        "mrope_interleaved",
        "grid_vision",
        "image_token_id",
        "video_token_id",
    ):
        assert field in qwen_config
        assert field not in CUDA_RUNTIME_PARAMETERS
    assert '"hidden_size"' not in CUDA_REGISTRY
    assert "nlohmann::json::parse" not in CUDA_REGISTRY
    assert "Config::from_json" not in CUDA_REGISTRY
    for namespace, stem in models.items():
        model_dir = CUDA_MODELS / namespace
        header = model_dir / f"{stem}.h"
        source = model_dir / f"{stem}.cpp"
        assert header.is_file()
        assert source.is_file()
        assert f"{namespace}::load_runtime_parameters" in CUDA_REGISTRY
        header_source = header.read_text(encoding="utf-8")
        assert "CudaRuntimeParameters load_runtime_parameters(" in header_source
        assert "const mfq::ModelSource& source" in header_source
        assert "CudaRuntimeParameters load_runtime_parameters(" in source.read_text(encoding="utf-8")


def test_cuda_model_runtime_uses_compiled_causal_lm_adapters() -> None:
    cmake = (ROOT / "cpp_runtime" / "cmake" / "CudaRuntime.cmake").read_text(
        encoding="utf-8"
    )
    adapters = {
        "flash_next": "qwen4_causal_lm",
        "deepseek_v41": "deepseek_v41_causal_lm",
        "deepseek_v4": "deepseek_v4_causal_lm",
        "glm_dsa": "glm_dsa_causal_lm",
        "gemma4": "gemma4_causal_lm",
        "qwen35": "qwen35_causal_lm",
    }

    assert not list(CUDA_MODELS.rglob("construction.h"))
    assert not list(CUDA_MODELS.rglob("model_loader.h"))
    assert not list((ROOT / "cpp_runtime" / "backends" / "cuda").rglob("*.inc"))
    for namespace, stem in adapters.items():
        model_dir = CUDA_MODELS / namespace
        header = model_dir / f"{stem}.h"
        source = model_dir / f"{stem}.cpp"
        assert header.is_file()
        assert source.is_file()
        assert f'models/{namespace}/{stem}.cpp' in cmake
        assert f'#include "{stem}.h"' in source.read_text(encoding="utf-8")
        assert not stem.startswith("cuda_")

    for concrete_definition in (
        "struct Glm5NextBlock",
        "struct Qwen4Block",
        "struct Dsv4Block",
        "struct GlmDsaBlock",
        "struct QuantLinear",
        "class MoeExpertCache",
    ):
        assert concrete_definition not in CUDA_RUNTIME_SOURCE
    assert "std::make_unique<FullBlock>" in CUDA_TRANSFORMER_LOADER
    assert "std::make_unique<LinearAttentionBlock>" in CUDA_QWEN_LINEAR
    assert 'type == "linear_attention"' not in CUDA_TRANSFORMER_LOADER
    assert len(CUDA_RUNTIME_SOURCE.splitlines()) < 12_000


def test_cuda_ops_and_execution_are_real_compilation_units() -> None:
    cuda = ROOT / "cpp_runtime" / "backends" / "cuda"
    cmake = (ROOT / "cpp_runtime" / "cmake" / "CudaRuntime.cmake").read_text(
        encoding="utf-8"
    )
    required = (
        "ops/cuda_quantized_ops.cpp",
        "runtime/cuda_execution.cpp",
        "runtime/cuda_model.cpp",
        "runtime/cuda_model_loader.cpp",
        "runtime/cuda_transformer.cpp",
        "runtime/cuda_transformer_loader.cpp",
        "runtime/mtp.cpp",
        "runtime/server_components.cpp",
        "runtime/diagnostics/backend_checks.cpp",
        "runtime/diagnostics/model_checks.cpp",
    )
    for relative in required:
        assert (cuda / relative).is_file()
        assert relative in cmake
    assert "struct QuantLinear" in (
        cuda / "ops" / "cuda_quantized_ops.h"
    ).read_text(encoding="utf-8")
    assert "struct QuantLinear" not in CUDA_RUNTIME_SOURCE
    assert '#include "cuda_quantized_ops.cpp"' not in CUDA_BACKEND_SOURCE
    assert not re.search(r'#include\s+["<][^">]+\.inc[">]', CUDA_BACKEND_SOURCE)


def test_cuda_qwen_speculation_is_model_owned() -> None:
    assert "struct LinearAttentionBlock final" in CUDA_QWEN_LINEAR
    assert "LinearAttentionBlock::rollback_speculative" in CUDA_QWEN_LINEAR
    assert "replay_recurrent_cuda(" in CUDA_QWEN_LINEAR
    assert "forward_speculative(" not in CUDA_TRANSFORMER_HEADER
    assert "struct LinearBlock" not in CUDA_TRANSFORMER_HEADER
    assert "MFQ_QWEN_MTP_BATCH" not in CUDA_TRANSFORMER_HEADER
    assert "for (int accepted_drafts : {0, 1, 2})" in CUDA_RUNTIME_SOURCE


def test_cuda_mtp_generation_loop_is_architecture_independent_and_reversible() -> None:
    generation = CUDA_MTP_SOURCE
    assert "run_cuda_mtp_generation(" in generation
    assert "int32_t run_cuda_mtp_generation(" not in CUDA_RUNTIME_SOURCE
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
    assert "retains_partial_target_prefix()" in generation
    assert "CompactDistribution" in generation
    assert "mfq_tensor_backend::topk(" in generation


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
