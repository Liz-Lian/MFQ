from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NINT = (ROOT / "mfq/kernels/cuda/nint_matmul.cu").read_text()
BINDINGS = (ROOT / "mfq/kernels/cuda/mfq_cuda.cpp").read_text()
EXTENSION = (ROOT / "mfq/kernels/cuda/_ext.py").read_text()
CMAKE = (ROOT / "cpp_runtime/cmake/CudaRuntime.cmake").read_text()
MOE = (ROOT / "mfq/kernels/cuda/moe.cu").read_text()
MOE_PYTHON = (ROOT / "mfq/kernels/cuda/moe.py").read_text()
RUNTIME = (
    ROOT / "cpp_runtime/backends/cuda/apps/mfq_decode.cpp"
).read_text()
METAL_NINT = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_nint.cpp"
).read_text()
METAL_NINT_PYTHON = (
    ROOT / "mfq/kernels/metal/nint.py"
).read_text()
METAL_MOE_PYTHON = (
    ROOT / "mfq/kernels/metal/moe.py"
).read_text()
METAL_GROUPED_PYTHON = (
    ROOT / "mfq/kernels/metal/grouped_linear.py"
).read_text()
MLX_LINEAR_PYTHON = (
    ROOT / "mfq/runtime/mlx_linear.py"
).read_text()
METAL_GROUPED = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_grouped_linear.cpp"
).read_text()
METAL_MOE = (
    ROOT / "cpp_runtime/backends/metal/ops/mlx_moe.cpp"
).read_text()
METAL_MFE_KERNELS = (
    ROOT / "cpp_runtime/backends/metal/kernels/mfq_mfe_prefill.metal"
).read_text()


def test_nint_has_one_metadata_driven_small_m_compute_kernel():
    kernels = re.findall(
        r"__global__\s+void(?:\s+__launch_bounds__\([^)]*\))?\s+"
        r"([A-Za-z0-9_]+)",
        NINT,
    )
    assert kernels.count("nint_matmul_kernel") == 1
    assert kernels.count("nint_quantize_activation_kernel") == 1
    assert kernels.count("nint_decode_rows_kernel") == 1
    assert not any(
        token in name
        for name in kernels
        for token in ("nint2", "nint3", "nint4", "nint5", "nint6", "nint7")
    )


def test_mfe_nint_reuses_the_one_runtime_metadata_compute_kernel():
    kernels = re.findall(
        r"__global__\s+void(?:\s+__launch_bounds__\([^)]*\))?\s+"
        r"([A-Za-z0-9_]+)",
        MOE,
    )
    assert "mfe_nint_matmul_kernel" not in kernels
    assert "launch_nint_matmul_routed_cuda" in MOE
    assert "launch_nint_matmul_routed_cuda" in NINT
    assert "const int32_t * __restrict__ route_ids" in NINT
    assert "gs >= 4 && gs <= 64" in MOE
    assert "gs == 16 ||" not in MOE
    assert "nint_moe_mmvq_mixed_q_kernel" not in MOE
    assert "MFQ_MIXED_Q_MOE_TOKEN_SWITCH" not in MOE
    assert "nint_moe_grouped_matmul_hetero_qx_cuda" not in MOE_PYTHON
    assert "nint_moe_grouped_matmul_pool_ws_cuda" not in MOE_PYTHON
    for name in (
        "nint_moe_mmvq_kernel",
        "nint_moe_mmvq_ksplit_kernel",
        "nint_moe_grouped_tile_kernel",
        "nint_moe_hetero_mmvq_kernel",
        "nint_moe_hetero_grouped_tile_kernel",
        "nint_moe_group32_mmq_kernel",
        "nint_moe_hetero_mma_kernel",
        "nint_moe_grouped_matmul_hetero_qx_cuda",
        "nint_moe_grouped_matmul_hetero_f16_cuda",
        "nint_moe_grouped_matmul_pool_ws_cuda",
        "nint_moe_quantize_24_28_ws_cuda",
        "nint_moe_quantize_multi_ws_cuda",
    ):
        assert name not in MOE
        assert f'm.def("{name}"' not in BINDINGS


def test_retired_nint_kernel_family_is_not_compiled_or_bound():
    retired = (
        "nint_small_m",
        "nint_gemv_packed_bits",
        "nint_gemv_packed_batch",
        "nint_gemv_packed_int6",
        "nint_mmq_packed",
        "nint5_gs28_q5",
        "nint_dequant_full_packed_compact",
    )
    for token in retired:
        assert token not in NINT
        assert token not in BINDINGS
    assert "nint_small_m.cu" not in EXTENSION
    assert "nint_small_m.cu" not in CMAKE


def test_public_cuda_nint_surface_is_canonical():
    bound = set(re.findall(r'm\.def\("(nint[^\"]+)', BINDINGS))
    ordinary = {
        name
        for name in bound
        if not name.startswith("nint8_") and not name.startswith("nint_moe_")
    }
    assert ordinary == {
        "nint_cublas_gemm_nt_f16acc_cuda",
        "nint_decode_cuda",
        "nint_embedding_cuda",
        "nint_matmul_ws_cuda",
    }


def test_cpp_runtime_keeps_only_canonical_nint_row_state():
    nint_cpu = RUNTIME[RUNTIME.index("struct NintCpu {") : RUNTIME.index("struct Nint8ZeroCpu {")]
    nint_weight = RUNTIME[
        RUNTIME.index("struct NintWeight {") : RUNTIME.index("static NintWeight to_device_nint")
    ]
    assert "qbytes" not in nint_cpu
    assert "mixed_q" not in nint_cpu
    assert "mixed_q" not in nint_weight
    assert "repack_nint_cpu_rows" in RUNTIME
    assert "copy_nint_packed_bits" in RUNTIME
    assert "mfe_nint_matmul_ws_cuda(" in RUNTIME
    for retired in (
        "MoeHeteroWorkspace",
        "pure_nint_candidate_",
        "hetero_host_map_",
        "nint_moe_grouped_matmul_hetero",
        "initialize_mfe_dispatch",
    ):
        assert retired not in RUNTIME
    assert '.rfind("NINT", 0)' not in RUNTIME


def test_cpp_runtime_accepts_canonical_and_legacy_mfe_delta_magics():
    assert 'std::memcmp(blob.data(), "MFD1", 4) == 0' in RUNTIME
    assert 'std::memcmp(blob.data(), "NID2", 4) == 0' in RUNTIME


def test_metal_nint_uses_one_metadata_driven_compute_kernel():
    assert METAL_NINT.count('"mfq_cpp_nint_matmul"') == 1
    assert "row_q_layout[output]" in METAL_NINT
    assert "row_q_byte_offsets[output]" in METAL_NINT
    assert "auto output = routed_matmul(" in METAL_NINT
    for retired in (
        "mfq_cpp_single_row_grouped_nint",
        "partitioned_nint4_qkv_kernel",
        "mfq_cpp_interleaved_grouped_nint4",
        "has_single_row_nint_fast_path",
    ):
        assert retired not in METAL_GROUPED


def test_python_metal_mfe_reuses_the_ordinary_nint_kernel():
    assert METAL_NINT_PYTHON.count('"mfq_nint_matmul"') == 1
    assert "def nint_routed_matmul(" in METAL_NINT_PYTHON
    assert "nint_routed_matmul(" in METAL_MOE_PYTHON
    assert "routed_to_nint" in METAL_MOE_PYTHON
    assert "mfq_nint4_grouped_qmv" not in METAL_MOE_PYTHON
    assert "_GROUPED_NINT4_QMV" not in METAL_MOE_PYTHON


def test_python_ordinary_grouped_linear_keeps_nint_out_of_heterogeneous_kernel():
    assert "NINT projections must reuse the unified NINT matmul kernel" in (
        METAL_GROUPED_PYTHON
    )
    assert "and not contains_nint" in MLX_LINEAR_PYTHON


def test_metal_mfe_has_no_second_dense_nint_compute_kernel():
    assert "mfq_dense_nint" not in METAL_MFE_KERNELS
    assert "descriptors[base + kNintV2] = 1;" in METAL_MOE
    assert "cohort.weight.routed_matmul(" in METAL_MOE
    assert "family != kFamilyNint" in METAL_MOE
    assert "mfq_moe_nint_" not in METAL_MOE
    assert "if (family == 0u)" not in METAL_MOE
    assert "grouped_nint4_group24" not in METAL_MOE
    assert "nint_profile_mask" not in METAL_MOE
    assert METAL_NINT.count("nint_matmul_kernel()(") == 2
