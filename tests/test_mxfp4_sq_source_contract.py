from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metal_mxfp4_sq_uses_one_profile_independent_compute_kernel() -> None:
    source = (
        ROOT / "cpp_runtime/backends/metal/ops/mlx_mxfp4_sq.cpp"
    ).read_text()
    assert source.count('"mfq_cpp_mxfp4_sq_matmul"') == 1
    assert '"mfq_cpp_mxfp4_sq2_' not in source
    assert '"mfq_cpp_mxfp4_sq3_' not in source
    assert "uint bits = uint(blob[2]) - 48u;" in source
    assert 'arguments.emplace_back("ROUTED", 1)' in source
    assert '"mfq_cpp_mxfp4_sq_moe' not in source


def test_metal_runtime_builds_only_the_unified_mxfp4_sq_implementation() -> None:
    cmake = (
        ROOT / "cpp_runtime/backends/metal/CMakeLists.txt"
    ).read_text()
    runtime_sources = cmake.split("add_library(mfq-metal-runtime STATIC", 1)[1]
    runtime_sources = runtime_sources.split(")", 1)[0]
    assert "ops/mlx_mxfp4_sq.cpp" in runtime_sources
    assert "ops/mlx_mxfp4_sq2.cpp" not in runtime_sources
    assert "ops/mlx_mxfp4_sq3.cpp" not in runtime_sources


def test_cuda_dense_and_routed_sq_share_one_compute_kernel_definition() -> None:
    source = (ROOT / "mfq/kernels/cuda/mxfp4_sq.cu").read_text()
    assert source.count("__global__ void sq_mmq(") == 1
    assert "sq_mmq<1, __half, true>" in source
    assert "template<int BITS" not in source
    assert "__global__ void sq2_" not in source
    assert "__global__ void sq3_" not in source
    assert "__global__ void mxfp4_sq_moe" not in source


def test_cpp_runtimes_route_mfe_sq_through_the_shared_linear_kernel() -> None:
    metal = (
        ROOT / "cpp_runtime/backends/metal/ops/mlx_moe.cpp"
    ).read_text()
    cuda = (
        ROOT / "cpp_runtime/backends/cuda/apps/mfq_decode.cpp"
    ).read_text()
    assert "cohort.weight.routed_matmul(" in metal
    assert "impl->grouped_mmq = false;" in metal
    assert "mxfp4_sq_moe_matmul_cuda(" in cuda
    assert 'pool.dtype == "MXFP4-SQ"' in cuda
