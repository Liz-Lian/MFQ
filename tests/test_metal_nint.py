"""Apple-silicon tests for the unified metadata-driven NINT path."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
try:
    mx.device_info()
except RuntimeError:
    pytest.skip("Metal device unavailable", allow_module_level=True)

from mfq.formats import io  # noqa: E402
from mfq.formats.header import FileHeader  # noqa: E402
from mfq.formats.nint import NINT2_SPEC, NintSpec, NintTensor  # noqa: E402
from mfq.kernels.metal import nint as metal_nint  # noqa: E402
from mfq.kernels.metal.nint import (  # noqa: E402
    MetalNintWeight,
    nint_backward_input,
    nint_dequantize,
    nint_dequantize_matmul,
    nint_embedding,
    nint_gemm,
    nint_gemv,
    nint_matmul,
    nint_mmq,
    nint_routed_matmul,
    nint_swiglu,
)
from mfq.quantize import nint_quant  # noqa: E402
from mfq.runtime.mlx_linear import MlxNintLinear, MlxNintModel  # noqa: E402


def _array(value: mx.array) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


def _random(seed: int, shape: tuple[int, ...], scale: float = 0.1) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=shape).astype(
        np.float32
    )


def _mixed_tensor(seed: int = 1) -> NintTensor:
    tensor = nint_quant.quantize(
        _random(seed, (12, 77)),
        NintSpec(4, 24, 6),
    )
    tensor.row_q_bits = np.asarray(
        [1, 2, 3, 4, 5, 6, 7, 8, 2, 4, 6, 8], dtype=np.uint8
    )
    for row, bits in enumerate(tensor.row_q_bits):
        tensor.q[row] &= (1 << int(bits)) - 1
    tensor.row_sub_bits = np.asarray(
        [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], dtype=np.uint8
    )
    for row, bits in enumerate(tensor.row_sub_bits):
        tensor.sub_scale[row] &= (1 << int(bits)) - 1
        tensor.sub_min[row] &= (1 << int(bits)) - 1
    return tensor


def _legacy_blob(tensor: NintTensor) -> bytes:
    spec = tensor.spec
    out, groups, _ = tensor.q.shape
    parts = [
        struct.pack(
            "<BBiii",
            spec.bits,
            spec.sub_bits,
            spec.groupsize,
            tensor.axis,
            tensor.neuron_len,
        ),
        struct.pack("<I", len(tensor.shape)),
        struct.pack(f"<{len(tensor.shape)}q", *tensor.shape),
        struct.pack("<II", out, groups),
        np.asarray(tensor.neuron_scale, dtype="<f2").tobytes(),
        np.asarray(tensor.neuron_min, dtype="<f2").tobytes(),
        io.pack_bits(tensor.sub_scale, spec.sub_bits),
        io.pack_bits(tensor.sub_min, spec.sub_bits),
        io.pack_bits(tensor.q, spec.bits),
    ]
    return b"".join(parts)


def test_python_metal_has_one_nint_matmul_kernel():
    kernels = {
        name
        for name in vars(metal_nint)
        if name.startswith("_NINT_") and name.endswith("_KERNEL")
    }
    assert kernels == {"_NINT_MATMUL_KERNEL", "_NINT_ROW_DECODE_KERNEL"}


@pytest.mark.parametrize(
    "spec,width",
    [
        (NINT2_SPEC, 70),
        (NintSpec(3, 24, 5), 77),
        (NintSpec(4, 24, 6), 79),
        (NintSpec(5, 28, 7), 83),
        (NintSpec(6, 26, 7), 81),
        (NintSpec(8, 48, 7), 97),
    ],
)
@pytest.mark.parametrize("rows", [1, 3, 9])
def test_uniform_presets_share_one_matmul(
    spec: NintSpec,
    width: int,
    rows: int,
):
    tensor = nint_quant.quantize(_random(100 + spec.bits, (23, width)), spec)
    source = _random(200 + rows, (rows, width))
    actual = _array(
        nint_matmul(
            MetalNintWeight.from_tensor(tensor),
            source,
            dequantize_threshold=None,
        )
    )
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("from_blob", [False, True])
def test_mixed_qk_uses_same_matmul(from_blob: bool):
    tensor = _mixed_tensor(3)
    weight = (
        MetalNintWeight.from_blob(io.pack_nint(tensor))
        if from_blob
        else MetalNintWeight.from_tensor(tensor)
    )
    source = _random(4, (6, tensor.neuron_len))
    actual = _array(nint_matmul(weight, source, dequantize_threshold=None))
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(_array(weight.row_q_bits), tensor.row_q_bits)


def test_tiny_float32_input_uses_same_address_space_agnostic_matmul():
    tensor = nint_quant.quantize(
        _random(12, (8, 8)),
        NintSpec(4, 8, 4),
    )
    source = _random(13, (1, 8))
    actual = _array(
        nint_matmul(
            MetalNintWeight.from_tensor(tensor),
            source,
            dequantize_threshold=None,
        )
    )
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("routed_input", [False, True])
def test_mixed_qk_routed_reuses_same_matmul(routed_input: bool):
    tensor = _mixed_tensor(13)
    weight = MetalNintWeight.from_tensor(tensor)
    out_per_expert = 6
    expert_map = np.asarray([-1, 0, -1, -1, 1, 9], dtype=np.int32)
    ids = np.asarray([[1, 4, 5], [4, 1, -1]], dtype=np.int32)
    shared = _random(14, (2, tensor.neuron_len))
    source = (
        np.stack((shared, shared * 0.5, shared * -0.25), axis=1)
        if routed_input
        else shared
    )
    actual = _array(
        nint_routed_matmul(
            weight,
            source,
            ids,
            expert_map,
            out_per_expert,
        )
    )
    dense = nint_quant.dequantize(tensor).reshape(
        2,
        out_per_expert,
        tensor.neuron_len,
    )
    expected = np.zeros_like(actual)
    for token in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            local = int(expert_map[expert]) if 0 <= expert < expert_map.size else -1
            if 0 <= local < dense.shape[0]:
                row = source[token, route] if routed_input else source[token]
                expected[token, route] = row @ dense[local].T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("from_blob", [False, True])
def test_mixed_q_runtime_rows_are_byte_aligned_and_contiguous(from_blob: bool):
    tensor = _mixed_tensor(4)
    weight = (
        MetalNintWeight.from_blob(io.pack_nint(tensor))
        if from_blob
        else MetalNintWeight.from_tensor(tensor)
    )
    row_bytes = (
        tensor.q.shape[1]
        * tensor.q.shape[2]
        * tensor.row_q_bits.astype(np.uint64)
        + 7
    ) // 8
    expected_offsets = np.zeros(tensor.q.shape[0], dtype=np.uint64)
    np.cumsum(row_bytes[:-1], out=expected_offsets[1:])
    np.testing.assert_array_equal(
        _array(weight.row_q_byte_offsets),
        expected_offsets.astype(np.uint32),
    )
    np.testing.assert_array_equal(
        _array(weight.row_q_layout) >> 4,
        np.zeros(tensor.q.shape[0], dtype=np.uint8),
    )


def test_legacy_uniform_blob_is_normalized_to_row_metadata():
    tensor = nint_quant.quantize(_random(5, (17, 73)), NintSpec(4, 24, 6))
    weight = MetalNintWeight.from_blob(_legacy_blob(tensor))
    source = _random(6, (4, 73))
    actual = _array(nint_matmul(weight, source, dequantize_threshold=None))
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(
        _array(weight.row_q_bits), np.full(17, 4, dtype=np.uint8)
    )


def test_explicit_entry_points_reuse_common_matmul():
    tensor = nint_quant.quantize(_random(7, (19, 75)), NintSpec(5, 24, 6))
    weight = MetalNintWeight.from_tensor(tensor)
    for rows, operation in ((1, nint_gemv), (4, nint_mmq), (17, nint_gemm)):
        source = _random(10 + rows, (rows, 75))
        actual = _array(operation(weight, source))
        expected = source @ nint_quant.dequantize(tensor).T
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_matmul_preserves_prefix_shape_and_fp16():
    tensor = nint_quant.quantize(_random(30, (13, 41)), NintSpec(6, 24, 7))
    source = _random(31, (2, 3, 41)).astype(np.float16)
    actual = _array(nint_matmul(MetalNintWeight.from_tensor(tensor), source))
    expected = source.astype(np.float32) @ nint_quant.dequantize(tensor).T
    assert actual.shape == (2, 3, 13)
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("bits", [2, 4, 5, 8])
def test_backward_and_custom_vjp(bits: int):
    tensor = nint_quant.quantize(
        _random(40 + bits, (17, 48)), NintSpec(bits, 24, 6)
    )
    weight = MetalNintWeight.from_tensor(tensor)
    source = mx.array(_random(50 + bits, (4, 48)))
    cotangent = mx.array(_random(60 + bits, (4, 17)))
    direct = nint_backward_input(weight, cotangent)
    _, vjp = mx.vjp(lambda value: nint_matmul(weight, value), [source], [cotangent])
    mx.eval(direct, vjp[0])
    np.testing.assert_allclose(_array(vjp[0]), _array(direct), rtol=0, atol=0)


def test_dequantize_embedding_and_dense_gemm():
    tensor = _mixed_tensor(70)
    weight = MetalNintWeight.from_tensor(tensor)
    dense = _array(nint_dequantize(weight, dtype=mx.float32))
    expected_dense = nint_quant.dequantize(tensor)
    np.testing.assert_allclose(dense, expected_dense, rtol=0, atol=1e-6)

    ids = np.asarray([0, 5, 11], dtype=np.int32)
    selected = _array(nint_embedding(weight, ids, dtype=mx.float32))
    np.testing.assert_allclose(selected, expected_dense[ids], rtol=0, atol=1e-6)

    source = _random(71, (65, tensor.neuron_len)).astype(np.float16)
    actual = _array(nint_dequantize_matmul(weight, source))
    reference = source.astype(np.float32) @ expected_dense.T
    np.testing.assert_allclose(actual, reference, rtol=3e-3, atol=3e-3)


def test_swiglu_reuses_two_common_projections():
    gate = _mixed_tensor(80)
    up = _mixed_tensor(81)
    source = _random(82, (4, gate.neuron_len))
    actual = _array(
        nint_swiglu(
            MetalNintWeight.from_tensor(gate),
            MetalNintWeight.from_tensor(up),
            source,
        )
    )
    gate_value = source @ nint_quant.dequantize(gate).T
    up_value = source @ nint_quant.dequantize(up).T
    expected = gate_value / (1.0 + np.exp(-gate_value)) * up_value
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=4e-5)


def test_mlx_linear_keeps_weight_packed():
    tensor = _mixed_tensor(90)
    linear = MlxNintLinear(tensor)
    assert isinstance(linear.packed_weight, MetalNintWeight)
    assert linear.packed_nbytes < tensor.q.size * np.dtype(np.float32).itemsize
    source = _random(91, (3, tensor.neuron_len))
    expected = source @ nint_quant.dequantize(tensor).T
    np.testing.assert_allclose(_array(linear(source)), expected, rtol=2e-5, atol=2e-5)


def test_mlx_model_roundtrip(tmp_path: Path):
    tensor = _mixed_tensor(100)
    path = tmp_path / "metal-unified-nint.mfq"
    io.save(
        path,
        FileHeader(model_arch="metal-test", num_tensors=1),
        {"model.embed_tokens.weight": tensor},
    )
    with MlxNintModel.from_mfq(path) as model:
        source = _random(101, (3, tensor.neuron_len))
        actual = _array(model.linear("model.embed_tokens.weight")(source))
        expected = source @ nint_quant.dequantize(tensor).T
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        ids = np.asarray([0, 5, 11], dtype=np.int32)
        np.testing.assert_allclose(
            _array(model.embedding("model.embed_tokens.weight")(ids)),
            nint_quant.dequantize(tensor)[ids].astype(np.float16),
            rtol=0,
            atol=1e-4,
        )
