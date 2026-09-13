from __future__ import annotations

import numpy as np
import pytest

from mfq.formats import io
from mfq.formats.mfe import MfePool, MfeTensor, merge_nint_pools
from mfq.formats.mx import MxTensor
from mfq.formats.nepq import NEPQ0_L, NEPQ0_S, NEPQ1_L, NEPQ1_S
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import quantize
from tests.mixed_family_fixtures import FLAT_FAMILIES, make_flat_family
from tests.test_formats.test_nepq import _tensor as make_nepq


@pytest.mark.parametrize("family", FLAT_FAMILIES)
def test_mfe_roundtrips_every_flat_compact_family(family: str):
    tensor = make_flat_family(family)
    container = MfeTensor(
        (2, 3, 96),
        (MfePool(np.arange(2, dtype=np.int32), tensor),),
    )
    blob = io.pack_mfe(container)
    restored = io.unpack_mfe(blob)
    assert blob[:4] == b"MFE1"
    assert restored.expert_profiles == (family, family)
    assert io.pack_mfe(restored) == blob


@pytest.mark.parametrize(
    "spec",
    (NEPQ0_S, NEPQ0_L, NEPQ1_S, NEPQ1_L),
)
def test_mfe_roundtrips_every_cross_expert_family(spec):
    tensor = make_nepq(spec)
    container = MfeTensor(
        tensor.shape,
        (MfePool(np.arange(2, dtype=np.int32), tensor),),
    )
    blob = io.pack_mfe(container)
    restored = io.unpack_mfe(blob)
    assert restored.expert_profiles == (spec.label, spec.label)
    assert io.pack_mfe(restored) == blob


@pytest.mark.parametrize(
    "spec",
    (
        NintSpec(4, 24, 6),
        NintSpec(5, 28, 6),
        NintSpec(6, 24, 6),
        NintSpec(8, 24, 8),
    ),
)
def test_mfe_roundtrips_every_nint_family(spec):
    rng = np.random.default_rng(100 + spec.bits)
    values = rng.normal(0, 0.04, (6, 96)).astype(np.float32)
    tensor = quantize(values, spec, axis=0)
    container = MfeTensor(
        (2, 3, 96),
        (MfePool(np.arange(2, dtype=np.int32), tensor),),
    )
    restored = io.unpack_mfe(io.pack_mfe(container))
    assert restored.expert_profiles == ("NINT", "NINT")


def test_nint_pool_merge_partitions_disjoint_k_selector_windows():
    rng = np.random.default_rng(111)
    pools = tuple(
        MfePool(
            np.asarray([expert], dtype=np.int32),
            quantize(
                rng.normal(0, 0.04, (2, 48)).astype(np.float32),
                NintSpec(4, 24, sub_bits),
                axis=0,
            ),
        )
        for expert, sub_bits in enumerate((1, 4, 7))
    )

    merged = merge_nint_pools(pools, out_per_expert=2, neuron_len=48)

    assert len(merged) == 2
    np.testing.assert_array_equal(merged[0].expert_ids, [0, 1])
    np.testing.assert_array_equal(merged[0].tensor.row_q_bits, [4, 4, 4, 4])
    np.testing.assert_array_equal(merged[0].tensor.row_sub_bits, [1, 1, 4, 4])
    assert merged[0].tensor.spec == NintSpec(4, 24, 2)
    np.testing.assert_array_equal(merged[1].expert_ids, [2])


def test_legacy_nim2_pool_name_is_canonicalized_at_the_boundary():
    rng = np.random.default_rng(173)
    values = rng.normal(0, 0.04, (6, 48)).astype(np.float32)
    tensor = quantize(values, NintSpec(4, 24, 6), axis=0)
    container = MfeTensor(
        (2, 3, 48),
        (MfePool(np.arange(2, dtype=np.int32), tensor),),
    )
    legacy = bytearray(io.pack_mfe(container))
    legacy[:4] = b"NIM2"
    legacy[io._MFE_HDR.size + 4 : io._MFE_HDR.size + 8] = (5).to_bytes(
        4, "little"
    )
    dtype_offset = io._MFE_HDR.size + io._MFE_POOL_HDR.size + 2 * 4
    assert legacy[dtype_offset : dtype_offset + 4] == b"NINT"
    legacy[dtype_offset : dtype_offset + 4] = b"NINT4"

    shape, pools = io.view_mfe_blob(legacy)
    restored = io.unpack_mfe(legacy)

    assert shape == container.shape
    assert tuple(pool.dtype for pool in pools) == ("NINT",)
    assert restored.expert_profiles == ("NINT", "NINT")


def test_mfe_roundtrips_native_mxfp8_and_dense_expert_cohorts():
    values = np.tile(np.arange(128, dtype=np.uint8) % 127, (2, 1))
    scales = np.full((1, 1), 127, dtype=np.uint8)
    mxfp8 = MxTensor("MXFP8", (2, 128), values, scales)
    bf16 = np.full((2, 128), 0x3F80, dtype="<u2").view(io.BFloat16Array)
    f16 = np.full((2, 128), np.float16(0.5), dtype=np.float16)
    container = MfeTensor(
        (3, 2, 128),
        (
            MfePool(np.asarray([1], dtype=np.int32), bf16),
            MfePool(np.asarray([2], dtype=np.int32), f16),
            MfePool(np.asarray([0], dtype=np.int32), mxfp8),
        ),
    )
    blob = io.pack_mfe(container)
    restored = io.unpack_mfe(blob)
    assert restored.expert_profiles == ("MXFP8", "BF16", "F16")
    assert io.pack_mfe(restored) == blob
