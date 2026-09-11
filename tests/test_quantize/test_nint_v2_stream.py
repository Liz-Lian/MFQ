from __future__ import annotations

import numpy as np
import torch

from mfq.formats.io import unpack_nint
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import (
    dequantize,
    quantize,
)
from mfq.quantize.nint import (
    allocate_row_profiles,
    candidate_profiles,
    measure_row_profile_losses,
    profile_variable_bits,
)
from mfq.tools.quantize_hf_to_mfq import _write_nint_axis0_blob


def test_stream_writer_packs_mixed_neuron_metadata_across_chunk_boundaries(tmp_path):
    rng = np.random.default_rng(20260911)
    weight = rng.normal(0, 0.05, size=(6, 73)).astype(np.float32)
    neuron_importance = np.asarray([1.0, 2.0, 3.0, 10.0, 50.0, 100.0], dtype=np.float32)
    spec = NintSpec(4, 24, 6)
    profiles = candidate_profiles(spec)
    losses = measure_row_profile_losses(weight, spec, profiles)
    losses *= neuron_importance[:, None]
    allocation = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=96,
        groups_per_row=4,
        target_variable_bits=(
            weight.shape[0]
            * profile_variable_bits(
                spec.bits,
                spec.sub_bits,
                values_per_row=96,
                groups_per_row=4,
            )
        ),
    )
    output = tmp_path / "mixed-nint.blob"

    _write_nint_axis0_blob(
        torch.from_numpy(weight),
        weight.shape,
        spec,
        output,
        row_chunk=2,
        quant_backend="cpu",
        device="cpu",
        neuron_importance_rows=lambda start, end: neuron_importance[start:end],
    )

    restored = unpack_nint(output.read_bytes())
    expected = quantize(
        weight,
        spec,
        row_sub_bits=allocation.row_sub_bits,
        row_q_bits=allocation.row_q_bits,
    )
    np.testing.assert_array_equal(restored.row_sub_bits, allocation.row_sub_bits)
    np.testing.assert_array_equal(restored.row_q_bits, allocation.row_q_bits)
    assert np.unique(restored.row_q_bits).size > 1
    assert np.unique(restored.row_sub_bits).size > 1
    np.testing.assert_allclose(dequantize(restored), dequantize(expected), rtol=0, atol=0)


def test_data_free_stream_writer_matches_uniform_encoded_bit_budget(tmp_path):
    rng = np.random.default_rng(20260912)
    weight = rng.normal(0, 0.05, size=(11, 73)).astype(np.float32)
    spec = NintSpec(4, 24, 6)
    uniform_path = tmp_path / "uniform-nint.blob"
    mixed_path = tmp_path / "data-free-nint-v2.blob"

    uniform_size = _write_nint_axis0_blob(
        torch.from_numpy(weight),
        weight.shape,
        spec,
        uniform_path,
        row_chunk=11,
        quant_backend="cpu",
        device="cpu",
    )
    mixed_size = _write_nint_axis0_blob(
        torch.from_numpy(weight),
        weight.shape,
        spec,
        mixed_path,
        row_chunk=11,
        quant_backend="cpu",
        device="cpu",
        nint_data_free=True,
    )

    restored = unpack_nint(mixed_path.read_bytes())
    profiles = candidate_profiles(spec)
    losses = measure_row_profile_losses(weight, spec, profiles)
    allocation = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=96,
        groups_per_row=4,
    )
    assert restored.format_version == 2
    assert allocation.actual_variable_bits <= allocation.target_variable_bits
    # The q/k selectors are format metadata, not part of the encoded-weight
    # bitrate constraint.  They may make the physical blob a few bytes larger.
    assert mixed_size <= uniform_size + 32
