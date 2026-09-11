import numpy as np
import torch

from mfq.formats.nint import NintSpec, NintTensor
from mfq.quantize.gptq import GptqConfig
from mfq.quantize.gsq import GsqConfig
from mfq.quantize.nint_quant import dequantize
from mfq.quantize.nint_solver import quantize_nint_gptq, quantize_nint_gsq
from mfq.quantize.nint import quantize_nint_data_free


def _matrix():
    torch.manual_seed(0)
    weight = torch.randn(4, 16)
    latent = torch.randn(128, 4)
    inputs = latent @ torch.randn(4, 16) + 0.05 * torch.randn(128, 16)
    return weight, inputs


def _row_bits():
    return (
        np.asarray([2, 3, 2, 4], dtype=np.uint8),
        np.asarray([5, 4, 6, 7], dtype=np.uint8),
    )


def test_gptq_seals_directly_into_mixed_qk_nintv2():
    weight, inputs = _matrix()
    row_q_bits, row_sub_bits = _row_bits()
    result = quantize_nint_gptq(
        weight,
        NintSpec(2, 8, 5),
        calibration_inputs=inputs,
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
        config=GptqConfig(block_size=8),
    )

    assert isinstance(result.encoded, NintTensor)
    np.testing.assert_array_equal(result.encoded.row_q_bits, row_q_bits)
    np.testing.assert_array_equal(result.encoded.row_sub_bits, row_sub_bits)
    assert result.loss.total < result.baseline_loss.total
    np.testing.assert_allclose(
        dequantize(result.encoded), result.reconstruction.cpu().numpy(), rtol=0, atol=0
    )
    for row, bits in enumerate(result.encoded.row_q_bits):
        assert int(result.encoded.q[row].max()) <= (1 << int(bits)) - 1


def test_gsq_learns_nint_neuron_anchors_without_changing_qk_map():
    weight, inputs = _matrix()
    row_q_bits, row_sub_bits = _row_bits()
    gptq = quantize_nint_gptq(
        weight,
        NintSpec(2, 8, 5),
        calibration_inputs=inputs,
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
        config=GptqConfig(block_size=8),
    )
    result = quantize_nint_gsq(
        weight,
        NintSpec(2, 8, 5),
        calibration_inputs=inputs,
        row_q_bits=row_q_bits,
        row_sub_bits=row_sub_bits,
        config=GsqConfig(
            steps=20,
            row_chunk_size=2,
            hard_eval_interval=4,
            use_gumbel_noise=False,
        ),
        gptq_config=GptqConfig(block_size=8),
    )

    assert isinstance(result.encoded, NintTensor)
    assert result.loss.total <= result.baseline_loss.total
    np.testing.assert_array_equal(result.encoded.row_q_bits, row_q_bits)
    np.testing.assert_array_equal(result.encoded.row_sub_bits, row_sub_bits)
    np.testing.assert_array_equal(result.encoded.sub_scale, gptq.encoded.sub_scale)
    np.testing.assert_array_equal(result.encoded.sub_min, gptq.encoded.sub_min)
    np.testing.assert_allclose(
        dequantize(result.encoded), result.reconstruction.cpu().numpy(), rtol=0, atol=0
    )


def test_data_free_nintv2_measures_allocates_and_quantizes():
    torch.manual_seed(31)
    weight = torch.randn(12, 48)
    encoded, allocation = quantize_nint_data_free(
        weight,
        NintSpec(4, 24, 6),
    )

    assert isinstance(encoded, NintTensor)
    np.testing.assert_array_equal(encoded.row_q_bits, allocation.row_q_bits)
    np.testing.assert_array_equal(encoded.row_sub_bits, allocation.row_sub_bits)
    assert allocation.actual_variable_bits <= allocation.target_variable_bits
    assert allocation.selected_loss <= allocation.uniform_loss
    assert allocation.solver.startswith("separable-lp-lower-hull+integer-rounding")
    assert encoded.format_version == 2
