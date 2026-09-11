import torch

from mfq.quantize.gptq import GptqConfig, quantize_gptq
from mfq.quantize.gsq import GsqConfig, quantize_gsq


def _correlated_inputs(seed: int = 0):
    torch.manual_seed(seed)
    weight = torch.randn(4, 16)
    latent = torch.randn(128, 4)
    inputs = latent @ torch.randn(4, 16) + 0.05 * torch.randn(128, 16)
    return weight, inputs


def test_activation_and_hessian_paths_are_equivalent():
    weight, inputs = _correlated_inputs()
    hessian = inputs.mT @ inputs / inputs.shape[0]
    config = GptqConfig(block_size=8)

    activation_result = quantize_gptq(
        weight,
        q_bits=3,
        group_size=8,
        calibration_inputs=inputs,
        config=config,
    )
    hessian_result = quantize_gptq(
        weight,
        q_bits=3,
        group_size=8,
        hessian=hessian,
        config=config,
    )

    torch.testing.assert_close(
        activation_result.reconstruction,
        hessian_result.reconstruction,
        rtol=0,
        atol=0,
    )
    assert abs(activation_result.loss.total - hessian_result.loss.total) < 1.0e-6


def test_gptq_propagates_hessian_error_and_beats_rtn_on_correlated_inputs():
    weight, inputs = _correlated_inputs()
    result = quantize_gptq(
        weight,
        q_bits=2,
        group_size=8,
        calibration_inputs=inputs,
        fit_iterations=0,
        config=GptqConfig(block_size=8),
    )

    assert result.metrics["accepted"] == 1.0
    assert result.loss.total < result.baseline_loss.total * 0.5
    assert result.grid.codes.dtype == torch.int16
    assert torch.all(result.grid.codes >= 0)
    assert torch.all(result.grid.codes <= 3)
    torch.testing.assert_close(result.encoded.dequantize(), result.reconstruction)


def test_gsq_improves_hard_codes_and_scales_without_layout_change():
    torch.manual_seed(2)
    weight = torch.randn(5, 12)
    inputs = torch.randn(96, 12)
    gptq = quantize_gptq(
        weight,
        q_bits=3,
        group_size=4,
        calibration_inputs=inputs,
        config=GptqConfig(block_size=4),
    )
    result = quantize_gsq(
        weight,
        q_bits=3,
        group_size=4,
        calibration_inputs=inputs,
        config=GsqConfig(
            steps=30,
            row_chunk_size=3,
            hard_eval_interval=5,
            use_gumbel_noise=False,
        ),
        gptq_config=GptqConfig(block_size=4),
    )

    assert result.metrics["accepted"] == 1.0
    assert result.loss.total < result.baseline_loss.total
    assert result.grid.codes.shape == gptq.grid.codes.shape
    assert result.grid.q_bits.tolist() == gptq.grid.q_bits.tolist()
    assert result.grid.group_size == gptq.grid.group_size
    assert not torch.equal(result.grid.scale_parameters, gptq.grid.scale_parameters)


def test_gsq_seeded_gumbel_path_is_reproducible_and_hard():
    torch.manual_seed(7)
    weight = torch.randn(3, 8)
    inputs = torch.randn(32, 8)
    config = GsqConfig(
        steps=8,
        row_chunk_size=2,
        hard_eval_interval=2,
        seed=91,
        use_gumbel_noise=True,
        use_gptq_init=False,
    )
    first = quantize_gsq(
        weight,
        q_bits=2,
        group_size=4,
        calibration_inputs=inputs,
        config=config,
    )
    second = quantize_gsq(
        weight,
        q_bits=2,
        group_size=4,
        calibration_inputs=inputs,
        config=config,
    )

    torch.testing.assert_close(first.grid.codes, second.grid.codes, rtol=0, atol=0)
    torch.testing.assert_close(
        first.grid.scale_parameters, second.grid.scale_parameters, rtol=0, atol=0
    )
    assert first.loss.total == second.loss.total
    assert torch.all(first.grid.codes >= 0)
    assert torch.all(first.grid.codes <= 3)


def test_mixed_width_grid_handles_tail_padding_and_singular_hessian():
    weight = torch.linspace(0.25, 2.5, 30).reshape(3, 10)
    result = quantize_gptq(
        weight,
        q_bits=torch.tensor([2, 3, 4]),
        group_size=4,
        calibration_inputs=torch.zeros(16, 10),
        config=GptqConfig(block_size=4),
    )

    assert result.grid.padded_count == 12
    assert result.reconstruction.shape == weight.shape
    assert torch.isfinite(result.reconstruction).all()
    for row, bits in enumerate([2, 3, 4]):
        assert int(result.grid.codes[row].min()) >= 0
        assert int(result.grid.codes[row].max()) <= (1 << bits) - 1
