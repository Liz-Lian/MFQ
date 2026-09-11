from __future__ import annotations

import numpy as np

from mfq.calibration.allocator import GroupCandidate, allocate_lp_rounded
from mfq.formats.nint import NintSpec
from mfq.quantize.nint_quant import dequantize, quantize
from mfq.quantize.nint import (
    _allocate_separable_lp_rounded,
    allocate_row_profiles,
    candidate_profiles,
    measure_row_profile_losses,
    profile_variable_bits,
)


def test_separable_allocator_matches_expanded_lp_reference() -> None:
    rng = np.random.default_rng(20260911)
    for _ in range(32):
        rows = int(rng.integers(2, 9))
        profile_count = int(rng.integers(3, 9))
        costs = np.sort(
            rng.choice(np.arange(10, 101), size=profile_count, replace=False)
        ).astype(np.int64)
        losses = rng.lognormal(0.0, 1.0, size=(rows, profile_count))
        target = int(rng.integers(rows * costs[0], rows * costs[-1] + 1))

        selected, actual, selected_loss = _allocate_separable_lp_rounded(
            losses,
            costs,
            target,
        )
        reference = allocate_lp_rounded(
            (
                GroupCandidate(
                    group=f"row:{row}",
                    profile=f"profile:{profile}",
                    specs={"index": profile},
                    storage_bits=int(costs[profile]),
                    train_loss=float(losses[row, profile]),
                    validation_loss=float(losses[row, profile]),
                )
                for row in range(rows)
                for profile in range(profile_count)
            ),
            target,
        )
        reference_selected = np.asarray(
            [reference.selected[f"row:{row}"].specs["index"] for row in range(rows)]
        )

        np.testing.assert_array_equal(selected, reference_selected)
        assert actual == reference.actual_storage_bits
        np.testing.assert_allclose(selected_loss, reference.train_loss, rtol=1e-12)


def test_joint_qk_allocation_moves_budget_to_important_neurons() -> None:
    spec = NintSpec(4, 24, 6)
    profiles = candidate_profiles(spec)
    importance = np.asarray([0.05, 0.2, 2.0, 20.0], dtype=np.float64)
    losses = np.asarray(
        [
            [
                value * (4.0 ** (-q_bits) + 0.2 * 4.0 ** (-sub_bits))
                for q_bits, sub_bits in profiles
            ]
            for value in importance
        ]
    )

    result = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=24,
        groups_per_row=1,
    )

    assert result.actual_variable_bits <= result.target_variable_bits
    assert result.selected_loss < result.uniform_loss
    assert result.row_q_bits[-1] > result.row_q_bits[0]
    assert (
        sum(
            profile_variable_bits(
                int(q_bits),
                int(sub_bits),
                values_per_row=24,
                groups_per_row=1,
            )
            for q_bits, sub_bits in zip(
                result.row_q_bits, result.row_sub_bits, strict=True
            )
        )
        == result.actual_variable_bits
    )


def test_joint_qk_allocation_keeps_uniform_profile_without_signal() -> None:
    spec = NintSpec(4, 24, 6)
    profiles = candidate_profiles(spec)
    losses = np.zeros((7, len(profiles)), dtype=np.float64)

    result = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=72,
        groups_per_row=3,
    )

    np.testing.assert_array_equal(result.row_q_bits, np.full(7, 4))
    np.testing.assert_array_equal(result.row_sub_bits, np.full(7, 6))
    assert result.actual_variable_bits == result.target_variable_bits


def test_joint_qk_allocation_respects_an_explicit_tight_payload_budget() -> None:
    spec = NintSpec(4, 24, 6)
    profiles = candidate_profiles(spec)
    rows = 7
    values_per_row = 72
    groups_per_row = 3
    losses = np.zeros((rows, len(profiles)), dtype=np.float64)
    uniform_bits = rows * profile_variable_bits(
        spec.bits,
        spec.sub_bits,
        values_per_row=values_per_row,
        groups_per_row=groups_per_row,
    )
    tight_budget = uniform_bits - rows * 5

    result = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=values_per_row,
        groups_per_row=groups_per_row,
        target_variable_bits=tight_budget,
    )

    assert result.actual_variable_bits <= tight_budget
    assert "uniform-fallback" not in result.solver


def test_candidate_profiles_follow_the_header_k_window() -> None:
    assert candidate_profiles(NintSpec(4, 24, 7))[0] == (1, 6)
    assert candidate_profiles(NintSpec(4, 24, 7))[-1] == (8, 8)
    assert candidate_profiles(NintSpec(4, 24, 1))[0] == (1, 1)
    assert candidate_profiles(NintSpec(4, 24, 1))[-1] == (8, 3)


def test_profile_measurement_matches_direct_weighted_row_error() -> None:
    rng = np.random.default_rng(418)
    weight = rng.normal(size=(5, 24)).astype(np.float32)
    importance = np.linspace(0.25, 2.0, 24, dtype=np.float32)
    spec = NintSpec(4, 24, 6)
    profiles = ((3, 5), (4, 6), (5, 7))

    losses = measure_row_profile_losses(
        weight,
        spec,
        profiles,
        importance=importance,
    )
    encoded = quantize(weight, spec, importance=importance)
    expected = (
        (dequantize(encoded) - weight).astype(np.float64) ** 2
        * importance[None, :]
    ).sum(axis=1)

    np.testing.assert_allclose(losses[:, 1], expected, rtol=1e-6, atol=1e-8)
