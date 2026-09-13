"""Rate-distortion allocation for the NINT format-v2 ``(q, k)`` space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mfq.formats.nint import NintSpec, NintTensor


@dataclass(frozen=True)
class NintAllocation:
    """One budget-safe assignment selected from measured row distortions."""

    row_q_bits: np.ndarray
    row_sub_bits: np.ndarray
    target_variable_bits: int
    actual_variable_bits: int
    selected_loss: float
    uniform_loss: float
    solver: str


def candidate_profiles(spec: NintSpec) -> tuple[tuple[int, int], ...]:
    """Return every q width and k width encodable by the NINT v2 header."""

    minimum_k = max(1, int(spec.sub_bits) - 1)
    maximum_k = min(8, int(spec.sub_bits) + 2)
    return tuple(
        (q_bits, sub_bits)
        for q_bits in range(1, 9)
        for sub_bits in range(minimum_k, maximum_k + 1)
    )


def profile_variable_bits(
    q_bits: int,
    sub_bits: int,
    *,
    values_per_row: int,
    groups_per_row: int,
) -> int:
    """Return packed q plus scale/minimum bits, excluding fixed row fields."""

    if not 1 <= int(q_bits) <= 8 or not 1 <= int(sub_bits) <= 8:
        raise ValueError("NINT q and k widths must lie in [1, 8]")
    if values_per_row <= 0 or groups_per_row <= 0:
        raise ValueError("NINT row geometry must be positive")
    return int(q_bits) * int(values_per_row) + 2 * int(sub_bits) * int(groups_per_row)


def _allocate_separable_lp_rounded(
    losses: np.ndarray,
    costs: np.ndarray,
    target: int,
) -> tuple[np.ndarray, int, float]:
    """Solve the row-separable LP and round its sole fractional row down.

    Each row contributes a small discrete rate-distortion curve.  Its lower
    convex envelope turns the LP relaxation into ordered, divisible upgrade
    segments.  Globally consuming those segments by marginal distortion
    reduction is equivalent to solving the expanded linear program, without
    materializing one variable and one sparse constraint entry per candidate.
    """

    rows, profile_count = losses.shape
    if costs.shape != (profile_count,):
        raise ValueError("NINT profile costs do not match the loss table")

    # Equal-cost candidates are interchangeable to the budget constraint; keep
    # the lowest-loss representative, with profile order as the stable tie-break.
    cost_order = np.argsort(costs, kind="stable")
    unique_points: list[tuple[int, int]] = []
    cursor = 0
    while cursor < profile_count:
        cost = int(costs[cost_order[cursor]])
        end = cursor + 1
        while end < profile_count and int(costs[cost_order[end]]) == cost:
            end += 1
        unique_points.append((cost, cursor))
        cursor = end

    hulls: list[list[int]] = []
    segments: list[tuple[float, int, int, int]] = []
    selected_positions = np.zeros(rows, dtype=np.int32)
    selected_indices = np.empty(rows, dtype=np.int32)
    minimum_bits = 0

    for row in range(rows):
        points: list[int] = []
        best_loss = np.inf
        for cost, order_start in unique_points:
            order_end = order_start + 1
            while (
                order_end < profile_count
                and int(costs[cost_order[order_end]]) == cost
            ):
                order_end += 1
            index = min(
                (int(cost_order[position]) for position in range(order_start, order_end)),
                key=lambda candidate: (float(losses[row, candidate]), candidate),
            )
            loss = float(losses[row, index])
            # A more expensive point with no lower distortion is dominated.
            if loss >= best_loss:
                continue
            points.append(index)
            best_loss = loss

        hull: list[int] = []
        for index in points:
            while len(hull) >= 2:
                first, middle = hull[-2], hull[-1]
                left = (float(losses[row, middle]) - float(losses[row, first])) * (
                    int(costs[index]) - int(costs[middle])
                )
                right = (float(losses[row, index]) - float(losses[row, middle])) * (
                    int(costs[middle]) - int(costs[first])
                )
                scale = max(1.0, abs(left), abs(right))
                if left < right - 1.0e-14 * scale:
                    break
                hull.pop()
            hull.append(index)

        if not hull:
            raise RuntimeError("NINT row has no feasible rate-distortion point")
        hulls.append(hull)
        selected_indices[row] = hull[0]
        minimum_bits += int(costs[hull[0]])
        for position in range(len(hull) - 1):
            lower = hull[position]
            upper = hull[position + 1]
            delta_bits = int(costs[upper]) - int(costs[lower])
            gain = (
                float(losses[row, lower]) - float(losses[row, upper])
            ) / float(delta_bits)
            segments.append((gain, row, position, delta_bits))

    if target < minimum_bits:
        raise ValueError(
            f"target budget {target} bits is below minimum feasible {minimum_bits} bits"
        )
    maximum_bits = sum(int(costs[hull[-1]]) for hull in hulls)
    remaining = min(int(target), maximum_bits) - minimum_bits

    # Convex-envelope slopes are monotonically decreasing within each row, so
    # this global order automatically respects every row's prefix constraint.
    segments.sort(key=lambda item: (-item[0], item[1], item[2]))
    fractional_row: int | None = None
    for _gain, row, position, delta_bits in segments:
        if int(selected_positions[row]) != position:
            continue
        if delta_bits <= remaining:
            selected_positions[row] = position + 1
            selected_indices[row] = hulls[row][position + 1]
            remaining -= delta_bits
            continue
        fractional_row = row
        break

    # Match allocate_lp_rounded(): round the fractional segment to its cheaper
    # endpoint, then spend the residual capacity on the best discrete candidate
    # available to that same row.
    if fractional_row is not None:
        current = int(selected_indices[fractional_row])
        available = int(costs[current]) + remaining
        feasible = np.flatnonzero(costs <= available)
        replacement = min(
            (int(index) for index in feasible),
            key=lambda index: (
                float(losses[fractional_row, index]),
                int(costs[index]),
                index,
            ),
        )
        selected_indices[fractional_row] = replacement

    actual = int(costs[selected_indices].sum())
    selected_loss = float(losses[np.arange(rows), selected_indices].sum())
    return selected_indices, actual, selected_loss


def allocate_row_profiles(
    row_losses: np.ndarray,
    profiles: tuple[tuple[int, int], ...],
    spec: NintSpec,
    *,
    values_per_row: int,
    groups_per_row: int,
    target_variable_bits: int | None = None,
) -> NintAllocation:
    """Choose one measured ``(q, k)`` candidate per row under one bit budget.

    The supplied loss table should already contain the complete NAQ objective:
    input-channel weighting and output-neuron importance.  The corresponding
    uniform NINT preset is always retained as a non-regression fallback.
    """

    losses = np.asarray(row_losses, dtype=np.float64)
    if losses.ndim != 2 or losses.shape[0] == 0:
        raise ValueError("NINTv2 row losses must be a non-empty matrix")
    if losses.shape[1] != len(profiles) or not profiles:
        raise ValueError("NINTv2 loss columns must match candidate profiles")
    if not np.isfinite(losses).all() or np.any(losses < 0):
        raise ValueError("NINTv2 row losses must be finite and non-negative")
    if len(set(profiles)) != len(profiles):
        raise ValueError("NINTv2 candidate profiles must be unique")

    rows = int(losses.shape[0])
    costs = np.asarray(
        [
            profile_variable_bits(
                q_bits,
                sub_bits,
                values_per_row=values_per_row,
                groups_per_row=groups_per_row,
            )
            for q_bits, sub_bits in profiles
        ],
        dtype=np.int64,
    )
    try:
        uniform_index = profiles.index((int(spec.bits), int(spec.sub_bits)))
    except ValueError as exc:
        raise ValueError("NINTv2 candidates omit the uniform base profile") from exc
    uniform_bits = rows * int(costs[uniform_index])
    target = uniform_bits if target_variable_bits is None else int(target_variable_bits)
    if target <= 0:
        raise ValueError("NINT target bit budget must be positive")

    selected_indices, actual, selected_loss = _allocate_separable_lp_rounded(
        losses,
        costs,
        target,
    )
    uniform_loss = float(losses[:, uniform_index].sum())
    meaningful_gain = (
        selected_loss < uniform_loss * (1.0 - 1.0e-12)
        if uniform_loss > 0.0
        else False
    )
    uniform_fits = uniform_bits <= target
    if actual > target:
        if not uniform_fits:
            raise RuntimeError(
                "NINT allocator exceeded the exact bit budget and the "
                "uniform profile does not fit"
            )
        row_q_bits = np.full(rows, int(spec.bits), dtype=np.uint8)
        row_sub_bits = np.full(rows, int(spec.sub_bits), dtype=np.uint8)
        actual = uniform_bits
        selected_loss = uniform_loss
        solver = "separable-lp-lower-hull+integer-rounding+uniform-fallback"
    elif not meaningful_gain and uniform_fits:
        row_q_bits = np.full(rows, int(spec.bits), dtype=np.uint8)
        row_sub_bits = np.full(rows, int(spec.sub_bits), dtype=np.uint8)
        actual = uniform_bits
        selected_loss = uniform_loss
        solver = "separable-lp-lower-hull+integer-rounding+uniform-fallback"
    else:
        row_q_bits = np.asarray(
            [int(profiles[index][0]) for index in selected_indices], dtype=np.uint8
        )
        row_sub_bits = np.asarray(
            [int(profiles[index][1]) for index in selected_indices], dtype=np.uint8
        )
        solver = "separable-lp-lower-hull+integer-rounding"

    return NintAllocation(
        row_q_bits=np.ascontiguousarray(row_q_bits),
        row_sub_bits=np.ascontiguousarray(row_sub_bits),
        target_variable_bits=target,
        actual_variable_bits=actual,
        selected_loss=selected_loss,
        uniform_loss=uniform_loss,
        solver=solver,
    )


def measure_row_profile_losses(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    profiles: tuple[tuple[int, int], ...],
    *,
    importance: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Measure the complete row objective for every legal q+k candidate."""

    target = torch.device(device)
    value = torch.as_tensor(weight, dtype=torch.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("NINTv2 profile measurement requires a non-empty matrix")
    rows, columns = map(int, value.shape)
    losses = np.empty((rows, len(profiles)), dtype=np.float64)

    if target.type in {"cuda", "mps"}:
        from mfq.quantize.nint_quant_torch import quantize_axis0

        accelerated = value.to(device=target, dtype=torch.float32)
        for index, (q_bits, sub_bits) in enumerate(profiles):
            row_loss = quantize_axis0(
                accelerated,
                NintSpec(q_bits, spec.groupsize, sub_bits),
                device=target,
                importance=importance,
                return_row_sse=True,
                row_sse_only=True,
            )
            losses[:, index] = row_loss.detach().cpu().to(torch.float64).numpy()
        return losses

    from mfq.quantize.nint_quant import dequantize, quantize

    array = np.ascontiguousarray(value.cpu().numpy(), dtype=np.float32)
    if importance is None:
        importance_rows = None
    else:
        importance_rows = np.asarray(
            importance.detach().cpu().numpy()
            if isinstance(importance, torch.Tensor)
            else importance,
            dtype=np.float32,
        )
        if importance_rows.shape == (columns,):
            importance_rows = np.broadcast_to(importance_rows, (rows, columns))
        elif importance_rows.shape != (rows, columns):
            raise ValueError(
                "NINTv2 importance must have shape [input] or [output,input]"
            )
    for index, (q_bits, sub_bits) in enumerate(profiles):
        encoded = quantize(
            array,
            NintSpec(q_bits, spec.groupsize, sub_bits),
            axis=0,
            importance=importance_rows,
        )
        error = (dequantize(encoded) - array).astype(np.float64) ** 2
        if importance_rows is not None:
            error *= importance_rows
        losses[:, index] = error.sum(axis=1)
    return losses


def quantize_nint_data_free(
    weight: torch.Tensor | np.ndarray,
    spec: NintSpec,
    *,
    device: str | torch.device = "cpu",
    target_variable_bits: int | None = None,
) -> tuple[NintTensor, NintAllocation]:
    """Quantize by measuring every row/q/k reconstruction and allocating bits.

    This is the simple calibration-free NINT baseline: the established NINT
    quantizer supplies every candidate reconstruction, the resulting row SSE
    table drives the budget allocation, and the same quantizer emits the final
    mixed-q+k tensor.
    """

    value = torch.as_tensor(weight, dtype=torch.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("data-free NINT quantization requires a non-empty matrix")
    rows, columns = map(int, value.shape)
    profiles = candidate_profiles(spec)
    losses = measure_row_profile_losses(
        value,
        spec,
        profiles,
        device=device,
    )
    groups = (columns + int(spec.groupsize) - 1) // int(spec.groupsize)
    allocation = allocate_row_profiles(
        losses,
        profiles,
        spec,
        values_per_row=groups * int(spec.groupsize),
        groups_per_row=groups,
        target_variable_bits=target_variable_bits,
    )
    target = torch.device(device)
    if target.type in {"cuda", "mps"}:
        from mfq.quantize.nint_quant_torch import quantize_axis0

        encoded = quantize_axis0(
            value.to(target),
            spec,
            device=target,
            row_q_bits=allocation.row_q_bits,
            row_sub_bits=allocation.row_sub_bits,
        )
    else:
        from mfq.quantize.nint_quant import quantize

        encoded = quantize(
            np.ascontiguousarray(value.cpu().numpy(), dtype=np.float32),
            spec,
            axis=0,
            row_q_bits=allocation.row_q_bits,
            row_sub_bits=allocation.row_sub_bits,
        )
    if allocation.row_q_bits.shape != (rows,):
        raise RuntimeError("NINT allocation returned an invalid row map")
    return encoded, allocation


__all__ = [
    "NintAllocation",
    "allocate_row_profiles",
    "candidate_profiles",
    "measure_row_profile_losses",
    "profile_variable_bits",
    "quantize_nint_data_free",
]
