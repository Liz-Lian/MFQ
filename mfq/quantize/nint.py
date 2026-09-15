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
    allocation_groups: np.ndarray | None = None,
) -> NintAllocation:
    """Choose one measured ``(q, k)`` candidate per row under one bit budget.

    The supplied NAQ loss table uses output-neuron importance as its allocation
    weight; ordinary input-channel importance has already served its separate
    role while fitting each candidate reconstruction. Logical allocation
    groups retain independent shares of the uniform preset budget, preventing
    physically fused projections from transferring bits between semantics.
    The corresponding uniform NINT preset is retained per group as a
    non-regression fallback.
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

    if allocation_groups is not None:
        raw_groups = np.asarray(allocation_groups)
        if raw_groups.shape != (rows,):
            raise ValueError(
                f"NINT allocation groups must have shape {(rows,)}, got "
                f"{raw_groups.shape}"
            )
        if not np.issubdtype(raw_groups.dtype, np.integer):
            if (
                not np.isfinite(raw_groups).all()
                or np.any(raw_groups != np.rint(raw_groups))
            ):
                raise ValueError("NINT allocation groups must contain integers")
        group_ids = np.asarray(raw_groups, dtype=np.int64)
        if np.any(group_ids < 0):
            raise ValueError("NINT allocation groups must be non-negative")
        unique_groups = np.unique(group_ids)
        if unique_groups.size > 1:
            if target != uniform_bits:
                raise ValueError(
                    "grouped NINT allocation requires the matching uniform-profile "
                    "budget so every logical group retains its own nominal share"
                )
            row_q_bits = np.empty(rows, dtype=np.uint8)
            row_sub_bits = np.empty(rows, dtype=np.uint8)
            actual = 0
            selected_loss = 0.0
            uniform_loss = 0.0
            solvers: list[str] = []
            for group_id in unique_groups:
                row_ids = np.flatnonzero(group_ids == group_id)
                group_budget = int(row_ids.size) * int(costs[uniform_index])
                result = allocate_row_profiles(
                    losses[row_ids],
                    profiles,
                    spec,
                    values_per_row=values_per_row,
                    groups_per_row=groups_per_row,
                    target_variable_bits=group_budget,
                )
                row_q_bits[row_ids] = result.row_q_bits
                row_sub_bits[row_ids] = result.row_sub_bits
                actual += result.actual_variable_bits
                selected_loss += result.selected_loss
                uniform_loss += result.uniform_loss
                solvers.append(result.solver)
            return NintAllocation(
                row_q_bits=np.ascontiguousarray(row_q_bits),
                row_sub_bits=np.ascontiguousarray(row_sub_bits),
                target_variable_bits=target,
                actual_variable_bits=actual,
                selected_loss=selected_loss,
                uniform_loss=uniform_loss,
                solver=(
                    f"logical-groups:{unique_groups.size}["
                    + ",".join(solvers)
                    + "]"
                ),
            )

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
    neuron_importance: torch.Tensor | np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Fit every q+k candidate and measure its row-allocation objective.

    ``importance`` is the ordinary input-channel imatrix used by the weight
    fitter.  ``neuron_importance`` is the independent output-row factor used
    only by precision allocation.
    """

    target = torch.device(device)
    value = torch.as_tensor(weight, dtype=torch.float32)
    if value.ndim != 2 or not value.shape[0] or not value.shape[1]:
        raise ValueError("NINTv2 profile measurement requires a non-empty matrix")
    rows, columns = map(int, value.shape)
    if neuron_importance is None:
        neuron_rows = None
    else:
        neuron_rows = np.asarray(
            neuron_importance.detach().cpu().numpy()
            if isinstance(neuron_importance, torch.Tensor)
            else neuron_importance,
            dtype=np.float64,
        ).reshape(-1)
        if neuron_rows.shape != (rows,):
            raise ValueError("NINTv2 neuron importance must have shape [output]")
        if not np.isfinite(neuron_rows).all() or np.any(neuron_rows < 0):
            raise ValueError(
                "NINTv2 neuron importance must be finite and non-negative"
            )

    if target.type in {"cuda", "mps"}:
        from mfq.quantize.nint_quant_torch import quantize_axis0

        accelerated = value.to(device=target, dtype=torch.float32)
        device_losses = []
        for q_bits, sub_bits in profiles:
            row_loss = quantize_axis0(
                accelerated,
                NintSpec(q_bits, spec.groupsize, sub_bits),
                device=target,
                importance=importance,
                return_row_sse=True,
                row_sse_only=True,
                row_sse_weighted_by_importance=(neuron_rows is None),
            )
            device_losses.append(row_loss)
        losses = (
            torch.stack(device_losses, dim=1)
            .detach()
            .cpu()
            .to(torch.float64)
            .numpy()
        )
        if neuron_rows is not None:
            losses *= neuron_rows[:, None]
        return losses

    from mfq.quantize.nint_quant import dequantize, quantize

    losses = np.empty((rows, len(profiles)), dtype=np.float64)
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
        if importance_rows is not None and neuron_rows is None:
            error *= importance_rows
        losses[:, index] = error.sum(axis=1)
    if neuron_rows is not None:
        losses *= neuron_rows[:, None]
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
