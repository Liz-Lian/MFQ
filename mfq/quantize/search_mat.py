"""Per-tensor NINT search over uniform template seeds.

Given weights and a target bpw, search a free ``k in [2,8]`` for every entry in the uniform template catalog
(the ``(bits, gs)`` pairs in :data:`mfq.formats.nint.UNIFORM_TEMPLATE_CATALOG`) and select the
:class:`~mfq.formats.nint.NintSpec` with the highest SNR. This is the core of MFQ per-tensor mixed precision
(development documentation v2 sections 1.7 and 2.2).

The catalog bounds the search space; it does not enumerate runtime formats.
Per-neuron q/k choices are encoded as row metadata and consumed by one kernel.
``gs`` remains a tensor-level choice because heterogeneous group boundaries
cannot be made row-transparent in the same packed stream.

The search quantizes and dequantizes the entire matrix to evaluate SNR without subset sampling, ensuring rigorous reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mfq.formats.nint import UNIFORM_TEMPLATE_CATALOG, NintSpec
from mfq.quantize import nint_quant
from mfq.utils.tensor import snr

_K_MIN, _K_MAX = 2, 8


@dataclass(frozen=True)
class SearchResult:
    """Search result containing the best spec, its SNR and bpw, and all candidates.

    ``evaluated`` is a list of ``(spec, snr_db, bpw)`` tuples in descending SNR order.
    SNR compares the original values against quantization/dequantization of the **entire weight tensor**, not a subsample.
    """

    spec: NintSpec
    snr_db: float
    bpw: float
    evaluated: list[tuple[NintSpec, float, float]]


def _eval_spec(W: np.ndarray, spec: NintSpec, axis: int) -> float:
    return snr(W, nint_quant.dequantize(nint_quant.quantize(W, spec, axis=axis)))


def search(
    weight: np.ndarray,
    target_bpw: float,
    axis: int = 0,
    profiles: tuple[tuple[int, int], ...] = UNIFORM_TEMPLATE_CATALOG,
) -> SearchResult:
    """Search a free k for every profile and return the complete :class:`SearchResult` with the best SNR under budget.

    Full evaluation: compute SNR by quantizing and dequantizing all of ``weight`` without row subsampling.
    """

    W = np.asarray(weight, dtype=np.float32)
    if W.ndim < 2:
        raise ValueError(f"search 需要 ndim>=2 的张量，得到 shape {W.shape}")
    Wt = np.moveaxis(W, axis, 0)
    out = Wt.shape[0]
    neuron_len = Wt.size // out

    evaluated: list[tuple[NintSpec, float, float]] = []
    for bits, gs in profiles:
        for k in range(_K_MIN, _K_MAX + 1):
            s = NintSpec(bits=bits, groupsize=gs, sub_bits=k)
            b = s.bpw(neuron_len)
            if b > target_bpw + 1e-9:
                continue
            evaluated.append((s, _eval_spec(W, s, axis), b))

    if not evaluated:
        raise ValueError(
            f"目标 bpw {target_bpw} 在 profiles={profiles} 下无合法 (bits,gs,k)"
        )
    evaluated.sort(key=lambda t: t[1], reverse=True)
    best_spec_, best_snr, best_bpw = evaluated[0]
    return SearchResult(spec=best_spec_, snr_db=best_snr, bpw=best_bpw, evaluated=evaluated)


def best_spec(
    weight: np.ndarray,
    target_bpw: float,
    axis: int = 0,
    profiles: tuple[tuple[int, int], ...] = UNIFORM_TEMPLATE_CATALOG,
) -> NintSpec:
    """Return the NintSpec with the highest SNR among candidates at or below ``target_bpw``."""

    return search(weight, target_bpw, axis, profiles).spec
