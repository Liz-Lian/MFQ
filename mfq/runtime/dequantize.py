"""Reference NINT dequantization with one optional backend override.

Uniform presets and heterogeneous q/k tensors share one runtime contract.
The override is consequently registered for NINT itself, never for a bit-width
or group-size label.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from mfq.formats.nint import NintTensor
from mfq.quantize import nint_quant

_BackendFn = Callable[[NintTensor], np.ndarray]
_DEFAULT: _BackendFn = nint_quant.dequantize
_BACKEND: _BackendFn | None = None


def register_backend(fn: _BackendFn) -> None:
    """Register the process-wide NINT dequantization backend."""
    global _BACKEND
    _BACKEND = fn


def clear_backends() -> None:
    """Clear all registered hardware backends and return to the NumPy default."""
    global _BACKEND
    _BACKEND = None


def dequantize(tensor: NintTensor) -> np.ndarray:
    """Dequantize with the common backend or the NumPy reference."""
    return (_BACKEND or _DEFAULT)(tensor)
