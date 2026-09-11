"""Legacy MFQ format-name compatibility at the container boundary.

Canonical runtime code sees only family-level public dtypes.  Historical
tensor records exposed an implementation version or quantizer preset as the
dtype.  Those spellings remain readable here and nowhere else.
"""

from __future__ import annotations


NINT_DTYPE = "NINT"
MFE_DTYPE = "MFE"
MFE_DELTA_DTYPE = "MFED"
NVQ_DTYPE = "NVQ"
NPQ_DTYPE = "NPQ"
NEPQ_DTYPE = "NEPQ"
MXFP4_SQ_DTYPE = "MXFP4-SQ"

_LEGACY_NVQ_DTYPES = frozenset(
    {
        "NIQ2",
        "NIQ2J",
        "NIQ3",
        "NVQ1-L",
        "NVQ1-S",
        "NVQ2",
        "NVQ2J",
        "NVQ2J-L",
        "NVQ2J-XL",
        "NVQ3",
        "NVQ3J",
        "NVQ3J-512",
        "NVQ3J-L",
    }
)
_LEGACY_NPQ_DTYPES = frozenset({"NPQ0-L", "NPQ0-S"})
_LEGACY_NEPQ_DTYPES = frozenset(
    {
        "NEPQ0-L",
        "NEPQ0-S",
        "NEPQ1-L",
        "NEPQ1-S",
        "NEPQ0-A",
        "NEPQ1-A",
    }
)


def is_legacy_nint_dtype(dtype: str) -> bool:
    """Return whether *dtype* is one historical NINT record spelling."""

    return dtype == "NINTv2" or (
        len(dtype) == 5
        and dtype.startswith("NINT")
        and "1" <= dtype[4] <= "8"
    )


def canonical_dtype(dtype: str) -> str:
    """Map a stored legacy dtype to the current public runtime name."""

    if dtype == NINT_DTYPE or is_legacy_nint_dtype(dtype):
        return NINT_DTYPE
    if dtype == "NINTM":
        return MFE_DTYPE
    if dtype == "NINTMD":
        return MFE_DELTA_DTYPE
    if dtype == NVQ_DTYPE or dtype in _LEGACY_NVQ_DTYPES:
        return NVQ_DTYPE
    if dtype == NPQ_DTYPE or dtype in _LEGACY_NPQ_DTYPES:
        return NPQ_DTYPE
    if dtype == NEPQ_DTYPE or dtype in _LEGACY_NEPQ_DTYPES:
        return NEPQ_DTYPE
    if dtype == MXFP4_SQ_DTYPE or dtype in {"MXFP4-SQ2", "MXFP4-SQ3"}:
        return MXFP4_SQ_DTYPE
    return dtype


def is_nint_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes NINT."""

    return canonical_dtype(dtype) == NINT_DTYPE


def is_mfe_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes an MFE container."""

    return canonical_dtype(dtype) == MFE_DTYPE


def is_vq_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes an NVQ family."""

    return canonical_dtype(dtype) == NVQ_DTYPE


def is_npq_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes an NPQ family."""

    return canonical_dtype(dtype) == NPQ_DTYPE


def is_nepq_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes an NEPQ family."""

    return canonical_dtype(dtype) == NEPQ_DTYPE


def is_mxfp4_sq_dtype(dtype: str) -> bool:
    """Return whether a canonical or stored dtype denotes MXFP4-SQ."""

    return canonical_dtype(dtype) == MXFP4_SQ_DTYPE


__all__ = [
    "MFE_DELTA_DTYPE",
    "MFE_DTYPE",
    "MXFP4_SQ_DTYPE",
    "NEPQ_DTYPE",
    "NINT_DTYPE",
    "NPQ_DTYPE",
    "NVQ_DTYPE",
    "canonical_dtype",
    "is_legacy_nint_dtype",
    "is_mfe_dtype",
    "is_mxfp4_sq_dtype",
    "is_nepq_dtype",
    "is_nint_dtype",
    "is_npq_dtype",
    "is_vq_dtype",
]
