"""Native-output scalar quantization for E4M3 weight tensors.

``MXFP8-SQ`` and ``FP8-128SQ`` are separate public tensor formats.  They
share only the scalar symbol stream: every output neuron carries a three-bit
``q - 1`` descriptor selecting SQ1 through SQ8.  SQ1--SQ7 reconstruct one
legal E4M3 code through a matrix-local scalar palette; SQ8 stores the source
E4M3 byte unchanged.

The scale contracts deliberately remain distinct.  MXFP8-SQ stores E8M0
microscales and their native block geometry.  FP8-128SQ stores a 128x128
BF16/F16/F32 multiplier grid.  A runtime therefore never guesses scale
semantics from the scale tensor shape.

Design intent: both formats provide high-fidelity, fine-grained requantization
of native QAT weights while preserving legal E4M3 reconstruction values and
the source format's scale geometry.  MXFP8-SQ is the native-MXFP8 variant and
FP8-128SQ is the 128x128 block-FP8 variant.  Keeping those native reconstruction
domains allows CUDA backends to use the corresponding hardware-accelerated
MXFP8/FP8 execution paths rather than expanding the model into a non-native
codebook representation.  The SQ bitstream is storage, not a replacement
arithmetic format.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Sequence, TypeAlias

import numpy as np


MXFP8_SQ_DTYPE = "MXFP8-SQ"
FP8_128SQ_DTYPE = "FP8-128SQ"
FP8_SQ_DTYPES = frozenset({MXFP8_SQ_DTYPE, FP8_128SQ_DTYPE})

_MXFP8_MAGIC = b"M8SQ"
_FP8_128_MAGIC = b"F8SQ"
_VERSION = 1
_HEADER = struct.Struct("<4sBBBBHHQQQQ")
_PALETTE_NBYTES = 256
_SCALE_KIND = {
    "F8_E8M0": 1,
    "BF16": 2,
    "F16": 3,
    "F32": 4,
}
_SCALE_DTYPE = {value: key for key, value in _SCALE_KIND.items()}
_SCALE_ITEMSIZE = {
    "F8_E8M0": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
}
_MXFP8_BLOCK_SHAPES = frozenset({(1, 32), (32, 32), (128, 128)})


@dataclass(frozen=True)
class Fp8SqDescriptor:
    """Exact stored rate and q-distribution summary."""

    aggregate_bpw: float
    distribution_entropy: float
    format_version: int


class _Fp8SqProperties:
    payload: bytes
    output_size: int
    input_size: int
    scale_shape: tuple[int, int]
    block_shape: tuple[int, int]
    row_q_bits: tuple[int, ...]
    scale_dtype: str
    format_version: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_size, self.input_size

    @property
    def profile(self) -> str:
        first = self.row_q_bits[0]
        if all(value == first for value in self.row_q_bits):
            return f"SQ{first}"
        return "SQ1/SQ2/SQ3/SQ4/SQ5/SQ6/SQ7/SQ8"

    @property
    def descriptor(self) -> Fp8SqDescriptor:
        q = np.asarray(self.row_q_bits, dtype=np.uint8)
        _, counts = np.unique(q, return_counts=True)
        probability = counts.astype(np.float64) / float(q.size)
        return Fp8SqDescriptor(
            aggregate_bpw=(len(self.payload) - _HEADER.size)
            * 8.0
            / float(self.output_size * self.input_size),
            distribution_entropy=float(
                -np.sum(probability * np.log2(probability))
            ),
            format_version=self.format_version,
        )


@dataclass(frozen=True)
class Mxfp8SqTensor(_Fp8SqProperties):
    """One validated MXFP8-SQ tensor with native E8M0 microscales."""

    payload: bytes
    output_size: int
    input_size: int
    scale_shape: tuple[int, int]
    block_shape: tuple[int, int]
    row_q_bits: tuple[int, ...]
    scale_dtype: str = "F8_E8M0"
    format_version: int = _VERSION

    @property
    def dtype(self) -> str:
        return MXFP8_SQ_DTYPE


@dataclass(frozen=True)
class Fp8_128SqTensor(_Fp8SqProperties):
    """One validated FP8-128SQ tensor with a 128x128 multiplier grid."""

    payload: bytes
    output_size: int
    input_size: int
    scale_shape: tuple[int, int]
    block_shape: tuple[int, int]
    row_q_bits: tuple[int, ...]
    scale_dtype: str
    format_version: int = _VERSION

    @property
    def dtype(self) -> str:
        return FP8_128SQ_DTYPE


Fp8SqTensor: TypeAlias = Mxfp8SqTensor | Fp8_128SqTensor


@dataclass(frozen=True)
class Fp8SqLayout:
    """Validated offsets for either native-output FP8-SQ format."""

    dtype: str
    version: int
    shape: tuple[int, int]
    scale_dtype: str
    scale_shape: tuple[int, int]
    block_shape: tuple[int, int]
    row_q_bits: tuple[int, ...]
    row_q_offset: int
    palettes_offset: int
    symbols_offset: int
    row_symbol_byte_offsets: tuple[int, ...]
    scales_offset: int
    payload_nbytes: int


def _packed_nbytes(count: int, bits: int) -> int:
    return (int(count) * int(bits) + 7) // 8


def _aligned_selector_nbytes(rows: int) -> int:
    return (_packed_nbytes(rows, 3) + 3) & ~3


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    source = np.asarray(values, dtype=np.uint8).reshape(-1)
    if source.size and int(source.max()) >= 1 << bits:
        raise ValueError(f"value exceeds {bits}-bit stream")
    output = bytearray(_packed_nbytes(source.size, bits))
    for index, raw_value in enumerate(source.tolist()):
        value = int(raw_value)
        bit = index * bits
        output[bit // 8] |= (value << (bit % 8)) & 0xFF
        if bit % 8 + bits > 8:
            output[bit // 8 + 1] |= value >> (8 - bit % 8)
    return bytes(output)


def _unpack_bits(
    payload: bytes | memoryview,
    offset: int,
    count: int,
    bits: int,
) -> np.ndarray:
    source = memoryview(payload)
    result = np.empty(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    for index in range(count):
        bit = index * bits
        value = int(source[offset + bit // 8])
        if bit % 8 + bits > 8:
            value |= int(source[offset + bit // 8 + 1]) << 8
        result[index] = (value >> (bit % 8)) & mask
    return result


def _palette_slice(bits: int) -> slice:
    if bits < 1 or bits > 7:
        raise ValueError("FP8-SQ palettes exist only for SQ1 through SQ7")
    start = (1 << bits) - 2
    return slice(start, start + (1 << bits))


def _validate_palette_bytes(palettes: np.ndarray) -> np.ndarray:
    raw = np.ascontiguousarray(palettes, dtype=np.uint8).reshape(-1)
    if raw.size != _PALETTE_NBYTES:
        raise ValueError(f"FP8-SQ palette region must contain {_PALETTE_NBYTES} bytes")
    if np.any((raw[:254] & 0x7F) == 0x7F):
        raise ValueError("FP8-SQ palette contains an E4M3 NaN code")
    if np.any(raw[254:]):
        raise ValueError("FP8-SQ palette padding must be zero")
    for bits in range(1, 8):
        values = raw[_palette_slice(bits)]
        if np.unique(values).size != values.size:
            raise ValueError(f"FP8-SQ SQ{bits} palette contains duplicate codes")
    return raw


def _validate_scale_contract(
    dtype: str,
    shape: tuple[int, int],
    scale_dtype: str,
    scale_shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> None:
    rows, columns = shape
    block_rows, block_columns = block_shape
    if rows <= 0 or columns <= 0:
        raise ValueError("FP8-SQ requires a non-empty matrix")
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("FP8-SQ block dimensions must be positive")
    expected = (
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    )
    if scale_shape != expected:
        raise ValueError(f"FP8-SQ scale shape {scale_shape} != {expected}")
    if dtype == MXFP8_SQ_DTYPE:
        if scale_dtype != "F8_E8M0":
            raise ValueError("MXFP8-SQ requires E8M0 microscales")
        if block_shape not in _MXFP8_BLOCK_SHAPES:
            raise ValueError(f"unsupported MXFP8-SQ block geometry: {block_shape}")
        if columns % block_columns:
            raise ValueError("MXFP8-SQ columns must contain complete native blocks")
    elif dtype == FP8_128SQ_DTYPE:
        if block_shape != (128, 128):
            raise ValueError("FP8-128SQ requires 128x128 scale blocks")
        if scale_dtype not in {"BF16", "F16", "F32"}:
            raise ValueError("FP8-128SQ scale storage must be BF16, F16, or F32")
    else:
        raise ValueError(f"unsupported FP8-SQ dtype: {dtype}")


def _layout_from_fields(
    *,
    dtype: str,
    version: int,
    shape: tuple[int, int],
    scale_dtype: str,
    scale_shape: tuple[int, int],
    block_shape: tuple[int, int],
    row_q_bits: Sequence[int],
) -> Fp8SqLayout:
    rows, columns = (int(value) for value in shape)
    scale_shape = tuple(int(value) for value in scale_shape)
    block_shape = tuple(int(value) for value in block_shape)
    q = tuple(int(value) for value in row_q_bits)
    if len(q) != rows or any(value < 1 or value > 8 for value in q):
        raise ValueError("FP8-SQ needs one SQ1..SQ8 descriptor per neuron")
    _validate_scale_contract(dtype, (rows, columns), scale_dtype, scale_shape, block_shape)
    row_q_offset = _HEADER.size
    palettes_offset = row_q_offset + _aligned_selector_nbytes(rows)
    symbols_offset = palettes_offset + _PALETTE_NBYTES
    row_offsets = [0]
    for bits in q:
        row_offsets.append(row_offsets[-1] + _packed_nbytes(columns, bits))
    scales_offset = symbols_offset + row_offsets[-1]
    scale_nbytes = math.prod(scale_shape) * _SCALE_ITEMSIZE[scale_dtype]
    return Fp8SqLayout(
        dtype=dtype,
        version=version,
        shape=(rows, columns),
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
        block_shape=block_shape,
        row_q_bits=q,
        row_q_offset=row_q_offset,
        palettes_offset=palettes_offset,
        symbols_offset=symbols_offset,
        row_symbol_byte_offsets=tuple(row_offsets),
        scales_offset=scales_offset,
        payload_nbytes=scales_offset + scale_nbytes,
    )


def parse_fp8_sq_layout(
    dtype: str,
    blob: bytes | memoryview,
) -> Fp8SqLayout:
    """Validate one payload and return all runtime-relevant offsets."""

    source = memoryview(blob)
    if len(source) < _HEADER.size:
        raise ValueError("truncated FP8-SQ header")
    (
        magic,
        version,
        scale_kind,
        flags,
        reserved,
        block_rows,
        block_columns,
        rows,
        columns,
        scale_rows,
        scale_columns,
    ) = _HEADER.unpack_from(source)
    expected_magic = {
        MXFP8_SQ_DTYPE: _MXFP8_MAGIC,
        FP8_128SQ_DTYPE: _FP8_128_MAGIC,
    }.get(dtype)
    if expected_magic is None:
        raise ValueError(f"unsupported FP8-SQ dtype: {dtype}")
    if magic != expected_magic or version != _VERSION or flags or reserved:
        raise ValueError(
            f"invalid {dtype} header: magic={magic!r}, version={version}"
        )
    scale_dtype = _SCALE_DTYPE.get(int(scale_kind))
    if scale_dtype is None:
        raise ValueError(f"invalid {dtype} scale kind: {scale_kind}")
    if not 0 < rows <= 0x7FFFFFFF or not 0 < columns <= 0x7FFFFFFF:
        raise ValueError("invalid FP8-SQ matrix dimensions")
    used_selector_nbytes = _packed_nbytes(rows, 3)
    selector_nbytes = _aligned_selector_nbytes(rows)
    if len(source) < _HEADER.size + selector_nbytes + _PALETTE_NBYTES:
        raise ValueError("truncated FP8-SQ descriptor or palette region")
    selectors = _unpack_bits(source, _HEADER.size, rows, 3)
    if rows * 3 % 8:
        used = rows * 3 % 8
        if source[_HEADER.size + used_selector_nbytes - 1] & ~((1 << used) - 1):
            raise ValueError("non-zero FP8-SQ descriptor padding")
    if any(source[_HEADER.size + used_selector_nbytes : _HEADER.size + selector_nbytes]):
        raise ValueError("non-zero FP8-SQ descriptor alignment padding")
    layout = _layout_from_fields(
        dtype=dtype,
        version=version,
        shape=(int(rows), int(columns)),
        scale_dtype=scale_dtype,
        scale_shape=(int(scale_rows), int(scale_columns)),
        block_shape=(int(block_rows), int(block_columns)),
        row_q_bits=selectors + 1,
    )
    if len(source) != layout.payload_nbytes:
        raise ValueError(
            f"{dtype} payload size {len(source)} != {layout.payload_nbytes}"
        )
    palettes = np.frombuffer(
        source,
        dtype=np.uint8,
        count=_PALETTE_NBYTES,
        offset=layout.palettes_offset,
    )
    _validate_palette_bytes(palettes)
    for row, bits in enumerate(layout.row_q_bits):
        if bits != 8:
            continue
        begin = layout.symbols_offset + layout.row_symbol_byte_offsets[row]
        raw_codes = np.frombuffer(
            source,
            dtype=np.uint8,
            count=layout.shape[1],
            offset=begin,
        )
        if np.any((raw_codes & 0x7F) == 0x7F):
            raise ValueError("FP8-SQ SQ8 stream contains an E4M3 NaN code")
    scale_bytes = source[layout.scales_offset :]
    if scale_dtype == "F8_E8M0" and 255 in scale_bytes:
        raise ValueError("MXFP8-SQ contains an E8M0 NaN scale")
    if scale_dtype == "BF16":
        scale_values = (
            np.frombuffer(scale_bytes, dtype="<u2").astype(np.uint32) << 16
        ).view(np.float32)
    elif scale_dtype == "F16":
        scale_values = np.frombuffer(scale_bytes, dtype="<f2").astype(np.float32)
    elif scale_dtype == "F32":
        scale_values = np.frombuffer(scale_bytes, dtype="<f4")
    else:
        scale_values = None
    if scale_values is not None and (
        not np.isfinite(scale_values).all() or np.any(scale_values <= 0)
    ):
        raise ValueError("FP8-128SQ scale multipliers must be positive and finite")
    return layout


def _tensor_from_layout(payload: bytes, layout: Fp8SqLayout) -> Fp8SqTensor:
    fields = dict(
        payload=payload,
        output_size=layout.shape[0],
        input_size=layout.shape[1],
        scale_shape=layout.scale_shape,
        block_shape=layout.block_shape,
        row_q_bits=layout.row_q_bits,
        scale_dtype=layout.scale_dtype,
        format_version=layout.version,
    )
    if layout.dtype == MXFP8_SQ_DTYPE:
        return Mxfp8SqTensor(**fields)
    return Fp8_128SqTensor(**fields)


def unpack_fp8_sq(dtype: str, blob: bytes | memoryview) -> Fp8SqTensor:
    """Deserialize either public FP8-SQ format without merging their contracts."""

    payload = bytes(blob)
    return _tensor_from_layout(payload, parse_fp8_sq_layout(dtype, payload))


def pack_fp8_sq(tensor: Fp8SqTensor) -> bytes:
    """Serialize a tensor after checking its detached metadata."""

    parsed = unpack_fp8_sq(tensor.dtype, tensor.payload)
    fields = (
        "output_size",
        "input_size",
        "scale_shape",
        "block_shape",
        "row_q_bits",
        "scale_dtype",
        "format_version",
    )
    if any(getattr(parsed, field) != getattr(tensor, field) for field in fields):
        raise ValueError(f"{tensor.dtype} tensor metadata does not match its payload")
    return parsed.payload


def build_fp8_sq(
    *,
    dtype: str,
    row_q_bits: Sequence[int],
    input_size: int,
    block_shape: tuple[int, int],
    scale_dtype: str,
    scale_shape: tuple[int, int],
    palettes: np.ndarray,
    row_symbols: Sequence[bytes | bytearray | memoryview],
    scale_bytes: bytes | bytearray | memoryview,
) -> Fp8SqTensor:
    """Build one solver-independent native-output FP8-SQ payload."""

    q = tuple(int(value) for value in row_q_bits)
    layout = _layout_from_fields(
        dtype=dtype,
        version=_VERSION,
        shape=(len(q), int(input_size)),
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
        block_shape=block_shape,
        row_q_bits=q,
    )
    palette_bytes = _validate_palette_bytes(palettes).tobytes()
    if len(row_symbols) != len(q):
        raise ValueError("FP8-SQ requires one symbol stream per neuron")
    streams: list[bytes] = []
    for row, (bits, raw_stream) in enumerate(zip(q, row_symbols, strict=True)):
        stream = bytes(raw_stream)
        expected = _packed_nbytes(input_size, bits)
        if len(stream) != expected:
            raise ValueError(
                f"FP8-SQ row {row} needs {expected} symbol bytes for SQ{bits}"
            )
        streams.append(stream)
    raw_scales = bytes(scale_bytes)
    expected_scales = math.prod(scale_shape) * _SCALE_ITEMSIZE[scale_dtype]
    if len(raw_scales) != expected_scales:
        raise ValueError(
            f"FP8-SQ scale payload has {len(raw_scales)} bytes, expected {expected_scales}"
        )
    if scale_dtype == "F8_E8M0" and 255 in raw_scales:
        raise ValueError("MXFP8-SQ contains an E8M0 NaN scale")
    magic = _MXFP8_MAGIC if dtype == MXFP8_SQ_DTYPE else _FP8_128_MAGIC
    header = _HEADER.pack(
        magic,
        _VERSION,
        _SCALE_KIND[scale_dtype],
        0,
        0,
        *block_shape,
        len(q),
        int(input_size),
        *scale_shape,
    )
    payload = b"".join(
        (
            header,
            _pack_bits(np.asarray(q, dtype=np.uint8) - 1, 3).ljust(
                _aligned_selector_nbytes(len(q)), b"\0"
            ),
            palette_bytes,
            b"".join(streams),
            raw_scales,
        )
    )
    if len(payload) != layout.payload_nbytes:
        raise RuntimeError("internal FP8-SQ payload accounting mismatch")
    return unpack_fp8_sq(dtype, payload)


def fp8_sq_palettes(tensor: Fp8SqTensor) -> dict[int, np.ndarray]:
    """Return copies of the seven scalar palettes indexed by q."""

    layout = parse_fp8_sq_layout(tensor.dtype, tensor.payload)
    raw = np.frombuffer(
        tensor.payload,
        dtype=np.uint8,
        count=_PALETTE_NBYTES,
        offset=layout.palettes_offset,
    )
    return {bits: raw[_palette_slice(bits)].copy() for bits in range(1, 8)}


def decode_fp8_sq_codes(tensor: Fp8SqTensor) -> np.ndarray:
    """Reconstruct the exact native E4M3 byte matrix, without scaling it."""

    layout = parse_fp8_sq_layout(tensor.dtype, tensor.payload)
    palettes = fp8_sq_palettes(tensor)
    result = np.empty(layout.shape, dtype=np.uint8)
    for row, bits in enumerate(layout.row_q_bits):
        begin = layout.symbols_offset + layout.row_symbol_byte_offsets[row]
        if bits == 8:
            result[row] = np.frombuffer(
                tensor.payload,
                dtype=np.uint8,
                count=layout.shape[1],
                offset=begin,
            )
        else:
            symbols = _unpack_bits(tensor.payload, begin, layout.shape[1], bits)
            result[row] = palettes[bits][symbols]
    if np.any((result & 0x7F) == 0x7F):
        raise ValueError("FP8-SQ reconstructed an illegal E4M3 code")
    return result


def fp8_sq_scale_bytes(tensor: Fp8SqTensor) -> bytes:
    """Return an exact copy of the format-specific scale payload."""

    layout = parse_fp8_sq_layout(tensor.dtype, tensor.payload)
    return tensor.payload[layout.scales_offset :]


def fp8_sq_blob_nbytes(
    dtype: str,
    rows: int,
    columns: int,
    row_q_bits: int | Sequence[int] | np.ndarray,
    *,
    scale_dtype: str,
    scale_shape: tuple[int, int],
    block_shape: tuple[int, int],
) -> int:
    """Return the exact encoded size for one q allocation."""

    if isinstance(row_q_bits, (int, np.integer)):
        q = (int(row_q_bits),) * int(rows)
    else:
        q = tuple(int(value) for value in np.asarray(row_q_bits).reshape(-1))
    return _layout_from_fields(
        dtype=dtype,
        version=_VERSION,
        shape=(int(rows), int(columns)),
        scale_dtype=scale_dtype,
        scale_shape=scale_shape,
        block_shape=block_shape,
        row_q_bits=q,
    ).payload_nbytes


__all__ = [
    "FP8_128SQ_DTYPE",
    "FP8_SQ_DTYPES",
    "MXFP8_SQ_DTYPE",
    "Fp8SqDescriptor",
    "Fp8SqLayout",
    "Fp8SqTensor",
    "Fp8_128SqTensor",
    "Mxfp8SqTensor",
    "build_fp8_sq",
    "decode_fp8_sq_codes",
    "fp8_sq_blob_nbytes",
    "fp8_sq_palettes",
    "fp8_sq_scale_bytes",
    "pack_fp8_sq",
    "parse_fp8_sq_layout",
    "unpack_fp8_sq",
]
