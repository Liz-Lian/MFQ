"""Unified, per-neuron MXFP4-SQ tensor container.

The current payload assigns one two-bit ``q-1`` descriptor to every neuron:

* q=1/2/3 select the corresponding scalar palette budget;
* q=4 is the lossless native-MXFP4 endpoint and retains the original E2M1
  values plus block-32 E8M0 scales.

All four choices are one public ``MXFP4-SQ`` format.  Historical uniform SQ2
and SQ3 payloads remain readable at this compatibility boundary.

Design intent: this is a high-fidelity, fine-grained requantization format for
weights that were natively QAT-trained in MXFP4.  Its reconstructed values stay
on the legal MXFP4/E2M1 lattice with native E8M0 block scales, so a CUDA backend
can retain native MXFP4 hardware acceleration instead of expanding the model
into a non-native codebook representation.  The packed SQ payload is the
storage layer; preserving the native reconstruction domain is what makes that
accelerated execution possible.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Sequence

import numpy as np


_HEADER = struct.Struct("<4sBBHQQ")
_CURRENT_MAGIC = b"SQV2"
_LEGACY_MAGIC = {2: b"SQ2\0", 3: b"SQ3\0"}
_LEGACY_SQ3_MAGIC = b"SQ31"


@dataclass(frozen=True)
class Mxfp4SqDescriptor:
    """Exact rate and q-distribution summary for one MXFP4-SQ tensor."""

    aggregate_bpw: float
    distribution_entropy: float
    format_version: int


@dataclass(frozen=True)
class Mxfp4SqTensor:
    """Validated, self-describing MXFP4-SQ payload."""

    payload: bytes
    bits: int
    output_size: int
    input_size: int
    matrix_scale_base: int
    row_q_bits: tuple[int, ...] = ()
    format_version: int = 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.output_size, self.input_size

    @property
    def profile(self) -> str:
        if self.bits in {1, 2, 3, 4}:
            return f"SQ{self.bits}"
        return "SQ1/SQ2/SQ3/SQ4"

    @property
    def descriptor(self) -> Mxfp4SqDescriptor:
        q = np.asarray(self.row_q_bits, dtype=np.uint8)
        if q.shape != (self.output_size,):
            q = np.full(self.output_size, self.bits, dtype=np.uint8)
        _, counts = np.unique(q, return_counts=True)
        probabilities = counts.astype(np.float64) / float(self.output_size)
        return Mxfp4SqDescriptor(
            aggregate_bpw=(len(self.payload) - _HEADER.size)
            * 8.0
            / float(self.output_size * self.input_size),
            distribution_entropy=float(
                -np.sum(probabilities * np.log2(probabilities))
            ),
            format_version=self.format_version,
        )


@dataclass(frozen=True)
class _Layout:
    version: int
    bits: int
    output_size: int
    input_size: int
    base: int
    row_q_offset: int
    symbols_offset: int
    selectors_offset: int
    scales_offset: int
    palettes_offset: int
    native_scales_offset: int
    payload_nbytes: int
    sq_rows: int
    sq4_rows: int
    row_q_bits: tuple[int, ...]


def _packed_nbytes(count: int, bits: int) -> int:
    return (int(count) * int(bits) + 7) // 8


def _aligned_q_selector_nbytes(outputs: int) -> int:
    return (_packed_nbytes(outputs, 2) + 3) & ~3


def _unpack_bits(payload: bytes, offset: int, count: int, bits: int) -> np.ndarray:
    result = np.empty(count, dtype=np.uint8)
    mask = (1 << bits) - 1
    for index in range(count):
        bit = index * bits
        value = payload[offset + bit // 8]
        if bit % 8 + bits > 8:
            value |= payload[offset + bit // 8 + 1] << 8
        result[index] = (value >> (bit % 8)) & mask
    return result


def _pack_bits(values: np.ndarray, bits: int) -> bytes:
    source = np.asarray(values, dtype=np.uint8).reshape(-1)
    result = bytearray(_packed_nbytes(source.size, bits))
    for index, value in enumerate(source.tolist()):
        bit = index * bits
        result[bit // 8] |= (value << (bit % 8)) & 0xFF
        if bit % 8 + bits > 8:
            result[bit // 8 + 1] |= value >> (8 - bit % 8)
    return bytes(result)


def _legacy_layout(bits: int, outputs: int, width: int, base: int) -> _Layout:
    weights = outputs * width
    symbols = _HEADER.size
    selectors = symbols + _packed_nbytes(weights, bits)
    scales = selectors + _packed_nbytes(weights // 32, 1)
    palettes = scales + outputs * 2
    total = palettes + outputs * 5
    return _Layout(
        1,
        bits,
        outputs,
        width,
        base,
        0,
        symbols,
        selectors,
        scales,
        palettes,
        total,
        total,
        outputs,
        0,
        (bits,) * outputs,
    )


def _adaptive_layout(
    outputs: int,
    width: int,
    base: int,
    row_q_bits: Sequence[int],
) -> _Layout:
    q = tuple(int(value) for value in row_q_bits)
    if len(q) != outputs or any(value < 1 or value > 4 for value in q):
        raise ValueError("MXFP4-SQ q descriptors must contain one value in [1,4] per row")
    sq4_rows = sum(value == 4 for value in q)
    sq_rows = outputs - sq4_rows
    blocks = width // 32
    row_q_offset = _HEADER.size
    symbols = row_q_offset + _aligned_q_selector_nbytes(outputs)
    selectors = symbols + sum(q) * width // 8
    scales = selectors + _packed_nbytes(sq_rows * blocks, 1)
    palettes = scales + sq_rows * 2
    native_scales = palettes + sq_rows * 5
    total = native_scales + sq4_rows * blocks
    return _Layout(
        2,
        0,
        outputs,
        width,
        base,
        row_q_offset,
        symbols,
        selectors,
        scales,
        palettes,
        native_scales,
        total,
        sq_rows,
        sq4_rows,
        q,
    )


def _parse_layout(payload: bytes) -> _Layout:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated MXFP4-SQ header")
    magic, version, base, reserved, outputs, width = _HEADER.unpack_from(payload)
    if reserved != 0 or not 0 <= base <= 251:
        raise ValueError("invalid MXFP4-SQ header metadata")
    if not 0 < outputs <= 0x7FFFFFFF:
        raise ValueError("invalid MXFP4-SQ output size")
    if not 0 < width <= 0x7FFFFFFF or width % 32:
        raise ValueError("MXFP4-SQ input size must be a positive multiple of 32")
    if magic in {b"SQ2\0", b"SQ3\0", _LEGACY_SQ3_MAGIC}:
        bits = 2 if magic == b"SQ2\0" else 3
        if version != 1:
            raise ValueError("invalid legacy MXFP4-SQ version")
        result = _legacy_layout(bits, outputs, width, base)
    elif magic == _CURRENT_MAGIC:
        if version != 2:
            raise ValueError("invalid adaptive MXFP4-SQ version")
        used_selector_bytes = _packed_nbytes(outputs, 2)
        selector_bytes = _aligned_q_selector_nbytes(outputs)
        if len(payload) < _HEADER.size + selector_bytes:
            raise ValueError("truncated MXFP4-SQ q descriptors")
        selectors = _unpack_bits(payload, _HEADER.size, outputs, 2)
        if outputs * 2 % 8:
            used = outputs * 2 % 8
            if payload[_HEADER.size + used_selector_bytes - 1] & ~((1 << used) - 1):
                raise ValueError("non-zero MXFP4-SQ q-descriptor padding")
        if any(payload[_HEADER.size + used_selector_bytes : _HEADER.size + selector_bytes]):
            raise ValueError("non-zero MXFP4-SQ q-descriptor alignment padding")
        result = _adaptive_layout(outputs, width, base, selectors + 1)
    else:
        raise ValueError(f"invalid MXFP4-SQ payload magic: {magic!r}")
    if len(payload) != result.payload_nbytes:
        raise ValueError(
            "MXFP4-SQ payload length mismatch: "
            f"expected {result.payload_nbytes}, got {len(payload)}"
        )
    if result.sq4_rows and 255 in payload[result.native_scales_offset :]:
        raise ValueError("MXFP4-SQ contains an E8M0 NaN native scale")
    return result


def unpack_mxfp4_sq(blob: bytes | memoryview) -> Mxfp4SqTensor:
    """Validate an adaptive payload or a legacy uniform SQ2/SQ3 payload."""

    payload = bytes(blob)
    layout = _parse_layout(payload)
    return Mxfp4SqTensor(
        payload=payload,
        bits=layout.bits,
        output_size=layout.output_size,
        input_size=layout.input_size,
        matrix_scale_base=layout.base,
        row_q_bits=layout.row_q_bits,
        format_version=layout.version,
    )


def pack_mxfp4_sq(tensor: Mxfp4SqTensor) -> bytes:
    """Serialize one validated tensor using the canonical public container."""

    parsed = unpack_mxfp4_sq(tensor.payload)
    if (
        parsed.bits != tensor.bits
        or parsed.output_size != tensor.output_size
        or parsed.input_size != tensor.input_size
        or parsed.matrix_scale_base != tensor.matrix_scale_base
        or parsed.format_version != tensor.format_version
        or (tensor.row_q_bits and parsed.row_q_bits != tuple(tensor.row_q_bits))
    ):
        raise ValueError("MXFP4-SQ tensor metadata does not match its payload")
    if parsed.payload[:4] == _LEGACY_SQ3_MAGIC:
        return _LEGACY_MAGIC[3] + parsed.payload[4:]
    return parsed.payload


def build_mxfp4_sq(
    *,
    row_q_bits: Sequence[int],
    input_size: int,
    matrix_scale_base: int,
    row_symbols: Sequence[bytes | bytearray | memoryview],
    sq_block_selectors: np.ndarray,
    sq_state_scales: np.ndarray,
    sq_state_palettes: np.ndarray,
    sq4_native_scales: np.ndarray,
) -> Mxfp4SqTensor:
    """Build the canonical adaptive payload from already-solved row streams.

    SQ rows occur in their original row order in the three ``sq_*`` arrays;
    SQ4 rows occur in row order in ``sq4_native_scales``.  This constructor is
    intentionally solver-independent so SQ1 palette search, NAQ allocation,
    and future workflows can share the same wire format.
    """

    q = tuple(int(value) for value in row_q_bits)
    outputs = len(q)
    if outputs == 0 or input_size <= 0 or input_size % 32:
        raise ValueError("MXFP4-SQ requires a non-empty block-32 matrix")
    layout = _adaptive_layout(outputs, input_size, matrix_scale_base, q)
    if len(row_symbols) != outputs:
        raise ValueError("MXFP4-SQ requires one symbol byte stream per neuron")
    symbol_rows: list[bytes] = []
    for row, (bits, stream) in enumerate(zip(q, row_symbols, strict=True)):
        encoded = bytes(stream)
        expected = input_size * bits // 8
        if len(encoded) != expected:
            raise ValueError(
                f"MXFP4-SQ row {row} needs {expected} symbol bytes for q={bits}"
            )
        symbol_rows.append(encoded)

    blocks = input_size // 32
    selectors = np.asarray(sq_block_selectors)
    state_scales = np.asarray(sq_state_scales)
    state_palettes = np.asarray(sq_state_palettes)
    native_scales = np.asarray(sq4_native_scales)
    if selectors.shape != (layout.sq_rows, blocks):
        raise ValueError("MXFP4-SQ block selectors must be [sq_rows, blocks]")
    if state_scales.shape != (layout.sq_rows, 8) or np.any(state_scales > 3):
        raise ValueError("MXFP4-SQ state scales must be [sq_rows,8] values in [0,3]")
    if state_palettes.shape != (layout.sq_rows, 8) or np.any(state_palettes > 31):
        raise ValueError("MXFP4-SQ state palettes must be [sq_rows,8] values in [0,31]")
    if native_scales.shape != (layout.sq4_rows, blocks):
        raise ValueError("MXFP4-SQ SQ4 scales must be [sq4_rows, blocks]")
    if np.any(selectors < 0) or np.any(selectors > 1):
        raise ValueError("MXFP4-SQ block selectors must be binary")
    for name, values in (
        ("state scales", state_scales),
        ("state palettes", state_palettes),
        ("SQ4 scales", native_scales),
    ):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"MXFP4-SQ {name} must use integer storage")
    if np.any(native_scales > 254):
        raise ValueError("MXFP4-SQ SQ4 scales must be valid E8M0 values")

    header = _HEADER.pack(
        _CURRENT_MAGIC,
        2,
        matrix_scale_base,
        0,
        outputs,
        input_size,
    )
    payload = b"".join(
        (
            header,
            _pack_bits(np.asarray(q, dtype=np.uint8) - 1, 2).ljust(
                _aligned_q_selector_nbytes(outputs), b"\0"
            ),
            b"".join(symbol_rows),
            _pack_bits(selectors, 1),
            _pack_bits(state_scales, 2),
            _pack_bits(state_palettes, 5),
            np.ascontiguousarray(native_scales, dtype=np.uint8).tobytes(),
        )
    )
    if len(payload) != layout.payload_nbytes:
        raise RuntimeError("internal MXFP4-SQ adaptive payload accounting mismatch")
    return unpack_mxfp4_sq(payload)


def concatenate_mxfp4_sq_rows(
    tensors: Sequence[Mxfp4SqTensor],
    row_indices: Sequence[int] | None = None,
) -> Mxfp4SqTensor:
    """Concatenate or select encoded rows without reconstructing their values.

    Legacy uniform SQ2/SQ3 payloads are promoted to the adaptive v2 wire
    format.  The operation copies packed symbols and metadata exactly; it does
    not invoke a quantizer or alter any decoded MXFP4 value.
    """

    sources = tuple(tensors)
    if not sources:
        raise ValueError("MXFP4-SQ row concatenation requires at least one tensor")

    records: list[
        tuple[
            int,
            bytes,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ] = []
    input_size = int(sources[0].input_size)
    matrix_scale_base = int(sources[0].matrix_scale_base)
    blocks = input_size // 32
    for tensor in sources:
        payload = pack_mxfp4_sq(tensor)
        layout = _parse_layout(payload)
        if layout.input_size != input_size:
            raise ValueError("MXFP4-SQ tensors must share one input size")
        if layout.base != matrix_scale_base:
            raise ValueError("MXFP4-SQ tensors must share one matrix scale base")

        selectors = _unpack_bits(
            payload,
            layout.selectors_offset,
            layout.sq_rows * blocks,
            1,
        ).reshape(layout.sq_rows, blocks)
        state_scales = _unpack_bits(
            payload,
            layout.scales_offset,
            layout.sq_rows * 8,
            2,
        ).reshape(layout.sq_rows, 8)
        state_palettes = _unpack_bits(
            payload,
            layout.palettes_offset,
            layout.sq_rows * 8,
            5,
        ).reshape(layout.sq_rows, 8)
        native_scales = np.frombuffer(
            payload,
            dtype=np.uint8,
            count=layout.sq4_rows * blocks,
            offset=layout.native_scales_offset,
        ).reshape(layout.sq4_rows, blocks)

        symbol_offset = layout.symbols_offset
        sq_row = 0
        native_row = 0
        for q in layout.row_q_bits:
            row_nbytes = input_size * int(q) // 8
            row_symbols = payload[symbol_offset : symbol_offset + row_nbytes]
            symbol_offset += row_nbytes
            if q == 4:
                records.append(
                    (
                        q,
                        row_symbols,
                        None,
                        None,
                        None,
                        native_scales[native_row].copy(),
                    )
                )
                native_row += 1
            else:
                records.append(
                    (
                        q,
                        row_symbols,
                        selectors[sq_row].copy(),
                        state_scales[sq_row].copy(),
                        state_palettes[sq_row].copy(),
                        None,
                    )
                )
                sq_row += 1

    if row_indices is None:
        selected = records
    else:
        indices = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0 or np.any(indices < 0) or np.any(indices >= len(records)):
            raise ValueError("MXFP4-SQ row indices are empty or out of range")
        selected = [records[int(index)] for index in indices]

    row_q_bits = tuple(record[0] for record in selected)
    row_symbols = tuple(record[1] for record in selected)
    sq_records = [record for record in selected if record[0] < 4]
    native_records = [record for record in selected if record[0] == 4]
    return build_mxfp4_sq(
        row_q_bits=row_q_bits,
        input_size=input_size,
        matrix_scale_base=matrix_scale_base,
        row_symbols=row_symbols,
        sq_block_selectors=(
            np.stack([record[2] for record in sq_records]).astype(np.uint8)
            if sq_records
            else np.empty((0, blocks), dtype=np.uint8)
        ),
        sq_state_scales=(
            np.stack([record[3] for record in sq_records]).astype(np.uint8)
            if sq_records
            else np.empty((0, 8), dtype=np.uint8)
        ),
        sq_state_palettes=(
            np.stack([record[4] for record in sq_records]).astype(np.uint8)
            if sq_records
            else np.empty((0, 8), dtype=np.uint8)
        ),
        sq4_native_scales=(
            np.stack([record[5] for record in native_records]).astype(np.uint8)
            if native_records
            else np.empty((0, blocks), dtype=np.uint8)
        ),
    )


__all__ = [
    "Mxfp4SqDescriptor",
    "Mxfp4SqTensor",
    "build_mxfp4_sq",
    "concatenate_mxfp4_sq_rows",
    "pack_mxfp4_sq",
    "unpack_mxfp4_sq",
]
