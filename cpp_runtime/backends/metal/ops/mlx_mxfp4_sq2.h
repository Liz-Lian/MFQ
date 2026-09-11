#pragma once

#include "mlx_mxfp4_sq.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

// Frozen format-level fixed32 catalog for MXFP4-SQ2. Each five-bit state code
// selects four ordinary E2M1 nibbles; the table is not stored per tensor and
// the decoded values remain native block-32 MXFP4.
inline constexpr std::array<std::uint8_t, 128> kMxfp4Sq2PaletteNibbles{
    15, 13, 0,  5,  15, 13, 1,  6,  15, 12, 2,  6,  14, 11, 0,  3,  14, 11, 1,
    5,  14, 11, 2,  6,  14, 10, 1,  4,  14, 10, 1,  5,  14, 10, 3,  6,  14, 10,
    4,  7,  14, 9,  5,  7,  13, 10, 1,  4,  13, 9,  3,  6,  13, 0,  5,  7,  12,
    9,  2,  5,  12, 9,  2,  6,  15, 12, 3,  7,  11, 0,  3,  6,  12, 9,  3,  6,
    13, 9,  2,  6,  14, 10, 3,  7,  15, 11, 4,  7,  15, 13, 9,  4,  15, 11, 1,
    5,  15, 11, 2,  6,  15, 12, 1,  6,  15, 12, 2,  7,  15, 13, 0,  6,  14, 11,
    0,  4,  13, 1,  5,  7,  15, 14, 13, 12, 15, 14, 13, 11,
};

// Source-compatible reader name for experimental callers.  It is an alias,
// not a second format class or execution implementation.
using MlxMxfp4Sq2Weight = MlxMxfp4SqWeight;

} // namespace mfq::metal
