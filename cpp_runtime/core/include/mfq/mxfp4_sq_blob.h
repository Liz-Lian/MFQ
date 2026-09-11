#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace mfq::sq {

// Shared SQ2/SQ3 v1 wire layout.  The public dtype is MXFP4-SQ; the symbol
// width is payload metadata and never selects a different runtime container.
struct Layout {
    int bits = 0;
    int outputs = 0;
    int width = 0;
    int base = 0;
    std::size_t symbols = 0;
    std::size_t selectors = 0;
    std::size_t scales = 0;
    std::size_t palettes = 0;
    std::size_t bytes = 0;
};

inline Layout layout(
    std::int64_t bits,
    std::int64_t outputs,
    std::int64_t width,
    std::int64_t base) {
    if ((bits != 2 && bits != 3) || outputs <= 0 || width <= 0 ||
        outputs > std::numeric_limits<int>::max() ||
        width > std::numeric_limits<int>::max() || width % 32 != 0 ||
        base < 0 || base > 251) {
        throw std::runtime_error(
            "invalid MXFP4-SQ geometry or scale base");
    }
    const auto weights =
        static_cast<std::uint64_t>(outputs) *
        static_cast<std::uint64_t>(width);
    if (weights > std::numeric_limits<std::size_t>::max() ||
        weights > (std::numeric_limits<std::uint64_t>::max() - 7) /
            static_cast<std::uint64_t>(bits)) {
        throw std::runtime_error("MXFP4-SQ payload size overflow");
    }
    const auto selectors =
        std::uint64_t{24} + (weights * static_cast<std::uint64_t>(bits) + 7) / 8;
    const auto scales = selectors + (weights / 32 + 7) / 8;
    const auto palettes = scales + static_cast<std::uint64_t>(outputs) * 2;
    const auto bytes = palettes + static_cast<std::uint64_t>(outputs) * 5;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("MXFP4-SQ payload size overflow");
    }
    return {
        static_cast<int>(bits),
        static_cast<int>(outputs),
        static_cast<int>(width),
        static_cast<int>(base),
        24,
        static_cast<std::size_t>(selectors),
        static_cast<std::size_t>(scales),
        static_cast<std::size_t>(palettes),
        static_cast<std::size_t>(bytes),
    };
}

inline Layout parse(const std::uint8_t* data, std::size_t bytes) {
    const bool sq2_magic = bytes >= 4 && data != nullptr &&
        data[0] == 'S' && data[1] == 'Q' && data[2] == '2' && data[3] == 0;
    // SQ31 was the original Metal-only SQ3 spelling.  It remains readable,
    // while all new producers use the backend-neutral SQ3\0 spelling.
    const bool sq3_magic = bytes >= 4 && data != nullptr &&
        data[0] == 'S' && data[1] == 'Q' && data[2] == '3' &&
        (data[3] == 0 || data[3] == '1');
    if (bytes < 24 || (!sq2_magic && !sq3_magic) || data[4] != 1 ||
        data[6] != 0 || data[7] != 0) {
        throw std::runtime_error("invalid MXFP4-SQ v1 header");
    }
    const auto read_dimension = [&](int offset) {
        std::uint64_t value = 0;
        for (int byte = 0; byte < 8; ++byte) {
            value |= static_cast<std::uint64_t>(data[offset + byte])
                << (byte * 8);
        }
        if (value > static_cast<std::uint64_t>(
                std::numeric_limits<int>::max())) {
            throw std::runtime_error("MXFP4-SQ dimension overflow");
        }
        return static_cast<std::int64_t>(value);
    };
    const auto result = layout(
        data[2] - '0',
        read_dimension(8),
        read_dimension(16),
        data[5]);
    if (bytes != result.bytes) {
        throw std::runtime_error(
            "truncated or trailing MXFP4-SQ payload");
    }
    return result;
}

} // namespace mfq::sq
