#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace mfq::fp8sq {

// Design intent: MXFP8-SQ and FP8-128SQ are high-fidelity, fine-grained
// requantization containers for native QAT weights.  They preserve legal E4M3
// reconstruction values and, respectively, native MXFP8/E8M0 or 128x128
// block-FP8 scale geometry so CUDA runtimes can retain the corresponding
// hardware-native acceleration.  Their packed SQ streams are storage
// representations; they do not define replacement arithmetic formats.

inline constexpr std::size_t kHeaderBytes = 44;
inline constexpr std::size_t kPaletteBytes = 256;

enum class Family : std::uint8_t {
    Mxfp8,
    Fp8Block128,
};

enum class ScaleKind : std::uint8_t {
    E8m0 = 1,
    Bf16 = 2,
    F16 = 3,
    F32 = 4,
};

struct Layout {
    Family family = Family::Mxfp8;
    ScaleKind scale_kind = ScaleKind::E8m0;
    int version = 0;
    int outputs = 0;
    int width = 0;
    int block_rows = 0;
    int block_columns = 0;
    int scale_rows = 0;
    int scale_columns = 0;
    std::size_t q_selectors = 0;
    std::size_t palettes = 0;
    std::size_t symbols = 0;
    std::size_t scales = 0;
    std::size_t bytes = 0;
};

struct RowMetadata {
    std::vector<std::uint8_t> q;
    std::vector<std::uint32_t> symbol_byte_offsets;
};

inline bool is_dtype(std::string_view dtype) noexcept {
    return dtype == "MXFP8-SQ" || dtype == "FP8-128SQ";
}

inline std::uint16_t read_u16(const std::uint8_t* data, std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
        static_cast<std::uint16_t>(data[offset + 1]) << 8;
}

inline std::uint64_t read_u64(const std::uint8_t* data, std::size_t offset) {
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte) {
        value |= static_cast<std::uint64_t>(data[offset + byte]) << (byte * 8);
    }
    return value;
}

inline std::uint32_t read_u32(const std::uint8_t* data, std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
        static_cast<std::uint32_t>(data[offset + 1]) << 8 |
        static_cast<std::uint32_t>(data[offset + 2]) << 16 |
        static_cast<std::uint32_t>(data[offset + 3]) << 24;
}

inline std::size_t packed_nbytes(std::uint64_t count, unsigned bits) {
    if (bits == 0 || bits > 8 ||
        count > (std::numeric_limits<std::uint64_t>::max() - 7) / bits) {
        throw std::runtime_error("FP8-SQ packed stream size overflow");
    }
    const auto bytes = (count * bits + 7) / 8;
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("FP8-SQ packed stream exceeds address space");
    }
    return static_cast<std::size_t>(bytes);
}

inline std::uint8_t read_q(const std::uint8_t* selectors, std::size_t row) {
    const auto bit = row * 3;
    const auto shift = static_cast<unsigned>(bit & 7);
    unsigned value = selectors[bit / 8];
    if (shift + 3 > 8) {
        value |= static_cast<unsigned>(selectors[bit / 8 + 1]) << 8;
    }
    return static_cast<std::uint8_t>(1u + ((value >> shift) & 7u));
}

inline std::size_t scale_itemsize(ScaleKind kind) {
    switch (kind) {
        case ScaleKind::E8m0:
            return 1;
        case ScaleKind::Bf16:
        case ScaleKind::F16:
            return 2;
        case ScaleKind::F32:
            return 4;
    }
    throw std::runtime_error("invalid FP8-SQ scale kind");
}

inline bool is_mxfp8_block(int rows, int columns) noexcept {
    return (rows == 1 && columns == 32) ||
        (rows == 32 && columns == 32) ||
        (rows == 128 && columns == 128);
}

inline Layout parse(
    std::string_view dtype,
    const std::uint8_t* data,
    std::size_t bytes) {
    if (!is_dtype(dtype) || data == nullptr || bytes < kHeaderBytes) {
        throw std::runtime_error("invalid FP8-SQ header");
    }
    const bool mxfp8 = dtype == "MXFP8-SQ";
    const std::array<std::uint8_t, 4> expected = mxfp8
        ? std::array<std::uint8_t, 4>{'M', '8', 'S', 'Q'}
        : std::array<std::uint8_t, 4>{'F', '8', 'S', 'Q'};
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (data[index] != expected[index]) {
            throw std::runtime_error("FP8-SQ dtype and payload magic disagree");
        }
    }
    if (data[4] != 1 || data[6] != 0 || data[7] != 0) {
        throw std::runtime_error("unsupported FP8-SQ header version or flags");
    }
    const auto scale_kind = static_cast<ScaleKind>(data[5]);
    const auto block_rows = static_cast<int>(read_u16(data, 8));
    const auto block_columns = static_cast<int>(read_u16(data, 10));
    const auto outputs64 = read_u64(data, 12);
    const auto width64 = read_u64(data, 20);
    const auto scale_rows64 = read_u64(data, 28);
    const auto scale_columns64 = read_u64(data, 36);
    const auto int_limit = static_cast<std::uint64_t>(
        std::numeric_limits<int>::max());
    if (outputs64 == 0 || width64 == 0 || scale_rows64 == 0 ||
        scale_columns64 == 0 || outputs64 > int_limit || width64 > int_limit ||
        scale_rows64 > int_limit || scale_columns64 > int_limit ||
        block_rows <= 0 || block_columns <= 0) {
        throw std::runtime_error("invalid FP8-SQ dimensions");
    }
    const auto outputs = static_cast<int>(outputs64);
    const auto width = static_cast<int>(width64);
    const auto scale_rows = static_cast<int>(scale_rows64);
    const auto scale_columns = static_cast<int>(scale_columns64);
    const auto expected_scale_rows =
        (outputs + block_rows - 1) / block_rows;
    const auto expected_scale_columns =
        (width + block_columns - 1) / block_columns;
    if (scale_rows != expected_scale_rows ||
        scale_columns != expected_scale_columns) {
        throw std::runtime_error("FP8-SQ scale shape disagrees with block geometry");
    }
    if (mxfp8) {
        if (scale_kind != ScaleKind::E8m0 ||
            !is_mxfp8_block(block_rows, block_columns) ||
            width % block_columns != 0) {
            throw std::runtime_error("invalid MXFP8-SQ scale contract");
        }
    } else if ((scale_kind != ScaleKind::Bf16 &&
                scale_kind != ScaleKind::F16 &&
                scale_kind != ScaleKind::F32) ||
               block_rows != 128 || block_columns != 128) {
        throw std::runtime_error("invalid FP8-128SQ scale contract");
    }

    const auto used_q_bytes = packed_nbytes(outputs64, 3);
    const auto q_bytes = (used_q_bytes + 3) & ~std::size_t{3};
    const auto palettes = kHeaderBytes + q_bytes;
    const auto symbols = palettes + kPaletteBytes;
    if (bytes < symbols) {
        throw std::runtime_error("truncated FP8-SQ descriptors or palettes");
    }
    if ((outputs64 * 3) % 8 != 0) {
        const auto used = static_cast<unsigned>((outputs64 * 3) % 8);
        const auto mask = static_cast<std::uint8_t>(~((1u << used) - 1u));
        if ((data[kHeaderBytes + used_q_bytes - 1] & mask) != 0) {
            throw std::runtime_error("non-zero FP8-SQ q-selector padding");
        }
    }
    for (std::size_t byte = used_q_bytes; byte < q_bytes; ++byte) {
        if (data[kHeaderBytes + byte] != 0) {
            throw std::runtime_error("non-zero FP8-SQ q-selector alignment padding");
        }
    }

    std::uint64_t symbol_bytes = 0;
    for (int row = 0; row < outputs; ++row) {
        symbol_bytes += packed_nbytes(
            static_cast<std::uint64_t>(width),
            read_q(data + kHeaderBytes, static_cast<std::size_t>(row)));
    }
    if (symbol_bytes > std::numeric_limits<std::size_t>::max() - symbols) {
        throw std::runtime_error("FP8-SQ symbol stream size overflow");
    }
    const auto scales = symbols + static_cast<std::size_t>(symbol_bytes);
    if (scale_rows64 > std::numeric_limits<std::uint64_t>::max() / scale_columns64) {
        throw std::runtime_error("FP8-SQ scale count overflow");
    }
    const auto scale_count = scale_rows64 * scale_columns64;
    const auto itemsize = scale_itemsize(scale_kind);
    if (scale_count >
        (std::numeric_limits<std::size_t>::max() - scales) / itemsize) {
        throw std::runtime_error("FP8-SQ scale payload size overflow");
    }
    const auto expected_bytes = scales +
        static_cast<std::size_t>(scale_count) * itemsize;
    if (bytes != expected_bytes) {
        throw std::runtime_error("truncated or trailing FP8-SQ payload");
    }

    for (int q = 1; q <= 7; ++q) {
        const auto begin = palettes + (std::size_t{1} << q) - 2;
        const auto count = std::size_t{1} << q;
        for (std::size_t left = 0; left < count; ++left) {
            const auto code = data[begin + left];
            if ((code & 0x7Fu) == 0x7Fu) {
                throw std::runtime_error("FP8-SQ palette contains E4M3 NaN");
            }
            for (std::size_t right = 0; right < left; ++right) {
                if (data[begin + right] == code) {
                    throw std::runtime_error("FP8-SQ palette contains duplicate codes");
                }
            }
        }
    }
    if (data[palettes + 254] != 0 || data[palettes + 255] != 0) {
        throw std::runtime_error("non-zero FP8-SQ palette padding");
    }
    std::size_t row_symbol_offset = 0;
    for (int row = 0; row < outputs; ++row) {
        const auto q = read_q(data + kHeaderBytes, static_cast<std::size_t>(row));
        if (q == 8) {
            const auto* codes = data + symbols + row_symbol_offset;
            for (int column = 0; column < width; ++column) {
                if ((codes[column] & 0x7Fu) == 0x7Fu) {
                    throw std::runtime_error(
                        "FP8-SQ SQ8 stream contains an E4M3 NaN code");
                }
            }
        }
        row_symbol_offset += packed_nbytes(
            static_cast<std::uint64_t>(width), q);
    }
    if (scale_kind == ScaleKind::E8m0) {
        for (std::size_t offset = scales; offset < expected_bytes; ++offset) {
            if (data[offset] == 255u) {
                throw std::runtime_error("MXFP8-SQ contains an E8M0 NaN scale");
            }
        }
    } else {
        for (std::uint64_t index = 0; index < scale_count; ++index) {
            const auto offset = scales + static_cast<std::size_t>(index) * itemsize;
            bool valid = false;
            if (scale_kind == ScaleKind::Bf16) {
                const auto raw = read_u16(data, offset);
                valid = (raw & 0x8000u) == 0 && (raw & 0x7FFFu) != 0 &&
                    (raw & 0x7F80u) != 0x7F80u;
            } else if (scale_kind == ScaleKind::F16) {
                const auto raw = read_u16(data, offset);
                valid = (raw & 0x8000u) == 0 && (raw & 0x7FFFu) != 0 &&
                    (raw & 0x7C00u) != 0x7C00u;
            } else {
                const auto raw = read_u32(data, offset);
                valid = (raw & 0x80000000u) == 0 && (raw & 0x7FFFFFFFu) != 0 &&
                    (raw & 0x7F800000u) != 0x7F800000u;
            }
            if (!valid) {
                throw std::runtime_error(
                    "FP8-128SQ scale must be positive and finite");
            }
        }
    }
    return {
        mxfp8 ? Family::Mxfp8 : Family::Fp8Block128,
        scale_kind,
        1,
        outputs,
        width,
        block_rows,
        block_columns,
        scale_rows,
        scale_columns,
        kHeaderBytes,
        palettes,
        symbols,
        scales,
        expected_bytes,
    };
}

inline RowMetadata row_metadata(
    const std::uint8_t* data,
    const Layout& layout) {
    if (data == nullptr || layout.outputs <= 0 || layout.width <= 0) {
        throw std::runtime_error("invalid FP8-SQ row metadata source");
    }
    RowMetadata result;
    result.q.resize(static_cast<std::size_t>(layout.outputs));
    result.symbol_byte_offsets.resize(static_cast<std::size_t>(layout.outputs));
    std::size_t offset = 0;
    for (std::size_t row = 0; row < result.q.size(); ++row) {
        const auto q = read_q(data + layout.q_selectors, row);
        if (offset > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("FP8-SQ row offsets exceed uint32");
        }
        result.q[row] = q;
        result.symbol_byte_offsets[row] = static_cast<std::uint32_t>(offset);
        offset += packed_nbytes(
            static_cast<std::uint64_t>(layout.width), q);
    }
    if (layout.symbols + offset != layout.scales) {
        throw std::runtime_error("inconsistent FP8-SQ row metadata");
    }
    return result;
}

} // namespace mfq::fp8sq
