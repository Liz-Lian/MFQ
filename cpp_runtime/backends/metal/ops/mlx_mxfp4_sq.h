#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_mxfp4_sq_dtype(std::string_view dtype) noexcept;

// One public MXFP4-SQ runtime weight. SQ2/SQ3 are payload profiles selected
// by the self-describing blob; ordinary Linear and routed MFE both execute
// through the same metadata-driven Metal matmul kernel.
class MlxMxfp4SqWeight {
public:
    static MlxMxfp4SqWeight from_blob(
        const std::vector<std::uint8_t>& blob);
    static MlxMxfp4SqWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& rows,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array matmul(const mlx::core::array& input) const;
    // Route a cohort packed as [local_experts * out_per_expert, K].
    // expert_map maps global/physical expert IDs to cohort-local IDs and
    // uses -1 for experts owned by another cohort.  This is another launch
    // mode of the same packed matmul kernel used by ordinary Linear.
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        int out_per_expert) const;
    mlx::core::array backward_input(
        const mlx::core::array& output_gradient) const;

    int bits() const noexcept {
        return bits_;
    }
    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    std::uint8_t matrix_scale_base() const noexcept {
        return matrix_scale_base_;
    }
    std::size_t packed_nbytes() const noexcept {
        return blob_.nbytes();
    }

private:
    MlxMxfp4SqWeight(
        mlx::core::array blob,
        int bits,
        int input_size,
        int output_size,
        std::uint8_t matrix_scale_base);

    mlx::core::array blob_;
    int bits_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
    std::uint8_t matrix_scale_base_ = 0;
};

} // namespace mfq::metal
