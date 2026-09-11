#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <mlx/mlx.h>

namespace mfq::metal {

bool is_nint_dtype(std::string_view dtype) noexcept;

class MlxNintWeight {
public:
    static MlxNintWeight from_blob(
        std::span<const std::uint8_t> blob);

    mlx::core::array matmul(const mlx::core::array& input) const;
    mlx::core::array matmul_add(
        const mlx::core::array& input,
        const mlx::core::array& residual) const;
    // Routed MFE projection over this packed expert cohort.  This is another
    // invocation mode of the ordinary NINT matmul kernel, not a separate MoE
    // decoder. expert_map maps global expert IDs to cohort-local rows and -1
    // for experts owned by another MFE cohort.
    mlx::core::array routed_matmul(
        const mlx::core::array& input,
        const mlx::core::array& expert_ids,
        const mlx::core::array& expert_map,
        int out_per_expert) const;
    // Decode-only fast path for a single FP16 row. Supported layouts compute
    // the LM-head projection and greedy argmax without materializing logits.
    std::optional<mlx::core::array> greedy_argmax(
        const mlx::core::array& input) const;
    bool can_fuse_swiglu(
        const MlxNintWeight& up) const noexcept;
    mlx::core::array swiglu(
        const MlxNintWeight& up,
        const mlx::core::array& input) const;
    mlx::core::array dequantize(
        mlx::core::Dtype dtype = mlx::core::float16) const;
    mlx::core::array embedding(
        const mlx::core::array& token_ids,
        mlx::core::Dtype dtype = mlx::core::float16) const;
    // O-LoRA-style grouped projection:
    // [..., M, G, K] x [G * O, K] -> [..., M, G, O].
    // Reuses the routed mode of the same metadata-driven NINT matmul kernel;
    // the optional return type is retained for the packed-linear interface.
    std::optional<mlx::core::array> grouped_row_matmul(
        const mlx::core::array& input,
        int group_count) const;

    int bits() const noexcept {
        return bits_;
    }
    int group_size() const noexcept {
        return group_size_;
    }
    int groups() const noexcept {
        return groups_;
    }
    int input_size() const noexcept {
        return input_size_;
    }
    int output_size() const noexcept {
        return output_size_;
    }
    std::size_t packed_nbytes() const noexcept;

    // Read-only packed storage views used by fused/grouped Metal kernels.
    // The arrays remain owned by this weight.
    const mlx::core::array& packed_values() const noexcept {
        return q_packed_;
    }
    const mlx::core::array& sub_scales() const noexcept {
        return sub_scale_;
    }
    const mlx::core::array& sub_mins() const noexcept {
        return sub_min_;
    }
    const mlx::core::array& neuron_scales() const noexcept {
        return neuron_scale_;
    }
    const mlx::core::array& neuron_mins() const noexcept {
        return neuron_min_;
    }
    const mlx::core::array& row_q_layout() const noexcept {
        return row_q_layout_;
    }
    const mlx::core::array& row_q_byte_offsets() const noexcept {
        return row_q_byte_offsets_;
    }
    bool has_uniform_q_bits() const noexcept {
        return uniform_q_bits_;
    }

private:
    mlx::core::array matmul_impl(
        const mlx::core::array& input,
        const mlx::core::array* residual) const;

    MlxNintWeight(
        mlx::core::array q_packed,
        mlx::core::array sub_scale,
        mlx::core::array sub_min,
        mlx::core::array neuron_scale,
        mlx::core::array neuron_min,
        mlx::core::array row_q_layout,
        mlx::core::array row_q_byte_offsets,
        int bits,
        int group_size,
        int groups,
        int input_size,
        int output_size,
        bool uniform_q_bits);

    mlx::core::array q_packed_;
    mlx::core::array sub_scale_;
    mlx::core::array sub_min_;
    mlx::core::array neuron_scale_;
    mlx::core::array neuron_min_;
    mlx::core::array row_q_layout_;
    mlx::core::array row_q_byte_offsets_;
    int bits_ = 0;
    int group_size_ = 0;
    int groups_ = 0;
    int input_size_ = 0;
    int output_size_ = 0;
    bool uniform_q_bits_ = false;
};

} // namespace mfq::metal
