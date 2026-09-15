#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

// Runtime-independent token constraint used to carry a chat-template grammar
// from the HTTP server into the backend sampler.
struct MfqTokenConstraint {
    std::function<bool(std::int64_t)> allows;
    std::function<void(float *, std::size_t)> apply;
    std::function<void(std::int64_t)> accept;
    // Speculative decoding needs an independent cursor for validating a
    // proposed token chain without advancing the committed grammar state.
    std::function<std::shared_ptr<MfqTokenConstraint>()> clone;

    explicit operator bool() const noexcept {
        return static_cast<bool>(apply);
    }
};

using MfqTokenConstraintPtr = std::shared_ptr<MfqTokenConstraint>;

inline bool mfq_token_constraint_supports_speculation(
    const MfqTokenConstraintPtr& constraint) noexcept {
    return !constraint ||
        (constraint->allows && constraint->apply &&
         constraint->accept && constraint->clone);
}
