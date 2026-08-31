#ifndef QWEN38_CODEX_ACTUAL_WORK_GATE_TOKEN_SPANS_HPP_
#define QWEN38_CODEX_ACTUAL_WORK_GATE_TOKEN_SPANS_HPP_

#include <cstdint>
#include <span>
#include <vector>

namespace qwen38::gate {

struct TokenSpan final {
  std::uint64_t begin;
  std::uint64_t end;

  friend constexpr bool operator==(const TokenSpan&, const TokenSpan&) noexcept =
      default;
};

enum class NormalizationStatus : std::uint8_t {
  kOk = 0,
  kInvalidSpan = 1,
  kLimitExceeded = 2,
};

struct NormalizationResult final {
  NormalizationStatus status;
  std::vector<TokenSpan> spans;
  std::uint64_t covered_tokens;

  friend constexpr bool operator==(const NormalizationResult&,
                                   const NormalizationResult&) noexcept =
      default;
};

[[nodiscard]] NormalizationResult NormalizeTokenSpans(
    std::span<const TokenSpan> spans, std::uint64_t token_limit);

}  // namespace qwen38::gate

#endif  // QWEN38_CODEX_ACTUAL_WORK_GATE_TOKEN_SPANS_HPP_
