#include "qwen38_gate/token_spans.hpp"

#include <array>
#include <limits>
#include <vector>

namespace qwen38::gate::test {

[[nodiscard]] bool RunAuthoredTests() {
  constexpr std::array<TokenSpan, 5> success_input{
      TokenSpan{4, 8}, TokenSpan{0, 4}, TokenSpan{1, 2},
      TokenSpan{3, 6}, TokenSpan{9, 10},
  };
  const NormalizationResult success_expected{
      NormalizationStatus::kOk,
      {TokenSpan{0, 8}, TokenSpan{9, 10}},
      9,
  };
  if (NormalizeTokenSpans(success_input, 10) != success_expected) {
    return false;
  }

  constexpr std::array<TokenSpan, 1> invalid_input{TokenSpan{9, 8}};
  const NormalizationResult invalid_expected{
      NormalizationStatus::kInvalidSpan, {}, 0};
  if (NormalizeTokenSpans(invalid_input, 8) != invalid_expected) {
    return false;
  }

  constexpr std::array<TokenSpan, 1> limit_input{TokenSpan{7, 9}};
  const NormalizationResult limit_expected{
      NormalizationStatus::kLimitExceeded, {}, 0};
  if (NormalizeTokenSpans(limit_input, 8) != limit_expected) {
    return false;
  }

  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  constexpr std::array<TokenSpan, 4> boundary_input{
      TokenSpan{0, 0}, TokenSpan{maximum, maximum},
      TokenSpan{maximum - 2, maximum}, TokenSpan{maximum - 1, maximum},
  };
  const NormalizationResult boundary_expected{
      NormalizationStatus::kOk, {TokenSpan{maximum - 2, maximum}}, 2};
  return NormalizeTokenSpans(boundary_input, maximum) == boundary_expected;
}

}  // namespace qwen38::gate::test
