#include "qwen38_gate/token_spans.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace qwen38::gate::test {
[[nodiscard]] bool RunAuthoredTests();
}  // namespace qwen38::gate::test

namespace {

using qwen38::gate::NormalizationResult;
using qwen38::gate::NormalizationStatus;
using qwen38::gate::NormalizeTokenSpans;
using qwen38::gate::TokenSpan;

static_assert(
    std::is_same_v<std::underlying_type_t<NormalizationStatus>, std::uint8_t>);
static_assert(static_cast<std::uint8_t>(NormalizationStatus::kOk) == 0);
static_assert(static_cast<std::uint8_t>(NormalizationStatus::kInvalidSpan) == 1);
static_assert(
    static_cast<std::uint8_t>(NormalizationStatus::kLimitExceeded) == 2);

[[nodiscard]] bool RecordCheck(bool passed, const char* expression,
                               int line) noexcept {
  if (!passed) {
    std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, line,
                 expression);
  }
  return passed;
}

#define CHECK(expression)                                                     \
  do {                                                                        \
    if (!RecordCheck(static_cast<bool>(expression), #expression, __LINE__)) { \
      return false;                                                           \
    }                                                                         \
  } while (false)

[[nodiscard]] bool ExpectResult(
    const NormalizationResult& actual, NormalizationStatus status,
    std::initializer_list<TokenSpan> expected_spans,
    std::uint64_t covered_tokens) {
  const std::vector<TokenSpan> expected(expected_spans);
  CHECK(actual.status == status);
  CHECK(actual.spans == expected);
  CHECK(actual.covered_tokens == covered_tokens);
  return true;
}

[[nodiscard]] bool MergesUnsortedOverlapAndAdjacency() {
  constexpr std::array<TokenSpan, 6> input{
      TokenSpan{8, 12}, TokenSpan{0, 4},   TokenSpan{4, 8},
      TokenSpan{3, 6},  TokenSpan{20, 20}, TokenSpan{16, 18},
  };
  return ExpectResult(NormalizeTokenSpans(input, 20),
                      NormalizationStatus::kOk,
                      {TokenSpan{0, 12}, TokenSpan{16, 18}}, 14);
}

[[nodiscard]] bool KeepsSeparatedSpansAndCountsUniqueCoverage() {
  constexpr std::array<TokenSpan, 4> input{
      TokenSpan{5, 7}, TokenSpan{1, 2}, TokenSpan{9, 10}, TokenSpan{5, 7},
  };
  return ExpectResult(NormalizeTokenSpans(input, 10),
                      NormalizationStatus::kOk,
                      {TokenSpan{1, 2}, TokenSpan{5, 7}, TokenSpan{9, 10}}, 4);
}

[[nodiscard]] bool AcceptsEmptyInputAndExactLimit() {
  CHECK(ExpectResult(NormalizeTokenSpans(std::span<const TokenSpan>{}, 0),
                     NormalizationStatus::kOk, {}, 0));
  constexpr std::array<TokenSpan, 3> input{
      TokenSpan{8, 10}, TokenSpan{10, 10}, TokenSpan{0, 0},
  };
  return ExpectResult(NormalizeTokenSpans(input, 10),
                      NormalizationStatus::kOk, {TokenSpan{8, 10}}, 2);
}

[[nodiscard]] bool ReportsTheFirstFailureInInputOrder() {
  constexpr std::array<TokenSpan, 3> invalid_first{
      TokenSpan{0, 1}, TokenSpan{9, 8}, TokenSpan{11, 12},
  };
  CHECK(ExpectResult(NormalizeTokenSpans(invalid_first, 10),
                     NormalizationStatus::kInvalidSpan, {}, 0));

  constexpr std::array<TokenSpan, 3> limit_first{
      TokenSpan{0, 1}, TokenSpan{9, 11}, TokenSpan{7, 6},
  };
  CHECK(ExpectResult(NormalizeTokenSpans(limit_first, 10),
                     NormalizationStatus::kLimitExceeded, {}, 0));

  constexpr std::array<TokenSpan, 1> empty_beyond_limit{
      TokenSpan{11, 11},
  };
  return ExpectResult(NormalizeTokenSpans(empty_beyond_limit, 10),
                      NormalizationStatus::kLimitExceeded, {}, 0);
}

[[nodiscard]] bool HandlesMaximumEndpointWithoutOverflow() {
  constexpr std::uint64_t maximum =
      std::numeric_limits<std::uint64_t>::max();
  constexpr std::array<TokenSpan, 3> input{
      TokenSpan{maximum - 2, maximum}, TokenSpan{0, maximum - 2},
      TokenSpan{5, 5},
  };
  return ExpectResult(NormalizeTokenSpans(input, maximum),
                      NormalizationStatus::kOk,
                      {TokenSpan{0, maximum}}, maximum);
}

}  // namespace

int main() {
  bool passed = true;
  if (!qwen38::gate::test::RunAuthoredTests()) {
    std::fputs("authored tests failed\n", stderr);
    passed = false;
  }
  if (!MergesUnsortedOverlapAndAdjacency()) {
    passed = false;
  }
  if (!KeepsSeparatedSpansAndCountsUniqueCoverage()) {
    passed = false;
  }
  if (!AcceptsEmptyInputAndExactLimit()) {
    passed = false;
  }
  if (!ReportsTheFirstFailureInInputOrder()) {
    passed = false;
  }
  if (!HandlesMaximumEndpointWithoutOverflow()) {
    passed = false;
  }
  if (!passed) {
    return EXIT_FAILURE;
  }
  std::puts("QWEN38_CPP_MULTI_FILE_GATE=passed");
  return EXIT_SUCCESS;
}
