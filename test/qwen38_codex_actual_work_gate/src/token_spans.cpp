#include "qwen38_gate/token_spans.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace qwen38::gate {

NormalizationResult NormalizeTokenSpans(std::span<const TokenSpan> spans,
                                        std::uint64_t token_limit) {
  std::vector<TokenSpan> normalized;
  normalized.reserve(spans.size());

  for (const TokenSpan& span : spans) {
    if (span.begin > span.end) {
      return {NormalizationStatus::kInvalidSpan, {}, 0};
    }
    if (span.begin > token_limit || span.end > token_limit) {
      return {NormalizationStatus::kLimitExceeded, {}, 0};
    }
    if (span.begin != span.end) {
      normalized.push_back(span);
    }
  }

  std::sort(normalized.begin(), normalized.end(),
            [](const TokenSpan& lhs, const TokenSpan& rhs) noexcept {
              if (lhs.begin != rhs.begin) {
                return lhs.begin < rhs.begin;
              }
              return lhs.end < rhs.end;
            });

  std::vector<TokenSpan> merged;
  merged.reserve(normalized.size());
  for (const TokenSpan& span : normalized) {
    if (merged.empty() || span.begin > merged.back().end) {
      merged.push_back(span);
      continue;
    }
    merged.back().end = std::max(merged.back().end, span.end);
  }

  std::uint64_t covered_tokens = 0;
  for (const TokenSpan& span : merged) {
    covered_tokens += span.end - span.begin;
  }
  return {NormalizationStatus::kOk, std::move(merged), covered_tokens};
}

}  // namespace qwen38::gate
