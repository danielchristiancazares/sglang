#ifndef SGLANG_BENCHMARK_SSE_PARSER_HPP_
#define SGLANG_BENCHMARK_SSE_PARSER_HPP_

#include <chrono>
#include <cstddef>
#include <functional>
#include <string>
#include <string_view>

#include <sglang/benchmark/config.hpp>

namespace sglang::benchmark {

using SseClock = std::chrono::steady_clock;
using SseTimePoint = SseClock::time_point;

enum class SseEventKind : unsigned char {
  kData,
  kDone,
};

struct SseEvent final {
  SseEventKind kind{SseEventKind::kData};
  std::string_view data;
  SseTimePoint line_completed_at{};
};

// The view in SseEvent remains valid only for the duration of the callback.
// Returning false asks the parser and its transport adapter to stop early.
using SseEventCallback = std::function<bool(const SseEvent &)>;

enum class SseParseStatus : unsigned char {
  kContinue,
  kDone,
  kStopped,
  kError,
};

class SseParser final {
public:
  explicit SseParser(std::size_t max_line_bytes = 8U * 1024U * 1024U);

  [[nodiscard]] SseParseStatus feed(std::string_view bytes,
                                    SseTimePoint received_at,
                                    const SseEventCallback &on_event);
  [[nodiscard]] SseParseStatus finish(SseTimePoint eof_at,
                                      const SseEventCallback &on_event);

  void reset() noexcept;
  [[nodiscard]] bool done() const noexcept { return done_; }
  [[nodiscard]] bool finished() const noexcept { return finished_; }
  [[nodiscard]] std::string_view error() const noexcept { return error_; }

private:
  [[nodiscard]] SseParseStatus complete_line(SseTimePoint completed_at,
                                             const SseEventCallback &on_event);
  [[nodiscard]] SseParseStatus fail(std::string_view message);

  std::size_t max_line_bytes_;
  std::string line_;
  std::string error_;
  bool skip_lf_{false};
  bool done_{false};
  bool stopped_{false};
  bool finished_{false};
};

[[nodiscard]] std::string_view
sse_parse_status_name(SseParseStatus status) noexcept;

} // namespace sglang::benchmark

#endif // SGLANG_BENCHMARK_SSE_PARSER_HPP_
