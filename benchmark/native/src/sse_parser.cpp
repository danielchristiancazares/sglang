#include "sglang/benchmark/sse_parser.hpp"

#include <stdexcept>

namespace sglang::benchmark {

SseParser::SseParser(std::size_t max_line_bytes)
    : max_line_bytes_(max_line_bytes) {
  if (max_line_bytes_ == 0) {
    throw std::invalid_argument("SSE maximum line size must be positive");
  }
}

SseParseStatus SseParser::fail(std::string_view message) {
  constexpr std::size_t kMaximumErrorBytes = 384;
  error_.assign(message.substr(0, kMaximumErrorBytes));
  return SseParseStatus::kError;
}

SseParseStatus SseParser::complete_line(SseTimePoint completed_at,
                                        const SseEventCallback &on_event) {
  if (!line_.starts_with("data:")) {
    line_.clear();
    return SseParseStatus::kContinue;
  }

  std::string_view data(line_);
  data.remove_prefix(5);
  if (!data.empty() && data.front() == ' ') {
    data.remove_prefix(1);
  }

  const bool is_done = data == "[DONE]";
  if (!on_event) {
    line_.clear();
    return fail("SSE event callback is empty");
  }

  const bool keep_reading = on_event(SseEvent{
      is_done ? SseEventKind::kDone : SseEventKind::kData, data, completed_at});
  line_.clear();
  if (is_done) {
    done_ = true;
    return SseParseStatus::kDone;
  }
  if (!keep_reading) {
    stopped_ = true;
    return SseParseStatus::kStopped;
  }
  return SseParseStatus::kContinue;
}

SseParseStatus SseParser::feed(std::string_view bytes, SseTimePoint received_at,
                               const SseEventCallback &on_event) {
  if (!error_.empty()) {
    return SseParseStatus::kError;
  }
  if (done_) {
    return SseParseStatus::kDone;
  }
  if (stopped_) {
    return SseParseStatus::kStopped;
  }
  if (finished_) {
    return bytes.empty() ? SseParseStatus::kContinue
                         : fail("SSE bytes arrived after end of input");
  }

  for (char byte : bytes) {
    if (skip_lf_) {
      skip_lf_ = false;
      if (byte == '\n') {
        continue;
      }
    }

    if (byte == '\r' || byte == '\n') {
      const SseParseStatus status = complete_line(received_at, on_event);
      if (byte == '\r') {
        skip_lf_ = true;
      }
      if (status != SseParseStatus::kContinue) {
        return status;
      }
      continue;
    }

    if (line_.size() >= max_line_bytes_) {
      return fail("SSE line exceeds the configured byte limit");
    }
    line_.push_back(byte);
  }
  return SseParseStatus::kContinue;
}

SseParseStatus SseParser::finish(SseTimePoint eof_at,
                                 const SseEventCallback &on_event) {
  if (!error_.empty()) {
    return SseParseStatus::kError;
  }
  if (done_) {
    finished_ = true;
    return SseParseStatus::kDone;
  }
  if (stopped_) {
    finished_ = true;
    return SseParseStatus::kStopped;
  }
  if (finished_) {
    return SseParseStatus::kContinue;
  }

  finished_ = true;
  skip_lf_ = false;
  if (line_.empty()) {
    return SseParseStatus::kContinue;
  }
  return complete_line(eof_at, on_event);
}

void SseParser::reset() noexcept {
  line_.clear();
  error_.clear();
  skip_lf_ = false;
  done_ = false;
  stopped_ = false;
  finished_ = false;
}

std::string_view sse_parse_status_name(SseParseStatus status) noexcept {
  switch (status) {
  case SseParseStatus::kContinue:
    return "continue";
  case SseParseStatus::kDone:
    return "done";
  case SseParseStatus::kStopped:
    return "stopped";
  case SseParseStatus::kError:
    return "error";
  }
  return "unknown";
}

} // namespace sglang::benchmark
