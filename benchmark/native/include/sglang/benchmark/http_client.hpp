#ifndef SGLANG_BENCHMARK_HTTP_CLIENT_HPP_
#define SGLANG_BENCHMARK_HTTP_CLIENT_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sglang/benchmark/config.hpp>

namespace sglang::benchmark {

using HttpClock = std::chrono::steady_clock;
using HttpTimePoint = HttpClock::time_point;

struct HttpHeader final {
  std::string name;
  std::string value;
};

struct HttpRequest final {
  std::string method{"GET"};
  std::string url;
  std::vector<HttpHeader> headers;
  std::string body;

  // A benchmark's single --timeout value can be assigned to both fields.
  std::chrono::milliseconds connect_timeout{std::chrono::minutes(10)};
  std::chrono::milliseconds io_timeout{std::chrono::minutes(10)};
  std::size_t max_header_bytes{1024U * 1024U};
  std::size_t max_body_bytes{256U * 1024U * 1024U};
  bool capture_body{true};
};

struct HttpResponse final {
  int status_code{0};
  std::string reason;
  std::vector<HttpHeader> headers;
  std::string body;
  std::uint64_t body_bytes{0};
  bool body_complete{false};
  bool stopped_early{false};
  HttpTimePoint request_started_at{};
  HttpTimePoint headers_completed_at{};
  HttpTimePoint completed_at{};

  [[nodiscard]] std::string_view header(std::string_view name) const noexcept;
};

enum class HttpErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidRequest,
  kInvalidUrl,
  kNameResolution,
  kSocket,
  kConnectTimeout,
  kConnect,
  kSendTimeout,
  kSend,
  kReceiveTimeout,
  kReceive,
  kMalformedResponse,
  kHeaderLimit,
  kBodyLimit,
  kTruncatedBody,
  kCallback,
};

struct HttpError final {
  HttpErrorCode code{HttpErrorCode::kNone};
  std::string message;
};

struct HttpResult final {
  HttpResponse response;
  HttpError error;

  [[nodiscard]] bool ok() const noexcept {
    return error.code == HttpErrorCode::kNone;
  }
};

// Returning false closes the connection and returns a successful response with
// stopped_early=true. Chunks contain response-body bytes after HTTP transfer
// decoding, and their timestamp is the receive completion time of those bytes.
using HttpBodyChunkCallback =
    std::function<bool(std::string_view, HttpTimePoint)>;

class HttpTransport {
public:
  virtual ~HttpTransport() = default;
  [[nodiscard]] virtual HttpResult
  perform(const HttpRequest &request,
          const HttpBodyChunkCallback &on_body_chunk = {}) = 0;
};

class SocketHttpTransport final : public HttpTransport {
public:
  SocketHttpTransport() = default;
  ~SocketHttpTransport() override = default;
  SocketHttpTransport(const SocketHttpTransport &) = delete;
  SocketHttpTransport &operator=(const SocketHttpTransport &) = delete;

  [[nodiscard]] HttpResult
  perform(const HttpRequest &request,
          const HttpBodyChunkCallback &on_body_chunk = {}) override;
};

[[nodiscard]] std::string_view
http_error_code_name(HttpErrorCode code) noexcept;

} // namespace sglang::benchmark

#endif // SGLANG_BENCHMARK_HTTP_CLIENT_HPP_
