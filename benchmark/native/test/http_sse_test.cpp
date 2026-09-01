#include "sglang/benchmark/http_client.hpp"
#include "sglang/benchmark/sse_parser.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

using sglang::benchmark::HttpBodyChunkCallback;
using sglang::benchmark::HttpClock;
using sglang::benchmark::HttpErrorCode;
using sglang::benchmark::HttpHeader;
using sglang::benchmark::HttpRequest;
using sglang::benchmark::HttpResult;
using sglang::benchmark::HttpTimePoint;
using sglang::benchmark::SocketHttpTransport;
using sglang::benchmark::SseEvent;
using sglang::benchmark::SseEventKind;
using sglang::benchmark::SseParser;
using sglang::benchmark::SseParseStatus;

#ifdef _WIN32
using TestSocket = SOCKET;
constexpr TestSocket kInvalidTestSocket = INVALID_SOCKET;
void close_test_socket(TestSocket socket) { closesocket(socket); }
class TestNetwork final {
public:
  TestNetwork() {
    WSADATA data{};
    ready_ = WSAStartup(MAKEWORD(2, 2), &data) == 0;
  }
  ~TestNetwork() {
    if (ready_) {
      WSACleanup();
    }
  }
  [[nodiscard]] bool ready() const noexcept { return ready_; }

private:
  bool ready_{false};
};
#else
using TestSocket = int;
constexpr TestSocket kInvalidTestSocket = -1;
void close_test_socket(TestSocket socket) { close(socket); }
class TestNetwork final {
public:
  [[nodiscard]] constexpr bool ready() const noexcept { return true; }
};
#endif

[[nodiscard]] TestNetwork &test_network() {
  static TestNetwork network;
  return network;
}

struct ResponseSegment final {
  std::string bytes;
  std::chrono::milliseconds delay_before{0};
};

class LoopbackServer final {
public:
  explicit LoopbackServer(std::vector<ResponseSegment> response,
                          bool ipv6 = false)
      : response_(std::move(response)), ipv6_(ipv6) {
    if (!test_network().ready()) {
      return;
    }
    listener_ = ::socket(ipv6 ? AF_INET6 : AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (listener_ == kInvalidTestSocket) {
      return;
    }
    const int reuse = 1;
    static_cast<void>(setsockopt(listener_, SOL_SOCKET, SO_REUSEADDR,
                                 reinterpret_cast<const char *>(&reuse),
                                 sizeof(reuse)));

    if (ipv6) {
      sockaddr_in6 address{};
      address.sin6_family = AF_INET6;
      address.sin6_addr = in6addr_loopback;
      address.sin6_port = 0;
      if (::bind(listener_, reinterpret_cast<const sockaddr *>(&address),
                 sizeof(address)) != 0) {
        close_test_socket(listener_);
        listener_ = kInvalidTestSocket;
        return;
      }
      socklen_type length = sizeof(address);
      if (getsockname(listener_, reinterpret_cast<sockaddr *>(&address),
                      &length) != 0) {
        close_test_socket(listener_);
        listener_ = kInvalidTestSocket;
        return;
      }
      port_ = ntohs(address.sin6_port);
    } else {
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      address.sin_port = 0;
      if (::bind(listener_, reinterpret_cast<const sockaddr *>(&address),
                 sizeof(address)) != 0) {
        close_test_socket(listener_);
        listener_ = kInvalidTestSocket;
        return;
      }
      socklen_type length = sizeof(address);
      if (getsockname(listener_, reinterpret_cast<sockaddr *>(&address),
                      &length) != 0) {
        close_test_socket(listener_);
        listener_ = kInvalidTestSocket;
        return;
      }
      port_ = ntohs(address.sin_port);
    }
    if (::listen(listener_, 1) != 0) {
      close_test_socket(listener_);
      listener_ = kInvalidTestSocket;
      return;
    }
    const TestSocket accepted_listener = listener_;
    worker_ =
        std::thread([this, accepted_listener] { serve(accepted_listener); });
  }

  LoopbackServer(const LoopbackServer &) = delete;
  LoopbackServer &operator=(const LoopbackServer &) = delete;

  ~LoopbackServer() { join(); }

  [[nodiscard]] bool ready() const noexcept {
    return listener_ != kInvalidTestSocket;
  }
  [[nodiscard]] std::string url(std::string_view path = "/probe") const {
    return std::string(ipv6_ ? "http://[::1]:" : "http://127.0.0.1:") +
           std::to_string(port_) + std::string(path);
  }
  void join() {
    if (listener_ != kInvalidTestSocket) {
#ifdef _WIN32
      static_cast<void>(shutdown(listener_, SD_BOTH));
#else
      static_cast<void>(shutdown(listener_, SHUT_RDWR));
#endif
      close_test_socket(listener_);
      listener_ = kInvalidTestSocket;
    }
    if (worker_.joinable()) {
      worker_.join();
    }
  }
  [[nodiscard]] const std::string &request() const noexcept { return request_; }

private:
#ifdef _WIN32
  using socklen_type = int;
#else
  using socklen_type = socklen_t;
#endif

  static void send_bytes(TestSocket socket, std::string_view bytes) {
    std::size_t position = 0;
    while (position < bytes.size()) {
      const int count = static_cast<int>(bytes.size() - position);
#ifdef MSG_NOSIGNAL
      constexpr int kFlags = MSG_NOSIGNAL;
#else
      constexpr int kFlags = 0;
#endif
      const int sent = ::send(socket, bytes.data() + position, count, kFlags);
      if (sent <= 0) {
        return;
      }
      position += static_cast<std::size_t>(sent);
    }
  }

  void serve(TestSocket listener) {
    TestSocket client = ::accept(listener, nullptr, nullptr);
    if (client == kInvalidTestSocket) {
      return;
    }
#if !defined(_WIN32) && defined(SO_NOSIGPIPE)
    const int no_sigpipe = 1;
    static_cast<void>(setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe,
                                 sizeof(no_sigpipe)));
#endif
    std::array<char, 4096> input{};
    std::size_t required_size = 0;
    for (;;) {
      const int received =
          ::recv(client, input.data(), static_cast<int>(input.size()), 0);
      if (received <= 0) {
        break;
      }
      request_.append(input.data(), static_cast<std::size_t>(received));
      const std::size_t header_end = request_.find("\r\n\r\n");
      if (header_end == std::string::npos) {
        continue;
      }
      if (required_size == 0) {
        required_size = header_end + 4;
        constexpr std::string_view kLength = "Content-Length:";
        const std::size_t length_position = request_.find(kLength);
        if (length_position != std::string::npos &&
            length_position < header_end) {
          const std::size_t value_begin = length_position + kLength.size();
          const std::size_t value_end = request_.find("\r\n", value_begin);
          const std::string text =
              request_.substr(value_begin, value_end - value_begin);
          required_size += static_cast<std::size_t>(std::stoull(text));
        }
      }
      if (request_.size() >= required_size) {
        break;
      }
    }
    for (const ResponseSegment &segment : response_) {
      if (segment.delay_before.count() != 0) {
        std::this_thread::sleep_for(segment.delay_before);
      }
      send_bytes(client, segment.bytes);
    }
    close_test_socket(client);
  }

  std::vector<ResponseSegment> response_;
  bool ipv6_{false};
  TestSocket listener_{kInvalidTestSocket};
  std::uint16_t port_{0};
  std::thread worker_;
  std::string request_;
};

[[nodiscard]] std::optional<std::uint16_t> closed_loopback_port() {
  if (!test_network().ready()) {
    return std::nullopt;
  }
  const TestSocket socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (socket == kInvalidTestSocket) {
    return std::nullopt;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(socket, reinterpret_cast<const sockaddr *>(&address),
             sizeof(address)) != 0) {
    close_test_socket(socket);
    return std::nullopt;
  }
#ifdef _WIN32
  int length = sizeof(address);
#else
  socklen_t length = sizeof(address);
#endif
  if (getsockname(socket, reinterpret_cast<sockaddr *>(&address), &length) !=
      0) {
    close_test_socket(socket);
    return std::nullopt;
  }
  const std::uint16_t port = ntohs(address.sin_port);
  close_test_socket(socket);
  return port;
}

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {     \
    return false;                                                              \
  }

[[nodiscard]] HttpRequest request_for(const LoopbackServer &server) {
  HttpRequest request;
  request.method = "POST";
  request.url = server.url("/v1/probe?mode=host");
  request.headers = {HttpHeader{"Content-Type", "application/json"},
                     HttpHeader{"X-Probe", "native"}};
  request.body = "{\"value\":7}";
  request.connect_timeout = std::chrono::seconds(2);
  request.io_timeout = std::chrono::seconds(2);
  return request;
}

[[nodiscard]] bool ContentLengthPostAndHeaders() {
  LoopbackServer server({ResponseSegment{"HTTP/1.1 200 Fine\r\nContent-Length: "
                                         "5\r\nX-Mixed: Value\r\n\r\nhello"}});
  CHECK(server.ready());
  SocketHttpTransport transport;
  std::string callbacks;
  std::vector<HttpTimePoint> callback_times;
  HttpRequest request = request_for(server);
  const HttpResult result = transport.perform(
      request, [&](std::string_view bytes, HttpTimePoint received_at) {
        callbacks.append(bytes);
        callback_times.push_back(received_at);
        return true;
      });
  server.join();
  CHECK(result.ok());
  CHECK(result.response.status_code == 200);
  CHECK(result.response.reason == "Fine");
  CHECK(result.response.header("x-mIxEd") == "Value");
  CHECK(result.response.body == "hello");
  CHECK(result.response.body_bytes == 5);
  CHECK(result.response.body_complete);
  CHECK(!result.response.stopped_early);
  CHECK(callbacks == "hello");
  CHECK(!callback_times.empty());
  CHECK(callback_times.front() >= result.response.request_started_at);
  CHECK(result.response.completed_at >= result.response.headers_completed_at);
  CHECK(server.request().starts_with("POST /v1/probe?mode=host HTTP/1.1\r\n"));
  CHECK(server.request().find("Content-Type: application/json\r\n") !=
        std::string::npos);
  CHECK(server.request().find("X-Probe: native\r\n") != std::string::npos);
  CHECK(server.request().find("Host: 127.0.0.1:") != std::string::npos);
  CHECK(server.request().find("Content-Length: 11\r\n") != std::string::npos);
  CHECK(server.request().find("Connection: close\r\n") != std::string::npos);
  CHECK(server.request().ends_with("\r\n\r\n{\"value\":7}"));
  return true;
}

[[nodiscard]] bool ChunkedExtensionsTrailersAndDechunkedCallbacks() {
  LoopbackServer server({
      ResponseSegment{"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                      "4;source=test\r\nWi"},
      ResponseSegment{"ki\r\n5\r\nped"},
      ResponseSegment{"ia\r\n0;final=yes\r\nX-Trailer: present\r\n\r\n"},
  });
  CHECK(server.ready());
  SocketHttpTransport transport;
  std::string callbacks;
  HttpRequest request = request_for(server);
  const HttpResult result =
      transport.perform(request, [&](std::string_view bytes, HttpTimePoint) {
        callbacks.append(bytes);
        CHECK(bytes.find("\r\n") == std::string_view::npos);
        return true;
      });
  server.join();
  CHECK(result.ok());
  CHECK(result.response.body_complete);
  CHECK(result.response.body == "Wikipedia");
  CHECK(callbacks == "Wikipedia");
  CHECK(result.response.body_bytes == 9);
  return true;
}

[[nodiscard]] bool GetInformationalResponseAndHostnameResolution() {
  LoopbackServer server(
      {ResponseSegment{"HTTP/1.1 100 Continue\r\nStage: interim\r\n\r\n"
                       "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"}});
  CHECK(server.ready());
  SocketHttpTransport transport;
  HttpRequest request;
  request.method = "GET";
  request.url = server.url("?ready=yes");
  const std::size_t host_begin = request.url.find("127.0.0.1");
  CHECK(host_begin != std::string::npos);
  request.url.replace(host_begin, std::strlen("127.0.0.1"), "localhost");
  request.connect_timeout = std::chrono::seconds(2);
  request.io_timeout = std::chrono::seconds(2);
  const HttpResult result = transport.perform(request);
  server.join();
  if (!result.ok()) {
    std::printf(
        "hostname/interim transport error: %.*s: %s\n",
        static_cast<int>(
            sglang::benchmark::http_error_code_name(result.error.code).size()),
        sglang::benchmark::http_error_code_name(result.error.code).data(),
        result.error.message.c_str());
  }
  CHECK(result.ok());
  CHECK(result.response.status_code == 200);
  CHECK(result.response.body == "ok");
  CHECK(server.request().starts_with("GET /?ready=yes HTTP/1.1\r\n"));
  CHECK(server.request().find("Host: localhost:") != std::string::npos);
  CHECK(server.request().find("Content-Length: 0\r\n") != std::string::npos);
  return true;
}

[[nodiscard]] bool CloseDelimitedAndCaptureCanBeDisabled() {
  LoopbackServer server({ResponseSegment{
      "HTTP/1.0 201 Created\r\nContent-Type: text/plain\r\n\r\nclose-body"}});
  CHECK(server.ready());
  SocketHttpTransport transport;
  HttpRequest request = request_for(server);
  request.capture_body = false;
  std::string callbacks;
  const HttpResult result =
      transport.perform(request, [&](std::string_view bytes, HttpTimePoint) {
        callbacks.append(bytes);
        return true;
      });
  server.join();
  CHECK(result.ok());
  CHECK(result.response.status_code == 201);
  CHECK(result.response.body.empty());
  CHECK(result.response.body_bytes == 10);
  CHECK(result.response.body_complete);
  CHECK(callbacks == "close-body");
  return true;
}

[[nodiscard]] bool CallbackCanStopAStreamingBody() {
  LoopbackServer server({
      ResponseSegment{"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                      "3\r\none\r\n"},
      ResponseSegment{"3\r\ntwo\r\n0\r\n\r\n", std::chrono::milliseconds(80)},
  });
  CHECK(server.ready());
  SocketHttpTransport transport;
  HttpRequest request = request_for(server);
  const HttpResult result = transport.perform(
      request, [](std::string_view, HttpTimePoint) { return false; });
  server.join();
  CHECK(result.ok());
  CHECK(result.response.stopped_early);
  CHECK(!result.response.body_complete);
  CHECK(result.response.body == "one");
  CHECK(result.response.body_bytes == 3);
  return true;
}

[[nodiscard]] bool TruncationLimitsAndTimeoutsFailClosed() {
  {
    LoopbackServer server(
        {ResponseSegment{"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nshort"}});
    CHECK(server.ready());
    SocketHttpTransport transport;
    const HttpResult result = transport.perform(request_for(server));
    server.join();
    CHECK(!result.ok());
    CHECK(result.error.code == HttpErrorCode::kTruncatedBody);
  }
  {
    LoopbackServer server({ResponseSegment{
        "HTTP/1.1 200 OK\r\nContent-Length: 20\r\n\r\n01234567890123456789"}});
    CHECK(server.ready());
    SocketHttpTransport transport;
    HttpRequest request = request_for(server);
    request.max_body_bytes = 8;
    const HttpResult result = transport.perform(request);
    server.join();
    CHECK(!result.ok());
    CHECK(result.error.code == HttpErrorCode::kBodyLimit);
  }
  {
    LoopbackServer server(
        {ResponseSegment{"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n",
                         std::chrono::milliseconds(100)}});
    CHECK(server.ready());
    SocketHttpTransport transport;
    HttpRequest request = request_for(server);
    request.io_timeout = std::chrono::milliseconds(20);
    const HttpResult result = transport.perform(request);
    server.join();
    CHECK(!result.ok());
    CHECK(result.error.code == HttpErrorCode::kReceiveTimeout);
  }
  return true;
}

[[nodiscard]] bool InvalidRequestsStayBoundedAndOffline() {
  SocketHttpTransport transport;
  HttpRequest request;
  request.url = "https://example.invalid/";
  HttpResult result = transport.perform(request);
  CHECK(!result.ok());
  CHECK(result.error.code == HttpErrorCode::kInvalidUrl);
  CHECK(result.error.message.size() <= 768);

  request.url = "http://127.0.0.1:80/";
  request.method = "DELETE";
  result = transport.perform(request);
  CHECK(!result.ok());
  CHECK(result.error.code == HttpErrorCode::kInvalidRequest);

  request.method = "GET";
  request.connect_timeout = std::chrono::milliseconds(0);
  result = transport.perform(request);
  CHECK(!result.ok());
  CHECK(result.error.code == HttpErrorCode::kInvalidRequest);

  request.connect_timeout = std::chrono::seconds(1);
  request.headers = {HttpHeader{"Bad\r\nName", "value"}};
  result = transport.perform(request);
  CHECK(!result.ok());
  CHECK(result.error.code == HttpErrorCode::kInvalidRequest);

  if (const auto port = closed_loopback_port(); port.has_value()) {
    request = HttpRequest{};
    request.url = "http://127.0.0.1:" + std::to_string(*port) + "/refused";
    request.connect_timeout = std::chrono::seconds(2);
    request.io_timeout = std::chrono::seconds(2);
    result = transport.perform(request);
    CHECK(!result.ok());
    // Local firewall policy may reject a closed port immediately or silently
    // drop it through the configured deadline. Both are bounded failures.
    CHECK(result.error.code == HttpErrorCode::kConnect ||
          result.error.code == HttpErrorCode::kConnectTimeout);
  }
  return true;
}

[[nodiscard]] bool IPv6LoopbackWhenAvailable() {
  LoopbackServer server(
      {ResponseSegment{"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n"}},
      true);
  if (!server.ready()) {
    return true;
  }
  SocketHttpTransport transport;
  const HttpResult result = transport.perform(request_for(server));
  server.join();
  CHECK(result.ok());
  CHECK(result.response.status_code == 204);
  CHECK(result.response.body_complete);
  CHECK(server.request().find("Host: [::1]:") != std::string::npos);
  return true;
}

[[nodiscard]] bool SseArbitraryBoundariesAndTimestamps() {
  SseParser parser;
  const HttpTimePoint first = HttpClock::now();
  const HttpTimePoint second = first + std::chrono::milliseconds(1);
  const HttpTimePoint third = second + std::chrono::milliseconds(1);
  struct OwnedEvent final {
    SseEventKind kind;
    std::string data;
    HttpTimePoint at;
  };
  std::vector<OwnedEvent> events;
  const auto callback = [&](const SseEvent &event) {
    events.push_back(OwnedEvent{event.kind, std::string(event.data),
                                event.line_completed_at});
    return true;
  };
  CHECK(parser.feed("da", first, callback) == SseParseStatus::kContinue);
  CHECK(parser.feed("ta: {\"x\":1}\r", second, callback) ==
        SseParseStatus::kContinue);
  CHECK(parser.feed("\n: comment\ndata: two\r\n", third, callback) ==
        SseParseStatus::kContinue);
  CHECK(events.size() == 2);
  CHECK(events[0].kind == SseEventKind::kData);
  CHECK(events[0].data == "{\"x\":1}");
  CHECK(events[0].at == second);
  CHECK(events[1].data == "two");
  CHECK(events[1].at == third);
  return true;
}

[[nodiscard]] bool SseDoneEofStopResetAndLimits() {
  const HttpTimePoint now = HttpClock::now();
  {
    SseParser parser;
    std::vector<std::string> values;
    const auto callback = [&](const SseEvent &event) {
      values.emplace_back(event.data);
      return true;
    };
    CHECK(parser.feed("data:first", now, callback) ==
          SseParseStatus::kContinue);
    CHECK(parser.finish(now, callback) == SseParseStatus::kContinue);
    CHECK(values == std::vector<std::string>{"first"});
    CHECK(parser.finished());
    parser.reset();
    CHECK(parser.feed("data: [DONE]\nignored", now, callback) ==
          SseParseStatus::kDone);
    CHECK(parser.done());
    CHECK(values.back() == "[DONE]");
  }
  {
    SseParser parser;
    CHECK(parser.feed("data: stop\n", now, [](const SseEvent &) {
      return false;
    }) == SseParseStatus::kStopped);
    CHECK(parser.finish(now, [](const SseEvent &) { return true; }) ==
          SseParseStatus::kStopped);
  }
  {
    SseParser parser(3);
    CHECK(parser.feed("data", now, [](const SseEvent &) { return true; }) ==
          SseParseStatus::kError);
    CHECK(!parser.error().empty());
    CHECK(parser.error().size() <= 384);
  }
  return true;
}

using Test = bool (*)();
constexpr std::array<std::pair<std::string_view, Test>, 10> kTests{{
    {"ContentLengthPostAndHeaders", ContentLengthPostAndHeaders},
    {"ChunkedExtensionsTrailersAndDechunkedCallbacks",
     ChunkedExtensionsTrailersAndDechunkedCallbacks},
    {"GetInformationalResponseAndHostnameResolution",
     GetInformationalResponseAndHostnameResolution},
    {"CloseDelimitedAndCaptureCanBeDisabled",
     CloseDelimitedAndCaptureCanBeDisabled},
    {"CallbackCanStopAStreamingBody", CallbackCanStopAStreamingBody},
    {"TruncationLimitsAndTimeoutsFailClosed",
     TruncationLimitsAndTimeoutsFailClosed},
    {"InvalidRequestsStayBoundedAndOffline",
     InvalidRequestsStayBoundedAndOffline},
    {"IPv6LoopbackWhenAvailable", IPv6LoopbackWhenAvailable},
    {"SseArbitraryBoundariesAndTimestamps",
     SseArbitraryBoundariesAndTimestamps},
    {"SseDoneEofStopResetAndLimits", SseDoneEofStopResetAndLimits},
}};

} // namespace

int main() {
  for (const auto &[name, test] : kTests) {
    if (!test()) {
      std::printf("FAILED: %.*s\n", static_cast<int>(name.size()), name.data());
      return 1;
    }
  }
  std::printf("http_sse_test: %zu/%zu passed\n", kTests.size(), kTests.size());
  return 0;
}
