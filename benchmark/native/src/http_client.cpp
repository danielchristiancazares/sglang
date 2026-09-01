#include "sglang/benchmark/http_client.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
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
#include <fcntl.h>
#include <netdb.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace sglang::benchmark {
namespace {

constexpr std::size_t kMaximumErrorBytes = 768;
constexpr auto kMaximumTimeout = std::chrono::hours(24);
constexpr std::size_t kReceiveBufferBytes = 64U * 1024U;

[[nodiscard]] char ascii_lower(char value) noexcept {
  if (value >= 'A' && value <= 'Z') {
    return static_cast<char>(value + ('a' - 'A'));
  }
  return value;
}

[[nodiscard]] bool ascii_iequals(std::string_view left,
                                 std::string_view right) noexcept {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (ascii_lower(left[index]) != ascii_lower(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::string_view trim_ows(std::string_view value) noexcept {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
    value.remove_prefix(1);
  }
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
    value.remove_suffix(1);
  }
  return value;
}

[[nodiscard]] bool is_token_character(unsigned char value) noexcept {
  if ((value >= '0' && value <= '9') || (value >= 'A' && value <= 'Z') ||
      (value >= 'a' && value <= 'z')) {
    return true;
  }
  switch (value) {
  case '!':
  case '#':
  case '$':
  case '%':
  case '&':
  case '\'':
  case '*':
  case '+':
  case '-':
  case '.':
  case '^':
  case '_':
  case '`':
  case '|':
  case '~':
    return true;
  default:
    return false;
  }
}

[[nodiscard]] bool valid_header_name(std::string_view name) noexcept {
  return !name.empty() && std::all_of(name.begin(), name.end(), [](char byte) {
    return is_token_character(static_cast<unsigned char>(byte));
  });
}

[[nodiscard]] bool valid_header_value(std::string_view value) noexcept {
  return std::all_of(value.begin(), value.end(), [](char byte) {
    const auto input = static_cast<unsigned char>(byte);
    return input == '\t' || input >= 0x20U;
  });
}

[[nodiscard]] std::string bounded_message(std::string message) {
  if (message.size() > kMaximumErrorBytes) {
    message.resize(kMaximumErrorBytes);
  }
  return message;
}

[[nodiscard]] HttpResult failure(HttpErrorCode code, std::string message,
                                 HttpResponse response = {}) {
  response.completed_at = HttpClock::now();
  return HttpResult{std::move(response),
                    HttpError{code, bounded_message(std::move(message))}};
}

struct ParsedUrl final {
  std::string host;
  std::string service;
  std::string host_header;
  std::string target;
};

[[nodiscard]] bool parse_port(std::string_view text, std::uint16_t &port) {
  if (text.empty()) {
    return false;
  }
  unsigned int value = 0;
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size() ||
      value == 0 || value > 65535U) {
    return false;
  }
  port = static_cast<std::uint16_t>(value);
  return true;
}

[[nodiscard]] bool has_forbidden_url_byte(std::string_view value) noexcept {
  return std::any_of(value.begin(), value.end(), [](char byte) {
    const auto input = static_cast<unsigned char>(byte);
    return input <= 0x20U || input == 0x7fU;
  });
}

[[nodiscard]] bool parse_url(std::string_view url, ParsedUrl &parsed,
                             std::string &error) {
  constexpr std::string_view kScheme = "http://";
  if (url.size() < kScheme.size() ||
      !ascii_iequals(url.substr(0, kScheme.size()), kScheme)) {
    error = "URL must use the plain http:// scheme";
    return false;
  }
  url.remove_prefix(kScheme.size());
  if (url.empty() || has_forbidden_url_byte(url)) {
    error = "HTTP URL contains an empty authority or a forbidden byte";
    return false;
  }

  const std::size_t target_begin = url.find_first_of("/?#");
  const std::string_view authority = url.substr(0, target_begin);
  std::string_view suffix = target_begin == std::string_view::npos
                                ? std::string_view{}
                                : url.substr(target_begin);
  if (authority.empty() || authority.find('@') != std::string_view::npos) {
    error = "HTTP URL authority is empty or contains unsupported user info";
    return false;
  }
  if (suffix.find('#') != std::string_view::npos) {
    error = "HTTP URL fragments are not sent in benchmark requests";
    return false;
  }

  std::string_view host;
  std::string_view port_text;
  bool bracketed = false;
  bool explicit_port = false;
  if (authority.front() == '[') {
    bracketed = true;
    const std::size_t closing = authority.find(']');
    if (closing == std::string_view::npos || closing == 1) {
      error = "HTTP URL has an invalid bracketed IPv6 host";
      return false;
    }
    host = authority.substr(1, closing - 1);
    const std::string_view remainder = authority.substr(closing + 1);
    if (!remainder.empty()) {
      if (remainder.front() != ':' || remainder.size() == 1) {
        error = "HTTP URL has invalid text after its bracketed host";
        return false;
      }
      explicit_port = true;
      port_text = remainder.substr(1);
    }
  } else {
    const std::size_t colon = authority.find(':');
    if (colon != std::string_view::npos) {
      if (authority.find(':', colon + 1) != std::string_view::npos) {
        error = "IPv6 URL hosts must be enclosed in brackets";
        return false;
      }
      host = authority.substr(0, colon);
      port_text = authority.substr(colon + 1);
      explicit_port = true;
    } else {
      host = authority;
    }
  }
  if (host.empty()) {
    error = "HTTP URL host is empty";
    return false;
  }

  std::uint16_t port = 80;
  if (explicit_port && !parse_port(port_text, port)) {
    error = "HTTP URL port must be an integer from 1 through 65535";
    return false;
  }

  parsed.host.assign(host);
  parsed.service = std::to_string(port);
  if (bracketed) {
    parsed.host_header = "[" + parsed.host + "]";
  } else {
    parsed.host_header = parsed.host;
  }
  if (explicit_port || port != 80) {
    parsed.host_header += ":" + parsed.service;
  }
  if (suffix.empty()) {
    parsed.target = "/";
  } else if (suffix.front() == '?') {
    parsed.target = "/";
    parsed.target.append(suffix);
  } else {
    parsed.target.assign(suffix);
  }
  return true;
}

[[nodiscard]] bool parse_decimal_u64(std::string_view text,
                                     std::uint64_t &value) noexcept {
  if (text.empty()) {
    return false;
  }
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

[[nodiscard]] bool build_request_head(const HttpRequest &request,
                                      const ParsedUrl &url, std::string &output,
                                      std::string &error) {
  if (request.method != "GET" && request.method != "POST") {
    error = "HTTP benchmark requests support only GET and POST";
    return false;
  }

  bool has_host = false;
  bool has_content_length = false;
  bool has_connection = false;
  output = request.method + " " + url.target + " HTTP/1.1\r\n";
  for (const HttpHeader &header : request.headers) {
    if (!valid_header_name(header.name) || !valid_header_value(header.value)) {
      error = "caller HTTP header has an invalid name or value";
      return false;
    }
    if (ascii_iequals(header.name, "Transfer-Encoding")) {
      error = "caller Transfer-Encoding is unsupported; Content-Length is "
              "managed by the transport";
      return false;
    }
    if (ascii_iequals(header.name, "Host")) {
      if (has_host || trim_ows(header.value).empty()) {
        error = "caller Host header is duplicated or empty";
        return false;
      }
      has_host = true;
    } else if (ascii_iequals(header.name, "Content-Length")) {
      if (has_content_length) {
        error = "caller Content-Length header is duplicated";
        return false;
      }
      std::uint64_t supplied = 0;
      if (!parse_decimal_u64(trim_ows(header.value), supplied) ||
          supplied != request.body.size()) {
        error = "caller Content-Length does not match the request body";
        return false;
      }
      has_content_length = true;
    } else if (ascii_iequals(header.name, "Connection")) {
      if (has_connection || !ascii_iequals(trim_ows(header.value), "close")) {
        error = "caller Connection header must be the single value close";
        return false;
      }
      has_connection = true;
    }
    output += header.name;
    output += ": ";
    output += header.value;
    output += "\r\n";
  }
  if (!has_host) {
    output += "Host: " + url.host_header + "\r\n";
  }
  if (!has_content_length) {
    output += "Content-Length: " + std::to_string(request.body.size()) + "\r\n";
  }
  if (!has_connection) {
    output += "Connection: close\r\n";
  }
  output += "\r\n";

  if (output.size() > request.max_header_bytes) {
    error = "serialized HTTP request headers exceed the configured byte limit";
    return false;
  }
  return true;
}

#ifdef _WIN32
using NativeSocket = SOCKET;
constexpr NativeSocket kInvalidSocket = INVALID_SOCKET;

[[nodiscard]] int last_socket_error() noexcept { return WSAGetLastError(); }
[[nodiscard]] bool socket_error_interrupted(int code) noexcept {
  return code == WSAEINTR;
}
[[nodiscard]] bool socket_error_would_block(int code) noexcept {
  return code == WSAEWOULDBLOCK || code == WSAEINPROGRESS || code == WSAEINVAL;
}
void close_socket(NativeSocket socket) noexcept { closesocket(socket); }

class WinsockState final {
public:
  WinsockState() noexcept {
    WSADATA data{};
    error_ = WSAStartup(MAKEWORD(2, 2), &data);
  }
  ~WinsockState() {
    if (error_ == 0) {
      WSACleanup();
    }
  }
  [[nodiscard]] int error() const noexcept { return error_; }

private:
  int error_{0};
};

[[nodiscard]] int ensure_socket_runtime() noexcept {
  static WinsockState state;
  return state.error();
}
#else
using NativeSocket = int;
constexpr NativeSocket kInvalidSocket = -1;

[[nodiscard]] int last_socket_error() noexcept { return errno; }
[[nodiscard]] bool socket_error_interrupted(int code) noexcept {
  return code == EINTR;
}
[[nodiscard]] bool socket_error_would_block(int code) noexcept {
  return code == EINPROGRESS || code == EWOULDBLOCK || code == EAGAIN;
}
void close_socket(NativeSocket socket) noexcept { close(socket); }
[[nodiscard]] int ensure_socket_runtime() noexcept { return 0; }
#endif

[[nodiscard]] std::string socket_error_message(std::string_view operation,
                                               int code) {
  return std::string(operation) + " failed with socket error " +
         std::to_string(code);
}

class SocketHandle final {
public:
  SocketHandle() = default;
  explicit SocketHandle(NativeSocket value) noexcept : value_(value) {}
  SocketHandle(const SocketHandle &) = delete;
  SocketHandle &operator=(const SocketHandle &) = delete;
  SocketHandle(SocketHandle &&other) noexcept
      : value_(std::exchange(other.value_, kInvalidSocket)) {}
  SocketHandle &operator=(SocketHandle &&other) noexcept {
    if (this != &other) {
      reset();
      value_ = std::exchange(other.value_, kInvalidSocket);
    }
    return *this;
  }
  ~SocketHandle() { reset(); }

  [[nodiscard]] NativeSocket get() const noexcept { return value_; }
  [[nodiscard]] bool valid() const noexcept { return value_ != kInvalidSocket; }
  void reset(NativeSocket value = kInvalidSocket) noexcept {
    if (valid()) {
      close_socket(value_);
    }
    value_ = value;
  }

private:
  NativeSocket value_{kInvalidSocket};
};

[[nodiscard]] bool set_nonblocking(NativeSocket socket, std::string &error) {
#ifdef _WIN32
  u_long enabled = 1;
  if (ioctlsocket(socket, FIONBIO, &enabled) != 0) {
    error = socket_error_message("ioctlsocket(FIONBIO)", last_socket_error());
    return false;
  }
#else
  const int flags = fcntl(socket, F_GETFL, 0);
  if (flags < 0 || fcntl(socket, F_SETFL, flags | O_NONBLOCK) < 0) {
    error = socket_error_message("fcntl(O_NONBLOCK)", last_socket_error());
    return false;
  }
#ifdef SO_NOSIGPIPE
  const int enabled = 1;
  static_cast<void>(
      setsockopt(socket, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled)));
#endif
#endif
  return true;
}

enum class WaitStatus : unsigned char { kReady, kTimeout, kError };

enum class WaitKind : unsigned char { kRead, kWrite, kConnect };

[[nodiscard]] WaitStatus wait_socket(NativeSocket socket, WaitKind kind,
                                     HttpTimePoint deadline,
                                     std::string &error) {
#ifndef _WIN32
  if (socket < 0 || socket >= FD_SETSIZE) {
    error = "socket descriptor exceeds select()'s FD_SETSIZE limit";
    return WaitStatus::kError;
  }
#endif
  for (;;) {
    const auto now = HttpClock::now();
    if (now >= deadline) {
      return WaitStatus::kTimeout;
    }
    auto remaining =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
    if (remaining.count() <= 0) {
      remaining = std::chrono::microseconds(1);
    }
    timeval timeout{};
    timeout.tv_sec = static_cast<long>(remaining.count() / 1000000);
    timeout.tv_usec = static_cast<long>(remaining.count() % 1000000);

    fd_set read_set;
    fd_set write_set;
    fd_set exception_set;
    FD_ZERO(&read_set);
    FD_ZERO(&write_set);
    FD_ZERO(&exception_set);
    if (kind == WaitKind::kRead) {
      FD_SET(socket, &read_set);
    } else {
      FD_SET(socket, &write_set);
      if (kind == WaitKind::kConnect) {
        FD_SET(socket, &exception_set);
      }
    }
#ifdef _WIN32
    const int selected =
        select(0, kind == WaitKind::kRead ? &read_set : nullptr,
               kind == WaitKind::kRead ? nullptr : &write_set,
               kind == WaitKind::kConnect ? &exception_set : nullptr, &timeout);
#else
    const int selected =
        select(socket + 1, kind == WaitKind::kRead ? &read_set : nullptr,
               kind == WaitKind::kRead ? nullptr : &write_set,
               kind == WaitKind::kConnect ? &exception_set : nullptr, &timeout);
#endif
    if (selected > 0) {
      return WaitStatus::kReady;
    }
    if (selected == 0) {
      return WaitStatus::kTimeout;
    }
    const int code = last_socket_error();
    if (socket_error_interrupted(code)) {
      continue;
    }
    error = socket_error_message("select", code);
    return WaitStatus::kError;
  }
}

[[nodiscard]] WaitStatus wait_connect_socket(NativeSocket socket,
                                             HttpTimePoint deadline,
                                             std::string &error) {
#ifdef _WIN32
  const WSAEVENT event = WSACreateEvent();
  if (event == WSA_INVALID_EVENT) {
    error = socket_error_message("WSACreateEvent", last_socket_error());
    return WaitStatus::kError;
  }
  const auto close_event = [&]() noexcept {
    static_cast<void>(WSAEventSelect(socket, nullptr, 0));
    static_cast<void>(WSACloseEvent(event));
  };
  if (WSAEventSelect(socket, event, FD_CONNECT) == SOCKET_ERROR) {
    error =
        socket_error_message("WSAEventSelect(FD_CONNECT)", last_socket_error());
    close_event();
    return WaitStatus::kError;
  }

  const auto now = HttpClock::now();
  if (now >= deadline) {
    close_event();
    return WaitStatus::kTimeout;
  }
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - now + std::chrono::milliseconds(1));
  const DWORD timeout = static_cast<DWORD>(std::min<std::int64_t>(
      remaining.count(), std::numeric_limits<DWORD>::max()));
  const DWORD waited =
      WSAWaitForMultipleEvents(1, &event, FALSE, timeout, FALSE);
  if (waited == WSA_WAIT_TIMEOUT) {
    close_event();
    return WaitStatus::kTimeout;
  }
  if (waited == WSA_WAIT_FAILED) {
    error =
        socket_error_message("WSAWaitForMultipleEvents", last_socket_error());
    close_event();
    return WaitStatus::kError;
  }

  WSANETWORKEVENTS events{};
  if (WSAEnumNetworkEvents(socket, event, &events) == SOCKET_ERROR) {
    error = socket_error_message("WSAEnumNetworkEvents", last_socket_error());
    close_event();
    return WaitStatus::kError;
  }
  close_event();
  if ((events.lNetworkEvents & FD_CONNECT) == 0) {
    error = "Winsock signaled an unexpected event while connecting";
    return WaitStatus::kError;
  }
  const int connect_error = events.iErrorCode[FD_CONNECT_BIT];
  if (connect_error != 0) {
    error = socket_error_message("connect", connect_error);
    return WaitStatus::kError;
  }
  return WaitStatus::kReady;
#else
  return wait_socket(socket, WaitKind::kConnect, deadline, error);
#endif
}

[[nodiscard]] bool socket_connect_completed(NativeSocket socket,
                                            std::string &error) {
  int socket_error = 0;
#ifdef _WIN32
  int length = sizeof(socket_error);
#else
  socklen_t length = sizeof(socket_error);
#endif
  if (getsockopt(socket, SOL_SOCKET, SO_ERROR,
                 reinterpret_cast<char *>(&socket_error), &length) != 0) {
    error = socket_error_message("getsockopt(SO_ERROR)", last_socket_error());
    return false;
  }
  if (socket_error != 0) {
    error = socket_error_message("connect", socket_error);
    return false;
  }
  return true;
}

struct AddressInfoDeleter final {
  void operator()(addrinfo *value) const noexcept {
    if (value != nullptr) {
      freeaddrinfo(value);
    }
  }
};

[[nodiscard]] bool connect_socket(const ParsedUrl &url,
                                  std::chrono::milliseconds timeout,
                                  SocketHandle &output, HttpErrorCode &code,
                                  std::string &error) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_protocol = IPPROTO_TCP;
  addrinfo *addresses_raw = nullptr;
  const int lookup = getaddrinfo(url.host.c_str(), url.service.c_str(), &hints,
                                 &addresses_raw);
  if (lookup != 0) {
#ifdef _WIN32
    const char *detail = gai_strerrorA(lookup);
#else
    const char *detail = gai_strerror(lookup);
#endif
    code = HttpErrorCode::kNameResolution;
    error = "getaddrinfo failed for " + url.host + ": " +
            (detail == nullptr ? std::to_string(lookup) : std::string(detail));
    return false;
  }
  const std::unique_ptr<addrinfo, AddressInfoDeleter> addresses(addresses_raw);
  const HttpTimePoint deadline = HttpClock::now() + timeout;
  std::string last_error = "no address candidates were returned";
  std::size_t addresses_remaining = 0;
  for (const addrinfo *address = addresses.get(); address != nullptr;
       address = address->ai_next) {
    ++addresses_remaining;
  }

  for (const addrinfo *address = addresses.get(); address != nullptr;
       address = address->ai_next) {
    const std::size_t candidates_including_current = addresses_remaining;
    --addresses_remaining;
    if (HttpClock::now() >= deadline) {
      code = HttpErrorCode::kConnectTimeout;
      error = "HTTP connection timed out while trying resolved addresses";
      return false;
    }
    SocketHandle candidate(::socket(address->ai_family, address->ai_socktype,
                                    address->ai_protocol));
    if (!candidate.valid()) {
      last_error = socket_error_message("socket", last_socket_error());
      continue;
    }
    if (!set_nonblocking(candidate.get(), last_error)) {
      continue;
    }
    const int connected = ::connect(candidate.get(), address->ai_addr,
#ifdef _WIN32
                                    static_cast<int>(address->ai_addrlen)
#else
                                    address->ai_addrlen
#endif
    );
    if (connected == 0) {
      output = std::move(candidate);
      return true;
    }
    const int connect_error = last_socket_error();
    if (!socket_error_would_block(connect_error)) {
      last_error = socket_error_message("connect", connect_error);
      continue;
    }
    const HttpTimePoint now = HttpClock::now();
    HttpTimePoint address_deadline = deadline;
    if (candidates_including_current > 1) {
      auto share = (deadline - now) / static_cast<HttpClock::duration::rep>(
                                          candidates_including_current);
      if (share < std::chrono::milliseconds(1)) {
        share = std::chrono::milliseconds(1);
      }
      address_deadline = std::min(deadline, now + share);
    }
    const WaitStatus waited =
        wait_connect_socket(candidate.get(), address_deadline, last_error);
    if (waited == WaitStatus::kTimeout) {
      // Winsock can retain an asynchronous connect failure in SO_ERROR
      // without placing the socket in select()'s exception set. Probe the
      // completion state before classifying the elapsed wait as a timeout.
      std::string completion_error;
      if (!socket_connect_completed(candidate.get(), completion_error)) {
        last_error = std::move(completion_error);
        continue;
      }
      if (HttpClock::now() >= deadline || candidates_including_current <= 1) {
        code = HttpErrorCode::kConnectTimeout;
        error = "HTTP connection timed out for " + url.host + ':' + url.service;
        return false;
      }
      last_error = "connection attempt timed out for one resolved address";
      continue;
    }
    if (waited == WaitStatus::kError ||
        !socket_connect_completed(candidate.get(), last_error)) {
      continue;
    }
    output = std::move(candidate);
    return true;
  }

  code = HttpErrorCode::kConnect;
  error = "unable to connect to " + url.host + ':' + url.service + ": " +
          last_error;
  return false;
}

[[nodiscard]] bool send_all(NativeSocket socket, std::string_view bytes,
                            HttpTimePoint deadline, HttpErrorCode &code,
                            std::string &error) {
  std::size_t sent = 0;
  while (sent < bytes.size()) {
    const WaitStatus waited =
        wait_socket(socket, WaitKind::kWrite, deadline, error);
    if (waited == WaitStatus::kTimeout) {
      code = HttpErrorCode::kSendTimeout;
      error = "HTTP request send timed out";
      return false;
    }
    if (waited == WaitStatus::kError) {
      code = HttpErrorCode::kSend;
      return false;
    }
    const std::size_t remaining = bytes.size() - sent;
    const int amount = static_cast<int>(
        std::min<std::size_t>(remaining, std::numeric_limits<int>::max()));
#ifdef MSG_NOSIGNAL
    constexpr int kSendFlags = MSG_NOSIGNAL;
#else
    constexpr int kSendFlags = 0;
#endif
    const int result = ::send(socket, bytes.data() + sent, amount, kSendFlags);
    if (result > 0) {
      sent += static_cast<std::size_t>(result);
      continue;
    }
    if (result == 0) {
      code = HttpErrorCode::kSend;
      error = "HTTP socket closed while sending the request";
      return false;
    }
    const int socket_error = last_socket_error();
    if (socket_error_interrupted(socket_error) ||
        socket_error_would_block(socket_error)) {
      continue;
    }
    code = HttpErrorCode::kSend;
    error = socket_error_message("send", socket_error);
    return false;
  }
  return true;
}

enum class ReceiveStatus : unsigned char {
  kData,
  kClosed,
  kTimeout,
  kError,
};

struct ReceiveResult final {
  ReceiveStatus status{ReceiveStatus::kError};
  std::size_t size{0};
  HttpTimePoint received_at{};
  std::string error;
};

[[nodiscard]] ReceiveResult receive_some(NativeSocket socket,
                                         std::span<char> buffer,
                                         std::chrono::milliseconds timeout) {
  const HttpTimePoint deadline = HttpClock::now() + timeout;
  for (;;) {
    std::string wait_error;
    const WaitStatus waited =
        wait_socket(socket, WaitKind::kRead, deadline, wait_error);
    if (waited == WaitStatus::kTimeout) {
      return ReceiveResult{ReceiveStatus::kTimeout, 0, HttpClock::now(), {}};
    }
    if (waited == WaitStatus::kError) {
      return ReceiveResult{ReceiveStatus::kError, 0, HttpClock::now(),
                           std::move(wait_error)};
    }
    const int capacity = static_cast<int>(
        std::min<std::size_t>(buffer.size(), std::numeric_limits<int>::max()));
    const int result = ::recv(socket, buffer.data(), capacity, 0);
    const HttpTimePoint received_at = HttpClock::now();
    if (result > 0) {
      return ReceiveResult{ReceiveStatus::kData,
                           static_cast<std::size_t>(result),
                           received_at,
                           {}};
    }
    if (result == 0) {
      return ReceiveResult{ReceiveStatus::kClosed, 0, received_at, {}};
    }
    const int socket_error = last_socket_error();
    if (socket_error_interrupted(socket_error) ||
        socket_error_would_block(socket_error)) {
      continue;
    }
    return ReceiveResult{ReceiveStatus::kError, 0, received_at,
                         socket_error_message("recv", socket_error)};
  }
}

[[nodiscard]] bool parse_response_head(std::string_view bytes,
                                       HttpResponse &response,
                                       std::string &error) {
  const std::size_t first_end = bytes.find("\r\n");
  if (first_end == std::string_view::npos) {
    error = "HTTP response is missing a complete status line";
    return false;
  }
  const std::string_view status_line = bytes.substr(0, first_end);
  if (!(status_line.starts_with("HTTP/1.1 ") ||
        status_line.starts_with("HTTP/1.0 "))) {
    error = "HTTP response status line has an unsupported protocol version";
    return false;
  }
  const std::string_view status = status_line.substr(9, 3);
  if (status.size() != 3 ||
      !std::all_of(status.begin(), status.end(),
                   [](char value) { return value >= '0' && value <= '9'; })) {
    error = "HTTP response status code is malformed";
    return false;
  }
  response.status_code =
      (status[0] - '0') * 100 + (status[1] - '0') * 10 + (status[2] - '0');
  if (response.status_code < 100 || response.status_code > 599) {
    error = "HTTP response status code is outside the valid range";
    return false;
  }
  if (status_line.size() > 12) {
    if (status_line[12] != ' ') {
      error = "HTTP response status line has invalid text after the code";
      return false;
    }
    response.reason.assign(status_line.substr(13));
    if (!valid_header_value(response.reason)) {
      error = "HTTP response reason phrase contains a forbidden byte";
      return false;
    }
  }

  std::size_t position = first_end + 2;
  while (position < bytes.size()) {
    const std::size_t line_end = bytes.find("\r\n", position);
    const std::size_t end =
        line_end == std::string_view::npos ? bytes.size() : line_end;
    const std::string_view line = bytes.substr(position, end - position);
    if (line.empty()) {
      position = end + (line_end == std::string_view::npos ? 0 : 2);
      continue;
    }
    if (line.front() == ' ' || line.front() == '\t') {
      error = "HTTP obsolete folded response headers are unsupported";
      return false;
    }
    const std::size_t colon = line.find(':');
    if (colon == std::string_view::npos) {
      error = "HTTP response header is missing a colon";
      return false;
    }
    const std::string_view name = line.substr(0, colon);
    const std::string_view value = trim_ows(line.substr(colon + 1));
    if (!valid_header_name(name) || !valid_header_value(value)) {
      error = "HTTP response contains an invalid header name or value";
      return false;
    }
    response.headers.push_back(
        HttpHeader{std::string(name), std::string(value)});
    if (line_end == std::string_view::npos) {
      break;
    }
    position = line_end + 2;
  }
  return true;
}

enum class BodyMode : unsigned char { kNone, kContentLength, kChunked, kClose };

struct BodyFraming final {
  BodyMode mode{BodyMode::kClose};
  std::uint64_t content_length{0};
};

[[nodiscard]] bool split_comma_tokens(std::string_view value,
                                      std::vector<std::string_view> &tokens) {
  std::size_t begin = 0;
  for (;;) {
    const std::size_t comma = value.find(',', begin);
    const std::string_view token = trim_ows(value.substr(
        begin, comma == std::string_view::npos ? value.size() - begin
                                               : comma - begin));
    if (token.empty()) {
      return false;
    }
    tokens.push_back(token);
    if (comma == std::string_view::npos) {
      return true;
    }
    begin = comma + 1;
  }
}

[[nodiscard]] bool determine_body_framing(const HttpResponse &response,
                                          BodyFraming &framing,
                                          std::string &error) {
  if ((response.status_code >= 100 && response.status_code < 200) ||
      response.status_code == 204 || response.status_code == 304) {
    framing.mode = BodyMode::kNone;
    return true;
  }

  std::vector<std::string_view> transfer_tokens;
  bool has_content_length = false;
  std::uint64_t content_length = 0;
  for (const HttpHeader &header : response.headers) {
    if (ascii_iequals(header.name, "Transfer-Encoding")) {
      if (!split_comma_tokens(header.value, transfer_tokens)) {
        error = "HTTP Transfer-Encoding contains an empty coding";
        return false;
      }
    } else if (ascii_iequals(header.name, "Content-Length")) {
      std::uint64_t current = 0;
      if (!parse_decimal_u64(trim_ows(header.value), current)) {
        error = "HTTP Content-Length is not a valid unsigned integer";
        return false;
      }
      if (has_content_length && current != content_length) {
        error = "HTTP response contains conflicting Content-Length values";
        return false;
      }
      has_content_length = true;
      content_length = current;
    }
  }

  if (!transfer_tokens.empty()) {
    if (has_content_length) {
      error =
          "HTTP response contains both Transfer-Encoding and Content-Length";
      return false;
    }
    if (transfer_tokens.size() != 1 ||
        !ascii_iequals(transfer_tokens.front(), "chunked")) {
      error = "HTTP response uses an unsupported transfer coding";
      return false;
    }
    framing.mode = BodyMode::kChunked;
    return true;
  }
  if (has_content_length) {
    framing.mode = BodyMode::kContentLength;
    framing.content_length = content_length;
    return true;
  }
  framing.mode = BodyMode::kClose;
  return true;
}

enum class DeliveryStatus : unsigned char { kContinue, kStopped, kError };

[[nodiscard]] DeliveryStatus
deliver_body(std::string_view bytes, HttpTimePoint received_at,
             const HttpRequest &request, const HttpBodyChunkCallback &callback,
             HttpResponse &response, HttpError &error) {
  if (bytes.empty()) {
    return DeliveryStatus::kContinue;
  }
  if (bytes.size() >
      request.max_body_bytes - static_cast<std::size_t>(response.body_bytes)) {
    error = HttpError{HttpErrorCode::kBodyLimit,
                      "HTTP response body exceeds the configured byte limit"};
    return DeliveryStatus::kError;
  }
  if (request.capture_body) {
    response.body.append(bytes);
  }
  response.body_bytes += bytes.size();
  if (callback) {
    try {
      if (!callback(bytes, received_at)) {
        response.stopped_early = true;
        return DeliveryStatus::kStopped;
      }
    } catch (const std::exception &exception) {
      error =
          HttpError{HttpErrorCode::kCallback,
                    bounded_message(std::string("HTTP body callback failed: ") +
                                    exception.what())};
      return DeliveryStatus::kError;
    } catch (...) {
      error = HttpError{HttpErrorCode::kCallback,
                        "HTTP body callback failed with an unknown exception"};
      return DeliveryStatus::kError;
    }
  }
  return DeliveryStatus::kContinue;
}

class ChunkedDecoder final {
public:
  ChunkedDecoder(std::size_t maximum_body_bytes,
                 std::size_t maximum_trailer_bytes)
      : maximum_body_bytes_(maximum_body_bytes),
        maximum_trailer_bytes_(maximum_trailer_bytes) {}

  enum class Status : unsigned char {
    kNeedInput,
    kComplete,
    kStopped,
    kError,
  };

  template <typename Deliver>
  [[nodiscard]] Status feed(std::string_view bytes, HttpTimePoint received_at,
                            Deliver &&deliver) {
    std::size_t position = 0;
    while (position < bytes.size()) {
      if (state_ == State::kComplete) {
        return Status::kComplete;
      }
      if (state_ == State::kData) {
        const std::size_t available = bytes.size() - position;
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(available, chunk_remaining_));
        const DeliveryStatus delivered =
            deliver(bytes.substr(position, count), received_at);
        position += count;
        chunk_remaining_ -= count;
        decoded_bytes_ += count;
        if (delivered == DeliveryStatus::kStopped) {
          return Status::kStopped;
        }
        if (delivered == DeliveryStatus::kError) {
          return Status::kError;
        }
        if (chunk_remaining_ == 0) {
          state_ = State::kDataCr;
        }
        continue;
      }
      if (state_ == State::kDataCr) {
        if (bytes[position++] != '\r') {
          return fail("HTTP chunk data is not followed by CRLF");
        }
        state_ = State::kDataLf;
        continue;
      }
      if (state_ == State::kDataLf) {
        if (bytes[position++] != '\n') {
          return fail("HTTP chunk data is not followed by CRLF");
        }
        state_ = State::kSizeLine;
        continue;
      }

      const char byte = bytes[position++];
      if (line_saw_cr_) {
        line_saw_cr_ = false;
        if (byte != '\n') {
          return fail("HTTP chunk metadata line has a bare carriage return");
        }
        const Status completed = complete_metadata_line();
        if (completed != Status::kNeedInput) {
          return completed;
        }
        continue;
      }
      if (byte == '\r') {
        line_saw_cr_ = true;
        continue;
      }
      if (byte == '\n') {
        return fail("HTTP chunk metadata line uses LF without CR");
      }
      const std::size_t limit = state_ == State::kTrailers
                                    ? maximum_trailer_bytes_
                                    : maximum_trailer_bytes_;
      if (line_.size() >= limit) {
        return fail("HTTP chunk metadata exceeds the configured header limit");
      }
      line_.push_back(byte);
    }
    return state_ == State::kComplete ? Status::kComplete : Status::kNeedInput;
  }

  [[nodiscard]] bool complete() const noexcept {
    return state_ == State::kComplete;
  }
  [[nodiscard]] std::string_view error() const noexcept { return error_; }
  [[nodiscard]] HttpErrorCode error_code() const noexcept {
    return error_code_;
  }

private:
  enum class State : unsigned char {
    kSizeLine,
    kData,
    kDataCr,
    kDataLf,
    kTrailers,
    kComplete,
  };

  [[nodiscard]] Status
  fail(std::string_view message,
       HttpErrorCode code = HttpErrorCode::kMalformedResponse) {
    error_code_ = code;
    error_.assign(message.substr(0, kMaximumErrorBytes));
    return Status::kError;
  }

  [[nodiscard]] Status complete_metadata_line() {
    if (state_ == State::kSizeLine) {
      const std::size_t semicolon = line_.find(';');
      const std::string_view size_text =
          std::string_view(line_).substr(0, semicolon);
      if (size_text.empty()) {
        return fail("HTTP chunk size is empty");
      }
      std::uint64_t size = 0;
      for (char digit : size_text) {
        unsigned int value = 0;
        if (digit >= '0' && digit <= '9') {
          value = static_cast<unsigned int>(digit - '0');
        } else if (digit >= 'a' && digit <= 'f') {
          value = static_cast<unsigned int>(digit - 'a' + 10);
        } else if (digit >= 'A' && digit <= 'F') {
          value = static_cast<unsigned int>(digit - 'A' + 10);
        } else {
          return fail("HTTP chunk size contains a non-hexadecimal byte");
        }
        if (size > (std::numeric_limits<std::uint64_t>::max() - value) / 16U) {
          return fail("HTTP chunk size overflows 64 bits");
        }
        size = size * 16U + value;
      }
      if (semicolon != std::string::npos) {
        const std::string_view extension =
            std::string_view(line_).substr(semicolon + 1);
        if (extension.empty() || !valid_header_value(extension)) {
          return fail("HTTP chunk extension contains an invalid byte");
        }
      }
      line_.clear();
      if (size > maximum_body_bytes_ - decoded_bytes_) {
        return fail("HTTP chunked body exceeds the configured byte limit",
                    HttpErrorCode::kBodyLimit);
      }
      if (size == 0) {
        state_ = State::kTrailers;
      } else {
        chunk_remaining_ = size;
        state_ = State::kData;
      }
      return Status::kNeedInput;
    }

    if (state_ == State::kTrailers) {
      trailer_bytes_ += line_.size() + 2;
      if (trailer_bytes_ > maximum_trailer_bytes_) {
        return fail("HTTP chunk trailers exceed the configured header limit",
                    HttpErrorCode::kHeaderLimit);
      }
      if (line_.empty()) {
        state_ = State::kComplete;
        return Status::kComplete;
      }
      if (line_.front() == ' ' || line_.front() == '\t') {
        return fail("HTTP chunk trailer uses obsolete line folding");
      }
      const std::size_t colon = line_.find(':');
      if (colon == std::string::npos ||
          !valid_header_name(std::string_view(line_).substr(0, colon)) ||
          !valid_header_value(
              trim_ows(std::string_view(line_).substr(colon + 1)))) {
        return fail("HTTP chunk trailer has an invalid name or value");
      }
      const std::string_view name = std::string_view(line_).substr(0, colon);
      if (ascii_iequals(name, "Content-Length") ||
          ascii_iequals(name, "Transfer-Encoding") ||
          ascii_iequals(name, "Host") || ascii_iequals(name, "Trailer")) {
        return fail("HTTP chunk trailer attempts to change message framing");
      }
      line_.clear();
      return Status::kNeedInput;
    }
    return fail("HTTP chunk decoder reached an invalid metadata state");
  }

  State state_{State::kSizeLine};
  std::size_t maximum_body_bytes_;
  std::size_t maximum_trailer_bytes_;
  std::uint64_t decoded_bytes_{0};
  std::uint64_t chunk_remaining_{0};
  std::size_t trailer_bytes_{0};
  std::string line_;
  std::string error_;
  HttpErrorCode error_code_{HttpErrorCode::kMalformedResponse};
  bool line_saw_cr_{false};
};

[[nodiscard]] bool valid_timeout(std::chrono::milliseconds timeout) noexcept {
  return timeout.count() > 0 && timeout <= kMaximumTimeout;
}

} // namespace

std::string_view HttpResponse::header(std::string_view name) const noexcept {
  for (const HttpHeader &item : headers) {
    if (ascii_iequals(item.name, name)) {
      return item.value;
    }
  }
  return {};
}

HttpResult
SocketHttpTransport::perform(const HttpRequest &request,
                             const HttpBodyChunkCallback &on_body_chunk) {
  HttpResponse response;
  response.request_started_at = HttpClock::now();
  if (!valid_timeout(request.connect_timeout) ||
      !valid_timeout(request.io_timeout)) {
    return failure(HttpErrorCode::kInvalidRequest,
                   "HTTP connect and I/O timeouts must be within (0, 24 hours]",
                   std::move(response));
  }
  if (request.max_header_bytes < 16 || request.max_body_bytes == 0) {
    return failure(
        HttpErrorCode::kInvalidRequest,
        "HTTP header and body byte limits must be positive and usable",
        std::move(response));
  }

  ParsedUrl parsed_url;
  std::string detail;
  if (!parse_url(request.url, parsed_url, detail)) {
    return failure(HttpErrorCode::kInvalidUrl, std::move(detail),
                   std::move(response));
  }
  std::string request_head;
  if (!build_request_head(request, parsed_url, request_head, detail)) {
    return failure(HttpErrorCode::kInvalidRequest, std::move(detail),
                   std::move(response));
  }

  const int runtime_error = ensure_socket_runtime();
  if (runtime_error != 0) {
    return failure(
        HttpErrorCode::kSocket,
        socket_error_message("socket runtime initialization", runtime_error),
        std::move(response));
  }
  SocketHandle socket;
  HttpErrorCode operation_code = HttpErrorCode::kConnect;
  if (!connect_socket(parsed_url, request.connect_timeout, socket,
                      operation_code, detail)) {
    return failure(operation_code, std::move(detail), std::move(response));
  }

  const HttpTimePoint send_deadline = HttpClock::now() + request.io_timeout;
  if (!send_all(socket.get(), request_head, send_deadline, operation_code,
                detail) ||
      !send_all(socket.get(), request.body, send_deadline, operation_code,
                detail)) {
    return failure(operation_code, std::move(detail), std::move(response));
  }

  std::vector<char> receive_buffer(kReceiveBufferBytes);
  std::string header_buffer;
  std::string initial_body;
  HttpTimePoint initial_body_at{};
  bool final_headers = false;
  while (!final_headers) {
    ReceiveResult received =
        receive_some(socket.get(), receive_buffer, request.io_timeout);
    if (received.status == ReceiveStatus::kTimeout) {
      return failure(HttpErrorCode::kReceiveTimeout,
                     "HTTP response header receive timed out",
                     std::move(response));
    }
    if (received.status == ReceiveStatus::kError) {
      return failure(HttpErrorCode::kReceive, std::move(received.error),
                     std::move(response));
    }
    if (received.status == ReceiveStatus::kClosed) {
      return failure(HttpErrorCode::kMalformedResponse,
                     "HTTP connection closed before final response headers",
                     std::move(response));
    }
    header_buffer.append(receive_buffer.data(), received.size);

    for (;;) {
      const std::size_t header_end = header_buffer.find("\r\n\r\n");
      if (header_end == std::string::npos) {
        if (header_buffer.size() > request.max_header_bytes) {
          return failure(
              HttpErrorCode::kHeaderLimit,
              "HTTP response headers exceed the configured byte limit",
              std::move(response));
        }
        break;
      }
      if (header_end + 4 > request.max_header_bytes) {
        return failure(HttpErrorCode::kHeaderLimit,
                       "HTTP response headers exceed the configured byte limit",
                       std::move(response));
      }
      HttpResponse parsed_response;
      parsed_response.request_started_at = response.request_started_at;
      if (!parse_response_head(
              std::string_view(header_buffer).substr(0, header_end),
              parsed_response, detail)) {
        return failure(HttpErrorCode::kMalformedResponse, std::move(detail),
                       std::move(response));
      }
      header_buffer.erase(0, header_end + 4);
      if (parsed_response.status_code >= 100 &&
          parsed_response.status_code < 200 &&
          parsed_response.status_code != 101) {
        continue;
      }
      response = std::move(parsed_response);
      response.headers_completed_at = received.received_at;
      initial_body = std::move(header_buffer);
      initial_body_at = received.received_at;
      final_headers = true;
      break;
    }
  }

  BodyFraming framing;
  if (!determine_body_framing(response, framing, detail)) {
    return failure(HttpErrorCode::kMalformedResponse, std::move(detail),
                   std::move(response));
  }
  if (framing.mode == BodyMode::kContentLength &&
      framing.content_length > request.max_body_bytes) {
    return failure(HttpErrorCode::kBodyLimit,
                   "HTTP Content-Length exceeds the configured body limit",
                   std::move(response));
  }

  HttpError delivery_error;
  const auto deliver = [&](std::string_view bytes, HttpTimePoint received_at) {
    return deliver_body(bytes, received_at, request, on_body_chunk, response,
                        delivery_error);
  };
  const auto stopped_result = [&]() {
    response.completed_at = HttpClock::now();
    return HttpResult{std::move(response), {}};
  };
  const auto delivery_failure = [&]() {
    return failure(delivery_error.code, std::move(delivery_error.message),
                   std::move(response));
  };

  if (framing.mode == BodyMode::kNone) {
    response.body_complete = true;
    response.completed_at = HttpClock::now();
    return HttpResult{std::move(response), {}};
  }

  if (framing.mode == BodyMode::kContentLength) {
    std::uint64_t remaining = framing.content_length;
    if (!initial_body.empty() && remaining != 0) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(initial_body.size(), remaining));
      const DeliveryStatus status = deliver(
          std::string_view(initial_body).substr(0, count), initial_body_at);
      remaining -= count;
      if (status == DeliveryStatus::kStopped) {
        return stopped_result();
      }
      if (status == DeliveryStatus::kError) {
        return delivery_failure();
      }
    }
    while (remaining != 0) {
      ReceiveResult received =
          receive_some(socket.get(), receive_buffer, request.io_timeout);
      if (received.status == ReceiveStatus::kTimeout) {
        return failure(HttpErrorCode::kReceiveTimeout,
                       "HTTP response body receive timed out",
                       std::move(response));
      }
      if (received.status == ReceiveStatus::kError) {
        return failure(HttpErrorCode::kReceive, std::move(received.error),
                       std::move(response));
      }
      if (received.status == ReceiveStatus::kClosed) {
        return failure(
            HttpErrorCode::kTruncatedBody,
            "HTTP connection closed before Content-Length bytes arrived",
            std::move(response));
      }
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(received.size, remaining));
      const DeliveryStatus status = deliver(
          std::string_view(receive_buffer.data(), count), received.received_at);
      remaining -= count;
      if (status == DeliveryStatus::kStopped) {
        return stopped_result();
      }
      if (status == DeliveryStatus::kError) {
        return delivery_failure();
      }
    }
    response.body_complete = true;
    response.completed_at = HttpClock::now();
    return HttpResult{std::move(response), {}};
  }

  if (framing.mode == BodyMode::kChunked) {
    ChunkedDecoder decoder(request.max_body_bytes, request.max_header_bytes);
    if (!initial_body.empty()) {
      const ChunkedDecoder::Status status =
          decoder.feed(initial_body, initial_body_at, deliver);
      if (status == ChunkedDecoder::Status::kStopped) {
        return stopped_result();
      }
      if (status == ChunkedDecoder::Status::kError) {
        if (delivery_error.code != HttpErrorCode::kNone) {
          return delivery_failure();
        }
        return failure(decoder.error_code(), std::string(decoder.error()),
                       std::move(response));
      }
    }
    while (!decoder.complete()) {
      ReceiveResult received =
          receive_some(socket.get(), receive_buffer, request.io_timeout);
      if (received.status == ReceiveStatus::kTimeout) {
        return failure(HttpErrorCode::kReceiveTimeout,
                       "HTTP chunked body receive timed out",
                       std::move(response));
      }
      if (received.status == ReceiveStatus::kError) {
        return failure(HttpErrorCode::kReceive, std::move(received.error),
                       std::move(response));
      }
      if (received.status == ReceiveStatus::kClosed) {
        return failure(HttpErrorCode::kTruncatedBody,
                       "HTTP connection closed before the final chunk trailer",
                       std::move(response));
      }
      const ChunkedDecoder::Status status =
          decoder.feed(std::string_view(receive_buffer.data(), received.size),
                       received.received_at, deliver);
      if (status == ChunkedDecoder::Status::kStopped) {
        return stopped_result();
      }
      if (status == ChunkedDecoder::Status::kError) {
        if (delivery_error.code != HttpErrorCode::kNone) {
          return delivery_failure();
        }
        return failure(decoder.error_code(), std::string(decoder.error()),
                       std::move(response));
      }
    }
    response.body_complete = true;
    response.completed_at = HttpClock::now();
    return HttpResult{std::move(response), {}};
  }

  if (!initial_body.empty()) {
    const DeliveryStatus status = deliver(initial_body, initial_body_at);
    if (status == DeliveryStatus::kStopped) {
      return stopped_result();
    }
    if (status == DeliveryStatus::kError) {
      return delivery_failure();
    }
  }
  for (;;) {
    ReceiveResult received =
        receive_some(socket.get(), receive_buffer, request.io_timeout);
    if (received.status == ReceiveStatus::kTimeout) {
      return failure(HttpErrorCode::kReceiveTimeout,
                     "close-delimited HTTP body receive timed out",
                     std::move(response));
    }
    if (received.status == ReceiveStatus::kError) {
      return failure(HttpErrorCode::kReceive, std::move(received.error),
                     std::move(response));
    }
    if (received.status == ReceiveStatus::kClosed) {
      response.body_complete = true;
      response.completed_at = received.received_at;
      return HttpResult{std::move(response), {}};
    }
    const DeliveryStatus status =
        deliver(std::string_view(receive_buffer.data(), received.size),
                received.received_at);
    if (status == DeliveryStatus::kStopped) {
      return stopped_result();
    }
    if (status == DeliveryStatus::kError) {
      return delivery_failure();
    }
  }
}

std::string_view http_error_code_name(HttpErrorCode code) noexcept {
  switch (code) {
  case HttpErrorCode::kNone:
    return "none";
  case HttpErrorCode::kInvalidRequest:
    return "invalid_request";
  case HttpErrorCode::kInvalidUrl:
    return "invalid_url";
  case HttpErrorCode::kNameResolution:
    return "name_resolution";
  case HttpErrorCode::kSocket:
    return "socket";
  case HttpErrorCode::kConnectTimeout:
    return "connect_timeout";
  case HttpErrorCode::kConnect:
    return "connect";
  case HttpErrorCode::kSendTimeout:
    return "send_timeout";
  case HttpErrorCode::kSend:
    return "send";
  case HttpErrorCode::kReceiveTimeout:
    return "receive_timeout";
  case HttpErrorCode::kReceive:
    return "receive";
  case HttpErrorCode::kMalformedResponse:
    return "malformed_response";
  case HttpErrorCode::kHeaderLimit:
    return "header_limit";
  case HttpErrorCode::kBodyLimit:
    return "body_limit";
  case HttpErrorCode::kTruncatedBody:
    return "truncated_body";
  case HttpErrorCode::kCallback:
    return "callback";
  }
  return "unknown";
}

} // namespace sglang::benchmark
