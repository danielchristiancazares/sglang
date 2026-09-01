#include "sglang/benchmark/json.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

namespace sglang::benchmark {
namespace {

[[noreturn]] void throw_utf8_error(std::size_t offset) {
  throw std::invalid_argument("malformed UTF-8 at byte " +
                              std::to_string(offset));
}

bool is_continuation(std::uint8_t byte) noexcept {
  return (byte & 0xc0U) == 0x80U;
}

std::string json_error_message(std::size_t offset, std::string_view message) {
  return "JSON parse error at byte " + std::to_string(offset) + ": " +
         std::string(message);
}

void append_utf8(std::uint32_t code_point, std::string &output) {
  if (code_point <= 0x7fU) {
    output.push_back(static_cast<char>(code_point));
  } else if (code_point <= 0x7ffU) {
    output.push_back(static_cast<char>(0xc0U | (code_point >> 6U)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
  } else if (code_point <= 0xffffU) {
    output.push_back(static_cast<char>(0xe0U | (code_point >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
  } else {
    output.push_back(static_cast<char>(0xf0U | (code_point >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 12U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | ((code_point >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (code_point & 0x3fU)));
  }
}

class Parser {
public:
  explicit Parser(std::string_view input) : input_(input) {}

  JsonValue parse_document() {
    skip_whitespace();
    JsonValue result = parse_value(0);
    skip_whitespace();
    if (position_ != input_.size()) {
      fail("trailing data");
    }
    return result;
  }

private:
  static constexpr std::size_t kMaximumDepth = 512;

  [[noreturn]] void fail(std::string_view message) const {
    throw JsonError(position_, std::string(message));
  }

  void skip_whitespace() noexcept {
    while (position_ < input_.size()) {
      const char c = input_[position_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
        break;
      }
      ++position_;
    }
  }

  bool consume(char expected) noexcept {
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  JsonValue parse_value(std::size_t depth) {
    if (depth > kMaximumDepth) {
      fail("nesting depth exceeds 512");
    }
    if (position_ == input_.size()) {
      fail("expected a value");
    }

    switch (input_[position_]) {
    case 'n':
      consume_literal("null");
      return nullptr;
    case 't':
      consume_literal("true");
      return true;
    case 'f':
      consume_literal("false");
      return false;
    case '"':
      return JsonValue(parse_string());
    case '[':
      return parse_array(depth);
    case '{':
      return parse_object(depth);
    default:
      if (input_[position_] == '-' ||
          (input_[position_] >= '0' && input_[position_] <= '9')) {
        return parse_number();
      }
      fail("expected a value");
    }
  }

  void consume_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      fail("invalid literal");
    }
    position_ += literal.size();
  }

  std::uint16_t parse_hex_quad() {
    if (input_.size() - position_ < 4) {
      fail("incomplete Unicode escape");
    }
    std::uint16_t value = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = input_[position_++];
      value = static_cast<std::uint16_t>(value << 4U);
      if (c >= '0' && c <= '9') {
        value = static_cast<std::uint16_t>(value | (c - '0'));
      } else if (c >= 'a' && c <= 'f') {
        value = static_cast<std::uint16_t>(value | (c - 'a' + 10));
      } else if (c >= 'A' && c <= 'F') {
        value = static_cast<std::uint16_t>(value | (c - 'A' + 10));
      } else {
        fail("invalid hexadecimal digit in Unicode escape");
      }
    }
    return value;
  }

  void parse_unicode_escape(std::string &output) {
    const std::uint16_t first = parse_hex_quad();
    if (first >= 0xd800U && first <= 0xdbffU) {
      if (input_.size() - position_ < 2 || input_[position_] != '\\' ||
          input_[position_ + 1] != 'u') {
        fail("high surrogate requires a low surrogate");
      }
      position_ += 2;
      const std::uint16_t second = parse_hex_quad();
      if (second < 0xdc00U || second > 0xdfffU) {
        fail("invalid low surrogate");
      }
      const std::uint32_t code_point =
          0x10000U + ((static_cast<std::uint32_t>(first) - 0xd800U) << 10U) +
          (static_cast<std::uint32_t>(second) - 0xdc00U);
      append_utf8(code_point, output);
      return;
    }
    if (first >= 0xdc00U && first <= 0xdfffU) {
      fail("unpaired low surrogate");
    }
    append_utf8(first, output);
  }

  std::string parse_string() {
    const std::size_t string_offset = position_;
    ++position_; // Opening quote.
    std::string output;

    while (position_ < input_.size()) {
      const auto byte = static_cast<unsigned char>(input_[position_++]);
      if (byte == static_cast<unsigned char>('"')) {
        try {
          static_cast<void>(utf8_code_point_count(output));
        } catch (const std::invalid_argument &) {
          throw JsonError(string_offset, "string contains malformed UTF-8");
        }
        return output;
      }
      if (byte < 0x20U) {
        fail("unescaped control character in string");
      }
      if (byte != static_cast<unsigned char>('\\')) {
        output.push_back(static_cast<char>(byte));
        continue;
      }
      if (position_ == input_.size()) {
        fail("incomplete string escape");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
      case '"':
        output.push_back('"');
        break;
      case '\\':
        output.push_back('\\');
        break;
      case '/':
        output.push_back('/');
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      case 'u':
        parse_unicode_escape(output);
        break;
      default:
        fail("invalid string escape");
      }
    }
    fail("unterminated string");
  }

  JsonValue parse_number() {
    const std::size_t start = position_;
    consume('-');
    if (position_ == input_.size()) {
      fail("incomplete number");
    }

    if (consume('0')) {
      if (position_ < input_.size() && input_[position_] >= '0' &&
          input_[position_] <= '9') {
        fail("leading zero in number");
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        fail("expected digit in number");
      }
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }

    bool is_integer = true;
    if (consume('.')) {
      is_integer = false;
      const std::size_t fraction_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction_start) {
        fail("fraction requires a digit");
      }
    }

    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      is_integer = false;
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_start = position_;
      while (position_ < input_.size() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent_start) {
        fail("exponent requires a digit");
      }
    }

    const std::string_view token = input_.substr(start, position_ - start);
    if (is_integer) {
      std::int64_t value = 0;
      const auto result =
          std::from_chars(token.data(), token.data() + token.size(), value);
      if (result.ec == std::errc::result_out_of_range) {
        fail("integer exceeds int64 range");
      }
      if (result.ec != std::errc{} ||
          result.ptr != token.data() + token.size()) {
        fail("invalid integer");
      }
      return JsonValue(value);
    }

    double value = 0.0;
    const auto result =
        std::from_chars(token.data(), token.data() + token.size(), value,
                        std::chars_format::general);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size() ||
        !std::isfinite(value)) {
      fail("floating-point number is outside the finite double range");
    }
    return JsonValue(value);
  }

  JsonValue parse_array(std::size_t depth) {
    ++position_; // Opening bracket.
    skip_whitespace();
    JsonValue::array values;
    if (consume(']')) {
      return JsonValue(std::move(values));
    }

    while (true) {
      values.push_back(parse_value(depth + 1));
      skip_whitespace();
      if (consume(']')) {
        return JsonValue(std::move(values));
      }
      if (!consume(',')) {
        fail("expected ',' or ']' in array");
      }
      skip_whitespace();
    }
  }

  JsonValue parse_object(std::size_t depth) {
    ++position_; // Opening brace.
    skip_whitespace();
    JsonValue::object values;
    if (consume('}')) {
      return JsonValue(std::move(values));
    }

    while (true) {
      if (position_ == input_.size() || input_[position_] != '"') {
        fail("expected a string key");
      }
      std::string key = parse_string();
      if (values.find(key) != values.end()) {
        fail("duplicate object key");
      }
      skip_whitespace();
      if (!consume(':')) {
        fail("expected ':' after object key");
      }
      skip_whitespace();
      JsonValue value = parse_value(depth + 1);
      values.emplace(std::move(key), std::move(value));
      skip_whitespace();
      if (consume('}')) {
        return JsonValue(std::move(values));
      }
      if (!consume(',')) {
        fail("expected ',' or '}' in object");
      }
      skip_whitespace();
    }
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

void append_escaped_string(std::string_view value, std::string &output) {
  static_cast<void>(utf8_code_point_count(value));
  constexpr char kHex[] = "0123456789abcdef";
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
    case '"':
      output += "\\\"";
      break;
    case '\\':
      output += "\\\\";
      break;
    case '\b':
      output += "\\b";
      break;
    case '\f':
      output += "\\f";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      if (byte < 0x20U) {
        output += "\\u00";
        output.push_back(kHex[byte >> 4U]);
        output.push_back(kHex[byte & 0x0fU]);
      } else {
        output.push_back(static_cast<char>(byte));
      }
    }
  }
  output.push_back('"');
}

void append_json(const JsonValue &value, std::string &output) {
  if (value.is_null()) {
    output += "null";
  } else if (value.is_bool()) {
    output += value.as_bool() ? "true" : "false";
  } else if (value.is_int()) {
    std::array<char, 32> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value.as_int());
    if (result.ec != std::errc{}) {
      throw std::runtime_error("failed to serialize JSON integer");
    }
    output.append(buffer.data(), result.ptr);
  } else if (value.is_double()) {
    const double number = value.as_double();
    if (!std::isfinite(number)) {
      throw std::domain_error("JSON double must be finite");
    }
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), number,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (result.ec != std::errc{}) {
      throw std::runtime_error("failed to serialize JSON double");
    }
    const std::string_view rendered(buffer.data(), result.ptr);
    output.append(rendered);
    if (rendered.find_first_of(".eE") == std::string_view::npos) {
      output += ".0";
    }
  } else if (value.is_string()) {
    append_escaped_string(value.as_string(), output);
  } else if (value.is_array()) {
    output.push_back('[');
    bool first = true;
    for (const JsonValue &element : value.as_array()) {
      if (!first) {
        output.push_back(',');
      }
      first = false;
      append_json(element, output);
    }
    output.push_back(']');
  } else {
    output.push_back('{');
    bool first = true;
    for (const auto &[key, element] : value.as_object()) {
      if (!first) {
        output.push_back(',');
      }
      first = false;
      append_escaped_string(key, output);
      output.push_back(':');
      append_json(element, output);
    }
    output.push_back('}');
  }
}

template <typename T>
const T &require_type(const JsonValue::storage &storage,
                      std::string_view name) {
  const T *value = std::get_if<T>(&storage);
  if (value == nullptr) {
    throw std::logic_error("JSON value is not " + std::string(name));
  }
  return *value;
}

template <typename T>
T &require_type(JsonValue::storage &storage, std::string_view name) {
  T *value = std::get_if<T>(&storage);
  if (value == nullptr) {
    throw std::logic_error("JSON value is not " + std::string(name));
  }
  return *value;
}

} // namespace

std::size_t utf8_code_point_count(std::string_view text) {
  std::size_t count = 0;
  std::size_t position = 0;
  while (position < text.size()) {
    const auto first = static_cast<std::uint8_t>(text[position]);
    if (first <= 0x7fU) {
      ++position;
    } else if (first >= 0xc2U && first <= 0xdfU) {
      if (text.size() - position < 2 ||
          !is_continuation(static_cast<std::uint8_t>(text[position + 1]))) {
        throw_utf8_error(position);
      }
      position += 2;
    } else if (first >= 0xe0U && first <= 0xefU) {
      if (text.size() - position < 3) {
        throw_utf8_error(position);
      }
      const auto second = static_cast<std::uint8_t>(text[position + 1]);
      const auto third = static_cast<std::uint8_t>(text[position + 2]);
      const bool valid_second =
          (first == 0xe0U && second >= 0xa0U && second <= 0xbfU) ||
          (first == 0xedU && second >= 0x80U && second <= 0x9fU) ||
          ((first >= 0xe1U && first <= 0xecU) && is_continuation(second)) ||
          ((first >= 0xeeU && first <= 0xefU) && is_continuation(second));
      if (!valid_second || !is_continuation(third)) {
        throw_utf8_error(position);
      }
      position += 3;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      if (text.size() - position < 4) {
        throw_utf8_error(position);
      }
      const auto second = static_cast<std::uint8_t>(text[position + 1]);
      const auto third = static_cast<std::uint8_t>(text[position + 2]);
      const auto fourth = static_cast<std::uint8_t>(text[position + 3]);
      const bool valid_second =
          (first == 0xf0U && second >= 0x90U && second <= 0xbfU) ||
          ((first >= 0xf1U && first <= 0xf3U) && is_continuation(second)) ||
          (first == 0xf4U && second >= 0x80U && second <= 0x8fU);
      if (!valid_second || !is_continuation(third) ||
          !is_continuation(fourth)) {
        throw_utf8_error(position);
      }
      position += 4;
    } else {
      throw_utf8_error(position);
    }
    ++count;
  }
  return count;
}

JsonError::JsonError(std::size_t offset, std::string message)
    : std::runtime_error(json_error_message(offset, message)), offset_(offset) {
}

JsonValue::JsonValue(std::nullptr_t) noexcept : value_(nullptr) {}

JsonValue::JsonValue(bool value) noexcept : value_(value) {}

JsonValue::JsonValue(double value) : value_(value) {
  if (!std::isfinite(value)) {
    throw std::domain_error("JSON double must be finite");
  }
}

JsonValue::JsonValue(const char *value)
    : JsonValue(value == nullptr
                    ? throw std::invalid_argument("JSON string pointer is null")
                    : std::string(value)) {}

JsonValue::JsonValue(std::string value) : value_(std::move(value)) {
  static_cast<void>(utf8_code_point_count(std::get<std::string>(value_)));
}

JsonValue::JsonValue(std::string_view value) : JsonValue(std::string(value)) {}

JsonValue::JsonValue(array value) : value_(std::move(value)) {}

JsonValue::JsonValue(object value) : value_(std::move(value)) {
  for (const auto &[key, element] : std::get<object>(value_)) {
    static_cast<void>(element);
    static_cast<void>(utf8_code_point_count(key));
  }
}

JsonValue JsonValue::parse(std::string_view input) {
  return Parser(input).parse_document();
}

std::string JsonValue::dump() const {
  std::string output;
  append_json(*this, output);
  return output;
}

bool JsonValue::is_null() const noexcept {
  return std::holds_alternative<std::nullptr_t>(value_);
}

bool JsonValue::is_bool() const noexcept {
  return std::holds_alternative<bool>(value_);
}

bool JsonValue::is_int() const noexcept {
  return std::holds_alternative<std::int64_t>(value_);
}

bool JsonValue::is_double() const noexcept {
  return std::holds_alternative<double>(value_);
}

bool JsonValue::is_number() const noexcept { return is_int() || is_double(); }

bool JsonValue::is_string() const noexcept {
  return std::holds_alternative<std::string>(value_);
}

bool JsonValue::is_array() const noexcept {
  return std::holds_alternative<array>(value_);
}

bool JsonValue::is_object() const noexcept {
  return std::holds_alternative<object>(value_);
}

bool JsonValue::as_bool() const {
  return require_type<bool>(value_, "a boolean");
}

std::int64_t JsonValue::as_int() const {
  return require_type<std::int64_t>(value_, "an integer");
}

double JsonValue::as_double() const {
  if (const auto *value = std::get_if<double>(&value_)) {
    return *value;
  }
  if (const auto *value = std::get_if<std::int64_t>(&value_)) {
    return static_cast<double>(*value);
  }
  throw std::logic_error("JSON value is not a number");
}

const std::string &JsonValue::as_string() const {
  return require_type<std::string>(value_, "a string");
}

std::string &JsonValue::as_string() {
  return require_type<std::string>(value_, "a string");
}

const JsonValue::array &JsonValue::as_array() const {
  return require_type<array>(value_, "an array");
}

JsonValue::array &JsonValue::as_array() {
  return require_type<array>(value_, "an array");
}

const JsonValue::object &JsonValue::as_object() const {
  return require_type<object>(value_, "an object");
}

JsonValue::object &JsonValue::as_object() {
  return require_type<object>(value_, "an object");
}

const JsonValue *JsonValue::find(std::string_view key) const noexcept {
  const auto *values = std::get_if<object>(&value_);
  if (values == nullptr) {
    return nullptr;
  }
  const auto iterator = values->find(key);
  return iterator == values->end() ? nullptr : &iterator->second;
}

JsonValue *JsonValue::find(std::string_view key) noexcept {
  auto *values = std::get_if<object>(&value_);
  if (values == nullptr) {
    return nullptr;
  }
  const auto iterator = values->find(key);
  return iterator == values->end() ? nullptr : &iterator->second;
}

const JsonValue &JsonValue::at(std::string_view key) const {
  const JsonValue *result = find(key);
  if (result == nullptr) {
    throw std::out_of_range("JSON object key was not found: " +
                            std::string(key));
  }
  return *result;
}

JsonValue &JsonValue::at(std::string_view key) {
  JsonValue *result = find(key);
  if (result == nullptr) {
    throw std::out_of_range("JSON object key was not found: " +
                            std::string(key));
  }
  return *result;
}

const JsonValue &JsonValue::at(std::size_t index) const {
  return as_array().at(index);
}

JsonValue &JsonValue::at(std::size_t index) { return as_array().at(index); }

bool JsonValue::contains(std::string_view key) const noexcept {
  return find(key) != nullptr;
}

std::size_t JsonValue::size() const noexcept {
  if (const auto *value = std::get_if<std::string>(&value_)) {
    return value->size();
  }
  if (const auto *value = std::get_if<array>(&value_)) {
    return value->size();
  }
  if (const auto *value = std::get_if<object>(&value_)) {
    return value->size();
  }
  return 0;
}

} // namespace sglang::benchmark
