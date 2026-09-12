#include "sglang/native/json_value.hpp"

#include <charconv>
#include <cmath>
#include <new>
#include <system_error>
#include <utility>

namespace sglang::native {
namespace {

[[nodiscard]] constexpr bool is_whitespace(char value) noexcept {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

[[nodiscard]] constexpr bool is_hex(char value) noexcept {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

[[nodiscard]] constexpr uint32_t hex_value(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return static_cast<uint32_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint32_t>(value - 'a' + 10);
  }
  return static_cast<uint32_t>(value - 'A' + 10);
}

[[nodiscard]] bool append_utf8(std::string &output, uint32_t scalar) {
  if (scalar <= 0x7fU) {
    output.push_back(static_cast<char>(scalar));
    return true;
  }
  if (scalar <= 0x7ffU) {
    output.push_back(static_cast<char>(0xc0U | (scalar >> 6U)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  if (scalar >= 0xd800U && scalar <= 0xdfffU) {
    return false;
  }
  if (scalar <= 0xffffU) {
    output.push_back(static_cast<char>(0xe0U | (scalar >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  if (scalar <= 0x10ffffU) {
    output.push_back(static_cast<char>(0xf0U | (scalar >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 12U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  return false;
}

class Parser final {
public:
  explicit Parser(std::string_view input) noexcept : input_(input) {}

  [[nodiscard]] NativeJsonParseResult parse() {
    NativeJsonParseResult result;
    skip_whitespace();
    if (!parse_value(result.value, 0U)) {
      result.error = error_;
      return result;
    }
    skip_whitespace();
    if (!at_end()) {
      result.error =
          NativeJsonError{NativeJsonErrorCode::kTrailingData, 0, position_};
    }
    return result;
  }

private:
  [[nodiscard]] bool at_end() const noexcept {
    return position_ == input_.size();
  }

  void skip_whitespace() noexcept {
    while (!at_end() && is_whitespace(input_[position_])) {
      ++position_;
    }
  }

  [[nodiscard]] bool consume(char expected) noexcept {
    if (at_end() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] bool fail(NativeJsonErrorCode code) noexcept {
    if (error_.code == NativeJsonErrorCode::kOk) {
      error_ = NativeJsonError{code, 0, position_};
    }
    return false;
  }

  [[nodiscard]] bool parse_value(NativeJsonValue &value, size_t depth) {
    if (depth > kNativeJsonMaximumDepth) {
      return fail(NativeJsonErrorCode::kDepthExceeded);
    }
    if (at_end()) {
      return fail(NativeJsonErrorCode::kSyntax);
    }
    switch (input_[position_]) {
    case 'n':
      if (!consume_literal("null")) {
        return false;
      }
      value = NativeJsonValue(nullptr);
      return true;
    case 't':
      if (!consume_literal("true")) {
        return false;
      }
      value = NativeJsonValue(true);
      return true;
    case 'f':
      if (!consume_literal("false")) {
        return false;
      }
      value = NativeJsonValue(false);
      return true;
    case '"': {
      std::string text;
      if (!parse_string(text)) {
        return false;
      }
      value = NativeJsonValue(std::move(text));
      return true;
    }
    case '[':
      return parse_array(value, depth);
    case '{':
      return parse_object(value, depth);
    default:
      if (input_[position_] == '-' ||
          (input_[position_] >= '0' && input_[position_] <= '9')) {
        return parse_number(value);
      }
      return fail(NativeJsonErrorCode::kSyntax);
    }
  }

  [[nodiscard]] bool consume_literal(std::string_view literal) noexcept {
    if (input_.substr(position_, literal.size()) != literal) {
      return fail(NativeJsonErrorCode::kSyntax);
    }
    position_ += literal.size();
    return true;
  }

  [[nodiscard]] bool parse_hex4(uint32_t &value) noexcept {
    if (input_.size() - position_ < 4U) {
      return fail(NativeJsonErrorCode::kSyntax);
    }
    value = 0;
    for (uint32_t index = 0; index < 4U; ++index) {
      const char digit = input_[position_++];
      if (!is_hex(digit)) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      value = (value << 4U) | hex_value(digit);
    }
    return true;
  }

  [[nodiscard]] bool parse_string(std::string &output) {
    if (!consume('"')) {
      return fail(NativeJsonErrorCode::kSyntax);
    }
    while (!at_end()) {
      const unsigned char value =
          static_cast<unsigned char>(input_[position_++]);
      if (value == '"') {
        return true;
      }
      if (value < 0x20U) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      if (value != '\\') {
        if (value >= 0x80U && !append_valid_utf8(value, output)) {
          return false;
        }
        if (value < 0x80U) {
          output.push_back(static_cast<char>(value));
        }
        continue;
      }
      if (at_end()) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      const char escape = input_[position_++];
      switch (escape) {
      case '"':
      case '\\':
      case '/':
        output.push_back(escape);
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
      case 'u': {
        uint32_t scalar = 0;
        if (!parse_hex4(scalar)) {
          return false;
        }
        if (scalar >= 0xd800U && scalar <= 0xdbffU) {
          if (input_.size() - position_ < 6U || input_[position_] != '\\' ||
              input_[position_ + 1U] != 'u') {
            return fail(NativeJsonErrorCode::kInvalidUtf8);
          }
          position_ += 2U;
          uint32_t low = 0;
          if (!parse_hex4(low) || low < 0xdc00U || low > 0xdfffU) {
            return fail(NativeJsonErrorCode::kInvalidUtf8);
          }
          scalar = 0x10000U + ((scalar - 0xd800U) << 10U) + (low - 0xdc00U);
        }
        if (!append_utf8(output, scalar)) {
          return fail(NativeJsonErrorCode::kInvalidUtf8);
        }
        break;
      }
      default:
        return fail(NativeJsonErrorCode::kSyntax);
      }
    }
    return fail(NativeJsonErrorCode::kSyntax);
  }

  [[nodiscard]] bool append_valid_utf8(unsigned char lead,
                                       std::string &output) {
    size_t bytes = 0;
    uint32_t scalar = 0;
    uint32_t minimum = 0;
    if (lead >= 0xc2U && lead <= 0xdfU) {
      bytes = 2U;
      scalar = lead & 0x1fU;
      minimum = 0x80U;
    } else if (lead >= 0xe0U && lead <= 0xefU) {
      bytes = 3U;
      scalar = lead & 0x0fU;
      minimum = 0x800U;
    } else if (lead >= 0xf0U && lead <= 0xf4U) {
      bytes = 4U;
      scalar = lead & 0x07U;
      minimum = 0x10000U;
    } else {
      return fail(NativeJsonErrorCode::kInvalidUtf8);
    }
    const size_t begin = position_ - 1U;
    if (input_.size() - begin < bytes) {
      return fail(NativeJsonErrorCode::kInvalidUtf8);
    }
    for (size_t index = 1U; index < bytes; ++index) {
      const unsigned char continuation =
          static_cast<unsigned char>(input_[position_++]);
      if ((continuation & 0xc0U) != 0x80U) {
        return fail(NativeJsonErrorCode::kInvalidUtf8);
      }
      scalar = (scalar << 6U) | (continuation & 0x3fU);
    }
    if (scalar < minimum || scalar > 0x10ffffU ||
        (scalar >= 0xd800U && scalar <= 0xdfffU)) {
      return fail(NativeJsonErrorCode::kInvalidUtf8);
    }
    output.append(input_.substr(begin, bytes));
    return true;
  }

  [[nodiscard]] bool parse_number(NativeJsonValue &value) noexcept {
    const size_t begin = position_;
    static_cast<void>(consume('-'));
    if (at_end()) {
      return fail(NativeJsonErrorCode::kSyntax);
    }
    if (consume('0')) {
      if (!at_end() && input_[position_] >= '0' && input_[position_] <= '9') {
        return fail(NativeJsonErrorCode::kSyntax);
      }
    } else {
      if (input_[position_] < '1' || input_[position_] > '9') {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      while (!at_end() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
    }

    bool integer = true;
    if (consume('.')) {
      integer = false;
      const size_t fraction_begin = position_;
      while (!at_end() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == fraction_begin) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
    }
    if (!at_end() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      integer = false;
      ++position_;
      if (!at_end() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const size_t exponent_begin = position_;
      while (!at_end() && input_[position_] >= '0' &&
             input_[position_] <= '9') {
        ++position_;
      }
      if (position_ == exponent_begin) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
    }

    const std::string_view token = input_.substr(begin, position_ - begin);
    if (integer) {
      int64_t parsed = 0;
      const auto status =
          std::from_chars(token.data(), token.data() + token.size(), parsed);
      if (status.ec == std::errc::result_out_of_range) {
        return fail(NativeJsonErrorCode::kIntegerOutOfRange);
      }
      if (status.ec != std::errc{} ||
          status.ptr != token.data() + token.size()) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      value = NativeJsonValue(parsed);
      return true;
    }
    double parsed = 0.0;
    const auto status =
        std::from_chars(token.data(), token.data() + token.size(), parsed,
                        std::chars_format::general);
    if (status.ec != std::errc{} || status.ptr != token.data() + token.size() ||
        !std::isfinite(parsed)) {
      return fail(NativeJsonErrorCode::kNumberOutOfRange);
    }
    value = NativeJsonValue(parsed);
    return true;
  }

  [[nodiscard]] bool parse_array(NativeJsonValue &value, size_t depth) {
    static_cast<void>(consume('['));
    skip_whitespace();
    NativeJsonValue::Array values;
    if (consume(']')) {
      value = NativeJsonValue(std::move(values));
      return true;
    }
    for (;;) {
      NativeJsonValue element;
      if (!parse_value(element, depth + 1U)) {
        return false;
      }
      values.push_back(std::move(element));
      skip_whitespace();
      if (consume(']')) {
        value = NativeJsonValue(std::move(values));
        return true;
      }
      if (!consume(',')) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      skip_whitespace();
    }
  }

  [[nodiscard]] bool parse_object(NativeJsonValue &value, size_t depth) {
    static_cast<void>(consume('{'));
    skip_whitespace();
    NativeJsonValue::Object values;
    if (consume('}')) {
      value = NativeJsonValue(std::move(values));
      return true;
    }
    for (;;) {
      std::string key;
      if (!parse_string(key)) {
        return false;
      }
      skip_whitespace();
      if (!consume(':')) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      skip_whitespace();
      NativeJsonValue element;
      if (!parse_value(element, depth + 1U)) {
        return false;
      }
      if (!values.emplace(std::move(key), std::move(element)).second) {
        return fail(NativeJsonErrorCode::kDuplicateKey);
      }
      skip_whitespace();
      if (consume('}')) {
        value = NativeJsonValue(std::move(values));
        return true;
      }
      if (!consume(',')) {
        return fail(NativeJsonErrorCode::kSyntax);
      }
      skip_whitespace();
    }
  }

  std::string_view input_;
  size_t position_{0};
  NativeJsonError error_{};
};

} // namespace

bool NativeJsonValue::is_null() const noexcept {
  return std::holds_alternative<std::nullptr_t>(value_);
}

bool NativeJsonValue::is_bool() const noexcept {
  return std::holds_alternative<bool>(value_);
}

bool NativeJsonValue::is_int() const noexcept {
  return std::holds_alternative<int64_t>(value_);
}

bool NativeJsonValue::is_double() const noexcept {
  return std::holds_alternative<double>(value_);
}

bool NativeJsonValue::is_string() const noexcept {
  return std::holds_alternative<std::string>(value_);
}

bool NativeJsonValue::is_array() const noexcept {
  return std::holds_alternative<Array>(value_);
}

bool NativeJsonValue::is_object() const noexcept {
  return std::holds_alternative<Object>(value_);
}

const bool *NativeJsonValue::bool_value() const noexcept {
  return std::get_if<bool>(&value_);
}

const int64_t *NativeJsonValue::int_value() const noexcept {
  return std::get_if<int64_t>(&value_);
}

const double *NativeJsonValue::double_value() const noexcept {
  return std::get_if<double>(&value_);
}

const std::string *NativeJsonValue::string_value() const noexcept {
  return std::get_if<std::string>(&value_);
}

const NativeJsonValue::Array *NativeJsonValue::array_value() const noexcept {
  return std::get_if<Array>(&value_);
}

const NativeJsonValue::Object *NativeJsonValue::object_value() const noexcept {
  return std::get_if<Object>(&value_);
}

const NativeJsonValue *
NativeJsonValue::find(std::string_view key) const noexcept {
  const Object *object = object_value();
  if (object == nullptr) {
    return nullptr;
  }
  const auto found = object->find(key);
  return found == object->end() ? nullptr : &found->second;
}

NativeJsonParseResult parse_native_json(std::string_view input) noexcept {
  try {
    return Parser(input).parse();
  } catch (const std::bad_alloc &) {
    NativeJsonParseResult result;
    result.error.code = NativeJsonErrorCode::kAllocationFailed;
    return result;
  } catch (...) {
    NativeJsonParseResult result;
    result.error.code = NativeJsonErrorCode::kInternalFailure;
    return result;
  }
}

std::string_view
native_json_error_code_name(NativeJsonErrorCode code) noexcept {
  switch (code) {
  case NativeJsonErrorCode::kOk:
    return "ok";
  case NativeJsonErrorCode::kSyntax:
    return "syntax";
  case NativeJsonErrorCode::kTrailingData:
    return "trailing_data";
  case NativeJsonErrorCode::kDuplicateKey:
    return "duplicate_key";
  case NativeJsonErrorCode::kInvalidUtf8:
    return "invalid_utf8";
  case NativeJsonErrorCode::kDepthExceeded:
    return "depth_exceeded";
  case NativeJsonErrorCode::kIntegerOutOfRange:
    return "integer_out_of_range";
  case NativeJsonErrorCode::kNumberOutOfRange:
    return "number_out_of_range";
  case NativeJsonErrorCode::kAllocationFailed:
    return "allocation_failed";
  case NativeJsonErrorCode::kInternalFailure:
    return "internal_failure";
  default:
    return "invalid_native_json_error";
  }
}

} // namespace sglang::native
