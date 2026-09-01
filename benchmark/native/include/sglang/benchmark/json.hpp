#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <sglang/benchmark/config.hpp>

namespace sglang::benchmark {

// Counts Unicode scalar values in a UTF-8 string. Throws std::invalid_argument
// when the input is not shortest-form UTF-8 or encodes a surrogate/non-scalar.
[[nodiscard]] std::size_t utf8_code_point_count(std::string_view text);

class JsonError final : public std::runtime_error {
public:
  JsonError(std::size_t offset, std::string message);

  [[nodiscard]] std::size_t offset() const noexcept { return offset_; }

private:
  std::size_t offset_;
};

class JsonValue {
public:
  using array = std::vector<JsonValue>;
  using object = std::map<std::string, JsonValue, std::less<>>;
  using storage = std::variant<std::nullptr_t, bool, std::int64_t, double,
                               std::string, array, object>;

  JsonValue(std::nullptr_t = nullptr) noexcept;
  JsonValue(bool value) noexcept;

  template <std::integral Integer>
    requires(!std::same_as<std::remove_cv_t<Integer>, bool>)
  JsonValue(Integer value) {
    static_assert(sizeof(Integer) <= sizeof(std::uint64_t));
    if constexpr (std::is_unsigned_v<Integer>) {
      if (static_cast<std::uint64_t>(value) >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::int64_t>::max())) {
        throw std::out_of_range("JSON integer exceeds int64 range");
      }
    }
    value_ = static_cast<std::int64_t>(value);
  }

  JsonValue(double value);
  JsonValue(const char *value);
  JsonValue(std::string value);
  JsonValue(std::string_view value);
  JsonValue(array value);
  JsonValue(object value);

  [[nodiscard]] static JsonValue parse(std::string_view input);
  [[nodiscard]] std::string dump() const;

  [[nodiscard]] bool is_null() const noexcept;
  [[nodiscard]] bool is_bool() const noexcept;
  [[nodiscard]] bool is_int() const noexcept;
  [[nodiscard]] bool is_double() const noexcept;
  [[nodiscard]] bool is_number() const noexcept;
  [[nodiscard]] bool is_string() const noexcept;
  [[nodiscard]] bool is_array() const noexcept;
  [[nodiscard]] bool is_object() const noexcept;

  [[nodiscard]] bool as_bool() const;
  [[nodiscard]] std::int64_t as_int() const;
  [[nodiscard]] double as_double() const;
  [[nodiscard]] const std::string &as_string() const;
  [[nodiscard]] std::string &as_string();
  [[nodiscard]] const array &as_array() const;
  [[nodiscard]] array &as_array();
  [[nodiscard]] const object &as_object() const;
  [[nodiscard]] object &as_object();

  [[nodiscard]] const JsonValue *find(std::string_view key) const noexcept;
  [[nodiscard]] JsonValue *find(std::string_view key) noexcept;
  [[nodiscard]] const JsonValue &at(std::string_view key) const;
  [[nodiscard]] JsonValue &at(std::string_view key);
  [[nodiscard]] const JsonValue &at(std::size_t index) const;
  [[nodiscard]] JsonValue &at(std::size_t index);
  [[nodiscard]] bool contains(std::string_view key) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept;

  [[nodiscard]] const storage &value() const noexcept { return value_; }
  [[nodiscard]] storage &value() noexcept { return value_; }

  friend bool operator==(const JsonValue &, const JsonValue &) = default;

private:
  storage value_;
};

// A short spelling for callers that prefer the data-format name.
using Json = JsonValue;

} // namespace sglang::benchmark
