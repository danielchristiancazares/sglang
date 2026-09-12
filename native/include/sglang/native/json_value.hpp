#ifndef SGLANG_NATIVE_JSON_VALUE_HPP_
#define SGLANG_NATIVE_JSON_VALUE_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace sglang::native {

inline constexpr size_t kNativeJsonMaximumDepth = 512;

enum class NativeJsonErrorCode : uint32_t {
  kOk = 0,
  kSyntax,
  kTrailingData,
  kDuplicateKey,
  kInvalidUtf8,
  kDepthExceeded,
  kIntegerOutOfRange,
  kNumberOutOfRange,
  kAllocationFailed,
  kInternalFailure,
};

struct NativeJsonError final {
  NativeJsonErrorCode code{NativeJsonErrorCode::kOk};
  uint32_t reserved{0};
  uint64_t byte_offset{0};
};

class NativeJsonValue final {
public:
  using Array = std::vector<NativeJsonValue>;
  using Object = std::map<std::string, NativeJsonValue, std::less<>>;
  using Storage = std::variant<std::nullptr_t, bool, int64_t, double,
                               std::string, Array, Object>;

  NativeJsonValue() noexcept = default;
  explicit NativeJsonValue(std::nullptr_t) noexcept : value_(nullptr) {}
  explicit NativeJsonValue(bool value) noexcept : value_(value) {}
  explicit NativeJsonValue(int64_t value) noexcept : value_(value) {}
  explicit NativeJsonValue(double value) noexcept : value_(value) {}
  explicit NativeJsonValue(std::string value) : value_(std::move(value)) {}
  explicit NativeJsonValue(Array value) : value_(std::move(value)) {}
  explicit NativeJsonValue(Object value) : value_(std::move(value)) {}

  [[nodiscard]] bool is_null() const noexcept;
  [[nodiscard]] bool is_bool() const noexcept;
  [[nodiscard]] bool is_int() const noexcept;
  [[nodiscard]] bool is_double() const noexcept;
  [[nodiscard]] bool is_string() const noexcept;
  [[nodiscard]] bool is_array() const noexcept;
  [[nodiscard]] bool is_object() const noexcept;

  [[nodiscard]] const bool *bool_value() const noexcept;
  [[nodiscard]] const int64_t *int_value() const noexcept;
  [[nodiscard]] const double *double_value() const noexcept;
  [[nodiscard]] const std::string *string_value() const noexcept;
  [[nodiscard]] const Array *array_value() const noexcept;
  [[nodiscard]] const Object *object_value() const noexcept;
  [[nodiscard]] const NativeJsonValue *
  find(std::string_view key) const noexcept;

private:
  Storage value_{nullptr};
};

struct NativeJsonParseResult final {
  NativeJsonValue value;
  NativeJsonError error;

  [[nodiscard]] bool ok() const noexcept {
    return error.code == NativeJsonErrorCode::kOk;
  }
};

[[nodiscard]] std::string_view
native_json_error_code_name(NativeJsonErrorCode code) noexcept;

// Strict RFC 8259 JSON. The public boundary is non-throwing; allocation and
// implementation exceptions are converted into structured parse errors.
[[nodiscard]] NativeJsonParseResult
parse_native_json(std::string_view input) noexcept;

} // namespace sglang::native

#endif // SGLANG_NATIVE_JSON_VALUE_HPP_
