#include "sglang/native/json_value.hpp"
#include "sglang/native/sha256.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>

namespace {

using sglang::native::NativeJsonErrorCode;
using sglang::native::NativeJsonValue;
using sglang::native::parse_native_json;
using sglang::native::sha256;
using sglang::native::sha256_hex;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {   \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] bool StrictDocumentParses() {
  const auto result = parse_native_json(
      R"({"null":null,"bool":true,"int":-7,"double":1.25e2,"text":"a\u03b1","array":[1,false]})");
  CHECK(result.ok());
  CHECK(result.value.is_object());
  const NativeJsonValue *null_value = result.value.find("null");
  const NativeJsonValue *bool_value = result.value.find("bool");
  const NativeJsonValue *int_value = result.value.find("int");
  const NativeJsonValue *double_value = result.value.find("double");
  const NativeJsonValue *text = result.value.find("text");
  const NativeJsonValue *array = result.value.find("array");
  CHECK(null_value != nullptr && null_value->is_null());
  CHECK(bool_value != nullptr && bool_value->bool_value() != nullptr &&
        *bool_value->bool_value());
  CHECK(int_value != nullptr && int_value->int_value() != nullptr &&
        *int_value->int_value() == -7);
  CHECK(double_value != nullptr && double_value->double_value() != nullptr &&
        *double_value->double_value() == 125.0);
  CHECK(text != nullptr && text->string_value() != nullptr &&
        *text->string_value() == "a\xce\xb1");
  CHECK(array != nullptr && array->array_value() != nullptr &&
        array->array_value()->size() == 2U);
  return true;
}

[[nodiscard]] bool MalformedInputsFailClosed() {
  CHECK(parse_native_json(R"({"x":1,"x":2})").error.code ==
        NativeJsonErrorCode::kDuplicateKey);
  CHECK(parse_native_json("01").error.code == NativeJsonErrorCode::kSyntax);
  CHECK(parse_native_json("9223372036854775808").error.code ==
        NativeJsonErrorCode::kIntegerOutOfRange);
  CHECK(parse_native_json("1e9999").error.code ==
        NativeJsonErrorCode::kNumberOutOfRange);
  CHECK(parse_native_json("true false").error.code ==
        NativeJsonErrorCode::kTrailingData);
  CHECK(parse_native_json(R"("\udc00")").error.code ==
        NativeJsonErrorCode::kInvalidUtf8);

  std::string malformed_utf8{"\""};
  malformed_utf8.push_back(static_cast<char>(0xc0U));
  malformed_utf8.push_back(static_cast<char>(0x80U));
  malformed_utf8.push_back('"');
  CHECK(parse_native_json(malformed_utf8).error.code ==
        NativeJsonErrorCode::kInvalidUtf8);
  return true;
}

[[nodiscard]] bool DepthLimitFailsClosed() {
  std::string document;
  document.append(514U, '[');
  document += '0';
  document.append(514U, ']');
  CHECK(parse_native_json(document).error.code ==
        NativeJsonErrorCode::kDepthExceeded);
  return true;
}

[[nodiscard]] bool Sha256KnownVectorsPass() {
  CHECK(sha256_hex(sha256("")) ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha256_hex(sha256("abc")) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  sglang::native::Sha256 streaming;
  CHECK(streaming.update("a"));
  CHECK(streaming.update("bc"));
  CHECK(sha256_hex(streaming.finalize()) ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(!streaming.update("after-finalize"));
  return true;
}

} // namespace

int main() {
  const std::array tests{
      std::pair{"strict document parses", &StrictDocumentParses},
      std::pair{"malformed inputs fail closed", &MalformedInputsFailClosed},
      std::pair{"depth limit fails closed", &DepthLimitFailsClosed},
      std::pair{"SHA-256 known vectors pass", &Sha256KnownVectorsPass},
  };
  size_t passed = 0;
  for (const auto &[name, test] : tests) {
    if (!test()) {
      std::printf("[  FAILED  ] %s\n", name);
      return 1;
    }
    ++passed;
    std::printf("[       OK ] %s\n", name);
  }
  std::printf("[  PASSED  ] %zu tests\n", passed);
  return 0;
}
