#include "sglang/benchmark/json.hpp"
#include "sglang/benchmark/sha256.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using sglang::benchmark::JsonError;
using sglang::benchmark::JsonValue;
using sglang::benchmark::Sha256;
using sglang::benchmark::sha256_hex;
using sglang::benchmark::utf8_code_point_count;

void expect(bool condition, std::string_view description) {
  if (!condition) {
    throw std::runtime_error("assertion failed: " + std::string(description));
  }
}

template <typename Exception, typename Callable>
void expect_throws(Callable &&callable, std::string_view description) {
  try {
    callable();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error("expected exception: " + std::string(description));
}

void test_json_values_and_accessors() {
  JsonValue::object fields{{"z", nullptr},
                           {"message", "hello"},
                           {"count", 7},
                           {"ratio", 1.5},
                           {"flags", JsonValue::array{true, false}}};
  JsonValue value(std::move(fields));

  expect(value.is_object(), "object predicate");
  expect(value.size() == 5, "object size");
  expect(value.contains("count"), "object contains");
  expect(value.find("missing") == nullptr, "missing object member");
  expect(value.at("count").is_int(), "integer predicate");
  expect(value.at("count").as_int() == 7, "integer accessor");
  expect(value.at("count").as_double() == 7.0, "numeric widening accessor");
  expect(value.at("ratio").is_double(), "double predicate");
  expect(value.at("ratio").as_double() == 1.5, "double accessor");
  expect(value.at("message").as_string() == "hello", "string accessor");
  expect(value.at("flags").at(0).as_bool(), "array index accessor");
  expect(!value.at("flags").at(1).as_bool(), "false boolean accessor");
  expect(value.at("z").is_null(), "null predicate");
  expect(
      value.dump() ==
          R"({"count":7,"flags":[true,false],"message":"hello","ratio":1.5,"z":null})",
      "compact sorted object dump");

  value.at("message").as_string() = "updated";
  expect(value.at("message").as_string() == "updated", "mutable accessor");
  expect_throws<std::out_of_range>(
      [&] { static_cast<void>(value.at("absent")); }, "checked missing key");
  expect_throws<std::out_of_range>(
      [&] { static_cast<void>(value.at("flags").at(2)); },
      "checked array index");
  expect_throws<std::logic_error>([&] { static_cast<void>(value.as_bool()); },
                                  "typed accessor mismatch");

  const JsonValue unsigned_max(
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()));
  expect(unsigned_max.as_int() == std::numeric_limits<std::int64_t>::max(),
         "unsigned integer boundary");
  expect_throws<std::out_of_range>(
      [] { JsonValue ignored(std::numeric_limits<std::uint64_t>::max()); },
      "unsigned constructor overflow");
  expect_throws<std::domain_error>(
      [] { JsonValue ignored(std::numeric_limits<double>::infinity()); },
      "infinite constructor");
  expect_throws<std::domain_error>(
      [] { JsonValue ignored(std::numeric_limits<double>::quiet_NaN()); },
      "NaN constructor");
}

void test_json_parsing_and_escaping() {
  const std::string document =
      R"( { "text": "quote:\" slash:\\ solidus:\/ controls:\b\f\n\r\t", "unicode": "\u20ac \ud83d\ude03", "maximum": 9223372036854775807, "minimum": -9223372036854775808, "number": -1.25e+2 } )";
  const JsonValue parsed = JsonValue::parse(document);
  expect(parsed.at("maximum").as_int() ==
             std::numeric_limits<std::int64_t>::max(),
         "maximum parsed integer");
  expect(parsed.at("minimum").as_int() ==
             std::numeric_limits<std::int64_t>::min(),
         "minimum parsed integer");
  expect(parsed.at("number").as_double() == -125.0, "parsed exponent");
  expect(parsed.at("unicode").as_string() ==
             std::string("\xe2\x82\xac \xf0\x9f\x98\x83"),
         "Unicode escape and surrogate decoding");
  expect(parsed.at("text").as_string() ==
             std::string("quote:\" slash:\\ solidus:/ controls:\b\f\n\r\t"),
         "JSON escape decoding");
  expect(JsonValue::parse(parsed.dump()) == parsed, "parse-dump round trip");
  expect(JsonValue::parse("\"\xe2\x82\xac\"").as_string() ==
             std::string("\xe2\x82\xac"),
         "raw UTF-8 parse");

  const std::string controls("\0\x01\b\f\n\r\t", 7);
  expect(JsonValue(controls).dump() == R"("\u0000\u0001\b\f\n\r\t")",
         "control character escaping");
  expect(JsonValue(1.0).dump() == "1.0", "integral-looking double dump");
  expect(JsonValue(-0.0).dump() == "-0.0", "negative zero double dump");
  expect(std::signbit(JsonValue::parse("-0.0").as_double()),
         "negative zero parse");

  const std::vector<std::string> invalid_documents = {
      "",
      "true false",
      "True",
      "nul",
      "+1",
      "01",
      "-01",
      "1.",
      "1e",
      "1e+",
      "--1",
      "9223372036854775808",
      "-9223372036854775809",
      "1e9999",
      R"({"a":1,"a":2})",
      R"({"a":1,"\u0061":2})",
      R"({"a" 1})",
      R"({"a":1,})",
      R"([1,])",
      R"([1 2])",
      R"("\x")",
      R"("\u12xz")",
      R"("\ud800")",
      R"("\ud800\u0041")",
      R"("\udc00")",
      std::string("\"line\nfeed\""),
      std::string("\"\xc0\xaf\""),
  };
  for (const std::string &invalid : invalid_documents) {
    expect_throws<JsonError>(
        [&] { static_cast<void>(JsonValue::parse(invalid)); }, invalid);
  }

  try {
    static_cast<void>(JsonValue::parse("[] trailing"));
    throw std::runtime_error("trailing data was accepted");
  } catch (const JsonError &error) {
    expect(error.offset() == 3, "parse error byte offset");
  }
}

void test_utf8_validation() {
  const std::string valid =
      std::string("A") + "\xc2\xa2" + "\xe2\x82\xac" + "\xf0\x90\x8d\x88";
  expect(utf8_code_point_count(valid) == 4, "UTF-8 scalar count");
  expect(utf8_code_point_count(std::string_view("a\0b", 3)) == 3,
         "UTF-8 embedded zero");

  const std::vector<std::string> malformed = {
      std::string("\x80", 1),
      std::string("\xc0\xaf", 2),
      std::string("\xc2", 1),
      std::string("\xe0\x80\x80", 3),
      std::string("\xed\xa0\x80", 3),
      std::string("\xe2\x28\xa1", 3),
      std::string("\xf0\x80\x80\x80", 4),
      std::string("\xf4\x90\x80\x80", 4),
      std::string("\xf0\x9f\x98", 3),
      std::string("\xf5\x80\x80\x80", 4),
  };
  for (const std::string &text : malformed) {
    expect_throws<std::invalid_argument>(
        [&] { static_cast<void>(utf8_code_point_count(text)); },
        "malformed UTF-8");
    expect_throws<std::invalid_argument>([&] { JsonValue ignored(text); },
                                         "malformed JSON string constructor");
  }

  JsonValue::object invalid_key{{std::string("\xc0\xaf", 2), nullptr}};
  expect_throws<std::invalid_argument>(
      [&] { JsonValue ignored(std::move(invalid_key)); },
      "malformed UTF-8 object key");

  JsonValue mutated("valid");
  mutated.as_string() = std::string("\xed\xa0\x80", 3);
  expect_throws<std::invalid_argument>(
      [&] { static_cast<void>(mutated.dump()); },
      "malformed UTF-8 introduced through mutable access");
}

void test_sha256_vectors() {
  expect(sha256_hex("") ==
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
         "SHA-256 empty vector");
  expect(sha256_hex("abc") ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "SHA-256 abc vector");
  expect(
      sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
      "SHA-256 multi-block vector");

  std::string million_a(1'000'000, 'a');
  expect(sha256_hex(million_a) ==
             "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
         "SHA-256 million-a vector");

  const std::array<std::byte, 1> zero{std::byte{0}};
  expect(sha256_hex(std::span<const std::byte>(zero)) ==
             "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d",
         "SHA-256 binary zero vector");

  Sha256 chunked;
  chunked.update("a").update("").update("b").update("c");
  const std::string first_final = chunked.final_hex();
  expect(first_final ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "chunked SHA-256");
  expect(chunked.final_hex() == first_final, "idempotent SHA-256 finalization");
  expect_throws<std::logic_error>([&] { chunked.update("later"); },
                                  "update after SHA-256 finalization");
}

} // namespace

int main() {
  try {
    test_json_values_and_accessors();
    test_json_parsing_and_escaping();
    test_utf8_validation();
    test_sha256_vectors();
    std::cout << "json_sha256_test: PASS\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "json_sha256_test: FAIL: " << error.what() << '\n';
    return 1;
  }
}
