#pragma once

#include <cstdio>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace native_test {
inline void check(bool value, const char *expression, const char *file, int line) {
  if (!value) {
    std::fprintf(stderr, "%s:%d: %s\n", file, line, expression);
    throw std::runtime_error("test assertion failed");
  }
}
template <class Function> bool throws(Function &&function) {
  try { std::invoke(std::forward<Function>(function)); }
  catch (const std::exception &) { return true; }
  return false;
}
template <class Tests> int run(std::string_view name, const Tests &tests) {
  std::size_t passed = 0;
  for (const auto &[label, function] : tests) {
    try {
      function();
      ++passed;
    } catch (const std::exception &error) {
      std::fprintf(stderr, "%.*s FAILED %s: %s\n", static_cast<int>(name.size()),
                   name.data(), label, error.what());
      return 1;
    }
  }
  std::printf("%.*s: %zu/%zu passed\n", static_cast<int>(name.size()),
              name.data(), passed, tests.size());
  return 0;
}
} // namespace native_test

#define REQUIRE(...) \
  native_test::check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
