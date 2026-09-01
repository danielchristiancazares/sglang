#pragma once

namespace sglang::benchmark {

#if defined(_MSVC_LANG)
inline constexpr long kCppLanguageVersion = _MSVC_LANG;
#else
inline constexpr long kCppLanguageVersion = __cplusplus;
#endif

static_assert(kCppLanguageVersion >= 202302L,
              "The native benchmark clients require C++23 or newer");

} // namespace sglang::benchmark
