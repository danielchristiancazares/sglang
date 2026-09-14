#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "mlx/ops.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;
namespace native = sglang::mlx_qwen38;
using Member = mx::array native::FullAttn::*;
constexpr std::array<Member, 6> kBuffers{
    &native::FullAttn::keys, &native::FullAttn::key_scales,
    &native::FullAttn::key_biases, &native::FullAttn::values,
    &native::FullAttn::value_scales, &native::FullAttn::value_biases};

void Require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

template <class Function> void Reject(Function function, const char* message) {
  try { function(); } catch (const std::runtime_error&) { return; }
  throw std::runtime_error(message);
}

native::FullAttn Q8(int capacity) {
  native::FullAttn cache;
  cache.cache_bits = 8;
  cache.cache_capacity = capacity;
  cache.cache_length = cache.offset = 17;
  for (auto member : kBuffers) {
    const bool packed = member == &native::FullAttn::keys ||
                        member == &native::FullAttn::values;
    cache.*member = mx::full({1, 4, capacity, packed ? 64 : 4},
        packed ? 12345.0f : 0.25f, packed ? mx::uint32 : native::activation_dtype());
  }
  return cache;
}

void Change(native::FullAttn& cache, Member member, int token) {
  auto& source = cache.*member;
  source = mx::slice_update(source, mx::full({1, 1, 1, 1}, 27.0f, source.dtype()),
      {0, 0, token, 0}, {1, 1, token + 1, 1});
}

int main() {
  try {
    const auto base = Q8(64);
    const auto digest = native::attention_cache_digest(base);
    Require(digest == native::attention_cache_digest(base), "unstable Q8 digest");
    for (auto member : kBuffers) {
      auto changed = base;
      Change(changed, member, 16);
      Require(native::attention_cache_digest(changed) != digest,
              "active Q8 buffer omitted from digest");
      auto suffix = base;
      Change(suffix, member, 17);
      Require(native::attention_cache_digest(suffix) == digest,
              "unused Q8 suffix entered digest");
      auto malformed = base;
      malformed.*member = mx::zeros({1}, (base.*member).dtype());
      Reject([&] { native::attention_cache_digest(malformed); },
             "invalid Q8 buffer shape accepted");
      auto wrong_dtype = base;
      wrong_dtype.*member = mx::astype(base.*member, mx::float32);
      Reject([&] { native::attention_cache_digest(wrong_dtype); },
             "invalid Q8 buffer dtype accepted");
    }
    auto poison = base;
    for (auto member : {&native::FullAttn::key_scales, &native::FullAttn::key_biases,
                        &native::FullAttn::value_scales, &native::FullAttn::value_biases}) {
      auto& source = poison.*member;
      source = mx::slice_update(source,
          mx::full({1, 4, 47, 4}, std::numeric_limits<float>::quiet_NaN(), source.dtype()),
          {0, 0, 17, 0}, {1, 4, 64, 4});
    }
    Require(native::attention_cache_digest(poison) == digest,
            "poisoned inactive coefficients entered digest");
    const auto larger = Q8(128);
    Require(native::attention_cache_digest(larger) != digest,
            "capacity-aware digest omitted allocation shape");
    Require(native::attention_cache_digest(larger, false) ==
            native::attention_cache_digest(base, false), "active digest changed with reserve");
    auto metadata = base;
    ++metadata.offset;
    Require(native::attention_cache_digest(metadata) != digest, "offset omitted");
    metadata = base;
    --metadata.cache_length;
    Require(native::attention_cache_digest(metadata) != digest, "length omitted");
    metadata = base;
    metadata.cache_bits = 4;
    Reject([&] { native::attention_cache_digest(metadata); }, "unknown format accepted");
    metadata = base;
    metadata.cache_length = 65;
    Reject([&] { native::attention_cache_digest(metadata); }, "invalid extent accepted");
    native::FullAttn empty;
    const auto empty_dense = native::attention_cache_digest(empty);
    empty.cache_bits = 8;
    Require(native::attention_cache_digest(empty) != empty_dense,
            "empty format domains collide");
    auto dense = base;
    dense.cache_bits = 16;
    dense.keys = mx::zeros({1, 4, 64, 256}, native::activation_dtype());
    dense.values = mx::ones({1, 4, 64, 256}, native::activation_dtype());
    const auto dense_digest = native::attention_cache_digest(dense);
    Change(dense, &native::FullAttn::values, 17);
    Require(native::attention_cache_digest(dense) == dense_digest,
            "dense inactive suffix entered digest");
    Change(dense, &native::FullAttn::values, 16);
    Require(native::attention_cache_digest(dense) != dense_digest,
            "active dense value omitted");
    std::cout << "attention_cache_digest=pass all_six_q8_buffers=checked"
              << " inactive_suffix=excluded reserve_independence=checked\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
