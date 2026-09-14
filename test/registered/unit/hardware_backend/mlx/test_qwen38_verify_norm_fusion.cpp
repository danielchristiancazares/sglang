#include <cstdint>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;
namespace native = sglang::mlx_qwen38;

namespace {
mx::array Values(const mx::Shape& shape, int multiplier, mx::Dtype dtype) {
  std::size_t size = 1;
  for (int extent : shape) size *= static_cast<std::size_t>(extent);
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i) {
    const int centered = (static_cast<int>(i % 521) * multiplier + 19) % 521 - 260;
    values[i] = static_cast<float>(centered) / 256.0f;
  }
  return mx::astype(mx::array(values.data(), shape, mx::float32), dtype);
}

struct Comparison {
  std::string_view label;
  mx::array actual;
  mx::array expected;
};

bool Check(int tokens) {
  std::vector<Comparison> pending;
  auto x = Values({1, tokens, 5120}, 37, sglang::mlx_qwen38::activation_dtype());
  auto residual = Values(x.shape(), 71, sglang::mlx_qwen38::activation_dtype());
  auto weight = Values({5120}, 97, sglang::mlx_qwen38::activation_dtype());
  auto fused = native::residual_rms_norm(x, residual, weight, 1e-6f);
  auto summed = x + residual;
  pending.push_back({"residual", fused.first, summed});
  pending.push_back({"residual_norm", fused.second,
                     mx::fast::rms_norm(summed, weight, 1e-6f)});

  // Splitting the packed convolution output leaves strided token rows.
  // These are the shapes passed by speculative gated-delta verification.
  auto qkv = Values({1, tokens, 10240}, 43, sglang::mlx_qwen38::activation_dtype());
  auto slices = mx::split(qkv, mx::Shape{2048, 4096}, -1);
  auto q = mx::reshape(slices[0], {1, tokens, 16, 128});
  auto k = mx::reshape(slices[1], {1, tokens, 16, 128});
  const float inv = 1.0f / std::sqrt(128.0f);
  auto normalized = native::normalize_gated_delta_qk(q, k, inv * inv, inv, 1e-6f);
  pending.push_back({"q", normalized.first,
                     (inv * inv) * mx::fast::rms_norm(q, std::nullopt, 1e-6f)});
  pending.push_back({"k", normalized.second,
                     inv * mx::fast::rms_norm(k, std::nullopt, 1e-6f)});

  auto recurrent = Values({1, tokens, 48, 128}, 59, mx::float32);
  auto z = Values(recurrent.shape(), 61, sglang::mlx_qwen38::activation_dtype());
  auto gate_weight = Values({128}, 89, sglang::mlx_qwen38::activation_dtype());
  auto gate = native::gated_delta_norm_gate(recurrent, z, gate_weight, 1e-6f);
  auto reference_gate = mx::astype(
      native::silu(mx::astype(z, mx::float32)) *
          mx::astype(mx::fast::rms_norm(recurrent, gate_weight, 1e-6f), mx::float32),
      z.dtype());
  pending.push_back({"gate", gate, reference_gate});

  std::vector<mx::array> outstanding;
  for (const auto& entry : pending) outstanding.push_back(entry.actual);
  mx::async_eval(outstanding);
  for (const auto& entry : pending) {
    if (entry.actual.shape() != entry.expected.shape() ||
        entry.actual.dtype() != entry.expected.dtype()) {
      std::cerr << entry.label << " metadata mismatch, tokens=" << tokens << '\n';
      return false;
    }
    auto equal = mx::all(mx::equal(entry.actual, entry.expected));
    auto finite = mx::all(mx::isfinite(entry.actual));
    mx::eval(equal, finite);
    if (!equal.item<bool>() || !finite.item<bool>()) {
      std::cerr << entry.label << " value mismatch, tokens=" << tokens << '\n';
      return false;
    }
  }
  std::cout << "verify_norm_tokens=" << tokens << " exact=1 finite=1\n";
  return true;
}
}  // namespace

int main() {
  for (int tokens : {1, 2, 3, 8, 9}) {
    if (!Check(tokens)) return 1;
  }
  return 0;
}
