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

namespace {

std::vector<float> MakeValues(std::size_t size, int multiplier) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i) {
    const int centered =
        (static_cast<int>(i % 521) * multiplier + 19) % 223 - 111;
    values[i] = static_cast<float>(centered) / 128.0f;
  }
  return values;
}

bool ExactEqual(
    std::string_view label,
    const mx::array& actual,
    const mx::array& expected) {
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype()) {
    std::cerr << label << " metadata mismatch\n";
    return false;
  }
  mx::array actual_f = mx::astype(actual, mx::float32);
  mx::array expected_f = mx::astype(expected, mx::float32);
  mx::eval(actual_f, expected_f);
  const float* actual_data = actual_f.data<float>();
  const float* expected_data = expected_f.data<float>();
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (actual_data[i] != expected_data[i]) {
      std::cerr << label << " mismatch at " << i << ": " << actual_data[i]
                << " != " << expected_data[i] << '\n';
      return false;
    }
  }
  return true;
}

using ArrayPair = std::pair<mx::array, mx::array>;

ArrayPair Reference(
    const mx::array& q,
    const mx::array& k,
    float q_scale,
    float k_scale) {
  return {
      q_scale * mx::fast::rms_norm(q, std::nullopt, 1e-6f),
      k_scale * mx::fast::rms_norm(k, std::nullopt, 1e-6f)};
}

bool CheckCase(int rows, int width, float q_scale, float k_scale) {
  const std::size_t size =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(width);
  const auto q_values = MakeValues(size, 37);
  const auto k_values = MakeValues(size, 71);
  mx::array q = mx::astype(
      mx::array(q_values.data(), {rows, 1, width}, mx::float32),
      mx::bfloat16);
  mx::array k = mx::astype(
      mx::array(k_values.data(), {rows, 1, width}, mx::float32),
      mx::bfloat16);

  auto actual = sglang::mlx_qwen38::normalize_gated_delta_qk(
      q, k, q_scale, k_scale, 1e-6f);
  auto expected = Reference(q, k, q_scale, k_scale);
  return ExactEqual("q", actual.first, expected.first) &&
      ExactEqual("k", actual.second, expected.second);
}

bool CheckOutstandingOutputs() {
  constexpr int kRows = 16;
  constexpr int kWidth = 128;
  constexpr int kOutstanding = 12;
  const auto q_values = MakeValues(kRows * kWidth, 43);
  mx::array q = mx::astype(
      mx::array(q_values.data(), {1, 1, kRows, kWidth}, mx::float32),
      mx::bfloat16);

  std::vector<std::pair<ArrayPair, ArrayPair>> pending;
  pending.reserve(kOutstanding);
  for (int i = 0; i < kOutstanding; ++i) {
    const auto k_values = MakeValues(kRows * kWidth, 47 + i * 2);
    mx::array k = mx::astype(
        mx::array(k_values.data(), {1, 1, kRows, kWidth}, mx::float32),
        mx::bfloat16);
    const float inv = 1.0f / std::sqrt(static_cast<float>(kWidth));
    auto actual = sglang::mlx_qwen38::normalize_gated_delta_qk(
        q, k, inv * inv, inv, 1e-6f);
    pending.emplace_back(std::move(actual), Reference(q, k, inv * inv, inv));
  }

  for (auto& [actual, expected] : pending) {
    if (!ExactEqual("outstanding q", actual.first, expected.first) ||
        !ExactEqual("outstanding k", actual.second, expected.second)) {
      return false;
    }
  }
  return true;
}

bool CheckFullAttentionNormRope() {
  constexpr int kBatch = 1;
  constexpr int kQueryHeads = 24;
  constexpr int kKvHeads = 4;
  constexpr int kWidth = 256;
  constexpr int kRopeDims = 64;
  constexpr int kOffset = 6237;
  constexpr float kEps = 1e-6f;
  constexpr float kRopeTheta = 10000000.0f;

  const auto qg_values = MakeValues(
      static_cast<std::size_t>(kBatch * kQueryHeads * 2 * kWidth), 29);
  const auto k_values = MakeValues(
      static_cast<std::size_t>(kBatch * kKvHeads * kWidth), 61);
  const auto q_weight_values = MakeValues(kWidth, 7);
  const auto k_weight_values = MakeValues(kWidth, 11);
  mx::array qg = mx::astype(
      mx::array(
          qg_values.data(),
          {kBatch, 1, kQueryHeads, 2 * kWidth},
          mx::float32),
      mx::bfloat16);
  mx::array k = mx::astype(
      mx::array(
          k_values.data(), {kBatch, 1, kKvHeads, kWidth}, mx::float32),
      mx::bfloat16);
  mx::array q_weight = mx::astype(
      mx::array(q_weight_values.data(), {kWidth}, mx::float32),
      mx::bfloat16);
  mx::array k_weight = mx::astype(
      mx::array(k_weight_values.data(), {kWidth}, mx::float32),
      mx::bfloat16);

  auto actual = sglang::mlx_qwen38::full_attn_qk_norm_rope(
      qg,
      k,
      q_weight,
      k_weight,
      kEps,
      kRopeTheta,
      kRopeDims,
      kOffset);
  mx::array q = mx::split(qg, 2, -1)[0];
  q = mx::fast::rms_norm(q, q_weight, kEps);
  mx::array expected_k = mx::fast::rms_norm(k, k_weight, kEps);
  q = mx::transpose(q, {0, 2, 1, 3});
  expected_k = mx::transpose(expected_k, {0, 2, 1, 3});
  q = mx::fast::rope(
      q,
      kRopeDims,
      /*traditional=*/false,
      kRopeTheta,
      1.0f,
      kOffset);
  expected_k = mx::fast::rope(
      expected_k,
      kRopeDims,
      /*traditional=*/false,
      kRopeTheta,
      1.0f,
      kOffset);
  return ExactEqual("full-attention q", actual.first, q) &&
      ExactEqual("full-attention k", actual.second, expected_k);
}

}  // namespace

int main() {
  const float inv = 1.0f / std::sqrt(128.0f);
  if (!CheckCase(16, 128, inv * inv, inv) ||
      !CheckCase(3, 257, 0.03125f, 0.176776692f) ||
      !CheckOutstandingOutputs() || !CheckFullAttentionNormRope()) {
    return 1;
  }
  std::cout << "qwen38 q/k normalization and RoPE parity passed\n";
  return 0;
}
