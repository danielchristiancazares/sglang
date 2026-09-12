#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/fast.h"
#include "mlx/mlx.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

std::vector<float> MakeValues(std::size_t count, int period, float scale) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered = static_cast<int>(index % period) - period / 2;
    values[index] = static_cast<float>(centered) * scale;
  }
  return values;
}

bool CheckParity(int query_tokens, int prefix_length, int cache_capacity) {
  constexpr int kQueryHeads = 24;
  constexpr int kKeyValueHeads = 4;
  constexpr int kHeadDimension = 256;
  const int active_length = prefix_length + query_tokens;
  if (active_length > cache_capacity) {
    return false;
  }

  std::vector<float> query_values = MakeValues(
      static_cast<std::size_t>(kQueryHeads) * query_tokens * kHeadDimension,
      113,
      0.001953125f);
  std::vector<float> key_values = MakeValues(
      static_cast<std::size_t>(kKeyValueHeads) * cache_capacity *
          kHeadDimension,
      127,
      0.0015625f);
  std::vector<float> value_values = MakeValues(
      static_cast<std::size_t>(kKeyValueHeads) * cache_capacity *
          kHeadDimension,
      109,
      0.00125f);

  mx::array queries = mx::astype(
      mx::array(
          query_values.data(),
          {1, kQueryHeads, query_tokens, kHeadDimension},
          mx::float32),
      mx::bfloat16);
  mx::array key_cache = mx::astype(
      mx::array(
          key_values.data(),
          {1, kKeyValueHeads, cache_capacity, kHeadDimension},
          mx::float32),
      mx::bfloat16);
  mx::array value_cache = mx::astype(
      mx::array(
          value_values.data(),
          {1, kKeyValueHeads, cache_capacity, kHeadDimension},
          mx::float32),
      mx::bfloat16);

  mx::array active_keys = mx::slice(
      key_cache,
      {0, 0, 0, 0},
      {1, kKeyValueHeads, active_length, kHeadDimension});
  mx::array active_values = mx::slice(
      value_cache,
      {0, 0, 0, 0},
      {1, kKeyValueHeads, active_length, kHeadDimension});
  mx::array expected = mx::fast::scaled_dot_product_attention(
      queries, active_keys, active_values, 0.0625f, "causal");
  mx::array actual = sglang::mlx_qwen38::fixed_prefill_attention(
      queries,
      key_cache,
      value_cache,
      prefix_length,
      active_length);
  mx::array expected_float = mx::astype(expected, mx::float32);
  mx::array actual_float = mx::astype(actual, mx::float32);
  mx::eval(expected, actual, expected_float, actual_float);

  float maximum_absolute_error = 0.0f;
  std::size_t non_finite = 0;
  const auto* expected_data = expected_float.data<float>();
  const auto* actual_data = actual_float.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    non_finite += !std::isfinite(actual_data[index]);
    maximum_absolute_error = std::max(
        maximum_absolute_error,
        std::abs(expected_data[index] - actual_data[index]));
  }
  std::cout << "query_tokens=" << query_tokens
            << " prefix_length=" << prefix_length
            << " max_abs=" << maximum_absolute_error
            << " non_finite=" << non_finite << '\n';
  return expected.shape() == actual.shape() && non_finite == 0 &&
      maximum_absolute_error <= 0.00390625f;
}

bool RejectsInvalidQueryDtype() {
  mx::array queries = mx::zeros({1, 24, 7, 256}, mx::float32);
  mx::array keys = mx::zeros({1, 4, 256, 256}, mx::bfloat16);
  mx::array values = mx::zeros({1, 4, 256, 256}, mx::bfloat16);
  try {
    (void)sglang::mlx_qwen38::fixed_prefill_attention(
        queries, keys, values, 1, 8);
  } catch (const std::runtime_error& error) {
    return std::string_view(error.what()) ==
        "invalid fixed prefill attention inputs";
  }
  return false;
}

} // namespace

int main() {
  if (!CheckParity(7, 1, 256) || !CheckParity(17, 65, 256) ||
      !CheckParity(64, 65, 256) || !CheckParity(7, 8191, 16384) ||
      !CheckParity(1024, 0, 1024) || !RejectsInvalidQueryDtype()) {
    return 1;
  }
  return 0;
}
