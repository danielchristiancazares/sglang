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

bool CheckQ8Parity(int query_tokens, int prefix_length, int cache_capacity) {
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
  mx::array dense_keys = mx::astype(
      mx::array(
          key_values.data(),
          {1, kKeyValueHeads, cache_capacity, kHeadDimension},
          mx::float32),
      mx::bfloat16);
  mx::array dense_values = mx::astype(
      mx::array(
          value_values.data(),
          {1, kKeyValueHeads, cache_capacity, kHeadDimension},
          mx::float32),
      mx::bfloat16);
  std::vector<mx::array> quantized_keys =
      mx::quantize(dense_keys, 64, 8, "affine");
  std::vector<mx::array> quantized_values =
      mx::quantize(dense_values, 64, 8, "affine");
  if (quantized_keys.size() != 3 || quantized_values.size() != 3) {
    return false;
  }
  mx::array dequantized_keys = mx::dequantize(
      quantized_keys[0],
      quantized_keys[1],
      quantized_keys[2],
      64,
      8,
      "affine",
      std::nullopt,
      mx::bfloat16);
  mx::array dequantized_values = mx::dequantize(
      quantized_values[0],
      quantized_values[1],
      quantized_values[2],
      64,
      8,
      "affine",
      std::nullopt,
      mx::bfloat16);
  mx::array active_keys = mx::slice(
      dequantized_keys,
      {0, 0, 0, 0},
      {1, kKeyValueHeads, active_length, kHeadDimension});
  mx::array active_values = mx::slice(
      dequantized_values,
      {0, 0, 0, 0},
      {1, kKeyValueHeads, active_length, kHeadDimension});
  mx::array expected = mx::fast::scaled_dot_product_attention(
      queries, active_keys, active_values, 0.0625f, "causal");
  mx::array actual = sglang::mlx_qwen38::fixed_q8_attention(
      queries,
      quantized_keys[0],
      quantized_keys[1],
      quantized_keys[2],
      quantized_values[0],
      quantized_values[1],
      quantized_values[2],
      prefix_length,
      active_length);
  mx::array expected_float = mx::astype(expected, mx::float32);
  mx::array actual_float = mx::astype(actual, mx::float32);
  mx::eval(expected_float, actual_float);

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
  std::cout << "q8_query_tokens=" << query_tokens
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

bool CheckCacheGrowthPolicy() {
  using sglang::mlx_qwen38::attention_cache_append_requires_large_growth;
  using sglang::mlx_qwen38::attention_cache_growth_capacity;
  using sglang::mlx_qwen38::post_growth_prefill_chunk_size;
  using sglang::mlx_qwen38::serialized_attention_cache_growth_chunk_size;
  const bool expected =
      attention_cache_growth_capacity(0, 1, 0) == 256 &&
      attention_cache_growth_capacity(256, 257, 0) == 512 &&
      attention_cache_growth_capacity(8192, 8193, 131072) == 131072 &&
      attention_cache_growth_capacity(131072, 131073, 131072) == 262144 &&
      attention_cache_growth_capacity(8192, 9000, 8000) == 16384 &&
      serialized_attention_cache_growth_chunk_size(32768, 32768, 1024) ==
          1024 &&
      serialized_attention_cache_growth_chunk_size(65024, 65536, 1024) ==
          512 &&
      serialized_attention_cache_growth_chunk_size(65535, 65536, 1024) == 1 &&
      serialized_attention_cache_growth_chunk_size(65536, 65536, 1024) == 1 &&
      serialized_attention_cache_growth_chunk_size(65537, 131072, 1024) ==
          1024 &&
      !attention_cache_append_requires_large_growth(32768, 32768, 1024) &&
      !attention_cache_append_requires_large_growth(65535, 65536, 1) &&
      attention_cache_append_requires_large_growth(65536, 65536, 1) &&
      attention_cache_append_requires_large_growth(65024, 65536, 1024) &&
      !attention_cache_append_requires_large_growth(65537, 131072, 1024) &&
      post_growth_prefill_chunk_size(
          65536, 65536, 65536, 65536, 1024, 512) == 1024 &&
      post_growth_prefill_chunk_size(
          65537, 131072, 65536, 65536, 1024, 512) == 1024 &&
      post_growth_prefill_chunk_size(
          65538, 131072, 65537, 131072, 1024, 0) == 1024 &&
      post_growth_prefill_chunk_size(
          65538, 131072, 65537, 131072, 1024, 512) == 512 &&
      post_growth_prefill_chunk_size(
          65538, 131072, 65537, 131072, 256, 512) == 256;
  bool rejected_invalid = false;
  try {
    (void)attention_cache_growth_capacity(256, 256, 0);
  } catch (const std::runtime_error& error) {
    rejected_invalid = std::string_view(error.what()) ==
        "invalid attention cache growth request";
  }
  bool rejected_invalid_extent = false;
  try {
    (void)serialized_attention_cache_growth_chunk_size(257, 256, 1);
  } catch (const std::runtime_error& error) {
    rejected_invalid_extent = std::string_view(error.what()) ==
        "invalid attention cache extent";
  }
  bool rejected_invalid_append = false;
  try {
    (void)attention_cache_append_requires_large_growth(0, 0, 0);
  } catch (const std::runtime_error& error) {
    rejected_invalid_append = std::string_view(error.what()) ==
        "invalid attention cache extent";
  }
  bool rejected_invalid_post_growth = false;
  try {
    (void)post_growth_prefill_chunk_size(
        65538, 131072, 65537, 131072, 1024, 1025);
  } catch (const std::runtime_error& error) {
    rejected_invalid_post_growth = std::string_view(error.what()) ==
        "invalid post-growth prefill chunk request";
  }
  std::cout << "cache_growth_policy=" << (expected ? "pass" : "fail")
            << " rejected_invalid=" << (rejected_invalid ? 1 : 0)
            << " rejected_extent=" << (rejected_invalid_extent ? 1 : 0)
            << " rejected_append=" << (rejected_invalid_append ? 1 : 0)
            << " rejected_post_growth="
            << (rejected_invalid_post_growth ? 1 : 0)
            << '\n';
  return expected && rejected_invalid && rejected_invalid_extent &&
      rejected_invalid_append && rejected_invalid_post_growth;
}

bool CheckAppendOnlyRollbackInvariant() {
  constexpr int kPromptLength = 255;
  constexpr int kInitialCapacity = 256;
  constexpr int kGrownCapacity = 512;
  constexpr int kFirstAppendLength = 3;
  constexpr int kReplacementLength = 2;

  std::vector<float> prompt_values(kPromptLength);
  for (int index = 0; index < kPromptLength; ++index) {
    prompt_values[static_cast<std::size_t>(index)] =
        static_cast<float>((index % 31) - 15) * 0.125f;
  }
  const float first_append[kFirstAppendLength] = {31.0f, 32.0f, 33.0f};
  const float replacement[kReplacementLength] = {-41.0f, -42.0f};

  mx::array cache = mx::zeros({1, 1, kInitialCapacity, 1}, mx::bfloat16);
  mx::array prompt = mx::astype(
      mx::array(
          prompt_values.data(), {1, 1, kPromptLength, 1}, mx::float32),
      mx::bfloat16);
  cache = mx::slice_update(
      cache, prompt, {0, 0, 0, 0}, {1, 1, kPromptLength, 1});
  mx::eval(cache);

  mx::array grown = mx::zeros({1, 1, kGrownCapacity, 1}, mx::bfloat16);
  grown = mx::slice_update(
      grown,
      mx::slice(cache, {0, 0, 0, 0}, {1, 1, kPromptLength, 1}),
      {0, 0, 0, 0},
      {1, 1, kPromptLength, 1});
  cache = std::move(grown);
  mx::eval(cache);

  cache = mx::slice_update(
      cache,
      mx::astype(
          mx::array(
              first_append, {1, 1, kFirstAppendLength, 1}, mx::float32),
          mx::bfloat16),
      {0, 0, kPromptLength, 0},
      {1, 1, kPromptLength + kFirstAppendLength, 1});
  mx::eval(cache);

  // Restoring an append-only snapshot changes only the logical length. The
  // following write begins at that saved length and replaces the abandoned
  // suffix while retaining the original prefix in the current allocation.
  cache = mx::slice_update(
      cache,
      mx::astype(
          mx::array(
              replacement, {1, 1, kReplacementLength, 1}, mx::float32),
          mx::bfloat16),
      {0, 0, kPromptLength, 0},
      {1, 1, kPromptLength + kReplacementLength, 1});
  mx::array active = mx::astype(
      mx::slice(
          cache,
          {0, 0, 0, 0},
          {1, 1, kPromptLength + kReplacementLength, 1}),
      mx::float32);
  mx::eval(active);

  const float* const actual = active.data<float>();
  bool prefix_exact = true;
  for (int index = 0; index < kPromptLength; ++index) {
    prefix_exact = prefix_exact &&
        actual[index] == prompt_values[static_cast<std::size_t>(index)];
  }
  const bool suffix_exact =
      actual[kPromptLength] == replacement[0] &&
      actual[kPromptLength + 1] == replacement[1];
  std::cout << "append_only_prefix=" << (prefix_exact ? "exact" : "changed")
            << " replacement_suffix=" << (suffix_exact ? "exact" : "changed")
            << '\n';
  return prefix_exact && suffix_exact;
}

} // namespace

int main() {
  if (!CheckParity(7, 1, 256) || !CheckParity(17, 65, 256) ||
      !CheckParity(64, 65, 256) || !CheckParity(7, 8191, 16384) ||
      !CheckParity(1024, 0, 1024) || !CheckQ8Parity(17, 65, 256) ||
      !CheckQ8Parity(1024, 0, 1024) ||
      !CheckQ8Parity(1, 8191, 16384) || !RejectsInvalidQueryDtype() ||
      !CheckCacheGrowthPolicy() || !CheckAppendOnlyRollbackInvariant()) {
    return 1;
  }
  return 0;
}
