#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

bool SameBfloat16(const mx::array& lhs, const mx::array& rhs) {
  mx::eval(lhs, rhs);
  if (lhs.shape() != rhs.shape() || lhs.dtype() != mx::bfloat16 ||
      rhs.dtype() != mx::bfloat16) {
    return false;
  }
  const auto* lhs_data = lhs.data<mx::bfloat16_t>();
  const auto* rhs_data = rhs.data<mx::bfloat16_t>();
  return std::equal(lhs_data, lhs_data + lhs.size(), rhs_data);
}

bool SameInt32(const mx::array& lhs, const mx::array& rhs) {
  mx::eval(lhs, rhs);
  if (lhs.shape() != rhs.shape() || lhs.dtype() != mx::int32 ||
      rhs.dtype() != mx::int32) {
    return false;
  }
  const auto* lhs_data = lhs.data<std::int32_t>();
  const auto* rhs_data = rhs.data<std::int32_t>();
  return std::equal(lhs_data, lhs_data + lhs.size(), rhs_data);
}

bool CheckInitialPromptAlignment() {
  const float hidden_values[12] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f};
  const std::int32_t tokens[4] = {10, 11, 12, 13};
  const float norm_values[3] = {1.0f, 0.5f, 2.0f};
  const mx::array hidden = mx::astype(
      mx::array(hidden_values, {1, 4, 3}, mx::float32), mx::bfloat16);
  const mx::array token_ids(tokens, {1, 4}, mx::int32);
  const mx::array norm = mx::astype(
      mx::array(norm_values, {3}, mx::float32), mx::bfloat16);

  auto [aligned_hidden, aligned_tokens] =
      sglang::mlx_qwen38::align_mtp_committed_history(
          hidden,
          token_ids,
          mx::array(0),
          /*has_previous_hidden=*/false,
          norm,
          1e-6f,
          /*post_norm_hidden=*/false);
  const mx::array expected_hidden = mx::slice(hidden, {0, 0, 0}, {1, 3, 3});
  const mx::array expected_tokens = mx::slice(token_ids, {0, 1}, {1, 4});
  std::cout << "initial prompt history rows=" << aligned_tokens.shape()[1]
            << '\n';
  return SameBfloat16(aligned_hidden, expected_hidden) &&
      SameInt32(aligned_tokens, expected_tokens);
}

bool CheckIncrementalAndPostNormAlignment() {
  const float hidden_values[6] = {4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  const float previous_values[3] = {1.0f, 2.0f, 3.0f};
  const float norm_values[3] = {1.0f, 0.5f, 2.0f};
  const std::int32_t tokens[2] = {20, 21};
  const mx::array hidden = mx::astype(
      mx::array(hidden_values, {1, 2, 3}, mx::float32), mx::bfloat16);
  const mx::array previous = mx::astype(
      mx::array(previous_values, {1, 3}, mx::float32), mx::bfloat16);
  const mx::array norm = mx::astype(
      mx::array(norm_values, {3}, mx::float32), mx::bfloat16);
  const mx::array token_ids(tokens, {1, 2}, mx::int32);
  const mx::array raw_expected = mx::concatenate(
      {mx::expand_dims(previous, 1),
       mx::slice(hidden, {0, 0, 0}, {1, 1, 3})},
      1);
  const mx::array normalized_expected =
      mx::fast::rms_norm(raw_expected, norm, 1e-6f);

  auto [raw_hidden, raw_tokens] =
      sglang::mlx_qwen38::align_mtp_committed_history(
          hidden,
          token_ids,
          previous,
          /*has_previous_hidden=*/true,
          norm,
          1e-6f,
          /*post_norm_hidden=*/false);
  auto [normalized_hidden, normalized_tokens] =
      sglang::mlx_qwen38::align_mtp_committed_history(
          hidden,
          token_ids,
          previous,
          /*has_previous_hidden=*/true,
          norm,
          1e-6f,
          /*post_norm_hidden=*/true);
  std::cout << "incremental history rows=" << raw_tokens.shape()[1] << '\n';
  return SameBfloat16(raw_hidden, raw_expected) &&
      SameBfloat16(normalized_hidden, normalized_expected) &&
      SameInt32(raw_tokens, token_ids) &&
      SameInt32(normalized_tokens, token_ids);
}

bool CheckSingleTokenAndInvalidInputs() {
  const float hidden_values[3] = {1.0f, 2.0f, 3.0f};
  const float norm_values[3] = {1.0f, 1.0f, 1.0f};
  const std::int32_t token = 10;
  const std::int32_t bad_tokens[2] = {10, 11};
  const mx::array hidden = mx::astype(
      mx::array(hidden_values, {1, 1, 3}, mx::float32), mx::bfloat16);
  const mx::array norm = mx::astype(
      mx::array(norm_values, {3}, mx::float32), mx::bfloat16);
  const mx::array token_ids(&token, {1, 1}, mx::int32);

  auto [empty_hidden, empty_tokens] =
      sglang::mlx_qwen38::align_mtp_committed_history(
          hidden,
          token_ids,
          mx::array(0),
          /*has_previous_hidden=*/false,
          norm,
          1e-6f,
          /*post_norm_hidden=*/true);
  bool rejects_bad_length = false;
  try {
    (void)sglang::mlx_qwen38::align_mtp_committed_history(
        hidden,
        mx::array(bad_tokens, {1, 2}, mx::int32),
        mx::array(0),
        /*has_previous_hidden=*/false,
        norm,
        1e-6f,
        /*post_norm_hidden=*/false);
  } catch (const std::runtime_error& error) {
    rejects_bad_length = std::string_view(error.what()) ==
        "invalid committed MTP history inputs";
  }
  return empty_hidden.shape() == mx::Shape{1, 0, 3} &&
      empty_tokens.shape() == mx::Shape{1, 0} && rejects_bad_length;
}

} // namespace

int main() {
  if (!CheckInitialPromptAlignment() ||
      !CheckIncrementalAndPostNormAlignment() ||
      !CheckSingleTokenAndInvalidInputs()) {
    return 1;
  }
  std::cout << "qwen38 committed MTP history alignment passed\n";
  return 0;
}
