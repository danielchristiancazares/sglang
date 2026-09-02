#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/mlx.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

bool CheckSelectedRowParity() {
  constexpr int kVocabularySize = 32;
  constexpr int kHiddenSize = 512;
  std::vector<float> values(kVocabularySize * kHiddenSize);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const int centered = static_cast<int>(index % 257) - 128;
    values[index] = static_cast<float>(centered) * 0.00390625f;
  }

  mx::array dense = mx::astype(
      mx::array(
          values.data(), {kVocabularySize, kHiddenSize}, mx::float32),
      mx::bfloat16);
  std::vector<mx::array> quantized = mx::quantize(dense, 64, 4, "affine");
  sglang::mlx_qwen38::QLinear embedding{
      quantized[0], quantized[1], quantized[2], 64, 4, true};
  const std::vector<std::int32_t> token_values{31, 0, 7, 7, 18, 3};
  mx::array signed_tokens(
      token_values.data(), {1, static_cast<int>(token_values.size())},
      mx::int32);
  mx::array unsigned_tokens = mx::astype(signed_tokens, mx::uint32);

  mx::array full = mx::dequantize(
      embedding.w,
      embedding.scales,
      embedding.biases,
      embedding.group_size,
      embedding.bits,
      "affine",
      std::nullopt,
      mx::bfloat16);
  mx::array expected = mx::take(full, signed_tokens, 0);
  mx::array signed_actual =
      sglang::mlx_qwen38::quantized_embedding_rows(embedding, signed_tokens);
  mx::array unsigned_actual = sglang::mlx_qwen38::quantized_embedding_rows(
      embedding, unsigned_tokens);
  mx::eval(expected, signed_actual, unsigned_actual);

  std::size_t signed_bit_mismatches = 0;
  std::size_t unsigned_bit_mismatches = 0;
  const auto* expected_bits = expected.data<std::uint16_t>();
  const auto* signed_actual_bits = signed_actual.data<std::uint16_t>();
  const auto* unsigned_actual_bits = unsigned_actual.data<std::uint16_t>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    signed_bit_mismatches += expected_bits[index] != signed_actual_bits[index];
    unsigned_bit_mismatches +=
        expected_bits[index] != unsigned_actual_bits[index];
  }
  std::cout << "selected_rows=" << token_values.size()
            << " signed_bit_mismatches=" << signed_bit_mismatches
            << " unsigned_bit_mismatches=" << unsigned_bit_mismatches
            << '\n';
  return expected.shape() == signed_actual.shape() &&
      expected.shape() == unsigned_actual.shape() &&
      signed_bit_mismatches == 0 && unsigned_bit_mismatches == 0;
}

bool RejectsInvalidTokenType() {
  mx::array weight = mx::zeros({8, 64}, mx::uint32);
  mx::array scales = mx::zeros({8, 8}, mx::bfloat16);
  mx::array biases = mx::zeros({8, 8}, mx::bfloat16);
  sglang::mlx_qwen38::QLinear embedding{
      weight, scales, biases, 64, 4, true};
  mx::array tokens = mx::zeros({1, 1}, mx::float32);
  try {
    (void)sglang::mlx_qwen38::quantized_embedding_rows(embedding, tokens);
  } catch (const std::runtime_error& error) {
    return std::string_view(error.what()) ==
        "invalid quantized embedding inputs";
  }
  return false;
}

} // namespace

int main() {
  if (!CheckSelectedRowParity() || !RejectsInvalidTokenType()) {
    return 1;
  }
  return 0;
}
