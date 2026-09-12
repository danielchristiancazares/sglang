#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

constexpr int kHeadDim = 128;
constexpr int kFrequencyCount = kHeadDim / 2;
constexpr double kBase = 10000000.0;
constexpr double kFactor = 32.0;
constexpr double kOriginalPositions = 8192.0;

std::vector<float> MakeFrequencies() {
  const auto correction_dimension = [](double rotations) {
    return kHeadDim *
           std::log(kOriginalPositions / (rotations * 2.0 * std::numbers::pi)) /
           (2.0 * std::log(kBase));
  };
  const int low =
      std::max(static_cast<int>(std::floor(correction_dimension(32.0))), 0);
  const int high = std::min(
      static_cast<int>(std::ceil(correction_dimension(1.0))), kHeadDim - 1);
  std::vector<float> frequencies(kFrequencyCount);
  for (int index = 0; index < kFrequencyCount; ++index) {
    const double extrapolated = 1.0 / std::pow(kBase, (2.0 * index) / kHeadDim);
    const double interpolated = extrapolated / kFactor;
    const double ramp = std::clamp((static_cast<double>(index) - low) /
                                       static_cast<double>(high - low),
                                   0.0, 1.0);
    frequencies[static_cast<std::size_t>(index)] =
        static_cast<float>(extrapolated * (1.0 - ramp) + interpolated * ramp);
  }
  return frequencies;
}

bool CheckOffset(int offset, float tolerance) {
  constexpr int kLength = 2;
  std::vector<float> input(kLength * kHeadDim);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = std::sin(static_cast<float>(index) * 0.071f) * 0.25f;
  }
  mx::array source(input.data(), {1, 1, kLength, kHeadDim}, mx::float32);
  mx::array actual = sglang::mlx_qwen38::dspark_yarn_rope(source, offset);
  mx::eval(actual);

  const std::vector<float> frequencies = MakeFrequencies();
  const float attention_factor =
      static_cast<float>(0.1 * std::log(kFactor) + 1.0);
  std::vector<float> expected(input.size());
  for (int row = 0; row < kLength; ++row) {
    for (int index = 0; index < kFrequencyCount; ++index) {
      const std::size_t first =
          static_cast<std::size_t>(row * kHeadDim + index);
      const std::size_t second = first + kFrequencyCount;
      const float angle = static_cast<float>(offset + row) * frequencies[index];
      const float cosine = std::cos(angle);
      const float sine = std::sin(angle);
      expected[first] =
          (input[first] * cosine - input[second] * sine) * attention_factor;
      expected[second] =
          (input[second] * cosine + input[first] * sine) * attention_factor;
    }
  }

  float maximum_absolute_error = 0.0f;
  const float *actual_data = actual.data<float>();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum_absolute_error = std::max(
        maximum_absolute_error, std::abs(expected[index] - actual_data[index]));
  }
  std::cout << "offset=" << offset << " max_abs=" << maximum_absolute_error
            << '\n';
  return std::isfinite(maximum_absolute_error) &&
         maximum_absolute_error <= tolerance;
}

bool RejectsInvalidWidth() {
  mx::array input = mx::zeros({1, 1, 1, 64}, mx::float32);
  try {
    (void)sglang::mlx_qwen38::dspark_yarn_rope(input, 0);
  } catch (const std::runtime_error &error) {
    return std::string_view(error.what()) == "invalid DSpark YaRN RoPE input";
  }
  return false;
}

bool CheckConfidence() {
  const std::vector<float> hidden_values = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> markov_values = {0.5f, -0.5f};
  const std::vector<float> weight_values = {0.25f, -0.5f, 1.0f};
  const std::vector<float> bias_values = {0.125f};
  mx::array hidden = mx::astype(
      mx::array(hidden_values.data(), {1, 2, 2}, mx::float32), mx::bfloat16);
  mx::array markov = mx::astype(
      mx::array(markov_values.data(), {1, 2, 1}, mx::float32), mx::bfloat16);
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {1, 3}, mx::float32), mx::bfloat16);
  mx::array bias =
      mx::astype(mx::array(bias_values.data(), {1}, mx::float32), mx::bfloat16);
  mx::array confidence =
      sglang::mlx_qwen38::dspark_confidence(hidden, markov, weight, bias);
  mx::eval(confidence);

  const float kExpected[] = {
      1.0f / (1.0f + std::exp(0.125f)),
      1.0f / (1.0f + std::exp(1.625f)),
  };
  if (confidence.shape() != mx::Shape{1, 2} ||
      confidence.dtype() != mx::float32) {
    return false;
  }
  const float *const actual = confidence.data<float>();
  return std::abs(actual[0] - kExpected[0]) <= 1e-6f &&
         std::abs(actual[1] - kExpected[1]) <= 1e-6f;
}

bool RejectsInvalidConfidenceShape() {
  mx::array hidden = mx::zeros({1, 2, 2}, mx::bfloat16);
  mx::array markov = mx::zeros({1, 1, 1}, mx::bfloat16);
  mx::array weight = mx::zeros({1, 3}, mx::bfloat16);
  mx::array bias = mx::zeros({1}, mx::bfloat16);
  try {
    (void)sglang::mlx_qwen38::dspark_confidence(hidden, markov, weight, bias);
  } catch (const std::runtime_error &error) {
    return std::string_view(error.what()) == "invalid DSpark confidence inputs";
  }
  return false;
}

bool CheckConfidenceBudget() {
  constexpr float kCostRatio = 1.694f;
  const float high_confidence[] = {
      1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  const float low_confidence[] = {
      0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  return sglang::mlx_qwen38::dspark_select_verify_draft_tokens(
             high_confidence, 7, kCostRatio) == 7 &&
         sglang::mlx_qwen38::dspark_select_verify_draft_tokens(
             low_confidence, 7, kCostRatio) == 1;
}

bool RejectsInvalidConfidenceBudget() {
  const float invalid_confidence[] = {
      1.01f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  try {
    (void)sglang::mlx_qwen38::dspark_select_verify_draft_tokens(
        invalid_confidence, 7, 1.694f);
  } catch (const std::runtime_error &error) {
    return std::string_view(error.what()) ==
           "invalid DSpark confidence budget inputs";
  }
  return false;
}

} // namespace

int main() {
  try {
    const bool origin_matches = CheckOffset(0, 2e-5f);
    const bool yarn_range_matches = CheckOffset(9000, 2e-4f);
    const bool context_limit_matches = CheckOffset(131071, 3e-3f);
    if (!origin_matches || !yarn_range_matches || !context_limit_matches ||
        !RejectsInvalidWidth() || !CheckConfidence() ||
        !RejectsInvalidConfidenceShape() || !CheckConfidenceBudget() ||
        !RejectsInvalidConfidenceBudget()) {
      return 1;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "qwen38 DSpark YaRN/confidence parity passed\n";
  return 0;
}
