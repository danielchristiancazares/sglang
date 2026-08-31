#include <cstddef>
#include <iostream>
#include <string_view>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

std::vector<float> MakeValues(std::size_t size, int multiplier) {
  std::vector<float> values(size);
  for (std::size_t i = 0; i < size; ++i) {
    const int centered =
        (static_cast<int>(i % 251) * multiplier + 17) % 101 - 50;
    values[i] = static_cast<float>(centered) / 64.0f;
  }
  return values;
}

bool ExactEqual(
    std::string_view label,
    const mx::array& actual,
    const mx::array& expected) {
  mx::array actual_float = mx::astype(actual, mx::float32);
  mx::array expected_float = mx::astype(expected, mx::float32);
  mx::eval(actual_float, expected_float);
  if (actual_float.shape() != expected_float.shape()) {
    std::cerr << label << " shape mismatch\n";
    return false;
  }
  const auto size = actual_float.size();
  const float* actual_data = actual_float.data<float>();
  const float* expected_data = expected_float.data<float>();
  for (std::size_t i = 0; i < size; ++i) {
    if (actual_data[i] != expected_data[i]) {
      std::cerr << label << " mismatch at " << i << ": " << actual_data[i]
                << " != " << expected_data[i] << '\n';
      return false;
    }
  }
  return true;
}

bool CheckCase(int batch, int kernel_size, int channels) {
  const std::size_t state_size = static_cast<std::size_t>(batch) *
      static_cast<std::size_t>(kernel_size - 1) * channels;
  const std::size_t qkv_size = static_cast<std::size_t>(batch) * channels;
  const std::size_t weight_size =
      static_cast<std::size_t>(channels) * kernel_size;
  const auto state_values = MakeValues(state_size, 29);
  const auto qkv_values = MakeValues(qkv_size, 43);
  const auto weight_values = MakeValues(weight_size, 61);

  mx::array state = mx::astype(
      mx::array(
          state_values.data(), {batch, kernel_size - 1, channels}, mx::float32),
      mx::bfloat16);
  mx::array qkv = mx::astype(
      mx::array(qkv_values.data(), {batch, 1, channels}, mx::float32),
      mx::bfloat16);
  mx::array weight = mx::astype(
      mx::array(weight_values.data(), {channels, kernel_size, 1}, mx::float32),
      mx::bfloat16);

  auto actual = sglang::mlx_qwen38::causal_conv_decode(state, qkv, weight);
  mx::array conv_input = mx::concatenate({state, qkv}, 1);
  mx::array expected_conv =
      mx::conv1d(conv_input, weight, 1, 0, 1, channels);
  mx::array expected_state = mx::slice(
      conv_input,
      {0, 1, 0},
      {batch, kernel_size, channels});

  return ExactEqual("convolution", actual.first, expected_conv) &&
      ExactEqual("state", actual.second, expected_state);
}

}  // namespace

int main() {
  if (!CheckCase(1, 4, 10240) || !CheckCase(2, 3, 257)) {
    return 1;
  }
  std::cout << "qwen38 causal-convolution decode parity passed\n";
  return 0;
}
