#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

int ParsePositive(const char *text, std::string_view name) {
  int value = 0;
  const std::string_view input(text);
  const auto [end, error] =
      std::from_chars(input.data(), input.data() + input.size(), value);
  if (error != std::errc() || end != input.data() + input.size() ||
      value <= 0) {
    throw std::runtime_error(std::string(name) + " must be positive");
  }
  return value;
}

std::uint32_t Next(std::uint64_t &state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return static_cast<std::uint32_t>(state);
}

mx::array Stock(const sglang::mlx_qwen38::QLinear &linear,
                const mx::array &input) {
  return mx::quantized_matmul(input, linear.w, linear.scales, linear.biases,
                              true, linear.group_size, linear.bits, "affine");
}

std::uint64_t Digest(const mx::array &output) {
  const auto *bytes =
      reinterpret_cast<const unsigned char *>(output.data<mx::bfloat16_t>());
  constexpr std::uint64_t kOffset = UINT64_C(14695981039346656037);
  constexpr std::uint64_t kPrime = UINT64_C(1099511628211);
  std::uint64_t digest = kOffset;
  for (std::size_t index = 0; index < output.size() * sizeof(mx::bfloat16_t);
       ++index) {
    digest ^= bytes[index];
    digest *= kPrime;
  }
  return digest;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      std::cerr << "usage: " << argv[0]
                << " stock|shared K N WARMUP ITERATIONS\n";
      return 2;
    }
    const std::string_view mode(argv[1]);
    if (mode != "stock" && mode != "shared") {
      throw std::runtime_error("mode must be stock or shared");
    }
    const int input_features = ParsePositive(argv[2], "K");
    const int output_features = ParsePositive(argv[3], "N");
    const int warmup = ParsePositive(argv[4], "WARMUP");
    const int iterations = ParsePositive(argv[5], "ITERATIONS");
    if (input_features % 512 != 0 || output_features % 16 != 0) {
      throw std::runtime_error("K must be divisible by 512 and N by 16");
    }

    const std::size_t packed_columns =
        static_cast<std::size_t>(input_features) / 8;
    const std::size_t packed_elements =
        static_cast<std::size_t>(output_features) * packed_columns;
    const std::size_t parameter_columns =
        static_cast<std::size_t>(input_features) / 64;
    const std::size_t parameter_elements =
        static_cast<std::size_t>(output_features) * parameter_columns;

    std::vector<std::uint32_t> packed(packed_elements);
    std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    for (auto &value : packed) {
      value = Next(state);
    }
    std::vector<float> scale_values(parameter_elements);
    std::vector<float> bias_values(parameter_elements);
    for (std::size_t index = 0; index < parameter_elements; ++index) {
      scale_values[index] =
          0.0025f + static_cast<float>(index % 17) * 0.000125f;
      bias_values[index] = (static_cast<float>(index % 11) - 5.0f) * 0.00025f;
    }
    std::vector<float> input_values(static_cast<std::size_t>(2) *
                                    input_features);
    for (std::size_t index = 0; index < input_values.size(); ++index) {
      input_values[index] =
          std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
    }

    const mx::array weights(packed.data(),
                            {output_features, static_cast<int>(packed_columns)},
                            mx::uint32);
    const mx::array scales = mx::astype(
        mx::array(scale_values.data(),
                  {output_features, static_cast<int>(parameter_columns)},
                  mx::float32),
        mx::bfloat16);
    const mx::array biases = mx::astype(
        mx::array(bias_values.data(),
                  {output_features, static_cast<int>(parameter_columns)},
                  mx::float32),
        mx::bfloat16);
    const mx::array input = mx::astype(
        mx::array(input_values.data(), {1, 2, input_features}, mx::float32),
        mx::bfloat16);
    const sglang::mlx_qwen38::QLinear linear{weights, scales, biases,
                                             64,      4,      true};
    mx::eval(weights, scales, biases, input);
    mx::synchronize();

    const mx::array expected = Stock(linear, input);
    const mx::array actual =
        sglang::mlx_qwen38::affine_q4_qmv_batch_two(linear, input);
    const mx::array expected_float = mx::astype(expected, mx::float32);
    const mx::array actual_float = mx::astype(actual, mx::float32);
    mx::eval(expected, actual, expected_float, actual_float);
    mx::synchronize();

    float maximum_absolute_error = 0.0f;
    std::size_t mismatches = 0;
    const float *expected_data = expected_float.data<float>();
    const float *actual_data = actual_float.data<float>();
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const float error = std::abs(expected_data[index] - actual_data[index]);
      maximum_absolute_error = std::max(maximum_absolute_error, error);
      mismatches += error != 0.0f;
    }

    const auto run = [&]() {
      return mode == "stock"
                 ? Stock(linear, input)
                 : sglang::mlx_qwen38::affine_q4_qmv_batch_two(linear, input);
    };
    mx::array output(0);
    for (int iteration = 0; iteration < warmup; ++iteration) {
      output = run();
      mx::eval(output);
      mx::synchronize();
    }
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      output = run();
      mx::eval(output);
      mx::synchronize();
    }
    const double milliseconds = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - started)
                                    .count() *
                                1000.0 / static_cast<double>(iterations);

    std::cout << std::fixed << std::setprecision(9) << "K=" << input_features
              << '\n'
              << "N=" << output_features << '\n'
              << "M=2\n"
              << "mode=" << mode << '\n'
              << "mean_ms=" << milliseconds << '\n'
              << "max_abs=" << maximum_absolute_error << '\n'
              << "mismatches=" << mismatches << '\n'
              << "output_digest_fnv1a64=" << std::hex << std::setw(16)
              << std::setfill('0') << Digest(output) << std::dec << '\n';
    return std::isfinite(maximum_absolute_error) ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
