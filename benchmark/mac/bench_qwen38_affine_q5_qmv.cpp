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

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: " << argv[0] << " K N WARMUP ITERATIONS\n";
      return 2;
    }
    const int input_features = ParsePositive(argv[1], "K");
    const int output_features = ParsePositive(argv[2], "N");
    const int warmup = ParsePositive(argv[3], "WARMUP");
    const int iterations = ParsePositive(argv[4], "ITERATIONS");
    if (input_features % 512 != 0 || output_features % 16 != 0) {
      throw std::runtime_error("K must be divisible by 512 and N by 16");
    }

    const std::size_t packed_columns =
        static_cast<std::size_t>(input_features) * 5 / 32;
    const std::size_t packed_elements =
        static_cast<std::size_t>(output_features) * packed_columns;
    const std::size_t parameter_columns =
        static_cast<std::size_t>(input_features) / 64;
    const std::size_t parameter_elements =
        static_cast<std::size_t>(output_features) * parameter_columns;

    std::vector<std::uint32_t> packed(packed_elements);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
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
    std::vector<float> input_values(input_features);
    for (int index = 0; index < input_features; ++index) {
      input_values[static_cast<std::size_t>(index)] =
          std::sin(static_cast<float>(index) * 0.017f) * 0.25f;
    }

    mx::array weights(packed.data(),
                      {output_features, static_cast<int>(packed_columns)},
                      mx::uint32);
    mx::array scales = mx::astype(
        mx::array(scale_values.data(),
                  {output_features, static_cast<int>(parameter_columns)},
                  mx::float32),
        mx::bfloat16);
    mx::array biases = mx::astype(
        mx::array(bias_values.data(),
                  {output_features, static_cast<int>(parameter_columns)},
                  mx::float32),
        mx::bfloat16);
    mx::array input = mx::astype(
        mx::array(input_values.data(), {1, 1, input_features}, mx::float32),
        mx::bfloat16);
    sglang::mlx_qwen38::QLinear linear{weights, scales, biases, 64, 5, true};
    mx::eval(weights, scales, biases, input);
    mx::synchronize();

    mx::array output(0);
    for (int iteration = 0; iteration < warmup; ++iteration) {
      output = sglang::mlx_qwen38::affine_q5_qmv_batch_one(linear, input);
      mx::eval(output);
      mx::synchronize();
    }

    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
      output = sglang::mlx_qwen38::affine_q5_qmv_batch_one(linear, input);
      mx::eval(output);
      mx::synchronize();
    }
    const double elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      started)
            .count();
    const double milliseconds =
        elapsed_seconds * 1000.0 / static_cast<double>(iterations);
    const double streamed_bytes = static_cast<double>(output_features) *
                                  input_features * (5.0 / 8.0 + 4.0 / 64.0);
    const double gigabytes_per_second = streamed_bytes / (milliseconds * 1.0e6);
    const auto *output_values = output.data<mlx::core::bfloat16_t>();
    const auto *output_bytes =
        reinterpret_cast<const unsigned char *>(output_values);
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    std::uint64_t digest = kFnvOffset;
    for (std::size_t index = 0;
         index < output.size() * sizeof(mlx::core::bfloat16_t); ++index) {
      digest ^= output_bytes[index];
      digest *= kFnvPrime;
    }
    const float first = output_values[0];

    std::cout << std::fixed << std::setprecision(9) << "K=" << input_features
              << '\n'
              << "N=" << output_features << '\n'
              << "warmup=" << warmup << '\n'
              << "iterations=" << iterations << '\n'
              << "mean_ms=" << milliseconds << '\n'
              << "effective_gbps=" << gigabytes_per_second << '\n'
              << "first=" << first << '\n'
              << "output_digest_fnv1a64=" << std::hex << std::setw(16)
              << std::setfill('0') << digest << std::dec << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
