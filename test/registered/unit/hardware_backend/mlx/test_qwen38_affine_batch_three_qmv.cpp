#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;
namespace qwen = sglang::mlx_qwen38;

namespace {

qwen::QLinear MakeLinear(int k, int n, std::uint64_t state) {
  std::vector<std::uint32_t> weights(static_cast<std::size_t>(n) * k / 8);
  for (auto& value : weights) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    value = static_cast<std::uint32_t>(state);
  }
  std::vector<float> scales(static_cast<std::size_t>(n) * k / 64);
  std::vector<float> biases(scales.size());
  for (std::size_t i = 0; i < scales.size(); ++i) {
    scales[i] = 0.0025f + static_cast<float>(i % 17) * 0.000125f;
    biases[i] = (static_cast<float>(i % 11) - 5.0f) * 0.00025f;
  }
  return {mx::array(weights.data(), {n, k / 8}, mx::uint32),
          mx::astype(mx::array(scales.data(), {n, k / 64}, mx::float32),
                     mx::bfloat16),
          mx::astype(mx::array(biases.data(), {n, k / 64}, mx::float32),
                     mx::bfloat16), 64, 4, true};
}

mx::array Input(int k) {
  std::vector<float> values(3 * k);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.3125f;
  }
  return mx::astype(mx::array(values.data(), {1, 3, k}, mx::float32),
                    mx::bfloat16);
}

mx::array Stock(const qwen::QLinear& linear, const mx::array& x) {
  return mx::quantized_matmul(x, linear.w, linear.scales, linear.biases,
                              true, 64, 4, "affine");
}

mx::array Evaluate(const qwen::QLinear& gate, const qwen::QLinear& up,
                   const mx::array& x, bool mlp, int mode) {
  if (mode == 1) {
    return mlp ? qwen::silu(qwen::affine_q4_qmv_batch_three(gate, x)) *
                       qwen::affine_q4_qmv_batch_three(up, x)
                 : qwen::affine_q4_qmv_batch_three(gate, x);
  }
  if (mode == 0) {
    return mlp ? qwen::silu(Stock(gate, x)) * Stock(up, x) : Stock(gate, x);
  }
  std::vector<mx::array> rows;
  for (int row = 0; row < 3; ++row) {
    const auto single = mx::slice(x, {0, row, 0}, {1, row + 1, x.shape()[2]});
    rows.push_back(mlp ? qwen::silu(Stock(gate, single)) * Stock(up, single)
                         : Stock(gate, single));
  }
  return mx::concatenate(rows, 1);
}

void RequireExact(const mx::array& expected, const mx::array& actual) {
  mx::eval(expected, actual);
  if (expected.shape() != actual.shape() || expected.dtype() != mx::bfloat16 ||
      actual.dtype() != mx::bfloat16) {
    throw std::runtime_error("output shape or dtype mismatch");
  }
  const auto* a = expected.data<mx::bfloat16_t>();
  const auto* b = actual.data<mx::bfloat16_t>();
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (a[i] != b[i]) {
      std::cerr << "mismatch index=" << i << " expected=" << float(a[i])
                << " actual=" << float(b[i]) << '\n';
      throw std::runtime_error("batch-three output differs from serial MLX");
    }
  }
}

std::uint64_t Digest(const mx::array& output) {
  mx::eval(output);
  const auto* bytes = reinterpret_cast<const unsigned char*>(
      output.data<mx::bfloat16_t>());
  std::uint64_t result = UINT64_C(14695981039346656037);
  for (std::size_t i = 0; i < output.nbytes(); ++i) {
    result = (result ^ bytes[i]) * UINT64_C(1099511628211);
  }
  return result;
}

void Check(int k, int n, bool mlp, bool boundary = false) {
  auto gate = MakeLinear(k, n, UINT64_C(0xd1b54a32d192ed03));
  auto up = MakeLinear(k, n, UINT64_C(0x9e3779b97f4a7c15));
  auto x = Input(k);
  if (boundary) {
    gate.scales = mx::zeros({n, k / 64}, mx::bfloat16);
    up.scales = gate.scales;
    gate.biases = mx::full({n, k / 64}, -6.84375f, mx::bfloat16);
    up.biases = mx::ones({n, k / 64}, mx::bfloat16);
    std::vector<float> values(3 * k, 0.0f);
    values[0] = 1.0f;
    values[k] = 1.0f;
    values[k + 1] = 1.0f / 256.0f;
    values[k + 2] = -1.0f;
    values[2 * k] = -1.0f;
    values[2 * k + 1] = -1.0f / 256.0f;
    values[2 * k + 2] = 1.0f;
    x = mx::astype(mx::array(values.data(), {1, 3, k}, mx::float32),
                   mx::bfloat16);
  }
  const auto expected = Evaluate(gate, up, x, mlp, 2);
  const auto actual = Evaluate(gate, up, x, mlp, 1);
  RequireExact(expected, actual);
  // Replay the same compiled Metal specialization with a different input.
  const auto next = mx::negative(x);
  RequireExact(Evaluate(gate, up, next, mlp, 2),
               Evaluate(gate, up, next, mlp, 1));
  RequireExact(expected, actual);
  if (!mlp) {
    if (setenv("SGLANG_MLX_NATIVE_Q4_BATCH_THREE_QMV", "1", 1) != 0) {
      throw std::runtime_error("cannot set dispatch switch");
    }
    RequireExact(actual, gate(x));
    for (int rows : {1, 2, 4}) {
      const auto other = mx::full({1, rows, k}, 0.25f, mx::bfloat16);
      RequireExact(Stock(gate, other), gate(other));
    }
    if (unsetenv("SGLANG_MLX_NATIVE_Q4_BATCH_THREE_QMV") != 0) {
      throw std::runtime_error("cannot clear dispatch switch");
    }
    RequireExact(Stock(gate, x), gate(x));
  }
  std::cout << "K=" << k << " N=" << n << " mlp=" << mlp
            << " boundary=" << boundary << " exact=1 digest=" << std::hex
            << Digest(actual) << std::dec << '\n';
}

void CheckInvalid() {
  const auto valid = MakeLinear(512, 64, 42);
  const auto x = Input(512);
  const auto rejects = [](const qwen::QLinear& q, const mx::array& input) {
    try {
      (void)qwen::affine_q4_qmv_batch_three(q, input);
    } catch (const std::runtime_error&) {
      return;
    }
    throw std::runtime_error("invalid Q4 input accepted");
  };
  for (const auto& invalid : {mx::zeros({3, 512}, mx::bfloat16),
                              mx::zeros({2, 3, 512}, mx::bfloat16),
                              mx::zeros({1, 2, 512}, mx::bfloat16),
                              mx::zeros({1, 3, 256}, mx::bfloat16),
                              mx::astype(x, mx::float32)}) {
    rejects(valid, invalid);
  }
  for (int field = 0; field < 8; ++field) {
    auto invalid = valid;
    switch (field) {
      case 0: invalid.valid = false; break;
      case 1: invalid.bits = 5; break;
      case 2: invalid.group_size = 32; break;
      case 3: invalid.w = mx::astype(valid.w, mx::int32); break;
      case 4: invalid.w = mx::zeros({63, 64}, mx::uint32); break;
      case 5: invalid.scales = mx::zeros({64, 7}, mx::bfloat16); break;
      case 6: invalid.biases = mx::zeros({512}, mx::bfloat16); break;
      case 7: invalid.biases = mx::astype(valid.biases, mx::float32); break;
    }
    rejects(invalid, x);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 5) {
      const int k = std::stoi(argv[1]);
      const int n = std::stoi(argv[2]);
      const bool mlp = std::stoi(argv[3]) != 0;
      const int mode = std::stoi(argv[4]);
      if (k <= 0 || n <= 0 || mode < 0 || mode > 2) return 2;
      auto gate = MakeLinear(k, n, 42);
      const auto up = MakeLinear(k, n, 123);
      const auto x = Input(k);
      mx::array output(0);
      for (int i = 0; i < 128; ++i) {
        output = Evaluate(gate, up, x, mlp, mode);
        mx::eval(output);
        mx::synchronize();
      }
      const auto started = std::chrono::steady_clock::now();
      for (int i = 0; i < 100; ++i) {
        output = Evaluate(gate, up, x, mlp, mode);
        mx::eval(output);
        mx::synchronize();
      }
      const double ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count() / 100;
      std::cout << std::setprecision(12) << "K=" << k << " N=" << n
                << " mlp=" << mlp << " mode=" << mode << " ms=" << ms
                << " digest=" << std::hex << Digest(output) << std::dec << '\n';
      return 0;
    }
    if (argc != 1) return 2;
    CheckInvalid();
    for (const auto& [k, n] : {std::pair{512, 64}, std::pair{5120, 128},
                              std::pair{5120, 17408},
                              std::pair{17408, 5120}}) {
      Check(k, n, false);
      Check(k, n, true);
    }
    Check(512, 64, true, true);
    Check(5120, 248320, false);
    std::cout << "batch-three Q4 parity, dispatch and replay passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
