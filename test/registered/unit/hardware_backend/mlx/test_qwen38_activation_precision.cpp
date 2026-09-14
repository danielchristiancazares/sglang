#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "mlx/mlx.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;
namespace native = sglang::mlx_qwen38;

namespace {
void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void CheckConversion() {
  const auto dtype = native::activation_dtype();
  const bool half = dtype == mx::float16;
  const float values[] = {0.0f, -0.0f, 1.0f, -1.0f, 0x1p-24f,
                          0x1p-25f, -0x1p-25f, 0x1p-100f};
  const auto source = mx::astype(mx::array(values, {8}, mx::float32), mx::bfloat16);
  const auto converted = native::convert_activation_parameter(source, "boundary");
  Require(converted.value.dtype() == dtype, "activation dtype was not applied");
  const auto expected = mx::astype(source, dtype);
  auto equal = mx::all(mx::equal(converted.value, expected));
  mx::eval(equal);
  Require(equal.item<bool>(), "parameter conversion differs from MLX cast");
  Require(converted.changed_values == (half ? 3U : 0U), "incorrect conversion count");
  Require(converted.maximum_absolute_error == (half ? 0x1p-25f : 0.0f),
          "incorrect conversion maximum error");
  const std::uint32_t codes[] = {0x01234567U, 0x89abcdefU, 0xffffffffU};
  const auto packed = native::convert_activation_parameter(
      mx::array(codes, {3}, mx::uint32), "packed");
  mx::eval(packed.value);
  Require(packed.value.dtype() == mx::uint32 && packed.changed_values == 0 &&
              packed.maximum_absolute_error == 0.0f &&
              std::equal(codes, codes + 3, packed.value.data<std::uint32_t>()),
          "packed weight codes changed");
  const auto fp32 = native::convert_activation_parameter(
      mx::array({0x1p100f}, mx::float32), "fp32-state");
  mx::eval(fp32.value);
  Require(fp32.value.dtype() == mx::float32 && fp32.value.item<float>() == 0x1p100f,
          "FP32 state was narrowed");
  const auto empty = native::convert_activation_parameter(
      mx::zeros({0}, mx::bfloat16), "empty");
  Require(empty.value.size() == 0 && empty.value.dtype() == dtype &&
              empty.changed_values == 0, "empty conversion failed");
  if (half) {
    for (float value : {65536.0f, -65536.0f, std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
      bool rejected = false;
      try {
        (void)native::convert_activation_parameter(
            mx::astype(mx::array(value), mx::bfloat16), "invalid-range");
      } catch (const std::runtime_error&) { rejected = true; }
      Require(rejected, "unsupported FP16 parameter was accepted");
    }
  }
  Require(setenv("SGLANG_MLX_NATIVE_ACTIVATION_DTYPE", half ? "bfloat16" : "float16", 1) == 0,
          "cannot test immutable dtype");
  Require(native::activation_dtype() == dtype, "kernel dtype changed during process lifetime");
  std::cout << "parameter_conversion=pass packed_codes=unchanged immutable_dtype=pass\n";
}

void EqualFinite(const mx::array& expected, const mx::array& actual, const char* label) {
  auto lhs = mx::astype(expected, mx::float32);
  auto rhs = mx::astype(actual, mx::float32);
  mx::eval(lhs, rhs);
  Require(lhs.shape() == rhs.shape(), "precision test shape mismatch");
  std::size_t mismatches = 0;
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    const float a = lhs.data<float>()[i], b = rhs.data<float>()[i];
    if (!std::isfinite(a) || !std::isfinite(b) || a != b) {
      if (mismatches++ < 4) std::cerr << label << " index=" << i << " expected=" << a
                                     << " actual=" << b << '\n';
    }
  }
  std::cout << label << " elements=" << lhs.size() << " mismatches=" << mismatches << '\n';
  Require(mismatches == 0, "finite activation domain differs from MLX");
}

void CheckHalfDomain() {
  if (native::activation_dtype() != mx::float16) return;
  constexpr int count = 65536, k = 512, groups = k / 64;
  std::vector<std::uint16_t> codes(count);
  for (int i = 0; i < count; ++i)
    codes[i] = (i & 0x7c00) == 0x7c00 ? 0 : static_cast<std::uint16_t>(i);
  const auto values = mx::view(mx::array(codes.data(), {count}, mx::uint16), mx::float16);
  const auto reference = native::silu(values);
  auto state = mx::zeros({1, 1, count}, mx::float16);
  auto input = mx::reshape(values, {1, 1, count});
  std::vector<float> conv_weights(count * 2, 0.0f);
  for (int i = 0; i < count; ++i) conv_weights[i * 2 + 1] = 1.0f;
  auto weights = mx::astype(mx::array(conv_weights.data(), {count, 2, 1}, mx::float32), mx::float16);
  const auto conv = native::causal_conv_decode_silu(state, input, weights);
  EqualFinite(reference, mx::reshape(conv.first, {count}), "half-convolution-domain");
  std::vector<std::uint16_t> gate_bias(count * groups, 0);
  std::vector<float> up_bias(count * groups, 0.0f);
  for (int i = 0; i < count; ++i) {
    gate_bias[i * groups] = codes[i];
    up_bias[i * groups] = 1.0f;
  }
  auto packed = mx::zeros({count, k / 8}, mx::uint32);
  auto zero = mx::zeros({count, groups}, mx::float16);
  native::QLinear gate{packed, zero,
      mx::view(mx::array(gate_bias.data(), {count, groups}, mx::uint16), mx::float16), 64, 4, true};
  native::QLinear up{packed, zero,
      mx::astype(mx::array(up_bias.data(), {count, groups}, mx::float32), mx::float16), 64, 4, true};
  std::vector<float> one_hot(k, 0.0f); one_hot[0] = 1.0f;
  auto x = mx::astype(mx::array(one_hot.data(), {1, 1, k}, mx::float32), mx::float16);
  EqualFinite(reference, mx::reshape(native::affine_q4_fused_swiglu_batch_one(gate, up, x), {count}),
              "half-fused-separate-parameters-domain");
  Require(native::prepare_fused_q4_raw_decode_parameters(gate, up), "raw parameters failed");
  EqualFinite(reference, mx::reshape(native::affine_q4_fused_swiglu_batch_one(gate, up, x), {count}),
              "half-fused-packed-parameters-domain");
  auto two = mx::concatenate({x, x}, 1);
  auto ref_two = mx::reshape(mx::concatenate({reference, reference}), {1, 2, count});
  for (const char* scalar : {"0", "1"}) {
    Require(setenv("SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU_BATCH_TWO_SCALAR_INPUTS", scalar, 1) == 0,
            "cannot select fused input path");
    EqualFinite(ref_two, native::affine_q4_fused_swiglu_batch_two(gate, up, two),
                "half-fused-two-row-domain");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "invalid") {
      bool rejected = false;
      try { (void)native::activation_dtype(); }
      catch (const std::runtime_error&) { rejected = true; }
      Require(rejected, "invalid activation configuration was accepted");
      return 0;
    }
    Require(argc == 1, "unexpected precision test argument");
    CheckConversion();
    CheckHalfDomain();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
