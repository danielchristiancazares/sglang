#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace mx = mlx::core;

namespace {

mx::array MakeBfloat16(std::vector<float> &values, const mx::Shape &shape,
                       float frequency, float offset = 0.0f) {
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] =
        offset + std::sin(static_cast<float>(index) * frequency) * 0.125f;
  }
  return mx::astype(mx::array(values.data(), shape, mx::float32), sglang::mlx_qwen38::activation_dtype());
}

bool CheckProductionShapePrefixes() {
  constexpr int kBatch = 1;
  constexpr int kTime = 8;
  constexpr int kKeyHeads = 16;
  constexpr int kValueHeads = 48;
  constexpr int kKeyDim = 128;
  constexpr int kValueDim = 128;
  std::vector<float> q_values(kBatch * kTime * kKeyHeads * kKeyDim);
  std::vector<float> k_values(q_values.size());
  std::vector<float> v_values(kBatch * kTime * kValueHeads * kValueDim);
  std::vector<float> decay_values(kBatch * kTime * kValueHeads);
  std::vector<float> beta_values(decay_values.size());
  std::vector<float> state_values(kBatch * kValueHeads * kValueDim * kKeyDim);

  mx::array q =
      MakeBfloat16(q_values, {kBatch, kTime, kKeyHeads, kKeyDim}, 0.0017f);
  mx::array k =
      MakeBfloat16(k_values, {kBatch, kTime, kKeyHeads, kKeyDim}, 0.0023f);
  mx::array v =
      MakeBfloat16(v_values, {kBatch, kTime, kValueHeads, kValueDim}, 0.0029f);
  mx::array beta =
      MakeBfloat16(beta_values, {kBatch, kTime, kValueHeads}, 0.013f, 0.5f);
  for (std::size_t index = 0; index < decay_values.size(); ++index) {
    decay_values[index] =
        0.75f + 0.2f * std::sin(static_cast<float>(index) * 0.011f);
  }
  mx::array decay(decay_values.data(), {kBatch, kTime, kValueHeads},
                  mx::float32);
  for (std::size_t index = 0; index < state_values.size(); ++index) {
    state_values[index] =
        std::cos(static_cast<float>(index) * 0.0003f) * 0.0625f;
  }
  mx::array state(state_values.data(),
                  {kBatch, kValueHeads, kValueDim, kKeyDim}, mx::float32);

  const auto full = sglang::mlx_qwen38::gated_delta_update_outputs(
      q, k, v, decay, beta, state, true);
  if (full.size() != 3 || full[2].dtype() != mx::float32) {
    std::cerr << "gated-delta tape output contract mismatch\n";
    return false;
  }
  for (int prefix = 1; prefix < kTime; ++prefix) {
    const mx::Shape q_end{kBatch, prefix, kKeyHeads, kKeyDim};
    const mx::Shape v_end{kBatch, prefix, kValueHeads, kValueDim};
    const mx::Shape scalar_end{kBatch, prefix, kValueHeads};
    const auto direct = sglang::mlx_qwen38::gated_delta_update_outputs(
        mx::slice(q, {0, 0, 0, 0}, q_end), mx::slice(k, {0, 0, 0, 0}, q_end),
        mx::slice(v, {0, 0, 0, 0}, v_end),
        mx::slice(decay, {0, 0, 0}, scalar_end),
        mx::slice(beta, {0, 0, 0}, scalar_end), state, false);
    mx::array replay = sglang::mlx_qwen38::gated_delta_commit(k, decay, full[2],
                                                              state, prefix);
    mx::array direct_state = mx::astype(direct[1], mx::float32);
    replay = mx::astype(replay, mx::float32);
    mx::eval(direct_state, replay);

    float maximum_absolute_error = 0.0f;
    const float *direct_data = direct_state.data<float>();
    const float *replay_data = replay.data<float>();
    for (std::size_t index = 0; index < direct_state.size(); ++index) {
      maximum_absolute_error =
          std::max(maximum_absolute_error,
                   std::abs(direct_data[index] - replay_data[index]));
    }
    std::cout << "prefix=" << prefix << " max_abs=" << maximum_absolute_error
              << '\n';
    if (maximum_absolute_error != 0.0f) {
      return false;
    }
  }
  return true;
}

} // namespace

int main() {
  try {
    if (!CheckProductionShapePrefixes()) {
      return 1;
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "qwen38 gated-delta commit parity passed\n";
  return 0;
}
