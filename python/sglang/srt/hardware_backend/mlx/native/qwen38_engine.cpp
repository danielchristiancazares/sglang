#include "qwen38_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/transforms.h"

namespace sglang {
namespace mlx_qwen38 {
namespace {

using mlx::core::array;
using mlx::core::astype;
using mlx::core::async_eval;
using mlx::core::concatenate;
using mlx::core::eval;
using mlx::core::expand_dims;
using mlx::core::logaddexp;
using mlx::core::matmul;
using mlx::core::quantized_matmul;
using mlx::core::reshape;
using mlx::core::sigmoid;
using mlx::core::slice;
using mlx::core::slice_update;
using mlx::core::split;
using mlx::core::sum;
using mlx::core::take;
using mlx::core::transpose;
using mlx::core::zeros;
namespace mx = mlx::core;

// cfg_ is Engine's first member, so this initializer runs before the MLX array
// members can create the Metal device and cache its command-buffer limits.
MlxQwen38Config configure_mlx_runtime(MlxQwen38Config cfg) {
  if (std::getenv("MLX_MAX_MB_PER_BUFFER") == nullptr &&
      setenv("MLX_MAX_MB_PER_BUFFER", "128", 0) != 0) {
    throw std::runtime_error("failed to set MLX command-buffer byte budget");
  }
  return cfg;
}

std::string layer_key(int i, const std::string& rest) {
  return "language_model.model.layers." + std::to_string(i) + rest;
}

array last_token(const array& hidden) {
  auto shape = hidden.shape();
  int seq = static_cast<int>(shape[1]);
  return squeeze(slice(hidden, {0, seq - 1, 0}, {1, seq, shape[2]}), 1);
}

// Same Metal body as mlx-lm's gated_delta_kernel (scalar g, no mask).
constexpr const char* kGatedDeltaSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
)";

constexpr const char* kCausalConvDecodeSiluSource = R"(
        auto channel = thread_position_in_grid.x;
        auto batch = thread_position_in_grid.y;
        auto state_base = state + batch * (K - 1) * D + channel;
        auto qkv_base = qkv + batch * D + channel;
        auto weight_base = weight + channel * K;

        float acc = 0.0f;
        for (int tap = 0; tap < K - 1; ++tap) {
          acc += static_cast<float>(state_base[tap * D]) * weight_base[tap];
        }
        acc += static_cast<float>(qkv_base[0]) * weight_base[K - 1];
        InT conv_value = static_cast<InT>(acc);
        auto sigmoid_low =
            1 / (1 + metal::precise::exp(metal::abs(conv_value)));
        InT sigmoid_value =
            (conv_value < 0) ? sigmoid_low : 1 - sigmoid_low;
        conv_out[batch * D + channel] =
            static_cast<InT>(conv_value * sigmoid_value);

        auto next_state_base = next_state + batch * (K - 1) * D + channel;
        for (int tap = 0; tap < K - 2; ++tap) {
          next_state_base[tap * D] = state_base[(tap + 1) * D];
        }
        next_state_base[(K - 2) * D] = qkv_base[0];
)";

constexpr const char* kResidualRmsNormSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        InT cached[Iterations][N_READS];
        float acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              InT value = x[index] + residual[index];
              cached[iteration][i] = value;
              residual_out[index] = value;
              float value_f = static_cast<float>(value);
              acc += value_f * value_f;
            }
          }
        }

        acc = simd_sum(acc);
        if (simd_lane == 0) {
          local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          acc = simd_sum(
              simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
          if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              norm_out[index] = weight[base + i] * static_cast<InT>(
                  static_cast<float>(cached[iteration][i]) * local_inv_mean[0]);
            }
          }
        }
)";

constexpr const char* kGatedDeltaQkNormSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[2];
        threadgroup float local_q_sums[SIMD_SIZE];
        threadgroup float local_k_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        float cached_q[Iterations][N_READS];
        float cached_k[Iterations][N_READS];
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float q_value = static_cast<float>(q[index]);
              float k_value = static_cast<float>(k[index]);
              cached_q[iteration][i] = q_value;
              cached_k[iteration][i] = k_value;
              q_acc += q_value * q_value;
              k_acc += k_value * k_value;
            }
          }
        }

        q_acc = simd_sum(q_acc);
        k_acc = simd_sum(k_acc);
        float q_inv_mean;
        float k_inv_mean;
        if (SIMD_GROUPS == 1) {
          q_inv_mean = metal::precise::rsqrt(q_acc / D + eps);
          k_inv_mean = metal::precise::rsqrt(k_acc / D + eps);
        } else {
          if (simd_lane == 0) {
            local_q_sums[simd_group] = q_acc;
            local_k_sums[simd_group] = k_acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group == 0) {
            q_acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_q_sums[simd_lane] : 0.0f);
            k_acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_k_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
              local_inv_mean[0] = metal::precise::rsqrt(q_acc / D + eps);
              local_inv_mean[1] = metal::precise::rsqrt(k_acc / D + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          q_inv_mean = local_inv_mean[0];
          k_inv_mean = local_inv_mean[1];
        }

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              InT q_norm = static_cast<InT>(
                  cached_q[iteration][i] * q_inv_mean);
              InT k_norm = static_cast<InT>(
                  cached_k[iteration][i] * k_inv_mean);
              q_out[index] = q_scale * static_cast<float>(q_norm);
              k_out[index] = k_scale * static_cast<float>(k_norm);
            }
          }
        }
)";

constexpr const char* kFullAttnQkNormRopeSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;
        constexpr int HeadsPerBatch = Nq + Nk;
        auto batch = row / HeadsPerBatch;
        auto combined_head = row % HeadsPerBatch;
        bool is_q = combined_head < Nq;
        auto head = is_q ? combined_head : combined_head - Nq;
        const device InT* input = is_q
            ? qg + (batch * Nq + head) * (2 * D)
            : k + (batch * Nk + head) * D;
        const device InT* weight = is_q ? q_weight : k_weight;
        device InT* output = is_q
            ? q_out + (batch * Nq + head) * D
            : k_out + (batch * Nk + head) * D;

        float acc = 0.0f;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              float value = static_cast<float>(input[base + i]);
              acc += value * value;
            }
          }
        }

        acc = simd_sum(acc);
        if (simd_lane == 0) {
          local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          acc = simd_sum(
              simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
          if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS) {
          for (int i = 0; i < N_READS; ++i) {
            auto index = base + i;
            if (index < D && index >= RopeDims) {
              output[index] = weight[index] * static_cast<InT>(
                  static_cast<float>(input[index]) * local_inv_mean[0]);
            }
          }
        }

        constexpr int RopePairs = RopeDims / 2;
        if (lid < RopePairs) {
          auto index_1 = lid;
          auto index_2 = lid + RopePairs;
          InT x1_norm = weight[index_1] * static_cast<InT>(
              static_cast<float>(input[index_1]) * local_inv_mean[0]);
          InT x2_norm = weight[index_2] * static_cast<InT>(
              static_cast<float>(input[index_2]) * local_inv_mean[0]);
          float d = static_cast<float>(lid) / static_cast<float>(RopePairs);
          float inv_freq = metal::exp2(-d * log2_base);
          float L = 1.0f * static_cast<float>(rope_offset);
          float theta = L * inv_freq;
          float costheta = metal::fast::cos(theta);
          float sintheta = metal::fast::sin(theta);
          float x1 = static_cast<float>(x1_norm);
          float x2 = static_cast<float>(x2_norm);
          output[index_1] = static_cast<InT>(
              x1 * costheta - x2 * sintheta);
          output[index_2] = static_cast<InT>(
              x1 * sintheta + x2 * costheta);
        }
)";

constexpr const char* kGatedDeltaNormGateSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        float cached[Iterations][N_READS];
        float acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float value = static_cast<float>(recurrent_out[index]);
              cached[iteration][i] = value;
              acc += value * value;
            }
          }
        }

        acc = simd_sum(acc);
        float inv_mean;
        if (SIMD_GROUPS == 1) {
          inv_mean = metal::precise::rsqrt(acc / D + eps);
        } else {
          if (simd_lane == 0) {
            local_sums[simd_group] = acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group == 0) {
            acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
              local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          inv_mean = local_inv_mean[0];
        }

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float normed =
                  static_cast<float>(weight[base + i]) *
                  (cached[iteration][i] * inv_mean);
              float z_value = static_cast<float>(z[index]);
              float sigmoid_low =
                  1.0f /
                  (1.0f + metal::precise::exp(metal::abs(z_value)));
              float sigmoid_value =
                  (z_value < 0.0f) ? sigmoid_low : 1.0f - sigmoid_low;
              gated_out[index] = static_cast<GateT>(
                  (z_value * sigmoid_value) * normed);
            }
          }
        }
)";

std::vector<array> compute_g_fn(const std::vector<array>& xs) {
  return {mx::exp(-mx::exp(astype(xs[0], mx::float32)) * softplus(xs[1] + xs[2]))};
}

const std::function<std::vector<array>(const std::vector<array>&)>&
compiled_compute_g() {
  static auto fn = mx::compile(compute_g_fn, /*shapeless=*/true);
  return fn;
}

const mx::fast::CustomKernelFunction& gated_delta_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      kGatedDeltaSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& causal_conv_decode_silu_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_causal_conv_decode_silu",
      {"state", "qkv", "weight"},
      {"conv_out", "next_state"},
      kCausalConvDecodeSiluSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& residual_rms_norm_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_residual_rms_norm",
      {"x", "residual", "weight", "eps"},
      {"residual_out", "norm_out"},
      kResidualRmsNormSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_qk_norm_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_qk_norm",
      {"q", "k", "q_scale", "k_scale", "eps"},
      {"q_out", "k_out"},
      kGatedDeltaQkNormSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& full_attn_qk_norm_rope_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_full_attn_qk_norm_rope",
      {"qg", "k", "q_weight", "k_weight", "eps", "log2_base", "rope_offset"},
      {"q_out", "k_out"},
      kFullAttnQkNormRopeSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_norm_gate_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_norm_gate",
      {"recurrent_out", "z", "weight", "eps"},
      {"gated_out"},
      kGatedDeltaNormGateSource);
  return kernel;
}

} // namespace

std::pair<array, array> causal_conv_decode_silu(
    const array& state,
    const array& qkv,
    const array& weight) {
  int B = static_cast<int>(qkv.shape()[0]);
  int K = static_cast<int>(state.shape()[1]) + 1;
  int D = static_cast<int>(qkv.shape()[2]);
  auto outs = causal_conv_decode_silu_metal()(
      {state, qkv, weight},
      {{B, 1, D}, {B, K - 1, D}},
      {qkv.dtype(), state.dtype()},
      {D, B, 1},
      {std::min(D, 256), 1, 1},
      {
          {"InT", mx::fast::TemplateArg{qkv.dtype()}},
          {"K", mx::fast::TemplateArg{K}},
          {"D", mx::fast::TemplateArg{D}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> residual_rms_norm(
    const array& x,
    const array& residual,
    const array& weight,
    float eps) {
  if (x.shape() != residual.shape() || x.ndim() == 0 || weight.ndim() != 1 ||
      x.shape().back() != weight.shape()[0] || x.dtype() != residual.dtype() ||
      x.dtype() != weight.dtype()) {
    throw std::runtime_error("invalid residual RMSNorm inputs");
  }
  const int D = static_cast<int>(x.shape().back());
  const int rows = static_cast<int>(x.size() / static_cast<size_t>(D));
  int threads = 1024;
  if (D <= 4096) {
    const int threads_needed = (D + 3) / 4;
    threads = ((threads_needed + 31) / 32) * 32;
  }
  auto outs = residual_rms_norm_metal()(
      {x, residual, weight, array(eps)},
      {x.shape(), x.shape()},
      {x.dtype(), x.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{x.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> normalize_gated_delta_qk(
    const array& q,
    const array& k,
    float q_scale,
    float k_scale,
    float eps) {
  if (q.shape() != k.shape() || q.ndim() == 0 || q.dtype() != k.dtype()) {
    throw std::runtime_error("invalid gated-delta q/k normalization inputs");
  }
  const int D = static_cast<int>(q.shape().back());
  if (D <= 0 || D > 4096) {
    throw std::runtime_error("gated-delta q/k normalization width out of range");
  }
  const int rows = static_cast<int>(q.size() / static_cast<size_t>(D));
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  auto outs = gated_delta_qk_norm_metal()(
      {q, k, array(q_scale), array(k_scale), array(eps)},
      {q.shape(), k.shape()},
      {mx::float32, mx::float32},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{q.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> full_attn_qk_norm_rope(
    const array& qg,
    const array& k,
    const array& q_weight,
    const array& k_weight,
    float eps,
    float rope_theta,
    int rope_dims,
    int rope_offset) {
  if (qg.ndim() != 4 || k.ndim() != 4 || qg.shape()[0] != k.shape()[0] ||
      qg.shape()[1] != 1 || k.shape()[1] != 1 ||
      qg.shape().back() != 2 * k.shape().back() ||
      q_weight.ndim() != 1 || k_weight.ndim() != 1 ||
      q_weight.shape()[0] != k.shape().back() ||
      k_weight.shape()[0] != k.shape().back() || qg.dtype() != k.dtype() ||
      qg.dtype() != q_weight.dtype() || qg.dtype() != k_weight.dtype()) {
    throw std::runtime_error("invalid full-attention q/k norm/RoPE inputs");
  }
  const int B = static_cast<int>(qg.shape()[0]);
  const int Nq = static_cast<int>(qg.shape()[2]);
  const int Nk = static_cast<int>(k.shape()[2]);
  const int D = static_cast<int>(k.shape().back());
  if (D <= 0 || D > 4096 || rope_dims <= 0 || rope_dims > D ||
      rope_dims % 2 != 0) {
    throw std::runtime_error("full-attention q/k norm/RoPE width out of range");
  }
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  const int rows = B * (Nq + Nk);
  auto outs = full_attn_qk_norm_rope_metal()(
      {qg,
       k,
       q_weight,
       k_weight,
       array(eps),
       array(std::log2(rope_theta)),
       array(rope_offset)},
      {{B, Nq, 1, D}, {B, Nk, 1, D}},
      {qg.dtype(), k.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{qg.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"RopeDims", mx::fast::TemplateArg{rope_dims}},
          {"Nq", mx::fast::TemplateArg{Nq}},
          {"Nk", mx::fast::TemplateArg{Nk}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

array gated_delta_norm_gate(
    const array& recurrent_out,
    const array& z,
    const array& weight,
    float eps) {
  if (recurrent_out.shape() != z.shape() || recurrent_out.ndim() == 0 ||
      weight.ndim() != 1 || recurrent_out.shape().back() != weight.shape()[0] ||
      recurrent_out.dtype() != mx::float32 || z.dtype() != weight.dtype()) {
    throw std::runtime_error("invalid gated-delta norm/gate inputs");
  }
  const int D = static_cast<int>(recurrent_out.shape().back());
  if (D <= 0 || D > 4096) {
    throw std::runtime_error("gated-delta norm/gate width out of range");
  }
  const int rows =
      static_cast<int>(recurrent_out.size() / static_cast<size_t>(D));
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  auto outs = gated_delta_norm_gate_metal()(
      {recurrent_out, z, weight, array(eps)},
      {z.shape()},
      {z.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"GateT", mx::fast::TemplateArg{z.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return outs[0];
}

array silu(const array& x) {
  return x * sigmoid(x);
}

array softplus(const array& x) {
  return logaddexp(x, array(0.0f));
}

array QLinear::operator()(const array& x) const {
  if (!valid) {
    throw std::runtime_error("QLinear used before load");
  }
  return quantized_matmul(
      x, w, scales, biases, /*transpose=*/true, group_size, bits, "affine");
}

std::pair<array, array> gated_delta_step(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  // q,k: [B, H, Dk]; v: [B, H, Dv]; g,beta: [B, H]; state: [B, H, Dv, Dk]
  array decay = expand_dims(expand_dims(g, -1), -1);
  array st = state * decay;
  array k_exp = expand_dims(k, -2);
  array kv_mem = sum(st * k_exp, -1);
  array delta = (v - kv_mem) * expand_dims(beta, -1);
  st = st + k_exp * expand_dims(delta, -1);
  array q_exp = expand_dims(q, -2);
  array y = sum(st * q_exp, -1);
  return {astype(y, q.dtype()), st};
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  int B = static_cast<int>(q.shape()[0]);
  int T = static_cast<int>(q.shape()[1]);
  int Hk = static_cast<int>(q.shape()[2]);
  int Dk = static_cast<int>(q.shape()[3]);
  int Hv = static_cast<int>(v.shape()[2]);
  int Dv = static_cast<int>(v.shape()[3]);
  if (Dk % 32 != 0) {
    throw std::runtime_error("gated_delta Metal kernel requires Dk multiple of 32");
  }
  if (Hv % Hk != 0) {
    throw std::runtime_error("Hv must be a multiple of Hk");
  }
  auto outs = gated_delta_metal()(
      {q, k, v, g, beta, state, array(T, mx::int32)},
      {{B, T, Hv, Dv}, state.shape()},
      {q.dtype(), state.dtype()},
      {32, Dv, B * Hv},
      {32, 4, 1},
      {
          {"InT", mx::fast::TemplateArg{q.dtype()}},
          {"StT", mx::fast::TemplateArg{state.dtype()}},
          {"Dk", mx::fast::TemplateArg{Dk}},
          {"Dv", mx::fast::TemplateArg{Dv}},
          {"Hk", mx::fast::TemplateArg{Hk}},
          {"Hv", mx::fast::TemplateArg{Hv}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

Engine::Engine(MlxQwen38Config cfg, const std::string& model_dir)
    : cfg_(configure_mlx_runtime(cfg)) {
  if (cfg_.hidden_size <= 0 || cfg_.num_hidden_layers <= 0) {
    throw std::runtime_error("invalid Qwen3.8 config");
  }
  layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
  load_weights(model_dir);
  reset();
}

void Engine::reset() {
  for (auto& layer : layers_) {
    if (layer.is_linear) {
      layer.linear.has_state = false;
    } else {
      layer.attn.offset = 0;
      layer.attn.cache_length = 0;
    }
  }
  reset_decode_pipeline();
  request_boundary_pending_ = false;
  token_history_.clear();
  if (mtp_valid_) {
    mtp_reset();
  }
}

void Engine::begin_request() {
  request_boundary_pending_ = true;
}

void Engine::reset_decode_pipeline() {
  pending_tok_ = array(0);
  last_hidden_ = array(0);
  last_emitted_ = -1;
  decode_scheduled_ = false;
  last_emitted_in_state_ = false;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
}

void Engine::record_processed_token(int32_t token) {
  token_history_.push_back(token);
  last_emitted_in_state_ = true;
}

QLinear Engine::load_qlinear(
    const std::unordered_map<std::string, array>& weights,
    const std::string& prefix) {
  QLinear q;
  q.w = require(weights, prefix + ".weight");
  q.scales = require(weights, prefix + ".scales");
  q.biases = require(weights, prefix + ".biases");
  q.group_size = cfg_.quant_group_size;

  const auto& weight_shape = q.w.shape();
  const auto& scale_shape = q.scales.shape();
  if (weight_shape.size() < 2 || scale_shape.size() < 2 ||
      q.biases.shape() != scale_shape) {
    throw std::runtime_error("invalid affine tensors for " + prefix);
  }
  const int64_t packed_bits = static_cast<int64_t>(weight_shape.back()) * 32;
  const int64_t input_features =
      static_cast<int64_t>(scale_shape.back()) * q.group_size;
  if (input_features <= 0 || packed_bits % input_features != 0) {
    throw std::runtime_error("cannot infer affine bit width for " + prefix);
  }
  q.bits = static_cast<int>(packed_bits / input_features);
  if (q.bits < 2 || q.bits > 8 || q.bits == 7) {
    throw std::runtime_error(
        "unsupported affine bit width " + std::to_string(q.bits) + " for " +
        prefix);
  }
  q.valid = true;
  return q;
}

array Engine::require(
    const std::unordered_map<std::string, array>& weights,
    const std::string& key) const {
  auto it = weights.find(key);
  if (it == weights.end()) {
    throw std::runtime_error("missing weight " + key);
  }
  return it->second;
}

void Engine::load_weights(const std::string& model_dir) {
  std::unordered_map<std::string, array> weights;
  DIR* dir = opendir(model_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("cannot open model dir " + model_dir);
  }
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() < 12 || name.substr(name.size() - 12) != ".safetensors") {
      continue;
    }
    auto loaded = mx::load_safetensors(model_dir + "/" + name);
    for (auto& kv : loaded.first) {
      if (kv.first.rfind("vision_tower", 0) == 0) {
        continue;
      }
      weights.emplace(kv.first, std::move(kv.second));
    }
  }
  closedir(dir);
  if (weights.empty()) {
    throw std::runtime_error("no language-model safetensors in " + model_dir);
  }

  bool shift_norms = false;
  for (auto& kv : weights) {
    if (kv.first.find("conv1d.weight") != std::string::npos) {
      auto shape = kv.second.shape();
      if (shape.size() == 3 && shape[2] != 1) {
        kv.second = transpose(kv.second, {0, 2, 1});
        shift_norms = true;
      }
    }
  }
  if (shift_norms) {
    for (auto& kv : weights) {
      const auto& k = kv.first;
      if (k.size() >= 7 && k.substr(k.size() - 7) == ".weight") {
        bool is_norm = k.find("layernorm") != std::string::npos ||
            k.find(".q_norm.") != std::string::npos ||
            k.find(".k_norm.") != std::string::npos ||
            k == "language_model.model.norm.weight";
        if (is_norm && kv.second.ndim() == 1) {
          kv.second = kv.second + array(1.0f);
        }
      }
    }
  }

  embed_tokens_ = load_qlinear(weights, "language_model.model.embed_tokens");
  embed_table_ = mx::dequantize(
      embed_tokens_.w,
      embed_tokens_.scales,
      embed_tokens_.biases,
      embed_tokens_.group_size,
      embed_tokens_.bits,
      "affine",
      std::nullopt,
      mx::bfloat16);
  eval(embed_table_);
  lm_head_ = load_qlinear(weights, "language_model.lm_head");
  final_norm_ = require(weights, "language_model.model.norm.weight");

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    DecoderLayer layer;
    layer.is_linear = ((i + 1) % cfg_.full_attention_interval) != 0;
    layer.input_norm = require(weights, layer_key(i, ".input_layernorm.weight"));
    layer.post_norm = require(weights, layer_key(i, ".post_attention_layernorm.weight"));
    layer.gate_proj = load_qlinear(weights, layer_key(i, ".mlp.gate_proj"));
    layer.up_proj = load_qlinear(weights, layer_key(i, ".mlp.up_proj"));
    layer.down_proj = load_qlinear(weights, layer_key(i, ".mlp.down_proj"));
    if (layer.is_linear) {
      layer.linear.in_proj_qkv =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_qkv"));
      layer.linear.in_proj_z =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_z"));
      layer.linear.in_proj_b =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_b"));
      layer.linear.in_proj_a =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_a"));
      layer.linear.out_proj =
          load_qlinear(weights, layer_key(i, ".linear_attn.out_proj"));
      layer.linear.conv1d = require(weights, layer_key(i, ".linear_attn.conv1d.weight"));
      layer.linear.A_log = require(weights, layer_key(i, ".linear_attn.A_log"));
      layer.linear.dt_bias = require(weights, layer_key(i, ".linear_attn.dt_bias"));
      layer.linear.norm = require(weights, layer_key(i, ".linear_attn.norm.weight"));
    } else {
      layer.attn.q_proj = load_qlinear(weights, layer_key(i, ".self_attn.q_proj"));
      layer.attn.k_proj = load_qlinear(weights, layer_key(i, ".self_attn.k_proj"));
      layer.attn.v_proj = load_qlinear(weights, layer_key(i, ".self_attn.v_proj"));
      layer.attn.o_proj = load_qlinear(weights, layer_key(i, ".self_attn.o_proj"));
      layer.attn.q_norm = require(weights, layer_key(i, ".self_attn.q_norm.weight"));
      layer.attn.k_norm = require(weights, layer_key(i, ".self_attn.k_norm.weight"));
    }
    layers_[static_cast<size_t>(i)] = std::move(layer);
  }
}

array Engine::embed(const array& tokens) const {
  return take(embed_table_, tokens, 0);
}

array Engine::logits(const array& hidden) const {
  array n = mx::fast::rms_norm(hidden, final_norm_, cfg_.rms_norm_eps);
  return lm_head_(n);
}

array Engine::mlp(const DecoderLayer& layer, const array& x) const {
  return layer.down_proj(silu(layer.gate_proj(x)) * layer.up_proj(x));
}

array Engine::full_attn(FullAttn& attn, const array& x) {
  auto xshape = x.shape();
  int B = static_cast<int>(xshape[0]);
  int L = static_cast<int>(xshape[1]);
  int n_q = cfg_.num_attention_heads;
  int n_kv = cfg_.num_key_value_heads;
  int hd = cfg_.head_dim;
  int rope_dims = static_cast<int>(std::lround(hd * cfg_.partial_rotary_factor));

  array qg = attn.q_proj(x);
  qg = reshape(qg, {B, L, n_q, hd * 2});
  auto q_gate = split(qg, 2, -1);
  array queries = q_gate[0];
  array gate = reshape(q_gate[1], {B, L, n_q * hd});

  array keys = reshape(attn.k_proj(x), {B, L, n_kv, hd});
  array values = reshape(attn.v_proj(x), {B, L, n_kv, hd});

  if (L == 1) {
    auto normalized = full_attn_qk_norm_rope(
        qg,
        keys,
        attn.q_norm,
        attn.k_norm,
        cfg_.rms_norm_eps,
        cfg_.rope_theta,
        rope_dims,
        attn.offset);
    queries = normalized.first;
    keys = normalized.second;
  } else {
    queries = mx::fast::rms_norm(queries, attn.q_norm, cfg_.rms_norm_eps);
    keys = mx::fast::rms_norm(keys, attn.k_norm, cfg_.rms_norm_eps);
    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});

    queries = mx::fast::rope(
        queries,
        rope_dims,
        /*traditional=*/false,
        cfg_.rope_theta,
        1.0f,
        attn.offset);
    keys = mx::fast::rope(
        keys,
        rope_dims,
        /*traditional=*/false,
        cfg_.rope_theta,
        1.0f,
        attn.offset);
  }
  values = transpose(values, {0, 2, 1, 3});

  const int needed = attn.cache_length + L;
  if (attn.cache_capacity < needed) {
    int capacity = std::max(256, attn.cache_capacity);
    while (capacity < needed) {
      capacity *= 2;
    }
    array new_keys = zeros({B, n_kv, capacity, hd}, keys.dtype());
    array new_values = zeros({B, n_kv, capacity, hd}, values.dtype());
    if (attn.cache_length > 0) {
      auto active_keys = slice(
          attn.keys, {0, 0, 0, 0}, {B, n_kv, attn.cache_length, hd});
      auto active_values = slice(
          attn.values, {0, 0, 0, 0}, {B, n_kv, attn.cache_length, hd});
      new_keys = slice_update(
          new_keys,
          active_keys,
          {0, 0, 0, 0},
          {B, n_kv, attn.cache_length, hd});
      new_values = slice_update(
          new_values,
          active_values,
          {0, 0, 0, 0},
          {B, n_kv, attn.cache_length, hd});
      eval(new_keys, new_values);
    }
    attn.keys = new_keys;
    attn.values = new_values;
    attn.cache_capacity = capacity;
  }
  attn.keys = slice_update(
      attn.keys,
      keys,
      {0, 0, attn.cache_length, 0},
      {B, n_kv, needed, hd});
  attn.values = slice_update(
      attn.values,
      values,
      {0, 0, attn.cache_length, 0},
      {B, n_kv, needed, hd});
  attn.cache_length = needed;
  attn.offset += L;

  keys = slice(attn.keys, {0, 0, 0, 0}, {B, n_kv, needed, hd});
  values = slice(attn.values, {0, 0, 0, 0}, {B, n_kv, needed, hd});

  std::string mask_mode = (L > 1) ? "causal" : "";
  array output = mx::fast::scaled_dot_product_attention(
      queries, keys, values, 1.0f / std::sqrt(static_cast<float>(hd)), mask_mode);
  output = reshape(transpose(output, {0, 2, 1, 3}), {B, L, n_q * hd});
  return attn.o_proj(output * sigmoid(gate));
}

array Engine::gated_delta(LinearAttn& lin, const array& x) {
  auto xshape = x.shape();
  int B = static_cast<int>(xshape[0]);
  int S = static_cast<int>(xshape[1]);
  int hk = cfg_.linear_num_key_heads;
  int hv = cfg_.linear_num_value_heads;
  int dk = cfg_.linear_key_head_dim;
  int dv = cfg_.linear_value_head_dim;
  int key_dim = hk * dk;
  int value_dim = hv * dv;
  int conv_dim = key_dim * 2 + value_dim;
  int ksz = cfg_.linear_conv_kernel_dim;

  array qkv = lin.in_proj_qkv(x);
  array z = reshape(lin.in_proj_z(x), {B, S, hv, dv});
  array b = lin.in_proj_b(x);
  array a = lin.in_proj_a(x);

  if (!lin.has_state) {
    lin.conv_state = zeros({B, ksz - 1, conv_dim}, x.dtype());
    lin.rec_state = zeros({B, hv, dv, dk}, mx::float32);
    lin.has_state = true;
  }
  array conv_out = qkv;
  if (S == 1 && ksz > 1) {
    auto conv = causal_conv_decode_silu(lin.conv_state, qkv, lin.conv1d);
    conv_out = conv.first;
    lin.conv_state = conv.second;
  } else {
    array conv_input = concatenate({lin.conv_state, qkv}, 1);
    lin.conv_state = slice(
        conv_input, {0, S, 0}, {B, S + (ksz - 1), conv_dim});
    conv_out = silu(mx::conv1d(conv_input, lin.conv1d, 1, 0, 1, conv_dim));
  }

  auto qkv_split = split(conv_out, mx::Shape{key_dim, 2 * key_dim}, -1);
  array q = reshape(qkv_split[0], {B, S, hk, dk});
  array k = reshape(qkv_split[1], {B, S, hk, dk});
  array v = reshape(qkv_split[2], {B, S, hv, dv});

  float inv = 1.0f / std::sqrt(static_cast<float>(dk));
  if (S == 1) {
    auto normalized = normalize_gated_delta_qk(q, k, inv * inv, inv, 1e-6f);
    q = normalized.first;
    k = normalized.second;
  } else {
    q = (inv * inv) * mx::fast::rms_norm(q, std::nullopt, 1e-6f);
    k = inv * mx::fast::rms_norm(k, std::nullopt, 1e-6f);
  }

  array beta = sigmoid(b);
  array g = compiled_compute_g()({lin.A_log, a, lin.dt_bias})[0];
  auto updated = gated_delta_update(q, k, v, g, beta, lin.rec_state);
  lin.rec_state = updated.second;
  array out = updated.first;
  array gated = S == 1
      ? gated_delta_norm_gate(out, z, lin.norm, cfg_.rms_norm_eps)
      : astype(
            silu(astype(z, mx::float32)) *
                astype(
                    mx::fast::rms_norm(out, lin.norm, cfg_.rms_norm_eps),
                    mx::float32),
            x.dtype());
  return lin.out_proj(reshape(gated, {B, S, value_dim}));
}

array Engine::forward_hidden(const array& tokens) {
  array h = embed(tokens);
  const bool single_token = tokens.shape()[1] == 1;
  if (!single_token) {
    for (auto& layer : layers_) {
      array n = mx::fast::rms_norm(h, layer.input_norm, cfg_.rms_norm_eps);
      array r =
          layer.is_linear ? gated_delta(layer.linear, n) : full_attn(layer.attn, n);
      h = h + r;
      array n2 = mx::fast::rms_norm(h, layer.post_norm, cfg_.rms_norm_eps);
      h = h + mlp(layer, n2);
    }
    return h;
  }

  array n = mx::fast::rms_norm(
      h, layers_.front().input_norm, cfg_.rms_norm_eps);
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    array r =
        layer.is_linear ? gated_delta(layer.linear, n) : full_attn(layer.attn, n);
    auto post_norm =
        residual_rms_norm(h, r, layer.post_norm, cfg_.rms_norm_eps);
    h = post_norm.first;
    array m = mlp(layer, post_norm.second);
    if (i + 1 == layers_.size()) {
      h = h + m;
    } else {
      auto next_norm = residual_rms_norm(
          h, m, layers_[i + 1].input_norm, cfg_.rms_norm_eps);
      h = next_norm.first;
      n = next_norm.second;
    }
  }
  return h;
}

array Engine::greedy_token(const array& hidden) {
  last_hidden_ = last_token(hidden);
  return mx::argmax(logits(last_hidden_), -1);
}

void Engine::snapshot() {
  snap_.resize(layers_.size());
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    auto& s = snap_[i];
    s.is_linear = layer.is_linear;
    if (layer.is_linear) {
      s.has_state = layer.linear.has_state;
      s.conv = layer.linear.conv_state;
      s.rec = layer.linear.rec_state;
    } else {
      s.offset = layer.attn.offset;
      s.cache_length = layer.attn.cache_length;
      s.cache_capacity = layer.attn.cache_capacity;
      s.keys = layer.attn.keys;
      s.values = layer.attn.values;
    }
  }
}

void Engine::restore() {
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    const auto& s = snap_[i];
    if (layer.is_linear) {
      layer.linear.has_state = s.has_state;
      layer.linear.conv_state = s.conv;
      layer.linear.rec_state = s.rec;
    } else {
      layer.attn.offset = s.offset;
      layer.attn.cache_length = s.cache_length;
      layer.attn.cache_capacity = s.cache_capacity;
      layer.attn.keys = s.keys;
      layer.attn.values = s.values;
    }
  }
}

void Engine::forward_argmax(const int32_t* tokens, int n, int32_t* out) {
  array ids(tokens, {1, n}, mx::int32);
  array hidden = forward_hidden(ids);
  last_hidden_ = last_token(hidden);
  array tok = mx::argmax(logits(hidden), -1);
  eval(tok);
  auto* data = tok.data<int32_t>();
  for (int i = 0; i < n; ++i) {
    out[i] = data[i];
  }
}

void Engine::mtp_reset() {
  mtp_layer_.attn.cache_length = 0;
  int seq = 0;
  for (const auto& layer : layers_) {
    if (!layer.is_linear) {
      seq = layer.attn.offset;
      break;
    }
  }
  mtp_layer_.attn.offset = seq;
}

array Engine::mtp_forward(const array& token_embed, const array& hidden) {
  array h = mtp_fc_(concatenate(
      {mx::fast::rms_norm(token_embed, mtp_pre_emb_, cfg_.rms_norm_eps),
       mx::fast::rms_norm(hidden, mtp_pre_hid_, cfg_.rms_norm_eps)},
      -1));
  array n = mx::fast::rms_norm(h, mtp_layer_.input_norm, cfg_.rms_norm_eps);
  h = h + full_attn(mtp_layer_.attn, n);
  array n2 = mx::fast::rms_norm(h, mtp_layer_.post_norm, cfg_.rms_norm_eps);
  h = h + mlp(mtp_layer_, n2);
  return mx::fast::rms_norm(h, mtp_norm_, cfg_.rms_norm_eps);
}

int Engine::mtp_draft(int32_t bonus, int32_t* drafts, int n_draft) {
  mtp_reset();
  array hid = last_hidden_;
  if (hid.ndim() == 1) {
    hid = reshape(hid, {1, 1, hid.shape()[0]});
  } else if (hid.ndim() == 2) {
    hid = expand_dims(hid, 1);
  }
  int32_t tok = bonus;
  for (int i = 0; i < n_draft; ++i) {
    array ids(&tok, {1, 1}, mx::int32);
    hid = mtp_forward(embed(ids), hid);
    array next = mx::argmax(lm_head_(hid), -1);
    eval(next);
    tok = next.item<int32_t>();
    drafts[i] = tok;
  }
  return n_draft;
}

void Engine::spec_refill(int32_t token) {
  if (spec_buf_pos_ < spec_buf_n_ && token == last_emitted_) {
    return;
  }
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  const int n_draft = mtp_block_ - 1;
  int32_t drafts[8];
  mtp_draft(token, drafts, n_draft);
  snapshot();
  int32_t input[8];
  input[0] = token;
  for (int i = 0; i < n_draft; ++i) {
    input[i + 1] = drafts[i];
  }
  const int n_in = n_draft + 1;
  int32_t pred[8];
  forward_argmax(input, n_in, pred);
  int n_correct = 0;
  for (int i = 0; i < n_draft; ++i) {
    if (pred[i] != drafts[i]) {
      break;
    }
    ++n_correct;
  }
  if (n_correct < n_draft) {
    restore();
    forward_argmax(input, n_correct + 1, pred);
  }
  for (int i = 0; i < n_correct; ++i) {
    spec_buf_[i] = drafts[i];
  }
  spec_buf_[n_correct] = pred[n_correct];
  spec_buf_n_ = n_correct + 1;
  spec_buf_pos_ = 0;
}

void Engine::load_mtp(const std::string& mtp_dir) {
  std::unordered_map<std::string, array> weights;
  DIR* dir = opendir(mtp_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("cannot open MTP dir " + mtp_dir);
  }
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() < 12 || name.substr(name.size() - 12) != ".safetensors") {
      continue;
    }
    auto loaded = mx::load_safetensors(mtp_dir + "/" + name);
    for (auto& kv : loaded.first) {
      weights.emplace(kv.first, std::move(kv.second));
    }
  }
  closedir(dir);
  if (weights.empty()) {
    throw std::runtime_error("no MTP safetensors in " + mtp_dir);
  }
  mtp_fc_ = load_qlinear(weights, "fc");
  mtp_pre_emb_ = require(weights, "pre_fc_norm_embedding.weight");
  mtp_pre_hid_ = require(weights, "pre_fc_norm_hidden.weight");
  mtp_norm_ = require(weights, "norm.weight");
  mtp_layer_.is_linear = false;
  mtp_layer_.input_norm = require(weights, "layers.0.input_layernorm.weight");
  mtp_layer_.post_norm = require(weights, "layers.0.post_attention_layernorm.weight");
  mtp_layer_.gate_proj = load_qlinear(weights, "layers.0.mlp.gate_proj");
  mtp_layer_.up_proj = load_qlinear(weights, "layers.0.mlp.up_proj");
  mtp_layer_.down_proj = load_qlinear(weights, "layers.0.mlp.down_proj");
  mtp_layer_.attn.q_proj = load_qlinear(weights, "layers.0.self_attn.q_proj");
  mtp_layer_.attn.k_proj = load_qlinear(weights, "layers.0.self_attn.k_proj");
  mtp_layer_.attn.v_proj = load_qlinear(weights, "layers.0.self_attn.v_proj");
  mtp_layer_.attn.o_proj = load_qlinear(weights, "layers.0.self_attn.o_proj");
  mtp_layer_.attn.q_norm = require(weights, "layers.0.self_attn.q_norm.weight");
  mtp_layer_.attn.k_norm = require(weights, "layers.0.self_attn.k_norm.weight");
  mtp_block_ = 3;
  mtp_valid_ = true;
  mtp_reset();
}

int32_t Engine::emit_scheduled() {
  eval(pending_tok_);
  last_emitted_ = pending_tok_.item<int32_t>();
  return last_emitted_;
}

int32_t Engine::prefill(const int32_t* tokens, int n, bool schedule_decode) {
  if (n <= 0) {
    throw std::runtime_error("prefill requires at least one token");
  }

  const int32_t* new_tokens = tokens;
  int new_token_count = n;
  if (request_boundary_pending_) {
    const bool can_reuse = !mtp_valid_ && token_history_.size() < size_t(n) &&
        std::equal(token_history_.begin(), token_history_.end(), tokens);
    if (can_reuse) {
      new_tokens += token_history_.size();
      new_token_count -= static_cast<int>(token_history_.size());
      reset_decode_pipeline();
      request_boundary_pending_ = false;
    } else {
      reset();
    }
  }

  token_history_.insert(
      token_history_.end(), new_tokens, new_tokens + new_token_count);
  array ids(new_tokens, {1, new_token_count}, mx::int32);
  array first = greedy_token(forward_hidden(ids));
  async_eval(first);
  if (!schedule_decode) {
    pending_tok_ = first;
    decode_scheduled_ = false;
    last_emitted_in_state_ = false;
    return emit_scheduled();
  }
  array following = greedy_token(forward_hidden(reshape(first, {1, 1})));
  async_eval(following);
  pending_tok_ = first;
  int32_t out = emit_scheduled();
  record_processed_token(out);
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

int32_t Engine::decode(int32_t token) {
  if (mtp_valid_) {
    if (spec_buf_pos_ >= spec_buf_n_ || token != last_emitted_) {
      spec_refill(token);
    }
    last_emitted_ = spec_buf_[spec_buf_pos_++];
    decode_scheduled_ = false;
    return last_emitted_;
  }
  if (decode_scheduled_ && token == last_emitted_) {
    array following = greedy_token(forward_hidden(reshape(pending_tok_, {1, 1})));
    async_eval(following);
    int32_t out = emit_scheduled();
    pending_tok_ = following;
    record_processed_token(out);
    return out;
  }
  if (!last_emitted_in_state_ || token != last_emitted_) {
    token_history_.push_back(token);
  }
  array ids = (token == last_emitted_) ? reshape(pending_tok_, {1, 1})
                                       : array(&token, {1, 1}, mx::int32);
  array next = greedy_token(forward_hidden(ids));
  async_eval(next);
  array following = greedy_token(forward_hidden(reshape(next, {1, 1})));
  async_eval(following);
  pending_tok_ = next;
  int32_t out = emit_scheduled();
  record_processed_token(out);
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

} // namespace mlx_qwen38
} // namespace sglang
