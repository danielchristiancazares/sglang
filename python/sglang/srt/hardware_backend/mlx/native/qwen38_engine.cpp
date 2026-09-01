#include "qwen38_engine.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/random.h"
#include "mlx/stream.h"
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

bool native_sampling_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_SAMPLING");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_state_trace_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_TRACE_STATE");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_spec_trace_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TRACE_SPEC");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_small_batch_qmm_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_SMALL_BATCH_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_m8_ksplit_qmm_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_M8_KSPLIT_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_qmm_trace_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TRACE_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_dflash_tape_commit_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DFLASH_TAPE_COMMIT");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

int native_dspark_verify_draft_tokens() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS");
  if (value == nullptr || *value == '\0') {
    return 7;
  }
  int count = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (error != std::errc() || end != text.data() + text.size() || count < 1 ||
      count > 7) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS must be an integer "
        "from 1 through 7");
  }
  return count;
}

enum class LinearAttnOverrideScope {
  kOutProjection,
  kQkvProjection,
  kZProjection,
  kQkvAndZProjections,
  kAllProjections,
};

LinearAttnOverrideScope linear_attn_override_scope() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "all") {
    return LinearAttnOverrideScope::kAllProjections;
  }
  if (std::string_view(value) == "qkv") {
    return LinearAttnOverrideScope::kQkvProjection;
  }
  if (std::string_view(value) == "z") {
    return LinearAttnOverrideScope::kZProjection;
  }
  if (std::string_view(value) == "qkvz") {
    return LinearAttnOverrideScope::kQkvAndZProjections;
  }
  throw std::runtime_error(
      "SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE must be all, qkv, z, or qkvz");
}

size_t linear_attn_projection_count(LinearAttnOverrideScope scope) {
  switch (scope) {
    case LinearAttnOverrideScope::kOutProjection:
    case LinearAttnOverrideScope::kQkvProjection:
    case LinearAttnOverrideScope::kZProjection:
      return 1;
    case LinearAttnOverrideScope::kQkvAndZProjections:
      return 2;
    case LinearAttnOverrideScope::kAllProjections:
      return 5;
  }
  throw std::runtime_error("invalid linear-attention override scope");
}

bool is_linear_attn_override_tensor(
    std::string_view key, LinearAttnOverrideScope scope) {
  const bool is_quant_tensor = key.ends_with(".weight") ||
      key.ends_with(".scales") || key.ends_with(".biases");
  const bool is_out_projection =
      key.find(".linear_attn.out_proj.") != std::string_view::npos;
  const bool is_qkv_projection =
      key.find(".linear_attn.in_proj_qkv.") != std::string_view::npos;
  const bool is_z_projection =
      key.find(".linear_attn.in_proj_z.") != std::string_view::npos;
  if (!is_quant_tensor) {
    return false;
  }
  switch (scope) {
    case LinearAttnOverrideScope::kOutProjection:
      return is_out_projection;
    case LinearAttnOverrideScope::kQkvProjection:
      return is_qkv_projection;
    case LinearAttnOverrideScope::kZProjection:
      return is_z_projection;
    case LinearAttnOverrideScope::kQkvAndZProjections:
      return is_qkv_projection || is_z_projection;
    case LinearAttnOverrideScope::kAllProjections:
      return is_out_projection || is_qkv_projection || is_z_projection ||
          key.find(".linear_attn.in_proj_a.") != std::string_view::npos ||
          key.find(".linear_attn.in_proj_b.") != std::string_view::npos;
  }
  throw std::runtime_error("invalid linear-attention override scope");
}

size_t apply_linear_attn_override(
    std::unordered_map<std::string, array>& weights,
    const std::string& model_dir,
    LinearAttnOverrideScope scope) {
  DIR* dir = opendir(model_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error(
        "cannot open linear-attention override dir " + model_dir);
  }
  std::vector<std::string> shards;
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() >= 12 &&
        name.substr(name.size() - 12) == ".safetensors") {
      shards.push_back(model_dir + "/" + name);
    }
  }
  closedir(dir);

  size_t replaced = 0;
  for (const std::string& shard : shards) {
    auto loaded = mx::load_safetensors(shard);
    for (auto& kv : loaded.first) {
      if (!is_linear_attn_override_tensor(kv.first, scope)) {
        continue;
      }
      auto target = weights.find(kv.first);
      if (target == weights.end()) {
        throw std::runtime_error(
            "linear-attention override has unknown weight " + kv.first);
      }
      target->second = std::move(kv.second);
      ++replaced;
    }
  }
  return replaced;
}

std::uint64_t native_sampling_seed() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_SAMPLING_SEED");
  if (value == nullptr || *value == '\0') {
    return 67396869;
  }
  const std::string_view text(value);
  std::uint64_t seed = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), seed);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_SAMPLING_SEED must be an unsigned integer");
  }
  return seed;
}

int native_reasoning_token_limit() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  const std::string_view text(value);
  int limit = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), limit);
  if (error != std::errc() || end != text.data() + text.size() || limit <= 0) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS must be a positive integer");
  }
  return limit;
}

std::string layer_key(int i, const std::string& rest) {
  return "language_model.model.layers." + std::to_string(i) + rest;
}

array last_token(const array& hidden) {
  auto shape = hidden.shape();
  int seq = static_cast<int>(shape[1]);
  return squeeze(slice(hidden, {0, seq - 1, 0}, {1, seq, shape[2]}), 1);
}

constexpr const char* kGatedDeltaHeader = R"(
#define SGLANG_STORE_GATED_DELTA(value, time_index)
)";

constexpr const char* kGatedDeltaTapeHeader = R"(
#define SGLANG_STORE_GATED_DELTA(value, time_index) \
  if (thread_index_in_simdgroup == 0) { \
    delta_out[(((b_idx * T + (time_index)) * Hv + hv_idx) * Dv) + dv_idx] = \
        (value); \
  }
)";

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
            SGLANG_STORE_GATED_DELTA(delta, t);

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

constexpr const char* kGatedDeltaCommitSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        constexpr int ValuesPerThread = Dk / 32;

        auto input_state = state_in + (n * Dv + dv_idx) * Dk;
        auto output_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[ValuesPerThread];
#pragma unroll
        for (int index = 0; index < ValuesPerThread; ++index) {
          const int dimension = ValuesPerThread * dk_idx + index;
          state[index] = static_cast<float>(input_state[dimension]);
        }

        for (int time_index = 0; time_index < token_count; ++time_index) {
          const float decay_value = decay[
              (b_idx * T + time_index) * Hv + hv_idx];
          const float delta_value = delta[
              ((b_idx * T + time_index) * Hv + hv_idx) * Dv + dv_idx];
          auto key = keys +
              (b_idx * T + time_index) * Hk * Dk + hk_idx * Dk;
#pragma unroll
          for (int index = 0; index < ValuesPerThread; ++index) {
            const int dimension = ValuesPerThread * dk_idx + index;
            state[index] = state[index] * decay_value;
            state[index] =
                state[index] + static_cast<float>(key[dimension]) * delta_value;
          }
        }

#pragma unroll
        for (int index = 0; index < ValuesPerThread; ++index) {
          const int dimension = ValuesPerThread * dk_idx + index;
          output_state[dimension] = state[index];
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

constexpr const char* kAffineSmallBatchQmmHeader = R"(
#include <metal_simdgroup>
)";

constexpr const char* kAffineSmallBatchQmmSource = R"(
        constexpr ushort OutputTile = 64;
        constexpr ushort RowTile = 8;
        constexpr ushort KTile = 64;
        constexpr ushort Threads = 128;
        constexpr ushort ValuesPerWord = 32 / Bits;
        constexpr ushort WordsPerRow = KTile / ValuesPerWord;
        constexpr ushort QuantGroupsPerTile = KTile / 64;

        threadgroup bfloat staged_x[RowTile * KTile];
        threadgroup bfloat staged_w[OutputTile * KTile];
        threadgroup float staged_scales[
            OutputTile * QuantGroupsPerTile];
        threadgroup float staged_biases[
            OutputTile * QuantGroupsPerTile];
        threadgroup float staged_y[RowTile * OutputTile];

        const ushort tid = thread_position_in_threadgroup.x;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.x * OutputTile;
        const uint packed_k = K * Bits / 32;
        const uint groups_per_row = K / 64;

        simdgroup_float8x8 accumulators[2];
        accumulators[0] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        accumulators[1] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);

        for (uint k_start = 0; k_start < K; k_start += KTile) {
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (ushort index = tid; index < RowTile * KTile;
               index += Threads) {
            const ushort row = index / KTile;
            const ushort column = index % KTile;
            staged_x[index] = row < M
                ? x[row * K + k_start + column]
                : static_cast<bfloat>(0.0f);
          }

          if (tid < OutputTile * QuantGroupsPerTile) {
            const ushort output = tid / QuantGroupsPerTile;
            const ushort group = tid % QuantGroupsPerTile;
            const uint parameter_index =
                (output_start + output) * groups_per_row +
                k_start / 64 + group;
            staged_scales[tid] = static_cast<float>(scales[parameter_index]);
            staged_biases[tid] = static_cast<float>(biases[parameter_index]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          constexpr ushort PackedWords = OutputTile * WordsPerRow;
          for (ushort index = tid; index < PackedWords;
               index += Threads) {
            const ushort output = index / WordsPerRow;
            const ushort word_column = index % WordsPerRow;
            const uint packed = w[
                (output_start + output) * packed_k +
                k_start / ValuesPerWord + word_column];
            const ushort group =
                word_column * ValuesPerWord / 64;
            const ushort parameter = output * QuantGroupsPerTile + group;
            const float scale = staged_scales[parameter];
            const float bias = staged_biases[parameter];
#pragma unroll
            for (ushort value = 0; value < ValuesPerWord; ++value) {
              const uint quantized =
                  (packed >> (value * Bits)) & ((1u << Bits) - 1u);
              staged_w[
                  output * KTile + word_column * ValuesPerWord + value] =
                  static_cast<bfloat>(scale * quantized + bias);
            }
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          threadgroup const bfloat* input_tile = staged_x;
          threadgroup const bfloat* weight_tile =
              staged_w + simd_id * 16 * KTile;
#pragma unroll
          for (ushort k_step = 0; k_step < KTile / 8; ++k_step) {
            simdgroup_bfloat8x8 input_fragment;
            simdgroup_bfloat8x8 weight_fragment;
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(
                input_fragment, input_tile + k_step * 8, KTile, 0, false);
#pragma unroll
            for (ushort output_step = 0; output_step < 2; ++output_step) {
              simdgroup_load(
                  weight_fragment,
                  weight_tile + output_step * 8 * KTile + k_step * 8,
                  KTile,
                  0,
                  true);
              simdgroup_multiply_accumulate(
                  accumulators[output_step],
                  input_fragment,
                  weight_fragment,
                  accumulators[output_step]);
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort output_step = 0; output_step < 2; ++output_step) {
          simdgroup_store(
              accumulators[output_step],
              staged_y + simd_id * 16 + output_step * 8,
              OutputTile,
              0,
              false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort index = tid; index < M * OutputTile; index += Threads) {
          const ushort row = index / OutputTile;
          const ushort column = index % OutputTile;
          y[row * N + output_start + column] =
              static_cast<bfloat>(staged_y[row * OutputTile + column]);
        }
)";

constexpr const char* kAffineM8KsplitQmmSource = R"(
        constexpr ushort Rows = 8;
        constexpr ushort OutputTile = 32;
        constexpr ushort KTile = 32;
        constexpr ushort KStep = 8;
        constexpr ushort SimdGroups = 16;
        constexpr uint K = KConst;
        constexpr uint PackedK = K / 8;
        constexpr uint QuantGroups = K / 64;
        constexpr uint KChunk = K / SimdGroups;

        constexpr uint StagedElements =
            SimdGroups * KTile * OutputTile;
        // Weight staging and cross-SIMDgroup reduction have disjoint
        // lifetimes. Reuse the 32 KiB staging allocation for FP32 partials.
        threadgroup float storage[StagedElements / 2];
        threadgroup bfloat* staged_w =
            reinterpret_cast<threadgroup bfloat*>(storage);
        threadgroup float* partial = storage;

        const ushort tid = thread_position_in_threadgroup.x;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.y * OutputTile;
        const uint k_begin = simd_id * KChunk;
        const uint k_end = k_begin + KChunk;

        simdgroup_bfloat8x8 input_fragment;
        simdgroup_bfloat8x8 weight_fragment;
        simdgroup_float8x8 accumulators[4];
#pragma unroll
        for (ushort output_step = 0; output_step < 4; ++output_step) {
          accumulators[output_step] =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
        }

        const ushort output_column = lane;
        for (uint k_start = k_begin; k_start < k_end; k_start += KTile) {
#pragma unroll
          for (ushort pack_in_tile = 0; pack_in_tile < 4; ++pack_in_tile) {
            const uint k_base = k_start + pack_in_tile * 8;
            const uint output = output_start + output_column;
            const uint packed = w[output * PackedK + k_base / 8];
            const uint parameter = output * QuantGroups + k_base / 64;
            const float scale = static_cast<float>(scales[parameter]);
            const float bias = static_cast<float>(biases[parameter]);
#pragma unroll
            for (ushort value = 0; value < 8; ++value) {
              const uint quantized = (packed >> (value * 4)) & 0xfu;
              staged_w[
                  simd_id * KTile * OutputTile +
                  (pack_in_tile * 8 + value) * OutputTile + output_column] =
                  static_cast<bfloat>(scale * quantized + bias);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
          for (ushort k_step = 0; k_step < KTile / KStep; ++k_step) {
            simdgroup_load(
                input_fragment, x + k_start + k_step * KStep, K);
#pragma unroll
            for (ushort output_step = 0; output_step < 4; ++output_step) {
              simdgroup_load(
                  weight_fragment,
                  staged_w + simd_id * KTile * OutputTile +
                      k_step * KStep * OutputTile +
                      output_step * 8,
                  OutputTile);
              simdgroup_multiply_accumulate(
                  accumulators[output_step],
                  input_fragment,
                  weight_fragment,
                  accumulators[output_step]);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort output_step = 0; output_step < 4; ++output_step) {
          simdgroup_store(
              accumulators[output_step],
              partial + simd_id * Rows * OutputTile + output_step * 8,
              OutputTile);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort offset = tid; offset < Rows * OutputTile;
             offset += SimdGroups * 32) {
          float value = 0.0f;
#pragma unroll
          for (ushort group = 0; group < SimdGroups; ++group) {
            value += partial[group * Rows * OutputTile + offset];
          }
          const ushort row = offset / OutputTile;
          const ushort column = offset % OutputTile;
          y[row * N_size + output_start + column] =
              static_cast<bfloat>(value);
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
      kGatedDeltaSource,
      kGatedDeltaHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_tape_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_step_tape",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out", "delta_out"},
      kGatedDeltaSource,
      kGatedDeltaTapeHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_commit_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_commit",
      {"keys", "decay", "delta", "state_in", "T", "token_count"},
      {"state_out"},
      kGatedDeltaCommitSource);
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

const mx::fast::CustomKernelFunction& affine_small_batch_qmm_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_affine_small_batch_qmm",
      {"w", "scales", "biases", "x", "K", "N", "M"},
      {"y"},
      kAffineSmallBatchQmmSource,
      kAffineSmallBatchQmmHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_m8_ksplit_qmm_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_affine_m8_ksplit_qmm",
      {"w", "scales", "biases", "x", "N_size"},
      {"y"},
      kAffineM8KsplitQmmSource,
      kAffineSmallBatchQmmHeader);
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

array dspark_yarn_rope(const array& x, int offset) {
  constexpr int kHeadDim = 128;
  constexpr int kFrequencyCount = kHeadDim / 2;
  constexpr double kBase = 10000000.0;
  constexpr double kFactor = 32.0;
  constexpr double kOriginalPositions = 8192.0;
  constexpr double kBetaFast = 32.0;
  constexpr double kBetaSlow = 1.0;
  if (x.ndim() == 0 || x.shape().back() != kHeadDim || offset < 0) {
    throw std::runtime_error("invalid DSpark YaRN RoPE input");
  }

  static const array frequencies = [] {
    const auto correction_dimension = [](double rotations) {
      return kHeadDim *
          std::log(kOriginalPositions / (rotations * 2.0 * std::numbers::pi)) /
          (2.0 * std::log(kBase));
    };
    const int low = std::max(
        static_cast<int>(std::floor(correction_dimension(kBetaFast))), 0);
    const int high = std::min(
        static_cast<int>(std::ceil(correction_dimension(kBetaSlow))),
        kHeadDim - 1);
    std::vector<float> values(kFrequencyCount);
    for (int index = 0; index < kFrequencyCount; ++index) {
      const double position_frequency =
          std::pow(kBase, (2.0 * index) / kHeadDim);
      const double extrapolated = 1.0 / position_frequency;
      const double interpolated = extrapolated / kFactor;
      const double ramp = std::clamp(
          (static_cast<double>(index) - low) /
              static_cast<double>(high - low),
          0.0,
          1.0);
      const double extrapolation_mask = 1.0 - ramp;
      const double inverse_frequency =
          interpolated * (1.0 - extrapolation_mask) +
          extrapolated * extrapolation_mask;
      values[static_cast<size_t>(index)] =
          static_cast<float>(1.0 / inverse_frequency);
    }
    array result(values.begin(), {kFrequencyCount}, mx::float32);
    eval(result);
    return result;
  }();
  static const float attention_factor =
      static_cast<float>(0.1 * std::log(kFactor) + 1.0);
  array rotated = mx::fast::rope(
      x,
      kHeadDim,
      /*traditional=*/false,
      std::nullopt,
      1.0f,
      offset,
      frequencies);
  return rotated * array(attention_factor, rotated.dtype());
}

array dspark_confidence(
    const array& hidden,
    const array& markov_embeddings,
    const array& weight,
    const array& bias) {
  if (hidden.ndim() != 3 || markov_embeddings.ndim() != 3 ||
      hidden.shape()[0] != markov_embeddings.shape()[0] ||
      hidden.shape()[1] != markov_embeddings.shape()[1] ||
      hidden.dtype() != mx::bfloat16 ||
      markov_embeddings.dtype() != mx::bfloat16 ||
      weight.dtype() != mx::bfloat16 || bias.dtype() != mx::bfloat16 ||
      weight.shape() != mx::Shape{
          1, hidden.shape()[2] + markov_embeddings.shape()[2]} ||
      bias.shape() != mx::Shape{1}) {
    throw std::runtime_error("invalid DSpark confidence inputs");
  }
  array features = concatenate({hidden, markov_embeddings}, -1);
  array raw = mx::matmul(features, transpose(weight)) + bias;
  return squeeze(sigmoid(astype(raw, mx::float32)), -1);
}

array QLinear::operator()(const array& x) const {
  if (!valid) {
    throw std::runtime_error("QLinear used before load");
  }
  if (w.dtype() == mx::bfloat16) {
    if (x.ndim() == 0 || x.dtype() != mx::bfloat16 || w.ndim() != 2 ||
        x.shape().back() != w.shape()[1]) {
      throw std::runtime_error("invalid dense QLinear inputs");
    }
    return matmul(x, transpose(w, {1, 0}));
  }
  if (w.dtype() != mx::uint32) {
    throw std::runtime_error("unsupported QLinear weight dtype");
  }
  if (native_small_batch_qmm_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] >= 6 && x.shape()[1] <= 8) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (native_m8_ksplit_qmm_enabled() && x.shape()[1] == 8 &&
        x.dtype() == mx::bfloat16 && w.dtype() == mx::uint32 &&
        scales.dtype() == mx::bfloat16 && biases.dtype() == mx::bfloat16 &&
        group_size == 64 && bits == 4 && input_features % 512 == 0 &&
        output_features % 32 == 0) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmm m8_ksplit rows=8 K=%d N=%d bits=4\n",
            input_features,
            output_features);
      }
      return affine_qmm_m8_ksplit(*this, x);
    }
    if (x.dtype() == mx::bfloat16 && w.dtype() == mx::uint32 &&
        scales.dtype() == mx::bfloat16 && biases.dtype() == mx::bfloat16 &&
        group_size == 64 && (bits == 2 || bits == 4) &&
        input_features % 64 == 0 && output_features % 64 == 0 &&
        output_features >= 6144) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmm custom rows=%d K=%d N=%d bits=%d\n",
            static_cast<int>(x.shape()[1]),
            input_features,
            output_features,
            bits);
      }
      return affine_qmm_small_batch(*this, x);
    }
    if (native_qmm_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_qmm fallback rows=%d K=%d N=%d bits=%d\n",
          static_cast<int>(x.shape()[1]),
          input_features,
          output_features,
          bits);
    }
  }
  return quantized_matmul(
      x, w, scales, biases, /*transpose=*/true, group_size, bits, "affine");
}

array affine_qmm_small_batch(const QLinear& linear, const array& x) {
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] < 2 || x.shape()[1] > 8 ||
      x.dtype() != mx::bfloat16 || linear.w.dtype() != mx::uint32 ||
      linear.scales.dtype() != mx::bfloat16 ||
      linear.biases.dtype() != mx::bfloat16 || linear.group_size != 64 ||
      (linear.bits != 2 && linear.bits != 4)) {
    throw std::runtime_error("invalid small-batch affine QMM inputs");
  }
  const int rows = static_cast<int>(x.shape()[1]);
  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features % 64 != 0 || output_features % 64 != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported small-batch affine QMM shape");
  }

  auto outputs = affine_small_batch_qmm_metal()(
      {linear.w,
       linear.scales,
       linear.biases,
       x,
       array(input_features, mx::int32),
       array(output_features, mx::int32),
       array(rows, mx::int32)},
      {{1, rows, output_features}},
      {x.dtype()},
      {output_features / 64 * 128, 1, 1},
      {128, 1, 1},
      {{"Bits", mx::fast::TemplateArg{linear.bits}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_qmm_m8_ksplit(const QLinear& linear, const array& x) {
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 8 || x.dtype() != mx::bfloat16 ||
      linear.w.dtype() != mx::uint32 ||
      linear.scales.dtype() != mx::bfloat16 ||
      linear.biases.dtype() != mx::bfloat16 || linear.group_size != 64 ||
      linear.bits != 4) {
    throw std::runtime_error("invalid M8 K-split affine QMM inputs");
  }
  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features % 512 != 0 || output_features % 32 != 0 ||
      linear.w.shape()[1] * 8 != input_features ||
      linear.scales.shape() != mx::Shape{
          output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported M8 K-split affine QMM shape");
  }

  auto outputs = affine_m8_ksplit_qmm_metal()(
      {linear.w,
       linear.scales,
       linear.biases,
       x,
       array(output_features, mx::int32)},
      {{1, 8, output_features}},
      {x.dtype()},
      {512, output_features / 32, 1},
      {512, 1, 1},
      {{"KConst", mx::fast::TemplateArg{input_features}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
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

std::vector<array> gated_delta_update_outputs(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state,
    bool capture_delta) {
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
  std::vector<mx::Shape> output_shapes = {
      {B, T, Hv, Dv}, state.shape()};
  std::vector<mx::Dtype> output_dtypes = {q.dtype(), state.dtype()};
  if (capture_delta) {
    output_shapes.push_back({B, T, Hv, Dv});
    output_dtypes.push_back(mx::float32);
  }
  const auto& kernel =
      capture_delta ? gated_delta_tape_metal() : gated_delta_metal();
  return kernel(
      {q, k, v, g, beta, state, array(T, mx::int32)},
      output_shapes,
      output_dtypes,
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
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  auto outputs = gated_delta_update_outputs(
      q, k, v, g, beta, state, false);
  return {outputs[0], outputs[1]};
}

array gated_delta_commit(
    const array& keys,
    const array& decay,
    const array& delta,
    const array& state,
    int token_count) {
  if (keys.ndim() != 4 || decay.ndim() != 3 || delta.ndim() != 4 ||
      state.ndim() != 4 || keys.shape()[0] != delta.shape()[0] ||
      keys.shape()[0] != state.shape()[0] ||
      keys.shape()[0] != decay.shape()[0] ||
      keys.shape()[1] != delta.shape()[1] ||
      keys.shape()[1] != decay.shape()[1] ||
      delta.shape()[2] != state.shape()[1] ||
      delta.shape()[2] != decay.shape()[2] ||
      delta.shape()[3] != state.shape()[2] ||
      keys.shape()[3] != state.shape()[3] ||
      state.shape()[1] % keys.shape()[2] != 0 ||
      keys.shape()[3] % 32 != 0 || token_count <= 0 ||
      token_count > keys.shape()[1] || decay.dtype() != mx::float32 ||
      delta.dtype() != mx::float32 || state.dtype() != mx::float32) {
    throw std::runtime_error("invalid gated-delta commit tape");
  }
  const int batch = static_cast<int>(keys.shape()[0]);
  const int tape_length = static_cast<int>(keys.shape()[1]);
  const int key_heads = static_cast<int>(keys.shape()[2]);
  const int key_dim = static_cast<int>(keys.shape()[3]);
  const int value_heads = static_cast<int>(state.shape()[1]);
  const int value_dim = static_cast<int>(state.shape()[2]);
  auto outputs = gated_delta_commit_metal()(
      {keys,
       decay,
       delta,
       state,
       array(tape_length, mx::int32),
       array(token_count, mx::int32)},
      {state.shape()},
      {state.dtype()},
      {32, value_dim, batch * value_heads},
      {32, 4, 1},
      {
          {"Dk", mx::fast::TemplateArg{key_dim}},
          {"Dv", mx::fast::TemplateArg{value_dim}},
          {"Hk", mx::fast::TemplateArg{key_heads}},
          {"Hv", mx::fast::TemplateArg{value_heads}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

Engine::Engine(MlxQwen38Config cfg, const std::string& model_dir)
    : cfg_(configure_mlx_runtime(cfg)) {
  if (cfg_.hidden_size <= 0 || cfg_.num_hidden_layers <= 0) {
    throw std::runtime_error("invalid Qwen3.8 config");
  }
  layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
  load_weights(model_dir);
  sampling_enabled_ = native_sampling_enabled();
  if (sampling_enabled_) {
    sampling_seed_ = native_sampling_seed();
  }
  max_reasoning_tokens_ = native_reasoning_token_limit();
  reset();
}

void Engine::reset() {
  if (native_state_trace_enabled()) {
    std::fprintf(
        stderr, "qwen38_native reset history=%zu pending=%d\n",
        token_history_.size(), request_boundary_pending_ ? 1 : 0);
  }
  for (auto& layer : layers_) {
    if (layer.is_linear) {
      layer.linear.has_state = false;
    } else {
      layer.attn.offset = 0;
      layer.attn.cache_length = 0;
    }
  }
  if (sampling_enabled_) {
    mx::random::seed(sampling_seed_);
  }
  reset_decode_pipeline();
  request_boundary_pending_ = false;
  token_history_.clear();
  selected_reasoning_tokens_ = 0;
  reasoning_open_ = false;
  reasoning_cap_selected_ = false;
  prompt_snapshot_valid_ = false;
  prompt_snapshot_history_.clear();
  snap_.clear();
  if (mtp_valid_) {
    mtp_reset();
  }
  if (dflash_valid_) {
    dflash_reset();
  }
  if (dspark_valid_) {
    dspark_reset();
  }
}

void Engine::begin_request() {
  if (native_state_trace_enabled()) {
    std::fprintf(
        stderr, "qwen38_native begin_request history=%zu\n",
        token_history_.size());
  }
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

  const char* const out_proj_override =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_OUT_PROJ_OVERRIDE_PATH");
  const char* const linear_attn_override =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_PATH");
  const bool has_out_proj_override =
      out_proj_override != nullptr && *out_proj_override != '\0';
  const bool has_linear_attn_override =
      linear_attn_override != nullptr && *linear_attn_override != '\0';
  if (has_out_proj_override && has_linear_attn_override) {
    throw std::runtime_error(
        "linear-attention override paths are mutually exclusive");
  }
  if (has_out_proj_override || has_linear_attn_override) {
    const LinearAttnOverrideScope scope = has_linear_attn_override
        ? linear_attn_override_scope()
        : LinearAttnOverrideScope::kOutProjection;
    const char* const override_path =
        has_linear_attn_override ? linear_attn_override : out_proj_override;
    const size_t replaced =
        apply_linear_attn_override(weights, override_path, scope);
    const size_t linear_layers = static_cast<size_t>(
        cfg_.num_hidden_layers -
        cfg_.num_hidden_layers / cfg_.full_attention_interval);
    const size_t projections_per_layer = linear_attn_projection_count(scope);
    const size_t expected = linear_layers * projections_per_layer * 3;
    if (replaced != expected) {
      throw std::runtime_error(
          "linear-attention override replaced " + std::to_string(replaced) +
          " tensors; expected " + std::to_string(expected));
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

array Engine::gated_delta(
    LinearAttn& lin, const array& x, LinearCommitTape* commit_tape) {
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
  if (commit_tape != nullptr) {
    commit_tape->conv_tokens = qkv;
    commit_tape->valid = false;
  }
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
  auto updated = gated_delta_update_outputs(
      q, k, v, g, beta, lin.rec_state, commit_tape != nullptr);
  lin.rec_state = updated[1];
  array out = updated[0];
  if (commit_tape != nullptr) {
    commit_tape->keys = k;
    commit_tape->decay = g;
    commit_tape->delta = updated[2];
    commit_tape->valid = true;
  }
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
  return forward_hidden_impl(tokens, nullptr, nullptr);
}

TargetForward Engine::forward_hidden_captured(
    const array& tokens, bool capture_commit_tape) {
  TargetForward result;
  result.captured.reserve(5);
  if (capture_commit_tape) {
    result.linear_tapes.resize(layers_.size());
  }
  result.hidden = forward_hidden_impl(
      tokens,
      &result.captured,
      capture_commit_tape ? &result.linear_tapes : nullptr);
  if (result.captured.size() != 5) {
    throw std::runtime_error(
        "speculative draft capture requires layers 5, 19, 33, 47, and 61");
  }
  return result;
}

array Engine::forward_hidden_impl(
    const array& tokens,
    std::vector<array>* captured,
    std::vector<LinearCommitTape>* commit_tapes) {
  array h = embed(tokens);
  const auto capture = [&captured, &h](size_t layer_index) {
    if (captured != nullptr &&
        (layer_index == 5 || layer_index == 19 || layer_index == 33 ||
         layer_index == 47 || layer_index == 61)) {
      captured->push_back(h);
    }
  };
  const bool single_token = tokens.shape()[1] == 1;
  if (!single_token) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto& layer = layers_[i];
      array n = mx::fast::rms_norm(h, layer.input_norm, cfg_.rms_norm_eps);
      LinearCommitTape* const commit_tape =
          commit_tapes != nullptr && layer.is_linear
          ? &(*commit_tapes)[i]
          : nullptr;
      array r = layer.is_linear
          ? gated_delta(layer.linear, n, commit_tape)
          : full_attn(layer.attn, n);
      h = h + r;
      array n2 = mx::fast::rms_norm(h, layer.post_norm, cfg_.rms_norm_eps);
      h = h + mlp(layer, n2);
      capture(i);
    }
    return h;
  }

  array n = mx::fast::rms_norm(
      h, layers_.front().input_norm, cfg_.rms_norm_eps);
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    LinearCommitTape* const commit_tape =
        commit_tapes != nullptr && layer.is_linear
        ? &(*commit_tapes)[i]
        : nullptr;
    array r = layer.is_linear
        ? gated_delta(layer.linear, n, commit_tape)
        : full_attn(layer.attn, n);
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
    capture(i);
  }
  return h;
}

array Engine::select_token(const array& hidden) {
  last_hidden_ = last_token(hidden);
  if (reasoning_open_) {
    if (max_reasoning_tokens_ > 0 && !reasoning_cap_selected_ &&
        selected_reasoning_tokens_ >= max_reasoning_tokens_) {
      reasoning_cap_selected_ = true;
      reasoning_open_ = false;
      return array({248069}, mx::int32);
    }
    ++selected_reasoning_tokens_;
  }
  array token_logits = logits(last_hidden_);
  if (!sampling_enabled_) {
    return mx::argmax(token_logits, -1);
  }

  // Qwen3.8's model sampling contract is temperature 1.0, top-k 20, top-p
  // 0.95. Keep the vocabulary partition and normalization on Metal, retain
  // the model's full-vocabulary nucleus threshold, then run Gumbel-max over
  // the surviving top-k candidates. This preserves the native graph's
  // asynchronous two-token pipeline and transfers no logits to the host.
  constexpr int kTopK = 20;
  constexpr float kTopP = 0.95f;
  const auto shape = token_logits.shape();
  const int batch = static_cast<int>(shape[0]);
  const int vocab = static_cast<int>(shape[1]);
  if (vocab < kTopK) {
    throw std::runtime_error("native sampling vocabulary is smaller than top-k");
  }

  array partitioned = mx::argpartition(token_logits, vocab - kTopK, -1);
  array candidate_ids =
      slice(partitioned, {0, vocab - kTopK}, {batch, vocab});
  array candidate_logits = astype(
      mx::take_along_axis(token_logits, candidate_ids, -1), mx::float32);
  array order = mx::argsort(-candidate_logits, -1);
  candidate_ids = mx::take_along_axis(candidate_ids, order, -1);
  candidate_logits = mx::take_along_axis(candidate_logits, order, -1);

  array candidate_probs = mx::exp(
      candidate_logits -
      mx::logsumexp(astype(token_logits, mx::float32), -1, true));
  array cumulative = mx::cumsum(candidate_probs, -1);
  array keep = mx::less_equal(
      cumulative - candidate_probs, array(kTopP, mx::float32));
  array filtered = mx::where(
      keep,
      candidate_logits,
      array(-std::numeric_limits<float>::infinity(), mx::float32));
  array noise = mx::random::gumbel(filtered.shape(), mx::float32);
  array selected_rank = mx::argmax(filtered + noise, -1);
  return squeeze(
      mx::take_along_axis(candidate_ids, expand_dims(selected_rank, -1), -1),
      -1);
}

array Engine::sampling_probabilities(const array& token_logits) {
  constexpr int kTopK = 20;
  constexpr float kTopP = 0.95f;
  const auto shape = token_logits.shape();
  const int vocab = static_cast<int>(shape.back());
  if (vocab < kTopK) {
    throw std::runtime_error("native sampling vocabulary is smaller than top-k");
  }

  array partitioned = mx::argpartition(token_logits, vocab - kTopK, -1);
  mx::Shape start(shape.size(), 0);
  mx::Shape end(shape);
  start.back() = vocab - kTopK;
  array candidate_ids = slice(partitioned, start, end);
  array candidate_logits = astype(
      mx::take_along_axis(token_logits, candidate_ids, -1), mx::float32);
  array order = mx::argsort(-candidate_logits, -1);
  candidate_ids = mx::take_along_axis(candidate_ids, order, -1);
  candidate_logits = mx::take_along_axis(candidate_logits, order, -1);

  array candidate_probs = mx::exp(
      candidate_logits -
      mx::logsumexp(astype(token_logits, mx::float32), -1, true));
  array cumulative = mx::cumsum(candidate_probs, -1);
  array keep = mx::less_equal(
      cumulative - candidate_probs, array(kTopP, mx::float32));
  array filtered = mx::where(
      keep,
      candidate_logits,
      array(-std::numeric_limits<float>::infinity(), mx::float32));
  array support_probs = mx::softmax(filtered, -1, true);
  return mx::put_along_axis(
      mx::zeros(token_logits.shape(), mx::float32),
      candidate_ids,
      support_probs,
      -1);
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

void Engine::load_dflash2(
    const std::unordered_map<std::string, array>& weights) {
  constexpr size_t kDenseTensorCount = 81;
  constexpr size_t kAffineTensorCount = 175;
  constexpr int kHiddenSize = 5120;
  constexpr int kIntermediateSize = 17408;
  constexpr int kDraftLayers = 5;
  constexpr int kDraftHeads = 32;
  constexpr int kDraftKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSelectorRank = 256;
  constexpr int kDynamicWidth = 1280;
  const bool affine = weights.size() == kAffineTensorCount;
  if (!affine && weights.size() != kDenseTensorCount) {
    throw std::runtime_error(
        "DFlash2 checkpoint contains " + std::to_string(weights.size()) +
        " tensors; expected 81 dense or 175 affine tensors");
  }
  if (cfg_.hidden_size != kHiddenSize || cfg_.num_hidden_layers != 64 ||
      cfg_.vocab_size != 248320) {
    throw std::runtime_error(
        "Qwen3.8-27B DFlash2 requires target shape 64x5120x248320");
  }

  const auto dense = [this, &weights](
                         const std::string& name,
                         const mx::Shape& shape) -> array {
    array value = require(weights, name);
    if (value.shape() != shape || value.dtype() != mx::bfloat16) {
      throw std::runtime_error("invalid DFlash2 tensor " + name);
    }
    return value;
  };
  const auto linear = [this, &weights, &dense, affine](
                          const std::string& prefix,
                          int output_features,
                          int input_features) -> QLinear {
    if (!affine) {
      QLinear value;
      value.w = dense(
          prefix + ".weight", {output_features, input_features});
      value.valid = true;
      return value;
    }
    QLinear value = load_qlinear(weights, prefix);
    if (value.bits != 4 || value.group_size != 64 ||
        value.w.dtype() != mx::uint32 ||
        value.scales.dtype() != mx::bfloat16 ||
        value.biases.dtype() != mx::bfloat16 ||
        value.w.shape()[0] != output_features ||
        value.scales.shape()[0] != output_features ||
        value.scales.shape().back() * value.group_size != input_features) {
      throw std::runtime_error("invalid DFlash2 affine linear " + prefix);
    }
    return value;
  };

  dflash_fc_ = linear("fc", kHiddenSize, kDraftLayers * kHiddenSize);
  dflash_hidden_norm_ =
      dense("hidden_norm.weight", {kHiddenSize});
  dflash_norm_ = dense("norm.weight", {kHiddenSize});
  dflash_selector_hidden_ = linear(
      "candidate_selector.hidden_projection", kSelectorRank, kHiddenSize);
  dflash_predecessor_ = dense(
      "candidate_selector.predecessor_codebook",
      {cfg_.vocab_size, kSelectorRank});
  dflash_successor_ = dense(
      "candidate_selector.successor_codebook",
      {cfg_.vocab_size, kSelectorRank});

  dflash_layers_.clear();
  dflash_layers_.resize(kDraftLayers);
  for (int index = 0; index < kDraftLayers; ++index) {
    DFlashLayer& layer = dflash_layers_[static_cast<size_t>(index)];
    const std::string prefix = "layers." + std::to_string(index);
    layer.input_norm =
        dense(prefix + ".input_layernorm.weight", {kHiddenSize});
    layer.post_norm =
        dense(prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    layer.gate_proj =
        linear(prefix + ".mlp.gate_proj", kIntermediateSize, kHiddenSize);
    layer.up_proj =
        linear(prefix + ".mlp.up_proj", kIntermediateSize, kHiddenSize);
    layer.down_proj =
        linear(prefix + ".mlp.down_proj", kHiddenSize, kIntermediateSize);
    layer.attn.q_proj =
        linear(prefix + ".self_attn.q_proj", kDraftHeads * kHeadDim, kHiddenSize);
    layer.attn.k_proj = linear(
        prefix + ".self_attn.k_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.v_proj = linear(
        prefix + ".self_attn.v_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.o_proj = linear(
        prefix + ".self_attn.o_proj", kHiddenSize, kDraftHeads * kHeadDim);
    layer.attn.q_norm =
        dense(prefix + ".self_attn.q_norm.weight", {kHeadDim});
    layer.attn.k_norm =
        dense(prefix + ".self_attn.k_norm.weight", {kHeadDim});
    layer.attention_conv.base_kernel = dense(
        prefix + ".attention_conv.base_kernel", {2, 2, kHiddenSize});
    layer.attention_conv.kernel_projection = linear(
        prefix + ".attention_conv.kernel_projection",
        kDynamicWidth,
        kHiddenSize);
    layer.mlp_conv.base_kernel = dense(
        prefix + ".mlp_conv.base_kernel", {2, 2, kHiddenSize});
    layer.mlp_conv.kernel_projection = linear(
        prefix + ".mlp_conv.kernel_projection", kDynamicWidth, kHiddenSize);
  }

  mtp_valid_ = false;
  dflash_valid_ = true;
  dspark_valid_ = false;
  dflash_reset();
}

void Engine::load_dspark(
    const std::unordered_map<std::string, array>& weights) {
  constexpr size_t kDenseTensorCount = 62;
  constexpr size_t kAffineTensorCount = 136;
  constexpr int kHiddenSize = 5120;
  constexpr int kIntermediateSize = 17408;
  constexpr int kDraftLayers = 5;
  constexpr int kDraftHeads = 32;
  constexpr int kDraftKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kMarkovRank = 256;
  const bool affine = weights.size() == kAffineTensorCount;
  if (!affine && weights.size() != kDenseTensorCount) {
    throw std::runtime_error(
        "DSpark checkpoint contains " + std::to_string(weights.size()) +
        " tensors; expected 62 dense or 136 affine tensors");
  }
  if (cfg_.hidden_size != kHiddenSize || cfg_.num_hidden_layers != 64 ||
      cfg_.vocab_size != 248320) {
    throw std::runtime_error(
        "Qwen3.8-27B DSpark requires target shape 64x5120x248320");
  }

  const auto dense = [this, &weights](
                         const std::string& name,
                         const mx::Shape& shape) -> array {
    array value = require(weights, name);
    if (value.shape() != shape || value.dtype() != mx::bfloat16) {
      throw std::runtime_error("invalid DSpark tensor " + name);
    }
    return value;
  };
  const auto linear = [this, &weights, &dense, affine](
                          const std::string& prefix,
                          int output_features,
                          int input_features) -> QLinear {
    if (!affine) {
      QLinear value;
      value.w = dense(
          prefix + ".weight", {output_features, input_features});
      value.valid = true;
      return value;
    }
    QLinear value = load_qlinear(weights, prefix);
    if (value.bits != 4 || value.group_size != 64 ||
        value.w.dtype() != mx::uint32 ||
        value.scales.dtype() != mx::bfloat16 ||
        value.biases.dtype() != mx::bfloat16 ||
        value.w.shape()[0] != output_features ||
        value.scales.shape()[0] != output_features ||
        value.scales.shape().back() * value.group_size != input_features) {
      throw std::runtime_error("invalid DSpark affine linear " + prefix);
    }
    return value;
  };

  dspark_fc_ = linear("fc", kHiddenSize, kDraftLayers * kHiddenSize);
  dspark_hidden_norm_ = dense("hidden_norm.weight", {kHiddenSize});
  dspark_norm_ = dense("norm.weight", {kHiddenSize});
  dspark_markov_w1_ = dense(
      "markov_head.markov_w1.weight", {cfg_.vocab_size, kMarkovRank});
  dspark_markov_w2_ =
      linear("markov_head.markov_w2", cfg_.vocab_size, kMarkovRank);
  dspark_confidence_weight_ = dense(
      "confidence_head.proj.weight", {1, kHiddenSize + kMarkovRank});
  dspark_confidence_bias_ = dense("confidence_head.proj.bias", {1});
  dspark_verify_draft_tokens_ = native_dspark_verify_draft_tokens();

  dspark_layers_.clear();
  dspark_layers_.resize(kDraftLayers);
  for (int index = 0; index < kDraftLayers; ++index) {
    DSparkLayer& layer = dspark_layers_[static_cast<size_t>(index)];
    const std::string prefix = "layers." + std::to_string(index);
    layer.input_norm =
        dense(prefix + ".input_layernorm.weight", {kHiddenSize});
    layer.post_norm =
        dense(prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    layer.gate_proj =
        linear(prefix + ".mlp.gate_proj", kIntermediateSize, kHiddenSize);
    layer.up_proj =
        linear(prefix + ".mlp.up_proj", kIntermediateSize, kHiddenSize);
    layer.down_proj =
        linear(prefix + ".mlp.down_proj", kHiddenSize, kIntermediateSize);
    layer.attn.q_proj = linear(
        prefix + ".self_attn.q_proj", kDraftHeads * kHeadDim, kHiddenSize);
    layer.attn.k_proj = linear(
        prefix + ".self_attn.k_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.v_proj = linear(
        prefix + ".self_attn.v_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.o_proj = linear(
        prefix + ".self_attn.o_proj", kHiddenSize, kDraftHeads * kHeadDim);
    layer.attn.q_norm =
        dense(prefix + ".self_attn.q_norm.weight", {kHeadDim});
    layer.attn.k_norm =
        dense(prefix + ".self_attn.k_norm.weight", {kHeadDim});
  }

  mtp_valid_ = false;
  dflash_valid_ = false;
  dspark_valid_ = true;
  dspark_reset();
}

void Engine::dspark_reset() {
  dspark_context_offset_ = 0;
  for (DSparkLayer& layer : dspark_layers_) {
    layer.attn.keys = array(0);
    layer.attn.values = array(0);
    layer.attn.cache_length = 0;
    layer.attn.cache_capacity = 0;
  }
}

void Engine::dspark_append_layer_context(
    DSparkAttention& attn,
    const array& projected,
    int position_offset) {
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  const int batch = static_cast<int>(projected.shape()[0]);
  const int length = static_cast<int>(projected.shape()[1]);
  if (length <= 0) {
    return;
  }
  if (attn.cache_length != position_offset) {
    throw std::runtime_error("noncontiguous DSpark context append");
  }

  array keys = reshape(
      attn.k_proj(projected), {batch, length, kKvHeads, kHeadDim});
  keys = mx::fast::rms_norm(keys, attn.k_norm, cfg_.rms_norm_eps);
  keys = dspark_yarn_rope(transpose(keys, {0, 2, 1, 3}), position_offset);
  array values = transpose(
      reshape(
          attn.v_proj(projected),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  const int needed = attn.cache_length + length;
  if (attn.cache_capacity < needed) {
    int capacity = std::max(256, attn.cache_capacity);
    while (capacity < needed) {
      capacity *= 2;
    }
    array new_keys =
        zeros({batch, kKvHeads, capacity, kHeadDim}, keys.dtype());
    array new_values =
        zeros({batch, kKvHeads, capacity, kHeadDim}, values.dtype());
    if (attn.cache_length > 0) {
      array active_keys = slice(
          attn.keys,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      array active_values = slice(
          attn.values,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      new_keys = slice_update(
          new_keys,
          active_keys,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      new_values = slice_update(
          new_values,
          active_values,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
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
      {batch, kKvHeads, needed, kHeadDim});
  attn.values = slice_update(
      attn.values,
      values,
      {0, 0, attn.cache_length, 0},
      {batch, kKvHeads, needed, kHeadDim});
  attn.cache_length = needed;
  async_eval(attn.keys, attn.values);
}

void Engine::dspark_append_context(
    const std::vector<array>& captured, int token_count) {
  if (captured.size() != 5 || token_count <= 0) {
    throw std::runtime_error("invalid DSpark target context capture");
  }
  const int available = static_cast<int>(captured.front().shape()[1]);
  if (token_count > available) {
    throw std::runtime_error("DSpark committed context exceeds target capture");
  }
  std::vector<array> selected;
  selected.reserve(captured.size());
  for (const array& value : captured) {
    if (value.ndim() != 3 || value.shape()[0] != 1 ||
        value.shape()[1] != available || value.shape()[2] != cfg_.hidden_size) {
      throw std::runtime_error("inconsistent DSpark target capture shape");
    }
    selected.push_back(slice(
        value,
        {0, 0, 0},
        {1, token_count, cfg_.hidden_size}));
  }
  array projected = mx::fast::rms_norm(
      dspark_fc_(concatenate(selected, -1)),
      dspark_hidden_norm_,
      cfg_.rms_norm_eps);
  for (DSparkLayer& layer : dspark_layers_) {
    dspark_append_layer_context(
        layer.attn, projected, dspark_context_offset_);
  }
  dspark_context_offset_ += token_count;
}

void Engine::dflash_reset() {
  dflash_context_offset_ = 0;
  for (DFlashLayer& layer : dflash_layers_) {
    layer.attn.keys = array(0);
    layer.attn.values = array(0);
    layer.attn.positions = array(0);
    layer.attn.cache_length = 0;
  }
}

void Engine::dflash_append_layer_context(
    DFlashAttention& attn,
    const array& projected,
    int position_offset) {
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSinkSize = 64;
  constexpr int kWindowSize = 2048;
  const int batch = static_cast<int>(projected.shape()[0]);
  const int length = static_cast<int>(projected.shape()[1]);
  if (length <= 0) {
    return;
  }

  array keys = reshape(
      attn.k_proj(projected), {batch, length, kKvHeads, kHeadDim});
  keys = mx::fast::rms_norm(keys, attn.k_norm, 1e-6f);
  keys = transpose(keys, {0, 2, 1, 3});
  keys = mx::fast::rope(
      keys,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      position_offset);
  array values = transpose(
      reshape(
          attn.v_proj(projected),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});
  array positions = mx::arange(
      position_offset, position_offset + length, mx::int32);

  if (attn.cache_length == 0) {
    attn.keys = keys;
    attn.values = values;
    attn.positions = positions;
    attn.cache_length = length;
  } else {
    attn.keys = concatenate({attn.keys, keys}, 2);
    attn.values = concatenate({attn.values, values}, 2);
    attn.positions = concatenate({attn.positions, positions}, 0);
    attn.cache_length += length;
  }

  const int maximum = kSinkSize + kWindowSize;
  if (attn.cache_length > maximum) {
    const int tail_start = attn.cache_length - kWindowSize;
    array sink_keys = slice(
        attn.keys, {0, 0, 0, 0}, {batch, kKvHeads, kSinkSize, kHeadDim});
    array tail_keys = slice(
        attn.keys,
        {0, 0, tail_start, 0},
        {batch, kKvHeads, attn.cache_length, kHeadDim});
    array sink_values = slice(
        attn.values, {0, 0, 0, 0}, {batch, kKvHeads, kSinkSize, kHeadDim});
    array tail_values = slice(
        attn.values,
        {0, 0, tail_start, 0},
        {batch, kKvHeads, attn.cache_length, kHeadDim});
    array sink_positions =
        slice(attn.positions, {0}, {kSinkSize});
    array tail_positions = slice(
        attn.positions, {tail_start}, {attn.cache_length});
    attn.keys = concatenate({sink_keys, tail_keys}, 2);
    attn.values = concatenate({sink_values, tail_values}, 2);
    attn.positions = concatenate({sink_positions, tail_positions}, 0);
    attn.cache_length = maximum;
  }
  async_eval(attn.keys, attn.values, attn.positions);
}

void Engine::dflash_append_context(
    const std::vector<array>& captured, int token_count) {
  constexpr int kSinkSize = 64;
  constexpr int kWindowSize = 2048;
  if (captured.size() != 5 || token_count <= 0) {
    throw std::runtime_error("invalid DFlash2 target context capture");
  }
  const int available = static_cast<int>(captured.front().shape()[1]);
  if (token_count > available) {
    throw std::runtime_error("DFlash2 committed context exceeds target capture");
  }
  for (const array& value : captured) {
    if (value.ndim() != 3 || value.shape()[0] != 1 ||
        value.shape()[1] != available || value.shape()[2] != cfg_.hidden_size) {
      throw std::runtime_error("inconsistent DFlash2 target capture shape");
    }
  }

  std::vector<std::pair<int, int>> spans;
  const bool empty = dflash_layers_.front().attn.cache_length == 0;
  if (empty && token_count > kSinkSize + kWindowSize) {
    spans.emplace_back(0, kSinkSize);
    spans.emplace_back(token_count - kWindowSize, token_count);
  } else if (!empty && token_count > kWindowSize) {
    spans.emplace_back(token_count - kWindowSize, token_count);
  } else {
    spans.emplace_back(0, token_count);
  }

  for (const auto& [start, end] : spans) {
    std::vector<array> selected;
    selected.reserve(captured.size());
    for (const array& value : captured) {
      selected.push_back(slice(
          value,
          {0, start, 0},
          {1, end, cfg_.hidden_size}));
    }
    array projected = mx::fast::rms_norm(
        dflash_fc_(concatenate(selected, -1)),
        dflash_hidden_norm_,
        cfg_.rms_norm_eps);
    for (DFlashLayer& layer : dflash_layers_) {
      dflash_append_layer_context(
          layer.attn, projected, dflash_context_offset_ + start);
    }
  }
  dflash_context_offset_ += token_count;
}

void Engine::draft_append_context(
    const std::vector<array>& captured, int token_count) {
  if (dspark_valid_) {
    dspark_append_context(captured, token_count);
    return;
  }
  if (dflash_valid_) {
    dflash_append_context(captured, token_count);
    return;
  }
  throw std::runtime_error("speculative draft context is unavailable");
}

array Engine::dflash_grouped_convolve(
    const array& hidden, const array& dynamic, const array& base) const {
  constexpr int kGroupSize = 16;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  const int hidden_size = static_cast<int>(hidden.shape()[2]);
  const int groups = hidden_size / kGroupSize;
  array blocks = reshape(
      hidden, {batch, length, groups, kGroupSize});
  array output = mx::zeros(blocks.shape(), hidden.dtype());
  for (int offset = 0; offset < 2; ++offset) {
    array values = blocks;
    if (offset != 0) {
      array leading =
          mx::zeros({batch, offset, groups, kGroupSize}, hidden.dtype());
      array preceding = slice(
          blocks,
          {0, 0, 0, 0},
          {batch, length - offset, groups, kGroupSize});
      values = concatenate({leading, preceding}, 1);
    }
    array kernel = reshape(
        astype(take(base, offset, 0), hidden.dtype()),
        {1, 1, groups, kGroupSize});
    array dynamic_offset = expand_dims(take(dynamic, offset, 2), -1);
    output = output + (kernel + dynamic_offset) * values;
  }
  return reshape(output, hidden.shape());
}

std::pair<array, array> Engine::dflash_conv_prepare(
    const DFlashDynamicConv& conv, const array& hidden) const {
  constexpr int kGroupSize = 16;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  const int groups = static_cast<int>(hidden.shape()[2]) / kGroupSize;
  array dynamic = reshape(
      conv.kernel_projection(hidden), {batch, length, 2, 2, groups});
  array prepared = dflash_grouped_convolve(
      hidden, take(dynamic, 0, 2), take(conv.base_kernel, 0, 0));
  return {prepared, take(dynamic, 1, 2)};
}

array Engine::dflash_conv_finish(
    const DFlashDynamicConv& conv,
    const array& hidden,
    const array& dynamic) const {
  return dflash_grouped_convolve(
      hidden, dynamic, take(conv.base_kernel, 1, 0));
}

array Engine::dflash_attention(
    DFlashAttention& attn, const array& hidden) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSlidingWindow = 2048;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  if (attn.cache_length <= 0) {
    throw std::runtime_error("DFlash2 draft context cache is empty");
  }

  array queries = reshape(
      attn.q_proj(hidden), {batch, length, kHeads, kHeadDim});
  queries = mx::fast::rms_norm(queries, attn.q_norm, 1e-6f);
  queries = transpose(queries, {0, 2, 1, 3});
  queries = mx::fast::rope(
      queries,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      dflash_context_offset_);

  array noise_keys = reshape(
      attn.k_proj(hidden), {batch, length, kKvHeads, kHeadDim});
  noise_keys = mx::fast::rms_norm(noise_keys, attn.k_norm, 1e-6f);
  noise_keys = transpose(noise_keys, {0, 2, 1, 3});
  noise_keys = mx::fast::rope(
      noise_keys,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      dflash_context_offset_);
  array noise_values = transpose(
      reshape(
          attn.v_proj(hidden),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  array keys = concatenate({attn.keys, noise_keys}, 2);
  array values = concatenate({attn.values, noise_values}, 2);
  array block_positions = mx::arange(
      dflash_context_offset_,
      dflash_context_offset_ + length,
      mx::int32);
  array key_positions = concatenate({attn.positions, block_positions}, 0);
  array query_grid = expand_dims(block_positions, 1);
  array key_grid = expand_dims(key_positions, 0);
  array context_mask = mx::logical_and(
      mx::less(key_grid, array(dflash_context_offset_, mx::int32)),
      mx::less(query_grid - key_grid, array(kSlidingWindow, mx::int32)));
  array block_mask =
      mx::greater_equal(key_grid, array(dflash_context_offset_, mx::int32));
  array mask = mx::logical_or(context_mask, block_mask);

  array output = mx::fast::scaled_dot_product_attention(
      queries,
      keys,
      values,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      "",
      mask);
  output = reshape(
      transpose(output, {0, 2, 1, 3}),
      {batch, length, kHeads * kHeadDim});
  return attn.o_proj(output);
}

array Engine::dflash_forward(int32_t anchor) {
  constexpr int kBlockSize = 8;
  constexpr int32_t kMaskToken = 248070;
  int32_t tokens[kBlockSize];
  tokens[0] = anchor;
  std::fill(tokens + 1, tokens + kBlockSize, kMaskToken);
  array ids(tokens, {1, kBlockSize}, mx::int32);
  array hidden = embed(ids);
  for (DFlashLayer& layer : dflash_layers_) {
    array residual = hidden;
    auto attention_prepared = dflash_conv_prepare(
        layer.attention_conv,
        mx::fast::rms_norm(
            hidden, layer.input_norm, cfg_.rms_norm_eps));
    array attention_output = dflash_attention(
        layer.attn, attention_prepared.first);
    hidden = residual + dflash_conv_finish(
        layer.attention_conv,
        attention_output,
        attention_prepared.second);

    residual = hidden;
    auto mlp_prepared = dflash_conv_prepare(
        layer.mlp_conv,
        mx::fast::rms_norm(
            hidden, layer.post_norm, cfg_.rms_norm_eps));
    array mlp_output = layer.down_proj(
        silu(layer.gate_proj(mlp_prepared.first)) *
        layer.up_proj(mlp_prepared.first));
    hidden = residual + dflash_conv_finish(
        layer.mlp_conv, mlp_output, mlp_prepared.second);
  }
  hidden = mx::fast::rms_norm(
      hidden, dflash_norm_, cfg_.rms_norm_eps);
  return slice(
      hidden, {0, 1, 0}, {1, kBlockSize, cfg_.hidden_size});
}

array Engine::dspark_attention(
    DSparkAttention& attn, const array& hidden) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  if (attn.cache_length <= 0 ||
      attn.cache_length != dspark_context_offset_ ||
      attn.cache_capacity < attn.cache_length) {
    throw std::runtime_error("invalid DSpark draft context cache");
  }

  array queries = reshape(
      attn.q_proj(hidden), {batch, length, kHeads, kHeadDim});
  queries = mx::fast::rms_norm(queries, attn.q_norm, cfg_.rms_norm_eps);
  queries = dspark_yarn_rope(
      transpose(queries, {0, 2, 1, 3}), dspark_context_offset_);

  array noise_keys = reshape(
      attn.k_proj(hidden), {batch, length, kKvHeads, kHeadDim});
  noise_keys =
      mx::fast::rms_norm(noise_keys, attn.k_norm, cfg_.rms_norm_eps);
  noise_keys = dspark_yarn_rope(
      transpose(noise_keys, {0, 2, 1, 3}), dspark_context_offset_);
  array noise_values = transpose(
      reshape(
          attn.v_proj(hidden),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  array context_keys = slice(
      attn.keys,
      {0, 0, 0, 0},
      {batch, kKvHeads, attn.cache_length, kHeadDim});
  array context_values = slice(
      attn.values,
      {0, 0, 0, 0},
      {batch, kKvHeads, attn.cache_length, kHeadDim});
  array keys = concatenate({context_keys, noise_keys}, 2);
  array values = concatenate({context_values, noise_values}, 2);
  array output = mx::fast::scaled_dot_product_attention(
      queries,
      keys,
      values,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)));
  output = reshape(
      transpose(output, {0, 2, 1, 3}),
      {batch, length, kHeads * kHeadDim});
  return attn.o_proj(output);
}

array Engine::dspark_forward(int32_t anchor) {
  constexpr int kBlockSize = 7;
  constexpr int32_t kMaskToken = 248070;
  int32_t tokens[kBlockSize];
  tokens[0] = anchor;
  std::fill(tokens + 1, tokens + kBlockSize, kMaskToken);
  array ids(tokens, {1, kBlockSize}, mx::int32);
  array hidden = embed(ids);
  for (DSparkLayer& layer : dspark_layers_) {
    array residual = hidden;
    array normalized = mx::fast::rms_norm(
        hidden, layer.input_norm, cfg_.rms_norm_eps);
    hidden = residual + dspark_attention(layer.attn, normalized);

    residual = hidden;
    normalized = mx::fast::rms_norm(
        hidden, layer.post_norm, cfg_.rms_norm_eps);
    hidden = residual + layer.down_proj(
        silu(layer.gate_proj(normalized)) * layer.up_proj(normalized));
  }
  return mx::fast::rms_norm(hidden, dspark_norm_, cfg_.rms_norm_eps);
}

std::pair<array, array> Engine::dspark_propose(
    const array& hidden, int32_t anchor, bool sampled) {
  constexpr int kBlockSize = 7;
  constexpr int kMarkovRank = 256;
  if (hidden.shape() != mx::Shape{1, kBlockSize, cfg_.hidden_size}) {
    throw std::runtime_error("invalid DSpark proposal hidden state");
  }
  array base_logits = lm_head_(hidden);
  array previous(&anchor, {1}, mx::int32);
  std::vector<array> path;
  std::vector<array> probabilities;
  path.reserve(kBlockSize);
  probabilities.reserve(kBlockSize);
  for (int position = 0; position < kBlockSize; ++position) {
    array base_row = reshape(
        slice(
            base_logits,
            {0, position, 0},
            {1, position + 1, cfg_.vocab_size}),
        {1, cfg_.vocab_size});
    array markov_embedding = reshape(
        take(dspark_markov_w1_, previous, 0), {1, kMarkovRank});
    array corrected = astype(base_row, mx::float32) +
        astype(dspark_markov_w2_(markov_embedding), mx::float32);
    if (sampled) {
      array probs = mx::softmax(corrected, -1, true);
      previous = astype(
          mx::random::categorical(mx::log(probs), -1), mx::int32);
      probabilities.push_back(probs);
    } else {
      previous = astype(mx::argmax(corrected, -1), mx::int32);
    }
    path.push_back(previous);
  }
  return {
      astype(mx::stack(path, 1), mx::int32),
      sampled ? mx::stack(probabilities, 1) : array(0)};
}

std::tuple<array, array, array> Engine::dflash_select(
    const array& hidden, const array& draft_logits, int32_t anchor) {
  constexpr int kTopK = 16;
  constexpr int kSelectorRank = 256;
  const int length = static_cast<int>(hidden.shape()[1]);
  const int vocab = static_cast<int>(draft_logits.shape()[2]);
  array partitioned = mx::argpartition(draft_logits, vocab - kTopK, -1);
  array candidates = slice(
      partitioned, {0, 0, vocab - kTopK}, {1, length, vocab});
  array unary = mx::take_along_axis(draft_logits, candidates, -1);
  array projected = dflash_selector_hidden_(hidden);
  array predecessor(&anchor, {1}, mx::int32);
  std::vector<array> path;
  std::vector<array> probabilities;
  path.reserve(static_cast<size_t>(length));
  probabilities.reserve(static_cast<size_t>(length));
  for (int position = 0; position < length; ++position) {
    array ids = reshape(
        slice(
            candidates,
            {0, position, 0},
            {1, position + 1, kTopK}),
        {1, kTopK});
    array unary_row = reshape(
        slice(
            unary,
            {0, position, 0},
            {1, position + 1, kTopK}),
        {1, kTopK});
    array hidden_row = reshape(
        slice(
            projected,
            {0, position, 0},
            {1, position + 1, kSelectorRank}),
        {1, kSelectorRank});
    array predecessor_embedding = take(
        dflash_predecessor_, predecessor, 0);
    array successor_embedding = take(dflash_successor_, ids, 0);
    array edges = sum(
        expand_dims(predecessor_embedding * hidden_row, 1) *
            successor_embedding,
        -1);
    array scores = astype(unary_row, mx::float32) +
        astype(edges, mx::float32);
    array probs = mx::softmax(scores, -1, true);
    array selected = mx::random::categorical(mx::log(probs), -1);
    predecessor = squeeze(
        mx::take_along_axis(ids, expand_dims(selected, -1), -1), -1);
    path.push_back(predecessor);
    probabilities.push_back(probs);
  }
  return {
      astype(mx::stack(path, 1), mx::int32),
      astype(candidates, mx::int32),
      mx::stack(probabilities, 1)};
}

void Engine::draft_commit_verified_prefix(
    const TargetForward& verified, int token_count) {
  if (token_count <= 0 || verified.hidden.ndim() != 3 ||
      token_count > verified.hidden.shape()[1] ||
      verified.linear_tapes.size() != layers_.size() ||
      snap_.size() != layers_.size()) {
    throw std::runtime_error("invalid speculative verified-prefix commit");
  }

  std::vector<array> committed_states;
  committed_states.reserve(layers_.size() * 2);
  for (size_t index = 0; index < layers_.size(); ++index) {
    DecoderLayer& layer = layers_[index];
    const LayerSnap& snapshot_state = snap_[index];
    if (layer.is_linear) {
      const LinearCommitTape& tape = verified.linear_tapes[index];
      if (!snapshot_state.has_state || !tape.valid ||
          snapshot_state.conv.ndim() != 3 ||
          tape.conv_tokens.ndim() != 3 ||
          snapshot_state.conv.shape()[0] != tape.conv_tokens.shape()[0] ||
          snapshot_state.conv.shape()[2] != tape.conv_tokens.shape()[2] ||
          token_count > tape.conv_tokens.shape()[1]) {
        throw std::runtime_error("invalid speculative linear commit tape");
      }
      const int batch = static_cast<int>(snapshot_state.conv.shape()[0]);
      const int window = static_cast<int>(snapshot_state.conv.shape()[1]);
      const int width = static_cast<int>(snapshot_state.conv.shape()[2]);
      array committed_tokens = slice(
          tape.conv_tokens,
          {0, 0, 0},
          {batch, token_count, width});
      array combined = concatenate(
          {snapshot_state.conv, committed_tokens}, 1);
      layer.linear.conv_state = slice(
          combined,
          {0, token_count, 0},
          {batch, token_count + window, width});
      layer.linear.rec_state = gated_delta_commit(
          tape.keys,
          tape.decay,
          tape.delta,
          snapshot_state.rec,
          token_count);
      layer.linear.has_state = true;
      committed_states.push_back(layer.linear.conv_state);
      committed_states.push_back(layer.linear.rec_state);
      continue;
    }

    const int committed_length = snapshot_state.cache_length + token_count;
    if (layer.attn.cache_length < committed_length ||
        layer.attn.offset < snapshot_state.offset + token_count) {
      throw std::runtime_error("invalid speculative attention commit state");
    }
    layer.attn.cache_length = committed_length;
    layer.attn.offset = snapshot_state.offset + token_count;
  }

  const int hidden_size = static_cast<int>(verified.hidden.shape()[2]);
  last_hidden_ = reshape(
      slice(
          verified.hidden,
          {0, token_count - 1, 0},
          {1, token_count, hidden_size}),
      {1, hidden_size});
  draft_append_context(verified.captured, token_count);
  async_eval(std::move(committed_states));
}

void Engine::verify_speculative_block(
    int32_t token,
    const array& draft_tokens,
    const array& proposal_indices,
    const array& proposal_probs,
    const array& confidence,
    bool dense_proposal,
    bool greedy,
    const char* trace_tag) {
  constexpr int kMaxDraftTokens = 7;
  const bool trace = native_spec_trace_enabled();
  const auto started = std::chrono::steady_clock::now();
  if (draft_tokens.ndim() != 2 || draft_tokens.shape()[0] != 1 ||
      draft_tokens.shape()[1] < 1 ||
      draft_tokens.shape()[1] > kMaxDraftTokens ||
      draft_tokens.dtype() != mx::int32) {
    throw std::runtime_error("invalid speculative draft token block");
  }
  const int draft_token_count = static_cast<int>(draft_tokens.shape()[1]);
  const bool has_confidence = confidence.ndim() != 0;
  if (has_confidence &&
      (confidence.shape() != mx::Shape{1, draft_token_count} ||
       confidence.dtype() != mx::float32)) {
    throw std::runtime_error("invalid speculative confidence block");
  }
  if (greedy) {
    if (has_confidence) {
      eval(draft_tokens, confidence);
    } else {
      eval(draft_tokens);
    }
  } else if (dense_proposal) {
    if (proposal_probs.shape() !=
            mx::Shape{1, draft_token_count, cfg_.vocab_size} ||
        proposal_probs.dtype() != mx::float32) {
      throw std::runtime_error("invalid dense speculative proposal");
    }
    if (has_confidence) {
      eval(draft_tokens, proposal_probs, confidence);
    } else {
      eval(draft_tokens, proposal_probs);
    }
  } else {
    if (proposal_indices.shape() != proposal_probs.shape() ||
        proposal_indices.ndim() != 3 || proposal_indices.shape()[0] != 1 ||
        proposal_indices.shape()[1] != draft_token_count ||
        proposal_indices.dtype() != mx::int32 ||
        proposal_probs.dtype() != mx::float32) {
      throw std::runtime_error("invalid sparse speculative proposal");
    }
    if (has_confidence) {
      eval(draft_tokens, proposal_indices, proposal_probs, confidence);
    } else {
      eval(draft_tokens, proposal_indices, proposal_probs);
    }
  }
  const auto draft_done = std::chrono::steady_clock::now();
  if (trace && has_confidence) {
    const float* const values = confidence.data<float>();
    std::fprintf(stderr, "qwen38_%s confidence=", trace_tag);
    for (int index = 0; index < draft_token_count; ++index) {
      std::fprintf(stderr, "%s%.6f", index == 0 ? "" : ",", values[index]);
    }
    std::fprintf(stderr, "\n");
  }
  const int32_t* const drafted = draft_tokens.data<int32_t>();
  int32_t input[kMaxDraftTokens + 1];
  input[0] = token;
  for (int index = 0; index < draft_token_count; ++index) {
    input[index + 1] = drafted[index];
  }

  snapshot();
  const bool tape_commit = native_dflash_tape_commit_enabled();
  TargetForward verified = forward_hidden_captured(
      array(input, {1, draft_token_count + 1}, mx::int32), tape_commit);
  last_hidden_ = last_token(verified.hidden);
  array target_logits = logits(verified.hidden);
  int accepted = 0;
  array next(0);
  auto verify_done = draft_done;
  if (greedy) {
    array target_tokens = astype(mx::argmax(target_logits, -1), mx::int32);
    array target_rows = slice(
        target_tokens, {0, 0}, {1, draft_token_count});
    array accepted_flags = astype(
        mx::equal(target_rows, draft_tokens), mx::int32);
    array accepted_array = sum(mx::cumprod(accepted_flags, -1), -1);
    eval(target_tokens, accepted_array);
    verify_done = std::chrono::steady_clock::now();
    accepted = std::clamp(
        accepted_array.item<int32_t>(), 0, draft_token_count);
    next = reshape(
        slice(
            target_tokens,
            {0, accepted},
            {1, accepted + 1}),
        {1});
  } else {
    array target_probs = sampling_probabilities(target_logits);
    array proposal_column = expand_dims(draft_tokens, -1);
    array target_rows = slice(
        target_probs,
        {0, 0, 0},
        {1, draft_token_count, cfg_.vocab_size});
    array p = squeeze(
        mx::take_along_axis(target_rows, proposal_column, -1), -1);
    array q = dense_proposal
        ? squeeze(
              mx::take_along_axis(
                  proposal_probs, proposal_column, -1),
              -1)
        : sum(
              proposal_probs *
                  mx::equal(proposal_indices, proposal_column),
              -1);
    array accepted_flags = astype(
        mx::less(
            mx::random::uniform(q.shape(), mx::float32) * q,
            p),
        mx::int32);
    array accepted_array = sum(mx::cumprod(accepted_flags, -1), -1);
    eval(accepted_array);
    verify_done = std::chrono::steady_clock::now();
    accepted = std::clamp(
        accepted_array.item<int32_t>(), 0, draft_token_count);

    array next_probs(0);
    if (accepted == draft_token_count) {
      next_probs = reshape(
          slice(
              target_probs,
              {0, draft_token_count, 0},
              {1, draft_token_count + 1, cfg_.vocab_size}),
          {1, cfg_.vocab_size});
    } else {
      array target_row = reshape(
          slice(
              target_probs,
              {0, accepted, 0},
              {1, accepted + 1, cfg_.vocab_size}),
          {1, cfg_.vocab_size});
      array residual(0);
      if (dense_proposal) {
        array proposal_row = reshape(
            slice(
                proposal_probs,
                {0, accepted, 0},
                {1, accepted + 1, cfg_.vocab_size}),
            {1, cfg_.vocab_size});
        residual = target_row - proposal_row;
      } else {
        const int support = static_cast<int>(proposal_probs.shape()[2]);
        array indices = reshape(
            slice(
                proposal_indices,
                {0, accepted, 0},
                {1, accepted + 1, support}),
            {1, support});
        array proposal_values = reshape(
            slice(
                proposal_probs,
                {0, accepted, 0},
                {1, accepted + 1, support}),
            {1, support});
        array residual_values =
            mx::take_along_axis(target_row, indices, -1) - proposal_values;
        residual = mx::put_along_axis(
            target_row, indices, residual_values, -1);
      }
      residual = mx::maximum(residual, array(0.0f, mx::float32));
      array total = sum(residual, -1, true);
      next_probs = mx::where(
          mx::greater(total, array(0.0f, mx::float32)),
          residual / mx::maximum(total, array(1e-30f, mx::float32)),
          target_row);
    }
    next = astype(
        mx::random::categorical(mx::log(next_probs), -1), mx::int32);
  }
  eval(next);
  const auto sample_done = std::chrono::steady_clock::now();

  if (accepted < draft_token_count) {
    if (tape_commit) {
      draft_commit_verified_prefix(verified, accepted + 1);
    } else {
      restore();
      TargetForward committed = forward_hidden_captured(
          array(input, {1, accepted + 1}, mx::int32));
      last_hidden_ = last_token(committed.hidden);
      draft_append_context(committed.captured, accepted + 1);
    }
  } else {
    draft_append_context(verified.captured, draft_token_count + 1);
  }

  for (int index = 0; index < accepted; ++index) {
    spec_buf_[index] = drafted[index];
  }
  spec_buf_[accepted] = next.item<int32_t>();
  spec_buf_n_ = accepted + 1;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;

  if (reasoning_open_) {
    for (int index = 0; index < spec_buf_n_; ++index) {
      if (spec_buf_[index] == 248069) {
        reasoning_open_ = false;
        break;
      }
      ++selected_reasoning_tokens_;
    }
  }
  if (trace) {
    mx::synchronize();
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed_ms = [](auto begin, auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::fprintf(
        stderr,
        "qwen38_%s accepted=%d width=%d drafts=%d draft_ms=%.3f "
        "verify_ms=%.3f "
        "sample_ms=%.3f commit_ms=%.3f total_ms=%.3f\n",
        trace_tag,
        accepted,
        spec_buf_n_,
        draft_token_count,
        elapsed_ms(started, draft_done),
        elapsed_ms(draft_done, verify_done),
        elapsed_ms(verify_done, sample_done),
        elapsed_ms(sample_done, finished),
        elapsed_ms(started, finished));
  }
}

void Engine::dflash_spec_refill(int32_t token) {
  constexpr int kDraftTokens = 7;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  if (reasoning_open_ && max_reasoning_tokens_ > 0 &&
      selected_reasoning_tokens_ + kDraftTokens + 1 >=
          max_reasoning_tokens_) {
    int32_t input[1] = {token};
    TargetForward target = forward_hidden_captured(
        array(input, {1, 1}, mx::int32));
    last_hidden_ = last_token(target.hidden);
    draft_append_context(target.captured, 1);
    array next = select_token(target.hidden);
    eval(next);
    spec_buf_[0] = next.item<int32_t>();
    spec_buf_n_ = 1;
    return;
  }

  array draft_hidden = dflash_forward(token);
  array draft_logits = lm_head_(draft_hidden);
  auto [draft_tokens, draft_indices, draft_probs] =
      dflash_select(draft_hidden, draft_logits, token);
  verify_speculative_block(
      token,
      draft_tokens,
      draft_indices,
      draft_probs,
      array(0),
      /*dense_proposal=*/false,
      /*greedy=*/false,
      "dflash");
}

void Engine::dspark_spec_refill(int32_t token) {
  constexpr int kMaxDraftTokens = 7;
  const int draft_token_count = dspark_verify_draft_tokens_;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  if (reasoning_open_ && max_reasoning_tokens_ > 0 &&
      selected_reasoning_tokens_ + draft_token_count + 1 >=
          max_reasoning_tokens_) {
    int32_t input[1] = {token};
    TargetForward target = forward_hidden_captured(
        array(input, {1, 1}, mx::int32));
    last_hidden_ = last_token(target.hidden);
    draft_append_context(target.captured, 1);
    array next = select_token(target.hidden);
    eval(next);
    spec_buf_[0] = next.item<int32_t>();
    spec_buf_n_ = 1;
    return;
  }

  const bool sampled = sampling_enabled_;
  array draft_hidden = dspark_forward(token);
  auto [draft_tokens, draft_probs] =
      dspark_propose(draft_hidden, token, sampled);
  // Keep the seven-position proposal fixed while profiling target verifier
  // prefixes, so acceptance and target geometry are the only changed inputs.
  if (draft_token_count < kMaxDraftTokens) {
    draft_tokens = slice(
        draft_tokens, {0, 0}, {1, draft_token_count});
    if (sampled) {
      draft_probs = slice(
          draft_probs,
          {0, 0, 0},
          {1, draft_token_count, cfg_.vocab_size});
    }
    draft_hidden = slice(
        draft_hidden,
        {0, 0, 0},
        {1, draft_token_count, cfg_.hidden_size});
  }
  array confidence(0);
  if (native_spec_trace_enabled()) {
    array anchor(&token, {1, 1}, mx::int32);
    array previous = anchor;
    if (draft_token_count > 1) {
      previous = concatenate(
          {
              anchor,
              slice(draft_tokens, {0, 0}, {1, draft_token_count - 1}),
          },
          1);
    }
    array markov_embeddings = take(dspark_markov_w1_, previous, 0);
    confidence = dspark_confidence(
        draft_hidden,
        markov_embeddings,
        dspark_confidence_weight_,
        dspark_confidence_bias_);
  }
  verify_speculative_block(
      token,
      draft_tokens,
      array(0),
      draft_probs,
      confidence,
      /*dense_proposal=*/sampled,
      /*greedy=*/!sampled,
      "dspark");
}

void Engine::spec_refill(int32_t token) {
  if (spec_buf_pos_ < spec_buf_n_ && token == last_emitted_) {
    return;
  }
  if (dspark_valid_) {
    dspark_spec_refill(token);
    return;
  }
  if (dflash_valid_) {
    dflash_spec_refill(token);
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
  if (weights.contains("markov_head.markov_w1.weight")) {
    load_dspark(weights);
    return;
  }
  if (weights.contains("candidate_selector.predecessor_codebook")) {
    load_dflash2(weights);
    return;
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
  dflash_valid_ = false;
  dspark_valid_ = false;
  mtp_valid_ = true;
  mtp_reset();
}

int32_t Engine::emit_scheduled() {
  eval(pending_tok_);
  last_emitted_ = pending_tok_.item<int32_t>();
  if (last_emitted_ == 248069) {
    reasoning_open_ = false;
  }
  return last_emitted_;
}

int32_t Engine::prefill(const int32_t* tokens, int n, bool schedule_decode) {
  if (n <= 0) {
    throw std::runtime_error("prefill requires at least one token");
  }
  if (has_mtp() && schedule_decode) {
    throw std::runtime_error(
        "speculative prefill requires schedule_decode=false");
  }

  const int32_t* new_tokens = tokens;
  int new_token_count = n;
  const bool starts_request =
      request_boundary_pending_ || token_history_.empty();
  if (request_boundary_pending_) {
    const std::size_t history_compare_count =
        std::min(token_history_.size(), static_cast<std::size_t>(n));
    const auto mismatch = std::mismatch(
        token_history_.begin(),
        token_history_.begin() + history_compare_count,
        tokens,
        tokens + history_compare_count);
    const std::size_t common_prefix =
        static_cast<std::size_t>(mismatch.first - token_history_.begin());
    const bool can_reuse_current =
        !has_mtp() && token_history_.size() < size_t(n) &&
        std::equal(token_history_.begin(), token_history_.end(), tokens);
    const std::size_t snapshot_compare_count =
        std::min(prompt_snapshot_history_.size(), static_cast<std::size_t>(n));
    const auto snapshot_mismatch = std::mismatch(
        prompt_snapshot_history_.begin(),
        prompt_snapshot_history_.begin() + snapshot_compare_count,
        tokens,
        tokens + snapshot_compare_count);
    const std::size_t snapshot_common = static_cast<std::size_t>(
        snapshot_mismatch.first - prompt_snapshot_history_.begin());
    const bool can_reuse_snapshot =
        !can_reuse_current && !has_mtp() && prompt_snapshot_valid_ &&
        prompt_snapshot_history_.size() < size_t(n) &&
        std::equal(
            prompt_snapshot_history_.begin(), prompt_snapshot_history_.end(),
            tokens);
    if (native_state_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_native prefill history=%zu input=%d common=%zu "
          "snapshot=%zu snapshot_common=%zu reuse_current=%d "
          "reuse_snapshot=%d mtp=%d\n",
          token_history_.size(), n, common_prefix,
          prompt_snapshot_history_.size(), snapshot_common,
          can_reuse_current ? 1 : 0, can_reuse_snapshot ? 1 : 0,
          has_mtp() ? 1 : 0);
    }
    if (can_reuse_current) {
      new_tokens += token_history_.size();
      new_token_count -= static_cast<int>(token_history_.size());
      reset_decode_pipeline();
      request_boundary_pending_ = false;
    } else if (can_reuse_snapshot) {
      restore();
      token_history_ = prompt_snapshot_history_;
      new_tokens += token_history_.size();
      new_token_count -= static_cast<int>(token_history_.size());
      reset_decode_pipeline();
      request_boundary_pending_ = false;
    } else {
      reset();
    }
  }

  if (starts_request) {
    reasoning_open_ = false;
    for (int index = 0; index < n; ++index) {
      if (tokens[index] == 248068) {
        reasoning_open_ = true;
      } else if (tokens[index] == 248069) {
        reasoning_open_ = false;
      }
    }
    selected_reasoning_tokens_ = 0;
    reasoning_cap_selected_ = false;
    if (native_state_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_native request_state input=%d reasoning_open=%d limit=%d\n",
          n, reasoning_open_ ? 1 : 0, max_reasoning_tokens_);
    }
  }

  token_history_.insert(
      token_history_.end(), new_tokens, new_tokens + new_token_count);
  array hidden(0);
  if (dflash_valid_ || dspark_valid_) {
    constexpr int kDraftPrefillChunkSize = 2048;
    for (int offset = 0; offset < new_token_count;
         offset += kDraftPrefillChunkSize) {
      const int chunk_size = std::min(
          kDraftPrefillChunkSize, new_token_count - offset);
      array chunk_ids(
          new_tokens + offset, {1, chunk_size}, mx::int32);
      TargetForward target = forward_hidden_captured(chunk_ids);
      hidden = target.hidden;
      draft_append_context(target.captured, chunk_size);
      mx::synchronize();
    }
  } else {
    array ids(new_tokens, {1, new_token_count}, mx::int32);
    hidden = forward_hidden(ids);
  }
  if (!has_mtp()) {
    snapshot();
    prompt_snapshot_history_ = token_history_;
    prompt_snapshot_valid_ = true;
  }
  array first = select_token(hidden);
  async_eval(first);
  if (!schedule_decode) {
    pending_tok_ = first;
    decode_scheduled_ = false;
    last_emitted_in_state_ = false;
    return emit_scheduled();
  }
  array following = select_token(forward_hidden(reshape(first, {1, 1})));
  async_eval(following);
  pending_tok_ = first;
  int32_t out = emit_scheduled();
  record_processed_token(out);
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

int32_t Engine::decode(int32_t token) {
  if (has_mtp()) {
    if (spec_buf_pos_ >= spec_buf_n_ || token != last_emitted_) {
      spec_refill(token);
    }
    last_emitted_ = spec_buf_[spec_buf_pos_++];
    decode_scheduled_ = false;
    return last_emitted_;
  }
  if (decode_scheduled_ && token == last_emitted_) {
    array following =
        select_token(forward_hidden(reshape(pending_tok_, {1, 1})));
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
  array next = select_token(forward_hidden(ids));
  async_eval(next);
  array following = select_token(forward_hidden(reshape(next, {1, 1})));
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
