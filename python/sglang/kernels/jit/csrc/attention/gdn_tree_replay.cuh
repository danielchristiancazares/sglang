#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::gdn_tree_replay {

constexpr int32_t kDim = 128;
constexpr int32_t kMaxNodes = 32;
constexpr int32_t kValueTile = 8;
constexpr int32_t kMainThreads = kValueTile * 32;
constexpr int32_t kPairThreads = 256;

__device__ __forceinline__ float WarpSum(float value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float Softplus(float x) {
  return x <= 20.0f ? log1pf(expf(x)) : x;
}

// One block owns a request/key-head.  It normalizes q/k, builds the two
// pairwise dot tables used by the low-rank tree recurrence, and writes raw k
// into the per-node commit ring exactly once (key heads are GQA-shared).
__global__ void BuildPairDotsAndRawK(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ parent,
    __nv_bfloat16* __restrict__ rawk_cache,
    float* __restrict__ inv_norms,
    float* __restrict__ pair_dots,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t rawk_slot_stride,
    int64_t rawk_head_stride,
    int64_t rawk_node_stride,
    int32_t batch_size,
    int32_t num_heads,
    int32_t num_nodes,
    int32_t max_tree_depth) {
  const int32_t block = static_cast<int32_t>(blockIdx.x);
  const int32_t batch = block / num_heads;
  const int32_t head = block % num_heads;
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid >> 5;
  const int32_t lane = tid & 31;
  constexpr int32_t kWarps = kPairThreads / 32;
  const int32_t slot = state_indices[batch];

  __shared__ float inv_q[kMaxNodes];
  __shared__ float inv_k[kMaxNodes];

  for (int32_t node = warp; node < num_nodes; node += kWarps) {
    const int64_t token = static_cast<int64_t>(batch) * num_nodes + node;
    const int64_t q_base = token * q_token_stride + head * kDim;
    const int64_t k_base = token * k_token_stride + head * kDim;
    float q_norm2 = 0.0f;
    float k_norm2 = 0.0f;
#pragma unroll 4
    for (int32_t dim = lane; dim < kDim; dim += 32) {
      const float qv = __bfloat162float(q[q_base + dim]);
      const float kv = __bfloat162float(k[k_base + dim]);
      q_norm2 += qv * qv;
      k_norm2 += kv * kv;
    }
    q_norm2 = WarpSum(q_norm2);
    k_norm2 = WarpSum(k_norm2);
    if (lane == 0) {
      const float iq = rsqrtf(q_norm2 + 1.0e-6f);
      const float ik = rsqrtf(k_norm2 + 1.0e-6f);
      inv_q[node] = iq;
      inv_k[node] = ik;
      const int64_t norm_offset =
          ((static_cast<int64_t>(batch) * num_heads + head) * num_nodes + node) * 2;
      inv_norms[norm_offset] = iq;
      inv_norms[norm_offset + 1] = ik;
    }
  }
  __syncthreads();

  // Slot zero is the node itself (k_node dot q_node). Later slots walk only
  // the node's strict ancestry. The dense implementation evaluated every
  // node/node pair even though the recurrence never reads non-ancestors.
  const int32_t pair_count = num_nodes * max_tree_depth;
  for (int32_t pair = warp; pair < pair_count; pair += kWarps) {
    const int32_t node = pair / max_tree_depth;
    const int32_t depth_slot = pair - node * max_tree_depth;
    int32_t ancestor = node;
    for (int32_t depth = 0; depth < depth_slot && ancestor >= 0; ++depth) {
      ancestor = parent[static_cast<int64_t>(batch) * num_nodes + ancestor];
    }
    const int64_t pair_offset =
        (((static_cast<int64_t>(batch) * num_heads + head) * num_nodes + node) *
             max_tree_depth +
         depth_slot) *
        2;
    if (ancestor < 0) {
      if (lane == 0) {
        pair_dots[pair_offset] = 0.0f;
        pair_dots[pair_offset + 1] = 0.0f;
      }
      continue;
    }
    const int64_t node_token =
        static_cast<int64_t>(batch) * num_nodes + node;
    const int64_t ancestor_token =
        static_cast<int64_t>(batch) * num_nodes + ancestor;
    const int64_t node_k_base = node_token * k_token_stride + head * kDim;
    const int64_t node_q_base = node_token * q_token_stride + head * kDim;
    const int64_t ancestor_k_base =
        ancestor_token * k_token_stride + head * kDim;
    float kk = 0.0f;
    float kq = 0.0f;
#pragma unroll 4
    for (int32_t dim = lane; dim < kDim; dim += 32) {
      const float ka = __bfloat162float(k[ancestor_k_base + dim]);
      if (depth_slot > 0) {
        kk += ka * __bfloat162float(k[node_k_base + dim]);
      }
      kq += ka * __bfloat162float(q[node_q_base + dim]);
    }
    kk = WarpSum(kk);
    kq = WarpSum(kq);
    if (lane == 0) {
      pair_dots[pair_offset] =
          depth_slot > 0 ? kk * inv_k[ancestor] * inv_k[node] : 0.0f;
      pair_dots[pair_offset + 1] = kq * inv_k[ancestor] * inv_q[node];
    }
  }

  if (slot >= 0) {
    const int32_t total = num_nodes * kDim;
    for (int32_t linear = tid; linear < total; linear += kPairThreads) {
      const int32_t node = linear / kDim;
      const int32_t dim = linear - node * kDim;
      const int64_t source =
          (static_cast<int64_t>(batch) * num_nodes + node) * k_token_stride +
          head * kDim + dim;
      const int64_t target = static_cast<int64_t>(slot) * rawk_slot_stride +
                             static_cast<int64_t>(head) * rawk_head_stride +
                             static_cast<int64_t>(node) * rawk_node_stride + dim;
      rawk_cache[target] = k[source];
    }
  }
}

// Gate and beta depend on (request, value head, node), not on the 16 value
// tiles used by the main recurrence. Compute their transcendental path once,
// retain alpha/beta in a compact workspace, and write commit records once.
__global__ void BuildReplayParams(
    const float* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ dt_bias,
    const __nv_bfloat16* __restrict__ b,
    const int32_t* __restrict__ state_indices,
    float* __restrict__ g_cache,
    float* __restrict__ beta_cache,
    float* __restrict__ replay_params,
    int64_t a_token_stride,
    int64_t b_token_stride,
    int64_t g_slot_stride,
    int64_t g_head_stride,
    int64_t beta_slot_stride,
    int64_t beta_head_stride,
    int32_t batch_size,
    int32_t num_value_heads,
    int32_t num_nodes) {
  const int32_t linear =
      static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const int32_t total = batch_size * num_value_heads * num_nodes;
  if (linear >= total) return;

  const int32_t node = linear % num_nodes;
  const int32_t head_batch = linear / num_nodes;
  const int32_t value_head = head_batch % num_value_heads;
  const int32_t batch = head_batch / num_value_heads;
  const int64_t token = static_cast<int64_t>(batch) * num_nodes + node;
  const float x = __bfloat162float(a[token * a_token_stride + value_head]) +
                  __bfloat162float(dt_bias[value_head]);
  const float gate = -expf(A_log[value_head]) * Softplus(x);
  const float beta_logit =
      __bfloat162float(b[token * b_token_stride + value_head]);
  const float beta_value = 1.0f / (1.0f + expf(-beta_logit));
  replay_params[static_cast<int64_t>(linear) * 2] = expf(gate);
  replay_params[static_cast<int64_t>(linear) * 2 + 1] = beta_value;

  const int32_t slot = state_indices[batch];
  if (slot >= 0) {
    g_cache[static_cast<int64_t>(slot) * g_slot_stride +
            static_cast<int64_t>(value_head) * g_head_stride + node] = gate;
    beta_cache[static_cast<int64_t>(slot) * beta_slot_stride +
               static_cast<int64_t>(value_head) * beta_head_stride + node] =
        beta_value;
  }
}

// One block owns (request, value-head, eight V columns).  Each warp retains
// one checkpoint column in registers, projects every node against that one
// root state, then evaluates the branch recurrence as ancestor rank updates.
__global__ void GdnTreeReplayMain(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ checkpoint_state,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ parent,
    __nv_bfloat16* __restrict__ rawv_cache,
    const float* __restrict__ inv_norms,
    const float* __restrict__ replay_params,
    const float* __restrict__ pair_dots,
    __nv_bfloat16* __restrict__ output,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t v_token_stride,
    int64_t output_token_stride,
    int64_t state_slot_stride,
    int64_t rawv_slot_stride,
    int64_t rawv_head_stride,
    int64_t rawv_node_stride,
    int32_t batch_size,
    int32_t num_key_heads,
    int32_t num_value_heads,
    int32_t num_nodes,
    int32_t max_tree_depth,
    float q_scale) {
  const int32_t batch = static_cast<int32_t>(blockIdx.x);
  const int32_t value_head = static_cast<int32_t>(blockIdx.y);
  const int32_t value_tile = static_cast<int32_t>(blockIdx.z);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid >> 5;
  const int32_t lane = tid & 31;
  const int32_t value_index = value_tile * kValueTile + warp;
  const int32_t key_head = value_head / (num_value_heads / num_key_heads);
  const int32_t slot = state_indices[batch];

  __shared__ float base_k[kMaxNodes * kValueTile];
  __shared__ float base_q[kMaxNodes * kValueTile];
  __shared__ float delta[kMaxNodes * kValueTile];
  __shared__ float alpha[kMaxNodes];
  __shared__ float beta[kMaxNodes];
  __shared__ float state_scale[kMaxNodes];
  __shared__ int32_t parents[kMaxNodes];

  if (tid < num_nodes) {
    const int32_t node = tid;
    const int64_t token = static_cast<int64_t>(batch) * num_nodes + node;
    const int64_t param_offset =
        ((static_cast<int64_t>(batch) * num_value_heads + value_head) *
             num_nodes +
         node) *
        2;
    alpha[node] = replay_params[param_offset];
    beta[node] = replay_params[param_offset + 1];
    parents[node] = node == 0 ? -1 : parent[token];
  }
  __syncthreads();

  if (tid == 0) {
    for (int32_t node = 0; node < num_nodes; ++node) {
      const int32_t p = parents[node];
      state_scale[node] = alpha[node] * (p >= 0 ? state_scale[p] : 1.0f);
    }
  }
  __syncthreads();

  float state_values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (slot >= 0 && value_index < kDim) {
#pragma unroll
    for (int32_t chunk = 0; chunk < 4; ++chunk) {
      const int32_t dim = lane + chunk * 32;
      const int64_t state_offset =
          static_cast<int64_t>(slot) * state_slot_stride +
          static_cast<int64_t>(value_head) * kDim * kDim +
          static_cast<int64_t>(value_index) * kDim + dim;
      state_values[chunk] = checkpoint_state[state_offset];
    }
  }

  if (value_index < kDim) {
    for (int32_t node = 0; node < num_nodes; ++node) {
      const int64_t token = static_cast<int64_t>(batch) * num_nodes + node;
      const int64_t q_base = token * q_token_stride + key_head * kDim;
      const int64_t k_base = token * k_token_stride + key_head * kDim;
      const int64_t norm_base =
          ((static_cast<int64_t>(batch) * num_key_heads + key_head) * num_nodes +
           node) *
          2;
      const float inv_q = inv_norms[norm_base];
      const float inv_k = inv_norms[norm_base + 1];
      float projection_q = 0.0f;
      float projection_k = 0.0f;
#pragma unroll
      for (int32_t chunk = 0; chunk < 4; ++chunk) {
        const int32_t dim = lane + chunk * 32;
        const float h = state_values[chunk];
        projection_q += h * __bfloat162float(q[q_base + dim]);
        projection_k += h * __bfloat162float(k[k_base + dim]);
      }
      projection_q = WarpSum(projection_q);
      projection_k = WarpSum(projection_k);
      if (lane == 0) {
        base_q[node * kValueTile + warp] = projection_q * inv_q * q_scale;
        base_k[node * kValueTile + warp] = projection_k * inv_k;
      }
    }
  }
  __syncthreads();

  if (lane == 0 && value_index < kDim) {
    for (int32_t node = 0; node < num_nodes; ++node) {
      const int64_t token = static_cast<int64_t>(batch) * num_nodes + node;
      const int64_t pair_row =
          ((static_cast<int64_t>(batch) * num_key_heads + key_head) * num_nodes +
           node) *
          max_tree_depth * 2;
      float predicted =
          state_scale[node] * base_k[node * kValueTile + warp];
      int32_t ancestor = parents[node];
      int32_t ancestor_slot = 1;
      float coefficient = alpha[node];
      while (ancestor >= 0 && ancestor_slot < max_tree_depth) {
        predicted += coefficient * pair_dots[pair_row + ancestor_slot * 2] *
                     delta[ancestor * kValueTile + warp];
        coefficient *= alpha[ancestor];
        ancestor = parents[ancestor];
        ++ancestor_slot;
      }

      const float raw_value = __bfloat162float(
          v[token * v_token_stride + value_head * kDim + value_index]);
      const float node_delta = beta[node] * (raw_value - predicted);
      delta[node * kValueTile + warp] = node_delta;

      float out = state_scale[node] * base_q[node * kValueTile + warp] +
                  pair_dots[pair_row + 1] * q_scale * node_delta;
      ancestor = parents[node];
      ancestor_slot = 1;
      coefficient = alpha[node];
      while (ancestor >= 0 && ancestor_slot < max_tree_depth) {
        out += coefficient * pair_dots[pair_row + ancestor_slot * 2 + 1] *
               q_scale * delta[ancestor * kValueTile + warp];
        coefficient *= alpha[ancestor];
        ancestor = parents[ancestor];
        ++ancestor_slot;
      }
      output[token * output_token_stride + value_head * kDim + value_index] =
          __float2bfloat16(out);

      if (slot >= 0) {
        rawv_cache[static_cast<int64_t>(slot) * rawv_slot_stride +
                   static_cast<int64_t>(value_head) * rawv_head_stride +
                   static_cast<int64_t>(node) * rawv_node_stride + value_index] =
            v[token * v_token_stride + value_head * kDim + value_index];
      }
    }
  }
}

// Commit is deliberately path-shaped even when verify is a tree.  It replays
// the raw records selected by accept_index in generation order, so the
// persistent fp32 checkpoint is the same state an ordinary sequential decode
// would have produced.  No rejected branch can leak into future decoding.
__global__ void GdnTreeReplayCommit(
    float* __restrict__ checkpoint_state,
    const __nv_bfloat16* __restrict__ rawv_cache,
    const __nv_bfloat16* __restrict__ rawk_cache,
    const float* __restrict__ g_cache,
    const float* __restrict__ beta_cache,
    const int32_t* __restrict__ state_indices,
    const int32_t* __restrict__ accept_index,
    const int32_t* __restrict__ accept_lens,
    const int64_t* __restrict__ track_indices,
    const int32_t* __restrict__ track_nodes,
    int64_t state_layer_stride,
    int64_t state_slot_stride,
    int64_t rawv_layer_stride,
    int64_t rawv_slot_stride,
    int64_t rawv_head_stride,
    int64_t rawv_node_stride,
    int64_t rawk_layer_stride,
    int64_t rawk_slot_stride,
    int64_t rawk_head_stride,
    int64_t rawk_node_stride,
    int64_t g_layer_stride,
    int64_t g_slot_stride,
    int64_t g_head_stride,
    int64_t beta_layer_stride,
    int64_t beta_slot_stride,
    int64_t beta_head_stride,
    int32_t num_layers,
    int32_t num_key_heads,
    int32_t num_value_heads,
    int32_t num_tree_nodes,
    int32_t max_accept_depth,
    bool has_track) {
  const int32_t batch = static_cast<int32_t>(blockIdx.x);
  const int32_t layer_value_head = static_cast<int32_t>(blockIdx.y);
  const int32_t layer = layer_value_head / num_value_heads;
  const int32_t value_head = layer_value_head % num_value_heads;
  const int32_t value_tile = static_cast<int32_t>(blockIdx.z);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t warp = tid >> 5;
  const int32_t lane = tid & 31;
  const int32_t value_index = value_tile * kValueTile + warp;
  const int32_t key_head = value_head / (num_value_heads / num_key_heads);
  const int32_t slot = state_indices[batch];
  if (slot < 0 || value_index >= kDim) return;

  const int32_t commit_count = accept_lens[batch];
  const int64_t track_slot = has_track ? track_indices[batch] : -1;
  const int32_t track_node = has_track ? track_nodes[batch] : -1;
  const int64_t state_layer_base = static_cast<int64_t>(layer) * state_layer_stride;
  const int64_t state_column_base =
      state_layer_base + static_cast<int64_t>(slot) * state_slot_stride +
      static_cast<int64_t>(value_head) * kDim * kDim +
      static_cast<int64_t>(value_index) * kDim;

  float state_values[4];
#pragma unroll
  for (int32_t chunk = 0; chunk < 4; ++chunk) {
    state_values[chunk] = checkpoint_state[
        state_column_base + lane + chunk * 32];
  }

  for (int32_t ordinal = 0;
       ordinal < commit_count && ordinal < max_accept_depth;
       ++ordinal) {
    const int32_t global_node =
        accept_index[static_cast<int64_t>(batch) * max_accept_depth + ordinal];
    const int32_t node = global_node - batch * num_tree_nodes;
    if (node < 0 || node >= num_tree_nodes) break;

    float k_values[4];
    float norm2 = 0.0f;
#pragma unroll
    for (int32_t chunk = 0; chunk < 4; ++chunk) {
      const int32_t dim = lane + chunk * 32;
      const int64_t offset =
          static_cast<int64_t>(layer) * rawk_layer_stride +
          static_cast<int64_t>(slot) * rawk_slot_stride +
          static_cast<int64_t>(key_head) * rawk_head_stride +
          static_cast<int64_t>(node) * rawk_node_stride + dim;
      const float value = __bfloat162float(rawk_cache[offset]);
      k_values[chunk] = value;
      norm2 += value * value;
    }
    norm2 = WarpSum(norm2);
    const float inv_norm = rsqrtf(__shfl_sync(0xffffffffu, norm2, 0) + 1.0e-6f);
    const float gate = g_cache[
        static_cast<int64_t>(layer) * g_layer_stride +
        static_cast<int64_t>(slot) * g_slot_stride +
        static_cast<int64_t>(value_head) * g_head_stride + node];
    const float beta = beta_cache[
        static_cast<int64_t>(layer) * beta_layer_stride +
        static_cast<int64_t>(slot) * beta_slot_stride +
        static_cast<int64_t>(value_head) * beta_head_stride + node];
    const float decay = expf(gate);

    float projection = 0.0f;
#pragma unroll
    for (int32_t chunk = 0; chunk < 4; ++chunk) {
      state_values[chunk] *= decay;
      projection += state_values[chunk] * k_values[chunk] * inv_norm;
    }
    projection = WarpSum(projection);
    const float raw_value = __bfloat162float(rawv_cache[
        static_cast<int64_t>(layer) * rawv_layer_stride +
        static_cast<int64_t>(slot) * rawv_slot_stride +
        static_cast<int64_t>(value_head) * rawv_head_stride +
        static_cast<int64_t>(node) * rawv_node_stride + value_index]);
    const float residual = beta * (raw_value - __shfl_sync(0xffffffffu, projection, 0));
#pragma unroll
    for (int32_t chunk = 0; chunk < 4; ++chunk) {
      state_values[chunk] += k_values[chunk] * inv_norm * residual;
    }

    if (has_track && node == track_node && track_slot >= 0) {
      const int64_t track_column_base =
          state_layer_base + static_cast<int64_t>(track_slot) * state_slot_stride +
          static_cast<int64_t>(value_head) * kDim * kDim +
          static_cast<int64_t>(value_index) * kDim;
#pragma unroll
      for (int32_t chunk = 0; chunk < 4; ++chunk) {
        checkpoint_state[track_column_base + lane + chunk * 32] =
            state_values[chunk];
      }
    }
  }

#pragma unroll
  for (int32_t chunk = 0; chunk < 4; ++chunk) {
    checkpoint_state[state_column_base + lane + chunk * 32] =
        state_values[chunk];
  }
}

}  // namespace device::gdn_tree_replay

struct GdnTreeReplayVerifyKernel {
  static void run(
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView checkpoint_state,
      const tvm::ffi::TensorView state_indices,
      const tvm::ffi::TensorView parent,
      const tvm::ffi::TensorView rawv_cache,
      const tvm::ffi::TensorView rawk_cache,
      const tvm::ffi::TensorView g_cache,
      const tvm::ffi::TensorView beta_cache,
      const tvm::ffi::TensorView inv_norms,
      const tvm::ffi::TensorView replay_params,
      const tvm::ffi::TensorView pair_dots,
      const tvm::ffi::TensorView output,
      int64_t max_tree_depth,
      double scale) {
    using namespace host;

    auto tokens = SymbolicSize{"tokens"};
    auto batch = SymbolicSize{"batch"};
    auto nodes = SymbolicSize{"nodes"};
    auto heads = SymbolicSize{"key_heads"};
    auto value_heads = SymbolicSize{"value_heads"};
    auto slots = SymbolicSize{"slots"};
    auto record_len = SymbolicSize{"record_len"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({value_heads})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .with_strides({1})
        .verify(A_log);
    TensorMatcher({value_heads})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({1})
        .verify(dt_bias);
    TensorMatcher({tokens, value_heads})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(a);
    TensorMatcher({tokens, value_heads})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, 1})
        .verify(b);
    TensorMatcher({tokens, heads, device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, device::gdn_tree_replay::kDim, 1})
        .verify(q);
    TensorMatcher({tokens, heads, device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, device::gdn_tree_replay::kDim, 1})
        .verify(k);
    TensorMatcher({tokens, value_heads, device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, device::gdn_tree_replay::kDim, 1})
        .verify(v);
    TensorMatcher({tokens, value_heads, device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, device::gdn_tree_replay::kDim, 1})
        .verify(output);
    TensorMatcher({slots, value_heads, device::gdn_tree_replay::kDim,
                   device::gdn_tree_replay::kDim})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .with_strides({-1, device::gdn_tree_replay::kDim *
                               device::gdn_tree_replay::kDim,
                       device::gdn_tree_replay::kDim, 1})
        .verify(checkpoint_state);
    TensorMatcher({batch})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({1})
        .verify(state_indices);
    TensorMatcher({batch, nodes})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({nodes, 1})
        .verify(parent);
    TensorMatcher({slots, value_heads, record_len,
                   device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, -1, device::gdn_tree_replay::kDim, 1})
        .verify(rawv_cache);
    TensorMatcher({slots, heads, record_len, device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, -1, device::gdn_tree_replay::kDim, 1})
        .verify(rawk_cache);
    TensorMatcher({slots, value_heads, record_len})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .with_strides({-1, -1, 1})
        .verify(g_cache)
        .verify(beta_cache);
    TensorMatcher({batch, heads, nodes, 2})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(inv_norms);
    auto tree_depth = SymbolicSize{"tree_depth"};
    TensorMatcher({batch, value_heads, nodes, 2})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(replay_params);
    TensorMatcher({batch, heads, nodes, tree_depth, 2})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .verify(pair_dots);

    const int64_t B = batch.unwrap();
    const int64_t N = nodes.unwrap();
    const int64_t H = heads.unwrap();
    const int64_t HV = value_heads.unwrap();
    RuntimeCheck(B > 0, "GDN tree verify requires a non-empty batch");
    RuntimeCheck(N > 0 && N <= device::gdn_tree_replay::kMaxNodes,
                 "GDN tree verify supports 1..",
                 device::gdn_tree_replay::kMaxNodes,
                 " nodes, got ", N);
    RuntimeCheck(tokens.unwrap() == B * N,
                 "tokens must equal batch * nodes");
    RuntimeCheck(record_len.unwrap() >= N,
                 "GDN tree replay ring is narrower than the proposal tree");
    RuntimeCheck(H > 0 && HV % H == 0,
                 "value heads must be divisible by key heads");
    RuntimeCheck(max_tree_depth > 0 && max_tree_depth <= N,
                 "invalid GDN tree maximum depth ", max_tree_depth,
                 " for ", N, " nodes");
    RuntimeCheck(tree_depth.unwrap() == max_tree_depth,
                 "pair-dot tree depth does not match max_tree_depth");

    const auto device = device_.unwrap();
    LaunchKernel(
        static_cast<uint32_t>(B * H),
        device::gdn_tree_replay::kPairThreads,
        device)(
        device::gdn_tree_replay::BuildPairDotsAndRawK,
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const int32_t*>(state_indices.data_ptr()),
        static_cast<const int32_t*>(parent.data_ptr()),
        static_cast<__nv_bfloat16*>(rawk_cache.data_ptr()),
        static_cast<float*>(inv_norms.data_ptr()),
        static_cast<float*>(pair_dots.data_ptr()),
        q.stride(0),
        k.stride(0),
        rawk_cache.stride(0),
        rawk_cache.stride(1),
        rawk_cache.stride(2),
        static_cast<int32_t>(B),
        static_cast<int32_t>(H),
        static_cast<int32_t>(N),
        static_cast<int32_t>(max_tree_depth));

    const int64_t replay_param_items = B * HV * N;
    LaunchKernel(
        static_cast<uint32_t>(div_ceil(replay_param_items,
                                      device::gdn_tree_replay::kPairThreads)),
        device::gdn_tree_replay::kPairThreads,
        device)(
        device::gdn_tree_replay::BuildReplayParams,
        static_cast<const float*>(A_log.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const __nv_bfloat16*>(dt_bias.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        static_cast<const int32_t*>(state_indices.data_ptr()),
        static_cast<float*>(g_cache.data_ptr()),
        static_cast<float*>(beta_cache.data_ptr()),
        static_cast<float*>(replay_params.data_ptr()),
        a.stride(0),
        b.stride(0),
        g_cache.stride(0),
        g_cache.stride(1),
        beta_cache.stride(0),
        beta_cache.stride(1),
        static_cast<int32_t>(B),
        static_cast<int32_t>(HV),
        static_cast<int32_t>(N));

    LaunchKernel(
        dim3(static_cast<uint32_t>(B), static_cast<uint32_t>(HV),
             device::gdn_tree_replay::kDim /
                 device::gdn_tree_replay::kValueTile),
        dim3(device::gdn_tree_replay::kMainThreads),
        device)(
        device::gdn_tree_replay::GdnTreeReplayMain,
        static_cast<const __nv_bfloat16*>(q.data_ptr()),
        static_cast<const __nv_bfloat16*>(k.data_ptr()),
        static_cast<const __nv_bfloat16*>(v.data_ptr()),
        static_cast<const float*>(checkpoint_state.data_ptr()),
        static_cast<const int32_t*>(state_indices.data_ptr()),
        static_cast<const int32_t*>(parent.data_ptr()),
        static_cast<__nv_bfloat16*>(rawv_cache.data_ptr()),
        static_cast<const float*>(inv_norms.data_ptr()),
        static_cast<const float*>(replay_params.data_ptr()),
        static_cast<const float*>(pair_dots.data_ptr()),
        static_cast<__nv_bfloat16*>(output.data_ptr()),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        output.stride(0),
        checkpoint_state.stride(0),
        rawv_cache.stride(0),
        rawv_cache.stride(1),
        rawv_cache.stride(2),
        static_cast<int32_t>(B),
        static_cast<int32_t>(H),
        static_cast<int32_t>(HV),
        static_cast<int32_t>(N),
        static_cast<int32_t>(max_tree_depth),
        static_cast<float>(scale));
  }
};

struct GdnTreeReplayCommitKernel {
  static void run(
      const tvm::ffi::TensorView checkpoint_state,
      const tvm::ffi::TensorView rawv_cache,
      const tvm::ffi::TensorView rawk_cache,
      const tvm::ffi::TensorView g_cache,
      const tvm::ffi::TensorView beta_cache,
      const tvm::ffi::TensorView state_indices,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView accept_lens,
      const tvm::ffi::TensorView track_indices,
      const tvm::ffi::TensorView track_nodes,
      int64_t num_tree_nodes,
      bool has_track) {
    using namespace host;

    auto layers = SymbolicSize{"layers"};
    auto slots = SymbolicSize{"slots"};
    auto value_heads = SymbolicSize{"value_heads"};
    auto heads = SymbolicSize{"key_heads"};
    auto record_len = SymbolicSize{"record_len"};
    auto batch = SymbolicSize{"batch"};
    auto max_depth = SymbolicSize{"max_depth"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();

    TensorMatcher({layers, slots, value_heads, device::gdn_tree_replay::kDim,
                   device::gdn_tree_replay::kDim})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .with_strides({-1, -1,
                       device::gdn_tree_replay::kDim *
                           device::gdn_tree_replay::kDim,
                       device::gdn_tree_replay::kDim, 1})
        .verify(checkpoint_state);
    TensorMatcher({layers, slots, value_heads, record_len,
                   device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, -1, -1, device::gdn_tree_replay::kDim, 1})
        .verify(rawv_cache);
    TensorMatcher({layers, slots, heads, record_len,
                   device::gdn_tree_replay::kDim})
        .with_dtype<bf16_t>()
        .with_device(device_)
        .with_strides({-1, -1, -1, device::gdn_tree_replay::kDim, 1})
        .verify(rawk_cache);
    TensorMatcher({layers, slots, value_heads, record_len})
        .with_dtype<fp32_t>()
        .with_device(device_)
        .with_strides({-1, -1, -1, 1})
        .verify(g_cache)
        .verify(beta_cache);
    TensorMatcher({batch})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({1})
        .verify(state_indices)
        .verify(accept_lens)
        .verify(track_nodes);
    if (has_track) {
      TensorMatcher({batch})
          .with_dtype<int64_t>()
          .with_device(device_)
          .with_strides({1})
          .verify(track_indices);
    }
    TensorMatcher({batch, max_depth})
        .with_dtype<int32_t>()
        .with_device(device_)
        .with_strides({max_depth, 1})
        .verify(accept_index);

    const int64_t B = batch.unwrap();
    const int64_t L = layers.unwrap();
    const int64_t H = heads.unwrap();
    const int64_t HV = value_heads.unwrap();
    RuntimeCheck(B > 0, "GDN tree commit requires a non-empty batch");
    RuntimeCheck(num_tree_nodes > 0 && num_tree_nodes <= record_len.unwrap(),
                 "invalid GDN tree node count ", num_tree_nodes);
    RuntimeCheck(H > 0 && HV % H == 0,
                 "value heads must be divisible by key heads");

    LaunchKernel(
        dim3(static_cast<uint32_t>(B), static_cast<uint32_t>(L * HV),
             device::gdn_tree_replay::kDim /
                 device::gdn_tree_replay::kValueTile),
        dim3(device::gdn_tree_replay::kMainThreads),
        device_.unwrap())(
        device::gdn_tree_replay::GdnTreeReplayCommit,
        static_cast<float*>(checkpoint_state.data_ptr()),
        static_cast<const __nv_bfloat16*>(rawv_cache.data_ptr()),
        static_cast<const __nv_bfloat16*>(rawk_cache.data_ptr()),
        static_cast<const float*>(g_cache.data_ptr()),
        static_cast<const float*>(beta_cache.data_ptr()),
        static_cast<const int32_t*>(state_indices.data_ptr()),
        static_cast<const int32_t*>(accept_index.data_ptr()),
        static_cast<const int32_t*>(accept_lens.data_ptr()),
        static_cast<const int64_t*>(track_indices.data_ptr()),
        static_cast<const int32_t*>(track_nodes.data_ptr()),
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        rawv_cache.stride(0),
        rawv_cache.stride(1),
        rawv_cache.stride(2),
        rawv_cache.stride(3),
        rawk_cache.stride(0),
        rawk_cache.stride(1),
        rawk_cache.stride(2),
        rawk_cache.stride(3),
        g_cache.stride(0),
        g_cache.stride(1),
        g_cache.stride(2),
        beta_cache.stride(0),
        beta_cache.stride(1),
        beta_cache.stride(2),
        static_cast<int32_t>(L),
        static_cast<int32_t>(H),
        static_cast<int32_t>(HV),
        static_cast<int32_t>(num_tree_nodes),
        static_cast<int32_t>(max_depth.unwrap()),
        has_track);
  }
};

}  // namespace sglang
