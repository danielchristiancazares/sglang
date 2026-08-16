#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::exact_tree_sampling {

constexpr int32_t kBlockThreads = 256;
constexpr int32_t kValuesPerThread = 4;
constexpr int32_t kMaxTreeNodes = 32;

using BlockReduce = cub::BlockReduce<float, kBlockThreads>;
using BlockScan = cub::BlockScan<float, kBlockThreads>;

union SamplingTempStorage {
  typename BlockReduce::TempStorage reduce;
  typename BlockScan::TempStorage scan;
};

__device__ __forceinline__ bool IsRejectedToken(
    int32_t token,
    const int64_t* __restrict__ rejected_tokens,
    int32_t rejected_count) {
#pragma unroll
  for (int32_t i = 0; i < kMaxTreeNodes; ++i) {
    if (i < rejected_count && rejected_tokens[i] == token) return true;
  }
  return false;
}

__global__ void ExactTreeSpeculativeSamplingCuda(
    int32_t* __restrict__ predicts,
    int32_t* __restrict__ accept_index,
    int32_t* __restrict__ accept_token_num,
    const int64_t* __restrict__ candidates,
    const int64_t* __restrict__ retrieve_index,
    const int64_t* __restrict__ retrieve_next_token,
    const int64_t* __restrict__ retrieve_next_sibling,
    const float* __restrict__ uniform_samples,
    const float* __restrict__ uniform_samples_final,
    const float* __restrict__ target_probs,
    int32_t num_nodes,
    int32_t max_accept_depth,
    int32_t vocab_size) {
  const int32_t batch_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t node_base = static_cast<int64_t>(batch_idx) * num_nodes;
  const int64_t accept_base =
      static_cast<int64_t>(batch_idx) * max_accept_depth;

  __shared__ SamplingTempStorage temp;
  __shared__ int64_t rejected_tokens[kMaxTreeNodes];
  __shared__ int32_t current_node;
  __shared__ int32_t last_accepted_global_index;
  __shared__ int32_t rejected_count;
  __shared__ int32_t num_accepted;
  __shared__ float residual_mass;
  __shared__ float cdf_base;
  __shared__ int32_t sampled_token;

  // The tree metadata is topological.  Thread zero consumes one uniform per
  // accepted prefix and records only the siblings ruled out at the terminal
  // prefix.  Their target mass is removed from the final residual below.
  if (tid == 0) {
    current_node = 0;
    rejected_count = 0;
    num_accepted = 0;
    last_accepted_global_index =
        static_cast<int32_t>(retrieve_index[node_base]);
    accept_index[accept_base] = last_accepted_global_index;

    for (int32_t depth = 1; depth < max_accept_depth; ++depth) {
      int64_t child = retrieve_next_token[node_base + current_node];
      if (child < 0) break;

      const float coin = uniform_samples[node_base + current_node];
      float cumulative_mass = 0.0f;
      int32_t selected = -1;
      rejected_count = 0;

      while (child >= 0) {
        const int64_t child_offset = node_base + child;
        const int64_t token = candidates[child_offset];
        const int64_t prob_offset =
            (node_base + current_node) * static_cast<int64_t>(vocab_size) +
            token;
        cumulative_mass += target_probs[prob_offset];
        if (coin <= cumulative_mass) {
          selected = static_cast<int32_t>(child);
          break;
        }
        if (rejected_count < kMaxTreeNodes) {
          rejected_tokens[rejected_count++] = token;
        }
        child = retrieve_next_sibling[child_offset];
      }

      if (selected < 0) break;

      const int64_t selected_offset = node_base + selected;
      const int64_t token = candidates[selected_offset];
      predicts[last_accepted_global_index] = static_cast<int32_t>(token);
      ++num_accepted;
      last_accepted_global_index =
          static_cast<int32_t>(retrieve_index[selected_offset]);
      accept_index[accept_base + num_accepted] =
          last_accepted_global_index;
      current_node = selected;
      // Rejected siblings belonged to the parent distribution that has now
      // selected a child; they do not constrain the child's distribution.
      rejected_count = 0;
    }

    accept_token_num[batch_idx] = num_accepted;
  }
  __syncthreads();

  const int64_t probability_row = node_base + current_node;

  // Pass one: normalize p after removing only the terminal sibling set.
  float local_mass = 0.0f;
  const int32_t chunk_width = kBlockThreads * kValuesPerThread;
  for (int32_t base = 0; base < vocab_size; base += chunk_width) {
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      const int32_t token = base + tid * kValuesPerThread + j;
      if (token < vocab_size &&
          !IsRejectedToken(token, rejected_tokens, rejected_count)) {
        local_mass += target_probs[
            probability_row * static_cast<int64_t>(vocab_size) + token];
      }
    }
  }
  const float block_mass = BlockReduce(temp.reduce).Sum(local_mass);
  if (tid == 0) {
    residual_mass = block_mass;
    cdf_base = 0.0f;
    sampled_token = vocab_size - 1;
  }
  __syncthreads();

  const float target = uniform_samples_final[batch_idx] * residual_mass;

  // Pass two: vectorized, token-ordered block scan.  Each thread owns four
  // consecutive vocabulary IDs, so the scan plus its local four-value walk is
  // an exact CDF traversal without a vocabulary-sized residual allocation.
  for (int32_t base = 0; base < vocab_size; base += chunk_width) {
    float values[kValuesPerThread];
    float thread_mass = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      const int32_t token = base + tid * kValuesPerThread + j;
      float value = 0.0f;
      if (token < vocab_size &&
          !IsRejectedToken(token, rejected_tokens, rejected_count)) {
        value = target_probs[
            probability_row * static_cast<int64_t>(vocab_size) + token];
      }
      values[j] = value;
      thread_mass += value;
    }

    float thread_prefix = 0.0f;
    float chunk_mass = 0.0f;
    BlockScan(temp.scan).ExclusiveSum(thread_mass, thread_prefix, chunk_mass);
    __syncthreads();

    float cursor = cdf_base + thread_prefix;
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      cursor += values[j];
      if (values[j] > 0.0f && cursor > target) {
        atomicMin(&sampled_token, base + tid * kValuesPerThread + j);
        break;
      }
    }
    __syncthreads();
    if (sampled_token < vocab_size - 1) break;
    if (tid == 0) cdf_base += chunk_mass;
    __syncthreads();
  }

  if (tid == 0) {
    predicts[last_accepted_global_index] = sampled_token;
  }
}

__global__ void ExactTreeSworSamplingCuda(
    int32_t* __restrict__ predicts,
    int32_t* __restrict__ accept_index,
    int32_t* __restrict__ accept_token_num,
    const int64_t* __restrict__ candidates,
    const int64_t* __restrict__ retrieve_index,
    const int64_t* __restrict__ retrieve_next_token,
    const int64_t* __restrict__ retrieve_next_sibling,
    const float* __restrict__ uniform_samples,
    const float* __restrict__ uniform_samples_final,
    float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    int32_t num_nodes,
    int32_t max_accept_depth,
    int32_t vocab_size) {
  const int32_t batch_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int64_t node_base = static_cast<int64_t>(batch_idx) * num_nodes;
  const int64_t accept_base =
      static_cast<int64_t>(batch_idx) * max_accept_depth;

  __shared__ SamplingTempStorage temp;
  __shared__ int64_t rejected_tokens[kMaxTreeNodes];
  __shared__ int32_t current_node;
  __shared__ int32_t current_child;
  __shared__ int32_t last_accepted_global_index;
  __shared__ int32_t rejected_count;
  __shared__ int32_t num_accepted;
  __shared__ int32_t selected_child;
  __shared__ float draft_mass;
  __shared__ float residual_mass;
  __shared__ float cdf_base;
  __shared__ int32_t sampled_token;

  if (tid == 0) {
    current_node = 0;
    current_child = -1;
    rejected_count = 0;
    num_accepted = 0;
    selected_child = -1;
    last_accepted_global_index =
        static_cast<int32_t>(retrieve_index[node_base]);
    accept_index[accept_base] = last_accepted_global_index;
  }
  __syncthreads();

  const int32_t chunk_width = kBlockThreads * kValuesPerThread;
  for (int32_t depth = 1; depth < max_accept_depth; ++depth) {
    if (tid == 0) {
      current_child = static_cast<int32_t>(
          retrieve_next_token[node_base + current_node]);
      rejected_count = 0;
      draft_mass = 1.0f;
      selected_child = -1;
    }
    __syncthreads();

    while (current_child >= 0 && selected_child < 0) {
      if (tid == 0) {
        const int64_t child_offset = node_base + current_child;
        const int64_t token = candidates[child_offset];
        const int64_t row_offset =
            (node_base + current_node) * static_cast<int64_t>(vocab_size);
        const float q = draft_probs[row_offset + token] / draft_mass;
        const float p = target_probs[row_offset + token];
        const float coin = uniform_samples[child_offset];
        if (q > 0.0f && coin * q < p) {
          selected_child = current_child;
        }
      }
      __syncthreads();
      if (selected_child >= 0) break;

      // The sibling was rejected. Update R <- normalize(max(R - D, 0))
      // over the complete vocabulary. D is the original parent q with earlier
      // siblings removed and the remaining mass renormalized.
      const int64_t probability_row = node_base + current_node;
      const int64_t row_offset =
          probability_row * static_cast<int64_t>(vocab_size);
      float local_mass = 0.0f;
      for (int32_t base = 0; base < vocab_size; base += chunk_width) {
#pragma unroll
        for (int32_t j = 0; j < kValuesPerThread; ++j) {
          const int32_t token = base + tid * kValuesPerThread + j;
          if (token < vocab_size &&
              !IsRejectedToken(token, rejected_tokens, rejected_count)) {
            const float value = target_probs[row_offset + token] -
                                draft_probs[row_offset + token] / draft_mass;
            local_mass += value > 0.0f ? value : 0.0f;
          }
        }
      }
      const float block_mass = BlockReduce(temp.reduce).Sum(local_mass);
      if (tid == 0) residual_mass = block_mass;
      __syncthreads();

      for (int32_t base = 0; base < vocab_size; base += chunk_width) {
#pragma unroll
        for (int32_t j = 0; j < kValuesPerThread; ++j) {
          const int32_t token = base + tid * kValuesPerThread + j;
          if (token < vocab_size) {
            float value = 0.0f;
            if (!IsRejectedToken(token, rejected_tokens, rejected_count)) {
              value = target_probs[row_offset + token] -
                      draft_probs[row_offset + token] / draft_mass;
              value = value > 0.0f ? value : 0.0f;
            }
            target_probs[row_offset + token] =
                residual_mass > 0.0f ? value / residual_mass : 0.0f;
          }
        }
      }
      __syncthreads();

      if (tid == 0) {
        const int64_t child_offset = node_base + current_child;
        const int64_t token = candidates[child_offset];
        if (rejected_count < kMaxTreeNodes) {
          rejected_tokens[rejected_count++] = token;
        }
        draft_mass -= draft_probs[row_offset + token];
        draft_mass = draft_mass > 0.0f ? draft_mass : 1.0f;
        current_child = static_cast<int32_t>(
            retrieve_next_sibling[child_offset]);
      }
      __syncthreads();
    }

    if (selected_child < 0) break;
    if (tid == 0) {
      const int64_t selected_offset = node_base + selected_child;
      const int64_t token = candidates[selected_offset];
      predicts[last_accepted_global_index] = static_cast<int32_t>(token);
      ++num_accepted;
      last_accepted_global_index =
          static_cast<int32_t>(retrieve_index[selected_offset]);
      accept_index[accept_base + num_accepted] =
          last_accepted_global_index;
      current_node = selected_child;
    }
    __syncthreads();
  }

  if (tid == 0) accept_token_num[batch_idx] = num_accepted;
  __syncthreads();

  const int64_t probability_row = node_base + current_node;
  const int64_t row_offset =
      probability_row * static_cast<int64_t>(vocab_size);
  float local_mass = 0.0f;
  for (int32_t base = 0; base < vocab_size; base += chunk_width) {
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      const int32_t token = base + tid * kValuesPerThread + j;
      if (token < vocab_size) local_mass += target_probs[row_offset + token];
    }
  }
  const float block_mass = BlockReduce(temp.reduce).Sum(local_mass);
  if (tid == 0) {
    residual_mass = block_mass;
    cdf_base = 0.0f;
    sampled_token = vocab_size - 1;
  }
  __syncthreads();

  const float target = uniform_samples_final[batch_idx] * residual_mass;
  for (int32_t base = 0; base < vocab_size; base += chunk_width) {
    float values[kValuesPerThread];
    float thread_mass = 0.0f;
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      const int32_t token = base + tid * kValuesPerThread + j;
      const float value =
          token < vocab_size ? target_probs[row_offset + token] : 0.0f;
      values[j] = value;
      thread_mass += value;
    }

    float thread_prefix = 0.0f;
    float chunk_mass = 0.0f;
    BlockScan(temp.scan).ExclusiveSum(thread_mass, thread_prefix, chunk_mass);
    __syncthreads();
    float cursor = cdf_base + thread_prefix;
#pragma unroll
    for (int32_t j = 0; j < kValuesPerThread; ++j) {
      cursor += values[j];
      if (values[j] > 0.0f && cursor > target) {
        atomicMin(&sampled_token, base + tid * kValuesPerThread + j);
        break;
      }
    }
    __syncthreads();
    if (sampled_token < vocab_size - 1) break;
    if (tid == 0) cdf_base += chunk_mass;
    __syncthreads();
  }

  if (tid == 0) predicts[last_accepted_global_index] = sampled_token;
}

}  // namespace device::exact_tree_sampling

struct ExactTreeSpeculativeSamplingKernel {
  static void run(
      const tvm::ffi::TensorView predicts,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView accept_token_num,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView retrieve_index,
      const tvm::ffi::TensorView retrieve_next_token,
      const tvm::ffi::TensorView retrieve_next_sibling,
      const tvm::ffi::TensorView uniform_samples,
      const tvm::ffi::TensorView uniform_samples_final,
      const tvm::ffi::TensorView target_probs) {
    using namespace host;

    auto batch_size = SymbolicSize{"batch_size"};
    auto num_nodes = SymbolicSize{"num_nodes"};
    auto max_accept_depth = SymbolicSize{"max_accept_depth"};
    auto vocab_size = SymbolicSize{"vocab_size"};
    auto num_predicts = SymbolicSize{"num_predicts"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({num_predicts})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device_)
        .verify(predicts);
    TensorMatcher({batch_size, max_accept_depth})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(accept_index);
    TensorMatcher({batch_size})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(accept_token_num);
    TensorMatcher({batch_size, num_nodes})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(candidates)
        .verify(retrieve_index)
        .verify(retrieve_next_token)
        .verify(retrieve_next_sibling);
    TensorMatcher({batch_size, num_nodes})
        .with_dtype<float>()
        .with_device(device_)
        .verify(uniform_samples);
    TensorMatcher({batch_size})
        .with_dtype<float>()
        .with_device(device_)
        .verify(uniform_samples_final);
    TensorMatcher({batch_size, num_nodes, vocab_size})
        .with_dtype<float>()
        .with_device(device_)
        .verify(target_probs);

    const int64_t bs = batch_size.unwrap();
    const int64_t nodes = num_nodes.unwrap();
    const int64_t depth = max_accept_depth.unwrap();
    const int64_t vocab = vocab_size.unwrap();
    RuntimeCheck(bs > 0, "exact tree sampling requires a non-empty batch");
    RuntimeCheck(nodes > 0, "exact tree sampling requires at least one node");
    RuntimeCheck(
        nodes <= device::exact_tree_sampling::kMaxTreeNodes,
        "exact tree sampling supports at most ",
        device::exact_tree_sampling::kMaxTreeNodes,
        " nodes, got ",
        nodes);
    RuntimeCheck(depth > 0 && depth <= nodes, "invalid maximum accept depth ", depth);
    RuntimeCheck(vocab > 0, "exact tree sampling requires a non-empty vocabulary");
    RuntimeCheck(
        num_predicts.unwrap() >= bs * nodes,
        "predicts buffer must cover batch_size * num_nodes");

    const auto device = device_.unwrap();
    LaunchKernel(
        static_cast<uint32_t>(bs),
        device::exact_tree_sampling::kBlockThreads,
        device)(
        device::exact_tree_sampling::ExactTreeSpeculativeSamplingCuda,
        static_cast<int32_t*>(predicts.data_ptr()),
        static_cast<int32_t*>(accept_index.data_ptr()),
        static_cast<int32_t*>(accept_token_num.data_ptr()),
        static_cast<const int64_t*>(candidates.data_ptr()),
        static_cast<const int64_t*>(retrieve_index.data_ptr()),
        static_cast<const int64_t*>(retrieve_next_token.data_ptr()),
        static_cast<const int64_t*>(retrieve_next_sibling.data_ptr()),
        static_cast<const float*>(uniform_samples.data_ptr()),
        static_cast<const float*>(uniform_samples_final.data_ptr()),
        static_cast<const float*>(target_probs.data_ptr()),
        static_cast<int32_t>(nodes),
        static_cast<int32_t>(depth),
        static_cast<int32_t>(vocab));
  }
};

struct ExactTreeSworSamplingKernel {
  static void run(
      const tvm::ffi::TensorView predicts,
      const tvm::ffi::TensorView accept_index,
      const tvm::ffi::TensorView accept_token_num,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView retrieve_index,
      const tvm::ffi::TensorView retrieve_next_token,
      const tvm::ffi::TensorView retrieve_next_sibling,
      const tvm::ffi::TensorView uniform_samples,
      const tvm::ffi::TensorView uniform_samples_final,
      const tvm::ffi::TensorView target_probs,
      const tvm::ffi::TensorView draft_probs) {
    using namespace host;

    auto batch_size = SymbolicSize{"batch_size"};
    auto num_nodes = SymbolicSize{"num_nodes"};
    auto max_accept_depth = SymbolicSize{"max_accept_depth"};
    auto vocab_size = SymbolicSize{"vocab_size"};
    auto num_predicts = SymbolicSize{"num_predicts"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({num_predicts})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device_)
        .verify(predicts);
    TensorMatcher({batch_size, max_accept_depth})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(accept_index);
    TensorMatcher({batch_size})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(accept_token_num);
    TensorMatcher({batch_size, num_nodes})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(candidates)
        .verify(retrieve_index)
        .verify(retrieve_next_token)
        .verify(retrieve_next_sibling);
    TensorMatcher({batch_size, num_nodes})
        .with_dtype<float>()
        .with_device(device_)
        .verify(uniform_samples);
    TensorMatcher({batch_size})
        .with_dtype<float>()
        .with_device(device_)
        .verify(uniform_samples_final);
    TensorMatcher({batch_size, num_nodes, vocab_size})
        .with_dtype<float>()
        .with_device(device_)
        .verify(target_probs)
        .verify(draft_probs);

    const int64_t bs = batch_size.unwrap();
    const int64_t nodes = num_nodes.unwrap();
    const int64_t depth = max_accept_depth.unwrap();
    const int64_t vocab = vocab_size.unwrap();
    RuntimeCheck(bs > 0, "exact tree SWOR sampling requires a non-empty batch");
    RuntimeCheck(nodes > 0, "exact tree SWOR sampling requires at least one node");
    RuntimeCheck(
        nodes <= device::exact_tree_sampling::kMaxTreeNodes,
        "exact tree SWOR sampling supports at most ",
        device::exact_tree_sampling::kMaxTreeNodes,
        " nodes, got ",
        nodes);
    RuntimeCheck(depth > 0 && depth <= nodes, "invalid maximum accept depth ", depth);
    RuntimeCheck(vocab > 0, "exact tree SWOR sampling requires a non-empty vocabulary");
    RuntimeCheck(
        num_predicts.unwrap() >= bs * nodes,
        "predicts buffer must cover batch_size * num_nodes");

    const auto device = device_.unwrap();
    LaunchKernel(
        static_cast<uint32_t>(bs),
        device::exact_tree_sampling::kBlockThreads,
        device)(
        device::exact_tree_sampling::ExactTreeSworSamplingCuda,
        static_cast<int32_t*>(predicts.data_ptr()),
        static_cast<int32_t*>(accept_index.data_ptr()),
        static_cast<int32_t*>(accept_token_num.data_ptr()),
        static_cast<const int64_t*>(candidates.data_ptr()),
        static_cast<const int64_t*>(retrieve_index.data_ptr()),
        static_cast<const int64_t*>(retrieve_next_token.data_ptr()),
        static_cast<const int64_t*>(retrieve_next_sibling.data_ptr()),
        static_cast<const float*>(uniform_samples.data_ptr()),
        static_cast<const float*>(uniform_samples_final.data_ptr()),
        static_cast<float*>(target_probs.data_ptr()),
        static_cast<const float*>(draft_probs.data_ptr()),
        static_cast<int32_t>(nodes),
        static_cast<int32_t>(depth),
        static_cast<int32_t>(vocab));
  }
};

}  // namespace sglang
