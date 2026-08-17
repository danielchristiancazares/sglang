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
constexpr int32_t kMaxSparseTargetSupport = 64;
constexpr int32_t kMaxOverlapGridAxis = 8;

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

__global__ void ProposalOverlapCuda(
    const float* __restrict__ target_probs,
    const float* __restrict__ draft_probs,
    const float* __restrict__ temperature_scales,
    const int32_t* __restrict__ top_ks,
    float* __restrict__ output,
    int32_t num_rows,
    int32_t vocab_size,
    int32_t num_scales,
    int32_t num_top_ks) {
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (row >= num_rows) return;

  __shared__ int32_t support_count;
  __shared__ int64_t support_tokens[kMaxSparseTargetSupport];
  __shared__ float support_probs[kMaxSparseTargetSupport];
  if (tid == 0) support_count = 0;
  __syncthreads();

  const int64_t row_offset = static_cast<int64_t>(row) * vocab_size;
  for (int32_t token = tid; token < vocab_size; token += kBlockThreads) {
    const float q = draft_probs[row_offset + token];
    if (q > 0.0f) {
      const int32_t slot = atomicAdd(&support_count, 1);
      if (slot < kMaxSparseTargetSupport) {
        support_tokens[slot] = token;
        support_probs[slot] = q;
      }
    }
  }
  __syncthreads();

  if (tid != 0) return;
  if (support_count > kMaxSparseTargetSupport) {
    for (int32_t scale = 0; scale < num_scales; ++scale) {
      for (int32_t top_k = 0; top_k < num_top_ks; ++top_k) {
        const int64_t out =
            (static_cast<int64_t>(row) * num_scales * num_top_ks +
             scale * num_top_ks + top_k) *
            3;
        output[out] = -1.0f;
        output[out + 1] = -1.0f;
        output[out + 2] = static_cast<float>(support_count);
      }
    }
    return;
  }

  // q is sparse (the production path uses at most top-k 64). Sort once by
  // descending mass so every requested support width is an ordered prefix.
  for (int32_t i = 1; i < support_count; ++i) {
    const int64_t token = support_tokens[i];
    const float probability = support_probs[i];
    int32_t j = i - 1;
    while (j >= 0 &&
           (support_probs[j] < probability ||
            (support_probs[j] == probability && support_tokens[j] > token))) {
      support_tokens[j + 1] = support_tokens[j];
      support_probs[j + 1] = support_probs[j];
      --j;
    }
    support_tokens[j + 1] = token;
    support_probs[j + 1] = probability;
  }

  for (int32_t scale_index = 0; scale_index < num_scales; ++scale_index) {
    const float inverse_scale = 1.0f / temperature_scales[scale_index];
    for (int32_t top_k_index = 0; top_k_index < num_top_ks; ++top_k_index) {
      int32_t retained = top_ks[top_k_index];
      retained = retained < support_count ? retained : support_count;
      float mass = 0.0f;
      for (int32_t i = 0; i < retained; ++i) {
        mass += powf(support_probs[i], inverse_scale);
      }

      float overlap = 0.0f;
      float outside_target = 0.0f;
      if (mass > 0.0f) {
        for (int32_t i = 0; i < retained; ++i) {
          const float calibrated =
              powf(support_probs[i], inverse_scale) / mass;
          const float p = target_probs[row_offset + support_tokens[i]];
          overlap += fminf(p, calibrated);
          if (p == 0.0f) outside_target += calibrated;
        }
      }

      const int64_t out =
          (static_cast<int64_t>(row) * num_scales * num_top_ks +
           scale_index * num_top_ks + top_k_index) *
          3;
      output[out] = overlap;
      output[out + 1] = outside_target;
      output[out + 2] = static_cast<float>(support_count);
    }
  }
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
  __shared__ int64_t target_support_tokens[kMaxSparseTargetSupport];
  __shared__ float target_residual[kMaxSparseTargetSupport];
  __shared__ int32_t current_node;
  __shared__ int32_t current_child;
  __shared__ int32_t last_accepted_global_index;
  __shared__ int32_t rejected_count;
  __shared__ int32_t draft_support_remaining;
  __shared__ int32_t target_support_count;
  __shared__ int32_t support_node;
  __shared__ int32_t sparse_mode;
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
    target_support_count = 0;
    support_node = -1;
    sparse_mode = 0;
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
      selected_child = -1;
      target_support_count = 0;
      support_node = current_node;
      sparse_mode = 0;
    }
    __syncthreads();

    // Reconstruct the exact proposal state for this accepted parent.  Positive
    // q entries follow weighted SWOR.  After all of them have appeared, the
    // generator orders the remaining tokens uniformly without replacement;
    // verification switches to the same D = Uniform(unrejected) law.
    const int64_t parent_row_offset =
        (node_base + current_node) * static_cast<int64_t>(vocab_size);
    float local_draft_mass = 0.0f;
    float local_support_count = 0.0f;
    for (int32_t token = tid; token < vocab_size; token += kBlockThreads) {
      const float q_value = draft_probs[parent_row_offset + token];
      local_draft_mass += q_value;
      local_support_count += q_value > 0.0f ? 1.0f : 0.0f;
      const float p_value = target_probs[parent_row_offset + token];
      if (p_value > 0.0f) {
        const int32_t slot = atomicAdd(&target_support_count, 1);
        if (slot < kMaxSparseTargetSupport) {
          target_support_tokens[slot] = token;
          target_residual[slot] = p_value;
        }
      }
    }
    const float parent_draft_mass = BlockReduce(temp.reduce).Sum(local_draft_mass);
    if (tid == 0) draft_mass = parent_draft_mass;
    __syncthreads();
    const float parent_support_count =
        BlockReduce(temp.reduce).Sum(local_support_count);
    if (tid == 0) {
      draft_support_remaining = static_cast<int32_t>(parent_support_count);
      sparse_mode = target_support_count <= kMaxSparseTargetSupport;
      if (sparse_mode) {
        // Atomic compaction is unordered. Sort the tiny support once so final
        // CDF sampling remains bit-for-bit compatible with vocabulary order.
        for (int32_t i = 1; i < target_support_count; ++i) {
          const int64_t token = target_support_tokens[i];
          const float value = target_residual[i];
          int32_t j = i - 1;
          while (j >= 0 && target_support_tokens[j] > token) {
            target_support_tokens[j + 1] = target_support_tokens[j];
            target_residual[j + 1] = target_residual[j];
            --j;
          }
          target_support_tokens[j + 1] = token;
          target_residual[j + 1] = value;
        }
      }
    }
    __syncthreads();

    while (current_child >= 0 && selected_child < 0) {
      if (tid == 0) {
        const int64_t child_offset = node_base + current_child;
        const int64_t token = candidates[child_offset];
        const int64_t row_offset =
            (node_base + current_node) * static_cast<int64_t>(vocab_size);
        const float q = draft_support_remaining > 0
                            ? draft_probs[row_offset + token] / draft_mass
                            : 1.0f / static_cast<float>(vocab_size - rejected_count);
        float p = target_probs[row_offset + token];
        if (sparse_mode) {
          p = 0.0f;
          for (int32_t i = 0; i < target_support_count; ++i) {
            if (target_support_tokens[i] == token) {
              p = target_residual[i];
              break;
            }
          }
        }
        const float coin = uniform_samples[child_offset];
        if (q > 0.0f && coin * q < p) {
          selected_child = current_child;
        }
      }
      __syncthreads();
      if (selected_child >= 0) break;

      // The sibling was rejected. Update R <- normalize(max(R - D, 0)).
      // D is q renormalized over its remaining support, then uniform over all
      // unrejected tokens once that support is exhausted.
      const int64_t probability_row = node_base + current_node;
      const int64_t row_offset =
          probability_row * static_cast<int64_t>(vocab_size);
      if (sparse_mode) {
        float value = 0.0f;
        if (tid < target_support_count) {
          const int32_t token =
              static_cast<int32_t>(target_support_tokens[tid]);
          float proposal = 0.0f;
          if (!IsRejectedToken(token, rejected_tokens, rejected_count)) {
            proposal = draft_support_remaining > 0
                           ? draft_probs[row_offset + token] / draft_mass
                           : 1.0f /
                                 static_cast<float>(vocab_size - rejected_count);
          }
          value = target_residual[tid] - proposal;
          value = value > 0.0f ? value : 0.0f;
          target_residual[tid] = value;
        }
        const float block_mass = BlockReduce(temp.reduce).Sum(value);
        if (tid == 0) residual_mass = block_mass;
        __syncthreads();
        if (tid < target_support_count) {
          target_residual[tid] =
              residual_mass > 0.0f ? target_residual[tid] / residual_mass : 0.0f;
        }
        __syncthreads();
      } else {
        float local_mass = 0.0f;
        for (int32_t base = 0; base < vocab_size; base += chunk_width) {
#pragma unroll
          for (int32_t j = 0; j < kValuesPerThread; ++j) {
            const int32_t token = base + tid * kValuesPerThread + j;
            if (token < vocab_size &&
                !IsRejectedToken(token, rejected_tokens, rejected_count)) {
              const float proposal =
                  draft_support_remaining > 0
                      ? draft_probs[row_offset + token] / draft_mass
                      : 1.0f / static_cast<float>(vocab_size - rejected_count);
              const float value = target_probs[row_offset + token] - proposal;
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
                const float proposal =
                    draft_support_remaining > 0
                        ? draft_probs[row_offset + token] / draft_mass
                        : 1.0f / static_cast<float>(vocab_size - rejected_count);
                value = target_probs[row_offset + token] - proposal;
                value = value > 0.0f ? value : 0.0f;
              }
              target_probs[row_offset + token] =
                  residual_mass > 0.0f ? value / residual_mass : 0.0f;
            }
          }
        }
        __syncthreads();
      }

      if (tid == 0) {
        const int64_t child_offset = node_base + current_child;
        const int64_t token = candidates[child_offset];
        if (rejected_count < kMaxTreeNodes) {
          rejected_tokens[rejected_count++] = token;
        }
        const float rejected_q = draft_probs[row_offset + token];
        if (rejected_q > 0.0f) {
          --draft_support_remaining;
          draft_mass -= rejected_q;
          if (draft_support_remaining == 0) draft_mass = 0.0f;
        }
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

  // When traversal stopped at the parent whose sparse residual is resident in
  // shared memory, finish there directly. Full-accept cycles end on a newly
  // selected child and deliberately take the dense fallback below.
  if (sparse_mode && support_node == current_node) {
    if (tid == 0) {
      float total = 0.0f;
      int32_t fallback_token = vocab_size - 1;
      for (int32_t i = 0; i < target_support_count; ++i) {
        total += target_residual[i];
        if (target_residual[i] > 0.0f) {
          fallback_token = static_cast<int32_t>(target_support_tokens[i]);
        }
      }
      const float target = uniform_samples_final[batch_idx] * total;
      float cursor = 0.0f;
      sampled_token = fallback_token;
      for (int32_t i = 0; i < target_support_count; ++i) {
        cursor += target_residual[i];
        if (target_residual[i] > 0.0f && cursor > target) {
          sampled_token = static_cast<int32_t>(target_support_tokens[i]);
          break;
        }
      }
      predicts[last_accepted_global_index] = sampled_token;
    }
    return;
  }

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

struct ProposalOverlapKernel {
  static void run(
      const tvm::ffi::TensorView target_probs,
      const tvm::ffi::TensorView draft_probs,
      const tvm::ffi::TensorView temperature_scales,
      const tvm::ffi::TensorView top_ks,
      const tvm::ffi::TensorView output) {
    using namespace host;

    auto batch_size = SymbolicSize{"batch_size"};
    auto num_nodes = SymbolicSize{"num_nodes"};
    auto vocab_size = SymbolicSize{"vocab_size"};
    auto num_scales = SymbolicSize{"num_scales"};
    auto num_top_ks = SymbolicSize{"num_top_ks"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({batch_size, num_nodes, vocab_size})
        .with_dtype<float>()
        .with_device<kDLGPU>(device_)
        .verify(target_probs)
        .verify(draft_probs);
    TensorMatcher({num_scales})
        .with_dtype<float>()
        .with_device(device_)
        .verify(temperature_scales);
    TensorMatcher({num_top_ks})
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(top_ks);
    TensorMatcher({batch_size, num_nodes, num_scales, num_top_ks, 3})
        .with_dtype<float>()
        .with_device(device_)
        .verify(output);

    const int64_t bs = batch_size.unwrap();
    const int64_t nodes = num_nodes.unwrap();
    const int64_t vocab = vocab_size.unwrap();
    const int64_t scales = num_scales.unwrap();
    const int64_t top_k_count = num_top_ks.unwrap();
    RuntimeCheck(bs > 0 && nodes > 0 && vocab > 0, "invalid proposal overlap shape");
    RuntimeCheck(
        scales > 0 && scales <= device::exact_tree_sampling::kMaxOverlapGridAxis,
        "proposal overlap supports 1..",
        device::exact_tree_sampling::kMaxOverlapGridAxis,
        " temperature scales");
    RuntimeCheck(
        top_k_count > 0 &&
            top_k_count <= device::exact_tree_sampling::kMaxOverlapGridAxis,
        "proposal overlap supports 1..",
        device::exact_tree_sampling::kMaxOverlapGridAxis,
        " top-k widths");

    const auto device = device_.unwrap();
    LaunchKernel(
        static_cast<uint32_t>(bs * nodes),
        device::exact_tree_sampling::kBlockThreads,
        device)(
        device::exact_tree_sampling::ProposalOverlapCuda,
        static_cast<const float*>(target_probs.data_ptr()),
        static_cast<const float*>(draft_probs.data_ptr()),
        static_cast<const float*>(temperature_scales.data_ptr()),
        static_cast<const int32_t*>(top_ks.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int32_t>(bs * nodes),
        static_cast<int32_t>(vocab),
        static_cast<int32_t>(scales),
        static_cast<int32_t>(top_k_count));
  }
};

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
