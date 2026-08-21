#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cfloat>
#include <cstdint>
#include <math_constants.h>

namespace sglang {

namespace device::draft_topk1_delta {

constexpr uint32_t kBlockSize = 256;
constexpr uint32_t kTokensPerSplit = 8192;

struct Candidate {
  float value;
  int32_t index;
};

SGL_DEVICE Candidate better(Candidate lhs, Candidate rhs) {
  if (rhs.value > lhs.value || (rhs.value == lhs.value && rhs.index < lhs.index)) {
    return rhs;
  }
  return lhs;
}

SGL_DEVICE Candidate warp_reduce(Candidate candidate) {
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    Candidate other{
        __shfl_down_sync(0xffffffff, candidate.value, offset), __shfl_down_sync(0xffffffff, candidate.index, offset)};
    candidate = better(candidate, other);
  }
  return candidate;
}

template <bool kUseAdditive, bool kUseBias>
__global__ void PartialArgmaxKernel(
    const float* __restrict__ logits,
    const float* __restrict__ additive,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ partial_values,
    int32_t* __restrict__ partial_indices,
    uint32_t vocab_size) {
  const uint32_t row = blockIdx.x;
  const uint32_t split = blockIdx.y;
  const uint32_t begin = split * kTokensPerSplit;
  const uint32_t end = min(begin + kTokensPerSplit, vocab_size);
  const uint64_t row_offset = static_cast<uint64_t>(row) * vocab_size;

  Candidate candidate{-CUDART_INF_F, INT32_MAX};
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    q[row_offset + token] = 0.0f;
    float value = logits[row_offset + token];
    if constexpr (kUseAdditive) {
      value += additive[row_offset + token];
    }
    if constexpr (kUseBias) {
      value += bias[row_offset + token];
    }
    if (value == value) {
      candidate = better(candidate, Candidate{value, static_cast<int32_t>(token)});
    }
  }

  candidate = warp_reduce(candidate);
  __shared__ Candidate warp_candidates[kBlockSize / 32];
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  if (lane == 0) warp_candidates[warp] = candidate;
  __syncthreads();

  if (warp == 0) {
    Candidate block_candidate = lane < kBlockSize / 32 ? warp_candidates[lane] : Candidate{-CUDART_INF_F, INT32_MAX};
    block_candidate = warp_reduce(block_candidate);
    if (lane == 0) {
      const uint32_t offset = row * gridDim.y + split;
      partial_values[offset] = block_candidate.value;
      partial_indices[offset] = block_candidate.index;
    }
  }
}

__global__ void FinalizeArgmaxKernel(
    const float* __restrict__ partial_values,
    const int32_t* __restrict__ partial_indices,
    float* __restrict__ q,
    float* __restrict__ topk_p,
    int64_t* __restrict__ topk_index,
    uint32_t vocab_size,
    uint32_t num_splits) {
  const uint32_t row = blockIdx.x;
  Candidate candidate{-CUDART_INF_F, INT32_MAX};
  for (uint32_t split = threadIdx.x; split < num_splits; split += blockDim.x) {
    const uint32_t offset = row * num_splits + split;
    candidate = better(candidate, Candidate{partial_values[offset], partial_indices[offset]});
  }

  candidate = warp_reduce(candidate);
  __shared__ Candidate warp_candidates[kBlockSize / 32];
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp = threadIdx.x >> 5;
  if (lane == 0) warp_candidates[warp] = candidate;
  __syncthreads();

  if (warp == 0) {
    Candidate block_candidate = lane < kBlockSize / 32 ? warp_candidates[lane] : Candidate{-CUDART_INF_F, INT32_MAX};
    block_candidate = warp_reduce(block_candidate);
    if (lane == 0) {
      const int32_t winner = block_candidate.index == INT32_MAX ? 0 : block_candidate.index;
      const uint64_t output = row;
      topk_p[output] = 1.0f;
      topk_index[output] = static_cast<int64_t>(winner);
      q[static_cast<uint64_t>(row) * vocab_size + winner] = 1.0f;
    }
  }
}

}  // namespace device::draft_topk1_delta

template <bool kUseAdditive, bool kUseBias>
struct DraftTopK1DeltaKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView additive,
      const tvm::ffi::TensorView bias,
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView topk_p,
      const tvm::ffi::TensorView topk_index,
      const tvm::ffi::TensorView partial_values,
      const tvm::ffi::TensorView partial_indices) {
    using namespace host;

    auto rows = SymbolicSize{"rows"};
    auto vocab_size = SymbolicSize{"vocab_size"};
    auto num_splits = SymbolicSize{"num_splits"};
    TensorMatcher({rows, vocab_size}).with_dtype<float>().verify(logits);
    const auto device = logits.device();
    CHECK_HOST(device.device_type == kDLCUDA) << "draft top-k1 delta requires CUDA tensors";
    TensorMatcher({rows, vocab_size}).with_dtype<float>().with_device(device).verify(q);
    if constexpr (kUseAdditive) {
      TensorMatcher({rows, vocab_size}).with_dtype<float>().with_device(device).verify(additive);
    }
    if constexpr (kUseBias) {
      TensorMatcher({rows, vocab_size}).with_dtype<float>().with_device(device).verify(bias);
    }
    TensorMatcher({rows, 1}).with_dtype<float>().with_device(device).verify(topk_p);
    TensorMatcher({rows, 1}).with_dtype<int64_t>().with_device(device).verify(topk_index);
    TensorMatcher({rows, num_splits}).with_dtype<float>().with_device(device).verify(partial_values);
    TensorMatcher({rows, num_splits}).with_dtype<int32_t>().with_device(device).verify(partial_indices);

    const int64_t row_count = rows.unwrap();
    const int64_t vocab = vocab_size.unwrap();
    const int64_t splits = num_splits.unwrap();
    CHECK_HOST(row_count > 0) << "draft top-k1 delta requires a non-empty batch";
    CHECK_HOST(vocab > 0 && vocab <= INT32_MAX) << "draft top-k1 delta vocabulary is out of range: " << vocab;
    CHECK_HOST(splits == div_ceil(vocab, device::draft_topk1_delta::kTokensPerSplit))
        << "draft top-k1 delta partial shape does not match vocabulary";
    CHECK_HOST(splits <= device::draft_topk1_delta::kBlockSize)
        << "draft top-k1 delta supports at most " << device::draft_topk1_delta::kBlockSize << " splits, got " << splits;

    LaunchKernel(
        dim3(static_cast<uint32_t>(row_count), static_cast<uint32_t>(splits)),
        device::draft_topk1_delta::kBlockSize,
        device)(
        device::draft_topk1_delta::PartialArgmaxKernel<kUseAdditive, kUseBias>,
        static_cast<const float*>(logits.data_ptr()),
        static_cast<const float*>(additive.data_ptr()),
        static_cast<const float*>(bias.data_ptr()),
        static_cast<float*>(q.data_ptr()),
        static_cast<float*>(partial_values.data_ptr()),
        static_cast<int32_t*>(partial_indices.data_ptr()),
        static_cast<uint32_t>(vocab));
    LaunchKernel(static_cast<uint32_t>(row_count), device::draft_topk1_delta::kBlockSize, device)(
        device::draft_topk1_delta::FinalizeArgmaxKernel,
        static_cast<const float*>(partial_values.data_ptr()),
        static_cast<const int32_t*>(partial_indices.data_ptr()),
        static_cast<float*>(q.data_ptr()),
        static_cast<float*>(topk_p.data_ptr()),
        static_cast<int64_t*>(topk_index.data_ptr()),
        static_cast<uint32_t>(vocab),
        static_cast<uint32_t>(splits));
  }
};

}  // namespace sglang
