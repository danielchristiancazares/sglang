#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <type_traits>

namespace sglang {

namespace device::dspark_sparse_accept {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;
constexpr uint32_t kNormalizerThreads = 256;
constexpr uint32_t kValuesPerSplit = 8192;

template <typename T>
SGL_DEVICE float to_float(T value) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    return __bfloat162float(value);
  } else if constexpr (std::is_same_v<T, __half>) {
    return __half2float(value);
  } else {
    return value;
  }
}

SGL_DEVICE float warp_sum(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return __shfl_sync(kFullWarpMask, value, 0);
}

SGL_DEVICE float warp_max(float value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  return __shfl_sync(kFullWarpMask, value, 0);
}

SGL_DEVICE uint64_t warp_min(uint64_t value) {
#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const uint64_t other = __shfl_down_sync(kFullWarpMask, value, offset);
    value = value < other ? value : other;
  }
  return __shfl_sync(kFullWarpMask, value, 0);
}

SGL_DEVICE float block_max(float value) {
  __shared__ float warp_values[kNormalizerThreads / kWarpSize];
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;

#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
  }
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();

  if (warp == 0) {
    value = lane < kNormalizerThreads / kWarpSize ? warp_values[lane] : -CUDART_INF_F;
#pragma unroll
    for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(kFullWarpMask, value, offset));
    }
    if (lane == 0) warp_values[0] = value;
  }
  __syncthreads();
  return warp_values[0];
}

SGL_DEVICE float block_sum(float value) {
  __shared__ float warp_values[kNormalizerThreads / kWarpSize];
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t warp = threadIdx.x / kWarpSize;

#pragma unroll
  for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  if (lane == 0) warp_values[warp] = value;
  __syncthreads();

  if (warp == 0) {
    value = lane < kNormalizerThreads / kWarpSize ? warp_values[lane] : 0.0F;
#pragma unroll
    for (uint32_t offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(kFullWarpMask, value, offset);
    }
    if (lane == 0) warp_values[0] = value;
  }
  __syncthreads();
  return warp_values[0];
}

template <typename DraftT>
__global__ void DraftLogNormalizerPartialsKernel(
    const DraftT* __restrict__ draft_logits,
    const float* __restrict__ draft_temperatures,
    float* __restrict__ partial_max,
    float* __restrict__ partial_sum,
    uint32_t gamma,
    uint32_t vocab_size,
    uint32_t num_splits) {
  const uint32_t split = blockIdx.x;
  const uint32_t row = blockIdx.y;
  const uint32_t request = row / gamma;
  const float temperature = fmaxf(draft_temperatures[request], 1.0e-5F);
  const uint32_t begin = split * kValuesPerSplit;
  const uint32_t end = min(begin + kValuesPerSplit, vocab_size);
  const uint64_t row_offset = static_cast<uint64_t>(row) * vocab_size;

  float local_max = -CUDART_INF_F;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const float value = to_float(draft_logits[row_offset + token]) / temperature;
    local_max = fmaxf(local_max, value);
  }
  const float split_max = block_max(local_max);

  float local_sum = 0.0F;
  for (uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
    const float value = to_float(draft_logits[row_offset + token]) / temperature;
    local_sum += expf(value - split_max);
  }
  const float split_sum = block_sum(local_sum);

  if (threadIdx.x == 0) {
    const uint64_t partial_offset = static_cast<uint64_t>(row) * num_splits + split;
    partial_max[partial_offset] = split_max;
    partial_sum[partial_offset] = split_sum;
  }
}

__global__ void DraftLogNormalizerFinalizeKernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    float* __restrict__ log_normalizers,
    uint32_t num_splits) {
  const uint32_t row = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t offset = static_cast<uint64_t>(row) * num_splits + lane;
  const float local_max = lane < num_splits ? partial_max[offset] : -CUDART_INF_F;
  const float row_max = warp_max(local_max);
  const float local_sum = lane < num_splits ? partial_sum[offset] * expf(local_max - row_max) : 0.0F;
  const float row_sum = warp_sum(local_sum);
  if (lane == 0) log_normalizers[row] = row_max + logf(row_sum);
}

template <typename TargetT>
SGL_DEVICE float target_probability_for_lane(
    const TargetT* __restrict__ target_topk_logits,
    const float* __restrict__ target_temperatures,
    const int32_t* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    uint32_t request,
    uint32_t row,
    uint32_t num_slots,
    uint32_t width) {
  const uint32_t lane = threadIdx.x;
  const int32_t requested_top_k = top_ks[request];
  const uint32_t positive_top_k = static_cast<uint32_t>(requested_top_k > 0 ? requested_top_k : 1);
  const uint32_t top_k = width < positive_top_k ? width : positive_top_k;
  const bool valid = lane < top_k;
  const uint64_t offset = (static_cast<uint64_t>(request) * num_slots + row) * width + lane;
  const float temperature = fmaxf(target_temperatures[request], 1.0e-5F);
  const float scaled_logit = valid ? to_float(target_topk_logits[offset]) / temperature : -CUDART_INF_F;
  const float row_max = warp_max(scaled_logit);
  const float weight = valid ? expf(scaled_logit - row_max) : 0.0F;
  const float weight_sum = warp_sum(weight);
  const float probability = weight_sum > 0.0F ? weight / weight_sum : 0.0F;

  float prefix = probability;
#pragma unroll
  for (uint32_t distance = 1; distance < kWarpSize; distance <<= 1) {
    const float prior = __shfl_up_sync(kFullWarpMask, prefix, distance);
    if (lane >= distance) prefix += prior;
  }

  const float top_p = top_ps[request];
  const bool crosses_after_lane = valid && prefix - probability < top_p;
  const uint32_t prefix_mask = __ballot_sync(kFullWarpMask, crosses_after_lane);
  const uint32_t cutoff_lane = prefix_mask == 0U ? 0U : 31U - static_cast<uint32_t>(__clz(prefix_mask));
  const float cutoff_probability = __shfl_sync(kFullWarpMask, probability, cutoff_lane);
  const float retained = valid && probability >= cutoff_probability ? probability : 0.0F;
  const float retained_sum = warp_sum(retained);
  return retained_sum > 0.0F ? retained / retained_sum : probability;
}

template <typename TargetT, typename DraftT>
__global__ void SparseChainAcceptKernel(
    const TargetT* __restrict__ target_topk_logits,
    const int64_t* __restrict__ target_topk_indices,
    const DraftT* __restrict__ draft_logits,
    const float* __restrict__ draft_log_normalizers,
    const int64_t* __restrict__ candidates,
    const float* __restrict__ target_temperatures,
    const float* __restrict__ draft_temperatures,
    const int32_t* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ uniform_samples,
    const float* __restrict__ uniform_samples_final,
    const int64_t* __restrict__ verify_lens,
    int32_t* __restrict__ correct_len,
    int64_t* __restrict__ bonus,
    int32_t* __restrict__ cap_trim_lens,
    uint32_t num_slots,
    uint32_t gamma,
    uint32_t width,
    uint32_t vocab_size) {
  const uint32_t request = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t candidate_row = static_cast<uint64_t>(request) * num_slots;
  uint32_t current_probability_row = 0;
  uint32_t accepted = 0;
  bool rejected = false;

  for (uint32_t step = 1; step < num_slots; ++step) {
    if (!rejected) {
      const float p_lane = target_probability_for_lane(
          target_topk_logits, target_temperatures, top_ks, top_ps, request, current_probability_row, num_slots, width);
      int64_t candidate = lane == 0 ? candidates[candidate_row + step] : 0;
      candidate = __shfl_sync(kFullWarpMask, candidate, 0);
      const uint64_t support_offset =
          (static_cast<uint64_t>(request) * num_slots + current_probability_row) * width + lane;
      const int64_t support_token = lane < width ? target_topk_indices[support_offset] : -1;
      const float candidate_p = warp_sum(support_token == candidate ? p_lane : 0.0F);

      bool accept = false;
      if (lane == 0) {
        const uint64_t draft_row = static_cast<uint64_t>(request) * gamma + current_probability_row;
        const float temperature = fmaxf(draft_temperatures[request], 1.0e-5F);
        const float draft_logit =
            to_float(draft_logits[draft_row * vocab_size + static_cast<uint64_t>(candidate)]) / temperature;
        const float candidate_q = expf(draft_logit - draft_log_normalizers[draft_row]);
        const float coin = uniform_samples[static_cast<uint64_t>(request) * gamma + step - 1];
        accept = coin * candidate_q < candidate_p;
      }
      accept = __shfl_sync(kFullWarpMask, static_cast<int>(accept), 0) != 0;
      if (accept) {
        ++accepted;
        current_probability_row = step;
      } else {
        rejected = true;
      }
    }
  }

  const float p_lane = target_probability_for_lane(
      target_topk_logits, target_temperatures, top_ks, top_ps, request, current_probability_row, num_slots, width);
  const uint64_t support_offset = (static_cast<uint64_t>(request) * num_slots + current_probability_row) * width + lane;
  const int64_t support_token = lane < width ? target_topk_indices[support_offset] : -1;
  float sample_weight = lane < width ? p_lane : 0.0F;
  if (rejected && lane < width) {
    const uint64_t draft_row = static_cast<uint64_t>(request) * gamma + current_probability_row;
    const float temperature = fmaxf(draft_temperatures[request], 1.0e-5F);
    const float draft_logit =
        to_float(draft_logits[draft_row * vocab_size + static_cast<uint64_t>(support_token)]) / temperature;
    float q = expf(draft_logit - draft_log_normalizers[draft_row]);
    if (q != q) q = 0.0F;
    sample_weight = fmaxf(p_lane - q, 0.0F);
  }

  const float weight_sum = warp_sum(sample_weight);
  const float target = uniform_samples_final[request] * weight_sum;
  float prefix = 0.0F;
#pragma unroll
  for (uint32_t source_lane = 0; source_lane < kWarpSize; ++source_lane) {
    const int64_t source_token = __shfl_sync(kFullWarpMask, support_token, source_lane);
    const float source_weight = __shfl_sync(kFullWarpMask, sample_weight, source_lane);
    if (source_lane < width && source_token <= support_token) prefix += source_weight;
  }
  const uint64_t selected = lane < width && prefix > target ? static_cast<uint64_t>(support_token) : UINT64_MAX;
  const uint64_t sampled = warp_min(selected);

  if (lane == 0) {
    const int32_t raw_correct = static_cast<int32_t>(accepted);
    const int32_t max_correct =
        verify_lens == nullptr ? static_cast<int32_t>(gamma) : static_cast<int32_t>(verify_lens[request] - 1);
    const int32_t capped_correct = raw_correct < max_correct ? raw_correct : max_correct;
    const int32_t trim = raw_correct - capped_correct;
    int64_t output_bonus = sampled == UINT64_MAX ? static_cast<int64_t>(vocab_size - 1) : static_cast<int64_t>(sampled);
    if (capped_correct < raw_correct) {
      output_bonus = candidates[candidate_row + capped_correct + 1];
    }
    correct_len[request] = capped_correct;
    bonus[request] = output_bonus;
    cap_trim_lens[request] = trim;
  }
}

}  // namespace device::dspark_sparse_accept

struct DSparkSparseAcceptKernel {
  static void
  run(const tvm::ffi::TensorView target_topk_logits,
      const tvm::ffi::TensorView target_topk_indices,
      const tvm::ffi::TensorView draft_logits,
      const tvm::ffi::TensorView candidates,
      const tvm::ffi::TensorView target_temperatures,
      const tvm::ffi::TensorView draft_temperatures,
      const tvm::ffi::TensorView top_ks,
      const tvm::ffi::TensorView top_ps,
      const tvm::ffi::TensorView uniform_samples,
      const tvm::ffi::TensorView uniform_samples_final,
      const tvm::ffi::Optional<tvm::ffi::TensorView> verify_lens,
      const tvm::ffi::TensorView partial_max,
      const tvm::ffi::TensorView partial_sum,
      const tvm::ffi::TensorView draft_log_normalizers,
      const tvm::ffi::TensorView correct_len,
      const tvm::ffi::TensorView bonus,
      const tvm::ffi::TensorView cap_trim_lens) {
    using namespace host;

    auto batch = SymbolicSize{"batch"};
    auto slots = SymbolicSize{"slots"};
    auto gamma = SymbolicSize{"gamma"};
    auto width = SymbolicSize{"width"};
    auto vocab = SymbolicSize{"vocab"};
    auto device = SymbolicDevice{};
    auto target_dtype = SymbolicDType{};
    auto draft_dtype = SymbolicDType{};
    device.set_options<kDLCUDA>();

    TensorMatcher({batch, slots, width})
        .with_dtype<fp32_t, fp16_t, bf16_t>(target_dtype)
        .with_device(device)
        .verify(target_topk_logits);
    TensorMatcher({batch, slots, width}).with_dtype<int64_t>().with_device(device).verify(target_topk_indices);
    TensorMatcher({batch, gamma, vocab})
        .with_dtype<fp32_t, fp16_t, bf16_t>(draft_dtype)
        .with_device(device)
        .verify(draft_logits);
    TensorMatcher({batch, slots}).with_dtype<int64_t>().with_device(device).verify(candidates);
    TensorMatcher({batch})
        .with_dtype<float>()
        .with_device(device)
        .verify(target_temperatures)
        .verify(draft_temperatures)
        .verify(top_ps)
        .verify(uniform_samples_final);
    TensorMatcher({batch})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(top_ks)
        .verify(correct_len)
        .verify(cap_trim_lens);
    TensorMatcher({batch, gamma}).with_dtype<float>().with_device(device).verify(uniform_samples);
    TensorMatcher({batch}).with_dtype<int64_t>().with_device(device).verify(bonus);

    const int64_t batch_size = batch.unwrap();
    const int64_t num_slots = slots.unwrap();
    const int64_t gamma_size = gamma.unwrap();
    const int64_t support_width = width.unwrap();
    const int64_t vocab_size = vocab.unwrap();
    const int64_t draft_rows = batch_size * gamma_size;
    const int64_t num_splits = (vocab_size + device::dspark_sparse_accept::kValuesPerSplit - 1) /
                               device::dspark_sparse_accept::kValuesPerSplit;

    RuntimeCheck(batch_size > 0, "DSpark sparse accept requires a non-empty batch");
    RuntimeCheck(num_slots == gamma_size + 1, "DSpark sparse accept requires slots == gamma + 1");
    RuntimeCheck(
        support_width > 0 && support_width <= device::dspark_sparse_accept::kWarpSize,
        "DSpark sparse accept support width must be in [1, 32], got ",
        support_width);
    RuntimeCheck(vocab_size > 0, "DSpark sparse accept requires a non-empty vocabulary");
    RuntimeCheck(
        num_splits > 0 && num_splits <= device::dspark_sparse_accept::kWarpSize,
        "DSpark sparse accept supports at most ",
        device::dspark_sparse_accept::kWarpSize * device::dspark_sparse_accept::kValuesPerSplit,
        " vocabulary entries, got ",
        vocab_size);

    TensorMatcher({draft_rows, num_splits})
        .with_dtype<float>()
        .with_device(device)
        .verify(partial_max)
        .verify(partial_sum);
    TensorMatcher({draft_rows}).with_dtype<float>().with_device(device).verify(draft_log_normalizers);

    const int64_t* verify_lens_ptr = nullptr;
    if (verify_lens.has_value()) {
      TensorMatcher({batch}).with_dtype<int64_t>().with_device(device).verify(verify_lens.value());
      verify_lens_ptr = static_cast<const int64_t*>(verify_lens.value().data_ptr());
    }

    const auto launch_for_types = [&](auto target_type, auto draft_type) {
      using TargetT = decltype(target_type);
      using DraftT = decltype(draft_type);
      LaunchKernel(
          dim3(static_cast<uint32_t>(num_splits), static_cast<uint32_t>(draft_rows)),
          device::dspark_sparse_accept::kNormalizerThreads,
          device.unwrap())(
          device::dspark_sparse_accept::DraftLogNormalizerPartialsKernel<DraftT>,
          static_cast<const DraftT*>(draft_logits.data_ptr()),
          static_cast<const float*>(draft_temperatures.data_ptr()),
          static_cast<float*>(partial_max.data_ptr()),
          static_cast<float*>(partial_sum.data_ptr()),
          static_cast<uint32_t>(gamma_size),
          static_cast<uint32_t>(vocab_size),
          static_cast<uint32_t>(num_splits));
      LaunchKernel(static_cast<uint32_t>(draft_rows), device::dspark_sparse_accept::kWarpSize, device.unwrap())(
          device::dspark_sparse_accept::DraftLogNormalizerFinalizeKernel,
          static_cast<const float*>(partial_max.data_ptr()),
          static_cast<const float*>(partial_sum.data_ptr()),
          static_cast<float*>(draft_log_normalizers.data_ptr()),
          static_cast<uint32_t>(num_splits));
      LaunchKernel(static_cast<uint32_t>(batch_size), device::dspark_sparse_accept::kWarpSize, device.unwrap())(
          device::dspark_sparse_accept::SparseChainAcceptKernel<TargetT, DraftT>,
          static_cast<const TargetT*>(target_topk_logits.data_ptr()),
          static_cast<const int64_t*>(target_topk_indices.data_ptr()),
          static_cast<const DraftT*>(draft_logits.data_ptr()),
          static_cast<const float*>(draft_log_normalizers.data_ptr()),
          static_cast<const int64_t*>(candidates.data_ptr()),
          static_cast<const float*>(target_temperatures.data_ptr()),
          static_cast<const float*>(draft_temperatures.data_ptr()),
          static_cast<const int32_t*>(top_ks.data_ptr()),
          static_cast<const float*>(top_ps.data_ptr()),
          static_cast<const float*>(uniform_samples.data_ptr()),
          static_cast<const float*>(uniform_samples_final.data_ptr()),
          verify_lens_ptr,
          static_cast<int32_t*>(correct_len.data_ptr()),
          static_cast<int64_t*>(bonus.data_ptr()),
          static_cast<int32_t*>(cap_trim_lens.data_ptr()),
          static_cast<uint32_t>(num_slots),
          static_cast<uint32_t>(gamma_size),
          static_cast<uint32_t>(support_width),
          static_cast<uint32_t>(vocab_size));
    };

    const auto launch_for_draft = [&](auto target_type) {
      if (is_type<fp32_t>(draft_logits.dtype())) {
        launch_for_types(target_type, float{});
      } else if (is_type<fp16_t>(draft_logits.dtype())) {
        launch_for_types(target_type, __half{});
      } else {
        launch_for_types(target_type, __nv_bfloat16{});
      }
    };
    if (is_type<fp32_t>(target_topk_logits.dtype())) {
      launch_for_draft(float{});
    } else if (is_type<fp16_t>(target_topk_logits.dtype())) {
      launch_for_draft(__half{});
    } else {
      launch_for_draft(__nv_bfloat16{});
    }
  }
};

}  // namespace sglang
