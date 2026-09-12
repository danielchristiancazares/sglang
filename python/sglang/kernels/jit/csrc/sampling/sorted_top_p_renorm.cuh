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

namespace device::sorted_top_p_renorm {

constexpr uint32_t kWarpSize = 32;
constexpr uint32_t kFullWarpMask = 0xffffffffU;

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

template <typename LogitT, bool kSample>
__global__ void SortedTopKTopPNormalizeKernel(
    const LogitT* __restrict__ logits,
    const int64_t* __restrict__ indices,
    const float* __restrict__ temperatures,
    const int32_t* __restrict__ top_ks,
    const float* __restrict__ top_ps,
    const float* __restrict__ exp_noise,
    const bool* __restrict__ greedy_mask,
    float* __restrict__ probabilities,
    int64_t* __restrict__ retained_indices,
    int64_t* __restrict__ sampled_tokens,
    uint32_t width,
    uint32_t vocab_size) {
  const uint32_t row = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t compact_offset = static_cast<uint64_t>(row) * width + lane;

  const int32_t requested_top_k = top_ks[row];
  const uint32_t positive_top_k = static_cast<uint32_t>(requested_top_k > 0 ? requested_top_k : 1);
  const uint32_t top_k = width < positive_top_k ? width : positive_top_k;
  const bool valid = lane < top_k;
  const float temperature = fmaxf(temperatures[row], 1.0e-5F);
  const float scaled_logit = valid ? to_float(logits[compact_offset]) / temperature : -CUDART_INF_F;
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

  const float top_p = top_ps[row];
  const bool crosses_after_lane = valid && prefix - probability < top_p;
  const uint32_t prefix_mask = __ballot_sync(kFullWarpMask, crosses_after_lane);
  const uint32_t cutoff_lane = prefix_mask == 0U ? 0U : 31U - static_cast<uint32_t>(__clz(prefix_mask));
  const float cutoff_probability = __shfl_sync(kFullWarpMask, probability, cutoff_lane);
  const float retained = valid && probability >= cutoff_probability ? probability : 0.0F;
  const float retained_sum = warp_sum(retained);
  const float normalized = retained_sum > 0.0F ? retained / retained_sum : probability;

  if (lane < width) probabilities[compact_offset] = normalized;

  if constexpr (kSample) {
    const int64_t token = indices[compact_offset];
    if (lane < width) retained_indices[compact_offset] = token;

    const bool greedy = greedy_mask[row];
    float score = -CUDART_INF_F;
    if (valid && normalized > 0.0F) {
      const float noise = exp_noise[static_cast<uint64_t>(row) * vocab_size + token];
      score = greedy ? scaled_logit : normalized / noise;
    }
    const float best_score = warp_max(score);
    const uint64_t candidate =
        valid && normalized > 0.0F && score == best_score ? static_cast<uint64_t>(token) : UINT64_MAX;
    const uint64_t sampled = warp_min(candidate);
    if (lane == 0) sampled_tokens[row] = sampled == UINT64_MAX ? 0 : static_cast<int64_t>(sampled);
  }
}

__global__ void SortedTopPRenormKernel(
    const float* __restrict__ probs, const float* __restrict__ top_ps, float* __restrict__ output, uint32_t width) {
  const uint32_t row = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  const uint64_t offset = static_cast<uint64_t>(row) * width + lane;
  const float probability = lane < width ? probs[offset] : 0.0F;

  float prefix = probability;
#pragma unroll
  for (uint32_t distance = 1; distance < kWarpSize; distance <<= 1) {
    const float prior = __shfl_up_sync(kFullWarpMask, prefix, distance);
    if (lane >= distance) prefix += prior;
  }

  const float top_p = top_ps[row];
  const bool crosses_after_lane = lane < width && prefix - probability < top_p;
  const uint32_t prefix_mask = __ballot_sync(kFullWarpMask, crosses_after_lane);
  const uint32_t cutoff_lane = prefix_mask == 0U ? 0U : 31U - static_cast<uint32_t>(__clz(prefix_mask));
  const float cutoff_probability = __shfl_sync(kFullWarpMask, probability, cutoff_lane);

  // Match threshold-based top-p semantics: if the cutoff probability is tied,
  // retain every equal-probability entry instead of truncating the tie by rank.
  const float retained = lane < width && probability >= cutoff_probability ? probability : 0.0F;
  const float retained_sum = warp_sum(retained);
  if (lane < width) {
    output[offset] = retained_sum > 0.0F ? retained / retained_sum : probability;
  }
}

}  // namespace device::sorted_top_p_renorm

namespace detail::sorted_top_p_renorm {

template <bool kSample>
void run_sorted_top_k_top_p(
    const tvm::ffi::TensorView logits,
    const tvm::ffi::Optional<tvm::ffi::TensorView> indices,
    const tvm::ffi::TensorView temperatures,
    const tvm::ffi::TensorView top_ks,
    const tvm::ffi::TensorView top_ps,
    const tvm::ffi::Optional<tvm::ffi::TensorView> exp_noise,
    const tvm::ffi::Optional<tvm::ffi::TensorView> greedy_mask,
    const tvm::ffi::TensorView probabilities,
    const tvm::ffi::Optional<tvm::ffi::TensorView> retained_indices,
    const tvm::ffi::Optional<tvm::ffi::TensorView> sampled_tokens) {
  using namespace host;

  auto rows = SymbolicSize{"rows"};
  auto width = SymbolicSize{"width"};
  auto vocab_size = SymbolicSize{"vocab_size"};
  auto device = SymbolicDevice{};
  auto logits_dtype = SymbolicDType{};
  device.set_options<kDLCUDA>();

  TensorMatcher({rows, width}).with_dtype<fp32_t, fp16_t, bf16_t>(logits_dtype).with_device(device).verify(logits);
  TensorMatcher({rows}).with_dtype<float>().with_device(device).verify(temperatures).verify(top_ps);
  TensorMatcher({rows}).with_dtype<int32_t>().with_device(device).verify(top_ks);
  TensorMatcher({rows, width}).with_dtype<float>().with_device(device).verify(probabilities);

  RuntimeCheck(rows.unwrap() > 0, "sorted top-k/top-p requires a non-empty batch");
  RuntimeCheck(
      width.unwrap() > 0 && width.unwrap() <= device::sorted_top_p_renorm::kWarpSize,
      "sorted top-k/top-p width must be in [1, 32], got ",
      width.unwrap());

  const int64_t* indices_ptr = nullptr;
  const float* noise_ptr = nullptr;
  const bool* greedy_ptr = nullptr;
  int64_t* retained_indices_ptr = nullptr;
  int64_t* sampled_tokens_ptr = nullptr;
  if constexpr (kSample) {
    RuntimeCheck(
        indices.has_value() && exp_noise.has_value() && greedy_mask.has_value() && retained_indices.has_value() &&
            sampled_tokens.has_value(),
        "sorted top-k/top-p sampling requires indices, noise, greedy mask, and outputs");
    TensorMatcher({rows, width}).with_dtype<int64_t>().with_device(device).verify(indices.value());
    TensorMatcher({rows, vocab_size}).with_dtype<float>().with_device(device).verify(exp_noise.value());
    TensorMatcher({rows}).with_device(device).verify(greedy_mask.value());
    const auto greedy_dtype = greedy_mask.value().dtype();
    RuntimeCheck(
        greedy_dtype.code == kDLBool && greedy_dtype.bits == 8 && greedy_dtype.lanes == 1,
        "greedy mask must have bool dtype, got ",
        greedy_dtype);
    TensorMatcher({rows, width}).with_dtype<int64_t>().with_device(device).verify(retained_indices.value());
    TensorMatcher({rows}).with_dtype<int64_t>().with_device(device).verify(sampled_tokens.value());
    indices_ptr = static_cast<const int64_t*>(indices.value().data_ptr());
    noise_ptr = static_cast<const float*>(exp_noise.value().data_ptr());
    greedy_ptr = static_cast<const bool*>(greedy_mask.value().data_ptr());
    retained_indices_ptr = static_cast<int64_t*>(retained_indices.value().data_ptr());
    sampled_tokens_ptr = static_cast<int64_t*>(sampled_tokens.value().data_ptr());
  } else {
    vocab_size.set_value(1);
  }

  const auto launch = [&](auto logit_type) {
    using LogitT = decltype(logit_type);
    LaunchKernel(static_cast<uint32_t>(rows.unwrap()), device::sorted_top_p_renorm::kWarpSize, device.unwrap())(
        device::sorted_top_p_renorm::SortedTopKTopPNormalizeKernel<LogitT, kSample>,
        static_cast<const LogitT*>(logits.data_ptr()),
        indices_ptr,
        static_cast<const float*>(temperatures.data_ptr()),
        static_cast<const int32_t*>(top_ks.data_ptr()),
        static_cast<const float*>(top_ps.data_ptr()),
        noise_ptr,
        greedy_ptr,
        static_cast<float*>(probabilities.data_ptr()),
        retained_indices_ptr,
        sampled_tokens_ptr,
        static_cast<uint32_t>(width.unwrap()),
        static_cast<uint32_t>(vocab_size.unwrap()));
  };

  if (is_type<fp32_t>(logits.dtype())) {
    launch(float{});
  } else if (is_type<fp16_t>(logits.dtype())) {
    launch(__half{});
  } else {
    launch(__nv_bfloat16{});
  }
}

}  // namespace detail::sorted_top_p_renorm

struct SortedTopPRenormKernel {
  static void
  run(const tvm::ffi::TensorView probs, const tvm::ffi::TensorView top_ps, const tvm::ffi::TensorView output) {
    using namespace host;

    auto rows = SymbolicSize{"rows"};
    auto width = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({rows, width}).with_dtype<float>().with_device(device).verify(probs).verify(output);
    TensorMatcher({rows}).with_dtype<float>().with_device(device).verify(top_ps);

    const int64_t row_count = rows.unwrap();
    const int64_t row_width = width.unwrap();
    RuntimeCheck(row_count > 0, "sorted top-p requires a non-empty batch");
    RuntimeCheck(
        row_width > 0 && row_width <= device::sorted_top_p_renorm::kWarpSize,
        "sorted top-p width must be in [1, 32], got ",
        row_width);

    LaunchKernel(static_cast<uint32_t>(row_count), device::sorted_top_p_renorm::kWarpSize, device.unwrap())(
        device::sorted_top_p_renorm::SortedTopPRenormKernel,
        static_cast<const float*>(probs.data_ptr()),
        static_cast<const float*>(top_ps.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<uint32_t>(row_width));
  }
};

struct SortedTopKTopPNormalizeKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView temperatures,
      const tvm::ffi::TensorView top_ks,
      const tvm::ffi::TensorView top_ps,
      const tvm::ffi::TensorView probabilities) {
    detail::sorted_top_p_renorm::run_sorted_top_k_top_p<false>(
        logits,
        std::nullopt,
        temperatures,
        top_ks,
        top_ps,
        std::nullopt,
        std::nullopt,
        probabilities,
        std::nullopt,
        std::nullopt);
  }
};

struct SortedTopKTopPSampleKernel {
  static void
  run(const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView temperatures,
      const tvm::ffi::TensorView top_ks,
      const tvm::ffi::TensorView top_ps,
      const tvm::ffi::TensorView exp_noise,
      const tvm::ffi::TensorView greedy_mask,
      const tvm::ffi::TensorView probabilities,
      const tvm::ffi::TensorView retained_indices,
      const tvm::ffi::TensorView sampled_tokens) {
    detail::sorted_top_p_renorm::run_sorted_top_k_top_p<true>(
        logits,
        indices,
        temperatures,
        top_ks,
        top_ps,
        exp_noise,
        greedy_mask,
        probabilities,
        retained_indices,
        sampled_tokens);
  }
};

}  // namespace sglang
