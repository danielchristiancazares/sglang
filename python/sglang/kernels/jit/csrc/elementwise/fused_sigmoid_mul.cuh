#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

struct FusedSigmoidMulParams {
  const bf16_t* __restrict__ attn;
  const bf16_t* __restrict__ gate;
  bf16_t* __restrict__ output;
  int64_t num_vecs;
};

template <bool kUsePDL>
__global__ void fused_sigmoid_mul_kernel(const __grid_constant__ FusedSigmoidMulParams params) {
  using namespace device;
  constexpr uint32_t kVecSize = kMaxVecBytes / sizeof(bf16_t);
  using vec_t = AlignedVector<bf16_t, kVecSize>;

  const int64_t vec_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (vec_id >= params.num_vecs) return;

  PDLWaitPrimary<kUsePDL>();
  const auto attn = load_as<vec_t>(params.attn, vec_id);
  const auto gate = load_as<vec_t>(params.gate, vec_id);
  vec_t output;
#pragma unroll
  for (uint32_t i = 0; i < kVecSize; ++i) {
    const float a = cast<fp32_t>(attn[i]);
    const float g = cast<fp32_t>(gate[i]);
    const float sigmoid = 1.0f / (1.0f + expf(-g));
    output[i] = cast<bf16_t>(a * sigmoid);
  }
  store_as<vec_t>(params.output, output, vec_id);
  PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
struct FusedSigmoidMulKernel {
  static constexpr int64_t kVecSize = device::kMaxVecBytes / sizeof(bf16_t);
  static constexpr uint32_t kBlockSize = 256;

  static void run(
      const tvm::ffi::TensorView attn,
      const tvm::ffi::TensorView gate,
      const tvm::ffi::TensorView output) {
    using namespace host;

    auto numel = SymbolicSize{"numel"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({numel})
        .with_dtype<bf16_t>()
        .with_device(device)
        .verify(attn)
        .verify(gate)
        .verify(output);

    const int64_t n = numel.unwrap();
    if (n == 0) return;
    RuntimeCheck(n % kVecSize == 0, "sigmoid-mul numel must be divisible by the vector width");

    const auto params = FusedSigmoidMulParams{
        .attn = static_cast<const bf16_t*>(attn.data_ptr()),
        .gate = static_cast<const bf16_t*>(gate.data_ptr()),
        .output = static_cast<bf16_t*>(output.data_ptr()),
        .num_vecs = n / kVecSize,
    };
    const auto blocks = div_ceil(params.num_vecs, static_cast<int64_t>(kBlockSize));
    LaunchKernel(blocks, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(fused_sigmoid_mul_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
