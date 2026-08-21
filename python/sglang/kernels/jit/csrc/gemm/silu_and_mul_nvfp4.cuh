#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/type.cuh>

#include <tensorrt_llm/kernels/quantization_utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace sglang {

struct SiluAndMulNVFP4Params {
  const void* input;
  const float* global_scale;
  void* output;
  uint8_t* output_scale;
  uint32_t num_rows;
  uint32_t hidden_dim;
  uint32_t padded_rows;
  uint32_t padded_scale_cols;
};

namespace silu_and_mul_nvfp4_detail {

template <typename T>
using PackedVec = tensorrt_llm::kernels::PackedVec<T, 16>;

struct alignas(32) PackedLoad {
  uint4 lower;
  uint4 upper;
};

template <typename T>
SGL_DEVICE PackedVec<T> load_packed(const T* ptr) {
  const auto raw = *reinterpret_cast<const PackedLoad*>(ptr);
  PackedVec<T> value;
  auto* words = reinterpret_cast<uint4*>(&value);
  words[0] = raw.lower;
  words[1] = raw.upper;
  return value;
}

template <typename T>
SGL_DEVICE T canonicalize_quantizer_input(T value) {
  if constexpr (std::is_same_v<T, bf16_t>) {
    const uint16_t bits = __bfloat16_as_ushort(value);
    if ((bits & 0x7f80u) == 0 && (bits & 0x007fu) != 0) {
      return __ushort_as_bfloat16(bits & 0x8000u);
    }
  }
  return value;
}

template <typename T>
SGL_DEVICE void apply_exact_silu_and_mul(PackedVec<T>& gate, const PackedVec<T>& up) {
  using namespace device;
  auto* gate_values = reinterpret_cast<T*>(&gate);
  const auto* up_values = reinterpret_cast<const T*>(&up);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float gate_f32 = cast<fp32_t>(gate_values[i]);
    const float up_f32 = cast<fp32_t>(up_values[i]);
    const T activated = cast<T>(gate_f32 / (1.0f + expf(-gate_f32)));
    const T product = cast<T>(cast<fp32_t>(activated) * up_f32);
    gate_values[i] = canonicalize_quantizer_input(product);
  }
}

}  // namespace silu_and_mul_nvfp4_detail

template <typename T, bool kUsePDL, bool kDisableQuantFastMath>
__global__ __launch_bounds__(512, 4) void silu_and_mul_nvfp4_kernel(
    const SiluAndMulNVFP4Params __grid_constant__ params) {
  using namespace silu_and_mul_nvfp4_detail;

  constexpr uint32_t kValuesPerBlock = 16;
  const uint32_t scale_cols = params.hidden_dim / kValuesPerBlock;
  const uint64_t total_work = static_cast<uint64_t>(params.padded_rows) * params.padded_scale_cols;

  device::PDLWaitPrimary<kUsePDL>();

  for (uint64_t work_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       work_idx < total_work;
       work_idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const uint32_t row = static_cast<uint32_t>(work_idx / params.padded_scale_cols);
    const uint32_t scale_col = static_cast<uint32_t>(work_idx % params.padded_scale_cols);
    const int64_t scale_offset =
        tensorrt_llm::kernels::get_sf_out_offset_128x4(row, scale_col, scale_cols);

    if (row >= params.num_rows || scale_col >= scale_cols) {
      params.output_scale[scale_offset] = 0;
      continue;
    }

    const auto* input = static_cast<const T*>(params.input);
    const uint64_t row_offset = static_cast<uint64_t>(row) * params.hidden_dim * 2;
    const uint64_t value_offset = row_offset + static_cast<uint64_t>(scale_col) * kValuesPerBlock;
    auto gate = load_packed(input + value_offset);
    const auto up = load_packed(input + value_offset + params.hidden_dim);
    apply_exact_silu_and_mul(gate, up);

    auto* scale_out = params.output_scale + scale_offset;
    const uint64_t packed =
        tensorrt_llm::kernels::
            cvt_warp_fp16_to_fp4<T, 16, 16, false, kDisableQuantFastMath, std::false_type>(
                gate, params.global_scale[0], scale_out);
    const uint64_t output_offset = static_cast<uint64_t>(row) * scale_cols + scale_col;
    static_cast<uint64_t*>(params.output)[output_offset] = packed;
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

template <typename T, bool kUsePDL, bool kDisableQuantFastMath>
struct SiluAndMulNVFP4Kernel {
  static void run(
      const tvm::ffi::TensorView input,
      const tvm::ffi::TensorView global_scale,
      const tvm::ffi::TensorView output,
      const tvm::ffi::TensorView output_scale) {
    using namespace host;

    auto M = SymbolicSize{"num_rows"};
    auto DIn = SymbolicSize{"input_width"};
    auto DOut = SymbolicSize{"packed_output_width"};
    auto MScale = SymbolicSize{"padded_scale_rows"};
    auto DScale = SymbolicSize{"padded_scale_cols"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, DIn}).with_strides({DIn, 1}).with_dtype<T>().with_device(device).verify(input);
    TensorMatcher({1}).with_dtype<float>().with_device(device).verify(global_scale);
    TensorMatcher({M, DOut}).with_dtype<uint8_t>().with_device(device).verify(output);
    TensorMatcher({MScale, DScale}).with_dtype<uint8_t>().with_device(device).verify(output_scale);

    CHECK_HOST(M.unwrap() > 0) << "silu_and_mul_nvfp4: num_rows must be positive";
    CHECK_HOST(DIn.unwrap() > 0 && DIn.unwrap() % 32 == 0)
        << "silu_and_mul_nvfp4: input width must be positive and divisible by 32";

    const int64_t hidden_dim = DIn.unwrap() / 2;
    const int64_t scale_cols = hidden_dim / 16;
    const int64_t padded_rows = div_ceil(M.unwrap(), int64_t{128}) * 128;
    const int64_t padded_scale_cols = div_ceil(scale_cols, int64_t{4}) * 4;

    CHECK_HOST(DOut.unwrap() * 2 == hidden_dim)
        << "silu_and_mul_nvfp4: packed output width mismatch";
    CHECK_HOST(MScale.unwrap() == padded_rows && DScale.unwrap() == padded_scale_cols)
        << "silu_and_mul_nvfp4: scale output shape mismatch";
    CHECK_HOST(M.unwrap() <= UINT32_MAX && hidden_dim <= UINT32_MAX && padded_rows <= UINT32_MAX &&
               padded_scale_cols <= UINT32_MAX)
        << "silu_and_mul_nvfp4: dimensions exceed uint32 indexing";
    CHECK_HOST(runtime::get_cc_major(device.unwrap().device_id) >= 10)
        << "silu_and_mul_nvfp4: NVFP4 conversion requires SM100 or newer";
    CHECK_HOST(
        reinterpret_cast<uintptr_t>(input.data_ptr()) %
            alignof(silu_and_mul_nvfp4_detail::PackedLoad) ==
        0)
        << "silu_and_mul_nvfp4: input must be 32-byte aligned";
    CHECK_HOST(reinterpret_cast<uintptr_t>(output.data_ptr()) % alignof(uint64_t) == 0)
        << "silu_and_mul_nvfp4: output must be 8-byte aligned";

    const auto params = SiluAndMulNVFP4Params{
        .input = input.data_ptr(),
        .global_scale = static_cast<const float*>(global_scale.data_ptr()),
        .output = output.data_ptr(),
        .output_scale = static_cast<uint8_t*>(output_scale.data_ptr()),
        .num_rows = static_cast<uint32_t>(M.unwrap()),
        .hidden_dim = static_cast<uint32_t>(hidden_dim),
        .padded_rows = static_cast<uint32_t>(padded_rows),
        .padded_scale_cols = static_cast<uint32_t>(padded_scale_cols),
    };

    constexpr uint32_t kBlockSize = 512;
    const uint64_t total_work = static_cast<uint64_t>(padded_rows) * padded_scale_cols;
    const uint32_t sm_count = runtime::get_sm_count(device.unwrap().device_id);
    const uint32_t grid = static_cast<uint32_t>(
        std::min<uint64_t>(div_ceil(total_work, uint64_t{kBlockSize}), static_cast<uint64_t>(sm_count) * 4));
    LaunchKernel(grid, kBlockSize, device.unwrap())
        .enable_pdl(kUsePDL)(silu_and_mul_nvfp4_kernel<T, kUsePDL, kDisableQuantFastMath>, params);
  }
};

}  // namespace sglang
