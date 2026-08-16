#include <torch/library.h>

#include "../../../aot/csrc/gemm/fp8_gemm_kernel.cu"

TORCH_LIBRARY_FRAGMENT(sgl_kernel_windows, m) {
  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, ScalarType out_dtype, Tensor? "
      "bias) -> Tensor");
  m.impl("fp8_scaled_mm", torch::kCUDA, &fp8_scaled_mm);
}
