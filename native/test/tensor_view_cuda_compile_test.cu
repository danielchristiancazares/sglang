#include "sglang/native/tensor_view.h"
#include "sglang/native/tensor_view.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

using sglang::native::DType;
using sglang::native::TensorValidationError;

static_assert(SGL_NATIVE_TENSOR_ABI_MAJOR == 1);
static_assert(SGL_NATIVE_TENSOR_ABI_MINOR == 0);
static_assert(sizeof(SglNativeTensorMetadataV1) == 184);
static_assert(sizeof(SglNativeConstTensorViewV1) == 192);
static_assert(offsetof(SglNativeTensorMetadataV1, extents) == 40);
static_assert(offsetof(SglNativeTensorMetadataV1, strides) == 104);
static_assert(offsetof(SglNativeConstTensorViewV1, allocation_base) == 184);
static_assert(sizeof(TensorValidationError) == 32);
static_assert(sglang::native::dtype_element_bits(DType::kNvFp4E2M1) == 4);
static_assert(std::is_trivially_copyable_v<SglNativeMutableTensorViewV1>);

__device__ uint64_t tensor_view_compile_probe(
    SglNativeTensorMetadataV1 metadata) {
  return metadata.allocation_bytes + metadata.storage_offset_elements +
         static_cast<uint64_t>(metadata.rank);
}
