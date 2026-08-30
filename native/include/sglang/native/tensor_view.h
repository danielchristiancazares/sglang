#ifndef SGLANG_NATIVE_TENSOR_VIEW_H_
#define SGLANG_NATIVE_TENSOR_VIEW_H_

#include <stdint.h>

#define SGL_NATIVE_TENSOR_ABI_MAJOR 1u
#define SGL_NATIVE_TENSOR_ABI_MINOR 0u
#define SGL_NATIVE_TENSOR_MAX_RANK 8u
#define SGL_NATIVE_TENSOR_METADATA_V1_SIZE 184u
#define SGL_NATIVE_TENSOR_VIEW_V1_SIZE 192u

typedef uint32_t SglNativeDType;
typedef uint32_t SglNativeDeviceKind;

#define SGL_NATIVE_DTYPE_INVALID ((SglNativeDType)0u)
#define SGL_NATIVE_DTYPE_BOOL8 ((SglNativeDType)1u)
#define SGL_NATIVE_DTYPE_UINT8 ((SglNativeDType)2u)
#define SGL_NATIVE_DTYPE_INT8 ((SglNativeDType)3u)
#define SGL_NATIVE_DTYPE_UINT16 ((SglNativeDType)4u)
#define SGL_NATIVE_DTYPE_INT16 ((SglNativeDType)5u)
#define SGL_NATIVE_DTYPE_UINT32 ((SglNativeDType)6u)
#define SGL_NATIVE_DTYPE_INT32 ((SglNativeDType)7u)
#define SGL_NATIVE_DTYPE_UINT64 ((SglNativeDType)8u)
#define SGL_NATIVE_DTYPE_INT64 ((SglNativeDType)9u)
#define SGL_NATIVE_DTYPE_FLOAT16 ((SglNativeDType)10u)
#define SGL_NATIVE_DTYPE_BFLOAT16 ((SglNativeDType)11u)
#define SGL_NATIVE_DTYPE_FLOAT32 ((SglNativeDType)12u)
#define SGL_NATIVE_DTYPE_FLOAT8_E4M3FN ((SglNativeDType)13u)
#define SGL_NATIVE_DTYPE_FLOAT8_E5M2 ((SglNativeDType)14u)
#define SGL_NATIVE_DTYPE_NVFP4_E2M1 ((SglNativeDType)15u)

#define SGL_NATIVE_DEVICE_INVALID ((SglNativeDeviceKind)0u)
#define SGL_NATIVE_DEVICE_CPU ((SglNativeDeviceKind)1u)
#define SGL_NATIVE_DEVICE_CUDA ((SglNativeDeviceKind)2u)
#define SGL_NATIVE_DEVICE_CUDA_HOST ((SglNativeDeviceKind)3u)

typedef struct SglNativeTensorMetadataV1 SglNativeTensorMetadataV1;
typedef struct SglNativeConstTensorViewV1 SglNativeConstTensorViewV1;
typedef struct SglNativeMutableTensorViewV1 SglNativeMutableTensorViewV1;

#if defined(_MSC_VER)
#pragma pack(push, 8)
#endif

struct SglNativeTensorMetadataV1 {
  uint32_t struct_size;
  uint16_t abi_major;
  uint16_t abi_minor;
  SglNativeDType dtype;
  SglNativeDeviceKind device_kind;
  int32_t device_ordinal;
  uint32_t rank;
  uint64_t allocation_bytes;
  uint64_t storage_offset_elements;
  int64_t extents[SGL_NATIVE_TENSOR_MAX_RANK];
  int64_t strides[SGL_NATIVE_TENSOR_MAX_RANK];
  uint64_t reserved[2];
};

struct SglNativeConstTensorViewV1 {
  SglNativeTensorMetadataV1 metadata;
  const void* allocation_base;
};

struct SglNativeMutableTensorViewV1 {
  SglNativeTensorMetadataV1 metadata;
  void* allocation_base;
};

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

#endif  // SGLANG_NATIVE_TENSOR_VIEW_H_
