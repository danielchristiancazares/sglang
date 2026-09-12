#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include "gptq_marlin_repack.cuh"
#include <cstdint>

namespace sglang {

namespace device::nvfp4_marlin_relayout {

constexpr uint32_t kTransposeTile = 32;
constexpr uint32_t kTransposeRowsPerThread = 4;
constexpr uint32_t kInverseBlockSize = 256;
constexpr uint32_t kTileK = 16;
constexpr uint32_t kTileN = 64;
constexpr uint32_t kPackedValues = 8;
constexpr uint32_t kWordsPerTile = kTileK * kTileN / kPackedValues;
constexpr uint32_t kScaleBlockSize = 256;
constexpr uint32_t kCutlassScaleTileN = 128;
constexpr uint32_t kCutlassScaleTileGroups = 4;

__device__ __forceinline__ uint8_t CutlassScaleToMarlin(uint8_t raw) {
  const uint8_t value = raw & 0x7f;
  if (value == 0) return 0;
  if (value < 8) {
    switch (value) {
      case 1:
        return 104;
      case 2:
        return 112;
      case 3:
        return 116;
      case 4:
        return 120;
      case 5:
        return 122;
      case 6:
        return 124;
      default:
        return 126;
    }
  }
  return value < 127 ? static_cast<uint8_t>(value + 120) : 255;
}

__device__ __forceinline__ uint8_t MarlinScaleToCutlass(uint8_t value) {
  switch (value) {
    case 0:
      return 0;
    case 104:
      return 1;
    case 112:
      return 2;
    case 116:
      return 3;
    case 120:
      return 4;
    case 122:
      return 5;
    case 124:
      return 6;
    case 126:
      return 7;
    case 255:
      return 127;
    default:
      return value >= 128 && value <= 246 ? static_cast<uint8_t>(value - 120) : 0;
  }
}

__device__ __forceinline__ uint64_t RawScaleToCutlassIndex(uint32_t n, uint32_t group, uint32_t groups) {
  // ModelOpt's CUTLASS path stores [N, K/16] scales as
  // [N/128, (K/16)/4, 32, 4, 4]. The source Parameter is overwritten by
  // that post-load swizzle, so this path indexes the live representation
  // rather than the checkpoint's row-major representation.
  const uint32_t group_tiles = groups / kCutlassScaleTileGroups;
  const uint32_t n_tile = n / kCutlassScaleTileN;
  const uint32_t n_in_tile = n % kCutlassScaleTileN;
  const uint32_t n_quarter = n_in_tile / 32;
  const uint32_t n_lane = n_in_tile % 32;
  const uint32_t group_tile = group / kCutlassScaleTileGroups;
  const uint32_t group_lane = group % kCutlassScaleTileGroups;
  return (((static_cast<uint64_t>(n_tile) * group_tiles + group_tile) * 32 + n_lane) * kCutlassScaleTileGroups +
          n_quarter) *
             kCutlassScaleTileGroups +
         group_lane;
}

__device__ __forceinline__ void
CutlassIndexToRawScale(uint64_t cutlass_index, uint32_t groups, uint32_t& n, uint32_t& group) {
  const uint32_t group_lane = cutlass_index % kCutlassScaleTileGroups;
  cutlass_index /= kCutlassScaleTileGroups;
  const uint32_t n_quarter = cutlass_index % kCutlassScaleTileGroups;
  cutlass_index /= kCutlassScaleTileGroups;
  const uint32_t n_lane = cutlass_index % 32;
  cutlass_index /= 32;
  const uint32_t group_tiles = groups / kCutlassScaleTileGroups;
  const uint32_t group_tile = cutlass_index % group_tiles;
  const uint32_t n_tile = cutlass_index / group_tiles;
  n = n_tile * kCutlassScaleTileN + n_quarter * 32 + n_lane;
  group = group_tile * kCutlassScaleTileGroups + group_lane;
}

__global__ void CutlassToMarlinScaleKernel(
    const uint8_t* __restrict__ cutlass,
    uint8_t* __restrict__ marlin,
    uint32_t size_n,
    uint32_t groups,
    uint64_t scale_count) {
  const uint64_t marlin_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (marlin_index >= scale_count) return;

  constexpr uint8_t kProcessPermutation[4] = {0, 2, 1, 3};
  const uint64_t permuted_index = (marlin_index & ~uint64_t{3}) + kProcessPermutation[marlin_index & 3];
  const uint32_t within_tile = static_cast<uint32_t>(permuted_index & 63);
  const uint64_t transposed_index = (permuted_index & ~uint64_t{63}) + (within_tile >> 3) + ((within_tile & 7) << 3);
  const uint32_t group = static_cast<uint32_t>(transposed_index / size_n);
  const uint32_t n = static_cast<uint32_t>(transposed_index % size_n);
  const uint64_t cutlass_index = RawScaleToCutlassIndex(n, group, groups);
  marlin[marlin_index] = CutlassScaleToMarlin(cutlass[cutlass_index]);
}

__global__ void MarlinToCutlassScaleKernel(
    const uint8_t* __restrict__ marlin,
    uint8_t* __restrict__ cutlass,
    uint32_t size_n,
    uint32_t groups,
    uint64_t scale_count) {
  const uint64_t cutlass_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cutlass_index >= scale_count) return;

  constexpr uint8_t kProcessPermutation[4] = {0, 2, 1, 3};
  uint32_t n = 0;
  uint32_t group = 0;
  CutlassIndexToRawScale(cutlass_index, groups, n, group);
  const uint64_t transposed_index = static_cast<uint64_t>(group) * size_n + n;
  const uint32_t within_tile = static_cast<uint32_t>(transposed_index & 63);
  const uint64_t permuted_index = (transposed_index & ~uint64_t{63}) + ((within_tile & 7) << 3) + (within_tile >> 3);
  const uint64_t marlin_index = (permuted_index & ~uint64_t{3}) + kProcessPermutation[permuted_index & 3];
  cutlass[cutlass_index] = MarlinScaleToCutlass(marlin[marlin_index]);
}

__global__ void
TransposeInt32Kernel(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, uint32_t rows, uint32_t cols) {
  __shared__ uint32_t tile[kTransposeTile][kTransposeTile + 1];

  const uint32_t input_col = blockIdx.x * kTransposeTile + threadIdx.x;
  const uint32_t input_row = blockIdx.y * kTransposeTile + threadIdx.y;
#pragma unroll
  for (uint32_t offset = 0; offset < kTransposeTile; offset += kTransposeTile / kTransposeRowsPerThread) {
    if (input_row + offset < rows && input_col < cols) {
      tile[threadIdx.y + offset][threadIdx.x] = input[(input_row + offset) * cols + input_col];
    }
  }
  __syncthreads();

  const uint32_t output_col = blockIdx.y * kTransposeTile + threadIdx.x;
  const uint32_t output_row = blockIdx.x * kTransposeTile + threadIdx.y;
#pragma unroll
  for (uint32_t offset = 0; offset < kTransposeTile; offset += kTransposeTile / kTransposeRowsPerThread) {
    if (output_row + offset < cols && output_col < rows) {
      output[(output_row + offset) * rows + output_col] = tile[threadIdx.x][threadIdx.y + offset];
    }
  }
}

__global__ void MarlinToTransposedCutlassKernel(
    const uint32_t* __restrict__ marlin, uint32_t* __restrict__ transposed_cutlass, uint32_t size_n, uint32_t size_k) {
  const uint64_t raw_word_index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint32_t packed_k_count = size_k / kPackedValues;
  const uint64_t raw_word_count = static_cast<uint64_t>(packed_k_count) * size_n;
  if (raw_word_index >= raw_word_count) return;

  const uint32_t packed_k = static_cast<uint32_t>(raw_word_index / size_n);
  const uint32_t n = static_cast<uint32_t>(raw_word_index % size_n);
  const uint32_t n_tiles = size_n / kTileN;
  const uint32_t tile_k = packed_k / 2;
  const uint32_t tile_n = n / kTileN;
  const uint32_t n_in_tile = n % kTileN;
  const uint32_t warp_id = n_in_tile / 16;
  const uint32_t tensor_col = n_in_tile % 8;
  const uint32_t second_n = (n_in_tile % 16) >= 8;
  const uint32_t second_k = packed_k & 1;

  uint32_t raw = 0;
#pragma unroll
  for (uint32_t pair = 0; pair < 4; ++pair) {
    const uint32_t thread_id = tensor_col * 4 + pair;
    const uint64_t marlin_word_index =
        (static_cast<uint64_t>(tile_k) * n_tiles + tile_n) * kWordsPerTile + thread_id * 4 + warp_id;
    const uint32_t packed = marlin[marlin_word_index];
    const uint32_t even_nibble = second_k ? (second_n ? 3 : 1) : (second_n ? 2 : 0);
    const uint32_t odd_nibble = even_nibble + 4;
    raw |= ((packed >> (even_nibble * 4)) & 0xf) << (pair * 8);
    raw |= ((packed >> (odd_nibble * 4)) & 0xf) << (pair * 8 + 4);
  }
  transposed_cutlass[raw_word_index] = raw;
}

}  // namespace device::nvfp4_marlin_relayout

void nvfp4_marlin_relayout_inplace(
    tvm::ffi::TensorView weight, tvm::ffi::TensorView scratch, int64_t size_n, int64_t size_k, bool to_marlin) {
  using namespace host;

  RuntimeCheck(size_n > 0 && size_n % device::marlin::tile_n_size == 0);
  RuntimeCheck(size_k > 0 && size_k % device::marlin::tile_k_size == 0);
  const int64_t weight_bytes = size_n * size_k / 2;
  auto bytes = SymbolicSize{"bytes"};
  bytes.set_value(weight_bytes);
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({bytes}).with_dtype<uint8_t>().with_device(device).verify(weight);
  auto scratch_bytes = SymbolicSize{"scratch_bytes"};
  TensorMatcher({scratch_bytes}).with_dtype<uint8_t>().with_device(device).verify(scratch);
  RuntimeCheck(
      scratch_bytes.unwrap() >= weight_bytes,
      "NVFP4 relayout scratch has ",
      scratch_bytes.unwrap(),
      " bytes, expected at least ",
      weight_bytes);

  const DLDevice dl_device = device.unwrap();
  const int dev = dl_device.device_id;
  const cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  const uint32_t packed_k = static_cast<uint32_t>(size_k / 8);
  const dim3 transpose_threads(
      device::nvfp4_marlin_relayout::kTransposeTile,
      device::nvfp4_marlin_relayout::kTransposeTile / device::nvfp4_marlin_relayout::kTransposeRowsPerThread);

  if (to_marlin) {
    const dim3 transpose_blocks(
        div_ceil(packed_k, device::nvfp4_marlin_relayout::kTransposeTile),
        div_ceil(static_cast<uint32_t>(size_n), device::nvfp4_marlin_relayout::kTransposeTile));
    LaunchKernel(transpose_blocks, transpose_threads, stream)(
        device::nvfp4_marlin_relayout::TransposeInt32Kernel,
        reinterpret_cast<const uint32_t*>(weight.data_ptr()),
        reinterpret_cast<uint32_t*>(scratch.data_ptr()),
        static_cast<uint32_t>(size_n),
        packed_k);

    int blocks = 0;
    int max_shared_mem = 0;
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev));
    RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    RuntimeDeviceCheck(cudaFuncSetAttribute(
        device::marlin::gptq_marlin_repack_kernel<device::marlin::repack_threads, 4, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_shared_mem));
    LaunchKernel(blocks, device::marlin::repack_threads, stream, static_cast<std::size_t>(max_shared_mem))(
        device::marlin::gptq_marlin_repack_kernel<device::marlin::repack_threads, 4, false>,
        reinterpret_cast<const uint32_t*>(scratch.data_ptr()),
        static_cast<const uint32_t*>(nullptr),
        reinterpret_cast<uint32_t*>(weight.data_ptr()),
        static_cast<int>(size_k),
        static_cast<int>(size_n));
  } else {
    const uint64_t raw_word_count = static_cast<uint64_t>(packed_k) * size_n;
    const uint32_t inverse_blocks =
        static_cast<uint32_t>(div_ceil(raw_word_count, device::nvfp4_marlin_relayout::kInverseBlockSize));
    LaunchKernel(inverse_blocks, device::nvfp4_marlin_relayout::kInverseBlockSize, stream)(
        device::nvfp4_marlin_relayout::MarlinToTransposedCutlassKernel,
        reinterpret_cast<const uint32_t*>(weight.data_ptr()),
        reinterpret_cast<uint32_t*>(scratch.data_ptr()),
        static_cast<uint32_t>(size_n),
        static_cast<uint32_t>(size_k));

    const dim3 transpose_blocks(
        div_ceil(static_cast<uint32_t>(size_n), device::nvfp4_marlin_relayout::kTransposeTile),
        div_ceil(packed_k, device::nvfp4_marlin_relayout::kTransposeTile));
    LaunchKernel(transpose_blocks, transpose_threads, stream)(
        device::nvfp4_marlin_relayout::TransposeInt32Kernel,
        reinterpret_cast<const uint32_t*>(scratch.data_ptr()),
        reinterpret_cast<uint32_t*>(weight.data_ptr()),
        packed_k,
        static_cast<uint32_t>(size_n));
  }
}

void nvfp4_marlin_scale_relayout_inplace(
    tvm::ffi::TensorView scale, tvm::ffi::TensorView scratch, int64_t size_n, int64_t size_k, bool to_marlin) {
  using namespace host;

  RuntimeCheck(size_n > 0 && size_n % device::nvfp4_marlin_relayout::kCutlassScaleTileN == 0);
  RuntimeCheck(size_k > 0 && size_k % device::nvfp4_marlin_relayout::kTileK == 0);
  RuntimeCheck(
      (size_k / device::nvfp4_marlin_relayout::kTileK) % device::nvfp4_marlin_relayout::kCutlassScaleTileGroups == 0);
  const int64_t scale_count = size_n * size_k / device::nvfp4_marlin_relayout::kTileK;
  auto scales = SymbolicSize{"scales"};
  scales.set_value(scale_count);
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({scales}).with_dtype<uint8_t>().with_device(device).verify(scale);
  auto scratch_bytes = SymbolicSize{"scratch_bytes"};
  TensorMatcher({scratch_bytes}).with_dtype<uint8_t>().with_device(device).verify(scratch);
  RuntimeCheck(
      scratch_bytes.unwrap() >= scale_count,
      "NVFP4 scale relayout scratch has ",
      scratch_bytes.unwrap(),
      " bytes, expected at least ",
      scale_count);

  const DLDevice dl_device = device.unwrap();
  const cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  const uint32_t blocks = static_cast<uint32_t>(
      div_ceil(static_cast<uint64_t>(scale_count), device::nvfp4_marlin_relayout::kScaleBlockSize));
  if (to_marlin) {
    LaunchKernel(blocks, device::nvfp4_marlin_relayout::kScaleBlockSize, stream)(
        device::nvfp4_marlin_relayout::CutlassToMarlinScaleKernel,
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<uint8_t*>(scratch.data_ptr()),
        static_cast<uint32_t>(size_n),
        static_cast<uint32_t>(size_k / device::nvfp4_marlin_relayout::kTileK),
        static_cast<uint64_t>(scale_count));
  } else {
    LaunchKernel(blocks, device::nvfp4_marlin_relayout::kScaleBlockSize, stream)(
        device::nvfp4_marlin_relayout::MarlinToCutlassScaleKernel,
        reinterpret_cast<const uint8_t*>(scale.data_ptr()),
        reinterpret_cast<uint8_t*>(scratch.data_ptr()),
        static_cast<uint32_t>(size_n),
        static_cast<uint32_t>(size_k / device::nvfp4_marlin_relayout::kTileK),
        static_cast<uint64_t>(scale_count));
  }
  RuntimeDeviceCheck(cudaMemcpyAsync(
      scale.data_ptr(), scratch.data_ptr(), static_cast<std::size_t>(scale_count), cudaMemcpyDeviceToDevice, stream));
}

}  // namespace sglang
