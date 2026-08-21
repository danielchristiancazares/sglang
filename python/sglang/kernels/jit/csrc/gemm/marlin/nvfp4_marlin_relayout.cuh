#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cstdint>

#include "gptq_marlin_repack.cuh"

namespace sglang {

namespace device::nvfp4_marlin_relayout {

constexpr uint32_t kTransposeTile = 32;
constexpr uint32_t kTransposeRowsPerThread = 4;
constexpr uint32_t kInverseBlockSize = 256;
constexpr uint32_t kTileK = 16;
constexpr uint32_t kTileN = 64;
constexpr uint32_t kPackedValues = 8;
constexpr uint32_t kWordsPerTile = kTileK * kTileN / kPackedValues;

__global__ void TransposeInt32Kernel(
    const uint32_t* __restrict__ input,
    uint32_t* __restrict__ output,
    uint32_t rows,
    uint32_t cols) {
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
    const uint32_t* __restrict__ marlin,
    uint32_t* __restrict__ transposed_cutlass,
    uint32_t size_n,
    uint32_t size_k) {
  const uint64_t raw_word_index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
        (static_cast<uint64_t>(tile_k) * n_tiles + tile_n) * kWordsPerTile +
        thread_id * 4 + warp_id;
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
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView scratch,
    int64_t size_n,
    int64_t size_k,
    bool to_marlin) {
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
      device::nvfp4_marlin_relayout::kTransposeTile /
          device::nvfp4_marlin_relayout::kTransposeRowsPerThread);

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
    const uint32_t inverse_blocks = static_cast<uint32_t>(
        div_ceil(raw_word_count, device::nvfp4_marlin_relayout::kInverseBlockSize));
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

}  // namespace sglang
