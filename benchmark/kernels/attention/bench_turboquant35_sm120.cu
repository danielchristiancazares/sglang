// Copyright 2026 SGLang Team
//
// Native SM120 admission benchmark for a TurboQuant35-style KV cache.
// This is intentionally isolated from serving dispatch: it measures the real
// Qwen3.8-27B target shape before a new cache ABI is allowed into the runtime.
// The codec follows the Apache-2.0 reference in mitkox/vllm-turboquant at
// c6b2ee90d17eecb43c0afa273ae5ef8ecc8a1a3d, implemented here directly in
// CUDA for the RTX 5090 rather than importing its Triton/Python integration.

#define NOMINMAX

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kHeadDim = 256;
constexpr int kGroupDim = 128;
constexpr int kKvHeads = 4;
constexpr int kQueryHeads = 24;
constexpr int kQueries = 3;
constexpr int kRows = kQueries * kQueryHeads;
constexpr int kQueriesPerKvHead = kQueryHeads / kKvHeads;
constexpr int kPageSize = 64;
constexpr int kPackedBytes = 120;
constexpr int kGroup0MseBytes = 48;
constexpr int kGroup0QjlOffset = 48;
constexpr int kGroup0VectorNormOffset = 64;
constexpr int kGroup0ResidualNormOffset = 66;
constexpr int kGroup1Offset = 68;
constexpr int kGroup1MseBytes = 32;
constexpr int kGroup1QjlOffset = 100;
constexpr int kGroup1VectorNormOffset = 116;
constexpr int kGroup1ResidualNormOffset = 118;
constexpr int kQjlBytes = 16;
constexpr int kTurboDomains = 2 * kHeadDim;
constexpr int kTurboSegmentTokens = 512;
constexpr int kShortFp8SegmentTokens = 512;
constexpr int kLongFp8SegmentTokens = 2048;
constexpr float kAttentionScale = 1.0F / 16.0F;
constexpr float kInverseSqrtGroup = 0.08838834764831845F;
constexpr float kQjlScale = 1.2533141373155001F / kGroupDim;

static_assert(kQueryHeads % kKvHeads == 0);
static_assert(kGroup1Offset + kGroup1MseBytes == kGroup1QjlOffset);
static_assert(kGroup1QjlOffset + kQjlBytes == kGroup1VectorNormOffset);
static_assert(kGroup1ResidualNormOffset + sizeof(__half) == kPackedBytes);

__constant__ std::int8_t device_mse_signs[kHeadDim];
__constant__ std::int8_t device_qjl_signs[kHeadDim];
__constant__ float device_centroids_3bit[8];
__constant__ float device_centroids_2bit[4];

void check_cuda(cudaError_t status, const char *expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#undef CHECK_CUDA
#define CHECK_CUDA(expression) check_cuda((expression), #expression)

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CHECK_CUDA(cudaMalloc(&data_, count * sizeof(T)));
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr)
      cudaFree(data_);
  }

  [[nodiscard]] T *get() const { return data_; }
  [[nodiscard]] std::size_t count() const { return count_; }
  [[nodiscard]] std::size_t bytes() const { return count_ * sizeof(T); }

  [[nodiscard]] std::vector<T> copy_to_host(cudaStream_t stream) const {
    std::vector<T> host(count_);
    CHECK_CUDA(cudaMemcpyAsync(host.data(), data_, bytes(),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return host;
  }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

class CapturedGraph {
public:
  CapturedGraph() = default;
  CapturedGraph(const CapturedGraph &) = delete;
  CapturedGraph &operator=(const CapturedGraph &) = delete;

  ~CapturedGraph() {
    if (executable_ != nullptr)
      cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr)
      cudaGraphDestroy(graph_);
  }

  template <typename Launch> void capture(cudaStream_t stream, Launch launch) {
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    launch();
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph_));
    CHECK_CUDA(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
  }

  void launch(cudaStream_t stream) const {
    CHECK_CUDA(cudaGraphLaunch(executable_, stream));
  }

  [[nodiscard]] std::size_t kernel_nodes() const {
    std::size_t node_count = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph_, nullptr, &node_count));
    std::vector<cudaGraphNode_t> nodes(node_count);
    CHECK_CUDA(cudaGraphGetNodes(graph_, nodes.data(), &node_count));
    std::size_t kernels = 0;
    for (cudaGraphNode_t node : nodes) {
      cudaGraphNodeType type{};
      CHECK_CUDA(cudaGraphNodeGetType(node, &type));
      if (type == cudaGraphNodeTypeKernel)
        ++kernels;
    }
    return kernels;
  }

private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__device__ __forceinline__ float deterministic_value(std::size_t index,
                                                     std::uint32_t seed) {
  const std::uint32_t folded = static_cast<std::uint32_t>(index) ^
                               static_cast<std::uint32_t>(index >> 32);
  const int centered = static_cast<int>(mix32(folded + seed) & 0xffffU) -
                       static_cast<int>(0x8000U);
  return static_cast<float>(centered) / 32768.0F;
}

__global__ void fill_half_and_fp8(__half *raw, __nv_fp8_e4m3 *fp8,
                                  std::size_t count, std::uint32_t seed) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    const float value = deterministic_value(index, seed);
    raw[index] = __float2half_rn(value);
    fp8[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void fill_half(__half *values, std::size_t count,
                          std::uint32_t seed) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x)
    values[index] = __float2half_rn(deterministic_value(index, seed));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ __forceinline__ void fwht_group_128(float *values) {
  const int lane = static_cast<int>(threadIdx.x);
#pragma unroll
  for (int stride = 1; stride < kGroupDim; stride *= 2) {
    const float self = values[lane];
    const float other = values[lane ^ stride];
    __syncthreads();
    values[lane] = (lane & stride) == 0 ? self + other : other - self;
    __syncthreads();
  }
}

__device__ __forceinline__ void fwht_two_groups(float *first, float *second) {
  const int coordinate = static_cast<int>(threadIdx.x);
  const int group_base = coordinate & ~(kGroupDim - 1);
  const int local = coordinate & (kGroupDim - 1);
#pragma unroll
  for (int stride = 1; stride < kGroupDim; stride *= 2) {
    const int partner = group_base + (local ^ stride);
    const float first_self = first[coordinate];
    const float first_other = first[partner];
    const float second_self = second[coordinate];
    const float second_other = second[partner];
    __syncthreads();
    first[coordinate] = (local & stride) == 0 ? first_self + first_other
                                              : first_other - first_self;
    second[coordinate] = (local & stride) == 0 ? second_self + second_other
                                               : second_other - second_self;
    __syncthreads();
  }
}

__device__ __forceinline__ float load_half_bytes(const std::uint8_t *base,
                                                 int offset) {
  return __half2float(*reinterpret_cast<const __half *>(base + offset));
}

__device__ __forceinline__ void store_half_bytes(std::uint8_t *base, int offset,
                                                 float value) {
  *reinterpret_cast<__half *>(base + offset) = __float2half_rn(value);
}

__device__ __forceinline__ int unpack_mse_index(const std::uint8_t *base,
                                                int coordinate) {
  const bool first_group = coordinate < kGroupDim;
  const int local = coordinate & (kGroupDim - 1);
  const int bits = first_group ? 3 : 2;
  const int group_offset = first_group ? 0 : kGroup1Offset;
  const int bit_position = local * bits;
  const int byte_position = bit_position / 8;
  const int bit_offset = bit_position & 7;
  const std::uint16_t packed =
      static_cast<std::uint16_t>(base[group_offset + byte_position]) |
      (static_cast<std::uint16_t>(base[group_offset + byte_position + 1]) << 8);
  return static_cast<int>((packed >> bit_offset) & ((1U << bits) - 1U));
}

__device__ __forceinline__ float unpack_qjl_sign(const std::uint8_t *base,
                                                 int coordinate) {
  const bool first_group = coordinate < kGroupDim;
  const int local = coordinate & (kGroupDim - 1);
  const int offset = first_group ? kGroup0QjlOffset : kGroup1QjlOffset;
  const int bit = (base[offset + local / 8] >> (local & 7)) & 1;
  return bit == 0 ? -1.0F : 1.0F;
}

__device__ __forceinline__ float centroid_for(int coordinate, int index) {
  return coordinate < kGroupDim ? device_centroids_3bit[index]
                                : device_centroids_2bit[index];
}

__global__ void encode_turboquant35(const __half *keys, const __half *values,
                                    std::uint8_t *key_cache,
                                    std::uint8_t *value_cache,
                                    int token_count) {
  __shared__ float transform[kGroupDim];
  __shared__ float residual_transform[kGroupDim];
  __shared__ float reduction[kGroupDim];
  __shared__ std::uint8_t indices[kGroupDim];
  __shared__ std::uint8_t qjl_bits[kGroupDim];

  const int local_vector = static_cast<int>(blockIdx.x);
  const int vectors_per_kind = token_count * kKvHeads;
  const bool is_value = local_vector >= vectors_per_kind;
  const int vector = is_value ? local_vector - vectors_per_kind : local_vector;
  const int token = vector / kKvHeads;
  const int head = vector % kKvHeads;
  const __half *source = is_value ? values : keys;
  std::uint8_t *cache = is_value ? value_cache : key_cache;
  const int source_base = (token * kKvHeads + head) * kHeadDim;
  std::uint8_t *packed = cache + (token * kKvHeads + head) * kPackedBytes;
  const int lane = static_cast<int>(threadIdx.x);

#pragma unroll
  for (int group = 0; group < 2; ++group) {
    const int coordinate = group * kGroupDim + lane;
    const float input = __half2float(source[source_base + coordinate]);
    reduction[lane] = input * input;
    __syncthreads();
#pragma unroll
    for (int stride = kGroupDim / 2; stride > 0; stride /= 2) {
      if (lane < stride)
        reduction[lane] += reduction[lane + stride];
      __syncthreads();
    }
    const float vector_norm = sqrtf(fmaxf(reduction[0], 1.0e-24F));
    const float unit = input / vector_norm;
    transform[lane] = unit * static_cast<float>(device_mse_signs[coordinate]);
    __syncthreads();
    fwht_group_128(transform);
    const float rotated = transform[lane] * kInverseSqrtGroup;

    int nearest_index = 0;
    float nearest =
        group == 0 ? device_centroids_3bit[0] : device_centroids_2bit[0];
    float nearest_distance = fabsf(rotated - nearest);
    const int levels = group == 0 ? 8 : 4;
    for (int level = 1; level < levels; ++level) {
      const float candidate = group == 0 ? device_centroids_3bit[level]
                                         : device_centroids_2bit[level];
      const float distance = fabsf(rotated - candidate);
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest = candidate;
        nearest_index = level;
      }
    }
    indices[lane] = static_cast<std::uint8_t>(nearest_index);
    transform[lane] = nearest;
    __syncthreads();

    fwht_group_128(transform);
    const float approximation =
        transform[lane] * kInverseSqrtGroup *
        static_cast<float>(device_mse_signs[coordinate]);
    const float residual = unit - approximation;
    reduction[lane] = residual * residual;
    residual_transform[lane] =
        residual * static_cast<float>(device_qjl_signs[coordinate]);
    __syncthreads();
#pragma unroll
    for (int stride = kGroupDim / 2; stride > 0; stride /= 2) {
      if (lane < stride)
        reduction[lane] += reduction[lane + stride];
      __syncthreads();
    }
    const float residual_norm = sqrtf(fmaxf(reduction[0], 0.0F));
    fwht_group_128(residual_transform);
    qjl_bits[lane] = residual_transform[lane] >= 0.0F ? 1U : 0U;
    __syncthreads();

    const int mse_bits = group == 0 ? 3 : 2;
    const int mse_bytes = group == 0 ? kGroup0MseBytes : kGroup1MseBytes;
    const int mse_offset = group == 0 ? 0 : kGroup1Offset;
    if (lane < mse_bytes) {
      std::uint8_t output = 0;
#pragma unroll
      for (int bit = 0; bit < 8; ++bit) {
        const int global_bit = lane * 8 + bit;
        const int source_coordinate = global_bit / mse_bits;
        const int source_bit = global_bit % mse_bits;
        output |= static_cast<std::uint8_t>(
            ((indices[source_coordinate] >> source_bit) & 1U) << bit);
      }
      packed[mse_offset + lane] = output;
    }
    if (lane < kQjlBytes) {
      std::uint8_t output = 0;
#pragma unroll
      for (int bit = 0; bit < 8; ++bit)
        output |= static_cast<std::uint8_t>(qjl_bits[lane * 8 + bit] << bit);
      const int qjl_offset = group == 0 ? kGroup0QjlOffset : kGroup1QjlOffset;
      packed[qjl_offset + lane] = output;
    }
    if (lane == 0) {
      const int vector_norm_offset =
          group == 0 ? kGroup0VectorNormOffset : kGroup1VectorNormOffset;
      const int residual_norm_offset =
          group == 0 ? kGroup0ResidualNormOffset : kGroup1ResidualNormOffset;
      store_half_bytes(packed, vector_norm_offset, vector_norm);
      store_half_bytes(packed, residual_norm_offset, residual_norm);
    }
    __syncthreads();
  }
}

__global__ void transform_turboquant_queries(const __half *queries,
                                             float *mse_queries,
                                             float *qjl_queries) {
  __shared__ float mse[kHeadDim];
  __shared__ float qjl[kHeadDim];
  const int row = static_cast<int>(blockIdx.x);
  const int coordinate = static_cast<int>(threadIdx.x);
  const float value = __half2float(queries[row * kHeadDim + coordinate]);
  mse[coordinate] = value * static_cast<float>(device_mse_signs[coordinate]);
  qjl[coordinate] = value * static_cast<float>(device_qjl_signs[coordinate]);
  __syncthreads();
  fwht_two_groups(mse, qjl);
  mse_queries[row * kHeadDim + coordinate] =
      mse[coordinate] * kInverseSqrtGroup;
  qjl_queries[row * kHeadDim + coordinate] = qjl[coordinate] * kQjlScale;
}

template <int SegmentTokens>
__global__ void turboquant_attention_partials(
    const float *mse_queries, const float *qjl_queries,
    const std::uint8_t *key_cache, const std::uint8_t *value_cache,
    float *partial_domains, float *partial_stats, int sequence_length,
    int segment_count) {
  static_assert(SegmentTokens % 8 == 0);
  __shared__ float shared_mse_query[kHeadDim];
  __shared__ float shared_qjl_query[kHeadDim];
  __shared__ float logits[8];
  __shared__ float weights[8];
  __shared__ float value_vector_norm_0[8];
  __shared__ float value_residual_norm_0[8];
  __shared__ float value_vector_norm_1[8];
  __shared__ float value_residual_norm_1[8];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float accumulator_scale;

  const int segment = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int coordinate = static_cast<int>(threadIdx.x);
  const int warp = coordinate / 32;
  const int lane = coordinate & 31;
  const int query_head = row % kQueryHeads;
  const int kv_head = query_head / kQueriesPerKvHead;
  const int segment_begin = segment * SegmentTokens;
  const int segment_end = min(segment_begin + SegmentTokens, sequence_length);

  shared_mse_query[coordinate] = mse_queries[row * kHeadDim + coordinate];
  shared_qjl_query[coordinate] = qjl_queries[row * kHeadDim + coordinate];
  if (coordinate == 0) {
    running_max = -INFINITY;
    running_sum = 0.0F;
  }
  __syncthreads();

  float mse_accumulator = 0.0F;
  float qjl_accumulator = 0.0F;
  for (int tile_begin = segment_begin; tile_begin < segment_end;
       tile_begin += 8) {
    const int token = tile_begin + warp;
    const bool valid = token < segment_end;
    float logit = 0.0F;
    const std::uint8_t *key = nullptr;
    float vector_norm_0 = 0.0F;
    float residual_norm_0 = 0.0F;
    float vector_norm_1 = 0.0F;
    float residual_norm_1 = 0.0F;
    if (valid) {
      key = key_cache + (static_cast<std::size_t>(token) * kKvHeads + kv_head) *
                            kPackedBytes;
      if (lane == 0) {
        vector_norm_0 = load_half_bytes(key, kGroup0VectorNormOffset);
        residual_norm_0 = load_half_bytes(key, kGroup0ResidualNormOffset);
        vector_norm_1 = load_half_bytes(key, kGroup1VectorNormOffset);
        residual_norm_1 = load_half_bytes(key, kGroup1ResidualNormOffset);
      }
      vector_norm_0 = __shfl_sync(0xffffffffU, vector_norm_0, 0);
      residual_norm_0 = __shfl_sync(0xffffffffU, residual_norm_0, 0);
      vector_norm_1 = __shfl_sync(0xffffffffU, vector_norm_1, 0);
      residual_norm_1 = __shfl_sync(0xffffffffU, residual_norm_1, 0);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int key_coordinate = lane + element * 32;
        const bool first_group = key_coordinate < kGroupDim;
        const float vector_norm = first_group ? vector_norm_0 : vector_norm_1;
        const float residual_norm =
            first_group ? residual_norm_0 : residual_norm_1;
        const int index = unpack_mse_index(key, key_coordinate);
        const float centroid = centroid_for(key_coordinate, index);
        const float sign = unpack_qjl_sign(key, key_coordinate);
        logit += vector_norm *
                 (centroid * shared_mse_query[key_coordinate] +
                  residual_norm * sign * shared_qjl_query[key_coordinate]);
      }
      logit = warp_sum(logit);
    }
    if (lane == 0)
      logits[warp] = valid ? logit * kAttentionScale : -INFINITY;
    __syncthreads();

    if (coordinate == 0) {
      float tile_max = logits[0];
#pragma unroll
      for (int index = 1; index < 8; ++index)
        tile_max = fmaxf(tile_max, logits[index]);
      const float next_max = fmaxf(running_max, tile_max);
      accumulator_scale =
          isinf(running_max) ? 0.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        weights[index] =
            isinf(logits[index]) ? 0.0F : expf(logits[index] - next_max);
        tile_sum += weights[index];
      }
      running_sum = running_sum * accumulator_scale + tile_sum;
      running_max = next_max;
    }
    if (coordinate < 8) {
      const int value_token = tile_begin + coordinate;
      if (value_token < segment_end) {
        const std::uint8_t *value =
            value_cache +
            (static_cast<std::size_t>(value_token) * kKvHeads + kv_head) *
                kPackedBytes;
        value_vector_norm_0[coordinate] =
            load_half_bytes(value, kGroup0VectorNormOffset);
        value_residual_norm_0[coordinate] =
            load_half_bytes(value, kGroup0ResidualNormOffset);
        value_vector_norm_1[coordinate] =
            load_half_bytes(value, kGroup1VectorNormOffset);
        value_residual_norm_1[coordinate] =
            load_half_bytes(value, kGroup1ResidualNormOffset);
      } else {
        value_vector_norm_0[coordinate] = 0.0F;
        value_residual_norm_0[coordinate] = 0.0F;
        value_vector_norm_1[coordinate] = 0.0F;
        value_residual_norm_1[coordinate] = 0.0F;
      }
    }
    __syncthreads();

    mse_accumulator *= accumulator_scale;
    qjl_accumulator *= accumulator_scale;
    const bool first_group = coordinate < kGroupDim;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int value_token = tile_begin + index;
      if (value_token < segment_end) {
        const std::uint8_t *value =
            value_cache +
            (static_cast<std::size_t>(value_token) * kKvHeads + kv_head) *
                kPackedBytes;
        const float vector_norm = first_group ? value_vector_norm_0[index]
                                              : value_vector_norm_1[index];
        const float residual_norm = first_group ? value_residual_norm_0[index]
                                                : value_residual_norm_1[index];
        const int mse_index = unpack_mse_index(value, coordinate);
        mse_accumulator +=
            weights[index] * vector_norm * centroid_for(coordinate, mse_index);
        qjl_accumulator += weights[index] * vector_norm * residual_norm *
                           kQjlScale * unpack_qjl_sign(value, coordinate);
      }
    }
    __syncthreads();
  }

  const std::size_t partition =
      static_cast<std::size_t>(row) * segment_count + segment;
  const std::size_t domain_base = partition * kTurboDomains;
  partial_domains[domain_base + coordinate] = mse_accumulator;
  partial_domains[domain_base + kHeadDim + coordinate] = qjl_accumulator;
  if (coordinate == 0) {
    partial_stats[partition * 2] = running_max;
    partial_stats[partition * 2 + 1] = running_sum;
  }
}

template <int SegmentTokens>
__global__ void
fp8_attention_partials(const __half *queries, const __nv_fp8_e4m3 *key_cache,
                       const __nv_fp8_e4m3 *value_cache, float *partial_values,
                       float *partial_stats, int sequence_length,
                       int segment_count) {
  static_assert(SegmentTokens % 8 == 0);
  __shared__ float query[kHeadDim];
  __shared__ float logits[8];
  __shared__ float weights[8];
  __shared__ float running_max;
  __shared__ float running_sum;
  __shared__ float accumulator_scale;

  const int segment = static_cast<int>(blockIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  const int coordinate = static_cast<int>(threadIdx.x);
  const int warp = coordinate / 32;
  const int lane = coordinate & 31;
  const int query_head = row % kQueryHeads;
  const int kv_head = query_head / kQueriesPerKvHead;
  const int segment_begin = segment * SegmentTokens;
  const int segment_end = min(segment_begin + SegmentTokens, sequence_length);

  query[coordinate] = __half2float(queries[row * kHeadDim + coordinate]);
  if (coordinate == 0) {
    running_max = -INFINITY;
    running_sum = 0.0F;
  }
  __syncthreads();

  float accumulator = 0.0F;
  for (int tile_begin = segment_begin; tile_begin < segment_end;
       tile_begin += 8) {
    const int token = tile_begin + warp;
    const bool valid = token < segment_end;
    float logit = 0.0F;
    if (valid) {
      const __nv_fp8_e4m3 *key =
          key_cache +
          (static_cast<std::size_t>(token) * kKvHeads + kv_head) * kHeadDim;
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        const int key_coordinate = lane + element * 32;
        logit +=
            static_cast<float>(key[key_coordinate]) * query[key_coordinate];
      }
      logit = warp_sum(logit);
    }
    if (lane == 0)
      logits[warp] = valid ? logit * kAttentionScale : -INFINITY;
    __syncthreads();

    if (coordinate == 0) {
      float tile_max = logits[0];
#pragma unroll
      for (int index = 1; index < 8; ++index)
        tile_max = fmaxf(tile_max, logits[index]);
      const float next_max = fmaxf(running_max, tile_max);
      accumulator_scale =
          isinf(running_max) ? 0.0F : expf(running_max - next_max);
      float tile_sum = 0.0F;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        weights[index] =
            isinf(logits[index]) ? 0.0F : expf(logits[index] - next_max);
        tile_sum += weights[index];
      }
      running_sum = running_sum * accumulator_scale + tile_sum;
      running_max = next_max;
    }
    __syncthreads();

    accumulator *= accumulator_scale;
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      const int value_token = tile_begin + index;
      if (value_token < segment_end) {
        const __nv_fp8_e4m3 *value =
            value_cache +
            (static_cast<std::size_t>(value_token) * kKvHeads + kv_head) *
                kHeadDim;
        accumulator += weights[index] * static_cast<float>(value[coordinate]);
      }
    }
    __syncthreads();
  }

  const std::size_t partition =
      static_cast<std::size_t>(row) * segment_count + segment;
  partial_values[partition * kHeadDim + coordinate] = accumulator;
  if (coordinate == 0) {
    partial_stats[partition * 2] = running_max;
    partial_stats[partition * 2 + 1] = running_sum;
  }
}

__global__ void finalize_turboquant_attention(const float *partial_domains,
                                              const float *partial_stats,
                                              float *output,
                                              int segment_count) {
  __shared__ float mse[kHeadDim];
  __shared__ float qjl[kHeadDim];
  __shared__ float global_max;
  __shared__ float global_sum;
  const int row = static_cast<int>(blockIdx.x);
  const int coordinate = static_cast<int>(threadIdx.x);

  if (coordinate == 0) {
    float maximum = -INFINITY;
    for (int segment = 0; segment < segment_count; ++segment) {
      const std::size_t partition =
          static_cast<std::size_t>(row) * segment_count + segment;
      maximum = fmaxf(maximum, partial_stats[partition * 2]);
    }
    float sum = 0.0F;
    for (int segment = 0; segment < segment_count; ++segment) {
      const std::size_t partition =
          static_cast<std::size_t>(row) * segment_count + segment;
      sum += partial_stats[partition * 2 + 1] *
             expf(partial_stats[partition * 2] - maximum);
    }
    global_max = maximum;
    global_sum = sum;
  }
  __syncthreads();

  float mse_total = 0.0F;
  float qjl_total = 0.0F;
  for (int segment = 0; segment < segment_count; ++segment) {
    const std::size_t partition =
        static_cast<std::size_t>(row) * segment_count + segment;
    const float scale =
        expf(partial_stats[partition * 2] - global_max) / global_sum;
    const std::size_t domain_base = partition * kTurboDomains;
    mse_total += partial_domains[domain_base + coordinate] * scale;
    qjl_total += partial_domains[domain_base + kHeadDim + coordinate] * scale;
  }
  mse[coordinate] = mse_total;
  qjl[coordinate] = qjl_total;
  __syncthreads();
  fwht_two_groups(mse, qjl);
  output[row * kHeadDim + coordinate] =
      mse[coordinate] * kInverseSqrtGroup *
          static_cast<float>(device_mse_signs[coordinate]) +
      qjl[coordinate] * static_cast<float>(device_qjl_signs[coordinate]);
}

__global__ void finalize_fp8_attention(const float *partial_values,
                                       const float *partial_stats,
                                       float *output, int segment_count) {
  __shared__ float global_max;
  __shared__ float global_sum;
  const int row = static_cast<int>(blockIdx.x);
  const int coordinate = static_cast<int>(threadIdx.x);
  if (coordinate == 0) {
    float maximum = -INFINITY;
    for (int segment = 0; segment < segment_count; ++segment) {
      const std::size_t partition =
          static_cast<std::size_t>(row) * segment_count + segment;
      maximum = fmaxf(maximum, partial_stats[partition * 2]);
    }
    float sum = 0.0F;
    for (int segment = 0; segment < segment_count; ++segment) {
      const std::size_t partition =
          static_cast<std::size_t>(row) * segment_count + segment;
      sum += partial_stats[partition * 2 + 1] *
             expf(partial_stats[partition * 2] - maximum);
    }
    global_max = maximum;
    global_sum = sum;
  }
  __syncthreads();

  float total = 0.0F;
  for (int segment = 0; segment < segment_count; ++segment) {
    const std::size_t partition =
        static_cast<std::size_t>(row) * segment_count + segment;
    const float scale =
        expf(partial_stats[partition * 2] - global_max) / global_sum;
    total += partial_values[partition * kHeadDim + coordinate] * scale;
  }
  output[row * kHeadDim + coordinate] = total;
}

std::uint32_t host_mix32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

std::array<std::int8_t, kHeadDim> make_signs() {
  constexpr std::array<std::uint32_t, 2> mse_seeds = {
      20250428U + 101U + kGroupDim, 20250428U + 211U + kGroupDim};
  std::array<std::int8_t, kHeadDim> signs{};
  for (int group = 0; group < 2; ++group) {
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      signs[static_cast<std::size_t>(group * kGroupDim + coordinate)] =
          (host_mix32(mse_seeds[static_cast<std::size_t>(group)] +
                      static_cast<std::uint32_t>(coordinate)) &
           1U) != 0
              ? 1
              : -1;
    }
  }
  return signs;
}

std::array<std::int8_t, kHeadDim> make_qjl_signs() {
  constexpr std::array<std::uint32_t, 2> qjl_seeds = {
      20250428U + 10000U + 307U + kGroupDim,
      20250428U + 10000U + 401U + kGroupDim};
  std::array<std::int8_t, kHeadDim> signs{};
  for (int group = 0; group < 2; ++group) {
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      signs[static_cast<std::size_t>(group * kGroupDim + coordinate)] =
          (host_mix32(qjl_seeds[static_cast<std::size_t>(group)] +
                      static_cast<std::uint32_t>(coordinate)) &
           1U) != 0
              ? 1
              : -1;
    }
  }
  return signs;
}

std::vector<float> make_centroids(int bits) {
  constexpr int grid_points = 32768;
  constexpr double epsilon = 1.0e-6;
  const int levels = 1 << bits;
  std::vector<double> grid(grid_points);
  std::vector<double> weights(grid_points);
  for (int index = 0; index < grid_points; ++index) {
    const double fraction =
        static_cast<double>(index) / static_cast<double>(grid_points - 1);
    const double value = (-1.0 + epsilon) + fraction * (2.0 - 2.0 * epsilon);
    grid[static_cast<std::size_t>(index)] = value;
    weights[static_cast<std::size_t>(index)] =
        std::pow(std::max(1.0 - value * value, epsilon),
                 0.5 * static_cast<double>(kGroupDim - 3));
  }

  std::vector<double> centroids(static_cast<std::size_t>(levels));
  for (int level = 0; level < levels; ++level) {
    const double fraction =
        static_cast<double>(level) / static_cast<double>(levels - 1);
    centroids[static_cast<std::size_t>(level)] =
        (-1.0 + 1.0 / static_cast<double>(levels + 1)) +
        fraction * (2.0 - 2.0 / static_cast<double>(levels + 1));
  }

  for (int iteration = 0; iteration < 200; ++iteration) {
    std::vector<double> masses(static_cast<std::size_t>(levels), 0.0);
    std::vector<double> sums(static_cast<std::size_t>(levels), 0.0);
    for (int index = 0; index < grid_points; ++index) {
      const double value = grid[static_cast<std::size_t>(index)];
      int assignment = 0;
      while (assignment + 1 < levels &&
             value >=
                 0.5 * (centroids[static_cast<std::size_t>(assignment)] +
                        centroids[static_cast<std::size_t>(assignment + 1)]))
        ++assignment;
      const double weight = weights[static_cast<std::size_t>(index)];
      masses[static_cast<std::size_t>(assignment)] += weight;
      sums[static_cast<std::size_t>(assignment)] += weight * value;
    }
    double maximum_change = 0.0;
    std::vector<double> next = centroids;
    for (int level = 0; level < levels; ++level) {
      const std::size_t offset = static_cast<std::size_t>(level);
      if (masses[offset] > 1.0e-18)
        next[offset] = sums[offset] / masses[offset];
      maximum_change =
          std::max(maximum_change, std::abs(next[offset] - centroids[offset]));
    }
    centroids = std::move(next);
    if (maximum_change < 1.0e-10)
      break;
  }

  std::vector<float> result(static_cast<std::size_t>(levels));
  std::transform(centroids.begin(), centroids.end(), result.begin(),
                 [](double value) { return static_cast<float>(value); });
  return result;
}

struct HostCodecTables {
  std::array<std::int8_t, kHeadDim> mse_signs;
  std::array<std::int8_t, kHeadDim> qjl_signs;
  std::array<float, 8> centroids_3bit;
  std::array<float, 4> centroids_2bit;
};

HostCodecTables initialize_codec_tables() {
  HostCodecTables tables{};
  tables.mse_signs = make_signs();
  tables.qjl_signs = make_qjl_signs();
  const auto centroids_3bit = make_centroids(3);
  const auto centroids_2bit = make_centroids(2);
  std::copy(centroids_3bit.begin(), centroids_3bit.end(),
            tables.centroids_3bit.begin());
  std::copy(centroids_2bit.begin(), centroids_2bit.end(),
            tables.centroids_2bit.begin());
  CHECK_CUDA(cudaMemcpyToSymbol(device_mse_signs, tables.mse_signs.data(),
                                sizeof(tables.mse_signs)));
  CHECK_CUDA(cudaMemcpyToSymbol(device_qjl_signs, tables.qjl_signs.data(),
                                sizeof(tables.qjl_signs)));
  CHECK_CUDA(cudaMemcpyToSymbol(device_centroids_3bit, centroids_3bit.data(),
                                centroids_3bit.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpyToSymbol(device_centroids_2bit, centroids_2bit.data(),
                                centroids_2bit.size() * sizeof(float)));

  std::cout << "centroids_3bit=";
  for (std::size_t index = 0; index < centroids_3bit.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << centroids_3bit[index];
  }
  std::cout << '\n' << "centroids_2bit=";
  for (std::size_t index = 0; index < centroids_2bit.size(); ++index) {
    if (index != 0)
      std::cout << ',';
    std::cout << centroids_2bit[index];
  }
  std::cout << '\n';
  return tables;
}

void host_fwht(std::array<float, kGroupDim> &values) {
  for (int stride = 1; stride < kGroupDim; stride *= 2) {
    for (int base = 0; base < kGroupDim; base += 2 * stride) {
      for (int offset = 0; offset < stride; ++offset) {
        const int left = base + offset;
        const int right = left + stride;
        const float left_value = values[static_cast<std::size_t>(left)];
        const float right_value = values[static_cast<std::size_t>(right)];
        values[static_cast<std::size_t>(left)] = left_value + right_value;
        values[static_cast<std::size_t>(right)] = left_value - right_value;
      }
    }
  }
}

float host_pairwise_sum(std::array<float, kGroupDim> values) {
  for (int stride = kGroupDim / 2; stride > 0; stride /= 2) {
    for (int index = 0; index < stride; ++index)
      values[static_cast<std::size_t>(index)] +=
          values[static_cast<std::size_t>(index + stride)];
  }
  return values[0];
}

void host_store_half(std::array<std::uint8_t, kPackedBytes> &packed, int offset,
                     float value) {
  const __half half_value = __float2half_rn(value);
  std::memcpy(packed.data() + offset, &half_value, sizeof(half_value));
}

float host_load_half(const std::array<std::uint8_t, kPackedBytes> &packed,
                     int offset) {
  __half value{};
  std::memcpy(&value, packed.data() + offset, sizeof(value));
  return __half2float(value);
}

std::array<std::uint8_t, kPackedBytes>
host_encode_turboquant35(const std::array<__half, kHeadDim> &source,
                         const HostCodecTables &tables) {
  std::array<std::uint8_t, kPackedBytes> packed{};
  for (int group = 0; group < 2; ++group) {
    std::array<float, kGroupDim> input{};
    std::array<float, kGroupDim> norm_terms{};
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      const float value = __half2float(
          source[static_cast<std::size_t>(group * kGroupDim + coordinate)]);
      input[static_cast<std::size_t>(coordinate)] = value;
      norm_terms[static_cast<std::size_t>(coordinate)] = value * value;
    }
    const float vector_norm =
        std::sqrt(std::max(host_pairwise_sum(norm_terms), 1.0e-24F));
    std::array<float, kGroupDim> rotated{};
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      const int global_coordinate = group * kGroupDim + coordinate;
      rotated[static_cast<std::size_t>(coordinate)] =
          input[static_cast<std::size_t>(coordinate)] / vector_norm *
          static_cast<float>(
              tables.mse_signs[static_cast<std::size_t>(global_coordinate)]);
    }
    host_fwht(rotated);

    std::array<std::uint8_t, kGroupDim> indices{};
    std::array<float, kGroupDim> approximation{};
    const int levels = group == 0 ? 8 : 4;
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      const float value =
          rotated[static_cast<std::size_t>(coordinate)] * kInverseSqrtGroup;
      int nearest_index = 0;
      float nearest =
          group == 0 ? tables.centroids_3bit[0] : tables.centroids_2bit[0];
      float distance = std::abs(value - nearest);
      for (int level = 1; level < levels; ++level) {
        const float candidate =
            group == 0 ? tables.centroids_3bit[static_cast<std::size_t>(level)]
                       : tables.centroids_2bit[static_cast<std::size_t>(level)];
        const float candidate_distance = std::abs(value - candidate);
        if (candidate_distance < distance) {
          distance = candidate_distance;
          nearest = candidate;
          nearest_index = level;
        }
      }
      indices[static_cast<std::size_t>(coordinate)] =
          static_cast<std::uint8_t>(nearest_index);
      approximation[static_cast<std::size_t>(coordinate)] = nearest;
    }
    host_fwht(approximation);

    std::array<float, kGroupDim> residual_transform{};
    std::array<float, kGroupDim> residual_terms{};
    for (int coordinate = 0; coordinate < kGroupDim; ++coordinate) {
      const int global_coordinate = group * kGroupDim + coordinate;
      const float reconstructed =
          approximation[static_cast<std::size_t>(coordinate)] *
          kInverseSqrtGroup *
          static_cast<float>(
              tables.mse_signs[static_cast<std::size_t>(global_coordinate)]);
      const float residual =
          input[static_cast<std::size_t>(coordinate)] / vector_norm -
          reconstructed;
      residual_terms[static_cast<std::size_t>(coordinate)] =
          residual * residual;
      residual_transform[static_cast<std::size_t>(coordinate)] =
          residual *
          static_cast<float>(
              tables.qjl_signs[static_cast<std::size_t>(global_coordinate)]);
    }
    const float residual_norm =
        std::sqrt(std::max(host_pairwise_sum(residual_terms), 0.0F));
    host_fwht(residual_transform);

    const int mse_bits = group == 0 ? 3 : 2;
    const int mse_bytes = group == 0 ? kGroup0MseBytes : kGroup1MseBytes;
    const int mse_offset = group == 0 ? 0 : kGroup1Offset;
    for (int byte = 0; byte < mse_bytes; ++byte) {
      std::uint8_t output = 0;
      for (int bit = 0; bit < 8; ++bit) {
        const int global_bit = byte * 8 + bit;
        const int coordinate = global_bit / mse_bits;
        const int source_bit = global_bit % mse_bits;
        output |= static_cast<std::uint8_t>(
            ((indices[static_cast<std::size_t>(coordinate)] >> source_bit) & 1U)
            << bit);
      }
      packed[static_cast<std::size_t>(mse_offset + byte)] = output;
    }
    const int qjl_offset = group == 0 ? kGroup0QjlOffset : kGroup1QjlOffset;
    for (int byte = 0; byte < kQjlBytes; ++byte) {
      std::uint8_t output = 0;
      for (int bit = 0; bit < 8; ++bit) {
        const int coordinate = byte * 8 + bit;
        if (residual_transform[static_cast<std::size_t>(coordinate)] >= 0.0F)
          output |= static_cast<std::uint8_t>(1U << bit);
      }
      packed[static_cast<std::size_t>(qjl_offset + byte)] = output;
    }
    host_store_half(
        packed, group == 0 ? kGroup0VectorNormOffset : kGroup1VectorNormOffset,
        vector_norm);
    host_store_half(packed,
                    group == 0 ? kGroup0ResidualNormOffset
                               : kGroup1ResidualNormOffset,
                    residual_norm);
  }
  return packed;
}

struct BenchmarkState {
  explicit BenchmarkState(int sequence_length, int minimum_segment_tokens)
      : max_sequence(sequence_length),
        max_segments((sequence_length + minimum_segment_tokens - 1) /
                     minimum_segment_tokens),
        raw_keys(static_cast<std::size_t>(sequence_length) * kKvHeads *
                 kHeadDim),
        raw_values(static_cast<std::size_t>(sequence_length) * kKvHeads *
                   kHeadDim),
        fp8_keys(static_cast<std::size_t>(sequence_length) * kKvHeads *
                 kHeadDim),
        fp8_values(static_cast<std::size_t>(sequence_length) * kKvHeads *
                   kHeadDim),
        turbo_keys(static_cast<std::size_t>(sequence_length) * kKvHeads *
                   kPackedBytes),
        turbo_values(static_cast<std::size_t>(sequence_length) * kKvHeads *
                     kPackedBytes),
        queries(static_cast<std::size_t>(kRows) * kHeadDim),
        transformed_mse_queries(static_cast<std::size_t>(kRows) * kHeadDim),
        transformed_qjl_queries(static_cast<std::size_t>(kRows) * kHeadDim),
        turbo_partials(static_cast<std::size_t>(kRows) * max_segments *
                       kTurboDomains),
        fp8_partials(static_cast<std::size_t>(kRows) * max_segments * kHeadDim),
        turbo_stats(static_cast<std::size_t>(kRows) * max_segments * 2),
        fp8_stats(static_cast<std::size_t>(kRows) * max_segments * 2),
        turbo_output(static_cast<std::size_t>(kRows) * kHeadDim),
        fp8_output(static_cast<std::size_t>(kRows) * kHeadDim),
        update_keys(static_cast<std::size_t>(kQueries) * kKvHeads *
                    kPackedBytes),
        update_values(static_cast<std::size_t>(kQueries) * kKvHeads *
                      kPackedBytes) {}

  int max_sequence;
  int max_segments;
  DeviceBuffer<__half> raw_keys;
  DeviceBuffer<__half> raw_values;
  DeviceBuffer<__nv_fp8_e4m3> fp8_keys;
  DeviceBuffer<__nv_fp8_e4m3> fp8_values;
  DeviceBuffer<std::uint8_t> turbo_keys;
  DeviceBuffer<std::uint8_t> turbo_values;
  DeviceBuffer<__half> queries;
  DeviceBuffer<float> transformed_mse_queries;
  DeviceBuffer<float> transformed_qjl_queries;
  DeviceBuffer<float> turbo_partials;
  DeviceBuffer<float> fp8_partials;
  DeviceBuffer<float> turbo_stats;
  DeviceBuffer<float> fp8_stats;
  DeviceBuffer<float> turbo_output;
  DeviceBuffer<float> fp8_output;
  DeviceBuffer<std::uint8_t> update_keys;
  DeviceBuffer<std::uint8_t> update_values;
};

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1U) != 0)
    return values[middle];
  return 0.5 * (values[middle - 1] + values[middle]);
}

double mean(const std::vector<double> &values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

double time_graph(const CapturedGraph &graph, cudaStream_t stream,
                  int repetitions) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&end));
  CHECK_CUDA(cudaEventRecord(begin, stream));
  for (int iteration = 0; iteration < repetitions; ++iteration)
    graph.launch(stream);
  CHECK_CUDA(cudaEventRecord(end, stream));
  CHECK_CUDA(cudaEventSynchronize(end));
  float elapsed_ms = 0.0F;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CHECK_CUDA(cudaEventDestroy(begin));
  CHECK_CUDA(cudaEventDestroy(end));
  return static_cast<double>(elapsed_ms) * 1000.0 /
         static_cast<double>(repetitions);
}

std::uint64_t fnv1a(const std::vector<float> &values) {
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(values.data());
  const std::size_t byte_count = values.size() * sizeof(float);
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

struct ErrorMetrics {
  double relative_l2;
  double cosine;
  double maximum_absolute;
};

ErrorMetrics compare_outputs(const std::vector<float> &reference,
                             const std::vector<float> &candidate) {
  if (reference.size() != candidate.size())
    throw std::runtime_error("output size mismatch");
  double difference_squared = 0.0;
  double reference_squared = 0.0;
  double candidate_squared = 0.0;
  double dot = 0.0;
  double maximum_absolute = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const double expected = static_cast<double>(reference[index]);
    const double actual = static_cast<double>(candidate[index]);
    if (!std::isfinite(expected) || !std::isfinite(actual))
      throw std::runtime_error("non-finite attention output");
    const double difference = actual - expected;
    difference_squared += difference * difference;
    reference_squared += expected * expected;
    candidate_squared += actual * actual;
    dot += expected * actual;
    maximum_absolute = std::max(maximum_absolute, std::abs(difference));
  }
  return {std::sqrt(difference_squared / reference_squared),
          dot / std::sqrt(reference_squared * candidate_squared),
          maximum_absolute};
}

double initialize_caches(BenchmarkState &state, cudaStream_t stream) {
  constexpr int threads = 256;
  const std::size_t elements =
      static_cast<std::size_t>(state.max_sequence) * kKvHeads * kHeadDim;
  const int blocks = static_cast<int>(
      std::min<std::size_t>(65535, (elements + threads - 1) / threads));
  fill_half_and_fp8<<<blocks, threads, 0, stream>>>(
      state.raw_keys.get(), state.fp8_keys.get(), elements, 0x1234abcdU);
  fill_half_and_fp8<<<blocks, threads, 0, stream>>>(
      state.raw_values.get(), state.fp8_values.get(), elements, 0x89abcdefU);
  const std::size_t query_elements = static_cast<std::size_t>(kRows) * kHeadDim;
  fill_half<<<static_cast<int>((query_elements + threads - 1) / threads),
              threads, 0, stream>>>(state.queries.get(), query_elements,
                                    0x31415926U);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&end));
  CHECK_CUDA(cudaEventRecord(begin, stream));
  encode_turboquant35<<<2 * state.max_sequence * kKvHeads, kGroupDim, 0,
                        stream>>>(state.raw_keys.get(), state.raw_values.get(),
                                  state.turbo_keys.get(),
                                  state.turbo_values.get(), state.max_sequence);
  CHECK_CUDA(cudaEventRecord(end, stream));
  CHECK_CUDA(cudaEventSynchronize(end));
  CHECK_CUDA(cudaGetLastError());
  float elapsed_ms = 0.0F;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
  CHECK_CUDA(cudaEventDestroy(begin));
  CHECK_CUDA(cudaEventDestroy(end));
  return static_cast<double>(elapsed_ms) * 1000.0;
}

void qualify_codec_encoding(const BenchmarkState &state,
                            const HostCodecTables &tables,
                            cudaStream_t stream) {
  std::array<__half, kHeadDim> source{};
  std::array<std::uint8_t, kPackedBytes> actual{};
  CHECK_CUDA(cudaMemcpyAsync(source.data(), state.raw_keys.get(),
                             sizeof(source), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(actual.data(), state.turbo_keys.get(),
                             sizeof(actual), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const auto expected = host_encode_turboquant35(source, tables);

  std::size_t payload_mismatches = 0;
  for (int offset = 0; offset < kPackedBytes; ++offset) {
    const bool norm_byte = (offset >= kGroup0VectorNormOffset &&
                            offset < kGroup0ResidualNormOffset + 2) ||
                           (offset >= kGroup1VectorNormOffset &&
                            offset < kGroup1ResidualNormOffset + 2);
    if (!norm_byte && actual[static_cast<std::size_t>(offset)] !=
                          expected[static_cast<std::size_t>(offset)])
      ++payload_mismatches;
  }
  double maximum_norm_difference = 0.0;
  constexpr std::array<int, 4> norm_offsets = {
      kGroup0VectorNormOffset, kGroup0ResidualNormOffset,
      kGroup1VectorNormOffset, kGroup1ResidualNormOffset};
  for (const int offset : norm_offsets) {
    maximum_norm_difference = std::max(
        maximum_norm_difference,
        std::abs(static_cast<double>(host_load_half(actual, offset)) -
                 static_cast<double>(host_load_half(expected, offset))));
  }
  if (payload_mismatches != 0 || maximum_norm_difference > 0.0078125)
    throw std::runtime_error("CPU/CUDA TurboQuant encoding parity failed");
  std::cout << "codec_encode_parity=pass,payload_mismatches="
            << payload_mismatches
            << ",max_norm_difference=" << maximum_norm_difference << '\n';
}

struct UpdateResult {
  double median_microseconds;
  std::vector<double> samples;
};

UpdateResult benchmark_update(BenchmarkState &state, cudaStream_t stream) {
  CapturedGraph graph;
  graph.capture(stream, [&state, stream] {
    encode_turboquant35<<<2 * kQueries * kKvHeads, kGroupDim, 0, stream>>>(
        state.raw_keys.get(), state.raw_values.get(), state.update_keys.get(),
        state.update_values.get(), kQueries);
  });
  if (graph.kernel_nodes() != 1)
    throw std::runtime_error("TurboQuant update graph must have one kernel");
  for (int warmup = 0; warmup < 20; ++warmup)
    graph.launch(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  std::vector<double> samples;
  for (int round = 0; round < 7; ++round)
    samples.push_back(time_graph(graph, stream, 1000));
  return {median(samples), samples};
}

struct ShapeResult {
  int sequence_length;
  int fp8_segment_tokens;
  int turbo_segment_tokens;
  double fp8_microseconds;
  double turbo_microseconds;
  ErrorMetrics error;
  std::uint64_t fp8_hash;
  std::uint64_t turbo_hash;
  std::vector<double> fp8_samples;
  std::vector<double> turbo_samples;
};

template <int TurboSegmentTokens, int Fp8SegmentTokens>
ShapeResult benchmark_shape(BenchmarkState &state, int sequence_length,
                            cudaStream_t stream, int rounds) {
  const int turbo_segment_count =
      (sequence_length + TurboSegmentTokens - 1) / TurboSegmentTokens;
  const int fp8_segment_count =
      (sequence_length + Fp8SegmentTokens - 1) / Fp8SegmentTokens;
  if (turbo_segment_count > state.max_segments ||
      fp8_segment_count > state.max_segments)
    throw std::runtime_error("partial workspace is too small");
  const dim3 turbo_partial_grid(static_cast<unsigned int>(turbo_segment_count),
                                static_cast<unsigned int>(kRows));
  const dim3 fp8_partial_grid(static_cast<unsigned int>(fp8_segment_count),
                              static_cast<unsigned int>(kRows));

  CapturedGraph turbo_graph;
  turbo_graph.capture(stream, [&] {
    transform_turboquant_queries<<<kRows, kHeadDim, 0, stream>>>(
        state.queries.get(), state.transformed_mse_queries.get(),
        state.transformed_qjl_queries.get());
    turboquant_attention_partials<TurboSegmentTokens>
        <<<turbo_partial_grid, kHeadDim, 0, stream>>>(
            state.transformed_mse_queries.get(),
            state.transformed_qjl_queries.get(), state.turbo_keys.get(),
            state.turbo_values.get(), state.turbo_partials.get(),
            state.turbo_stats.get(), sequence_length, turbo_segment_count);
    finalize_turboquant_attention<<<kRows, kHeadDim, 0, stream>>>(
        state.turbo_partials.get(), state.turbo_stats.get(),
        state.turbo_output.get(), turbo_segment_count);
  });
  if (turbo_graph.kernel_nodes() != 3)
    throw std::runtime_error(
        "TurboQuant attention graph must have three kernels");

  CapturedGraph fp8_graph;
  fp8_graph.capture(stream, [&] {
    fp8_attention_partials<Fp8SegmentTokens>
        <<<fp8_partial_grid, kHeadDim, 0, stream>>>(
            state.queries.get(), state.fp8_keys.get(), state.fp8_values.get(),
            state.fp8_partials.get(), state.fp8_stats.get(), sequence_length,
            fp8_segment_count);
    finalize_fp8_attention<<<kRows, kHeadDim, 0, stream>>>(
        state.fp8_partials.get(), state.fp8_stats.get(), state.fp8_output.get(),
        fp8_segment_count);
  });
  if (fp8_graph.kernel_nodes() != 2)
    throw std::runtime_error("FP8 attention graph must have two kernels");

  for (int warmup = 0; warmup < 10; ++warmup) {
    turbo_graph.launch(stream);
    fp8_graph.launch(stream);
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  const int repetitions = sequence_length >= 100000 ? 20 : 200;
  std::vector<double> turbo_samples;
  std::vector<double> fp8_samples;
  for (int round = 0; round < rounds; ++round) {
    if ((round & 1) == 0) {
      fp8_samples.push_back(time_graph(fp8_graph, stream, repetitions));
      turbo_samples.push_back(time_graph(turbo_graph, stream, repetitions));
    } else {
      turbo_samples.push_back(time_graph(turbo_graph, stream, repetitions));
      fp8_samples.push_back(time_graph(fp8_graph, stream, repetitions));
    }
  }

  turbo_graph.launch(stream);
  fp8_graph.launch(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const auto first_turbo = state.turbo_output.copy_to_host(stream);
  const auto first_fp8 = state.fp8_output.copy_to_host(stream);
  turbo_graph.launch(stream);
  fp8_graph.launch(stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const auto second_turbo = state.turbo_output.copy_to_host(stream);
  const auto second_fp8 = state.fp8_output.copy_to_host(stream);
  if (std::memcmp(first_turbo.data(), second_turbo.data(),
                  first_turbo.size() * sizeof(float)) != 0 ||
      std::memcmp(first_fp8.data(), second_fp8.data(),
                  first_fp8.size() * sizeof(float)) != 0)
    throw std::runtime_error("CUDA graph replay is not bit deterministic");

  return {sequence_length,       Fp8SegmentTokens,
          TurboSegmentTokens,    median(fp8_samples),
          median(turbo_samples), compare_outputs(first_fp8, first_turbo),
          fnv1a(first_fp8),      fnv1a(first_turbo),
          fp8_samples,           turbo_samples};
}

void print_result(const ShapeResult &result) {
  std::cout << std::fixed << std::setprecision(6)
            << "result,sequence=" << result.sequence_length
            << ",fp8_segment=" << result.fp8_segment_tokens
            << ",turboquant35_segment=" << result.turbo_segment_tokens
            << ",fp8_median_us=" << result.fp8_microseconds
            << ",fp8_mean_us=" << mean(result.fp8_samples)
            << ",turboquant35_median_us=" << result.turbo_microseconds
            << ",turboquant35_mean_us=" << mean(result.turbo_samples)
            << ",delta_us="
            << result.turbo_microseconds - result.fp8_microseconds
            << ",ratio=" << result.turbo_microseconds / result.fp8_microseconds
            << ",relative_l2=" << result.error.relative_l2
            << ",cosine=" << result.error.cosine
            << ",max_abs=" << result.error.maximum_absolute
            << ",fp8_hash=" << std::hex << result.fp8_hash
            << ",turbo_hash=" << result.turbo_hash << std::dec << '\n';
  std::cout << "samples,sequence=" << result.sequence_length
            << ",fp8_segment=" << result.fp8_segment_tokens
            << ",turboquant35_segment=" << result.turbo_segment_tokens
            << ",fp8_us=";
  for (std::size_t index = 0; index < result.fp8_samples.size(); ++index) {
    if (index != 0)
      std::cout << '|';
    std::cout << result.fp8_samples[index];
  }
  std::cout << ",turboquant35_us=";
  for (std::size_t index = 0; index < result.turbo_samples.size(); ++index) {
    if (index != 0)
      std::cout << '|';
    std::cout << result.turbo_samples[index];
  }
  std::cout << '\n';
}

struct Options {
  bool quick = false;
  bool qualify_only = false;
};

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--quick")
      options.quick = true;
    else if (argument == "--qualify-only")
      options.qualify_only = true;
    else
      throw std::runtime_error("usage: bench_turboquant35_sm120.exe "
                               "[--quick] [--qualify-only]");
  }
  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 0)
      throw std::runtime_error("this admission benchmark requires SM120");
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "gpu=" << properties.name << ",sm=" << properties.major
              << properties.minor
              << ",multiprocessors=" << properties.multiProcessorCount
              << ",free_mib=" << free_bytes / (1024 * 1024)
              << ",total_mib=" << total_bytes / (1024 * 1024) << '\n';
    std::cout << "shape=B1,Q3,QH24,KVH4,D256,page" << kPageSize << '\n';
    std::cout << "codec=turboquant35,packed_bytes_per_head=" << kPackedBytes
              << ",fp8_bytes_per_head=" << kHeadDim
              << ",kv_bytes_per_token=" << 2 * kKvHeads * kPackedBytes
              << ",fp8_kv_bytes_per_token=" << 2 * kKvHeads * kHeadDim
              << ",reduction_percent=" << std::fixed << std::setprecision(3)
              << 100.0 * (1.0 - static_cast<double>(kPackedBytes) / kHeadDim)
              << '\n';
    const std::uint64_t per_layer_saving = static_cast<std::uint64_t>(200000) *
                                           2U * kKvHeads *
                                           (kHeadDim - kPackedBytes);
    std::cout << "projected_200k_saving_mib_per_global_layer="
              << static_cast<double>(per_layer_saving) / (1024.0 * 1024.0)
              << ",qwen_16_layer_saving_gib="
              << static_cast<double>(per_layer_saving * 16U) /
                     (1024.0 * 1024.0 * 1024.0)
              << '\n';

    const HostCodecTables codec_tables = initialize_codec_tables();
    const int maximum_sequence = options.qualify_only ? 257 : 199000;
    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    BenchmarkState state(maximum_sequence, 512);
    const double full_encode_us = initialize_caches(state, stream);
    qualify_codec_encoding(state, codec_tables, stream);
    std::cout << std::fixed << std::setprecision(3)
              << "full_cache_encode_us=" << full_encode_us
              << ",encoded_vectors=" << 2 * maximum_sequence * kKvHeads
              << ",us_per_vector="
              << full_encode_us /
                     static_cast<double>(2 * maximum_sequence * kKvHeads)
              << '\n';
    const UpdateResult update = benchmark_update(state, stream);
    std::cout << "graph_update_q3_kv_median_us=" << update.median_microseconds
              << ",mean_us=" << mean(update.samples)
              << ",vectors=" << 2 * kQueries * kKvHeads << ",samples_us=";
    for (std::size_t index = 0; index < update.samples.size(); ++index) {
      if (index != 0)
        std::cout << '|';
      std::cout << update.samples[index];
    }
    std::cout << '\n';

    if (options.qualify_only) {
      print_result(benchmark_shape<kTurboSegmentTokens, kShortFp8SegmentTokens>(
          state, 257, stream, 1));
    } else if (options.quick) {
      print_result(benchmark_shape<kTurboSegmentTokens, kShortFp8SegmentTokens>(
          state, 6213, stream, 3));
    } else {
      print_result(benchmark_shape<kTurboSegmentTokens, kShortFp8SegmentTokens>(
          state, 6213, stream, 7));
      print_result(benchmark_shape<kTurboSegmentTokens, kLongFp8SegmentTokens>(
          state, 199000, stream, 7));
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "final_free_mib=" << free_bytes / (1024 * 1024) << '\n';
    std::cout << "TurboQuant35 SM120 admission benchmark passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
