#include "sglang/native/linear_verify_rng.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace sglang::native {
namespace {

struct ByteRange final {
  uintptr_t begin;
  uintptr_t end;
  LinearVerifyRngArgument argument;
};

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           LinearVerifyRngArgument argument = LinearVerifyRngArgument::kNone,
           int32_t native_code = 0, uint64_t actual = 0,
           uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,        operation,
                            native_code, static_cast<uint32_t>(argument),
                            actual,      required};
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] const std::byte *
tensor_data(const GraphStableTensorView<D, Rank, Access> &view) noexcept {
  return view.data_bytes();
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError
validate_tensor(const GraphStableTensorView<D, Rank, Access> &view,
                const CudaExecutionContext &context,
                LinearVerifyRngArgument argument,
                const std::array<int64_t, Rank> &expected_extents,
                uint64_t expected_bytes, ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearVerifyRng;
  if (!context.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (!view.is_row_major_contiguous()) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const std::span<const int64_t, Rank> extents = view.extents();
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension]) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, 0, static_cast<uint64_t>(extents[dimension]),
                        static_cast<uint64_t>(expected_extents[dimension]));
    }
  }
  if (view.allocation_bytes() != expected_bytes) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, view.allocation_bytes(), expected_bytes);
  }

  const auto *const data = tensor_data(view);
  if (data == nullptr || range == nullptr ||
      expected_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t bytes = static_cast<uintptr_t>(expected_bytes);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + bytes, argument};
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(const ByteRange &left,
                            const ByteRange &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

template <std::size_t Count>
[[nodiscard]] NativeRuntimeError
validate_non_aliasing(const std::array<ByteRange, Count> &ranges) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearVerifyRng;
  for (std::size_t left = 0; left < Count; ++left) {
    for (std::size_t right = left + 1; right < Count; ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                          ranges[left].argument, 0,
                          static_cast<uint64_t>(ranges[right].argument), 0);
      }
    }
  }
  return native_runtime_ok();
}

[[nodiscard]] NativeRuntimeError
validate_current_device(const CudaExecutionContext &context) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearVerifyRng;
  if (!context.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation);
  }
  int current_device = -1;
  const cudaError_t get_device = cudaGetDevice(&current_device);
  if (get_device != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
                      LinearVerifyRngArgument::kNone,
                      static_cast<int32_t>(get_device));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      LinearVerifyRngArgument::kNone, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  return native_runtime_ok();
}

[[nodiscard]] NativeRuntimeError
validate_seeded_layout(const CudaExecutionContext &context,
                       const SeededLinearVerifyRngBuffers &buffers,
                       LinearVerifyRngShape *shape) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearVerifyRng;
  const NativeRuntimeError device_status = validate_current_device(context);
  if (!is_ok(device_status)) {
    return device_status;
  }
  if (shape == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation);
  }
  const auto accept_extents = buffers.accept_uniforms.extents();
  if (accept_extents[0] != 1) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      LinearVerifyRngArgument::kAcceptUniforms, 0,
                      static_cast<uint64_t>(accept_extents[0]), 1);
  }
  *shape = LinearVerifyRngShape{static_cast<uint64_t>(accept_extents[1])};
  const NativeRuntimeError shape_status =
      validate_linear_verify_rng_shape(*shape);
  if (!is_ok(shape_status)) {
    return shape_status;
  }

  const int64_t slots = static_cast<int64_t>(shape->num_slots);
  const uint64_t coin_bytes = shape->num_slots * sizeof(float);
  std::array<ByteRange, 5> ranges{};
  ByteRange range{};
  NativeRuntimeError status =
      validate_tensor(buffers.seed, context, LinearVerifyRngArgument::kSeed,
                      std::array<int64_t, 1>{1}, sizeof(uint64_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[0] = range;
  status = validate_tensor(buffers.sequence_length, context,
                           LinearVerifyRngArgument::kSequenceLength,
                           std::array<int64_t, 1>{1}, sizeof(uint64_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[1] = range;
  status =
      validate_tensor(buffers.accept_uniforms, context,
                      LinearVerifyRngArgument::kAcceptUniforms,
                      std::array<int64_t, 2>{1, slots}, coin_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[2] = range;
  status = validate_tensor(buffers.bonus_uniforms, context,
                           LinearVerifyRngArgument::kBonusUniforms,
                           std::array<int64_t, 1>{1}, sizeof(float), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[3] = range;
  status = validate_tensor(buffers.device_status, context,
                           LinearVerifyRngArgument::kDeviceStatus,
                           std::array<int64_t, 1>{1}, sizeof(uint32_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[4] = range;
  return validate_non_aliasing(ranges);
}

[[nodiscard]] NativeRuntimeError
validate_stateful_layout(const CudaExecutionContext &context,
                         const StatefulLinearVerifyRngBuffers &buffers,
                         LinearVerifyRngShape *shape) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearVerifyRng;
  const NativeRuntimeError device_status = validate_current_device(context);
  if (!is_ok(device_status)) {
    return device_status;
  }
  if (shape == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation);
  }
  const auto accept_extents = buffers.accept_uniforms.extents();
  if (accept_extents[0] != 1) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      LinearVerifyRngArgument::kAcceptUniforms, 0,
                      static_cast<uint64_t>(accept_extents[0]), 1);
  }
  *shape = LinearVerifyRngShape{static_cast<uint64_t>(accept_extents[1])};
  const NativeRuntimeError shape_status =
      validate_linear_verify_rng_shape(*shape);
  if (!is_ok(shape_status)) {
    return shape_status;
  }

  const int64_t slots = static_cast<int64_t>(shape->num_slots);
  const uint64_t coin_bytes = shape->num_slots * sizeof(float);
  std::array<ByteRange, 4> ranges{};
  ByteRange range{};
  NativeRuntimeError status = validate_tensor(
      buffers.state, context, LinearVerifyRngArgument::kState,
      std::array<int64_t, 1>{4}, sizeof(LinearVerifyRngStateV1), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[0] = range;
  status =
      validate_tensor(buffers.accept_uniforms, context,
                      LinearVerifyRngArgument::kAcceptUniforms,
                      std::array<int64_t, 2>{1, slots}, coin_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[1] = range;
  status = validate_tensor(buffers.bonus_uniforms, context,
                           LinearVerifyRngArgument::kBonusUniforms,
                           std::array<int64_t, 1>{1}, sizeof(float), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[2] = range;
  status = validate_tensor(buffers.device_status, context,
                           LinearVerifyRngArgument::kDeviceStatus,
                           std::array<int64_t, 1>{1}, sizeof(uint32_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[3] = range;
  return validate_non_aliasing(ranges);
}

__device__ __forceinline__ uint32_t rotate_left_32(uint32_t value,
                                                   uint32_t bits) {
  return (value << bits) | (value >> (32U - bits));
}

__device__ __forceinline__ uint32_t murmur_mix(uint32_t hash, uint32_t value) {
  value *= 0xcc9e2d51U;
  value = rotate_left_32(value, 15U);
  value *= 0x1b873593U;
  hash ^= value;
  hash = rotate_left_32(hash, 13U);
  return hash * 5U + 0xe6546b64U;
}

__device__ __forceinline__ uint32_t murmur_finalize(uint32_t hash) {
  hash ^= 16U;
  hash ^= hash >> 16U;
  hash *= 0x85ebca6bU;
  hash ^= hash >> 13U;
  hash *= 0xc2b2ae35U;
  hash ^= hash >> 16U;
  return hash;
}

__device__ __forceinline__ uint32_t seeded_hash(uint64_t seed,
                                                uint64_t sequence_length,
                                                uint32_t column) {
  uint32_t hash = 0U;
  hash = murmur_mix(hash, static_cast<uint32_t>(seed));
  hash = murmur_mix(hash, static_cast<uint32_t>(seed >> 32U));
  hash = murmur_mix(hash, static_cast<uint32_t>(sequence_length));
  hash = murmur_mix(hash, column);
  return murmur_finalize(hash);
}

__device__ __forceinline__ float seeded_coin(uint32_t hash) {
  constexpr float kLargestCoin = 1.0F - 0x1p-24F;
  const double uniform = static_cast<double>(hash) / 4294967295.0;
  const float coin = static_cast<float>(uniform);
  return coin < kLargestCoin ? coin : kLargestCoin;
}

struct PhiloxBlock final {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

__device__ __forceinline__ PhiloxBlock philox4x32_10(uint64_t counter,
                                                     uint64_t subsequence,
                                                     uint64_t seed) {
  constexpr uint32_t kMultiplier0 = 0xd2511f53U;
  constexpr uint32_t kMultiplier1 = 0xcd9e8d57U;
  constexpr uint32_t kWeyl0 = 0x9e3779b9U;
  constexpr uint32_t kWeyl1 = 0xbb67ae85U;

  PhiloxBlock block{static_cast<uint32_t>(counter),
                    static_cast<uint32_t>(counter >> 32U),
                    static_cast<uint32_t>(subsequence),
                    static_cast<uint32_t>(subsequence >> 32U)};
  uint32_t key0 = static_cast<uint32_t>(seed);
  uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
  for (uint32_t round = 0; round < 10U; ++round) {
    const uint32_t high0 = __umulhi(kMultiplier0, block.x);
    const uint32_t low0 = kMultiplier0 * block.x;
    const uint32_t high1 = __umulhi(kMultiplier1, block.z);
    const uint32_t low1 = kMultiplier1 * block.z;
    block =
        PhiloxBlock{high1 ^ block.y ^ key0, low1, high0 ^ block.w ^ key1, low0};
    key0 += kWeyl0;
    key1 += kWeyl1;
  }
  return block;
}

__device__ __forceinline__ float stateful_coin(uint32_t value) {
  return static_cast<float>(value >> 8U) * 0x1p-24F;
}

__global__ void seeded_linear_verify_rng_kernel(const uint64_t *seed,
                                                const uint64_t *sequence_length,
                                                float *accept_uniforms,
                                                float *bonus_uniforms,
                                                uint32_t *device_status,
                                                uint32_t num_slots) {
  const uint32_t draw = threadIdx.x;
  if (draw == 0U) {
    device_status[0] = static_cast<uint32_t>(LinearVerifyRngDeviceCode::kOk);
  }
  if (draw > num_slots) {
    return;
  }
  const float coin =
      seeded_coin(seeded_hash(seed[0], sequence_length[0], draw));
  if (draw < num_slots) {
    accept_uniforms[draw] = coin;
  } else {
    bonus_uniforms[0] = coin;
  }
}

__global__ void stateful_linear_verify_rng_kernel(uint64_t *state,
                                                  float *accept_uniforms,
                                                  float *bonus_uniforms,
                                                  uint32_t *device_status,
                                                  uint32_t num_slots) {
  __shared__ uint64_t seed;
  __shared__ uint64_t subsequence;
  __shared__ uint64_t first_counter;
  __shared__ uint32_t counter_groups;
  __shared__ uint32_t valid;

  const uint32_t thread = threadIdx.x;
  if (thread == 0U) {
    valid = 0U;
    const uint32_t groups = (num_slots + 4U) / 4U;
    LinearVerifyRngDeviceCode code = LinearVerifyRngDeviceCode::kOk;
    if (state[0] != kLinearVerifyRngStateDescriptorV1) {
      code = LinearVerifyRngDeviceCode::kInvalidStateDescriptor;
    } else if (state[3] > 0xffffffffffffffffULL - groups) {
      code = LinearVerifyRngDeviceCode::kCounterOverflow;
    }
    device_status[0] = static_cast<uint32_t>(code);
    if (code == LinearVerifyRngDeviceCode::kOk) {
      seed = state[1];
      subsequence = state[2];
      first_counter = state[3];
      counter_groups = groups;
      state[3] += groups;
      valid = 1U;
    }
  }
  __syncthreads();

  if (valid == 0U || thread >= counter_groups) {
    return;
  }
  const PhiloxBlock block =
      philox4x32_10(first_counter + thread, subsequence, seed);
  const uint32_t values[4]{block.x, block.y, block.z, block.w};
  for (uint32_t lane = 0; lane < 4U; ++lane) {
    const uint32_t draw = thread * 4U + lane;
    if (draw < num_slots) {
      accept_uniforms[draw] = stateful_coin(values[lane]);
    } else if (draw == num_slots) {
      bonus_uniforms[0] = stateful_coin(values[lane]);
    }
  }
}

} // namespace

NativeRuntimeError launch_seeded_linear_verify_rng(
    const CudaExecutionContext &context,
    const SeededLinearVerifyRngBuffers &buffers) noexcept {
  LinearVerifyRngShape shape{};
  const NativeRuntimeError layout_status =
      validate_seeded_layout(context, buffers, &shape);
  if (!is_ok(layout_status)) {
    return layout_status;
  }

  seeded_linear_verify_rng_kernel<<<1, kLinearVerifyRngThreads, 0,
                                    context.stream()>>>(
      reinterpret_cast<const uint64_t *>(buffers.seed.data_bytes()),
      reinterpret_cast<const uint64_t *>(buffers.sequence_length.data_bytes()),
      reinterpret_cast<float *>(buffers.accept_uniforms.data_bytes()),
      reinterpret_cast<float *>(buffers.bonus_uniforms.data_bytes()),
      reinterpret_cast<uint32_t *>(buffers.device_status.data_bytes()),
      static_cast<uint32_t>(shape.num_slots));
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchSeededLinearVerifyRng,
                      LinearVerifyRngArgument::kNone,
                      static_cast<int32_t>(launch_status));
  }
  return native_runtime_ok();
}

NativeRuntimeError launch_stateful_linear_verify_rng(
    const CudaExecutionContext &context,
    const StatefulLinearVerifyRngBuffers &buffers) noexcept {
  LinearVerifyRngShape shape{};
  const NativeRuntimeError layout_status =
      validate_stateful_layout(context, buffers, &shape);
  if (!is_ok(layout_status)) {
    return layout_status;
  }

  stateful_linear_verify_rng_kernel<<<1, kLinearVerifyRngThreads, 0,
                                      context.stream()>>>(
      reinterpret_cast<uint64_t *>(buffers.state.data_bytes()),
      reinterpret_cast<float *>(buffers.accept_uniforms.data_bytes()),
      reinterpret_cast<float *>(buffers.bonus_uniforms.data_bytes()),
      reinterpret_cast<uint32_t *>(buffers.device_status.data_bytes()),
      static_cast<uint32_t>(shape.num_slots));
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchStatefulLinearVerifyRng,
                      LinearVerifyRngArgument::kNone,
                      static_cast<int32_t>(launch_status));
  }
  return native_runtime_ok();
}

} // namespace sglang::native
