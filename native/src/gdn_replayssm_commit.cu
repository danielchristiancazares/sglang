#include "sglang/native/gdn_replayssm_commit.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace sglang::native {
namespace {

constexpr uint32_t kWarpThreads = 32;
constexpr uint32_t kCopyThreads = 256;

struct ByteRange final {
  uintptr_t begin;
  uintptr_t end;
  GdnReplaySsmArgument argument;
};

// ReplaySSM accepts envelope-strided layer/slot storage while requiring the
// inner kernel tile to stay contiguous.  A single bounding interval is too
// conservative for alias checks: two views may intentionally occupy disjoint
// bands of the same owner allocation even though their envelopes overlap.
// Retain the compact segment description so the validation path can distinguish
// those layouts without allocating host memory.
struct TensorFootprint final {
  ByteRange envelope;
  std::array<uint64_t, 5> extents;
  std::array<uint64_t, 5> strides;
  uint64_t element_bytes;
  uint64_t segment_bytes;
  uint64_t segment_count;
  uint32_t rank;
  uint32_t first_contiguous_dimension;
};

struct SegmentCursor final {
  const TensorFootprint *footprint;
  uint64_t flat_index;
  ByteRange range;
};

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           GdnReplaySsmArgument argument = GdnReplaySsmArgument::kNone,
           int32_t native_code = 0, uint64_t actual = 0,
           uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,        operation,
                            native_code, static_cast<uint32_t>(argument),
                            actual,      required};
}

[[nodiscard]] bool checked_multiply(uint64_t left, uint64_t right,
                                    uint64_t *product) noexcept {
  if (product == nullptr ||
      (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

template <uint32_t Rank>
[[nodiscard]] bool trailing_contiguous(std::span<const int64_t, Rank> extents,
                                       std::span<const int64_t, Rank> strides,
                                       uint32_t first_dimension) noexcept {
  uint64_t expected = 1;
  for (uint32_t dimension = Rank; dimension > first_dimension; --dimension) {
    const uint32_t index = dimension - 1;
    if (extents[index] > 1 &&
        static_cast<uint64_t>(strides[index]) != expected) {
      return false;
    }
    expected *= static_cast<uint64_t>(extents[index]);
  }
  return true;
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError validate_tensor(
    const GraphStableTensorView<D, Rank, Access> &view,
    const CudaExecutionContext &context, GdnReplaySsmArgument argument,
    const std::array<int64_t, Rank> &expected_extents,
    uint32_t first_contiguous_dimension, TensorFootprint *footprint) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateGdnReplaySsmCommit;
  if (!context.valid() || footprint == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  const auto extents = view.extents();
  const auto strides = view.strides();
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension]) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, static_cast<int32_t>(dimension),
                        static_cast<uint64_t>(extents[dimension]),
                        static_cast<uint64_t>(expected_extents[dimension]));
    }
  }
  if (!trailing_contiguous<Rank>(extents, strides,
                                 first_contiguous_dimension)) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const auto *data = view.data_bytes();
  if (data == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  uint64_t maximum_element = 0;
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    const uint64_t extent = static_cast<uint64_t>(extents[dimension]);
    const uint64_t stride = static_cast<uint64_t>(strides[dimension]);
    uint64_t contribution = 0;
    if (extent != 0 && (!checked_multiply(extent - 1, stride, &contribution) ||
                        maximum_element > std::numeric_limits<uint64_t>::max() -
                                              contribution)) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
    maximum_element += contribution;
  }
  const uint64_t element_bytes = dtype_element_bits(D) / 8U;
  uint64_t required_bytes = 0;
  uint64_t origin_bytes = 0;
  if (!checked_multiply(view.storage_offset_elements(), element_bytes,
                        &origin_bytes) ||
      !checked_multiply(maximum_element + 1, element_bytes, &required_bytes) ||
      origin_bytes > view.allocation_bytes() ||
      required_bytes > view.allocation_bytes() - origin_bytes) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation, argument, 0,
        view.allocation_bytes() - (origin_bytes <= view.allocation_bytes()
                                       ? origin_bytes
                                       : view.allocation_bytes()),
        required_bytes);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  if (required_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
      begin > std::numeric_limits<uintptr_t>::max() - required_bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }

  uint64_t segment_count = 1;
  for (uint32_t dimension = 0; dimension < first_contiguous_dimension;
       ++dimension) {
    if (!checked_multiply(segment_count,
                          static_cast<uint64_t>(extents[dimension]),
                          &segment_count)) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
  }
  uint64_t segment_elements = 1;
  for (uint32_t dimension = first_contiguous_dimension; dimension < Rank;
       ++dimension) {
    if (!checked_multiply(segment_elements,
                          static_cast<uint64_t>(extents[dimension]),
                          &segment_elements)) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
  }
  uint64_t segment_bytes = 0;
  if (!checked_multiply(segment_elements, element_bytes, &segment_bytes)) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  TensorFootprint result{
      ByteRange{begin, begin + static_cast<uintptr_t>(required_bytes),
                argument},
      {},
      {},
      element_bytes,
      segment_bytes,
      segment_count,
      Rank,
      first_contiguous_dimension};
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    result.extents[dimension] = static_cast<uint64_t>(extents[dimension]);
    result.strides[dimension] = static_cast<uint64_t>(strides[dimension]);
  }
  *footprint = result;
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(const ByteRange &left,
                            const ByteRange &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] ByteRange segment_at(const TensorFootprint &footprint,
                                   uint64_t flat_index) noexcept {
  uint64_t offset_elements = 0;
  for (uint32_t dimension = footprint.first_contiguous_dimension; dimension > 0;
       --dimension) {
    const uint32_t index = dimension - 1;
    const uint64_t coordinate = flat_index % footprint.extents[index];
    flat_index /= footprint.extents[index];
    offset_elements += coordinate * footprint.strides[index];
  }
  const uintptr_t begin =
      footprint.envelope.begin +
      static_cast<uintptr_t>(offset_elements * footprint.element_bytes);
  return ByteRange{begin,
                   begin + static_cast<uintptr_t>(footprint.segment_bytes),
                   footprint.envelope.argument};
}

[[nodiscard]] bool advance(SegmentCursor *cursor) noexcept {
  if (cursor == nullptr ||
      cursor->flat_index + 1 >= cursor->footprint->segment_count) {
    return false;
  }
  ++cursor->flat_index;
  cursor->range = segment_at(*cursor->footprint, cursor->flat_index);
  return true;
}

[[nodiscard]] bool overlaps(const TensorFootprint &left,
                            const TensorFootprint &right) noexcept {
  if (!overlaps(left.envelope, right.envelope)) {
    return false;
  }
  SegmentCursor left_cursor{&left, 0, segment_at(left, 0)};
  SegmentCursor right_cursor{&right, 0, segment_at(right, 0)};
  while (true) {
    if (overlaps(left_cursor.range, right_cursor.range)) {
      return true;
    }
    if (left_cursor.range.end <= right_cursor.range.begin) {
      if (!advance(&left_cursor)) {
        return false;
      }
    } else {
      if (!advance(&right_cursor)) {
        return false;
      }
    }
  }
}

[[nodiscard]] NativeRuntimeError
validate_layout(const CudaExecutionContext &context,
                const GdnReplaySsmCommitBuffers &buffers,
                GdnReplaySsmShape *shape) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateGdnReplaySsmCommit;
  if (!context.valid() || shape == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation);
  }
  int current_device = -1;
  const cudaError_t get_device = cudaGetDevice(&current_device);
  if (get_device != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
                      GdnReplaySsmArgument::kNone,
                      static_cast<int32_t>(get_device));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      GdnReplaySsmArgument::kNone, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(context.device_ordinal()));
  }

  const auto temporal = buffers.temporal.extents();
  const auto raw_values = buffers.raw_values.extents();
  const auto raw_keys = buffers.raw_keys.extents();
  *shape = GdnReplaySsmShape{
      static_cast<uint64_t>(temporal[0]),
      static_cast<uint64_t>(temporal[1]),
      static_cast<uint64_t>(buffers.state_indices.extents()[0]),
      static_cast<uint64_t>(temporal[2]),
      static_cast<uint64_t>(raw_keys[2]),
      static_cast<uint64_t>(temporal[4]),
      static_cast<uint64_t>(temporal[3]),
      static_cast<uint64_t>(raw_values[3])};
  const NativeRuntimeError shape_status = validate_gdn_replayssm_shape(*shape);
  if (!is_ok(shape_status)) {
    return shape_status;
  }

  const int64_t layers = static_cast<int64_t>(shape->num_layers);
  const int64_t slots = static_cast<int64_t>(shape->num_slots);
  const int64_t batch = static_cast<int64_t>(shape->batch_size);
  const int64_t hv = static_cast<int64_t>(shape->num_value_heads);
  const int64_t h = static_cast<int64_t>(shape->num_key_heads);
  const int64_t k = static_cast<int64_t>(shape->key_dimension);
  const int64_t v = static_cast<int64_t>(shape->value_dimension);
  const int64_t replay = static_cast<int64_t>(shape->replay_length);

  std::array<TensorFootprint, 11> core_ranges{};
  uint32_t core_count = 0;
  TensorFootprint footprint{};
  const auto add = [&](NativeRuntimeError status) noexcept -> bool {
    if (!is_ok(status))
      return false;
    core_ranges[core_count++] = footprint;
    return true;
  };
  NativeRuntimeError status = validate_tensor(
      buffers.temporal, context, GdnReplaySsmArgument::kTemporal,
      std::array<int64_t, 5>{layers, slots, hv, v, k}, 2, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(
      buffers.raw_values, context, GdnReplaySsmArgument::kRawValue,
      std::array<int64_t, 5>{layers, slots, hv, replay, v}, 2, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(
      buffers.raw_keys, context, GdnReplaySsmArgument::kRawKey,
      std::array<int64_t, 5>{layers, slots, h, replay, k}, 2, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(
      buffers.log_decay, context, GdnReplaySsmArgument::kLogDecay,
      std::array<int64_t, 4>{layers, slots, hv, replay}, 2, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.beta, context, GdnReplaySsmArgument::kBeta,
                           std::array<int64_t, 4>{layers, slots, hv, replay}, 2,
                           &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.state_indices, context,
                           GdnReplaySsmArgument::kStateIndices,
                           std::array<int64_t, 1>{batch}, 0, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.accept_lengths, context,
                           GdnReplaySsmArgument::kAcceptLengths,
                           std::array<int64_t, 1>{batch}, 0, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.last_correct_steps, context,
                           GdnReplaySsmArgument::kLastCorrectSteps,
                           std::array<int64_t, 1>{batch}, 0, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.track_indices, context,
                           GdnReplaySsmArgument::kTrackIndices,
                           std::array<int64_t, 1>{batch}, 0, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.track_steps, context,
                           GdnReplaySsmArgument::kTrackSteps,
                           std::array<int64_t, 1>{batch}, 0, &footprint);
  if (!add(status))
    return status;
  status = validate_tensor(buffers.device_status, context,
                           GdnReplaySsmArgument::kDeviceStatus,
                           std::array<int64_t, 1>{1}, 0, &footprint);
  if (!add(status))
    return status;

  for (uint32_t left = 0; left < core_count; ++left) {
    for (uint32_t right = left + 1; right < core_count; ++right) {
      if (overlaps(core_ranges[left], core_ranges[right])) {
        return make_error(
            NativeRuntimeCode::kInvalidArgument, kOperation,
            core_ranges[left].envelope.argument, 0,
            static_cast<uint64_t>(core_ranges[right].envelope.argument));
      }
    }
  }

  for (const GdnReplaySsmConvPair &pair : buffers.conv_pairs) {
    const auto conv = pair.conv_states.extents();
    const auto windows = pair.intermediate_conv_windows.extents();
    if (conv[0] != layers || conv[1] != slots || windows[0] != layers ||
        windows[1] != batch || windows[2] != replay || windows[3] != conv[2] ||
        windows[4] != conv[3]) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        GdnReplaySsmArgument::kConvStates);
    }
    TensorFootprint conv_range{};
    status = validate_tensor(
        pair.conv_states, context, GdnReplaySsmArgument::kConvStates,
        std::array<int64_t, 4>{layers, slots, conv[2], conv[3]}, 2,
        &conv_range);
    if (!is_ok(status))
      return status;
    TensorFootprint windows_range{};
    status = validate_tensor(
        pair.intermediate_conv_windows, context,
        GdnReplaySsmArgument::kIntermediateConv,
        std::array<int64_t, 5>{layers, batch, replay, conv[2], conv[3]}, 3,
        &windows_range);
    if (!is_ok(status))
      return status;
    if (overlaps(conv_range, windows_range)) {
      return make_error(
          NativeRuntimeCode::kInvalidArgument, kOperation,
          GdnReplaySsmArgument::kConvStates, 0,
          static_cast<uint64_t>(GdnReplaySsmArgument::kIntermediateConv));
    }
    for (uint32_t index = 0; index < core_count; ++index) {
      if (overlaps(conv_range, core_ranges[index]) ||
          overlaps(windows_range, core_ranges[index])) {
        return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                          GdnReplaySsmArgument::kConvStates);
      }
    }
    for (const GdnReplaySsmConvPair &prior :
         buffers.conv_pairs.first(&pair - buffers.conv_pairs.data())) {
      const auto prior_conv_extents = prior.conv_states.extents();
      TensorFootprint prior_conv{};
      status = validate_tensor(
          prior.conv_states, context, GdnReplaySsmArgument::kConvStates,
          std::array<int64_t, 4>{layers, slots, prior_conv_extents[2],
                                 prior_conv_extents[3]},
          2, &prior_conv);
      if (!is_ok(status))
        return status;
      TensorFootprint prior_windows{};
      status = validate_tensor(prior.intermediate_conv_windows, context,
                               GdnReplaySsmArgument::kIntermediateConv,
                               std::array<int64_t, 5>{layers, batch, replay,
                                                      prior_conv_extents[2],
                                                      prior_conv_extents[3]},
                               3, &prior_windows);
      if (!is_ok(status))
        return status;
      if (overlaps(conv_range, prior_conv) ||
          overlaps(conv_range, prior_windows) ||
          overlaps(windows_range, prior_conv) ||
          overlaps(windows_range, prior_windows)) {
        return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                          GdnReplaySsmArgument::kConvStates);
      }
    }
  }
  return native_runtime_ok();
}

__global__ void validate_content_kernel(
    const int32_t *state_indices, const int32_t *accept_lengths,
    const int32_t *last_correct_steps, const int32_t *track_indices,
    const int32_t *track_steps, uint32_t *device_status, uint32_t batch_size,
    uint32_t num_slots, uint32_t replay_length, bool require_ready_status) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  if (require_ready_status && device_status[0] != 0U)
    return;
  GdnReplaySsmDeviceCode code = GdnReplaySsmDeviceCode::kOk;
  for (uint32_t request = 0; request < batch_size; ++request) {
    const int32_t state = state_indices[request];
    const int32_t accept = accept_lengths[request];
    const int32_t last = last_correct_steps[request];
    const int32_t track = track_indices[request];
    const int32_t track_step = track_steps[request];
    if (state >= static_cast<int32_t>(num_slots)) {
      code = GdnReplaySsmDeviceCode::kStateIndexOutOfRange;
      break;
    }
    if (state < 0) {
      if (accept != 0 || last >= 0 || track >= 0 || track_step >= 0) {
        code = GdnReplaySsmDeviceCode::kStateIndexOutOfRange;
        break;
      }
      continue;
    }
    if (accept > static_cast<int32_t>(replay_length)) {
      code = GdnReplaySsmDeviceCode::kAcceptLengthOutOfRange;
      break;
    }
    if (accept < 0) {
      code = GdnReplaySsmDeviceCode::kAcceptLengthOutOfRange;
      break;
    }
    if (last >= static_cast<int32_t>(replay_length)) {
      code = GdnReplaySsmDeviceCode::kLastCorrectStepOutOfRange;
      break;
    }
    if (track >= static_cast<int32_t>(num_slots)) {
      code = GdnReplaySsmDeviceCode::kTrackIndexOutOfRange;
      break;
    }
    if (track_step >= static_cast<int32_t>(replay_length)) {
      code = GdnReplaySsmDeviceCode::kTrackStepOutOfRange;
      break;
    }
    if (last != accept - 1) {
      code = GdnReplaySsmDeviceCode::kLastCorrectStepOutOfRange;
      break;
    }
    if ((track >= 0) != (track_step >= 0)) {
      code = track < 0 ? GdnReplaySsmDeviceCode::kTrackIndexOutOfRange
                       : GdnReplaySsmDeviceCode::kTrackStepOutOfRange;
      break;
    }
    if (track_step >= accept) {
      code = GdnReplaySsmDeviceCode::kTrackStepOutOfRange;
      break;
    }
  }
  device_status[0] = static_cast<uint32_t>(code);
}

__device__ __forceinline__ float add_round_to_nearest(float left, float right) {
  return __fadd_rn(left, right);
}

__device__ __forceinline__ float multiply_round_to_nearest(float left,
                                                           float right) {
  return __fmul_rn(left, right);
}

__device__ __forceinline__ float subtract_round_to_nearest(float left,
                                                           float right) {
  return __fsub_rn(left, right);
}

__device__ __forceinline__ float approximate_ftz_sqrt(float value) {
  float result;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float full_precision_divide(float numerator,
                                                       float denominator) {
  float result;
  asm("div.full.f32 %0, %1, %2;"
      : "=f"(result)
      : "f"(numerator), "f"(denominator));
  return result;
}

__device__ __forceinline__ float approximate_exp(float exponent) {
  constexpr float kLog2E = 1.4426950408889634074F;
  const float base_two_exponent = multiply_round_to_nearest(exponent, kLog2E);
  float result;
  asm("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(base_two_exponent));
  return result;
}

__device__ __forceinline__ float warp_butterfly_sum(float value) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    value =
        add_round_to_nearest(value, __shfl_xor_sync(0xffffffffU, value, mask));
  }
  return value;
}

__global__ void gdn_replayssm_fold_kernel(
    float *temporal, const __nv_bfloat16 *raw_values,
    const __nv_bfloat16 *raw_keys, const float *log_decay, const float *beta,
    const int32_t *state_indices, const int32_t *accept_lengths,
    const int32_t *track_indices, const int32_t *track_steps,
    const uint32_t *device_status, uint32_t num_slots, uint32_t batch_size,
    uint32_t num_value_heads, uint32_t num_key_heads, uint32_t key_dimension,
    uint32_t value_dimension, uint32_t replay_length,
    uint64_t temporal_layer_stride, uint64_t temporal_slot_stride,
    uint64_t raw_value_layer_stride, uint64_t raw_value_slot_stride,
    uint64_t raw_key_layer_stride, uint64_t raw_key_slot_stride,
    uint64_t log_decay_layer_stride, uint64_t log_decay_slot_stride,
    uint64_t beta_layer_stride, uint64_t beta_slot_stride) {
  if (device_status[0] != 0U)
    return;
  constexpr uint32_t kValuesPerProgram = 32;
  constexpr uint32_t kKeysPerLane = 4;
  const uint32_t lane = threadIdx.x;
  const uint32_t value_base = blockIdx.x * kValuesPerProgram;
  const uint32_t request = blockIdx.y;
  const uint32_t packed = blockIdx.z;
  const uint32_t layer = packed / num_value_heads;
  const uint32_t value_head = packed % num_value_heads;
  if (request >= batch_size)
    return;
  const int32_t state_slot = state_indices[request];
  const int32_t accept_length = accept_lengths[request];
  if (state_slot < 0 || accept_length <= 0)
    return;
  const uint32_t key_head = value_head / (num_value_heads / num_key_heads);
  const uint64_t temporal_head_base =
      static_cast<uint64_t>(layer) * temporal_layer_stride +
      static_cast<uint64_t>(state_slot) * temporal_slot_stride +
      static_cast<uint64_t>(value_head) * value_dimension * key_dimension;
  const int32_t track_slot = track_indices[request];
  const int32_t track_step = track_steps[request];
  const bool active_value = value_base + lane < value_dimension;
  const uint64_t local_value = active_value ? value_base + lane : 0U;
  float state[kKeysPerLane][kValuesPerProgram];
#pragma unroll
  for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
    const uint32_t key = lane + key_item * kWarpThreads;
#pragma unroll
    for (uint32_t value_item = 0; value_item < kValuesPerProgram;
         ++value_item) {
      const uint32_t value = value_base + value_item;
      state[key_item][value_item] =
          key < key_dimension && value < value_dimension
              ? temporal[temporal_head_base +
                         static_cast<uint64_t>(value) * key_dimension + key]
              : 0.0F;
    }
  }

  for (uint32_t step = 0; step < static_cast<uint32_t>(accept_length); ++step) {
    const uint64_t raw_key_base =
        static_cast<uint64_t>(layer) * raw_key_layer_stride +
        static_cast<uint64_t>(state_slot) * raw_key_slot_stride +
        (static_cast<uint64_t>(key_head) * replay_length + step) *
            key_dimension;
    float normalized_key[kKeysPerLane];
#pragma unroll
    for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
      const uint32_t key = lane + key_item * kWarpThreads;
      normalized_key[key_item] =
          key < key_dimension ? __bfloat162float(raw_keys[raw_key_base + key])
                              : 0.0F;
    }
    // Match Triton's four-element per-lane reduction tree before its
    // one-warp XOR butterfly.  The cached sm_120 oracle starts with K1*K1,
    // then folds K0, K2, and K3 through fused multiply-adds.
    float squared_norm =
        multiply_round_to_nearest(normalized_key[1], normalized_key[1]);
    squared_norm =
        __fmaf_rn(normalized_key[0], normalized_key[0], squared_norm);
    squared_norm =
        __fmaf_rn(normalized_key[2], normalized_key[2], squared_norm);
    squared_norm =
        __fmaf_rn(normalized_key[3], normalized_key[3], squared_norm);
    squared_norm = warp_butterfly_sum(squared_norm);
    const float norm =
        approximate_ftz_sqrt(add_round_to_nearest(squared_norm, 1.0e-6F));
#pragma unroll
    for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
      normalized_key[key_item] =
          full_precision_divide(normalized_key[key_item], norm);
    }
    const uint64_t raw_value_offset =
        static_cast<uint64_t>(layer) * raw_value_layer_stride +
        static_cast<uint64_t>(state_slot) * raw_value_slot_stride +
        (static_cast<uint64_t>(value_head) * replay_length + step) *
            value_dimension +
        local_value;
    float current_value =
        active_value ? __bfloat162float(raw_values[raw_value_offset]) : 0.0F;
    const uint64_t gate_offset =
        static_cast<uint64_t>(layer) * log_decay_layer_stride +
        static_cast<uint64_t>(state_slot) * log_decay_slot_stride +
        static_cast<uint64_t>(value_head) * replay_length + step;
    const float decay = approximate_exp(log_decay[gate_offset]);
#pragma unroll
    for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
#pragma unroll
      for (uint32_t value_item = 0; value_item < kValuesPerProgram;
           ++value_item) {
        state[key_item][value_item] =
            multiply_round_to_nearest(state[key_item][value_item], decay);
      }
    }
    float dot[kValuesPerProgram];
#pragma unroll
    for (uint32_t value_item = 0; value_item < kValuesPerProgram;
         ++value_item) {
      const float product0 =
          multiply_round_to_nearest(state[0][value_item], normalized_key[0]);
      const float product1 =
          multiply_round_to_nearest(state[1][value_item], normalized_key[1]);
      const float product2 =
          multiply_round_to_nearest(state[2][value_item], normalized_key[2]);
      const float product3 =
          multiply_round_to_nearest(state[3][value_item], normalized_key[3]);
      float local = add_round_to_nearest(product0, product1);
      local = add_round_to_nearest(product2, local);
      local = add_round_to_nearest(product3, local);
      dot[value_item] = warp_butterfly_sum(local);
    }
    current_value = subtract_round_to_nearest(current_value, dot[lane]);
    const uint64_t beta_offset =
        static_cast<uint64_t>(layer) * beta_layer_stride +
        static_cast<uint64_t>(state_slot) * beta_slot_stride +
        static_cast<uint64_t>(value_head) * replay_length + step;
    current_value = multiply_round_to_nearest(current_value, beta[beta_offset]);
    const float value_lane = current_value;
#pragma unroll
    for (uint32_t value_item = 0; value_item < kValuesPerProgram;
         ++value_item) {
      const float value_update =
          __shfl_sync(0xffffffffU, value_lane, value_item);
#pragma unroll
      for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
        state[key_item][value_item] =
            __fmaf_rn(normalized_key[key_item], value_update,
                      state[key_item][value_item]);
      }
    }
    if (track_slot >= 0 && static_cast<int32_t>(step) == track_step) {
      const uint64_t track_head_base =
          static_cast<uint64_t>(layer) * temporal_layer_stride +
          static_cast<uint64_t>(track_slot) * temporal_slot_stride +
          static_cast<uint64_t>(value_head) * value_dimension * key_dimension;
#pragma unroll
      for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
        const uint32_t key = lane + key_item * kWarpThreads;
#pragma unroll
        for (uint32_t value_item = 0; value_item < kValuesPerProgram;
             ++value_item) {
          const uint32_t value = value_base + value_item;
          if (key < key_dimension && value < value_dimension) {
            temporal[track_head_base +
                     static_cast<uint64_t>(value) * key_dimension + key] =
                state[key_item][value_item];
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t key_item = 0; key_item < kKeysPerLane; ++key_item) {
    const uint32_t key = lane + key_item * kWarpThreads;
#pragma unroll
    for (uint32_t value_item = 0; value_item < kValuesPerProgram;
         ++value_item) {
      const uint32_t value = value_base + value_item;
      if (key < key_dimension && value < value_dimension) {
        temporal[temporal_head_base +
                 static_cast<uint64_t>(value) * key_dimension + key] =
            state[key_item][value_item];
      }
    }
  }
}

__global__ void gdn_conv_scatter_kernel(
    __nv_bfloat16 *conv_states, const __nv_bfloat16 *windows,
    const int32_t *state_indices, const int32_t *last_correct_steps,
    const int32_t *track_indices, const int32_t *track_steps,
    const uint32_t *device_status, uint32_t num_layers, uint32_t num_slots,
    uint32_t batch_size, uint32_t replay_length, uint32_t conv_dimension,
    uint32_t conv_window, uint64_t conv_layer_stride, uint64_t conv_slot_stride,
    uint64_t windows_layer_stride, uint64_t windows_request_stride,
    uint64_t windows_step_stride) {
  if (device_status[0] != 0U)
    return;
  const uint64_t elements_per_entry =
      static_cast<uint64_t>(conv_dimension) * conv_window;
  const uint64_t element =
      static_cast<uint64_t>(blockIdx.x) * kCopyThreads + threadIdx.x;
  const uint32_t request = blockIdx.y;
  const uint32_t layer = blockIdx.z;
  if (element >= elements_per_entry || request >= batch_size ||
      layer >= num_layers) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(element / conv_window);
  const uint32_t window = static_cast<uint32_t>(element % conv_window);
  const int32_t state_destination = state_indices[request];
  const int32_t state_step = last_correct_steps[request];
  if (state_destination >= 0 && state_step >= 0) {
    const uint64_t source_offset =
        static_cast<uint64_t>(layer) * windows_layer_stride +
        static_cast<uint64_t>(request) * windows_request_stride +
        static_cast<uint64_t>(state_step) * windows_step_stride +
        static_cast<uint64_t>(dimension) * conv_window + window;
    const uint64_t destination_offset =
        static_cast<uint64_t>(layer) * conv_layer_stride +
        static_cast<uint64_t>(state_destination) * conv_slot_stride + element;
    conv_states[destination_offset] = windows[source_offset];
  }
  const int32_t track_destination = track_indices[request];
  const int32_t track_step = track_steps[request];
  if (track_destination >= 0 && track_step >= 0) {
    const uint64_t source_offset =
        static_cast<uint64_t>(layer) * windows_layer_stride +
        static_cast<uint64_t>(request) * windows_request_stride +
        static_cast<uint64_t>(track_step) * windows_step_stride +
        static_cast<uint64_t>(dimension) * conv_window + window;
    const uint64_t destination_offset =
        static_cast<uint64_t>(layer) * conv_layer_stride +
        static_cast<uint64_t>(track_destination) * conv_slot_stride + element;
    conv_states[destination_offset] = windows[source_offset];
  }
}

[[nodiscard]] NativeRuntimeError
launch_impl(const CudaExecutionContext &context,
            const GdnReplaySsmCommitBuffers &buffers,
            bool require_ready_status) noexcept {
  GdnReplaySsmShape shape{};
  const NativeRuntimeError validation =
      validate_gdn_replayssm_commit_buffers(context, buffers, &shape);
  if (!is_ok(validation))
    return validation;

  validate_content_kernel<<<1, 1, 0, context.stream()>>>(
      reinterpret_cast<const int32_t *>(buffers.state_indices.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.accept_lengths.data_bytes()),
      reinterpret_cast<const int32_t *>(
          buffers.last_correct_steps.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.track_indices.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.track_steps.data_bytes()),
      reinterpret_cast<uint32_t *>(buffers.device_status.data_bytes()),
      static_cast<uint32_t>(shape.batch_size),
      static_cast<uint32_t>(shape.num_slots),
      static_cast<uint32_t>(shape.replay_length), require_ready_status);
  cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchGdnReplaySsmCommit,
                      GdnReplaySsmArgument::kNone,
                      static_cast<int32_t>(launch_status));
  }

  const auto temporal_strides = buffers.temporal.strides();
  const auto raw_value_strides = buffers.raw_values.strides();
  const auto raw_key_strides = buffers.raw_keys.strides();
  const auto gate_strides = buffers.log_decay.strides();
  const auto beta_strides = buffers.beta.strides();
  const dim3 fold_grid(
      (static_cast<uint32_t>(shape.value_dimension) + kWarpThreads - 1) /
          kWarpThreads,
      static_cast<uint32_t>(shape.batch_size),
      static_cast<uint32_t>(shape.num_layers * shape.num_value_heads));
  gdn_replayssm_fold_kernel<<<fold_grid, kWarpThreads, 0, context.stream()>>>(
      reinterpret_cast<float *>(buffers.temporal.data_bytes()),
      reinterpret_cast<const __nv_bfloat16 *>(buffers.raw_values.data_bytes()),
      reinterpret_cast<const __nv_bfloat16 *>(buffers.raw_keys.data_bytes()),
      reinterpret_cast<const float *>(buffers.log_decay.data_bytes()),
      reinterpret_cast<const float *>(buffers.beta.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.state_indices.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.accept_lengths.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.track_indices.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.track_steps.data_bytes()),
      reinterpret_cast<const uint32_t *>(buffers.device_status.data_bytes()),
      static_cast<uint32_t>(shape.num_slots),
      static_cast<uint32_t>(shape.batch_size),
      static_cast<uint32_t>(shape.num_value_heads),
      static_cast<uint32_t>(shape.num_key_heads),
      static_cast<uint32_t>(shape.key_dimension),
      static_cast<uint32_t>(shape.value_dimension),
      static_cast<uint32_t>(shape.replay_length),
      static_cast<uint64_t>(temporal_strides[0]),
      static_cast<uint64_t>(temporal_strides[1]),
      static_cast<uint64_t>(raw_value_strides[0]),
      static_cast<uint64_t>(raw_value_strides[1]),
      static_cast<uint64_t>(raw_key_strides[0]),
      static_cast<uint64_t>(raw_key_strides[1]),
      static_cast<uint64_t>(gate_strides[0]),
      static_cast<uint64_t>(gate_strides[1]),
      static_cast<uint64_t>(beta_strides[0]),
      static_cast<uint64_t>(beta_strides[1]));
  launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchGdnReplaySsmCommit,
                      GdnReplaySsmArgument::kTemporal,
                      static_cast<int32_t>(launch_status));
  }

  for (const GdnReplaySsmConvPair &pair : buffers.conv_pairs) {
    const auto conv_extents = pair.conv_states.extents();
    const auto conv_strides = pair.conv_states.strides();
    const auto window_strides = pair.intermediate_conv_windows.strides();
    const uint64_t elements =
        static_cast<uint64_t>(conv_extents[2]) * conv_extents[3];
    const dim3 conv_grid(
        static_cast<uint32_t>((elements + kCopyThreads - 1) / kCopyThreads),
        static_cast<uint32_t>(shape.batch_size),
        static_cast<uint32_t>(shape.num_layers));
    gdn_conv_scatter_kernel<<<conv_grid, kCopyThreads, 0, context.stream()>>>(
        reinterpret_cast<__nv_bfloat16 *>(pair.conv_states.data_bytes()),
        reinterpret_cast<const __nv_bfloat16 *>(
            pair.intermediate_conv_windows.data_bytes()),
        reinterpret_cast<const int32_t *>(buffers.state_indices.data_bytes()),
        reinterpret_cast<const int32_t *>(
            buffers.last_correct_steps.data_bytes()),
        reinterpret_cast<const int32_t *>(buffers.track_indices.data_bytes()),
        reinterpret_cast<const int32_t *>(buffers.track_steps.data_bytes()),
        reinterpret_cast<const uint32_t *>(buffers.device_status.data_bytes()),
        static_cast<uint32_t>(shape.num_layers),
        static_cast<uint32_t>(shape.num_slots),
        static_cast<uint32_t>(shape.batch_size),
        static_cast<uint32_t>(shape.replay_length),
        static_cast<uint32_t>(conv_extents[2]),
        static_cast<uint32_t>(conv_extents[3]),
        static_cast<uint64_t>(conv_strides[0]),
        static_cast<uint64_t>(conv_strides[1]),
        static_cast<uint64_t>(window_strides[0]),
        static_cast<uint64_t>(window_strides[1]),
        static_cast<uint64_t>(window_strides[2]));
    launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
      return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                        NativeRuntimeOperation::kLaunchGdnReplaySsmCommit,
                        GdnReplaySsmArgument::kConvStates,
                        static_cast<int32_t>(launch_status));
    }
  }
  return native_runtime_ok();
}

struct ReplaySsmCapturePayload final {
  const CudaExecutionContext *context;
  const GdnReplaySsmCommitBuffers *buffers;
};

[[nodiscard]] NativeRuntimeError capture_replayssm_body(void *opaque) noexcept {
  const auto *payload = static_cast<const ReplaySsmCapturePayload *>(opaque);
  return launch_gdn_replayssm_commit_if_ready(*payload->context,
                                              *payload->buffers);
}

[[nodiscard]] constexpr NativeRuntimeError
capture_ownership_error(GdnReplaySsmArgument argument) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kForeignSlice,
                            NativeRuntimeOperation::kGraphCaptureBegin,
                            0,
                            static_cast<uint32_t>(argument),
                            0,
                            0};
}

} // namespace

NativeRuntimeResult<CudaCapturedGraph> capture_gdn_replayssm_commit_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const GdnReplaySsmCommitBuffers &buffers) noexcept {
  using GraphResult = NativeRuntimeResult<CudaCapturedGraph>;
  if (!context.valid() || !arena.valid()) {
    return GraphResult::failure(NativeRuntimeError{
        NativeRuntimeCode::kInvalidArgument,
        NativeRuntimeOperation::kGraphCaptureBegin, 0, 0, 0, 0});
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return GraphResult::failure(
        NativeRuntimeError{NativeRuntimeCode::kDeviceMismatch,
                           NativeRuntimeOperation::kGraphCaptureBegin, 0,
                           static_cast<uint32_t>(GdnReplaySsmArgument::kNone),
                           static_cast<uint64_t>(context.device_ordinal()),
                           static_cast<uint64_t>(arena.device_ordinal())});
  }
  const std::array<bool, 11> owned{buffers.temporal.belongs_to(arena),
                                   buffers.raw_values.belongs_to(arena),
                                   buffers.raw_keys.belongs_to(arena),
                                   buffers.log_decay.belongs_to(arena),
                                   buffers.beta.belongs_to(arena),
                                   buffers.state_indices.belongs_to(arena),
                                   buffers.accept_lengths.belongs_to(arena),
                                   buffers.last_correct_steps.belongs_to(arena),
                                   buffers.track_indices.belongs_to(arena),
                                   buffers.track_steps.belongs_to(arena),
                                   buffers.device_status.belongs_to(arena)};
  constexpr std::array<GdnReplaySsmArgument, 11> arguments{
      GdnReplaySsmArgument::kTemporal,
      GdnReplaySsmArgument::kRawValue,
      GdnReplaySsmArgument::kRawKey,
      GdnReplaySsmArgument::kLogDecay,
      GdnReplaySsmArgument::kBeta,
      GdnReplaySsmArgument::kStateIndices,
      GdnReplaySsmArgument::kAcceptLengths,
      GdnReplaySsmArgument::kLastCorrectSteps,
      GdnReplaySsmArgument::kTrackIndices,
      GdnReplaySsmArgument::kTrackSteps,
      GdnReplaySsmArgument::kDeviceStatus};
  for (std::size_t index = 0; index < owned.size(); ++index) {
    if (!owned[index]) {
      return GraphResult::failure(capture_ownership_error(arguments[index]));
    }
  }
  for (const GdnReplaySsmConvPair &pair : buffers.conv_pairs) {
    if (!pair.conv_states.belongs_to(arena)) {
      return GraphResult::failure(
          capture_ownership_error(GdnReplaySsmArgument::kConvStates));
    }
    if (!pair.intermediate_conv_windows.belongs_to(arena)) {
      return GraphResult::failure(
          capture_ownership_error(GdnReplaySsmArgument::kIntermediateConv));
    }
  }

  ReplaySsmCapturePayload payload{&context, &buffers};
  return CudaCapturedGraph::capture(context, arena, &capture_replayssm_body,
                                    &payload);
}

NativeRuntimeError
launch_gdn_replayssm_commit(const CudaExecutionContext &context,
                            const GdnReplaySsmCommitBuffers &buffers) noexcept {
  return launch_impl(context, buffers, false);
}

NativeRuntimeError
validate_gdn_replayssm_commit_buffers(const CudaExecutionContext &context,
                                      const GdnReplaySsmCommitBuffers &buffers,
                                      GdnReplaySsmShape *shape) noexcept {
  return validate_layout(context, buffers, shape);
}

NativeRuntimeError launch_gdn_replayssm_commit_if_ready(
    const CudaExecutionContext &context,
    const GdnReplaySsmCommitBuffers &buffers) noexcept {
  return launch_impl(context, buffers, true);
}

} // namespace sglang::native
