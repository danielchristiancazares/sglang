#include "sglang/native/dspark_target_verify.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace sglang::native {
namespace {

struct ByteRange final {
  uintptr_t begin;
  uintptr_t end;
  DsparkTargetVerifyArgument argument;
};

[[nodiscard]] constexpr NativeRuntimeError make_error(
    NativeRuntimeCode code, NativeRuntimeOperation operation,
    DsparkTargetVerifyArgument argument = DsparkTargetVerifyArgument::kNone,
    int32_t native_code = 0, uint64_t actual = 0,
    uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,        operation,
                            native_code, static_cast<uint32_t>(argument),
                            actual,      required};
}

[[nodiscard]] constexpr bool
has_flag(uint32_t flags, DsparkTargetVerifySamplingFlag flag) noexcept {
  return (flags & static_cast<uint32_t>(flag)) != 0U;
}

[[nodiscard]] bool checked_multiply(uint64_t left, uint64_t right,
                                    uint64_t *product) noexcept {
  if (product == nullptr ||
      (left != 0U && right > std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError
validate_tensor(const GraphStableTensorView<D, Rank, Access> &view,
                const CudaExecutionContext &context,
                const GraphArenaLease &cycle_arena,
                DsparkTargetVerifyArgument argument,
                const std::array<int64_t, Rank> &expected_extents,
                bool require_cycle_arena, ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkTargetVerify;
  if (!context.valid() || range == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (require_cycle_arena && !view.belongs_to(cycle_arena)) {
    return make_error(NativeRuntimeCode::kForeignSlice, kOperation, argument);
  }
  if (!view.is_row_major_contiguous()) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const std::span<const int64_t, Rank> extents = view.extents();
  uint64_t elements = 1U;
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension]) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, 0, static_cast<uint64_t>(extents[dimension]),
                        static_cast<uint64_t>(expected_extents[dimension]));
    }
    if (!checked_multiply(elements,
                          static_cast<uint64_t>(expected_extents[dimension]),
                          &elements)) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
  }
  constexpr uint64_t kElementBits = dtype_element_bits(D);
  uint64_t bits = 0U;
  if (!checked_multiply(elements, kElementBits, &bits) || bits % 8U != 0U) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  const uint64_t bytes = bits / 8U;
  if (view.allocation_bytes() != bytes) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, view.allocation_bytes(), bytes);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(view.data_bytes());
  if (begin == 0U || bytes > std::numeric_limits<uintptr_t>::max() ||
      begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + static_cast<uintptr_t>(bytes), argument};
  return native_runtime_ok();
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError
validate_strided_tensor(const GraphStableTensorView<D, Rank, Access> &view,
                        const CudaExecutionContext &context,
                        const GraphArenaLease &cycle_arena,
                        DsparkTargetVerifyArgument argument,
                        const std::array<int64_t, Rank> &expected_extents,
                        uint32_t first_contiguous_dimension,
                        bool require_cycle_arena, ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkTargetVerify;
  if (!context.valid() || range == nullptr ||
      first_contiguous_dimension >= Rank) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (require_cycle_arena && !view.belongs_to(cycle_arena)) {
    return make_error(NativeRuntimeCode::kForeignSlice, kOperation, argument);
  }
  const auto extents = view.extents();
  const auto strides = view.strides();
  uint64_t maximum_element = 0U;
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension] ||
        strides[dimension] <= 0) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, static_cast<int32_t>(dimension),
                        static_cast<uint64_t>(extents[dimension]),
                        static_cast<uint64_t>(expected_extents[dimension]));
    }
    uint64_t contribution = 0U;
    if (!checked_multiply(static_cast<uint64_t>(extents[dimension] - 1),
                          static_cast<uint64_t>(strides[dimension]),
                          &contribution) ||
        maximum_element > std::numeric_limits<uint64_t>::max() - contribution) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
    maximum_element += contribution;
  }
  uint64_t expected_stride = 1U;
  for (uint32_t reverse = Rank; reverse > first_contiguous_dimension;
       --reverse) {
    const uint32_t dimension = reverse - 1U;
    if (extents[dimension] > 1 &&
        static_cast<uint64_t>(strides[dimension]) != expected_stride) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, static_cast<int32_t>(dimension));
    }
    if (!checked_multiply(expected_stride,
                          static_cast<uint64_t>(extents[dimension]),
                          &expected_stride)) {
      return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                        argument);
    }
  }
  constexpr uint64_t kElementBytes = dtype_element_bits(D) / 8U;
  uint64_t origin_bytes = 0U;
  uint64_t required_bytes = 0U;
  if (!checked_multiply(view.storage_offset_elements(), kElementBytes,
                        &origin_bytes) ||
      !checked_multiply(maximum_element + 1U, kElementBytes, &required_bytes) ||
      origin_bytes > view.allocation_bytes() ||
      required_bytes > view.allocation_bytes() - origin_bytes) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(view.data_bytes());
  if (begin == 0U ||
      required_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
      begin > std::numeric_limits<uintptr_t>::max() - required_bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + static_cast<uintptr_t>(required_bytes),
                     argument};
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(const ByteRange &left,
                            const ByteRange &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

template <typename View, uint32_t Rank>
[[nodiscard]] NativeRuntimeError validate_optional(
    const View *view, uint32_t flags, DsparkTargetVerifySamplingFlag flag,
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    DsparkTargetVerifyArgument argument,
    const std::array<int64_t, Rank> &extents, ByteRange *range,
    bool *present) noexcept {
  const bool flag_present = has_flag(flags, flag);
  const bool pointer_present = view != nullptr;
  if (flag_present != pointer_present || present == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kValidateDsparkTargetVerify,
                      argument, 0, pointer_present, flag_present);
  }
  *present = pointer_present;
  return pointer_present ? validate_tensor(*view, context, cycle_arena,
                                           argument, extents, true, range)
                         : native_runtime_ok();
}

template <DType D>
[[nodiscard]] NativeRuntimeError validate_kv_layer(
    const GraphStableTensorView<D, 1, TensorAccess::kReadWrite> &view,
    const DsparkTargetVerifyKvPoolDescriptor &descriptor,
    const CudaExecutionContext &context,
    const DsparkModelGraphResourceLease &model_resources,
    DsparkTargetVerifyArgument argument, ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkTargetVerify;
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (view.owner_identity() != model_resources.model_storage_identity()) {
    return make_error(NativeRuntimeCode::kForeignSlice, kOperation, argument);
  }
  const uint64_t row_elements =
      static_cast<uint64_t>(descriptor.heads) * descriptor.head_dimension;
  if (descriptor.capacity_tokens >
      std::numeric_limits<uint64_t>::max() - descriptor.page_size) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  const uint64_t physical_capacity =
      descriptor.capacity_tokens + descriptor.page_size;
  uint64_t elements = 0U;
  if (!checked_multiply(physical_capacity, row_elements, &elements)) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  constexpr uint64_t element_bytes = dtype_element_bits(D) / 8U;
  uint64_t required_bytes = 0U;
  if (!checked_multiply(elements, element_bytes, &required_bytes) ||
      view.allocation_bytes() != required_bytes ||
      !view.is_row_major_contiguous() ||
      view.extents()[0] != static_cast<int64_t>(elements)) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, view.allocation_bytes(), required_bytes);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(view.data_bytes());
  if (begin == 0U ||
      required_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
      begin > std::numeric_limits<uintptr_t>::max() - required_bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + static_cast<uintptr_t>(required_bytes),
                     argument};
  return native_runtime_ok();
}

[[nodiscard]] NativeRuntimeError validate_kv_pool(
    const DsparkTargetVerifyKvPool &pool, const CudaExecutionContext &context,
    const DsparkModelGraphResourceLease &model_resources, uint32_t layers,
    uint32_t heads, uint32_t head_dimension,
    DsparkTargetVerifyArgument argument, std::span<ByteRange> ranges) noexcept {
  const DsparkTargetVerifyKvPoolKind kind =
      argument == DsparkTargetVerifyArgument::kTargetKvPool
          ? DsparkTargetVerifyKvPoolKind::kTarget
          : DsparkTargetVerifyKvPoolKind::kDraft;
  NativeRuntimeError status =
      validate_dspark_target_verify_kv_pool_descriptor(pool.descriptor, kind);
  if (!is_ok(status)) {
    return status;
  }
  if (pool.descriptor.layers != layers || pool.descriptor.heads != heads ||
      pool.descriptor.head_dimension != head_dimension) {
    return make_error(NativeRuntimeCode::kInvalidState,
                      NativeRuntimeOperation::kValidateDsparkTargetVerify,
                      argument);
  }
  status = validate_dspark_target_verify_kv_layer_ids(pool.layer_ids, kind);
  if (!is_ok(status)) {
    return status;
  }
  const bool fp8 =
      pool.descriptor.dtype == DsparkTargetVerifyKvDType::kFloat8E4M3Fn;
  const std::size_t expected_layers = pool.descriptor.layers;
  if ((fp8 && (pool.fp8_keys.size() != expected_layers ||
               pool.fp8_values.size() != expected_layers ||
               !pool.bfloat16_keys.empty() || !pool.bfloat16_values.empty())) ||
      (!fp8 && (pool.bfloat16_keys.size() != expected_layers ||
                pool.bfloat16_values.size() != expected_layers ||
                !pool.fp8_keys.empty() || !pool.fp8_values.empty())) ||
      ranges.size() < 2U * expected_layers) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kValidateDsparkTargetVerify,
                      argument);
  }
  for (std::size_t layer = 0; layer < expected_layers; ++layer) {
    const std::size_t key_index = 2U * layer;
    const std::size_t value_index = key_index + 1U;
    status =
        fp8 ? validate_kv_layer(pool.fp8_keys[layer], pool.descriptor, context,
                                model_resources, argument, &ranges[key_index])
            : validate_kv_layer(pool.bfloat16_keys[layer], pool.descriptor,
                                context, model_resources, argument,
                                &ranges[key_index]);
    if (!is_ok(status)) {
      return status;
    }
    status = fp8 ? validate_kv_layer(pool.fp8_values[layer], pool.descriptor,
                                     context, model_resources, argument,
                                     &ranges[value_index])
                 : validate_kv_layer(pool.bfloat16_values[layer],
                                     pool.descriptor, context, model_resources,
                                     argument, &ranges[value_index]);
    if (!is_ok(status)) {
      return status;
    }
    if (overlaps(ranges[key_index], ranges[value_index])) {
      return make_error(NativeRuntimeCode::kInvalidArgument,
                        NativeRuntimeOperation::kValidateDsparkTargetVerify,
                        argument, static_cast<int32_t>(layer));
    }
  }
  return native_runtime_ok();
}

[[nodiscard]] NativeRuntimeError
validate_non_aliasing(std::span<const ByteRange> ranges) noexcept {
  for (std::size_t left = 0; left < ranges.size(); ++left) {
    for (std::size_t right = left + 1; right < ranges.size(); ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return make_error(NativeRuntimeCode::kInvalidArgument,
                          NativeRuntimeOperation::kValidateDsparkTargetVerify,
                          ranges[left].argument, 0,
                          static_cast<uint64_t>(ranges[right].argument));
      }
    }
  }
  return native_runtime_ok();
}

struct TargetCapturePayload final {
  const CudaExecutionContext *context;
  const DsparkTargetVerifyPlan *plan;
  const DsparkTargetVerifyInputs *inputs;
  const DsparkTargetVerifySamplingBuffers *sampling;
  const DsparkTargetVerifyOutputs *outputs;
};

[[nodiscard]] NativeRuntimeError capture_target_body(void *opaque) noexcept {
  const auto *payload = static_cast<const TargetCapturePayload *>(opaque);
  const NativeRuntimeError launch = payload->plan->enqueue(
      *payload->context, *payload->inputs, *payload->sampling,
      *payload->outputs, payload->plan->implementation);
  if (!is_ok(launch)) {
    return launch;
  }
  return native_runtime_ok();
}

} // namespace

NativeRuntimeError validate_dspark_target_verify_buffers(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const DsparkModelGraphResourceLease &model_resources,
    const DsparkTargetVerifyInputs &inputs,
    const DsparkTargetVerifySamplingBuffers &sampling,
    const DsparkTargetVerifyOutputs &outputs) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkTargetVerify;
  if (!context.valid() || !cycle_arena.valid() || !model_resources.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation,
                      DsparkTargetVerifyArgument::kCycleArena);
  }
  if (context.device_ordinal() != cycle_arena.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      DsparkTargetVerifyArgument::kCycleArena, 0,
                      static_cast<uint64_t>(cycle_arena.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (model_resources.device_ordinal() != context.device_ordinal() ||
      model_resources.cycle_arena_identity() != cycle_arena.owner_identity() ||
      model_resources.model_storage_identity() == nullptr ||
      model_resources.model_storage_identity() ==
          cycle_arena.owner_identity()) {
    return make_error(NativeRuntimeCode::kForeignSlice, kOperation,
                      DsparkTargetVerifyArgument::kModelStorage);
  }
  int current_device = -1;
  const cudaError_t get_device = cudaGetDevice(&current_device);
  if (get_device != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
                      DsparkTargetVerifyArgument::kNone,
                      static_cast<int32_t>(get_device));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      DsparkTargetVerifyArgument::kNone, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(context.device_ordinal()));
  }

  NativeRuntimeError status =
      validate_dspark_target_verify_sampling_descriptor(sampling.descriptor);
  if (!is_ok(status)) {
    return status;
  }
  status = validate_dspark_target_verify_recurrent_pool_descriptor(
      outputs.recurrent);
  if (!is_ok(status)) {
    return status;
  }

  std::array<ByteRange, 20> cycle_ranges{};
  uint32_t cycle_count = 0U;
  std::array<ByteRange, 2U * (kDsparkTargetNumFullAttentionLayers +
                              kDsparkTargetNumDraftLayers) +
                            6U>
      model_ranges{};
  uint32_t model_count = 0U;
  ByteRange range{};
  const auto add_cycle = [&](NativeRuntimeError validation) noexcept -> bool {
    if (!is_ok(validation)) {
      return false;
    }
    cycle_ranges[cycle_count++] = range;
    return true;
  };
  const auto add_model = [&](NativeRuntimeError validation) noexcept -> bool {
    if (!is_ok(validation)) {
      return false;
    }
    model_ranges[model_count++] = range;
    return true;
  };
  const auto validate_model_tensor =
      [&](const auto &view, DsparkTargetVerifyArgument argument,
          const auto &extents, uint32_t first_contiguous_dimension) noexcept {
        NativeRuntimeError validation = validate_strided_tensor(
            view, context, cycle_arena, argument, extents,
            first_contiguous_dimension, false, &range);
        if (is_ok(validation) &&
            view.owner_identity() != model_resources.model_storage_identity()) {
          validation = make_error(NativeRuntimeCode::kForeignSlice, kOperation,
                                  argument);
        }
        return validation;
      };

  constexpr int64_t kWidth = kDsparkProductionNumVerifyTokens;
  constexpr int64_t kVocab = kDsparkProductionVocabSize;
  constexpr int64_t kPackedHidden = 25600;
  constexpr int64_t kGdnLayers = kDsparkTargetNumGdnLayers;
  const int64_t mamba_slots =
      static_cast<int64_t>(outputs.recurrent.mamba_cache_slots) + 1;
  const int64_t speculative_slots =
      static_cast<int64_t>(outputs.recurrent.speculative_request_slots) + 1;
  constexpr int64_t kValueHeads = kGdnProductionNumValueHeads;
  constexpr int64_t kKeyHeads = kGdnProductionNumKeyHeads;
  constexpr int64_t kValueDimension = kGdnProductionValueDimension;
  constexpr int64_t kKeyDimension = kGdnProductionKeyDimension;
  constexpr int64_t kConvWindow = kGdnProductionConvWindow;

  status = validate_tensor(inputs.input_tokens, context, cycle_arena,
                           DsparkTargetVerifyArgument::kInputTokens,
                           std::array<int64_t, 1>{kWidth}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(inputs.positions, context, cycle_arena,
                           DsparkTargetVerifyArgument::kPositions,
                           std::array<int64_t, 1>{kWidth}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(inputs.target_cache_locations, context, cycle_arena,
                           DsparkTargetVerifyArgument::kTargetCacheLocations,
                           std::array<int64_t, 1>{kWidth}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(inputs.target_sequence_lengths, context, cycle_arena,
                           DsparkTargetVerifyArgument::kTargetSequenceLengths,
                           std::array<int64_t, 1>{1}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(inputs.target_request_slots, context, cycle_arena,
                           DsparkTargetVerifyArgument::kTargetRequestSlots,
                           std::array<int64_t, 1>{1}, true, &range);
  if (!add_cycle(status))
    return status;

  bool present = false;
  status = validate_optional(
      sampling.additive_penalties, sampling.descriptor.flags,
      DsparkTargetVerifySamplingFlag::kAdditivePenalties, context, cycle_arena,
      DsparkTargetVerifyArgument::kAdditivePenalties,
      std::array<int64_t, 2>{1, kVocab}, &range, &present);
  if (!is_ok(status))
    return status;
  if (present)
    cycle_ranges[cycle_count++] = range;
  status = validate_optional(
      sampling.scaling_penalties, sampling.descriptor.flags,
      DsparkTargetVerifySamplingFlag::kScalingPenalties, context, cycle_arena,
      DsparkTargetVerifyArgument::kScalingPenalties,
      std::array<int64_t, 2>{1, kVocab}, &range, &present);
  if (!is_ok(status))
    return status;
  if (present)
    cycle_ranges[cycle_count++] = range;
  status = validate_optional(
      sampling.grammar_mask, sampling.descriptor.flags,
      DsparkTargetVerifySamplingFlag::kGrammarMask, context, cycle_arena,
      DsparkTargetVerifyArgument::kGrammarMask,
      std::array<int64_t, 2>{1, (kVocab + 31) / 32}, &range, &present);
  if (!is_ok(status))
    return status;
  if (present)
    cycle_ranges[cycle_count++] = range;
  status =
      validate_optional(sampling.logit_bias, sampling.descriptor.flags,
                        DsparkTargetVerifySamplingFlag::kLogitBias, context,
                        cycle_arena, DsparkTargetVerifyArgument::kLogitBias,
                        std::array<int64_t, 2>{1, kVocab}, &range, &present);
  if (!is_ok(status))
    return status;
  if (present)
    cycle_ranges[cycle_count++] = range;

  status =
      validate_tensor(outputs.target_probabilities, context, cycle_arena,
                      DsparkTargetVerifyArgument::kTargetProbabilities,
                      std::array<int64_t, 3>{1, kWidth, kVocab}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(outputs.target_hidden, context, cycle_arena,
                           DsparkTargetVerifyArgument::kTargetHidden,
                           std::array<int64_t, 3>{1, kWidth, kPackedHidden},
                           true, &range);
  if (!add_cycle(status))
    return status;
  const uint32_t target_range_begin = model_count;
  status = validate_kv_pool(
      outputs.kv.target, context, model_resources,
      kDsparkTargetNumFullAttentionLayers, kDsparkTargetNumKvHeads,
      kDsparkTargetAttentionHeadDimension,
      DsparkTargetVerifyArgument::kTargetKvPool,
      std::span<ByteRange>(model_ranges.data() + target_range_begin,
                           model_ranges.size() - target_range_begin));
  if (!is_ok(status))
    return status;
  model_count += 2U * kDsparkTargetNumFullAttentionLayers;
  const uint32_t draft_range_begin = model_count;
  status = validate_kv_pool(
      outputs.kv.draft, context, model_resources, kDsparkTargetNumDraftLayers,
      kDsparkTargetDraftNumKvHeads, kDsparkTargetDraftHeadDimension,
      DsparkTargetVerifyArgument::kDraftKvPool,
      std::span<ByteRange>(model_ranges.data() + draft_range_begin,
                           model_ranges.size() - draft_range_begin));
  if (!is_ok(status))
    return status;
  model_count += 2U * kDsparkTargetNumDraftLayers;
  status = validate_tensor(outputs.published_outputs, context, cycle_arena,
                           DsparkTargetVerifyArgument::kPublishedOutputs,
                           std::array<int64_t, 1>{1}, true, &range);
  if (!add_cycle(status))
    return status;
  status = validate_tensor(outputs.device_status, context, cycle_arena,
                           DsparkTargetVerifyArgument::kDeviceStatus,
                           std::array<int64_t, 1>{1}, true, &range);
  if (!add_cycle(status))
    return status;

  if (outputs.replay_raw_values.extents()[1] != mamba_slots ||
      outputs.replay_raw_keys.extents()[1] != mamba_slots ||
      outputs.replay_log_decay.extents()[1] != mamba_slots ||
      outputs.replay_beta.extents()[1] != mamba_slots ||
      outputs.replay_convolution.empty() ||
      outputs.replay_convolution.front().conv_states.extents()[1] !=
          mamba_slots ||
      outputs.replay_convolution.front()
              .intermediate_conv_windows.extents()[1] != speculative_slots) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      DsparkTargetVerifyArgument::kReplaySlots);
  }

  status = validate_model_tensor(
      outputs.replay_raw_values, DsparkTargetVerifyArgument::kReplayRawValues,
      std::array<int64_t, 5>{kGdnLayers, mamba_slots, kValueHeads, kWidth,
                             kValueDimension},
      2U);
  if (!add_model(status))
    return status;
  status = validate_model_tensor(
      outputs.replay_raw_keys, DsparkTargetVerifyArgument::kReplayRawKeys,
      std::array<int64_t, 5>{kGdnLayers, mamba_slots, kKeyHeads, kWidth,
                             kKeyDimension},
      2U);
  if (!add_model(status))
    return status;
  status = validate_model_tensor(
      outputs.replay_log_decay, DsparkTargetVerifyArgument::kReplayLogDecay,
      std::array<int64_t, 4>{kGdnLayers, mamba_slots, kValueHeads, kWidth}, 2U);
  if (!add_model(status))
    return status;
  status = validate_model_tensor(
      outputs.replay_beta, DsparkTargetVerifyArgument::kReplayBeta,
      std::array<int64_t, 4>{kGdnLayers, mamba_slots, kValueHeads, kWidth}, 2U);
  if (!add_model(status))
    return status;

  if (outputs.replay_convolution.size() != 1U) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      DsparkTargetVerifyArgument::kReplayConvWindows, 0,
                      outputs.replay_convolution.size(), 1U);
  }
  const DsparkTargetVerifyConvBuffers &convolution =
      outputs.replay_convolution.front();
  status = validate_model_tensor(
      convolution.conv_states, DsparkTargetVerifyArgument::kReplayConvWindows,
      std::array<int64_t, 4>{kGdnLayers, mamba_slots,
                             kDsparkTargetGdnConvolutionChannels, kConvWindow},
      2U);
  if (!add_model(status))
    return status;
  status = validate_model_tensor(
      convolution.intermediate_conv_windows,
      DsparkTargetVerifyArgument::kReplayConvWindows,
      std::array<int64_t, 5>{kGdnLayers, speculative_slots, kWidth,
                             kDsparkTargetGdnConvolutionChannels, kConvWindow},
      3U);
  if (!is_ok(status))
    return status;
  model_ranges[model_count++] = range;

  status = validate_non_aliasing(
      std::span<const ByteRange>(cycle_ranges.data(), cycle_count));
  if (!is_ok(status))
    return status;
  status = validate_non_aliasing(
      std::span<const ByteRange>(model_ranges.data(), model_count));
  if (!is_ok(status))
    return status;
  for (uint32_t cycle_index = 0; cycle_index < cycle_count; ++cycle_index) {
    for (uint32_t model_index = 0; model_index < model_count; ++model_index) {
      if (overlaps(cycle_ranges[cycle_index], model_ranges[model_index])) {
        return make_error(
            NativeRuntimeCode::kInvalidArgument, kOperation,
            cycle_ranges[cycle_index].argument, 0,
            static_cast<uint64_t>(model_ranges[model_index].argument));
      }
    }
  }
  return native_runtime_ok();
}

DsparkCapturedTargetVerifyGraph::DsparkCapturedTargetVerifyGraph(
    CudaCapturedGraph graph, DsparkModelGraphResourceLease resources) noexcept
    : graph_(std::move(graph)), resources_(std::move(resources)) {}

DsparkCapturedTargetVerifyGraph::DsparkCapturedTargetVerifyGraph(
    DsparkCapturedTargetVerifyGraph &&other) noexcept
    : graph_(std::move(other.graph_)), resources_(std::move(other.resources_)) {
}

bool DsparkCapturedTargetVerifyGraph::valid() const noexcept {
  return graph_.valid() && resources_.valid() &&
         graph_.retained_owner_identity() == resources_.resource_identity();
}

DsparkTargetVerifyGraphSource
DsparkCapturedTargetVerifyGraph::source() const noexcept {
  return DsparkTargetVerifyGraphSource{DsparkModelGraphBinding{
      DsparkModelGraphKind::kTargetVerify, 0U,
      dspark_required_model_graph_outputs(DsparkModelGraphKind::kTargetVerify),
      dspark_production_model_graph_shape(), graph_, resources_}};
}

NativeRuntimeError DsparkCapturedTargetVerifyGraph::close() noexcept {
  const NativeRuntimeError status = graph_.close();
  if (!is_ok(status)) {
    return status;
  }
  resources_ = {};
  return native_runtime_ok();
}

NativeRuntimeResult<DsparkCapturedTargetVerifyGraph>
capture_dspark_target_verify_graph(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const DsparkModelGraphResourceOwner &model_resources,
    const DsparkTargetVerifyPlan &plan, const DsparkTargetVerifyInputs &inputs,
    const DsparkTargetVerifySamplingBuffers &sampling,
    const DsparkTargetVerifyOutputs &outputs) noexcept {
  using SourceResult = NativeRuntimeResult<DsparkCapturedTargetVerifyGraph>;
  NativeRuntimeError status = validate_dspark_target_verify_plan(plan);
  if (!is_ok(status)) {
    return SourceResult::failure(status);
  }
  if (!model_resources.valid()) {
    return SourceResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kValidateDsparkTargetVerify,
                   DsparkTargetVerifyArgument::kResourceOwner));
  }
  DsparkModelGraphResourceLease lease = model_resources.acquire_lease();
  if (lease.device_ordinal() != context.device_ordinal()) {
    return SourceResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kValidateDsparkTargetVerify,
                   DsparkTargetVerifyArgument::kResourceOwner, 0,
                   static_cast<uint64_t>(lease.device_ordinal()),
                   static_cast<uint64_t>(context.device_ordinal())));
  }
  if (lease.cycle_arena_identity() != cycle_arena.owner_identity()) {
    return SourceResult::failure(
        make_error(NativeRuntimeCode::kForeignSlice,
                   NativeRuntimeOperation::kValidateDsparkTargetVerify,
                   DsparkTargetVerifyArgument::kCycleArena));
  }
  status = validate_dspark_target_verify_buffers(context, cycle_arena, lease,
                                                 inputs, sampling, outputs);
  if (!is_ok(status)) {
    return SourceResult::failure(status);
  }

  TargetCapturePayload payload{&context, &plan, &inputs, &sampling, &outputs};
  return std::move(CudaCapturedGraph::capture_retaining(
                       context, cycle_arena, lease.retain(),
                       &capture_target_body, &payload))
      .match(
          [&lease](CudaCapturedGraph &&graph) noexcept -> SourceResult {
            if (!graph.valid()) {
              return SourceResult::failure(
                  make_error(NativeRuntimeCode::kInvalidState,
                             NativeRuntimeOperation::kGraphCaptureEnd));
            }
            return SourceResult::success(
                DsparkCapturedTargetVerifyGraph(std::move(graph), lease));
          },
          [](NativeRuntimeError &&error) noexcept -> SourceResult {
            return SourceResult::failure(error);
          });
}

} // namespace sglang::native
