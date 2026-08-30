#include "sglang/native/linear_rejection_sampling.hpp"

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
  LinearRejectionSamplingArgument argument;
};

struct KernelState final {
  uint32_t current_prob_row;
  uint32_t last_accept_out_index;
  uint32_t all_drafts_correct;
  uint32_t inputs_valid;
  float target_mass;
};

[[nodiscard]] constexpr NativeRuntimeError make_error(
    NativeRuntimeCode code, NativeRuntimeOperation operation,
    LinearRejectionSamplingArgument argument =
        LinearRejectionSamplingArgument::kNone,
    int32_t native_code = 0, uint64_t actual = 0,
    uint64_t required = 0) noexcept {
  return NativeRuntimeError{
      code,
      operation,
      native_code,
      static_cast<uint32_t>(argument),
      actual,
      required};
}

[[nodiscard]] bool checked_multiply(
    uint64_t left, uint64_t right, uint64_t* product) noexcept {
  if (product == nullptr ||
      (left != 0 &&
       right > std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] const std::byte* tensor_data(
    const GraphStableTensorView<D, Rank, Access>& view) noexcept {
  return view.data_bytes();
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError validate_tensor(
    const GraphStableTensorView<D, Rank, Access>& view,
    const CudaExecutionContext& context,
    LinearRejectionSamplingArgument argument,
    const std::array<int64_t, Rank>& expected_extents,
    uint64_t expected_bytes, ByteRange* range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearRejectionSampling;
  if (!context.valid()) {
    return make_error(
        NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(
        NativeRuntimeCode::kDeviceMismatch, kOperation, argument, 0,
        static_cast<uint64_t>(view.device_ordinal()),
        static_cast<uint64_t>(context.device_ordinal()));
  }
  if (!view.is_row_major_contiguous()) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation, argument);
  }
  const std::span<const int64_t, Rank> extents = view.extents();
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension]) {
      return make_error(
          NativeRuntimeCode::kInvalidArgument, kOperation, argument, 0,
          static_cast<uint64_t>(extents[dimension]),
          static_cast<uint64_t>(expected_extents[dimension]));
    }
  }
  if (view.allocation_bytes() != expected_bytes) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation, argument, 0,
        view.allocation_bytes(), expected_bytes);
  }

  const auto* const data = tensor_data(view);
  if (data == nullptr || range == nullptr ||
      expected_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation, argument);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t bytes = static_cast<uintptr_t>(expected_bytes);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return make_error(
        NativeRuntimeCode::kArithmeticOverflow, kOperation, argument);
  }
  *range = ByteRange{begin, begin + bytes, argument};
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(
    const ByteRange& left, const ByteRange& right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] NativeRuntimeError validate_layout(
    const CudaExecutionContext& context,
    const LinearRejectionSamplingBuffers& buffers,
    LinearRejectionSamplingShape* shape) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateLinearRejectionSampling;
  if (!context.valid() || shape == nullptr) {
    return make_error(
        NativeRuntimeCode::kInvalidState, kOperation);
  }

  int current_device = -1;
  const cudaError_t get_device = cudaGetDevice(&current_device);
  if (get_device != cudaSuccess) {
    return make_error(
        NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
        LinearRejectionSamplingArgument::kNone,
        static_cast<int32_t>(get_device));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(
        NativeRuntimeCode::kDeviceMismatch, kOperation,
        LinearRejectionSamplingArgument::kNone, 0,
        static_cast<uint64_t>(current_device),
        static_cast<uint64_t>(context.device_ordinal()));
  }

  const auto proposal_extents = buffers.proposal_tokens.extents();
  const auto target_extents = buffers.target_probs.extents();
  if (proposal_extents[0] != 1) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation,
        LinearRejectionSamplingArgument::kProposalTokens, 0,
        static_cast<uint64_t>(proposal_extents[0]), 1);
  }
  if (target_extents[0] != 1) {
    return make_error(
        NativeRuntimeCode::kInvalidArgument, kOperation,
        LinearRejectionSamplingArgument::kTargetProbs, 0,
        static_cast<uint64_t>(target_extents[0]), 1);
  }
  *shape = LinearRejectionSamplingShape{
      static_cast<uint64_t>(proposal_extents[1]),
      static_cast<uint64_t>(target_extents[2]),
      static_cast<uint64_t>(buffers.out_tokens.extents()[0])};
  const NativeRuntimeError shape_status =
      validate_linear_rejection_sampling_shape(*shape);
  if (!is_ok(shape_status)) {
    return shape_status;
  }

  const uint64_t num_slots = shape->num_slots;
  const uint64_t vocab_size = shape->vocab_size;
  const uint64_t num_draft_rows = num_slots - 1;
  uint64_t int32_slots_bytes = 0;
  uint64_t int64_slots_bytes = 0;
  uint64_t float_slots_bytes = 0;
  uint64_t target_elements = 0;
  uint64_t target_bytes = 0;
  uint64_t draft_elements = 0;
  uint64_t draft_bytes = 0;
  if (!checked_multiply(num_slots, sizeof(int32_t),
                        &int32_slots_bytes) ||
      !checked_multiply(num_slots, sizeof(int64_t),
                        &int64_slots_bytes) ||
      !checked_multiply(num_slots, sizeof(float),
                        &float_slots_bytes) ||
      !checked_multiply(num_slots, vocab_size, &target_elements) ||
      !checked_multiply(target_elements, sizeof(float), &target_bytes) ||
      !checked_multiply(num_draft_rows, vocab_size, &draft_elements) ||
      !checked_multiply(draft_elements, sizeof(float), &draft_bytes)) {
    return make_error(
        NativeRuntimeCode::kArithmeticOverflow, kOperation);
  }

  const int64_t slots = static_cast<int64_t>(num_slots);
  const int64_t draft_rows = static_cast<int64_t>(num_draft_rows);
  const int64_t vocab = static_cast<int64_t>(vocab_size);
  std::array<ByteRange, 10> ranges{};
  uint32_t range_index = 0;

  ByteRange range{};
  NativeRuntimeError status = validate_tensor(
      buffers.out_tokens, context,
      LinearRejectionSamplingArgument::kOutTokens,
      std::array<int64_t, 1>{slots}, int32_slots_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.accept_indices, context,
      LinearRejectionSamplingArgument::kAcceptIndices,
      std::array<int64_t, 2>{1, slots}, int32_slots_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.num_correct_drafts, context,
      LinearRejectionSamplingArgument::kNumCorrectDrafts,
      std::array<int64_t, 1>{1}, sizeof(int32_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.proposal_tokens, context,
      LinearRejectionSamplingArgument::kProposalTokens,
      std::array<int64_t, 2>{1, slots}, int64_slots_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.proposal_out_indices, context,
      LinearRejectionSamplingArgument::kProposalOutIndices,
      std::array<int64_t, 2>{1, slots}, int64_slots_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.accept_uniforms, context,
      LinearRejectionSamplingArgument::kAcceptUniforms,
      std::array<int64_t, 2>{1, slots}, float_slots_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.bonus_uniforms, context,
      LinearRejectionSamplingArgument::kBonusUniforms,
      std::array<int64_t, 1>{1}, sizeof(float), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.target_probs, context,
      LinearRejectionSamplingArgument::kTargetProbs,
      std::array<int64_t, 3>{1, slots, vocab}, target_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.draft_probs, context,
      LinearRejectionSamplingArgument::kDraftProbs,
      std::array<int64_t, 3>{1, draft_rows, vocab}, draft_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;
  status = validate_tensor(
      buffers.device_status, context,
      LinearRejectionSamplingArgument::kDeviceStatus,
      std::array<int64_t, 1>{1}, sizeof(uint32_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[range_index++] = range;

  for (uint32_t left = 0; left < range_index; ++left) {
    for (uint32_t right = left + 1; right < range_index; ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return make_error(
            NativeRuntimeCode::kInvalidArgument, kOperation,
            ranges[left].argument, 0,
            static_cast<uint64_t>(ranges[right].argument), 0);
      }
    }
  }
  return native_runtime_ok();
}

__device__ __forceinline__ float residual_mass(
    const float* target_probs, const float* draft_probs, uint64_t offset,
    bool all_drafts_correct) {
  const float target = target_probs[offset];
  if (all_drafts_correct) {
    return target;
  }
  const float raw_draft = draft_probs[offset];
  const float draft = raw_draft == raw_draft ? raw_draft : 0.0F;
  const float difference = target - draft;
  return difference > 0.0F ? difference : 0.0F;
}

__global__ void linear_rejection_sampling_kernel(
    int32_t* out_tokens, int32_t* accept_indices,
    int32_t* num_correct_drafts, const int64_t* proposal_tokens,
    const int64_t* proposal_out_indices, const float* accept_uniforms,
    const float* bonus_uniforms, const float* target_probs,
    const float* draft_probs, uint32_t* device_status,
    uint32_t num_slots, uint32_t vocab_size) {
  __shared__ KernelState state;
  __shared__ float thread_masses[kLinearRejectionSamplingThreads];
  __shared__ uint32_t thread_bonus_tokens[
      kLinearRejectionSamplingThreads];

  const uint32_t thread = threadIdx.x;
  if (thread == 0) {
    LinearRejectionSamplingDeviceCode code =
        LinearRejectionSamplingDeviceCode::kOk;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
      const int64_t out_index = proposal_out_indices[slot];
      if (out_index < 0 ||
          out_index >= static_cast<int64_t>(num_slots)) {
        code = LinearRejectionSamplingDeviceCode::
            kProposalOutIndexOutOfRange;
        break;
      }
      for (uint32_t prior = 0; prior < slot; ++prior) {
        if (proposal_out_indices[prior] == out_index) {
          code = LinearRejectionSamplingDeviceCode::
              kDuplicateProposalOutIndex;
          break;
        }
      }
      if (code != LinearRejectionSamplingDeviceCode::kOk) {
        break;
      }
      const int64_t proposal_token = proposal_tokens[slot];
      if (proposal_token < 0 ||
          proposal_token >= static_cast<int64_t>(vocab_size)) {
        code =
            LinearRejectionSamplingDeviceCode::kProposalTokenOutOfRange;
        break;
      }
    }

    device_status[0] = static_cast<uint32_t>(code);
    state.inputs_valid =
        code == LinearRejectionSamplingDeviceCode::kOk ? 1U : 0U;
    if (state.inputs_valid != 0U) {
      uint32_t current_prob_row = 0;
      uint32_t num_correct = 0;
      uint32_t last_accept_out_index =
          static_cast<uint32_t>(proposal_out_indices[0]);
      bool all_drafts_correct = true;
      accept_indices[0] =
          static_cast<int32_t>(last_accept_out_index);

      for (uint32_t step = 1; step < num_slots; ++step) {
        const uint32_t proposal_token =
            static_cast<uint32_t>(proposal_tokens[step]);
        const uint64_t prob_offset =
            static_cast<uint64_t>(current_prob_row) * vocab_size +
            proposal_token;
        const float target = target_probs[prob_offset];
        const float draft = draft_probs[prob_offset];
        const float coin = accept_uniforms[step - 1];
        if (coin * draft < target) {
          out_tokens[last_accept_out_index] =
              static_cast<int32_t>(proposal_token);
          ++num_correct;
          current_prob_row = step;
          last_accept_out_index =
              static_cast<uint32_t>(proposal_out_indices[step]);
          accept_indices[num_correct] =
              static_cast<int32_t>(last_accept_out_index);
        } else {
          all_drafts_correct = false;
          break;
        }
      }
      num_correct_drafts[0] = static_cast<int32_t>(num_correct);
      state.current_prob_row = current_prob_row;
      state.last_accept_out_index = last_accept_out_index;
      state.all_drafts_correct = all_drafts_correct ? 1U : 0U;
    }
  }
  __syncthreads();

  if (state.inputs_valid == 0U) {
    return;
  }

  const uint64_t segment_size =
      (static_cast<uint64_t>(vocab_size) +
       kLinearRejectionSamplingThreads - 1) /
      kLinearRejectionSamplingThreads;
  const uint64_t begin = static_cast<uint64_t>(thread) * segment_size;
  uint64_t end = begin + segment_size;
  if (end > vocab_size) {
    end = vocab_size;
  }
  const uint64_t target_row_offset =
      static_cast<uint64_t>(state.current_prob_row) * vocab_size;
  // On full accept this is the one-past draft row; residual_mass returns
  // target p before dereferencing it.
  const uint64_t draft_row_offset =
      static_cast<uint64_t>(state.current_prob_row) * vocab_size;
  const bool all_drafts_correct = state.all_drafts_correct != 0U;

  float local_mass = 0.0F;
  for (uint64_t token = begin; token < end; ++token) {
    local_mass += residual_mass(
        target_probs + target_row_offset,
        draft_probs + draft_row_offset, token, all_drafts_correct);
  }
  thread_masses[thread] = local_mass;
  __syncthreads();

  if (thread == 0) {
    float total_mass = 0.0F;
    for (uint32_t index = 0;
         index < kLinearRejectionSamplingThreads; ++index) {
      const float partition_mass = thread_masses[index];
      thread_masses[index] = total_mass;
      total_mass += partition_mass;
    }
    state.target_mass = bonus_uniforms[0] * total_mass;
  }
  __syncthreads();

  float cumulative_mass = thread_masses[thread];
  uint32_t bonus_token = vocab_size - 1;
  for (uint64_t token = begin; token < end; ++token) {
    cumulative_mass += residual_mass(
        target_probs + target_row_offset,
        draft_probs + draft_row_offset, token, all_drafts_correct);
    if (cumulative_mass > state.target_mass) {
      bonus_token = static_cast<uint32_t>(token);
      break;
    }
  }
  thread_bonus_tokens[thread] = bonus_token;
  __syncthreads();

  if (thread == 0) {
    uint32_t first_bonus_token = vocab_size - 1;
    for (uint32_t index = 0;
         index < kLinearRejectionSamplingThreads; ++index) {
      if (thread_bonus_tokens[index] < first_bonus_token) {
        first_bonus_token = thread_bonus_tokens[index];
      }
    }
    out_tokens[state.last_accept_out_index] =
        static_cast<int32_t>(first_bonus_token);
  }
}

}  // namespace

NativeRuntimeError launch_linear_rejection_sampling(
    const CudaExecutionContext& context,
    const LinearRejectionSamplingBuffers& buffers) noexcept {
  LinearRejectionSamplingShape shape{};
  const NativeRuntimeError layout_status =
      validate_layout(context, buffers, &shape);
  if (!is_ok(layout_status)) {
    return layout_status;
  }

  linear_rejection_sampling_kernel
      <<<1, kLinearRejectionSamplingThreads, 0, context.stream()>>>(
          reinterpret_cast<int32_t*>(buffers.out_tokens.data_bytes()),
          reinterpret_cast<int32_t*>(buffers.accept_indices.data_bytes()),
          reinterpret_cast<int32_t*>(
              buffers.num_correct_drafts.data_bytes()),
          reinterpret_cast<const int64_t*>(
              buffers.proposal_tokens.data_bytes()),
          reinterpret_cast<const int64_t*>(
              buffers.proposal_out_indices.data_bytes()),
          reinterpret_cast<const float*>(
              buffers.accept_uniforms.data_bytes()),
          reinterpret_cast<const float*>(
              buffers.bonus_uniforms.data_bytes()),
          reinterpret_cast<const float*>(
              buffers.target_probs.data_bytes()),
          reinterpret_cast<const float*>(
              buffers.draft_probs.data_bytes()),
          reinterpret_cast<uint32_t*>(
              buffers.device_status.data_bytes()),
          static_cast<uint32_t>(shape.num_slots),
          static_cast<uint32_t>(shape.vocab_size));
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(
        NativeRuntimeCode::kCudaRuntimeFailure,
        NativeRuntimeOperation::kLaunchLinearRejectionSampling,
        LinearRejectionSamplingArgument::kNone,
        static_cast<int32_t>(launch_status));
  }
  return native_runtime_ok();
}

}  // namespace sglang::native
