#include "sglang/native/linear_rejection_sampling.hpp"

#include <limits>

namespace sglang::native {
namespace {

[[nodiscard]] constexpr NativeRuntimeError shape_error(
    LinearRejectionSamplingArgument argument, uint64_t actual,
    uint64_t required) noexcept {
  return NativeRuntimeError{
      NativeRuntimeCode::kInvalidArgument,
      NativeRuntimeOperation::kValidateLinearRejectionSampling,
      0,
      static_cast<uint32_t>(argument),
      actual,
      required};
}

}  // namespace

std::string_view linear_rejection_sampling_argument_name(
    LinearRejectionSamplingArgument argument) noexcept {
  switch (argument) {
    case LinearRejectionSamplingArgument::kNone:
      return "none";
    case LinearRejectionSamplingArgument::kNumSlots:
      return "num_slots";
    case LinearRejectionSamplingArgument::kVocabSize:
      return "vocab_size";
    case LinearRejectionSamplingArgument::kOutTokens:
      return "out_tokens";
    case LinearRejectionSamplingArgument::kAcceptIndices:
      return "accept_indices";
    case LinearRejectionSamplingArgument::kNumCorrectDrafts:
      return "num_correct_drafts";
    case LinearRejectionSamplingArgument::kProposalTokens:
      return "proposal_tokens";
    case LinearRejectionSamplingArgument::kProposalOutIndices:
      return "proposal_out_indices";
    case LinearRejectionSamplingArgument::kAcceptUniforms:
      return "accept_uniforms";
    case LinearRejectionSamplingArgument::kBonusUniforms:
      return "bonus_uniforms";
    case LinearRejectionSamplingArgument::kTargetProbs:
      return "target_probs";
    case LinearRejectionSamplingArgument::kDraftProbs:
      return "draft_probs";
    case LinearRejectionSamplingArgument::kDeviceStatus:
      return "device_status";
    default:
      return "invalid_linear_rejection_sampling_argument";
  }
}

std::string_view linear_rejection_sampling_device_code_name(
    LinearRejectionSamplingDeviceCode code) noexcept {
  switch (code) {
    case LinearRejectionSamplingDeviceCode::kOk:
      return "ok";
    case LinearRejectionSamplingDeviceCode::kProposalOutIndexOutOfRange:
      return "proposal_out_index_out_of_range";
    case LinearRejectionSamplingDeviceCode::kDuplicateProposalOutIndex:
      return "duplicate_proposal_out_index";
    case LinearRejectionSamplingDeviceCode::kProposalTokenOutOfRange:
      return "proposal_token_out_of_range";
    default:
      return "invalid_linear_rejection_sampling_device_code";
  }
}

NativeRuntimeError validate_linear_rejection_sampling_shape(
    LinearRejectionSamplingShape shape) noexcept {
  if (shape.num_slots < 2) {
    return shape_error(
        LinearRejectionSamplingArgument::kNumSlots, shape.num_slots, 2);
  }
  if (shape.num_slots > kLinearRejectionSamplingMaxNumSlots) {
    return shape_error(
        LinearRejectionSamplingArgument::kNumSlots, shape.num_slots,
        kLinearRejectionSamplingMaxNumSlots);
  }
  if (shape.vocab_size == 0) {
    return shape_error(
        LinearRejectionSamplingArgument::kVocabSize, shape.vocab_size, 1);
  }
  if (shape.vocab_size >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return shape_error(
        LinearRejectionSamplingArgument::kVocabSize, shape.vocab_size,
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
  }
  if (shape.num_out_tokens != shape.num_slots) {
    return shape_error(
        LinearRejectionSamplingArgument::kOutTokens,
        shape.num_out_tokens, shape.num_slots);
  }
  return native_runtime_ok();
}

}  // namespace sglang::native
