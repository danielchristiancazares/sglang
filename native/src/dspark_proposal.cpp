#include "sglang/native/dspark_proposal.hpp"

#include <limits>

namespace sglang::native {
namespace {

[[nodiscard]] constexpr NativeRuntimeError
proposal_error(DsparkProposalArgument argument, uint64_t actual,
               uint64_t required) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kValidateDsparkProposal,
                            0,
                            static_cast<uint32_t>(argument),
                            actual,
                            required};
}

} // namespace

std::string_view
dspark_lm_head_tactic_name(DsparkLmHeadTactic tactic) noexcept {
  switch (tactic) {
  case DsparkLmHeadTactic::kInvalid:
    return "invalid";
  case DsparkLmHeadTactic::kDense:
    return "dense";
  case DsparkLmHeadTactic::kPackedNvFp4:
    return "packed_nvfp4";
  default:
    return "invalid_dspark_lm_head_tactic";
  }
}

std::string_view
dspark_proposal_argument_name(DsparkProposalArgument argument) noexcept {
  switch (argument) {
  case DsparkProposalArgument::kNone:
    return "none";
  case DsparkProposalArgument::kBatchSize:
    return "batch_size";
  case DsparkProposalArgument::kGamma:
    return "gamma";
  case DsparkProposalArgument::kVocabSize:
    return "vocab_size";
  case DsparkProposalArgument::kMarkovRank:
    return "markov_rank";
  case DsparkProposalArgument::kNumVerifyTokens:
    return "num_verify_tokens";
  case DsparkProposalArgument::kLmHeadTactic:
    return "lm_head_tactic";
  case DsparkProposalArgument::kLogitsDType:
    return "logits_dtype";
  case DsparkProposalArgument::kLayerCapture:
    return "layer_capture";
  case DsparkProposalArgument::kBaseLogits:
    return "base_logits";
  case DsparkProposalArgument::kMarkovW1:
    return "markov_w1";
  case DsparkProposalArgument::kMarkovW2:
    return "markov_w2";
  case DsparkProposalArgument::kAnchorTokens:
    return "anchor_tokens";
  case DsparkProposalArgument::kTemperatures:
    return "temperatures";
  case DsparkProposalArgument::kGreedyMask:
    return "greedy_mask";
  case DsparkProposalArgument::kRngState:
    return "rng_state";
  case DsparkProposalArgument::kProposalTokens:
    return "proposal_tokens";
  case DsparkProposalArgument::kCorrectedLogits:
    return "corrected_logits";
  case DsparkProposalArgument::kLogNormalizers:
    return "log_normalizers";
  case DsparkProposalArgument::kDeviceStatus:
    return "device_status";
  default:
    return "invalid_dspark_proposal_argument";
  }
}

std::string_view
dspark_proposal_device_code_name(DsparkProposalDeviceCode code) noexcept {
  switch (code) {
  case DsparkProposalDeviceCode::kOk:
    return "ok";
  case DsparkProposalDeviceCode::kInvalidRngStateDescriptor:
    return "invalid_rng_state_descriptor";
  case DsparkProposalDeviceCode::kRngCounterOverflow:
    return "rng_counter_overflow";
  case DsparkProposalDeviceCode::kAnchorTokenOutOfRange:
    return "anchor_token_out_of_range";
  case DsparkProposalDeviceCode::kNonPositiveTemperature:
    return "non_positive_temperature";
  default:
    return "invalid_dspark_proposal_device_code";
  }
}

NativeRuntimeError
validate_dspark_proposal_shape(DsparkProposalShape shape) noexcept {
  if (shape.batch_size != kDsparkProductionBatchSize) {
    return proposal_error(DsparkProposalArgument::kBatchSize, shape.batch_size,
                          kDsparkProductionBatchSize);
  }
  if (shape.gamma != kDsparkProductionGamma) {
    return proposal_error(DsparkProposalArgument::kGamma, shape.gamma,
                          kDsparkProductionGamma);
  }
  if (shape.vocab_size != kDsparkProductionVocabSize) {
    return proposal_error(DsparkProposalArgument::kVocabSize, shape.vocab_size,
                          kDsparkProductionVocabSize);
  }
  if (shape.markov_rank != kDsparkProductionMarkovRank) {
    return proposal_error(DsparkProposalArgument::kMarkovRank,
                          shape.markov_rank, kDsparkProductionMarkovRank);
  }
  if (shape.num_verify_tokens != kDsparkProductionNumVerifyTokens) {
    return proposal_error(DsparkProposalArgument::kNumVerifyTokens,
                          shape.num_verify_tokens,
                          kDsparkProductionNumVerifyTokens);
  }
  return native_runtime_ok();
}

NativeRuntimeError validate_dspark_lm_head_descriptor(
    const DsparkLmHeadDescriptor &descriptor) noexcept {
  const bool valid_logits = descriptor.logits_dtype == DType::kBFloat16 ||
                            descriptor.logits_dtype == DType::kFloat16;
  if (!valid_logits) {
    return proposal_error(DsparkProposalArgument::kLogitsDType,
                          static_cast<uint64_t>(descriptor.logits_dtype),
                          static_cast<uint64_t>(DType::kBFloat16));
  }
  if (descriptor.output_features != kDsparkProductionVocabSize ||
      descriptor.input_features == 0) {
    return proposal_error(DsparkProposalArgument::kVocabSize,
                          descriptor.output_features,
                          kDsparkProductionVocabSize);
  }
  switch (descriptor.tactic) {
  case DsparkLmHeadTactic::kDense:
    if ((descriptor.weight_dtype != DType::kBFloat16 &&
         descriptor.weight_dtype != DType::kFloat16) ||
        descriptor.packed_elements_per_byte != 1) {
      return proposal_error(DsparkProposalArgument::kLmHeadTactic,
                            static_cast<uint64_t>(descriptor.weight_dtype), 1);
    }
    break;
  case DsparkLmHeadTactic::kPackedNvFp4:
    if (descriptor.weight_dtype != DType::kNvFp4E2M1 ||
        descriptor.packed_elements_per_byte != 2) {
      return proposal_error(DsparkProposalArgument::kLmHeadTactic,
                            descriptor.packed_elements_per_byte, 2);
    }
    break;
  default:
    return proposal_error(DsparkProposalArgument::kLmHeadTactic,
                          static_cast<uint64_t>(descriptor.tactic),
                          static_cast<uint64_t>(DsparkLmHeadTactic::kDense));
  }
  return native_runtime_ok();
}

NativeRuntimeError validate_dspark_layer_capture_descriptor(
    const DsparkLayerCaptureDescriptor &descriptor) noexcept {
  constexpr std::array<uint32_t, kDsparkProductionTargetLayerCount>
      kProductionLayers{5, 19, 33, 47, 61};
  if (descriptor.num_layers != kProductionLayers.size() ||
      descriptor.target_num_layers != 64) {
    return proposal_error(DsparkProposalArgument::kLayerCapture,
                          descriptor.num_layers, kProductionLayers.size());
  }
  for (uint32_t index = 0; index < descriptor.num_layers; ++index) {
    if (descriptor.layer_ids[index] != kProductionLayers[index]) {
      return proposal_error(DsparkProposalArgument::kLayerCapture,
                            descriptor.layer_ids[index],
                            kProductionLayers[index]);
    }
  }
  return native_runtime_ok();
}

uint64_t
dspark_proposal_rng_counter_blocks(DsparkProposalShape shape) noexcept {
  if (!is_ok(validate_dspark_proposal_shape(shape))) {
    return 0;
  }
  constexpr uint64_t kPhiloxWidth = 4;
  const uint64_t blocks_per_row =
      (shape.vocab_size + kPhiloxWidth - 1) / kPhiloxWidth;
  if (blocks_per_row > std::numeric_limits<uint64_t>::max() / shape.gamma) {
    return 0;
  }
  return shape.gamma * blocks_per_row;
}

} // namespace sglang::native
