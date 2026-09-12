#include "sglang/native/dspark_target_verify.hpp"

#include <bit>
#include <limits>

namespace sglang::native {
namespace {

constexpr uint32_t kKnownSamplingFlags =
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kTopK) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kTopP) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kMinP) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kAdditivePenalties) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kScalingPenalties) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kGrammarMask) |
    static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kLogitBias);

[[nodiscard]] constexpr NativeRuntimeError
target_error(DsparkTargetVerifyArgument argument, uint64_t actual = 0,
             uint64_t required = 0) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kValidateDsparkTargetVerify,
                            0,
                            static_cast<uint32_t>(argument),
                            actual,
                            required};
}

[[nodiscard]] constexpr uint64_t float_bits(float value) noexcept {
  return std::bit_cast<uint32_t>(value);
}

[[nodiscard]] constexpr bool is_finite(float value) noexcept {
  return (std::bit_cast<uint32_t>(value) & 0x7f800000U) != 0x7f800000U;
}

[[nodiscard]] constexpr bool
same_shape(const DsparkTargetVerifyShape &left,
           const DsparkTargetVerifyShape &right) noexcept {
  return left.batch_size == right.batch_size && left.gamma == right.gamma &&
         left.verify_width == right.verify_width &&
         left.vocabulary_size == right.vocabulary_size &&
         left.hidden_size == right.hidden_size &&
         left.target_layers == right.target_layers &&
         left.full_attention_layers == right.full_attention_layers &&
         left.gdn_layers == right.gdn_layers &&
         left.attention_heads == right.attention_heads &&
         left.attention_kv_heads == right.attention_kv_heads &&
         left.attention_head_dimension == right.attention_head_dimension &&
         left.gdn_key_heads == right.gdn_key_heads &&
         left.gdn_value_heads == right.gdn_value_heads &&
         left.gdn_key_dimension == right.gdn_key_dimension &&
         left.gdn_value_dimension == right.gdn_value_dimension &&
         left.convolution_window == right.convolution_window &&
         left.captured_target_layers == right.captured_target_layers &&
         left.packed_target_hidden_width == right.packed_target_hidden_width &&
         left.draft_layers == right.draft_layers &&
         left.draft_kv_heads == right.draft_kv_heads &&
         left.draft_head_dimension == right.draft_head_dimension &&
         left.reserved == 0U;
}

[[nodiscard]] constexpr bool
has_flag(uint32_t flags, DsparkTargetVerifySamplingFlag flag) noexcept {
  return (flags & static_cast<uint32_t>(flag)) != 0U;
}

} // namespace

std::string_view dspark_target_verify_argument_name(
    DsparkTargetVerifyArgument argument) noexcept {
  switch (argument) {
  case DsparkTargetVerifyArgument::kNone:
    return "none";
  case DsparkTargetVerifyArgument::kContractVersion:
    return "contract_version";
  case DsparkTargetVerifyArgument::kShape:
    return "shape";
  case DsparkTargetVerifyArgument::kCaptureLayers:
    return "capture_layers";
  case DsparkTargetVerifyArgument::kKvCommitMode:
    return "kv_commit_mode";
  case DsparkTargetVerifyArgument::kSamplingFlags:
    return "sampling_flags";
  case DsparkTargetVerifyArgument::kTopK:
    return "top_k";
  case DsparkTargetVerifyArgument::kTopP:
    return "top_p";
  case DsparkTargetVerifyArgument::kMinP:
    return "min_p";
  case DsparkTargetVerifyArgument::kTemperature:
    return "temperature";
  case DsparkTargetVerifyArgument::kInputTokens:
    return "input_tokens";
  case DsparkTargetVerifyArgument::kPositions:
    return "positions";
  case DsparkTargetVerifyArgument::kTargetCacheLocations:
    return "target_cache_locations";
  case DsparkTargetVerifyArgument::kTargetSequenceLengths:
    return "target_sequence_lengths";
  case DsparkTargetVerifyArgument::kTargetRequestSlots:
    return "target_request_slots";
  case DsparkTargetVerifyArgument::kAdditivePenalties:
    return "additive_penalties";
  case DsparkTargetVerifyArgument::kScalingPenalties:
    return "scaling_penalties";
  case DsparkTargetVerifyArgument::kGrammarMask:
    return "grammar_mask";
  case DsparkTargetVerifyArgument::kLogitBias:
    return "logit_bias";
  case DsparkTargetVerifyArgument::kTargetProbabilities:
    return "target_probabilities";
  case DsparkTargetVerifyArgument::kTargetHidden:
    return "target_hidden";
  case DsparkTargetVerifyArgument::kTargetKvPublication:
    return "target_kv_publication";
  case DsparkTargetVerifyArgument::kDraftKvPublication:
    return "draft_kv_prewrite_publication";
  case DsparkTargetVerifyArgument::kReplayRawValues:
    return "replay_raw_values";
  case DsparkTargetVerifyArgument::kReplayRawKeys:
    return "replay_raw_keys";
  case DsparkTargetVerifyArgument::kReplayLogDecay:
    return "replay_log_decay";
  case DsparkTargetVerifyArgument::kReplayBeta:
    return "replay_beta";
  case DsparkTargetVerifyArgument::kReplayConvWindows:
    return "replay_convolution_windows";
  case DsparkTargetVerifyArgument::kPublishedOutputs:
    return "published_outputs";
  case DsparkTargetVerifyArgument::kDeviceStatus:
    return "device_status";
  case DsparkTargetVerifyArgument::kCycleArena:
    return "cycle_arena";
  case DsparkTargetVerifyArgument::kResourceOwner:
    return "resource_owner";
  case DsparkTargetVerifyArgument::kEnqueue:
    return "enqueue";
  case DsparkTargetVerifyArgument::kModelStorage:
    return "model_storage";
  case DsparkTargetVerifyArgument::kReplaySlots:
    return "replay_slots";
  case DsparkTargetVerifyArgument::kTargetKvPool:
    return "target_kv_pool";
  case DsparkTargetVerifyArgument::kDraftKvPool:
    return "draft_kv_pool";
  case DsparkTargetVerifyArgument::kPageSize:
    return "page_size";
  case DsparkTargetVerifyArgument::kCacheCapacity:
    return "cache_capacity";
  default:
    return "invalid_dspark_target_verify_argument";
  }
}

std::string_view dspark_target_verify_kv_commit_mode_name(
    DsparkTargetVerifyKvCommitMode mode) noexcept {
  switch (mode) {
  case DsparkTargetVerifyKvCommitMode::kInvalid:
    return "invalid";
  case DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite:
    return "speculative_prewrite";
  default:
    return "invalid_dspark_target_verify_kv_commit_mode";
  }
}

std::string_view dspark_target_verify_device_code_name(
    DsparkTargetVerifyDeviceCode code) noexcept {
  switch (code) {
  case DsparkTargetVerifyDeviceCode::kOk:
    return "ok";
  case DsparkTargetVerifyDeviceCode::kInvalidInputToken:
    return "invalid_input_token";
  case DsparkTargetVerifyDeviceCode::kInvalidPosition:
    return "invalid_position";
  case DsparkTargetVerifyDeviceCode::kInvalidTargetCacheLocation:
    return "invalid_target_cache_location";
  case DsparkTargetVerifyDeviceCode::kInvalidTargetRequestSlot:
    return "invalid_target_request_slot";
  case DsparkTargetVerifyDeviceCode::kInvalidSamplingControl:
    return "invalid_sampling_control";
  default:
    return "invalid_dspark_target_verify_device_code";
  }
}

NativeRuntimeError validate_dspark_target_verify_plan(
    const DsparkTargetVerifyPlan &plan) noexcept {
  if (plan.contract_version != kDsparkTargetVerifyContractVersion) {
    return target_error(DsparkTargetVerifyArgument::kContractVersion,
                        plan.contract_version,
                        kDsparkTargetVerifyContractVersion);
  }
  if (!same_shape(plan.shape, dspark_production_target_verify_shape())) {
    return target_error(DsparkTargetVerifyArgument::kShape);
  }
  const NativeRuntimeError capture_status =
      validate_dspark_layer_capture_descriptor(plan.capture);
  if (!is_ok(capture_status)) {
    return target_error(DsparkTargetVerifyArgument::kCaptureLayers);
  }
  if (plan.kv_commit_mode !=
      DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite) {
    return target_error(
        DsparkTargetVerifyArgument::kKvCommitMode,
        static_cast<uint64_t>(plan.kv_commit_mode),
        static_cast<uint64_t>(
            DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite));
  }
  if (plan.enqueue == nullptr) {
    return target_error(DsparkTargetVerifyArgument::kEnqueue);
  }
  if (plan.implementation == nullptr) {
    return target_error(DsparkTargetVerifyArgument::kResourceOwner);
  }
  return native_runtime_ok();
}

NativeRuntimeError validate_dspark_target_verify_sampling_descriptor(
    const DsparkTargetVerifySamplingDescriptor &descriptor) noexcept {
  if ((descriptor.flags & ~kKnownSamplingFlags) != 0U ||
      descriptor.reserved != 0U) {
    return target_error(DsparkTargetVerifyArgument::kSamplingFlags,
                        descriptor.flags, kKnownSamplingFlags);
  }
  if (!is_finite(descriptor.temperature) || descriptor.temperature <= 0.0F) {
    return target_error(DsparkTargetVerifyArgument::kTemperature,
                        float_bits(descriptor.temperature), float_bits(1.0F));
  }
  if (has_flag(descriptor.flags, DsparkTargetVerifySamplingFlag::kTopK)) {
    if (descriptor.top_k == 0U || descriptor.top_k > kDsparkTargetTopKLimit ||
        descriptor.top_k > kDsparkProductionVocabSize) {
      return target_error(DsparkTargetVerifyArgument::kTopK, descriptor.top_k,
                          kDsparkTargetTopKLimit);
    }
  } else if (descriptor.top_k != kDsparkProductionVocabSize) {
    return target_error(DsparkTargetVerifyArgument::kTopK, descriptor.top_k,
                        kDsparkProductionVocabSize);
  }
  if (!is_finite(descriptor.top_p) || descriptor.top_p <= 0.0F ||
      descriptor.top_p > 1.0F ||
      (!has_flag(descriptor.flags, DsparkTargetVerifySamplingFlag::kTopP) &&
       descriptor.top_p != 1.0F)) {
    return target_error(DsparkTargetVerifyArgument::kTopP,
                        float_bits(descriptor.top_p), float_bits(1.0F));
  }
  if (!is_finite(descriptor.min_p) || descriptor.min_p < 0.0F ||
      descriptor.min_p > 1.0F ||
      (!has_flag(descriptor.flags, DsparkTargetVerifySamplingFlag::kMinP) &&
       descriptor.min_p != 0.0F)) {
    return target_error(DsparkTargetVerifyArgument::kMinP,
                        float_bits(descriptor.min_p), float_bits(0.0F));
  }
  return native_runtime_ok();
}

NativeRuntimeError validate_dspark_target_verify_kv_pool_descriptor(
    const DsparkTargetVerifyKvPoolDescriptor &descriptor,
    DsparkTargetVerifyKvPoolKind kind) noexcept {
  const bool target = kind == DsparkTargetVerifyKvPoolKind::kTarget;
  const bool draft = kind == DsparkTargetVerifyKvPoolKind::kDraft;
  const DsparkTargetVerifyArgument argument =
      draft ? DsparkTargetVerifyArgument::kDraftKvPool
            : DsparkTargetVerifyArgument::kTargetKvPool;
  if (!target && !draft) {
    return target_error(argument, static_cast<uint64_t>(kind));
  }
  const uint32_t layers = target ? kDsparkTargetNumFullAttentionLayers
                                 : kDsparkTargetNumDraftLayers;
  const uint32_t heads =
      target ? kDsparkTargetNumKvHeads : kDsparkTargetDraftNumKvHeads;
  const uint32_t head_dimension = target ? kDsparkTargetAttentionHeadDimension
                                         : kDsparkTargetDraftHeadDimension;
  if (descriptor.layers != layers || descriptor.heads != heads ||
      descriptor.head_dimension != head_dimension) {
    return target_error(argument);
  }
  if (descriptor.page_size != kDsparkTargetProductionKvPageSize) {
    return target_error(DsparkTargetVerifyArgument::kPageSize,
                        descriptor.page_size,
                        kDsparkTargetProductionKvPageSize);
  }
  if (descriptor.capacity_tokens != kDsparkTargetProductionKvCapacity) {
    return target_error(DsparkTargetVerifyArgument::kCacheCapacity,
                        descriptor.capacity_tokens,
                        kDsparkTargetProductionKvCapacity);
  }
  if (descriptor.dtype != DsparkTargetVerifyKvDType::kFloat8E4M3Fn &&
      descriptor.dtype != DsparkTargetVerifyKvDType::kBFloat16) {
    return target_error(argument, static_cast<uint64_t>(descriptor.dtype));
  }
  if (descriptor.layout != DsparkTargetVerifyKvLayout::kNhd &&
      descriptor.layout != DsparkTargetVerifyKvLayout::kHnd &&
      descriptor.layout != DsparkTargetVerifyKvLayout::kPageMajorLayerMajor) {
    return target_error(argument, static_cast<uint64_t>(descriptor.layout));
  }
  return native_runtime_ok();
}

NativeRuntimeResult<uint64_t> dspark_target_verify_kv_element_offset(
    const DsparkTargetVerifyKvPoolDescriptor &descriptor,
    DsparkTargetVerifyKvPoolKind kind, uint64_t token_location, uint32_t head,
    uint32_t dimension) noexcept {
  using OffsetResult = NativeRuntimeResult<uint64_t>;
  const NativeRuntimeError validation =
      validate_dspark_target_verify_kv_pool_descriptor(descriptor, kind);
  if (!is_ok(validation)) {
    return OffsetResult::failure(validation);
  }
  const uint64_t physical_capacity =
      descriptor.capacity_tokens + descriptor.page_size;
  if (token_location >= physical_capacity || head >= descriptor.heads ||
      dimension >= descriptor.head_dimension) {
    return OffsetResult::failure(
        target_error(kind == DsparkTargetVerifyKvPoolKind::kTarget
                         ? DsparkTargetVerifyArgument::kTargetKvPool
                         : DsparkTargetVerifyArgument::kDraftKvPool));
  }
  const uint64_t head_width = descriptor.head_dimension;
  const uint64_t token_width =
      static_cast<uint64_t>(descriptor.heads) * head_width;
  uint64_t offset = 0U;
  if (descriptor.layout == DsparkTargetVerifyKvLayout::kHnd) {
    const uint64_t page = token_location / descriptor.page_size;
    const uint64_t token_in_page = token_location % descriptor.page_size;
    const uint64_t page_width =
        static_cast<uint64_t>(descriptor.page_size) * token_width;
    offset = page * page_width +
             static_cast<uint64_t>(head) * descriptor.page_size * head_width +
             token_in_page * head_width + dimension;
  } else {
    // NHD and page-major [page, token, head, dimension] have the same linear
    // token/head ordering; page-major retains page boundaries for kernels.
    offset = token_location * token_width +
             static_cast<uint64_t>(head) * head_width + dimension;
  }
  return OffsetResult::success(std::move(offset));
}

NativeRuntimeError validate_dspark_target_verify_kv_layer_ids(
    std::span<const uint32_t> layer_ids,
    DsparkTargetVerifyKvPoolKind kind) noexcept {
  const bool target = kind == DsparkTargetVerifyKvPoolKind::kTarget;
  const bool draft = kind == DsparkTargetVerifyKvPoolKind::kDraft;
  const DsparkTargetVerifyArgument argument =
      draft ? DsparkTargetVerifyArgument::kDraftKvPool
            : DsparkTargetVerifyArgument::kTargetKvPool;
  if (!target && !draft) {
    return target_error(argument, static_cast<uint64_t>(kind));
  }
  const std::span<const uint32_t> required =
      target ? std::span<const uint32_t>(kDsparkTargetFullAttentionLayerIds)
             : std::span<const uint32_t>(kDsparkTargetDraftLayerIds);
  if (layer_ids.size() != required.size()) {
    return target_error(argument, layer_ids.size(), required.size());
  }
  for (std::size_t index = 0; index < required.size(); ++index) {
    if (layer_ids[index] != required[index]) {
      return target_error(argument, layer_ids[index], required[index]);
    }
  }
  return native_runtime_ok();
}

NativeRuntimeError validate_dspark_target_verify_recurrent_pool_descriptor(
    const DsparkTargetVerifyRecurrentPoolDescriptor &descriptor) noexcept {
  if (descriptor.mamba_cache_slots == 0U ||
      descriptor.speculative_request_slots == 0U ||
      descriptor.reserved[0] != 0U || descriptor.reserved[1] != 0U) {
    return target_error(DsparkTargetVerifyArgument::kReplaySlots);
  }
  return native_runtime_ok();
}

} // namespace sglang::native
