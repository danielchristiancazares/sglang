#include "sglang/native/dspark_target_verify.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using sglang::native::dspark_production_target_capture;
using sglang::native::dspark_production_target_verify_shape;
using sglang::native::dspark_target_verify_argument_name;
using sglang::native::dspark_target_verify_device_code_name;
using sglang::native::dspark_target_verify_kv_commit_mode_name;
using sglang::native::dspark_target_verify_kv_element_offset;
using sglang::native::DsparkTargetVerifyArgument;
using sglang::native::DsparkTargetVerifyDeviceCode;
using sglang::native::DsparkTargetVerifyKvCommitMode;
using sglang::native::DsparkTargetVerifyKvDType;
using sglang::native::DsparkTargetVerifyKvLayout;
using sglang::native::DsparkTargetVerifyKvPoolDescriptor;
using sglang::native::DsparkTargetVerifyKvPoolKind;
using sglang::native::DsparkTargetVerifyPlan;
using sglang::native::DsparkTargetVerifyRecurrentPoolDescriptor;
using sglang::native::DsparkTargetVerifySamplingDescriptor;
using sglang::native::DsparkTargetVerifySamplingFlag;
using sglang::native::is_ok;
using sglang::native::kDsparkProductionVocabSize;
using sglang::native::kDsparkTargetVerifyContractVersion;
using sglang::native::NativeRuntimeError;
using sglang::native::validate_dspark_target_verify_kv_layer_ids;
using sglang::native::validate_dspark_target_verify_kv_pool_descriptor;
using sglang::native::validate_dspark_target_verify_plan;
using sglang::native::validate_dspark_target_verify_recurrent_pool_descriptor;
using sglang::native::validate_dspark_target_verify_sampling_descriptor;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {   \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] NativeRuntimeError
unused_enqueue(const sglang::native::CudaExecutionContext &,
               const sglang::native::DsparkTargetVerifyInputs &,
               const sglang::native::DsparkTargetVerifySamplingBuffers &,
               const sglang::native::DsparkTargetVerifyOutputs &,
               void *) noexcept {
  return sglang::native::native_runtime_ok();
}

[[nodiscard]] bool ProductionShapeAndNamesAreStable() {
  constexpr auto shape = dspark_production_target_verify_shape();
  CHECK(shape.batch_size == 1U);
  CHECK(shape.gamma == 7U);
  CHECK(shape.verify_width == 8U);
  CHECK(shape.vocabulary_size == 248320U);
  CHECK(shape.hidden_size == 5120U);
  CHECK(shape.target_layers == 64U);
  CHECK(shape.full_attention_layers == 16U);
  CHECK(shape.gdn_layers == 48U);
  CHECK(shape.attention_heads == 24U);
  CHECK(shape.attention_kv_heads == 4U);
  CHECK(shape.attention_head_dimension == 256U);
  CHECK(shape.gdn_key_heads == 16U);
  CHECK(shape.gdn_value_heads == 48U);
  CHECK(shape.gdn_key_dimension == 128U);
  CHECK(shape.gdn_value_dimension == 128U);
  CHECK(shape.convolution_window == 3U);
  CHECK(shape.captured_target_layers == 5U);
  CHECK(shape.packed_target_hidden_width == 25600U);
  CHECK(shape.draft_layers == 5U);
  CHECK(shape.draft_kv_heads == 8U);
  CHECK(shape.draft_head_dimension == 128U);
  CHECK(sglang::native::kDsparkTargetGdnConvolutionChannels == 10240U);
  CHECK(sglang::native::kDsparkTargetGdnConvolutionKernelSize == 4U);
  CHECK(sglang::native::kDsparkTargetGdnConvolutionStateWidth == 3U);
  CHECK(shape.reserved == 0U);

  constexpr auto capture = dspark_production_target_capture();
  CHECK(capture.layer_ids == (std::array<uint32_t, 5>{5U, 19U, 33U, 47U, 61U}));
  CHECK(capture.num_layers == 5U);
  CHECK(capture.target_num_layers == 64U);
  CHECK(sglang::native::kDsparkTargetFullAttentionLayerIds ==
        (std::array<uint32_t, 16>{3U, 7U, 11U, 15U, 19U, 23U, 27U, 31U, 35U,
                                  39U, 43U, 47U, 51U, 55U, 59U, 63U}));
  CHECK(dspark_target_verify_argument_name(
            DsparkTargetVerifyArgument::kTargetProbabilities) ==
        "target_probabilities");
  CHECK(dspark_target_verify_argument_name(
            DsparkTargetVerifyArgument::kDraftKvPublication) ==
        "draft_kv_prewrite_publication");
  CHECK(dspark_target_verify_kv_commit_mode_name(
            DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite) ==
        "speculative_prewrite");
  CHECK(dspark_target_verify_device_code_name(
            DsparkTargetVerifyDeviceCode::kInvalidInputToken) ==
        "invalid_input_token");
  const DsparkTargetVerifyKvPoolDescriptor pool{
      16U,
      4U,
      256U,
      64U,
      200000U,
      DsparkTargetVerifyKvDType::kFloat8E4M3Fn,
      DsparkTargetVerifyKvLayout::kNhd};
  CHECK(pool.capacity_tokens + pool.page_size == 200064U);
  CHECK(sizeof(pool) == 32U);
  return true;
}

[[nodiscard]] bool PlanValidationIsProductionSpecific() {
  int implementation = 0;
  DsparkTargetVerifyPlan plan{
      kDsparkTargetVerifyContractVersion,
      DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite,
      dspark_production_target_verify_shape(),
      dspark_production_target_capture(),
      &unused_enqueue,
      &implementation};
  CHECK(is_ok(validate_dspark_target_verify_plan(plan)));

  ++plan.shape.gamma;
  CHECK(validate_dspark_target_verify_plan(plan).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kShape));
  plan.shape = dspark_production_target_verify_shape();
  plan.capture.layer_ids[2] = 34U;
  CHECK(validate_dspark_target_verify_plan(plan).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kCaptureLayers));
  plan.capture = dspark_production_target_capture();
  plan.kv_commit_mode = DsparkTargetVerifyKvCommitMode::kInvalid;
  CHECK(validate_dspark_target_verify_plan(plan).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kKvCommitMode));
  plan.kv_commit_mode = DsparkTargetVerifyKvCommitMode::kSpeculativePrewrite;
  plan.enqueue = nullptr;
  CHECK(validate_dspark_target_verify_plan(plan).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kEnqueue));
  plan.enqueue = &unused_enqueue;
  plan.implementation = nullptr;
  CHECK(validate_dspark_target_verify_plan(plan).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kResourceOwner));
  return true;
}

[[nodiscard]] bool SamplingValidationPreservesProductionSemantics() {
  constexpr uint32_t kTopK =
      static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kTopK);
  constexpr uint32_t kTopP =
      static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kTopP);
  constexpr uint32_t kAdditive =
      static_cast<uint32_t>(DsparkTargetVerifySamplingFlag::kAdditivePenalties);
  DsparkTargetVerifySamplingDescriptor sampled{
      kTopK | kTopP | kAdditive, 20U, 0.95F, 0.0F, 1.0F, 0U};
  CHECK(is_ok(validate_dspark_target_verify_sampling_descriptor(sampled)));

  sampled.top_k = 33U;
  CHECK(validate_dspark_target_verify_sampling_descriptor(sampled).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTopK));
  sampled.top_k = 20U;
  sampled.top_p = 0.0F;
  CHECK(validate_dspark_target_verify_sampling_descriptor(sampled).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTopP));
  sampled.top_p = 0.95F;
  sampled.temperature = -1.0F;
  CHECK(validate_dspark_target_verify_sampling_descriptor(sampled).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTemperature));

  const DsparkTargetVerifySamplingDescriptor unfiltered{
      0U, kDsparkProductionVocabSize, 1.0F, 0.0F, 1.0F, 0U};
  CHECK(is_ok(validate_dspark_target_verify_sampling_descriptor(unfiltered)));
  const DsparkTargetVerifySamplingDescriptor incoherent{0U,   20U,  1.0F,
                                                        0.0F, 1.0F, 0U};
  CHECK(validate_dspark_target_verify_sampling_descriptor(incoherent).detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTopK));
  return true;
}

[[nodiscard]] bool PoolDescriptorsAreProductionSpecific() {
  DsparkTargetVerifyKvPoolDescriptor target{
      16U,
      4U,
      256U,
      64U,
      200000U,
      DsparkTargetVerifyKvDType::kFloat8E4M3Fn,
      DsparkTargetVerifyKvLayout::kNhd};
  CHECK(is_ok(validate_dspark_target_verify_kv_pool_descriptor(
      target, DsparkTargetVerifyKvPoolKind::kTarget)));
  target.page_size = 32U;
  CHECK(validate_dspark_target_verify_kv_pool_descriptor(
            target, DsparkTargetVerifyKvPoolKind::kTarget)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kPageSize));
  target.page_size = 64U;
  target.capacity_tokens = 199936U;
  CHECK(validate_dspark_target_verify_kv_pool_descriptor(
            target, DsparkTargetVerifyKvPoolKind::kTarget)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kCacheCapacity));
  target.capacity_tokens = 200000U;
  target.dtype = DsparkTargetVerifyKvDType::kInvalid;
  CHECK(validate_dspark_target_verify_kv_pool_descriptor(
            target, DsparkTargetVerifyKvPoolKind::kTarget)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTargetKvPool));
  target.dtype = DsparkTargetVerifyKvDType::kBFloat16;
  target.layout = DsparkTargetVerifyKvLayout::kInvalid;
  CHECK(validate_dspark_target_verify_kv_pool_descriptor(
            target, DsparkTargetVerifyKvPoolKind::kTarget)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTargetKvPool));
  target.layout = DsparkTargetVerifyKvLayout::kNhd;

  uint64_t nhd_offset = 0U;
  CHECK(std::move(
            dspark_target_verify_kv_element_offset(
                target, DsparkTargetVerifyKvPoolKind::kTarget, 65U, 3U, 255U))
            .match(
                [&nhd_offset](uint64_t &&value) noexcept {
                  nhd_offset = value;
                  return true;
                },
                [](NativeRuntimeError &&) noexcept { return false; }));
  CHECK(nhd_offset == 65ULL * 4ULL * 256ULL + 3ULL * 256ULL + 255ULL);
  target.layout = DsparkTargetVerifyKvLayout::kPageMajorLayerMajor;
  uint64_t page_major_offset = 0U;
  CHECK(std::move(
            dspark_target_verify_kv_element_offset(
                target, DsparkTargetVerifyKvPoolKind::kTarget, 65U, 3U, 255U))
            .match(
                [&page_major_offset](uint64_t &&value) noexcept {
                  page_major_offset = value;
                  return true;
                },
                [](NativeRuntimeError &&) noexcept { return false; }));
  CHECK(page_major_offset == nhd_offset);
  target.layout = DsparkTargetVerifyKvLayout::kHnd;
  uint64_t hnd_offset = 0U;
  CHECK(std::move(
            dspark_target_verify_kv_element_offset(
                target, DsparkTargetVerifyKvPoolKind::kTarget, 65U, 3U, 255U))
            .match(
                [&hnd_offset](uint64_t &&value) noexcept {
                  hnd_offset = value;
                  return true;
                },
                [](NativeRuntimeError &&) noexcept { return false; }));
  CHECK(hnd_offset ==
        64ULL * 4ULL * 256ULL + 3ULL * 64ULL * 256ULL + 256ULL + 255ULL);
  bool accepted_sink_overflow = false;
  std::move(dspark_target_verify_kv_element_offset(
                target, DsparkTargetVerifyKvPoolKind::kTarget, 200064U, 0U, 0U))
      .match([&accepted_sink_overflow](
                 uint64_t &&) noexcept { accepted_sink_overflow = true; },
             [](NativeRuntimeError &&) noexcept {});
  CHECK(!accepted_sink_overflow);

  const auto target_layers = sglang::native::kDsparkTargetFullAttentionLayerIds;
  CHECK(is_ok(validate_dspark_target_verify_kv_layer_ids(
      target_layers, DsparkTargetVerifyKvPoolKind::kTarget)));
  auto reordered = target_layers;
  std::swap(reordered[0], reordered[1]);
  CHECK(validate_dspark_target_verify_kv_layer_ids(
            reordered, DsparkTargetVerifyKvPoolKind::kTarget)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kTargetKvPool));
  CHECK(is_ok(validate_dspark_target_verify_kv_layer_ids(
      sglang::native::kDsparkTargetDraftLayerIds,
      DsparkTargetVerifyKvPoolKind::kDraft)));

  DsparkTargetVerifyRecurrentPoolDescriptor recurrent{5U, 1U, {0U, 0U}};
  CHECK(is_ok(
      validate_dspark_target_verify_recurrent_pool_descriptor(recurrent)));
  recurrent.mamba_cache_slots = 0U;
  CHECK(validate_dspark_target_verify_recurrent_pool_descriptor(recurrent)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kReplaySlots));
  recurrent.mamba_cache_slots = 4U;
  CHECK(is_ok(
      validate_dspark_target_verify_recurrent_pool_descriptor(recurrent)));
  recurrent.reserved[1] = 1U;
  CHECK(validate_dspark_target_verify_recurrent_pool_descriptor(recurrent)
            .detail ==
        static_cast<uint32_t>(DsparkTargetVerifyArgument::kReplaySlots));
  return true;
}

static_assert(sizeof(sglang::native::DsparkTargetVerifyShape) == 88);
static_assert(sizeof(DsparkTargetVerifySamplingDescriptor) == 24);
static_assert(
    std::is_trivially_copyable_v<sglang::native::DsparkTargetVerifyShape>);

} // namespace

int main() {
  if (!ProductionShapeAndNamesAreStable() ||
      !PlanValidationIsProductionSpecific() ||
      !SamplingValidationPreservesProductionSemantics() ||
      !PoolDescriptorsAreProductionSpecific()) {
    return 1;
  }
  std::printf("[  PASSED  ] 4 tests\n");
  return 0;
}
