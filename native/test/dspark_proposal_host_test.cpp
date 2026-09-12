#include "sglang/native/dspark_proposal.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>

namespace {

using sglang::native::dspark_lm_head_tactic_name;
using sglang::native::dspark_proposal_argument_name;
using sglang::native::dspark_proposal_device_code_name;
using sglang::native::dspark_proposal_rng_counter_blocks;
using sglang::native::DsparkLayerCaptureDescriptor;
using sglang::native::DsparkLmHeadDescriptor;
using sglang::native::DsparkLmHeadTactic;
using sglang::native::DsparkProposalArgument;
using sglang::native::DsparkProposalDeviceCode;
using sglang::native::DsparkProposalRngStateV1;
using sglang::native::DsparkProposalShape;
using sglang::native::DType;
using sglang::native::is_ok;
using sglang::native::kDsparkProductionGamma;
using sglang::native::kDsparkProductionMarkovRank;
using sglang::native::kDsparkProductionNumVerifyTokens;
using sglang::native::kDsparkProductionVocabSize;
using sglang::native::make_dspark_proposal_rng_state_v1;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeOperation;
using sglang::native::validate_dspark_layer_capture_descriptor;
using sglang::native::validate_dspark_lm_head_descriptor;
using sglang::native::validate_dspark_proposal_shape;

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

[[nodiscard]] bool IdentifiersAreStable() {
  constexpr std::array<std::string_view, 3> tactic_names{"invalid", "dense",
                                                         "packed_nvfp4"};
  for (uint32_t index = 0; index < tactic_names.size(); ++index) {
    CHECK(dspark_lm_head_tactic_name(static_cast<DsparkLmHeadTactic>(index)) ==
          tactic_names[index]);
  }
  constexpr std::array<std::string_view, 20> argument_names{"none",
                                                            "batch_size",
                                                            "gamma",
                                                            "vocab_size",
                                                            "markov_rank",
                                                            "num_verify_tokens",
                                                            "lm_head_tactic",
                                                            "logits_dtype",
                                                            "layer_capture",
                                                            "base_logits",
                                                            "markov_w1",
                                                            "markov_w2",
                                                            "anchor_tokens",
                                                            "temperatures",
                                                            "greedy_mask",
                                                            "rng_state",
                                                            "proposal_tokens",
                                                            "corrected_logits",
                                                            "log_normalizers",
                                                            "device_status"};
  for (uint32_t index = 0; index < argument_names.size(); ++index) {
    CHECK(dspark_proposal_argument_name(static_cast<DsparkProposalArgument>(
              index)) == argument_names[index]);
  }
  CHECK(dspark_proposal_device_code_name(
            DsparkProposalDeviceCode::kInvalidRngStateDescriptor) ==
        "invalid_rng_state_descriptor");
  CHECK(dspark_proposal_device_code_name(
            DsparkProposalDeviceCode::kNonPositiveTemperature) ==
        "non_positive_temperature");
  return true;
}

[[nodiscard]] bool ProductionShapeIsClosed() {
  constexpr DsparkProposalShape production{
      1, kDsparkProductionGamma, kDsparkProductionVocabSize,
      kDsparkProductionMarkovRank, kDsparkProductionNumVerifyTokens};
  CHECK(is_ok(validate_dspark_proposal_shape(production)));
  CHECK(dspark_proposal_rng_counter_blocks(production) == 434560);

  auto wrong = production;
  wrong.batch_size = 2;
  const auto batch_error = validate_dspark_proposal_shape(wrong);
  CHECK(batch_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(batch_error.operation ==
        NativeRuntimeOperation::kValidateDsparkProposal);
  CHECK(batch_error.detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kBatchSize));

  wrong = production;
  wrong.gamma = 4;
  CHECK(validate_dspark_proposal_shape(wrong).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kGamma));
  wrong = production;
  wrong.vocab_size = kDsparkProductionVocabSize - 1;
  CHECK(validate_dspark_proposal_shape(wrong).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kVocabSize));
  wrong = production;
  wrong.markov_rank = 128;
  CHECK(validate_dspark_proposal_shape(wrong).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kMarkovRank));
  wrong = production;
  wrong.num_verify_tokens = 7;
  CHECK(validate_dspark_proposal_shape(wrong).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kNumVerifyTokens));
  return true;
}

[[nodiscard]] bool LmHeadTacticsAndLogitsDtypeAreValidated() {
  const DsparkLmHeadDescriptor dense{DsparkLmHeadTactic::kDense,
                                     DType::kBFloat16,
                                     DType::kBFloat16,
                                     0,
                                     5120,
                                     kDsparkProductionVocabSize,
                                     1};
  CHECK(is_ok(validate_dspark_lm_head_descriptor(dense)));

  auto fp16 = dense;
  fp16.weight_dtype = DType::kFloat16;
  fp16.logits_dtype = DType::kFloat16;
  CHECK(is_ok(validate_dspark_lm_head_descriptor(fp16)));

  const DsparkLmHeadDescriptor packed{DsparkLmHeadTactic::kPackedNvFp4,
                                      DType::kNvFp4E2M1,
                                      DType::kBFloat16,
                                      0,
                                      5120,
                                      kDsparkProductionVocabSize,
                                      2};
  CHECK(is_ok(validate_dspark_lm_head_descriptor(packed)));

  auto invalid = packed;
  invalid.logits_dtype = DType::kUInt8;
  CHECK(validate_dspark_lm_head_descriptor(invalid).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kLogitsDType));
  invalid = packed;
  invalid.packed_elements_per_byte = 1;
  CHECK(validate_dspark_lm_head_descriptor(invalid).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kLmHeadTactic));
  return true;
}

[[nodiscard]] bool LayerCaptureAndRngStateAreFrozen() {
  const DsparkLayerCaptureDescriptor capture{{5, 19, 33, 47, 61}, 5, 64};
  CHECK(is_ok(validate_dspark_layer_capture_descriptor(capture)));
  auto invalid = capture;
  invalid.layer_ids[3] = 46;
  CHECK(validate_dspark_layer_capture_descriptor(invalid).detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kLayerCapture));

  constexpr DsparkProposalRngStateV1 state =
      make_dspark_proposal_rng_state_v1(17, 23, 29);
  CHECK(state.seed == 17);
  CHECK(state.subsequence == 23);
  CHECK(state.counter == 29);
  return true;
}

static_assert(sizeof(DsparkProposalRngStateV1) == 32);
static_assert(std::is_trivially_copyable_v<DsparkProposalRngStateV1>);

} // namespace

int main() {
  if (!IdentifiersAreStable() || !ProductionShapeIsClosed() ||
      !LmHeadTacticsAndLogitsDtypeAreValidated() ||
      !LayerCaptureAndRngStateAreFrozen()) {
    return 1;
  }
  std::printf("[  PASSED  ] 4 tests\n");
  return 0;
}
