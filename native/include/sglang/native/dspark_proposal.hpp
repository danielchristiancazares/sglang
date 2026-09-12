#ifndef SGLANG_NATIVE_DSPARK_PROPOSAL_HPP_
#define SGLANG_NATIVE_DSPARK_PROPOSAL_HPP_

#include "sglang/native/cuda_graph_resources.hpp"

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kDsparkProductionBatchSize = 1;
inline constexpr uint32_t kDsparkProductionGamma = 7;
inline constexpr uint32_t kDsparkProductionNumVerifyTokens = 8;
inline constexpr uint32_t kDsparkProductionVocabSize = 248320;
inline constexpr uint32_t kDsparkProductionMarkovRank = 256;
inline constexpr uint32_t kDsparkProductionTargetLayerCount = 5;
inline constexpr uint32_t kDsparkProposalRngStateWords = 4;
inline constexpr uint64_t kDsparkProposalRngStateDescriptorV1 =
    0x3150534452474c53ULL; // Little-endian bytes: "SLGRDSP1".
static_assert(kDsparkProductionNumVerifyTokens == kDsparkProductionGamma + 1);

enum class DsparkLmHeadTactic : uint32_t {
  kInvalid = 0,
  kDense = 1,
  kPackedNvFp4 = 2,
};

enum class DsparkProposalArgument : uint32_t {
  kNone = 0,
  kBatchSize = 1,
  kGamma = 2,
  kVocabSize = 3,
  kMarkovRank = 4,
  kNumVerifyTokens = 5,
  kLmHeadTactic = 6,
  kLogitsDType = 7,
  kLayerCapture = 8,
  kBaseLogits = 9,
  kMarkovW1 = 10,
  kMarkovW2 = 11,
  kAnchorTokens = 12,
  kTemperatures = 13,
  kGreedyMask = 14,
  kRngState = 15,
  kProposalTokens = 16,
  kCorrectedLogits = 17,
  kLogNormalizers = 18,
  kDeviceStatus = 19,
};

enum class DsparkProposalDeviceCode : uint32_t {
  kOk = 0,
  kInvalidRngStateDescriptor = 0x00020001U,
  kRngCounterOverflow = 0x00020002U,
  kAnchorTokenOutOfRange = 0x00020003U,
  kNonPositiveTemperature = 0x00020004U,
};

struct DsparkProposalShape final {
  uint64_t batch_size;
  uint64_t gamma;
  uint64_t vocab_size;
  uint64_t markov_rank;
  uint64_t num_verify_tokens;
};

struct DsparkLmHeadDescriptor final {
  DsparkLmHeadTactic tactic;
  DType weight_dtype;
  DType logits_dtype;
  uint32_t reserved;
  uint64_t input_features;
  uint64_t output_features;
  uint64_t packed_elements_per_byte;
};

struct DsparkLayerCaptureDescriptor final {
  std::array<uint32_t, kDsparkProductionTargetLayerCount> layer_ids;
  uint32_t num_layers;
  uint32_t target_num_layers;
};

struct DsparkProposalRngStateV1 final {
  uint64_t descriptor;
  uint64_t seed;
  uint64_t subsequence;
  uint64_t counter;
};

using DsparkConstBFloat16Matrix =
    GraphStableTensorView<DType::kBFloat16, 2, TensorAccess::kReadOnly>;
using DsparkConstBFloat16Tensor3 =
    GraphStableTensorView<DType::kBFloat16, 3, TensorAccess::kReadOnly>;
using DsparkMutableBFloat16Tensor3 =
    GraphStableTensorView<DType::kBFloat16, 3, TensorAccess::kReadWrite>;
using DsparkConstFloat16Matrix =
    GraphStableTensorView<DType::kFloat16, 2, TensorAccess::kReadOnly>;
using DsparkConstFloat16Tensor3 =
    GraphStableTensorView<DType::kFloat16, 3, TensorAccess::kReadOnly>;
using DsparkMutableFloat16Tensor3 =
    GraphStableTensorView<DType::kFloat16, 3, TensorAccess::kReadWrite>;
using DsparkConstFloat32Vector =
    GraphStableTensorView<DType::kFloat32, 1, TensorAccess::kReadOnly>;
using DsparkMutableFloat32Matrix =
    GraphStableTensorView<DType::kFloat32, 2, TensorAccess::kReadWrite>;
using DsparkConstBool8Vector =
    GraphStableTensorView<DType::kBool8, 1, TensorAccess::kReadOnly>;
using DsparkConstInt64Vector =
    GraphStableTensorView<DType::kInt64, 1, TensorAccess::kReadOnly>;
using DsparkMutableInt64Matrix =
    GraphStableTensorView<DType::kInt64, 2, TensorAccess::kReadWrite>;
using DsparkMutableUInt64Vector =
    GraphStableTensorView<DType::kUInt64, 1, TensorAccess::kReadWrite>;
using DsparkMutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;

template <DType LogitsDType> struct DsparkProposalBuffers final {
  static_assert(LogitsDType == DType::kBFloat16 ||
                LogitsDType == DType::kFloat16);

  using ConstLogitsTensor3 =
      GraphStableTensorView<LogitsDType, 3, TensorAccess::kReadOnly>;
  using ConstLogitsMatrix =
      GraphStableTensorView<LogitsDType, 2, TensorAccess::kReadOnly>;
  using MutableLogitsTensor3 =
      GraphStableTensorView<LogitsDType, 3, TensorAccess::kReadWrite>;

  const ConstLogitsTensor3 &base_logits;
  const ConstLogitsMatrix &markov_w1;
  const ConstLogitsMatrix &markov_w2;
  const DsparkConstInt64Vector &anchor_tokens;
  const DsparkConstFloat32Vector &temperatures;
  const DsparkConstBool8Vector &greedy_mask;
  const DsparkMutableUInt64Vector &rng_state;
  const DsparkMutableInt64Matrix &proposal_tokens;
  const MutableLogitsTensor3 &corrected_logits;
  const DsparkMutableFloat32Matrix &log_normalizers;
  const DsparkMutableUInt32Vector &device_status;
};

// Owns one captured, guarded proposal source graph.  The graph retains the
// stream context and arena lease whose stable addresses its nodes reference.
// A cycle controller may clone graph(); the source bundle remains independently
// owned and may be closed after controller construction.
[[nodiscard]] constexpr DsparkProposalRngStateV1
make_dspark_proposal_rng_state_v1(uint64_t seed, uint64_t subsequence,
                                  uint64_t counter = 0) noexcept {
  return DsparkProposalRngStateV1{kDsparkProposalRngStateDescriptorV1, seed,
                                  subsequence, counter};
}

[[nodiscard]] std::string_view
dspark_lm_head_tactic_name(DsparkLmHeadTactic tactic) noexcept;
[[nodiscard]] std::string_view
dspark_proposal_argument_name(DsparkProposalArgument argument) noexcept;
[[nodiscard]] std::string_view
dspark_proposal_device_code_name(DsparkProposalDeviceCode code) noexcept;

[[nodiscard]] NativeRuntimeError
validate_dspark_proposal_shape(DsparkProposalShape shape) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_lm_head_descriptor(
    const DsparkLmHeadDescriptor &descriptor) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_layer_capture_descriptor(
    const DsparkLayerCaptureDescriptor &descriptor) noexcept;
[[nodiscard]] uint64_t
dspark_proposal_rng_counter_blocks(DsparkProposalShape shape) noexcept;

// Metadata failures return synchronously. Device-content failures publish a
// DsparkProposalDeviceCode and preserve all output buffers. Every successful
// replay reserves gamma * ceil(vocab_size / 4) Philox counter blocks.
[[nodiscard]] NativeRuntimeError launch_dspark_proposal_bfloat16(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept;
[[nodiscard]] NativeRuntimeError launch_dspark_proposal_bfloat16_if_ready(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept;
[[nodiscard]] NativeRuntimeError launch_dspark_proposal_float16(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept;
[[nodiscard]] NativeRuntimeError launch_dspark_proposal_float16_if_ready(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept;
[[nodiscard]] NativeRuntimeResult<CudaCapturedGraph>
capture_dspark_proposal_bfloat16_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept;
[[nodiscard]] NativeRuntimeResult<CudaCapturedGraph>
capture_dspark_proposal_float16_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept;

static_assert(sizeof(DsparkProposalShape) == 40);
static_assert(alignof(DsparkProposalShape) == 8);
static_assert(std::is_standard_layout_v<DsparkProposalShape>);
static_assert(std::is_trivially_copyable_v<DsparkProposalShape>);
static_assert(sizeof(DsparkLmHeadDescriptor) == 40);
static_assert(std::is_standard_layout_v<DsparkLmHeadDescriptor>);
static_assert(std::is_trivially_copyable_v<DsparkLmHeadDescriptor>);
static_assert(sizeof(DsparkLayerCaptureDescriptor) == 28);
static_assert(std::is_standard_layout_v<DsparkLayerCaptureDescriptor>);
static_assert(std::is_trivially_copyable_v<DsparkLayerCaptureDescriptor>);
static_assert(sizeof(DsparkProposalRngStateV1) == 32);
static_assert(std::is_standard_layout_v<DsparkProposalRngStateV1>);
static_assert(std::is_trivially_copyable_v<DsparkProposalRngStateV1>);
} // namespace sglang::native

#endif // SGLANG_NATIVE_DSPARK_PROPOSAL_HPP_
