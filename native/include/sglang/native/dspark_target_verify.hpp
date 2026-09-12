#ifndef SGLANG_NATIVE_DSPARK_TARGET_VERIFY_HPP_
#define SGLANG_NATIVE_DSPARK_TARGET_VERIFY_HPP_

#include "sglang/native/cuda_graph_resources.hpp"
#include "sglang/native/dspark_cycle_controller.hpp"
#include "sglang/native/gdn_replayssm_commit.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kDsparkTargetVerifyContractVersion = 1;
inline constexpr uint32_t kDsparkTargetNumLayers = 64;
inline constexpr uint32_t kDsparkTargetNumFullAttentionLayers = 16;
inline constexpr uint32_t kDsparkTargetNumGdnLayers = 48;
inline constexpr uint32_t kDsparkTargetNumAttentionHeads = 24;
inline constexpr uint32_t kDsparkTargetNumKvHeads = 4;
inline constexpr uint32_t kDsparkTargetAttentionHeadDimension = 256;
inline constexpr uint32_t kDsparkTargetGdnQueryHeads = 16;
inline constexpr uint32_t kDsparkTargetGdnQueryDimension = 128;
inline constexpr uint32_t kDsparkTargetGdnKeyHeads = 16;
inline constexpr uint32_t kDsparkTargetGdnKeyDimension = 128;
inline constexpr uint32_t kDsparkTargetGdnValueHeads = 48;
inline constexpr uint32_t kDsparkTargetGdnValueDimension = 128;
inline constexpr uint32_t kDsparkTargetGdnConvolutionChannels =
    kDsparkTargetGdnQueryHeads * kDsparkTargetGdnQueryDimension +
    kDsparkTargetGdnKeyHeads * kDsparkTargetGdnKeyDimension +
    kDsparkTargetGdnValueHeads * kDsparkTargetGdnValueDimension;
inline constexpr uint32_t kDsparkTargetGdnConvolutionKernelSize = 4;
inline constexpr uint32_t kDsparkTargetGdnConvolutionStateWidth =
    kDsparkTargetGdnConvolutionKernelSize - 1U;
inline constexpr uint32_t kDsparkTargetNumDraftLayers = 5;
inline constexpr uint32_t kDsparkTargetDraftNumKvHeads = 8;
inline constexpr uint32_t kDsparkTargetDraftHeadDimension = 128;
inline constexpr uint32_t kDsparkTargetTopKLimit = 32;
inline constexpr uint32_t kDsparkTargetProductionKvPageSize = 64;
inline constexpr uint64_t kDsparkTargetProductionKvCapacity = 200000;
inline constexpr std::array<uint32_t, kDsparkProductionTargetLayerCount>
    kDsparkTargetCaptureLayerIds{5, 19, 33, 47, 61};
inline constexpr std::array<uint32_t, kDsparkTargetNumFullAttentionLayers>
    kDsparkTargetFullAttentionLayerIds{3,  7,  11, 15, 19, 23, 27, 31,
                                       35, 39, 43, 47, 51, 55, 59, 63};
inline constexpr std::array<uint32_t, kDsparkTargetNumDraftLayers>
    kDsparkTargetDraftLayerIds{0, 1, 2, 3, 4};

enum class DsparkTargetVerifyKvCommitMode : uint32_t {
  kInvalid = 0,
  // Production parity: target verify projects and writes all eight draft-KV
  // rows.  The later KV-write stage publishes reachability for the accepted
  // prefix only; rejected speculative rows remain unreachable and reusable.
  kSpeculativePrewrite = 1,
};

enum class DsparkTargetVerifyArgument : uint32_t {
  kNone = 0,
  kContractVersion = 1,
  kShape = 2,
  kCaptureLayers = 3,
  kKvCommitMode = 4,
  kSamplingFlags = 5,
  kTopK = 6,
  kTopP = 7,
  kMinP = 8,
  kTemperature = 9,
  kInputTokens = 10,
  kPositions = 11,
  kTargetCacheLocations = 12,
  kTargetSequenceLengths = 13,
  kTargetRequestSlots = 14,
  kAdditivePenalties = 15,
  kScalingPenalties = 16,
  kGrammarMask = 17,
  kLogitBias = 18,
  kTargetProbabilities = 19,
  kTargetHidden = 20,
  kTargetKvPublication = 21,
  kDraftKvPublication = 22,
  kReplayRawValues = 23,
  kReplayRawKeys = 24,
  kReplayLogDecay = 25,
  kReplayBeta = 26,
  kReplayConvWindows = 27,
  kPublishedOutputs = 28,
  kDeviceStatus = 29,
  kCycleArena = 30,
  kResourceOwner = 31,
  kEnqueue = 32,
  kModelStorage = 33,
  kReplaySlots = 34,
  kTargetKvPool = 35,
  kDraftKvPool = 36,
  kPageSize = 37,
  kCacheCapacity = 38,
};

enum class DsparkTargetVerifySamplingFlag : uint32_t {
  kTopK = 1U << 0U,
  kTopP = 1U << 1U,
  kMinP = 1U << 2U,
  kAdditivePenalties = 1U << 3U,
  kScalingPenalties = 1U << 4U,
  kGrammarMask = 1U << 5U,
  kLogitBias = 1U << 6U,
};

enum class DsparkTargetVerifyDeviceCode : uint32_t {
  kOk = 0,
  kInvalidInputToken = 0x00050001U,
  kInvalidPosition = 0x00050002U,
  kInvalidTargetCacheLocation = 0x00050003U,
  kInvalidTargetRequestSlot = 0x00050004U,
  kInvalidSamplingControl = 0x00050005U,
};

struct DsparkTargetVerifyShape final {
  uint32_t batch_size;
  uint32_t gamma;
  uint32_t verify_width;
  uint32_t vocabulary_size;
  uint32_t hidden_size;
  uint32_t target_layers;
  uint32_t full_attention_layers;
  uint32_t gdn_layers;
  uint32_t attention_heads;
  uint32_t attention_kv_heads;
  uint32_t attention_head_dimension;
  uint32_t gdn_key_heads;
  uint32_t gdn_value_heads;
  uint32_t gdn_key_dimension;
  uint32_t gdn_value_dimension;
  uint32_t convolution_window;
  uint32_t captured_target_layers;
  uint32_t packed_target_hidden_width;
  uint32_t draft_layers;
  uint32_t draft_kv_heads;
  uint32_t draft_head_dimension;
  uint32_t reserved;
};

struct DsparkTargetVerifySamplingDescriptor final {
  uint32_t flags;
  uint32_t top_k;
  float top_p;
  float min_p;
  float temperature;
  uint32_t reserved;
};

enum class DsparkTargetVerifyKvDType : uint32_t {
  kInvalid = 0,
  kFloat8E4M3Fn = 1,
  kBFloat16 = 2,
};

enum class DsparkTargetVerifyKvLayout : uint32_t {
  kInvalid = 0,
  // One allocation per layer: [token_slot, head, dimension].
  kNhd = 1,
  // One allocation per layer: [page, head, token_in_page, dimension].
  kHnd = 2,
  // One allocation per layer: [page, token_in_page, head, dimension].
  kPageMajorLayerMajor = 3,
};

enum class DsparkTargetVerifyKvPoolKind : uint32_t {
  kInvalid = 0,
  kTarget = 1,
  kDraft = 2,
};

struct DsparkTargetVerifyKvPoolDescriptor final {
  uint32_t layers;
  uint32_t heads;
  uint32_t head_dimension;
  uint32_t page_size;
  // Logical scheduler pool size. SGLang allocates one additional physical
  // page as a padded sink, so each layer view contains
  // capacity_tokens + page_size token rows.
  uint64_t capacity_tokens;
  DsparkTargetVerifyKvDType dtype;
  DsparkTargetVerifyKvLayout layout;
};

struct DsparkTargetVerifyRecurrentPoolDescriptor final {
  // Logical configurable counts; each SGLang allocation includes one padded
  // sink slot, so the corresponding physical tensor extent is count + 1.
  uint32_t mamba_cache_slots;
  uint32_t speculative_request_slots;
  std::array<uint32_t, 2> reserved;
};

using DsparkTargetConstInt64Vector =
    GraphStableTensorView<DType::kInt64, 1, TensorAccess::kReadOnly>;
using DsparkTargetConstFloat32Vector =
    GraphStableTensorView<DType::kFloat32, 1, TensorAccess::kReadOnly>;
using DsparkTargetConstFloat32Matrix =
    GraphStableTensorView<DType::kFloat32, 2, TensorAccess::kReadOnly>;
using DsparkTargetConstUInt32Matrix =
    GraphStableTensorView<DType::kUInt32, 2, TensorAccess::kReadOnly>;
using DsparkTargetMutableFloat32Tensor3 =
    GraphStableTensorView<DType::kFloat32, 3, TensorAccess::kReadWrite>;
using DsparkTargetMutableBFloat16Tensor3 =
    GraphStableTensorView<DType::kBFloat16, 3, TensorAccess::kReadWrite>;
using DsparkTargetMutableFloat8Bytes =
    GraphStableTensorView<DType::kFloat8E4M3Fn, 1, TensorAccess::kReadWrite>;
using DsparkTargetMutableBFloat16Bytes =
    GraphStableTensorView<DType::kBFloat16, 1, TensorAccess::kReadWrite>;
using DsparkTargetMutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;
using DsparkTargetMutableBFloat16Tensor4 =
    GraphStableTensorView<DType::kBFloat16, 4, TensorAccess::kReadWrite>;
using DsparkTargetMutableBFloat16Tensor5 =
    GraphStableTensorView<DType::kBFloat16, 5, TensorAccess::kReadWrite>;
using DsparkTargetMutableFloat32Tensor4 =
    GraphStableTensorView<DType::kFloat32, 4, TensorAccess::kReadWrite>;
using DsparkTargetMutableBFloat16ConvTensor5 =
    GraphStableTensorView<DType::kBFloat16, 5, TensorAccess::kReadWrite>;

struct DsparkTargetVerifyConvBuffers final {
  const GdnMutableBFloat16Tensor4 &conv_states;
  const DsparkTargetMutableBFloat16ConvTensor5 &intermediate_conv_windows;
};

struct DsparkTargetVerifyKvPool final {
  DsparkTargetVerifyKvPoolDescriptor descriptor;
  std::span<const uint32_t> layer_ids;
  // Actual pool destinations, not compact eight-row output tensors. Every
  // layer supplies graph-stable K and V storage with the declared layout. The
  // model plan scatters through inputs.target_cache_locations.
  std::span<const DsparkTargetMutableFloat8Bytes> fp8_keys;
  std::span<const DsparkTargetMutableFloat8Bytes> fp8_values;
  std::span<const DsparkTargetMutableBFloat16Bytes> bfloat16_keys;
  std::span<const DsparkTargetMutableBFloat16Bytes> bfloat16_values;
};

struct DsparkTargetVerifyKvOutputs final {
  DsparkTargetVerifyKvPool target;
  DsparkTargetVerifyKvPool draft;
};

// Inputs whose addresses and shapes are fixed for every replay.  Sequence
// lengths describe the prefix before the eight verify rows.  target_cache_locs
// identifies the target KV destinations for those rows.
struct DsparkTargetVerifyInputs final {
  const DsparkTargetConstInt64Vector &input_tokens;
  const DsparkTargetConstInt64Vector &positions;
  const DsparkTargetConstInt64Vector &target_cache_locations;
  const DsparkTargetConstInt64Vector &target_sequence_lengths;
  const DsparkTargetConstInt64Vector &target_request_slots;
};

// Runtime sampling state is explicit.  Optional dense rows are represented by
// nullable pointers and must agree with descriptor.flags.  custom Python logit
// processors are intentionally outside this native contract.
struct DsparkTargetVerifySamplingBuffers final {
  DsparkTargetVerifySamplingDescriptor descriptor;
  const DsparkTargetConstFloat32Matrix *additive_penalties;
  const DsparkTargetConstFloat32Matrix *scaling_penalties;
  // Packed 32-bit mask as consumed by the qualified grammar backends.
  const DsparkTargetConstUInt32Matrix *grammar_mask;
  const DsparkTargetConstFloat32Matrix *logit_bias;
};

// All output rows are graph-stable and remain unpublished until the model plan
// validates its device inputs and completes every required write.
struct DsparkTargetVerifyOutputs final {
  DsparkTargetVerifyRecurrentPoolDescriptor recurrent;
  const DsparkTargetMutableFloat32Tensor3 &target_probabilities;
  const DsparkTargetMutableBFloat16Tensor3 &target_hidden;
  // Production cache writes. Each target/draft layer supplies distinct K and
  // V pool allocations; the model plan scatters all eight verify rows through
  // target_cache_locations according to the pool's declared physical layout.
  // Draft KV is speculative-prewritten for all eight rows.
  DsparkTargetVerifyKvOutputs kv;
  const DsparkTargetMutableBFloat16Tensor5 &replay_raw_values;
  const DsparkTargetMutableBFloat16Tensor5 &replay_raw_keys;
  const DsparkTargetMutableFloat32Tensor4 &replay_log_decay;
  const DsparkTargetMutableFloat32Tensor4 &replay_beta;
  std::span<const DsparkTargetVerifyConvBuffers> replay_convolution;
  const DsparkTargetMutableUInt32Vector &published_outputs;
  const DsparkTargetMutableUInt32Vector &device_status;
};

// The full 64-layer Qwen3.5 implementation is supplied as an externally owned
// AOT plan.  The callback must enqueue only allocation-free CUDA work on the
// supplied stream and must publish every output bit only after successful
// completion.  Resources are retained by DsparkModelGraphResourceOwner.
using DsparkTargetVerifyEnqueue = NativeRuntimeError (*)(
    const CudaExecutionContext &context, const DsparkTargetVerifyInputs &inputs,
    const DsparkTargetVerifySamplingBuffers &sampling,
    const DsparkTargetVerifyOutputs &outputs, void *implementation) noexcept;

struct DsparkTargetVerifyPlan final {
  uint32_t contract_version;
  DsparkTargetVerifyKvCommitMode kv_commit_mode;
  DsparkTargetVerifyShape shape;
  DsparkLayerCaptureDescriptor capture;
  DsparkTargetVerifyEnqueue enqueue;
  void *implementation;
};

class DsparkCapturedTargetVerifyGraph;
[[nodiscard]] NativeRuntimeResult<DsparkCapturedTargetVerifyGraph>
capture_dspark_target_verify_graph(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const DsparkModelGraphResourceOwner &model_resources,
    const DsparkTargetVerifyPlan &plan, const DsparkTargetVerifyInputs &inputs,
    const DsparkTargetVerifySamplingBuffers &sampling,
    const DsparkTargetVerifyOutputs &outputs) noexcept;

// Owns the captured graph. source() borrows graph_ and is valid only while this
// wrapper remains alive; the cycle controller clones it and retains resources.
class DsparkCapturedTargetVerifyGraph final {
public:
  DsparkCapturedTargetVerifyGraph(const DsparkCapturedTargetVerifyGraph &) =
      delete;
  DsparkCapturedTargetVerifyGraph &
  operator=(const DsparkCapturedTargetVerifyGraph &) = delete;
  DsparkCapturedTargetVerifyGraph(
      DsparkCapturedTargetVerifyGraph &&other) noexcept;
  DsparkCapturedTargetVerifyGraph &
  operator=(DsparkCapturedTargetVerifyGraph &&) = delete;
  ~DsparkCapturedTargetVerifyGraph() noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] DsparkTargetVerifyGraphSource source() const noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;

private:
  DsparkCapturedTargetVerifyGraph(
      CudaCapturedGraph graph,
      DsparkModelGraphResourceLease resources) noexcept;

  CudaCapturedGraph graph_;
  DsparkModelGraphResourceLease resources_;

  friend NativeRuntimeResult<DsparkCapturedTargetVerifyGraph>
  capture_dspark_target_verify_graph(
      const CudaExecutionContext &, const GraphArenaLease &,
      const DsparkModelGraphResourceOwner &, const DsparkTargetVerifyPlan &,
      const DsparkTargetVerifyInputs &,
      const DsparkTargetVerifySamplingBuffers &,
      const DsparkTargetVerifyOutputs &) noexcept;
};

[[nodiscard]] constexpr DsparkTargetVerifyShape
dspark_production_target_verify_shape() noexcept {
  return DsparkTargetVerifyShape{kDsparkProductionBatchSize,
                                 kDsparkProductionGamma,
                                 kDsparkProductionNumVerifyTokens,
                                 kDsparkProductionVocabSize,
                                 5120U,
                                 kDsparkTargetNumLayers,
                                 kDsparkTargetNumFullAttentionLayers,
                                 kDsparkTargetNumGdnLayers,
                                 kDsparkTargetNumAttentionHeads,
                                 kDsparkTargetNumKvHeads,
                                 kDsparkTargetAttentionHeadDimension,
                                 kGdnProductionNumKeyHeads,
                                 kGdnProductionNumValueHeads,
                                 kGdnProductionKeyDimension,
                                 kGdnProductionValueDimension,
                                 kDsparkTargetGdnConvolutionStateWidth,
                                 kDsparkProductionTargetLayerCount,
                                 25600U,
                                 kDsparkTargetNumDraftLayers,
                                 kDsparkTargetDraftNumKvHeads,
                                 kDsparkTargetDraftHeadDimension,
                                 0U};
}

[[nodiscard]] constexpr DsparkLayerCaptureDescriptor
dspark_production_target_capture() noexcept {
  return DsparkLayerCaptureDescriptor{kDsparkTargetCaptureLayerIds,
                                      kDsparkProductionTargetLayerCount,
                                      kDsparkTargetNumLayers};
}

[[nodiscard]] std::string_view dspark_target_verify_argument_name(
    DsparkTargetVerifyArgument argument) noexcept;
[[nodiscard]] std::string_view dspark_target_verify_kv_commit_mode_name(
    DsparkTargetVerifyKvCommitMode mode) noexcept;
[[nodiscard]] std::string_view dspark_target_verify_device_code_name(
    DsparkTargetVerifyDeviceCode code) noexcept;
[[nodiscard]] NativeRuntimeError
validate_dspark_target_verify_plan(const DsparkTargetVerifyPlan &plan) noexcept;
[[nodiscard]] NativeRuntimeError
validate_dspark_target_verify_sampling_descriptor(
    const DsparkTargetVerifySamplingDescriptor &descriptor) noexcept;
[[nodiscard]] NativeRuntimeError
validate_dspark_target_verify_kv_pool_descriptor(
    const DsparkTargetVerifyKvPoolDescriptor &descriptor,
    DsparkTargetVerifyKvPoolKind kind) noexcept;
[[nodiscard]] NativeRuntimeResult<uint64_t>
dspark_target_verify_kv_element_offset(
    const DsparkTargetVerifyKvPoolDescriptor &descriptor,
    DsparkTargetVerifyKvPoolKind kind, uint64_t token_location, uint32_t head,
    uint32_t dimension) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_target_verify_kv_layer_ids(
    std::span<const uint32_t> layer_ids,
    DsparkTargetVerifyKvPoolKind kind) noexcept;
[[nodiscard]] NativeRuntimeError
validate_dspark_target_verify_recurrent_pool_descriptor(
    const DsparkTargetVerifyRecurrentPoolDescriptor &descriptor) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_target_verify_buffers(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const DsparkModelGraphResourceLease &model_resources,
    const DsparkTargetVerifyInputs &inputs,
    const DsparkTargetVerifySamplingBuffers &sampling,
    const DsparkTargetVerifyOutputs &outputs) noexcept;

// Captures one real model-owned target-verify source graph. CUDA capture only
// records work; output publication is qualified by a separate instantiate /
// launch / synchronize parity test rather than read at capture time.
static_assert(sizeof(DsparkTargetVerifyShape) == 88);
static_assert(std::is_standard_layout_v<DsparkTargetVerifyShape>);
static_assert(std::is_trivially_copyable_v<DsparkTargetVerifyShape>);
static_assert(sizeof(DsparkTargetVerifySamplingDescriptor) == 24);
static_assert(std::is_standard_layout_v<DsparkTargetVerifySamplingDescriptor>);
static_assert(
    std::is_trivially_copyable_v<DsparkTargetVerifySamplingDescriptor>);
static_assert(sizeof(DsparkTargetVerifyKvPoolDescriptor) == 32);
static_assert(std::is_standard_layout_v<DsparkTargetVerifyKvPoolDescriptor>);
static_assert(std::is_trivially_copyable_v<DsparkTargetVerifyKvPoolDescriptor>);
static_assert(sizeof(DsparkTargetVerifyRecurrentPoolDescriptor) == 16);
static_assert(
    std::is_standard_layout_v<DsparkTargetVerifyRecurrentPoolDescriptor>);
static_assert(
    std::is_trivially_copyable_v<DsparkTargetVerifyRecurrentPoolDescriptor>);

} // namespace sglang::native

#endif // SGLANG_NATIVE_DSPARK_TARGET_VERIFY_HPP_
