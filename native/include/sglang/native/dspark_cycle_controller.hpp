#ifndef SGLANG_NATIVE_DSPARK_CYCLE_CONTROLLER_HPP_
#define SGLANG_NATIVE_DSPARK_CYCLE_CONTROLLER_HPP_

#include "sglang/native/cuda_graph_resources.hpp"
#include "sglang/native/dspark_proposal.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kDsparkCycleAbiVersion = 1;
inline constexpr uint32_t kDsparkCycleNumStages = 8;
inline constexpr uint32_t kDsparkCycleMaxOwnedGraphs =
    kDsparkCycleNumStages + 1;
inline constexpr uint32_t kDsparkCycleProductionNumOutTokens =
    kDsparkProductionNumVerifyTokens;

enum class DsparkCycleStage : uint32_t {
  kDraftProposal = 1,
  kTargetVerify = 2,
  kVerifyRng = 3,
  kRejectionSampling = 4,
  kReplaySsmCommit = 5,
  kDraftExtend = 6,
  kKvWrite = 7,
  kCompactResult = 8,
};

enum class DsparkCycleArgument : uint32_t {
  kNone = 0,
  kAbiVersion = 1,
  kStages = 2,
  kGraph = 3,
  kDeviceOrdinal = 4,
  kDeviceResult = 5,
  kOutTokens = 6,
  kNumCorrectDrafts = 7,
  kDeviceStatus = 8,
  kRequestSlot = 9,
  kResourceOwner = 10,
  kCycleArena = 11,
  kRequiredOutputs = 12,
  kProductionShape = 13,
  kModelKind = 14,
  kStableAddressProvenance = 15,
};

enum class DsparkCycleDeviceCode : uint32_t {
  kOk = 0,
  kNumCorrectDraftsOutOfRange = 0x00040001U,
};

struct DsparkCycleChildGraph final {
  DsparkCycleStage stage;
  uint32_t reserved;
  cudaGraph_t graph;
};

// This is the only host-visible result transferred by a replay. The controller
// clears the shared status before the first guarded stage; the first subsequent
// nonzero status is retained through compact publication. accepted_token_count
// includes the bonus token; num_correct_drafts excludes it.
struct DsparkCycleResultV1 final {
  uint32_t abi_version;
  uint32_t device_status;
  int32_t accepted_token_count;
  int32_t num_correct_drafts;
  int32_t output_token;
  int32_t request_slot;
};

struct DsparkCycleDescriptor final {
  uint32_t abi_version;
  uint32_t device_ordinal;
  std::span<const DsparkCycleChildGraph> stages;
  void *device_status;
  const void *device_result;
};

using DsparkCycleConstInt32Vector =
    GraphStableTensorView<DType::kInt32, 1, TensorAccess::kReadOnly>;
using DsparkCycleMutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;

struct DsparkCycleCompactResultBuffers final {
  // The accepted tokens are stored first, followed by the bonus token at
  // out_tokens[num_correct_drafts].  result_words owns exactly one
  // DsparkCycleResultV1 in its six uint32 words. request_slot identifies the
  // scheduler-owned request receiving the compact result.
  const DsparkCycleConstInt32Vector &out_tokens;
  const DsparkCycleConstInt32Vector &num_correct_drafts;
  const DsparkCycleConstInt32Vector &request_slot;
  const DsparkCycleMutableUInt32Vector &device_status;
  const DsparkCycleMutableUInt32Vector &result_words;
};

class DsparkCycleCompactGraph final {
public:
  DsparkCycleCompactGraph(const DsparkCycleCompactGraph &) = delete;
  DsparkCycleCompactGraph &operator=(const DsparkCycleCompactGraph &) = delete;
  DsparkCycleCompactGraph(DsparkCycleCompactGraph &&other) noexcept;
  DsparkCycleCompactGraph &operator=(DsparkCycleCompactGraph &&) = delete;
  ~DsparkCycleCompactGraph() noexcept;

  [[nodiscard]] static NativeRuntimeResult<DsparkCycleCompactGraph>
  capture(const CudaExecutionContext &context, const GraphArenaLease &arena,
          const DsparkCycleCompactResultBuffers &buffers) noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const CudaCapturedGraph &graph() const noexcept;

private:
  DsparkCycleCompactGraph(CudaCapturedGraph graph,
                          CudaPinnedHostBuffer capture_result) noexcept;

  CudaCapturedGraph graph_;
  CudaPinnedHostBuffer capture_result_;
};

struct DsparkCycleCoreGraphs final {
  cudaGraph_t draft_proposal;
  cudaGraph_t target_verify;
  cudaGraph_t verify_rng;
  cudaGraph_t rejection_sampling;
  cudaGraph_t replayssm_commit;
  cudaGraph_t draft_extend;
  cudaGraph_t kv_write;
  cudaGraph_t compact_result;
};

inline constexpr uint64_t kDsparkModelGraphResourceOwnerDescriptorV1 =
    0x31524f4750534453ULL; // Little-endian bytes: "SDSPGOR1".

enum class DsparkModelGraphKind : uint32_t {
  kInvalid = 0,
  kTargetVerify = 1,
  kDraftExtend = 2,
  kKvWrite = 3,
};

enum class DsparkModelGraphOutput : uint64_t {
  kNone = 0,
  kProposalLayout = 1ULL << 0U,
  kTargetProbabilities = 1ULL << 1U,
  kTargetHidden = 1ULL << 2U,
  kTargetKv = 1ULL << 3U,
  kReplaySsmInputs = 1ULL << 4U,
  kDraftHidden = 1ULL << 5U,
  kNextBaseLogits = 1ULL << 6U,
  kDraftKv = 1ULL << 7U,
  kPrefixValidCommit = 1ULL << 8U,
};

[[nodiscard]] constexpr uint64_t
operator|(DsparkModelGraphOutput lhs, DsparkModelGraphOutput rhs) noexcept {
  return static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs);
}
[[nodiscard]] constexpr uint64_t
operator|(uint64_t lhs, DsparkModelGraphOutput rhs) noexcept {
  return lhs | static_cast<uint64_t>(rhs);
}

struct DsparkModelGraphProductionShape final {
  uint32_t batch_size;
  uint32_t gamma;
  uint32_t verify_width;
  uint32_t vocabulary_size;
  uint32_t hidden_size;
  uint32_t target_capture_layers;
  uint32_t target_hidden_width;
  uint32_t reserved;
};

struct DsparkModelGraphResourceOwnerState;

// A copyable lease over model weights, KV pools, backend workspaces, and other
// stable resources that a captured model graph addresses outside the cycle IO
// arena. retained_resources may be any non-null shared owner for that complete
// resource aggregate; a no-op-deleter observer is not an ownership contract.
// The controller retains one lease for every model stage after cloning the
// CUDA graph, so callers may close the source graph and drop their lease.
class DsparkModelGraphResourceLease final {
public:
  DsparkModelGraphResourceLease() noexcept = default;
  DsparkModelGraphResourceLease(
      const DsparkModelGraphResourceLease &) noexcept = default;
  DsparkModelGraphResourceLease(DsparkModelGraphResourceLease &&) noexcept =
      default;
  DsparkModelGraphResourceLease &
  operator=(const DsparkModelGraphResourceLease &) noexcept = default;
  DsparkModelGraphResourceLease &
  operator=(DsparkModelGraphResourceLease &&) noexcept = default;
  ~DsparkModelGraphResourceLease() noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] uint64_t descriptor() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;
  [[nodiscard]] const void *cycle_arena_identity() const noexcept;
  [[nodiscard]] const void *model_storage_identity() const noexcept;
  [[nodiscard]] const void *resource_identity() const noexcept;
  [[nodiscard]] std::shared_ptr<const void> retain() const noexcept;

private:
  explicit DsparkModelGraphResourceLease(
      std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept;

  std::shared_ptr<const DsparkModelGraphResourceOwnerState> state_;

  friend class DsparkModelGraphResourceOwner;
  friend class DsparkModelStorageArena;
};

class DsparkModelGraphResourceOwner final {
public:
  DsparkModelGraphResourceOwner() noexcept = default;
  DsparkModelGraphResourceOwner(const DsparkModelGraphResourceOwner &) = delete;
  DsparkModelGraphResourceOwner &
  operator=(const DsparkModelGraphResourceOwner &) = delete;
  DsparkModelGraphResourceOwner(DsparkModelGraphResourceOwner &&) noexcept =
      default;
  DsparkModelGraphResourceOwner &
  operator=(DsparkModelGraphResourceOwner &&) noexcept = default;
  ~DsparkModelGraphResourceOwner() noexcept = default;

  [[nodiscard]] static NativeRuntimeResult<DsparkModelGraphResourceOwner>
  create(const CudaExecutionContext &context,
         const GraphArenaLease &cycle_arena,
         std::shared_ptr<const void> retained_resources) noexcept;
  [[nodiscard]] DsparkModelGraphResourceLease acquire_lease() const noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] uint64_t active_leases() const noexcept;
  [[nodiscard]] const void *resource_identity() const noexcept;

private:
  [[nodiscard]] static NativeRuntimeResult<DsparkModelGraphResourceOwner>
  create_with_model_storage(
      const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
      const GraphArenaLease &model_storage,
      std::shared_ptr<const void> retained_resources) noexcept;
  explicit DsparkModelGraphResourceOwner(
      std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept;

  std::shared_ptr<const DsparkModelGraphResourceOwnerState> state_;

  friend class DsparkModelStorageArena;
};

class DsparkModelStorageArena;

// One model-owned CUDA allocation arena.  Every weight, cache, recurrent-state,
// or workspace view supplied to a model graph must originate from this lease.
// Separating it from the cycle IO arena lets validation prove both lifetimes.
class DsparkModelStorageArena final {
public:
  DsparkModelStorageArena(const DsparkModelStorageArena &) = delete;
  DsparkModelStorageArena &operator=(const DsparkModelStorageArena &) = delete;
  DsparkModelStorageArena(DsparkModelStorageArena &&) noexcept = default;
  DsparkModelStorageArena &operator=(DsparkModelStorageArena &&) = delete;
  ~DsparkModelStorageArena() noexcept = default;

  [[nodiscard]] static NativeRuntimeResult<DsparkModelStorageArena>
  create(const CudaExecutionContext &context,
         const GraphArenaLease &cycle_arena, GraphArenaLease storage_arena,
         std::shared_ptr<const void> retained_resources) noexcept;
  [[nodiscard]] DsparkModelGraphResourceOwner model_resources() const noexcept;
  [[nodiscard]] const GraphArenaLease &storage() const noexcept;
  [[nodiscard]] bool valid() const noexcept;

private:
  DsparkModelStorageArena(
      GraphArenaLease storage_arena,
      std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept;

  GraphArenaLease storage_arena_;
  std::shared_ptr<const DsparkModelGraphResourceOwnerState> state_;
};

struct DsparkModelGraphBinding final {
  DsparkModelGraphKind kind;
  uint32_t reserved;
  uint64_t published_outputs;
  DsparkModelGraphProductionShape shape;
  const CudaCapturedGraph &graph;
  DsparkModelGraphResourceLease resources;
};

[[nodiscard]] constexpr DsparkModelGraphProductionShape
dspark_production_model_graph_shape() noexcept {
  return DsparkModelGraphProductionShape{kDsparkProductionBatchSize,
                                         kDsparkProductionGamma,
                                         kDsparkProductionNumVerifyTokens,
                                         kDsparkProductionVocabSize,
                                         5120U,
                                         kDsparkProductionTargetLayerCount,
                                         25600U,
                                         0U};
}
[[nodiscard]] constexpr uint64_t
dspark_required_model_graph_outputs(DsparkModelGraphKind kind) noexcept {
  switch (kind) {
  case DsparkModelGraphKind::kTargetVerify:
    return DsparkModelGraphOutput::kProposalLayout |
           DsparkModelGraphOutput::kTargetProbabilities |
           DsparkModelGraphOutput::kTargetHidden |
           DsparkModelGraphOutput::kTargetKv |
           DsparkModelGraphOutput::kReplaySsmInputs;
  case DsparkModelGraphKind::kDraftExtend:
    return DsparkModelGraphOutput::kDraftHidden |
           DsparkModelGraphOutput::kNextBaseLogits;
  case DsparkModelGraphKind::kKvWrite:
    return DsparkModelGraphOutput::kDraftKv |
           DsparkModelGraphOutput::kPrefixValidCommit;
  default:
    return 0;
  }
}
[[nodiscard]] const char *
dspark_model_graph_kind_name(DsparkModelGraphKind kind) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_model_graph_binding(
    const DsparkModelGraphBinding &binding, DsparkModelGraphKind required_kind,
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const void *required_resource_identity = nullptr) noexcept;

// Target verify, draft extend, and KV write are supplied by model-specific AOT
// code. Unlike an unowned cudaGraph_t, each source records its production
// shape, required output publication, cycle-IO provenance, and retained model
// owner.
struct DsparkTargetVerifyGraphSource final {
  DsparkModelGraphBinding binding;
};

struct DsparkDraftExtendGraphSource final {
  DsparkModelGraphBinding binding;
};

struct DsparkKvWriteGraphSource final {
  DsparkModelGraphBinding binding;
};

struct DsparkCycleGraphSources final {
  const CudaCapturedGraph &draft_proposal;
  DsparkTargetVerifyGraphSource target_verify;
  const CudaCapturedGraph &verify_rng;
  const CudaCapturedGraph &rejection_sampling;
  const CudaCapturedGraph &replayssm_commit;
  DsparkDraftExtendGraphSource draft_extend;
  DsparkKvWriteGraphSource kv_write;
  const DsparkCycleCompactGraph &compact_result;
};

[[nodiscard]] const char *
dspark_cycle_stage_name(DsparkCycleStage stage) noexcept;
[[nodiscard]] const char *
dspark_cycle_argument_name(DsparkCycleArgument argument) noexcept;
[[nodiscard]] const char *
dspark_cycle_device_code_name(DsparkCycleDeviceCode code) noexcept;
[[nodiscard]] NativeRuntimeError validate_dspark_cycle_descriptor(
    const DsparkCycleDescriptor &descriptor) noexcept;
[[nodiscard]] std::array<DsparkCycleChildGraph, kDsparkCycleNumStages>
make_dspark_cycle_child_graphs(const DsparkCycleCoreGraphs &graphs) noexcept;

// Publishes the only host-visible cycle summary.  A prior nonzero status is
// copied through without consuming acceptance outputs.  On success,
// output_token is the bonus token used to seed the next DSpark proposal.
[[nodiscard]] NativeRuntimeError launch_dspark_cycle_compact_result(
    const CudaExecutionContext &context,
    const DsparkCycleCompactResultBuffers &buffers) noexcept;
class DsparkCycleController final {
public:
  DsparkCycleController(const DsparkCycleController &) = delete;
  DsparkCycleController &operator=(const DsparkCycleController &) = delete;
  DsparkCycleController(DsparkCycleController &&other) noexcept;
  DsparkCycleController &operator=(DsparkCycleController &&) = delete;
  ~DsparkCycleController() noexcept;

  // Low-level graph-plumbing escape hatch for isolated controller tests only.
  // Production callers must use the typed source overload below so model
  // resources and required publications are validated and retained.
  [[nodiscard]] static NativeRuntimeResult<DsparkCycleController>
  create_unsafe_unowned_for_test(const DsparkCycleDescriptor &descriptor,
                                 const CudaExecutionContext &context,
                                 const GraphArenaLease &arena) noexcept;
  [[nodiscard]] static NativeRuntimeResult<DsparkCycleController>
  create(const DsparkCycleGraphSources &sources, void *device_status,
         const void *device_result, const CudaExecutionContext &context,
         const GraphArenaLease &arena) noexcept;
  [[nodiscard]] NativeRuntimeError launch() noexcept;
  [[nodiscard]] NativeRuntimeError synchronize() noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] const DsparkCycleResultV1 *result() const noexcept;

private:
  [[nodiscard]] static NativeRuntimeResult<DsparkCycleController> create_owned(
      const DsparkCycleDescriptor &descriptor,
      const CudaExecutionContext &context, const GraphArenaLease &arena,
      std::array<DsparkModelGraphResourceLease, 3> model_resources) noexcept;
  DsparkCycleController(
      std::array<cudaGraph_t, kDsparkCycleMaxOwnedGraphs> graphs,
      uint32_t graph_count, CudaGraphExecutable executable,
      CudaPinnedHostBuffer host_result,
      std::array<DsparkModelGraphResourceLease, 3> model_resources) noexcept;
  void destroy_source_graphs() noexcept;

  std::array<cudaGraph_t, kDsparkCycleMaxOwnedGraphs> graphs_;
  uint32_t graph_count_;
  CudaGraphExecutable executable_;
  CudaPinnedHostBuffer host_result_;
  std::array<DsparkModelGraphResourceLease, 3> model_resources_;
};

static_assert(sizeof(DsparkCycleChildGraph) == 16);
static_assert(std::is_standard_layout_v<DsparkCycleChildGraph>);
static_assert(std::is_trivially_copyable_v<DsparkCycleChildGraph>);
static_assert(sizeof(DsparkCycleResultV1) == 24);
static_assert(sizeof(DsparkCycleResultV1) == 6 * sizeof(uint32_t));
static_assert(std::is_standard_layout_v<DsparkCycleResultV1>);
static_assert(std::is_trivially_copyable_v<DsparkCycleResultV1>);
static_assert(std::is_standard_layout_v<DsparkCycleCoreGraphs>);
static_assert(std::is_trivially_copyable_v<DsparkCycleCoreGraphs>);
static_assert(sizeof(DsparkModelGraphProductionShape) == 32);
static_assert(std::is_standard_layout_v<DsparkModelGraphProductionShape>);
static_assert(std::is_trivially_copyable_v<DsparkModelGraphProductionShape>);
static_assert(static_cast<uint32_t>(DsparkCycleDeviceCode::kOk) == 0);

} // namespace sglang::native

#endif // SGLANG_NATIVE_DSPARK_CYCLE_CONTROLLER_HPP_
