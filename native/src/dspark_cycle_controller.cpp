#include "sglang/native/dspark_cycle_controller.hpp"

#include <array>
#include <exception>
#include <memory>
#include <optional>
#include <utility>

namespace sglang::native {

struct DsparkModelGraphResourceOwnerState final {
  uint64_t descriptor;
  int32_t device_ordinal;
  const void *cycle_arena_identity;
  const void *model_storage_identity;
  std::shared_ptr<const void> retained_resources;
};

namespace {

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           DsparkCycleArgument argument = DsparkCycleArgument::kNone,
           uint64_t actual = 0, uint64_t required = 0) noexcept {
  return NativeRuntimeError{
      code, operation, 0, static_cast<uint32_t>(argument), actual, required};
}

void destroy_graphs(std::array<cudaGraph_t, kDsparkCycleMaxOwnedGraphs> *graphs,
                    uint32_t *graph_count) noexcept {
  for (uint32_t index = 0; index < *graph_count; ++index) {
    if ((*graphs)[index] != nullptr &&
        cudaGraphDestroy((*graphs)[index]) != cudaSuccess) {
      std::terminate();
    }
    (*graphs)[index] = nullptr;
  }
  *graph_count = 0;
}

} // namespace

DsparkModelGraphResourceLease::DsparkModelGraphResourceLease(
    std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept
    : state_(std::move(state)) {}

bool DsparkModelGraphResourceLease::valid() const noexcept {
  return state_ != nullptr &&
         state_->descriptor == kDsparkModelGraphResourceOwnerDescriptorV1 &&
         state_->device_ordinal >= 0 &&
         state_->cycle_arena_identity != nullptr &&
         state_->model_storage_identity != nullptr &&
         state_->retained_resources != nullptr;
}

uint64_t DsparkModelGraphResourceLease::descriptor() const noexcept {
  return state_ == nullptr ? 0 : state_->descriptor;
}

int32_t DsparkModelGraphResourceLease::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

const void *
DsparkModelGraphResourceLease::cycle_arena_identity() const noexcept {
  return state_ == nullptr ? nullptr : state_->cycle_arena_identity;
}

const void *DsparkModelGraphResourceLease::resource_identity() const noexcept {
  return state_.get();
}

const void *
DsparkModelGraphResourceLease::model_storage_identity() const noexcept {
  return state_ == nullptr ? nullptr : state_->model_storage_identity;
}

std::shared_ptr<const void>
DsparkModelGraphResourceLease::retain() const noexcept {
  return state_;
}

DsparkModelGraphResourceOwner::DsparkModelGraphResourceOwner(
    std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept
    : state_(std::move(state)) {}

NativeRuntimeResult<DsparkModelGraphResourceOwner>
DsparkModelGraphResourceOwner::create(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    std::shared_ptr<const void> retained_resources) noexcept {
  // Compatibility constructor for synthetic controller fixtures whose model
  // marker graph addresses only cycle-arena storage. Real model graph factories
  // must use DsparkModelStorageArena and a distinct model allocation arena.
  return create_with_model_storage(context, cycle_arena, cycle_arena,
                                   std::move(retained_resources));
}

NativeRuntimeResult<DsparkModelGraphResourceOwner>
DsparkModelGraphResourceOwner::create_with_model_storage(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const GraphArenaLease &model_storage,
    std::shared_ptr<const void> retained_resources) noexcept {
  using OwnerResult = NativeRuntimeResult<DsparkModelGraphResourceOwner>;
  if (!context.valid() || !cycle_arena.valid() || !model_storage.valid() ||
      retained_resources == nullptr) {
    return OwnerResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCreate,
                   DsparkCycleArgument::kResourceOwner));
  }
  if (context.device_ordinal() != cycle_arena.device_ordinal() ||
      context.device_ordinal() != model_storage.device_ordinal()) {
    return OwnerResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCreate,
                   DsparkCycleArgument::kDeviceOrdinal,
                   static_cast<uint64_t>(cycle_arena.device_ordinal()),
                   static_cast<uint64_t>(context.device_ordinal())));
  }
  std::shared_ptr<const DsparkModelGraphResourceOwnerState> state;
  try {
    state = std::make_shared<const DsparkModelGraphResourceOwnerState>(
        DsparkModelGraphResourceOwnerState{
            kDsparkModelGraphResourceOwnerDescriptorV1,
            context.device_ordinal(), cycle_arena.owner_identity(),
            model_storage.owner_identity(), std::move(retained_resources)});
  } catch (...) {
    return OwnerResult::failure(
        make_error(NativeRuntimeCode::kHostAllocationFailed,
                   NativeRuntimeOperation::kGraphCreate,
                   DsparkCycleArgument::kResourceOwner));
  }
  return OwnerResult::success(DsparkModelGraphResourceOwner(std::move(state)));
}

DsparkModelGraphResourceLease
DsparkModelGraphResourceOwner::acquire_lease() const noexcept {
  return DsparkModelGraphResourceLease(state_);
}

bool DsparkModelGraphResourceOwner::valid() const noexcept {
  return acquire_lease().valid();
}

uint64_t DsparkModelGraphResourceOwner::active_leases() const noexcept {
  return state_ == nullptr || state_.use_count() <= 1
             ? 0
             : static_cast<uint64_t>(state_.use_count() - 1);
}

const void *DsparkModelGraphResourceOwner::resource_identity() const noexcept {
  return state_.get();
}

DsparkModelStorageArena::DsparkModelStorageArena(
    GraphArenaLease storage_arena,
    std::shared_ptr<const DsparkModelGraphResourceOwnerState> state) noexcept
    : storage_arena_(std::move(storage_arena)), state_(std::move(state)) {}

NativeRuntimeResult<DsparkModelStorageArena> DsparkModelStorageArena::create(
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    GraphArenaLease storage_arena,
    std::shared_ptr<const void> retained_resources) noexcept {
  using StorageResult = NativeRuntimeResult<DsparkModelStorageArena>;
  if (!context.valid() || !cycle_arena.valid() || !storage_arena.valid() ||
      retained_resources == nullptr ||
      storage_arena.owner_identity() == cycle_arena.owner_identity()) {
    return StorageResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCreate,
                   DsparkCycleArgument::kResourceOwner));
  }
  if (context.device_ordinal() != cycle_arena.device_ordinal() ||
      context.device_ordinal() != storage_arena.device_ordinal()) {
    return StorageResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCreate,
                   DsparkCycleArgument::kDeviceOrdinal));
  }
  return std::move(DsparkModelGraphResourceOwner::create_with_model_storage(
                       context, cycle_arena, storage_arena,
                       std::move(retained_resources)))
      .match(
          [&storage_arena](
              DsparkModelGraphResourceOwner &&owner) noexcept -> StorageResult {
            return StorageResult::success(DsparkModelStorageArena(
                std::move(storage_arena), std::move(owner.state_)));
          },
          [](NativeRuntimeError &&error) noexcept -> StorageResult {
            return StorageResult::failure(std::move(error));
          });
}

DsparkModelGraphResourceOwner
DsparkModelStorageArena::model_resources() const noexcept {
  return DsparkModelGraphResourceOwner(state_);
}

const GraphArenaLease &DsparkModelStorageArena::storage() const noexcept {
  return storage_arena_;
}

bool DsparkModelStorageArena::valid() const noexcept {
  return storage_arena_.valid() &&
         DsparkModelGraphResourceLease(state_).valid() &&
         storage_arena_.owner_identity() == state_->model_storage_identity;
}

const char *dspark_model_graph_kind_name(DsparkModelGraphKind kind) noexcept {
  switch (kind) {
  case DsparkModelGraphKind::kTargetVerify:
    return "target_verify";
  case DsparkModelGraphKind::kDraftExtend:
    return "draft_extend";
  case DsparkModelGraphKind::kKvWrite:
    return "kv_write";
  default:
    return "invalid_dspark_model_graph_kind";
  }
}

NativeRuntimeError validate_dspark_model_graph_binding(
    const DsparkModelGraphBinding &binding, DsparkModelGraphKind required_kind,
    const CudaExecutionContext &context, const GraphArenaLease &cycle_arena,
    const void *required_resource_identity) noexcept {
  if (binding.kind != required_kind || binding.reserved != 0U) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kModelKind,
                      static_cast<uint64_t>(binding.kind),
                      static_cast<uint64_t>(required_kind));
  }
  const DsparkModelGraphProductionShape required_shape =
      dspark_production_model_graph_shape();
  if (binding.shape.batch_size != required_shape.batch_size ||
      binding.shape.gamma != required_shape.gamma ||
      binding.shape.verify_width != required_shape.verify_width ||
      binding.shape.vocabulary_size != required_shape.vocabulary_size ||
      binding.shape.hidden_size != required_shape.hidden_size ||
      binding.shape.target_capture_layers !=
          required_shape.target_capture_layers ||
      binding.shape.target_hidden_width != required_shape.target_hidden_width ||
      binding.shape.reserved != 0U) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kProductionShape);
  }
  constexpr uint64_t kKnownOutputs =
      static_cast<uint64_t>(DsparkModelGraphOutput::kProposalLayout) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kTargetProbabilities) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kTargetHidden) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kTargetKv) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kReplaySsmInputs) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kDraftHidden) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kNextBaseLogits) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kDraftKv) |
      static_cast<uint64_t>(DsparkModelGraphOutput::kPrefixValidCommit);
  const uint64_t required_outputs =
      dspark_required_model_graph_outputs(required_kind);
  if ((binding.published_outputs & required_outputs) != required_outputs ||
      (binding.published_outputs & ~kKnownOutputs) != 0U) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kRequiredOutputs,
                      binding.published_outputs, required_outputs);
  }
  if (!binding.graph.valid()) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kGraph);
  }
  if (!binding.graph.belongs_to(cycle_arena)) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kCycleArena);
  }
  if (binding.graph.retained_owner_identity() !=
      binding.resources.resource_identity()) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kResourceOwner);
  }
  if (binding.graph.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kDeviceOrdinal,
                      static_cast<uint64_t>(binding.graph.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (!binding.resources.valid()) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kResourceOwner);
  }
  // Generic composition accepts compatibility marker graphs whose only
  // addresses live in the cycle arena. Real model factories validate their
  // external tensors against a distinct DsparkModelStorageArena before
  // capture; target verify does so in validate_dspark_target_verify_buffers.
  if (binding.resources.model_storage_identity() == nullptr) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kStableAddressProvenance);
  }
  if (binding.resources.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kDeviceOrdinal,
                      static_cast<uint64_t>(binding.resources.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (binding.resources.cycle_arena_identity() !=
      cycle_arena.owner_identity()) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kStableAddressProvenance);
  }
  if (required_resource_identity != nullptr &&
      binding.resources.resource_identity() != required_resource_identity) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kGraphAddChild,
                      DsparkCycleArgument::kResourceOwner);
  }
  return native_runtime_ok();
}

const char *dspark_cycle_stage_name(DsparkCycleStage stage) noexcept {
  switch (stage) {
  case DsparkCycleStage::kDraftProposal:
    return "draft_proposal";
  case DsparkCycleStage::kTargetVerify:
    return "target_verify";
  case DsparkCycleStage::kVerifyRng:
    return "verify_rng";
  case DsparkCycleStage::kRejectionSampling:
    return "rejection_sampling";
  case DsparkCycleStage::kReplaySsmCommit:
    return "replayssm_commit";
  case DsparkCycleStage::kDraftExtend:
    return "draft_extend";
  case DsparkCycleStage::kKvWrite:
    return "kv_write";
  case DsparkCycleStage::kCompactResult:
    return "compact_result";
  default:
    return "invalid_dspark_cycle_stage";
  }
}

const char *dspark_cycle_argument_name(DsparkCycleArgument argument) noexcept {
  switch (argument) {
  case DsparkCycleArgument::kNone:
    return "none";
  case DsparkCycleArgument::kAbiVersion:
    return "abi_version";
  case DsparkCycleArgument::kStages:
    return "stages";
  case DsparkCycleArgument::kGraph:
    return "graph";
  case DsparkCycleArgument::kDeviceOrdinal:
    return "device_ordinal";
  case DsparkCycleArgument::kDeviceResult:
    return "device_result";
  case DsparkCycleArgument::kOutTokens:
    return "out_tokens";
  case DsparkCycleArgument::kNumCorrectDrafts:
    return "num_correct_drafts";
  case DsparkCycleArgument::kDeviceStatus:
    return "device_status";
  case DsparkCycleArgument::kRequestSlot:
    return "request_slot";
  case DsparkCycleArgument::kResourceOwner:
    return "resource_owner";
  case DsparkCycleArgument::kCycleArena:
    return "cycle_arena";
  case DsparkCycleArgument::kRequiredOutputs:
    return "required_outputs";
  case DsparkCycleArgument::kProductionShape:
    return "production_shape";
  case DsparkCycleArgument::kModelKind:
    return "model_kind";
  case DsparkCycleArgument::kStableAddressProvenance:
    return "stable_address_provenance";
  default:
    return "invalid_dspark_cycle_argument";
  }
}

const char *dspark_cycle_device_code_name(DsparkCycleDeviceCode code) noexcept {
  switch (code) {
  case DsparkCycleDeviceCode::kOk:
    return "ok";
  case DsparkCycleDeviceCode::kNumCorrectDraftsOutOfRange:
    return "num_correct_drafts_out_of_range";
  default:
    return "invalid_dspark_cycle_device_code";
  }
}

std::array<DsparkCycleChildGraph, kDsparkCycleNumStages>
make_dspark_cycle_child_graphs(const DsparkCycleCoreGraphs &graphs) noexcept {
  return {{{DsparkCycleStage::kDraftProposal, 0, graphs.draft_proposal},
           {DsparkCycleStage::kTargetVerify, 0, graphs.target_verify},
           {DsparkCycleStage::kVerifyRng, 0, graphs.verify_rng},
           {DsparkCycleStage::kRejectionSampling, 0, graphs.rejection_sampling},
           {DsparkCycleStage::kReplaySsmCommit, 0, graphs.replayssm_commit},
           {DsparkCycleStage::kDraftExtend, 0, graphs.draft_extend},
           {DsparkCycleStage::kKvWrite, 0, graphs.kv_write},
           {DsparkCycleStage::kCompactResult, 0, graphs.compact_result}}};
}

NativeRuntimeError validate_dspark_cycle_descriptor(
    const DsparkCycleDescriptor &descriptor) noexcept {
  constexpr std::array<DsparkCycleStage, 8> required_stages{
      DsparkCycleStage::kDraftProposal,   DsparkCycleStage::kTargetVerify,
      DsparkCycleStage::kVerifyRng,       DsparkCycleStage::kRejectionSampling,
      DsparkCycleStage::kReplaySsmCommit, DsparkCycleStage::kDraftExtend,
      DsparkCycleStage::kKvWrite,         DsparkCycleStage::kCompactResult};

  if (descriptor.abi_version != kDsparkCycleAbiVersion) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphCreate,
                      DsparkCycleArgument::kAbiVersion, descriptor.abi_version,
                      kDsparkCycleAbiVersion);
  }
  if (descriptor.device_ordinal > static_cast<uint32_t>(INT32_MAX)) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphCreate,
                      DsparkCycleArgument::kDeviceOrdinal,
                      descriptor.device_ordinal, INT32_MAX);
  }
  if (descriptor.stages.size() != required_stages.size()) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphCreate,
                      DsparkCycleArgument::kStages, descriptor.stages.size(),
                      required_stages.size());
  }
  for (uint32_t index = 0; index < required_stages.size(); ++index) {
    const DsparkCycleChildGraph &child = descriptor.stages[index];
    if (child.stage != required_stages[index]) {
      return make_error(NativeRuntimeCode::kInvalidArgument,
                        NativeRuntimeOperation::kGraphAddChild,
                        DsparkCycleArgument::kStages,
                        static_cast<uint32_t>(child.stage),
                        static_cast<uint32_t>(required_stages[index]));
    }
    if (child.reserved != 0U || child.graph == nullptr) {
      return make_error(NativeRuntimeCode::kInvalidArgument,
                        NativeRuntimeOperation::kGraphAddChild,
                        DsparkCycleArgument::kGraph, index, 1);
    }
  }
  if (descriptor.device_result == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphMemcpyParamsSet,
                      DsparkCycleArgument::kDeviceResult);
  }
  if (descriptor.device_status == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphAddMemset,
                      DsparkCycleArgument::kDeviceStatus);
  }
  return native_runtime_ok();
}

DsparkCycleController::DsparkCycleController(
    std::array<cudaGraph_t, kDsparkCycleMaxOwnedGraphs> graphs,
    uint32_t graph_count, CudaGraphExecutable executable,
    CudaPinnedHostBuffer host_result,
    std::array<DsparkModelGraphResourceLease, 3> model_resources) noexcept
    : graphs_(graphs), graph_count_(graph_count),
      executable_(std::move(executable)), host_result_(std::move(host_result)),
      model_resources_(std::move(model_resources)) {}

DsparkCycleController::DsparkCycleController(
    DsparkCycleController &&other) noexcept
    : graphs_(std::exchange(other.graphs_, {})),
      graph_count_(std::exchange(other.graph_count_, 0)),
      executable_(std::move(other.executable_)),
      host_result_(std::move(other.host_result_)),
      model_resources_(std::move(other.model_resources_)) {}

DsparkCycleController::~DsparkCycleController() noexcept {
  if (!is_ok(close())) {
    std::terminate();
  }
}

void DsparkCycleController::destroy_source_graphs() noexcept {
  destroy_graphs(&graphs_, &graph_count_);
}

NativeRuntimeResult<DsparkCycleController> DsparkCycleController::create_owned(
    const DsparkCycleDescriptor &descriptor,
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    std::array<DsparkModelGraphResourceLease, 3> model_resources) noexcept {
  using ControllerResult = NativeRuntimeResult<DsparkCycleController>;
  const NativeRuntimeError validation =
      validate_dspark_cycle_descriptor(descriptor);
  if (!is_ok(validation)) {
    return ControllerResult::failure(validation);
  }
  if (context.device_ordinal() !=
          static_cast<int32_t>(descriptor.device_ordinal) ||
      arena.device_ordinal() !=
          static_cast<int32_t>(descriptor.device_ordinal)) {
    return ControllerResult::failure(make_error(
        NativeRuntimeCode::kDeviceMismatch,
        NativeRuntimeOperation::kGraphCreate,
        DsparkCycleArgument::kDeviceOrdinal, descriptor.device_ordinal,
        static_cast<uint64_t>(context.device_ordinal())));
  }

  std::array<cudaGraph_t, kDsparkCycleMaxOwnedGraphs> graphs{};
  uint32_t graph_count = 0;
  cudaGraph_t status_reset_graph = nullptr;
  cudaError_t graph_status = cudaGraphCreate(&status_reset_graph, 0);
  if (graph_status != cudaSuccess) {
    return ControllerResult::failure(NativeRuntimeError{
        NativeRuntimeCode::kCudaRuntimeFailure,
        NativeRuntimeOperation::kGraphCreate,
        static_cast<int32_t>(graph_status),
        static_cast<uint32_t>(DsparkCycleArgument::kDeviceStatus), 0, 0});
  }
  cudaMemsetParams reset_params{};
  reset_params.dst = descriptor.device_status;
  reset_params.pitch = 0;
  reset_params.value = 0;
  reset_params.elementSize = sizeof(uint32_t);
  reset_params.width = 1;
  reset_params.height = 1;
  cudaGraphNode_t reset_node = nullptr;
  graph_status = cudaGraphAddMemsetNode(&reset_node, status_reset_graph,
                                        nullptr, 0, &reset_params);
  if (graph_status != cudaSuccess) {
    if (cudaGraphDestroy(status_reset_graph) != cudaSuccess) {
      std::terminate();
    }
    return ControllerResult::failure(NativeRuntimeError{
        NativeRuntimeCode::kCudaRuntimeFailure,
        NativeRuntimeOperation::kGraphAddMemset,
        static_cast<int32_t>(graph_status),
        static_cast<uint32_t>(DsparkCycleArgument::kDeviceStatus), 0, 0});
  }
  (void)reset_node;
  graphs[0] = status_reset_graph;
  ++graph_count;
  for (uint32_t index = 0; index < descriptor.stages.size(); ++index) {
    const cudaError_t clone =
        cudaGraphClone(&graphs[index + 1], descriptor.stages[index].graph);
    if (clone != cudaSuccess) {
      destroy_graphs(&graphs, &graph_count);
      return ControllerResult::failure(NativeRuntimeError{
          NativeRuntimeCode::kCudaRuntimeFailure,
          NativeRuntimeOperation::kGraphClone, static_cast<int32_t>(clone),
          static_cast<uint32_t>(DsparkCycleArgument::kGraph), index,
          descriptor.stages.size()});
    }
    ++graph_count;
  }
  std::optional<CudaPinnedHostBuffer> host_result;
  NativeRuntimeError allocation_error = native_runtime_ok();
  std::move(CudaPinnedHostBuffer::allocate(sizeof(DsparkCycleResultV1)))
      .match(
          [&host_result](CudaPinnedHostBuffer &&value) noexcept {
            host_result.emplace(std::move(value));
          },
          [&allocation_error](NativeRuntimeError &&error) noexcept {
            allocation_error = error;
          });
  if (!is_ok(allocation_error)) {
    destroy_graphs(&graphs, &graph_count);
    return ControllerResult::failure(allocation_error);
  }
  auto parent = CudaGraphExecutable::instantiate_child_sequence(
      std::span<const cudaGraph_t>(graphs.data(), graph_count), context, arena);
  return std::move(parent).match(
      [&descriptor, &host_result, &graphs, &graph_count, &model_resources](
          CudaGraphExecutable &&executable) noexcept -> ControllerResult {
        const NativeRuntimeError binding =
            executable.bind_compact_device_to_host_result(
                host_result->data(), descriptor.device_result,
                sizeof(DsparkCycleResultV1));
        if (!is_ok(binding)) {
          if (!is_ok(executable.close())) {
            std::terminate();
          }
          destroy_graphs(&graphs, &graph_count);
          return ControllerResult::failure(binding);
        }
        return ControllerResult::success(DsparkCycleController(
            graphs, std::exchange(graph_count, 0), std::move(executable),
            std::move(*host_result), std::move(model_resources)));
      },
      [&graphs,
       &graph_count](NativeRuntimeError &&error) noexcept -> ControllerResult {
        destroy_graphs(&graphs, &graph_count);
        return ControllerResult::failure(std::move(error));
      });
}

NativeRuntimeResult<DsparkCycleController>
DsparkCycleController::create_unsafe_unowned_for_test(
    const DsparkCycleDescriptor &descriptor,
    const CudaExecutionContext &context,
    const GraphArenaLease &arena) noexcept {
  return create_owned(descriptor, context, arena, {});
}

NativeRuntimeResult<DsparkCycleController>
DsparkCycleController::create(const DsparkCycleGraphSources &sources,
                              void *device_status, const void *device_result,
                              const CudaExecutionContext &context,
                              const GraphArenaLease &arena) noexcept {
  using ControllerResult = NativeRuntimeResult<DsparkCycleController>;
  if (!context.valid() || !arena.valid()) {
    return ControllerResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCreate));
  }
  const std::array<const DsparkModelGraphBinding *, 3> model_bindings{
      &sources.target_verify.binding, &sources.draft_extend.binding,
      &sources.kv_write.binding};
  constexpr std::array<DsparkModelGraphKind, 3> required_kinds{
      DsparkModelGraphKind::kTargetVerify, DsparkModelGraphKind::kDraftExtend,
      DsparkModelGraphKind::kKvWrite};
  const void *required_resource_identity = nullptr;
  std::array<DsparkModelGraphResourceLease, 3> model_resources{};
  for (uint32_t index = 0; index < model_bindings.size(); ++index) {
    const NativeRuntimeError validation = validate_dspark_model_graph_binding(
        *model_bindings[index], required_kinds[index], context, arena,
        required_resource_identity);
    if (!is_ok(validation)) {
      return ControllerResult::failure(validation);
    }
    required_resource_identity =
        model_bindings[index]->resources.resource_identity();
    model_resources[index] = model_bindings[index]->resources;
  }
  const std::array<const CudaCapturedGraph *, kDsparkCycleNumStages>
      source_list{&sources.draft_proposal,
                  &sources.target_verify.binding.graph,
                  &sources.verify_rng,
                  &sources.rejection_sampling,
                  &sources.replayssm_commit,
                  &sources.draft_extend.binding.graph,
                  &sources.kv_write.binding.graph,
                  &sources.compact_result.graph()};
  for (uint32_t index = 0; index < source_list.size(); ++index) {
    const CudaCapturedGraph &source = *source_list[index];
    if (!source.valid() || !source.belongs_to(arena)) {
      return ControllerResult::failure(
          make_error(source.valid() ? NativeRuntimeCode::kForeignSlice
                                    : NativeRuntimeCode::kInvalidArgument,
                     NativeRuntimeOperation::kGraphAddChild,
                     DsparkCycleArgument::kGraph, index, source_list.size()));
    }
    if (source.device_ordinal() != context.device_ordinal()) {
      return ControllerResult::failure(
          make_error(NativeRuntimeCode::kDeviceMismatch,
                     NativeRuntimeOperation::kGraphAddChild,
                     DsparkCycleArgument::kDeviceOrdinal,
                     static_cast<uint64_t>(source.device_ordinal()),
                     static_cast<uint64_t>(context.device_ordinal())));
    }
  }
  const auto children = make_dspark_cycle_child_graphs(DsparkCycleCoreGraphs{
      sources.draft_proposal.graph(),
      sources.target_verify.binding.graph.graph(), sources.verify_rng.graph(),
      sources.rejection_sampling.graph(), sources.replayssm_commit.graph(),
      sources.draft_extend.binding.graph.graph(),
      sources.kv_write.binding.graph.graph(),
      sources.compact_result.graph().graph()});
  return create_owned(
      DsparkCycleDescriptor{kDsparkCycleAbiVersion,
                            static_cast<uint32_t>(context.device_ordinal()),
                            children, device_status, device_result},
      context, arena, std::move(model_resources));
}

NativeRuntimeError DsparkCycleController::launch() noexcept {
  return executable_.launch();
}

NativeRuntimeError DsparkCycleController::synchronize() noexcept {
  return executable_.synchronize();
}

NativeRuntimeError DsparkCycleController::close() noexcept {
  const NativeRuntimeError status = executable_.close();
  if (!is_ok(status)) {
    return status;
  }
  const NativeRuntimeError host_status = host_result_.close();
  if (!is_ok(host_status)) {
    return host_status;
  }
  destroy_source_graphs();
  model_resources_ = {};
  return native_runtime_ok();
}

bool DsparkCycleController::valid() const noexcept {
  return host_result_.valid() && executable_.valid();
}

const DsparkCycleResultV1 *DsparkCycleController::result() const noexcept {
  return valid() ? static_cast<const DsparkCycleResultV1 *>(host_result_.data())
                 : nullptr;
}

} // namespace sglang::native
