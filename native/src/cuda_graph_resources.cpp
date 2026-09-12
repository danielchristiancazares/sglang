#include "sglang/native/cuda_graph_resources.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

namespace sglang::native {
namespace detail {

struct CudaStreamState final {
  std::atomic<uint64_t> context_count;
  cudaStream_t stream;
  int32_t device_ordinal;
};

struct GraphArenaState final {
  std::atomic<uint64_t> lease_count;
  void *allocation_base;
  uint64_t capacity_bytes;
  uint64_t used_bytes;
  int32_t device_ordinal;
  bool sealed;
};

} // namespace detail
namespace {

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           int32_t native_code = 0, uint32_t detail = 0, uint64_t actual = 0,
           uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,   operation, native_code,
                            detail, actual,    required};
}

[[nodiscard]] constexpr NativeRuntimeError
invalid_state(NativeRuntimeOperation operation) noexcept {
  return make_error(NativeRuntimeCode::kInvalidState, operation);
}

[[nodiscard]] constexpr NativeRuntimeError
cuda_failure(NativeRuntimeOperation operation, cudaError_t error) noexcept {
  return make_error(NativeRuntimeCode::kCudaRuntimeFailure, operation,
                    static_cast<int32_t>(error));
}

[[nodiscard]] NativeRuntimeError
require_current_device(int32_t required_device,
                       NativeRuntimeOperation operation) noexcept {
  int current_device = -1;
  const cudaError_t result = cudaGetDevice(&current_device);
  if (result != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kGetDevice, result);
  }
  if (current_device != required_device) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, operation, 0, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(required_device));
  }
  return native_runtime_ok();
}

void retain_context(detail::CudaStreamState *state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->context_count.fetch_add(1, std::memory_order_relaxed);
  if (previous == std::numeric_limits<uint64_t>::max()) {
    std::terminate();
  }
}

void release_context(detail::CudaStreamState *state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->context_count.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 0) {
    std::terminate();
  }
}

void retain_arena(detail::GraphArenaState *state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->lease_count.fetch_add(1, std::memory_order_relaxed);
  if (previous == std::numeric_limits<uint64_t>::max()) {
    std::terminate();
  }
}

void release_arena(detail::GraphArenaState *state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->lease_count.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 0) {
    std::terminate();
  }
}

void require_cleanup_success(NativeRuntimeError status) noexcept {
  if (!is_ok(status)) {
    std::terminate();
  }
}

[[nodiscard]] constexpr bool is_power_of_two(uint64_t value) noexcept {
  return value != 0 && (value & (value - 1)) == 0;
}

[[nodiscard]] NativeRuntimeError
owner_metadata_conflict(TensorValidationField field, uint64_t actual,
                        uint64_t required) noexcept {
  return make_error(NativeRuntimeCode::kOwnerMetadataConflict,
                    NativeRuntimeOperation::kBindTensor, 0,
                    static_cast<uint32_t>(field), actual, required);
}

} // namespace

namespace detail {

NativeRuntimeError tensor_binding_error(TensorValidationError error) noexcept {
  return make_error(
      NativeRuntimeCode::kTensorValidationFailure,
      NativeRuntimeOperation::kBindTensor, static_cast<int32_t>(error.code),
      static_cast<uint32_t>(error.field), error.actual, error.required);
}

} // namespace detail

std::string_view native_runtime_code_name(NativeRuntimeCode code) noexcept {
  switch (code) {
  case NativeRuntimeCode::kOk:
    return "ok";
  case NativeRuntimeCode::kInvalidArgument:
    return "invalid_argument";
  case NativeRuntimeCode::kInvalidState:
    return "invalid_state";
  case NativeRuntimeCode::kHostAllocationFailed:
    return "host_allocation_failed";
  case NativeRuntimeCode::kDeviceMismatch:
    return "device_mismatch";
  case NativeRuntimeCode::kArithmeticOverflow:
    return "arithmetic_overflow";
  case NativeRuntimeCode::kOutOfCapacity:
    return "out_of_capacity";
  case NativeRuntimeCode::kAlreadySealed:
    return "already_sealed";
  case NativeRuntimeCode::kNotSealed:
    return "not_sealed";
  case NativeRuntimeCode::kResourceBusy:
    return "resource_busy";
  case NativeRuntimeCode::kForeignSlice:
    return "foreign_slice";
  case NativeRuntimeCode::kOwnerMetadataConflict:
    return "owner_metadata_conflict";
  case NativeRuntimeCode::kCudaRuntimeFailure:
    return "cuda_runtime_failure";
  case NativeRuntimeCode::kTensorValidationFailure:
    return "tensor_validation_failure";
  default:
    return "invalid_runtime_code";
  }
}

std::string_view
native_runtime_operation_name(NativeRuntimeOperation operation) noexcept {
  switch (operation) {
  case NativeRuntimeOperation::kNone:
    return "none";
  case NativeRuntimeOperation::kGetDevice:
    return "get_device";
  case NativeRuntimeOperation::kStreamCreate:
    return "stream_create";
  case NativeRuntimeOperation::kStreamSynchronize:
    return "stream_synchronize";
  case NativeRuntimeOperation::kStreamDestroy:
    return "stream_destroy";
  case NativeRuntimeOperation::kDeviceAllocate:
    return "device_allocate";
  case NativeRuntimeOperation::kDeviceFree:
    return "device_free";
  case NativeRuntimeOperation::kReserve:
    return "reserve";
  case NativeRuntimeOperation::kSeal:
    return "seal";
  case NativeRuntimeOperation::kAcquireLease:
    return "acquire_lease";
  case NativeRuntimeOperation::kBindTensor:
    return "bind_tensor";
  case NativeRuntimeOperation::kGraphInstantiate:
    return "graph_instantiate";
  case NativeRuntimeOperation::kEventCreate:
    return "event_create";
  case NativeRuntimeOperation::kGraphLaunch:
    return "graph_launch";
  case NativeRuntimeOperation::kEventRecord:
    return "event_record";
  case NativeRuntimeOperation::kEventSynchronize:
    return "event_synchronize";
  case NativeRuntimeOperation::kGraphDestroy:
    return "graph_destroy";
  case NativeRuntimeOperation::kEventDestroy:
    return "event_destroy";
  case NativeRuntimeOperation::kStreamGetFlags:
    return "stream_get_flags";
  case NativeRuntimeOperation::kValidateLinearRejectionSampling:
    return "validate_linear_rejection_sampling";
  case NativeRuntimeOperation::kLaunchLinearRejectionSampling:
    return "launch_linear_rejection_sampling";
  case NativeRuntimeOperation::kValidateLinearVerifyRng:
    return "validate_linear_verify_rng";
  case NativeRuntimeOperation::kLaunchSeededLinearVerifyRng:
    return "launch_seeded_linear_verify_rng";
  case NativeRuntimeOperation::kLaunchStatefulLinearVerifyRng:
    return "launch_stateful_linear_verify_rng";
  case NativeRuntimeOperation::kValidateDsparkProposal:
    return "validate_dspark_proposal";
  case NativeRuntimeOperation::kLaunchDsparkProposal:
    return "launch_dspark_proposal";
  case NativeRuntimeOperation::kValidateGdnReplaySsmCommit:
    return "validate_gdn_replayssm_commit";
  case NativeRuntimeOperation::kLaunchGdnReplaySsmCommit:
    return "launch_gdn_replayssm_commit";
  case NativeRuntimeOperation::kGraphCreate:
    return "graph_create";
  case NativeRuntimeOperation::kGraphAddChild:
    return "graph_add_child";
  case NativeRuntimeOperation::kGraphClone:
    return "graph_clone";
  case NativeRuntimeOperation::kGraphGetNodes:
    return "graph_get_nodes";
  case NativeRuntimeOperation::kGraphNodeType:
    return "graph_node_type";
  case NativeRuntimeOperation::kGraphMemcpyParams:
    return "graph_memcpy_params";
  case NativeRuntimeOperation::kGraphMemcpyParamsSet:
    return "graph_memcpy_params_set";
  case NativeRuntimeOperation::kHostAllocate:
    return "host_allocate";
  case NativeRuntimeOperation::kHostFree:
    return "host_free";
  case NativeRuntimeOperation::kValidateDsparkCycle:
    return "validate_dspark_cycle";
  case NativeRuntimeOperation::kLaunchDsparkCycleCompactResult:
    return "launch_dspark_cycle_compact_result";
  case NativeRuntimeOperation::kGraphAddMemset:
    return "graph_add_memset";
  case NativeRuntimeOperation::kGraphCaptureBegin:
    return "graph_capture_begin";
  case NativeRuntimeOperation::kGraphCaptureEnd:
    return "graph_capture_end";
  case NativeRuntimeOperation::kValidateDsparkTargetVerify:
    return "validate_dspark_target_verify";
  case NativeRuntimeOperation::kLaunchDsparkTargetVerify:
    return "launch_dspark_target_verify";
  default:
    return "invalid_runtime_operation";
  }
}

CudaExecutionContext::CudaExecutionContext(
    detail::CudaStreamState *state) noexcept
    : state_(state) {
  retain_context(state_);
}

CudaExecutionContext::CudaExecutionContext(
    const CudaExecutionContext &other) noexcept
    : state_(other.state_) {
  retain_context(state_);
}

CudaExecutionContext::CudaExecutionContext(
    CudaExecutionContext &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

CudaExecutionContext::~CudaExecutionContext() noexcept {
  release_context(state_);
}

bool CudaExecutionContext::valid() const noexcept { return state_ != nullptr; }

int32_t CudaExecutionContext::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

cudaStream_t CudaExecutionContext::stream() const noexcept {
  return state_ == nullptr ? nullptr : state_->stream;
}

NativeRuntimeResult<unsigned int>
CudaExecutionContext::stream_flags() const noexcept {
  using FlagsResult = NativeRuntimeResult<unsigned int>;
  if (state_ == nullptr) {
    return FlagsResult::failure(
        invalid_state(NativeRuntimeOperation::kStreamGetFlags));
  }
  const NativeRuntimeError current = require_current_device(
      state_->device_ordinal, NativeRuntimeOperation::kStreamGetFlags);
  if (!is_ok(current)) {
    return FlagsResult::failure(current);
  }
  unsigned int flags = 0;
  const cudaError_t result = cudaStreamGetFlags(state_->stream, &flags);
  if (result != cudaSuccess) {
    return FlagsResult::failure(
        cuda_failure(NativeRuntimeOperation::kStreamGetFlags, result));
  }
  return FlagsResult::success(std::move(flags));
}

NativeRuntimeError CudaExecutionContext::synchronize() const noexcept {
  if (state_ == nullptr) {
    return invalid_state(NativeRuntimeOperation::kStreamSynchronize);
  }
  const NativeRuntimeError current = require_current_device(
      state_->device_ordinal, NativeRuntimeOperation::kStreamSynchronize);
  if (!is_ok(current)) {
    return current;
  }
  const cudaError_t result = cudaStreamSynchronize(state_->stream);
  if (result != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kStreamSynchronize, result);
  }
  return native_runtime_ok();
}

CudaStream::CudaStream(detail::CudaStreamState *state) noexcept
    : state_(state) {}

CudaStream::CudaStream(CudaStream &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

CudaStream::~CudaStream() noexcept {
  if (state_ != nullptr) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<CudaStream> CudaStream::create_nonblocking() noexcept {
  using StreamResult = NativeRuntimeResult<CudaStream>;
  int device = -1;
  const cudaError_t get_device = cudaGetDevice(&device);
  if (get_device != cudaSuccess) {
    return StreamResult::failure(
        cuda_failure(NativeRuntimeOperation::kGetDevice, get_device));
  }

  auto *state = new (std::nothrow) detail::CudaStreamState;
  if (state != nullptr) {
    state->context_count.store(0, std::memory_order_relaxed);
    state->stream = nullptr;
    state->device_ordinal = static_cast<int32_t>(device);
    const cudaError_t create =
        cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking);
    if (create != cudaSuccess) {
      delete state;
      return StreamResult::failure(
          cuda_failure(NativeRuntimeOperation::kStreamCreate, create));
    }
    return StreamResult::success(CudaStream(state));
  }
  return StreamResult::failure(
      make_error(NativeRuntimeCode::kHostAllocationFailed,
                 NativeRuntimeOperation::kStreamCreate));
}

NativeRuntimeResult<CudaExecutionContext> CudaStream::context() const noexcept {
  using ContextResult = NativeRuntimeResult<CudaExecutionContext>;
  if (state_ == nullptr) {
    return ContextResult::failure(
        invalid_state(NativeRuntimeOperation::kAcquireLease));
  }
  return ContextResult::success(CudaExecutionContext(state_));
}

NativeRuntimeError CudaStream::close() noexcept {
  if (state_ == nullptr) {
    return native_runtime_ok();
  }
  const uint64_t active = state_->context_count.load(std::memory_order_acquire);
  if (active != 0) {
    return make_error(NativeRuntimeCode::kResourceBusy,
                      NativeRuntimeOperation::kStreamDestroy, 0, 0, active, 0);
  }
  const NativeRuntimeError current = require_current_device(
      state_->device_ordinal, NativeRuntimeOperation::kStreamDestroy);
  if (!is_ok(current)) {
    return current;
  }
  const cudaError_t destroy = cudaStreamDestroy(state_->stream);
  if (destroy != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kStreamDestroy, destroy);
  }
  delete std::exchange(state_, nullptr);
  return native_runtime_ok();
}

bool CudaStream::valid() const noexcept { return state_ != nullptr; }

int32_t CudaStream::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

GraphMemorySlice::GraphMemorySlice(detail::GraphArenaState *state,
                                   uint64_t offset_bytes,
                                   uint64_t size_bytes) noexcept
    : state_(state), offset_bytes_(offset_bytes), size_bytes_(size_bytes) {
  retain_arena(state_);
}

GraphMemorySlice::GraphMemorySlice(const GraphMemorySlice &other) noexcept
    : state_(other.state_), offset_bytes_(other.offset_bytes_),
      size_bytes_(other.size_bytes_) {
  retain_arena(state_);
}

GraphMemorySlice::GraphMemorySlice(GraphMemorySlice &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      offset_bytes_(std::exchange(other.offset_bytes_, 0)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

GraphMemorySlice::~GraphMemorySlice() noexcept { release_arena(state_); }

bool GraphMemorySlice::valid() const noexcept { return state_ != nullptr; }

uint64_t GraphMemorySlice::offset_bytes() const noexcept {
  return offset_bytes_;
}

uint64_t GraphMemorySlice::size_bytes() const noexcept { return size_bytes_; }

GraphArenaLease::GraphArenaLease(detail::GraphArenaState *state) noexcept
    : state_(state) {
  retain_arena(state_);
}

GraphArenaLease::GraphArenaLease(const GraphArenaLease &other) noexcept
    : state_(other.state_) {
  retain_arena(state_);
}

GraphArenaLease::GraphArenaLease(GraphArenaLease &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

GraphArenaLease::~GraphArenaLease() noexcept { release_arena(state_); }

bool GraphArenaLease::valid() const noexcept { return state_ != nullptr; }

int32_t GraphArenaLease::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

uint64_t GraphArenaLease::capacity_bytes() const noexcept {
  return state_ == nullptr ? 0 : state_->capacity_bytes;
}

const void *GraphArenaLease::owner_identity() const noexcept { return state_; }

NativeRuntimeError
GraphArenaLease::prepare_binding(const GraphMemorySlice &slice,
                                 SglNativeTensorMetadataV1 *metadata,
                                 const void **allocation_base) const noexcept {
  if (!valid() || !slice.valid() || metadata == nullptr ||
      allocation_base == nullptr) {
    return invalid_state(NativeRuntimeOperation::kBindTensor);
  }
  if (state_ != slice.state_) {
    return make_error(NativeRuntimeCode::kForeignSlice,
                      NativeRuntimeOperation::kBindTensor);
  }
  if (!state_->sealed) {
    return make_error(NativeRuntimeCode::kNotSealed,
                      NativeRuntimeOperation::kBindTensor);
  }
  if (metadata->device_kind != SGL_NATIVE_DEVICE_INVALID) {
    return owner_metadata_conflict(TensorValidationField::kDeviceKind,
                                   metadata->device_kind,
                                   SGL_NATIVE_DEVICE_INVALID);
  }
  if (metadata->device_ordinal != 0) {
    return owner_metadata_conflict(
        TensorValidationField::kDeviceOrdinal,
        static_cast<uint64_t>(metadata->device_ordinal), 0);
  }
  if (metadata->allocation_bytes != 0) {
    return owner_metadata_conflict(TensorValidationField::kAllocationBytes,
                                   metadata->allocation_bytes, 0);
  }

  metadata->device_kind = SGL_NATIVE_DEVICE_CUDA;
  metadata->device_ordinal = state_->device_ordinal;
  metadata->allocation_bytes = slice.size_bytes_;
  *allocation_base = static_cast<const std::byte *>(state_->allocation_base) +
                     slice.offset_bytes_;
  return native_runtime_ok();
}

NativeRuntimeError GraphArenaLease::prepare_const_binding(
    const GraphMemorySlice &slice, SglNativeTensorMetadataV1 metadata,
    SglNativeConstTensorViewV1 *raw) const noexcept {
  if (raw == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kBindTensor);
  }
  const void *allocation_base = nullptr;
  const NativeRuntimeError prepared =
      prepare_binding(slice, &metadata, &allocation_base);
  if (!is_ok(prepared)) {
    return prepared;
  }
  *raw = SglNativeConstTensorViewV1{metadata, allocation_base};
  return native_runtime_ok();
}

NativeRuntimeError GraphArenaLease::prepare_mutable_binding(
    const GraphMemorySlice &slice, SglNativeTensorMetadataV1 metadata,
    SglNativeMutableTensorViewV1 *raw) const noexcept {
  if (raw == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kBindTensor);
  }
  const void *allocation_base = nullptr;
  const NativeRuntimeError prepared =
      prepare_binding(slice, &metadata, &allocation_base);
  if (!is_ok(prepared)) {
    return prepared;
  }
  *raw = SglNativeMutableTensorViewV1{metadata,
                                      const_cast<void *>(allocation_base)};
  return native_runtime_ok();
}

GraphMemoryArena::GraphMemoryArena(detail::GraphArenaState *state) noexcept
    : state_(state) {}

GraphMemoryArena::GraphMemoryArena(GraphMemoryArena &&other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

GraphMemoryArena::~GraphMemoryArena() noexcept {
  if (state_ != nullptr) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<GraphMemoryArena>
GraphMemoryArena::allocate(const CudaExecutionContext &context,
                           uint64_t capacity_bytes) noexcept {
  using ArenaResult = NativeRuntimeResult<GraphMemoryArena>;
  if (!context.valid() || capacity_bytes == 0 ||
      capacity_bytes >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      capacity_bytes >
          static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ArenaResult::failure(make_error(
        NativeRuntimeCode::kInvalidArgument,
        NativeRuntimeOperation::kDeviceAllocate, 0, 0, capacity_bytes,
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())));
  }
  const NativeRuntimeError current = require_current_device(
      context.device_ordinal(), NativeRuntimeOperation::kDeviceAllocate);
  if (!is_ok(current)) {
    return ArenaResult::failure(current);
  }

  auto *state = new (std::nothrow) detail::GraphArenaState;
  if (state != nullptr) {
    state->lease_count.store(0, std::memory_order_relaxed);
    state->allocation_base = nullptr;
    state->capacity_bytes = capacity_bytes;
    state->used_bytes = 0;
    state->device_ordinal = context.device_ordinal();
    state->sealed = false;
    const cudaError_t allocate = cudaMalloc(
        &state->allocation_base, static_cast<std::size_t>(capacity_bytes));
    if (allocate != cudaSuccess) {
      delete state;
      return ArenaResult::failure(
          cuda_failure(NativeRuntimeOperation::kDeviceAllocate, allocate));
    }
    return ArenaResult::success(GraphMemoryArena(state));
  }
  return ArenaResult::failure(
      make_error(NativeRuntimeCode::kHostAllocationFailed,
                 NativeRuntimeOperation::kDeviceAllocate));
}

NativeRuntimeResult<GraphMemorySlice>
GraphMemoryArena::reserve(uint64_t size_bytes,
                          uint64_t alignment_bytes) noexcept {
  using SliceResult = NativeRuntimeResult<GraphMemorySlice>;
  if (state_ == nullptr) {
    return SliceResult::failure(
        invalid_state(NativeRuntimeOperation::kReserve));
  }
  if (state_->sealed) {
    return SliceResult::failure(make_error(NativeRuntimeCode::kAlreadySealed,
                                           NativeRuntimeOperation::kReserve));
  }
  if (size_bytes == 0 || !is_power_of_two(alignment_bytes) ||
      alignment_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return SliceResult::failure(make_error(NativeRuntimeCode::kInvalidArgument,
                                           NativeRuntimeOperation::kReserve, 0,
                                           0, alignment_bytes, 1));
  }

  const uintptr_t base = reinterpret_cast<uintptr_t>(state_->allocation_base);
  if (state_->used_bytes >
      static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - base)) {
    return SliceResult::failure(
        make_error(NativeRuntimeCode::kArithmeticOverflow,
                   NativeRuntimeOperation::kReserve));
  }
  const uintptr_t current = base + static_cast<uintptr_t>(state_->used_bytes);
  const uintptr_t mask = static_cast<uintptr_t>(alignment_bytes - 1);
  if (current > std::numeric_limits<uintptr_t>::max() - mask) {
    return SliceResult::failure(
        make_error(NativeRuntimeCode::kArithmeticOverflow,
                   NativeRuntimeOperation::kReserve));
  }
  const uintptr_t aligned = (current + mask) & ~mask;
  const uint64_t offset = static_cast<uint64_t>(aligned - base);
  if (offset > state_->capacity_bytes ||
      size_bytes > state_->capacity_bytes - offset) {
    uint64_t required = std::numeric_limits<uint64_t>::max();
    if (size_bytes <= std::numeric_limits<uint64_t>::max() - offset) {
      required = offset + size_bytes;
    }
    return SliceResult::failure(make_error(
        NativeRuntimeCode::kOutOfCapacity, NativeRuntimeOperation::kReserve, 0,
        0, required, state_->capacity_bytes));
  }

  state_->used_bytes = offset + size_bytes;
  return SliceResult::success(GraphMemorySlice(state_, offset, size_bytes));
}

NativeRuntimeError GraphMemoryArena::seal() noexcept {
  if (state_ == nullptr) {
    return invalid_state(NativeRuntimeOperation::kSeal);
  }
  if (state_->sealed) {
    return make_error(NativeRuntimeCode::kAlreadySealed,
                      NativeRuntimeOperation::kSeal);
  }
  state_->sealed = true;
  return native_runtime_ok();
}

NativeRuntimeResult<GraphArenaLease>
GraphMemoryArena::acquire_lease() const noexcept {
  using LeaseResult = NativeRuntimeResult<GraphArenaLease>;
  if (state_ == nullptr) {
    return LeaseResult::failure(
        invalid_state(NativeRuntimeOperation::kAcquireLease));
  }
  if (!state_->sealed) {
    return LeaseResult::failure(make_error(
        NativeRuntimeCode::kNotSealed, NativeRuntimeOperation::kAcquireLease));
  }
  return LeaseResult::success(GraphArenaLease(state_));
}

NativeRuntimeError GraphMemoryArena::close() noexcept {
  if (state_ == nullptr) {
    return native_runtime_ok();
  }
  const uint64_t active = state_->lease_count.load(std::memory_order_acquire);
  if (active != 0) {
    return make_error(NativeRuntimeCode::kResourceBusy,
                      NativeRuntimeOperation::kDeviceFree, 0, 0, active, 0);
  }
  const NativeRuntimeError current = require_current_device(
      state_->device_ordinal, NativeRuntimeOperation::kDeviceFree);
  if (!is_ok(current)) {
    return current;
  }
  const cudaError_t release = cudaFree(state_->allocation_base);
  if (release != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kDeviceFree, release);
  }
  delete std::exchange(state_, nullptr);
  return native_runtime_ok();
}

bool GraphMemoryArena::valid() const noexcept { return state_ != nullptr; }

bool GraphMemoryArena::sealed() const noexcept {
  return state_ != nullptr && state_->sealed;
}

int32_t GraphMemoryArena::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

uint64_t GraphMemoryArena::capacity_bytes() const noexcept {
  return state_ == nullptr ? 0 : state_->capacity_bytes;
}

uint64_t GraphMemoryArena::used_bytes() const noexcept {
  return state_ == nullptr ? 0 : state_->used_bytes;
}

CudaPinnedHostBuffer::CudaPinnedHostBuffer(void *data,
                                           uint64_t size_bytes) noexcept
    : data_(data), size_bytes_(size_bytes) {}

CudaPinnedHostBuffer::CudaPinnedHostBuffer(
    CudaPinnedHostBuffer &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

CudaPinnedHostBuffer::~CudaPinnedHostBuffer() noexcept {
  if (data_ != nullptr) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<CudaPinnedHostBuffer>
CudaPinnedHostBuffer::allocate(uint64_t size_bytes) noexcept {
  using BufferResult = NativeRuntimeResult<CudaPinnedHostBuffer>;
  if (size_bytes == 0 || size_bytes > std::numeric_limits<size_t>::max()) {
    return BufferResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kHostAllocate, 0, 0, size_bytes, 1));
  }
  void *data = nullptr;
  const cudaError_t status =
      cudaMallocHost(&data, static_cast<size_t>(size_bytes));
  if (status != cudaSuccess) {
    return BufferResult::failure(
        cuda_failure(NativeRuntimeOperation::kHostAllocate, status));
  }
  return BufferResult::success(CudaPinnedHostBuffer(data, size_bytes));
}

NativeRuntimeError CudaPinnedHostBuffer::close() noexcept {
  if (data_ == nullptr) {
    return native_runtime_ok();
  }
  const cudaError_t status = cudaFreeHost(data_);
  if (status != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kHostFree, status);
  }
  data_ = nullptr;
  size_bytes_ = 0;
  return native_runtime_ok();
}

bool CudaPinnedHostBuffer::valid() const noexcept { return data_ != nullptr; }

void *CudaPinnedHostBuffer::data() const noexcept { return data_; }

uint64_t CudaPinnedHostBuffer::size_bytes() const noexcept {
  return size_bytes_;
}

CudaCapturedGraph::CudaCapturedGraph(
    cudaGraph_t graph, CudaExecutionContext context, GraphArenaLease arena,
    std::shared_ptr<const void> retained_owner) noexcept
    : graph_(graph), context_(std::move(context)), arena_(std::move(arena)),
      retained_owner_(std::move(retained_owner)) {}

CudaCapturedGraph::CudaCapturedGraph(CudaCapturedGraph &&other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      context_(std::move(other.context_)), arena_(std::move(other.arena_)),
      retained_owner_(std::move(other.retained_owner_)) {
  other.context_.reset();
  other.arena_.reset();
}

CudaCapturedGraph::~CudaCapturedGraph() noexcept {
  if (graph_ != nullptr || context_.has_value() || arena_.has_value()) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<CudaCapturedGraph>
CudaCapturedGraph::capture(const CudaExecutionContext &context,
                           const GraphArenaLease &arena, CaptureBody body,
                           void *user_data) noexcept {
  return capture_impl(context, arena, {}, body, user_data);
}

NativeRuntimeResult<CudaCapturedGraph> CudaCapturedGraph::capture_retaining(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    std::shared_ptr<const void> retained_owner, CaptureBody body,
    void *user_data) noexcept {
  using GraphResult = NativeRuntimeResult<CudaCapturedGraph>;
  if (retained_owner == nullptr) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCaptureBegin));
  }
  return capture_impl(context, arena, std::move(retained_owner), body,
                      user_data);
}

NativeRuntimeResult<CudaCapturedGraph>
CudaCapturedGraph::capture_impl(const CudaExecutionContext &context,
                                const GraphArenaLease &arena,
                                std::shared_ptr<const void> retained_owner,
                                CaptureBody body, void *user_data) noexcept {
  using GraphResult = NativeRuntimeResult<CudaCapturedGraph>;
  if (!context.valid() || !arena.valid() || body == nullptr) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCaptureBegin));
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCaptureBegin, 0, 0,
                   static_cast<uint64_t>(context.device_ordinal()),
                   static_cast<uint64_t>(arena.device_ordinal())));
  }
  const NativeRuntimeError current = require_current_device(
      context.device_ordinal(), NativeRuntimeOperation::kGraphCaptureBegin);
  if (!is_ok(current)) {
    return GraphResult::failure(current);
  }

  cudaError_t status = cudaStreamBeginCapture(context.stream(),
                                              cudaStreamCaptureModeThreadLocal);
  if (status != cudaSuccess) {
    return GraphResult::failure(
        cuda_failure(NativeRuntimeOperation::kGraphCaptureBegin, status));
  }
  const NativeRuntimeError body_status = body(user_data);
  cudaGraph_t graph = nullptr;
  status = cudaStreamEndCapture(context.stream(), &graph);
  if (!is_ok(body_status)) {
    if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) {
      std::terminate();
    }
    return GraphResult::failure(body_status);
  }
  if (status != cudaSuccess || graph == nullptr) {
    if (graph != nullptr && cudaGraphDestroy(graph) != cudaSuccess) {
      std::terminate();
    }
    return GraphResult::failure(
        status == cudaSuccess
            ? invalid_state(NativeRuntimeOperation::kGraphCaptureEnd)
            : cuda_failure(NativeRuntimeOperation::kGraphCaptureEnd, status));
  }
  return GraphResult::success(
      CudaCapturedGraph(graph, CudaExecutionContext(context),
                        GraphArenaLease(arena), std::move(retained_owner)));
}

NativeRuntimeError CudaCapturedGraph::close() noexcept {
  if (graph_ == nullptr && !context_.has_value() && !arena_.has_value()) {
    return native_runtime_ok();
  }
  if (graph_ == nullptr || !context_.has_value() || !arena_.has_value()) {
    return invalid_state(NativeRuntimeOperation::kGraphDestroy);
  }
  const NativeRuntimeError current = require_current_device(
      context_->device_ordinal(), NativeRuntimeOperation::kGraphDestroy);
  if (!is_ok(current)) {
    return current;
  }
  const cudaError_t destroy = cudaGraphDestroy(graph_);
  if (destroy != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kGraphDestroy, destroy);
  }
  graph_ = nullptr;
  arena_.reset();
  context_.reset();
  retained_owner_.reset();
  return native_runtime_ok();
}

bool CudaCapturedGraph::valid() const noexcept {
  return graph_ != nullptr && context_.has_value() && context_->valid() &&
         arena_.has_value() && arena_->valid();
}

int32_t CudaCapturedGraph::device_ordinal() const noexcept {
  return context_.has_value() ? context_->device_ordinal() : -1;
}

cudaGraph_t CudaCapturedGraph::graph() const noexcept {
  return valid() ? graph_ : nullptr;
}

bool CudaCapturedGraph::belongs_to(
    const GraphArenaLease &arena) const noexcept {
  return valid() && arena_.has_value() && arena_->state_ != nullptr &&
         arena_->state_ == arena.state_;
}

const void *CudaCapturedGraph::retained_owner_identity() const noexcept {
  return valid() ? retained_owner_.get() : nullptr;
}

CudaGraphExecutable::CudaGraphExecutable(cudaGraph_t graph,
                                         cudaGraphExec_t executable,
                                         cudaEvent_t completion_event,
                                         CudaExecutionContext context,
                                         GraphArenaLease arena) noexcept
    : graph_(graph), executable_(executable),
      completion_event_(completion_event), context_(std::move(context)),
      arena_(std::move(arena)), in_flight_(false), completion_recorded_(false) {
}

CudaGraphExecutable::CudaGraphExecutable(CudaGraphExecutable &&other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)),
      executable_(std::exchange(other.executable_, nullptr)),
      completion_event_(std::exchange(other.completion_event_, nullptr)),
      context_(std::move(other.context_)), arena_(std::move(other.arena_)),
      in_flight_(std::exchange(other.in_flight_, false)),
      completion_recorded_(std::exchange(other.completion_recorded_, false)) {
  other.context_.reset();
  other.arena_.reset();
}

CudaGraphExecutable::~CudaGraphExecutable() noexcept {
  if (graph_ != nullptr || executable_ != nullptr ||
      completion_event_ != nullptr || context_.has_value() ||
      arena_.has_value()) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<CudaGraphExecutable>
CudaGraphExecutable::instantiate(cudaGraph_t graph,
                                 const CudaExecutionContext &context,
                                 const GraphArenaLease &arena) noexcept {
  using ExecutableResult = NativeRuntimeResult<CudaGraphExecutable>;
  if (graph == nullptr || !context.valid() || !arena.valid()) {
    return ExecutableResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphInstantiate));
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return ExecutableResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphInstantiate, 0, 0,
                   static_cast<uint64_t>(context.device_ordinal()),
                   static_cast<uint64_t>(arena.device_ordinal())));
  }
  const NativeRuntimeError current = require_current_device(
      context.device_ordinal(), NativeRuntimeOperation::kGraphInstantiate);
  if (!is_ok(current)) {
    return ExecutableResult::failure(current);
  }

  cudaEvent_t completion_event = nullptr;
  const cudaError_t create_event =
      cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming);
  if (create_event != cudaSuccess) {
    return ExecutableResult::failure(
        cuda_failure(NativeRuntimeOperation::kEventCreate, create_event));
  }

  cudaGraph_t retained_graph = nullptr;
  const cudaError_t clone = cudaGraphClone(&retained_graph, graph);
  if (clone != cudaSuccess) {
    const cudaError_t cleanup = cudaEventDestroy(completion_event);
    if (cleanup != cudaSuccess) {
      std::terminate();
    }
    return ExecutableResult::failure(
        cuda_failure(NativeRuntimeOperation::kGraphClone, clone));
  }

  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiate =
      cudaGraphInstantiateWithFlags(&executable, retained_graph, 0);
  if (instantiate != cudaSuccess) {
    const cudaError_t graph_cleanup = cudaGraphDestroy(retained_graph);
    if (graph_cleanup != cudaSuccess) {
      std::terminate();
    }
    const cudaError_t cleanup = cudaEventDestroy(completion_event);
    if (cleanup != cudaSuccess) {
      std::terminate();
    }
    return ExecutableResult::failure(
        cuda_failure(NativeRuntimeOperation::kGraphInstantiate, instantiate));
  }

  return ExecutableResult::success(CudaGraphExecutable(
      retained_graph, executable, completion_event,
      CudaExecutionContext(context), GraphArenaLease(arena)));
}

NativeRuntimeResult<CudaGraphExecutable>
CudaGraphExecutable::instantiate_child_sequence(
    std::span<const cudaGraph_t> child_graphs,
    const CudaExecutionContext &context,
    const GraphArenaLease &arena) noexcept {
  using ExecutableResult = NativeRuntimeResult<CudaGraphExecutable>;
  if (child_graphs.empty() || !context.valid() || !arena.valid()) {
    return ExecutableResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCreate));
  }
  for (const cudaGraph_t child : child_graphs) {
    if (child == nullptr) {
      return ExecutableResult::failure(
          make_error(NativeRuntimeCode::kInvalidArgument,
                     NativeRuntimeOperation::kGraphAddChild));
    }
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return ExecutableResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCreate, 0, 0,
                   static_cast<uint64_t>(context.device_ordinal()),
                   static_cast<uint64_t>(arena.device_ordinal())));
  }
  const NativeRuntimeError current = require_current_device(
      context.device_ordinal(), NativeRuntimeOperation::kGraphCreate);
  if (!is_ok(current)) {
    return ExecutableResult::failure(current);
  }

  cudaGraph_t parent = nullptr;
  const cudaError_t create = cudaGraphCreate(&parent, 0);
  if (create != cudaSuccess) {
    return ExecutableResult::failure(
        cuda_failure(NativeRuntimeOperation::kGraphCreate, create));
  }

  cudaGraphNode_t dependency = nullptr;
  for (const cudaGraph_t child : child_graphs) {
    cudaGraphNode_t node = nullptr;
    const cudaGraphNode_t *dependencies =
        dependency == nullptr ? nullptr : &dependency;
    const size_t dependency_count = dependency == nullptr ? 0U : 1U;
    const cudaError_t add = cudaGraphAddChildGraphNode(
        &node, parent, dependencies, dependency_count, child);
    if (add != cudaSuccess) {
      const cudaError_t cleanup = cudaGraphDestroy(parent);
      if (cleanup != cudaSuccess) {
        std::terminate();
      }
      return ExecutableResult::failure(
          cuda_failure(NativeRuntimeOperation::kGraphAddChild, add));
    }
    dependency = node;
  }

  ExecutableResult result = instantiate(parent, context, arena);
  const cudaError_t cleanup = cudaGraphDestroy(parent);
  if (cleanup != cudaSuccess) {
    std::terminate();
  }
  return result;
}

NativeRuntimeError CudaGraphExecutable::launch() noexcept {
  if (!valid()) {
    return invalid_state(NativeRuntimeOperation::kGraphLaunch);
  }
  const NativeRuntimeError current = require_current_device(
      context_->device_ordinal(), NativeRuntimeOperation::kGraphLaunch);
  if (!is_ok(current)) {
    return current;
  }

  const cudaError_t launch = cudaGraphLaunch(executable_, context_->stream());
  if (launch != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kGraphLaunch, launch);
  }
  in_flight_ = true;
  completion_recorded_ = false;

  const cudaError_t record =
      cudaEventRecord(completion_event_, context_->stream());
  if (record != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kEventRecord, record);
  }
  completion_recorded_ = true;
  return native_runtime_ok();
}

NativeRuntimeError CudaGraphExecutable::synchronize() noexcept {
  if (!valid()) {
    return invalid_state(NativeRuntimeOperation::kEventSynchronize);
  }
  if (!in_flight_) {
    return native_runtime_ok();
  }
  const NativeRuntimeError current = require_current_device(
      context_->device_ordinal(), NativeRuntimeOperation::kEventSynchronize);
  if (!is_ok(current)) {
    return current;
  }

  if (!completion_recorded_) {
    const NativeRuntimeError stream_status = context_->synchronize();
    if (!is_ok(stream_status)) {
      return stream_status;
    }
  } else {
    const cudaError_t wait = cudaEventSynchronize(completion_event_);
    if (wait != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kEventSynchronize, wait);
    }
  }
  in_flight_ = false;
  completion_recorded_ = false;
  return native_runtime_ok();
}

NativeRuntimeError CudaGraphExecutable::close() noexcept {
  if (graph_ == nullptr && executable_ == nullptr &&
      completion_event_ == nullptr && !context_.has_value() &&
      !arena_.has_value()) {
    return native_runtime_ok();
  }
  if (!context_.has_value()) {
    return invalid_state(NativeRuntimeOperation::kGraphDestroy);
  }
  const NativeRuntimeError current = require_current_device(
      context_->device_ordinal(), NativeRuntimeOperation::kGraphDestroy);
  if (!is_ok(current)) {
    return current;
  }

  if (in_flight_) {
    const NativeRuntimeError completed = synchronize();
    if (!is_ok(completed)) {
      return completed;
    }
  }
  if (executable_ != nullptr) {
    const cudaError_t destroy = cudaGraphExecDestroy(executable_);
    if (destroy != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphDestroy, destroy);
    }
    executable_ = nullptr;
  }
  if (graph_ != nullptr) {
    const cudaError_t destroy = cudaGraphDestroy(graph_);
    if (destroy != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphDestroy, destroy);
    }
    graph_ = nullptr;
  }
  if (completion_event_ != nullptr) {
    const cudaError_t destroy = cudaEventDestroy(completion_event_);
    if (destroy != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kEventDestroy, destroy);
    }
    completion_event_ = nullptr;
  }
  arena_.reset();
  context_.reset();
  return native_runtime_ok();
}

bool CudaGraphExecutable::valid() const noexcept {
  return graph_ != nullptr && executable_ != nullptr &&
         completion_event_ != nullptr && context_.has_value() &&
         context_->valid() && arena_.has_value() && arena_->valid();
}

int32_t CudaGraphExecutable::device_ordinal() const noexcept {
  return context_.has_value() ? context_->device_ordinal() : -1;
}

NativeRuntimeResult<cudaGraph_t>
CudaGraphExecutable::clone_graph() const noexcept {
  using GraphResult = NativeRuntimeResult<cudaGraph_t>;
  if (!valid()) {
    return GraphResult::failure(
        invalid_state(NativeRuntimeOperation::kGraphClone));
  }
  const NativeRuntimeError current = require_current_device(
      context_->device_ordinal(), NativeRuntimeOperation::kGraphClone);
  if (!is_ok(current)) {
    return GraphResult::failure(current);
  }
  cudaGraph_t clone = nullptr;
  const cudaError_t status = cudaGraphClone(&clone, graph_);
  if (status != cudaSuccess) {
    return GraphResult::failure(
        cuda_failure(NativeRuntimeOperation::kGraphClone, status));
  }
  return GraphResult::success(std::move(clone));
}

NativeRuntimeError CudaGraphExecutable::bind_compact_device_to_host_result(
    void *host_destination, const void *device_source,
    uint64_t size_bytes) noexcept {
  if (!valid() || host_destination == nullptr || device_source == nullptr ||
      size_bytes == 0 || size_bytes > std::numeric_limits<size_t>::max()) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphMemcpyParamsSet);
  }
  if (in_flight_) {
    return make_error(NativeRuntimeCode::kResourceBusy,
                      NativeRuntimeOperation::kGraphMemcpyParamsSet);
  }
  const NativeRuntimeError current =
      require_current_device(context_->device_ordinal(),
                             NativeRuntimeOperation::kGraphMemcpyParamsSet);
  if (!is_ok(current)) {
    return current;
  }

  std::vector<cudaGraphNode_t> pending;
  try {
    pending.push_back(nullptr);
  } catch (const std::bad_alloc &) {
    return make_error(NativeRuntimeCode::kHostAllocationFailed,
                      NativeRuntimeOperation::kGraphGetNodes);
  }
  pending.pop_back();
  auto append_graph_nodes =
      [&pending](cudaGraph_t graph) noexcept -> NativeRuntimeError {
    size_t count = 0;
    cudaError_t status = cudaGraphGetNodes(graph, nullptr, &count);
    if (status != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphGetNodes, status);
    }
    const size_t begin = pending.size();
    try {
      pending.resize(begin + count);
    } catch (const std::bad_alloc &) {
      return make_error(NativeRuntimeCode::kHostAllocationFailed,
                        NativeRuntimeOperation::kGraphGetNodes);
    }
    status = cudaGraphGetNodes(graph, pending.data() + begin, &count);
    if (status != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphGetNodes, status);
    }
    pending.resize(begin + count);
    return native_runtime_ok();
  };
  NativeRuntimeError enumeration = append_graph_nodes(graph_);
  if (!is_ok(enumeration)) {
    return enumeration;
  }

  cudaGraphNode_t match = nullptr;
  for (size_t index = 0; index < pending.size(); ++index) {
    const cudaGraphNode_t node = pending[index];
    cudaGraphNodeType type{};
    cudaError_t status = cudaGraphNodeGetType(node, &type);
    if (status != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphNodeType, status);
    }
    if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child = nullptr;
      status = cudaGraphChildGraphNodeGetGraph(node, &child);
      if (status != cudaSuccess) {
        return cuda_failure(NativeRuntimeOperation::kGraphGetNodes, status);
      }
      enumeration = append_graph_nodes(child);
      if (!is_ok(enumeration)) {
        return enumeration;
      }
      continue;
    }
    if (type != cudaGraphNodeTypeMemcpy) {
      continue;
    }
    cudaMemcpy3DParms params{};
    status = cudaGraphMemcpyNodeGetParams(node, &params);
    if (status != cudaSuccess) {
      return cuda_failure(NativeRuntimeOperation::kGraphMemcpyParams, status);
    }
    if (params.kind != cudaMemcpyDeviceToHost ||
        params.srcPtr.ptr != device_source ||
        params.extent.width != size_bytes || params.extent.height != 1U ||
        params.extent.depth != 1U) {
      continue;
    }
    if (match != nullptr) {
      return make_error(NativeRuntimeCode::kInvalidArgument,
                        NativeRuntimeOperation::kGraphMemcpyParamsSet, 0, 0, 2,
                        1);
    }
    match = node;
  }
  if (match == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kGraphMemcpyParamsSet, 0, 0, 0,
                      1);
  }

  cudaMemcpy3DParms updated{};
  updated.srcPtr = cudaPitchedPtr{const_cast<void *>(device_source),
                                  static_cast<size_t>(size_bytes),
                                  static_cast<size_t>(size_bytes), 1};
  updated.dstPtr =
      cudaPitchedPtr{host_destination, static_cast<size_t>(size_bytes),
                     static_cast<size_t>(size_bytes), 1};
  updated.extent = cudaExtent{static_cast<size_t>(size_bytes), 1, 1};
  updated.kind = cudaMemcpyDeviceToHost;
  const cudaError_t status =
      cudaGraphExecMemcpyNodeSetParams(executable_, match, &updated);
  if (status != cudaSuccess) {
    return cuda_failure(NativeRuntimeOperation::kGraphMemcpyParamsSet, status);
  }
  return native_runtime_ok();
}

} // namespace sglang::native
