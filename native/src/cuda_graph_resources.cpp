#include "sglang/native/cuda_graph_resources.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <string_view>
#include <utility>

namespace sglang::native {
namespace detail {

struct CudaStreamState final {
  std::atomic<uint64_t> context_count;
  cudaStream_t stream;
  int32_t device_ordinal;
};

struct GraphArenaState final {
  std::atomic<uint64_t> lease_count;
  void* allocation_base;
  uint64_t capacity_bytes;
  uint64_t used_bytes;
  int32_t device_ordinal;
  bool sealed;
};

}  // namespace detail
namespace {

[[nodiscard]] constexpr NativeRuntimeError make_error(
    NativeRuntimeCode code, NativeRuntimeOperation operation,
    int32_t native_code = 0, uint32_t detail = 0, uint64_t actual = 0,
    uint64_t required = 0) noexcept {
  return NativeRuntimeError{
      code, operation, native_code, detail, actual, required};
}

[[nodiscard]] constexpr NativeRuntimeError invalid_state(
    NativeRuntimeOperation operation) noexcept {
  return make_error(NativeRuntimeCode::kInvalidState, operation);
}

[[nodiscard]] constexpr NativeRuntimeError cuda_failure(
    NativeRuntimeOperation operation, cudaError_t error) noexcept {
  return make_error(NativeRuntimeCode::kCudaRuntimeFailure, operation,
                    static_cast<int32_t>(error));
}

[[nodiscard]] NativeRuntimeError require_current_device(
    int32_t required_device, NativeRuntimeOperation operation) noexcept {
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

void retain_context(detail::CudaStreamState* state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->context_count.fetch_add(1, std::memory_order_relaxed);
  if (previous == std::numeric_limits<uint64_t>::max()) {
    std::terminate();
  }
}

void release_context(detail::CudaStreamState* state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->context_count.fetch_sub(1, std::memory_order_acq_rel);
  if (previous == 0) {
    std::terminate();
  }
}

void retain_arena(detail::GraphArenaState* state) noexcept {
  if (state == nullptr) {
    return;
  }
  const uint64_t previous =
      state->lease_count.fetch_add(1, std::memory_order_relaxed);
  if (previous == std::numeric_limits<uint64_t>::max()) {
    std::terminate();
  }
}

void release_arena(detail::GraphArenaState* state) noexcept {
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

[[nodiscard]] NativeRuntimeError owner_metadata_conflict(
    TensorValidationField field, uint64_t actual,
    uint64_t required) noexcept {
  return make_error(
      NativeRuntimeCode::kOwnerMetadataConflict,
      NativeRuntimeOperation::kBindTensor, 0, static_cast<uint32_t>(field),
      actual, required);
}

}  // namespace

namespace detail {

NativeRuntimeError tensor_binding_error(
    TensorValidationError error) noexcept {
  return make_error(
      NativeRuntimeCode::kTensorValidationFailure,
      NativeRuntimeOperation::kBindTensor,
      static_cast<int32_t>(error.code), static_cast<uint32_t>(error.field),
      error.actual, error.required);
}

}  // namespace detail

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

std::string_view native_runtime_operation_name(
    NativeRuntimeOperation operation) noexcept {
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
    default:
      return "invalid_runtime_operation";
  }
}

CudaExecutionContext::CudaExecutionContext(
    detail::CudaStreamState* state) noexcept
    : state_(state) {
  retain_context(state_);
}

CudaExecutionContext::CudaExecutionContext(
    const CudaExecutionContext& other) noexcept
    : state_(other.state_) {
  retain_context(state_);
}

CudaExecutionContext::CudaExecutionContext(
    CudaExecutionContext&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

CudaExecutionContext::~CudaExecutionContext() noexcept {
  release_context(state_);
}

bool CudaExecutionContext::valid() const noexcept {
  return state_ != nullptr;
}

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

CudaStream::CudaStream(detail::CudaStreamState* state) noexcept
    : state_(state) {}

CudaStream::CudaStream(CudaStream&& other) noexcept
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

  auto* state =
      new (std::nothrow) detail::CudaStreamState;
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
  return StreamResult::failure(make_error(
      NativeRuntimeCode::kHostAllocationFailed,
      NativeRuntimeOperation::kStreamCreate));
}

NativeRuntimeResult<CudaExecutionContext> CudaStream::context()
    const noexcept {
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
  const uint64_t active =
      state_->context_count.load(std::memory_order_acquire);
  if (active != 0) {
    return make_error(NativeRuntimeCode::kResourceBusy,
                      NativeRuntimeOperation::kStreamDestroy, 0, 0, active,
                      0);
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

bool CudaStream::valid() const noexcept {
  return state_ != nullptr;
}

int32_t CudaStream::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

GraphMemorySlice::GraphMemorySlice(detail::GraphArenaState* state,
                                   uint64_t offset_bytes,
                                   uint64_t size_bytes) noexcept
    : state_(state),
      offset_bytes_(offset_bytes),
      size_bytes_(size_bytes) {
  retain_arena(state_);
}

GraphMemorySlice::GraphMemorySlice(
    const GraphMemorySlice& other) noexcept
    : state_(other.state_),
      offset_bytes_(other.offset_bytes_),
      size_bytes_(other.size_bytes_) {
  retain_arena(state_);
}

GraphMemorySlice::GraphMemorySlice(GraphMemorySlice&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)),
      offset_bytes_(std::exchange(other.offset_bytes_, 0)),
      size_bytes_(std::exchange(other.size_bytes_, 0)) {}

GraphMemorySlice::~GraphMemorySlice() noexcept {
  release_arena(state_);
}

bool GraphMemorySlice::valid() const noexcept {
  return state_ != nullptr;
}

uint64_t GraphMemorySlice::offset_bytes() const noexcept {
  return offset_bytes_;
}

uint64_t GraphMemorySlice::size_bytes() const noexcept {
  return size_bytes_;
}

GraphArenaLease::GraphArenaLease(detail::GraphArenaState* state) noexcept
    : state_(state) {
  retain_arena(state_);
}

GraphArenaLease::GraphArenaLease(const GraphArenaLease& other) noexcept
    : state_(other.state_) {
  retain_arena(state_);
}

GraphArenaLease::GraphArenaLease(GraphArenaLease&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

GraphArenaLease::~GraphArenaLease() noexcept {
  release_arena(state_);
}

bool GraphArenaLease::valid() const noexcept {
  return state_ != nullptr;
}

int32_t GraphArenaLease::device_ordinal() const noexcept {
  return state_ == nullptr ? -1 : state_->device_ordinal;
}

uint64_t GraphArenaLease::capacity_bytes() const noexcept {
  return state_ == nullptr ? 0 : state_->capacity_bytes;
}

NativeRuntimeError GraphArenaLease::prepare_binding(
    const GraphMemorySlice& slice, SglNativeTensorMetadataV1* metadata,
    const void** allocation_base) const noexcept {
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
    return owner_metadata_conflict(
        TensorValidationField::kDeviceKind, metadata->device_kind,
        SGL_NATIVE_DEVICE_INVALID);
  }
  if (metadata->device_ordinal != 0) {
    return owner_metadata_conflict(
        TensorValidationField::kDeviceOrdinal,
        static_cast<uint64_t>(metadata->device_ordinal), 0);
  }
  if (metadata->allocation_bytes != 0) {
    return owner_metadata_conflict(
        TensorValidationField::kAllocationBytes,
        metadata->allocation_bytes, 0);
  }

  metadata->device_kind = SGL_NATIVE_DEVICE_CUDA;
  metadata->device_ordinal = state_->device_ordinal;
  metadata->allocation_bytes = slice.size_bytes_;
  *allocation_base =
      static_cast<const std::byte*>(state_->allocation_base) +
      slice.offset_bytes_;
  return native_runtime_ok();
}

NativeRuntimeError GraphArenaLease::prepare_const_binding(
    const GraphMemorySlice& slice, SglNativeTensorMetadataV1 metadata,
    SglNativeConstTensorViewV1* raw) const noexcept {
  if (raw == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kBindTensor);
  }
  const void* allocation_base = nullptr;
  const NativeRuntimeError prepared =
      prepare_binding(slice, &metadata, &allocation_base);
  if (!is_ok(prepared)) {
    return prepared;
  }
  *raw = SglNativeConstTensorViewV1{metadata, allocation_base};
  return native_runtime_ok();
}

NativeRuntimeError GraphArenaLease::prepare_mutable_binding(
    const GraphMemorySlice& slice, SglNativeTensorMetadataV1 metadata,
    SglNativeMutableTensorViewV1* raw) const noexcept {
  if (raw == nullptr) {
    return make_error(NativeRuntimeCode::kInvalidArgument,
                      NativeRuntimeOperation::kBindTensor);
  }
  const void* allocation_base = nullptr;
  const NativeRuntimeError prepared =
      prepare_binding(slice, &metadata, &allocation_base);
  if (!is_ok(prepared)) {
    return prepared;
  }
  *raw = SglNativeMutableTensorViewV1{
      metadata, const_cast<void*>(allocation_base)};
  return native_runtime_ok();
}

GraphMemoryArena::GraphMemoryArena(detail::GraphArenaState* state) noexcept
    : state_(state) {}

GraphMemoryArena::GraphMemoryArena(GraphMemoryArena&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

GraphMemoryArena::~GraphMemoryArena() noexcept {
  if (state_ != nullptr) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<GraphMemoryArena> GraphMemoryArena::allocate(
    const CudaExecutionContext& context,
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

  auto* state =
      new (std::nothrow) detail::GraphArenaState;
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
  return ArenaResult::failure(make_error(
      NativeRuntimeCode::kHostAllocationFailed,
      NativeRuntimeOperation::kDeviceAllocate));
}

NativeRuntimeResult<GraphMemorySlice> GraphMemoryArena::reserve(
    uint64_t size_bytes, uint64_t alignment_bytes) noexcept {
  using SliceResult = NativeRuntimeResult<GraphMemorySlice>;
  if (state_ == nullptr) {
    return SliceResult::failure(
        invalid_state(NativeRuntimeOperation::kReserve));
  }
  if (state_->sealed) {
    return SliceResult::failure(make_error(
        NativeRuntimeCode::kAlreadySealed,
        NativeRuntimeOperation::kReserve));
  }
  if (size_bytes == 0 || !is_power_of_two(alignment_bytes) ||
      alignment_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return SliceResult::failure(make_error(
        NativeRuntimeCode::kInvalidArgument,
        NativeRuntimeOperation::kReserve, 0, 0, alignment_bytes, 1));
  }

  const uintptr_t base =
      reinterpret_cast<uintptr_t>(state_->allocation_base);
  if (state_->used_bytes >
      static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max() - base)) {
    return SliceResult::failure(make_error(
        NativeRuntimeCode::kArithmeticOverflow,
        NativeRuntimeOperation::kReserve));
  }
  const uintptr_t current =
      base + static_cast<uintptr_t>(state_->used_bytes);
  const uintptr_t mask = static_cast<uintptr_t>(alignment_bytes - 1);
  if (current > std::numeric_limits<uintptr_t>::max() - mask) {
    return SliceResult::failure(make_error(
        NativeRuntimeCode::kArithmeticOverflow,
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
        NativeRuntimeCode::kOutOfCapacity,
        NativeRuntimeOperation::kReserve, 0, 0, required,
        state_->capacity_bytes));
  }

  state_->used_bytes = offset + size_bytes;
  return SliceResult::success(
      GraphMemorySlice(state_, offset, size_bytes));
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

NativeRuntimeResult<GraphArenaLease> GraphMemoryArena::acquire_lease()
    const noexcept {
  using LeaseResult = NativeRuntimeResult<GraphArenaLease>;
  if (state_ == nullptr) {
    return LeaseResult::failure(
        invalid_state(NativeRuntimeOperation::kAcquireLease));
  }
  if (!state_->sealed) {
    return LeaseResult::failure(make_error(
        NativeRuntimeCode::kNotSealed,
        NativeRuntimeOperation::kAcquireLease));
  }
  return LeaseResult::success(GraphArenaLease(state_));
}

NativeRuntimeError GraphMemoryArena::close() noexcept {
  if (state_ == nullptr) {
    return native_runtime_ok();
  }
  const uint64_t active =
      state_->lease_count.load(std::memory_order_acquire);
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

bool GraphMemoryArena::valid() const noexcept {
  return state_ != nullptr;
}

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

CudaGraphExecutable::CudaGraphExecutable(
    cudaGraphExec_t executable, cudaEvent_t completion_event,
    CudaExecutionContext context, GraphArenaLease arena) noexcept
    : executable_(executable),
      completion_event_(completion_event),
      context_(std::move(context)),
      arena_(std::move(arena)),
      in_flight_(false),
      completion_recorded_(false) {}

CudaGraphExecutable::CudaGraphExecutable(
    CudaGraphExecutable&& other) noexcept
    : executable_(std::exchange(other.executable_, nullptr)),
      completion_event_(std::exchange(other.completion_event_, nullptr)),
      context_(std::move(other.context_)),
      arena_(std::move(other.arena_)),
      in_flight_(std::exchange(other.in_flight_, false)),
      completion_recorded_(
          std::exchange(other.completion_recorded_, false)) {
  other.context_.reset();
  other.arena_.reset();
}

CudaGraphExecutable::~CudaGraphExecutable() noexcept {
  if (executable_ != nullptr || completion_event_ != nullptr ||
      context_.has_value() || arena_.has_value()) {
    require_cleanup_success(close());
  }
}

NativeRuntimeResult<CudaGraphExecutable>
CudaGraphExecutable::instantiate(
    cudaGraph_t graph, const CudaExecutionContext& context,
    const GraphArenaLease& arena) noexcept {
  using ExecutableResult = NativeRuntimeResult<CudaGraphExecutable>;
  if (graph == nullptr || !context.valid() || !arena.valid()) {
    return ExecutableResult::failure(make_error(
        NativeRuntimeCode::kInvalidArgument,
        NativeRuntimeOperation::kGraphInstantiate));
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return ExecutableResult::failure(make_error(
        NativeRuntimeCode::kDeviceMismatch,
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

  cudaGraphExec_t executable = nullptr;
  const cudaError_t instantiate =
      cudaGraphInstantiateWithFlags(&executable, graph, 0);
  if (instantiate != cudaSuccess) {
    const cudaError_t cleanup = cudaEventDestroy(completion_event);
    if (cleanup != cudaSuccess) {
      std::terminate();
    }
    return ExecutableResult::failure(cuda_failure(
        NativeRuntimeOperation::kGraphInstantiate, instantiate));
  }

  return ExecutableResult::success(CudaGraphExecutable(
      executable, completion_event, CudaExecutionContext(context),
      GraphArenaLease(arena)));
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

  const cudaError_t launch =
      cudaGraphLaunch(executable_, context_->stream());
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
      context_->device_ordinal(),
      NativeRuntimeOperation::kEventSynchronize);
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
      return cuda_failure(
          NativeRuntimeOperation::kEventSynchronize, wait);
    }
  }
  in_flight_ = false;
  completion_recorded_ = false;
  return native_runtime_ok();
}

NativeRuntimeError CudaGraphExecutable::close() noexcept {
  if (executable_ == nullptr && completion_event_ == nullptr &&
      !context_.has_value() && !arena_.has_value()) {
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
  return executable_ != nullptr && completion_event_ != nullptr &&
         context_.has_value() && context_->valid() &&
         arena_.has_value() && arena_->valid();
}

int32_t CudaGraphExecutable::device_ordinal() const noexcept {
  return context_.has_value() ? context_->device_ordinal() : -1;
}

}  // namespace sglang::native
