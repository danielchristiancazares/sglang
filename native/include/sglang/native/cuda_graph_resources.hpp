#ifndef SGLANG_NATIVE_CUDA_GRAPH_RESOURCES_HPP_
#define SGLANG_NATIVE_CUDA_GRAPH_RESOURCES_HPP_

#include "sglang/native/result.hpp"
#include "sglang/native/tensor_view.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#if !defined(_WIN64)
#error "The native CUDA resource foundation requires Windows x64."
#endif

namespace sglang::native {

enum class NativeRuntimeCode : uint32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kInvalidState = 2,
  kHostAllocationFailed = 3,
  kDeviceMismatch = 4,
  kArithmeticOverflow = 5,
  kOutOfCapacity = 6,
  kAlreadySealed = 7,
  kNotSealed = 8,
  kResourceBusy = 9,
  kForeignSlice = 10,
  kOwnerMetadataConflict = 11,
  kCudaRuntimeFailure = 12,
  kTensorValidationFailure = 13,
};

enum class NativeRuntimeOperation : uint32_t {
  kNone = 0,
  kGetDevice = 1,
  kStreamCreate = 2,
  kStreamSynchronize = 3,
  kStreamDestroy = 4,
  kDeviceAllocate = 5,
  kDeviceFree = 6,
  kReserve = 7,
  kSeal = 8,
  kAcquireLease = 9,
  kBindTensor = 10,
  kGraphInstantiate = 11,
  kEventCreate = 12,
  kGraphLaunch = 13,
  kEventRecord = 14,
  kEventSynchronize = 15,
  kGraphDestroy = 16,
  kEventDestroy = 17,
  kStreamGetFlags = 18,
  kValidateLinearRejectionSampling = 19,
  kLaunchLinearRejectionSampling = 20,
  kValidateLinearVerifyRng = 21,
  kLaunchSeededLinearVerifyRng = 22,
  kLaunchStatefulLinearVerifyRng = 23,
};

struct NativeRuntimeError final {
  NativeRuntimeCode code;
  NativeRuntimeOperation operation;
  int32_t native_code;
  uint32_t detail;
  uint64_t actual;
  uint64_t required;
};

[[nodiscard]] constexpr NativeRuntimeError native_runtime_ok() noexcept {
  return NativeRuntimeError{
      NativeRuntimeCode::kOk, NativeRuntimeOperation::kNone, 0, 0, 0, 0};
}

[[nodiscard]] constexpr bool is_ok(NativeRuntimeError status) noexcept {
  return status.code == NativeRuntimeCode::kOk;
}

[[nodiscard]] std::string_view native_runtime_code_name(
    NativeRuntimeCode code) noexcept;
[[nodiscard]] std::string_view native_runtime_operation_name(
    NativeRuntimeOperation operation) noexcept;

template <typename T>
using NativeRuntimeResult = Result<T, NativeRuntimeError>;

namespace detail {
struct CudaStreamState;
struct GraphArenaState;

[[nodiscard]] NativeRuntimeError tensor_binding_error(
    TensorValidationError error) noexcept;
}  // namespace detail

class CudaStream;
class GraphMemoryArena;
class GraphArenaLease;
class GraphMemorySlice;
class CudaGraphExecutable;

template <DType D, uint32_t Rank, TensorAccess Access>
class GraphStableTensorView;

class CudaExecutionContext final {
 public:
  CudaExecutionContext(const CudaExecutionContext& other) noexcept;
  CudaExecutionContext(CudaExecutionContext&& other) noexcept;
  CudaExecutionContext& operator=(const CudaExecutionContext&) = delete;
  CudaExecutionContext& operator=(CudaExecutionContext&&) = delete;
  ~CudaExecutionContext() noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;
  [[nodiscard]] cudaStream_t stream() const noexcept;
  [[nodiscard]] NativeRuntimeResult<unsigned int> stream_flags()
      const noexcept;
  [[nodiscard]] NativeRuntimeError synchronize() const noexcept;

 private:
  explicit CudaExecutionContext(detail::CudaStreamState* state) noexcept;

  detail::CudaStreamState* state_;

  friend class CudaStream;
  friend class CudaGraphExecutable;
};

class CudaStream final {
 public:
  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;
  CudaStream(CudaStream&& other) noexcept;
  CudaStream& operator=(CudaStream&&) = delete;
  ~CudaStream() noexcept;

  [[nodiscard]] static NativeRuntimeResult<CudaStream> create_nonblocking()
      noexcept;
  [[nodiscard]] NativeRuntimeResult<CudaExecutionContext> context()
      const noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;

 private:
  explicit CudaStream(detail::CudaStreamState* state) noexcept;

  detail::CudaStreamState* state_;
};

class GraphMemorySlice final {
 public:
  GraphMemorySlice(const GraphMemorySlice& other) noexcept;
  GraphMemorySlice(GraphMemorySlice&& other) noexcept;
  GraphMemorySlice& operator=(const GraphMemorySlice&) = delete;
  GraphMemorySlice& operator=(GraphMemorySlice&&) = delete;
  ~GraphMemorySlice() noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] uint64_t offset_bytes() const noexcept;
  [[nodiscard]] uint64_t size_bytes() const noexcept;

 private:
  GraphMemorySlice(detail::GraphArenaState* state, uint64_t offset_bytes,
                   uint64_t size_bytes) noexcept;

  detail::GraphArenaState* state_;
  uint64_t offset_bytes_;
  uint64_t size_bytes_;

  friend class GraphMemoryArena;
  friend class GraphArenaLease;
};

class GraphArenaLease final {
 public:
  GraphArenaLease(const GraphArenaLease& other) noexcept;
  GraphArenaLease(GraphArenaLease&& other) noexcept;
  GraphArenaLease& operator=(const GraphArenaLease&) = delete;
  GraphArenaLease& operator=(GraphArenaLease&&) = delete;
  ~GraphArenaLease() noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;
  [[nodiscard]] uint64_t capacity_bytes() const noexcept;

  template <DType D, uint32_t Rank>
  [[nodiscard]] NativeRuntimeResult<
      GraphStableTensorView<D, Rank, TensorAccess::kReadOnly>>
  bind_const(const GraphMemorySlice& slice,
             SglNativeTensorMetadataV1 metadata) const noexcept;

  template <DType D, uint32_t Rank>
  [[nodiscard]] NativeRuntimeResult<
      GraphStableTensorView<D, Rank, TensorAccess::kReadWrite>>
  bind_mutable(const GraphMemorySlice& slice,
               SglNativeTensorMetadataV1 metadata) const noexcept;

 private:
  explicit GraphArenaLease(detail::GraphArenaState* state) noexcept;

  [[nodiscard]] NativeRuntimeError prepare_binding(
      const GraphMemorySlice& slice, SglNativeTensorMetadataV1* metadata,
      const void** allocation_base) const noexcept;
  [[nodiscard]] NativeRuntimeError prepare_const_binding(
      const GraphMemorySlice& slice, SglNativeTensorMetadataV1 metadata,
      SglNativeConstTensorViewV1* raw) const noexcept;
  [[nodiscard]] NativeRuntimeError prepare_mutable_binding(
      const GraphMemorySlice& slice, SglNativeTensorMetadataV1 metadata,
      SglNativeMutableTensorViewV1* raw) const noexcept;

  detail::GraphArenaState* state_;

  friend class GraphMemoryArena;
  friend class CudaGraphExecutable;
  template <DType D, uint32_t Rank, TensorAccess Access>
  friend class GraphStableTensorView;
};

class GraphMemoryArena final {
 public:
  GraphMemoryArena(const GraphMemoryArena&) = delete;
  GraphMemoryArena& operator=(const GraphMemoryArena&) = delete;
  GraphMemoryArena(GraphMemoryArena&& other) noexcept;
  GraphMemoryArena& operator=(GraphMemoryArena&&) = delete;
  ~GraphMemoryArena() noexcept;

  [[nodiscard]] static NativeRuntimeResult<GraphMemoryArena> allocate(
      const CudaExecutionContext& context, uint64_t capacity_bytes) noexcept;
  [[nodiscard]] NativeRuntimeResult<GraphMemorySlice> reserve(
      uint64_t size_bytes, uint64_t alignment_bytes) noexcept;
  [[nodiscard]] NativeRuntimeError seal() noexcept;
  [[nodiscard]] NativeRuntimeResult<GraphArenaLease> acquire_lease()
      const noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool sealed() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;
  [[nodiscard]] uint64_t capacity_bytes() const noexcept;
  [[nodiscard]] uint64_t used_bytes() const noexcept;

 private:
  explicit GraphMemoryArena(detail::GraphArenaState* state) noexcept;

  detail::GraphArenaState* state_;
};

template <DType D, uint32_t Rank, TensorAccess Access>
class GraphStableTensorView final {
  static_assert(dtype_element_bits(D) != 0);
  static_assert(Rank <= SGL_NATIVE_TENSOR_MAX_RANK);

 public:
  GraphStableTensorView(const GraphStableTensorView&) noexcept = default;
  GraphStableTensorView(GraphStableTensorView&&) noexcept = default;
  GraphStableTensorView& operator=(const GraphStableTensorView&) = delete;
  GraphStableTensorView& operator=(GraphStableTensorView&&) = delete;
  ~GraphStableTensorView() noexcept = default;

  [[nodiscard]] constexpr DType dtype() const noexcept { return D; }
  [[nodiscard]] constexpr uint32_t rank() const noexcept { return Rank; }
  [[nodiscard]] DeviceKind device_kind() const noexcept {
    return view_.device_kind();
  }
  [[nodiscard]] int32_t device_ordinal() const noexcept {
    return view_.device_ordinal();
  }
  [[nodiscard]] std::span<const int64_t, Rank> extents() const noexcept {
    return view_.extents();
  }
  [[nodiscard]] std::span<const int64_t, Rank> strides() const noexcept {
    return view_.strides();
  }
  [[nodiscard]] uint64_t allocation_bytes() const noexcept {
    return view_.allocation_bytes();
  }
  [[nodiscard]] bool is_empty() const noexcept { return view_.is_empty(); }
  [[nodiscard]] bool is_row_major_contiguous() const noexcept {
    return view_.is_row_major_contiguous();
  }

  [[nodiscard]] const std::byte* data_bytes() const noexcept
    requires(Access == TensorAccess::kReadOnly)
  {
    return view_.data_bytes();
  }

  [[nodiscard]] std::byte* data_bytes() const noexcept
    requires(Access == TensorAccess::kReadWrite)
  {
    return view_.data_bytes();
  }

 private:
  using TypedView = TypedTensorView<D, Rank, Access>;

  GraphStableTensorView(GraphArenaLease lease, TypedView view) noexcept
      : lease_(std::move(lease)), view_(view) {}

  GraphArenaLease lease_;
  TypedView view_;

  friend class GraphArenaLease;
};

class CudaGraphExecutable final {
 public:
  CudaGraphExecutable(const CudaGraphExecutable&) = delete;
  CudaGraphExecutable& operator=(const CudaGraphExecutable&) = delete;
  CudaGraphExecutable(CudaGraphExecutable&& other) noexcept;
  CudaGraphExecutable& operator=(CudaGraphExecutable&&) = delete;
  ~CudaGraphExecutable() noexcept;

  [[nodiscard]] static NativeRuntimeResult<CudaGraphExecutable> instantiate(
      cudaGraph_t graph, const CudaExecutionContext& context,
      const GraphArenaLease& arena) noexcept;
  [[nodiscard]] NativeRuntimeError launch() noexcept;
  [[nodiscard]] NativeRuntimeError synchronize() noexcept;
  [[nodiscard]] NativeRuntimeError close() noexcept;
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] int32_t device_ordinal() const noexcept;

 private:
  CudaGraphExecutable(cudaGraphExec_t executable, cudaEvent_t completion_event,
                      CudaExecutionContext context,
                      GraphArenaLease arena) noexcept;

  cudaGraphExec_t executable_;
  cudaEvent_t completion_event_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphArenaLease> arena_;
  bool in_flight_;
  bool completion_recorded_;
};

template <DType D, uint32_t Rank>
NativeRuntimeResult<
    GraphStableTensorView<D, Rank, TensorAccess::kReadOnly>>
GraphArenaLease::bind_const(const GraphMemorySlice& slice,
                            SglNativeTensorMetadataV1 metadata) const noexcept {
  using StableView =
      GraphStableTensorView<D, Rank, TensorAccess::kReadOnly>;
  using StableResult = NativeRuntimeResult<StableView>;

  SglNativeConstTensorViewV1 raw{};
  const NativeRuntimeError prepared =
      prepare_const_binding(slice, metadata, &raw);
  if (!is_ok(prepared)) {
    return StableResult::failure(prepared);
  }

  return std::move(validate(&raw)).match(
      [this](ValidatedConstTensorView validated) noexcept -> StableResult {
        return std::move(narrow<D, Rank>(validated))
            .match(
                [this](TypedConstTensorView<D, Rank> typed) noexcept
                    -> StableResult {
                  return StableResult::success(
                      StableView(GraphArenaLease(*this), typed));
                },
                [](TensorValidationError error) noexcept -> StableResult {
                  return StableResult::failure(
                      detail::tensor_binding_error(error));
                });
      },
      [](TensorValidationError error) noexcept -> StableResult {
        return StableResult::failure(detail::tensor_binding_error(error));
      });
}

template <DType D, uint32_t Rank>
NativeRuntimeResult<
    GraphStableTensorView<D, Rank, TensorAccess::kReadWrite>>
GraphArenaLease::bind_mutable(
    const GraphMemorySlice& slice,
    SglNativeTensorMetadataV1 metadata) const noexcept {
  using StableView =
      GraphStableTensorView<D, Rank, TensorAccess::kReadWrite>;
  using StableResult = NativeRuntimeResult<StableView>;

  SglNativeMutableTensorViewV1 raw{};
  const NativeRuntimeError prepared =
      prepare_mutable_binding(slice, metadata, &raw);
  if (!is_ok(prepared)) {
    return StableResult::failure(prepared);
  }

  return std::move(validate(&raw)).match(
      [this](ValidatedMutableTensorView validated) noexcept -> StableResult {
        return std::move(narrow<D, Rank>(validated))
            .match(
                [this](TypedMutableTensorView<D, Rank> typed) noexcept
                    -> StableResult {
                  return StableResult::success(
                      StableView(GraphArenaLease(*this), typed));
                },
                [](TensorValidationError error) noexcept -> StableResult {
                  return StableResult::failure(
                      detail::tensor_binding_error(error));
                });
      },
      [](TensorValidationError error) noexcept -> StableResult {
        return StableResult::failure(detail::tensor_binding_error(error));
      });
}

static_assert(sizeof(NativeRuntimeError) == 32);
static_assert(alignof(NativeRuntimeError) == 8);
static_assert(std::is_standard_layout_v<NativeRuntimeError>);
static_assert(std::is_trivially_copyable_v<NativeRuntimeError>);
static_assert(!std::is_default_constructible_v<CudaExecutionContext>);
static_assert(!std::is_default_constructible_v<GraphMemorySlice>);
static_assert(!std::is_default_constructible_v<GraphArenaLease>);
static_assert(!std::is_constructible_v<CudaExecutionContext, cudaStream_t,
                                       int32_t>);
static_assert(!std::is_constructible_v<GraphMemorySlice, void*, uint64_t,
                                       uint64_t>);

}  // namespace sglang::native

#endif  // SGLANG_NATIVE_CUDA_GRAPH_RESOURCES_HPP_
