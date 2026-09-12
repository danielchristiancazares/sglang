#include "sglang/native/cuda_graph_resources.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <utility>

namespace {

using sglang::native::CudaCapturedGraph;
using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaStream;
using sglang::native::DType;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::GraphStableTensorView;
using sglang::native::is_ok;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::native_runtime_code_name;
using sglang::native::native_runtime_operation_name;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::NativeRuntimeResult;
using sglang::native::TensorAccess;
using sglang::native::TensorValidationCode;
using sglang::native::TensorValidationField;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

void print_error(NativeRuntimeError error) noexcept {
  const auto code = native_runtime_code_name(error.code);
  const auto operation = native_runtime_operation_name(error.operation);
  std::printf("runtime error code=%.*s operation=%.*s native=%d detail=%u "
              "actual=%llu required=%llu\n",
              static_cast<int>(code.size()), code.data(),
              static_cast<int>(operation.size()), operation.data(),
              error.native_code, error.detail,
              static_cast<unsigned long long>(error.actual),
              static_cast<unsigned long long>(error.required));
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {   \
      return false;                                                            \
    }                                                                          \
  } while (false)

#define CHECK_CUDA(expression)                                                 \
  do {                                                                         \
    const cudaError_t cuda_status = (expression);                              \
    if (cuda_status != cudaSuccess) {                                          \
      std::printf("%s:%d: CUDA failure %d: %s\n", __FILE__, __LINE__,          \
                  static_cast<int>(cuda_status),                               \
                  cudaGetErrorString(cuda_status));                            \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] bool check_status(NativeRuntimeError status,
                                const char *expression, int line) noexcept {
  if (is_ok(status)) {
    return true;
  }
  std::printf("%s:%d: failed status: %s\n", __FILE__, line, expression);
  print_error(status);
  return false;
}

#define CHECK_STATUS(expression)                                               \
  do {                                                                         \
    if (!check_status((expression), #expression, __LINE__)) {                  \
      return false;                                                            \
    }                                                                          \
  } while (false)

template <typename T>
[[nodiscard]] bool take_result(NativeRuntimeResult<T> result,
                               std::optional<T> *output) noexcept {
  return std::move(result).match(
      [output](T &&value) noexcept {
        output->emplace(std::move(value));
        return true;
      },
      [](NativeRuntimeError &&error) noexcept {
        print_error(error);
        return false;
      });
}

template <typename T>
[[nodiscard]] bool
expect_result_error(NativeRuntimeResult<T> result, NativeRuntimeCode code,
                    NativeRuntimeOperation operation) noexcept {
  return std::move(result).match(
      [](T &&) noexcept { return false; },
      [code, operation](NativeRuntimeError &&error) noexcept {
        return error.code == code && error.operation == operation;
      });
}

[[nodiscard]] SglNativeTensorMetadataV1
vector_metadata(SglNativeDType dtype, int64_t elements) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 1;
  metadata.extents[0] = elements;
  metadata.strides[0] = 1;
  return metadata;
}

[[nodiscard]] bool StreamContextLifetime() {
  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(stream->valid());
  CHECK(context->valid());
  CHECK(stream->device_ordinal() == context->device_ordinal());
  std::optional<unsigned int> stream_flags;
  CHECK(take_result(context->stream_flags(), &stream_flags));
  CHECK((*stream_flags & cudaStreamNonBlocking) != 0);

  const NativeRuntimeError busy = stream->close();
  CHECK(busy.code == NativeRuntimeCode::kResourceBusy);
  CHECK(busy.operation == NativeRuntimeOperation::kStreamDestroy);
  CHECK(busy.actual == 1);

  CHECK_STATUS(context->synchronize());
  context.reset();
  CHECK_STATUS(stream->close());
  CHECK(!stream->valid());
  stream.reset();
  return true;
}

[[nodiscard]] bool ArenaReservationLifecycle() {
  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> first;
  std::optional<GraphMemorySlice> second;
  std::optional<GraphArenaLease> lease;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(expect_result_error(GraphMemoryArena::allocate(*context, 0),
                            NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kDeviceAllocate));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 1024), &arena));
  CHECK(expect_result_error(arena->reserve(16, 3),
                            NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kReserve));
  CHECK(take_result(arena->reserve(128, 256), &first));
  CHECK(take_result(arena->reserve(128, 64), &second));
  CHECK(first->offset_bytes() % 256 == 0);
  CHECK(second->offset_bytes() >= first->offset_bytes() + first->size_bytes());
  CHECK(arena->used_bytes() == second->offset_bytes() + second->size_bytes());
  CHECK(expect_result_error(arena->reserve(1024, 1),
                            NativeRuntimeCode::kOutOfCapacity,
                            NativeRuntimeOperation::kReserve));
  CHECK(expect_result_error(arena->acquire_lease(),
                            NativeRuntimeCode::kNotSealed,
                            NativeRuntimeOperation::kAcquireLease));

  CHECK_STATUS(arena->seal());
  CHECK(arena->sealed());
  const NativeRuntimeError sealed_again = arena->seal();
  CHECK(sealed_again.code == NativeRuntimeCode::kAlreadySealed);
  CHECK(expect_result_error(arena->reserve(1, 1),
                            NativeRuntimeCode::kAlreadySealed,
                            NativeRuntimeOperation::kReserve));
  CHECK(take_result(arena->acquire_lease(), &lease));

  const NativeRuntimeError busy = arena->close();
  CHECK(busy.code == NativeRuntimeCode::kResourceBusy);
  CHECK(busy.actual == 3);
  lease.reset();
  second.reset();
  first.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

using MutableFloatVector =
    GraphStableTensorView<DType::kFloat32, 1, TensorAccess::kReadWrite>;

[[nodiscard]] bool OwnerBackedBinding() {
  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> first_arena;
  std::optional<GraphMemoryArena> second_arena;
  std::optional<GraphMemorySlice> first_slice;
  std::optional<GraphMemorySlice> adjacent_slice;
  std::optional<GraphMemorySlice> foreign_slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> first_view;
  std::optional<MutableFloatVector> adjacent_view;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 2048), &first_arena));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 1024), &second_arena));
  CHECK(take_result(first_arena->reserve(256, 256), &first_slice));
  CHECK(take_result(first_arena->reserve(256, 256), &adjacent_slice));
  CHECK(take_result(second_arena->reserve(256, 256), &foreign_slice));
  CHECK_STATUS(first_arena->seal());
  CHECK_STATUS(second_arena->seal());
  CHECK(take_result(first_arena->acquire_lease(), &lease));

  const auto metadata = vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, 64);
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(*first_slice, metadata),
      &first_view));
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(*adjacent_slice, metadata),
      &adjacent_view));
  CHECK(first_view->device_ordinal() == context->device_ordinal());
  CHECK(first_view->allocation_bytes() == 256);
  CHECK(first_view->data_bytes() != adjacent_view->data_bytes());
  CHECK(first_view->data_bytes() + first_view->allocation_bytes() <=
        adjacent_view->data_bytes());

  CHECK(expect_result_error(
      lease->bind_mutable<DType::kFloat32, 1>(*foreign_slice, metadata),
      NativeRuntimeCode::kForeignSlice, NativeRuntimeOperation::kBindTensor));

  auto conflicting = metadata;
  conflicting.device_kind = SGL_NATIVE_DEVICE_CUDA;
  auto conflict =
      lease->bind_mutable<DType::kFloat32, 1>(*first_slice, conflicting);
  CHECK(std::move(conflict).match(
      [](MutableFloatVector &&) noexcept { return false; },
      [](NativeRuntimeError &&error) noexcept {
        return error.code == NativeRuntimeCode::kOwnerMetadataConflict &&
               error.operation == NativeRuntimeOperation::kBindTensor &&
               error.detail ==
                   static_cast<uint32_t>(TensorValidationField::kDeviceKind);
      }));

  auto oversized = metadata;
  oversized.extents[0] = 65;
  auto invalid =
      lease->bind_mutable<DType::kFloat32, 1>(*first_slice, oversized);
  CHECK(std::move(invalid).match(
      [](MutableFloatVector &&) noexcept { return false; },
      [](NativeRuntimeError &&error) noexcept {
        return error.code == NativeRuntimeCode::kTensorValidationFailure &&
               error.operation == NativeRuntimeOperation::kBindTensor &&
               error.native_code ==
                   static_cast<int32_t>(TensorValidationCode::kOutOfBounds) &&
               error.detail ==
                   static_cast<uint32_t>(TensorValidationField::kSpan);
      }));

  CHECK(first_arena->close().code == NativeRuntimeCode::kResourceBusy);
  adjacent_view.reset();
  first_view.reset();
  lease.reset();
  foreign_slice.reset();
  adjacent_slice.reset();
  first_slice.reset();
  CHECK_STATUS(second_arena->close());
  second_arena.reset();
  CHECK_STATUS(first_arena->close());
  first_arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

__global__ void increment_kernel(float *values, uint32_t count) {
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] += 1.0F;
  }
}

__global__ void add_u32_kernel(uint32_t *value, uint32_t increment) {
  if (threadIdx.x == 0U) {
    value[0] += increment;
  }
}

struct IncrementCapturePayload final {
  const CudaExecutionContext *context;
  float *values;
  uint32_t count;
};

[[nodiscard]] NativeRuntimeError capture_increment(void *opaque) noexcept {
  const auto *payload = static_cast<const IncrementCapturePayload *>(opaque);
  increment_kernel<<<1, payload->count, 0, payload->context->stream()>>>(
      payload->values, payload->count);
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return NativeRuntimeError{NativeRuntimeCode::kCudaRuntimeFailure,
                              NativeRuntimeOperation::kGraphCaptureEnd,
                              static_cast<int32_t>(status),
                              0,
                              0,
                              0};
  }
  return sglang::native::native_runtime_ok();
}

struct RawGraph final {
  cudaGraph_t value = nullptr;

  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

[[nodiscard]] bool GraphReplayRetainsResources() {
  constexpr uint32_t kElements = 256;
  constexpr uint64_t kBytes = kElements * sizeof(float);

  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> view;
  std::optional<CudaGraphExecutable> executable;
  RawGraph graph;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 4096), &arena));
  CHECK(take_result(arena->reserve(kBytes, 256), &slice));
  CHECK_STATUS(arena->seal());
  CHECK(take_result(arena->acquire_lease(), &lease));
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(
          *slice, vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, kElements)),
      &view));
  float *const device_values = reinterpret_cast<float *>(view->data_bytes());

  CHECK_CUDA(cudaMemsetAsync(device_values, 0, static_cast<std::size_t>(kBytes),
                             context->stream()));
  CHECK_STATUS(context->synchronize());
  CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                    cudaStreamCaptureModeThreadLocal));
  increment_kernel<<<1, kElements, 0, context->stream()>>>(device_values,
                                                           kElements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamEndCapture(context->stream(), &graph.value));
  CHECK(take_result(
      CudaGraphExecutable::instantiate(graph.value, *context, *lease),
      &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));
  RawGraph executable_clone;
  CHECK(std::move(executable->clone_graph())
            .match(
                [&executable_clone](cudaGraph_t &&value) noexcept {
                  executable_clone.value = value;
                  return true;
                },
                [](NativeRuntimeError &&error) noexcept {
                  print_error(error);
                  return false;
                }));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(executable_clone.value, nullptr)));

  view.reset();
  lease.reset();
  slice.reset();
  const NativeRuntimeError retained = arena->close();
  CHECK(retained.code == NativeRuntimeCode::kResourceBusy);
  CHECK(retained.actual == 1);
  CHECK(stream->close().code == NativeRuntimeCode::kResourceBusy);

  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());

  std::array<float, kElements> host_values{};
  CHECK_CUDA(cudaMemcpy(host_values.data(), device_values,
                        static_cast<std::size_t>(kBytes),
                        cudaMemcpyDeviceToHost));
  for (const float value : host_values) {
    CHECK(value == 3.0F);
  }

  CHECK_STATUS(executable->close());
  executable.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

[[nodiscard]] bool CapturedGraphRetainsResources() {
  constexpr uint32_t kElements = 256;
  constexpr uint64_t kBytes = kElements * sizeof(float);

  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> view;
  std::optional<CudaCapturedGraph> graph;
  std::optional<CudaGraphExecutable> executable;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 4096), &arena));
  CHECK(take_result(arena->reserve(kBytes, 256), &slice));
  CHECK_STATUS(arena->seal());
  CHECK(take_result(arena->acquire_lease(), &lease));
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(
          *slice, vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, kElements)),
      &view));
  float *const device_values = reinterpret_cast<float *>(view->data_bytes());
  CHECK_CUDA(cudaMemset(device_values, 0, static_cast<size_t>(kBytes)));

  IncrementCapturePayload payload{&*context, device_values, kElements};
  CHECK(take_result(CudaCapturedGraph::capture(*context, *lease,
                                               &capture_increment, &payload),
                    &graph));
  CHECK(graph->valid());
  CHECK(graph->device_ordinal() == context->device_ordinal());
  CHECK(graph->graph() != nullptr);
  CHECK(take_result(
      CudaGraphExecutable::instantiate(graph->graph(), *context, *lease),
      &executable));

  view.reset();
  lease.reset();
  slice.reset();
  const NativeRuntimeError twice_retained = arena->close();
  CHECK(twice_retained.code == NativeRuntimeCode::kResourceBusy);
  CHECK(twice_retained.actual == 2U);
  CHECK(stream->close().code == NativeRuntimeCode::kResourceBusy);

  CHECK_STATUS(graph->close());
  graph.reset();
  const NativeRuntimeError executable_retained = arena->close();
  CHECK(executable_retained.code == NativeRuntimeCode::kResourceBusy);
  CHECK(executable_retained.actual == 1U);

  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  std::array<float, kElements> host_values{};
  CHECK_CUDA(cudaMemcpy(host_values.data(), device_values,
                        static_cast<size_t>(kBytes), cudaMemcpyDeviceToHost));
  for (const float value : host_values) {
    CHECK(value == 2.0F);
  }

  CHECK_STATUS(executable->close());
  executable.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

[[nodiscard]] bool ChildGraphSequencePreservesOrder() {
  constexpr uint32_t kElements = 256;
  constexpr uint64_t kBytes = kElements * sizeof(float);

  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> view;
  std::optional<CudaGraphExecutable> executable;
  RawGraph first_graph;
  RawGraph second_graph;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 4096), &arena));
  CHECK(take_result(arena->reserve(kBytes, 256), &slice));
  CHECK_STATUS(arena->seal());
  CHECK(take_result(arena->acquire_lease(), &lease));
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(
          *slice, vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, kElements)),
      &view));
  float *const device_values = reinterpret_cast<float *>(view->data_bytes());

  CHECK_CUDA(cudaMemsetAsync(device_values, 0, static_cast<std::size_t>(kBytes),
                             context->stream()));
  CHECK_STATUS(context->synchronize());
  CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                    cudaStreamCaptureModeThreadLocal));
  increment_kernel<<<1, kElements, 0, context->stream()>>>(device_values,
                                                           kElements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamEndCapture(context->stream(), &first_graph.value));
  CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                    cudaStreamCaptureModeThreadLocal));
  increment_kernel<<<1, kElements, 0, context->stream()>>>(device_values,
                                                           kElements);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaStreamEndCapture(context->stream(), &second_graph.value));

  const std::array<cudaGraph_t, 2> children{first_graph.value,
                                            second_graph.value};
  CHECK(take_result(CudaGraphExecutable::instantiate_child_sequence(
                        children, *context, *lease),
                    &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(first_graph.value, nullptr)));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(second_graph.value, nullptr)));

  view.reset();
  lease.reset();
  slice.reset();
  CHECK(arena->close().code == NativeRuntimeCode::kResourceBusy);

  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());

  std::array<float, kElements> host_values{};
  CHECK_CUDA(cudaMemcpy(host_values.data(), device_values,
                        static_cast<std::size_t>(kBytes),
                        cudaMemcpyDeviceToHost));
  for (const float value : host_values) {
    CHECK(value == 4.0F);
  }

  CHECK_STATUS(executable->close());
  executable.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

[[nodiscard]] bool CompactDeviceToHostResultCanBeBound() {
  constexpr uint32_t kWords = 4;
  constexpr uint64_t kBytes = kWords * sizeof(float);

  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> view;
  std::optional<CudaGraphExecutable> executable;
  RawGraph graph;
  float *host_result = nullptr;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 256), &arena));
  CHECK(take_result(arena->reserve(kBytes, 16), &slice));
  CHECK_STATUS(arena->seal());
  CHECK(take_result(arena->acquire_lease(), &lease));
  CHECK(take_result(
      lease->bind_mutable<DType::kFloat32, 1>(
          *slice, vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, kWords)),
      &view));
  float *const device_values = reinterpret_cast<float *>(view->data_bytes());
  CHECK_CUDA(cudaMallocHost(&host_result, static_cast<size_t>(kBytes)));
  float *capture_destination = nullptr;
  CHECK_CUDA(cudaMallocHost(&capture_destination, static_cast<size_t>(kBytes)));
  for (uint32_t index = 0; index < kWords; ++index) {
    capture_destination[index] = 0.0F;
  }

  CHECK_CUDA(cudaMemsetAsync(device_values, 0, static_cast<std::size_t>(kBytes),
                             context->stream()));
  CHECK_STATUS(context->synchronize());
  CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                    cudaStreamCaptureModeThreadLocal));
  increment_kernel<<<1, kWords, 0, context->stream()>>>(device_values, kWords);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpyAsync(capture_destination, device_values,
                             static_cast<size_t>(kBytes),
                             cudaMemcpyDeviceToHost, context->stream()));
  CHECK_CUDA(cudaStreamEndCapture(context->stream(), &graph.value));
  CHECK(take_result(
      CudaGraphExecutable::instantiate(graph.value, *context, *lease),
      &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));
  CHECK_STATUS(executable->bind_compact_device_to_host_result(
      host_result, device_values, kBytes));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  for (uint32_t index = 0; index < kWords; ++index) {
    CHECK(host_result[index] == 1.0F);
    CHECK(capture_destination[index] == 0.0F);
  }

  CHECK_STATUS(executable->close());
  executable.reset();
  view.reset();
  lease.reset();
  slice.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  CHECK_CUDA(cudaFreeHost(host_result));
  CHECK_CUDA(cudaFreeHost(capture_destination));
  return true;
}

[[nodiscard]] bool EightStageCycleComposition() {
  constexpr uint32_t kStageCount = 8;
  constexpr uint64_t kBytes = sizeof(uint32_t);

  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;
  std::optional<GraphMemorySlice> slice;
  std::optional<GraphArenaLease> lease;
  std::optional<MutableFloatVector> storage;
  std::optional<CudaGraphExecutable> executable;
  std::array<RawGraph, kStageCount> graphs;
  std::array<cudaGraph_t, kStageCount> children{};
  uint32_t *host_result = nullptr;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK(take_result(GraphMemoryArena::allocate(*context, 256), &arena));
  CHECK(take_result(arena->reserve(kBytes, alignof(uint32_t)), &slice));
  CHECK_STATUS(arena->seal());
  CHECK(take_result(arena->acquire_lease(), &lease));
  CHECK(take_result(lease->bind_mutable<DType::kFloat32, 1>(
                        *slice, vector_metadata(SGL_NATIVE_DTYPE_FLOAT32, 1)),
                    &storage));
  auto *const device_result =
      reinterpret_cast<uint32_t *>(storage->data_bytes());
  CHECK_CUDA(cudaMallocHost(&host_result, static_cast<size_t>(kBytes)));
  *host_result = 0U;
  CHECK_CUDA(cudaMemsetAsync(device_result, 0, static_cast<size_t>(kBytes),
                             context->stream()));
  CHECK_STATUS(context->synchronize());

  for (uint32_t stage = 0; stage < kStageCount - 1; ++stage) {
    CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                      cudaStreamCaptureModeThreadLocal));
    add_u32_kernel<<<1, 1, 0, context->stream()>>>(device_result, stage + 1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamEndCapture(context->stream(), &graphs[stage].value));
    children[stage] = graphs[stage].value;
  }
  CHECK_CUDA(cudaStreamBeginCapture(context->stream(),
                                    cudaStreamCaptureModeThreadLocal));
  add_u32_kernel<<<1, 1, 0, context->stream()>>>(device_result, kStageCount);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpyAsync(host_result, device_result,
                             static_cast<size_t>(kBytes),
                             cudaMemcpyDeviceToHost, context->stream()));
  CHECK_CUDA(
      cudaStreamEndCapture(context->stream(), &graphs[kStageCount - 1].value));
  children[kStageCount - 1] = graphs[kStageCount - 1].value;

  CHECK(take_result(CudaGraphExecutable::instantiate_child_sequence(
                        children, *context, *lease),
                    &executable));
  for (RawGraph &graph : graphs) {
    CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));
  }
  CHECK_STATUS(executable->bind_compact_device_to_host_result(
      host_result, device_result, kBytes));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  CHECK(*host_result == 36U);

  CHECK_STATUS(executable->close());
  executable.reset();
  storage.reset();
  lease.reset();
  slice.reset();
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  CHECK_CUDA(cudaFreeHost(host_result));
  return true;
}

[[nodiscard]] bool DeviceMismatchFailsClosed() {
  int device_count = 0;
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::printf("[  SKIPPED ] DeviceMismatchFailsClosed requires 2 GPUs\n");
    return true;
  }

  int original_device = -1;
  CHECK_CUDA(cudaGetDevice(&original_device));
  const int alternate_device = original_device == 0 ? 1 : 0;
  std::optional<CudaStream> stream;
  std::optional<CudaExecutionContext> context;
  std::optional<GraphMemoryArena> arena;

  CHECK(take_result(CudaStream::create_nonblocking(), &stream));
  CHECK(take_result(stream->context(), &context));
  CHECK_CUDA(cudaSetDevice(alternate_device));

  auto allocation = GraphMemoryArena::allocate(*context, 256);
  CHECK(std::move(allocation)
            .match([](GraphMemoryArena &&) noexcept { return false; },
                   [alternate_device,
                    original_device](NativeRuntimeError &&error) noexcept {
                     return error.code == NativeRuntimeCode::kDeviceMismatch &&
                            error.operation ==
                                NativeRuntimeOperation::kDeviceAllocate &&
                            error.actual ==
                                static_cast<uint64_t>(alternate_device) &&
                            error.required ==
                                static_cast<uint64_t>(original_device);
                   }));
  const NativeRuntimeError sync = context->synchronize();
  CHECK(sync.code == NativeRuntimeCode::kDeviceMismatch);

  CHECK_CUDA(cudaSetDevice(original_device));
  CHECK_STATUS(context->synchronize());
  CHECK(take_result(GraphMemoryArena::allocate(*context, 256), &arena));
  CHECK_STATUS(arena->close());
  arena.reset();
  context.reset();
  CHECK_STATUS(stream->close());
  stream.reset();
  return true;
}

struct TestCase final {
  const char *name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"StreamContextLifetime", StreamContextLifetime},
    {"ArenaReservationLifecycle", ArenaReservationLifecycle},
    {"OwnerBackedBinding", OwnerBackedBinding},
    {"GraphReplayRetainsResources", GraphReplayRetainsResources},
    {"CapturedGraphRetainsResources", CapturedGraphRetainsResources},
    {"ChildGraphSequencePreservesOrder", ChildGraphSequencePreservesOrder},
    {"CompactDeviceToHostResultCanBeBound",
     CompactDeviceToHostResultCanBeBound},
    {"EightStageCycleComposition", EightStageCycleComposition},
    {"DeviceMismatchFailsClosed", DeviceMismatchFailsClosed},
};

} // namespace

int main() {
  uint32_t passed = 0;
  for (const auto &test : kTests) {
    std::printf("[ RUN      ] %s\n", test.name);
    if (!test.function()) {
      std::printf("[  FAILED  ] %s\n", test.name);
      return 1;
    }
    ++passed;
    std::printf("[       OK ] %s\n", test.name);
  }
  std::printf("[  PASSED  ] %u tests\n", passed);
  return 0;
}
