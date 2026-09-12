#include "sglang/native/dspark_cycle_controller.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <utility>

namespace sglang::native {
namespace {

struct ByteRange final {
  uintptr_t begin;
  uintptr_t end;
  DsparkCycleArgument argument;
};

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           DsparkCycleArgument argument = DsparkCycleArgument::kNone,
           int32_t native_code = 0, uint64_t actual = 0,
           uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,        operation,
                            native_code, static_cast<uint32_t>(argument),
                            actual,      required};
}

template <DType D, TensorAccess Access>
[[nodiscard]] NativeRuntimeError
validate_vector(const GraphStableTensorView<D, 1, Access> &view,
                const CudaExecutionContext &context,
                DsparkCycleArgument argument, int64_t expected_elements,
                ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkCycle;
  if (!context.valid() || range == nullptr || expected_elements <= 0) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (!view.is_row_major_contiguous() ||
      view.extents()[0] != expected_elements) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, static_cast<uint64_t>(view.extents()[0]),
                      static_cast<uint64_t>(expected_elements));
  }
  constexpr uint64_t kElementBytes = dtype_element_bits(D) / 8U;
  const uint64_t expected_bytes =
      static_cast<uint64_t>(expected_elements) * kElementBytes;
  if (view.allocation_bytes() != expected_bytes) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, view.allocation_bytes(), expected_bytes);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(view.data_bytes());
  if (begin == 0 || expected_bytes > std::numeric_limits<uintptr_t>::max() ||
      begin > std::numeric_limits<uintptr_t>::max() - expected_bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + static_cast<uintptr_t>(expected_bytes),
                     argument};
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(const ByteRange &left,
                            const ByteRange &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] NativeRuntimeError
validate_buffers(const CudaExecutionContext &context,
                 const DsparkCycleCompactResultBuffers &buffers) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkCycle;
  if (!context.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation);
  }
  int current_device = -1;
  const cudaError_t current = cudaGetDevice(&current_device);
  if (current != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
                      DsparkCycleArgument::kNone,
                      static_cast<int32_t>(current));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      DsparkCycleArgument::kDeviceOrdinal, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(context.device_ordinal()));
  }

  std::array<ByteRange, 5> ranges{};
  NativeRuntimeError status = validate_vector(
      buffers.out_tokens, context, DsparkCycleArgument::kOutTokens,
      kDsparkCycleProductionNumOutTokens, &ranges[0]);
  if (!is_ok(status)) {
    return status;
  }
  status =
      validate_vector(buffers.num_correct_drafts, context,
                      DsparkCycleArgument::kNumCorrectDrafts, 1, &ranges[1]);
  if (!is_ok(status)) {
    return status;
  }
  status = validate_vector(buffers.request_slot, context,
                           DsparkCycleArgument::kRequestSlot, 1, &ranges[2]);
  if (!is_ok(status)) {
    return status;
  }
  status = validate_vector(buffers.device_status, context,
                           DsparkCycleArgument::kDeviceStatus, 1, &ranges[3]);
  if (!is_ok(status)) {
    return status;
  }
  status = validate_vector(
      buffers.result_words, context, DsparkCycleArgument::kDeviceResult,
      sizeof(DsparkCycleResultV1) / sizeof(uint32_t), &ranges[4]);
  if (!is_ok(status)) {
    return status;
  }
  for (uint32_t left = 0; left < ranges.size(); ++left) {
    for (uint32_t right = left + 1; right < ranges.size(); ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                          ranges[left].argument, 0,
                          static_cast<uint64_t>(ranges[right].argument));
      }
    }
  }
  return native_runtime_ok();
}

__global__ void compact_cycle_result_kernel(const int32_t *out_tokens,
                                            const int32_t *num_correct_drafts,
                                            const int32_t *request_slot,
                                            uint32_t *device_status,
                                            DsparkCycleResultV1 *result) {
  if (blockIdx.x != 0U || threadIdx.x != 0U) {
    return;
  }

  uint32_t status = device_status[0];
  int32_t correct = 0;
  int32_t accepted = 0;
  int32_t output = -1;
  const int32_t slot = request_slot[0];
  if (status == 0U) {
    correct = num_correct_drafts[0];
    if (correct < 0 ||
        correct >= static_cast<int32_t>(kDsparkCycleProductionNumOutTokens)) {
      status = static_cast<uint32_t>(
          DsparkCycleDeviceCode::kNumCorrectDraftsOutOfRange);
      device_status[0] = status;
      correct = 0;
    } else {
      accepted = correct + 1;
      output = out_tokens[correct];
    }
  }
  *result = DsparkCycleResultV1{
      kDsparkCycleAbiVersion, status, accepted, correct, output, slot};
}

struct CompactCapturePayload final {
  const CudaExecutionContext *context;
  const DsparkCycleCompactResultBuffers *buffers;
  const CudaPinnedHostBuffer *host_destination;
};

[[nodiscard]] NativeRuntimeError capture_compact_body(void *opaque) noexcept {
  const auto *payload = static_cast<const CompactCapturePayload *>(opaque);
  const NativeRuntimeError launch =
      launch_dspark_cycle_compact_result(*payload->context, *payload->buffers);
  if (!is_ok(launch)) {
    return launch;
  }
  const cudaError_t copy = cudaMemcpyAsync(
      payload->host_destination->data(),
      payload->buffers->result_words.data_bytes(), sizeof(DsparkCycleResultV1),
      cudaMemcpyDeviceToHost, payload->context->stream());
  return copy == cudaSuccess
             ? native_runtime_ok()
             : make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                          NativeRuntimeOperation::kGraphCaptureEnd,
                          DsparkCycleArgument::kDeviceResult,
                          static_cast<int32_t>(copy));
}

} // namespace

DsparkCycleCompactGraph::DsparkCycleCompactGraph(
    CudaCapturedGraph graph, CudaPinnedHostBuffer capture_result) noexcept
    : graph_(std::move(graph)), capture_result_(std::move(capture_result)) {}

DsparkCycleCompactGraph::DsparkCycleCompactGraph(
    DsparkCycleCompactGraph &&other) noexcept
    : graph_(std::move(other.graph_)),
      capture_result_(std::move(other.capture_result_)) {}

DsparkCycleCompactGraph::~DsparkCycleCompactGraph() noexcept {
  if (!is_ok(close())) {
    std::terminate();
  }
}

NativeRuntimeResult<DsparkCycleCompactGraph> DsparkCycleCompactGraph::capture(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const DsparkCycleCompactResultBuffers &buffers) noexcept {
  using GraphResult = NativeRuntimeResult<DsparkCycleCompactGraph>;
  if (!context.valid() || !arena.valid()) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCaptureBegin));
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCaptureBegin,
                   DsparkCycleArgument::kDeviceOrdinal, 0,
                   static_cast<uint64_t>(context.device_ordinal()),
                   static_cast<uint64_t>(arena.device_ordinal())));
  }
  const std::array<bool, 5> owned{buffers.out_tokens.belongs_to(arena),
                                  buffers.num_correct_drafts.belongs_to(arena),
                                  buffers.request_slot.belongs_to(arena),
                                  buffers.device_status.belongs_to(arena),
                                  buffers.result_words.belongs_to(arena)};
  constexpr std::array<DsparkCycleArgument, 5> arguments{
      DsparkCycleArgument::kOutTokens, DsparkCycleArgument::kNumCorrectDrafts,
      DsparkCycleArgument::kRequestSlot, DsparkCycleArgument::kDeviceStatus,
      DsparkCycleArgument::kDeviceResult};
  for (std::size_t index = 0; index < owned.size(); ++index) {
    if (!owned[index]) {
      return GraphResult::failure(make_error(
          NativeRuntimeCode::kForeignSlice,
          NativeRuntimeOperation::kGraphCaptureBegin, arguments[index]));
    }
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
    return GraphResult::failure(allocation_error);
  }
  CompactCapturePayload payload{&context, &buffers, &*host_result};
  return std::move(CudaCapturedGraph::capture(context, arena,
                                              &capture_compact_body, &payload))
      .match(
          [&host_result](CudaCapturedGraph &&graph) noexcept -> GraphResult {
            return GraphResult::success(DsparkCycleCompactGraph(
                std::move(graph), std::move(*host_result)));
          },
          [](NativeRuntimeError &&error) noexcept -> GraphResult {
            return GraphResult::failure(std::move(error));
          });
}

NativeRuntimeError DsparkCycleCompactGraph::close() noexcept {
  const NativeRuntimeError graph_status = graph_.close();
  if (!is_ok(graph_status)) {
    return graph_status;
  }
  return capture_result_.close();
}

bool DsparkCycleCompactGraph::valid() const noexcept {
  return graph_.valid() && capture_result_.valid();
}

const CudaCapturedGraph &DsparkCycleCompactGraph::graph() const noexcept {
  return graph_;
}

NativeRuntimeError launch_dspark_cycle_compact_result(
    const CudaExecutionContext &context,
    const DsparkCycleCompactResultBuffers &buffers) noexcept {
  const NativeRuntimeError validation = validate_buffers(context, buffers);
  if (!is_ok(validation)) {
    return validation;
  }
  compact_cycle_result_kernel<<<1, 1, 0, context.stream()>>>(
      reinterpret_cast<const int32_t *>(buffers.out_tokens.data_bytes()),
      reinterpret_cast<const int32_t *>(
          buffers.num_correct_drafts.data_bytes()),
      reinterpret_cast<const int32_t *>(buffers.request_slot.data_bytes()),
      reinterpret_cast<uint32_t *>(buffers.device_status.data_bytes()),
      reinterpret_cast<DsparkCycleResultV1 *>(
          buffers.result_words.data_bytes()));
  const cudaError_t launch = cudaGetLastError();
  if (launch != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchDsparkCycleCompactResult,
                      DsparkCycleArgument::kNone, static_cast<int32_t>(launch));
  }
  return native_runtime_ok();
}

} // namespace sglang::native
