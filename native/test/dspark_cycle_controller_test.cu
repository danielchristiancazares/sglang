#include "sglang/native/dspark_cycle_controller.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace {

using sglang::native::CudaCapturedGraph;
using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaPinnedHostBuffer;
using sglang::native::CudaStream;
using sglang::native::dspark_production_model_graph_shape;
using sglang::native::dspark_required_model_graph_outputs;
using sglang::native::DsparkCycleChildGraph;
using sglang::native::DsparkCycleCompactGraph;
using sglang::native::DsparkCycleCompactResultBuffers;
using sglang::native::DsparkCycleController;
using sglang::native::DsparkCycleCoreGraphs;
using sglang::native::DsparkCycleDescriptor;
using sglang::native::DsparkCycleDeviceCode;
using sglang::native::DsparkCycleGraphSources;
using sglang::native::DsparkCycleResultV1;
using sglang::native::DsparkCycleStage;
using sglang::native::DsparkDraftExtendGraphSource;
using sglang::native::DsparkKvWriteGraphSource;
using sglang::native::DsparkModelGraphBinding;
using sglang::native::DsparkModelGraphKind;
using sglang::native::DsparkModelGraphResourceOwner;
using sglang::native::DsparkTargetVerifyGraphSource;
using sglang::native::DType;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::is_ok;
using sglang::native::kDsparkCycleAbiVersion;
using sglang::native::kDsparkCycleProductionNumOutTokens;
using sglang::native::launch_dspark_cycle_compact_result;
using sglang::native::make_dspark_cycle_child_graphs;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeResult;
using sglang::native::TensorAccess;

[[nodiscard]] bool check(bool condition, const char *expression, int line) {
  if (!condition) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return condition;
}

#define CHECK(expression)                                                      \
  do {                                                                         \
    if (!check(static_cast<bool>(expression), #expression, __LINE__)) {        \
      return false;                                                            \
    }                                                                          \
  } while (false)

#define CHECK_CUDA(expression)                                                 \
  do {                                                                         \
    const cudaError_t status = (expression);                                   \
    if (!check(status == cudaSuccess, #expression, __LINE__)) {                \
      return false;                                                            \
    }                                                                          \
  } while (false)

#define CHECK_STATUS(expression) CHECK(is_ok(expression))

template <typename T>
[[nodiscard]] bool take_result(NativeRuntimeResult<T> result,
                               std::optional<T> *output) noexcept {
  return std::move(result).match(
      [output](T &&value) noexcept {
        output->emplace(std::move(value));
        return true;
      },
      [](NativeRuntimeError &&) noexcept { return false; });
}

[[nodiscard]] SglNativeTensorMetadataV1
vector_metadata(SglNativeDType dtype, int64_t elements,
                uint64_t storage_offset_elements = 0) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 1;
  metadata.extents[0] = elements;
  metadata.strides[0] = 1;
  metadata.storage_offset_elements = storage_offset_elements;
  return metadata;
}

template <DType D, uint32_t Rank, typename T>
[[nodiscard]] bool copy_to_const(const sglang::native::GraphStableTensorView<
                                     D, Rank, TensorAccess::kReadOnly> &view,
                                 std::span<const T> values) noexcept {
  return cudaMemcpy(const_cast<std::byte *>(view.data_bytes()), values.data(),
                    values.size_bytes(), cudaMemcpyHostToDevice) == cudaSuccess;
}

template <DType D, uint32_t Rank, typename T>
[[nodiscard]] bool copy_to_mutable(const sglang::native::GraphStableTensorView<
                                       D, Rank, TensorAccess::kReadWrite> &view,
                                   std::span<const T> values) noexcept {
  return cudaMemcpy(view.data_bytes(), values.data(), values.size_bytes(),
                    cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T, DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] bool
copy_to_host(std::span<T> values,
             const sglang::native::GraphStableTensorView<D, Rank, Access>
                 &view) noexcept {
  return cudaMemcpy(values.data(), view.data_bytes(), values.size_bytes(),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

struct RawGraph final {
  cudaGraph_t value = nullptr;
  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

[[nodiscard]] DsparkModelGraphBinding
model_binding(DsparkModelGraphKind kind, const CudaCapturedGraph &graph,
              const DsparkModelGraphResourceOwner &owner) noexcept {
  return DsparkModelGraphBinding{kind,
                                 0U,
                                 dspark_required_model_graph_outputs(kind),
                                 dspark_production_model_graph_shape(),
                                 graph,
                                 owner.acquire_lease()};
}

class CompactResultFixture final {
public:
  CompactResultFixture(const CompactResultFixture &) = delete;
  CompactResultFixture &operator=(const CompactResultFixture &) = delete;
  CompactResultFixture(CompactResultFixture &&) = delete;
  CompactResultFixture &operator=(CompactResultFixture &&) = delete;
  CompactResultFixture() = default;

  [[nodiscard]] bool
  initialize(bool alias_count_and_status = false,
             int64_t result_elements = sizeof(DsparkCycleResultV1) /
                                       sizeof(uint32_t)) noexcept {
    constexpr uint64_t kOutBytes =
        kDsparkCycleProductionNumOutTokens * sizeof(int32_t);
    constexpr uint64_t kCountBytes = sizeof(int32_t);
    constexpr uint64_t kRequestSlotBytes = sizeof(int32_t);
    constexpr uint64_t kStatusBytes = sizeof(uint32_t);
    constexpr uint64_t kResultBytes = sizeof(DsparkCycleResultV1);
    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(GraphMemoryArena::allocate(*context_, 512), &arena_) ||
        !take_result(arena_->reserve(kOutBytes, alignof(int32_t)),
                     &out_tokens_slice_) ||
        !take_result(arena_->reserve(kCountBytes, alignof(int32_t)),
                     &num_correct_drafts_slice_)) {
      return false;
    }
    if (alias_count_and_status) {
      device_status_slice_.emplace(*num_correct_drafts_slice_);
    } else if (!take_result(arena_->reserve(kStatusBytes, alignof(uint32_t)),
                            &device_status_slice_)) {
      return false;
    }
    if (!take_result(arena_->reserve(kRequestSlotBytes, alignof(int32_t)),
                     &request_slot_slice_)) {
      return false;
    }
    if (!take_result(arena_->reserve(kResultBytes, alignof(uint32_t)),
                     &result_words_slice_)) {
      return false;
    }
    if (!is_ok(arena_->seal()) ||
        !take_result(arena_->acquire_lease(), &lease_) ||
        !take_result(lease_->bind_const<DType::kInt32, 1>(
                         *out_tokens_slice_,
                         vector_metadata(SGL_NATIVE_DTYPE_INT32,
                                         kDsparkCycleProductionNumOutTokens)),
                     &out_tokens_) ||
        !take_result(lease_->bind_const<DType::kInt32, 1>(
                         *num_correct_drafts_slice_,
                         vector_metadata(SGL_NATIVE_DTYPE_INT32, 1)),
                     &num_correct_drafts_) ||
        !take_result(lease_->bind_const<DType::kInt32, 1>(
                         *request_slot_slice_,
                         vector_metadata(SGL_NATIVE_DTYPE_INT32, 1)),
                     &request_slot_) ||
        !take_result(lease_->bind_mutable<DType::kUInt32, 1>(
                         *device_status_slice_,
                         vector_metadata(SGL_NATIVE_DTYPE_UINT32, 1)),
                     &device_status_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt32, 1>(
                *result_words_slice_,
                vector_metadata(SGL_NATIVE_DTYPE_UINT32, result_elements)),
            &result_words_)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] DsparkCycleCompactResultBuffers buffers() const noexcept {
    return {*out_tokens_, *num_correct_drafts_, *request_slot_, *device_status_,
            *result_words_};
  }

  [[nodiscard]] bool write_inputs(
      std::span<const int32_t, kDsparkCycleProductionNumOutTokens> out_tokens,
      int32_t num_correct_drafts, uint32_t device_status,
      int32_t request_slot = 0) noexcept {
    const std::array<uint32_t, sizeof(DsparkCycleResultV1) / sizeof(uint32_t)>
        sentinels{0xccccccccU, 0xccccccccU, 0xccccccccU,
                  0xccccccccU, 0xccccccccU, 0xccccccccU};
    return copy_to_mutable(*result_words_,
                           std::span<const uint32_t>(sentinels)) &&
           copy_to_const(*out_tokens_, std::span<const int32_t>(out_tokens)) &&
           copy_to_const(*num_correct_drafts_,
                         std::span<const int32_t>(&num_correct_drafts, 1)) &&
           copy_to_const(*request_slot_,
                         std::span<const int32_t>(&request_slot, 1)) &&
           copy_to_mutable(*device_status_,
                           std::span<const uint32_t>(&device_status, 1));
  }

  [[nodiscard]] bool read_result(DsparkCycleResultV1 *result) const noexcept {
    return result != nullptr &&
           copy_to_host(
               std::span<uint32_t>(reinterpret_cast<uint32_t *>(result),
                                   sizeof(*result) / sizeof(uint32_t)),
               *result_words_);
  }

  [[nodiscard]] bool read_status(uint32_t *status) const noexcept {
    return status != nullptr &&
           copy_to_host(std::span<uint32_t>(status, 1), *device_status_);
  }

  [[nodiscard]] uintptr_t result_address() const noexcept {
    return reinterpret_cast<uintptr_t>(result_words_->data_bytes());
  }
  [[nodiscard]] const void *result_data() const noexcept {
    return result_words_->data_bytes();
  }
  [[nodiscard]] int32_t *out_tokens_data() const noexcept {
    return reinterpret_cast<int32_t *>(
        const_cast<std::byte *>(out_tokens_->data_bytes()));
  }

  [[nodiscard]] const CudaExecutionContext &context() const noexcept {
    return *context_;
  }

  [[nodiscard]] const GraphArenaLease &lease() const noexcept {
    return *lease_;
  }
  [[nodiscard]] void *device_status_data() const noexcept {
    return device_status_->data_bytes();
  }

private:
  std::optional<CudaStream> stream_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphMemoryArena> arena_;
  std::optional<GraphMemorySlice> out_tokens_slice_;
  std::optional<GraphMemorySlice> num_correct_drafts_slice_;
  std::optional<GraphMemorySlice> device_status_slice_;
  std::optional<GraphMemorySlice> request_slot_slice_;
  std::optional<GraphMemorySlice> result_words_slice_;
  std::optional<GraphArenaLease> lease_;
  std::optional<sglang::native::DsparkCycleConstInt32Vector> out_tokens_;
  std::optional<sglang::native::DsparkCycleConstInt32Vector>
      num_correct_drafts_;
  std::optional<sglang::native::DsparkCycleConstInt32Vector> request_slot_;
  std::optional<sglang::native::DsparkCycleMutableUInt32Vector> device_status_;
  std::optional<sglang::native::DsparkCycleMutableUInt32Vector> result_words_;
};

[[nodiscard]] bool CompactResultPublishesBoundariesAndBonus() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize());
  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> tokens{
      101, 102, 103, 104, 105, 106, 107, 108};

  CHECK(fixture.write_inputs(tokens, 0, 0, 41));
  CHECK_STATUS(
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers()));
  CHECK_STATUS(fixture.context().synchronize());
  DsparkCycleResultV1 result{};
  CHECK(fixture.read_result(&result));
  CHECK(result.abi_version == kDsparkCycleAbiVersion);
  CHECK(result.device_status == 0U);
  CHECK(result.accepted_token_count == 1);
  CHECK(result.num_correct_drafts == 0);
  CHECK(result.output_token == tokens[0]);
  CHECK(result.request_slot == 41);

  CHECK(fixture.write_inputs(tokens, 7, 0, 42));
  CHECK_STATUS(
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers()));
  CHECK_STATUS(fixture.context().synchronize());
  CHECK(fixture.read_result(&result));
  CHECK(result.device_status == 0U);
  CHECK(result.accepted_token_count == 8);
  CHECK(result.num_correct_drafts == 7);
  CHECK(result.output_token == tokens[7]);
  CHECK(result.request_slot == 42);
  return true;
}

[[nodiscard]] bool CompactResultPreservesUpstreamStatus() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize());
  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> tokens{
      201, 202, 203, 204, 205, 206, 207, 208};
  constexpr uint32_t kUpstreamStatus = 0xabcddcbaU;
  CHECK(fixture.write_inputs(tokens, 5, kUpstreamStatus, 43));
  CHECK_STATUS(
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers()));
  CHECK_STATUS(fixture.context().synchronize());
  DsparkCycleResultV1 result{};
  uint32_t status = 0;
  CHECK(fixture.read_result(&result));
  CHECK(fixture.read_status(&status));
  CHECK(result.abi_version == kDsparkCycleAbiVersion);
  CHECK(result.device_status == kUpstreamStatus);
  CHECK(result.accepted_token_count == 0);
  CHECK(result.num_correct_drafts == 0);
  CHECK(result.output_token == -1);
  CHECK(result.request_slot == 43);
  CHECK(status == kUpstreamStatus);
  return true;
}

[[nodiscard]] bool CompactResultRejectsInvalidCounts() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize());
  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> tokens{
      301, 302, 303, 304, 305, 306, 307, 308};
  constexpr std::array<int32_t, 2> invalid_counts{
      -1, static_cast<int32_t>(kDsparkCycleProductionNumOutTokens)};
  for (const int32_t count : invalid_counts) {
    CHECK(fixture.write_inputs(tokens, count, 0));
    CHECK_STATUS(launch_dspark_cycle_compact_result(fixture.context(),
                                                    fixture.buffers()));
    CHECK_STATUS(fixture.context().synchronize());
    DsparkCycleResultV1 result{};
    uint32_t status = 0;
    CHECK(fixture.read_result(&result));
    CHECK(fixture.read_status(&status));
    const uint32_t expected = static_cast<uint32_t>(
        DsparkCycleDeviceCode::kNumCorrectDraftsOutOfRange);
    CHECK(result.device_status == expected);
    CHECK(result.accepted_token_count == 0);
    CHECK(result.num_correct_drafts == 0);
    CHECK(result.output_token == -1);
    CHECK(result.request_slot == 0);
    CHECK(status == expected);
  }
  return true;
}

[[nodiscard]] bool CompactResultCapturedReplayUsesStableAddresses() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize());
  const uintptr_t result_address = fixture.result_address();
  std::optional<CudaPinnedHostBuffer> host_result;
  CHECK(take_result(CudaPinnedHostBuffer::allocate(sizeof(DsparkCycleResultV1)),
                    &host_result));
  RawGraph graph;
  CHECK_CUDA(cudaStreamBeginCapture(fixture.context().stream(),
                                    cudaStreamCaptureModeThreadLocal));
  CHECK_STATUS(
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers()));
  CHECK_CUDA(cudaMemcpyAsync(
      host_result->data(), fixture.result_data(), sizeof(DsparkCycleResultV1),
      cudaMemcpyDeviceToHost, fixture.context().stream()));
  CHECK_CUDA(cudaStreamEndCapture(fixture.context().stream(), &graph.value));
  size_t node_count = 0;
  CHECK_CUDA(cudaGraphGetNodes(graph.value, nullptr, &node_count));
  std::array<cudaGraphNode_t, 2> nodes{};
  CHECK(node_count == nodes.size());
  CHECK_CUDA(cudaGraphGetNodes(graph.value, nodes.data(), &node_count));
  uint32_t compact_copy_count = 0;
  for (const cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CHECK_CUDA(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeMemcpy) {
      continue;
    }
    cudaMemcpy3DParms params{};
    CHECK_CUDA(cudaGraphMemcpyNodeGetParams(node, &params));
    CHECK(params.kind == cudaMemcpyDeviceToHost);
    CHECK(params.srcPtr.ptr == fixture.result_data());
    CHECK(params.extent.width == sizeof(DsparkCycleResultV1));
    CHECK(params.extent.height == 1U);
    CHECK(params.extent.depth == 1U);
    ++compact_copy_count;
  }
  CHECK(compact_copy_count == 1U);
  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(CudaGraphExecutable::instantiate(
                        graph.value, fixture.context(), fixture.lease()),
                    &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));

  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> first_tokens{
      401, 402, 403, 404, 405, 406, 407, 408};
  CHECK(fixture.write_inputs(first_tokens, 2, 0));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  const DsparkCycleResultV1 first =
      *static_cast<const DsparkCycleResultV1 *>(host_result->data());
  CHECK(first.accepted_token_count == 3);
  CHECK(first.output_token == first_tokens[2]);
  CHECK(first.request_slot == 0);
  CHECK(fixture.result_address() == result_address);

  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> second_tokens{
      501, 502, 503, 504, 505, 506, 507, 508};
  CHECK(fixture.write_inputs(second_tokens, 6, 0));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  const DsparkCycleResultV1 second =
      *static_cast<const DsparkCycleResultV1 *>(host_result->data());
  CHECK(second.accepted_token_count == 7);
  CHECK(second.output_token == second_tokens[6]);
  CHECK(second.request_slot == 0);
  CHECK(second.output_token != first.output_token);
  CHECK(fixture.result_address() == result_address);
  CHECK_STATUS(executable->close());
  CHECK_STATUS(host_result->close());
  return true;
}

[[nodiscard]] bool CompactResultRejectsAliasing() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize(true));
  const NativeRuntimeError status =
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers());
  CHECK(status.code == NativeRuntimeCode::kInvalidArgument);
  return true;
}

[[nodiscard]] bool CompactResultRejectsMalformedLayout() {
  CompactResultFixture fixture;
  CHECK(fixture.initialize(false,
                           sizeof(DsparkCycleResultV1) / sizeof(uint32_t) - 1));
  const NativeRuntimeError status =
      launch_dspark_cycle_compact_result(fixture.context(), fixture.buffers());
  CHECK(status.code == NativeRuntimeCode::kInvalidArgument);
  return true;
}

__global__ void append_stage_digit(int32_t *value, uint32_t digit) {
  if (threadIdx.x == 0U) {
    value[0] = value[0] * 10 + static_cast<int32_t>(digit);
  }
}

struct StageCapturePayload final {
  const CudaExecutionContext *context;
  int32_t *value;
  uint32_t digit;
};

[[nodiscard]] NativeRuntimeError capture_stage_digit(void *opaque) noexcept {
  const auto *payload = static_cast<const StageCapturePayload *>(opaque);
  append_stage_digit<<<1, 1, 0, payload->context->stream()>>>(payload->value,
                                                              payload->digit);
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess
             ? sglang::native::native_runtime_ok()
             : NativeRuntimeError{
                   NativeRuntimeCode::kCudaRuntimeFailure,
                   sglang::native::NativeRuntimeOperation::kGraphCaptureEnd,
                   static_cast<int32_t>(status),
                   0,
                   0,
                   0};
}

[[nodiscard]] bool ControllerComposesRealCompactResult() {
  CompactResultFixture fixture;
  std::optional<DsparkCycleController> controller;
  std::array<RawGraph, sglang::native::kDsparkCycleNumStages> graphs;
  std::array<DsparkCycleChildGraph, sglang::native::kDsparkCycleNumStages>
      children{};
  std::optional<CudaPinnedHostBuffer> capture_result;

  CHECK(fixture.initialize());
  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> tokens{
      601, 602, 603, 604, 605, 606, 607, 0};
  CHECK(fixture.write_inputs(tokens, 7, 0xabcddcbaU));
  CHECK(take_result(CudaPinnedHostBuffer::allocate(sizeof(DsparkCycleResultV1)),
                    &capture_result));
  *static_cast<DsparkCycleResultV1 *>(capture_result->data()) = {};

  for (uint32_t index = 0; index < graphs.size(); ++index) {
    CHECK_CUDA(cudaStreamBeginCapture(fixture.context().stream(),
                                      cudaStreamCaptureModeThreadLocal));
    if (index + 1U < graphs.size()) {
      append_stage_digit<<<1, 1, 0, fixture.context().stream()>>>(
          fixture.out_tokens_data() + (kDsparkCycleProductionNumOutTokens - 1U),
          index + 1U);
      CHECK_CUDA(cudaGetLastError());
    } else {
      CHECK_STATUS(launch_dspark_cycle_compact_result(fixture.context(),
                                                      fixture.buffers()));
      CHECK_CUDA(cudaMemcpyAsync(capture_result->data(), fixture.result_data(),
                                 sizeof(DsparkCycleResultV1),
                                 cudaMemcpyDeviceToHost,
                                 fixture.context().stream()));
    }
    CHECK_CUDA(
        cudaStreamEndCapture(fixture.context().stream(), &graphs[index].value));
  }

  children = make_dspark_cycle_child_graphs(DsparkCycleCoreGraphs{
      graphs[0].value, graphs[1].value, graphs[2].value, graphs[3].value,
      graphs[4].value, graphs[5].value, graphs[6].value, graphs[7].value});

  const DsparkCycleDescriptor descriptor{
      kDsparkCycleAbiVersion,
      static_cast<uint32_t>(fixture.context().device_ordinal()), children,
      fixture.device_status_data(), fixture.result_data()};
  CHECK(take_result(DsparkCycleController::create_unsafe_unowned_for_test(
                        descriptor, fixture.context(), fixture.lease()),
                    &controller));
  for (RawGraph &graph : graphs) {
    CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));
  }
  CHECK_STATUS(controller->launch());
  CHECK_STATUS(controller->synchronize());
  const DsparkCycleResultV1 *result = controller->result();
  CHECK(result != nullptr);
  CHECK(result->abi_version == kDsparkCycleAbiVersion);
  CHECK(result->device_status == 0U);
  CHECK(result->accepted_token_count == 8);
  CHECK(result->num_correct_drafts == 7);
  CHECK(result->output_token == 1234567);
  CHECK(result->request_slot == 0);
  CHECK(static_cast<const DsparkCycleResultV1 *>(capture_result->data())
            ->abi_version == 0U);

  CHECK(fixture.write_inputs(tokens, 7, 0xdeadbeefU));
  CHECK_STATUS(controller->launch());
  CHECK_STATUS(controller->synchronize());
  result = controller->result();
  CHECK(result != nullptr);
  CHECK(result->device_status == 0U);
  CHECK(result->output_token == 1234567);

  CHECK_STATUS(controller->close());
  controller.reset();
  CHECK_STATUS(capture_result->close());
  return true;
}

[[nodiscard]] bool ControllerAcceptsOwnedSourceGraphs() {
  CompactResultFixture fixture;
  std::optional<DsparkCycleController> controller;
  std::optional<DsparkModelGraphResourceOwner> model_owner;
  std::array<std::optional<CudaCapturedGraph>,
             sglang::native::kDsparkCycleNumStages - 1U>
      graphs;
  std::optional<DsparkCycleCompactGraph> compact_graph;

  CHECK(fixture.initialize());
  auto retained_fixture_resources =
      std::make_shared<const uint32_t>(0x12345678U);
  CHECK(take_result(
      DsparkModelGraphResourceOwner::create(fixture.context(), fixture.lease(),
                                            retained_fixture_resources),
      &model_owner));
  CHECK(model_owner->valid());
  CHECK(model_owner->active_leases() == 0U);
  const std::array<int32_t, kDsparkCycleProductionNumOutTokens> tokens{
      701, 702, 703, 704, 705, 706, 707, 0};
  CHECK(fixture.write_inputs(tokens, 7, 0xabcddcbaU));
  for (uint32_t index = 0; index < graphs.size(); ++index) {
    StageCapturePayload payload{&fixture.context(),
                                fixture.out_tokens_data() +
                                    (kDsparkCycleProductionNumOutTokens - 1U),
                                index + 1U};
    const bool is_model_graph = index == 1U || index == 5U || index == 6U;
    CHECK(take_result(
        is_model_graph
            ? CudaCapturedGraph::capture_retaining(
                  fixture.context(), fixture.lease(),
                  model_owner->acquire_lease().retain(), &capture_stage_digit,
                  &payload)
            : CudaCapturedGraph::capture(fixture.context(), fixture.lease(),
                                         &capture_stage_digit, &payload),
        &graphs[index]));
  }
  CHECK(take_result(DsparkCycleCompactGraph::capture(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &compact_graph));

  const DsparkCycleGraphSources sources{
      *graphs[0],
      DsparkTargetVerifyGraphSource{model_binding(
          DsparkModelGraphKind::kTargetVerify, *graphs[1], *model_owner)},
      *graphs[2],
      *graphs[3],
      *graphs[4],
      DsparkDraftExtendGraphSource{model_binding(
          DsparkModelGraphKind::kDraftExtend, *graphs[5], *model_owner)},
      DsparkKvWriteGraphSource{model_binding(DsparkModelGraphKind::kKvWrite,
                                             *graphs[6], *model_owner)},
      *compact_graph};
  CHECK(take_result(
      DsparkCycleController::create(sources, fixture.device_status_data(),
                                    fixture.result_data(), fixture.context(),
                                    fixture.lease()),
      &controller));
  CHECK(model_owner->active_leases() == 9U);
  for (auto &graph : graphs) {
    CHECK_STATUS(graph->close());
    graph.reset();
  }
  CHECK_STATUS(compact_graph->close());
  compact_graph.reset();
  CHECK(model_owner->active_leases() == 6U);
  model_owner.reset();
  CHECK(fixture.write_inputs(tokens, 7, 0xdeadbeefU));
  CHECK_STATUS(controller->launch());
  CHECK_STATUS(controller->synchronize());
  const DsparkCycleResultV1 *result = controller->result();
  CHECK(result != nullptr);
  CHECK(result->device_status == 0U);
  CHECK(result->accepted_token_count == 8);
  CHECK(result->num_correct_drafts == 7);
  CHECK(result->output_token == 1234567);
  CHECK_STATUS(controller->close());
  controller.reset();
  return true;
}

[[nodiscard]] bool TypedControllerRejectsInvalidAndForeignSources() {
  CompactResultFixture fixture;
  CompactResultFixture foreign_fixture;
  std::optional<DsparkModelGraphResourceOwner> model_owner;
  std::optional<DsparkModelGraphResourceOwner> foreign_model_owner;
  std::array<std::optional<CudaCapturedGraph>,
             sglang::native::kDsparkCycleNumStages - 1U>
      graphs;
  std::optional<CudaCapturedGraph> foreign_graph;
  std::optional<DsparkCycleCompactGraph> compact_graph;

  CHECK(fixture.initialize());
  CHECK(foreign_fixture.initialize());
  auto retained_fixture_resources =
      std::make_shared<const uint32_t>(0x12345678U);
  auto retained_foreign_resources =
      std::make_shared<const uint32_t>(0x87654321U);
  CHECK(take_result(
      DsparkModelGraphResourceOwner::create(fixture.context(), fixture.lease(),
                                            retained_fixture_resources),
      &model_owner));
  CHECK(take_result(
      DsparkModelGraphResourceOwner::create(fixture.context(), fixture.lease(),
                                            retained_foreign_resources),
      &foreign_model_owner));
  CHECK(model_owner->valid());
  CHECK(foreign_model_owner->valid());
  for (uint32_t index = 0; index < graphs.size(); ++index) {
    StageCapturePayload payload{&fixture.context(),
                                fixture.out_tokens_data() +
                                    (kDsparkCycleProductionNumOutTokens - 1U),
                                index + 1U};
    const bool is_model_graph = index == 1U || index == 5U || index == 6U;
    CHECK(take_result(
        is_model_graph
            ? CudaCapturedGraph::capture_retaining(
                  fixture.context(), fixture.lease(),
                  model_owner->acquire_lease().retain(), &capture_stage_digit,
                  &payload)
            : CudaCapturedGraph::capture(fixture.context(), fixture.lease(),
                                         &capture_stage_digit, &payload),
        &graphs[index]));
  }
  StageCapturePayload foreign_payload{
      &foreign_fixture.context(),
      foreign_fixture.out_tokens_data() +
          (kDsparkCycleProductionNumOutTokens - 1U),
      1U};
  CHECK(take_result(CudaCapturedGraph::capture(
                        foreign_fixture.context(), foreign_fixture.lease(),
                        &capture_stage_digit, &foreign_payload),
                    &foreign_graph));
  CHECK(take_result(DsparkCycleCompactGraph::capture(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &compact_graph));

  const auto make_sources = [&]() noexcept {
    return DsparkCycleGraphSources{
        *graphs[0],
        DsparkTargetVerifyGraphSource{model_binding(
            DsparkModelGraphKind::kTargetVerify, *graphs[1], *model_owner)},
        *graphs[2],
        *graphs[3],
        *graphs[4],
        DsparkDraftExtendGraphSource{model_binding(
            DsparkModelGraphKind::kDraftExtend, *graphs[5], *model_owner)},
        DsparkKvWriteGraphSource{model_binding(DsparkModelGraphKind::kKvWrite,
                                               *graphs[6], *model_owner)},
        *compact_graph};
  };

  CHECK_STATUS(graphs[0]->close());
  bool accepted_invalid_source = false;
  bool invalid_source_close_ok = true;
  NativeRuntimeError invalid_error{};
  std::move(DsparkCycleController::create(
                make_sources(), fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_invalid_source, &invalid_source_close_ok](
              DsparkCycleController &&controller) noexcept {
            accepted_invalid_source = true;
            invalid_source_close_ok = is_ok(controller.close());
          },
          [&invalid_error](NativeRuntimeError &&error) noexcept {
            invalid_error = error;
          });
  CHECK(!accepted_invalid_source);
  CHECK(invalid_source_close_ok);
  CHECK(invalid_error.code == NativeRuntimeCode::kInvalidArgument);

  graphs[0].reset();
  graphs[0].emplace(std::move(*foreign_graph));
  foreign_graph.reset();
  bool accepted_foreign_source = false;
  bool foreign_source_close_ok = true;
  NativeRuntimeError foreign_error{};
  std::move(DsparkCycleController::create(
                make_sources(), fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_foreign_source, &foreign_source_close_ok](
              DsparkCycleController &&controller) noexcept {
            accepted_foreign_source = true;
            foreign_source_close_ok = is_ok(controller.close());
          },
          [&foreign_error](NativeRuntimeError &&error) noexcept {
            foreign_error = error;
          });
  CHECK(!accepted_foreign_source);
  CHECK(foreign_source_close_ok);
  CHECK(foreign_error.code == NativeRuntimeCode::kForeignSlice);

  graphs[0].reset();
  StageCapturePayload replacement_payload{
      &fixture.context(),
      fixture.out_tokens_data() + (kDsparkCycleProductionNumOutTokens - 1U),
      1U};
  CHECK(take_result(
      CudaCapturedGraph::capture(fixture.context(), fixture.lease(),
                                 &capture_stage_digit, &replacement_payload),
      &graphs[0]));
  std::optional<CudaCapturedGraph> unowned_target_graph;
  StageCapturePayload unowned_target_payload{
      &fixture.context(),
      fixture.out_tokens_data() + (kDsparkCycleProductionNumOutTokens - 1U),
      2U};
  CHECK(take_result(
      CudaCapturedGraph::capture(fixture.context(), fixture.lease(),
                                 &capture_stage_digit, &unowned_target_payload),
      &unowned_target_graph));
  const DsparkCycleGraphSources unowned_graph_sources{
      *graphs[0],
      DsparkTargetVerifyGraphSource{
          model_binding(DsparkModelGraphKind::kTargetVerify,
                        *unowned_target_graph, *model_owner)},
      *graphs[2],
      *graphs[3],
      *graphs[4],
      DsparkDraftExtendGraphSource{model_binding(
          DsparkModelGraphKind::kDraftExtend, *graphs[5], *model_owner)},
      DsparkKvWriteGraphSource{model_binding(DsparkModelGraphKind::kKvWrite,
                                             *graphs[6], *model_owner)},
      *compact_graph};
  bool accepted_unowned_graph = false;
  bool unowned_graph_close_ok = true;
  NativeRuntimeError unowned_graph_error{};
  std::move(DsparkCycleController::create(
                unowned_graph_sources, fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_unowned_graph, &unowned_graph_close_ok](
              DsparkCycleController &&controller) noexcept {
            accepted_unowned_graph = true;
            unowned_graph_close_ok = is_ok(controller.close());
          },
          [&unowned_graph_error](NativeRuntimeError &&error) noexcept {
            unowned_graph_error = error;
          });
  CHECK(!accepted_unowned_graph);
  CHECK(unowned_graph_close_ok);
  CHECK(unowned_graph_error.code == NativeRuntimeCode::kForeignSlice);
  CHECK(unowned_graph_error.detail ==
        static_cast<uint32_t>(
            sglang::native::DsparkCycleArgument::kResourceOwner));

  auto wrong_owner_sources = make_sources();
  wrong_owner_sources.kv_write.binding.resources =
      foreign_model_owner->acquire_lease();
  bool accepted_wrong_owner = false;
  bool wrong_owner_close_ok = true;
  NativeRuntimeError wrong_owner_error{};
  std::move(DsparkCycleController::create(
                wrong_owner_sources, fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_wrong_owner,
           &wrong_owner_close_ok](DsparkCycleController &&controller) noexcept {
            accepted_wrong_owner = true;
            wrong_owner_close_ok = is_ok(controller.close());
          },
          [&wrong_owner_error](NativeRuntimeError &&error) noexcept {
            wrong_owner_error = error;
          });
  CHECK(!accepted_wrong_owner);
  CHECK(wrong_owner_close_ok);
  CHECK(wrong_owner_error.code == NativeRuntimeCode::kForeignSlice);
  CHECK(wrong_owner_error.detail ==
        static_cast<uint32_t>(
            sglang::native::DsparkCycleArgument::kResourceOwner));

  auto missing_output_sources = make_sources();
  missing_output_sources.target_verify.binding.published_outputs = 0U;
  bool accepted_missing_output = false;
  bool missing_output_close_ok = true;
  NativeRuntimeError missing_output_error{};
  std::move(DsparkCycleController::create(
                missing_output_sources, fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_missing_output, &missing_output_close_ok](
              DsparkCycleController &&controller) noexcept {
            accepted_missing_output = true;
            missing_output_close_ok = is_ok(controller.close());
          },
          [&missing_output_error](NativeRuntimeError &&error) noexcept {
            missing_output_error = error;
          });
  CHECK(!accepted_missing_output);
  CHECK(missing_output_close_ok);
  CHECK(missing_output_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(missing_output_error.detail ==
        static_cast<uint32_t>(
            sglang::native::DsparkCycleArgument::kRequiredOutputs));

  auto wrong_shape_sources = make_sources();
  wrong_shape_sources.draft_extend.binding.shape.verify_width = 7U;
  bool accepted_wrong_shape = false;
  bool wrong_shape_close_ok = true;
  NativeRuntimeError wrong_shape_error{};
  std::move(DsparkCycleController::create(
                wrong_shape_sources, fixture.device_status_data(),
                fixture.result_data(), fixture.context(), fixture.lease()))
      .match(
          [&accepted_wrong_shape,
           &wrong_shape_close_ok](DsparkCycleController &&controller) noexcept {
            accepted_wrong_shape = true;
            wrong_shape_close_ok = is_ok(controller.close());
          },
          [&wrong_shape_error](NativeRuntimeError &&error) noexcept {
            wrong_shape_error = error;
          });
  CHECK(!accepted_wrong_shape);
  CHECK(wrong_shape_close_ok);
  CHECK(wrong_shape_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(wrong_shape_error.detail ==
        static_cast<uint32_t>(
            sglang::native::DsparkCycleArgument::kProductionShape));
  return true;
}

} // namespace

int main() {
  struct TestCase final {
    const char *name;
    bool (*function)();
  };
  constexpr TestCase kTests[]{
      {"CompactResultPublishesBoundariesAndBonus",
       CompactResultPublishesBoundariesAndBonus},
      {"CompactResultPreservesUpstreamStatus",
       CompactResultPreservesUpstreamStatus},
      {"CompactResultRejectsInvalidCounts", CompactResultRejectsInvalidCounts},
      {"CompactResultCapturedReplayUsesStableAddresses",
       CompactResultCapturedReplayUsesStableAddresses},
      {"CompactResultRejectsAliasing", CompactResultRejectsAliasing},
      {"CompactResultRejectsMalformedLayout",
       CompactResultRejectsMalformedLayout},
      {"ControllerComposesRealCompactResult",
       ControllerComposesRealCompactResult},
      {"ControllerAcceptsOwnedSourceGraphs",
       ControllerAcceptsOwnedSourceGraphs},
      {"TypedControllerRejectsInvalidAndForeignSources",
       TypedControllerRejectsInvalidAndForeignSources},
  };
  for (const TestCase &test : kTests) {
    std::printf("[ RUN      ] %s\n", test.name);
    if (!test.function()) {
      std::printf("[  FAILED  ] %s\n", test.name);
      return 1;
    }
    std::printf("[       OK ] %s\n", test.name);
  }
  std::printf("[  PASSED  ] %zu tests\n", std::size(kTests));
  return 0;
}
