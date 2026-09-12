#include "sglang/native/gdn_replayssm_commit.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using sglang::native::capture_gdn_replayssm_commit_graph;
using sglang::native::CudaCapturedGraph;
using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaStream;
using sglang::native::DType;
using sglang::native::GdnConstBFloat16ConvTensor5;
using sglang::native::GdnConstBFloat16Tensor5;
using sglang::native::GdnConstFloat32Tensor4;
using sglang::native::GdnConstInt32Vector;
using sglang::native::GdnMutableBFloat16Tensor4;
using sglang::native::GdnMutableFloat32Tensor5;
using sglang::native::GdnMutableUInt32Vector;
using sglang::native::GdnReplaySsmArgument;
using sglang::native::GdnReplaySsmCommitBuffers;
using sglang::native::GdnReplaySsmConvPair;
using sglang::native::GdnReplaySsmShape;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::is_ok;
using sglang::native::launch_gdn_replayssm_commit;
using sglang::native::launch_gdn_replayssm_commit_if_ready;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::NativeRuntimeResult;
using sglang::native::validate_gdn_replayssm_commit_buffers;

constexpr uint32_t kLayers = 2;
constexpr uint32_t kSlots = 4;
constexpr uint32_t kBatch = 2;
constexpr uint32_t kValueHeads = 2;
constexpr uint32_t kKeyHeads = 1;
constexpr uint32_t kKeyDimension = 128;
constexpr uint32_t kValueDimension = 128;
constexpr uint32_t kReplay = 8;
constexpr uint32_t kConvDimension = 3;
constexpr uint32_t kConvWindow = 2;
constexpr uint32_t kSentinelStatus = 0xf00dba5eU;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
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
  if (is_ok(status))
    return true;
  std::printf("%s:%d: native failure for %s: code=%u operation=%u "
              "detail=%u native=%d actual=%llu required=%llu\n",
              __FILE__, line, expression, static_cast<unsigned>(status.code),
              static_cast<unsigned>(status.operation), status.detail,
              status.native_code,
              static_cast<unsigned long long>(status.actual),
              static_cast<unsigned long long>(status.required));
  return false;
}

#define CHECK_STATUS(expression)                                               \
  do {                                                                         \
    if (!check_status((expression), #expression, __LINE__))                    \
      return false;                                                            \
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
        return check_status(error, "NativeRuntimeResult", __LINE__);
      });
}

[[nodiscard]] SglNativeTensorMetadataV1
metadata(SglNativeDType dtype, std::span<const int64_t> extents) noexcept {
  auto result = make_tensor_metadata_v1();
  result.dtype = dtype;
  result.rank = static_cast<uint32_t>(extents.size());
  int64_t stride = 1;
  for (std::size_t reverse = extents.size(); reverse > 0; --reverse) {
    const std::size_t dimension = reverse - 1;
    result.extents[dimension] = extents[dimension];
    result.strides[dimension] = stride;
    stride *= extents[dimension];
  }
  return result;
}

template <std::size_t Rank>
[[nodiscard]] SglNativeTensorMetadataV1
metadata_with_strides(SglNativeDType dtype,
                      const std::array<int64_t, Rank> &extents,
                      const std::array<int64_t, Rank> &strides,
                      uint64_t storage_offset_elements = 0) noexcept {
  auto result = make_tensor_metadata_v1();
  result.dtype = dtype;
  result.rank = Rank;
  result.storage_offset_elements = storage_offset_elements;
  for (std::size_t dimension = 0; dimension < Rank; ++dimension) {
    result.extents[dimension] = extents[dimension];
    result.strides[dimension] = strides[dimension];
  }
  return result;
}

template <std::size_t Rank>
[[nodiscard]] SglNativeTensorMetadataV1
metadata(SglNativeDType dtype,
         const std::array<int64_t, Rank> &extents) noexcept {
  return metadata(dtype, std::span<const int64_t>(extents));
}

[[nodiscard]] uint64_t product(std::span<const int64_t> extents) noexcept {
  uint64_t result = 1;
  for (const int64_t extent : extents)
    result *= static_cast<uint64_t>(extent);
  return result;
}

template <std::size_t Rank>
[[nodiscard]] uint64_t
product(const std::array<int64_t, Rank> &extents) noexcept {
  return product(std::span<const int64_t>(extents));
}

struct RawGraph final {
  cudaGraph_t value = nullptr;
  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

class Fixture final {
public:
  Fixture(const Fixture &) = delete;
  Fixture &operator=(const Fixture &) = delete;
  Fixture(Fixture &&) = delete;
  Fixture &operator=(Fixture &&) = delete;
  Fixture() = default;

  [[nodiscard]] bool initialize() noexcept {
    constexpr uint64_t temporal_elements = static_cast<uint64_t>(kLayers) *
                                           kSlots * kValueHeads *
                                           kValueDimension * kKeyDimension;
    constexpr uint64_t raw_value_elements = static_cast<uint64_t>(kLayers) *
                                            kSlots * kValueHeads * kReplay *
                                            kValueDimension;
    constexpr uint64_t raw_key_elements = static_cast<uint64_t>(kLayers) *
                                          kSlots * kKeyHeads * kReplay *
                                          kKeyDimension;
    constexpr uint64_t gate_elements =
        static_cast<uint64_t>(kLayers) * kSlots * kValueHeads * kReplay;
    constexpr uint64_t conv_elements =
        static_cast<uint64_t>(kLayers) * kSlots * kConvDimension * kConvWindow;
    constexpr uint64_t window_elements = static_cast<uint64_t>(kLayers) *
                                         kBatch * kReplay * kConvDimension *
                                         kConvWindow;
    constexpr uint64_t capacity = temporal_elements * sizeof(float) +
                                  (raw_value_elements + raw_key_elements +
                                   conv_elements + window_elements) *
                                      sizeof(uint16_t) +
                                  gate_elements * sizeof(float) * 2 +
                                  kBatch * sizeof(int32_t) * 5 +
                                  sizeof(uint32_t) + 16ULL * 256ULL;
    constexpr uint64_t test_only_capacity =
        7ULL * kSlots * kValueHeads * kValueDimension * kKeyDimension *
            sizeof(float) +
        (kBatch + 1ULL) * sizeof(int32_t) +
        4ULL * kSlots * kConvDimension * kConvWindow * sizeof(uint16_t) +
        2ULL * window_elements * sizeof(uint16_t) + 4ULL * 256ULL;

    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(GraphMemoryArena::allocate(*context_,
                                                capacity + test_only_capacity),
                     &arena_)) {
      return false;
    }
    if (!reserve(temporal_elements * sizeof(float), &temporal_slice_) ||
        !reserve(raw_value_elements * sizeof(uint16_t), &raw_value_slice_) ||
        !reserve(raw_key_elements * sizeof(uint16_t), &raw_key_slice_) ||
        !reserve(gate_elements * sizeof(float), &log_decay_slice_) ||
        !reserve(gate_elements * sizeof(float), &beta_slice_) ||
        !reserve(kBatch * sizeof(int32_t), &state_indices_slice_) ||
        !reserve(kBatch * sizeof(int32_t), &accept_lengths_slice_) ||
        !reserve(kBatch * sizeof(int32_t), &last_correct_steps_slice_) ||
        !reserve(kBatch * sizeof(int32_t), &track_indices_slice_) ||
        !reserve(kBatch * sizeof(int32_t), &track_steps_slice_) ||
        !reserve(conv_elements * sizeof(uint16_t), &conv_states_slice_) ||
        !reserve(window_elements * sizeof(uint16_t), &conv_windows_slice_) ||
        !reserve(sizeof(uint32_t), &device_status_slice_) ||
        !reserve(7ULL * kSlots * kValueHeads * kValueDimension * kKeyDimension *
                     sizeof(float),
                 &offset_temporal_slice_) ||
        !reserve((kBatch + 1ULL) * sizeof(int32_t), &offset_indices_slice_) ||
        !reserve(4ULL * kSlots * kConvDimension * kConvWindow *
                     sizeof(uint16_t),
                 &alias_conv_slice_) ||
        !reserve(window_elements * sizeof(uint16_t), &alias_windows0_slice_) ||
        !reserve(window_elements * sizeof(uint16_t), &alias_windows1_slice_)) {
      return false;
    }
    if (!check_status(arena_->seal(), "arena seal", __LINE__) ||
        !take_result(arena_->acquire_lease(), &lease_)) {
      return false;
    }

    const std::array<int64_t, 5> temporal_shape{kLayers, kSlots, kValueHeads,
                                                kValueDimension, kKeyDimension};
    const std::array<int64_t, 5> raw_value_shape{kLayers, kSlots, kValueHeads,
                                                 kReplay, kValueDimension};
    const std::array<int64_t, 5> raw_key_shape{kLayers, kSlots, kKeyHeads,
                                               kReplay, kKeyDimension};
    const std::array<int64_t, 4> gate_shape{kLayers, kSlots, kValueHeads,
                                            kReplay};
    const std::array<int64_t, 1> batch_shape{kBatch};
    const std::array<int64_t, 4> conv_shape{kLayers, kSlots, kConvDimension,
                                            kConvWindow};
    const std::array<int64_t, 5> window_shape{kLayers, kBatch, kReplay,
                                              kConvDimension, kConvWindow};
    const std::array<int64_t, 1> status_shape{1};

    return take_result(lease_->bind_mutable<DType::kFloat32, 5>(
                           *temporal_slice_,
                           metadata(SGL_NATIVE_DTYPE_FLOAT32, temporal_shape)),
                       &temporal_) &&
           take_result(
               lease_->bind_const<DType::kBFloat16, 5>(
                   *raw_value_slice_,
                   metadata(SGL_NATIVE_DTYPE_BFLOAT16, raw_value_shape)),
               &raw_values_) &&
           take_result(lease_->bind_const<DType::kBFloat16, 5>(
                           *raw_key_slice_,
                           metadata(SGL_NATIVE_DTYPE_BFLOAT16, raw_key_shape)),
                       &raw_keys_) &&
           take_result(lease_->bind_const<DType::kFloat32, 4>(
                           *log_decay_slice_,
                           metadata(SGL_NATIVE_DTYPE_FLOAT32, gate_shape)),
                       &log_decay_) &&
           take_result(lease_->bind_const<DType::kFloat32, 4>(
                           *beta_slice_,
                           metadata(SGL_NATIVE_DTYPE_FLOAT32, gate_shape)),
                       &beta_) &&
           bind_vector(*state_indices_slice_, &state_indices_, batch_shape) &&
           bind_vector(*accept_lengths_slice_, &accept_lengths_, batch_shape) &&
           bind_vector(*last_correct_steps_slice_, &last_correct_steps_,
                       batch_shape) &&
           bind_vector(*track_indices_slice_, &track_indices_, batch_shape) &&
           bind_vector(*track_steps_slice_, &track_steps_, batch_shape) &&
           take_result(lease_->bind_mutable<DType::kBFloat16, 4>(
                           *conv_states_slice_,
                           metadata(SGL_NATIVE_DTYPE_BFLOAT16, conv_shape)),
                       &conv_states_) &&
           take_result(lease_->bind_const<DType::kBFloat16, 5>(
                           *conv_windows_slice_,
                           metadata(SGL_NATIVE_DTYPE_BFLOAT16, window_shape)),
                       &conv_windows_) &&
           take_result(lease_->bind_mutable<DType::kUInt32, 1>(
                           *device_status_slice_,
                           metadata(SGL_NATIVE_DTYPE_UINT32, status_shape)),
                       &device_status_);
  }

  [[nodiscard]] const CudaExecutionContext &context() const noexcept {
    return *context_;
  }
  [[nodiscard]] const GraphArenaLease &lease() const noexcept {
    return *lease_;
  }
  [[nodiscard]] const GraphMemorySlice &offset_temporal_slice() const noexcept {
    return *offset_temporal_slice_;
  }
  [[nodiscard]] const GraphMemorySlice &offset_indices_slice() const noexcept {
    return *offset_indices_slice_;
  }
  [[nodiscard]] const GraphMemorySlice &alias_conv_slice() const noexcept {
    return *alias_conv_slice_;
  }
  [[nodiscard]] const GraphMemorySlice &alias_windows0_slice() const noexcept {
    return *alias_windows0_slice_;
  }
  [[nodiscard]] const GraphMemorySlice &alias_windows1_slice() const noexcept {
    return *alias_windows1_slice_;
  }
  [[nodiscard]] GdnReplaySsmCommitBuffers buffers() const noexcept {
    conv_pair_.emplace(GdnReplaySsmConvPair{*conv_states_, *conv_windows_});
    return GdnReplaySsmCommitBuffers{
        *temporal_,
        *raw_values_,
        *raw_keys_,
        *log_decay_,
        *beta_,
        *state_indices_,
        *accept_lengths_,
        *last_correct_steps_,
        *track_indices_,
        *track_steps_,
        std::span<const GdnReplaySsmConvPair>(&*conv_pair_, 1),
        *device_status_};
  }

  template <typename T>
  [[nodiscard]] bool copy_temporal(const std::vector<T> &values) noexcept {
    return copy_to_mutable(*temporal_, values);
  }
  [[nodiscard]] bool
  copy_raw_values(const std::vector<uint16_t> &values) noexcept {
    return copy_to_const(*raw_values_, values);
  }
  [[nodiscard]] bool
  copy_raw_keys(const std::vector<uint16_t> &values) noexcept {
    return copy_to_const(*raw_keys_, values);
  }
  [[nodiscard]] bool copy_log_decay(const std::vector<float> &values) noexcept {
    return copy_to_const(*log_decay_, values);
  }
  [[nodiscard]] bool copy_beta(const std::vector<float> &values) noexcept {
    return copy_to_const(*beta_, values);
  }
  [[nodiscard]] bool
  copy_indices(const std::array<int32_t, kBatch> &states,
               const std::array<int32_t, kBatch> &accepts,
               const std::array<int32_t, kBatch> &lasts,
               const std::array<int32_t, kBatch> &tracks,
               const std::array<int32_t, kBatch> &steps) noexcept {
    return copy_to_const(*state_indices_, states) &&
           copy_to_const(*accept_lengths_, accepts) &&
           copy_to_const(*last_correct_steps_, lasts) &&
           copy_to_const(*track_indices_, tracks) &&
           copy_to_const(*track_steps_, steps);
  }
  [[nodiscard]] bool
  copy_conv_states(const std::vector<uint16_t> &values) noexcept {
    return copy_to_mutable(*conv_states_, values);
  }
  [[nodiscard]] bool
  copy_conv_windows(const std::vector<uint16_t> &values) noexcept {
    return copy_to_const(*conv_windows_, values);
  }
  [[nodiscard]] bool set_status(uint32_t status) noexcept {
    return copy_to_mutable(*device_status_,
                           std::span<const uint32_t>(&status, 1));
  }
  [[nodiscard]] bool synchronize() const noexcept {
    return check_status(context_->synchronize(), "context synchronize",
                        __LINE__);
  }

  [[nodiscard]] std::vector<float> temporal() const {
    return copy_from(*temporal_);
  }
  [[nodiscard]] std::vector<uint16_t> conv_states() const {
    return copy_from(*conv_states_);
  }
  [[nodiscard]] uint32_t status() const {
    return copy_from(*device_status_)[0];
  }

private:
  template <typename OptionalView, std::size_t Rank>
  [[nodiscard]] bool
  bind_vector(const GraphMemorySlice &slice, OptionalView *output,
              const std::array<int64_t, Rank> &shape) noexcept {
    return take_result(lease_->bind_const<DType::kInt32, 1>(
                           slice, metadata(SGL_NATIVE_DTYPE_INT32, shape)),
                       output);
  }

  [[nodiscard]] bool reserve(uint64_t bytes,
                             std::optional<GraphMemorySlice> *output) noexcept {
    return take_result(arena_->reserve(bytes, 256), output);
  }

  template <typename View, typename T>
  [[nodiscard]] bool copy_to_const(const View &view,
                                   std::span<const T> values) noexcept {
    if (values.size_bytes() != view.allocation_bytes())
      return false;
    return cudaMemcpyAsync(const_cast<std::byte *>(view.data_bytes()),
                           values.data(), values.size_bytes(),
                           cudaMemcpyHostToDevice,
                           context_->stream()) == cudaSuccess;
  }

  template <typename View, typename T>
  [[nodiscard]] bool copy_to_mutable(const View &view,
                                     std::span<const T> values) noexcept {
    if (values.size_bytes() != view.allocation_bytes())
      return false;
    return cudaMemcpyAsync(view.data_bytes(), values.data(),
                           values.size_bytes(), cudaMemcpyHostToDevice,
                           context_->stream()) == cudaSuccess;
  }

  template <typename View, typename Container>
  [[nodiscard]] bool copy_to_const(const View &view,
                                   const Container &values) noexcept {
    using Value = typename Container::value_type;
    return copy_to_const(view, std::span<const Value>(values));
  }

  template <typename View, typename Container>
  [[nodiscard]] bool copy_to_mutable(const View &view,
                                     const Container &values) noexcept {
    using Value = typename Container::value_type;
    return copy_to_mutable(view, std::span<const Value>(values));
  }

  template <typename View>
  [[nodiscard]] auto copy_from(const View &view) const {
    using Value = std::conditional_t<
        std::is_same_v<View, GdnMutableFloat32Tensor5>, float,
        std::conditional_t<std::is_same_v<View, GdnMutableUInt32Vector>,
                           uint32_t, uint16_t>>;
    std::vector<Value> result(view.allocation_bytes() / sizeof(Value));
    if (cudaMemcpy(result.data(), view.data_bytes(), view.allocation_bytes(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      std::abort();
    }
    return result;
  }

  std::optional<CudaStream> stream_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphMemoryArena> arena_;
  std::optional<GraphMemorySlice> temporal_slice_;
  std::optional<GraphMemorySlice> raw_value_slice_;
  std::optional<GraphMemorySlice> raw_key_slice_;
  std::optional<GraphMemorySlice> log_decay_slice_;
  std::optional<GraphMemorySlice> beta_slice_;
  std::optional<GraphMemorySlice> state_indices_slice_;
  std::optional<GraphMemorySlice> accept_lengths_slice_;
  std::optional<GraphMemorySlice> last_correct_steps_slice_;
  std::optional<GraphMemorySlice> track_indices_slice_;
  std::optional<GraphMemorySlice> track_steps_slice_;
  std::optional<GraphMemorySlice> conv_states_slice_;
  std::optional<GraphMemorySlice> conv_windows_slice_;
  std::optional<GraphMemorySlice> device_status_slice_;
  std::optional<GraphMemorySlice> offset_temporal_slice_;
  std::optional<GraphMemorySlice> offset_indices_slice_;
  std::optional<GraphMemorySlice> alias_conv_slice_;
  std::optional<GraphMemorySlice> alias_windows0_slice_;
  std::optional<GraphMemorySlice> alias_windows1_slice_;
  std::optional<GraphArenaLease> lease_;
  std::optional<GdnMutableFloat32Tensor5> temporal_;
  std::optional<GdnConstBFloat16Tensor5> raw_values_;
  std::optional<GdnConstBFloat16Tensor5> raw_keys_;
  std::optional<GdnConstFloat32Tensor4> log_decay_;
  std::optional<GdnConstFloat32Tensor4> beta_;
  std::optional<GdnConstInt32Vector> state_indices_;
  std::optional<GdnConstInt32Vector> accept_lengths_;
  std::optional<GdnConstInt32Vector> last_correct_steps_;
  std::optional<GdnConstInt32Vector> track_indices_;
  std::optional<GdnConstInt32Vector> track_steps_;
  std::optional<GdnMutableBFloat16Tensor4> conv_states_;
  std::optional<GdnConstBFloat16ConvTensor5> conv_windows_;
  std::optional<GdnMutableUInt32Vector> device_status_;
  mutable std::optional<GdnReplaySsmConvPair> conv_pair_;
};

[[nodiscard]] uint16_t bfloat16_bits(float value) noexcept {
  return __bfloat16_as_ushort(__float2bfloat16(value));
}

[[nodiscard]] float from_bfloat16(uint16_t bits) noexcept {
  return __bfloat162float(__ushort_as_bfloat16(bits));
}

[[nodiscard]] std::size_t temporal_offset(uint32_t layer, uint32_t slot,
                                          uint32_t head, uint32_t value,
                                          uint32_t key) noexcept {
  return ((((static_cast<std::size_t>(layer) * kSlots + slot) * kValueHeads +
            head) *
               kValueDimension +
           value) *
              kKeyDimension +
          key);
}

[[nodiscard]] std::size_t raw_value_offset(uint32_t layer, uint32_t slot,
                                           uint32_t head, uint32_t step,
                                           uint32_t value) noexcept {
  return ((((static_cast<std::size_t>(layer) * kSlots + slot) * kValueHeads +
            head) *
               kReplay +
           step) *
              kValueDimension +
          value);
}

[[nodiscard]] std::size_t raw_key_offset(uint32_t layer, uint32_t slot,
                                         uint32_t step, uint32_t key) noexcept {
  return ((((static_cast<std::size_t>(layer) * kSlots + slot) * kKeyHeads) *
               kReplay +
           step) *
              kKeyDimension +
          key);
}

[[nodiscard]] std::size_t gate_offset(uint32_t layer, uint32_t slot,
                                      uint32_t head, uint32_t step) noexcept {
  return (
      ((static_cast<std::size_t>(layer) * kSlots + slot) * kValueHeads + head) *
          kReplay +
      step);
}

[[nodiscard]] std::size_t conv_offset(uint32_t layer, uint32_t slot,
                                      uint32_t dimension,
                                      uint32_t window) noexcept {
  return (((static_cast<std::size_t>(layer) * kSlots + slot) * kConvDimension +
           dimension) *
              kConvWindow +
          window);
}

[[nodiscard]] std::size_t window_offset(uint32_t layer, uint32_t request,
                                        uint32_t step, uint32_t dimension,
                                        uint32_t window) noexcept {
  return (
      (((static_cast<std::size_t>(layer) * kBatch + request) * kReplay + step) *
           kConvDimension +
       dimension) *
          kConvWindow +
      window);
}

struct Inputs final {
  std::vector<float> temporal;
  std::vector<uint16_t> raw_values;
  std::vector<uint16_t> raw_keys;
  std::vector<float> log_decay;
  std::vector<float> beta;
  std::vector<uint16_t> conv_states;
  std::vector<uint16_t> conv_windows;
};

[[nodiscard]] Inputs make_inputs() {
  Inputs inputs{
      std::vector<float>(static_cast<std::size_t>(kLayers) * kSlots *
                         kValueHeads * kValueDimension * kKeyDimension),
      std::vector<uint16_t>(static_cast<std::size_t>(kLayers) * kSlots *
                            kValueHeads * kReplay * kValueDimension),
      std::vector<uint16_t>(static_cast<std::size_t>(kLayers) * kSlots *
                            kKeyHeads * kReplay * kKeyDimension),
      std::vector<float>(static_cast<std::size_t>(kLayers) * kSlots *
                         kValueHeads * kReplay),
      std::vector<float>(static_cast<std::size_t>(kLayers) * kSlots *
                         kValueHeads * kReplay),
      std::vector<uint16_t>(static_cast<std::size_t>(kLayers) * kSlots *
                            kConvDimension * kConvWindow),
      std::vector<uint16_t>(static_cast<std::size_t>(kLayers) * kBatch *
                            kReplay * kConvDimension * kConvWindow)};
  for (std::size_t index = 0; index < inputs.temporal.size(); ++index) {
    inputs.temporal[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) * 0.015625F;
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (uint32_t slot = 0; slot < kSlots; ++slot) {
      for (uint32_t step = 0; step < kReplay; ++step) {
        for (uint32_t key = 0; key < kKeyDimension; ++key) {
          const float raw_key =
              0.03125F *
              static_cast<float>(
                  static_cast<int>((key + 3 * step + layer) % 11) - 5);
          inputs.raw_keys[raw_key_offset(layer, slot, step, key)] =
              bfloat16_bits(raw_key);
        }
        for (uint32_t head = 0; head < kValueHeads; ++head) {
          inputs.log_decay[gate_offset(layer, slot, head, step)] = 0.0F;
          inputs.beta[gate_offset(layer, slot, head, step)] =
              0.25F + 0.125F * static_cast<float>((head + step) % 3);
          for (uint32_t value = 0; value < kValueDimension; ++value) {
            const float raw =
                0.03125F * static_cast<float>(1 + layer + head + step) +
                0.00390625F * static_cast<float>(value % 5);
            inputs
                .raw_values[raw_value_offset(layer, slot, head, step, value)] =
                bfloat16_bits(raw);
          }
        }
      }
    }
  }
  for (std::size_t index = 0; index < inputs.conv_states.size(); ++index) {
    inputs.conv_states[index] = bfloat16_bits(-2.0F);
  }
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (uint32_t request = 0; request < kBatch; ++request) {
      for (uint32_t step = 0; step < kReplay; ++step) {
        for (uint32_t dimension = 0; dimension < kConvDimension; ++dimension) {
          for (uint32_t window = 0; window < kConvWindow; ++window) {
            const float value =
                static_cast<float>(1000 * layer + 100 * request + 10 * step +
                                   2 * dimension + window);
            inputs.conv_windows[window_offset(layer, request, step, dimension,
                                              window)] = bfloat16_bits(value);
          }
        }
      }
    }
  }
  return inputs;
}

void fold_reference(std::vector<float> *temporal, const Inputs &inputs,
                    uint32_t state_slot, uint32_t accept_length,
                    int32_t track_slot, int32_t track_step) {
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (uint32_t head = 0; head < kValueHeads; ++head) {
      std::array<float, kValueDimension * kKeyDimension> state{};
      for (uint32_t value = 0; value < kValueDimension; ++value) {
        for (uint32_t key = 0; key < kKeyDimension; ++key) {
          state[value * kKeyDimension + key] =
              (*temporal)[temporal_offset(layer, state_slot, head, value, key)];
        }
      }
      for (uint32_t step = 0; step < accept_length; ++step) {
        std::array<float, kKeyDimension> normalized_key{};
        float squared_norm = 0.0F;
        for (uint32_t key = 0; key < kKeyDimension; ++key) {
          const float raw_key = from_bfloat16(
              inputs.raw_keys[raw_key_offset(layer, state_slot, step, key)]);
          normalized_key[key] = raw_key;
          squared_norm += raw_key * raw_key;
        }
        const float norm = squared_norm == 1.0F
                               ? 1.000000476837158203125F
                               : std::sqrt(squared_norm + 1.0e-6F);
        for (float &key : normalized_key)
          key /= norm;
        const float beta =
            inputs.beta[gate_offset(layer, state_slot, head, step)];
        for (uint32_t value = 0; value < kValueDimension; ++value) {
          float dot = 0.0F;
          for (uint32_t key = 0; key < kKeyDimension; ++key) {
            dot += state[value * kKeyDimension + key] * normalized_key[key];
          }
          const float raw = from_bfloat16(inputs.raw_values[raw_value_offset(
              layer, state_slot, head, step, value)]);
          const float corrected = (raw - dot) * beta;
          for (uint32_t key = 0; key < kKeyDimension; ++key) {
            state[value * kKeyDimension + key] +=
                normalized_key[key] * corrected;
          }
        }
        if (track_slot >= 0 && static_cast<int32_t>(step) == track_step) {
          for (uint32_t value = 0; value < kValueDimension; ++value) {
            for (uint32_t key = 0; key < kKeyDimension; ++key) {
              (*temporal)[temporal_offset(
                  layer, static_cast<uint32_t>(track_slot), head, value, key)] =
                  state[value * kKeyDimension + key];
            }
          }
        }
      }
      for (uint32_t value = 0; value < kValueDimension; ++value) {
        for (uint32_t key = 0; key < kKeyDimension; ++key) {
          (*temporal)[temporal_offset(layer, state_slot, head, value, key)] =
              state[value * kKeyDimension + key];
        }
      }
    }
  }
}

void scatter_conv_reference(std::vector<uint16_t> *conv, const Inputs &inputs,
                            uint32_t request, int32_t destination,
                            int32_t step) {
  if (destination < 0 || step < 0)
    return;
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (uint32_t dimension = 0; dimension < kConvDimension; ++dimension) {
      for (uint32_t window = 0; window < kConvWindow; ++window) {
        (*conv)[conv_offset(layer, static_cast<uint32_t>(destination),
                            dimension, window)] =
            inputs.conv_windows[window_offset(layer, request,
                                              static_cast<uint32_t>(step),
                                              dimension, window)];
      }
    }
  }
}

[[nodiscard]] bool upload(Fixture *fixture, const Inputs &inputs,
                          const std::array<int32_t, kBatch> &states,
                          const std::array<int32_t, kBatch> &accepts,
                          const std::array<int32_t, kBatch> &lasts,
                          const std::array<int32_t, kBatch> &tracks,
                          const std::array<int32_t, kBatch> &steps,
                          uint32_t status) noexcept {
  return fixture->copy_temporal(inputs.temporal) &&
         fixture->copy_raw_values(inputs.raw_values) &&
         fixture->copy_raw_keys(inputs.raw_keys) &&
         fixture->copy_log_decay(inputs.log_decay) &&
         fixture->copy_beta(inputs.beta) &&
         fixture->copy_indices(states, accepts, lasts, tracks, steps) &&
         fixture->copy_conv_states(inputs.conv_states) &&
         fixture->copy_conv_windows(inputs.conv_windows) &&
         fixture->set_status(status) && fixture->synchronize();
}

[[nodiscard]] bool close_enough(float actual, float expected) noexcept {
  return std::abs(actual - expected) <=
         2.0F * std::numeric_limits<float>::epsilon();
}

[[nodiscard]] bool matches_selected_slots(
    const std::vector<float> &actual, const std::vector<float> &expected,
    std::span<const uint32_t> selected_slots, const char *label) noexcept {
  for (uint32_t layer = 0; layer < kLayers; ++layer) {
    for (const uint32_t slot : selected_slots) {
      for (uint32_t head = 0; head < kValueHeads; ++head) {
        for (uint32_t value = 0; value < kValueDimension; ++value) {
          for (uint32_t key = 0; key < kKeyDimension; ++key) {
            const std::size_t index =
                temporal_offset(layer, slot, head, value, key);
            if (!close_enough(actual[index], expected[index])) {
              std::printf("%s mismatch L=%u S=%u H=%u V=%u K=%u "
                          "index=%llu actual=%.9g expected=%.9g delta=%.9g\n",
                          label, layer, slot, head, value, key,
                          static_cast<unsigned long long>(index),
                          static_cast<double>(actual[index]),
                          static_cast<double>(expected[index]),
                          static_cast<double>(actual[index] - expected[index]));
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool AcceptedPrefixTrackAndConvolutionMatch() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 3};
  constexpr std::array<int32_t, kBatch> accepts{3, 1};
  constexpr std::array<int32_t, kBatch> lasts{2, 0};
  constexpr std::array<int32_t, kBatch> tracks{2, -1};
  constexpr std::array<int32_t, kBatch> steps{1, -1};
  CHECK(upload(&fixture, inputs, states, accepts, lasts, tracks, steps,
               kSentinelStatus));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0U);

  auto expected_temporal = inputs.temporal;
  fold_reference(&expected_temporal, inputs, 1, 3, 2, 1);
  fold_reference(&expected_temporal, inputs, 3, 1, -1, -1);
  const auto actual_temporal = fixture.temporal();
  CHECK(actual_temporal.size() == expected_temporal.size());
  constexpr std::array<uint32_t, 3> touched_slots{1, 2, 3};
  CHECK(matches_selected_slots(actual_temporal, expected_temporal,
                               touched_slots, "temporal"));

  auto expected_conv = inputs.conv_states;
  scatter_conv_reference(&expected_conv, inputs, 0, 1, 2);
  scatter_conv_reference(&expected_conv, inputs, 0, 2, 1);
  scatter_conv_reference(&expected_conv, inputs, 1, 3, 0);
  CHECK(fixture.conv_states() == expected_conv);
  return true;
}

[[nodiscard]] bool AllAcceptedLengthsWork() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, -1};
  constexpr std::array<int32_t, kBatch> absent{-1, -1};
  for (uint32_t length = 0; length <= kReplay; ++length) {
    const std::array<int32_t, kBatch> accepts{static_cast<int32_t>(length), 0};
    const std::array<int32_t, kBatch> lasts{
        length == 0 ? -1 : static_cast<int32_t>(length - 1), -1};
    CHECK(upload(&fixture, inputs, states, accepts, lasts, absent, absent,
                 kSentinelStatus));
    CHECK_STATUS(
        launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
    CHECK(fixture.synchronize());

    auto expected_temporal = inputs.temporal;
    fold_reference(&expected_temporal, inputs, 1, length, -1, -1);
    const auto actual_temporal = fixture.temporal();
    constexpr std::array<uint32_t, 2> checked_slots{1, 3};
    CHECK(matches_selected_slots(actual_temporal, expected_temporal,
                                 checked_slots, "accept-length"));
    auto expected_conv = inputs.conv_states;
    scatter_conv_reference(&expected_conv, inputs, 0, 1, lasts[0]);
    CHECK(fixture.conv_states() == expected_conv);
  }
  return true;
}

[[nodiscard]] bool MalformedContentPreservesOutputs() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 99};
  constexpr std::array<int32_t, kBatch> accepts{3, 1};
  constexpr std::array<int32_t, kBatch> lasts{2, 0};
  constexpr std::array<int32_t, kBatch> absent{-1, -1};
  CHECK(upload(&fixture, inputs, states, accepts, lasts, absent, absent,
               kSentinelStatus));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0x00030001U);
  CHECK(fixture.temporal() == inputs.temporal);
  CHECK(fixture.conv_states() == inputs.conv_states);
  return true;
}

[[nodiscard]] bool ReadyStatusCompositionAndGraphReplayWork() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 3};
  constexpr std::array<int32_t, kBatch> accepts{1, 1};
  constexpr std::array<int32_t, kBatch> lasts{0, 0};
  constexpr std::array<int32_t, kBatch> absent{-1, -1};
  constexpr uint32_t upstream = 0x12345678U;
  CHECK(upload(&fixture, inputs, states, accepts, lasts, absent, absent,
               upstream));
  CHECK_STATUS(launch_gdn_replayssm_commit_if_ready(fixture.context(),
                                                    fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == upstream);
  CHECK(fixture.temporal() == inputs.temporal);
  CHECK(fixture.conv_states() == inputs.conv_states);

  CHECK(fixture.set_status(0U));
  CHECK(fixture.synchronize());
  RawGraph graph;
  CHECK_CUDA(cudaStreamBeginCapture(fixture.context().stream(),
                                    cudaStreamCaptureModeThreadLocal));
  CHECK_STATUS(launch_gdn_replayssm_commit_if_ready(fixture.context(),
                                                    fixture.buffers()));
  CHECK_CUDA(cudaStreamEndCapture(fixture.context().stream(), &graph.value));
  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(CudaGraphExecutable::instantiate(
                        graph.value, fixture.context(), fixture.lease()),
                    &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));

  CHECK(upload(&fixture, inputs, states, accepts, lasts, absent, absent, 0U));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  const auto first = fixture.temporal();
  CHECK(first != inputs.temporal);

  CHECK(upload(&fixture, inputs, states, accepts, lasts, absent, absent, 0U));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  CHECK(fixture.temporal() == first);
  CHECK_STATUS(executable->close());
  return true;
}

[[nodiscard]] bool CapturedReplaySsmFactoryOwnsConvolutionGraph() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 3};
  constexpr std::array<int32_t, kBatch> accepts{3, 1};
  constexpr std::array<int32_t, kBatch> lasts{2, 0};
  constexpr std::array<int32_t, kBatch> tracks{2, -1};
  constexpr std::array<int32_t, kBatch> steps{1, -1};
  CHECK(upload(&fixture, inputs, states, accepts, lasts, tracks, steps, 0U));
  CHECK(fixture.synchronize());

  std::optional<CudaCapturedGraph> graph;
  CHECK(take_result(capture_gdn_replayssm_commit_graph(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &graph));
  CHECK(graph->valid());
  size_t node_count = 0;
  CHECK_CUDA(cudaGraphGetNodes(graph->graph(), nullptr, &node_count));
  CHECK(node_count == 3U);
  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(CudaGraphExecutable::instantiate(
                        graph->graph(), fixture.context(), fixture.lease()),
                    &executable));
  CHECK_STATUS(graph->close());
  graph.reset();

  CHECK(upload(&fixture, inputs, states, accepts, lasts, tracks, steps, 0U));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  CHECK(fixture.status() == 0U);
  auto expected_temporal = inputs.temporal;
  fold_reference(&expected_temporal, inputs, 1, 3, 2, 1);
  fold_reference(&expected_temporal, inputs, 3, 1, -1, -1);
  constexpr std::array<uint32_t, 3> touched_slots{1, 2, 3};
  CHECK(matches_selected_slots(fixture.temporal(), expected_temporal,
                               touched_slots, "captured-temporal"));
  auto expected_conv = inputs.conv_states;
  scatter_conv_reference(&expected_conv, inputs, 0, 1, 2);
  scatter_conv_reference(&expected_conv, inputs, 0, 2, 1);
  scatter_conv_reference(&expected_conv, inputs, 1, 3, 0);
  CHECK(fixture.conv_states() == expected_conv);
  CHECK_STATUS(executable->close());
  executable.reset();
  return true;
}

[[nodiscard]] bool ReplaySsmCaptureRejectsForeignConvolutionArena() {
  Fixture fixture;
  CHECK(fixture.initialize());
  std::optional<GraphMemoryArena> other_arena;
  std::optional<GraphMemorySlice> other_slice;
  std::optional<GraphArenaLease> other_lease;
  std::optional<GdnMutableBFloat16Tensor4> other_conv;
  const uint64_t bytes = static_cast<uint64_t>(kLayers) * kSlots *
                         kConvDimension * kConvWindow * sizeof(uint16_t);
  CHECK(take_result(GraphMemoryArena::allocate(fixture.context(), bytes),
                    &other_arena));
  CHECK(take_result(other_arena->reserve(bytes, 256), &other_slice));
  CHECK_STATUS(other_arena->seal());
  CHECK(take_result(other_arena->acquire_lease(), &other_lease));
  const std::array<int64_t, 4> shape{kLayers, kSlots, kConvDimension,
                                     kConvWindow};
  CHECK(
      take_result(other_lease->bind_mutable<DType::kBFloat16, 4>(
                      *other_slice, metadata(SGL_NATIVE_DTYPE_BFLOAT16, shape)),
                  &other_conv));

  const auto base = fixture.buffers();
  const std::array<GdnReplaySsmConvPair, 1> pairs{GdnReplaySsmConvPair{
      *other_conv, base.conv_pairs[0].intermediate_conv_windows}};
  const GdnReplaySsmCommitBuffers mixed{base.temporal,
                                        base.raw_values,
                                        base.raw_keys,
                                        base.log_decay,
                                        base.beta,
                                        base.state_indices,
                                        base.accept_lengths,
                                        base.last_correct_steps,
                                        base.track_indices,
                                        base.track_steps,
                                        pairs,
                                        base.device_status};
  NativeRuntimeError error = sglang::native::native_runtime_ok();
  CHECK(std::move(capture_gdn_replayssm_commit_graph(fixture.context(),
                                                     fixture.lease(), mixed))
            .match([](CudaCapturedGraph &&) noexcept { return false; },
                   [&error](NativeRuntimeError &&value) noexcept {
                     error = value;
                     return true;
                   }));
  CHECK(error.code == NativeRuntimeCode::kForeignSlice);
  CHECK(error.operation == NativeRuntimeOperation::kGraphCaptureBegin);
  CHECK(error.detail ==
        static_cast<uint32_t>(GdnReplaySsmArgument::kConvStates));
  other_conv.reset();
  other_lease.reset();
  other_slice.reset();
  CHECK_STATUS(other_arena->close());
  return true;
}

[[nodiscard]] bool LayoutValidationCoversOffsetsEnvelopesAndAliases() {
  Fixture fixture;
  CHECK(fixture.initialize());

  // Model the real per-layer cache envelope: this two-layer view starts at
  // physical layer two inside a larger allocation while retaining the owner's
  // full layer stride.
  const std::array<int64_t, 5> temporal_extents{kLayers, kSlots, kValueHeads,
                                                kValueDimension, kKeyDimension};
  const std::array<int64_t, 5> temporal_strides{
      4LL * kSlots * kValueHeads * kValueDimension * kKeyDimension,
      kValueHeads * kValueDimension * kKeyDimension,
      kValueDimension * kKeyDimension, kKeyDimension, 1};
  std::optional<GdnMutableFloat32Tensor5> offset_temporal;
  CHECK(take_result(
      fixture.lease().bind_mutable<DType::kFloat32, 5>(
          fixture.offset_temporal_slice(),
          metadata_with_strides(
              SGL_NATIVE_DTYPE_FLOAT32, temporal_extents, temporal_strides,
              2ULL * kSlots * kValueHeads * kValueDimension * kKeyDimension)),
      &offset_temporal));
  const auto base_buffers = fixture.buffers();
  const GdnReplaySsmCommitBuffers offset_buffers{
      *offset_temporal,
      base_buffers.raw_values,
      base_buffers.raw_keys,
      base_buffers.log_decay,
      base_buffers.beta,
      base_buffers.state_indices,
      base_buffers.accept_lengths,
      base_buffers.last_correct_steps,
      base_buffers.track_indices,
      base_buffers.track_steps,
      base_buffers.conv_pairs,
      base_buffers.device_status};
  GdnReplaySsmShape shape{};
  CHECK(is_ok(validate_gdn_replayssm_commit_buffers(fixture.context(),
                                                    offset_buffers, &shape)));
  CHECK(shape.num_layers == kLayers);
  CHECK(shape.num_slots == kSlots);
  const std::array<int64_t, 1> batch_extent{kBatch};
  const std::array<int64_t, 1> vector_stride{1};
  std::optional<GdnConstInt32Vector> offset_indices;
  CHECK(take_result(fixture.lease().bind_const<DType::kInt32, 1>(
                        fixture.offset_indices_slice(),
                        metadata_with_strides(SGL_NATIVE_DTYPE_INT32,
                                              batch_extent, vector_stride, 1)),
                    &offset_indices));
  const GdnReplaySsmCommitBuffers offset_index_buffers{
      base_buffers.temporal,       base_buffers.raw_values,
      base_buffers.raw_keys,       base_buffers.log_decay,
      base_buffers.beta,           *offset_indices,
      base_buffers.accept_lengths, base_buffers.last_correct_steps,
      base_buffers.track_indices,  base_buffers.track_steps,
      base_buffers.conv_pairs,     base_buffers.device_status};
  CHECK(is_ok(validate_gdn_replayssm_commit_buffers(
      fixture.context(), offset_index_buffers, &shape)));

  const std::array<int64_t, 4> conv_extents{kLayers, kSlots, kConvDimension,
                                            kConvWindow};
  const std::array<int64_t, 4> even_conv_strides{
      2LL * kSlots * kConvDimension * kConvWindow, kConvDimension * kConvWindow,
      kConvWindow, 1};
  const std::array<int64_t, 4> odd_conv_strides = even_conv_strides;
  const std::array<int64_t, 5> window_extents{kLayers, kBatch, kReplay,
                                              kConvDimension, kConvWindow};
  const std::array<int64_t, 5> window_strides{
      kBatch * kReplay * kConvDimension * kConvWindow,
      kReplay * kConvDimension * kConvWindow, kConvDimension * kConvWindow,
      kConvWindow, 1};

  std::optional<GdnMutableBFloat16Tensor4> even_conv;
  std::optional<GdnMutableBFloat16Tensor4> odd_conv;
  std::optional<GdnConstBFloat16ConvTensor5> windows0;
  std::optional<GdnConstBFloat16ConvTensor5> windows1;
  CHECK(take_result(fixture.lease().bind_mutable<DType::kBFloat16, 4>(
                        fixture.alias_conv_slice(),
                        metadata_with_strides(SGL_NATIVE_DTYPE_BFLOAT16,
                                              conv_extents, even_conv_strides)),
                    &even_conv));
  CHECK(take_result(
      fixture.lease().bind_mutable<DType::kBFloat16, 4>(
          fixture.alias_conv_slice(),
          metadata_with_strides(SGL_NATIVE_DTYPE_BFLOAT16, conv_extents,
                                odd_conv_strides,
                                kSlots * kConvDimension * kConvWindow)),
      &odd_conv));
  CHECK(take_result(fixture.lease().bind_const<DType::kBFloat16, 5>(
                        fixture.alias_windows0_slice(),
                        metadata_with_strides(SGL_NATIVE_DTYPE_BFLOAT16,
                                              window_extents, window_strides)),
                    &windows0));
  CHECK(take_result(fixture.lease().bind_const<DType::kBFloat16, 5>(
                        fixture.alias_windows1_slice(),
                        metadata_with_strides(SGL_NATIVE_DTYPE_BFLOAT16,
                                              window_extents, window_strides)),
                    &windows1));

  const std::array<GdnReplaySsmConvPair, 2> disjoint_pairs{
      GdnReplaySsmConvPair{*even_conv, *windows0},
      GdnReplaySsmConvPair{*odd_conv, *windows1}};
  // The two mutable views share one allocation and their envelope intervals
  // overlap, but one occupies even layer bands and the other odd layer bands.
  // Exact segment-aware alias validation must admit this layout.
  const GdnReplaySsmCommitBuffers disjoint_buffers{
      base_buffers.temporal,
      base_buffers.raw_values,
      base_buffers.raw_keys,
      base_buffers.log_decay,
      base_buffers.beta,
      base_buffers.state_indices,
      base_buffers.accept_lengths,
      base_buffers.last_correct_steps,
      base_buffers.track_indices,
      base_buffers.track_steps,
      disjoint_pairs,
      base_buffers.device_status};
  CHECK(is_ok(validate_gdn_replayssm_commit_buffers(fixture.context(),
                                                    disjoint_buffers, &shape)));
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> null_states{-1, -1};
  constexpr std::array<int32_t, kBatch> zero_accepts{0, 0};
  constexpr std::array<int32_t, kBatch> absent{-1, -1};
  CHECK(upload(&fixture, inputs, null_states, zero_accepts, absent, absent,
               absent, 0U));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), disjoint_buffers));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0U);

  const std::array<GdnReplaySsmConvPair, 2> aliased_pairs{
      GdnReplaySsmConvPair{*even_conv, *windows0},
      GdnReplaySsmConvPair{*even_conv, *windows1}};
  const GdnReplaySsmCommitBuffers aliased_buffers{
      base_buffers.temporal,
      base_buffers.raw_values,
      base_buffers.raw_keys,
      base_buffers.log_decay,
      base_buffers.beta,
      base_buffers.state_indices,
      base_buffers.accept_lengths,
      base_buffers.last_correct_steps,
      base_buffers.track_indices,
      base_buffers.track_steps,
      aliased_pairs,
      base_buffers.device_status};
  const NativeRuntimeError alias_error = validate_gdn_replayssm_commit_buffers(
      fixture.context(), aliased_buffers, &shape);
  CHECK(alias_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(
      alias_error.detail ==
      static_cast<uint32_t>(sglang::native::GdnReplaySsmArgument::kConvStates));

  const std::array<int64_t, 5> bad_temporal_strides{
      static_cast<int64_t>(kSlots * kValueHeads * (kValueDimension + 1) *
                           kKeyDimension),
      static_cast<int64_t>(kValueHeads * (kValueDimension + 1) * kKeyDimension),
      static_cast<int64_t>((kValueDimension + 1) * kKeyDimension),
      kKeyDimension + 1, 1};
  std::optional<GdnMutableFloat32Tensor5> bad_temporal;
  CHECK(take_result(
      fixture.lease().bind_mutable<DType::kFloat32, 5>(
          fixture.offset_temporal_slice(),
          metadata_with_strides(SGL_NATIVE_DTYPE_FLOAT32, temporal_extents,
                                bad_temporal_strides)),
      &bad_temporal));
  const GdnReplaySsmCommitBuffers bad_layout_buffers{
      *bad_temporal,
      base_buffers.raw_values,
      base_buffers.raw_keys,
      base_buffers.log_decay,
      base_buffers.beta,
      base_buffers.state_indices,
      base_buffers.accept_lengths,
      base_buffers.last_correct_steps,
      base_buffers.track_indices,
      base_buffers.track_steps,
      base_buffers.conv_pairs,
      base_buffers.device_status};
  const NativeRuntimeError layout_error = validate_gdn_replayssm_commit_buffers(
      fixture.context(), bad_layout_buffers, &shape);
  CHECK(layout_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(layout_error.detail == 2U);
  return true;
}

[[nodiscard]] bool MalformedTrackSentinelCombinationsPreserveOutputs() {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 3};
  constexpr std::array<int32_t, kBatch> accepts{3, 1};
  constexpr std::array<int32_t, kBatch> lasts{2, 0};

  const auto run = [&](std::array<int32_t, kBatch> tracks,
                       std::array<int32_t, kBatch> steps,
                       uint32_t expected_status) noexcept -> bool {
    if (!upload(&fixture, inputs, states, accepts, lasts, tracks, steps,
                kSentinelStatus) ||
        !check_status(
            launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()),
            "launch_gdn_replayssm_commit", __LINE__) ||
        !fixture.synchronize() || fixture.status() != expected_status ||
        fixture.temporal() != inputs.temporal ||
        fixture.conv_states() != inputs.conv_states) {
      return false;
    }
    return true;
  };
  CHECK(run({-1, -1}, {0, -1}, 0x00030004U));
  CHECK(run({2, -1}, {-1, -1}, 0x00030005U));
  CHECK(run({2, -1}, {3, -1}, 0x00030005U));
  constexpr std::array<int32_t, kBatch> bad_lasts{1, 0};
  constexpr std::array<int32_t, kBatch> absent{-1, -1};
  CHECK(upload(&fixture, inputs, states, accepts, bad_lasts, absent, absent,
               kSentinelStatus));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0x00030003U);
  CHECK(fixture.temporal() == inputs.temporal);
  CHECK(fixture.conv_states() == inputs.conv_states);

  constexpr std::array<int32_t, kBatch> null_states{-1, 3};
  constexpr std::array<int32_t, kBatch> null_accepts{1, 1};
  constexpr std::array<int32_t, kBatch> null_lasts{0, 0};
  CHECK(upload(&fixture, inputs, null_states, null_accepts, null_lasts, absent,
               absent, kSentinelStatus));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0x00030001U);
  CHECK(fixture.temporal() == inputs.temporal);
  CHECK(fixture.conv_states() == inputs.conv_states);
  return true;
}

[[nodiscard]] bool TritonOutputMatches(const char *path) {
  Fixture fixture;
  CHECK(fixture.initialize());
  const Inputs inputs = make_inputs();
  constexpr std::array<int32_t, kBatch> states{1, 3};
  constexpr std::array<int32_t, kBatch> accepts{3, 1};
  constexpr std::array<int32_t, kBatch> lasts{2, 0};
  constexpr std::array<int32_t, kBatch> tracks{2, -1};
  constexpr std::array<int32_t, kBatch> steps{1, -1};
  CHECK(upload(&fixture, inputs, states, accepts, lasts, tracks, steps,
               kSentinelStatus));
  CHECK_STATUS(
      launch_gdn_replayssm_commit(fixture.context(), fixture.buffers()));
  CHECK(fixture.synchronize());
  CHECK(fixture.status() == 0U);

  std::ifstream input(path, std::ios::binary | std::ios::ate);
  CHECK(input.is_open());
  const std::streamoff file_bytes = input.tellg();
  const auto actual = fixture.temporal();
  CHECK(file_bytes ==
        static_cast<std::streamoff>(actual.size() * sizeof(float)));
  std::vector<float> expected(actual.size());
  input.seekg(0);
  input.read(reinterpret_cast<char *>(expected.data()), file_bytes);
  CHECK(input.good());
  constexpr std::array<uint32_t, 3> touched_slots{1, 2, 3};
  CHECK(matches_selected_slots(actual, expected, touched_slots, "triton"));
  return true;
}

struct TestCase final {
  const char *name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"AcceptedPrefixTrackAndConvolutionMatch",
     AcceptedPrefixTrackAndConvolutionMatch},
    {"AllAcceptedLengthsWork", AllAcceptedLengthsWork},
    {"MalformedContentPreservesOutputs", MalformedContentPreservesOutputs},
    {"ReadyStatusCompositionAndGraphReplayWork",
     ReadyStatusCompositionAndGraphReplayWork},
    {"CapturedReplaySsmFactoryOwnsConvolutionGraph",
     CapturedReplaySsmFactoryOwnsConvolutionGraph},
    {"ReplaySsmCaptureRejectsForeignConvolutionArena",
     ReplaySsmCaptureRejectsForeignConvolutionArena},
    {"LayoutValidationCoversOffsetsEnvelopesAndAliases",
     LayoutValidationCoversOffsetsEnvelopesAndAliases},
    {"MalformedTrackSentinelCombinationsPreserveOutputs",
     MalformedTrackSentinelCombinationsPreserveOutputs},
};

} // namespace

int main(int argc, char **argv) {
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
  if (argc == 2) {
    std::printf("[ RUN      ] TritonOutputMatches\n");
    if (!TritonOutputMatches(argv[1])) {
      std::printf("[  FAILED  ] TritonOutputMatches\n");
      return 1;
    }
    ++passed;
    std::printf("[       OK ] TritonOutputMatches\n");
  } else if (argc != 1) {
    std::printf("usage: %s [triton-fp32-output.bin]\n", argv[0]);
    return 2;
  }
  std::printf("[  PASSED  ] %u tests\n", passed);
  return 0;
}
