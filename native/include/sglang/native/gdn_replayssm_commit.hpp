#ifndef SGLANG_NATIVE_GDN_REPLAYSSM_COMMIT_HPP_
#define SGLANG_NATIVE_GDN_REPLAYSSM_COMMIT_HPP_

#include "sglang/native/cuda_graph_resources.hpp"

#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kGdnProductionNumLayers = 48;
inline constexpr uint32_t kGdnProductionNumValueHeads = 48;
inline constexpr uint32_t kGdnProductionNumKeyHeads = 16;
inline constexpr uint32_t kGdnProductionKeyDimension = 128;
inline constexpr uint32_t kGdnProductionValueDimension = 128;
inline constexpr uint32_t kGdnProductionReplayLength = 8;
inline constexpr uint32_t kGdnProductionConvWindow = 3;

enum class GdnReplaySsmArgument : uint32_t {
  kNone = 0,
  kShape = 1,
  kTemporal = 2,
  kRawValue = 3,
  kRawKey = 4,
  kLogDecay = 5,
  kBeta = 6,
  kStateIndices = 7,
  kAcceptLengths = 8,
  kLastCorrectSteps = 9,
  kTrackIndices = 10,
  kTrackSteps = 11,
  kConvStates = 12,
  kIntermediateConv = 13,
  kDeviceStatus = 14,
};

enum class GdnReplaySsmDeviceCode : uint32_t {
  kOk = 0,
  kStateIndexOutOfRange = 0x00030001U,
  kAcceptLengthOutOfRange = 0x00030002U,
  kLastCorrectStepOutOfRange = 0x00030003U,
  kTrackIndexOutOfRange = 0x00030004U,
  kTrackStepOutOfRange = 0x00030005U,
};

struct GdnReplaySsmShape final {
  uint64_t num_layers;
  uint64_t num_slots;
  uint64_t batch_size;
  uint64_t num_value_heads;
  uint64_t num_key_heads;
  uint64_t key_dimension;
  uint64_t value_dimension;
  uint64_t replay_length;
};

using GdnMutableFloat32Tensor5 =
    GraphStableTensorView<DType::kFloat32, 5, TensorAccess::kReadWrite>;
using GdnConstBFloat16Tensor5 =
    GraphStableTensorView<DType::kBFloat16, 5, TensorAccess::kReadOnly>;
using GdnConstFloat32Tensor4 =
    GraphStableTensorView<DType::kFloat32, 4, TensorAccess::kReadOnly>;
using GdnConstInt32Vector =
    GraphStableTensorView<DType::kInt32, 1, TensorAccess::kReadOnly>;
using GdnMutableBFloat16Tensor4 =
    GraphStableTensorView<DType::kBFloat16, 4, TensorAccess::kReadWrite>;
using GdnConstBFloat16ConvTensor5 =
    GraphStableTensorView<DType::kBFloat16, 5, TensorAccess::kReadOnly>;
using GdnMutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;

struct GdnReplaySsmConvPair final {
  const GdnMutableBFloat16Tensor4 &conv_states;
  const GdnConstBFloat16ConvTensor5 &intermediate_conv_windows;
};

struct GdnReplaySsmCommitBuffers final {
  const GdnMutableFloat32Tensor5 &temporal;
  const GdnConstBFloat16Tensor5 &raw_values;
  const GdnConstBFloat16Tensor5 &raw_keys;
  const GdnConstFloat32Tensor4 &log_decay;
  const GdnConstFloat32Tensor4 &beta;
  const GdnConstInt32Vector &state_indices;
  const GdnConstInt32Vector &accept_lengths;
  const GdnConstInt32Vector &last_correct_steps;
  const GdnConstInt32Vector &track_indices;
  const GdnConstInt32Vector &track_steps;
  std::span<const GdnReplaySsmConvPair> conv_pairs;
  const GdnMutableUInt32Vector &device_status;
};

[[nodiscard]] std::string_view
gdn_replayssm_argument_name(GdnReplaySsmArgument argument) noexcept;
[[nodiscard]] std::string_view
gdn_replayssm_device_code_name(GdnReplaySsmDeviceCode code) noexcept;
[[nodiscard]] NativeRuntimeError
validate_gdn_replayssm_shape(GdnReplaySsmShape shape) noexcept;
[[nodiscard]] NativeRuntimeError
validate_gdn_production_replayssm_shape(GdnReplaySsmShape shape) noexcept;

// Host-only metadata validation.  This is deliberately separate from launch
// so adapters can reject malformed, aliased, or incompatible views before
// beginning CUDA graph capture.
[[nodiscard]] NativeRuntimeError
validate_gdn_replayssm_commit_buffers(const CudaExecutionContext &context,
                                      const GdnReplaySsmCommitBuffers &buffers,
                                      GdnReplaySsmShape *shape) noexcept;

// Device-content validation runs before every write.  On failure only the
// device status is published; temporal and convolution states are untouched.
// A negative state is the null-request sentinel and requires zero accepted
// tokens plus absent last/track fields.  Active requests require
// last_correct_step == accept_length - 1.  Track index/step are either both
// absent (-1) or both nonnegative, with track_step inside the accepted prefix.
[[nodiscard]] NativeRuntimeError
launch_gdn_replayssm_commit(const CudaExecutionContext &context,
                            const GdnReplaySsmCommitBuffers &buffers) noexcept;
[[nodiscard]] NativeRuntimeError launch_gdn_replayssm_commit_if_ready(
    const CudaExecutionContext &context,
    const GdnReplaySsmCommitBuffers &buffers) noexcept;

// Captures the guarded ReplaySSM fold and every convolution scatter into one
// source graph. Every fixed view and every view in every convolution pair must
// belong to arena, which the returned graph retains for its full lifetime.
[[nodiscard]] NativeRuntimeResult<CudaCapturedGraph>
capture_gdn_replayssm_commit_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const GdnReplaySsmCommitBuffers &buffers) noexcept;

static_assert(sizeof(GdnReplaySsmShape) == 64);
static_assert(alignof(GdnReplaySsmShape) == 8);
static_assert(std::is_standard_layout_v<GdnReplaySsmShape>);
static_assert(std::is_trivially_copyable_v<GdnReplaySsmShape>);

} // namespace sglang::native

#endif // SGLANG_NATIVE_GDN_REPLAYSSM_COMMIT_HPP_
