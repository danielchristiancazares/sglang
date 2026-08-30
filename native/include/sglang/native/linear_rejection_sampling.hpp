#ifndef SGLANG_NATIVE_LINEAR_REJECTION_SAMPLING_HPP_
#define SGLANG_NATIVE_LINEAR_REJECTION_SAMPLING_HPP_

#include "sglang/native/cuda_graph_resources.hpp"

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kLinearRejectionSamplingThreads = 256;
inline constexpr uint32_t kLinearRejectionSamplingMaxNumSlots = 64;

enum class LinearRejectionSamplingArgument : uint32_t {
  kNone = 0,
  kNumSlots = 1,
  kVocabSize = 2,
  kOutTokens = 3,
  kAcceptIndices = 4,
  kNumCorrectDrafts = 5,
  kProposalTokens = 6,
  kProposalOutIndices = 7,
  kAcceptUniforms = 8,
  kBonusUniforms = 9,
  kTargetProbs = 10,
  kDraftProbs = 11,
  kDeviceStatus = 12,
};

enum class LinearRejectionSamplingDeviceCode : uint32_t {
  kOk = 0,
  kProposalOutIndexOutOfRange = 1,
  kDuplicateProposalOutIndex = 2,
  kProposalTokenOutOfRange = 3,
};

struct LinearRejectionSamplingShape final {
  uint64_t num_slots;
  uint64_t vocab_size;
  uint64_t num_out_tokens;
};

using MutableInt32Vector =
    GraphStableTensorView<DType::kInt32, 1, TensorAccess::kReadWrite>;
using MutableInt32Matrix =
    GraphStableTensorView<DType::kInt32, 2, TensorAccess::kReadWrite>;
using MutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;
using ConstInt64Matrix =
    GraphStableTensorView<DType::kInt64, 2, TensorAccess::kReadOnly>;
using ConstFloat32Vector =
    GraphStableTensorView<DType::kFloat32, 1, TensorAccess::kReadOnly>;
using ConstFloat32Matrix =
    GraphStableTensorView<DType::kFloat32, 2, TensorAccess::kReadOnly>;
using ConstFloat32Tensor3 =
    GraphStableTensorView<DType::kFloat32, 3, TensorAccess::kReadOnly>;

struct LinearRejectionSamplingBuffers final {
  // accept_indices includes the bonus destination; num_correct_drafts does not.
  const MutableInt32Vector& out_tokens;
  const MutableInt32Matrix& accept_indices;
  const MutableInt32Vector& num_correct_drafts;
  const ConstInt64Matrix& proposal_tokens;
  const ConstInt64Matrix& proposal_out_indices;
  const ConstFloat32Matrix& accept_uniforms;
  const ConstFloat32Vector& bonus_uniforms;
  const ConstFloat32Tensor3& target_probs;
  const ConstFloat32Tensor3& draft_probs;
  const MutableUInt32Vector& device_status;
};

[[nodiscard]] std::string_view linear_rejection_sampling_argument_name(
    LinearRejectionSamplingArgument argument) noexcept;
[[nodiscard]] std::string_view linear_rejection_sampling_device_code_name(
    LinearRejectionSamplingDeviceCode code) noexcept;

[[nodiscard]] NativeRuntimeError validate_linear_rejection_sampling_shape(
    LinearRejectionSamplingShape shape) noexcept;

// Metadata failures return synchronously. Content failures are published to
// device_status and become host-visible after the context completes.
[[nodiscard]] NativeRuntimeError launch_linear_rejection_sampling(
    const CudaExecutionContext& context,
    const LinearRejectionSamplingBuffers& buffers) noexcept;

static_assert(sizeof(LinearRejectionSamplingShape) == 24);
static_assert(alignof(LinearRejectionSamplingShape) == 8);
static_assert(std::is_standard_layout_v<LinearRejectionSamplingShape>);
static_assert(std::is_trivially_copyable_v<LinearRejectionSamplingShape>);
static_assert(
    static_cast<uint32_t>(LinearRejectionSamplingDeviceCode::kOk) == 0);

}  // namespace sglang::native

#endif  // SGLANG_NATIVE_LINEAR_REJECTION_SAMPLING_HPP_
