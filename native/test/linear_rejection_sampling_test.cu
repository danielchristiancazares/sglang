#include "sglang/native/linear_rejection_sampling.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <initializer_list>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using sglang::native::ConstFloat32Matrix;
using sglang::native::ConstFloat32Tensor3;
using sglang::native::ConstFloat32Vector;
using sglang::native::ConstInt64Matrix;
using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaStream;
using sglang::native::DType;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::LinearRejectionSamplingArgument;
using sglang::native::LinearRejectionSamplingBuffers;
using sglang::native::LinearRejectionSamplingDeviceCode;
using sglang::native::MutableInt32Matrix;
using sglang::native::MutableInt32Vector;
using sglang::native::MutableUInt32Vector;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::NativeRuntimeResult;
using sglang::native::TensorAccess;
using sglang::native::is_ok;
using sglang::native::kLinearRejectionSamplingThreads;
using sglang::native::launch_linear_rejection_sampling;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::native_runtime_code_name;
using sglang::native::native_runtime_operation_name;

constexpr int32_t kOutputSentinel = -9137;
constexpr uint32_t kStatusSentinel = 0xffffffffU;

[[nodiscard]] bool record_check(bool passed, const char* expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

void print_error(NativeRuntimeError error) noexcept {
  const auto code = native_runtime_code_name(error.code);
  const auto operation = native_runtime_operation_name(error.operation);
  std::printf(
      "runtime error code=%.*s operation=%.*s native=%d detail=%u "
      "actual=%llu required=%llu\n",
      static_cast<int>(code.size()), code.data(),
      static_cast<int>(operation.size()), operation.data(),
      error.native_code, error.detail,
      static_cast<unsigned long long>(error.actual),
      static_cast<unsigned long long>(error.required));
}

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) { \
      return false;                                                         \
    }                                                                       \
  } while (false)

#define CHECK_CUDA(expression)                                               \
  do {                                                                       \
    const cudaError_t cuda_status = (expression);                             \
    if (cuda_status != cudaSuccess) {                                         \
      std::printf("%s:%d: CUDA failure %d: %s\n", __FILE__, __LINE__,        \
                  static_cast<int>(cuda_status),                              \
                  cudaGetErrorString(cuda_status));                           \
      return false;                                                          \
    }                                                                        \
  } while (false)

[[nodiscard]] bool check_status(NativeRuntimeError status,
                                const char* expression,
                                int line) noexcept {
  if (is_ok(status)) {
    return true;
  }
  std::printf("%s:%d: failed status: %s\n", __FILE__, line, expression);
  print_error(status);
  return false;
}

#define CHECK_STATUS(expression)                                             \
  do {                                                                       \
    if (!check_status((expression), #expression, __LINE__)) {                \
      return false;                                                          \
    }                                                                        \
  } while (false)

template <typename T>
[[nodiscard]] bool take_result(NativeRuntimeResult<T> result,
                               std::optional<T>* output) noexcept {
  return std::move(result).match(
      [output](T&& value) noexcept {
        output->emplace(std::move(value));
        return true;
      },
      [](NativeRuntimeError&& error) noexcept {
        print_error(error);
        return false;
      });
}

[[nodiscard]] SglNativeTensorMetadataV1 metadata_1d(
    SglNativeDType dtype, int64_t first) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 1;
  metadata.extents[0] = first;
  metadata.strides[0] = 1;
  return metadata;
}

[[nodiscard]] SglNativeTensorMetadataV1 metadata_2d(
    SglNativeDType dtype, int64_t first, int64_t second) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 2;
  metadata.extents[0] = first;
  metadata.extents[1] = second;
  metadata.strides[0] = second;
  metadata.strides[1] = 1;
  return metadata;
}

[[nodiscard]] SglNativeTensorMetadataV1 metadata_3d(
    SglNativeDType dtype, int64_t first, int64_t second,
    int64_t third) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 3;
  metadata.extents[0] = first;
  metadata.extents[1] = second;
  metadata.extents[2] = third;
  metadata.strides[0] = second * third;
  metadata.strides[1] = third;
  metadata.strides[2] = 1;
  return metadata;
}

struct HostInputs final {
  uint32_t num_slots;
  uint32_t vocab_size;
  std::vector<int64_t> proposal_tokens;
  std::vector<int64_t> proposal_out_indices;
  std::vector<float> accept_uniforms;
  std::array<float, 1> bonus_uniforms;
  std::vector<float> target_probs;
  std::vector<float> draft_probs;
};

struct HostOutputs final {
  std::vector<int32_t> out_tokens;
  std::vector<int32_t> accept_indices;
  std::array<int32_t, 1> num_correct_drafts;
  std::array<uint32_t, 1> device_status;
};

[[nodiscard]] HostOutputs sentinel_outputs(uint32_t num_slots) {
  return HostOutputs{
      std::vector<int32_t>(num_slots, kOutputSentinel),
      std::vector<int32_t>(num_slots, kOutputSentinel),
      {kOutputSentinel},
      {kStatusSentinel}};
}

[[nodiscard]] float probability_mass(
    const HostInputs& inputs, uint32_t row, uint32_t token,
    bool all_drafts_correct) {
  const uint64_t offset =
      static_cast<uint64_t>(row) * inputs.vocab_size + token;
  const float target = inputs.target_probs[offset];
  if (all_drafts_correct) {
    return target;
  }
  const float raw_draft = inputs.draft_probs[offset];
  const float draft = raw_draft == raw_draft ? raw_draft : 0.0F;
  const float difference = target - draft;
  return difference > 0.0F ? difference : 0.0F;
}

[[nodiscard]] HostOutputs reference_sampling(const HostInputs& inputs) {
  HostOutputs outputs = sentinel_outputs(inputs.num_slots);
  LinearRejectionSamplingDeviceCode code =
      LinearRejectionSamplingDeviceCode::kOk;
  for (uint32_t slot = 0; slot < inputs.num_slots; ++slot) {
    const int64_t out_index = inputs.proposal_out_indices[slot];
    if (out_index < 0 ||
        out_index >= static_cast<int64_t>(inputs.num_slots)) {
      code = LinearRejectionSamplingDeviceCode::
          kProposalOutIndexOutOfRange;
      break;
    }
    for (uint32_t prior = 0; prior < slot; ++prior) {
      if (inputs.proposal_out_indices[prior] == out_index) {
        code =
            LinearRejectionSamplingDeviceCode::kDuplicateProposalOutIndex;
        break;
      }
    }
    if (code != LinearRejectionSamplingDeviceCode::kOk) {
      break;
    }
    const int64_t proposal_token = inputs.proposal_tokens[slot];
    if (proposal_token < 0 ||
        proposal_token >= static_cast<int64_t>(inputs.vocab_size)) {
      code =
          LinearRejectionSamplingDeviceCode::kProposalTokenOutOfRange;
      break;
    }
  }
  outputs.device_status[0] = static_cast<uint32_t>(code);
  if (code != LinearRejectionSamplingDeviceCode::kOk) {
    return outputs;
  }

  uint32_t current_prob_row = 0;
  uint32_t num_correct = 0;
  uint32_t last_accept_out_index =
      static_cast<uint32_t>(inputs.proposal_out_indices[0]);
  bool all_drafts_correct = true;
  outputs.accept_indices[0] =
      static_cast<int32_t>(last_accept_out_index);
  for (uint32_t step = 1; step < inputs.num_slots; ++step) {
    const uint32_t proposal_token =
        static_cast<uint32_t>(inputs.proposal_tokens[step]);
    const uint64_t offset =
        static_cast<uint64_t>(current_prob_row) * inputs.vocab_size +
        proposal_token;
    const float target = inputs.target_probs[offset];
    const float draft = inputs.draft_probs[offset];
    if (inputs.accept_uniforms[step - 1] * draft < target) {
      outputs.out_tokens[last_accept_out_index] =
          static_cast<int32_t>(proposal_token);
      ++num_correct;
      current_prob_row = step;
      last_accept_out_index =
          static_cast<uint32_t>(inputs.proposal_out_indices[step]);
      outputs.accept_indices[num_correct] =
          static_cast<int32_t>(last_accept_out_index);
    } else {
      all_drafts_correct = false;
      break;
    }
  }
  outputs.num_correct_drafts[0] = static_cast<int32_t>(num_correct);

  constexpr uint32_t kThreads = kLinearRejectionSamplingThreads;
  const uint64_t segment_size =
      (static_cast<uint64_t>(inputs.vocab_size) + kThreads - 1) /
      kThreads;
  std::array<float, kThreads> partition_masses{};
  std::array<float, kThreads> partition_prefixes{};
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    const uint64_t begin =
        static_cast<uint64_t>(thread) * segment_size;
    const uint64_t end =
        std::min<uint64_t>(begin + segment_size, inputs.vocab_size);
    float mass = 0.0F;
    for (uint64_t token = begin; token < end; ++token) {
      mass += probability_mass(
          inputs, current_prob_row, static_cast<uint32_t>(token),
          all_drafts_correct);
    }
    partition_masses[thread] = mass;
  }

  float total_mass = 0.0F;
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    partition_prefixes[thread] = total_mass;
    total_mass += partition_masses[thread];
  }
  const float target_mass = inputs.bonus_uniforms[0] * total_mass;
  uint32_t bonus_token = inputs.vocab_size - 1;
  for (uint32_t thread = 0; thread < kThreads; ++thread) {
    const uint64_t begin =
        static_cast<uint64_t>(thread) * segment_size;
    const uint64_t end =
        std::min<uint64_t>(begin + segment_size, inputs.vocab_size);
    float cumulative_mass = partition_prefixes[thread];
    for (uint64_t token = begin; token < end; ++token) {
      cumulative_mass += probability_mass(
          inputs, current_prob_row, static_cast<uint32_t>(token),
          all_drafts_correct);
      if (cumulative_mass > target_mass) {
        bonus_token = std::min(
            bonus_token, static_cast<uint32_t>(token));
        break;
      }
    }
  }
  outputs.out_tokens[last_accept_out_index] =
      static_cast<int32_t>(bonus_token);
  return outputs;
}

[[nodiscard]] bool outputs_equal(const HostOutputs& actual,
                                 const HostOutputs& expected) {
  CHECK(actual.out_tokens == expected.out_tokens);
  CHECK(actual.accept_indices == expected.accept_indices);
  CHECK(actual.num_correct_drafts == expected.num_correct_drafts);
  CHECK(actual.device_status == expected.device_status);
  return true;
}

class SamplingFixture final {
 public:
  SamplingFixture(const SamplingFixture&) = delete;
  SamplingFixture& operator=(const SamplingFixture&) = delete;
  SamplingFixture(SamplingFixture&&) = delete;
  SamplingFixture& operator=(SamplingFixture&&) = delete;

  explicit SamplingFixture(uint32_t num_slots,
                           uint32_t vocab_size) noexcept
      : num_slots_(num_slots), vocab_size_(vocab_size) {}

  [[nodiscard]] bool initialize() noexcept {
    const uint64_t slots = num_slots_;
    const uint64_t vocab = vocab_size_;
    const uint64_t target_bytes =
        slots * vocab * sizeof(float);
    const uint64_t draft_bytes =
        (slots - 1) * vocab * sizeof(float);
    const uint64_t capacity =
        target_bytes + draft_bytes +
        slots * (2 * sizeof(int32_t) + 2 * sizeof(int64_t) +
                 sizeof(float)) +
        2 * sizeof(int32_t) + sizeof(float) + sizeof(uint32_t) +
        10 * 256;

    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(
            GraphMemoryArena::allocate(*context_, capacity), &arena_) ||
        !reserve(slots * sizeof(int32_t), &out_tokens_slice_) ||
        !reserve(slots * sizeof(int32_t), &accept_indices_slice_) ||
        !reserve(sizeof(int32_t), &num_correct_drafts_slice_) ||
        !reserve(slots * sizeof(int64_t), &proposal_tokens_slice_) ||
        !reserve(slots * sizeof(int64_t),
                 &proposal_out_indices_slice_) ||
        !reserve(slots * sizeof(float), &accept_uniforms_slice_) ||
        !reserve(sizeof(float), &bonus_uniforms_slice_) ||
        !reserve(target_bytes, &target_probs_slice_) ||
        !reserve(draft_bytes, &draft_probs_slice_) ||
        !reserve(sizeof(uint32_t), &device_status_slice_)) {
      return false;
    }
    if (!check_status(arena_->seal(), "arena_->seal()", __LINE__) ||
        !take_result(arena_->acquire_lease(), &lease_)) {
      return false;
    }

    if (!take_result(
            lease_->bind_mutable<DType::kInt32, 1>(
                *out_tokens_slice_,
                metadata_1d(SGL_NATIVE_DTYPE_INT32, num_slots_)),
            &out_tokens_) ||
        !take_result(
            lease_->bind_mutable<DType::kInt32, 2>(
                *accept_indices_slice_,
                metadata_2d(
                    SGL_NATIVE_DTYPE_INT32, 1, num_slots_)),
            &accept_indices_) ||
        !take_result(
            lease_->bind_mutable<DType::kInt32, 1>(
                *num_correct_drafts_slice_,
                metadata_1d(SGL_NATIVE_DTYPE_INT32, 1)),
            &num_correct_drafts_) ||
        !take_result(
            lease_->bind_const<DType::kInt64, 2>(
                *proposal_tokens_slice_,
                metadata_2d(
                    SGL_NATIVE_DTYPE_INT64, 1, num_slots_)),
            &proposal_tokens_) ||
        !take_result(
            lease_->bind_const<DType::kInt64, 2>(
                *proposal_out_indices_slice_,
                metadata_2d(
                    SGL_NATIVE_DTYPE_INT64, 1, num_slots_)),
            &proposal_out_indices_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 2>(
                *accept_uniforms_slice_,
                metadata_2d(
                    SGL_NATIVE_DTYPE_FLOAT32, 1, num_slots_)),
            &accept_uniforms_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 1>(
                *bonus_uniforms_slice_,
                metadata_1d(SGL_NATIVE_DTYPE_FLOAT32, 1)),
            &bonus_uniforms_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 3>(
                *target_probs_slice_,
                metadata_3d(
                    SGL_NATIVE_DTYPE_FLOAT32, 1, num_slots_,
                    vocab_size_)),
            &target_probs_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 3>(
                *draft_probs_slice_,
                metadata_3d(
                    SGL_NATIVE_DTYPE_FLOAT32, 1, num_slots_ - 1,
                    vocab_size_)),
            &draft_probs_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt32, 1>(
                *device_status_slice_,
                metadata_1d(SGL_NATIVE_DTYPE_UINT32, 1)),
            &device_status_)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] LinearRejectionSamplingBuffers buffers() const noexcept {
    return LinearRejectionSamplingBuffers{
        *out_tokens_,
        *accept_indices_,
        *num_correct_drafts_,
        *proposal_tokens_,
        *proposal_out_indices_,
        *accept_uniforms_,
        *bonus_uniforms_,
        *target_probs_,
        *draft_probs_,
        *device_status_};
  }

  [[nodiscard]] LinearRejectionSamplingBuffers buffers_with_status(
      const MutableUInt32Vector& status) const noexcept {
    return LinearRejectionSamplingBuffers{
        *out_tokens_,
        *accept_indices_,
        *num_correct_drafts_,
        *proposal_tokens_,
        *proposal_out_indices_,
        *accept_uniforms_,
        *bonus_uniforms_,
        *target_probs_,
        *draft_probs_,
        status};
  }

  [[nodiscard]] LinearRejectionSamplingBuffers buffers_with_accept_uniforms(
      const ConstFloat32Matrix& uniforms) const noexcept {
    return LinearRejectionSamplingBuffers{
        *out_tokens_,
        *accept_indices_,
        *num_correct_drafts_,
        *proposal_tokens_,
        *proposal_out_indices_,
        uniforms,
        *bonus_uniforms_,
        *target_probs_,
        *draft_probs_,
        *device_status_};
  }

  [[nodiscard]] bool copy_inputs(const HostInputs& inputs) noexcept {
    if (inputs.num_slots != num_slots_ ||
        inputs.vocab_size != vocab_size_ ||
        inputs.proposal_tokens.size() != num_slots_ ||
        inputs.proposal_out_indices.size() != num_slots_ ||
        inputs.accept_uniforms.size() != num_slots_ ||
        inputs.target_probs.size() !=
            static_cast<uint64_t>(num_slots_) * vocab_size_ ||
        inputs.draft_probs.size() !=
            static_cast<uint64_t>(num_slots_ - 1) * vocab_size_) {
      return false;
    }
    return copy_to_const(
               *proposal_tokens_,
               std::span<const int64_t>(inputs.proposal_tokens)) &&
           copy_to_const(
               *proposal_out_indices_,
               std::span<const int64_t>(
                   inputs.proposal_out_indices)) &&
           copy_to_const(
               *accept_uniforms_,
               std::span<const float>(inputs.accept_uniforms)) &&
           copy_to_const(
               *bonus_uniforms_,
               std::span<const float>(inputs.bonus_uniforms)) &&
           copy_to_const(
               *target_probs_,
               std::span<const float>(inputs.target_probs)) &&
           copy_to_const(
               *draft_probs_,
               std::span<const float>(inputs.draft_probs));
  }

  [[nodiscard]] bool reset_outputs() noexcept {
    const HostOutputs sentinels = sentinel_outputs(num_slots_);
    return copy_to_mutable(
               *out_tokens_,
               std::span<const int32_t>(sentinels.out_tokens)) &&
           copy_to_mutable(
               *accept_indices_,
               std::span<const int32_t>(sentinels.accept_indices)) &&
           copy_to_mutable(
               *num_correct_drafts_,
               std::span<const int32_t>(
                   sentinels.num_correct_drafts)) &&
           copy_to_mutable(
               *device_status_,
               std::span<const uint32_t>(sentinels.device_status));
  }

  [[nodiscard]] bool run(const HostInputs& inputs,
                         HostOutputs* outputs) noexcept {
    if (outputs == nullptr || !copy_inputs(inputs) || !reset_outputs()) {
      return false;
    }
    const auto sampling_buffers = buffers();
    if (!check_status(
            launch_linear_rejection_sampling(
                *context_, sampling_buffers),
            "launch_linear_rejection_sampling", __LINE__) ||
        !check_status(context_->synchronize(), "context_->synchronize()",
                      __LINE__)) {
      return false;
    }
    return read_outputs(outputs);
  }

  [[nodiscard]] bool read_outputs(HostOutputs* outputs) const noexcept {
    if (outputs == nullptr) {
      return false;
    }
    *outputs = sentinel_outputs(num_slots_);
    return copy_to_host(
               std::span<int32_t>(outputs->out_tokens), *out_tokens_) &&
           copy_to_host(
               std::span<int32_t>(outputs->accept_indices),
               *accept_indices_) &&
           copy_to_host(
               std::span<int32_t>(outputs->num_correct_drafts),
               *num_correct_drafts_) &&
           copy_to_host(
               std::span<uint32_t>(outputs->device_status),
               *device_status_);
  }

  [[nodiscard]] const CudaExecutionContext& context() const noexcept {
    return *context_;
  }

  [[nodiscard]] const GraphArenaLease& lease() const noexcept {
    return *lease_;
  }

  [[nodiscard]] const GraphMemorySlice& num_correct_drafts_slice()
      const noexcept {
    return *num_correct_drafts_slice_;
  }

  [[nodiscard]] const GraphMemorySlice& accept_uniforms_slice()
      const noexcept {
    return *accept_uniforms_slice_;
  }

  [[nodiscard]] uintptr_t out_tokens_address() const noexcept {
    return reinterpret_cast<uintptr_t>(out_tokens_->data_bytes());
  }

 private:
  [[nodiscard]] bool reserve(
      uint64_t bytes,
      std::optional<GraphMemorySlice>* output) noexcept {
    return take_result(arena_->reserve(bytes, 256), output);
  }

  template <DType D, uint32_t Rank>
  [[nodiscard]] bool copy_to_const(
      const sglang::native::GraphStableTensorView<
          D, Rank, TensorAccess::kReadOnly>& view,
      std::span<const std::conditional_t<
          D == DType::kInt64, int64_t, float>> values) noexcept {
    const std::size_t bytes = values.size_bytes();
    // Tests initialize owner-backed storage before exposing it as read-only
    // operator input; production adapters must perform the same ownership step.
    const cudaError_t result = cudaMemcpyAsync(
        const_cast<std::byte*>(view.data_bytes()), values.data(), bytes,
        cudaMemcpyHostToDevice, context_->stream());
    if (result != cudaSuccess) {
      std::printf("input copy failed: %s\n", cudaGetErrorString(result));
      return false;
    }
    return true;
  }

  template <DType D, uint32_t Rank, typename T>
  [[nodiscard]] bool copy_to_mutable(
      const sglang::native::GraphStableTensorView<
          D, Rank, TensorAccess::kReadWrite>& view,
      std::span<const T> values) noexcept {
    const cudaError_t result = cudaMemcpyAsync(
        view.data_bytes(), values.data(), values.size_bytes(),
        cudaMemcpyHostToDevice, context_->stream());
    if (result != cudaSuccess) {
      std::printf("output reset failed: %s\n", cudaGetErrorString(result));
      return false;
    }
    return true;
  }

  template <typename T, DType D, uint32_t Rank>
  [[nodiscard]] bool copy_to_host(
      std::span<T> values,
      const sglang::native::GraphStableTensorView<
          D, Rank, TensorAccess::kReadWrite>& view) const noexcept {
    const cudaError_t result = cudaMemcpy(
        values.data(), view.data_bytes(), values.size_bytes(),
        cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
      std::printf("output copy failed: %s\n", cudaGetErrorString(result));
      return false;
    }
    return true;
  }

  uint32_t num_slots_;
  uint32_t vocab_size_;
  std::optional<CudaStream> stream_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphMemoryArena> arena_;
  std::optional<GraphMemorySlice> out_tokens_slice_;
  std::optional<GraphMemorySlice> accept_indices_slice_;
  std::optional<GraphMemorySlice> num_correct_drafts_slice_;
  std::optional<GraphMemorySlice> proposal_tokens_slice_;
  std::optional<GraphMemorySlice> proposal_out_indices_slice_;
  std::optional<GraphMemorySlice> accept_uniforms_slice_;
  std::optional<GraphMemorySlice> bonus_uniforms_slice_;
  std::optional<GraphMemorySlice> target_probs_slice_;
  std::optional<GraphMemorySlice> draft_probs_slice_;
  std::optional<GraphMemorySlice> device_status_slice_;
  std::optional<GraphArenaLease> lease_;
  std::optional<MutableInt32Vector> out_tokens_;
  std::optional<MutableInt32Matrix> accept_indices_;
  std::optional<MutableInt32Vector> num_correct_drafts_;
  std::optional<ConstInt64Matrix> proposal_tokens_;
  std::optional<ConstInt64Matrix> proposal_out_indices_;
  std::optional<ConstFloat32Matrix> accept_uniforms_;
  std::optional<ConstFloat32Vector> bonus_uniforms_;
  std::optional<ConstFloat32Tensor3> target_probs_;
  std::optional<ConstFloat32Tensor3> draft_probs_;
  std::optional<MutableUInt32Vector> device_status_;
};

[[nodiscard]] HostInputs empty_inputs(uint32_t num_slots,
                                      uint32_t vocab_size) {
  HostInputs inputs{
      num_slots,
      vocab_size,
      std::vector<int64_t>(num_slots),
      std::vector<int64_t>(num_slots),
      std::vector<float>(num_slots),
      {0.0F},
      std::vector<float>(
          static_cast<uint64_t>(num_slots) * vocab_size),
      std::vector<float>(
          static_cast<uint64_t>(num_slots - 1) * vocab_size)};
  for (uint32_t slot = 0; slot < num_slots; ++slot) {
    inputs.proposal_out_indices[slot] = slot;
  }
  return inputs;
}

void set_distribution(std::vector<float>* probabilities, uint32_t row,
                      uint32_t vocab_size,
                      std::initializer_list<std::pair<uint32_t, float>>
                          values) {
  const uint64_t offset = static_cast<uint64_t>(row) * vocab_size;
  for (const auto [token, probability] : values) {
    (*probabilities)[offset + token] = probability;
  }
}

[[nodiscard]] bool LayoutAndContentFailuresAreClosed() {
  SamplingFixture fixture(3, 8);
  CHECK(fixture.initialize());

  HostInputs inputs = empty_inputs(3, 8);
  inputs.proposal_tokens = {2, 3, 4};
  inputs.accept_uniforms = {0.5F, 0.5F, 0.0F};
  inputs.bonus_uniforms = {0.5F};
  for (uint32_t row = 0; row < 3; ++row) {
    set_distribution(
        &inputs.target_probs, row, 8, {{3, 0.5F}, {5, 0.5F}});
  }
  for (uint32_t row = 0; row < 2; ++row) {
    set_distribution(
        &inputs.draft_probs, row, 8, {{3, 0.5F}, {5, 0.5F}});
  }

  const std::array<std::pair<
      LinearRejectionSamplingDeviceCode, std::vector<int64_t>>, 2>
      invalid_indices{{
          {LinearRejectionSamplingDeviceCode::
               kProposalOutIndexOutOfRange,
           {0, 3, 2}},
          {LinearRejectionSamplingDeviceCode::
               kDuplicateProposalOutIndex,
           {0, 0, 2}},
      }};
  for (const auto& [expected_code, indices] : invalid_indices) {
    inputs.proposal_out_indices = indices;
    HostOutputs actual = sentinel_outputs(3);
    CHECK(fixture.run(inputs, &actual));
    const HostOutputs expected = reference_sampling(inputs);
    CHECK(outputs_equal(actual, expected));
    CHECK(actual.device_status[0] ==
          static_cast<uint32_t>(expected_code));
  }

  inputs.proposal_out_indices = {0, 1, 2};
  inputs.proposal_tokens[2] = 8;
  HostOutputs actual = sentinel_outputs(3);
  CHECK(fixture.run(inputs, &actual));
  const HostOutputs expected = reference_sampling(inputs);
  CHECK(outputs_equal(actual, expected));
  CHECK(actual.device_status[0] ==
        static_cast<uint32_t>(
            LinearRejectionSamplingDeviceCode::
                kProposalTokenOutOfRange));

  std::optional<MutableUInt32Vector> aliased_status;
  CHECK(take_result(
      fixture.lease().bind_mutable<DType::kUInt32, 1>(
          fixture.num_correct_drafts_slice(),
          metadata_1d(SGL_NATIVE_DTYPE_UINT32, 1)),
      &aliased_status));
  const auto aliased_buffers =
      fixture.buffers_with_status(*aliased_status);
  const NativeRuntimeError alias_error =
      launch_linear_rejection_sampling(
          fixture.context(), aliased_buffers);
  CHECK(alias_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(alias_error.operation ==
        NativeRuntimeOperation::kValidateLinearRejectionSampling);
  CHECK(alias_error.detail ==
        static_cast<uint32_t>(
            LinearRejectionSamplingArgument::kNumCorrectDrafts));
  CHECK(alias_error.actual ==
        static_cast<uint64_t>(
            LinearRejectionSamplingArgument::kDeviceStatus));

  std::optional<ConstFloat32Matrix> short_uniforms;
  CHECK(take_result(
      fixture.lease().bind_const<DType::kFloat32, 2>(
          fixture.accept_uniforms_slice(),
          metadata_2d(SGL_NATIVE_DTYPE_FLOAT32, 1, 2)),
      &short_uniforms));
  const auto short_uniform_buffers =
      fixture.buffers_with_accept_uniforms(*short_uniforms);
  const NativeRuntimeError shape_error =
      launch_linear_rejection_sampling(
          fixture.context(), short_uniform_buffers);
  CHECK(shape_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(shape_error.operation ==
        NativeRuntimeOperation::kValidateLinearRejectionSampling);
  CHECK(shape_error.detail ==
        static_cast<uint32_t>(
            LinearRejectionSamplingArgument::kAcceptUniforms));
  CHECK(shape_error.actual == 2);
  CHECK(shape_error.required == 3);
  return true;
}

[[nodiscard]] bool HandCasesMatchOracle() {
  SamplingFixture fixture(3, 8);
  CHECK(fixture.initialize());

  std::vector<HostInputs> cases;
  HostInputs all_correct = empty_inputs(3, 8);
  all_correct.proposal_tokens = {7, 1, 5};
  all_correct.proposal_out_indices = {2, 0, 1};
  all_correct.accept_uniforms = {0.5F, 0.5F, 0.0F};
  all_correct.bonus_uniforms = {0.5F};
  set_distribution(
      &all_correct.target_probs, 0, 8, {{1, 0.5F}, {3, 0.5F}});
  set_distribution(
      &all_correct.target_probs, 1, 8, {{5, 0.5F}, {6, 0.5F}});
  set_distribution(
      &all_correct.target_probs, 2, 8, {{2, 0.25F}, {6, 0.75F}});
  set_distribution(
      &all_correct.draft_probs, 0, 8, {{1, 0.5F}, {3, 0.5F}});
  set_distribution(
      &all_correct.draft_probs, 1, 8, {{5, 0.5F}, {6, 0.5F}});
  cases.push_back(all_correct);

  HostInputs first_reject = empty_inputs(3, 8);
  first_reject.proposal_tokens = {0, 1, 2};
  first_reject.accept_uniforms = {0.5F, 0.0F, 0.0F};
  first_reject.bonus_uniforms = {0.5F};
  set_distribution(
      &first_reject.target_probs, 0, 8,
      {{1, 0.1F}, {4, 0.6F}, {7, 0.3F}});
  set_distribution(
      &first_reject.target_probs, 1, 8, {{0, 1.0F}});
  set_distribution(
      &first_reject.target_probs, 2, 8, {{0, 1.0F}});
  set_distribution(
      &first_reject.draft_probs, 0, 8,
      {{1, 0.4F}, {3, 0.5F}, {7, 0.1F}});
  set_distribution(
      &first_reject.draft_probs, 1, 8, {{0, 1.0F}});
  cases.push_back(first_reject);

  HostInputs accept_then_reject = empty_inputs(3, 8);
  accept_then_reject.proposal_tokens = {0, 2, 6};
  accept_then_reject.accept_uniforms = {0.25F, 1.0F, 0.0F};
  accept_then_reject.bonus_uniforms = {0.75F};
  set_distribution(
      &accept_then_reject.target_probs, 0, 8,
      {{2, 0.5F}, {4, 0.5F}});
  set_distribution(
      &accept_then_reject.target_probs, 1, 8,
      {{1, 0.6F}, {6, 0.2F}, {7, 0.2F}});
  set_distribution(
      &accept_then_reject.target_probs, 2, 8, {{0, 1.0F}});
  set_distribution(
      &accept_then_reject.draft_probs, 0, 8,
      {{2, 0.5F}, {4, 0.5F}});
  set_distribution(
      &accept_then_reject.draft_probs, 1, 8,
      {{1, 0.1F}, {6, 0.2F}, {5, 0.7F}});
  cases.push_back(accept_then_reject);

  HostInputs nan_draft = empty_inputs(3, 8);
  nan_draft.proposal_tokens = {0, 3, 4};
  nan_draft.accept_uniforms = {0.0F, 0.0F, 0.0F};
  nan_draft.bonus_uniforms = {0.1F};
  set_distribution(
      &nan_draft.target_probs, 0, 8,
      {{1, 0.25F}, {3, 0.25F}, {5, 0.5F}});
  set_distribution(
      &nan_draft.target_probs, 1, 8, {{0, 1.0F}});
  set_distribution(
      &nan_draft.target_probs, 2, 8, {{0, 1.0F}});
  set_distribution(
      &nan_draft.draft_probs, 0, 8,
      {{1, 0.5F}, {3, std::numeric_limits<float>::quiet_NaN()},
       {5, 0.5F}});
  set_distribution(
      &nan_draft.draft_probs, 1, 8, {{0, 1.0F}});
  cases.push_back(nan_draft);

  HostInputs zero_residual = empty_inputs(3, 8);
  zero_residual.proposal_tokens = {0, 2, 3};
  zero_residual.accept_uniforms = {1.0F, 0.0F, 0.0F};
  zero_residual.bonus_uniforms = {0.5F};
  set_distribution(
      &zero_residual.target_probs, 0, 8,
      {{2, 0.25F}, {3, 0.25F}, {4, 0.5F}});
  set_distribution(
      &zero_residual.target_probs, 1, 8, {{0, 1.0F}});
  set_distribution(
      &zero_residual.target_probs, 2, 8, {{0, 1.0F}});
  set_distribution(
      &zero_residual.draft_probs, 0, 8,
      {{2, 0.25F}, {3, 0.25F}, {4, 0.5F}});
  set_distribution(
      &zero_residual.draft_probs, 1, 8, {{0, 1.0F}});
  cases.push_back(zero_residual);

  for (const HostInputs& inputs : cases) {
    HostOutputs actual = sentinel_outputs(3);
    CHECK(fixture.run(inputs, &actual));
    CHECK(outputs_equal(actual, reference_sampling(inputs)));
  }
  CHECK(reference_sampling(zero_residual).out_tokens[0] == 7);
  return true;
}

class DeterministicGenerator final {
 public:
  explicit DeterministicGenerator(uint64_t seed) noexcept : state_(seed) {}

  [[nodiscard]] uint64_t next() noexcept {
    uint64_t value = state_;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    state_ = value;
    return value;
  }

  [[nodiscard]] float uniform() noexcept {
    return static_cast<float>((next() >> 40) & 0xffffffU) /
           16777216.0F;
  }

 private:
  uint64_t state_;
};

void fill_distribution(std::vector<float>* probabilities, uint32_t row,
                       uint32_t vocab_size,
                       DeterministicGenerator* generator) {
  const uint64_t offset = static_cast<uint64_t>(row) * vocab_size;
  float total = 0.0F;
  for (uint32_t token = 0; token < vocab_size; ++token) {
    const float value = 0.01F + generator->uniform();
    (*probabilities)[offset + token] = value;
    total += value;
  }
  for (uint32_t token = 0; token < vocab_size; ++token) {
    (*probabilities)[offset + token] /= total;
  }
}

[[nodiscard]] bool RandomizedParity() {
  constexpr uint32_t kNumSlots = 3;
  constexpr uint32_t kVocabSize = 257;
  constexpr uint32_t kCases = 256;
  SamplingFixture fixture(kNumSlots, kVocabSize);
  CHECK(fixture.initialize());
  DeterministicGenerator generator(0x53474c52534d504cULL);

  for (uint32_t case_index = 0; case_index < kCases; ++case_index) {
    HostInputs inputs = empty_inputs(kNumSlots, kVocabSize);
    for (uint32_t slot = 0; slot < kNumSlots; ++slot) {
      inputs.proposal_tokens[slot] =
          static_cast<int64_t>(generator.next() % kVocabSize);
      inputs.accept_uniforms[slot] = generator.uniform();
      fill_distribution(
          &inputs.target_probs, slot, kVocabSize, &generator);
      if (slot + 1 < kNumSlots) {
        fill_distribution(
            &inputs.draft_probs, slot, kVocabSize, &generator);
      }
    }
    inputs.bonus_uniforms[0] = generator.uniform();
    std::swap(
        inputs.proposal_out_indices[case_index % kNumSlots],
        inputs.proposal_out_indices[
            (case_index * 2 + 1) % kNumSlots]);
    if (case_index % 31 == 0) {
      const uint32_t proposal_token =
          static_cast<uint32_t>(inputs.proposal_tokens[1]);
      inputs.draft_probs[proposal_token] =
          std::numeric_limits<float>::quiet_NaN();
    }

    HostOutputs actual = sentinel_outputs(kNumSlots);
    CHECK(fixture.run(inputs, &actual));
    CHECK(outputs_equal(actual, reference_sampling(inputs)));
  }
  return true;
}

[[nodiscard]] bool run_slot_boundary(uint32_t num_slots) {
  constexpr uint32_t kVocabSize = 67;
  SamplingFixture fixture(num_slots, kVocabSize);
  CHECK(fixture.initialize());
  HostInputs inputs = empty_inputs(num_slots, kVocabSize);
  inputs.bonus_uniforms[0] = 0.5F;
  for (uint32_t slot = 0; slot < num_slots; ++slot) {
    inputs.proposal_tokens[slot] =
        static_cast<int64_t>((slot * 7 + 3) % kVocabSize);
    inputs.proposal_out_indices[slot] = num_slots - slot - 1;
    inputs.accept_uniforms[slot] = 0.0F;
    if (slot != 0) {
      const uint32_t proposal_token =
          static_cast<uint32_t>(inputs.proposal_tokens[slot]);
      set_distribution(
          &inputs.target_probs, slot - 1, kVocabSize,
          {{proposal_token, 1.0F}});
      set_distribution(
          &inputs.draft_probs, slot - 1, kVocabSize,
          {{proposal_token, 1.0F}});
    }
  }
  set_distribution(
      &inputs.target_probs, num_slots - 1, kVocabSize,
      {{kVocabSize - 1, 1.0F}});

  HostOutputs actual = sentinel_outputs(num_slots);
  CHECK(fixture.run(inputs, &actual));
  CHECK(outputs_equal(actual, reference_sampling(inputs)));
  CHECK(actual.num_correct_drafts[0] ==
        static_cast<int32_t>(num_slots - 1));
  return true;
}

[[nodiscard]] bool SlotBoundariesMatchOracle() {
  CHECK(run_slot_boundary(2));
  CHECK(run_slot_boundary(64));
  return true;
}

struct RawGraph final {
  cudaGraph_t value = nullptr;

  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

[[nodiscard]] HostInputs production_reject_case() {
  constexpr uint32_t kVocabSize = 248320;
  HostInputs inputs = empty_inputs(3, kVocabSize);
  inputs.proposal_tokens = {7, 12345, 67890};
  inputs.proposal_out_indices = {2, 0, 1};
  inputs.accept_uniforms = {0.4F, 0.5F, 0.0F};
  inputs.bonus_uniforms = {0.5F};
  set_distribution(
      &inputs.target_probs, 0, kVocabSize,
      {{12345, 0.75F}, {3, 0.25F}});
  set_distribution(
      &inputs.target_probs, 1, kVocabSize,
      {{42, 0.9F}, {67890, 0.1F}});
  set_distribution(
      &inputs.target_probs, 2, kVocabSize,
      {{99, 0.25F}, {150000, 0.75F}});
  set_distribution(
      &inputs.draft_probs, 0, kVocabSize,
      {{12345, 0.5F}, {3, 0.5F}});
  set_distribution(
      &inputs.draft_probs, 1, kVocabSize,
      {{67890, 0.8F}, {99, 0.2F}});
  return inputs;
}

[[nodiscard]] HostInputs production_all_correct_case() {
  constexpr uint32_t kVocabSize = 248320;
  HostInputs inputs = empty_inputs(3, kVocabSize);
  inputs.proposal_tokens = {11, 54321, 77777};
  inputs.proposal_out_indices = {1, 2, 0};
  inputs.accept_uniforms = {0.0F, 0.0F, 0.0F};
  inputs.bonus_uniforms = {0.75F};
  set_distribution(
      &inputs.target_probs, 0, kVocabSize,
      {{54321, 1.0F}});
  set_distribution(
      &inputs.target_probs, 1, kVocabSize,
      {{77777, 1.0F}});
  set_distribution(
      &inputs.target_probs, 2, kVocabSize,
      {{17, 0.25F}, {247000, 0.75F}});
  set_distribution(
      &inputs.draft_probs, 0, kVocabSize,
      {{54321, 1.0F}});
  set_distribution(
      &inputs.draft_probs, 1, kVocabSize,
      {{77777, 1.0F}});
  return inputs;
}

[[nodiscard]] bool ProductionShapeGraphReplay() {
  constexpr uint32_t kVocabSize = 248320;
  SamplingFixture fixture(3, kVocabSize);
  CHECK(fixture.initialize());
  const uintptr_t stable_out_address = fixture.out_tokens_address();

  HostInputs first = production_reject_case();
  CHECK(fixture.copy_inputs(first));
  CHECK(fixture.reset_outputs());
  CHECK_STATUS(fixture.context().synchronize());

  RawGraph graph;
  CHECK_CUDA(cudaStreamBeginCapture(
      fixture.context().stream(), cudaStreamCaptureModeThreadLocal));
  const auto capture_buffers = fixture.buffers();
  CHECK_STATUS(launch_linear_rejection_sampling(
      fixture.context(), capture_buffers));
  CHECK_CUDA(cudaStreamEndCapture(
      fixture.context().stream(), &graph.value));

  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(
      CudaGraphExecutable::instantiate(
          graph.value, fixture.context(), fixture.lease()),
      &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));

  CHECK(fixture.copy_inputs(first));
  CHECK(fixture.reset_outputs());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  HostOutputs first_actual = sentinel_outputs(3);
  CHECK(fixture.read_outputs(&first_actual));
  CHECK(outputs_equal(first_actual, reference_sampling(first)));
  CHECK(fixture.out_tokens_address() == stable_out_address);

  HostInputs second = production_all_correct_case();
  CHECK(fixture.copy_inputs(second));
  CHECK(fixture.reset_outputs());
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  HostOutputs second_actual = sentinel_outputs(3);
  CHECK(fixture.read_outputs(&second_actual));
  CHECK(outputs_equal(second_actual, reference_sampling(second)));
  CHECK(first_actual.out_tokens != second_actual.out_tokens);
  CHECK(fixture.out_tokens_address() == stable_out_address);

  CHECK_STATUS(executable->close());
  executable.reset();
  return true;
}

struct TestCase final {
  const char* name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"LayoutAndContentFailuresAreClosed",
     LayoutAndContentFailuresAreClosed},
    {"HandCasesMatchOracle", HandCasesMatchOracle},
    {"RandomizedParity", RandomizedParity},
    {"SlotBoundariesMatchOracle", SlotBoundariesMatchOracle},
    {"ProductionShapeGraphReplay", ProductionShapeGraphReplay},
};

}  // namespace

int main() {
  uint32_t passed = 0;
  for (const auto& test : kTests) {
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
