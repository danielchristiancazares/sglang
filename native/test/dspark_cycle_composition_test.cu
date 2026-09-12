#include "sglang/native/dspark_cycle_controller.hpp"
#include "sglang/native/gdn_replayssm_commit.hpp"
#include "sglang/native/linear_rejection_sampling.hpp"
#include "sglang/native/linear_verify_rng.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

using namespace sglang::native;

constexpr uint32_t kNumSlots = kDsparkProductionNumVerifyTokens;
constexpr uint32_t kGamma = kDsparkProductionGamma;
constexpr uint32_t kVocabSize = kDsparkProductionVocabSize;
constexpr uint32_t kMarkovRank = kDsparkProductionMarkovRank;
constexpr uint64_t kProposalCounterBlocks =
    static_cast<uint64_t>(kGamma) * ((kVocabSize + 3ULL) / 4ULL);
constexpr uint64_t kVerifyCounterBlocks = (kNumSlots + 4ULL) / 4ULL;
constexpr int32_t kBonusToken = 42;
constexpr int32_t kRequestSlot = 9;

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

template <std::size_t Rank>
[[nodiscard]] SglNativeTensorMetadataV1
metadata(SglNativeDType dtype,
         const std::array<int64_t, Rank> &extents) noexcept {
  auto result = make_tensor_metadata_v1();
  result.dtype = dtype;
  result.rank = static_cast<uint32_t>(Rank);
  int64_t stride = 1;
  for (std::size_t reverse = Rank; reverse > 0; --reverse) {
    const std::size_t dimension = reverse - 1;
    result.extents[dimension] = extents[dimension];
    result.strides[dimension] = stride;
    stride *= extents[dimension];
  }
  return result;
}

template <typename View, typename T>
[[nodiscard]] bool copy_to_device(const View &view,
                                  std::span<const T> values) noexcept {
  return values.size_bytes() == view.allocation_bytes() &&
         cudaMemcpy(const_cast<std::byte *>(view.data_bytes()), values.data(),
                    values.size_bytes(), cudaMemcpyHostToDevice) == cudaSuccess;
}

template <typename T, typename View>
[[nodiscard]] bool copy_to_host(std::span<T> values,
                                const View &view) noexcept {
  return values.size_bytes() == view.allocation_bytes() &&
         cudaMemcpy(values.data(), view.data_bytes(), values.size_bytes(),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

[[nodiscard]] NativeRuntimeError capture_error(cudaError_t status) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kCudaRuntimeFailure,
                            NativeRuntimeOperation::kGraphCaptureEnd,
                            static_cast<int32_t>(status),
                            0,
                            0,
                            0};
}

__global__ void prepare_target_verify_inputs(const int64_t *anchor_token,
                                             const int64_t *draft_tokens,
                                             int64_t *proposal_tokens,
                                             int64_t *proposal_out_indices,
                                             int32_t *stage_marker,
                                             const uint32_t *device_status) {
  if (blockIdx.x != 0U || threadIdx.x != 0U || device_status[0] != 0U) {
    return;
  }
  proposal_tokens[0] = anchor_token[0];
  proposal_out_indices[0] = 0;
  for (uint32_t step = 0; step < kGamma; ++step) {
    proposal_tokens[step + 1U] = draft_tokens[step];
    proposal_out_indices[step + 1U] = static_cast<int64_t>(step + 1U);
  }
  stage_marker[0] = 2;
}

__global__ void append_stage_marker(int32_t *stage_marker, uint32_t digit,
                                    const uint32_t *device_status) {
  if (blockIdx.x == 0U && threadIdx.x == 0U && device_status[0] == 0U) {
    stage_marker[0] = stage_marker[0] * 10 + static_cast<int32_t>(digit);
  }
}

struct TargetCapturePayload final {
  const CudaExecutionContext *context;
  const int64_t *anchor_token;
  const int64_t *draft_tokens;
  int64_t *proposal_tokens;
  int64_t *proposal_out_indices;
  int32_t *stage_marker;
  const uint32_t *device_status;
};

[[nodiscard]] NativeRuntimeError capture_target_body(void *opaque) noexcept {
  const auto *payload = static_cast<const TargetCapturePayload *>(opaque);
  prepare_target_verify_inputs<<<1, 1, 0, payload->context->stream()>>>(
      payload->anchor_token, payload->draft_tokens, payload->proposal_tokens,
      payload->proposal_out_indices, payload->stage_marker,
      payload->device_status);
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? native_runtime_ok() : capture_error(status);
}

struct MarkerCapturePayload final {
  const CudaExecutionContext *context;
  int32_t *stage_marker;
  const uint32_t *device_status;
  uint32_t digit;
};

[[nodiscard]] NativeRuntimeError capture_marker_body(void *opaque) noexcept {
  const auto *payload = static_cast<const MarkerCapturePayload *>(opaque);
  append_stage_marker<<<1, 1, 0, payload->context->stream()>>>(
      payload->stage_marker, payload->digit, payload->device_status);
  const cudaError_t status = cudaGetLastError();
  return status == cudaSuccess ? native_runtime_ok() : capture_error(status);
}

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

class UnifiedCycleFixture final {
public:
  UnifiedCycleFixture(const UnifiedCycleFixture &) = delete;
  UnifiedCycleFixture &operator=(const UnifiedCycleFixture &) = delete;
  UnifiedCycleFixture(UnifiedCycleFixture &&) = delete;
  UnifiedCycleFixture &operator=(UnifiedCycleFixture &&) = delete;
  UnifiedCycleFixture() = default;

  [[nodiscard]] bool initialize() noexcept {
    constexpr uint64_t kCapacity = 320ULL * 1024ULL * 1024ULL;
    constexpr uint64_t kDraftElements =
        static_cast<uint64_t>(kGamma) * kVocabSize;
    constexpr uint64_t kMarkovElements =
        static_cast<uint64_t>(kVocabSize) * kMarkovRank;
    constexpr uint64_t kTargetElements =
        static_cast<uint64_t>(kNumSlots) * kVocabSize;

    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(GraphMemoryArena::allocate(*context_, kCapacity),
                     &arena_) ||
        !reserve(kBaseLogits, kDraftElements * sizeof(uint16_t)) ||
        !reserve(kMarkovW1, kMarkovElements * sizeof(uint16_t)) ||
        !reserve(kMarkovW2, kMarkovElements * sizeof(uint16_t)) ||
        !reserve(kAnchorToken, sizeof(int64_t)) ||
        !reserve(kTemperature, sizeof(float)) ||
        !reserve(kGreedyMask, sizeof(uint8_t)) ||
        !reserve(kProposalRng, sizeof(DsparkProposalRngStateV1)) ||
        !reserve(kDraftTokens, kGamma * sizeof(int64_t)) ||
        !reserve(kCorrectedLogits, kDraftElements * sizeof(uint16_t)) ||
        !reserve(kLogNormalizers, kGamma * sizeof(float)) ||
        !reserve(kDeviceStatus, sizeof(uint32_t)) ||
        !reserve(kVerifyRng, sizeof(LinearVerifyRngStateV1)) ||
        !reserve(kAcceptUniforms, kNumSlots * sizeof(float)) ||
        !reserve(kBonusUniform, sizeof(float)) ||
        !reserve(kOutTokens, kNumSlots * sizeof(int32_t)) ||
        !reserve(kAcceptIndices, kNumSlots * sizeof(int32_t)) ||
        !reserve(kNumCorrectDrafts, sizeof(int32_t)) ||
        !reserve(kProposalTokens, kNumSlots * sizeof(int64_t)) ||
        !reserve(kProposalOutIndices, kNumSlots * sizeof(int64_t)) ||
        !reserve(kTargetProbabilities, kTargetElements * sizeof(float)) ||
        !reserve(kTemporal, 2U * sizeof(float)) ||
        !reserve(kRawValues, 16U * sizeof(uint16_t)) ||
        !reserve(kRawKeys, 16U * sizeof(uint16_t)) ||
        !reserve(kLogDecay, 16U * sizeof(float)) ||
        !reserve(kBeta, 16U * sizeof(float)) ||
        !reserve(kStateIndex, sizeof(int32_t)) ||
        !reserve(kAcceptLength, sizeof(int32_t)) ||
        !reserve(kLastCorrectStep, sizeof(int32_t)) ||
        !reserve(kTrackIndex, sizeof(int32_t)) ||
        !reserve(kTrackStep, sizeof(int32_t)) ||
        !reserve(kConvStates, 2U * sizeof(uint16_t)) ||
        !reserve(kConvWindows, 8U * sizeof(uint16_t)) ||
        !reserve(kRequestSlotSlice, sizeof(int32_t)) ||
        !reserve(kResultWords, sizeof(DsparkCycleResultV1)) ||
        !reserve(kStageMarker, sizeof(int32_t)) || !is_ok(arena_->seal()) ||
        !take_result(arena_->acquire_lease(), &lease_)) {
      return false;
    }

    if (!bind_const<DType::kBFloat16, 3>(
            kBaseLogits,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 3>{1, kGamma, kVocabSize}),
            &base_logits_) ||
        !bind_const<DType::kBFloat16, 2>(
            kMarkovW1,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 2>{kVocabSize, kMarkovRank}),
            &markov_w1_) ||
        !bind_const<DType::kBFloat16, 2>(
            kMarkovW2,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 2>{kVocabSize, kMarkovRank}),
            &markov_w2_) ||
        !bind_const<DType::kInt64, 1>(
            kAnchorToken,
            metadata(SGL_NATIVE_DTYPE_INT64, std::array<int64_t, 1>{1}),
            &anchor_token_) ||
        !bind_const<DType::kFloat32, 1>(
            kTemperature,
            metadata(SGL_NATIVE_DTYPE_FLOAT32, std::array<int64_t, 1>{1}),
            &temperature_) ||
        !bind_const<DType::kBool8, 1>(
            kGreedyMask,
            metadata(SGL_NATIVE_DTYPE_BOOL8, std::array<int64_t, 1>{1}),
            &greedy_mask_) ||
        !bind_mutable<DType::kUInt64, 1>(
            kProposalRng,
            metadata(SGL_NATIVE_DTYPE_UINT64, std::array<int64_t, 1>{4}),
            &proposal_rng_) ||
        !bind_mutable<DType::kInt64, 2>(
            kDraftTokens,
            metadata(SGL_NATIVE_DTYPE_INT64, std::array<int64_t, 2>{1, kGamma}),
            &draft_tokens_mutable_) ||
        !bind_const<DType::kInt64, 2>(
            kDraftTokens,
            metadata(SGL_NATIVE_DTYPE_INT64, std::array<int64_t, 2>{1, kGamma}),
            &draft_tokens_const_) ||
        !bind_mutable<DType::kBFloat16, 3>(
            kCorrectedLogits,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 3>{1, kGamma, kVocabSize}),
            &corrected_logits_mutable_) ||
        !bind_const<DType::kBFloat16, 3>(
            kCorrectedLogits,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 3>{1, kGamma, kVocabSize}),
            &corrected_logits_const_) ||
        !bind_mutable<DType::kFloat32, 2>(
            kLogNormalizers,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 2>{1, kGamma}),
            &log_normalizers_mutable_) ||
        !bind_const<DType::kFloat32, 2>(
            kLogNormalizers,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 2>{1, kGamma}),
            &log_normalizers_const_) ||
        !bind_mutable<DType::kUInt32, 1>(
            kDeviceStatus,
            metadata(SGL_NATIVE_DTYPE_UINT32, std::array<int64_t, 1>{1}),
            &device_status_) ||
        !bind_mutable<DType::kUInt64, 1>(
            kVerifyRng,
            metadata(SGL_NATIVE_DTYPE_UINT64, std::array<int64_t, 1>{4}),
            &verify_rng_) ||
        !bind_mutable<DType::kFloat32, 2>(
            kAcceptUniforms,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &accept_uniforms_mutable_) ||
        !bind_const<DType::kFloat32, 2>(
            kAcceptUniforms,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &accept_uniforms_const_) ||
        !bind_mutable<DType::kFloat32, 1>(
            kBonusUniform,
            metadata(SGL_NATIVE_DTYPE_FLOAT32, std::array<int64_t, 1>{1}),
            &bonus_uniform_mutable_) ||
        !bind_const<DType::kFloat32, 1>(
            kBonusUniform,
            metadata(SGL_NATIVE_DTYPE_FLOAT32, std::array<int64_t, 1>{1}),
            &bonus_uniform_const_) ||
        !bind_mutable<DType::kInt32, 1>(
            kOutTokens,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{kNumSlots}),
            &out_tokens_mutable_) ||
        !bind_const<DType::kInt32, 1>(
            kOutTokens,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{kNumSlots}),
            &out_tokens_const_) ||
        !bind_mutable<DType::kInt32, 2>(
            kAcceptIndices,
            metadata(SGL_NATIVE_DTYPE_INT32,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &accept_indices_) ||
        !bind_mutable<DType::kInt32, 1>(
            kNumCorrectDrafts,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{1}),
            &num_correct_mutable_) ||
        !bind_const<DType::kInt32, 1>(
            kNumCorrectDrafts,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{1}),
            &num_correct_const_) ||
        !bind_mutable<DType::kInt64, 2>(
            kProposalTokens,
            metadata(SGL_NATIVE_DTYPE_INT64,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &proposal_tokens_mutable_) ||
        !bind_const<DType::kInt64, 2>(
            kProposalTokens,
            metadata(SGL_NATIVE_DTYPE_INT64,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &proposal_tokens_const_) ||
        !bind_mutable<DType::kInt64, 2>(
            kProposalOutIndices,
            metadata(SGL_NATIVE_DTYPE_INT64,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &proposal_indices_mutable_) ||
        !bind_const<DType::kInt64, 2>(
            kProposalOutIndices,
            metadata(SGL_NATIVE_DTYPE_INT64,
                     std::array<int64_t, 2>{1, kNumSlots}),
            &proposal_indices_const_) ||
        !bind_const<DType::kFloat32, 3>(
            kTargetProbabilities,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 3>{1, kNumSlots, kVocabSize}),
            &target_probabilities_) ||
        !bind_mutable<DType::kFloat32, 5>(
            kTemporal,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 5>{1, 2, 1, 1, 1}),
            &temporal_) ||
        !bind_const<DType::kBFloat16, 5>(
            kRawValues,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 5>{1, 2, 1, 8, 1}),
            &raw_values_) ||
        !bind_const<DType::kBFloat16, 5>(
            kRawKeys,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 5>{1, 2, 1, 8, 1}),
            &raw_keys_) ||
        !bind_const<DType::kFloat32, 4>(
            kLogDecay,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 4>{1, 2, 1, 8}),
            &log_decay_) ||
        !bind_const<DType::kFloat32, 4>(
            kBeta,
            metadata(SGL_NATIVE_DTYPE_FLOAT32,
                     std::array<int64_t, 4>{1, 2, 1, 8}),
            &beta_) ||
        !bind_gdn_vector(kStateIndex, &state_index_) ||
        !bind_gdn_vector(kAcceptLength, &accept_length_) ||
        !bind_gdn_vector(kLastCorrectStep, &last_correct_step_) ||
        !bind_gdn_vector(kTrackIndex, &track_index_) ||
        !bind_gdn_vector(kTrackStep, &track_step_) ||
        !bind_mutable<DType::kBFloat16, 4>(
            kConvStates,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 4>{1, 2, 1, 1}),
            &conv_states_) ||
        !bind_const<DType::kBFloat16, 5>(
            kConvWindows,
            metadata(SGL_NATIVE_DTYPE_BFLOAT16,
                     std::array<int64_t, 5>{1, 1, 8, 1, 1}),
            &conv_windows_) ||
        !bind_const<DType::kInt32, 1>(
            kRequestSlotSlice,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{1}),
            &request_slot_) ||
        !bind_mutable<DType::kUInt32, 1>(
            kResultWords,
            metadata(SGL_NATIVE_DTYPE_UINT32, std::array<int64_t, 1>{6}),
            &result_words_) ||
        !bind_mutable<DType::kInt32, 1>(
            kStageMarker,
            metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{1}),
            &stage_marker_)) {
      return false;
    }
    return initialize_values();
  }

  [[nodiscard]] DsparkProposalBuffers<DType::kBFloat16>
  proposal_buffers() const noexcept {
    return {*base_logits_,
            *markov_w1_,
            *markov_w2_,
            *anchor_token_,
            *temperature_,
            *greedy_mask_,
            *proposal_rng_,
            *draft_tokens_mutable_,
            *corrected_logits_mutable_,
            *log_normalizers_mutable_,
            *device_status_};
  }

  [[nodiscard]] StatefulLinearVerifyRngBuffers rng_buffers() const noexcept {
    return {*verify_rng_, *accept_uniforms_mutable_, *bonus_uniform_mutable_,
            *device_status_};
  }

  [[nodiscard]] LinearRejectionSamplingCorrectedLogitBuffers<DType::kBFloat16>
  rejection_buffers() const noexcept {
    return {*out_tokens_mutable_,
            *accept_indices_,
            *num_correct_mutable_,
            *proposal_tokens_const_,
            *proposal_indices_const_,
            *accept_uniforms_const_,
            *bonus_uniform_const_,
            *target_probabilities_,
            *corrected_logits_const_,
            *log_normalizers_const_,
            *temperature_,
            *device_status_};
  }

  [[nodiscard]] GdnReplaySsmCommitBuffers gdn_buffers() const noexcept {
    conv_pair_.emplace(GdnReplaySsmConvPair{*conv_states_, *conv_windows_});
    return {*temporal_,
            *raw_values_,
            *raw_keys_,
            *log_decay_,
            *beta_,
            *state_index_,
            *accept_length_,
            *last_correct_step_,
            *track_index_,
            *track_step_,
            std::span<const GdnReplaySsmConvPair>(&*conv_pair_, 1),
            *device_status_};
  }

  [[nodiscard]] DsparkCycleCompactResultBuffers
  compact_buffers() const noexcept {
    return {*out_tokens_const_, *num_correct_const_, *request_slot_,
            *device_status_, *result_words_};
  }

  [[nodiscard]] TargetCapturePayload target_payload() const noexcept {
    return {
        &context(),
        reinterpret_cast<const int64_t *>(anchor_token_->data_bytes()),
        reinterpret_cast<const int64_t *>(draft_tokens_const_->data_bytes()),
        reinterpret_cast<int64_t *>(proposal_tokens_mutable_->data_bytes()),
        reinterpret_cast<int64_t *>(proposal_indices_mutable_->data_bytes()),
        reinterpret_cast<int32_t *>(stage_marker_->data_bytes()),
        reinterpret_cast<const uint32_t *>(device_status_->data_bytes())};
  }

  [[nodiscard]] MarkerCapturePayload
  marker_payload(uint32_t digit) const noexcept {
    return {&context(),
            reinterpret_cast<int32_t *>(stage_marker_->data_bytes()),
            reinterpret_cast<const uint32_t *>(device_status_->data_bytes()),
            digit};
  }

  [[nodiscard]] bool arm_replay() noexcept {
    constexpr uint32_t kStaleStatus = 0xdeadbeefU;
    constexpr int32_t kMarker = -1;
    const std::array<int32_t, kNumSlots> token_sentinels{-1, -1, -1, -1,
                                                         -1, -1, -1, -1};
    constexpr int32_t kCountSentinel = -1;
    return copy_to_device(*out_tokens_mutable_,
                          std::span<const int32_t>(token_sentinels)) &&
           copy_to_device(*num_correct_mutable_,
                          std::span<const int32_t>(&kCountSentinel, 1)) &&
           copy_to_device(*stage_marker_,
                          std::span<const int32_t>(&kMarker, 1)) &&
           copy_to_device(*device_status_,
                          std::span<const uint32_t>(&kStaleStatus, 1));
  }

  [[nodiscard]] bool
  read_proposal_rng(DsparkProposalRngStateV1 *state) const noexcept {
    std::array<uint64_t, 4> words{};
    if (state == nullptr ||
        !copy_to_host(std::span<uint64_t>(words), *proposal_rng_)) {
      return false;
    }
    *state = {words[0], words[1], words[2], words[3]};
    return true;
  }

  [[nodiscard]] bool
  read_verify_rng(LinearVerifyRngStateV1 *state) const noexcept {
    std::array<uint64_t, 4> words{};
    if (state == nullptr ||
        !copy_to_host(std::span<uint64_t>(words), *verify_rng_)) {
      return false;
    }
    *state = {words[0], words[1], words[2], words[3]};
    return true;
  }

  [[nodiscard]] bool
  read_full_proposal(std::array<int64_t, kNumSlots> *values) const noexcept {
    return values != nullptr &&
           copy_to_host(std::span<int64_t>(*values), *proposal_tokens_const_);
  }

  [[nodiscard]] bool read_marker(int32_t *value) const noexcept {
    return value != nullptr &&
           copy_to_host(std::span<int32_t>(value, 1), *stage_marker_);
  }

  [[nodiscard]] bool
  read_temporal(std::array<float, 2> *values) const noexcept {
    return values != nullptr &&
           copy_to_host(std::span<float>(*values), *temporal_);
  }

  [[nodiscard]] bool
  read_conv_states(std::array<uint16_t, 2> *values) const noexcept {
    return values != nullptr &&
           copy_to_host(std::span<uint16_t>(*values), *conv_states_);
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
  [[nodiscard]] const void *device_result_data() const noexcept {
    return result_words_->data_bytes();
  }
  [[nodiscard]] std::array<uintptr_t, 5> stable_addresses() const noexcept {
    return {
        reinterpret_cast<uintptr_t>(draft_tokens_mutable_->data_bytes()),
        reinterpret_cast<uintptr_t>(corrected_logits_mutable_->data_bytes()),
        reinterpret_cast<uintptr_t>(accept_uniforms_mutable_->data_bytes()),
        reinterpret_cast<uintptr_t>(out_tokens_mutable_->data_bytes()),
        reinterpret_cast<uintptr_t>(temporal_->data_bytes())};
  }

private:
  enum Slice : std::size_t {
    kBaseLogits,
    kMarkovW1,
    kMarkovW2,
    kAnchorToken,
    kTemperature,
    kGreedyMask,
    kProposalRng,
    kDraftTokens,
    kCorrectedLogits,
    kLogNormalizers,
    kDeviceStatus,
    kVerifyRng,
    kAcceptUniforms,
    kBonusUniform,
    kOutTokens,
    kAcceptIndices,
    kNumCorrectDrafts,
    kProposalTokens,
    kProposalOutIndices,
    kTargetProbabilities,
    kTemporal,
    kRawValues,
    kRawKeys,
    kLogDecay,
    kBeta,
    kStateIndex,
    kAcceptLength,
    kLastCorrectStep,
    kTrackIndex,
    kTrackStep,
    kConvStates,
    kConvWindows,
    kRequestSlotSlice,
    kResultWords,
    kStageMarker,
    kSliceCount
  };

  [[nodiscard]] bool reserve(Slice index, uint64_t bytes) noexcept {
    return take_result(arena_->reserve(bytes, 256), &slices_[index]);
  }

  template <DType D, uint32_t Rank, typename OptionalView>
  [[nodiscard]] bool bind_const(Slice index, SglNativeTensorMetadataV1 value,
                                OptionalView *view) noexcept {
    return take_result(lease_->bind_const<D, Rank>(*slices_[index], value),
                       view);
  }

  template <DType D, uint32_t Rank, typename OptionalView>
  [[nodiscard]] bool bind_mutable(Slice index, SglNativeTensorMetadataV1 value,
                                  OptionalView *view) noexcept {
    return take_result(lease_->bind_mutable<D, Rank>(*slices_[index], value),
                       view);
  }

  [[nodiscard]] bool
  bind_gdn_vector(Slice index,
                  std::optional<GdnConstInt32Vector> *view) noexcept {
    return bind_const<DType::kInt32, 1>(
        index, metadata(SGL_NATIVE_DTYPE_INT32, std::array<int64_t, 1>{1}),
        view);
  }

  [[nodiscard]] bool initialize_values() noexcept {
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(base_logits_->data_bytes()),
                          0, base_logits_->allocation_bytes()));
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(markov_w1_->data_bytes()), 0,
                          markov_w1_->allocation_bytes()));
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(markov_w2_->data_bytes()), 0,
                          markov_w2_->allocation_bytes()));
    const uint16_t eight = __bfloat16_as_ushort(__float2bfloat16(8.0F));
    const uint16_t one_bf16 = __bfloat16_as_ushort(__float2bfloat16(1.0F));
    for (uint32_t token = 0; token <= kGamma; ++token) {
      const uint64_t w1_element =
          static_cast<uint64_t>(token) * kMarkovRank + token;
      const uint64_t w2_element =
          static_cast<uint64_t>(token + 1U) * kMarkovRank + token;
      CHECK_CUDA(cudaMemcpy(const_cast<std::byte *>(markov_w1_->data_bytes()) +
                                w1_element * sizeof(uint16_t),
                            &one_bf16, sizeof(one_bf16),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(const_cast<std::byte *>(markov_w2_->data_bytes()) +
                                w2_element * sizeof(uint16_t),
                            &eight, sizeof(eight), cudaMemcpyHostToDevice));
    }

    constexpr int64_t kAnchor = 0;
    constexpr float kTemperatureValue = 1.0F;
    constexpr uint8_t kGreedy = 1;
    const DsparkProposalRngStateV1 proposal_state =
        make_dspark_proposal_rng_state_v1(17U, 23U, 29U);
    const std::array<uint64_t, 4> proposal_words{
        proposal_state.descriptor, proposal_state.seed,
        proposal_state.subsequence, proposal_state.counter};
    const LinearVerifyRngStateV1 verify_state =
        make_linear_verify_rng_state_v1(31U, 37U, 41U);
    const std::array<uint64_t, 4> verify_words{
        verify_state.descriptor, verify_state.seed, verify_state.subsequence,
        verify_state.counter};
    std::vector<float> target(static_cast<uint64_t>(kNumSlots) * kVocabSize,
                              0.0F);
    for (uint32_t row = 0; row < kGamma; ++row) {
      target[static_cast<uint64_t>(row) * kVocabSize + row + 1U] = 1.0F;
    }
    target[static_cast<uint64_t>(kGamma) * kVocabSize + kBonusToken] = 1.0F;

    const std::array<float, 2> temporal{0.0F, 0.0F};
    const std::array<uint16_t, 16> raw_values = [] {
      std::array<uint16_t, 16> values{};
      values.fill(__bfloat16_as_ushort(__float2bfloat16(2.0F)));
      return values;
    }();
    const std::array<uint16_t, 16> raw_keys = [] {
      std::array<uint16_t, 16> values{};
      values.fill(__bfloat16_as_ushort(__float2bfloat16(1.0F)));
      return values;
    }();
    const std::array<float, 16> log_decay{};
    const std::array<float, 16> beta = [] {
      std::array<float, 16> values{};
      values.fill(1.0F);
      return values;
    }();
    constexpr int32_t kStateIndexValue = 0;
    constexpr int32_t kAcceptLengthValue = 1;
    constexpr int32_t kLastCorrectStepValue = 0;
    constexpr int32_t kAbsent = -1;
    const std::array<uint16_t, 2> conv_states{};
    const std::array<uint16_t, 8> conv_windows = [] {
      std::array<uint16_t, 8> values{};
      for (uint32_t step = 0; step < values.size(); ++step) {
        values[step] = __bfloat16_as_ushort(__float2bfloat16(5.0F + step));
      }
      return values;
    }();
    constexpr std::array<uint32_t, 6> kResultSentinels{
        0xccccccccU, 0xccccccccU, 0xccccccccU,
        0xccccccccU, 0xccccccccU, 0xccccccccU};

    return copy_to_device(*anchor_token_,
                          std::span<const int64_t>(&kAnchor, 1)) &&
           copy_to_device(*temperature_,
                          std::span<const float>(&kTemperatureValue, 1)) &&
           copy_to_device(*greedy_mask_,
                          std::span<const uint8_t>(&kGreedy, 1)) &&
           copy_to_device(*proposal_rng_,
                          std::span<const uint64_t>(proposal_words)) &&
           copy_to_device(*verify_rng_,
                          std::span<const uint64_t>(verify_words)) &&
           copy_to_device(*target_probabilities_,
                          std::span<const float>(target)) &&
           copy_to_device(*temporal_, std::span<const float>(temporal)) &&
           copy_to_device(*raw_values_,
                          std::span<const uint16_t>(raw_values)) &&
           copy_to_device(*raw_keys_, std::span<const uint16_t>(raw_keys)) &&
           copy_to_device(*log_decay_, std::span<const float>(log_decay)) &&
           copy_to_device(*beta_, std::span<const float>(beta)) &&
           copy_to_device(*state_index_,
                          std::span<const int32_t>(&kStateIndexValue, 1)) &&
           copy_to_device(*accept_length_,
                          std::span<const int32_t>(&kAcceptLengthValue, 1)) &&
           copy_to_device(
               *last_correct_step_,
               std::span<const int32_t>(&kLastCorrectStepValue, 1)) &&
           copy_to_device(*track_index_,
                          std::span<const int32_t>(&kAbsent, 1)) &&
           copy_to_device(*track_step_,
                          std::span<const int32_t>(&kAbsent, 1)) &&
           copy_to_device(*conv_states_,
                          std::span<const uint16_t>(conv_states)) &&
           copy_to_device(*conv_windows_,
                          std::span<const uint16_t>(conv_windows)) &&
           copy_to_device(*request_slot_,
                          std::span<const int32_t>(&kRequestSlot, 1)) &&
           copy_to_device(*result_words_,
                          std::span<const uint32_t>(kResultSentinels)) &&
           arm_replay();
  }

  std::optional<CudaStream> stream_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphMemoryArena> arena_;
  std::array<std::optional<GraphMemorySlice>, kSliceCount> slices_;
  std::optional<GraphArenaLease> lease_;

  std::optional<DsparkConstBFloat16Tensor3> base_logits_;
  std::optional<DsparkConstBFloat16Matrix> markov_w1_;
  std::optional<DsparkConstBFloat16Matrix> markov_w2_;
  std::optional<DsparkConstInt64Vector> anchor_token_;
  std::optional<DsparkConstFloat32Vector> temperature_;
  std::optional<DsparkConstBool8Vector> greedy_mask_;
  std::optional<DsparkMutableUInt64Vector> proposal_rng_;
  std::optional<DsparkMutableInt64Matrix> draft_tokens_mutable_;
  std::optional<ConstInt64Matrix> draft_tokens_const_;
  std::optional<DsparkMutableBFloat16Tensor3> corrected_logits_mutable_;
  std::optional<ConstBFloat16Tensor3> corrected_logits_const_;
  std::optional<DsparkMutableFloat32Matrix> log_normalizers_mutable_;
  std::optional<ConstFloat32Matrix> log_normalizers_const_;
  std::optional<MutableUInt32Vector> device_status_;
  std::optional<LinearVerifyMutableUInt64Vector> verify_rng_;
  std::optional<LinearVerifyMutableFloat32Matrix> accept_uniforms_mutable_;
  std::optional<ConstFloat32Matrix> accept_uniforms_const_;
  std::optional<LinearVerifyMutableFloat32Vector> bonus_uniform_mutable_;
  std::optional<ConstFloat32Vector> bonus_uniform_const_;
  std::optional<MutableInt32Vector> out_tokens_mutable_;
  std::optional<DsparkCycleConstInt32Vector> out_tokens_const_;
  std::optional<MutableInt32Matrix> accept_indices_;
  std::optional<MutableInt32Vector> num_correct_mutable_;
  std::optional<DsparkCycleConstInt32Vector> num_correct_const_;
  std::optional<DsparkMutableInt64Matrix> proposal_tokens_mutable_;
  std::optional<ConstInt64Matrix> proposal_tokens_const_;
  std::optional<DsparkMutableInt64Matrix> proposal_indices_mutable_;
  std::optional<ConstInt64Matrix> proposal_indices_const_;
  std::optional<ConstFloat32Tensor3> target_probabilities_;
  std::optional<GdnMutableFloat32Tensor5> temporal_;
  std::optional<GdnConstBFloat16Tensor5> raw_values_;
  std::optional<GdnConstBFloat16Tensor5> raw_keys_;
  std::optional<GdnConstFloat32Tensor4> log_decay_;
  std::optional<GdnConstFloat32Tensor4> beta_;
  std::optional<GdnConstInt32Vector> state_index_;
  std::optional<GdnConstInt32Vector> accept_length_;
  std::optional<GdnConstInt32Vector> last_correct_step_;
  std::optional<GdnConstInt32Vector> track_index_;
  std::optional<GdnConstInt32Vector> track_step_;
  std::optional<GdnMutableBFloat16Tensor4> conv_states_;
  std::optional<GdnConstBFloat16ConvTensor5> conv_windows_;
  std::optional<DsparkCycleConstInt32Vector> request_slot_;
  std::optional<DsparkCycleMutableUInt32Vector> result_words_;
  std::optional<MutableInt32Vector> stage_marker_;
  mutable std::optional<GdnReplaySsmConvPair> conv_pair_;
};

[[nodiscard]] bool UnifiedRealFactoriesComposeAndOutliveSources() {
  UnifiedCycleFixture fixture;
  CHECK(fixture.initialize());
  const auto stable_addresses = fixture.stable_addresses();

  std::optional<CudaCapturedGraph> proposal;
  std::optional<CudaCapturedGraph> target;
  std::optional<CudaCapturedGraph> verify_rng;
  std::optional<CudaCapturedGraph> rejection;
  std::optional<CudaCapturedGraph> replayssm;
  std::optional<CudaCapturedGraph> draft_extend;
  std::optional<CudaCapturedGraph> kv_write;
  std::optional<DsparkCycleCompactGraph> compact;
  std::optional<DsparkCycleController> controller;
  std::optional<DsparkModelGraphResourceOwner> model_owner;
  auto retained_model_resources = std::make_shared<const uint32_t>(0x4453504bU);
  CHECK(take_result(
      DsparkModelGraphResourceOwner::create(fixture.context(), fixture.lease(),
                                            retained_model_resources),
      &model_owner));

  CHECK(take_result(
      capture_dspark_proposal_bfloat16_graph(fixture.context(), fixture.lease(),
                                             fixture.proposal_buffers()),
      &proposal));
  TargetCapturePayload target_payload = fixture.target_payload();
  CHECK(take_result(CudaCapturedGraph::capture_retaining(
                        fixture.context(), fixture.lease(),
                        model_owner->acquire_lease().retain(),
                        &capture_target_body, &target_payload),
                    &target));
  CHECK(take_result(
      capture_stateful_linear_verify_rng_graph(
          fixture.context(), fixture.lease(), fixture.rng_buffers()),
      &verify_rng));
  CHECK(take_result(
      capture_linear_rejection_sampling_from_bfloat16_logits_graph(
          fixture.context(), fixture.lease(), fixture.rejection_buffers()),
      &rejection));
  CHECK(take_result(capture_gdn_replayssm_commit_graph(fixture.context(),
                                                       fixture.lease(),
                                                       fixture.gdn_buffers()),
                    &replayssm));
  MarkerCapturePayload draft_payload = fixture.marker_payload(6U);
  CHECK(take_result(CudaCapturedGraph::capture_retaining(
                        fixture.context(), fixture.lease(),
                        model_owner->acquire_lease().retain(),
                        &capture_marker_body, &draft_payload),
                    &draft_extend));
  MarkerCapturePayload kv_payload = fixture.marker_payload(7U);
  CHECK(take_result(CudaCapturedGraph::capture_retaining(
                        fixture.context(), fixture.lease(),
                        model_owner->acquire_lease().retain(),
                        &capture_marker_body, &kv_payload),
                    &kv_write));
  CHECK(take_result(DsparkCycleCompactGraph::capture(fixture.context(),
                                                     fixture.lease(),
                                                     fixture.compact_buffers()),
                    &compact));

  const DsparkCycleGraphSources sources{
      *proposal,
      DsparkTargetVerifyGraphSource{model_binding(
          DsparkModelGraphKind::kTargetVerify, *target, *model_owner)},
      *verify_rng,
      *rejection,
      *replayssm,
      DsparkDraftExtendGraphSource{model_binding(
          DsparkModelGraphKind::kDraftExtend, *draft_extend, *model_owner)},
      DsparkKvWriteGraphSource{model_binding(DsparkModelGraphKind::kKvWrite,
                                             *kv_write, *model_owner)},
      *compact};
  CHECK(take_result(
      DsparkCycleController::create(sources, fixture.device_status_data(),
                                    fixture.device_result_data(),
                                    fixture.context(), fixture.lease()),
      &controller));
  CHECK(model_owner->active_leases() == 9U);

  CHECK_STATUS(proposal->close());
  CHECK_STATUS(target->close());
  CHECK_STATUS(verify_rng->close());
  CHECK_STATUS(rejection->close());
  CHECK_STATUS(replayssm->close());
  CHECK_STATUS(draft_extend->close());
  CHECK_STATUS(kv_write->close());
  CHECK_STATUS(compact->close());
  proposal.reset();
  target.reset();
  verify_rng.reset();
  rejection.reset();
  replayssm.reset();
  draft_extend.reset();
  kv_write.reset();
  compact.reset();
  CHECK(model_owner->active_leases() == 6U);
  model_owner.reset();

  for (uint64_t replay = 0; replay < 2U; ++replay) {
    CHECK(fixture.arm_replay());
    CHECK_STATUS(controller->launch());
    CHECK_STATUS(controller->synchronize());
    const DsparkCycleResultV1 *result = controller->result();
    CHECK(result != nullptr);
    CHECK(result->abi_version == kDsparkCycleAbiVersion);
    CHECK(result->device_status == 0U);
    CHECK(result->accepted_token_count == static_cast<int32_t>(kNumSlots));
    CHECK(result->num_correct_drafts == static_cast<int32_t>(kGamma));
    CHECK(result->output_token == kBonusToken);
    CHECK(result->request_slot == kRequestSlot);

    std::array<int64_t, kNumSlots> proposal_tokens{};
    CHECK(fixture.read_full_proposal(&proposal_tokens));
    for (uint32_t token = 0; token < proposal_tokens.size(); ++token) {
      CHECK(proposal_tokens[token] == static_cast<int64_t>(token));
    }
    int32_t marker = 0;
    CHECK(fixture.read_marker(&marker));
    CHECK(marker == 267);
    CHECK(fixture.stable_addresses() == stable_addresses);

    DsparkProposalRngStateV1 proposal_state{};
    LinearVerifyRngStateV1 verify_state{};
    CHECK(fixture.read_proposal_rng(&proposal_state));
    CHECK(fixture.read_verify_rng(&verify_state));
    CHECK(proposal_state.counter ==
          29U + (replay + 1U) * kProposalCounterBlocks);
    CHECK(verify_state.counter == 41U + (replay + 1U) * kVerifyCounterBlocks);
  }

  std::array<float, 2> temporal{};
  std::array<uint16_t, 2> conv_states{};
  CHECK(fixture.read_temporal(&temporal));
  CHECK(fixture.read_conv_states(&conv_states));
  CHECK(temporal[0] > 0.0F);
  CHECK(temporal[1] == 0.0F);
  CHECK(conv_states[0] == __bfloat16_as_ushort(__float2bfloat16(5.0F)));
  CHECK(conv_states[1] == 0U);

  CHECK_STATUS(controller->close());
  controller.reset();
  return true;
}

} // namespace

int main() {
  std::printf("[ RUN      ] UnifiedRealFactoriesComposeAndOutliveSources\n");
  if (!UnifiedRealFactoriesComposeAndOutliveSources()) {
    std::printf("[  FAILED  ] UnifiedRealFactoriesComposeAndOutliveSources\n");
    return 1;
  }
  std::printf("[       OK ] UnifiedRealFactoriesComposeAndOutliveSources\n");
  std::printf("[  PASSED  ] 1 test\n");
  return 0;
}
