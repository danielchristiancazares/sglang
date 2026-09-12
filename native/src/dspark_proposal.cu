#include "sglang/native/dspark_proposal.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace sglang::native {
namespace {

constexpr uint32_t kThreads = 256;
constexpr float kNegativeInfinity =
    std::bit_cast<float>(static_cast<uint32_t>(0xff800000U));
constexpr uint32_t kTokensPerThread =
    (kDsparkProductionVocabSize + kThreads - 1) / kThreads;
constexpr uint64_t kBlocksPerRow = (kDsparkProductionVocabSize + 3ULL) / 4ULL;
constexpr uint64_t kBlocksPerReplay = kDsparkProductionGamma * kBlocksPerRow;

struct ByteRange final {
  uintptr_t begin;
  uintptr_t end;
  DsparkProposalArgument argument;
};

struct PhiloxBlock final {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

struct ProposalSharedState final {
  uint64_t seed;
  uint64_t subsequence;
  uint64_t first_counter;
  int64_t previous_token;
  int64_t sampled_token;
  float row_max;
  float sum_exp;
  uint32_t valid;
};

[[nodiscard]] constexpr NativeRuntimeError
make_error(NativeRuntimeCode code, NativeRuntimeOperation operation,
           DsparkProposalArgument argument = DsparkProposalArgument::kNone,
           int32_t native_code = 0, uint64_t actual = 0,
           uint64_t required = 0) noexcept {
  return NativeRuntimeError{code,        operation,
                            native_code, static_cast<uint32_t>(argument),
                            actual,      required};
}

[[nodiscard]] bool checked_multiply(uint64_t left, uint64_t right,
                                    uint64_t *product) noexcept {
  if (product == nullptr ||
      (left != 0 && right > std::numeric_limits<uint64_t>::max() / left)) {
    return false;
  }
  *product = left * right;
  return true;
}

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] NativeRuntimeError
validate_tensor(const GraphStableTensorView<D, Rank, Access> &view,
                const CudaExecutionContext &context,
                DsparkProposalArgument argument,
                const std::array<int64_t, Rank> &expected_extents,
                uint64_t expected_bytes, ByteRange *range) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkProposal;
  if (!context.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation, argument);
  }
  if (view.device_kind() != DeviceKind::kCuda ||
      view.device_ordinal() != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation, argument,
                      0, static_cast<uint64_t>(view.device_ordinal()),
                      static_cast<uint64_t>(context.device_ordinal()));
  }
  if (!view.is_row_major_contiguous()) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const std::span<const int64_t, Rank> extents = view.extents();
  for (uint32_t dimension = 0; dimension < Rank; ++dimension) {
    if (extents[dimension] != expected_extents[dimension]) {
      return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                        argument, 0, static_cast<uint64_t>(extents[dimension]),
                        static_cast<uint64_t>(expected_extents[dimension]));
    }
  }
  if (view.allocation_bytes() != expected_bytes) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation, argument,
                      0, view.allocation_bytes(), expected_bytes);
  }
  const auto *const data = view.data_bytes();
  if (data == nullptr || range == nullptr ||
      expected_bytes >
          static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max())) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      argument);
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t bytes = static_cast<uintptr_t>(expected_bytes);
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation,
                      argument);
  }
  *range = ByteRange{begin, begin + bytes, argument};
  return native_runtime_ok();
}

[[nodiscard]] bool overlaps(const ByteRange &left,
                            const ByteRange &right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

template <DType LogitsDType>
[[nodiscard]] NativeRuntimeError
validate_layout(const CudaExecutionContext &context,
                const DsparkProposalBuffers<LogitsDType> &buffers) noexcept {
  constexpr NativeRuntimeOperation kOperation =
      NativeRuntimeOperation::kValidateDsparkProposal;
  if (!context.valid()) {
    return make_error(NativeRuntimeCode::kInvalidState, kOperation);
  }
  int current_device = -1;
  const cudaError_t get_device = cudaGetDevice(&current_device);
  if (get_device != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure, kOperation,
                      DsparkProposalArgument::kNone,
                      static_cast<int32_t>(get_device));
  }
  if (current_device != context.device_ordinal()) {
    return make_error(NativeRuntimeCode::kDeviceMismatch, kOperation,
                      DsparkProposalArgument::kNone, 0,
                      static_cast<uint64_t>(current_device),
                      static_cast<uint64_t>(context.device_ordinal()));
  }

  const auto base_extents = buffers.base_logits.extents();
  const auto w1_extents = buffers.markov_w1.extents();
  const auto w2_extents = buffers.markov_w2.extents();
  const DsparkProposalShape shape{static_cast<uint64_t>(base_extents[0]),
                                  static_cast<uint64_t>(base_extents[1]),
                                  static_cast<uint64_t>(base_extents[2]),
                                  static_cast<uint64_t>(w1_extents[1]),
                                  kDsparkProductionNumVerifyTokens};
  const NativeRuntimeError shape_status = validate_dspark_proposal_shape(shape);
  if (!is_ok(shape_status)) {
    return shape_status;
  }
  if (w1_extents[0] != static_cast<int64_t>(shape.vocab_size)) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      DsparkProposalArgument::kMarkovW1, 0,
                      static_cast<uint64_t>(w1_extents[0]), shape.vocab_size);
  }
  if (w2_extents[0] != static_cast<int64_t>(shape.vocab_size) ||
      w2_extents[1] != static_cast<int64_t>(shape.markov_rank)) {
    return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                      DsparkProposalArgument::kMarkovW2);
  }

  uint64_t logits_elements = 0;
  uint64_t logits_bytes = 0;
  uint64_t markov_elements = 0;
  uint64_t markov_bytes = 0;
  if (!checked_multiply(shape.gamma, shape.vocab_size, &logits_elements) ||
      !checked_multiply(logits_elements, sizeof(uint16_t), &logits_bytes) ||
      !checked_multiply(shape.vocab_size, shape.markov_rank,
                        &markov_elements) ||
      !checked_multiply(markov_elements, sizeof(uint16_t), &markov_bytes)) {
    return make_error(NativeRuntimeCode::kArithmeticOverflow, kOperation);
  }

  std::array<ByteRange, 11> ranges{};
  ByteRange range{};
  uint32_t index = 0;
  NativeRuntimeError status = validate_tensor(
      buffers.base_logits, context, DsparkProposalArgument::kBaseLogits,
      std::array<int64_t, 3>{1, kDsparkProductionGamma,
                             kDsparkProductionVocabSize},
      logits_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.markov_w1, context,
                           DsparkProposalArgument::kMarkovW1,
                           std::array<int64_t, 2>{kDsparkProductionVocabSize,
                                                  kDsparkProductionMarkovRank},
                           markov_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.markov_w2, context,
                           DsparkProposalArgument::kMarkovW2,
                           std::array<int64_t, 2>{kDsparkProductionVocabSize,
                                                  kDsparkProductionMarkovRank},
                           markov_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.anchor_tokens, context,
                           DsparkProposalArgument::kAnchorTokens,
                           std::array<int64_t, 1>{1}, sizeof(int64_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.temperatures, context,
                           DsparkProposalArgument::kTemperatures,
                           std::array<int64_t, 1>{1}, sizeof(float), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.greedy_mask, context,
                           DsparkProposalArgument::kGreedyMask,
                           std::array<int64_t, 1>{1}, sizeof(uint8_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.rng_state, context,
                           DsparkProposalArgument::kRngState,
                           std::array<int64_t, 1>{kDsparkProposalRngStateWords},
                           sizeof(DsparkProposalRngStateV1), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.proposal_tokens, context,
                           DsparkProposalArgument::kProposalTokens,
                           std::array<int64_t, 2>{1, kDsparkProductionGamma},
                           kDsparkProductionGamma * sizeof(int64_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.corrected_logits, context,
                           DsparkProposalArgument::kCorrectedLogits,
                           std::array<int64_t, 3>{1, kDsparkProductionGamma,
                                                  kDsparkProductionVocabSize},
                           logits_bytes, &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.log_normalizers, context,
                           DsparkProposalArgument::kLogNormalizers,
                           std::array<int64_t, 2>{1, kDsparkProductionGamma},
                           kDsparkProductionGamma * sizeof(float), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;
  status = validate_tensor(buffers.device_status, context,
                           DsparkProposalArgument::kDeviceStatus,
                           std::array<int64_t, 1>{1}, sizeof(uint32_t), &range);
  if (!is_ok(status)) {
    return status;
  }
  ranges[index++] = range;

  for (uint32_t left = 0; left < index; ++left) {
    for (uint32_t right = left + 1; right < index; ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return make_error(NativeRuntimeCode::kInvalidArgument, kOperation,
                          ranges[left].argument, 0,
                          static_cast<uint64_t>(ranges[right].argument), 0);
      }
    }
  }
  return native_runtime_ok();
}

__device__ __forceinline__ PhiloxBlock philox4x32_10(uint64_t counter,
                                                     uint64_t subsequence,
                                                     uint64_t seed) {
  constexpr uint32_t kMultiplier0 = 0xd2511f53U;
  constexpr uint32_t kMultiplier1 = 0xcd9e8d57U;
  constexpr uint32_t kWeyl0 = 0x9e3779b9U;
  constexpr uint32_t kWeyl1 = 0xbb67ae85U;

  PhiloxBlock block{static_cast<uint32_t>(counter),
                    static_cast<uint32_t>(counter >> 32U),
                    static_cast<uint32_t>(subsequence),
                    static_cast<uint32_t>(subsequence >> 32U)};
  uint32_t key0 = static_cast<uint32_t>(seed);
  uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
  for (uint32_t round = 0; round < 10U; ++round) {
    const uint32_t high0 = __umulhi(kMultiplier0, block.x);
    const uint32_t low0 = kMultiplier0 * block.x;
    const uint32_t high1 = __umulhi(kMultiplier1, block.z);
    const uint32_t low1 = kMultiplier1 * block.z;
    block =
        PhiloxBlock{high1 ^ block.y ^ key0, low1, high0 ^ block.w ^ key1, low0};
    key0 += kWeyl0;
    key1 += kWeyl1;
  }
  return block;
}

__device__ __forceinline__ uint32_t philox_lane(const PhiloxBlock &block,
                                                uint32_t lane) {
  switch (lane) {
  case 0:
    return block.x;
  case 1:
    return block.y;
  case 2:
    return block.z;
  default:
    return block.w;
  }
}

__device__ __forceinline__ float open_closed_uniform(uint32_t value) {
  return (static_cast<float>(value >> 8U) + 1.0F) * 0x1p-24F;
}

__device__ __forceinline__ float exponential_noise(uint64_t first_counter,
                                                   uint64_t subsequence,
                                                   uint64_t seed, uint32_t row,
                                                   uint32_t token) {
  const uint64_t block =
      first_counter + static_cast<uint64_t>(row) * kBlocksPerRow + token / 4U;
  const PhiloxBlock values = philox4x32_10(block, subsequence, seed);
  return -__logf(open_closed_uniform(philox_lane(values, token % 4U)));
}

template <typename Scalar> struct ScalarConvert;

template <> struct ScalarConvert<__nv_bfloat16> final {
  __device__ static float to_float(__nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ static __nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <> struct ScalarConvert<__half> final {
  __device__ static float to_float(__half value) { return __half2float(value); }
  __device__ static __half from_float(float value) {
    return __float2half_rn(value);
  }
};

struct Candidate final {
  float key;
  uint32_t token;
};

__device__ __forceinline__ Candidate choose_candidate(Candidate left,
                                                      Candidate right) {
  if (right.key > left.key ||
      (right.key == left.key && right.token < left.token)) {
    return right;
  }
  return left;
}

template <typename Scalar>
__global__ void
dspark_proposal_kernel(const Scalar *base_logits, const Scalar *markov_w1,
                       const Scalar *markov_w2, const int64_t *anchor_tokens,
                       const float *temperatures, const uint8_t *greedy_mask,
                       uint64_t *rng_state, int64_t *proposal_tokens,
                       Scalar *corrected_logits, float *log_normalizers,
                       uint32_t *device_status, bool require_ready_status) {
  __shared__ ProposalSharedState state;
  __shared__ float embedding[kDsparkProductionMarkovRank];
  __shared__ float thread_max[kThreads];
  __shared__ float thread_sum[kThreads];
  __shared__ Candidate thread_best[kThreads];
  __shared__ int64_t proposal_results[kDsparkProductionGamma];
  __shared__ float normalizer_results[kDsparkProductionGamma];

  const uint32_t thread = threadIdx.x;
  if (thread == 0U) {
    state.valid = 0U;
    if (require_ready_status && device_status[0] != 0U) {
      return;
    }
    DsparkProposalDeviceCode code = DsparkProposalDeviceCode::kOk;
    const int64_t anchor = anchor_tokens[0];
    const float temperature = temperatures[0];
    if (rng_state[0] != kDsparkProposalRngStateDescriptorV1) {
      code = DsparkProposalDeviceCode::kInvalidRngStateDescriptor;
    } else if (rng_state[3] > 0xffffffffffffffffULL - kBlocksPerReplay) {
      code = DsparkProposalDeviceCode::kRngCounterOverflow;
    } else if (anchor < 0 || anchor >= kDsparkProductionVocabSize) {
      code = DsparkProposalDeviceCode::kAnchorTokenOutOfRange;
    } else if (!(temperature > 0.0F)) {
      code = DsparkProposalDeviceCode::kNonPositiveTemperature;
    }
    device_status[0] = static_cast<uint32_t>(code);
    if (code == DsparkProposalDeviceCode::kOk) {
      state.seed = rng_state[1];
      state.subsequence = rng_state[2];
      state.first_counter = rng_state[3];
      state.previous_token = anchor;
      state.valid = 1U;
    }
  }
  __syncthreads();
  if (state.valid == 0U) {
    return;
  }

  const float temperature = temperatures[0];
  const bool greedy = greedy_mask[0] != 0U;
  for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
    if (thread < kDsparkProductionMarkovRank) {
      const uint64_t w1_offset = static_cast<uint64_t>(state.previous_token) *
                                     kDsparkProductionMarkovRank +
                                 thread;
      embedding[thread] = ScalarConvert<Scalar>::to_float(markov_w1[w1_offset]);
    }
    __syncthreads();

    float local_max = kNegativeInfinity;
    for (uint32_t local = 0; local < kTokensPerThread; ++local) {
      const uint32_t token = thread + local * kThreads;
      if (token >= kDsparkProductionVocabSize) {
        break;
      }
      float bias = 0.0F;
      const uint64_t w2_base =
          static_cast<uint64_t>(token) * kDsparkProductionMarkovRank;
      for (uint32_t rank = 0; rank < kDsparkProductionMarkovRank; ++rank) {
        bias = fmaf(ScalarConvert<Scalar>::to_float(markov_w2[w2_base + rank]),
                    embedding[rank], bias);
      }
      const uint64_t offset =
          static_cast<uint64_t>(row) * kDsparkProductionVocabSize + token;
      const Scalar corrected_storage = ScalarConvert<Scalar>::from_float(
          ScalarConvert<Scalar>::to_float(base_logits[offset]) + bias);
      corrected_logits[offset] = corrected_storage;
      const float scaled =
          ScalarConvert<Scalar>::to_float(corrected_storage) / temperature;
      if (scaled == scaled) {
        local_max = fmaxf(local_max, scaled);
      }
    }
    thread_max[thread] = local_max;
    __syncthreads();
    if (thread == 0U) {
      float row_max = kNegativeInfinity;
      for (uint32_t index = 0; index < kThreads; ++index) {
        row_max = fmaxf(row_max, thread_max[index]);
      }
      state.row_max = row_max;
      if (!isfinite(row_max)) {
        // Match the established DSpark fallback: an entirely non-finite row
        // emits token zero rather than an out-of-vocabulary sentinel.
        state.sampled_token = 0;
        state.previous_token = 0;
        proposal_results[row] = 0;
        normalizer_results[row] = kNegativeInfinity;
      }
    }
    __syncthreads();
    if (!isfinite(state.row_max)) {
      continue;
    }

    float local_sum = 0.0F;
    Candidate local_best{-1.0F, kDsparkProductionVocabSize};
    for (uint32_t local = 0; local < kTokensPerThread; ++local) {
      const uint32_t token = thread + local * kThreads;
      if (token >= kDsparkProductionVocabSize) {
        break;
      }
      const uint64_t offset =
          static_cast<uint64_t>(row) * kDsparkProductionVocabSize + token;
      const float corrected =
          ScalarConvert<Scalar>::to_float(corrected_logits[offset]);
      const float scaled = corrected / temperature;
      const float mass =
          scaled == scaled ? __expf(scaled - state.row_max) : 0.0F;
      local_sum += mass;
      const float noise = exponential_noise(
          state.first_counter, state.subsequence, state.seed, row, token);
      const Candidate candidate{mass / noise, token};
      local_best = greedy
                       ? choose_candidate(local_best, Candidate{scaled, token})
                       : choose_candidate(local_best, candidate);
    }
    thread_sum[thread] = local_sum;
    thread_best[thread] = local_best;
    __syncthreads();
    if (thread == 0U) {
      float sum = 0.0F;
      Candidate best{-1.0F, kDsparkProductionVocabSize};
      for (uint32_t index = 0; index < kThreads; ++index) {
        sum += thread_sum[index];
        best = choose_candidate(best, thread_best[index]);
      }
      state.sum_exp = sum;
      state.sampled_token = static_cast<int64_t>(best.token);
      if (!(sum > 0.0F) || best.token >= kDsparkProductionVocabSize) {
        state.sampled_token = 0;
        state.previous_token = 0;
        proposal_results[row] = 0;
        normalizer_results[row] = kNegativeInfinity;
      } else {
        state.previous_token = state.sampled_token;
        proposal_results[row] = state.sampled_token;
        normalizer_results[row] = state.row_max + __logf(sum);
      }
    }
    __syncthreads();
    if (state.valid == 0U) {
      return;
    }
  }

  if (thread == 0U) {
    rng_state[3] += kBlocksPerReplay;
    for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
      proposal_tokens[row] = proposal_results[row];
      log_normalizers[row] = normalizer_results[row];
    }
  }
}

template <DType LogitsDType, typename Scalar>
[[nodiscard]] NativeRuntimeError
launch_impl(const CudaExecutionContext &context,
            const DsparkProposalBuffers<LogitsDType> &buffers,
            bool require_ready_status) noexcept {
  const NativeRuntimeError layout_status = validate_layout(context, buffers);
  if (!is_ok(layout_status)) {
    return layout_status;
  }
  dspark_proposal_kernel<Scalar><<<1, kThreads, 0, context.stream()>>>(
      reinterpret_cast<const Scalar *>(buffers.base_logits.data_bytes()),
      reinterpret_cast<const Scalar *>(buffers.markov_w1.data_bytes()),
      reinterpret_cast<const Scalar *>(buffers.markov_w2.data_bytes()),
      reinterpret_cast<const int64_t *>(buffers.anchor_tokens.data_bytes()),
      reinterpret_cast<const float *>(buffers.temperatures.data_bytes()),
      reinterpret_cast<const uint8_t *>(buffers.greedy_mask.data_bytes()),
      reinterpret_cast<uint64_t *>(buffers.rng_state.data_bytes()),
      reinterpret_cast<int64_t *>(buffers.proposal_tokens.data_bytes()),
      reinterpret_cast<Scalar *>(buffers.corrected_logits.data_bytes()),
      reinterpret_cast<float *>(buffers.log_normalizers.data_bytes()),
      reinterpret_cast<uint32_t *>(buffers.device_status.data_bytes()),
      require_ready_status);
  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    return make_error(NativeRuntimeCode::kCudaRuntimeFailure,
                      NativeRuntimeOperation::kLaunchDsparkProposal,
                      DsparkProposalArgument::kNone,
                      static_cast<int32_t>(launch_status));
  }
  return native_runtime_ok();
}

template <DType LogitsDType> struct ProposalCapturePayload final {
  const CudaExecutionContext *context;
  const DsparkProposalBuffers<LogitsDType> *buffers;
};

template <DType LogitsDType>
[[nodiscard]] NativeRuntimeError capture_proposal_body(void *opaque) noexcept {
  const auto *payload =
      static_cast<const ProposalCapturePayload<LogitsDType> *>(opaque);
  if constexpr (LogitsDType == DType::kBFloat16) {
    return launch_dspark_proposal_bfloat16_if_ready(*payload->context,
                                                    *payload->buffers);
  } else {
    return launch_dspark_proposal_float16_if_ready(*payload->context,
                                                   *payload->buffers);
  }
}

template <DType LogitsDType>
[[nodiscard]] NativeRuntimeResult<CudaCapturedGraph>
capture_impl(const CudaExecutionContext &context, const GraphArenaLease &arena,
             const DsparkProposalBuffers<LogitsDType> &buffers) noexcept {
  using GraphResult = NativeRuntimeResult<CudaCapturedGraph>;
  if (!context.valid() || !arena.valid()) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kInvalidArgument,
                   NativeRuntimeOperation::kGraphCaptureBegin));
  }
  if (context.device_ordinal() != arena.device_ordinal()) {
    return GraphResult::failure(
        make_error(NativeRuntimeCode::kDeviceMismatch,
                   NativeRuntimeOperation::kGraphCaptureBegin,
                   DsparkProposalArgument::kNone, 0,
                   static_cast<uint64_t>(context.device_ordinal()),
                   static_cast<uint64_t>(arena.device_ordinal())));
  }
  const std::array<bool, 11> owned{buffers.base_logits.belongs_to(arena),
                                   buffers.markov_w1.belongs_to(arena),
                                   buffers.markov_w2.belongs_to(arena),
                                   buffers.anchor_tokens.belongs_to(arena),
                                   buffers.temperatures.belongs_to(arena),
                                   buffers.greedy_mask.belongs_to(arena),
                                   buffers.rng_state.belongs_to(arena),
                                   buffers.proposal_tokens.belongs_to(arena),
                                   buffers.corrected_logits.belongs_to(arena),
                                   buffers.log_normalizers.belongs_to(arena),
                                   buffers.device_status.belongs_to(arena)};
  constexpr std::array<DsparkProposalArgument, 11> arguments{
      DsparkProposalArgument::kBaseLogits,
      DsparkProposalArgument::kMarkovW1,
      DsparkProposalArgument::kMarkovW2,
      DsparkProposalArgument::kAnchorTokens,
      DsparkProposalArgument::kTemperatures,
      DsparkProposalArgument::kGreedyMask,
      DsparkProposalArgument::kRngState,
      DsparkProposalArgument::kProposalTokens,
      DsparkProposalArgument::kCorrectedLogits,
      DsparkProposalArgument::kLogNormalizers,
      DsparkProposalArgument::kDeviceStatus};
  for (uint32_t index = 0; index < owned.size(); ++index) {
    if (!owned[index]) {
      return GraphResult::failure(make_error(
          NativeRuntimeCode::kForeignSlice,
          NativeRuntimeOperation::kGraphCaptureBegin, arguments[index]));
    }
  }

  ProposalCapturePayload<LogitsDType> payload{&context, &buffers};
  return CudaCapturedGraph::capture(
      context, arena, &capture_proposal_body<LogitsDType>, &payload);
}

} // namespace

NativeRuntimeResult<CudaCapturedGraph> capture_dspark_proposal_bfloat16_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept {
  return capture_impl(context, arena, buffers);
}

NativeRuntimeResult<CudaCapturedGraph> capture_dspark_proposal_float16_graph(
    const CudaExecutionContext &context, const GraphArenaLease &arena,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept {
  return capture_impl(context, arena, buffers);
}

NativeRuntimeError launch_dspark_proposal_bfloat16(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept {
  return launch_impl<DType::kBFloat16, __nv_bfloat16>(context, buffers, false);
}

NativeRuntimeError launch_dspark_proposal_bfloat16_if_ready(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kBFloat16> &buffers) noexcept {
  return launch_impl<DType::kBFloat16, __nv_bfloat16>(context, buffers, true);
}

NativeRuntimeError launch_dspark_proposal_float16(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept {
  return launch_impl<DType::kFloat16, __half>(context, buffers, false);
}

NativeRuntimeError launch_dspark_proposal_float16_if_ready(
    const CudaExecutionContext &context,
    const DsparkProposalBuffers<DType::kFloat16> &buffers) noexcept {
  return launch_impl<DType::kFloat16, __half>(context, buffers, true);
}

} // namespace sglang::native
