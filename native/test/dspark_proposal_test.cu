#include "sglang/native/dspark_proposal.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using sglang::native::capture_dspark_proposal_bfloat16_graph;
using sglang::native::capture_dspark_proposal_float16_graph;
using sglang::native::CudaCapturedGraph;
using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaStream;
using sglang::native::DsparkConstBFloat16Matrix;
using sglang::native::DsparkConstBFloat16Tensor3;
using sglang::native::DsparkConstBool8Vector;
using sglang::native::DsparkConstFloat16Matrix;
using sglang::native::DsparkConstFloat16Tensor3;
using sglang::native::DsparkConstFloat32Vector;
using sglang::native::DsparkConstInt64Vector;
using sglang::native::DsparkMutableBFloat16Tensor3;
using sglang::native::DsparkMutableFloat16Tensor3;
using sglang::native::DsparkMutableFloat32Matrix;
using sglang::native::DsparkMutableInt64Matrix;
using sglang::native::DsparkMutableUInt32Vector;
using sglang::native::DsparkMutableUInt64Vector;
using sglang::native::DsparkProposalArgument;
using sglang::native::DsparkProposalBuffers;
using sglang::native::DsparkProposalDeviceCode;
using sglang::native::DsparkProposalRngStateV1;
using sglang::native::DType;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::is_ok;
using sglang::native::kDsparkProductionGamma;
using sglang::native::kDsparkProductionMarkovRank;
using sglang::native::kDsparkProductionVocabSize;
using sglang::native::kDsparkProposalRngStateDescriptorV1;
using sglang::native::launch_dspark_proposal_bfloat16;
using sglang::native::launch_dspark_proposal_bfloat16_if_ready;
using sglang::native::launch_dspark_proposal_float16;
using sglang::native::launch_dspark_proposal_float16_if_ready;
using sglang::native::make_dspark_proposal_rng_state_v1;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::NativeRuntimeResult;
using sglang::native::TensorAccess;

constexpr uint64_t kBlocksPerReplay = 434560ULL;
constexpr uint64_t kLogitsElements =
    static_cast<uint64_t>(kDsparkProductionGamma) * kDsparkProductionVocabSize;
constexpr uint64_t kMarkovElements =
    static_cast<uint64_t>(kDsparkProductionVocabSize) *
    kDsparkProductionMarkovRank;
constexpr uint16_t kOutputSentinelBits = 0x7e35U;
constexpr int64_t kTokenSentinel = -777;
constexpr float kNormalizerSentinel = 12345.0F;
constexpr uint32_t kStatusSentinel = 0xf00dba5eU;
constexpr float kMaxTrainedOracleCorrectedAbsError = 0.0625F;
constexpr float kMaxNativeNormalizerReproductionError = 2e-3F;
constexpr float kMaxPythonNormalizerParityError = 4e-3F;
constexpr std::array<char, 8> kTrainedOracleMagic{'S', 'G', 'D', 'S',
                                                  'O', 'R', '1', '\0'};

struct TrainedOracleV1 final {
  std::array<char, 8> magic;
  uint32_t version;
  uint32_t gamma;
  uint32_t vocab_size;
  uint32_t markov_rank;
  uint64_t seed;
  uint64_t subsequence;
  uint64_t counter;
  uint64_t final_counter;
  int64_t anchor_token;
  float temperature;
  uint32_t reserved;
  std::array<int64_t, kDsparkProductionGamma> tokens;
  std::array<float, kDsparkProductionGamma> log_normalizers;
  uint32_t trailing_padding;
};

static_assert(sizeof(TrainedOracleV1) == 160);
static_assert(std::is_standard_layout_v<TrainedOracleV1>);
static_assert(std::is_trivially_copyable_v<TrainedOracleV1>);
static_assert(offsetof(TrainedOracleV1, magic) == 0);
static_assert(offsetof(TrainedOracleV1, version) == 8);
static_assert(offsetof(TrainedOracleV1, gamma) == 12);
static_assert(offsetof(TrainedOracleV1, vocab_size) == 16);
static_assert(offsetof(TrainedOracleV1, markov_rank) == 20);
static_assert(offsetof(TrainedOracleV1, seed) == 24);
static_assert(offsetof(TrainedOracleV1, subsequence) == 32);
static_assert(offsetof(TrainedOracleV1, counter) == 40);
static_assert(offsetof(TrainedOracleV1, final_counter) == 48);
static_assert(offsetof(TrainedOracleV1, anchor_token) == 56);
static_assert(offsetof(TrainedOracleV1, temperature) == 64);
static_assert(offsetof(TrainedOracleV1, reserved) == 68);
static_assert(offsetof(TrainedOracleV1, tokens) == 72);
static_assert(offsetof(TrainedOracleV1, log_normalizers) == 128);
static_assert(offsetof(TrainedOracleV1, trailing_padding) == 156);

struct PhiloxBlock final {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;
};

[[nodiscard]] constexpr PhiloxBlock
reference_philox4x32_10(uint64_t counter, uint64_t subsequence,
                        uint64_t seed) noexcept {
  constexpr uint64_t kMultiplier0 = 0xd2511f53ULL;
  constexpr uint64_t kMultiplier1 = 0xcd9e8d57ULL;
  constexpr uint32_t kWeyl0 = 0x9e3779b9U;
  constexpr uint32_t kWeyl1 = 0xbb67ae85U;
  PhiloxBlock block{static_cast<uint32_t>(counter),
                    static_cast<uint32_t>(counter >> 32U),
                    static_cast<uint32_t>(subsequence),
                    static_cast<uint32_t>(subsequence >> 32U)};
  uint32_t key0 = static_cast<uint32_t>(seed);
  uint32_t key1 = static_cast<uint32_t>(seed >> 32U);
  for (uint32_t round = 0; round < 10U; ++round) {
    const uint64_t product0 = kMultiplier0 * block.x;
    const uint64_t product1 = kMultiplier1 * block.z;
    block = PhiloxBlock{static_cast<uint32_t>(product1 >> 32U) ^ block.y ^ key0,
                        static_cast<uint32_t>(product1),
                        static_cast<uint32_t>(product0 >> 32U) ^ block.w ^ key1,
                        static_cast<uint32_t>(product0)};
    key0 += kWeyl0;
    key1 += kWeyl1;
  }
  return block;
}

[[nodiscard]] float
reference_exponential_noise(const DsparkProposalRngStateV1 &state, uint32_t row,
                            uint32_t token) noexcept {
  constexpr uint64_t kBlocksPerRow = (kDsparkProductionVocabSize + 3ULL) / 4ULL;
  const PhiloxBlock block = reference_philox4x32_10(
      state.counter + static_cast<uint64_t>(row) * kBlocksPerRow + token / 4U,
      state.subsequence, state.seed);
  const uint32_t words[4]{block.x, block.y, block.z, block.w};
  const float uniform =
      (static_cast<float>(words[token % 4U] >> 8U) + 1.0F) * 0x1p-24F;
  return -std::log(uniform);
}

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
  if (is_ok(status)) {
    return true;
  }
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
        return check_status(error, "NativeRuntimeResult", __LINE__);
      });
}

template <typename T>
[[nodiscard]] bool read_exact_file(const char *path,
                                   std::span<T> destination) noexcept {
  if (path == nullptr) {
    return false;
  }
  std::FILE *file = nullptr;
  if (fopen_s(&file, path, "rb") != 0 || file == nullptr) {
    return false;
  }
  const std::size_t bytes = destination.size_bytes();
  const std::size_t read = std::fread(destination.data(), 1, bytes, file);
  const int trailing = std::fgetc(file);
  const bool closed = std::fclose(file) == 0;
  return read == bytes && trailing == EOF && closed;
}

[[nodiscard]] SglNativeTensorMetadataV1 metadata_1d(SglNativeDType dtype,
                                                    int64_t first) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 1;
  metadata.extents[0] = first;
  metadata.strides[0] = 1;
  return metadata;
}

[[nodiscard]] SglNativeTensorMetadataV1
metadata_2d(SglNativeDType dtype, int64_t first, int64_t second) noexcept {
  auto metadata = make_tensor_metadata_v1();
  metadata.dtype = dtype;
  metadata.rank = 2;
  metadata.extents[0] = first;
  metadata.extents[1] = second;
  metadata.strides[0] = second;
  metadata.strides[1] = 1;
  return metadata;
}

[[nodiscard]] SglNativeTensorMetadataV1 metadata_3d(SglNativeDType dtype,
                                                    int64_t first,
                                                    int64_t second,
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

struct RawGraph final {
  cudaGraph_t value = nullptr;

  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

template <DType LogitsDType> struct ProposalTypes;

template <> struct ProposalTypes<DType::kBFloat16> final {
  using ConstTensor3 = DsparkConstBFloat16Tensor3;
  using ConstMatrix = DsparkConstBFloat16Matrix;
  using MutableTensor3 = DsparkMutableBFloat16Tensor3;
  static constexpr SglNativeDType kRawDtype = SGL_NATIVE_DTYPE_BFLOAT16;
};

template <> struct ProposalTypes<DType::kFloat16> final {
  using ConstTensor3 = DsparkConstFloat16Tensor3;
  using ConstMatrix = DsparkConstFloat16Matrix;
  using MutableTensor3 = DsparkMutableFloat16Tensor3;
  static constexpr SglNativeDType kRawDtype = SGL_NATIVE_DTYPE_FLOAT16;
};

template <DType LogitsDType> class ProposalFixture final {
public:
  using Types = ProposalTypes<LogitsDType>;
  using ConstTensor3 = typename Types::ConstTensor3;
  using ConstMatrix = typename Types::ConstMatrix;
  using MutableTensor3 = typename Types::MutableTensor3;

  ProposalFixture(const ProposalFixture &) = delete;
  ProposalFixture &operator=(const ProposalFixture &) = delete;
  ProposalFixture(ProposalFixture &&) = delete;
  ProposalFixture &operator=(ProposalFixture &&) = delete;
  ProposalFixture() = default;

  [[nodiscard]] bool initialize() noexcept {
    constexpr uint64_t kLogitsBytes = kLogitsElements * sizeof(uint16_t);
    constexpr uint64_t kMarkovBytes = kMarkovElements * sizeof(uint16_t);
    constexpr uint64_t kCapacity =
        2ULL * kLogitsBytes + 2ULL * kMarkovBytes +
        sizeof(DsparkProposalRngStateV1) + sizeof(int64_t) + sizeof(float) +
        sizeof(uint8_t) + kDsparkProductionGamma * sizeof(int64_t) +
        kDsparkProductionGamma * sizeof(float) + sizeof(uint32_t) +
        13ULL * 256ULL;

    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(GraphMemoryArena::allocate(*context_, kCapacity),
                     &arena_) ||
        !reserve(kLogitsBytes, &base_logits_slice_) ||
        !reserve(kMarkovBytes, &markov_w1_slice_) ||
        !reserve(kMarkovBytes, &markov_w2_slice_) ||
        !reserve(sizeof(int64_t), &anchor_slice_) ||
        !reserve(sizeof(float), &temperature_slice_) ||
        !reserve(sizeof(uint8_t), &greedy_slice_) ||
        !reserve(sizeof(DsparkProposalRngStateV1), &rng_state_slice_) ||
        !reserve(kDsparkProductionGamma * sizeof(int64_t),
                 &proposal_tokens_slice_) ||
        !reserve(kLogitsBytes, &corrected_logits_slice_) ||
        !reserve(kDsparkProductionGamma * sizeof(float),
                 &log_normalizers_slice_) ||
        !reserve(sizeof(uint32_t), &device_status_slice_)) {
      return false;
    }
    if (!check_status(arena_->seal(), "arena_->seal()", __LINE__) ||
        !take_result(arena_->acquire_lease(), &lease_)) {
      return false;
    }

    if (!take_result(
            lease_->bind_const<LogitsDType, 3>(
                *base_logits_slice_,
                metadata_3d(Types::kRawDtype, 1, kDsparkProductionGamma,
                            kDsparkProductionVocabSize)),
            &base_logits_) ||
        !take_result(
            lease_->bind_const<LogitsDType, 2>(
                *markov_w1_slice_,
                metadata_2d(Types::kRawDtype, kDsparkProductionVocabSize,
                            kDsparkProductionMarkovRank)),
            &markov_w1_) ||
        !take_result(
            lease_->bind_const<LogitsDType, 2>(
                *markov_w2_slice_,
                metadata_2d(Types::kRawDtype, kDsparkProductionVocabSize,
                            kDsparkProductionMarkovRank)),
            &markov_w2_) ||
        !take_result(
            lease_->bind_const<DType::kInt64, 1>(
                *anchor_slice_, metadata_1d(SGL_NATIVE_DTYPE_INT64, 1)),
            &anchor_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 1>(
                *temperature_slice_, metadata_1d(SGL_NATIVE_DTYPE_FLOAT32, 1)),
            &temperature_) ||
        !take_result(
            lease_->bind_const<DType::kBool8, 1>(
                *greedy_slice_, metadata_1d(SGL_NATIVE_DTYPE_BOOL8, 1)),
            &greedy_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt64, 1>(
                *rng_state_slice_, metadata_1d(SGL_NATIVE_DTYPE_UINT64, 4)),
            &rng_state_) ||
        !take_result(
            lease_->bind_mutable<DType::kInt64, 2>(
                *proposal_tokens_slice_,
                metadata_2d(SGL_NATIVE_DTYPE_INT64, 1, kDsparkProductionGamma)),
            &proposal_tokens_) ||
        !take_result(
            lease_->bind_mutable<LogitsDType, 3>(
                *corrected_logits_slice_,
                metadata_3d(Types::kRawDtype, 1, kDsparkProductionGamma,
                            kDsparkProductionVocabSize)),
            &corrected_logits_) ||
        !take_result(lease_->bind_mutable<DType::kFloat32, 2>(
                         *log_normalizers_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_FLOAT32, 1,
                                     kDsparkProductionGamma)),
                     &log_normalizers_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt32, 1>(
                *device_status_slice_, metadata_1d(SGL_NATIVE_DTYPE_UINT32, 1)),
            &device_status_)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] DsparkProposalBuffers<LogitsDType> buffers() const noexcept {
    return DsparkProposalBuffers<LogitsDType>{
        *base_logits_,     *markov_w1_,       *markov_w2_,
        *anchor_,          *temperature_,     *greedy_,
        *rng_state_,       *proposal_tokens_, *corrected_logits_,
        *log_normalizers_, *device_status_};
  }

  [[nodiscard]] bool prepare_zero_weights() noexcept {
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(base_logits_->data_bytes()),
                          0, base_logits_->allocation_bytes()));
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(markov_w1_->data_bytes()), 0,
                          markov_w1_->allocation_bytes()));
    CHECK_CUDA(cudaMemset(const_cast<std::byte *>(markov_w2_->data_bytes()), 0,
                          markov_w2_->allocation_bytes()));
    return true;
  }

  [[nodiscard]] bool prepare_chain_weights() noexcept {
    CHECK(prepare_zero_weights());
    for (uint32_t token = 0; token <= kDsparkProductionGamma; ++token) {
      CHECK(set_w1(token, token, 1.0F));
      CHECK(set_w2(token + 1U, token, 8.0F));
    }
    return true;
  }

  [[nodiscard]] bool set_base_logit(uint32_t row, uint32_t token,
                                    float value) noexcept {
    return set_scalar(
        *base_logits_,
        static_cast<uint64_t>(row) * kDsparkProductionVocabSize + token, value);
  }

  [[nodiscard]] bool reset(int64_t anchor, float temperature, bool greedy,
                           DsparkProposalRngStateV1 state) noexcept {
    const uint8_t greedy_value = greedy ? 1U : 0U;
    const std::array<uint64_t, 4> state_words{state.descriptor, state.seed,
                                              state.subsequence, state.counter};
    const std::array<int64_t, kDsparkProductionGamma> tokens = [] {
      std::array<int64_t, kDsparkProductionGamma> values{};
      values.fill(kTokenSentinel);
      return values;
    }();
    const std::array<float, kDsparkProductionGamma> normalizers = [] {
      std::array<float, kDsparkProductionGamma> values{};
      values.fill(kNormalizerSentinel);
      return values;
    }();
    const uint32_t sentinel_word =
        static_cast<uint32_t>(kOutputSentinelBits) |
        (static_cast<uint32_t>(kOutputSentinelBits) << 16U);
    return copy_to_const(*anchor_, std::span<const int64_t>(&anchor, 1)) &&
           copy_to_const(*temperature_,
                         std::span<const float>(&temperature, 1)) &&
           copy_to_const(*greedy_,
                         std::span<const uint8_t>(&greedy_value, 1)) &&
           copy_to_mutable(*rng_state_,
                           std::span<const uint64_t>(state_words)) &&
           copy_to_mutable(*proposal_tokens_,
                           std::span<const int64_t>(tokens)) &&
           copy_to_mutable(*log_normalizers_,
                           std::span<const float>(normalizers)) &&
           copy_to_mutable(*device_status_,
                           std::span<const uint32_t>(&kStatusSentinel, 1)) &&
           cudaMemset(corrected_logits_->data_bytes(),
                      static_cast<int>(sentinel_word & 0xffU),
                      corrected_logits_->allocation_bytes()) == cudaSuccess;
  }

  [[nodiscard]] bool launch() noexcept {
    if constexpr (LogitsDType == DType::kBFloat16) {
      return check_status(launch_dspark_proposal_bfloat16(context(), buffers()),
                          "launch_dspark_proposal_bfloat16", __LINE__);
    } else {
      return check_status(launch_dspark_proposal_float16(context(), buffers()),
                          "launch_dspark_proposal_float16", __LINE__);
    }
  }

  [[nodiscard]] bool launch_if_ready() noexcept {
    if constexpr (LogitsDType == DType::kBFloat16) {
      return check_status(
          launch_dspark_proposal_bfloat16_if_ready(context(), buffers()),
          "launch_dspark_proposal_bfloat16_if_ready", __LINE__);
    } else {
      return check_status(
          launch_dspark_proposal_float16_if_ready(context(), buffers()),
          "launch_dspark_proposal_float16_if_ready", __LINE__);
    }
  }

  [[nodiscard]] bool write_status(uint32_t status) noexcept {
    return copy_to_mutable(*device_status_,
                           std::span<const uint32_t>(&status, 1));
  }

  [[nodiscard]] bool read_tokens(
      std::array<int64_t, kDsparkProductionGamma> *values) const noexcept {
    return values != nullptr &&
           copy_to_host(std::span<int64_t>(*values), *proposal_tokens_);
  }

  [[nodiscard]] bool read_normalizers(
      std::array<float, kDsparkProductionGamma> *values) const noexcept {
    return values != nullptr &&
           copy_to_host(std::span<float>(*values), *log_normalizers_);
  }

  [[nodiscard]] bool
  read_state(DsparkProposalRngStateV1 *state) const noexcept {
    if (state == nullptr) {
      return false;
    }
    std::array<uint64_t, 4> words{};
    if (!copy_to_host(std::span<uint64_t>(words), *rng_state_)) {
      return false;
    }
    *state = DsparkProposalRngStateV1{words[0], words[1], words[2], words[3]};
    return true;
  }

  [[nodiscard]] bool read_status(uint32_t *status) const noexcept {
    return status != nullptr &&
           copy_to_host(std::span<uint32_t>(status, 1), *device_status_);
  }

  [[nodiscard]] bool
  read_corrected_bits(uint32_t row, std::span<const uint32_t> tokens,
                      std::vector<uint16_t> *values) const noexcept {
    if (row >= kDsparkProductionGamma || values == nullptr) {
      return false;
    }
    values->resize(tokens.size());
    for (uint64_t index = 0; index < tokens.size(); ++index) {
      if (tokens[index] >= kDsparkProductionVocabSize) {
        return false;
      }
      const uint64_t offset =
          (static_cast<uint64_t>(row) * kDsparkProductionVocabSize +
           tokens[index]) *
          sizeof(uint16_t);
      const auto *source = corrected_logits_->data_bytes() + offset;
      if (cudaMemcpy(&(*values)[index], source, sizeof(uint16_t),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool
  read_corrected_row(uint32_t row,
                     std::vector<uint16_t> *values) const noexcept {
    if (row >= kDsparkProductionGamma || values == nullptr) {
      return false;
    }
    values->resize(kDsparkProductionVocabSize);
    const auto *source = corrected_logits_->data_bytes() +
                         static_cast<uint64_t>(row) *
                             kDsparkProductionVocabSize * sizeof(uint16_t);
    return cudaMemcpy(values->data(), source,
                      kDsparkProductionVocabSize * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  [[nodiscard]] uintptr_t corrected_address() const noexcept {
    return reinterpret_cast<uintptr_t>(corrected_logits_->data_bytes());
  }

  [[nodiscard]] uintptr_t proposal_address() const noexcept {
    return reinterpret_cast<uintptr_t>(proposal_tokens_->data_bytes());
  }

  [[nodiscard]] const CudaExecutionContext &context() const noexcept {
    return *context_;
  }

  [[nodiscard]] const GraphArenaLease &lease() const noexcept {
    return *lease_;
  }

private:
  [[nodiscard]] bool reserve(uint64_t bytes,
                             std::optional<GraphMemorySlice> *output) noexcept {
    return take_result(arena_->reserve(bytes, 256), output);
  }

  [[nodiscard]] bool set_w1(uint32_t token, uint32_t rank,
                            float value) noexcept {
    return set_scalar(
        *markov_w1_,
        static_cast<uint64_t>(token) * kDsparkProductionMarkovRank + rank,
        value);
  }

  [[nodiscard]] bool set_w2(uint32_t token, uint32_t rank,
                            float value) noexcept {
    return set_scalar(
        *markov_w2_,
        static_cast<uint64_t>(token) * kDsparkProductionMarkovRank + rank,
        value);
  }

  template <typename View>
  [[nodiscard]] bool set_scalar(const View &view, uint64_t element,
                                float value) noexcept {
    uint16_t bits = 0;
    if constexpr (LogitsDType == DType::kBFloat16) {
      bits = __bfloat16_as_ushort(__float2bfloat16(value));
    } else {
      bits = __half_as_ushort(__float2half(value));
    }
    auto *destination =
        const_cast<std::byte *>(view.data_bytes()) + element * sizeof(uint16_t);
    return cudaMemcpy(destination, &bits, sizeof(bits),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

  template <DType D, uint32_t Rank, typename T>
  [[nodiscard]] bool copy_to_const(const sglang::native::GraphStableTensorView<
                                       D, Rank, TensorAccess::kReadOnly> &view,
                                   std::span<const T> values) noexcept {
    return cudaMemcpy(const_cast<std::byte *>(view.data_bytes()), values.data(),
                      values.size_bytes(),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

  template <DType D, uint32_t Rank, typename T>
  [[nodiscard]] bool
  copy_to_mutable(const sglang::native::GraphStableTensorView<
                      D, Rank, TensorAccess::kReadWrite> &view,
                  std::span<const T> values) noexcept {
    return cudaMemcpy(view.data_bytes(), values.data(), values.size_bytes(),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

  template <typename T, DType D, uint32_t Rank, TensorAccess Access>
  [[nodiscard]] bool
  copy_to_host(std::span<T> values,
               const sglang::native::GraphStableTensorView<D, Rank, Access>
                   &view) const noexcept {
    return cudaMemcpy(values.data(), view.data_bytes(), values.size_bytes(),
                      cudaMemcpyDeviceToHost) == cudaSuccess;
  }

  std::optional<CudaStream> stream_;
  std::optional<CudaExecutionContext> context_;
  std::optional<GraphMemoryArena> arena_;
  std::optional<GraphMemorySlice> base_logits_slice_;
  std::optional<GraphMemorySlice> markov_w1_slice_;
  std::optional<GraphMemorySlice> markov_w2_slice_;
  std::optional<GraphMemorySlice> anchor_slice_;
  std::optional<GraphMemorySlice> temperature_slice_;
  std::optional<GraphMemorySlice> greedy_slice_;
  std::optional<GraphMemorySlice> rng_state_slice_;
  std::optional<GraphMemorySlice> proposal_tokens_slice_;
  std::optional<GraphMemorySlice> corrected_logits_slice_;
  std::optional<GraphMemorySlice> log_normalizers_slice_;
  std::optional<GraphMemorySlice> device_status_slice_;
  std::optional<GraphArenaLease> lease_;
  std::optional<ConstTensor3> base_logits_;
  std::optional<ConstMatrix> markov_w1_;
  std::optional<ConstMatrix> markov_w2_;
  std::optional<DsparkConstInt64Vector> anchor_;
  std::optional<DsparkConstFloat32Vector> temperature_;
  std::optional<DsparkConstBool8Vector> greedy_;
  std::optional<DsparkMutableUInt64Vector> rng_state_;
  std::optional<DsparkMutableInt64Matrix> proposal_tokens_;
  std::optional<MutableTensor3> corrected_logits_;
  std::optional<DsparkMutableFloat32Matrix> log_normalizers_;
  std::optional<DsparkMutableUInt32Vector> device_status_;
};

template <DType LogitsDType>
[[nodiscard]] bool GreedyChainPreservesDtypeAndDependency() {
  ProposalFixture<LogitsDType> fixture;
  CHECK(fixture.initialize());
  CHECK(fixture.prepare_chain_weights());
  const DsparkProposalRngStateV1 initial =
      make_dspark_proposal_rng_state_v1(17U, 23U, 29U);
  CHECK(fixture.reset(0, 1.0F, true, initial));
  CHECK(fixture.launch());
  CHECK_STATUS(fixture.context().synchronize());

  std::array<int64_t, kDsparkProductionGamma> tokens{};
  std::array<float, kDsparkProductionGamma> normalizers{};
  DsparkProposalRngStateV1 state{};
  uint32_t status = kStatusSentinel;
  CHECK(fixture.read_tokens(&tokens));
  CHECK(fixture.read_normalizers(&normalizers));
  CHECK(fixture.read_state(&state));
  CHECK(fixture.read_status(&status));
  for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
    CHECK(tokens[row] == static_cast<int64_t>(row + 1U));
    CHECK(std::isfinite(normalizers[row]));
  }
  CHECK(status == 0U);
  CHECK(state.counter == initial.counter + kBlocksPerReplay);

  const std::array<uint32_t, 4> probe_tokens{0U, 1U, 2U,
                                             kDsparkProductionVocabSize - 1U};
  std::vector<uint16_t> bits;
  CHECK(fixture.read_corrected_bits(0, probe_tokens, &bits));
  CHECK(bits.size() == probe_tokens.size());
  CHECK(bits[0] == 0U);
  CHECK(bits[1] != 0U);
  CHECK(bits[2] == 0U);
  CHECK(bits[3] == 0U);
  return true;
}

[[nodiscard]] bool InvalidInputsPreserveOutputsAndCounter() {
  ProposalFixture<DType::kBFloat16> fixture;
  CHECK(fixture.initialize());
  CHECK(fixture.prepare_zero_weights());

  const auto check_failure = [&](int64_t anchor, float temperature,
                                 DsparkProposalRngStateV1 state,
                                 DsparkProposalDeviceCode expected) -> bool {
    if (!fixture.reset(anchor, temperature, false, state) ||
        !fixture.launch() || !is_ok(fixture.context().synchronize())) {
      return false;
    }
    std::array<int64_t, kDsparkProductionGamma> tokens{};
    std::array<float, kDsparkProductionGamma> normalizers{};
    DsparkProposalRngStateV1 after{};
    uint32_t status = 0U;
    if (!fixture.read_tokens(&tokens) ||
        !fixture.read_normalizers(&normalizers) ||
        !fixture.read_state(&after) || !fixture.read_status(&status)) {
      return false;
    }
    return std::all_of(tokens.begin(), tokens.end(),
                       [](int64_t value) { return value == kTokenSentinel; }) &&
           std::all_of(
               normalizers.begin(), normalizers.end(),
               [](float value) { return value == kNormalizerSentinel; }) &&
           after.descriptor == state.descriptor && after.seed == state.seed &&
           after.subsequence == state.subsequence &&
           after.counter == state.counter &&
           status == static_cast<uint32_t>(expected);
  };

  auto invalid_descriptor = make_dspark_proposal_rng_state_v1(1U, 2U, 3U);
  invalid_descriptor.descriptor = 0U;
  CHECK(check_failure(0, 1.0F, invalid_descriptor,
                      DsparkProposalDeviceCode::kInvalidRngStateDescriptor));
  CHECK(check_failure(
      0, 1.0F,
      make_dspark_proposal_rng_state_v1(
          1U, 2U, std::numeric_limits<uint64_t>::max() - kBlocksPerReplay + 1U),
      DsparkProposalDeviceCode::kRngCounterOverflow));
  CHECK(check_failure(-1, 1.0F, make_dspark_proposal_rng_state_v1(1U, 2U, 3U),
                      DsparkProposalDeviceCode::kAnchorTokenOutOfRange));
  CHECK(check_failure(kDsparkProductionVocabSize, 1.0F,
                      make_dspark_proposal_rng_state_v1(1U, 2U, 3U),
                      DsparkProposalDeviceCode::kAnchorTokenOutOfRange));
  CHECK(check_failure(0, 0.0F, make_dspark_proposal_rng_state_v1(1U, 2U, 3U),
                      DsparkProposalDeviceCode::kNonPositiveTemperature));
  CHECK(check_failure(0, std::numeric_limits<float>::quiet_NaN(),
                      make_dspark_proposal_rng_state_v1(1U, 2U, 3U),
                      DsparkProposalDeviceCode::kNonPositiveTemperature));
  return true;
}

template <DType LogitsDType>
[[nodiscard]] bool GuardedEntrypointPreservesPublishedStateForDtype() {
  ProposalFixture<LogitsDType> fixture;
  if (!fixture.initialize() || !fixture.prepare_zero_weights()) {
    return false;
  }
  const DsparkProposalRngStateV1 initial =
      make_dspark_proposal_rng_state_v1(17U, 23U, 29U);
  if (!fixture.reset(0, 1.0F, false, initial)) {
    return false;
  }
  constexpr uint32_t kUpstreamStatus = 0xabcddcbaU;
  if (!fixture.write_status(kUpstreamStatus)) {
    return false;
  }
  std::optional<CudaCapturedGraph> graph;
  bool captured = false;
  if constexpr (LogitsDType == DType::kBFloat16) {
    captured =
        take_result(capture_dspark_proposal_bfloat16_graph(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &graph);
  } else {
    captured =
        take_result(capture_dspark_proposal_float16_graph(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &graph);
  }
  std::optional<CudaGraphExecutable> executable;
  if (!captured ||
      !take_result(CudaGraphExecutable::instantiate(
                       graph->graph(), fixture.context(), fixture.lease()),
                   &executable) ||
      !is_ok(graph->close()) || !is_ok(executable->launch()) ||
      !is_ok(executable->synchronize())) {
    return false;
  }

  std::array<int64_t, kDsparkProductionGamma> tokens{};
  std::array<float, kDsparkProductionGamma> normalizers{};
  DsparkProposalRngStateV1 after{};
  uint32_t status = 0U;
  if (!fixture.read_tokens(&tokens) ||
      !fixture.read_normalizers(&normalizers) || !fixture.read_state(&after) ||
      !fixture.read_status(&status)) {
    return false;
  }
  return std::all_of(tokens.begin(), tokens.end(),
                     [](int64_t value) { return value == kTokenSentinel; }) &&
         std::all_of(
             normalizers.begin(), normalizers.end(),
             [](float value) { return value == kNormalizerSentinel; }) &&
         after.descriptor == initial.descriptor && after.seed == initial.seed &&
         after.subsequence == initial.subsequence &&
         after.counter == initial.counter && status == kUpstreamStatus &&
         is_ok(executable->close());
}

[[nodiscard]] bool GuardedEntrypointsPreservePublishedState() {
  CHECK(GuardedEntrypointPreservesPublishedStateForDtype<DType::kBFloat16>());
  CHECK(GuardedEntrypointPreservesPublishedStateForDtype<DType::kFloat16>());
  return true;
}

[[nodiscard]] bool SampledTokenMatchesControlledPhiloxOracle() {
  ProposalFixture<DType::kBFloat16> fixture;
  CHECK(fixture.initialize());
  CHECK(fixture.prepare_zero_weights());
  const auto initial = make_dspark_proposal_rng_state_v1(
      0x0123456789abcdefULL, 0xfedcba9876543210ULL, 17U);
  constexpr std::array<uint32_t, 4> candidates{3U, 17U, 1009U, 248319U};
  constexpr std::array<float, 4> logits{0.25F, 0.5F, 1.25F, -0.75F};
  for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
    for (uint32_t index = 0; index < candidates.size(); ++index) {
      CHECK(fixture.set_base_logit(row, candidates[index], logits[index]));
    }
  }
  CHECK(fixture.reset(0, 1.0F, false, initial));
  CHECK(fixture.launch());
  CHECK_STATUS(fixture.context().synchronize());

  std::array<int64_t, kDsparkProductionGamma> actual{};
  CHECK(fixture.read_tokens(&actual));
  for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
    float best_key = -1.0F;
    uint32_t best_token = kDsparkProductionVocabSize;
    for (uint32_t token = 0; token < kDsparkProductionVocabSize; ++token) {
      float logit = 0.0F;
      for (uint32_t index = 0; index < candidates.size(); ++index) {
        if (token == candidates[index]) {
          logit = logits[index];
        }
      }
      const float mass = std::exp(logit - logits[2]);
      const float key = mass / reference_exponential_noise(initial, row, token);
      if (key > best_key || (key == best_key && token < best_token)) {
        best_key = key;
        best_token = token;
      }
    }
    CHECK(actual[row] == static_cast<int64_t>(best_token));
  }
  return true;
}

[[nodiscard]] bool SampledReplayIsStableAndResettable() {
  ProposalFixture<DType::kBFloat16> fixture;
  CHECK(fixture.initialize());
  CHECK(fixture.prepare_zero_weights());
  const auto initial = make_dspark_proposal_rng_state_v1(
      0x0123456789abcdefULL, 0xfedcba9876543210ULL, 11U);
  CHECK(fixture.reset(0, 1.0F, false, initial));
  CHECK(fixture.write_status(0U));
  const uintptr_t corrected_address = fixture.corrected_address();
  const uintptr_t proposal_address = fixture.proposal_address();

  std::optional<CudaCapturedGraph> graph;
  CHECK(take_result(capture_dspark_proposal_bfloat16_graph(
                        fixture.context(), fixture.lease(), fixture.buffers()),
                    &graph));
  CHECK(graph->valid());
  CHECK(graph->device_ordinal() == fixture.context().device_ordinal());
  size_t node_count = 0;
  CHECK_CUDA(cudaGraphGetNodes(graph->graph(), nullptr, &node_count));
  CHECK(node_count == 1U);

  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(CudaGraphExecutable::instantiate(
                        graph->graph(), fixture.context(), fixture.lease()),
                    &executable));
  CHECK_STATUS(graph->close());
  CHECK(!graph->valid());
  CHECK(graph->graph() == nullptr);

  std::array<int64_t, kDsparkProductionGamma> first{};
  std::array<int64_t, kDsparkProductionGamma> second{};
  for (uint64_t replay = 0; replay < 2U; ++replay) {
    CHECK_STATUS(executable->launch());
    CHECK_STATUS(executable->synchronize());
    auto *output = replay == 0U ? &first : &second;
    CHECK(fixture.read_tokens(output));
    CHECK(fixture.corrected_address() == corrected_address);
    CHECK(fixture.proposal_address() == proposal_address);
    DsparkProposalRngStateV1 state{};
    CHECK(fixture.read_state(&state));
    CHECK(state.counter == initial.counter + (replay + 1U) * kBlocksPerReplay);
    CHECK(std::all_of(output->begin(), output->end(), [](int64_t token) {
      return token >= 0 && token < kDsparkProductionVocabSize;
    }));
  }
  CHECK(first != second);

  CHECK(fixture.reset(0, 1.0F, false, initial));
  CHECK(fixture.write_status(0U));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  std::array<int64_t, kDsparkProductionGamma> reset{};
  CHECK(fixture.read_tokens(&reset));
  CHECK(reset == first);

  CHECK(fixture.reset(0, 1.0F, true, initial));
  CHECK(fixture.write_status(0U));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  std::array<int64_t, kDsparkProductionGamma> greedy{};
  CHECK(fixture.read_tokens(&greedy));
  CHECK(std::all_of(greedy.begin(), greedy.end(),
                    [](int64_t token) { return token == 0; }));
  CHECK_STATUS(executable->close());
  executable.reset();
  return true;
}

[[nodiscard]] bool TrainedGammaSevenOracleFixture(const char *weights_path,
                                                  const char *base_logits_path,
                                                  const char *expected_path) {
  constexpr uint64_t kWeightBytes = kMarkovElements * sizeof(uint16_t);
  constexpr uint64_t kBaseLogitBytes = kLogitsElements * sizeof(uint16_t);
  static_assert(2U * kWeightBytes <= std::numeric_limits<std::size_t>::max());
  static_assert(kBaseLogitBytes <= std::numeric_limits<std::size_t>::max());
  std::vector<std::byte> weights(static_cast<std::size_t>(2U * kWeightBytes));
  std::vector<std::byte> base_logits(static_cast<std::size_t>(kBaseLogitBytes));
  std::vector<std::byte> expected_bytes(
      sizeof(TrainedOracleV1) + static_cast<std::size_t>(kBaseLogitBytes));
  CHECK(read_exact_file(weights_path, std::span<std::byte>(weights)));
  CHECK(read_exact_file(base_logits_path, std::span<std::byte>(base_logits)));
  CHECK(read_exact_file(expected_path, std::span<std::byte>(expected_bytes)));

  TrainedOracleV1 expected{};
  std::memcpy(&expected, expected_bytes.data(), sizeof(expected));
  CHECK(expected.magic == kTrainedOracleMagic);
  CHECK(expected.version == 1U);
  CHECK(expected.gamma == kDsparkProductionGamma);
  CHECK(expected.vocab_size == kDsparkProductionVocabSize);
  CHECK(expected.markov_rank == kDsparkProductionMarkovRank);
  CHECK(expected.reserved == 0U);
  CHECK(expected.trailing_padding == 0U);
  CHECK(expected.counter <=
        std::numeric_limits<uint64_t>::max() - kBlocksPerReplay);
  CHECK(expected.final_counter == expected.counter + kBlocksPerReplay);
  CHECK(expected.anchor_token >= 0);
  CHECK(expected.anchor_token < kDsparkProductionVocabSize);
  CHECK(std::isfinite(expected.temperature));
  CHECK(expected.temperature > 0.0F);
  CHECK(std::all_of(expected.tokens.begin(), expected.tokens.end(),
                    [](int64_t token) {
                      return token >= 0 && token < kDsparkProductionVocabSize;
                    }));
  CHECK(std::all_of(expected.log_normalizers.begin(),
                    expected.log_normalizers.end(),
                    [](float value) { return std::isfinite(value); }));

  ProposalFixture<DType::kBFloat16> fixture;
  CHECK(fixture.initialize());
  const auto buffers = fixture.buffers();
  CHECK_CUDA(cudaMemcpy(
      const_cast<std::byte *>(buffers.base_logits.data_bytes()),
      base_logits.data(), base_logits.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(const_cast<std::byte *>(buffers.markov_w1.data_bytes()),
                        weights.data(), static_cast<std::size_t>(kWeightBytes),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(const_cast<std::byte *>(buffers.markov_w2.data_bytes()),
                        weights.data() + kWeightBytes,
                        static_cast<std::size_t>(kWeightBytes),
                        cudaMemcpyHostToDevice));
  const DsparkProposalRngStateV1 initial{kDsparkProposalRngStateDescriptorV1,
                                         expected.seed, expected.subsequence,
                                         expected.counter};
  CHECK(fixture.reset(expected.anchor_token, expected.temperature, false,
                      initial));
  CHECK(fixture.launch());
  CHECK_STATUS(fixture.context().synchronize());

  std::array<int64_t, kDsparkProductionGamma> tokens{};
  std::array<float, kDsparkProductionGamma> normalizers{};
  DsparkProposalRngStateV1 state{};
  uint32_t status = kStatusSentinel;
  CHECK(fixture.read_tokens(&tokens));
  CHECK(fixture.read_normalizers(&normalizers));
  CHECK(fixture.read_state(&state));
  CHECK(fixture.read_status(&status));
  CHECK(status == 0U);
  CHECK(tokens == expected.tokens);
  CHECK(state.descriptor == initial.descriptor);
  CHECK(state.seed == initial.seed);
  CHECK(state.subsequence == initial.subsequence);
  CHECK(state.counter == expected.final_counter);

  float max_observed_corrected_error = 0.0F;
  float max_observed_normalizer_error = 0.0F;
  for (uint32_t row = 0; row < kDsparkProductionGamma; ++row) {
    std::vector<uint16_t> actual_corrected;
    std::vector<uint16_t> expected_corrected(kDsparkProductionVocabSize);
    CHECK(fixture.read_corrected_row(row, &actual_corrected));
    std::memcpy(expected_corrected.data(),
                expected_bytes.data() + sizeof(TrainedOracleV1) +
                    static_cast<uint64_t>(row) * kDsparkProductionVocabSize *
                        sizeof(uint16_t),
                expected_corrected.size() * sizeof(uint16_t));
    float max_corrected_error = 0.0F;
    for (uint32_t token = 0; token < kDsparkProductionVocabSize; ++token) {
      const float actual_value =
          __bfloat162float(__ushort_as_bfloat16(actual_corrected[token]));
      const float expected_value =
          __bfloat162float(__ushort_as_bfloat16(expected_corrected[token]));
      if (!std::isfinite(actual_value) || !std::isfinite(expected_value)) {
        std::printf("corrected row %u token %u contains a non-finite value\n",
                    row, token);
        return false;
      }
      max_corrected_error = std::max(max_corrected_error,
                                     std::abs(actual_value - expected_value));
    }
    max_observed_corrected_error =
        std::max(max_observed_corrected_error, max_corrected_error);
    // The native scalar rank-256 FMA order and the PyTorch CUDA GEMM reduction
    // order are intentionally different. One BF16 output quantum is the
    // measured full-tensor bound; sampled tokens and RNG state remain exact.
    if (max_corrected_error > kMaxTrainedOracleCorrectedAbsError) {
      std::printf("corrected row %u max_abs_error=%.9g\n", row,
                  static_cast<double>(max_corrected_error));
      return false;
    }
    float row_max = -std::numeric_limits<float>::infinity();
    for (const uint16_t bits : actual_corrected) {
      row_max = std::max(row_max, __bfloat162float(__ushort_as_bfloat16(bits)) /
                                      expected.temperature);
    }
    float sum = 0.0F;
    for (const uint16_t bits : actual_corrected) {
      const float scaled =
          __bfloat162float(__ushort_as_bfloat16(bits)) / expected.temperature;
      sum += std::exp(scaled - row_max);
    }
    const float reproduced_normalizer = row_max + std::log(sum);
    CHECK(std::abs(normalizers[row] - reproduced_normalizer) <=
          kMaxNativeNormalizerReproductionError);
    const float normalizer_error =
        std::abs(normalizers[row] - expected.log_normalizers[row]);
    max_observed_normalizer_error =
        std::max(max_observed_normalizer_error, normalizer_error);
    CHECK(normalizer_error <= kMaxPythonNormalizerParityError);
  }
  std::printf("trained oracle max corrected error=%.9g, max normalizer "
              "error=%.9g\n",
              static_cast<double>(max_observed_corrected_error),
              static_cast<double>(max_observed_normalizer_error));
  return true;
}

[[nodiscard]] bool CaptureRejectsForeignArena() {
  ProposalFixture<DType::kBFloat16> fixture;
  CHECK(fixture.initialize());

  std::optional<GraphMemoryArena> other_arena;
  std::optional<GraphArenaLease> other_lease;
  CHECK(take_result(GraphMemoryArena::allocate(fixture.context(), 256),
                    &other_arena));
  CHECK_STATUS(other_arena->seal());
  CHECK(take_result(other_arena->acquire_lease(), &other_lease));

  NativeRuntimeError error = sglang::native::native_runtime_ok();
  std::move(capture_dspark_proposal_bfloat16_graph(
                fixture.context(), *other_lease, fixture.buffers()))
      .match([](CudaCapturedGraph &&) noexcept { return false; },
             [&error](NativeRuntimeError &&value) noexcept {
               error = value;
               return true;
             });
  CHECK(error.code == NativeRuntimeCode::kForeignSlice);
  CHECK(error.operation == NativeRuntimeOperation::kGraphCaptureBegin);
  CHECK(error.detail ==
        static_cast<uint32_t>(DsparkProposalArgument::kBaseLogits));
  other_lease.reset();
  CHECK_STATUS(other_arena->close());
  return true;
}

struct TestCase final {
  const char *name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"GreedyChainBfloat16",
     GreedyChainPreservesDtypeAndDependency<DType::kBFloat16>},
    {"GreedyChainFloat16",
     GreedyChainPreservesDtypeAndDependency<DType::kFloat16>},
    {"InvalidInputsPreserveOutputsAndCounter",
     InvalidInputsPreserveOutputsAndCounter},
    {"GuardedEntrypointsPreservePublishedState",
     GuardedEntrypointsPreservePublishedState},
    {"SampledTokenMatchesControlledPhiloxOracle",
     SampledTokenMatchesControlledPhiloxOracle},
    {"SampledReplayIsStableAndResettable", SampledReplayIsStableAndResettable},
    {"CaptureRejectsForeignArena", CaptureRejectsForeignArena},
};

} // namespace

int main(int argc, char **argv) {
  for (const TestCase &test : kTests) {
    std::printf("[ RUN      ] %s\n", test.name);
    if (!test.function()) {
      std::printf("[  FAILED  ] %s\n", test.name);
      return EXIT_FAILURE;
    }
    std::printf("[       OK ] %s\n", test.name);
  }
  bool trained_oracle_ran = false;
  if (argc != 1) {
    if (argc != 5 || std::string_view(argv[1]) != "--trained-oracle") {
      std::printf("usage: %s [--trained-oracle WEIGHTS BASE EXPECTED]\n",
                  argv[0]);
      return EXIT_FAILURE;
    }
    std::printf("[ RUN      ] TrainedGammaSevenOracleFixture\n");
    if (!TrainedGammaSevenOracleFixture(argv[2], argv[3], argv[4])) {
      std::printf("[  FAILED  ] TrainedGammaSevenOracleFixture\n");
      return EXIT_FAILURE;
    }
    std::printf("[       OK ] TrainedGammaSevenOracleFixture\n");
    trained_oracle_ran = true;
  }
  std::printf("[  PASSED  ] %zu tests\n",
              std::size(kTests) + (trained_oracle_ran ? 1U : 0U));
  return EXIT_SUCCESS;
}
