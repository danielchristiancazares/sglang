#include "sglang/native/linear_rejection_sampling.hpp"
#include "sglang/native/linear_verify_rng.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
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
using sglang::native::is_ok;
using sglang::native::kLinearVerifyRngStateDescriptorV1;
using sglang::native::launch_linear_rejection_sampling_if_ready;
using sglang::native::launch_seeded_linear_verify_rng;
using sglang::native::launch_stateful_linear_verify_rng;
using sglang::native::LinearRejectionSamplingBuffers;
using sglang::native::LinearVerifyConstUInt64Vector;
using sglang::native::LinearVerifyMutableFloat32Matrix;
using sglang::native::LinearVerifyMutableFloat32Vector;
using sglang::native::LinearVerifyMutableUInt64Vector;
using sglang::native::LinearVerifyRngArgument;
using sglang::native::LinearVerifyRngDeviceCode;
using sglang::native::LinearVerifyRngStateV1;
using sglang::native::make_linear_verify_rng_state_v1;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::MutableInt32Matrix;
using sglang::native::MutableInt32Vector;
using sglang::native::MutableUInt32Vector;
using sglang::native::native_runtime_code_name;
using sglang::native::native_runtime_operation_name;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::NativeRuntimeResult;
using sglang::native::SeededLinearVerifyRngBuffers;
using sglang::native::StatefulLinearVerifyRngBuffers;
using sglang::native::TensorAccess;

constexpr float kCoinSentinel = -17.0F;
constexpr int32_t kTokenSentinel = -9137;
constexpr uint32_t kStatusSentinel = 0xffffffffU;

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

[[nodiscard]] constexpr uint32_t rotate_left_32(uint32_t value,
                                                uint32_t bits) noexcept {
  return (value << bits) | (value >> (32U - bits));
}

[[nodiscard]] constexpr uint32_t murmur_mix(uint32_t hash,
                                            uint32_t value) noexcept {
  value *= 0xcc9e2d51U;
  value = rotate_left_32(value, 15U);
  value *= 0x1b873593U;
  hash ^= value;
  hash = rotate_left_32(hash, 13U);
  return hash * 5U + 0xe6546b64U;
}

[[nodiscard]] constexpr uint32_t murmur_finalize(uint32_t hash) noexcept {
  hash ^= 16U;
  hash ^= hash >> 16U;
  hash *= 0x85ebca6bU;
  hash ^= hash >> 13U;
  hash *= 0xc2b2ae35U;
  hash ^= hash >> 16U;
  return hash;
}

[[nodiscard]] constexpr uint32_t
reference_seeded_hash(uint64_t seed, uint64_t sequence_length,
                      uint32_t column) noexcept {
  uint32_t hash = 0U;
  hash = murmur_mix(hash, static_cast<uint32_t>(seed));
  hash = murmur_mix(hash, static_cast<uint32_t>(seed >> 32U));
  hash = murmur_mix(hash, static_cast<uint32_t>(sequence_length));
  hash = murmur_mix(hash, column);
  return murmur_finalize(hash);
}

[[nodiscard]] float reference_seeded_coin(uint64_t seed,
                                          uint64_t sequence_length,
                                          uint32_t column) noexcept {
  constexpr float kLargestCoin = 1.0F - 0x1p-24F;
  const double uniform =
      static_cast<double>(
          reference_seeded_hash(seed, sequence_length, column)) /
      static_cast<double>(std::numeric_limits<uint32_t>::max());
  return std::min(static_cast<float>(uniform), kLargestCoin);
}

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

[[nodiscard]] constexpr float reference_stateful_coin(uint32_t value) noexcept {
  return static_cast<float>(value >> 8U) * 0x1p-24F;
}

struct HostCoins final {
  std::vector<float> accept;
  std::array<float, 1> bonus;
  std::array<uint32_t, 1> status;
};

[[nodiscard]] HostCoins
reference_stateful_coins(const LinearVerifyRngStateV1 &state,
                         uint32_t num_slots) {
  HostCoins coins{std::vector<float>(num_slots), {0.0F}, {0U}};
  uint32_t draw = 0;
  const uint32_t groups = (num_slots + 4U) / 4U;
  for (uint32_t group = 0; group < groups; ++group) {
    const PhiloxBlock block = reference_philox4x32_10(
        state.counter + group, state.subsequence, state.seed);
    const uint32_t values[4]{block.x, block.y, block.z, block.w};
    for (uint32_t lane = 0; lane < 4U; ++lane, ++draw) {
      if (draw < num_slots) {
        coins.accept[draw] = reference_stateful_coin(values[lane]);
      } else if (draw == num_slots) {
        coins.bonus[0] = reference_stateful_coin(values[lane]);
      }
    }
  }
  return coins;
}

[[nodiscard]] bool coins_equal(const HostCoins &left, const HostCoins &right) {
  CHECK(left.accept == right.accept);
  CHECK(left.bonus == right.bonus);
  CHECK(left.status == right.status);
  return true;
}

class VerifyFixture final {
public:
  VerifyFixture(const VerifyFixture &) = delete;
  VerifyFixture &operator=(const VerifyFixture &) = delete;
  VerifyFixture(VerifyFixture &&) = delete;
  VerifyFixture &operator=(VerifyFixture &&) = delete;

  explicit VerifyFixture(uint32_t num_slots, uint32_t vocab_size) noexcept
      : num_slots_(num_slots), vocab_size_(vocab_size) {}

  [[nodiscard]] bool initialize() noexcept {
    const uint64_t slots = num_slots_;
    const uint64_t vocab = vocab_size_;
    const uint64_t target_bytes = slots * vocab * sizeof(float);
    const uint64_t draft_bytes = (slots - 1U) * vocab * sizeof(float);
    const uint64_t capacity =
        target_bytes + draft_bytes +
        slots * (2U * sizeof(int32_t) + 2U * sizeof(int64_t) + sizeof(float)) +
        sizeof(LinearVerifyRngStateV1) + 2U * sizeof(uint64_t) +
        2U * sizeof(int32_t) + sizeof(float) + sizeof(uint32_t) + 13U * 256U;

    if (!take_result(CudaStream::create_nonblocking(), &stream_) ||
        !take_result(stream_->context(), &context_) ||
        !take_result(GraphMemoryArena::allocate(*context_, capacity),
                     &arena_) ||
        !reserve(sizeof(uint64_t), &seed_slice_) ||
        !reserve(sizeof(uint64_t), &sequence_length_slice_) ||
        !reserve(sizeof(LinearVerifyRngStateV1), &state_slice_) ||
        !reserve(slots * sizeof(float), &accept_uniforms_slice_) ||
        !reserve(sizeof(float), &bonus_uniforms_slice_) ||
        !reserve(sizeof(uint32_t), &device_status_slice_) ||
        !reserve(slots * sizeof(int32_t), &out_tokens_slice_) ||
        !reserve(slots * sizeof(int32_t), &accept_indices_slice_) ||
        !reserve(sizeof(int32_t), &num_correct_drafts_slice_) ||
        !reserve(slots * sizeof(int64_t), &proposal_tokens_slice_) ||
        !reserve(slots * sizeof(int64_t), &proposal_out_indices_slice_) ||
        !reserve(target_bytes, &target_probs_slice_) ||
        !reserve(draft_bytes, &draft_probs_slice_)) {
      return false;
    }
    if (!check_status(arena_->seal(), "arena_->seal()", __LINE__) ||
        !take_result(arena_->acquire_lease(), &lease_)) {
      return false;
    }

    if (!take_result(lease_->bind_const<DType::kUInt64, 1>(
                         *seed_slice_, metadata_1d(SGL_NATIVE_DTYPE_UINT64, 1)),
                     &seed_) ||
        !take_result(lease_->bind_const<DType::kUInt64, 1>(
                         *sequence_length_slice_,
                         metadata_1d(SGL_NATIVE_DTYPE_UINT64, 1)),
                     &sequence_length_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt64, 1>(
                *state_slice_, metadata_1d(SGL_NATIVE_DTYPE_UINT64, 4)),
            &state_) ||
        !take_result(lease_->bind_mutable<DType::kFloat32, 2>(
                         *accept_uniforms_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_FLOAT32, 1, num_slots_)),
                     &rng_accept_uniforms_) ||
        !take_result(lease_->bind_const<DType::kFloat32, 2>(
                         *accept_uniforms_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_FLOAT32, 1, num_slots_)),
                     &sampler_accept_uniforms_) ||
        !take_result(lease_->bind_mutable<DType::kFloat32, 1>(
                         *bonus_uniforms_slice_,
                         metadata_1d(SGL_NATIVE_DTYPE_FLOAT32, 1)),
                     &rng_bonus_uniforms_) ||
        !take_result(lease_->bind_const<DType::kFloat32, 1>(
                         *bonus_uniforms_slice_,
                         metadata_1d(SGL_NATIVE_DTYPE_FLOAT32, 1)),
                     &sampler_bonus_uniforms_) ||
        !take_result(
            lease_->bind_mutable<DType::kUInt32, 1>(
                *device_status_slice_, metadata_1d(SGL_NATIVE_DTYPE_UINT32, 1)),
            &device_status_) ||
        !take_result(lease_->bind_mutable<DType::kInt32, 1>(
                         *out_tokens_slice_,
                         metadata_1d(SGL_NATIVE_DTYPE_INT32, num_slots_)),
                     &out_tokens_) ||
        !take_result(lease_->bind_mutable<DType::kInt32, 2>(
                         *accept_indices_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_INT32, 1, num_slots_)),
                     &accept_indices_) ||
        !take_result(lease_->bind_mutable<DType::kInt32, 1>(
                         *num_correct_drafts_slice_,
                         metadata_1d(SGL_NATIVE_DTYPE_INT32, 1)),
                     &num_correct_drafts_) ||
        !take_result(lease_->bind_const<DType::kInt64, 2>(
                         *proposal_tokens_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_INT64, 1, num_slots_)),
                     &proposal_tokens_) ||
        !take_result(lease_->bind_const<DType::kInt64, 2>(
                         *proposal_out_indices_slice_,
                         metadata_2d(SGL_NATIVE_DTYPE_INT64, 1, num_slots_)),
                     &proposal_out_indices_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 3>(
                *target_probs_slice_, metadata_3d(SGL_NATIVE_DTYPE_FLOAT32, 1,
                                                  num_slots_, vocab_size_)),
            &target_probs_) ||
        !take_result(
            lease_->bind_const<DType::kFloat32, 3>(
                *draft_probs_slice_, metadata_3d(SGL_NATIVE_DTYPE_FLOAT32, 1,
                                                 num_slots_ - 1, vocab_size_)),
            &draft_probs_)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] SeededLinearVerifyRngBuffers seeded_buffers() const noexcept {
    return SeededLinearVerifyRngBuffers{*seed_, *sequence_length_,
                                        *rng_accept_uniforms_,
                                        *rng_bonus_uniforms_, *device_status_};
  }

  [[nodiscard]] StatefulLinearVerifyRngBuffers
  stateful_buffers() const noexcept {
    return StatefulLinearVerifyRngBuffers{
        *state_, *rng_accept_uniforms_, *rng_bonus_uniforms_, *device_status_};
  }

  [[nodiscard]] LinearRejectionSamplingBuffers
  sampling_buffers() const noexcept {
    return LinearRejectionSamplingBuffers{*out_tokens_,
                                          *accept_indices_,
                                          *num_correct_drafts_,
                                          *proposal_tokens_,
                                          *proposal_out_indices_,
                                          *sampler_accept_uniforms_,
                                          *sampler_bonus_uniforms_,
                                          *target_probs_,
                                          *draft_probs_,
                                          *device_status_};
  }

  [[nodiscard]] bool copy_seeded_inputs(uint64_t seed,
                                        uint64_t sequence_length) noexcept {
    return copy_to_const(*seed_, std::span<const uint64_t>(&seed, 1)) &&
           copy_to_const(*sequence_length_,
                         std::span<const uint64_t>(&sequence_length, 1));
  }

  [[nodiscard]] bool copy_state(const LinearVerifyRngStateV1 &state) noexcept {
    const std::array<uint64_t, 4> words{state.descriptor, state.seed,
                                        state.subsequence, state.counter};
    return copy_to_mutable(*state_, std::span<const uint64_t>(words));
  }

  [[nodiscard]] bool read_state(LinearVerifyRngStateV1 *state) const noexcept {
    if (state == nullptr) {
      return false;
    }
    std::array<uint64_t, 4> words{};
    if (!copy_to_host(std::span<uint64_t>(words), *state_)) {
      return false;
    }
    *state = LinearVerifyRngStateV1{words[0], words[1], words[2], words[3]};
    return true;
  }

  [[nodiscard]] bool reset_coins() noexcept {
    const std::vector<float> accept(num_slots_, kCoinSentinel);
    const std::array<float, 1> bonus{kCoinSentinel};
    const std::array<uint32_t, 1> status{kStatusSentinel};
    return copy_to_mutable(*rng_accept_uniforms_,
                           std::span<const float>(accept)) &&
           copy_to_mutable(*rng_bonus_uniforms_,
                           std::span<const float>(bonus)) &&
           copy_to_mutable(*device_status_, std::span<const uint32_t>(status));
  }

  [[nodiscard]] bool read_coins(HostCoins *coins) const noexcept {
    if (coins == nullptr) {
      return false;
    }
    *coins = HostCoins{std::vector<float>(num_slots_), {0.0F}, {0U}};
    return copy_to_host(std::span<float>(coins->accept),
                        *rng_accept_uniforms_) &&
           copy_to_host(std::span<float>(coins->bonus), *rng_bonus_uniforms_) &&
           copy_to_host(std::span<uint32_t>(coins->status), *device_status_);
  }

  [[nodiscard]] bool
  copy_sampler_inputs(std::span<const int64_t> proposal_tokens,
                      std::span<const int64_t> proposal_out_indices,
                      std::span<const float> target_probs,
                      std::span<const float> draft_probs) noexcept {
    if (proposal_tokens.size() != num_slots_ ||
        proposal_out_indices.size() != num_slots_ ||
        target_probs.size() !=
            static_cast<uint64_t>(num_slots_) * vocab_size_ ||
        draft_probs.size() !=
            static_cast<uint64_t>(num_slots_ - 1U) * vocab_size_) {
      return false;
    }
    return copy_to_const(*proposal_tokens_, proposal_tokens) &&
           copy_to_const(*proposal_out_indices_, proposal_out_indices) &&
           copy_to_const(*target_probs_, target_probs) &&
           copy_to_const(*draft_probs_, draft_probs);
  }

  [[nodiscard]] bool reset_sampler_outputs() noexcept {
    const std::vector<int32_t> tokens(num_slots_, 0);
    const std::vector<int32_t> indices(num_slots_, -1);
    const std::array<int32_t, 1> count{0};
    return copy_to_mutable(*out_tokens_, std::span<const int32_t>(tokens)) &&
           copy_to_mutable(*accept_indices_,
                           std::span<const int32_t>(indices)) &&
           copy_to_mutable(*num_correct_drafts_,
                           std::span<const int32_t>(count));
  }

  [[nodiscard]] bool capture_sampler_output_initialization() noexcept {
    return cudaMemsetAsync(out_tokens_->data_bytes(), 0,
                           num_slots_ * sizeof(int32_t),
                           context_->stream()) == cudaSuccess &&
           cudaMemsetAsync(accept_indices_->data_bytes(), 0xff,
                           num_slots_ * sizeof(int32_t),
                           context_->stream()) == cudaSuccess &&
           cudaMemsetAsync(num_correct_drafts_->data_bytes(), 0,
                           sizeof(int32_t), context_->stream()) == cudaSuccess;
  }

  [[nodiscard]] bool read_sampler_outputs(std::vector<int32_t> *tokens,
                                          std::vector<int32_t> *indices,
                                          int32_t *count,
                                          uint32_t *status) const noexcept {
    if (tokens == nullptr || indices == nullptr || count == nullptr ||
        status == nullptr) {
      return false;
    }
    tokens->assign(num_slots_, kTokenSentinel);
    indices->assign(num_slots_, kTokenSentinel);
    std::array<int32_t, 1> count_value{};
    std::array<uint32_t, 1> status_value{};
    if (!copy_to_host(std::span<int32_t>(*tokens), *out_tokens_) ||
        !copy_to_host(std::span<int32_t>(*indices), *accept_indices_) ||
        !copy_to_host(std::span<int32_t>(count_value), *num_correct_drafts_) ||
        !copy_to_host(std::span<uint32_t>(status_value), *device_status_)) {
      return false;
    }
    *count = count_value[0];
    *status = status_value[0];
    return true;
  }

  [[nodiscard]] const CudaExecutionContext &context() const noexcept {
    return *context_;
  }

  [[nodiscard]] const GraphArenaLease &lease() const noexcept {
    return *lease_;
  }

  [[nodiscard]] const GraphMemorySlice &bonus_uniforms_slice() const noexcept {
    return *bonus_uniforms_slice_;
  }

  [[nodiscard]] uintptr_t accept_uniforms_address() const noexcept {
    return reinterpret_cast<uintptr_t>(rng_accept_uniforms_->data_bytes());
  }

  [[nodiscard]] uintptr_t out_tokens_address() const noexcept {
    return reinterpret_cast<uintptr_t>(out_tokens_->data_bytes());
  }

private:
  [[nodiscard]] bool reserve(uint64_t bytes,
                             std::optional<GraphMemorySlice> *output) noexcept {
    return take_result(arena_->reserve(bytes, 256), output);
  }

  template <DType D, uint32_t Rank, typename T>
  [[nodiscard]] bool copy_to_const(const sglang::native::GraphStableTensorView<
                                       D, Rank, TensorAccess::kReadOnly> &view,
                                   std::span<const T> values) noexcept {
    const cudaError_t result =
        cudaMemcpy(const_cast<std::byte *>(view.data_bytes()), values.data(),
                   values.size_bytes(), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::printf("input copy failed: %s\n", cudaGetErrorString(result));
      return false;
    }
    return true;
  }

  template <DType D, uint32_t Rank, typename T>
  [[nodiscard]] bool
  copy_to_mutable(const sglang::native::GraphStableTensorView<
                      D, Rank, TensorAccess::kReadWrite> &view,
                  std::span<const T> values) noexcept {
    const cudaError_t result =
        cudaMemcpy(view.data_bytes(), values.data(), values.size_bytes(),
                   cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
      std::printf("state/output copy failed: %s\n", cudaGetErrorString(result));
      return false;
    }
    return true;
  }

  template <typename T, DType D, uint32_t Rank, TensorAccess Access>
  [[nodiscard]] bool
  copy_to_host(std::span<T> values,
               const sglang::native::GraphStableTensorView<D, Rank, Access>
                   &view) const noexcept {
    const cudaError_t result =
        cudaMemcpy(values.data(), view.data_bytes(), values.size_bytes(),
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
  std::optional<GraphMemorySlice> seed_slice_;
  std::optional<GraphMemorySlice> sequence_length_slice_;
  std::optional<GraphMemorySlice> state_slice_;
  std::optional<GraphMemorySlice> accept_uniforms_slice_;
  std::optional<GraphMemorySlice> bonus_uniforms_slice_;
  std::optional<GraphMemorySlice> device_status_slice_;
  std::optional<GraphMemorySlice> out_tokens_slice_;
  std::optional<GraphMemorySlice> accept_indices_slice_;
  std::optional<GraphMemorySlice> num_correct_drafts_slice_;
  std::optional<GraphMemorySlice> proposal_tokens_slice_;
  std::optional<GraphMemorySlice> proposal_out_indices_slice_;
  std::optional<GraphMemorySlice> target_probs_slice_;
  std::optional<GraphMemorySlice> draft_probs_slice_;
  std::optional<GraphArenaLease> lease_;
  std::optional<LinearVerifyConstUInt64Vector> seed_;
  std::optional<LinearVerifyConstUInt64Vector> sequence_length_;
  std::optional<LinearVerifyMutableUInt64Vector> state_;
  std::optional<LinearVerifyMutableFloat32Matrix> rng_accept_uniforms_;
  std::optional<ConstFloat32Matrix> sampler_accept_uniforms_;
  std::optional<LinearVerifyMutableFloat32Vector> rng_bonus_uniforms_;
  std::optional<ConstFloat32Vector> sampler_bonus_uniforms_;
  std::optional<MutableUInt32Vector> device_status_;
  std::optional<MutableInt32Vector> out_tokens_;
  std::optional<MutableInt32Matrix> accept_indices_;
  std::optional<MutableInt32Vector> num_correct_drafts_;
  std::optional<ConstInt64Matrix> proposal_tokens_;
  std::optional<ConstInt64Matrix> proposal_out_indices_;
  std::optional<ConstFloat32Tensor3> target_probs_;
  std::optional<ConstFloat32Tensor3> draft_probs_;
};

[[nodiscard]] bool MetadataAndDeviceFailuresAreClosed() {
  VerifyFixture fixture(3, 8);
  CHECK(fixture.initialize());

  std::optional<MutableUInt32Vector> aliased_status;
  CHECK(take_result(fixture.lease().bind_mutable<DType::kUInt32, 1>(
                        fixture.bonus_uniforms_slice(),
                        metadata_1d(SGL_NATIVE_DTYPE_UINT32, 1)),
                    &aliased_status));
  const auto seeded = fixture.seeded_buffers();
  const SeededLinearVerifyRngBuffers aliased{
      seeded.seed, seeded.sequence_length, seeded.accept_uniforms,
      seeded.bonus_uniforms, *aliased_status};
  const NativeRuntimeError alias_error =
      launch_seeded_linear_verify_rng(fixture.context(), aliased);
  CHECK(alias_error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(alias_error.operation ==
        NativeRuntimeOperation::kValidateLinearVerifyRng);
  CHECK(alias_error.detail ==
        static_cast<uint32_t>(LinearVerifyRngArgument::kBonusUniforms));
  CHECK(alias_error.actual ==
        static_cast<uint64_t>(LinearVerifyRngArgument::kDeviceStatus));

  const LinearVerifyRngStateV1 invalid{0U, 11U, 13U, 17U};
  CHECK(fixture.copy_state(invalid));
  CHECK(fixture.reset_coins());
  CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                 fixture.stateful_buffers()));
  CHECK_STATUS(fixture.context().synchronize());
  LinearVerifyRngStateV1 actual_state{};
  HostCoins actual{};
  CHECK(fixture.read_state(&actual_state));
  CHECK(fixture.read_coins(&actual));
  CHECK(actual_state.descriptor == invalid.descriptor);
  CHECK(actual_state.seed == invalid.seed);
  CHECK(actual_state.subsequence == invalid.subsequence);
  CHECK(actual_state.counter == invalid.counter);
  CHECK(actual.accept == std::vector<float>(3, kCoinSentinel));
  CHECK(actual.bonus[0] == kCoinSentinel);
  CHECK(actual.status[0] ==
        static_cast<uint32_t>(
            LinearVerifyRngDeviceCode::kInvalidStateDescriptor));

  const LinearVerifyRngStateV1 overflow = make_linear_verify_rng_state_v1(
      11U, 13U, std::numeric_limits<uint64_t>::max());
  CHECK(fixture.copy_state(overflow));
  CHECK(fixture.reset_coins());
  CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                 fixture.stateful_buffers()));
  CHECK_STATUS(fixture.context().synchronize());
  CHECK(fixture.read_state(&actual_state));
  CHECK(fixture.read_coins(&actual));
  CHECK(actual_state.counter == overflow.counter);
  CHECK(actual.accept == std::vector<float>(3, kCoinSentinel));
  CHECK(actual.bonus[0] == kCoinSentinel);
  CHECK(actual.status[0] ==
        static_cast<uint32_t>(LinearVerifyRngDeviceCode::kCounterOverflow));
  return true;
}

[[nodiscard]] bool SeededCudaMatchesMurmurOracle() {
  static_assert(reference_seeded_hash(0U, 0U, 0U) == 0x8134cdf8U);
  static_assert(reference_seeded_hash(1U, 1U, 0U) == 0x292acae6U);
  static_assert(reference_seeded_hash(0x0123456789abcdefULL,
                                      0xfedcba9876543210ULL,
                                      3U) == 0x0b979070U);

  constexpr std::array<uint32_t, 3> slot_counts{2U, 3U, 64U};
  constexpr std::array<std::array<uint64_t, 2>, 4> inputs{
      {{0U, 0U},
       {1U, 1U},
       {12345U, 7U},
       {0x0123456789abcdefULL, 0xfedcba9876543210ULL}}};
  for (uint32_t slots : slot_counts) {
    VerifyFixture fixture(slots, 8);
    CHECK(fixture.initialize());
    for (const auto &input : inputs) {
      CHECK(fixture.copy_seeded_inputs(input[0], input[1]));
      CHECK(fixture.reset_coins());
      CHECK_STATUS(launch_seeded_linear_verify_rng(fixture.context(),
                                                   fixture.seeded_buffers()));
      CHECK_STATUS(fixture.context().synchronize());
      HostCoins actual{};
      CHECK(fixture.read_coins(&actual));
      CHECK(actual.status[0] == 0U);
      for (uint32_t slot = 0; slot < slots; ++slot) {
        CHECK(actual.accept[slot] ==
              reference_seeded_coin(input[0], input[1], slot));
        CHECK(std::isfinite(actual.accept[slot]));
        CHECK(actual.accept[slot] >= 0.0F);
        CHECK(actual.accept[slot] < 1.0F);
      }
      CHECK(actual.bonus[0] ==
            reference_seeded_coin(input[0], input[1], slots));
      CHECK(actual.bonus[0] >= 0.0F);
      CHECK(actual.bonus[0] < 1.0F);
    }
  }
  return true;
}

[[nodiscard]] bool StatefulCudaMatchesPhiloxOracle() {
  constexpr PhiloxBlock zero = reference_philox4x32_10(0U, 0U, 0U);
  static_assert(zero.x == 0x6627e8d5U);
  static_assert(zero.y == 0xe169c58dU);
  static_assert(zero.z == 0xbc57ac4cU);
  static_assert(zero.w == 0x9b00dbd8U);

  constexpr std::array<uint32_t, 3> slot_counts{2U, 3U, 64U};
  for (uint32_t slots : slot_counts) {
    VerifyFixture fixture(slots, 8);
    CHECK(fixture.initialize());
    const LinearVerifyRngStateV1 initial = make_linear_verify_rng_state_v1(
        0x0123456789abcdefULL, 0xfedcba9876543210ULL, 17U);
    CHECK(fixture.copy_state(initial));
    CHECK(fixture.reset_coins());
    CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                   fixture.stateful_buffers()));
    CHECK_STATUS(fixture.context().synchronize());
    HostCoins actual{};
    LinearVerifyRngStateV1 advanced{};
    CHECK(fixture.read_coins(&actual));
    CHECK(fixture.read_state(&advanced));
    CHECK(coins_equal(actual, reference_stateful_coins(initial, slots)));
    const uint64_t groups = (slots + 4U) / 4U;
    CHECK(advanced.descriptor == initial.descriptor);
    CHECK(advanced.seed == initial.seed);
    CHECK(advanced.subsequence == initial.subsequence);
    CHECK(advanced.counter == initial.counter + groups);

    CHECK(fixture.reset_coins());
    CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                   fixture.stateful_buffers()));
    CHECK_STATUS(fixture.context().synchronize());
    HostCoins second{};
    LinearVerifyRngStateV1 second_state{};
    CHECK(fixture.read_coins(&second));
    CHECK(fixture.read_state(&second_state));
    CHECK(coins_equal(second, reference_stateful_coins(advanced, slots)));
    CHECK(second.accept != actual.accept || second.bonus != actual.bonus);
    CHECK(second_state.counter == initial.counter + 2U * groups);

    CHECK(fixture.copy_state(initial));
    CHECK(fixture.reset_coins());
    CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                   fixture.stateful_buffers()));
    CHECK_STATUS(fixture.context().synchronize());
    HostCoins reset{};
    CHECK(fixture.read_coins(&reset));
    CHECK(coins_equal(reset, actual));
  }
  return true;
}

void set_probability(std::vector<float> *probabilities, uint32_t row,
                     uint32_t vocab_size, uint32_t token, float probability) {
  (*probabilities)[static_cast<uint64_t>(row) * vocab_size + token] =
      probability;
}

struct RawGraph final {
  cudaGraph_t value = nullptr;

  ~RawGraph() noexcept {
    if (value != nullptr && cudaGraphDestroy(value) != cudaSuccess) {
      std::terminate();
    }
  }
};

[[nodiscard]] bool check_composed_result(const HostCoins &coins,
                                         const std::vector<int32_t> &tokens,
                                         const std::vector<int32_t> &indices,
                                         int32_t count, uint32_t status) {
  CHECK(status == 0U);
  CHECK(indices[0] == 0);
  const bool first_accept = coins.accept[0] < 0.5F;
  if (!first_accept) {
    CHECK(tokens == std::vector<int32_t>({42, 0, 0}));
    CHECK(indices == std::vector<int32_t>({0, -1, -1}));
    CHECK(count == 0);
    return true;
  }
  const bool second_accept = coins.accept[1] < 0.5F;
  if (!second_accept) {
    CHECK(tokens == std::vector<int32_t>({12345, 99, 0}));
    CHECK(indices == std::vector<int32_t>({0, 1, -1}));
    CHECK(count == 1);
    return true;
  }
  CHECK(tokens == std::vector<int32_t>({12345, 67890, 247000}));
  CHECK(indices == std::vector<int32_t>({0, 1, 2}));
  CHECK(count == 2);
  return true;
}

[[nodiscard]] bool ProductionShapeGraphComposesRngAndSampler() {
  constexpr uint32_t kNumSlots = 3;
  constexpr uint32_t kVocabSize = 248320;
  VerifyFixture fixture(kNumSlots, kVocabSize);
  CHECK(fixture.initialize());
  const uintptr_t stable_coin_address = fixture.accept_uniforms_address();
  const uintptr_t stable_output_address = fixture.out_tokens_address();

  const std::array<int64_t, kNumSlots> proposals{7, 12345, 67890};
  const std::array<int64_t, kNumSlots> output_indices{0, 1, 2};
  std::vector<float> target(static_cast<uint64_t>(kNumSlots) * kVocabSize,
                            0.0F);
  std::vector<float> draft(static_cast<uint64_t>(kNumSlots - 1U) * kVocabSize,
                           0.0F);
  set_probability(&target, 0, kVocabSize, 12345, 0.5F);
  set_probability(&target, 0, kVocabSize, 42, 0.5F);
  set_probability(&draft, 0, kVocabSize, 12345, 1.0F);
  set_probability(&target, 1, kVocabSize, 67890, 0.5F);
  set_probability(&target, 1, kVocabSize, 99, 0.5F);
  set_probability(&draft, 1, kVocabSize, 67890, 1.0F);
  set_probability(&target, 2, kVocabSize, 247000, 1.0F);
  CHECK(fixture.copy_sampler_inputs(std::span<const int64_t>(proposals),
                                    std::span<const int64_t>(output_indices),
                                    std::span<const float>(target),
                                    std::span<const float>(draft)));

  const LinearVerifyRngStateV1 initial =
      make_linear_verify_rng_state_v1(0U, 0U, 0U);
  CHECK(fixture.copy_state(initial));
  CHECK(fixture.reset_coins());
  CHECK(fixture.reset_sampler_outputs());
  CHECK_STATUS(fixture.context().synchronize());

  RawGraph graph;
  CHECK_CUDA(cudaStreamBeginCapture(fixture.context().stream(),
                                    cudaStreamCaptureModeThreadLocal));
  CHECK(fixture.capture_sampler_output_initialization());
  CHECK_STATUS(launch_stateful_linear_verify_rng(fixture.context(),
                                                 fixture.stateful_buffers()));
  CHECK_STATUS(launch_linear_rejection_sampling_if_ready(
      fixture.context(), fixture.sampling_buffers()));
  CHECK_CUDA(cudaStreamEndCapture(fixture.context().stream(), &graph.value));

  std::optional<CudaGraphExecutable> executable;
  CHECK(take_result(CudaGraphExecutable::instantiate(
                        graph.value, fixture.context(), fixture.lease()),
                    &executable));
  CHECK_CUDA(cudaGraphDestroy(std::exchange(graph.value, nullptr)));

  HostCoins first_coins{};
  std::vector<int32_t> first_tokens;
  std::vector<int32_t> first_indices;
  int32_t first_count = -1;
  uint32_t first_status = kStatusSentinel;
  for (uint64_t replay = 0; replay < 3U; ++replay) {
    CHECK_STATUS(executable->launch());
    CHECK_STATUS(executable->synchronize());
    HostCoins coins{};
    std::vector<int32_t> tokens;
    std::vector<int32_t> indices;
    int32_t count = -1;
    uint32_t status = kStatusSentinel;
    CHECK(fixture.read_coins(&coins));
    CHECK(fixture.read_sampler_outputs(&tokens, &indices, &count, &status));
    const LinearVerifyRngStateV1 replay_state =
        make_linear_verify_rng_state_v1(0U, 0U, replay);
    CHECK(
        coins_equal(coins, reference_stateful_coins(replay_state, kNumSlots)));
    CHECK(check_composed_result(coins, tokens, indices, count, status));
    CHECK(fixture.accept_uniforms_address() == stable_coin_address);
    CHECK(fixture.out_tokens_address() == stable_output_address);
    if (replay == 0U) {
      first_coins = coins;
      first_tokens = tokens;
      first_indices = indices;
      first_count = count;
      first_status = status;
    }
  }

  LinearVerifyRngStateV1 advanced{};
  CHECK(fixture.read_state(&advanced));
  CHECK(advanced.counter == 3U);

  CHECK(fixture.copy_state(initial));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  HostCoins reset_coins{};
  std::vector<int32_t> reset_tokens;
  std::vector<int32_t> reset_indices;
  int32_t reset_count = -1;
  uint32_t reset_status = kStatusSentinel;
  CHECK(fixture.read_coins(&reset_coins));
  CHECK(fixture.read_sampler_outputs(&reset_tokens, &reset_indices,
                                     &reset_count, &reset_status));
  CHECK(coins_equal(reset_coins, first_coins));
  CHECK(reset_tokens == first_tokens);
  CHECK(reset_indices == first_indices);
  CHECK(reset_count == first_count);
  CHECK(reset_status == first_status);

  const HostCoins preserved_coins = reset_coins;
  const LinearVerifyRngStateV1 overflow = make_linear_verify_rng_state_v1(
      0U, 0U, std::numeric_limits<uint64_t>::max());
  CHECK(fixture.copy_state(overflow));
  CHECK_STATUS(executable->launch());
  CHECK_STATUS(executable->synchronize());
  HostCoins failed_coins{};
  std::vector<int32_t> failed_tokens;
  std::vector<int32_t> failed_indices;
  int32_t failed_count = -1;
  uint32_t failed_status = kStatusSentinel;
  CHECK(fixture.read_coins(&failed_coins));
  CHECK(fixture.read_sampler_outputs(&failed_tokens, &failed_indices,
                                     &failed_count, &failed_status));
  CHECK(failed_coins.accept == preserved_coins.accept);
  CHECK(failed_coins.bonus == preserved_coins.bonus);
  CHECK(failed_status ==
        static_cast<uint32_t>(LinearVerifyRngDeviceCode::kCounterOverflow));
  CHECK(failed_tokens == std::vector<int32_t>({0, 0, 0}));
  CHECK(failed_indices == std::vector<int32_t>({-1, -1, -1}));
  CHECK(failed_count == 0);
  CHECK(fixture.read_state(&advanced));
  CHECK(advanced.counter == overflow.counter);

  CHECK_STATUS(executable->close());
  executable.reset();
  return true;
}

struct TestCase final {
  const char *name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"MetadataAndDeviceFailuresAreClosed", MetadataAndDeviceFailuresAreClosed},
    {"SeededCudaMatchesMurmurOracle", SeededCudaMatchesMurmurOracle},
    {"StatefulCudaMatchesPhiloxOracle", StatefulCudaMatchesPhiloxOracle},
    {"ProductionShapeGraphComposesRngAndSampler",
     ProductionShapeGraphComposesRngAndSampler},
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
