#ifndef SGLANG_NATIVE_LINEAR_VERIFY_RNG_HPP_
#define SGLANG_NATIVE_LINEAR_VERIFY_RNG_HPP_

#include "sglang/native/cuda_graph_resources.hpp"

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace sglang::native {

inline constexpr uint32_t kLinearVerifyRngThreads = 128;
inline constexpr uint32_t kLinearVerifyRngMaxNumSlots = 64;
inline constexpr uint64_t kLinearVerifyRngStateDescriptorV1 =
    0x3156474e524c4753ULL; // Little-endian bytes: "SGLRNGV1".

enum class LinearVerifyRngArgument : uint32_t {
  kNone = 0,
  kNumSlots = 1,
  kSeed = 2,
  kSequenceLength = 3,
  kState = 4,
  kAcceptUniforms = 5,
  kBonusUniforms = 6,
  kDeviceStatus = 7,
};

enum class LinearVerifyRngDeviceCode : uint32_t {
  kOk = 0,
  kInvalidStateDescriptor = 0x00010001U,
  kCounterOverflow = 0x00010002U,
};

struct LinearVerifyRngShape final {
  uint64_t num_slots;
};

// The descriptor freezes both layout version 1 and the Philox4x32-10 mapping.
// Counter words are (counter_lo, counter_hi, subsequence_lo,
// subsequence_hi); key words are (seed_lo, seed_hi).
struct LinearVerifyRngStateV1 final {
  uint64_t descriptor;
  uint64_t seed;
  uint64_t subsequence;
  uint64_t counter;
};

using LinearVerifyConstUInt64Vector =
    GraphStableTensorView<DType::kUInt64, 1, TensorAccess::kReadOnly>;
using LinearVerifyMutableUInt64Vector =
    GraphStableTensorView<DType::kUInt64, 1, TensorAccess::kReadWrite>;
using LinearVerifyMutableFloat32Vector =
    GraphStableTensorView<DType::kFloat32, 1, TensorAccess::kReadWrite>;
using LinearVerifyMutableFloat32Matrix =
    GraphStableTensorView<DType::kFloat32, 2, TensorAccess::kReadWrite>;
using LinearVerifyMutableUInt32Vector =
    GraphStableTensorView<DType::kUInt32, 1, TensorAccess::kReadWrite>;

struct SeededLinearVerifyRngBuffers final {
  const LinearVerifyConstUInt64Vector &seed;
  const LinearVerifyConstUInt64Vector &sequence_length;
  const LinearVerifyMutableFloat32Matrix &accept_uniforms;
  const LinearVerifyMutableFloat32Vector &bonus_uniforms;
  const LinearVerifyMutableUInt32Vector &device_status;
};

struct StatefulLinearVerifyRngBuffers final {
  const LinearVerifyMutableUInt64Vector &state;
  const LinearVerifyMutableFloat32Matrix &accept_uniforms;
  const LinearVerifyMutableFloat32Vector &bonus_uniforms;
  const LinearVerifyMutableUInt32Vector &device_status;
};

[[nodiscard]] constexpr LinearVerifyRngStateV1
make_linear_verify_rng_state_v1(uint64_t seed, uint64_t subsequence,
                                uint64_t counter = 0) noexcept {
  return LinearVerifyRngStateV1{kLinearVerifyRngStateDescriptorV1, seed,
                                subsequence, counter};
}

[[nodiscard]] std::string_view
linear_verify_rng_argument_name(LinearVerifyRngArgument argument) noexcept;
[[nodiscard]] std::string_view
linear_verify_rng_device_code_name(LinearVerifyRngDeviceCode code) noexcept;

[[nodiscard]] NativeRuntimeError
validate_linear_verify_rng_shape(LinearVerifyRngShape shape) noexcept;

// Mirrors the established seeded SRT conversion: FP64 division by UINT32_MAX,
// conversion to FP32, then a clamp to the largest FP32 value below one.
[[nodiscard]] float linear_verify_seeded_coin_from_hash(uint32_t hash) noexcept;

// Metadata failures return synchronously. Device-state failures publish a
// LinearVerifyRngDeviceCode and preserve state plus both coin outputs.
[[nodiscard]] NativeRuntimeError launch_seeded_linear_verify_rng(
    const CudaExecutionContext &context,
    const SeededLinearVerifyRngBuffers &buffers) noexcept;
[[nodiscard]] NativeRuntimeError launch_stateful_linear_verify_rng(
    const CudaExecutionContext &context,
    const StatefulLinearVerifyRngBuffers &buffers) noexcept;

static_assert(sizeof(LinearVerifyRngShape) == 8);
static_assert(alignof(LinearVerifyRngShape) == 8);
static_assert(std::is_standard_layout_v<LinearVerifyRngShape>);
static_assert(std::is_trivially_copyable_v<LinearVerifyRngShape>);
static_assert(sizeof(LinearVerifyRngStateV1) == 32);
static_assert(alignof(LinearVerifyRngStateV1) == 8);
static_assert(std::is_standard_layout_v<LinearVerifyRngStateV1>);
static_assert(std::is_trivially_copyable_v<LinearVerifyRngStateV1>);
static_assert(static_cast<uint32_t>(LinearVerifyRngDeviceCode::kOk) == 0);

} // namespace sglang::native

#endif // SGLANG_NATIVE_LINEAR_VERIFY_RNG_HPP_
