#include "sglang/native/linear_rejection_sampling.hpp"
#include "sglang/native/linear_verify_rng.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using sglang::native::kLinearVerifyRngMaxNumSlots;
using sglang::native::kLinearVerifyRngStateDescriptorV1;
using sglang::native::linear_verify_rng_argument_name;
using sglang::native::linear_verify_rng_device_code_name;
using sglang::native::linear_verify_seeded_coin_from_hash;
using sglang::native::LinearVerifyRngArgument;
using sglang::native::LinearVerifyRngDeviceCode;
using sglang::native::LinearVerifyRngShape;
using sglang::native::LinearVerifyRngStateV1;
using sglang::native::make_linear_verify_rng_state_v1;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeOperation;
using sglang::native::validate_linear_verify_rng_shape;

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

[[nodiscard]] bool IdentifiersAndStateLayoutAreStable() {
  constexpr std::array<std::string_view, 8> argument_names{
      "none",  "num_slots",       "seed",           "sequence_length",
      "state", "accept_uniforms", "bonus_uniforms", "device_status"};
  for (uint32_t index = 0; index < argument_names.size(); ++index) {
    CHECK(linear_verify_rng_argument_name(static_cast<LinearVerifyRngArgument>(
              index)) == argument_names[index]);
  }
  CHECK(linear_verify_rng_argument_name(static_cast<LinearVerifyRngArgument>(
            99)) == "invalid_linear_verify_rng_argument");
  CHECK(linear_verify_rng_device_code_name(LinearVerifyRngDeviceCode::kOk) ==
        "ok");
  CHECK(linear_verify_rng_device_code_name(
            LinearVerifyRngDeviceCode::kInvalidStateDescriptor) ==
        "invalid_state_descriptor");
  CHECK(linear_verify_rng_device_code_name(
            LinearVerifyRngDeviceCode::kCounterOverflow) == "counter_overflow");
  CHECK(linear_verify_rng_device_code_name(
            static_cast<LinearVerifyRngDeviceCode>(99)) ==
        "invalid_linear_verify_rng_device_code");

  constexpr LinearVerifyRngStateV1 state = make_linear_verify_rng_state_v1(
      0x0123456789abcdefULL, 0xfedcba9876543210ULL, 17);
  CHECK(state.descriptor == kLinearVerifyRngStateDescriptorV1);
  CHECK(state.seed == 0x0123456789abcdefULL);
  CHECK(state.subsequence == 0xfedcba9876543210ULL);
  CHECK(state.counter == 17);
  return true;
}

[[nodiscard]] bool ShapeValidationIsClosed() {
  const auto too_small =
      validate_linear_verify_rng_shape(LinearVerifyRngShape{1});
  CHECK(too_small.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(too_small.operation ==
        NativeRuntimeOperation::kValidateLinearVerifyRng);
  CHECK(too_small.detail ==
        static_cast<uint32_t>(LinearVerifyRngArgument::kNumSlots));
  CHECK(too_small.actual == 1);
  CHECK(too_small.required == 2);

  const auto minimum =
      validate_linear_verify_rng_shape(LinearVerifyRngShape{2});
  CHECK(minimum.code == NativeRuntimeCode::kOk);
  const auto maximum = validate_linear_verify_rng_shape(
      LinearVerifyRngShape{kLinearVerifyRngMaxNumSlots});
  CHECK(maximum.code == NativeRuntimeCode::kOk);

  const auto too_large = validate_linear_verify_rng_shape(
      LinearVerifyRngShape{kLinearVerifyRngMaxNumSlots + 1ULL});
  CHECK(too_large.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(too_large.operation ==
        NativeRuntimeOperation::kValidateLinearVerifyRng);
  CHECK(too_large.detail ==
        static_cast<uint32_t>(LinearVerifyRngArgument::kNumSlots));
  CHECK(too_large.actual == kLinearVerifyRngMaxNumSlots + 1ULL);
  CHECK(too_large.required == kLinearVerifyRngMaxNumSlots);
  return true;
}

[[nodiscard]] bool SeededHashConversionIsHalfOpen() {
  constexpr float kLargestCoin = 1.0F - 0x1p-24F;
  CHECK(linear_verify_seeded_coin_from_hash(0U) == 0.0F);
  CHECK(linear_verify_seeded_coin_from_hash(1U) > 0.0F);
  CHECK(linear_verify_seeded_coin_from_hash(
            std::numeric_limits<uint32_t>::max()) == kLargestCoin);
  for (uint32_t distance = 0; distance < 129U; ++distance) {
    const uint32_t hash = std::numeric_limits<uint32_t>::max() - distance;
    const float coin = linear_verify_seeded_coin_from_hash(hash);
    CHECK(std::isfinite(coin));
    CHECK(coin >= 0.0F);
    CHECK(coin < 1.0F);
  }
  return true;
}

static_assert(sizeof(LinearVerifyRngStateV1) == 32);
static_assert(alignof(LinearVerifyRngStateV1) == 8);
static_assert(std::is_standard_layout_v<LinearVerifyRngStateV1>);
static_assert(std::is_trivially_copyable_v<LinearVerifyRngStateV1>);
static_assert(static_cast<uint32_t>(LinearVerifyRngDeviceCode::kOk) == 0U);
static_assert(
    static_cast<uint32_t>(LinearVerifyRngDeviceCode::kInvalidStateDescriptor) >
    static_cast<uint32_t>(sglang::native::LinearRejectionSamplingDeviceCode::
                              kProposalTokenOutOfRange));

} // namespace

int main() {
  if (!IdentifiersAndStateLayoutAreStable() || !ShapeValidationIsClosed() ||
      !SeededHashConversionIsHalfOpen()) {
    return 1;
  }
  std::printf("[  PASSED  ] 3 tests\n");
  return 0;
}
