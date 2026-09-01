#include "sglang/native/linear_verify_rng.hpp"

#include <algorithm>
#include <limits>

namespace sglang::native {
namespace {

[[nodiscard]] constexpr NativeRuntimeError
shape_error(LinearVerifyRngArgument argument, uint64_t actual,
            uint64_t required) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kValidateLinearVerifyRng,
                            0,
                            static_cast<uint32_t>(argument),
                            actual,
                            required};
}

} // namespace

std::string_view
linear_verify_rng_argument_name(LinearVerifyRngArgument argument) noexcept {
  switch (argument) {
  case LinearVerifyRngArgument::kNone:
    return "none";
  case LinearVerifyRngArgument::kNumSlots:
    return "num_slots";
  case LinearVerifyRngArgument::kSeed:
    return "seed";
  case LinearVerifyRngArgument::kSequenceLength:
    return "sequence_length";
  case LinearVerifyRngArgument::kState:
    return "state";
  case LinearVerifyRngArgument::kAcceptUniforms:
    return "accept_uniforms";
  case LinearVerifyRngArgument::kBonusUniforms:
    return "bonus_uniforms";
  case LinearVerifyRngArgument::kDeviceStatus:
    return "device_status";
  default:
    return "invalid_linear_verify_rng_argument";
  }
}

std::string_view
linear_verify_rng_device_code_name(LinearVerifyRngDeviceCode code) noexcept {
  switch (code) {
  case LinearVerifyRngDeviceCode::kOk:
    return "ok";
  case LinearVerifyRngDeviceCode::kInvalidStateDescriptor:
    return "invalid_state_descriptor";
  case LinearVerifyRngDeviceCode::kCounterOverflow:
    return "counter_overflow";
  default:
    return "invalid_linear_verify_rng_device_code";
  }
}

NativeRuntimeError
validate_linear_verify_rng_shape(LinearVerifyRngShape shape) noexcept {
  if (shape.num_slots < 2) {
    return shape_error(LinearVerifyRngArgument::kNumSlots, shape.num_slots, 2);
  }
  if (shape.num_slots > kLinearVerifyRngMaxNumSlots) {
    return shape_error(LinearVerifyRngArgument::kNumSlots, shape.num_slots,
                       kLinearVerifyRngMaxNumSlots);
  }
  return native_runtime_ok();
}

float linear_verify_seeded_coin_from_hash(uint32_t hash) noexcept {
  constexpr float kLargestCoin = 1.0F - 0x1p-24F;
  const double uniform =
      static_cast<double>(hash) /
      static_cast<double>(std::numeric_limits<uint32_t>::max());
  return std::min(static_cast<float>(uniform), kLargestCoin);
}

} // namespace sglang::native
