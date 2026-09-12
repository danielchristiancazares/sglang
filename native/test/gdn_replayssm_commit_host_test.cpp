#include "sglang/native/gdn_replayssm_commit.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

namespace {

using sglang::native::gdn_replayssm_argument_name;
using sglang::native::gdn_replayssm_device_code_name;
using sglang::native::GdnReplaySsmArgument;
using sglang::native::GdnReplaySsmDeviceCode;
using sglang::native::GdnReplaySsmShape;
using sglang::native::is_ok;
using sglang::native::kGdnProductionKeyDimension;
using sglang::native::kGdnProductionNumKeyHeads;
using sglang::native::kGdnProductionNumLayers;
using sglang::native::kGdnProductionNumValueHeads;
using sglang::native::kGdnProductionReplayLength;
using sglang::native::kGdnProductionValueDimension;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeOperation;
using sglang::native::validate_gdn_production_replayssm_shape;
using sglang::native::validate_gdn_replayssm_shape;

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

[[nodiscard]] constexpr GdnReplaySsmShape
production_shape(uint64_t slots = 5) noexcept {
  return GdnReplaySsmShape{kGdnProductionNumLayers,
                           slots,
                           1,
                           kGdnProductionNumValueHeads,
                           kGdnProductionNumKeyHeads,
                           kGdnProductionKeyDimension,
                           kGdnProductionValueDimension,
                           kGdnProductionReplayLength};
}

[[nodiscard]] bool IdentifiersAreStable() {
  constexpr std::array<std::string_view, 15> argument_names{
      "none",           "shape",
      "temporal",       "raw_values",
      "raw_keys",       "log_decay",
      "beta",           "state_indices",
      "accept_lengths", "last_correct_steps",
      "track_indices",  "track_steps",
      "conv_states",    "intermediate_conv_windows",
      "device_status"};
  for (uint32_t index = 0; index < argument_names.size(); ++index) {
    CHECK(gdn_replayssm_argument_name(static_cast<GdnReplaySsmArgument>(
              index)) == argument_names[index]);
  }
  CHECK(gdn_replayssm_argument_name(static_cast<GdnReplaySsmArgument>(99)) ==
        "invalid_gdn_replayssm_argument");
  CHECK(gdn_replayssm_device_code_name(GdnReplaySsmDeviceCode::kOk) == "ok");
  CHECK(gdn_replayssm_device_code_name(
            GdnReplaySsmDeviceCode::kTrackStepOutOfRange) ==
        "track_step_out_of_range");
  CHECK(gdn_replayssm_device_code_name(static_cast<GdnReplaySsmDeviceCode>(
            99)) == "invalid_gdn_replayssm_device_code");
  return true;
}

[[nodiscard]] bool GeneralShapeBoundsAreValidated() {
  constexpr GdnReplaySsmShape general{2, 7, 3, 6, 2, 128, 96, 8};
  CHECK(is_ok(validate_gdn_replayssm_shape(general)));

  auto invalid = general;
  invalid.num_layers = 0;
  auto error = validate_gdn_replayssm_shape(invalid);
  CHECK(error.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(error.operation == NativeRuntimeOperation::kValidateGdnReplaySsmCommit);
  CHECK(error.detail == static_cast<uint32_t>(GdnReplaySsmArgument::kShape));
  CHECK(error.required == 1);

  invalid = general;
  invalid.batch_size = 8;
  CHECK(validate_gdn_replayssm_shape(invalid).required == 7);
  invalid = general;
  invalid.num_value_heads = 5;
  CHECK(validate_gdn_replayssm_shape(invalid).actual == 5);
  invalid = general;
  invalid.key_dimension = 129;
  CHECK(validate_gdn_replayssm_shape(invalid).required == 128);
  invalid = general;
  invalid.value_dimension = 1025;
  CHECK(validate_gdn_replayssm_shape(invalid).required == 1024);
  invalid = general;
  invalid.replay_length = 65;
  CHECK(validate_gdn_replayssm_shape(invalid).required == 64);
  invalid = GdnReplaySsmShape{65, 1, 1, 1024, 1, 128, 1, 1};
  CHECK(validate_gdn_replayssm_shape(invalid).required == 1008);
  return true;
}

[[nodiscard]] bool ProductionShapeIsClosed() {
  CHECK(is_ok(validate_gdn_production_replayssm_shape(production_shape())));
  auto invalid = production_shape();
  invalid.num_layers = kGdnProductionNumLayers - 1;
  const auto layer_error = validate_gdn_production_replayssm_shape(invalid);
  CHECK(layer_error.actual == kGdnProductionNumLayers - 1);
  CHECK(layer_error.required == kGdnProductionNumLayers);
  invalid = production_shape();
  invalid.batch_size = 2;
  const auto batch_error = validate_gdn_production_replayssm_shape(invalid);
  CHECK(batch_error.actual == 2);
  CHECK(batch_error.required == 1);
  invalid = production_shape();
  invalid.replay_length = kGdnProductionReplayLength - 1;
  const auto replay_error = validate_gdn_production_replayssm_shape(invalid);
  CHECK(replay_error.actual == kGdnProductionReplayLength - 1);
  CHECK(replay_error.required == kGdnProductionReplayLength);
  return true;
}

struct TestCase final {
  const char *name;
  bool (*function)();
};

constexpr TestCase kTests[]{
    {"IdentifiersAreStable", IdentifiersAreStable},
    {"GeneralShapeBoundsAreValidated", GeneralShapeBoundsAreValidated},
    {"ProductionShapeIsClosed", ProductionShapeIsClosed},
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
