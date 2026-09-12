#include "sglang/native/gdn_replayssm_commit.hpp"

#include <iterator>
#include <limits>

namespace sglang::native {
namespace {

[[nodiscard]] constexpr NativeRuntimeError
shape_error(GdnReplaySsmArgument argument, uint64_t actual,
            uint64_t required) noexcept {
  return NativeRuntimeError{NativeRuntimeCode::kInvalidArgument,
                            NativeRuntimeOperation::kValidateGdnReplaySsmCommit,
                            0,
                            static_cast<uint32_t>(argument),
                            actual,
                            required};
}

} // namespace

std::string_view
gdn_replayssm_argument_name(GdnReplaySsmArgument argument) noexcept {
  switch (argument) {
  case GdnReplaySsmArgument::kNone:
    return "none";
  case GdnReplaySsmArgument::kShape:
    return "shape";
  case GdnReplaySsmArgument::kTemporal:
    return "temporal";
  case GdnReplaySsmArgument::kRawValue:
    return "raw_values";
  case GdnReplaySsmArgument::kRawKey:
    return "raw_keys";
  case GdnReplaySsmArgument::kLogDecay:
    return "log_decay";
  case GdnReplaySsmArgument::kBeta:
    return "beta";
  case GdnReplaySsmArgument::kStateIndices:
    return "state_indices";
  case GdnReplaySsmArgument::kAcceptLengths:
    return "accept_lengths";
  case GdnReplaySsmArgument::kLastCorrectSteps:
    return "last_correct_steps";
  case GdnReplaySsmArgument::kTrackIndices:
    return "track_indices";
  case GdnReplaySsmArgument::kTrackSteps:
    return "track_steps";
  case GdnReplaySsmArgument::kConvStates:
    return "conv_states";
  case GdnReplaySsmArgument::kIntermediateConv:
    return "intermediate_conv_windows";
  case GdnReplaySsmArgument::kDeviceStatus:
    return "device_status";
  default:
    return "invalid_gdn_replayssm_argument";
  }
}

std::string_view
gdn_replayssm_device_code_name(GdnReplaySsmDeviceCode code) noexcept {
  switch (code) {
  case GdnReplaySsmDeviceCode::kOk:
    return "ok";
  case GdnReplaySsmDeviceCode::kStateIndexOutOfRange:
    return "state_index_out_of_range";
  case GdnReplaySsmDeviceCode::kAcceptLengthOutOfRange:
    return "accept_length_out_of_range";
  case GdnReplaySsmDeviceCode::kLastCorrectStepOutOfRange:
    return "last_correct_step_out_of_range";
  case GdnReplaySsmDeviceCode::kTrackIndexOutOfRange:
    return "track_index_out_of_range";
  case GdnReplaySsmDeviceCode::kTrackStepOutOfRange:
    return "track_step_out_of_range";
  default:
    return "invalid_gdn_replayssm_device_code";
  }
}

NativeRuntimeError
validate_gdn_replayssm_shape(GdnReplaySsmShape shape) noexcept {
  const uint64_t fields[]{shape.num_layers,      shape.num_slots,
                          shape.batch_size,      shape.num_value_heads,
                          shape.num_key_heads,   shape.key_dimension,
                          shape.value_dimension, shape.replay_length};
  for (uint32_t field = 0; field < std::size(fields); ++field) {
    if (fields[field] == 0) {
      return shape_error(GdnReplaySsmArgument::kShape, field, 1);
    }
  }
  if (shape.batch_size > shape.num_slots || shape.batch_size > 65535) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.batch_size,
                       shape.batch_size > shape.num_slots ? shape.num_slots
                                                          : 65535);
  }
  if (shape.num_value_heads % shape.num_key_heads != 0) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_value_heads,
                       shape.num_key_heads);
  }
  if (shape.key_dimension > 128) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.key_dimension, 128);
  }
  if (shape.value_dimension > 1024) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.value_dimension,
                       1024);
  }
  if (shape.replay_length > 64) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.replay_length, 64);
  }
  if (shape.num_layers > 1024) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_layers, 1024);
  }
  if (shape.num_value_heads > 65535 / shape.num_layers) {
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_value_heads,
                       65535 / shape.num_layers);
  }
  if (shape.num_slots >
      static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    return shape_error(
        GdnReplaySsmArgument::kShape, shape.num_slots,
        static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
  }
  return native_runtime_ok();
}

NativeRuntimeError
validate_gdn_production_replayssm_shape(GdnReplaySsmShape shape) noexcept {
  const NativeRuntimeError general = validate_gdn_replayssm_shape(shape);
  if (!is_ok(general)) {
    return general;
  }
  const GdnReplaySsmShape production{kGdnProductionNumLayers,
                                     shape.num_slots,
                                     1,
                                     kGdnProductionNumValueHeads,
                                     kGdnProductionNumKeyHeads,
                                     kGdnProductionKeyDimension,
                                     kGdnProductionValueDimension,
                                     kGdnProductionReplayLength};
  if (shape.num_layers != production.num_layers)
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_layers,
                       production.num_layers);
  if (shape.batch_size != production.batch_size)
    return shape_error(GdnReplaySsmArgument::kShape, shape.batch_size,
                       production.batch_size);
  if (shape.num_value_heads != production.num_value_heads)
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_value_heads,
                       production.num_value_heads);
  if (shape.num_key_heads != production.num_key_heads)
    return shape_error(GdnReplaySsmArgument::kShape, shape.num_key_heads,
                       production.num_key_heads);
  if (shape.key_dimension != production.key_dimension)
    return shape_error(GdnReplaySsmArgument::kShape, shape.key_dimension,
                       production.key_dimension);
  if (shape.value_dimension != production.value_dimension)
    return shape_error(GdnReplaySsmArgument::kShape, shape.value_dimension,
                       production.value_dimension);
  if (shape.replay_length != production.replay_length)
    return shape_error(GdnReplaySsmArgument::kShape, shape.replay_length,
                       production.replay_length);
  return native_runtime_ok();
}

} // namespace sglang::native
