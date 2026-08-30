#include "sglang/native/linear_rejection_sampling.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <type_traits>

namespace {

using sglang::native::ConstFloat32Matrix;
using sglang::native::ConstFloat32Tensor3;
using sglang::native::ConstFloat32Vector;
using sglang::native::ConstInt64Matrix;
using sglang::native::LinearRejectionSamplingArgument;
using sglang::native::LinearRejectionSamplingBuffers;
using sglang::native::LinearRejectionSamplingDeviceCode;
using sglang::native::LinearRejectionSamplingShape;
using sglang::native::MutableInt32Matrix;
using sglang::native::MutableInt32Vector;
using sglang::native::MutableUInt32Vector;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeOperation;
using sglang::native::is_ok;
using sglang::native::kLinearRejectionSamplingMaxNumSlots;
using sglang::native::kLinearRejectionSamplingThreads;
using sglang::native::linear_rejection_sampling_argument_name;
using sglang::native::linear_rejection_sampling_device_code_name;
using sglang::native::validate_linear_rejection_sampling_shape;

[[nodiscard]] bool record_check(bool passed, const char* expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) { \
      return false;                                                         \
    }                                                                       \
  } while (false)

[[nodiscard]] bool StableIdentifiers() {
  constexpr std::array<std::string_view, 13> argument_names{
      "none",
      "num_slots",
      "vocab_size",
      "out_tokens",
      "accept_indices",
      "num_correct_drafts",
      "proposal_tokens",
      "proposal_out_indices",
      "accept_uniforms",
      "bonus_uniforms",
      "target_probs",
      "draft_probs",
      "device_status"};
  constexpr std::array<std::string_view, 4> device_code_names{
      "ok",
      "proposal_out_index_out_of_range",
      "duplicate_proposal_out_index",
      "proposal_token_out_of_range"};

  for (uint32_t index = 0; index < argument_names.size(); ++index) {
    CHECK(linear_rejection_sampling_argument_name(
              static_cast<LinearRejectionSamplingArgument>(index)) ==
          argument_names[index]);
  }
  for (uint32_t index = 0; index < device_code_names.size(); ++index) {
    CHECK(linear_rejection_sampling_device_code_name(
              static_cast<LinearRejectionSamplingDeviceCode>(index)) ==
          device_code_names[index]);
  }
  CHECK(linear_rejection_sampling_argument_name(
            static_cast<LinearRejectionSamplingArgument>(99)) ==
        "invalid_linear_rejection_sampling_argument");
  CHECK(linear_rejection_sampling_device_code_name(
            static_cast<LinearRejectionSamplingDeviceCode>(99)) ==
        "invalid_linear_rejection_sampling_device_code");
  return true;
}

[[nodiscard]] bool ShapeContract() {
  CHECK(kLinearRejectionSamplingThreads == 256);
  CHECK(kLinearRejectionSamplingMaxNumSlots == 64);
  CHECK(is_ok(validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{3, 248320, 3})));
  CHECK(is_ok(validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{
          kLinearRejectionSamplingMaxNumSlots,
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
          kLinearRejectionSamplingMaxNumSlots})));

  const auto too_few = validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{1, 248320, 1});
  CHECK(too_few.code == NativeRuntimeCode::kInvalidArgument);
  CHECK(too_few.operation ==
        NativeRuntimeOperation::kValidateLinearRejectionSampling);
  CHECK(too_few.detail == static_cast<uint32_t>(
                              LinearRejectionSamplingArgument::kNumSlots));
  CHECK(too_few.actual == 1);
  CHECK(too_few.required == 2);

  const auto too_many = validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{
          kLinearRejectionSamplingMaxNumSlots + 1, 248320,
          kLinearRejectionSamplingMaxNumSlots + 1});
  CHECK(too_many.detail == static_cast<uint32_t>(
                               LinearRejectionSamplingArgument::kNumSlots));

  const auto no_vocab = validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{3, 0, 3});
  CHECK(no_vocab.detail == static_cast<uint32_t>(
                              LinearRejectionSamplingArgument::kVocabSize));
  CHECK(no_vocab.required == 1);

  const auto large_vocab = validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{
          3,
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) + 1,
          3});
  CHECK(large_vocab.detail == static_cast<uint32_t>(
                                 LinearRejectionSamplingArgument::kVocabSize));

  const auto wrong_output = validate_linear_rejection_sampling_shape(
      LinearRejectionSamplingShape{3, 248320, 4});
  CHECK(wrong_output.detail ==
        static_cast<uint32_t>(
            LinearRejectionSamplingArgument::kOutTokens));
  CHECK(wrong_output.actual == 4);
  CHECK(wrong_output.required == 3);
  return true;
}

static_assert(sizeof(LinearRejectionSamplingShape) == 24);
static_assert(std::is_trivially_copyable_v<LinearRejectionSamplingShape>);
static_assert(!std::is_default_constructible_v<MutableInt32Vector>);
static_assert(!std::is_default_constructible_v<MutableInt32Matrix>);
static_assert(!std::is_default_constructible_v<MutableUInt32Vector>);
static_assert(!std::is_default_constructible_v<ConstInt64Matrix>);
static_assert(!std::is_default_constructible_v<ConstFloat32Vector>);
static_assert(!std::is_default_constructible_v<ConstFloat32Matrix>);
static_assert(!std::is_default_constructible_v<ConstFloat32Tensor3>);
static_assert(std::is_aggregate_v<LinearRejectionSamplingBuffers>);

}  // namespace

int main() {
  if (!StableIdentifiers() || !ShapeContract()) {
    return 1;
  }
  std::printf("[  PASSED  ] 2 tests\n");
  return 0;
}
