#include "sglang/native/dspark_cycle_controller.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>

namespace {

using sglang::native::dspark_cycle_stage_name;
using sglang::native::dspark_model_graph_kind_name;
using sglang::native::dspark_production_model_graph_shape;
using sglang::native::dspark_required_model_graph_outputs;
using sglang::native::DsparkCycleArgument;
using sglang::native::DsparkCycleChildGraph;
using sglang::native::DsparkCycleCoreGraphs;
using sglang::native::DsparkCycleDescriptor;
using sglang::native::DsparkCycleResultV1;
using sglang::native::DsparkCycleStage;
using sglang::native::DsparkModelGraphBinding;
using sglang::native::DsparkModelGraphKind;
using sglang::native::DsparkModelGraphOutput;
using sglang::native::DsparkModelGraphProductionShape;
using sglang::native::DsparkModelGraphResourceLease;
using sglang::native::DsparkModelStorageArena;
using sglang::native::kDsparkCycleAbiVersion;
using sglang::native::make_dspark_cycle_child_graphs;
using sglang::native::NativeRuntimeCode;
using sglang::native::validate_dspark_cycle_descriptor;

[[nodiscard]] bool check(bool condition, const char *expression, int line) {
  if (!condition) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return condition;
}

#define CHECK(expression)                                                      \
  do {                                                                         \
    if (!check(static_cast<bool>(expression), #expression, __LINE__)) {        \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] std::array<DsparkCycleChildGraph, 8> valid_stages() {
  return make_dspark_cycle_child_graphs(DsparkCycleCoreGraphs{
      reinterpret_cast<cudaGraph_t>(1), reinterpret_cast<cudaGraph_t>(2),
      reinterpret_cast<cudaGraph_t>(3), reinterpret_cast<cudaGraph_t>(4),
      reinterpret_cast<cudaGraph_t>(5), reinterpret_cast<cudaGraph_t>(6),
      reinterpret_cast<cudaGraph_t>(7), reinterpret_cast<cudaGraph_t>(8)});
}

[[nodiscard]] bool DescriptorValidation() {
  auto stages = valid_stages();
  uint32_t device_result[sizeof(DsparkCycleResultV1) / sizeof(uint32_t)]{};
  uint32_t device_status = 0;
  DsparkCycleDescriptor descriptor{kDsparkCycleAbiVersion, 0, stages,
                                   &device_status, device_result};
  CHECK(validate_dspark_cycle_descriptor(descriptor).code ==
        NativeRuntimeCode::kOk);

  descriptor.abi_version = 0;
  auto status = validate_dspark_cycle_descriptor(descriptor);
  CHECK(status.detail ==
        static_cast<uint32_t>(DsparkCycleArgument::kAbiVersion));
  descriptor.abi_version = kDsparkCycleAbiVersion;

  stages[2].stage = DsparkCycleStage::kDraftExtend;
  status = validate_dspark_cycle_descriptor(descriptor);
  CHECK(status.detail == static_cast<uint32_t>(DsparkCycleArgument::kStages));
  stages = valid_stages();
  descriptor.stages = stages;
  stages[4].graph = nullptr;
  status = validate_dspark_cycle_descriptor(descriptor);
  CHECK(status.detail == static_cast<uint32_t>(DsparkCycleArgument::kGraph));
  stages = valid_stages();
  descriptor.stages = stages;
  descriptor.device_result = nullptr;
  status = validate_dspark_cycle_descriptor(descriptor);
  CHECK(status.detail ==
        static_cast<uint32_t>(DsparkCycleArgument::kDeviceResult));
  descriptor.device_result = device_result;
  descriptor.device_status = nullptr;
  status = validate_dspark_cycle_descriptor(descriptor);
  CHECK(status.detail ==
        static_cast<uint32_t>(DsparkCycleArgument::kDeviceStatus));
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kResourceOwner)) == "resource_owner");
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kCycleArena)) == "cycle_arena");
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kRequiredOutputs)) == "required_outputs");
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kProductionShape)) == "production_shape");
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kModelKind)) == "model_kind");
  CHECK(std::string_view(sglang::native::dspark_cycle_argument_name(
            DsparkCycleArgument::kStableAddressProvenance)) ==
        "stable_address_provenance");
  CHECK(sizeof(DsparkCycleResultV1) == 6 * sizeof(uint32_t));
  return true;
}

[[nodiscard]] bool ModelGraphContractIsProductionSpecific() {
  constexpr DsparkModelGraphProductionShape shape =
      dspark_production_model_graph_shape();
  CHECK(shape.batch_size == 1U);
  CHECK(shape.gamma == 7U);
  CHECK(shape.verify_width == 8U);
  CHECK(shape.vocabulary_size == 248320U);
  CHECK(shape.hidden_size == 5120U);
  CHECK(shape.target_capture_layers == 5U);
  CHECK(shape.target_hidden_width == 25600U);
  CHECK(shape.reserved == 0U);
  CHECK(dspark_required_model_graph_outputs(
            DsparkModelGraphKind::kTargetVerify) ==
        (DsparkModelGraphOutput::kProposalLayout |
         DsparkModelGraphOutput::kTargetProbabilities |
         DsparkModelGraphOutput::kTargetHidden |
         DsparkModelGraphOutput::kTargetKv |
         DsparkModelGraphOutput::kReplaySsmInputs));
  CHECK(
      dspark_required_model_graph_outputs(DsparkModelGraphKind::kDraftExtend) ==
      (DsparkModelGraphOutput::kDraftHidden |
       DsparkModelGraphOutput::kNextBaseLogits));
  CHECK(dspark_required_model_graph_outputs(DsparkModelGraphKind::kKvWrite) ==
        (DsparkModelGraphOutput::kDraftKv |
         DsparkModelGraphOutput::kPrefixValidCommit));
  CHECK(std::string_view(dspark_model_graph_kind_name(
            DsparkModelGraphKind::kTargetVerify)) == "target_verify");
  CHECK(std::string_view(dspark_model_graph_kind_name(
            DsparkModelGraphKind::kDraftExtend)) == "draft_extend");
  CHECK(std::string_view(dspark_model_graph_kind_name(
            DsparkModelGraphKind::kKvWrite)) == "kv_write");
  CHECK(std::string_view(
            dspark_model_graph_kind_name(DsparkModelGraphKind::kInvalid)) ==
        "invalid_dspark_model_graph_kind");
  CHECK(std::is_default_constructible_v<DsparkModelGraphResourceLease>);
  CHECK(std::is_copy_constructible_v<DsparkModelGraphResourceLease>);
  CHECK(std::is_nothrow_move_constructible_v<DsparkModelGraphResourceLease>);
  CHECK(std::is_nothrow_move_constructible_v<DsparkModelStorageArena>);
  CHECK(!std::is_copy_constructible_v<DsparkModelStorageArena>);
  CHECK(!std::is_trivially_copyable_v<DsparkModelGraphBinding>);
  return true;
}

} // namespace

int main() {
  if (!DescriptorValidation() || !ModelGraphContractIsProductionSpecific()) {
    return 1;
  }
  std::printf("[  PASSED  ] 2 tests\n");
  return 0;
}
