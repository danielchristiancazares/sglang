#include "sglang/native/cuda_graph_resources.hpp"
#include "sglang/native/dspark_cycle_controller.hpp"
#include "sglang/native/dspark_proposal.hpp"
#include "sglang/native/gdn_replayssm_commit.hpp"
#include "sglang/native/linear_rejection_sampling.hpp"
#include "sglang/native/linear_verify_rng.hpp"
#include "sglang/native/tensor_view.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  using sglang::native::capture_dspark_proposal_bfloat16_graph;
  using sglang::native::capture_gdn_replayssm_commit_graph;
  using sglang::native::
      capture_linear_rejection_sampling_from_bfloat16_logits_graph;
  using sglang::native::capture_seeded_linear_verify_rng_graph;
  using sglang::native::capture_stateful_linear_verify_rng_graph;
  using sglang::native::kDsparkCycleAbiVersion;
  using sglang::native::kDsparkCycleNumStages;
  using sglang::native::kDsparkCycleProductionNumOutTokens;
  using sglang::native::kDsparkProductionGamma;
  using sglang::native::kDsparkProductionVocabSize;
  using sglang::native::kLinearRejectionSamplingMaxNumSlots;
  using sglang::native::kLinearVerifyRngMaxNumSlots;
  using sglang::native::kLinearVerifyRngStateDescriptorV1;
  using sglang::native::launch_dspark_cycle_compact_result;
  using sglang::native::launch_dspark_proposal_bfloat16;
  using sglang::native::launch_dspark_proposal_bfloat16_if_ready;
  using sglang::native::launch_linear_rejection_sampling_if_ready;
  using sglang::native::launch_seeded_linear_verify_rng;
  using sglang::native::launch_seeded_linear_verify_rng_if_ready;
  using sglang::native::launch_stateful_linear_verify_rng;
  using sglang::native::launch_stateful_linear_verify_rng_if_ready;

  static_assert(kLinearRejectionSamplingMaxNumSlots ==
                kLinearVerifyRngMaxNumSlots);
  static_assert(kDsparkCycleProductionNumOutTokens ==
                kDsparkProductionGamma + 1);
  static_assert(kDsparkCycleNumStages == 8);

  using SeededLaunch = decltype(&launch_seeded_linear_verify_rng);
  using StatefulLaunch = decltype(&launch_stateful_linear_verify_rng);
  using SamplingLaunch = decltype(&launch_linear_rejection_sampling_if_ready);
  SeededLaunch volatile seeded_launch = &launch_seeded_linear_verify_rng;
  StatefulLaunch volatile stateful_launch = &launch_stateful_linear_verify_rng;
  SeededLaunch volatile guarded_seeded_launch =
      &launch_seeded_linear_verify_rng_if_ready;
  StatefulLaunch volatile guarded_stateful_launch =
      &launch_stateful_linear_verify_rng_if_ready;
  SamplingLaunch volatile sampling_launch =
      &launch_linear_rejection_sampling_if_ready;
  using DsparkLaunch = decltype(&launch_dspark_proposal_bfloat16);
  DsparkLaunch volatile dspark_launch = &launch_dspark_proposal_bfloat16;
  DsparkLaunch volatile guarded_dspark_launch =
      &launch_dspark_proposal_bfloat16_if_ready;
  using CompactLaunch = decltype(&launch_dspark_cycle_compact_result);
  CompactLaunch volatile compact_launch = &launch_dspark_cycle_compact_result;
  using CaptureDspark = decltype(&capture_dspark_proposal_bfloat16_graph);
  CaptureDspark volatile capture_dspark =
      &capture_dspark_proposal_bfloat16_graph;
  using CaptureSeededRng = decltype(&capture_seeded_linear_verify_rng_graph);
  using CaptureStatefulRng =
      decltype(&capture_stateful_linear_verify_rng_graph);
  using CaptureRejection =
      decltype(&capture_linear_rejection_sampling_from_bfloat16_logits_graph);
  using CaptureReplaySsm = decltype(&capture_gdn_replayssm_commit_graph);
  CaptureSeededRng volatile capture_seeded_rng =
      &capture_seeded_linear_verify_rng_graph;
  CaptureStatefulRng volatile capture_stateful_rng =
      &capture_stateful_linear_verify_rng_graph;
  CaptureRejection volatile capture_rejection =
      &capture_linear_rejection_sampling_from_bfloat16_logits_graph;
  CaptureReplaySsm volatile capture_replayssm =
      &capture_gdn_replayssm_commit_graph;

  if (seeded_launch == nullptr || stateful_launch == nullptr ||
      guarded_seeded_launch == nullptr || guarded_stateful_launch == nullptr ||
      sampling_launch == nullptr || dspark_launch == nullptr ||
      guarded_dspark_launch == nullptr || compact_launch == nullptr ||
      capture_dspark == nullptr || capture_seeded_rng == nullptr ||
      capture_stateful_rng == nullptr || capture_rejection == nullptr ||
      capture_replayssm == nullptr) {
    return 1;
  }

  int cuda_runtime_version = 0;
  const cudaError_t version_status =
      cudaRuntimeGetVersion(&cuda_runtime_version);
  if (version_status != cudaSuccess) {
    std::fprintf(stderr, "cudaRuntimeGetVersion failed: %s\n",
                 cudaGetErrorString(version_status));
    return 2;
  }

  std::printf(
      "{\"name\":\"sglang-native\",\"tensor_abi\":\"%u.%u\","
      "\"metadata_bytes\":%zu,\"view_bytes\":%zu,"
      "\"max_linear_verify_slots\":%u,"
      "\"dspark_production_gamma\":%u,"
      "\"dspark_production_vocab_size\":%u,"
      "\"dspark_cycle_abi_version\":%u,"
      "\"rng_state_descriptor\":\"0x%016llx\","
      "\"cuda_headers_version\":%d,"
      "\"cuda_runtime_reported_version\":%d,"
      "\"capabilities\":[\"graph_arena\",\"linear_verify_rng\","
      "\"linear_rejection_sampling\",\"rng_sampler_graph_gate\","
      "\"dspark_proposal\",\"dspark_proposal_graph\","
      "\"linear_verify_rng_graph\","
      "\"corrected_logit_rejection_graph\","
      "\"gdn_replayssm_graph\","
      "\"dspark_compact_result_graph\","
      "\"dspark_cycle_controller\"]}\n",
      SGL_NATIVE_TENSOR_ABI_MAJOR, SGL_NATIVE_TENSOR_ABI_MINOR,
      sizeof(SglNativeTensorMetadataV1), sizeof(SglNativeConstTensorViewV1),
      kLinearVerifyRngMaxNumSlots, kDsparkProductionGamma,
      kDsparkProductionVocabSize, kDsparkCycleAbiVersion,
      static_cast<unsigned long long>(kLinearVerifyRngStateDescriptorV1),
      CUDART_VERSION, cuda_runtime_version);
  return 0;
}
