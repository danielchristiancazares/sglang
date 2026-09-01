#include "sglang/native/cuda_graph_resources.hpp"
#include "sglang/native/linear_rejection_sampling.hpp"
#include "sglang/native/linear_verify_rng.hpp"
#include "sglang/native/tensor_view.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
  using sglang::native::kLinearRejectionSamplingMaxNumSlots;
  using sglang::native::kLinearVerifyRngMaxNumSlots;
  using sglang::native::kLinearVerifyRngStateDescriptorV1;
  using sglang::native::launch_linear_rejection_sampling_if_ready;
  using sglang::native::launch_seeded_linear_verify_rng;
  using sglang::native::launch_stateful_linear_verify_rng;

  static_assert(kLinearRejectionSamplingMaxNumSlots ==
                kLinearVerifyRngMaxNumSlots);

  using SeededLaunch = decltype(&launch_seeded_linear_verify_rng);
  using StatefulLaunch = decltype(&launch_stateful_linear_verify_rng);
  using SamplingLaunch = decltype(&launch_linear_rejection_sampling_if_ready);
  SeededLaunch volatile seeded_launch = &launch_seeded_linear_verify_rng;
  StatefulLaunch volatile stateful_launch = &launch_stateful_linear_verify_rng;
  SamplingLaunch volatile sampling_launch =
      &launch_linear_rejection_sampling_if_ready;

  if (seeded_launch == nullptr || stateful_launch == nullptr ||
      sampling_launch == nullptr) {
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
      "\"rng_state_descriptor\":\"0x%016llx\","
      "\"cuda_headers_version\":%d,"
      "\"cuda_runtime_reported_version\":%d,"
      "\"capabilities\":[\"graph_arena\",\"linear_verify_rng\","
      "\"linear_rejection_sampling\",\"rng_sampler_graph_gate\"]}\n",
      SGL_NATIVE_TENSOR_ABI_MAJOR, SGL_NATIVE_TENSOR_ABI_MINOR,
      sizeof(SglNativeTensorMetadataV1), sizeof(SglNativeConstTensorViewV1),
      kLinearVerifyRngMaxNumSlots,
      static_cast<unsigned long long>(kLinearVerifyRngStateDescriptorV1),
      CUDART_VERSION, cuda_runtime_version);
  return 0;
}
