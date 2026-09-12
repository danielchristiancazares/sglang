// Copyright 2026 SGLang Team
//
// Isolated SM120 benchmark for the NVFP4 Marlin M<=8 thread-block geometry.
// The timing sweep uses uniform values so every layout remains bit exact;
// production-dispatch qualification separately randomizes every input.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../python/sglang/kernels/jit/csrc/gemm/marlin/gptq_marlin.cuh"

namespace {

void check_cuda(cudaError_t status, const char *expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#undef CHECK_CUDA
#define CHECK_CUDA(expression) check_cuda((expression), #expression)

__device__ std::uint32_t mix32(std::uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352dU;
  value ^= value >> 15;
  value *= 0x846ca68bU;
  return value ^ (value >> 16);
}

__global__ void fill_bf16(sglang::bf16_t *values, std::size_t count,
                          std::uint32_t seed) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    const int centered =
        static_cast<int>(mix32(static_cast<std::uint32_t>(index) + seed) &
                         2047U) -
        1024;
    values[index] = __float2bfloat16_rn(static_cast<float>(centered) / 2048.0F);
  }
}

__global__ void fill_bytes(std::uint8_t *values, std::size_t count,
                           std::uint32_t seed, bool fp8_scales) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += gridDim.x * blockDim.x) {
    const std::uint32_t mixed = mix32(static_cast<std::uint32_t>(index) + seed);
    values[index] = fp8_scales
                        ? static_cast<std::uint8_t>(0x20U + (mixed % 5U) * 8U)
                        : static_cast<std::uint8_t>(mixed);
  }
}

class DeviceAllocation {
public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    CHECK_CUDA(cudaMalloc(&data_, bytes));
  }

  DeviceAllocation(const DeviceAllocation &) = delete;
  DeviceAllocation &operator=(const DeviceAllocation &) = delete;

  ~DeviceAllocation() {
    if (data_ != nullptr)
      cudaFree(data_);
  }

  [[nodiscard]] void *get() const { return data_; }
  [[nodiscard]] std::size_t bytes() const { return bytes_; }

private:
  void *data_ = nullptr;
  std::size_t bytes_ = 0;
};

struct Shape {
  const char *name;
  int n;
  int k;
  int count;
};

enum class Config : int {
  k64x64 = 0,
  k128x64 = 1,
  k256x64 = 2,
  k64x128 = 3,
  k128x128 = 4,
  k256x128 = 5,
  k64x256 = 6,
  k128x256 = 7,
  k128x128b2 = 8,
  k128x64b2 = 9,
  k128x64b3 = 10,
  k64x128b2 = 11,
  k64x128b3 = 12,
  k64x256b2 = 13,
};

constexpr std::array<Config, 14> kConfigs = {
    Config::k64x64,     Config::k128x64,   Config::k256x64,   Config::k64x128,
    Config::k128x128,   Config::k256x128,  Config::k64x256,   Config::k128x256,
    Config::k128x128b2, Config::k128x64b2, Config::k128x64b3, Config::k64x128b2,
    Config::k64x128b3,  Config::k64x256b2};

const char *config_name(Config config) {
  switch (config) {
  case Config::k64x64:
    return "k64_n64_t64";
  case Config::k128x64:
    return "k128_n64_t128";
  case Config::k256x64:
    return "k256_n64_t256";
  case Config::k64x128:
    return "k64_n128_t128";
  case Config::k128x128:
    return "k128_n128_t256";
  case Config::k256x128:
    return "k256_n128_t512";
  case Config::k64x256:
    return "k64_n256_t256";
  case Config::k128x256:
    return "k128_n256_t512";
  case Config::k128x128b2:
    return "k128_n128_t256_b2";
  case Config::k128x64b2:
    return "k128_n64_t128_b2";
  case Config::k128x64b3:
    return "k128_n64_t128_b3";
  case Config::k64x128b2:
    return "k64_n128_t128_b2";
  case Config::k64x128b3:
    return "k64_n128_t128_b3";
  case Config::k64x256b2:
    return "k64_n256_t256_b2";
  }
  return "unknown";
}

template <int ThreadK, int ThreadN, int Threads, int BlocksPerSm = 1>
void launch_marlin(cudaStream_t stream, int sms, int max_shared_mem,
                   const void *a, const void *b, void *c, void *c_tmp,
                   const void *scales, const void *global_scale, int *workspace,
                   int m, int n, int k) {
  static_assert(ThreadK % 16 == 0);
  static_assert(ThreadN % 16 == 0);
  auto kernel = sglang::device::marlin::Marlin<
      sglang::bf16_t, sglang::host::kFE2M1f.id(), Threads, 1, ThreadN / 16,
      ThreadK / 16, true, sglang::device::marlin::pipe_stages, 1, false>;
  const int launch_shared_mem =
      BlocksPerSm == 1 ? max_shared_mem : max_shared_mem / BlocksPerSm - 1024;
  CHECK_CUDA(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, launch_shared_mem));
  kernel<<<sms * BlocksPerSm, Threads, launch_shared_mem, stream>>>(
      static_cast<const int4 *>(a), static_cast<const int4 *>(b),
      static_cast<int4 *>(c), static_cast<int4 *>(c_tmp),
      static_cast<const int4 *>(scales),
      static_cast<const std::uint16_t *>(global_scale), nullptr, nullptr,
      k / 16, m, n, k, k, workspace, false, true, launch_shared_mem);
}

void launch_config(Config config, cudaStream_t stream, int sms,
                   int max_shared_mem, const void *a, const void *b, void *c,
                   void *c_tmp, const void *scales, const void *global_scale,
                   int *workspace, int m, int n, int k) {
  switch (config) {
  case Config::k64x64:
    launch_marlin<64, 64, 64>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                              scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x64:
    launch_marlin<128, 64, 128>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                scales, global_scale, workspace, m, n, k);
    return;
  case Config::k256x64:
    launch_marlin<256, 64, 256>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                scales, global_scale, workspace, m, n, k);
    return;
  case Config::k64x128:
    launch_marlin<64, 128, 128>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x128:
    launch_marlin<128, 128, 256>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                 scales, global_scale, workspace, m, n, k);
    return;
  case Config::k256x128:
    launch_marlin<256, 128, 512>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                 scales, global_scale, workspace, m, n, k);
    return;
  case Config::k64x256:
    launch_marlin<64, 256, 256>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x256:
    launch_marlin<128, 256, 512>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                 scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x128b2:
    launch_marlin<128, 128, 256, 2>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                    scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x64b2:
    launch_marlin<128, 64, 128, 2>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                   scales, global_scale, workspace, m, n, k);
    return;
  case Config::k128x64b3:
    launch_marlin<128, 64, 128, 3>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                   scales, global_scale, workspace, m, n, k);
    return;
  case Config::k64x128b2:
    launch_marlin<64, 128, 128, 2>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                   scales, global_scale, workspace, m, n, k);
    return;
  case Config::k64x128b3:
    launch_marlin<64, 128, 128, 3>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                   scales, global_scale, workspace, m, n, k);
    return;
  case Config::k64x256b2:
    launch_marlin<64, 256, 256, 2>(stream, sms, max_shared_mem, a, b, c, c_tmp,
                                   scales, global_scale, workspace, m, n, k);
    return;
  }
  throw std::runtime_error("unknown Marlin configuration");
}

void launch_auto(cudaStream_t stream, int device, int sms, const void *a,
                 const void *b, void *c, void *c_tmp, const void *scales,
                 const void *global_scale, int *workspace, int m, int n,
                 int k) {
  sglang::device::marlin::marlin_mm<sglang::bf16_t>(
      a, b, c, c_tmp, const_cast<void *>(scales),
      const_cast<void *>(global_scale), nullptr, nullptr, nullptr, nullptr, m,
      n, k, k, workspace, sglang::host::kFE2M1f, false, true, false, k / 16, 16,
      device, stream, -1, -1, sms, false, true, false);
}

std::uint64_t fnv1a(const std::vector<std::uint16_t> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint16_t value : values) {
    hash ^= static_cast<std::uint8_t>(value);
    hash *= 1099511628211ULL;
    hash ^= static_cast<std::uint8_t>(value >> 8);
    hash *= 1099511628211ULL;
  }
  return hash;
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

float bf16_to_float(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

struct Difference {
  std::size_t mismatches = 0;
  float max_abs = 0.0F;
};

Difference compare_outputs(const std::vector<std::uint16_t> &lhs,
                           const std::vector<std::uint16_t> &rhs) {
  if (lhs.size() != rhs.size())
    throw std::runtime_error("output sizes differ");
  Difference difference;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index] != rhs[index])
      ++difference.mismatches;
    difference.max_abs =
        std::max(difference.max_abs, std::abs(bf16_to_float(lhs[index]) -
                                              bf16_to_float(rhs[index])));
  }
  return difference;
}

std::vector<std::uint16_t> copy_output(const DeviceAllocation &output,
                                       std::size_t elements) {
  std::vector<std::uint16_t> host_output(elements);
  CHECK_CUDA(cudaMemcpy(host_output.data(), output.get(), output.bytes(),
                        cudaMemcpyDeviceToHost));
  return host_output;
}

void fill_random_inputs(DeviceAllocation &a, DeviceAllocation &b,
                        DeviceAllocation &scales,
                        DeviceAllocation &global_scale, std::uint32_t seed,
                        cudaStream_t stream, int sms) {
  constexpr int threads = 256;
  const int blocks = sms * 4;
  fill_bf16<<<blocks, threads, 0, stream>>>(
      static_cast<sglang::bf16_t *>(a.get()), a.bytes() / sizeof(std::uint16_t),
      seed);
  fill_bytes<<<blocks, threads, 0, stream>>>(
      static_cast<std::uint8_t *>(b.get()), b.bytes(), seed ^ 0x9e3779b9U,
      false);
  fill_bytes<<<blocks, threads, 0, stream>>>(
      static_cast<std::uint8_t *>(scales.get()), scales.bytes(),
      seed ^ 0x243f6a88U, true);
  const std::uint16_t host_global_scale = 0x3c00U;
  CHECK_CUDA(cudaMemcpyAsync(global_scale.get(), &host_global_scale,
                             sizeof(host_global_scale), cudaMemcpyHostToDevice,
                             stream));
  CHECK_CUDA(cudaGetLastError());
}

void qualify_auto_dispatch(const Shape &shape, Config selected_config,
                           int expected_blocks, int expected_threads,
                           int expected_shared_mem, cudaStream_t stream,
                           int device, int sms, int max_shared_mem) {
  constexpr int m = 8;
  const std::size_t a_bytes =
      static_cast<std::size_t>(m) * shape.k * sizeof(std::uint16_t);
  const std::size_t b_bytes = static_cast<std::size_t>(shape.n) * shape.k / 2;
  const std::size_t scale_bytes =
      static_cast<std::size_t>(shape.n) * shape.k / 16;
  const std::size_t c_elements = static_cast<std::size_t>(m) * shape.n;
  const std::size_t c_bytes = c_elements * sizeof(std::uint16_t);
  const std::size_t c_tmp_bytes =
      static_cast<std::size_t>(sms) * 16 * 256 * sizeof(float);

  DeviceAllocation a(a_bytes);
  DeviceAllocation b(b_bytes);
  DeviceAllocation scales(scale_bytes);
  DeviceAllocation global_scale(sizeof(std::uint16_t));
  DeviceAllocation c(c_bytes);
  DeviceAllocation c_tmp(c_tmp_bytes);
  DeviceAllocation workspace(static_cast<std::size_t>(sms) * sizeof(int));

  fill_random_inputs(a, b, scales, global_scale, 0x13579bdfU, stream, sms);
  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  launch_config(Config::k128x128, stream, sms, max_shared_mem, a.get(), b.get(),
                c.get(), c_tmp.get(), scales.get(), global_scale.get(),
                static_cast<int *>(workspace.get()), m, shape.n, shape.k);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> baseline = copy_output(c, c_elements);

  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  launch_config(selected_config, stream, sms, max_shared_mem, a.get(), b.get(),
                c.get(), c_tmp.get(), scales.get(), global_scale.get(),
                static_cast<int *>(workspace.get()), m, shape.n, shape.k);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> selected = copy_output(c, c_elements);

  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  launch_auto(stream, device, sms, a.get(), b.get(), c.get(), c_tmp.get(),
              scales.get(), global_scale.get(),
              static_cast<int *>(workspace.get()), m, shape.n, shape.k);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> automatic = copy_output(c, c_elements);

  const Difference selected_vs_auto = compare_outputs(selected, automatic);
  const Difference baseline_vs_auto = compare_outputs(baseline, automatic);
  if (selected_vs_auto.mismatches != 0) {
    throw std::runtime_error(std::string(shape.name) +
                             " auto dispatch differs from selected kernel");
  }

  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  launch_auto(stream, device, sms, a.get(), b.get(), c.get(), c_tmp.get(),
              scales.get(), global_scale.get(),
              static_cast<int *>(workspace.get()), m, shape.n, shape.k);
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

  std::size_t node_count = 0;
  CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CHECK_CUDA(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  std::size_t kernel_count = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    CHECK_CUDA(cudaGraphNodeGetType(node, &type));
    if (type != cudaGraphNodeTypeKernel)
      continue;
    ++kernel_count;
    cudaKernelNodeParams params{};
    CHECK_CUDA(cudaGraphKernelNodeGetParams(node, &params));
    if (params.gridDim.x != static_cast<unsigned int>(expected_blocks) ||
        params.blockDim.x != static_cast<unsigned int>(expected_threads) ||
        params.sharedMemBytes !=
            static_cast<unsigned int>(expected_shared_mem)) {
      throw std::runtime_error(std::string(shape.name) +
                               " captured unexpected launch geometry");
    }
  }
  if (kernel_count != 1)
    throw std::runtime_error(std::string(shape.name) +
                             " capture did not contain exactly one kernel");

  CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> graph_initial = copy_output(c, c_elements);
  if (compare_outputs(automatic, graph_initial).mismatches != 0)
    throw std::runtime_error(std::string(shape.name) +
                             " initial graph replay is not bit exact");

  fill_random_inputs(a, b, scales, global_scale, 0x2468ace0U, stream, sms);
  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> graph_mutated = copy_output(c, c_elements);

  CHECK_CUDA(cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
  launch_config(selected_config, stream, sms, max_shared_mem, a.get(), b.get(),
                c.get(), c_tmp.get(), scales.get(), global_scale.get(),
                static_cast<int *>(workspace.get()), m, shape.n, shape.k);
  CHECK_CUDA(cudaStreamSynchronize(stream));
  const std::vector<std::uint16_t> selected_mutated =
      copy_output(c, c_elements);
  if (compare_outputs(selected_mutated, graph_mutated).mismatches != 0)
    throw std::runtime_error(std::string(shape.name) +
                             " mutated graph replay is not bit exact");

  CHECK_CUDA(cudaGraphExecDestroy(graph_exec));
  CHECK_CUDA(cudaGraphDestroy(graph));
  std::cout << "qualified," << shape.name << ",auto_vs_selected_mismatches,"
            << selected_vs_auto.mismatches << ",auto_vs_baseline_mismatches,"
            << baseline_vs_auto.mismatches << ",auto_vs_baseline_max_abs,"
            << baseline_vs_auto.max_abs << ",graph_nodes," << node_count
            << ",grid," << expected_blocks << ",threads," << expected_threads
            << ",shared," << expected_shared_mem << '\n';
}

struct ShapeResult {
  Shape shape;
  std::array<float, kConfigs.size()> median_us{};
};

ShapeResult benchmark_shape(const Shape &shape, int repetitions, int rounds,
                            cudaStream_t stream, int sms, int max_shared_mem) {
  constexpr int m = 8;
  const std::size_t a_bytes =
      static_cast<std::size_t>(m) * shape.k * sizeof(std::uint16_t);
  const std::size_t b_bytes = static_cast<std::size_t>(shape.n) * shape.k / 2;
  const std::size_t scale_bytes =
      static_cast<std::size_t>(shape.n) * shape.k / 16;
  const std::size_t c_bytes =
      static_cast<std::size_t>(m) * shape.n * sizeof(std::uint16_t);
  const std::size_t c_tmp_bytes =
      static_cast<std::size_t>(sms) * 16 * 256 * sizeof(float);
  const std::size_t workspace_words =
      std::max<std::size_t>(sms, static_cast<std::size_t>(shape.n / 64) * 16);

  DeviceAllocation a(a_bytes);
  DeviceAllocation b(b_bytes);
  DeviceAllocation scales(scale_bytes);
  DeviceAllocation global_scale(sizeof(std::uint16_t));
  DeviceAllocation c(c_bytes);
  DeviceAllocation c_tmp(c_tmp_bytes);
  DeviceAllocation workspace(workspace_words * sizeof(int));

  std::vector<std::uint16_t> host_a(static_cast<std::size_t>(m) * shape.k,
                                    0x3f80U);
  const std::uint16_t host_global_scale = 0x3f80U;
  CHECK_CUDA(cudaMemcpyAsync(a.get(), host_a.data(), a.bytes(),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(b.get(), 0x11, b.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(scales.get(), 0x38, scales.bytes(), stream));
  CHECK_CUDA(cudaMemcpyAsync(global_scale.get(), &host_global_scale,
                             sizeof(host_global_scale), cudaMemcpyHostToDevice,
                             stream));

  std::array<std::vector<float>, kConfigs.size()> samples;
  std::vector<std::uint16_t> reference;
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&end));

  for (int round = 0; round < rounds; ++round) {
    for (std::size_t offset = 0; offset < kConfigs.size(); ++offset) {
      const std::size_t index =
          (offset + static_cast<std::size_t>(round)) % kConfigs.size();
      const Config config = kConfigs[index];
      CHECK_CUDA(
          cudaMemsetAsync(workspace.get(), 0, workspace.bytes(), stream));
      CHECK_CUDA(cudaMemsetAsync(c.get(), 0, c.bytes(), stream));
      for (int warmup = 0; warmup < 32; ++warmup) {
        launch_config(config, stream, sms, max_shared_mem, a.get(), b.get(),
                      c.get(), c_tmp.get(), scales.get(), global_scale.get(),
                      static_cast<int *>(workspace.get()), m, shape.n, shape.k);
      }
      CHECK_CUDA(cudaGetLastError());
      CHECK_CUDA(cudaEventRecord(begin, stream));
      for (int iteration = 0; iteration < repetitions; ++iteration) {
        launch_config(config, stream, sms, max_shared_mem, a.get(), b.get(),
                      c.get(), c_tmp.get(), scales.get(), global_scale.get(),
                      static_cast<int *>(workspace.get()), m, shape.n, shape.k);
      }
      CHECK_CUDA(cudaEventRecord(end, stream));
      CHECK_CUDA(cudaEventSynchronize(end));
      float elapsed_ms = 0.0F;
      CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, begin, end));
      samples[index].push_back(elapsed_ms * 1000.0F /
                               static_cast<float>(repetitions));

      std::vector<std::uint16_t> output(static_cast<std::size_t>(m) * shape.n);
      CHECK_CUDA(cudaMemcpy(output.data(), c.get(), c.bytes(),
                            cudaMemcpyDeviceToHost));
      if (reference.empty()) {
        reference = output;
      } else {
        std::size_t mismatches = 0;
        for (std::size_t position = 0; position < output.size(); ++position) {
          if (output[position] != reference[position])
            ++mismatches;
        }
        if (mismatches != 0) {
          throw std::runtime_error(
              std::string(shape.name) + " " + config_name(config) +
              " produced " + std::to_string(mismatches) + " bit mismatches");
        }
      }
    }
  }

  CHECK_CUDA(cudaEventDestroy(begin));
  CHECK_CUDA(cudaEventDestroy(end));

  ShapeResult result{shape};
  for (std::size_t index = 0; index < kConfigs.size(); ++index) {
    result.median_us[index] = median(samples[index]);
    std::cout << shape.name << ',' << shape.n << ',' << shape.k << ','
              << config_name(kConfigs[index]) << ',' << std::fixed
              << std::setprecision(6) << result.median_us[index] << ','
              << std::hex << fnv1a(reference) << std::dec << '\n';
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool qualify_only =
        argc > 1 && std::strcmp(argv[1], "--qualify-only") == 0;
    const int repetitions =
        !qualify_only && argc > 1 ? std::max(1, std::atoi(argv[1])) : 1000;
    const int rounds =
        !qualify_only && argc > 2 ? std::max(1, std::atoi(argv[2])) : 7;
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    int major = 0;
    int minor = 0;
    int sms = 0;
    int max_shared_mem = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                      device));
    CHECK_CUDA(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                      device));
    CHECK_CUDA(
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaDeviceGetAttribute(
        &max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    if (major != 12 || minor != 0)
      throw std::runtime_error("this benchmark requires SM120");

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    constexpr std::array<Shape, 4> shapes = {
        Shape{"gate_up", 34816, 5120, 64},
        Shape{"linear_qkvz", 16384, 5120, 48},
        Shape{"full_qkv", 14336, 5120, 16},
        Shape{"attention_out", 5120, 6144, 64}};

    if (!qualify_only) {
      std::cout << "shape,n,k,config,median_us,reference_fnv1a\n";
      std::array<double, kConfigs.size()> weighted_us{};
      for (const Shape &shape : shapes) {
        const ShapeResult result = benchmark_shape(shape, repetitions, rounds,
                                                   stream, sms, max_shared_mem);
        for (std::size_t index = 0; index < kConfigs.size(); ++index) {
          weighted_us[index] +=
              static_cast<double>(shape.count) * result.median_us[index];
        }
      }

      std::cout << "weighted_192_projection_us\n";
      for (std::size_t index = 0; index < kConfigs.size(); ++index) {
        std::cout << config_name(kConfigs[index]) << ',' << std::fixed
                  << std::setprecision(6) << weighted_us[index] << '\n';
      }
    }

    qualify_auto_dispatch(shapes[0], Config::k256x64, sms, 256, max_shared_mem,
                          stream, device, sms, max_shared_mem);
    qualify_auto_dispatch(shapes[1], Config::k64x128b2, sms * 2, 128,
                          max_shared_mem / 2 - 1024, stream, device, sms,
                          max_shared_mem);
    CHECK_CUDA(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
