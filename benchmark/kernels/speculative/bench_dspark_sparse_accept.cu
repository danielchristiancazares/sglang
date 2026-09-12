// Copyright 2026 SGLang Team
//
// Standalone CUDA qualification and timing harness for DSpark's exact sparse
// rejection sampler. The production shape is Qwen3.8-27B: gamma 7, target
// top-k 20, and a 248,320-token vocabulary.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../../python/sglang/kernels/jit/csrc/speculative/dspark_sparse_accept.cuh"

namespace {

constexpr uint32_t kBatch = 1;
constexpr uint32_t kGamma = 7;
constexpr uint32_t kSlots = kGamma + 1;
constexpr uint32_t kWidth = 20;
constexpr uint32_t kVocab = 248320;
constexpr uint32_t kSplits =
    (kVocab + sglang::device::dspark_sparse_accept::kValuesPerSplit - 1) /
    sglang::device::dspark_sparse_accept::kValuesPerSplit;

void check_cuda(cudaError_t status, const char *expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#undef CHECK_CUDA
#define CHECK_CUDA(expression) check_cuda((expression), #expression)

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    CHECK_CUDA(cudaMalloc(&data_, count * sizeof(T)));
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr)
      cudaFree(data_);
  }

  [[nodiscard]] T *get() const { return data_; }
  [[nodiscard]] std::size_t count() const { return count_; }

  void copy_from(const std::vector<T> &host, cudaStream_t stream = nullptr) {
    if (host.size() != count_)
      throw std::runtime_error("host/device size mismatch");
    CHECK_CUDA(cudaMemcpyAsync(data_, host.data(), count_ * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  [[nodiscard]] std::vector<T>
  copy_to_host(cudaStream_t stream = nullptr) const {
    std::vector<T> host(count_);
    CHECK_CUDA(cudaMemcpyAsync(host.data(), data_, count_ * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    return host;
  }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

struct Inputs {
  std::vector<float> target_logits;
  std::vector<int64_t> target_indices;
  std::vector<float> draft_logits;
  std::vector<int64_t> candidates;
  std::vector<float> target_temperatures{1.0F};
  std::vector<float> draft_temperatures{1.0F};
  std::vector<int32_t> top_ks{static_cast<int32_t>(kWidth)};
  std::vector<float> top_ps{0.95F};
  std::vector<float> uniforms;
  std::vector<float> final_uniforms{0.0F};
  std::vector<int64_t> verify_lens{static_cast<int64_t>(kSlots)};
};

Inputs make_inputs() {
  Inputs input;
  input.target_logits.resize(kBatch * kSlots * kWidth);
  input.target_indices.resize(kBatch * kSlots * kWidth);
  input.draft_logits.resize(kBatch * kGamma * kVocab);
  input.candidates.resize(kBatch * kSlots);
  input.uniforms.assign(kBatch * kGamma, 0.0F);

  for (uint32_t row = 0; row < kSlots; ++row) {
    for (uint32_t lane = 0; lane < kWidth; ++lane) {
      const uint64_t offset = static_cast<uint64_t>(row) * kWidth + lane;
      input.target_logits[offset] = 4.0F - 0.16F * static_cast<float>(lane) +
                                    0.01F * static_cast<float>(row);
      input.target_indices[offset] =
          static_cast<int64_t>((row * 4099U + lane * 997U + 31U) % kVocab);
    }
  }

  for (uint32_t row = 0; row < kGamma; ++row) {
    for (uint32_t token = 0; token < kVocab; ++token) {
      const uint32_t mixed = (token * 17U + row * 29U + 11U) % 101U;
      input.draft_logits[static_cast<uint64_t>(row) * kVocab + token] =
          (static_cast<int32_t>(mixed) - 50) * 0.04F;
    }
  }

  input.candidates[0] = 7;
  for (uint32_t step = 1; step < kSlots; ++step) {
    input.candidates[step] =
        input.target_indices[static_cast<uint64_t>(step - 1) * kWidth];
  }
  return input;
}

struct DeviceState {
  DeviceBuffer<float> target_logits{kBatch * kSlots * kWidth};
  DeviceBuffer<int64_t> target_indices{kBatch * kSlots * kWidth};
  DeviceBuffer<float> draft_logits{kBatch * kGamma * kVocab};
  DeviceBuffer<int64_t> candidates{kBatch * kSlots};
  DeviceBuffer<float> target_temperatures{kBatch};
  DeviceBuffer<float> draft_temperatures{kBatch};
  DeviceBuffer<int32_t> top_ks{kBatch};
  DeviceBuffer<float> top_ps{kBatch};
  DeviceBuffer<float> uniforms{kBatch * kGamma};
  DeviceBuffer<float> final_uniforms{kBatch};
  DeviceBuffer<int64_t> verify_lens{kBatch};
  DeviceBuffer<float> partial_max{kBatch * kGamma * kSplits};
  DeviceBuffer<float> partial_sum{kBatch * kGamma * kSplits};
  DeviceBuffer<float> log_normalizers{kBatch * kGamma};
  DeviceBuffer<int32_t> correct_len{kBatch};
  DeviceBuffer<int64_t> bonus{kBatch};
  DeviceBuffer<int32_t> trim{kBatch};

  void copy_inputs(const Inputs &input, cudaStream_t stream = nullptr) {
    target_logits.copy_from(input.target_logits, stream);
    target_indices.copy_from(input.target_indices, stream);
    draft_logits.copy_from(input.draft_logits, stream);
    candidates.copy_from(input.candidates, stream);
    target_temperatures.copy_from(input.target_temperatures, stream);
    draft_temperatures.copy_from(input.draft_temperatures, stream);
    top_ks.copy_from(input.top_ks, stream);
    top_ps.copy_from(input.top_ps, stream);
    uniforms.copy_from(input.uniforms, stream);
    final_uniforms.copy_from(input.final_uniforms, stream);
    verify_lens.copy_from(input.verify_lens, stream);
  }
};

void launch(DeviceState &state, cudaStream_t stream) {
  using namespace sglang::device::dspark_sparse_accept;
  DraftLogNormalizerPartialsKernel<float>
      <<<dim3(kSplits, kBatch * kGamma), kNormalizerThreads, 0, stream>>>(
          state.draft_logits.get(), state.draft_temperatures.get(),
          state.partial_max.get(), state.partial_sum.get(), kGamma, kVocab,
          kSplits);
  DraftLogNormalizerFinalizeKernel<<<kBatch * kGamma, kWarpSize, 0, stream>>>(
      state.partial_max.get(), state.partial_sum.get(),
      state.log_normalizers.get(), kSplits);
  SparseChainAcceptKernel<float, float><<<kBatch, kWarpSize, 0, stream>>>(
      state.target_logits.get(), state.target_indices.get(),
      state.draft_logits.get(), state.log_normalizers.get(),
      state.candidates.get(), state.target_temperatures.get(),
      state.draft_temperatures.get(), state.top_ks.get(), state.top_ps.get(),
      state.uniforms.get(), state.final_uniforms.get(), state.verify_lens.get(),
      state.correct_len.get(), state.bonus.get(), state.trim.get(), kSlots,
      kGamma, kWidth, kVocab);
  CHECK_CUDA(cudaGetLastError());
}

std::vector<int64_t> retained_tokens_for_row(const Inputs &input,
                                             uint32_t row) {
  std::vector<float> probabilities(kWidth);
  float maximum = -std::numeric_limits<float>::infinity();
  for (uint32_t lane = 0; lane < kWidth; ++lane) {
    maximum = std::max(
        maximum,
        input.target_logits[static_cast<uint64_t>(row) * kWidth + lane]);
  }
  float sum = 0.0F;
  for (uint32_t lane = 0; lane < kWidth; ++lane) {
    probabilities[lane] = std::exp(
        input.target_logits[static_cast<uint64_t>(row) * kWidth + lane] -
        maximum);
    sum += probabilities[lane];
  }
  for (float &value : probabilities)
    value /= sum;

  float prefix = 0.0F;
  uint32_t cutoff = 0;
  for (uint32_t lane = 0; lane < kWidth; ++lane) {
    if (prefix < input.top_ps[0])
      cutoff = lane;
    prefix += probabilities[lane];
  }
  const float threshold = probabilities[cutoff];
  std::vector<int64_t> tokens;
  for (uint32_t lane = 0; lane < kWidth; ++lane) {
    if (probabilities[lane] >= threshold) {
      tokens.push_back(
          input.target_indices[static_cast<uint64_t>(row) * kWidth + lane]);
    }
  }
  return tokens;
}

void expect_output(DeviceState &state, int32_t expected_len,
                   int32_t expected_trim, int64_t expected_bonus,
                   const char *label, cudaStream_t stream) {
  const auto len = state.correct_len.copy_to_host(stream);
  const auto bonus = state.bonus.copy_to_host(stream);
  const auto trim = state.trim.copy_to_host(stream);
  if (len[0] != expected_len || trim[0] != expected_trim ||
      bonus[0] != expected_bonus) {
    throw std::runtime_error(std::string(label) + " output mismatch: got len=" +
                             std::to_string(len[0]) +
                             ", trim=" + std::to_string(trim[0]) +
                             ", bonus=" + std::to_string(bonus[0]) +
                             "; expected len=" + std::to_string(expected_len) +
                             ", trim=" + std::to_string(expected_trim) +
                             ", bonus=" + std::to_string(expected_bonus));
  }
}

void qualify_eager(DeviceState &state, Inputs &input, cudaStream_t stream) {
  state.copy_inputs(input, stream);
  launch(state, stream);
  const auto retained = retained_tokens_for_row(input, kGamma);
  const int64_t expected_bonus =
      *std::min_element(retained.begin(), retained.end());
  expect_output(state, kGamma, 0, expected_bonus, "all-accepted", stream);

  input.verify_lens[0] = 3;
  state.verify_lens.copy_from(input.verify_lens, stream);
  launch(state, stream);
  expect_output(state, 2, kGamma - 2, input.candidates[3], "capped", stream);

  input.verify_lens[0] = kSlots;
  input.candidates[1] = kVocab - 1;
  state.verify_lens.copy_from(input.verify_lens, stream);
  state.candidates.copy_from(input.candidates, stream);
  launch(state, stream);
  const auto rejected_support = retained_tokens_for_row(input, 0);
  const int64_t rejected_bonus =
      *std::min_element(rejected_support.begin(), rejected_support.end());
  expect_output(state, 0, 0, rejected_bonus, "rejected", stream);

  std::fill_n(input.draft_logits.begin(), kVocab,
              std::numeric_limits<float>::quiet_NaN());
  input.candidates[1] = input.target_indices[0];
  state.draft_logits.copy_from(input.draft_logits, stream);
  state.candidates.copy_from(input.candidates, stream);
  launch(state, stream);
  expect_output(state, 0, 0, rejected_bonus, "nan-q", stream);
}

void qualify_graph(DeviceState &state, Inputs input, cudaStream_t stream) {
  input.verify_lens[0] = kSlots;
  state.copy_inputs(input, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  launch(state, stream);
  CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

  std::size_t node_count = 0;
  CHECK_CUDA(cudaGraphGetNodes(graph, nullptr, &node_count));
  std::vector<cudaGraphNode_t> nodes(node_count);
  CHECK_CUDA(cudaGraphGetNodes(graph, nodes.data(), &node_count));
  std::size_t kernel_nodes = 0;
  for (cudaGraphNode_t node : nodes) {
    cudaGraphNodeType type{};
    CHECK_CUDA(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel)
      ++kernel_nodes;
  }
  if (kernel_nodes != 3) {
    throw std::runtime_error(
        "captured graph must contain exactly three kernel nodes, got " +
        std::to_string(kernel_nodes));
  }

  CHECK_CUDA(cudaGraphInstantiate(&executable, graph, 0));
  CHECK_CUDA(cudaGraphLaunch(executable, stream));
  const auto retained = retained_tokens_for_row(input, kGamma);
  expect_output(state, kGamma, 0,
                *std::min_element(retained.begin(), retained.end()),
                "graph-all-accepted", stream);

  input.verify_lens[0] = 3;
  state.verify_lens.copy_from(input.verify_lens, stream);
  CHECK_CUDA(cudaGraphLaunch(executable, stream));
  expect_output(state, 2, kGamma - 2, input.candidates[3],
                "graph-mutated-cutoff", stream);

  input.verify_lens[0] = kSlots;
  input.candidates[1] = kVocab - 1;
  state.verify_lens.copy_from(input.verify_lens, stream);
  state.candidates.copy_from(input.candidates, stream);
  CHECK_CUDA(cudaGraphLaunch(executable, stream));
  const auto rejected_support = retained_tokens_for_row(input, 0);
  expect_output(
      state, 0, 0,
      *std::min_element(rejected_support.begin(), rejected_support.end()),
      "graph-mutated-reject", stream);

  CHECK_CUDA(cudaGraphExecDestroy(executable));
  CHECK_CUDA(cudaGraphDestroy(graph));
  std::cout << "graph_kernel_nodes=" << kernel_nodes << '\n';
}

double time_kernels(DeviceState &state, const Inputs &input,
                    cudaStream_t stream) {
  state.copy_inputs(input, stream);
  for (int iteration = 0; iteration < 20; ++iteration)
    launch(state, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  constexpr int kIterations = 1000;
  CHECK_CUDA(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < kIterations; ++iteration)
    launch(state, stream);
  CHECK_CUDA(cudaEventRecord(stop, stream));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float milliseconds = 0.0F;
  CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return static_cast<double>(milliseconds) * 1000.0 / kIterations;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const bool qualify_only =
        argc == 2 && std::string(argv[1]) == "--qualify-only";
    if (argc > 2 || (argc == 2 && !qualify_only)) {
      std::cerr << "usage: " << argv[0] << " [--qualify-only]\n";
      return 2;
    }

    cudaStream_t stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceState state;
    Inputs input = make_inputs();
    qualify_eager(state, input, stream);
    qualify_graph(state, make_inputs(), stream);
    if (!qualify_only) {
      const double microseconds = time_kernels(state, make_inputs(), stream);
      std::cout << std::fixed << std::setprecision(3)
                << "production_sparse_triplet_us=" << microseconds << '\n';
    }
    CHECK_CUDA(cudaStreamDestroy(stream));
    std::cout << "DSpark sparse accept qualification passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DSpark sparse accept qualification failed: " << error.what()
              << '\n';
    return 1;
  }
}
