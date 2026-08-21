#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <flashinfer/air_top_p.cuh>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace sparse_top_p_detail {

namespace air = flashinfer::sampling::air_top_p;
static_assert(sizeof(air::Counter<float>) == 384);

template <uint32_t kMaxNonzero>
__global__
__launch_bounds__(1024) void sparse_air_apply_kernel(float* probs, air::Counter<float>* counters, uint32_t vocab_size) {
  using BlockReduce = cub::BlockReduce<float, 1024>;

  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float values[kMaxNonzero];
  __shared__ uint32_t indices[kMaxNonzero];
  __shared__ uint32_t count;
  __shared__ float norm;

  const uint32_t row = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  float* row_probs = probs + static_cast<uint64_t>(row) * vocab_size;
  const float threshold = air::twiddleOut<float>(counters[row].kthValueBits, false);

  if (tid == 0) count = 0;
  __syncthreads();

  // Match AIR apply's per-thread accumulation exactly while retaining sparse
  // coordinates for the one-pass store path.
  float thread_sum = 0.0f;
  for (uint32_t i = tid; i < vocab_size; i += blockDim.x) {
    const float value = row_probs[i];
    if (value > 0.0f) {
      const uint32_t slot = atomicAdd(&count, 1u);
      if (slot < kMaxNonzero) {
        values[slot] = value;
        indices[slot] = i;
      }
    }
    if (value >= threshold) thread_sum += value;
  }

  const float total_sum = BlockReduce(reduce_storage).Sum(thread_sum);
  if (tid == 0) norm = total_sum > 1e-8f ? 1.0f / total_sum : 1.0f;
  __syncthreads();

  const float row_norm = norm;
  if (count <= kMaxNonzero) {
    if (tid < count) {
      const float value = values[tid];
      row_probs[indices[tid]] = value >= threshold ? value * row_norm : 0.0f;
    }
  } else {
    // Exact top-k ties can widen support beyond k. The AIR threshold and
    // normalization are already available, so fall back to its dense store.
    for (uint32_t i = tid; i < vocab_size; i += blockDim.x) {
      const float value = row_probs[i];
      row_probs[i] = value >= threshold ? value * row_norm : 0.0f;
    }
  }
}

inline size_t workspace_size(uint32_t batch_size, uint32_t vocab_size) {
  auto align256 = [](size_t value) { return ((value + 255) / 256) * 256; };
  constexpr size_t kCounterSize = 384;
  const auto buf_len = std::max(align256(static_cast<size_t>(vocab_size) / 32), size_t{256});
  return align256(kCounterSize * batch_size) + align256(sizeof(float) * air::NUM_BUCKETS * batch_size) +
         align256(sizeof(air::IdxT) * air::NUM_BUCKETS * batch_size) +
         2 * align256(sizeof(float) * buf_len * batch_size);
}

}  // namespace sparse_top_p_detail

template <uint32_t kMaxNonzero>
struct SparseTopPRenormKernel {
  static_assert(kMaxNonzero > 0 && kMaxNonzero <= 1024);

  static void
  run(const tvm::ffi::TensorView probs, const tvm::ffi::TensorView top_ps, const tvm::ffi::TensorView workspace) {
    using namespace host;
    namespace air = sparse_top_p_detail::air;

    auto M = SymbolicSize{"num_rows"};
    auto V = SymbolicSize{"vocab_size"};
    auto W = SymbolicSize{"workspace_bytes"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({M, V}).with_dtype<float>().with_device(device).verify(probs);
    TensorMatcher({M}).with_dtype<float>().with_device(device).verify(top_ps);
    TensorMatcher({W}).with_dtype<uint8_t>().with_device(device).verify(workspace);

    CHECK_HOST(M.unwrap() > 0 && M.unwrap() <= UINT32_MAX) << "sparse_top_p_renorm: num_rows must fit uint32";
    CHECK_HOST(V.unwrap() >= air::NUM_BUCKETS && V.unwrap() <= UINT32_MAX)
        << "sparse_top_p_renorm: AIR requires vocab_size >= " << air::NUM_BUCKETS;

    const uint32_t batch_size = static_cast<uint32_t>(M.unwrap());
    const uint32_t vocab_size = static_cast<uint32_t>(V.unwrap());
    const size_t required_workspace = sparse_top_p_detail::workspace_size(batch_size, vocab_size);
    CHECK_HOST(static_cast<size_t>(W.unwrap()) >= required_workspace) << "sparse_top_p_renorm: workspace too small";

    auto* ws = static_cast<uint8_t*>(workspace.data_ptr());
    auto align256 = [](size_t value) { return ((value + 255) / 256) * 256; };
    constexpr size_t kCounterSize = 384;
    const size_t counters_size = align256(kCounterSize * batch_size);
    const size_t histogram_size = align256(sizeof(float) * air::NUM_BUCKETS * batch_size);
    const size_t count_histogram_size = align256(sizeof(air::IdxT) * air::NUM_BUCKETS * batch_size);
    const size_t buf_len = std::max(align256(static_cast<size_t>(vocab_size) / 32), size_t{256});
    const size_t buffer_size = align256(sizeof(float) * buf_len * batch_size);

    auto* counters = reinterpret_cast<air::Counter<float>*>(ws);
    auto* histograms = reinterpret_cast<float*>(ws + counters_size);
    auto* count_histograms = reinterpret_cast<air::IdxT*>(ws + counters_size + histogram_size);
    auto* buf1 = reinterpret_cast<float*>(ws + counters_size + histogram_size + count_histogram_size);
    auto* buf2 = reinterpret_cast<float*>(ws + counters_size + histogram_size + count_histogram_size + buffer_size);

    const auto dl_device = device.unwrap();
    const uint32_t sm_count = runtime::get_sm_count(dl_device.device_id);
    const uint32_t block_count = air::CalcAirTopPBlockNum<false, float>(batch_size, vocab_size, sm_count);

    LaunchKernel(batch_size, 256, dl_device)(
        air::AirTopPRenormInitKernel<float, float>,
        counters,
        static_cast<int>(vocab_size),
        static_cast<const float*>(probs.data_ptr()),
        static_cast<float*>(top_ps.data_ptr()),
        0.0f,
        histograms,
        count_histograms);

    for (int pass = 0; pass < air::NUM_PASSES<float>; ++pass) {
      LaunchKernel(dim3(block_count, batch_size), air::BLOCK_SIZE, dl_device)(
          air::AirTopPRenormRadixKernel<false, float>, counters, histograms, count_histograms, pass, buf1, buf2);
    }

    LaunchKernel(batch_size, 1024, dl_device)(
        sparse_top_p_detail::sparse_air_apply_kernel<kMaxNonzero>,
        static_cast<float*>(probs.data_ptr()),
        counters,
        vocab_size);
  }
};

}  // namespace sglang
