#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cassert>
#include <cstdint>

namespace sglang {

namespace flashinfer_page_table_detail {

/**
 * \brief Convert page-aligned request token slots to FlashInfer page IDs.
 *
 * \tparam kPageSize Physical and logical tokens per page.
 */
template <uint32_t kPageSize>
__global__ void build_page_table_kernel(
    int32_t* __restrict__ page_indices,
    const int32_t* __restrict__ req_to_token,
    const int64_t* __restrict__ req_pool_indices,
    const int32_t* __restrict__ page_lens,
    const int32_t* __restrict__ page_indptr,
    uint32_t batch_size,
    uint32_t req_to_token_stride,
    uint32_t total_pages,
    uint32_t physical_page_count) {
  const uint32_t row = blockIdx.x;
  if (row >= batch_size) {
    return;
  }

  const uint64_t request_offset = static_cast<uint64_t>(req_pool_indices[row]) * req_to_token_stride;
  const uint32_t output_offset = static_cast<uint32_t>(page_indptr[row]);
  const uint32_t num_pages = static_cast<uint32_t>(page_lens[row]);
  for (uint32_t page = threadIdx.x; page < num_pages; page += blockDim.x) {
    const uint32_t output_index = output_offset + page;
    if (output_index < total_pages) {
      const int32_t slot = req_to_token[request_offset + static_cast<uint64_t>(page) * kPageSize];
      assert(slot >= 0);
      assert((static_cast<uint32_t>(slot) & (kPageSize - 1)) == 0);
      const uint32_t physical_page = static_cast<uint32_t>(slot) / kPageSize;
      assert(physical_page < physical_page_count);
      page_indices[output_index] = static_cast<int32_t>(physical_page);
    }
  }
}

}  // namespace flashinfer_page_table_detail

/**
 * \brief Validate and launch the FlashInfer page-table builder.
 *
 * \tparam kPageSize Physical and logical tokens per page.
 */
template <uint32_t kPageSize>
struct FlashInferPageTableKernel {
  static_assert(kPageSize > 1 && (kPageSize & (kPageSize - 1)) == 0);

  /**
   * \param page_indices Output page IDs; capacity must cover `total_pages`.
   * \param req_to_token Request-to-physical-token mapping.
   * \param req_pool_indices Active request rows.
   * \param page_lens Logical page count per active request.
   * \param page_indptr Exclusive page offsets, shape `[batch + 1]`.
   * \param total_pages Sum of `page_lens`.
   * \param physical_page_count Number of addressable physical KV pages.
   */
  static void
  run(const tvm::ffi::TensorView page_indices,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView req_pool_indices,
      const tvm::ffi::TensorView page_lens,
      const tvm::ffi::TensorView page_indptr,
      int64_t total_pages,
      int64_t physical_page_count) {
    using namespace host;

    auto R = SymbolicSize{"request_pool_size"};
    auto C = SymbolicSize{"max_context_len"};
    auto B = SymbolicSize{"batch_size"};
    auto I = SymbolicSize{"indptr_size"};
    auto O = SymbolicSize{"output_capacity"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();

    TensorMatcher({R, C}).with_dtype<int32_t>().with_device(device).verify(req_to_token);
    TensorMatcher({B}).with_dtype<int64_t>().with_device(device).verify(req_pool_indices);
    TensorMatcher({B}).with_dtype<int32_t>().with_device(device).verify(page_lens);
    TensorMatcher({I}).with_dtype<int32_t>().with_device(device).verify(page_indptr);
    TensorMatcher({O}).with_dtype<int32_t>().with_device(device).verify(page_indices);

    CHECK_HOST(B.unwrap() > 0 && B.unwrap() <= UINT32_MAX) << "flashinfer_page_table: batch_size must fit uint32";
    CHECK_HOST(C.unwrap() > 0 && C.unwrap() <= UINT32_MAX) << "flashinfer_page_table: max_context_len must fit uint32";
    CHECK_HOST(I.unwrap() == B.unwrap() + 1) << "flashinfer_page_table: page_indptr must have batch_size + 1 entries";
    CHECK_HOST(total_pages >= 0 && total_pages <= UINT32_MAX) << "flashinfer_page_table: total_pages must fit uint32";
    CHECK_HOST(physical_page_count > 0 && physical_page_count <= UINT32_MAX)
        << "flashinfer_page_table: physical_page_count must fit uint32";
    CHECK_HOST(O.unwrap() >= total_pages) << "flashinfer_page_table: output capacity is smaller than total_pages";

    constexpr uint32_t kBlockSize = 256;
    LaunchKernel(static_cast<uint32_t>(B.unwrap()), kBlockSize, device.unwrap())(
        flashinfer_page_table_detail::build_page_table_kernel<kPageSize>,
        static_cast<int32_t*>(page_indices.data_ptr()),
        static_cast<const int32_t*>(req_to_token.data_ptr()),
        static_cast<const int64_t*>(req_pool_indices.data_ptr()),
        static_cast<const int32_t*>(page_lens.data_ptr()),
        static_cast<const int32_t*>(page_indptr.data_ptr()),
        static_cast<uint32_t>(B.unwrap()),
        static_cast<uint32_t>(C.unwrap()),
        static_cast<uint32_t>(total_pages),
        static_cast<uint32_t>(physical_page_count));
  }
};

}  // namespace sglang
