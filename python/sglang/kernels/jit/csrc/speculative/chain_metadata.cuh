#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace device::chain_metadata {

// Build every fixed-top-k1 chain artifact in one launch.  The general EAGLE
// tree builder has to discover the parent relation from selected indices;
// top-k1 makes that relation statically i -> i + 1.
__global__ void BuildChainMetadataKernel(
    const int64_t* __restrict__ bonus_tokens,
    const int64_t* __restrict__ draft_tokens,
    const int64_t* __restrict__ seq_lens,
    uint8_t* __restrict__ tree_mask,
    int64_t* __restrict__ positions,
    int64_t* __restrict__ retrieve_index,
    int64_t* __restrict__ retrieve_next_token,
    int64_t* __restrict__ retrieve_next_sibling,
    int64_t* __restrict__ output_tokens,
    int32_t num_slots) {
  const int32_t batch_idx = static_cast<int32_t>(blockIdx.x);
  const int32_t slot = static_cast<int32_t>(threadIdx.x);
  if (slot >= num_slots) return;

  const int64_t row_offset = static_cast<int64_t>(batch_idx) * num_slots;
  const int64_t item = row_offset + slot;

  output_tokens[item] =
      slot == 0 ? bonus_tokens[batch_idx]
                : draft_tokens[static_cast<int64_t>(batch_idx) * (num_slots - 1) + slot - 1];
  positions[item] = seq_lens[batch_idx] + slot;
  retrieve_index[item] = item;
  retrieve_next_token[item] = slot + 1 < num_slots ? slot + 1 : -1;
  retrieve_next_sibling[item] = -1;

  // QLEN_ONLY is a row-major [batch, query_slot, key_slot] causal chain.
  const int64_t mask_row =
      (static_cast<int64_t>(batch_idx) * num_slots + slot) * num_slots;
  for (int32_t key = 0; key < num_slots; ++key) {
    tree_mask[mask_row + key] = static_cast<uint8_t>(key <= slot);
  }
}

}  // namespace device::chain_metadata

struct ChainMetadataKernel {
  static void run(
      const tvm::ffi::TensorView bonus_tokens,
      const tvm::ffi::TensorView draft_tokens,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView tree_mask,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView retrieve_buf,
      const tvm::ffi::TensorView output_tokens) {
    using namespace host;

    auto batch_size = SymbolicSize{"batch_size"};
    auto num_steps = SymbolicSize{"num_steps"};
    auto num_slots = SymbolicSize{"num_slots"};
    auto device_ = SymbolicDevice{};

    TensorMatcher({batch_size})
        .with_dtype<int64_t>()
        .with_device<kDLGPU>(device_)
        .verify(bonus_tokens)
        .verify(seq_lens);
    TensorMatcher({batch_size, num_steps})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(draft_tokens);
    TensorMatcher({num_slots})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(positions)
        .verify(output_tokens);
    TensorMatcher({3, batch_size, -1})
        .with_dtype<int64_t>()
        .with_device(device_)
        .verify(retrieve_buf);
    TensorMatcher({-1}).with_device(device_).verify(tree_mask);

    const int64_t bs = batch_size.unwrap();
    const int64_t steps = num_steps.unwrap();
    const int64_t slots = steps + 1;
    RuntimeCheck(bs > 0, "chain metadata requires a non-empty batch");
    RuntimeCheck(slots <= 32, "chain metadata supports at most 32 verify slots, got ", slots);
    RuntimeCheck(
        num_slots.unwrap() == bs * slots,
        "positions/output size must be batch_size * (num_steps + 1)");
    RuntimeCheck(
        retrieve_buf.size(2) == slots,
        "retrieve buffer slot dimension must equal num_steps + 1");
    RuntimeCheck(
        tree_mask.size(0) >= bs * slots * slots,
        "QLEN_ONLY tree mask is too small for this chain");
    RuntimeCheck(
        tree_mask.dtype().bits == 8,
        "chain metadata expects a one-byte QLEN_ONLY mask");

    const auto device = device_.unwrap();
    LaunchKernel(static_cast<uint32_t>(bs), 32, device)(
        device::chain_metadata::BuildChainMetadataKernel,
        static_cast<const int64_t*>(bonus_tokens.data_ptr()),
        static_cast<const int64_t*>(draft_tokens.data_ptr()),
        static_cast<const int64_t*>(seq_lens.data_ptr()),
        static_cast<uint8_t*>(tree_mask.data_ptr()),
        static_cast<int64_t*>(positions.data_ptr()),
        static_cast<int64_t*>(retrieve_buf.data_ptr()),
        static_cast<int64_t*>(retrieve_buf.data_ptr()) + bs * slots,
        static_cast<int64_t*>(retrieve_buf.data_ptr()) + 2 * bs * slots,
        static_cast<int64_t*>(output_tokens.data_ptr()),
        static_cast<int32_t>(slots));
  }
};

}  // namespace sglang
