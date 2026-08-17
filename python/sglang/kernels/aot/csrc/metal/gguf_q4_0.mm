#include <torch/extension.h>

#include <ATen/mps/MPSStream.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace {

constexpr const char * kQ4Source = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {
    half d;
    uchar qs[16];
};

struct block_q4_1 {
    half d;
    half m;
    uchar qs[16];
};

struct block_q5_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

struct Q4Args {
    uint input_size;
    uint output_size;
    uint batch_size;
    uint blocks_per_row;
};

inline float4 dequantize_q4_0_t4(
        device const block_q4_0 * block,
        ushort chunk) {
    device const ushort * qs = reinterpret_cast<device const ushort *>(block->qs);
    const bool high = chunk >= 4;
    const float d1 = high ? float(block->d) / 16.0f : float(block->d);
    const float d2 = d1 / 256.0f;
    const float minimum = -8.0f * float(block->d);
    const ushort mask0 = high ? 0x00f0 : 0x000f;
    const ushort mask1 = mask0 << 8;
    const ushort local = chunk & 3;
    float4 value;
    for (ushort i = 0; i < 2; ++i) {
        const ushort packed = qs[2 * local + i];
        value[2 * i] = d1 * float(packed & mask0) + minimum;
        value[2 * i + 1] = d2 * float(packed & mask1) + minimum;
    }
    return value;
}

inline float dequant_value(device const block_q4_1 * block, uint index) {
    const uchar packed = block->qs[index & 15];
    const uint quant = index < 16 ? packed & 0x0f : packed >> 4;
    return float(block->d) * float(quant) + float(block->m);
}

inline uchar2 q5_k_scale_min(device const uchar * scales, uint index) {
    if (index < 4) {
        return uchar2(scales[index] & 63, scales[index + 4] & 63);
    }
    return uchar2(
        (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
        (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4));
}

inline float dequant_value(device const block_q5_K * block, uint index) {
    const uint group = index / 64;
    const uint within = index & 63;
    const uint lane = within & 31;
    const uint scale_index = group * 2 + (within >= 32 ? 1 : 0);
    const uchar2 scale_min = q5_k_scale_min(block->scales, scale_index);
    const uchar packed = block->qs[group * 32 + lane];
    const uint low = within < 32 ? packed & 0x0f : packed >> 4;
    const uint high_bit = 1u << (group * 2 + (within >= 32 ? 1 : 0));
    const uint quant = low + ((block->qh[lane] & high_bit) ? 16 : 0);
    return float(block->d) * float(scale_min[0]) * float(quant) -
        float(block->dmin) * float(scale_min[1]);
}

inline float dequant_value(device const block_q6_K * block, uint index) {
    const uint half_block = index / 128;
    const uint within = index & 127;
    const uint quadrant = within / 32;
    const uint lane = within & 31;
    const uint ql_offset = half_block * 64 + lane + ((quadrant & 1) ? 32 : 0);
    const uchar packed_low = block->ql[ql_offset];
    const uint low = quadrant < 2 ? packed_low & 0x0f : packed_low >> 4;
    const uchar packed_high = block->qh[half_block * 32 + lane];
    const uint high = (packed_high >> (quadrant * 2)) & 3;
    const int quant = int(low | (high << 4)) - 32;
    const int scale_index = int(half_block * 8 + lane / 16 + quadrant * 2);
    return float(block->d) * float(block->scales[scale_index]) * float(quant);
}

template <typename block_t, uint BlockSize, ushort BatchTile>
kernel void quant_small_batch_impl(
        device const block_t * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort simdgroups = 4;
    const uint row = group.x * simdgroups + simd_id;
    const uint batch_start = group.y * BatchTile;
    const bool valid_row = row < args.output_size;
    device const block_t * row_weights = valid_row
        ? weights + row * args.blocks_per_row
        : weights;

    float sums[BatchTile] = {[0 ... BatchTile - 1] = 0.0f};
    for (uint block_index = 0; block_index < args.blocks_per_row; ++block_index) {
        device const block_t * block = row_weights + block_index;
        for (uint local = lane; local < BlockSize; local += 32) {
            const float weight = dequant_value(block, local);
            const uint column = block_index * BlockSize + local;
#pragma unroll(BatchTile)
            for (ushort batch = 0; batch < BatchTile; ++batch) {
                const uint batch_index = batch_start + batch;
                if (batch_index < args.batch_size) {
                    sums[batch] += weight * input[batch_index * args.input_size + column];
                }
            }
        }
    }

#pragma unroll(BatchTile)
    for (ushort batch = 0; batch < BatchTile; ++batch) {
        sums[batch] = simd_sum(sums[batch]);
    }
    if (lane == 0 && valid_row) {
        for (ushort batch = 0; batch < BatchTile; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

#define INSTANTIATE_QUANT_BATCH(name, type, block_size, batch) \
template [[host_name(name)]] \
kernel decltype(quant_small_batch_impl<type, block_size, batch>) \
    quant_small_batch_impl<type, block_size, batch>;

INSTANTIATE_QUANT_BATCH("q4_1_batch_1", block_q4_1, 32, 1)
INSTANTIATE_QUANT_BATCH("q4_1_batch_4", block_q4_1, 32, 4)
INSTANTIATE_QUANT_BATCH("q4_1_batch_8", block_q4_1, 32, 8)
INSTANTIATE_QUANT_BATCH("q5_K_batch_1", block_q5_K, 256, 1)
INSTANTIATE_QUANT_BATCH("q5_K_batch_4", block_q5_K, 256, 4)
INSTANTIATE_QUANT_BATCH("q5_K_batch_8", block_q5_K, 256, 8)
INSTANTIATE_QUANT_BATCH("q6_K_batch_1", block_q6_K, 256, 1)
INSTANTIATE_QUANT_BATCH("q6_K_batch_4", block_q6_K, 256, 4)
INSTANTIATE_QUANT_BATCH("q6_K_batch_8", block_q6_K, 256, 8)

#undef INSTANTIATE_QUANT_BATCH

kernel void q6_K_batch_24_split(
        device const block_q6_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort subgroups = 4;
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 6;
    constexpr ushort batch_tile = subgroups * batches_per_subgroup;
    constexpr ushort rows_per_threadgroup = 4;
    const uint row = group.x * rows_per_threadgroup + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = group.y * batch_tile +
        subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    const bool active = valid_row && batch_start < args.batch_size;
    device const block_q6_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {
        [0 ... batches_per_subgroup - 1] = 0.0f
    };

    if (active) {
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q6_K * block = row_weights + block_index;
            for (uint local = thread_x; local < 256; local += lanes_per_subgroup) {
                const float weight = dequant_value(block, local);
                const uint column = block_index * 256 + local;
#pragma unroll(batches_per_subgroup)
                for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
                    const uint batch_index = batch_start + batch;
                    if (batch_index < args.batch_size) {
                        sums[batch] += weight *
                            input[batch_index * args.input_size + column];
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

kernel void q6_K_batch_24_split16(
        device const block_q6_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 16;
    constexpr ushort batches_per_subgroup = 12;
    constexpr ushort batch_tile = 24;
    const uint row = group.x * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = group.y * batch_tile +
        subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    const bool active = valid_row && batch_start < args.batch_size;
    device const block_q6_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {
        [0 ... batches_per_subgroup - 1] = 0.0f
    };
    device const float * input_rows[batches_per_subgroup];
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        const uint batch_index = batch_start + batch;
        input_rows[batch] = batch_index < args.batch_size
            ? input + batch_index * args.input_size : input;
    }
    if (active) {
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q6_K * block = row_weights + block_index;
            const float block_scale = float(block->d);
#pragma unroll(2)
            for (ushort half_block = 0; half_block < 2; ++half_block) {
#pragma unroll(4)
                for (ushort quadrant = 0; quadrant < 4; ++quadrant) {
#pragma unroll(2)
                    for (ushort lane_half = 0; lane_half < 2; ++lane_half) {
                        const uint lane32 = thread_x + lane_half * 16;
                        const uint local = half_block * 128 + quadrant * 32 + lane32;
                        const uint ql_offset = half_block * 64 + lane32
                            + ((quadrant & 1) ? 32 : 0);
                        const uchar packed_low = block->ql[ql_offset];
                        const uint low = quadrant < 2
                            ? packed_low & 0x0f : packed_low >> 4;
                        const uchar packed_high =
                            block->qh[half_block * 32 + lane32];
                        const uint high = (packed_high >> (quadrant * 2)) & 3;
                        const int quant = int(low | (high << 4)) - 32;
                        const uint scale_index =
                            half_block * 8 + lane_half + quadrant * 2;
                        const float weight = block_scale
                            * float(block->scales[scale_index]) * float(quant);
                        const uint column = block_index * 256 + local;
#pragma unroll(batches_per_subgroup)
                        for (ushort batch = 0;
                             batch < batches_per_subgroup;
                             ++batch) {
                            if (batch_start + batch < args.batch_size) {
                                sums[batch] += weight * input_rows[batch][column];
                            }
                        }
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 8);
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

kernel void q6_K_batch_24_vec4(
        device const block_q6_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 6;
    constexpr ushort batch_tile = 24;
    const uint row = group.x * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = group.y * batch_tile +
        subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    const bool active = valid_row;
    device const block_q6_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {
        [0 ... batches_per_subgroup - 1] = 0.0f
    };
    device const float * input_rows[batches_per_subgroup];
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        const uint batch_index = batch_start + batch;
        input_rows[batch] = input + batch_index * args.input_size;
    }
    if (active) {
        const uint lane_base = thread_x * 4;
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q6_K * block = row_weights + block_index;
            const float block_scale = float(block->d);
#pragma unroll(2)
            for (ushort half_block = 0; half_block < 2; ++half_block) {
#pragma unroll(4)
                for (ushort quadrant = 0; quadrant < 4; ++quadrant) {
                    const uint ql_offset = half_block * 64 + lane_base
                        + ((quadrant & 1) ? 32 : 0);
                    const uchar4 packed_low =
                        *reinterpret_cast<device const uchar4 *>(
                            block->ql + ql_offset);
                    const uchar4 low = quadrant < 2
                        ? packed_low & uchar4(0x0f) : packed_low >> 4;
                    const uchar4 packed_high =
                        *reinterpret_cast<device const uchar4 *>(
                            block->qh + half_block * 32 + lane_base);
                    const uchar4 high =
                        (packed_high >> (quadrant * 2)) & uchar4(3);
                    const int4 quant = int4(low | (high << 4)) - 32;
                    const uint scale_index = half_block * 8
                        + (lane_base / 16) + quadrant * 2;
                    const float4 weight = block_scale
                        * float(block->scales[scale_index]) * float4(quant);
                    const uint column = block_index * 256
                        + half_block * 128 + quadrant * 32 + lane_base;
#pragma unroll(batches_per_subgroup)
                    for (ushort batch = 0;
                         batch < batches_per_subgroup;
                         ++batch) {
                        sums[batch] += dot(
                            weight,
                            *reinterpret_cast<device const float4 *>(
                                input_rows[batch] + column));
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            const uint batch_index = batch_start + batch;
            output[batch_index * args.output_size + row] = sums[batch];
        }
    }
}

kernel void q6_K_batch_8_vec4(
        device const block_q6_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 2;
    const uint row = group * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    device const block_q6_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {0.0f, 0.0f};
    device const float * input_rows[batches_per_subgroup] = {
        input + batch_start * args.input_size,
        input + (batch_start + 1) * args.input_size,
    };
    if (valid_row) {
        const uint lane_base = thread_x * 4;
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q6_K * block = row_weights + block_index;
            const float block_scale = float(block->d);
#pragma unroll(2)
            for (ushort half_block = 0; half_block < 2; ++half_block) {
#pragma unroll(4)
                for (ushort quadrant = 0; quadrant < 4; ++quadrant) {
                    const uint ql_offset = half_block * 64 + lane_base
                        + ((quadrant & 1) ? 32 : 0);
                    const uchar4 packed_low =
                        *reinterpret_cast<device const uchar4 *>(
                            block->ql + ql_offset);
                    const uchar4 low = quadrant < 2
                        ? packed_low & uchar4(0x0f) : packed_low >> 4;
                    const uchar4 packed_high =
                        *reinterpret_cast<device const uchar4 *>(
                            block->qh + half_block * 32 + lane_base);
                    const uchar4 high =
                        (packed_high >> (quadrant * 2)) & uchar4(3);
                    const int4 quant = int4(low | (high << 4)) - 32;
                    const uint scale_index = half_block * 8
                        + (lane_base / 16) + quadrant * 2;
                    const float4 weight = block_scale
                        * float(block->scales[scale_index]) * float4(quant);
                    const uint column = block_index * 256
                        + half_block * 128 + quadrant * 32 + lane_base;
#pragma unroll(batches_per_subgroup)
                    for (ushort batch = 0;
                         batch < batches_per_subgroup;
                         ++batch) {
                        sums[batch] += dot(
                            weight,
                            *reinterpret_cast<device const float4 *>(
                                input_rows[batch] + column));
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
#pragma unroll(batches_per_subgroup)
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            output[(batch_start + batch) * args.output_size + row] = sums[batch];
        }
    }
}

kernel void q5_K_batch_8_vec4(
        device const block_q5_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 2;
    const uint row = group * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    device const block_q5_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {0.0f, 0.0f};
    device const float * input_rows[batches_per_subgroup] = {
        input + batch_start * args.input_size,
        input + (batch_start + 1) * args.input_size,
    };
    if (valid_row) {
        const uint lane_base = thread_x * 4;
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q5_K * block = row_weights + block_index;
            const float block_scale = float(block->d);
            const float block_min = float(block->dmin);
            const uchar4 packed_high =
                *reinterpret_cast<device const uchar4 *>(
                    block->qh + lane_base);
#pragma unroll(4)
            for (ushort group_index = 0; group_index < 4; ++group_index) {
                const uchar4 packed_low =
                    *reinterpret_cast<device const uchar4 *>(
                        block->qs + group_index * 32 + lane_base);
#pragma unroll(2)
                for (ushort nibble_half = 0; nibble_half < 2; ++nibble_half) {
                    const ushort scale_index = group_index * 2 + nibble_half;
                    const uchar2 scale_min =
                        q5_k_scale_min(block->scales, scale_index);
                    const uchar4 low = nibble_half == 0
                        ? packed_low & uchar4(0x0f) : packed_low >> 4;
                    const uchar4 high =
                        (packed_high >> scale_index) & uchar4(1);
                    const float4 weight =
                        block_scale * float(scale_min[0])
                            * float4(uint4(low) + uint4(high) * 16)
                        - block_min * float(scale_min[1]);
                    const uint column = block_index * 256
                        + group_index * 64 + nibble_half * 32 + lane_base;
#pragma unroll(batches_per_subgroup)
                    for (ushort batch = 0;
                         batch < batches_per_subgroup;
                         ++batch) {
                        sums[batch] += dot(
                            weight,
                            *reinterpret_cast<device const float4 *>(
                                input_rows[batch] + column));
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
#pragma unroll(batches_per_subgroup)
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            output[(batch_start + batch) * args.output_size + row] = sums[batch];
        }
    }
}

kernel void q5_K_batch_24_vec4(
        device const block_q5_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 6;
    constexpr ushort batch_tile = 24;
    const uint row = group.x * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = group.y * batch_tile
        + subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    device const block_q5_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batches_per_subgroup] = {
        [0 ... batches_per_subgroup - 1] = 0.0f
    };
    device const float * input_rows[batches_per_subgroup];
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        input_rows[batch] = input
            + (batch_start + batch) * args.input_size;
    }
    if (valid_row) {
        const uint lane_base = thread_x * 4;
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q5_K * block = row_weights + block_index;
            const float block_scale = float(block->d);
            const float block_min = float(block->dmin);
            const uchar4 packed_high =
                *reinterpret_cast<device const uchar4 *>(
                    block->qh + lane_base);
#pragma unroll(4)
            for (ushort group_index = 0; group_index < 4; ++group_index) {
                const uchar4 packed_low =
                    *reinterpret_cast<device const uchar4 *>(
                        block->qs + group_index * 32 + lane_base);
#pragma unroll(2)
                for (ushort nibble_half = 0;
                     nibble_half < 2;
                     ++nibble_half) {
                    const ushort scale_index =
                        group_index * 2 + nibble_half;
                    const uchar2 scale_min =
                        q5_k_scale_min(block->scales, scale_index);
                    const uchar4 low = nibble_half == 0
                        ? packed_low & uchar4(0x0f) : packed_low >> 4;
                    const uchar4 high =
                        (packed_high >> scale_index) & uchar4(1);
                    const float4 weight =
                        block_scale * float(scale_min[0])
                            * float4(uint4(low) + uint4(high) * 16)
                        - block_min * float(scale_min[1]);
                    const uint column = block_index * 256
                        + group_index * 64 + nibble_half * 32 + lane_base;
#pragma unroll(batches_per_subgroup)
                    for (ushort batch = 0;
                         batch < batches_per_subgroup;
                         ++batch) {
                        sums[batch] += dot(
                            weight,
                            *reinterpret_cast<device const float4 *>(
                                input_rows[batch] + column));
                    }
                }
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
#pragma unroll(batches_per_subgroup)
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            output[(batch_start + batch) * args.output_size + row] = sums[batch];
        }
    }
}

kernel void q6_K_batch_24_full(
        device const block_q6_K * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort batch_tile = 24;
    const uint row = group.x * 4 + simd_id;
    const uint batch_start = group.y * batch_tile;
    const bool valid_row = row < args.output_size;
    device const block_q6_K * row_weights = valid_row
        ? weights + row * args.blocks_per_row : weights;
    float sums[batch_tile] = {[0 ... batch_tile - 1] = 0.0f};
    if (valid_row) {
        for (uint block_index = 0;
             block_index < args.blocks_per_row;
             ++block_index) {
            device const block_q6_K * block = row_weights + block_index;
            for (uint local = lane; local < 256; local += 32) {
                const float weight = dequant_value(block, local);
                const uint column = block_index * 256 + local;
#pragma unroll(batch_tile)
                for (ushort batch = 0; batch < batch_tile; ++batch) {
                    const uint batch_index = batch_start + batch;
                    if (batch_index < args.batch_size) {
                        sums[batch] += weight *
                            input[batch_index * args.input_size + column];
                    }
                }
            }
        }
    }
#pragma unroll(batch_tile)
    for (ushort batch = 0; batch < batch_tile; ++batch) {
        sums[batch] = simd_sum(sums[batch]);
    }
    if (lane == 0 && valid_row) {
        for (ushort batch = 0; batch < batch_tile; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

kernel void dense_f32_batch_8(
        device const float * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort batch_tile = 8;
    constexpr ushort simdgroups = 4;
    const uint row = group.x * simdgroups + simd_id;
    const uint batch_start = group.y * batch_tile;
    const bool valid_row = row < args.output_size;
    float sums[batch_tile] = {[0 ... batch_tile - 1] = 0.0f};
    if (valid_row) {
        device const float * weight_row = weights + row * args.input_size;
        for (uint column = lane; column < args.input_size; column += 32) {
            const float weight = weight_row[column];
#pragma unroll(batch_tile)
            for (ushort batch = 0; batch < batch_tile; ++batch) {
                const uint batch_index = batch_start + batch;
                if (batch_index < args.batch_size) {
                    sums[batch] += weight *
                        input[batch_index * args.input_size + column];
                }
            }
        }
    }
#pragma unroll(batch_tile)
    for (ushort batch = 0; batch < batch_tile; ++batch) {
        sums[batch] = simd_sum(sums[batch]);
    }
    if (lane == 0 && valid_row) {
        for (ushort batch = 0; batch < batch_tile; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

template <ushort RowsPerBatchTile>
kernel void q4_0_small_batch_impl(
        device const block_q4_0 * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort simdgroups = 4;
    constexpr ushort threads_per_row = 8;
    constexpr ushort rows_per_simdgroup = 4;
    constexpr ushort rows_per_threadgroup = simdgroups * rows_per_simdgroup;
    constexpr ushort chunks_per_thread = 4;

    const ushort thread_x = lane % threads_per_row;
    const ushort thread_y = lane / threads_per_row;
    const uint row = group.x * rows_per_threadgroup +
                     simd_id * rows_per_simdgroup + thread_y;
    const uint batch_start = group.y * RowsPerBatchTile;
    const bool valid_row = row < args.output_size;

    device const block_q4_0 * quant = valid_row
        ? weights + row * args.blocks_per_row
        : weights;
    ushort chunk_in_block = thread_x;

    device const float4 * input_rows[RowsPerBatchTile];
    for (ushort batch = 0; batch < RowsPerBatchTile; ++batch) {
        const uint batch_index = batch_start + batch;
        input_rows[batch] = batch_index < args.batch_size
            ? reinterpret_cast<device const float4 *>(
                  input + batch_index * args.input_size) + thread_x
            : reinterpret_cast<device const float4 *>(input);
    }

    float sums[RowsPerBatchTile] = {
        [0 ... RowsPerBatchTile - 1] = 0.0f
    };

    for (uint chunk = thread_x;
         chunk * 4 < args.input_size;
         chunk += chunks_per_thread * threads_per_row) {
        float4 values[chunks_per_thread];
#pragma unroll(chunks_per_thread)
        for (ushort local = 0; local < chunks_per_thread; ++local) {
            values[local] = dequantize_q4_0_t4(quant, chunk_in_block);
            chunk_in_block += threads_per_row;
            if (chunk_in_block >= 8) {
                quant += chunk_in_block / 8;
                chunk_in_block %= 8;
            }
        }

#pragma unroll(chunks_per_thread)
        for (ushort local = 0; local < chunks_per_thread; ++local) {
#pragma unroll(RowsPerBatchTile)
            for (ushort batch = 0; batch < RowsPerBatchTile; ++batch) {
                sums[batch] += dot(values[local], input_rows[batch][local * threads_per_row]);
            }
        }

#pragma unroll(RowsPerBatchTile)
        for (ushort batch = 0; batch < RowsPerBatchTile; ++batch) {
            input_rows[batch] += chunks_per_thread * threads_per_row;
        }
    }

#pragma unroll(RowsPerBatchTile)
    for (ushort batch = 0; batch < RowsPerBatchTile; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }

    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < RowsPerBatchTile; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

template [[host_name("q4_0_batch_2")]]
kernel decltype(q4_0_small_batch_impl<2>) q4_0_small_batch_impl<2>;
template [[host_name("q4_0_batch_3")]]
kernel decltype(q4_0_small_batch_impl<3>) q4_0_small_batch_impl<3>;
template [[host_name("q4_0_batch_4")]]
kernel decltype(q4_0_small_batch_impl<4>) q4_0_small_batch_impl<4>;

kernel void q4_0_batch_8_split(
        device const block_q4_0 * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort simdgroups = 4;
    constexpr ushort threads_per_row_half = 8;
    constexpr ushort rows_per_simdgroup = 2;
    constexpr ushort rows_per_threadgroup = simdgroups * rows_per_simdgroup;
    constexpr ushort batches_per_half = 4;
    constexpr ushort batch_tile = 8;
    constexpr ushort chunks_per_thread = 4;

    const ushort row_in_simd = lane / 16;
    const ushort batch_half = (lane / 8) & 1;
    const ushort thread_x = lane & 7;
    const uint row = group.x * rows_per_threadgroup +
        simd_id * rows_per_simdgroup + row_in_simd;
    const uint batch_start = group.y * batch_tile +
        batch_half * batches_per_half;
    const bool valid_row = row < args.output_size;
    device const block_q4_0 * quant = valid_row
        ? weights + row * args.blocks_per_row : weights;
    ushort chunk_in_block = thread_x;

    device const float4 * input_rows[batches_per_half];
    for (ushort batch = 0; batch < batches_per_half; ++batch) {
        const uint batch_index = batch_start + batch;
        input_rows[batch] = batch_index < args.batch_size
            ? reinterpret_cast<device const float4 *>(
                  input + batch_index * args.input_size) + thread_x
            : reinterpret_cast<device const float4 *>(input);
    }
    float sums[batches_per_half] = {
        [0 ... batches_per_half - 1] = 0.0f
    };

    for (uint chunk = thread_x;
         chunk * 4 < args.input_size;
         chunk += chunks_per_thread * threads_per_row_half) {
        float4 values[chunks_per_thread];
#pragma unroll(chunks_per_thread)
        for (ushort local = 0; local < chunks_per_thread; ++local) {
            values[local] = dequantize_q4_0_t4(quant, chunk_in_block);
            chunk_in_block += threads_per_row_half;
            if (chunk_in_block >= 8) {
                quant += chunk_in_block / 8;
                chunk_in_block %= 8;
            }
        }
#pragma unroll(chunks_per_thread)
        for (ushort local = 0; local < chunks_per_thread; ++local) {
#pragma unroll(batches_per_half)
            for (ushort batch = 0; batch < batches_per_half; ++batch) {
                sums[batch] += dot(
                    values[local], input_rows[batch][local * threads_per_row_half]);
            }
        }
#pragma unroll(batches_per_half)
        for (ushort batch = 0; batch < batches_per_half; ++batch) {
            input_rows[batch] += chunks_per_thread * threads_per_row_half;
        }
    }

#pragma unroll(batches_per_half)
    for (ushort batch = 0; batch < batches_per_half; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < batches_per_half; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

kernel void q4_0_batch_24_split(
        device const block_q4_0 * weights,
        device const float * input,
        device float * output,
        constant Q4Args & args,
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    constexpr ushort lanes_per_subgroup = 8;
    constexpr ushort batches_per_subgroup = 6;
    constexpr ushort batch_tile = 24;
    constexpr ushort chunks_per_thread = 4;
    const uint row = group.x * 4 + simd_id;
    const ushort subgroup = lane / lanes_per_subgroup;
    const ushort thread_x = lane & (lanes_per_subgroup - 1);
    const uint batch_start = group.y * batch_tile +
        subgroup * batches_per_subgroup;
    const bool valid_row = row < args.output_size;
    const bool active = valid_row && batch_start < args.batch_size;
    device const block_q4_0 * quant = valid_row
        ? weights + row * args.blocks_per_row : weights;
    ushort chunk_in_block = thread_x;
    device const float4 * input_rows[batches_per_subgroup];
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        const uint batch_index = batch_start + batch;
        input_rows[batch] = batch_index < args.batch_size
            ? reinterpret_cast<device const float4 *>(
                  input + batch_index * args.input_size) + thread_x
            : reinterpret_cast<device const float4 *>(input);
    }
    float sums[batches_per_subgroup] = {
        [0 ... batches_per_subgroup - 1] = 0.0f
    };
    if (active) {
        for (uint chunk = thread_x;
             chunk * 4 < args.input_size;
             chunk += chunks_per_thread * lanes_per_subgroup) {
            float4 values[chunks_per_thread];
#pragma unroll(chunks_per_thread)
            for (ushort local = 0; local < chunks_per_thread; ++local) {
                values[local] = dequantize_q4_0_t4(quant, chunk_in_block);
                chunk_in_block += lanes_per_subgroup;
                if (chunk_in_block >= 8) {
                    quant += chunk_in_block / 8;
                    chunk_in_block %= 8;
                }
            }
#pragma unroll(chunks_per_thread)
            for (ushort local = 0; local < chunks_per_thread; ++local) {
#pragma unroll(batches_per_subgroup)
                for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
                    const uint batch_index = batch_start + batch;
                    if (batch_index < args.batch_size) {
                        sums[batch] += dot(
                            values[local],
                            input_rows[batch][local * lanes_per_subgroup]);
                    }
                }
            }
#pragma unroll(batches_per_subgroup)
            for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
                input_rows[batch] += chunks_per_thread * lanes_per_subgroup;
            }
        }
    }
#pragma unroll(batches_per_subgroup)
    for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
        sums[batch] += simd_shuffle_down(sums[batch], 4);
        sums[batch] += simd_shuffle_down(sums[batch], 2);
        sums[batch] += simd_shuffle_down(sums[batch], 1);
    }
    if (thread_x == 0 && valid_row) {
        for (ushort batch = 0; batch < batches_per_subgroup; ++batch) {
            const uint batch_index = batch_start + batch;
            if (batch_index < args.batch_size) {
                output[batch_index * args.output_size + row] = sums[batch];
            }
        }
    }
}

kernel void q4_0_embedding_f32(
        device const block_q4_0 * weights,
        device const long * token_ids,
        device float * output,
        constant Q4Args & args,
        uint index [[thread_position_in_grid]]) {
    const uint total = args.batch_size * args.input_size;
    if (index >= total) {
        return;
    }
    const uint batch = index / args.input_size;
    const uint column = index % args.input_size;
    const long token = token_ids[batch];
    if (token < 0 || uint(token) >= args.output_size) {
        output[index] = 0.0f;
        return;
    }
    device const block_q4_0 * block =
        weights + uint(token) * args.blocks_per_row + column / 32;
    const uint local = column % 32;
    const uchar packed = block->qs[local % 16];
    const uint quant = local < 16 ? packed & 0x0f : packed >> 4;
    output[index] = (float(quant) - 8.0f) * float(block->d);
}

struct GDNArgs {
    uint batch_size;
    uint num_k_heads;
    uint num_v_heads;
    uint key_dim;
    uint value_dim;
};

struct ConvArgs {
    uint num_tokens;
    uint channels;
    uint batch_size;
};

struct AttentionArgs {
    uint batch_size;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint cache_slots;
    uint req_stride;
    float scale;
};

struct NormArgs {
    uint rows;
    uint columns;
    float epsilon;
};

struct GatedNormArgs {
    uint batch_size;
    uint num_k_heads;
    uint num_v_heads;
    uint head_dim;
    float epsilon;
};

struct PackArgs {
    uint batch_size;
    uint key_dim;
    uint value_dim;
    uint num_v_heads;
};

struct GDNNormArgs {
    uint tokens;
    uint num_heads;
    uint head_dim;
};

struct FullAttentionPrepareArgs {
    uint tokens;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint rotary_dim;
    float epsilon;
};

inline float silu_f32(float value) {
    return value / (1.0f + exp(-value));
}

template <bool HasResidual>
kernel void gemma_rmsnorm_impl(
        device const float * input,
        device const float * residual,
        device const float * weight,
        device float * output,
        device float * residual_output,
        constant NormArgs & args,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.rows) {
        return;
    }
    device const float * input_row = input + row * args.columns;
    device const float * residual_row = residual + row * args.columns;
    device float * output_row = output + row * args.columns;
    device float * residual_out_row = residual_output + row * args.columns;
    float square_sum = 0.0f;
    for (uint column = tid; column < args.columns; column += 256) {
        const float value = input_row[column] +
            (HasResidual ? residual_row[column] : 0.0f);
        if (HasResidual) {
            residual_out_row[column] = value;
        }
        square_sum += value * value;
    }
    square_sum = simd_sum(square_sum);
    threadgroup float partial[8];
    if (lane == 0) {
        partial[simd_id] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id == 0) {
        float value = lane < 8 ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0) {
            partial[0] = rsqrt(value / float(args.columns) + args.epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inverse_rms = partial[0];
    for (uint column = tid; column < args.columns; column += 256) {
        const float value = HasResidual ? residual_out_row[column] : input_row[column];
        output_row[column] = value * inverse_rms * (1.0f + weight[column]);
    }
}

template [[host_name("gemma_rmsnorm_f32")]]
kernel decltype(gemma_rmsnorm_impl<false>) gemma_rmsnorm_impl<false>;
template [[host_name("gemma_fused_add_rmsnorm_f32")]]
kernel decltype(gemma_rmsnorm_impl<true>) gemma_rmsnorm_impl<true>;

kernel void silu_and_mul_f32(
        device const float * input,
        device float * output,
        constant NormArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint total = args.rows * args.columns;
    if (index >= total) {
        return;
    }
    const uint row = index / args.columns;
    const uint column = index - row * args.columns;
    const uint input_row = row * args.columns * 2;
    output[index] = silu_f32(input[input_row + column]) *
        input[input_row + args.columns + column];
}

kernel void gdn_gated_rmsnorm_reorder_f32(
        device const float * input,
        device const float * gate,
        device const float * weight,
        device float * output,
        constant GatedNormArgs & args,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint batch = group / args.num_v_heads;
    const uint value_head = group - batch * args.num_v_heads;
    if (batch >= args.batch_size) {
        return;
    }
    const uint base = (batch * args.num_v_heads + value_head) * args.head_dim;
    float square_sum = 0.0f;
    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        const float value = input[base + dim];
        square_sum += value * value;
    }
    const float inverse_rms =
        rsqrt(simd_sum(square_sum) / float(args.head_dim) + args.epsilon);
    const uint ratio = args.num_v_heads / args.num_k_heads;
    const uint key_head = value_head / ratio;
    const uint within_group = value_head - key_head * ratio;
    const uint tiled_head = within_group * args.num_k_heads + key_head;
    const uint output_base =
        (batch * args.num_v_heads + tiled_head) * args.head_dim;
    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        output[output_base + dim] = input[base + dim] * inverse_rms * weight[dim]
            * silu_f32(gate[base + dim]);
    }
}

kernel void pack_gdn_inputs_f32(
        device const float * qkvz,
        device const float * ba,
        device float * mixed_qkv,
        device float * gate,
        device float * b_out,
        device float * a_out,
        constant PackArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint mixed_dim = 2 * args.key_dim + args.value_dim;
    const uint qkvz_dim = mixed_dim + args.value_dim;
    const uint per_batch = mixed_dim + args.value_dim + 2 * args.num_v_heads;
    const uint total = args.batch_size * per_batch;
    if (index >= total) {
        return;
    }
    const uint batch = index / per_batch;
    const uint within = index - batch * per_batch;
    if (within < mixed_dim) {
        mixed_qkv[batch * mixed_dim + within] =
            qkvz[batch * qkvz_dim + within];
    } else if (within < mixed_dim + args.value_dim) {
        const uint offset = within - mixed_dim;
        gate[batch * args.value_dim + offset] =
            qkvz[batch * qkvz_dim + mixed_dim + offset];
    } else if (within < mixed_dim + args.value_dim + args.num_v_heads) {
        const uint offset = within - mixed_dim - args.value_dim;
        b_out[batch * args.num_v_heads + offset] =
            ba[batch * 2 * args.num_v_heads + offset];
    } else {
        const uint offset = within - mixed_dim - args.value_dim - args.num_v_heads;
        a_out[batch * args.num_v_heads + offset] =
            ba[batch * 2 * args.num_v_heads + args.num_v_heads + offset];
    }
}

kernel void normalize_gdn_qk_f32(
        device const float * query,
        device const float * key,
        device float * query_out,
        device float * key_out,
        constant GDNNormArgs & args,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint token = group / args.num_heads;
    const uint head = group - token * args.num_heads;
    if (token >= args.tokens) {
        return;
    }
    const uint base = (token * args.num_heads + head) * args.head_dim;
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        const float q_value = query[base + dim];
        const float k_value = key[base + dim];
        q_sum += q_value * q_value;
        k_sum += k_value * k_value;
    }
    const float q_scale = rsqrt(simd_sum(q_sum) / float(args.head_dim) + 1e-6f)
        / float(args.head_dim);
    const float k_scale = rsqrt(simd_sum(k_sum) / float(args.head_dim) + 1e-6f)
        / sqrt(float(args.head_dim));
    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        query_out[base + dim] = query[base + dim] * q_scale;
        key_out[base + dim] = key[base + dim] * k_scale;
    }
}

kernel void prepare_full_attention_f32(
        device const float * qkv,
        device const float * q_weight,
        device const float * k_weight,
        device const float * cos_sin_cache,
        device const long * positions,
        device float * query_out,
        device float * key_out,
        device float * value_out,
        device float * gate_out,
        constant FullAttentionPrepareArgs & args,
        uint group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint heads_per_token = args.num_q_heads + args.num_kv_heads;
    const uint token = group / heads_per_token;
    const uint packed_head = group - token * heads_per_token;
    if (token >= args.tokens) {
        return;
    }
    const bool is_query = packed_head < args.num_q_heads;
    const uint head = is_query ? packed_head : packed_head - args.num_q_heads;
    const uint q_gate_dim = 2 * args.num_q_heads * args.head_dim;
    const uint kv_dim = args.num_kv_heads * args.head_dim;
    const uint qkv_stride = q_gate_dim + 2 * kv_dim;
    const uint source_base = token * qkv_stride +
        (is_query ? head * 2 * args.head_dim
                  : q_gate_dim + head * args.head_dim);
    device const float * norm_weight = is_query ? q_weight : k_weight;

    float square_sum = 0.0f;
    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        const float value = qkv[source_base + dim];
        square_sum += value * value;
    }
    const float inverse_rms =
        rsqrt(simd_sum(square_sum) / float(args.head_dim) + args.epsilon);
    const uint output_base =
        (token * (is_query ? args.num_q_heads : args.num_kv_heads) + head)
        * args.head_dim;
    const uint half_rotary = args.rotary_dim / 2;
    const uint position = uint(positions[token]);
    device const float * cache_row = cos_sin_cache + position * args.rotary_dim;

    for (uint dim = lane; dim < args.head_dim; dim += 32) {
        float output_value;
        if (dim < args.rotary_dim) {
            const uint pair_dim = dim < half_rotary
                ? dim + half_rotary : dim - half_rotary;
            const uint frequency = dim < half_rotary ? dim : dim - half_rotary;
            const float first = qkv[source_base + (dim < half_rotary ? dim : pair_dim)]
                * inverse_rms
                * (1.0f + norm_weight[dim < half_rotary ? dim : pair_dim]);
            const float second = qkv[source_base + (dim < half_rotary ? pair_dim : dim)]
                * inverse_rms
                * (1.0f + norm_weight[dim < half_rotary ? pair_dim : dim]);
            const float cosine = cache_row[frequency];
            const float sine = cache_row[half_rotary + frequency];
            output_value = dim < half_rotary
                ? first * cosine - second * sine
                : second * cosine + first * sine;
        } else {
            output_value = qkv[source_base + dim] * inverse_rms
                * (1.0f + norm_weight[dim]);
        }
        if (is_query) {
            query_out[output_base + dim] = output_value;
            gate_out[output_base + dim] =
                qkv[source_base + args.head_dim + dim];
        } else {
            key_out[output_base + dim] = output_value;
            value_out[output_base + dim] =
                qkv[token * qkv_stride + q_gate_dim + kv_dim
                    + head * args.head_dim + dim];
        }
    }
}

kernel void sigmoid_mul_inplace_f32(
        device float * input,
        device const float * gate,
        constant NormArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint total = args.rows * args.columns;
    if (index < total) {
        input[index] *= 1.0f / (1.0f + exp(-gate[index]));
    }
}

kernel void causal_conv1d_decode_f32(
        device const float * input,
        device const float * weight,
        device float * state,
        device const int * cache_indices,
        device float * output,
        constant ConvArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint total = args.batch_size * args.channels;
    if (index >= total) {
        return;
    }
    const uint batch = index / args.channels;
    const uint channel = index % args.channels;
    const int slot = cache_indices[batch];
    if (slot < 0) {
        output[index] = 0.0f;
        return;
    }
    device float * history = state +
        (uint(slot) * args.channels + channel) * 3;
    const float current = input[index];
    const float value = history[0] * weight[channel * 4] +
        history[1] * weight[channel * 4 + 1] +
        history[2] * weight[channel * 4 + 2] +
        current * weight[channel * 4 + 3];
    history[0] = history[1];
    history[1] = history[2];
    history[2] = current;
    output[index] = silu_f32(value);
}

kernel void causal_conv1d_prefill_f32(
        device const float * input,
        device const float * weight,
        device float * state,
        device const int * cache_indices,
        device const int * query_start_loc,
        device const int * has_initial_state,
        device float * output,
        constant ConvArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint total = args.batch_size * args.channels;
    if (index >= total) {
        return;
    }
    const uint sequence = index / args.channels;
    const uint channel = index % args.channels;
    const int slot = cache_indices[sequence];
    if (slot < 0) {
        return;
    }
    device float * state_row = state +
        (uint(slot) * args.channels + channel) * 3;
    float h0 = has_initial_state[sequence] ? state_row[0] : 0.0f;
    float h1 = has_initial_state[sequence] ? state_row[1] : 0.0f;
    float h2 = has_initial_state[sequence] ? state_row[2] : 0.0f;
    const uint start = uint(query_start_loc[sequence]);
    const uint end = uint(query_start_loc[sequence + 1]);
    for (uint token = start; token < end; ++token) {
        const float current = input[channel * args.num_tokens + token];
        const float value = h0 * weight[channel * 4] +
            h1 * weight[channel * 4 + 1] +
            h2 * weight[channel * 4 + 2] +
            current * weight[channel * 4 + 3];
        output[channel * args.num_tokens + token] = silu_f32(value);
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    state_row[0] = h0;
    state_row[1] = h1;
    state_row[2] = h2;
}

kernel void gdn_decode_f32(
        device const float * query,
        device const float * key,
        device const float * value,
        device const float * a,
        device const float * b,
        device const float * A_log,
        device const float * dt_bias,
        device float * state,
        device const int * cache_indices,
        device float * output,
        constant GDNArgs & args,
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 local [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint packed_head = group.z;
    const uint batch = packed_head / args.num_v_heads;
    const uint value_head = packed_head % args.num_v_heads;
    const uint key_head = value_head / (args.num_v_heads / args.num_k_heads);
    const uint value_index = group.y * 4 + local.y;
    if (batch >= args.batch_size || value_index >= args.value_dim) {
        return;
    }

    const uint state_slot = uint(cache_indices[batch]);
    device float * state_row = state +
        ((state_slot * args.num_v_heads + value_head) * args.value_dim + value_index) * args.key_dim;
    device const float * q_row = query +
        (batch * args.num_k_heads + key_head) * args.key_dim;
    device const float * k_row = key +
        (batch * args.num_k_heads + key_head) * args.key_dim;

    const float gate_input = a[batch * args.num_v_heads + value_head] + dt_bias[value_head];
    const float softplus = gate_input > 20.0f
        ? gate_input
        : log(1.0f + exp(gate_input));
    const float decay = exp(-exp(A_log[value_head]) * softplus);
    const float beta = 1.0f /
        (1.0f + exp(-b[batch * args.num_v_heads + value_head]));

    float state_values[4];
    float remembered = 0.0f;
    for (uint index = 0; index < 4; ++index) {
        const uint key_index = lane + index * 32;
        state_values[index] = state_row[key_index] * decay;
        remembered += state_values[index] * k_row[key_index];
    }
    remembered = simd_sum(remembered);

    const float delta =
        (value[(batch * args.num_v_heads + value_head) * args.value_dim + value_index] - remembered) * beta;
    float result = 0.0f;
    for (uint index = 0; index < 4; ++index) {
        const uint key_index = lane + index * 32;
        state_values[index] += k_row[key_index] * delta;
        state_row[key_index] = state_values[index];
        result += state_values[index] * q_row[key_index];
    }
    result = simd_sum(result);
    if (lane == 0) {
        output[(batch * args.num_v_heads + value_head) * args.value_dim + value_index] = result;
    }
}

kernel void gdn_prefill_f32(
        device const float * query,
        device const float * key,
        device const float * value,
        device const float * g,
        device const float * beta,
        device float * state,
        device const int * cache_indices,
        device const int * query_start_loc,
        device float * output,
        constant GDNArgs & args,
        uint3 group [[threadgroup_position_in_grid]],
        ushort3 local [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint packed_head = group.z;
    const uint sequence = packed_head / args.num_v_heads;
    const uint value_head = packed_head % args.num_v_heads;
    const uint key_head = value_head / (args.num_v_heads / args.num_k_heads);
    const uint value_index = group.y * 4 + local.y;
    if (sequence >= args.batch_size || value_index >= args.value_dim) {
        return;
    }

    const uint start = uint(query_start_loc[sequence]);
    const uint end = uint(query_start_loc[sequence + 1]);
    const uint state_slot = uint(cache_indices[sequence]);
    device float * state_row = state +
        ((state_slot * args.num_v_heads + value_head) * args.value_dim + value_index) * args.key_dim;

    float state_values[4];
    for (uint index = 0; index < 4; ++index) {
        state_values[index] = state_row[lane + index * 32];
    }

    for (uint token = start; token < end; ++token) {
        device const float * q_row = query +
            (token * args.num_k_heads + key_head) * args.key_dim;
        device const float * k_row = key +
            (token * args.num_k_heads + key_head) * args.key_dim;
        const float decay = g[token * args.num_v_heads + value_head];
        const float beta_value = beta[token * args.num_v_heads + value_head];

        float remembered = 0.0f;
        for (uint index = 0; index < 4; ++index) {
            const uint key_index = lane + index * 32;
            state_values[index] *= decay;
            remembered += state_values[index] * k_row[key_index];
        }
        remembered = simd_sum(remembered);

        const float delta =
            (value[(token * args.num_v_heads + value_head) * args.value_dim + value_index] - remembered)
            * beta_value;
        float result = 0.0f;
        for (uint index = 0; index < 4; ++index) {
            const uint key_index = lane + index * 32;
            state_values[index] += k_row[key_index] * delta;
            result += state_values[index] * q_row[key_index];
        }
        result = simd_sum(result);
        if (lane == 0) {
            output[(token * args.num_v_heads + value_head) * args.value_dim + value_index] = result;
        }
    }

    for (uint index = 0; index < 4; ++index) {
        state_row[lane + index * 32] = state_values[index];
    }
}

kernel void store_decode_kv_f32(
        device const float * key,
        device const float * value,
        device float * key_cache,
        device float * value_cache,
        device const long * cache_locations,
        constant AttentionArgs & args,
        uint index [[thread_position_in_grid]]) {
    const uint row_size = args.num_kv_heads * args.head_dim;
    const uint total = args.batch_size * row_size;
    if (index >= total) {
        return;
    }
    const uint batch = index / row_size;
    const long slot = cache_locations[batch];
    if (slot < 0 || uint(slot) >= args.cache_slots) {
        return;
    }
    const uint within = index - batch * row_size;
    const uint cache_index = uint(slot) * row_size + within;
    key_cache[cache_index] = key[index];
    value_cache[cache_index] = value[index];
}

kernel void decode_gqa_f32(
        device const float * query,
        device const float * key_cache,
        device const float * value_cache,
        device const int * req_to_token,
        device const long * req_pool_indices,
        device const long * seq_lens,
        device float * output,
        constant AttentionArgs & args,
        threadgroup float * scratch [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd_id [[simdgroup_index_in_threadgroup]]) {
    const uint batch = group / args.num_q_heads;
    const uint query_head = group - batch * args.num_q_heads;
    if (batch >= args.batch_size) {
        return;
    }
    const uint kv_head = query_head / (args.num_q_heads / args.num_kv_heads);
    const uint seq_len = min(uint(max(seq_lens[batch], long(0))), args.cache_slots);
    const long req_slot = req_pool_indices[batch];
    if (req_slot < 0 || seq_len == 0) {
        return;
    }

    device const float * q_row = query +
        (batch * args.num_q_heads + query_head) * args.head_dim;
    const uint cache_row_size = args.num_kv_heads * args.head_dim;

    // One SIMD group computes one QK score at a time. Eight SIMD groups cover
    // eight cache positions concurrently while reusing the same query row.
    for (uint token = simd_id; token < seq_len; token += 8) {
        const int cache_slot = req_to_token[uint(req_slot) * args.req_stride + token];
        float dot_value = 0.0f;
        if (cache_slot >= 0 && uint(cache_slot) < args.cache_slots) {
            device const float * k_row = key_cache +
                uint(cache_slot) * cache_row_size + kv_head * args.head_dim;
            for (uint dim = lane; dim < args.head_dim; dim += 32) {
                dot_value += q_row[dim] * k_row[dim];
            }
        }
        dot_value = simd_sum(dot_value) * args.scale;
        if (lane == 0) {
            scratch[token] = dot_value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * reduction = scratch + args.cache_slots;
    float local_max = -INFINITY;
    for (uint token = tid; token < seq_len; token += 256) {
        local_max = max(local_max, scratch[token]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] = max(reduction[tid], reduction[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = reduction[0];

    float local_sum = 0.0f;
    for (uint token = tid; token < seq_len; token += 256) {
        const float probability = exp(scratch[token] - max_score);
        scratch[token] = probability;
        local_sum += probability;
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inverse_sum = 1.0f / reduction[0];

    if (tid < args.head_dim) {
        float result = 0.0f;
        for (uint token = 0; token < seq_len; ++token) {
            const int cache_slot =
                req_to_token[uint(req_slot) * args.req_stride + token];
            if (cache_slot >= 0 && uint(cache_slot) < args.cache_slots) {
                const uint value_index = uint(cache_slot) * cache_row_size +
                    kv_head * args.head_dim + tid;
                result += scratch[token] * value_cache[value_index];
            }
        }
        output[(batch * args.num_q_heads + query_head) * args.head_dim + tid] =
            result * inverse_sum;
    }
}
)METAL";

struct Q4Args {
    uint32_t input_size;
    uint32_t output_size;
    uint32_t batch_size;
    uint32_t blocks_per_row;
};

struct GDNArgs {
    uint32_t batch_size;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t key_dim;
    uint32_t value_dim;
};

struct ConvArgs {
    uint32_t num_tokens;
    uint32_t channels;
    uint32_t batch_size;
};

struct AttentionArgs {
    uint32_t batch_size;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t cache_slots;
    uint32_t req_stride;
    float scale;
};

struct NormArgs {
    uint32_t rows;
    uint32_t columns;
    float epsilon;
};

struct GatedNormArgs {
    uint32_t batch_size;
    uint32_t num_k_heads;
    uint32_t num_v_heads;
    uint32_t head_dim;
    float epsilon;
};

struct PackArgs {
    uint32_t batch_size;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t num_v_heads;
};

struct GDNNormArgs {
    uint32_t tokens;
    uint32_t num_heads;
    uint32_t head_dim;
};

struct FullAttentionPrepareArgs {
    uint32_t tokens;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    float epsilon;
};

struct Pipelines {
    id<MTLComputePipelineState> batch2 = nil;
    id<MTLComputePipelineState> batch3 = nil;
    id<MTLComputePipelineState> batch4 = nil;
    id<MTLComputePipelineState> batch8_split = nil;
    id<MTLComputePipelineState> batch24_split = nil;
    id<MTLComputePipelineState> q4_1_batch1 = nil;
    id<MTLComputePipelineState> q4_1_batch4 = nil;
    id<MTLComputePipelineState> q4_1_batch8 = nil;
    id<MTLComputePipelineState> q5_K_batch1 = nil;
    id<MTLComputePipelineState> q5_K_batch4 = nil;
    id<MTLComputePipelineState> q5_K_batch8 = nil;
    id<MTLComputePipelineState> q5_K_batch8_vec4 = nil;
    id<MTLComputePipelineState> q5_K_batch24_vec4 = nil;
    id<MTLComputePipelineState> q6_K_batch1 = nil;
    id<MTLComputePipelineState> q6_K_batch4 = nil;
    id<MTLComputePipelineState> q6_K_batch8 = nil;
    id<MTLComputePipelineState> q6_K_batch8_vec4 = nil;
    id<MTLComputePipelineState> q6_K_batch24_split = nil;
    id<MTLComputePipelineState> q6_K_batch24_split16 = nil;
    id<MTLComputePipelineState> q6_K_batch24_vec4 = nil;
    id<MTLComputePipelineState> q6_K_batch24_full = nil;
    id<MTLComputePipelineState> gdn_decode = nil;
    id<MTLComputePipelineState> gdn_prefill = nil;
    id<MTLComputePipelineState> conv_decode = nil;
    id<MTLComputePipelineState> conv_prefill = nil;
    id<MTLComputePipelineState> embedding = nil;
    id<MTLComputePipelineState> store_decode_kv = nil;
    id<MTLComputePipelineState> decode_gqa = nil;
    id<MTLComputePipelineState> gemma_rmsnorm = nil;
    id<MTLComputePipelineState> gemma_fused_add_rmsnorm = nil;
    id<MTLComputePipelineState> silu_and_mul = nil;
    id<MTLComputePipelineState> gdn_gated_norm_reorder = nil;
    id<MTLComputePipelineState> pack_gdn_inputs = nil;
    id<MTLComputePipelineState> normalize_gdn_qk = nil;
    id<MTLComputePipelineState> dense_f32_batch8 = nil;
    id<MTLComputePipelineState> prepare_full_attention = nil;
    id<MTLComputePipelineState> sigmoid_mul_inplace = nil;
};

Pipelines & pipelines() {
    static Pipelines value;
    static std::once_flag once;
    std::call_once(once, [&] {
        id<MTLDevice> device = at::mps::getCurrentMPSStream()->device();
        NSError * error = nil;
        MTLCompileOptions * options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        id<MTLLibrary> library = [device newLibraryWithSource:@(kQ4Source)
                                                      options:options
                                                        error:&error];
        if (library == nil) {
            throw std::runtime_error(
                "Failed to compile native Metal Q4_0 kernels: " +
                std::string(error.localizedDescription.UTF8String));
        }

        auto compile = [&](NSString * name) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (function == nil) {
                throw std::runtime_error(
                    "Native Metal function is missing: " +
                    std::string(name.UTF8String));
            }
            NSError * pipeline_error = nil;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function
                                                       error:&pipeline_error];
            if (pipeline == nil) {
                throw std::runtime_error(
                    "Failed to create native Metal Q4_0 pipeline: " +
                    std::string(pipeline_error.localizedDescription.UTF8String));
            }
            return pipeline;
        };

        value.batch2 = compile(@"q4_0_batch_2");
        value.batch3 = compile(@"q4_0_batch_3");
        value.batch4 = compile(@"q4_0_batch_4");
        value.batch8_split = compile(@"q4_0_batch_8_split");
        value.batch24_split = compile(@"q4_0_batch_24_split");
        value.q4_1_batch1 = compile(@"q4_1_batch_1");
        value.q4_1_batch4 = compile(@"q4_1_batch_4");
        value.q4_1_batch8 = compile(@"q4_1_batch_8");
        value.q5_K_batch1 = compile(@"q5_K_batch_1");
        value.q5_K_batch4 = compile(@"q5_K_batch_4");
        value.q5_K_batch8 = compile(@"q5_K_batch_8");
        value.q5_K_batch8_vec4 = compile(@"q5_K_batch_8_vec4");
        value.q5_K_batch24_vec4 = compile(@"q5_K_batch_24_vec4");
        value.q6_K_batch1 = compile(@"q6_K_batch_1");
        value.q6_K_batch4 = compile(@"q6_K_batch_4");
        value.q6_K_batch8 = compile(@"q6_K_batch_8");
        value.q6_K_batch8_vec4 = compile(@"q6_K_batch_8_vec4");
        value.q6_K_batch24_split = compile(@"q6_K_batch_24_split");
        value.q6_K_batch24_split16 = compile(@"q6_K_batch_24_split16");
        value.q6_K_batch24_vec4 = compile(@"q6_K_batch_24_vec4");
        value.q6_K_batch24_full = compile(@"q6_K_batch_24_full");
        value.gdn_decode = compile(@"gdn_decode_f32");
        value.gdn_prefill = compile(@"gdn_prefill_f32");
        value.conv_decode = compile(@"causal_conv1d_decode_f32");
        value.conv_prefill = compile(@"causal_conv1d_prefill_f32");
        value.embedding = compile(@"q4_0_embedding_f32");
        value.store_decode_kv = compile(@"store_decode_kv_f32");
        value.decode_gqa = compile(@"decode_gqa_f32");
        value.gemma_rmsnorm = compile(@"gemma_rmsnorm_f32");
        value.gemma_fused_add_rmsnorm = compile(@"gemma_fused_add_rmsnorm_f32");
        value.silu_and_mul = compile(@"silu_and_mul_f32");
        value.gdn_gated_norm_reorder =
            compile(@"gdn_gated_rmsnorm_reorder_f32");
        value.pack_gdn_inputs = compile(@"pack_gdn_inputs_f32");
        value.normalize_gdn_qk = compile(@"normalize_gdn_qk_f32");
        value.dense_f32_batch8 = compile(@"dense_f32_batch_8");
        value.prepare_full_attention = compile(@"prepare_full_attention_f32");
        value.sigmoid_mul_inplace = compile(@"sigmoid_mul_inplace_f32");
    });
    return value;
}

torch::Tensor q4_0_matmul(
    const torch::Tensor & packed_weight,
    const torch::Tensor & input,
    int64_t output_size,
    int64_t input_size) {
    TORCH_CHECK(packed_weight.device().is_mps(), "packed_weight must be on MPS");
    TORCH_CHECK(input.device().is_mps(), "input must be on MPS");
    TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
                "packed_weight must have dtype uint8");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype float32");
    TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input_size > 0 && input_size % 32 == 0,
                "input_size must be a positive multiple of 32");
    TORCH_CHECK(output_size > 0, "output_size must be positive");
    TORCH_CHECK(input.numel() % input_size == 0,
                "input element count must be divisible by input_size");

    const int64_t batch_size = input.numel() / input_size;
    TORCH_CHECK(batch_size >= 1,
                "native Q4_0 matmul requires a non-empty batch");
    const int64_t expected_weight_bytes = output_size * (input_size / 32) * 18;
    TORCH_CHECK(packed_weight.numel() >= expected_weight_bytes,
                "packed_weight is smaller than the Q4_0 matrix shape requires");

    auto output = torch::empty(
        {batch_size, output_size}, input.options().dtype(torch::kFloat32));

    id<MTLBuffer> weight_buffer =
        (__bridge id<MTLBuffer>)packed_weight.storage().data_ptr().get();
    id<MTLBuffer> input_buffer =
        (__bridge id<MTLBuffer>)input.storage().data_ptr().get();
    id<MTLBuffer> output_buffer =
        (__bridge id<MTLBuffer>)output.storage().data_ptr().get();

    const NSUInteger weight_offset =
        packed_weight.storage_offset() * packed_weight.element_size();
    const NSUInteger input_offset = input.storage_offset() * input.element_size();
    const NSUInteger output_offset = output.storage_offset() * output.element_size();

    const int64_t rows_per_tile = batch_size <= 2 ? 2 : (batch_size <= 6 ? 3 : 4);
    id<MTLComputePipelineState> pipeline =
        rows_per_tile == 2 ? pipelines().batch2
                           : (rows_per_tile == 3 ? pipelines().batch3
                                                 : pipelines().batch4);
    Q4Args args = {
        static_cast<uint32_t>(input_size),
        static_cast<uint32_t>(output_size),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(input_size / 32),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
        [encoder setBuffer:input_buffer offset:input_offset atIndex:1];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        const NSUInteger output_groups = (output_size + 15) / 16;
        const NSUInteger batch_groups = (batch_size + rows_per_tile - 1) / rows_per_tile;
        [encoder dispatchThreadgroups:MTLSizeMake(output_groups, batch_groups, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
    return output;
}

torch::Tensor dense_matmul(
    const torch::Tensor & weight,
    const torch::Tensor & input) {
    for (const auto & tensor : {weight, input}) {
        TORCH_CHECK(tensor.device().is_mps() &&
                        tensor.scalar_type() == torch::kFloat32 &&
                        tensor.is_contiguous(),
                    "native Metal dense matmul requires contiguous float32 MPS tensors");
    }
    TORCH_CHECK(weight.dim() == 2 && input.dim() == 2 &&
                    input.size(1) == weight.size(1),
                "dense matmul expects [output,input] weight and [batch,input] input");
    const int64_t batch_size = input.size(0);
    const int64_t input_size = input.size(1);
    const int64_t output_size = weight.size(0);
    auto output = torch::empty({batch_size, output_size}, input.options());
    Q4Args args = {
        static_cast<uint32_t>(input_size),
        static_cast<uint32_t>(output_size),
        static_cast<uint32_t>(batch_size),
        0,
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().dense_f32_batch8];
        [encoder setBuffer:buffer_of(weight) offset:offset_of(weight) atIndex:0];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:1];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(
                (output_size + 3) / 4, (batch_size + 7) / 8, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
    return output;
}

torch::Tensor quant_matmul(
    const torch::Tensor & packed_weight,
    const torch::Tensor & input,
    int64_t output_size,
    int64_t input_size,
    int64_t weight_type) {
    TORCH_CHECK(packed_weight.device().is_mps(), "packed_weight must be on MPS");
    TORCH_CHECK(input.device().is_mps(), "input must be on MPS");
    TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
                "packed_weight must have dtype uint8");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "input must have dtype float32");
    TORCH_CHECK(packed_weight.is_contiguous() && input.is_contiguous(),
                "quantized matmul inputs must be contiguous");
    TORCH_CHECK(output_size > 0 && input_size > 0,
                "matrix dimensions must be positive");
    TORCH_CHECK(input.numel() % input_size == 0,
                "input element count must be divisible by input_size");

    int64_t block_size = 0;
    int64_t type_size = 0;
    if (weight_type == 3) {          // GGML_TYPE_Q4_1
        block_size = 32;
        type_size = 20;
    } else if (weight_type == 13) {  // GGML_TYPE_Q5_K
        block_size = 256;
        type_size = 176;
    } else if (weight_type == 14) {  // GGML_TYPE_Q6_K
        block_size = 256;
        type_size = 210;
    } else {
        TORCH_CHECK(false, "native Metal quant_matmul received unsupported type");
    }
    TORCH_CHECK(input_size % block_size == 0,
                "input_size must be divisible by the quantization block size");
    const int64_t batch_size = input.numel() / input_size;
    TORCH_CHECK(batch_size >= 1, "quantized matmul requires a non-empty batch");
    const int64_t blocks_per_row = input_size / block_size;
    const int64_t expected_weight_bytes = output_size * blocks_per_row * type_size;
    TORCH_CHECK(packed_weight.numel() >= expected_weight_bytes,
                "packed_weight is smaller than the quantized matrix shape requires");

    auto output = torch::empty(
        {batch_size, output_size}, input.options().dtype(torch::kFloat32));
    id<MTLBuffer> weight_buffer =
        (__bridge id<MTLBuffer>)packed_weight.storage().data_ptr().get();
    id<MTLBuffer> input_buffer =
        (__bridge id<MTLBuffer>)input.storage().data_ptr().get();
    id<MTLBuffer> output_buffer =
        (__bridge id<MTLBuffer>)output.storage().data_ptr().get();
    const NSUInteger weight_offset =
        packed_weight.storage_offset() * packed_weight.element_size();
    const NSUInteger input_offset = input.storage_offset() * input.element_size();
    const NSUInteger output_offset = output.storage_offset() * output.element_size();

    const bool use_q6_batch24 = weight_type == 14 && batch_size >= 12;
    const bool use_q6_vec24 = weight_type == 14 && batch_size == 24;
    const bool use_q5_vec24 = weight_type == 13 && batch_size == 24;
    const int64_t batch_tile = (use_q6_batch24 || use_q5_vec24)
        ? 24 : (batch_size == 1 ? 1 : (batch_size <= 4 ? 4 : 8));
    Pipelines & p = pipelines();
    id<MTLComputePipelineState> pipeline = nil;
    if (weight_type == 3) {
        pipeline = batch_tile == 1 ? p.q4_1_batch1
            : (batch_tile == 4 ? p.q4_1_batch4 : p.q4_1_batch8);
    } else if (weight_type == 13) {
        pipeline = use_q5_vec24 ? p.q5_K_batch24_vec4
            : (batch_tile == 1 ? p.q5_K_batch1
                : (batch_tile == 4 ? p.q5_K_batch4
                    : (batch_size == 8 ? p.q5_K_batch8_vec4
                                       : p.q5_K_batch8)));
    } else {
        pipeline = use_q6_vec24 ? p.q6_K_batch24_vec4
            : (use_q6_batch24 ? p.q6_K_batch24_split16
            : (batch_tile == 1 ? p.q6_K_batch1
                : (batch_tile == 4 ? p.q6_K_batch4
                    : (batch_size == 8 ? p.q6_K_batch8_vec4
                                       : p.q6_K_batch8))));
    }
    Q4Args args = {
        static_cast<uint32_t>(input_size),
        static_cast<uint32_t>(output_size),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(blocks_per_row),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
        [encoder setBuffer:input_buffer offset:input_offset atIndex:1];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        const NSUInteger output_groups = (output_size + 3) / 4;
        const NSUInteger batch_groups = (batch_size + batch_tile - 1) / batch_tile;
        [encoder dispatchThreadgroups:MTLSizeMake(output_groups, batch_groups, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    });
    return output;
}

torch::Tensor q4_0_embedding(
    const torch::Tensor & packed_weight,
    const torch::Tensor & token_ids,
    int64_t vocab_size,
    int64_t hidden_size) {
    TORCH_CHECK(packed_weight.device().is_mps(), "packed_weight must be on MPS");
    TORCH_CHECK(token_ids.device().is_mps(), "token_ids must be on MPS");
    TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8,
                "packed_weight must have dtype uint8");
    TORCH_CHECK(token_ids.scalar_type() == torch::kInt64,
                "token_ids must have dtype int64");
    TORCH_CHECK(packed_weight.is_contiguous(), "packed_weight must be contiguous");
    TORCH_CHECK(token_ids.is_contiguous(), "token_ids must be contiguous");
    TORCH_CHECK(hidden_size > 0 && hidden_size % 32 == 0,
                "hidden_size must be a positive multiple of 32");
    TORCH_CHECK(vocab_size > 0, "vocab_size must be positive");

    const int64_t expected_weight_bytes = vocab_size * (hidden_size / 32) * 18;
    TORCH_CHECK(packed_weight.numel() >= expected_weight_bytes,
                "packed_weight is smaller than the Q4_0 embedding shape requires");
    auto output = torch::empty(
        {token_ids.numel(), hidden_size},
        packed_weight.options().dtype(torch::kFloat32));

    id<MTLBuffer> weight_buffer =
        (__bridge id<MTLBuffer>)packed_weight.storage().data_ptr().get();
    id<MTLBuffer> ids_buffer =
        (__bridge id<MTLBuffer>)token_ids.storage().data_ptr().get();
    id<MTLBuffer> output_buffer =
        (__bridge id<MTLBuffer>)output.storage().data_ptr().get();
    const NSUInteger weight_offset =
        packed_weight.storage_offset() * packed_weight.element_size();
    const NSUInteger ids_offset = token_ids.storage_offset() * token_ids.element_size();
    const NSUInteger output_offset = output.storage_offset() * output.element_size();
    Q4Args args = {
        static_cast<uint32_t>(hidden_size),
        static_cast<uint32_t>(vocab_size),
        static_cast<uint32_t>(token_ids.numel()),
        static_cast<uint32_t>(hidden_size / 32),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = pipelines().embedding;
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:0];
        [encoder setBuffer:ids_buffer offset:ids_offset atIndex:1];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        const NSUInteger total = token_ids.numel() * hidden_size;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output.view([&] {
        auto shape = token_ids.sizes().vec();
        shape.push_back(hidden_size);
        return shape;
    }());
}

torch::Tensor causal_conv1d_decode(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const torch::Tensor & state,
    const torch::Tensor & cache_indices) {
    for (const auto & tensor : {input, weight, state}) {
        TORCH_CHECK(tensor.device().is_mps(), "causal-conv tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal causal conv requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "causal-conv tensors must be contiguous");
    }
    TORCH_CHECK(cache_indices.device().is_mps(), "cache_indices must be on MPS");
    TORCH_CHECK(cache_indices.scalar_type() == torch::kInt32,
                "cache_indices must have dtype int32");
    TORCH_CHECK(input.dim() == 2, "decode input must have shape [batch, channels]");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 4,
                "causal-conv weight must have shape [channels, 4]");
    TORCH_CHECK(state.dim() == 3 && state.size(2) == 3,
                "causal-conv state must have shape [slots, channels, 3]");
    const int64_t batch_size = input.size(0);
    const int64_t channels = input.size(1);
    TORCH_CHECK(weight.size(0) == channels && state.size(1) == channels,
                "causal-conv channel dimensions must match");
    TORCH_CHECK(cache_indices.numel() >= batch_size,
                "cache_indices must cover the batch");

    auto output = torch::empty_like(input);
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    id<MTLBuffer> input_buffer = buffer_of(input);
    id<MTLBuffer> weight_buffer = buffer_of(weight);
    id<MTLBuffer> state_buffer = buffer_of(state);
    id<MTLBuffer> indices_buffer = buffer_of(cache_indices);
    id<MTLBuffer> output_buffer = buffer_of(output);
    const NSUInteger input_offset = offset_of(input);
    const NSUInteger weight_offset = offset_of(weight);
    const NSUInteger state_offset = offset_of(state);
    const NSUInteger indices_offset = offset_of(cache_indices);
    const NSUInteger output_offset = offset_of(output);
    ConvArgs args = {
        1,
        static_cast<uint32_t>(channels),
        static_cast<uint32_t>(batch_size),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = pipelines().conv_decode;
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:input_offset atIndex:0];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:1];
        [encoder setBuffer:state_buffer offset:state_offset atIndex:2];
        [encoder setBuffer:indices_buffer offset:indices_offset atIndex:3];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        const NSUInteger total = batch_size * channels;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output;
}

torch::Tensor causal_conv1d_prefill(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const torch::Tensor & state,
    const torch::Tensor & cache_indices,
    const torch::Tensor & query_start_loc,
    const torch::Tensor & has_initial_state) {
    for (const auto & tensor : {input, weight, state}) {
        TORCH_CHECK(tensor.device().is_mps(), "causal-conv tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal causal conv requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "causal-conv tensors must be contiguous");
    }
    for (const auto & tensor : {cache_indices, query_start_loc, has_initial_state}) {
        TORCH_CHECK(tensor.device().is_mps(), "causal-conv metadata must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
                    "causal-conv metadata must have dtype int32");
        TORCH_CHECK(tensor.is_contiguous(), "causal-conv metadata must be contiguous");
    }
    TORCH_CHECK(input.dim() == 2, "prefill input must have shape [channels, tokens]");
    TORCH_CHECK(weight.dim() == 2 && weight.size(1) == 4,
                "causal-conv weight must have shape [channels, 4]");
    TORCH_CHECK(state.dim() == 3 && state.size(2) == 3,
                "causal-conv state must have shape [slots, channels, 3]");
    const int64_t channels = input.size(0);
    const int64_t num_tokens = input.size(1);
    const int64_t batch_size = query_start_loc.numel() - 1;
    TORCH_CHECK(batch_size >= 1, "causal-conv prefill requires a sequence");
    TORCH_CHECK(weight.size(0) == channels && state.size(1) == channels,
                "causal-conv channel dimensions must match");
    TORCH_CHECK(cache_indices.numel() >= batch_size &&
                    has_initial_state.numel() >= batch_size,
                "causal-conv metadata must cover every sequence");

    auto output = torch::empty_like(input);
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    id<MTLBuffer> input_buffer = buffer_of(input);
    id<MTLBuffer> weight_buffer = buffer_of(weight);
    id<MTLBuffer> state_buffer = buffer_of(state);
    id<MTLBuffer> indices_buffer = buffer_of(cache_indices);
    id<MTLBuffer> starts_buffer = buffer_of(query_start_loc);
    id<MTLBuffer> initial_buffer = buffer_of(has_initial_state);
    id<MTLBuffer> output_buffer = buffer_of(output);
    const NSUInteger input_offset = offset_of(input);
    const NSUInteger weight_offset = offset_of(weight);
    const NSUInteger state_offset = offset_of(state);
    const NSUInteger indices_offset = offset_of(cache_indices);
    const NSUInteger starts_offset = offset_of(query_start_loc);
    const NSUInteger initial_offset = offset_of(has_initial_state);
    const NSUInteger output_offset = offset_of(output);
    ConvArgs args = {
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(channels),
        static_cast<uint32_t>(batch_size),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    id<MTLComputePipelineState> pipeline = pipelines().conv_prefill;
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input_buffer offset:input_offset atIndex:0];
        [encoder setBuffer:weight_buffer offset:weight_offset atIndex:1];
        [encoder setBuffer:state_buffer offset:state_offset atIndex:2];
        [encoder setBuffer:indices_buffer offset:indices_offset atIndex:3];
        [encoder setBuffer:starts_buffer offset:starts_offset atIndex:4];
        [encoder setBuffer:initial_buffer offset:initial_offset atIndex:5];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:6];
        [encoder setBytes:&args length:sizeof(args) atIndex:7];
        const NSUInteger total = batch_size * channels;
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output;
}

torch::Tensor gdn_decode(
    const torch::Tensor & query,
    const torch::Tensor & key,
    const torch::Tensor & value,
    const torch::Tensor & a,
    const torch::Tensor & b,
    const torch::Tensor & A_log,
    const torch::Tensor & dt_bias,
    const torch::Tensor & state,
    const torch::Tensor & cache_indices) {
    for (const auto & tensor : {query, key, value, a, b, A_log, dt_bias, state}) {
        TORCH_CHECK(tensor.device().is_mps(), "all GDN tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal GDN decode requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "native Metal GDN tensors must be contiguous");
    }
    TORCH_CHECK(cache_indices.device().is_mps(), "cache_indices must be on MPS");
    TORCH_CHECK(cache_indices.scalar_type() == torch::kInt32,
                "cache_indices must have dtype int32");
    TORCH_CHECK(cache_indices.is_contiguous(), "cache_indices must be contiguous");
    TORCH_CHECK(query.dim() == 4 && query.size(0) == 1,
                "query must have shape [1, batch, key_heads, key_dim]");
    TORCH_CHECK(key.sizes() == query.sizes(), "key must match query shape");
    TORCH_CHECK(value.dim() == 4 && value.size(0) == 1,
                "value must have shape [1, batch, value_heads, value_dim]");
    TORCH_CHECK(state.dim() == 4,
                "state must have shape [slots, value_heads, value_dim, key_dim]");

    const int64_t batch_size = query.size(1);
    const int64_t num_k_heads = query.size(2);
    const int64_t key_dim = query.size(3);
    const int64_t num_v_heads = value.size(2);
    const int64_t value_dim = value.size(3);
    TORCH_CHECK(key_dim == 128,
                "native Metal GDN decode currently requires key_dim=128");
    TORCH_CHECK(value.size(1) == batch_size,
                "query and value batch sizes must match");
    TORCH_CHECK(num_v_heads % num_k_heads == 0,
                "value heads must be divisible by key heads");
    TORCH_CHECK(state.size(1) == num_v_heads && state.size(2) == value_dim &&
                    state.size(3) == key_dim,
                "state shape does not match the GDN head geometry");
    TORCH_CHECK(a.numel() == batch_size * num_v_heads &&
                    b.numel() == batch_size * num_v_heads,
                "a and b must have one value per batch and value head");
    TORCH_CHECK(A_log.numel() == num_v_heads && dt_bias.numel() == num_v_heads,
                "A_log and dt_bias must have one value per value head");
    TORCH_CHECK(cache_indices.numel() >= batch_size,
                "cache_indices must cover the batch");

    auto output = torch::empty(
        {1, batch_size, num_v_heads, value_dim}, query.options());
    auto normalized_query = torch::empty_like(query);
    auto normalized_key = torch::empty_like(key);

    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    id<MTLBuffer> query_buffer = buffer_of(query);
    id<MTLBuffer> key_buffer = buffer_of(key);
    id<MTLBuffer> normalized_query_buffer = buffer_of(normalized_query);
    id<MTLBuffer> normalized_key_buffer = buffer_of(normalized_key);
    id<MTLBuffer> value_buffer = buffer_of(value);
    id<MTLBuffer> a_buffer = buffer_of(a);
    id<MTLBuffer> b_buffer = buffer_of(b);
    id<MTLBuffer> A_log_buffer = buffer_of(A_log);
    id<MTLBuffer> dt_bias_buffer = buffer_of(dt_bias);
    id<MTLBuffer> state_buffer = buffer_of(state);
    id<MTLBuffer> indices_buffer = buffer_of(cache_indices);
    id<MTLBuffer> output_buffer = buffer_of(output);
    const NSUInteger query_offset = offset_of(query);
    const NSUInteger key_offset = offset_of(key);
    const NSUInteger normalized_query_offset = offset_of(normalized_query);
    const NSUInteger normalized_key_offset = offset_of(normalized_key);
    const NSUInteger value_offset = offset_of(value);
    const NSUInteger a_offset = offset_of(a);
    const NSUInteger b_offset = offset_of(b);
    const NSUInteger A_log_offset = offset_of(A_log);
    const NSUInteger dt_bias_offset = offset_of(dt_bias);
    const NSUInteger state_offset = offset_of(state);
    const NSUInteger indices_offset = offset_of(cache_indices);
    const NSUInteger output_offset = offset_of(output);
    GDNArgs args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_k_heads),
        static_cast<uint32_t>(num_v_heads),
        static_cast<uint32_t>(key_dim),
        static_cast<uint32_t>(value_dim),
    };
    GDNNormArgs norm_args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_k_heads),
        static_cast<uint32_t>(key_dim),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    id<MTLComputePipelineState> gdn_pipeline = pipelines().gdn_decode;
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().normalize_gdn_qk];
        [encoder setBuffer:query_buffer offset:query_offset atIndex:0];
        [encoder setBuffer:key_buffer offset:key_offset atIndex:1];
        [encoder setBuffer:normalized_query_buffer
                 offset:normalized_query_offset atIndex:2];
        [encoder setBuffer:normalized_key_buffer
                 offset:normalized_key_offset atIndex:3];
        [encoder setBytes:&norm_args length:sizeof(norm_args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size * num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:gdn_pipeline];
        [encoder setBuffer:normalized_query_buffer
                 offset:normalized_query_offset atIndex:0];
        [encoder setBuffer:normalized_key_buffer
                 offset:normalized_key_offset atIndex:1];
        [encoder setBuffer:value_buffer offset:value_offset atIndex:2];
        [encoder setBuffer:a_buffer offset:a_offset atIndex:3];
        [encoder setBuffer:b_buffer offset:b_offset atIndex:4];
        [encoder setBuffer:A_log_buffer offset:A_log_offset atIndex:5];
        [encoder setBuffer:dt_bias_buffer offset:dt_bias_offset atIndex:6];
        [encoder setBuffer:state_buffer offset:state_offset atIndex:7];
        [encoder setBuffer:indices_buffer offset:indices_offset atIndex:8];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(
                1, (value_dim + 3) / 4, batch_size * num_v_heads)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    });
    return output;
}

torch::Tensor gdn_prefill(
    const torch::Tensor & query,
    const torch::Tensor & key,
    const torch::Tensor & value,
    const torch::Tensor & g,
    const torch::Tensor & beta,
    const torch::Tensor & state,
    const torch::Tensor & cache_indices,
    const torch::Tensor & query_start_loc) {
    for (const auto & tensor : {query, key, value, g, beta, state}) {
        TORCH_CHECK(tensor.device().is_mps(), "all GDN prefill tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal GDN prefill requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(),
                    "native Metal GDN prefill tensors must be contiguous");
    }
    for (const auto & tensor : {cache_indices, query_start_loc}) {
        TORCH_CHECK(tensor.device().is_mps(), "GDN prefill indices must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
                    "GDN prefill indices must have dtype int32");
        TORCH_CHECK(tensor.is_contiguous(), "GDN prefill indices must be contiguous");
    }
    TORCH_CHECK(query.dim() == 3,
                "query must have shape [tokens, key_heads, key_dim]");
    TORCH_CHECK(key.sizes() == query.sizes(), "key must match query shape");
    TORCH_CHECK(value.dim() == 3,
                "value must have shape [tokens, value_heads, value_dim]");
    TORCH_CHECK(state.dim() == 4,
                "state must have shape [slots, value_heads, value_dim, key_dim]");

    const int64_t num_tokens = query.size(0);
    const int64_t num_k_heads = query.size(1);
    const int64_t key_dim = query.size(2);
    const int64_t num_v_heads = value.size(1);
    const int64_t value_dim = value.size(2);
    const int64_t batch_size = query_start_loc.numel() - 1;
    TORCH_CHECK(batch_size >= 1, "GDN prefill requires at least one sequence");
    TORCH_CHECK(key_dim == 128,
                "native Metal GDN prefill currently requires key_dim=128");
    TORCH_CHECK(value.size(0) == num_tokens,
                "query and value token counts must match");
    TORCH_CHECK(num_v_heads % num_k_heads == 0,
                "value heads must be divisible by key heads");
    TORCH_CHECK(state.size(1) == num_v_heads && state.size(2) == value_dim &&
                    state.size(3) == key_dim,
                "state shape does not match the GDN head geometry");
    TORCH_CHECK(g.sizes() == beta.sizes() && g.dim() == 2 &&
                    g.size(0) == num_tokens && g.size(1) == num_v_heads,
                "g and beta must have shape [tokens, value_heads]");
    TORCH_CHECK(cache_indices.numel() >= batch_size,
                "cache_indices must cover every sequence");

    auto output = torch::empty(
        {num_tokens, num_v_heads, value_dim}, query.options());
    auto normalized_query = torch::empty_like(query);
    auto normalized_key = torch::empty_like(key);

    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    id<MTLBuffer> query_buffer = buffer_of(query);
    id<MTLBuffer> key_buffer = buffer_of(key);
    id<MTLBuffer> normalized_query_buffer = buffer_of(normalized_query);
    id<MTLBuffer> normalized_key_buffer = buffer_of(normalized_key);
    id<MTLBuffer> value_buffer = buffer_of(value);
    id<MTLBuffer> g_buffer = buffer_of(g);
    id<MTLBuffer> beta_buffer = buffer_of(beta);
    id<MTLBuffer> state_buffer = buffer_of(state);
    id<MTLBuffer> indices_buffer = buffer_of(cache_indices);
    id<MTLBuffer> starts_buffer = buffer_of(query_start_loc);
    id<MTLBuffer> output_buffer = buffer_of(output);
    const NSUInteger query_offset = offset_of(query);
    const NSUInteger key_offset = offset_of(key);
    const NSUInteger normalized_query_offset = offset_of(normalized_query);
    const NSUInteger normalized_key_offset = offset_of(normalized_key);
    const NSUInteger value_offset = offset_of(value);
    const NSUInteger g_offset = offset_of(g);
    const NSUInteger beta_offset = offset_of(beta);
    const NSUInteger state_offset = offset_of(state);
    const NSUInteger indices_offset = offset_of(cache_indices);
    const NSUInteger starts_offset = offset_of(query_start_loc);
    const NSUInteger output_offset = offset_of(output);
    GDNArgs args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_k_heads),
        static_cast<uint32_t>(num_v_heads),
        static_cast<uint32_t>(key_dim),
        static_cast<uint32_t>(value_dim),
    };
    GDNNormArgs norm_args = {
        static_cast<uint32_t>(num_tokens),
        static_cast<uint32_t>(num_k_heads),
        static_cast<uint32_t>(key_dim),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    id<MTLComputePipelineState> gdn_pipeline = pipelines().gdn_prefill;
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().normalize_gdn_qk];
        [encoder setBuffer:query_buffer offset:query_offset atIndex:0];
        [encoder setBuffer:key_buffer offset:key_offset atIndex:1];
        [encoder setBuffer:normalized_query_buffer
                 offset:normalized_query_offset atIndex:2];
        [encoder setBuffer:normalized_key_buffer
                 offset:normalized_key_offset atIndex:3];
        [encoder setBytes:&norm_args length:sizeof(norm_args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(num_tokens * num_k_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:gdn_pipeline];
        [encoder setBuffer:normalized_query_buffer
                 offset:normalized_query_offset atIndex:0];
        [encoder setBuffer:normalized_key_buffer
                 offset:normalized_key_offset atIndex:1];
        [encoder setBuffer:value_buffer offset:value_offset atIndex:2];
        [encoder setBuffer:g_buffer offset:g_offset atIndex:3];
        [encoder setBuffer:beta_buffer offset:beta_offset atIndex:4];
        [encoder setBuffer:state_buffer offset:state_offset atIndex:5];
        [encoder setBuffer:indices_buffer offset:indices_offset atIndex:6];
        [encoder setBuffer:starts_buffer offset:starts_offset atIndex:7];
        [encoder setBuffer:output_buffer offset:output_offset atIndex:8];
        [encoder setBytes:&args length:sizeof(args) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(
                1, (value_dim + 3) / 4, batch_size * num_v_heads)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
    });
    return output;
}

torch::Tensor gemma_rmsnorm(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    double epsilon) {
    for (const auto & tensor : {input, weight}) {
        TORCH_CHECK(tensor.device().is_mps(), "RMSNorm tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal RMSNorm requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "RMSNorm tensors must be contiguous");
    }
    TORCH_CHECK(input.dim() >= 1 && weight.dim() == 1,
                "RMSNorm input must have a final feature dimension");
    const int64_t columns = input.size(-1);
    TORCH_CHECK(weight.numel() == columns && input.numel() % columns == 0,
                "RMSNorm weight shape must match the input feature dimension");
    const int64_t rows = input.numel() / columns;
    auto output = torch::empty_like(input);
    NormArgs args = {
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(columns),
        static_cast<float>(epsilon),
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().gemma_rmsnorm];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:0];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:1];
        [encoder setBuffer:buffer_of(weight) offset:offset_of(weight) atIndex:2];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:3];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> gemma_fused_add_rmsnorm(
    const torch::Tensor & input,
    const torch::Tensor & residual,
    const torch::Tensor & weight,
    double epsilon) {
    for (const auto & tensor : {input, residual, weight}) {
        TORCH_CHECK(tensor.device().is_mps(), "RMSNorm tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal RMSNorm requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(), "RMSNorm tensors must be contiguous");
    }
    TORCH_CHECK(input.sizes() == residual.sizes(),
                "RMSNorm residual must match the input shape");
    const int64_t columns = input.size(-1);
    TORCH_CHECK(weight.numel() == columns && input.numel() % columns == 0,
                "RMSNorm weight shape must match the input feature dimension");
    const int64_t rows = input.numel() / columns;
    auto output = torch::empty_like(input);
    auto residual_output = torch::empty_like(input);
    NormArgs args = {
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(columns),
        static_cast<float>(epsilon),
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().gemma_fused_add_rmsnorm];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:0];
        [encoder setBuffer:buffer_of(residual) offset:offset_of(residual) atIndex:1];
        [encoder setBuffer:buffer_of(weight) offset:offset_of(weight) atIndex:2];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:3];
        [encoder setBuffer:buffer_of(residual_output)
                 offset:offset_of(residual_output) atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return {output, residual_output};
}

torch::Tensor silu_and_mul(const torch::Tensor & input) {
    TORCH_CHECK(input.device().is_mps() && input.scalar_type() == torch::kFloat32 &&
                    input.is_contiguous(),
                "native Metal SiLU-and-mul requires contiguous float32 MPS input");
    TORCH_CHECK(input.size(-1) % 2 == 0,
                "SiLU-and-mul input feature dimension must be even");
    const int64_t columns = input.size(-1) / 2;
    const int64_t rows = input.numel() / (2 * columns);
    auto shape = input.sizes().vec();
    shape.back() = columns;
    auto output = torch::empty(shape, input.options());
    NormArgs args = {
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(columns),
        0.0f,
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().silu_and_mul];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:0];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(rows * columns, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
pack_gdn_inputs(
    const torch::Tensor & qkvz,
    const torch::Tensor & ba,
    int64_t key_dim,
    int64_t value_dim,
    int64_t num_v_heads) {
    for (const auto & tensor : {qkvz, ba}) {
        TORCH_CHECK(tensor.device().is_mps() &&
                        tensor.scalar_type() == torch::kFloat32 &&
                        tensor.is_contiguous(),
                    "native Metal GDN pack requires contiguous float32 MPS tensors");
    }
    TORCH_CHECK(qkvz.dim() == 2 && ba.dim() == 2 && qkvz.size(0) == ba.size(0),
                "GDN pack inputs must be two-dimensional with matching batches");
    const int64_t batch_size = qkvz.size(0);
    const int64_t mixed_dim = 2 * key_dim + value_dim;
    TORCH_CHECK(qkvz.size(1) == mixed_dim + value_dim &&
                    ba.size(1) == 2 * num_v_heads,
                "GDN pack dimensions do not match the requested geometry");
    auto mixed = torch::empty({batch_size, mixed_dim}, qkvz.options());
    auto gate = torch::empty({batch_size, value_dim}, qkvz.options());
    auto b = torch::empty({batch_size, num_v_heads}, qkvz.options());
    auto a = torch::empty({batch_size, num_v_heads}, qkvz.options());
    PackArgs args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(key_dim),
        static_cast<uint32_t>(value_dim),
        static_cast<uint32_t>(num_v_heads),
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().pack_gdn_inputs];
        [encoder setBuffer:buffer_of(qkvz) offset:offset_of(qkvz) atIndex:0];
        [encoder setBuffer:buffer_of(ba) offset:offset_of(ba) atIndex:1];
        [encoder setBuffer:buffer_of(mixed) offset:offset_of(mixed) atIndex:2];
        [encoder setBuffer:buffer_of(gate) offset:offset_of(gate) atIndex:3];
        [encoder setBuffer:buffer_of(b) offset:offset_of(b) atIndex:4];
        [encoder setBuffer:buffer_of(a) offset:offset_of(a) atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        const NSUInteger total = batch_size *
            (mixed_dim + value_dim + 2 * num_v_heads);
        [encoder dispatchThreads:MTLSizeMake(total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return {mixed, gate, b, a};
}

torch::Tensor gdn_gated_rmsnorm_reorder(
    const torch::Tensor & input,
    const torch::Tensor & gate,
    const torch::Tensor & weight,
    int64_t num_k_heads,
    int64_t num_v_heads,
    int64_t head_dim,
    double epsilon) {
    for (const auto & tensor : {input, gate, weight}) {
        TORCH_CHECK(tensor.device().is_mps() &&
                        tensor.scalar_type() == torch::kFloat32 &&
                        tensor.is_contiguous(),
                    "native Metal gated RMSNorm requires contiguous float32 MPS tensors");
    }
    TORCH_CHECK(input.sizes() == gate.sizes() &&
                    input.numel() % (num_v_heads * head_dim) == 0 &&
                    weight.numel() == head_dim && num_v_heads % num_k_heads == 0,
                "gated RMSNorm geometry is inconsistent");
    const int64_t batch_size = input.numel() / (num_v_heads * head_dim);
    auto output = torch::empty_like(input);
    GatedNormArgs args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_k_heads),
        static_cast<uint32_t>(num_v_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<float>(epsilon),
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().gdn_gated_norm_reorder];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:0];
        [encoder setBuffer:buffer_of(gate) offset:offset_of(gate) atIndex:1];
        [encoder setBuffer:buffer_of(weight) offset:offset_of(weight) atIndex:2];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size * num_v_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    });
    return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
prepare_full_attention(
    const torch::Tensor & qkv,
    const torch::Tensor & q_weight,
    const torch::Tensor & k_weight,
    const torch::Tensor & cos_sin_cache,
    const torch::Tensor & positions,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t rotary_dim,
    double epsilon) {
    for (const auto & tensor : {qkv, q_weight, k_weight, cos_sin_cache}) {
        TORCH_CHECK(tensor.device().is_mps() &&
                        tensor.scalar_type() == torch::kFloat32 &&
                        tensor.is_contiguous(),
                    "native Metal attention prepare requires contiguous float32 MPS tensors");
    }
    TORCH_CHECK(positions.device().is_mps() &&
                    positions.scalar_type() == torch::kInt64 &&
                    positions.is_contiguous(),
                "attention positions must be contiguous int64 MPS data");
    TORCH_CHECK(qkv.dim() == 2 && q_weight.numel() == head_dim &&
                    k_weight.numel() == head_dim && rotary_dim > 0 &&
                    rotary_dim <= head_dim && rotary_dim % 2 == 0,
                "attention prepare geometry is inconsistent");
    const int64_t tokens = qkv.size(0);
    const int64_t expected_width =
        2 * num_q_heads * head_dim + 2 * num_kv_heads * head_dim;
    TORCH_CHECK(qkv.size(1) == expected_width && positions.numel() >= tokens &&
                    cos_sin_cache.dim() == 2 &&
                    cos_sin_cache.size(1) == rotary_dim,
                "attention prepare inputs do not match the requested geometry");
    auto query = torch::empty({tokens, num_q_heads, head_dim}, qkv.options());
    auto key = torch::empty({tokens, num_kv_heads, head_dim}, qkv.options());
    auto value = torch::empty_like(key);
    auto gate = torch::empty_like(query);
    FullAttentionPrepareArgs args = {
        static_cast<uint32_t>(tokens),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(rotary_dim),
        static_cast<float>(epsilon),
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().prepare_full_attention];
        [encoder setBuffer:buffer_of(qkv) offset:offset_of(qkv) atIndex:0];
        [encoder setBuffer:buffer_of(q_weight) offset:offset_of(q_weight) atIndex:1];
        [encoder setBuffer:buffer_of(k_weight) offset:offset_of(k_weight) atIndex:2];
        [encoder setBuffer:buffer_of(cos_sin_cache)
                 offset:offset_of(cos_sin_cache) atIndex:3];
        [encoder setBuffer:buffer_of(positions)
                 offset:offset_of(positions) atIndex:4];
        [encoder setBuffer:buffer_of(query) offset:offset_of(query) atIndex:5];
        [encoder setBuffer:buffer_of(key) offset:offset_of(key) atIndex:6];
        [encoder setBuffer:buffer_of(value) offset:offset_of(value) atIndex:7];
        [encoder setBuffer:buffer_of(gate) offset:offset_of(gate) atIndex:8];
        [encoder setBytes:&args length:sizeof(args) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(
                tokens * (num_q_heads + num_kv_heads), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    });
    return {query, key, value, gate};
}

torch::Tensor sigmoid_mul_inplace(
    const torch::Tensor & input,
    const torch::Tensor & gate) {
    for (const auto & tensor : {input, gate}) {
        TORCH_CHECK(tensor.device().is_mps() &&
                        tensor.scalar_type() == torch::kFloat32 &&
                        tensor.is_contiguous(),
                    "native Metal sigmoid-mul requires contiguous float32 MPS tensors");
    }
    TORCH_CHECK(input.sizes() == gate.sizes(),
                "sigmoid gate must match the input shape");
    NormArgs args = {
        static_cast<uint32_t>(input.numel()),
        1,
        0.0f,
    };
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };
    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipelines().sigmoid_mul_inplace];
        [encoder setBuffer:buffer_of(input) offset:offset_of(input) atIndex:0];
        [encoder setBuffer:buffer_of(gate) offset:offset_of(gate) atIndex:1];
        [encoder setBytes:&args length:sizeof(args) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(input.numel(), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return input;
}

torch::Tensor decode_gqa(
    const torch::Tensor & query,
    const torch::Tensor & key,
    const torch::Tensor & value,
    const torch::Tensor & key_cache,
    const torch::Tensor & value_cache,
    const torch::Tensor & cache_locations,
    const torch::Tensor & req_to_token,
    const torch::Tensor & req_pool_indices,
    const torch::Tensor & seq_lens,
    double scale) {
    for (const auto & tensor : {query, key, value, key_cache, value_cache}) {
        TORCH_CHECK(tensor.device().is_mps(), "decode attention tensors must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
                    "native Metal decode attention requires float32 tensors");
        TORCH_CHECK(tensor.is_contiguous(),
                    "native Metal decode attention tensors must be contiguous");
    }
    for (const auto & tensor : {cache_locations, req_pool_indices, seq_lens}) {
        TORCH_CHECK(tensor.device().is_mps(), "decode attention metadata must be on MPS");
        TORCH_CHECK(tensor.scalar_type() == torch::kInt64,
                    "decode attention metadata must have dtype int64");
        TORCH_CHECK(tensor.is_contiguous(),
                    "decode attention metadata must be contiguous");
    }
    TORCH_CHECK(req_to_token.device().is_mps(), "req_to_token must be on MPS");
    TORCH_CHECK(req_to_token.scalar_type() == torch::kInt32,
                "req_to_token must have dtype int32");
    TORCH_CHECK(req_to_token.is_contiguous(), "req_to_token must be contiguous");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
                "query/key/value must have shape [batch, heads, head_dim]");
    TORCH_CHECK(key_cache.dim() == 3 && value_cache.dim() == 3,
                "KV caches must have shape [slots, heads, head_dim]");
    TORCH_CHECK(key.sizes() == value.sizes(), "key and value shapes must match");
    TORCH_CHECK(key_cache.sizes() == value_cache.sizes(),
                "key and value cache shapes must match");

    const int64_t batch_size = query.size(0);
    const int64_t num_q_heads = query.size(1);
    const int64_t num_kv_heads = key.size(1);
    const int64_t head_dim = query.size(2);
    const int64_t cache_slots = key_cache.size(0);
    TORCH_CHECK(batch_size >= 1 && key.size(0) == batch_size,
                "query/key batch sizes must match");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "query heads must be divisible by KV heads");
    TORCH_CHECK(key.size(2) == head_dim && key_cache.size(1) == num_kv_heads &&
                    key_cache.size(2) == head_dim,
                "decode attention head geometry does not match the KV cache");
    TORCH_CHECK(head_dim > 0 && head_dim <= 256,
                "native Metal decode attention requires head_dim <= 256");
    TORCH_CHECK(cache_slots > 0 && cache_slots <= 7936,
                "native Metal decode attention supports at most 7936 cache slots");
    TORCH_CHECK(req_to_token.dim() == 2,
                "req_to_token must have shape [request_slots, context_length]");
    TORCH_CHECK(cache_locations.numel() >= batch_size &&
                    req_pool_indices.numel() >= batch_size &&
                    seq_lens.numel() >= batch_size,
                "decode attention metadata must cover every batch row");

    auto output = torch::empty_like(query);
    auto buffer_of = [](const torch::Tensor & tensor) {
        return (__bridge id<MTLBuffer>)tensor.storage().data_ptr().get();
    };
    auto offset_of = [](const torch::Tensor & tensor) -> NSUInteger {
        return tensor.storage_offset() * tensor.element_size();
    };

    AttentionArgs args = {
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(cache_slots),
        static_cast<uint32_t>(req_to_token.size(1)),
        static_cast<float>(scale),
    };

    at::mps::MPSStream * stream = at::mps::getCurrentMPSStream();
    Pipelines & p = pipelines();
    dispatch_sync(stream->queue(), ^{
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
        [encoder setComputePipelineState:p.store_decode_kv];
        [encoder setBuffer:buffer_of(key) offset:offset_of(key) atIndex:0];
        [encoder setBuffer:buffer_of(value) offset:offset_of(value) atIndex:1];
        [encoder setBuffer:buffer_of(key_cache) offset:offset_of(key_cache) atIndex:2];
        [encoder setBuffer:buffer_of(value_cache) offset:offset_of(value_cache) atIndex:3];
        [encoder setBuffer:buffer_of(cache_locations)
                 offset:offset_of(cache_locations) atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        const NSUInteger kv_total = batch_size * num_kv_heads * head_dim;
        [encoder dispatchThreads:MTLSizeMake(kv_total, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [encoder setComputePipelineState:p.decode_gqa];
        [encoder setBuffer:buffer_of(query) offset:offset_of(query) atIndex:0];
        [encoder setBuffer:buffer_of(key_cache) offset:offset_of(key_cache) atIndex:1];
        [encoder setBuffer:buffer_of(value_cache) offset:offset_of(value_cache) atIndex:2];
        [encoder setBuffer:buffer_of(req_to_token) offset:offset_of(req_to_token) atIndex:3];
        [encoder setBuffer:buffer_of(req_pool_indices)
                 offset:offset_of(req_pool_indices) atIndex:4];
        [encoder setBuffer:buffer_of(seq_lens) offset:offset_of(seq_lens) atIndex:5];
        [encoder setBuffer:buffer_of(output) offset:offset_of(output) atIndex:6];
        [encoder setBytes:&args length:sizeof(args) atIndex:7];
        [encoder setThreadgroupMemoryLength:(cache_slots + 256) * sizeof(float)
                                    atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(batch_size * num_q_heads, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    });
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("q4_0_matmul", &q4_0_matmul, "Native Metal GGUF Q4_0 matmul");
    module.def("quant_matmul", &quant_matmul,
               "Native Metal GGUF Q4_1/Q5_K/Q6_K matmul");
    module.def("dense_matmul", &dense_matmul,
               "Native Metal small-output float32 matmul");
    module.def("q4_0_embedding", &q4_0_embedding,
               "Native Metal GGUF Q4_0 embedding");
    module.def("causal_conv1d_decode", &causal_conv1d_decode,
               "Native Metal causal-conv decode");
    module.def("causal_conv1d_prefill", &causal_conv1d_prefill,
               "Native Metal causal-conv prefill");
    module.def("gdn_decode", &gdn_decode, "Native Metal Gated DeltaNet decode");
    module.def("gdn_prefill", &gdn_prefill, "Native Metal Gated DeltaNet prefill");
    module.def("decode_gqa", &decode_gqa,
               "Native Metal fused KV write and grouped-query decode attention");
    module.def("gemma_rmsnorm", &gemma_rmsnorm,
               "Native Metal Gemma RMSNorm");
    module.def("gemma_fused_add_rmsnorm", &gemma_fused_add_rmsnorm,
               "Native Metal fused residual add and Gemma RMSNorm");
    module.def("silu_and_mul", &silu_and_mul,
               "Native Metal SiLU-and-multiply activation");
    module.def("pack_gdn_inputs", &pack_gdn_inputs,
               "Native Metal GDN projection packing");
    module.def("gdn_gated_rmsnorm_reorder", &gdn_gated_rmsnorm_reorder,
               "Native Metal GDN gated RMSNorm and tiled-head reorder");
    module.def("prepare_full_attention", &prepare_full_attention,
               "Native Metal Q/K norm, partial RoPE, and QKV/gate unpack");
    module.def("sigmoid_mul_inplace", &sigmoid_mul_inplace,
               "Native Metal in-place sigmoid gate");
}
