#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach_time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t batch;
    uint32_t groups_per_row;
} Q4Shape;

static const char *kShader = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Q4Shape {
    uint rows;
    uint cols;
    uint batch;
    uint groups_per_row;
};

kernel void q4_g64_gemv(
    device const uchar *weights [[buffer(0)]],
    device const half2 *scale_bias [[buffer(1)]],
    device const half *x [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant Q4Shape &shape [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]) {
    float4 sums = 0.0f;
    const uint packed_row_stride = shape.cols / 2;
    const uint group = tid;

    if (group < shape.groups_per_row) {
        const uint weight_base = row * packed_row_stride + group * 32;
        const uint x_base = group * 64;
        const half2 sb = scale_bias[row * shape.groups_per_row + group];
        const float scale = float(sb.x);
        const float bias = float(sb.y);

        for (uint byte_idx = 0; byte_idx < 32; ++byte_idx) {
            const uchar packed = weights[weight_base + byte_idx];
            const float w0 = float(packed & 0x0f) * scale + bias;
            const float w1 = float(packed >> 4) * scale + bias;
            const uint col = x_base + byte_idx * 2;
            if (shape.batch > 0) {
                sums.x += w0 * float(x[col]) + w1 * float(x[col + 1]);
            }
            if (shape.batch > 1) {
                const uint off = shape.cols;
                sums.y += w0 * float(x[off + col]) + w1 * float(x[off + col + 1]);
            }
            if (shape.batch > 2) {
                const uint off = shape.cols * 2;
                sums.z += w0 * float(x[off + col]) + w1 * float(x[off + col + 1]);
            }
            if (shape.batch > 3) {
                const uint off = shape.cols * 3;
                sums.w += w0 * float(x[off + col]) + w1 * float(x[off + col + 1]);
            }
        }
    }

    sums = simd_sum(sums);
    threadgroup float4 partial[8];
    if (lane == 0) {
        partial[simd_id] = sums;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_id == 0) {
        float4 total = lane < 8 ? partial[lane] : float4(0.0f);
        total = simd_sum(total);
        if (lane == 0) {
            if (shape.batch > 0) y[row] = total.x;
            if (shape.batch > 1) y[shape.rows + row] = total.y;
            if (shape.batch > 2) y[shape.rows * 2 + row] = total.z;
            if (shape.batch > 3) y[shape.rows * 3 + row] = total.w;
        }
    }
}
)METAL";

static double seconds_now(void) {
    static mach_timebase_info_data_t timebase;
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)mach_absolute_time() * (double)timebase.numer /
           (double)timebase.denom / 1e9;
}

static id<MTLDevice> select_device(void) {
    NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
    id<MTLDevice> selected = nil;
    for (id<MTLDevice> device in devices) {
        fprintf(stderr,
                "Metal device: %s, low_power=%d, removable=%d, unified=%d, "
                "recommended_working_set=%.1f GiB\n",
                device.name.UTF8String,
                device.lowPower,
                device.removable,
                device.hasUnifiedMemory,
                (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0));
        if (selected == nil || (!device.lowPower && selected.lowPower)) selected = device;
    }
    return selected;
}

static int run_shape(id<MTLDevice> device,
                     id<MTLComputePipelineState> pipeline,
                     uint32_t rows,
                     uint32_t cols,
                     uint32_t batch,
                     uint32_t iterations) {
    if (cols % 64 != 0 || batch == 0 || batch > 4) return 2;

    const uint32_t groups = cols / 64;
    const size_t weight_bytes = (size_t)rows * cols / 2;
    const size_t scale_bytes = (size_t)rows * groups * sizeof(uint16_t) * 2;
    const size_t input_bytes = (size_t)batch * cols * sizeof(uint16_t);
    const size_t output_bytes = (size_t)batch * rows * sizeof(float);

    id<MTLBuffer> weights = [device newBufferWithLength:weight_bytes
                                                options:MTLResourceStorageModePrivate];
    id<MTLBuffer> scale_bias = [device newBufferWithLength:scale_bytes
                                                   options:MTLResourceStorageModePrivate];
    id<MTLBuffer> x = [device newBufferWithLength:input_bytes
                                          options:MTLResourceStorageModeShared];
    id<MTLBuffer> y = [device newBufferWithLength:output_bytes
                                          options:MTLResourceStorageModeShared];
    if (!weights || !scale_bias || !x || !y) {
        fprintf(stderr, "Metal allocation failed for %ux%u batch %u\n", rows, cols, batch);
        return 3;
    }

    uint16_t *x_data = (uint16_t *)x.contents;
    for (size_t i = 0; i < (size_t)batch * cols; ++i) x_data[i] = 0x3c00;

    Q4Shape shape = {rows, cols, batch, groups};
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return 4;

    const uint32_t warmup = 8;
    const double start = seconds_now();
    for (uint32_t iteration = 0; iteration < warmup + iterations; ++iteration) {
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:weights offset:0 atIndex:0];
            [encoder setBuffer:scale_bias offset:0 atIndex:1];
            [encoder setBuffer:x offset:0 atIndex:2];
            [encoder setBuffer:y offset:0 atIndex:3];
            [encoder setBytes:&shape length:sizeof(shape) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            if (command.status == MTLCommandBufferStatusError) {
                fprintf(stderr, "Metal command failed: %s\n", command.error.description.UTF8String);
                return 5;
            }
        }
        if (iteration + 1 == warmup) {
            // Start the measured interval after pipeline/cache warmup.
            (void)start;
        }
    }
    const double total = seconds_now() - start;

    // The timer includes warmup, so report the conservative per-dispatch wall time.
    const double ms = total * 1e3 / (double)(warmup + iterations);
    const double gib_per_s = ((double)weight_bytes / (1024.0 * 1024.0 * 1024.0)) /
                             (ms / 1e3);
    const double dense_tflops = (2.0 * (double)rows * cols * batch) / (ms * 1e9);
    const float checksum = ((float *)y.contents)[0] + ((float *)y.contents)[output_bytes / sizeof(float) - 1];
    printf("rows=%u cols=%u batch=%u: %.3f ms, %.1f GiB/s weight traffic, "
           "%.2f dense-equivalent TFLOP/s, checksum=%g\n",
           rows, cols, batch, ms, gib_per_s, dense_tflops, checksum);
    return 0;
}

int main(int argc, const char **argv) {
    @autoreleasepool {
        id<MTLDevice> device = select_device();
        if (!device) {
            fprintf(stderr, "No native Metal device is available.\n");
            return 1;
        }
        fprintf(stderr, "Selected native Metal device: %s\n", device.name.UTF8String);

        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        id<MTLLibrary> library = [device newLibraryWithSource:@(kShader)
                                                      options:options
                                                        error:&error];
        if (!library) {
            fprintf(stderr, "Metal shader compile failed: %s\n", error.description.UTF8String);
            return 1;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"q4_g64_gemv"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) {
            fprintf(stderr, "Metal pipeline creation failed: %s\n", error.description.UTF8String);
            return 1;
        }

        const uint32_t iterations = argc > 1 ? (uint32_t)strtoul(argv[1], NULL, 10) : 40;
        int status = run_shape(device, pipeline, 34816, 5120, 1, iterations);
        if (status == 0) status = run_shape(device, pipeline, 34816, 5120, 4, iterations);
        if (status == 0) status = run_shape(device, pipeline, 5120, 17408, 1, iterations);
        if (status == 0) status = run_shape(device, pipeline, 5120, 17408, 4, iterations);
        return status;
    }
}
