// Native-Metal Q4_K matrix benchmark for the Intel Mac Pro bring-up.
//
// Build against a standalone ggml checkout configured with:
//   -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
//
// This file intentionally uses ggml as a tensor dependency only. It does not
// contain or invoke a model runtime.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct Shape {
    int64_t output;
    int64_t input;
    int64_t batch;
};

bool benchmark_shape(
    ggml_backend_t backend,
    const Shape & shape,
    ggml_type weight_type,
    ggml_type activation_type,
    int iterations) {
    const size_t context_size = 4 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params params = {
        /* .mem_size   = */ context_size,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * context = ggml_init(params);
    if (context == nullptr) {
        std::fprintf(stderr, "ggml_init failed\n");
        return false;
    }

    ggml_tensor * weight =
        ggml_new_tensor_2d(context, weight_type, shape.input, shape.output);
    ggml_tensor * input =
        ggml_new_tensor_2d(context, activation_type, shape.input, shape.batch);
    ggml_tensor * output = ggml_mul_mat(context, weight, input);
    ggml_set_name(weight, "weight_q4_k");
    ggml_set_name(input, "input_f32");
    ggml_set_name(output, "output_f32");

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) {
        std::fprintf(stderr, "Metal allocation failed for %lldx%lld batch %lld\n",
                     static_cast<long long>(shape.output),
                     static_cast<long long>(shape.input),
                     static_cast<long long>(shape.batch));
        ggml_free(context);
        return false;
    }

    const size_t quantized_row_bytes = ggml_row_size(weight_type, shape.input);
    std::vector<float> source_row(shape.input);
    for (int64_t i = 0; i < shape.input; ++i) {
        source_row[i] = static_cast<float>((i % 31) - 15) / 32.0f;
    }
    std::vector<uint8_t> quantized_row(quantized_row_bytes);
    std::vector<float> imatrix(shape.input, 1.0f);
    const size_t written = ggml_quantize_chunk(
        weight_type,
        source_row.data(),
        quantized_row.data(),
        0,
        1,
        shape.input,
        ggml_quantize_requires_imatrix(weight_type) ? imatrix.data() : nullptr);
    if (written != quantized_row_bytes) {
        std::fprintf(stderr, "%s row quantization wrote %zu bytes, expected %zu\n",
                     ggml_type_name(weight_type),
                     written, quantized_row_bytes);
        ggml_backend_buffer_free(buffer);
        ggml_free(context);
        return false;
    }

    const size_t weight_bytes = quantized_row_bytes * shape.output;
    std::vector<uint8_t> quantized_weight(weight_bytes);
    for (int64_t row = 0; row < shape.output; ++row) {
        std::memcpy(quantized_weight.data() + row * quantized_row_bytes,
                    quantized_row.data(), quantized_row_bytes);
    }
    ggml_backend_tensor_set(weight, quantized_weight.data(), 0, weight_bytes);
    quantized_weight.clear();
    quantized_weight.shrink_to_fit();

    std::vector<float> input_f32(shape.input * shape.batch);
    for (size_t i = 0; i < input_f32.size(); ++i) {
        input_f32[i] = static_cast<float>((i % 17) - 8) / 16.0f;
    }
    if (activation_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> input_f16(input_f32.size());
        ggml_fp32_to_fp16_row(input_f32.data(), input_f16.data(), input_f32.size());
        ggml_backend_tensor_set(
            input, input_f16.data(), 0, input_f16.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(
            input, input_f32.data(), 0, input_f32.size() * sizeof(float));
    }
    ggml_backend_synchronize(backend);

    constexpr int warmup_iterations = 5;
    for (int i = 0; i < warmup_iterations; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "Metal warmup failed\n");
            ggml_backend_buffer_free(buffer);
            ggml_free(context);
            return false;
        }
    }
    ggml_backend_synchronize(backend);

    const int64_t start_us = ggml_time_us();
    for (int i = 0; i < iterations; ++i) {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "Metal benchmark dispatch failed\n");
            ggml_backend_buffer_free(buffer);
            ggml_free(context);
            return false;
        }
    }
    ggml_backend_synchronize(backend);
    const int64_t elapsed_us = ggml_time_us() - start_us;

    const double milliseconds = elapsed_us / 1000.0 / iterations;
    const double gib_per_second =
        (static_cast<double>(weight_bytes) / (1024.0 * 1024.0 * 1024.0)) /
        (milliseconds / 1000.0);
    const double dense_equivalent_tflops =
        2.0 * shape.output * shape.input * shape.batch / (milliseconds * 1.0e9);
    std::printf(
        "%s/%s output=%lld input=%lld batch=%lld: %.3f ms, %.1f GiB/s, "
        "%.2f dense-equivalent TFLOP/s\n",
        ggml_type_name(weight_type),
        ggml_type_name(activation_type),
        static_cast<long long>(shape.output),
        static_cast<long long>(shape.input),
        static_cast<long long>(shape.batch),
        milliseconds,
        gib_per_second,
        dense_equivalent_tflops);

    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 30;
    const char * requested_type = argc > 2 ? argv[2] : nullptr;
    const int64_t requested_batch = argc > 3 ? std::atoll(argv[3]) : 0;
    const ggml_type activation_type =
        argc > 4 && std::strcmp(argv[4], "f16") == 0 ? GGML_TYPE_F16
                                                     : GGML_TYPE_F32;
    ggml_backend_t backend = ggml_backend_metal_init();
    if (backend == nullptr) {
        std::fprintf(stderr, "No native Metal backend is available\n");
        return 1;
    }

    const Shape shapes[] = {
        {34816, 5120, 1},
        {34816, 5120, 4},
        {34816, 5120, 8},
        {34816, 5120, 16},
        {34816, 5120, 32},
        {5120, 17408, 1},
        {5120, 17408, 4},
        {5120, 17408, 8},
        {5120, 17408, 16},
        {5120, 17408, 32},
    };
    bool ok = true;
    const ggml_type weight_types[] = {
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q4_0,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ4_XS,
    };
    for (ggml_type weight_type : weight_types) {
        if (requested_type != nullptr &&
            std::strcmp(requested_type, ggml_type_name(weight_type)) != 0) {
            continue;
        }
        for (const Shape & shape : shapes) {
            if (requested_batch != 0 && shape.batch != requested_batch) {
                continue;
            }
            ok = benchmark_shape(
                     backend, shape, weight_type, activation_type, iterations) &&
                 ok;
        }
    }

    ggml_backend_free(backend);
    ggml_quantize_free();
    return ok ? 0 : 1;
}
