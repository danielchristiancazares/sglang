#include <torch/all.h>
#include <torch/library.h>

#include <optional>

torch::Tensor
ggml_dequantize(torch::Tensor W, int64_t type, int64_t m, int64_t n, std::optional<at::ScalarType> const& dtype);
torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row);
torch::Tensor ggml_mul_mat_a8(torch::Tensor W, torch::Tensor X, int64_t type, int64_t row);
torch::Tensor ggml_moe_a8(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens);
torch::Tensor ggml_moe_a8_vec(
    torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids, int64_t top_k, int64_t type, int64_t row, int64_t tokens);
int64_t ggml_moe_get_block_size(int64_t type);

TORCH_LIBRARY(sglang_windows_gguf, m) {
  m.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, "
      "ScalarType? dtype) -> Tensor");
  m.impl("ggml_dequantize", torch::kCUDA, &ggml_dequantize);

  m.def("ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  m.impl("ggml_mul_mat_vec_a8", torch::kCUDA, &ggml_mul_mat_vec_a8);

  m.def("ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  m.impl("ggml_mul_mat_a8", torch::kCUDA, &ggml_mul_mat_a8);

  m.def(
      "ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_padded, int type, SymInt row, "
      "int top_k, int tokens) -> Tensor");
  m.impl("ggml_moe_a8", torch::kCUDA, &ggml_moe_a8);

  m.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int top_k, "
      "int type, SymInt row, int tokens) -> Tensor");
  m.impl("ggml_moe_a8_vec", torch::kCUDA, &ggml_moe_a8_vec);

  m.def("ggml_moe_get_block_size(int type) -> int", &ggml_moe_get_block_size);
}
