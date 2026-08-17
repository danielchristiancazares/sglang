#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MlxQwen38Engine MlxQwen38Engine;

typedef struct MlxQwen38Config {
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t num_hidden_layers;
  int32_t num_attention_heads;
  int32_t num_key_value_heads;
  int32_t head_dim;
  int32_t vocab_size;
  int32_t full_attention_interval;
  int32_t linear_num_value_heads;
  int32_t linear_num_key_heads;
  int32_t linear_key_head_dim;
  int32_t linear_value_head_dim;
  int32_t linear_conv_kernel_dim;
  float rms_norm_eps;
  float rope_theta;
  float partial_rotary_factor;
  int32_t quant_group_size;
  int32_t quant_bits;
} MlxQwen38Config;

/* Fill `out` from a Hugging Face `config.json` text_config. Returns 0 on success. */
int mlx_qwen38_config_from_json(const char* json, MlxQwen38Config* out, char* err, int errlen);

/* Affine-q4 quantized matmul used by every linear in this checkpoint. */
int mlx_qwen38_qlinear(
    const float* x,
    int x_rows,
    int x_cols,
    const uint32_t* w,
    const float* scales,
    const float* biases,
    int out_features,
    int group_size,
    int bits,
    float* y,
    char* err,
    int errlen);

/* One Gated-DeltaNet recurrent step. Shapes: q/k [H, Dk], v [Hv, Dv],
   g/beta [Hv], state [Hv, Dv, Dk]. Hv must be a multiple of H. */
int mlx_qwen38_gated_delta_step(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    const float* state,
    int h_k,
    int h_v,
    int d_k,
    int d_v,
    float* y,
    float* state_out,
    char* err,
    int errlen);

/* Fused Metal Gated-DeltaNet over T steps. B=1.
   q/k [T, Hk, Dk], v [T, Hv, Dv], g/beta [T, Hv], state [Hv, Dv, Dk].
   Dk must be a multiple of 32. */
int mlx_qwen38_gated_delta_update(
    const float* q,
    const float* k,
    const float* v,
    const float* g,
    const float* beta,
    const float* state,
    int t,
    int h_k,
    int h_v,
    int d_k,
    int d_v,
    float* y,
    float* state_out,
    char* err,
    int errlen);

MlxQwen38Engine* mlx_qwen38_load(
    const MlxQwen38Config* cfg,
    const char* model_dir,
    char* err,
    int errlen);

void mlx_qwen38_reset(MlxQwen38Engine* engine);

/* Prefill `n` tokens, write the next-token id, and keep decode caches.
   schedule_decode != 0 starts the first decode graph before returning. */
int mlx_qwen38_prefill(
    MlxQwen38Engine* engine,
    const int32_t* tokens,
    int n,
    int schedule_decode,
    int32_t* next_token,
    char* err,
    int errlen);

int mlx_qwen38_decode(
    MlxQwen38Engine* engine,
    int32_t token,
    int32_t* next_token,
    char* err,
    int errlen);

int mlx_qwen38_load_mtp(
    MlxQwen38Engine* engine,
    const char* mtp_dir,
    char* err,
    int errlen);

int mlx_qwen38_has_mtp(MlxQwen38Engine* engine);

int mlx_qwen38_last_spec_width(MlxQwen38Engine* engine);

void mlx_qwen38_free(MlxQwen38Engine* engine);

#ifdef __cplusplus
}
#endif
