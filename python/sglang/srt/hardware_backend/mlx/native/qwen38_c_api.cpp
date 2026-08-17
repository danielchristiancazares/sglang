#include "qwen38_c_api.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "qwen38_engine.h"

namespace {

void set_err(char* err, int errlen, const std::string& msg) {
  if (err == nullptr || errlen <= 0) {
    return;
  }
  std::snprintf(err, static_cast<size_t>(errlen), "%s", msg.c_str());
}

int find_int(const char* json, const char* key, int fallback) {
  std::string pat = std::string("\"") + key + "\"";
  const char* p = std::strstr(json, pat.c_str());
  if (p == nullptr) {
    return fallback;
  }
  p = std::strchr(p + pat.size(), ':');
  if (p == nullptr) {
    return fallback;
  }
  return static_cast<int>(std::strtol(p + 1, nullptr, 10));
}

float find_float(const char* json, const char* key, float fallback) {
  std::string pat = std::string("\"") + key + "\"";
  const char* p = std::strstr(json, pat.c_str());
  if (p == nullptr) {
    return fallback;
  }
  p = std::strchr(p + pat.size(), ':');
  if (p == nullptr) {
    return fallback;
  }
  return std::strtof(p + 1, nullptr);
}

} // namespace

int mlx_qwen38_config_from_json(const char* json, MlxQwen38Config* out, char* err, int errlen) {
  if (json == nullptr || out == nullptr) {
    set_err(err, errlen, "null config json");
    return -1;
  }
  const char* text = std::strstr(json, "\"text_config\"");
  const char* src = text != nullptr ? text : json;
  std::memset(out, 0, sizeof(*out));
  out->hidden_size = find_int(src, "hidden_size", 0);
  out->intermediate_size = find_int(src, "intermediate_size", 0);
  out->num_hidden_layers = find_int(src, "num_hidden_layers", 0);
  out->num_attention_heads = find_int(src, "num_attention_heads", 0);
  out->num_key_value_heads = find_int(src, "num_key_value_heads", 0);
  out->head_dim = find_int(src, "head_dim", 0);
  out->vocab_size = find_int(src, "vocab_size", 0);
  out->full_attention_interval = find_int(src, "full_attention_interval", 4);
  out->linear_num_value_heads = find_int(src, "linear_num_value_heads", 0);
  out->linear_num_key_heads = find_int(src, "linear_num_key_heads", 0);
  out->linear_key_head_dim = find_int(src, "linear_key_head_dim", 0);
  out->linear_value_head_dim = find_int(src, "linear_value_head_dim", 0);
  out->linear_conv_kernel_dim = find_int(src, "linear_conv_kernel_dim", 4);
  out->rms_norm_eps = find_float(src, "rms_norm_eps", 1e-6f);
  out->rope_theta = find_float(src, "rope_theta", 100000.0f);
  out->partial_rotary_factor = find_float(src, "partial_rotary_factor", 0.25f);
  out->quant_group_size = 64;
  out->quant_bits = 4;
  if (out->hidden_size <= 0 || out->num_hidden_layers <= 0 || out->vocab_size <= 0) {
    set_err(err, errlen, "config.json missing required text_config fields");
    return -1;
  }
  return 0;
}

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
    int errlen) {
  try {
    using mlx::core::array;
    using mlx::core::eval;
    using mlx::core::float32;
    using mlx::core::quantize;
    using mlx::core::quantized_matmul;
    using mlx::core::reshape;
    array xa(x, {x_rows, x_cols}, float32);
    int packed_cols = (x_cols * bits + 31) / 32;
    array wa(w, {out_features, packed_cols}, mlx::core::uint32);
    int n_groups = x_cols / group_size;
    array sa(scales, {out_features, n_groups}, float32);
    array ba(biases, {out_features, n_groups}, float32);
    array out = quantized_matmul(
        xa, wa, sa, ba, true, group_size, bits, "affine");
    eval(out);
    auto* data = out.data<float>();
    std::memcpy(y, data, static_cast<size_t>(x_rows * out_features) * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

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
    int errlen) {
  try {
    using mlx::core::array;
    using mlx::core::eval;
    using mlx::core::float32;
    if (h_v % h_k != 0) {
      throw std::runtime_error("h_v must be a multiple of h_k");
    }
    array qa(q, {1, h_k, d_k}, float32);
    array ka(k, {1, h_k, d_k}, float32);
    array va(v, {1, h_v, d_v}, float32);
    array ga(g, {1, h_v}, float32);
    array ba(beta, {1, h_v}, float32);
    array st(state, {1, h_v, d_v, d_k}, float32);
    if (h_v != h_k) {
      int rep = h_v / h_k;
      qa = mlx::core::repeat(qa, rep, 1);
      ka = mlx::core::repeat(ka, rep, 1);
    }
    auto step = sglang::mlx_qwen38::gated_delta_step(qa, ka, va, ga, ba, st);
    eval(step.first, step.second);
    std::memcpy(y, step.first.data<float>(), static_cast<size_t>(h_v * d_v) * sizeof(float));
    std::memcpy(
        state_out,
        step.second.data<float>(),
        static_cast<size_t>(h_v * d_v * d_k) * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

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
    int errlen) {
  try {
    using mlx::core::array;
    using mlx::core::eval;
    using mlx::core::float32;
    if (t <= 0) {
      throw std::runtime_error("T must be positive");
    }
    if (h_v % h_k != 0) {
      throw std::runtime_error("h_v must be a multiple of h_k");
    }
    array qa(q, {1, t, h_k, d_k}, float32);
    array ka(k, {1, t, h_k, d_k}, float32);
    array va(v, {1, t, h_v, d_v}, float32);
    array ga(g, {1, t, h_v}, float32);
    array ba(beta, {1, t, h_v}, float32);
    array st(state, {1, h_v, d_v, d_k}, float32);
    auto step = sglang::mlx_qwen38::gated_delta_update(qa, ka, va, ga, ba, st);
    eval(step.first, step.second);
    std::memcpy(
        y,
        step.first.data<float>(),
        static_cast<size_t>(t) * static_cast<size_t>(h_v * d_v) * sizeof(float));
    std::memcpy(
        state_out,
        step.second.data<float>(),
        static_cast<size_t>(h_v * d_v * d_k) * sizeof(float));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

struct MlxQwen38Engine {
  sglang::mlx_qwen38::Engine impl;
  MlxQwen38Engine(const MlxQwen38Config& cfg, const std::string& dir) : impl(cfg, dir) {}
};

MlxQwen38Engine* mlx_qwen38_load(
    const MlxQwen38Config* cfg,
    const char* model_dir,
    char* err,
    int errlen) {
  try {
    if (cfg == nullptr || model_dir == nullptr) {
      throw std::runtime_error("null load arguments");
    }
    return new MlxQwen38Engine(*cfg, model_dir);
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return nullptr;
  }
}

void mlx_qwen38_reset(MlxQwen38Engine* engine) {
  if (engine != nullptr) {
    engine->impl.reset();
  }
}

int mlx_qwen38_prefill(
    MlxQwen38Engine* engine,
    const int32_t* tokens,
    int n,
    int schedule_decode,
    int32_t* next_token,
    char* err,
    int errlen) {
  try {
    if (engine == nullptr || tokens == nullptr || next_token == nullptr) {
      throw std::runtime_error("null prefill arguments");
    }
    *next_token = engine->impl.prefill(tokens, n, schedule_decode != 0);
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

int mlx_qwen38_decode(
    MlxQwen38Engine* engine,
    int32_t token,
    int32_t* next_token,
    char* err,
    int errlen) {
  try {
    if (engine == nullptr || next_token == nullptr) {
      throw std::runtime_error("null decode arguments");
    }
    *next_token = engine->impl.decode(token);
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

int mlx_qwen38_load_mtp(
    MlxQwen38Engine* engine,
    const char* mtp_dir,
    char* err,
    int errlen) {
  try {
    if (engine == nullptr || mtp_dir == nullptr) {
      throw std::runtime_error("null MTP load arguments");
    }
    engine->impl.load_mtp(mtp_dir);
    return 0;
  } catch (const std::exception& e) {
    set_err(err, errlen, e.what());
    return -1;
  }
}

int mlx_qwen38_has_mtp(MlxQwen38Engine* engine) {
  return engine != nullptr && engine->impl.has_mtp() ? 1 : 0;
}

int mlx_qwen38_last_spec_width(MlxQwen38Engine* engine) {
  return engine != nullptr ? engine->impl.last_spec_width() : 0;
}

void mlx_qwen38_free(MlxQwen38Engine* engine) {
  delete engine;
}
