#include "qwen38_engine.h"

#include <cmath>
#include <dirent.h>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/transforms.h"

namespace sglang {
namespace mlx_qwen38 {
namespace {

using mlx::core::array;
using mlx::core::astype;
using mlx::core::async_eval;
using mlx::core::concatenate;
using mlx::core::eval;
using mlx::core::expand_dims;
using mlx::core::logaddexp;
using mlx::core::matmul;
using mlx::core::quantized_matmul;
using mlx::core::reshape;
using mlx::core::sigmoid;
using mlx::core::slice;
using mlx::core::split;
using mlx::core::sum;
using mlx::core::take;
using mlx::core::transpose;
using mlx::core::zeros;
namespace mx = mlx::core;

std::string layer_key(int i, const std::string& rest) {
  return "language_model.model.layers." + std::to_string(i) + rest;
}

array last_token(const array& hidden) {
  auto shape = hidden.shape();
  int seq = static_cast<int>(shape[1]);
  return squeeze(slice(hidden, {0, seq - 1, 0}, {1, seq, shape[2]}), 1);
}

// Same Metal body as mlx-lm's gated_delta_kernel (scalar g, no mask).
constexpr const char* kGatedDeltaSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] * g_[hv_idx];
              kv_mem += state[i] * k_[s_idx];
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
)";

std::vector<array> compute_g_fn(const std::vector<array>& xs) {
  return {mx::exp(-mx::exp(astype(xs[0], mx::float32)) * softplus(xs[1] + xs[2]))};
}

const std::function<std::vector<array>(const std::vector<array>&)>&
compiled_compute_g() {
  static auto fn = mx::compile(compute_g_fn, /*shapeless=*/true);
  return fn;
}

const mx::fast::CustomKernelFunction& gated_delta_metal() {
  static const auto kernel = mx::fast::metal_kernel(
      "sglang_gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      kGatedDeltaSource);
  return kernel;
}

} // namespace

array silu(const array& x) {
  return x * sigmoid(x);
}

array softplus(const array& x) {
  return logaddexp(x, array(0.0f));
}

array QLinear::operator()(const array& x) const {
  if (!valid) {
    throw std::runtime_error("QLinear used before load");
  }
  return quantized_matmul(
      x, w, scales, biases, /*transpose=*/true, group_size, bits, "affine");
}

std::pair<array, array> gated_delta_step(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  // q,k: [B, H, Dk]; v: [B, H, Dv]; g,beta: [B, H]; state: [B, H, Dv, Dk]
  array decay = expand_dims(expand_dims(g, -1), -1);
  array st = state * decay;
  array k_exp = expand_dims(k, -2);
  array kv_mem = sum(st * k_exp, -1);
  array delta = (v - kv_mem) * expand_dims(beta, -1);
  st = st + k_exp * expand_dims(delta, -1);
  array q_exp = expand_dims(q, -2);
  array y = sum(st * q_exp, -1);
  return {astype(y, q.dtype()), st};
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  int B = static_cast<int>(q.shape()[0]);
  int T = static_cast<int>(q.shape()[1]);
  int Hk = static_cast<int>(q.shape()[2]);
  int Dk = static_cast<int>(q.shape()[3]);
  int Hv = static_cast<int>(v.shape()[2]);
  int Dv = static_cast<int>(v.shape()[3]);
  if (Dk % 32 != 0) {
    throw std::runtime_error("gated_delta Metal kernel requires Dk multiple of 32");
  }
  if (Hv % Hk != 0) {
    throw std::runtime_error("Hv must be a multiple of Hk");
  }
  auto outs = gated_delta_metal()(
      {q, k, v, g, beta, state, array(T, mx::int32)},
      {{B, T, Hv, Dv}, state.shape()},
      {q.dtype(), state.dtype()},
      {32, Dv, B * Hv},
      {32, 4, 1},
      {
          {"InT", mx::fast::TemplateArg{q.dtype()}},
          {"StT", mx::fast::TemplateArg{state.dtype()}},
          {"Dk", mx::fast::TemplateArg{Dk}},
          {"Dv", mx::fast::TemplateArg{Dv}},
          {"Hk", mx::fast::TemplateArg{Hk}},
          {"Hv", mx::fast::TemplateArg{Hv}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

Engine::Engine(MlxQwen38Config cfg, const std::string& model_dir) : cfg_(cfg) {
  if (cfg_.hidden_size <= 0 || cfg_.num_hidden_layers <= 0) {
    throw std::runtime_error("invalid Qwen3.8 config");
  }
  layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
  load_weights(model_dir);
  reset();
}

void Engine::reset() {
  for (auto& layer : layers_) {
    if (layer.is_linear) {
      layer.linear.has_state = false;
    } else {
      layer.attn.offset = 0;
      layer.attn.keys = array(0);
      layer.attn.values = array(0);
    }
  }
  pending_tok_ = array(0);
  last_hidden_ = array(0);
  last_emitted_ = -1;
  decode_scheduled_ = false;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  if (mtp_valid_) {
    mtp_reset();
  }
}

QLinear Engine::load_qlinear(
    const std::unordered_map<std::string, array>& weights,
    const std::string& prefix) {
  QLinear q;
  q.w = require(weights, prefix + ".weight");
  q.scales = require(weights, prefix + ".scales");
  q.biases = require(weights, prefix + ".biases");
  q.group_size = cfg_.quant_group_size;
  q.bits = cfg_.quant_bits;
  q.valid = true;
  return q;
}

array Engine::require(
    const std::unordered_map<std::string, array>& weights,
    const std::string& key) const {
  auto it = weights.find(key);
  if (it == weights.end()) {
    throw std::runtime_error("missing weight " + key);
  }
  return it->second;
}

void Engine::load_weights(const std::string& model_dir) {
  std::unordered_map<std::string, array> weights;
  DIR* dir = opendir(model_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("cannot open model dir " + model_dir);
  }
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() < 12 || name.substr(name.size() - 12) != ".safetensors") {
      continue;
    }
    auto loaded = mx::load_safetensors(model_dir + "/" + name);
    for (auto& kv : loaded.first) {
      if (kv.first.rfind("vision_tower", 0) == 0) {
        continue;
      }
      weights.emplace(kv.first, std::move(kv.second));
    }
  }
  closedir(dir);
  if (weights.empty()) {
    throw std::runtime_error("no language-model safetensors in " + model_dir);
  }

  bool shift_norms = false;
  for (auto& kv : weights) {
    if (kv.first.find("conv1d.weight") != std::string::npos) {
      auto shape = kv.second.shape();
      if (shape.size() == 3 && shape[2] != 1) {
        kv.second = transpose(kv.second, {0, 2, 1});
        shift_norms = true;
      }
    }
  }
  if (shift_norms) {
    for (auto& kv : weights) {
      const auto& k = kv.first;
      if (k.size() >= 7 && k.substr(k.size() - 7) == ".weight") {
        bool is_norm = k.find("layernorm") != std::string::npos ||
            k.find(".q_norm.") != std::string::npos ||
            k.find(".k_norm.") != std::string::npos ||
            k == "language_model.model.norm.weight";
        if (is_norm && kv.second.ndim() == 1) {
          kv.second = kv.second + array(1.0f);
        }
      }
    }
  }

  embed_tokens_ = load_qlinear(weights, "language_model.model.embed_tokens");
  embed_table_ = mx::dequantize(
      embed_tokens_.w,
      embed_tokens_.scales,
      embed_tokens_.biases,
      cfg_.quant_group_size,
      cfg_.quant_bits,
      "affine",
      std::nullopt,
      mx::bfloat16);
  eval(embed_table_);
  lm_head_ = load_qlinear(weights, "language_model.lm_head");
  final_norm_ = require(weights, "language_model.model.norm.weight");

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    DecoderLayer layer;
    layer.is_linear = ((i + 1) % cfg_.full_attention_interval) != 0;
    layer.input_norm = require(weights, layer_key(i, ".input_layernorm.weight"));
    layer.post_norm = require(weights, layer_key(i, ".post_attention_layernorm.weight"));
    layer.gate_proj = load_qlinear(weights, layer_key(i, ".mlp.gate_proj"));
    layer.up_proj = load_qlinear(weights, layer_key(i, ".mlp.up_proj"));
    layer.down_proj = load_qlinear(weights, layer_key(i, ".mlp.down_proj"));
    if (layer.is_linear) {
      layer.linear.in_proj_qkv =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_qkv"));
      layer.linear.in_proj_z =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_z"));
      layer.linear.in_proj_b =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_b"));
      layer.linear.in_proj_a =
          load_qlinear(weights, layer_key(i, ".linear_attn.in_proj_a"));
      layer.linear.out_proj =
          load_qlinear(weights, layer_key(i, ".linear_attn.out_proj"));
      layer.linear.conv1d = require(weights, layer_key(i, ".linear_attn.conv1d.weight"));
      layer.linear.A_log = require(weights, layer_key(i, ".linear_attn.A_log"));
      layer.linear.dt_bias = require(weights, layer_key(i, ".linear_attn.dt_bias"));
      layer.linear.norm = require(weights, layer_key(i, ".linear_attn.norm.weight"));
    } else {
      layer.attn.q_proj = load_qlinear(weights, layer_key(i, ".self_attn.q_proj"));
      layer.attn.k_proj = load_qlinear(weights, layer_key(i, ".self_attn.k_proj"));
      layer.attn.v_proj = load_qlinear(weights, layer_key(i, ".self_attn.v_proj"));
      layer.attn.o_proj = load_qlinear(weights, layer_key(i, ".self_attn.o_proj"));
      layer.attn.q_norm = require(weights, layer_key(i, ".self_attn.q_norm.weight"));
      layer.attn.k_norm = require(weights, layer_key(i, ".self_attn.k_norm.weight"));
    }
    layers_[static_cast<size_t>(i)] = std::move(layer);
  }
}

array Engine::embed(const array& tokens) const {
  return take(embed_table_, tokens, 0);
}

array Engine::logits(const array& hidden) const {
  array n = mx::fast::rms_norm(hidden, final_norm_, cfg_.rms_norm_eps);
  return lm_head_(n);
}

array Engine::mlp(const DecoderLayer& layer, const array& x) const {
  return layer.down_proj(silu(layer.gate_proj(x)) * layer.up_proj(x));
}

array Engine::full_attn(FullAttn& attn, const array& x) {
  auto xshape = x.shape();
  int B = static_cast<int>(xshape[0]);
  int L = static_cast<int>(xshape[1]);
  int n_q = cfg_.num_attention_heads;
  int n_kv = cfg_.num_key_value_heads;
  int hd = cfg_.head_dim;
  int rope_dims = static_cast<int>(std::lround(hd * cfg_.partial_rotary_factor));

  array qg = attn.q_proj(x);
  qg = reshape(qg, {B, L, n_q, hd * 2});
  auto q_gate = split(qg, 2, -1);
  array queries = q_gate[0];
  array gate = reshape(q_gate[1], {B, L, n_q * hd});

  array keys = reshape(attn.k_proj(x), {B, L, n_kv, hd});
  array values = reshape(attn.v_proj(x), {B, L, n_kv, hd});

  queries = mx::fast::rms_norm(queries, attn.q_norm, cfg_.rms_norm_eps);
  keys = mx::fast::rms_norm(keys, attn.k_norm, cfg_.rms_norm_eps);
  queries = transpose(queries, {0, 2, 1, 3});
  keys = transpose(keys, {0, 2, 1, 3});
  values = transpose(values, {0, 2, 1, 3});

  queries = mx::fast::rope(
      queries, rope_dims, /*traditional=*/false, cfg_.rope_theta, 1.0f, attn.offset);
  keys = mx::fast::rope(
      keys, rope_dims, /*traditional=*/false, cfg_.rope_theta, 1.0f, attn.offset);

  if (attn.offset > 0 && attn.keys.ndim() == 4) {
    keys = concatenate({attn.keys, keys}, 2);
    values = concatenate({attn.values, values}, 2);
  }
  attn.keys = keys;
  attn.values = values;
  attn.offset += L;

  std::string mask_mode = (L > 1) ? "causal" : "";
  array output = mx::fast::scaled_dot_product_attention(
      queries, keys, values, 1.0f / std::sqrt(static_cast<float>(hd)), mask_mode);
  output = reshape(transpose(output, {0, 2, 1, 3}), {B, L, n_q * hd});
  return attn.o_proj(output * sigmoid(gate));
}

array Engine::gated_delta(LinearAttn& lin, const array& x) {
  auto xshape = x.shape();
  int B = static_cast<int>(xshape[0]);
  int S = static_cast<int>(xshape[1]);
  int hk = cfg_.linear_num_key_heads;
  int hv = cfg_.linear_num_value_heads;
  int dk = cfg_.linear_key_head_dim;
  int dv = cfg_.linear_value_head_dim;
  int key_dim = hk * dk;
  int value_dim = hv * dv;
  int conv_dim = key_dim * 2 + value_dim;
  int ksz = cfg_.linear_conv_kernel_dim;

  array qkv = lin.in_proj_qkv(x);
  array z = reshape(lin.in_proj_z(x), {B, S, hv, dv});
  array b = lin.in_proj_b(x);
  array a = lin.in_proj_a(x);

  if (!lin.has_state) {
    lin.conv_state = zeros({B, ksz - 1, conv_dim}, x.dtype());
    lin.rec_state = zeros({B, hv, dv, dk}, mx::float32);
    lin.has_state = true;
  }
  array conv_input = concatenate({lin.conv_state, qkv}, 1);
  lin.conv_state = slice(
      conv_input, {0, S, 0}, {B, S + (ksz - 1), conv_dim});
  array conv_out = silu(mx::conv1d(conv_input, lin.conv1d, 1, 0, 1, conv_dim));

  auto qkv_split = split(conv_out, mx::Shape{key_dim, 2 * key_dim}, -1);
  array q = reshape(qkv_split[0], {B, S, hk, dk});
  array k = reshape(qkv_split[1], {B, S, hk, dk});
  array v = reshape(qkv_split[2], {B, S, hv, dv});

  float inv = 1.0f / std::sqrt(static_cast<float>(dk));
  q = (inv * inv) * mx::fast::rms_norm(q, std::nullopt, 1e-6f);
  k = inv * mx::fast::rms_norm(k, std::nullopt, 1e-6f);

  array beta = sigmoid(b);
  array g = compiled_compute_g()({lin.A_log, a, lin.dt_bias})[0];
  auto updated = gated_delta_update(q, k, v, g, beta, lin.rec_state);
  lin.rec_state = updated.second;
  array out = updated.first;
  array normed = mx::fast::rms_norm(out, lin.norm, cfg_.rms_norm_eps);
  array gated = silu(astype(z, mx::float32)) * astype(normed, mx::float32);
  gated = astype(gated, x.dtype());
  return lin.out_proj(reshape(gated, {B, S, value_dim}));
}

array Engine::forward_hidden(const array& tokens) {
  array h = embed(tokens);
  for (auto& layer : layers_) {
    array n = mx::fast::rms_norm(h, layer.input_norm, cfg_.rms_norm_eps);
    array r = layer.is_linear ? gated_delta(layer.linear, n) : full_attn(layer.attn, n);
    h = h + r;
    array n2 = mx::fast::rms_norm(h, layer.post_norm, cfg_.rms_norm_eps);
    h = h + mlp(layer, n2);
  }
  return h;
}

array Engine::greedy_token(const array& hidden) {
  last_hidden_ = last_token(hidden);
  return mx::argmax(logits(last_hidden_), -1);
}

void Engine::snapshot() {
  snap_.resize(layers_.size());
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    auto& s = snap_[i];
    s.is_linear = layer.is_linear;
    if (layer.is_linear) {
      s.has_state = layer.linear.has_state;
      s.conv = layer.linear.conv_state;
      s.rec = layer.linear.rec_state;
    } else {
      s.offset = layer.attn.offset;
      s.keys = layer.attn.keys;
      s.values = layer.attn.values;
    }
  }
}

void Engine::restore() {
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    const auto& s = snap_[i];
    if (layer.is_linear) {
      layer.linear.has_state = s.has_state;
      layer.linear.conv_state = s.conv;
      layer.linear.rec_state = s.rec;
    } else {
      layer.attn.offset = s.offset;
      layer.attn.keys = s.keys;
      layer.attn.values = s.values;
    }
  }
}

void Engine::forward_argmax(const int32_t* tokens, int n, int32_t* out) {
  array ids(tokens, {1, n}, mx::int32);
  array hidden = forward_hidden(ids);
  last_hidden_ = last_token(hidden);
  array tok = mx::argmax(logits(hidden), -1);
  eval(tok);
  auto* data = tok.data<int32_t>();
  for (int i = 0; i < n; ++i) {
    out[i] = data[i];
  }
}

void Engine::mtp_reset() {
  mtp_layer_.attn.keys = array(0);
  mtp_layer_.attn.values = array(0);
  int seq = 0;
  for (const auto& layer : layers_) {
    if (!layer.is_linear) {
      seq = layer.attn.offset;
      break;
    }
  }
  mtp_layer_.attn.offset = seq;
}

array Engine::mtp_forward(const array& token_embed, const array& hidden) {
  array h = mtp_fc_(concatenate(
      {mx::fast::rms_norm(token_embed, mtp_pre_emb_, cfg_.rms_norm_eps),
       mx::fast::rms_norm(hidden, mtp_pre_hid_, cfg_.rms_norm_eps)},
      -1));
  array n = mx::fast::rms_norm(h, mtp_layer_.input_norm, cfg_.rms_norm_eps);
  h = h + full_attn(mtp_layer_.attn, n);
  array n2 = mx::fast::rms_norm(h, mtp_layer_.post_norm, cfg_.rms_norm_eps);
  h = h + mlp(mtp_layer_, n2);
  return mx::fast::rms_norm(h, mtp_norm_, cfg_.rms_norm_eps);
}

int Engine::mtp_draft(int32_t bonus, int32_t* drafts, int n_draft) {
  mtp_reset();
  array hid = last_hidden_;
  if (hid.ndim() == 1) {
    hid = reshape(hid, {1, 1, hid.shape()[0]});
  } else if (hid.ndim() == 2) {
    hid = expand_dims(hid, 1);
  }
  int32_t tok = bonus;
  for (int i = 0; i < n_draft; ++i) {
    array ids(&tok, {1, 1}, mx::int32);
    hid = mtp_forward(embed(ids), hid);
    array next = mx::argmax(lm_head_(hid), -1);
    eval(next);
    tok = next.item<int32_t>();
    drafts[i] = tok;
  }
  return n_draft;
}

void Engine::spec_refill(int32_t token) {
  if (spec_buf_pos_ < spec_buf_n_ && token == last_emitted_) {
    return;
  }
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  const int n_draft = mtp_block_ - 1;
  int32_t drafts[8];
  mtp_draft(token, drafts, n_draft);
  snapshot();
  int32_t input[8];
  input[0] = token;
  for (int i = 0; i < n_draft; ++i) {
    input[i + 1] = drafts[i];
  }
  const int n_in = n_draft + 1;
  int32_t pred[8];
  forward_argmax(input, n_in, pred);
  int n_correct = 0;
  for (int i = 0; i < n_draft; ++i) {
    if (pred[i] != drafts[i]) {
      break;
    }
    ++n_correct;
  }
  if (n_correct < n_draft) {
    restore();
    forward_argmax(input, n_correct + 1, pred);
  }
  for (int i = 0; i < n_correct; ++i) {
    spec_buf_[i] = drafts[i];
  }
  spec_buf_[n_correct] = pred[n_correct];
  spec_buf_n_ = n_correct + 1;
  spec_buf_pos_ = 0;
}

void Engine::load_mtp(const std::string& mtp_dir) {
  std::unordered_map<std::string, array> weights;
  DIR* dir = opendir(mtp_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error("cannot open MTP dir " + mtp_dir);
  }
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() < 12 || name.substr(name.size() - 12) != ".safetensors") {
      continue;
    }
    auto loaded = mx::load_safetensors(mtp_dir + "/" + name);
    for (auto& kv : loaded.first) {
      weights.emplace(kv.first, std::move(kv.second));
    }
  }
  closedir(dir);
  if (weights.empty()) {
    throw std::runtime_error("no MTP safetensors in " + mtp_dir);
  }
  mtp_fc_ = load_qlinear(weights, "fc");
  mtp_pre_emb_ = require(weights, "pre_fc_norm_embedding.weight");
  mtp_pre_hid_ = require(weights, "pre_fc_norm_hidden.weight");
  mtp_norm_ = require(weights, "norm.weight");
  mtp_layer_.is_linear = false;
  mtp_layer_.input_norm = require(weights, "layers.0.input_layernorm.weight");
  mtp_layer_.post_norm = require(weights, "layers.0.post_attention_layernorm.weight");
  mtp_layer_.gate_proj = load_qlinear(weights, "layers.0.mlp.gate_proj");
  mtp_layer_.up_proj = load_qlinear(weights, "layers.0.mlp.up_proj");
  mtp_layer_.down_proj = load_qlinear(weights, "layers.0.mlp.down_proj");
  mtp_layer_.attn.q_proj = load_qlinear(weights, "layers.0.self_attn.q_proj");
  mtp_layer_.attn.k_proj = load_qlinear(weights, "layers.0.self_attn.k_proj");
  mtp_layer_.attn.v_proj = load_qlinear(weights, "layers.0.self_attn.v_proj");
  mtp_layer_.attn.o_proj = load_qlinear(weights, "layers.0.self_attn.o_proj");
  mtp_layer_.attn.q_norm = require(weights, "layers.0.self_attn.q_norm.weight");
  mtp_layer_.attn.k_norm = require(weights, "layers.0.self_attn.k_norm.weight");
  mtp_block_ = 3;
  mtp_valid_ = true;
  mtp_reset();
}

int32_t Engine::emit_scheduled() {
  eval(pending_tok_);
  last_emitted_ = pending_tok_.item<int32_t>();
  return last_emitted_;
}

int32_t Engine::prefill(const int32_t* tokens, int n, bool schedule_decode) {
  if (n <= 0) {
    throw std::runtime_error("prefill requires at least one token");
  }
  array ids(tokens, {1, n}, mx::int32);
  array first = greedy_token(forward_hidden(ids));
  async_eval(first);
  if (!schedule_decode) {
    pending_tok_ = first;
    decode_scheduled_ = false;
    return emit_scheduled();
  }
  array following = greedy_token(forward_hidden(reshape(first, {1, 1})));
  async_eval(following);
  pending_tok_ = first;
  int32_t out = emit_scheduled();
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

int32_t Engine::decode(int32_t token) {
  if (mtp_valid_) {
    if (spec_buf_pos_ >= spec_buf_n_ || token != last_emitted_) {
      spec_refill(token);
    }
    last_emitted_ = spec_buf_[spec_buf_pos_++];
    decode_scheduled_ = false;
    return last_emitted_;
  }
  if (decode_scheduled_ && token == last_emitted_) {
    array following = greedy_token(forward_hidden(reshape(pending_tok_, {1, 1})));
    async_eval(following);
    int32_t out = emit_scheduled();
    pending_tok_ = following;
    return out;
  }
  array ids = (token == last_emitted_) ? reshape(pending_tok_, {1, 1})
                                       : array(&token, {1, 1}, mx::int32);
  array next = greedy_token(forward_hidden(ids));
  async_eval(next);
  array following = greedy_token(forward_hidden(reshape(next, {1, 1})));
  async_eval(following);
  pending_tok_ = next;
  int32_t out = emit_scheduled();
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

} // namespace mlx_qwen38
} // namespace sglang
