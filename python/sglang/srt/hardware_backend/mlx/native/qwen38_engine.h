#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "mlx/array.h"
#include "mlx/ops.h"
#include "qwen38_c_api.h"

namespace sglang {
namespace mlx_qwen38 {

struct QLinear {
  mlx::core::array w{0};
  mlx::core::array scales{0};
  mlx::core::array biases{0};
  int group_size = 64;
  int bits = 4;
  bool valid = false;

  mlx::core::array operator()(const mlx::core::array& x) const;
};

struct FullAttn {
  QLinear q_proj;
  QLinear k_proj;
  QLinear v_proj;
  QLinear o_proj;
  mlx::core::array q_norm{0};
  mlx::core::array k_norm{0};
  mlx::core::array keys{0};
  mlx::core::array values{0};
  int offset = 0;
};

struct LinearAttn {
  QLinear in_proj_qkv;
  QLinear in_proj_z;
  QLinear in_proj_b;
  QLinear in_proj_a;
  QLinear out_proj;
  mlx::core::array conv1d{0};
  mlx::core::array A_log{0};
  mlx::core::array dt_bias{0};
  mlx::core::array norm{0};
  mlx::core::array conv_state{0};
  mlx::core::array rec_state{0};
  bool has_state = false;
};

struct DecoderLayer {
  bool is_linear = true;
  mlx::core::array input_norm{0};
  mlx::core::array post_norm{0};
  QLinear gate_proj;
  QLinear up_proj;
  QLinear down_proj;
  FullAttn attn;
  LinearAttn linear;
};

struct LayerSnap {
  mlx::core::array conv{0};
  mlx::core::array rec{0};
  mlx::core::array keys{0};
  mlx::core::array values{0};
  int offset = 0;
  bool has_state = false;
  bool is_linear = true;
};

class Engine {
 public:
  Engine(MlxQwen38Config cfg, const std::string& model_dir);

  void reset();
  int32_t prefill(const int32_t* tokens, int n, bool schedule_decode);
  int32_t decode(int32_t token);
  void load_mtp(const std::string& mtp_dir);
  bool has_mtp() const { return mtp_valid_; }
  int last_spec_width() const { return spec_buf_n_; }

  const MlxQwen38Config& config() const { return cfg_; }

 private:
  mlx::core::array embed(const mlx::core::array& tokens) const;
  mlx::core::array logits(const mlx::core::array& hidden) const;
  mlx::core::array mlp(const DecoderLayer& layer, const mlx::core::array& x) const;
  mlx::core::array full_attn(FullAttn& layer, const mlx::core::array& x);
  mlx::core::array gated_delta(LinearAttn& layer, const mlx::core::array& x);
  mlx::core::array forward_hidden(const mlx::core::array& tokens);
  mlx::core::array greedy_token(const mlx::core::array& hidden);
  int32_t emit_scheduled();
  void snapshot();
  void restore();
  void forward_argmax(const int32_t* tokens, int n, int32_t* out);
  void mtp_reset();
  int mtp_draft(int32_t bonus, int32_t* drafts, int n_draft);
  mlx::core::array mtp_forward(
      const mlx::core::array& token_embed, const mlx::core::array& hidden);
  void spec_refill(int32_t token);

  void load_weights(const std::string& model_dir);
  QLinear load_qlinear(
      const std::unordered_map<std::string, mlx::core::array>& weights,
      const std::string& prefix);
  mlx::core::array require(
      const std::unordered_map<std::string, mlx::core::array>& weights,
      const std::string& key) const;

  MlxQwen38Config cfg_;
  QLinear embed_tokens_;
  mlx::core::array embed_table_{0};
  QLinear lm_head_;
  mlx::core::array final_norm_{0};
  std::vector<DecoderLayer> layers_;
  mlx::core::array pending_tok_{0};
  mlx::core::array last_hidden_{0};
  int32_t last_emitted_ = -1;
  bool decode_scheduled_ = false;

  bool mtp_valid_ = false;
  int mtp_block_ = 3;
  DecoderLayer mtp_layer_;
  QLinear mtp_fc_;
  mlx::core::array mtp_pre_emb_{0};
  mlx::core::array mtp_pre_hid_{0};
  mlx::core::array mtp_norm_{0};
  std::vector<LayerSnap> snap_;
  int32_t spec_buf_[8]{};
  int spec_buf_n_ = 0;
  int spec_buf_pos_ = 0;
};

mlx::core::array silu(const mlx::core::array& x);
mlx::core::array softplus(const mlx::core::array& x);
std::pair<mlx::core::array, mlx::core::array> gated_delta_step(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& g,
    const mlx::core::array& beta,
    const mlx::core::array& state);
/* Metal fused recurrence. q/k [B,T,Hk,Dk], v [B,T,Hv,Dv], g/beta [B,T,Hv],
   state [B,Hv,Dv,Dk]. Dk must be a multiple of 32. */
std::pair<mlx::core::array, mlx::core::array> gated_delta_update(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& g,
    const mlx::core::array& beta,
    const mlx::core::array& state);

} // namespace mlx_qwen38
} // namespace sglang
