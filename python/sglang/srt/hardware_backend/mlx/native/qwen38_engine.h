#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
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
  mlx::core::array fused_q4_decode_params{0};
  bool fused_q4_decode_params_valid = false;

  mlx::core::array operator()(const mlx::core::array& x) const;
};

bool prepare_fused_q4_raw_decode_parameters(
    QLinear& gate, const QLinear& up);

mlx::core::array affine_qmm_small_batch(
    const QLinear& linear, const mlx::core::array& x);
mlx::core::array affine_qmm_m8_ksplit(
    const QLinear& linear, const mlx::core::array& x);
mlx::core::array affine_q4_qmv_batch_one(
    const QLinear& linear, const mlx::core::array& x);
mlx::core::array affine_q4_qmv_batch_two(
    const QLinear& linear, const mlx::core::array& x);
mlx::core::array affine_q4_fused_swiglu_batch_one(
    const QLinear& gate,
    const QLinear& up,
    const mlx::core::array& x);
mlx::core::array affine_q4_fused_swiglu_batch_two(
    const QLinear& gate,
    const QLinear& up,
    const mlx::core::array& x);
mlx::core::array affine_q5_qmv_batch_one(
    const QLinear& linear, const mlx::core::array& x);
mlx::core::array affine_q5_qmv_batch_two(
    const QLinear& linear, const mlx::core::array& x);
std::pair<mlx::core::array, mlx::core::array> align_mtp_committed_history(
    const mlx::core::array& target_hidden,
    const mlx::core::array& token_ids,
    const mlx::core::array& previous_hidden,
    bool has_previous_hidden,
    const mlx::core::array& final_norm,
    float rms_norm_eps,
    bool post_norm_hidden);
mlx::core::array quantized_embedding_rows(
    const QLinear& embedding, const mlx::core::array& tokens);
mlx::core::array fixed_prefill_attention(
    const mlx::core::array& queries,
    const mlx::core::array& key_cache,
    const mlx::core::array& value_cache,
    int prefix_length,
    int active_cache_length);
mlx::core::array dspark_yarn_rope(
    const mlx::core::array& x, int offset);
mlx::core::array dspark_confidence(
    const mlx::core::array& hidden,
    const mlx::core::array& markov_embeddings,
    const mlx::core::array& weight,
    const mlx::core::array& bias);
int dspark_select_verify_draft_tokens(
    const float* confidence,
    int count,
    float full_to_short_cost_ratio);

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
  int cache_length = 0;
  int cache_capacity = 0;
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

struct DFlashDynamicConv {
  QLinear kernel_projection;
  mlx::core::array base_kernel{0};
};

struct DFlashAttention {
  QLinear q_proj;
  QLinear k_proj;
  QLinear v_proj;
  QLinear o_proj;
  mlx::core::array q_norm{0};
  mlx::core::array k_norm{0};
  mlx::core::array keys{0};
  mlx::core::array values{0};
  mlx::core::array positions{0};
  int cache_length = 0;
};

struct DFlashLayer {
  mlx::core::array input_norm{0};
  mlx::core::array post_norm{0};
  QLinear gate_proj;
  QLinear up_proj;
  QLinear down_proj;
  DFlashAttention attn;
  DFlashDynamicConv attention_conv;
  DFlashDynamicConv mlp_conv;
};

struct DSparkAttention {
  QLinear q_proj;
  QLinear k_proj;
  QLinear v_proj;
  QLinear o_proj;
  mlx::core::array q_norm{0};
  mlx::core::array k_norm{0};
  mlx::core::array keys{0};
  mlx::core::array values{0};
  int cache_length = 0;
  int cache_capacity = 0;
};

struct DSparkLayer {
  mlx::core::array input_norm{0};
  mlx::core::array post_norm{0};
  QLinear gate_proj;
  QLinear up_proj;
  QLinear down_proj;
  DSparkAttention attn;
};

struct LinearCommitTape {
  mlx::core::array conv_tokens{0};
  mlx::core::array keys{0};
  mlx::core::array decay{0};
  mlx::core::array delta{0};
  bool valid = false;
};

struct TargetForward {
  mlx::core::array hidden{0};
  std::vector<mlx::core::array> captured;
  std::vector<LinearCommitTape> linear_tapes;
};

struct LayerSnap {
  mlx::core::array conv{0};
  mlx::core::array rec{0};
  mlx::core::array keys{0};
  mlx::core::array values{0};
  int offset = 0;
  int cache_length = 0;
  int cache_capacity = 0;
  bool has_state = false;
  bool is_linear = true;
};

class Engine {
 public:
  Engine(MlxQwen38Config cfg, const std::string& model_dir);

  void reset();
  void begin_request();
  int32_t prefill(const int32_t* tokens, int n, bool schedule_decode);
  int32_t decode(int32_t token);
  void load_mtp(const std::string& mtp_dir);
  bool has_mtp() const {
    return mtp_valid_ || dflash_valid_ || dspark_valid_;
  }
  int last_spec_width() const { return spec_buf_n_; }

  const MlxQwen38Config& config() const { return cfg_; }

 private:
  mlx::core::array embed(const mlx::core::array& tokens) const;
  mlx::core::array logits(const mlx::core::array& hidden) const;
  mlx::core::array mlp(const DecoderLayer& layer, const mlx::core::array& x) const;
  mlx::core::array full_attn(FullAttn& layer, const mlx::core::array& x);
  mlx::core::array gated_delta(
      LinearAttn& layer,
      const mlx::core::array& x,
      LinearCommitTape* commit_tape = nullptr);
  mlx::core::array forward_hidden(const mlx::core::array& tokens);
  TargetForward forward_hidden_captured(
      const mlx::core::array& tokens,
      bool capture_commit_tape = false);
  mlx::core::array forward_hidden_impl(
      const mlx::core::array& tokens,
      std::vector<mlx::core::array>* captured,
      std::vector<LinearCommitTape>* commit_tapes);
  mlx::core::array select_token(const mlx::core::array& hidden);
  mlx::core::array sampling_probabilities(const mlx::core::array& token_logits);
  int32_t emit_scheduled();
  void reset_decode_pipeline();
  void record_processed_token(int32_t token);
  void snapshot();
  void restore();
  void forward_argmax(const int32_t* tokens, int n, int32_t* out);
  int target_sequence_length() const;
  mlx::core::array mtp_seed_hidden() const;
  void mtp_reset();
  void mtp_append_history(
      const mlx::core::array& target_hidden,
      const mlx::core::array& token_ids);
  void mtp_append_prompt_history(
      const mlx::core::array& target_hidden,
      const int32_t* tokens,
      int token_count,
      const mlx::core::array& previous_hidden,
      bool has_previous_hidden);
  void mtp_begin_committed_cycle();
  void mtp_commit_cycle(
      const int32_t* tokens,
      int token_count,
      const mlx::core::array& committed_hidden);
  int mtp_draft(int32_t bonus, int32_t* drafts, int n_draft);
  mlx::core::array mtp_forward(
      const mlx::core::array& token_embed, const mlx::core::array& hidden);
  void spec_refill(int32_t token);
  void load_dflash2(
      const std::unordered_map<std::string, mlx::core::array>& weights);
  void dflash_reset();
  void dflash_append_context(
      const std::vector<mlx::core::array>& captured,
      int token_count);
  void dflash_append_layer_context(
      DFlashAttention& attn,
      const mlx::core::array& projected,
      int position_offset);
  mlx::core::array dflash_grouped_convolve(
      const mlx::core::array& hidden,
      const mlx::core::array& dynamic,
      const mlx::core::array& base) const;
  std::pair<mlx::core::array, mlx::core::array> dflash_conv_prepare(
      const DFlashDynamicConv& conv,
      const mlx::core::array& hidden) const;
  mlx::core::array dflash_conv_finish(
      const DFlashDynamicConv& conv,
      const mlx::core::array& hidden,
      const mlx::core::array& dynamic) const;
  mlx::core::array dflash_attention(
      DFlashAttention& attn,
      const mlx::core::array& hidden);
  mlx::core::array dflash_forward(int32_t anchor);
  std::tuple<mlx::core::array, mlx::core::array, mlx::core::array>
  dflash_select(
      const mlx::core::array& hidden,
      const mlx::core::array& draft_logits,
      int32_t anchor);
  void draft_commit_verified_prefix(
      const TargetForward& verified, int token_count);
  int32_t target_only_spec_refill(int32_t token);
  float dspark_anchor_score(int32_t token);
  void dflash_spec_refill(int32_t token);
  void load_dspark(
      const std::unordered_map<std::string, mlx::core::array>& weights);
  void dspark_reset();
  void dspark_append_context(
      const std::vector<mlx::core::array>& captured,
      int token_count);
  void dspark_append_layer_context(
      DSparkAttention& attn,
      const mlx::core::array& projected,
      int position_offset);
  mlx::core::array dspark_attention(
      DSparkAttention& attn,
      const mlx::core::array& hidden);
  mlx::core::array dspark_forward(int32_t anchor);
  std::pair<mlx::core::array, mlx::core::array> dspark_propose(
      const mlx::core::array& hidden,
      int32_t anchor,
      bool sampled);
  void dspark_spec_refill(int32_t token);
  void draft_append_context(
      const std::vector<mlx::core::array>& captured,
      int token_count);
  int verify_speculative_block(
      int32_t token,
      const mlx::core::array& draft_tokens,
      const mlx::core::array& proposal_indices,
      const mlx::core::array& proposal_probs,
      const mlx::core::array& confidence,
      bool dense_proposal,
      bool greedy,
      float confidence_cost_ratio,
      float sparse_mean_q_threshold,
      const char* trace_tag);

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
  bool last_emitted_in_state_ = false;
  bool request_boundary_pending_ = false;
  std::vector<int32_t> token_history_;
  bool sampling_enabled_ = false;
  std::uint64_t sampling_seed_ = 67396869;
  int max_reasoning_tokens_ = 0;
  int selected_reasoning_tokens_ = 0;
  bool reasoning_open_ = false;
  bool reasoning_cap_selected_ = false;
  int target_only_prefill_chunk_size_ = 0;
  bool quantized_embedding_enabled_ = false;
  bool prompt_snapshot_valid_ = false;
  std::vector<int32_t> prompt_snapshot_history_;

  bool mtp_valid_ = false;
  int mtp_block_ = 3;
  DecoderLayer mtp_layer_;
  QLinear mtp_fc_;
  mlx::core::array mtp_pre_emb_{0};
  mlx::core::array mtp_pre_hid_{0};
  mlx::core::array mtp_norm_{0};
  bool mtp_committed_history_enabled_ = false;
  LayerSnap mtp_cycle_snapshot_;
  mlx::core::array mtp_cycle_previous_hidden_{0};
  bool mtp_cycle_pending_ = false;
  bool dflash_valid_ = false;
  int dflash_context_offset_ = 0;
  QLinear dflash_fc_;
  mlx::core::array dflash_hidden_norm_{0};
  mlx::core::array dflash_norm_{0};
  QLinear dflash_selector_hidden_;
  mlx::core::array dflash_predecessor_{0};
  mlx::core::array dflash_successor_{0};
  float dflash_selector_temperature_ = 1.0f;
  float dflash_mean_q_threshold_ = 0.0f;
  std::vector<DFlashLayer> dflash_layers_;
  bool dspark_valid_ = false;
  int dspark_context_offset_ = 0;
  QLinear dspark_fc_;
  mlx::core::array dspark_hidden_norm_{0};
  mlx::core::array dspark_norm_{0};
  mlx::core::array dspark_markov_w1_{0};
  QLinear dspark_markov_w2_;
  mlx::core::array dspark_confidence_weight_{0};
  mlx::core::array dspark_confidence_bias_{0};
  int dspark_verify_draft_tokens_ = 7;
  float dspark_confidence_cost_ratio_ = 0.0f;
  int dspark_bypass_refills_ = 0;
  int dspark_bypass_remaining_ = 0;
  std::vector<DSparkLayer> dspark_layers_;
  std::vector<LayerSnap> snap_;
  int32_t spec_buf_[8]{};
  int spec_buf_n_ = 0;
  int spec_buf_pos_ = 0;
};

mlx::core::array silu(const mlx::core::array& x);
mlx::core::array softplus(const mlx::core::array& x);
std::pair<mlx::core::array, mlx::core::array> causal_conv_decode_silu(
    const mlx::core::array& state,
    const mlx::core::array& qkv,
    const mlx::core::array& weight);
std::pair<mlx::core::array, mlx::core::array> residual_rms_norm(
    const mlx::core::array& x,
    const mlx::core::array& residual,
    const mlx::core::array& weight,
    float eps);
std::pair<mlx::core::array, mlx::core::array> normalize_gated_delta_qk(
    const mlx::core::array& q,
    const mlx::core::array& k,
    float q_scale,
    float k_scale,
    float eps);
std::pair<mlx::core::array, mlx::core::array> full_attn_qk_norm_rope(
    const mlx::core::array& qg,
    const mlx::core::array& k,
    const mlx::core::array& q_weight,
    const mlx::core::array& k_weight,
    float eps,
    float rope_theta,
    int rope_dims,
    int rope_offset);
mlx::core::array gated_delta_norm_gate(
    const mlx::core::array& recurrent_out,
    const mlx::core::array& z,
    const mlx::core::array& weight,
    float eps);
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
std::vector<mlx::core::array> gated_delta_update_outputs(
    const mlx::core::array& q,
    const mlx::core::array& k,
    const mlx::core::array& v,
    const mlx::core::array& g,
    const mlx::core::array& beta,
    const mlx::core::array& state,
    bool capture_delta);
mlx::core::array gated_delta_commit(
    const mlx::core::array& keys,
    const mlx::core::array& decay,
    const mlx::core::array& delta,
    const mlx::core::array& state,
    int token_count);

} // namespace mlx_qwen38
} // namespace sglang
