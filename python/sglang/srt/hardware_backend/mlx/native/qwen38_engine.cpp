#include "qwen38_engine.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <dirent.h>
#include <functional>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/io.h"
#include "mlx/memory.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

namespace sglang {
namespace mlx_qwen38 {
// Kernel factories and request state share one immutable process-wide format.
// Keep BF16 as the default; FP16 is an explicit numerical execution variant.
mlx::core::Dtype activation_dtype() {
  static const auto dtype = [] {
    const char* value = std::getenv("SGLANG_MLX_NATIVE_ACTIVATION_DTYPE");
    if (value == nullptr || std::string_view(value) == "bfloat16")
      return mlx::core::bfloat16;
    if (std::string_view(value) == "float16") return mlx::core::float16;
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_ACTIVATION_DTYPE must be bfloat16 or float16");
  }();
  return dtype;
}

ActivationConversion convert_activation_parameter(
    const mlx::core::array& value, const std::string& name) {
  namespace mx = mlx::core;
  if (activation_dtype() != mx::float16 || value.dtype() != mx::bfloat16)
    return {value, 0, 0.0f};
  auto converted = mx::astype(value, mx::float16);
  if (value.size() == 0) return {converted, 0, 0.0f};
  if (value.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("FP16 parameter audit exceeds counter capacity: " + name);
  auto before = mx::astype(value, mx::float32);
  auto after = mx::astype(converted, mx::float32);
  auto changed = mx::sum(mx::astype(mx::not_equal(before, after), mx::uint32));
  auto error = mx::max(mx::abs(before - after));
  auto finite = mx::all(mx::isfinite(before) && mx::isfinite(after));
  mx::eval(converted, changed, error, finite);
  // BF16 normal values in FP16 range are representable exactly. Only the
  // smaller exponent range may round a tiny value by at most half an FP16 ULP.
  if (!finite.item<bool>() || error.item<float>() > 0x1p-25f)
    throw std::runtime_error("FP16 parameter is outside the supported range: " + name);
  return {std::move(converted), changed.item<std::uint32_t>(), error.item<float>()};
}

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
using mlx::core::slice_update;
using mlx::core::split;
using mlx::core::sum;
using mlx::core::take;
using mlx::core::transpose;
using mlx::core::zeros;
namespace mx = mlx::core;
auto load_activation_safetensors(const std::string& path) {
  auto loaded = mx::load_safetensors(path);
  if (activation_dtype() == mx::bfloat16) return loaded;
  std::uint64_t values = 0, changed = 0, packed_words = 0;
  float maximum_error = 0.0f;
  for (auto& [name, value] : loaded.first) {
    // The language-only engine never consumes the checkpoint's vision tower.
    if (name.starts_with("vision_tower")) continue;
    if (value.dtype() == mx::uint32) {
      packed_words += value.size();
      continue;
    }
    if (value.dtype() != mx::bfloat16) continue;
    const auto converted = convert_activation_parameter(value, name);
    values += value.size();
    changed += converted.changed_values;
    maximum_error = std::max(maximum_error, converted.maximum_absolute_error);
    value = converted.value;
  }
  std::fprintf(stderr,
      "qwen38_fp16_parameters path=%s values=%llu changed=%llu max_abs=%.9g unchanged_packed_words=%llu\n",
      path.c_str(), static_cast<unsigned long long>(values),
      static_cast<unsigned long long>(changed), maximum_error,
      static_cast<unsigned long long>(packed_words));
  return loaded;
}

mx::fast::CustomKernelFunction activation_metal_kernel(
    const std::string& name, const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs, const std::string& source,
    const std::string& header = "") {
  const bool half = activation_dtype() == mx::float16;
  const std::string prefix = half
      ? "#include <metal_simdgroup_matrix>\nusing Activation = half;\nusing ActivationMatrix = simdgroup_half8x8;\n#define SGLANG_NATIVE_FP16 1\n"
      : "#include <metal_simdgroup_matrix>\nusing Activation = bfloat;\nusing ActivationMatrix = simdgroup_bfloat8x8;\n#define SGLANG_NATIVE_FP16 0\n";
  return mx::fast::metal_kernel(name + (half ? "_fp16" : ""), inputs, outputs,
      source, prefix + header);
}


// cfg_ is Engine's first member, so this initializer runs before the MLX array
// members can create the Metal device and cache its command-buffer limits.
MlxQwen38Config configure_mlx_runtime(MlxQwen38Config cfg) {
  if (std::getenv("MLX_MAX_MB_PER_BUFFER") == nullptr &&
      setenv("MLX_MAX_MB_PER_BUFFER", "128", 0) != 0) {
    throw std::runtime_error("failed to set MLX command-buffer byte budget");
  }
  if (const char* value = std::getenv("SGLANG_MLX_CACHE_LIMIT_GB")) {
    const std::string_view text(value);
    double gib = 0.0;
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), gib);
    constexpr size_t kGiB = size_t{1024} * 1024 * 1024;
    if (error != std::errc{} || end != text.data() + text.size() ||
        !std::isfinite(gib) || gib < 0.0 ||
        gib > static_cast<double>(std::numeric_limits<size_t>::max() / kGiB)) {
      throw std::runtime_error(
          "SGLANG_MLX_CACHE_LIMIT_GB must be a nonnegative finite size");
    }
    mx::set_cache_limit(static_cast<size_t>(gib * kGiB));
  }
  return cfg;
}

bool native_sampling_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_SAMPLING");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_state_trace_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_TRACE_STATE");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_spec_trace_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TRACE_SPEC");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_small_batch_qmm_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_SMALL_BATCH_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_m8_ksplit_qmm_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_M8_KSPLIT_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q4_batch_one_qmv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q4_batch_two_qmv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_BATCH_TWO_QMV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q4_batch_three_qmv_enabled() {
  const char* value = std::getenv("SGLANG_MLX_NATIVE_Q4_BATCH_THREE_QMV");
  return value != nullptr && std::string_view(value) == "1";
}

bool supports_q4_batch_three(const QLinear& linear, const array& x) {
  if (!linear.valid || linear.bits != 4 || linear.group_size != 64 ||
      x.ndim() != 3 || x.shape()[0] != 1 || x.shape()[1] != 3 ||
      x.dtype() != activation_dtype() || linear.w.ndim() != 2 ||
      linear.w.dtype() != mx::uint32 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype()) {
    return false;
  }
  const int k = x.shape()[2];
  const int n = linear.w.shape()[0];
  return k > 0 && k % 512 == 0 && n > 0 && n % 16 == 0 &&
      linear.w.shape()[1] == k / 8 &&
      linear.scales.shape() == mx::Shape{n, k / 64} &&
      linear.biases.shape() == linear.scales.shape();
}

bool native_q4_fused_swiglu_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q4_fused_raw_params_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_FUSED_RAW_PARAMS");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_evict_q4_raw_params_at_mtp_growth_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_EVICT_Q4_RAW_PARAMS_AT_MTP_GROWTH");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

int native_post_growth_mtp_prefill_chunk_size() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_POST_GROWTH_MTP_PREFILL_CHUNK_SIZE");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "0") {
    return 0;
  }
  int chunk_size = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), chunk_size);
  constexpr int kMaximumChunkSize = 1024;
  if (error != std::errc() || end != text.data() + text.size() ||
      chunk_size < 1 || chunk_size > kMaximumChunkSize) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_POST_GROWTH_MTP_PREFILL_CHUNK_SIZE must be 0 or "
        "an integer from 1 through 1024");
  }
  return chunk_size;
}

bool native_q4_fused_swiglu_batch_two_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU_BATCH_TWO");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q4_fused_swiglu_batch_two_scalar_inputs_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU_BATCH_TWO_SCALAR_INPUTS");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_qmm_trace_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TRACE_QMM");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_async_verify_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_ASYNC_VERIFY");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_verify_fused_norms_enabled() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_VERIFY_FUSED_NORMS");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_two_token_causal_conv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TWO_TOKEN_CAUSAL_CONV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_dflash_tape_commit_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DFLASH_TAPE_COMMIT");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q5_batch_one_qmv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q5_batch_two_qmv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q5_BATCH_TWO_QMV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q5_multirow_qmv_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_Q5_MULTIROW_QMV");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_quantized_embedding_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_QUANTIZED_EMBEDDING");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q8_tiled_attention_enabled() {

  const char* value = std::getenv("SGLANG_MLX_NATIVE_Q8_TILED_ATTENTION");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_q8_split_verify_enabled() {
  const char* value = std::getenv("SGLANG_MLX_NATIVE_Q8_SPLIT_VERIFY");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_fixed_prefill_attention_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_FIXED_PREFILL_ATTENTION");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_append_only_attention_snapshot_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_APPEND_ONLY_ATTN_SNAPSHOT");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_serialize_attention_cache_growth_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_SERIALIZE_ATTN_CACHE_GROWTH");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

int native_attention_cache_reserve_capacity() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_ATTN_CACHE_RESERVE");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  int capacity = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), capacity);
  constexpr int kMinimumCapacity = 256;
  constexpr int kMaximumCapacity = 262144;
  constexpr int kCacheAlignment = 64;
  if (error != std::errc() || end != text.data() + text.size() ||
      (capacity != 0 &&
       (capacity < kMinimumCapacity || capacity > kMaximumCapacity ||
        capacity % kCacheAlignment != 0))) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_ATTN_CACHE_RESERVE must be 0 or a multiple of 64 "
        "from 256 through 262144");
  }
  return capacity;
}

int native_attention_cache_bits() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_ATTN_CACHE_BITS");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "0") {
    return 0;
  }
  if (std::string_view(value) == "8") {
    return 8;
  }
  throw std::runtime_error(
      "SGLANG_MLX_NATIVE_ATTN_CACHE_BITS must be 0 or 8");
}

int native_target_only_prefill_chunk_size() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  int chunk_size = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), chunk_size);
  if (error != std::errc() || end != text.data() + text.size() ||
      chunk_size < 1 || chunk_size > 8192) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE must be an integer "
        "from 1 through 8192");
  }
  return chunk_size;
}

int native_mtp_block_size() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MTP_BLOCK_SIZE");
  if (value == nullptr || *value == '\0') {
    return 3;
  }
  int block_size = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), block_size);
  if (error != std::errc() || end != text.data() + text.size() ||
      block_size < 2 || block_size > 8) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_MTP_BLOCK_SIZE must be an integer from 2 through "
        "8");
  }
  return block_size;
}

bool native_mtp_post_norm_seed_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MTP_POST_NORM_SEED");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_mtp_committed_history_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MTP_COMMITTED_HISTORY");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

bool native_mtp_prompt_cache_enabled() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MTP_PROMPT_CACHE");
  return value != nullptr && std::string_view(value) != "0" &&
      std::string_view(value) != "false";
}

float native_dflash_selector_temperature() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DFLASH_SELECTOR_TEMPERATURE");
  if (value == nullptr || *value == '\0') {
    return 1.0f;
  }
  float temperature = 0.0f;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), temperature);
  if (error != std::errc() || end != text.data() + text.size() ||
      !std::isfinite(temperature) || temperature <= 0.0f) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DFLASH_SELECTOR_TEMPERATURE must be a positive "
        "finite number");
  }
  return temperature;
}

float native_dflash_mean_q_threshold() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DFLASH_MEAN_Q_THRESHOLD");
  if (value == nullptr || *value == '\0') {
    return 0.0f;
  }
  float threshold = 0.0f;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), threshold);
  if (error != std::errc() || end != text.data() + text.size() ||
      !std::isfinite(threshold) || threshold <= 0.0f || threshold > 1.0f) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DFLASH_MEAN_Q_THRESHOLD must be a positive "
        "finite number no greater than one");
  }
  return threshold;
}

int native_dspark_verify_draft_tokens() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS");
  if (value == nullptr || *value == '\0') {
    return 7;
  }
  int count = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (error != std::errc() || end != text.data() + text.size() || count < 1 ||
      count > 7) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS must be an integer "
        "from 1 through 7");
  }
  return count;
}

float native_dspark_confidence_cost_ratio() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO");
  if (value == nullptr || *value == '\0') {
    return 0.0f;
  }
  float ratio = 0.0f;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), ratio);
  if (error != std::errc() || end != text.data() + text.size() ||
      !std::isfinite(ratio) || ratio <= 0.0f) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO must be a positive "
        "finite number");
  }
  return ratio;
}

int native_dspark_bypass_refills() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_DSPARK_BYPASS_REFILLS");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  int count = 0;
  const std::string_view text(value);
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (error != std::errc() || end != text.data() + text.size() || count < 0 ||
      count > 1024) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_DSPARK_BYPASS_REFILLS must be an integer from 0 "
        "through 1024");
  }
  return count;
}

enum class LinearAttnOverrideScope {
  kOutProjection,
  kQkvProjection,
  kZProjection,
  kQkvAndZProjections,
  kAllProjections,
};

LinearAttnOverrideScope linear_attn_override_scope() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE");
  if (value == nullptr || *value == '\0' || std::string_view(value) == "all") {
    return LinearAttnOverrideScope::kAllProjections;
  }
  if (std::string_view(value) == "qkv") {
    return LinearAttnOverrideScope::kQkvProjection;
  }
  if (std::string_view(value) == "z") {
    return LinearAttnOverrideScope::kZProjection;
  }
  if (std::string_view(value) == "qkvz") {
    return LinearAttnOverrideScope::kQkvAndZProjections;
  }
  throw std::runtime_error(
      "SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE must be all, qkv, z, or qkvz");
}

size_t linear_attn_projection_count(LinearAttnOverrideScope scope) {
  switch (scope) {
    case LinearAttnOverrideScope::kOutProjection:
    case LinearAttnOverrideScope::kQkvProjection:
    case LinearAttnOverrideScope::kZProjection:
      return 1;
    case LinearAttnOverrideScope::kQkvAndZProjections:
      return 2;
    case LinearAttnOverrideScope::kAllProjections:
      return 5;
  }
  throw std::runtime_error("invalid linear-attention override scope");
}

bool is_linear_attn_override_tensor(
    std::string_view key, LinearAttnOverrideScope scope) {
  const bool is_quant_tensor = key.ends_with(".weight") ||
      key.ends_with(".scales") || key.ends_with(".biases");
  const bool is_out_projection =
      key.find(".linear_attn.out_proj.") != std::string_view::npos;
  const bool is_qkv_projection =
      key.find(".linear_attn.in_proj_qkv.") != std::string_view::npos;
  const bool is_z_projection =
      key.find(".linear_attn.in_proj_z.") != std::string_view::npos;
  if (!is_quant_tensor) {
    return false;
  }
  switch (scope) {
    case LinearAttnOverrideScope::kOutProjection:
      return is_out_projection;
    case LinearAttnOverrideScope::kQkvProjection:
      return is_qkv_projection;
    case LinearAttnOverrideScope::kZProjection:
      return is_z_projection;
    case LinearAttnOverrideScope::kQkvAndZProjections:
      return is_qkv_projection || is_z_projection;
    case LinearAttnOverrideScope::kAllProjections:
      return is_out_projection || is_qkv_projection || is_z_projection ||
          key.find(".linear_attn.in_proj_a.") != std::string_view::npos ||
          key.find(".linear_attn.in_proj_b.") != std::string_view::npos;
  }
  throw std::runtime_error("invalid linear-attention override scope");
}

size_t apply_linear_attn_override(
    std::unordered_map<std::string, array>& weights,
    const std::string& model_dir,
    LinearAttnOverrideScope scope) {
  DIR* dir = opendir(model_dir.c_str());
  if (dir == nullptr) {
    throw std::runtime_error(
        "cannot open linear-attention override dir " + model_dir);
  }
  std::vector<std::string> shards;
  while (dirent* ent = readdir(dir)) {
    std::string name = ent->d_name;
    if (name.size() >= 12 &&
        name.substr(name.size() - 12) == ".safetensors") {
      shards.push_back(model_dir + "/" + name);
    }
  }
  closedir(dir);

  size_t replaced = 0;
  for (const std::string& shard : shards) {
    auto loaded = load_activation_safetensors(shard);
    for (auto& kv : loaded.first) {
      if (!is_linear_attn_override_tensor(kv.first, scope)) {
        continue;
      }
      auto target = weights.find(kv.first);
      if (target == weights.end()) {
        throw std::runtime_error(
            "linear-attention override has unknown weight " + kv.first);
      }
      target->second = std::move(kv.second);
      ++replaced;
    }
  }
  return replaced;
}

std::uint64_t native_sampling_seed() {
  const char* const value = std::getenv("SGLANG_MLX_NATIVE_SAMPLING_SEED");
  if (value == nullptr || *value == '\0') {
    return 67396869;
  }
  const std::string_view text(value);
  std::uint64_t seed = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), seed);
  if (error != std::errc() || end != text.data() + text.size()) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_SAMPLING_SEED must be an unsigned integer");
  }
  return seed;
}

int native_reasoning_token_limit() {
  const char* const value =
      std::getenv("SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS");
  if (value == nullptr || *value == '\0') {
    return 0;
  }
  const std::string_view text(value);
  int limit = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), limit);
  if (error != std::errc() || end != text.data() + text.size() || limit <= 0) {
    throw std::runtime_error(
        "SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS must be a positive integer");
  }
  return limit;
}

std::string layer_key(int i, const std::string& rest) {
  return "language_model.model.layers." + std::to_string(i) + rest;
}

array last_token(const array& hidden) {
  auto shape = hidden.shape();
  int seq = static_cast<int>(shape[1]);
  return squeeze(slice(hidden, {0, seq - 1, 0}, {1, seq, shape[2]}), 1);
}

constexpr const char* kGatedDeltaHeader = R"(
#define SGLANG_STORE_GATED_DELTA(value, time_index)
)";

constexpr const char* kGatedDeltaTapeHeader = R"(
#define SGLANG_STORE_GATED_DELTA(value, time_index) \
  if (thread_index_in_simdgroup == 0) { \
    delta_out[(((b_idx * T + (time_index)) * Hv + hv_idx) * Dv) + dv_idx] = \
        (value); \
  }
)";

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
            SGLANG_STORE_GATED_DELTA(delta, t);

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

constexpr const char* kGatedDeltaCommitSource = R"(
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        constexpr int ValuesPerThread = Dk / 32;

        auto input_state = state_in + (n * Dv + dv_idx) * Dk;
        auto output_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[ValuesPerThread];
#pragma unroll
        for (int index = 0; index < ValuesPerThread; ++index) {
          const int dimension = ValuesPerThread * dk_idx + index;
          state[index] = static_cast<float>(input_state[dimension]);
        }

        for (int time_index = 0; time_index < token_count; ++time_index) {
          const float decay_value = decay[
              (b_idx * T + time_index) * Hv + hv_idx];
          const float delta_value = delta[
              ((b_idx * T + time_index) * Hv + hv_idx) * Dv + dv_idx];
          auto key = keys +
              (b_idx * T + time_index) * Hk * Dk + hk_idx * Dk;
#pragma unroll
          for (int index = 0; index < ValuesPerThread; ++index) {
            const int dimension = ValuesPerThread * dk_idx + index;
            state[index] = state[index] * decay_value;
            state[index] =
                state[index] + static_cast<float>(key[dimension]) * delta_value;
          }
        }

#pragma unroll
        for (int index = 0; index < ValuesPerThread; ++index) {
          const int dimension = ValuesPerThread * dk_idx + index;
          output_state[dimension] = state[index];
        }
)";

constexpr const char* kCausalConvDecodeSiluSource = R"(
        auto channel = thread_position_in_grid.x;
        auto batch = thread_position_in_grid.y;
        auto state_base = state + batch * (K - 1) * D + channel;
        auto qkv_base = qkv + batch * D + channel;
        auto weight_base = weight + channel * K;

        float acc = 0.0f;
        for (int tap = 0; tap < K - 1; ++tap) {
          acc += static_cast<float>(state_base[tap * D]) * weight_base[tap];
        }
        acc += static_cast<float>(qkv_base[0]) * weight_base[K - 1];
        InT conv_value = static_cast<InT>(acc);
        auto sigmoid_low =
            #if SGLANG_NATIVE_FP16
            1 / (1 + metal::exp(metal::abs(conv_value)));
#else
            1 / (1 + metal::precise::exp(metal::abs(conv_value)));
#endif
        InT sigmoid_value =
            (conv_value < 0) ? sigmoid_low : 1 - sigmoid_low;
        conv_out[batch * D + channel] =
            static_cast<InT>(conv_value * sigmoid_value);

        auto next_state_base = next_state + batch * (K - 1) * D + channel;
        for (int tap = 0; tap < K - 2; ++tap) {
          next_state_base[tap * D] = state_base[(tap + 1) * D];
        }
        next_state_base[(K - 2) * D] = qkv_base[0];
)";

constexpr const char* kCausalConvTwoTokenSiluSource = R"(
        auto channel = thread_position_in_grid.x;
        auto batch = thread_position_in_grid.y;
        auto state_base = state + batch * (K - 1) * D + channel;
        auto qkv_base = qkv + batch * 2 * D + channel;
        auto weight_base = weight + channel * K;

        float first_acc = 0.0f;
        float second_acc = 0.0f;
        for (int tap = 0; tap < K - 1; ++tap) {
          first_acc +=
              static_cast<float>(state_base[tap * D]) * weight_base[tap];
          const InT second_input = tap + 1 < K - 1
              ? state_base[(tap + 1) * D]
              : qkv_base[0];
          second_acc +=
              static_cast<float>(second_input) * weight_base[tap];
        }
        first_acc +=
            static_cast<float>(qkv_base[0]) * weight_base[K - 1];
        second_acc +=
            static_cast<float>(qkv_base[D]) * weight_base[K - 1];

        InT first_value = static_cast<InT>(first_acc);
        auto first_sigmoid_low =
            #if SGLANG_NATIVE_FP16
            1 / (1 + metal::exp(metal::abs(first_value)));
#else
            1 / (1 + metal::precise::exp(metal::abs(first_value)));
#endif
        InT first_sigmoid =
            first_value < 0 ? first_sigmoid_low : 1 - first_sigmoid_low;
        conv_out[batch * 2 * D + channel] =
            static_cast<InT>(first_value * first_sigmoid);

        InT second_value = static_cast<InT>(second_acc);
        auto second_sigmoid_low =
            #if SGLANG_NATIVE_FP16
            1 / (1 + metal::exp(metal::abs(second_value)));
#else
            1 / (1 + metal::precise::exp(metal::abs(second_value)));
#endif
        InT second_sigmoid =
            second_value < 0 ? second_sigmoid_low : 1 - second_sigmoid_low;
        conv_out[(batch * 2 + 1) * D + channel] =
            static_cast<InT>(second_value * second_sigmoid);

        auto next_state_base = next_state + batch * (K - 1) * D + channel;
        for (int tap = 0; tap < K - 1; ++tap) {
          const int source = tap + 2;
          next_state_base[tap * D] = source < K - 1
              ? state_base[source * D]
              : qkv_base[(source - (K - 1)) * D];
        }
)";

constexpr const char* kResidualRmsNormSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        InT cached[Iterations][N_READS];
        float acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              InT value = x[index] + residual[index];
              cached[iteration][i] = value;
              residual_out[index] = value;
              float value_f = static_cast<float>(value);
              acc += value_f * value_f;
            }
          }
        }

        acc = simd_sum(acc);
        if (simd_lane == 0) {
          local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          acc = simd_sum(
              simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
          if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              norm_out[index] = weight[base + i] * static_cast<InT>(
                  static_cast<float>(cached[iteration][i]) * local_inv_mean[0]);
            }
          }
        }
)";

constexpr const char* kGatedDeltaQkNormSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[2];
        threadgroup float local_q_sums[SIMD_SIZE];
        threadgroup float local_k_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        float cached_q[Iterations][N_READS];
        float cached_k[Iterations][N_READS];
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float q_value = static_cast<float>(q[index]);
              float k_value = static_cast<float>(k[index]);
              cached_q[iteration][i] = q_value;
              cached_k[iteration][i] = k_value;
              q_acc += q_value * q_value;
              k_acc += k_value * k_value;
            }
          }
        }

        q_acc = simd_sum(q_acc);
        k_acc = simd_sum(k_acc);
        float q_inv_mean;
        float k_inv_mean;
        if (SIMD_GROUPS == 1) {
          q_inv_mean = metal::precise::rsqrt(q_acc / D + eps);
          k_inv_mean = metal::precise::rsqrt(k_acc / D + eps);
        } else {
          if (simd_lane == 0) {
            local_q_sums[simd_group] = q_acc;
            local_k_sums[simd_group] = k_acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group == 0) {
            q_acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_q_sums[simd_lane] : 0.0f);
            k_acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_k_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
              local_inv_mean[0] = metal::precise::rsqrt(q_acc / D + eps);
              local_inv_mean[1] = metal::precise::rsqrt(k_acc / D + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          q_inv_mean = local_inv_mean[0];
          k_inv_mean = local_inv_mean[1];
        }

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              InT q_norm = static_cast<InT>(
                  cached_q[iteration][i] * q_inv_mean);
              InT k_norm = static_cast<InT>(
                  cached_k[iteration][i] * k_inv_mean);
              q_out[index] = q_scale * static_cast<float>(q_norm);
              k_out[index] = k_scale * static_cast<float>(k_norm);
            }
          }
        }
)";

constexpr const char* kFullAttnQkNormRopeSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;
        constexpr int HeadsPerBatch = Nq + Nk;
        auto batch = row / HeadsPerBatch;
        auto combined_head = row % HeadsPerBatch;
        bool is_q = combined_head < Nq;
        auto head = is_q ? combined_head : combined_head - Nq;
        const device InT* input = is_q
            ? qg + (batch * Nq + head) * (2 * D)
            : k + (batch * Nk + head) * D;
        const device InT* weight = is_q ? q_weight : k_weight;
        device InT* output = is_q
            ? q_out + (batch * Nq + head) * D
            : k_out + (batch * Nk + head) * D;

        float acc = 0.0f;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              float value = static_cast<float>(input[base + i]);
              acc += value * value;
            }
          }
        }

        acc = simd_sum(acc);
        if (simd_lane == 0) {
          local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
          acc = simd_sum(
              simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
          if (simd_lane == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS) {
          for (int i = 0; i < N_READS; ++i) {
            auto index = base + i;
            if (index < D && index >= RopeDims) {
              output[index] = weight[index] * static_cast<InT>(
                  static_cast<float>(input[index]) * local_inv_mean[0]);
            }
          }
        }

        constexpr int RopePairs = RopeDims / 2;
        if (lid < RopePairs) {
          auto index_1 = lid;
          auto index_2 = lid + RopePairs;
          InT x1_norm = weight[index_1] * static_cast<InT>(
              static_cast<float>(input[index_1]) * local_inv_mean[0]);
          InT x2_norm = weight[index_2] * static_cast<InT>(
              static_cast<float>(input[index_2]) * local_inv_mean[0]);
          float d = static_cast<float>(lid) / static_cast<float>(RopePairs);
          float inv_freq = metal::exp2(-d * log2_base);
          float L = 1.0f * static_cast<float>(rope_offset);
          float theta = L * inv_freq;
          float costheta = metal::fast::cos(theta);
          float sintheta = metal::fast::sin(theta);
          float x1 = static_cast<float>(x1_norm);
          float x2 = static_cast<float>(x2_norm);
          output[index_1] = static_cast<InT>(
              x1 * costheta - x2 * sintheta);
          output[index_2] = static_cast<InT>(
              x1 * sintheta + x2 * costheta);
        }
)";

constexpr const char* kFixedPrefillAttentionHeader = R"(
#include <metal_simdgroup>

inline float sglang_attention_simd_max_16(float value) {
  value = max(value, simd_shuffle_xor(value, 8));
  value = max(value, simd_shuffle_xor(value, 4));
  value = max(value, simd_shuffle_xor(value, 2));
  return max(value, simd_shuffle_xor(value, 1));
}

inline float sglang_attention_simd_sum_16(float value) {
  value += simd_shuffle_xor(value, 8);
  value += simd_shuffle_xor(value, 4);
  value += simd_shuffle_xor(value, 2);
  return value + simd_shuffle_xor(value, 1);
}

inline Activation sglang_attention_dequantize_q8(
    device const uint* packed,
    device const Activation* scales,
    device const Activation* biases,
    uint kv_head,
    uint token,
    uint dimension,
    uint capacity) {
  constexpr uint HeadDim = 256;
  constexpr uint PackedValuesPerWord = 4;
  constexpr uint GroupSize = 64;
  const ulong row = ulong(kv_head) * ulong(capacity) + token;
  const uint word = packed[
      row * (HeadDim / PackedValuesPerWord) +
      dimension / PackedValuesPerWord];
  const uint shift = (dimension % PackedValuesPerWord) * 8;
  const uint quantized = (word >> shift) & 0xffu;
  const ulong parameter =
      row * (HeadDim / GroupSize) + dimension / GroupSize;
  return static_cast<Activation>(
      static_cast<float>(quantized) * static_cast<float>(scales[parameter]) +
      static_cast<float>(biases[parameter]));
}
)";

// This bounded Q8/C64 online-softmax geometry is the contiguous-cache MLX
// form of SGLang's existing extend_gqa_bf16_tiled_256 Metal kernel. That
// kernel adapts llama.cpp's kernel_flash_attn_ext_impl at commit
// 749f688fcaa4c472ec034b08cb8a907c45cfaa02; see THIRDPARTYNOTICES.txt.
constexpr const char* kFixedPrefillAttentionSource = R"(
        constexpr ushort QueryTile = 8;
        constexpr ushort KeyTile = 64;
        constexpr ushort HeadsPerKv = 6;
        constexpr ushort HeadDim = 256;

        threadgroup float shared_query[QueryTile * HeadDim];
        threadgroup float shared_output[QueryTile * HeadDim];
        threadgroup float shared_scores[QueryTile * KeyTile];
        threadgroup float shared_stats[QueryTile * 2];

        const ushort tid = thread_index_in_threadgroup;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint kv_head = threadgroup_position_in_grid.y;
        const uint attention_row_start =
            threadgroup_position_in_grid.x * QueryTile;
        const uint query_count = uint(query_tokens);
        const uint prefix = uint(prefix_length);
        const uint capacity = uint(cache_capacity);
        const uint active_length = uint(active_cache_length);
        const uint attention_rows = query_count * HeadsPerKv;
        const uint last_attention_row = min(
            attention_rows - 1, attention_row_start + QueryTile - 1);
        const uint group_kv_length = min(
            active_length,
            prefix + last_attention_row / HeadsPerKv + 1);

        for (uint index = tid; index < QueryTile * HeadDim; index += 128) {
          const uint row = index / HeadDim;
          const uint dimension = index - row * HeadDim;
          const uint attention_row = attention_row_start + row;
          float value = 0.0f;
          if (attention_row < attention_rows) {
            const uint query_token = attention_row / HeadsPerKv;
            const uint query_head =
                kv_head * HeadsPerKv + attention_row % HeadsPerKv;
            value = static_cast<float>(query[
                (query_head * query_count + query_token) * HeadDim +
                dimension]);
          }
          shared_query[index] = value;
          shared_output[index] = 0.0f;
        }
        if (tid < QueryTile) {
          shared_stats[2 * tid] = -INFINITY;
          shared_stats[2 * tid + 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_start = 0; key_start < group_kv_length;
             key_start += KeyTile) {
          simdgroup_float8x8 score_left =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
          simdgroup_float8x8 score_right =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
          for (ushort dimension_start = 0; dimension_start < HeadDim;
               dimension_start += 16) {
            simdgroup_float8x8 query_low;
            simdgroup_float8x8 query_high;
            simdgroup_load(
                query_low,
                shared_query + dimension_start,
                HeadDim,
                0,
                false);
            simdgroup_load(
                query_high,
                shared_query + dimension_start + 8,
                HeadDim,
                0,
                false);

#pragma unroll
            for (ushort key_half = 0; key_half < 2; ++key_half) {
              const uint key_base =
                  (kv_head * capacity + key_start + simd_id * 16 +
                   key_half * 8) * HeadDim + dimension_start;
              const device Activation* key_run = key_cache + key_base;
              ActivationMatrix key_low;
              ActivationMatrix key_high;
              simdgroup_barrier(mem_flags::mem_none);
              simdgroup_load(key_low, key_run, HeadDim, 0, true);
              simdgroup_load(key_high, key_run + 8, HeadDim, 0, true);
              simdgroup_barrier(mem_flags::mem_none);
              if (key_half == 0) {
                simdgroup_multiply_accumulate(
                    score_left, query_low, key_low, score_left);
                simdgroup_multiply_accumulate(
                    score_left, query_high, key_high, score_left);
              } else {
                simdgroup_multiply_accumulate(
                    score_right, query_low, key_low, score_right);
                simdgroup_multiply_accumulate(
                    score_right, query_high, key_high, score_right);
              }
            }
          }

          simdgroup_store(
              score_left,
              shared_scores + simd_id * 16,
              KeyTile,
              0,
              false);
          simdgroup_store(
              score_right,
              shared_scores + simd_id * 16 + 8,
              KeyTile,
              0,
              false);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const ushort softmax_row = tid / 16;
          const ushort softmax_lane = tid & 15;
          const uint attention_row = attention_row_start + softmax_row;
          const bool row_valid = attention_row < attention_rows;
          const uint query_token = row_valid
              ? attention_row / HeadsPerKv
              : 0;
          const uint causal_limit = row_valid
              ? min(active_length, prefix + query_token + 1)
              : 0;

          float row_max = -INFINITY;
          float scaled_scores[4];
#pragma unroll
          for (ushort index = 0; index < 4; ++index) {
            const ushort key_column = softmax_lane + 16 * index;
            const uint logical_token = key_start + key_column;
            const float score = row_valid && logical_token < causal_limit
                ? shared_scores[softmax_row * KeyTile + key_column] * 0.0625f
                : -INFINITY;
            scaled_scores[index] = score;
            row_max = max(row_max, score);
          }
          row_max = sglang_attention_simd_max_16(row_max);

          const float old_max = shared_stats[2 * softmax_row];
          const float old_sum = shared_stats[2 * softmax_row + 1];
          const bool block_valid = row_max != -INFINITY;
          const float next_max = block_valid ? max(old_max, row_max) : old_max;
          const float old_scale = old_sum == 0.0f
              ? 0.0f
              : (block_valid ? exp(old_max - next_max) : 1.0f);
          float block_sum = 0.0f;
#pragma unroll
          for (ushort index = 0; index < 4; ++index) {
            const ushort key_column = softmax_lane + 16 * index;
            const float probability = scaled_scores[index] == -INFINITY
                ? 0.0f
                : exp(scaled_scores[index] - next_max);
            shared_scores[softmax_row * KeyTile + key_column] = probability;
            block_sum += probability;
          }
          block_sum = sglang_attention_simd_sum_16(block_sum);
          for (ushort dimension = softmax_lane; dimension < HeadDim;
               dimension += 16) {
            shared_output[softmax_row * HeadDim + dimension] *= old_scale;
          }
          if (softmax_lane == 0 && row_valid) {
            shared_stats[2 * softmax_row] = next_max;
            shared_stats[2 * softmax_row + 1] =
                old_sum * old_scale + block_sum;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          simdgroup_float8x8 output_fragments[8];
#pragma unroll
          for (ushort output_block = 0; output_block < 8; ++output_block) {
            simdgroup_load(
                output_fragments[output_block],
                shared_output + simd_id * 64 + output_block * 8,
                HeadDim,
                0,
                false);
          }
          for (ushort key_block = 0; key_block < KeyTile; key_block += 8) {
            simdgroup_float8x8 probability_fragment;
            simdgroup_load(
                probability_fragment,
                shared_scores + key_block,
                KeyTile,
                0,
                false);
            const uint value_base =
                (kv_head * capacity + key_start + key_block) * HeadDim +
                simd_id * 64;
#pragma unroll
            for (ushort output_block = 0; output_block < 8; ++output_block) {
              ActivationMatrix value_fragment;
              simdgroup_barrier(mem_flags::mem_none);
              simdgroup_load(
                  value_fragment,
                  value_cache + value_base + output_block * 8,
                  HeadDim,
                  0,
                  false);
              simdgroup_barrier(mem_flags::mem_none);
              simdgroup_multiply_accumulate(
                  output_fragments[output_block],
                  probability_fragment,
                  value_fragment,
                  output_fragments[output_block]);
            }
          }
#pragma unroll
          for (ushort output_block = 0; output_block < 8; ++output_block) {
            simdgroup_store(
                output_fragments[output_block],
                shared_output + simd_id * 64 + output_block * 8,
                HeadDim,
                0,
                false);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint index = tid; index < QueryTile * HeadDim; index += 128) {
          const uint row = index / HeadDim;
          const uint dimension = index - row * HeadDim;
          const uint attention_row = attention_row_start + row;
          if (attention_row < attention_rows) {
            const uint query_token = attention_row / HeadsPerKv;
            const uint query_head =
                kv_head * HeadsPerKv + attention_row % HeadsPerKv;
            const float denominator = shared_stats[2 * row + 1];
            output[(query_head * query_count + query_token) * HeadDim +
                   dimension] = static_cast<Activation>(
                denominator == 0.0f
                    ? 0.0f
                    : shared_output[index] / denominator);
          }
        }
)";

// Affine-Q8/G64 cache variant of the fixed-memory attention kernel above.
// Only K/V tile loading changes: each tile is dequantized into bounded
// threadgroup storage before the same matrix multiply, online softmax, causal
// masking, and BF16 output path execute.
constexpr const char* kFixedQ8AttentionSource = R"(
        constexpr ushort QueryTile = 8;
        constexpr ushort KeyTile = 64;
        constexpr ushort HeadsPerKv = 6;
        constexpr ushort HeadDim = 256;

        threadgroup float shared_query[QueryTile * HeadDim];
        threadgroup float shared_output[QueryTile * HeadDim];
        threadgroup float shared_scores[QueryTile * KeyTile];
        threadgroup float shared_stats[QueryTile * 2];
        threadgroup Activation shared_dequant[4 * 8 * 16];

        const ushort tid = thread_index_in_threadgroup;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint kv_head = threadgroup_position_in_grid.y;
        const uint query_count = uint(query_tokens);
        const uint prefix = uint(prefix_length);
        const uint capacity = uint(cache_capacity);
        const uint active_length = uint(active_cache_length);
        const uint requested_key_splits = uint(key_splits);
        const bool split_decode = requested_key_splits > 1;
        const uint split_index = split_decode
            ? threadgroup_position_in_grid.x % requested_key_splits : 0;
        const uint query_tile = split_decode
            ? threadgroup_position_in_grid.x / requested_key_splits
            : threadgroup_position_in_grid.x;
        const uint attention_row_start = query_tile * QueryTile;
        const uint attention_rows = query_count * HeadsPerKv;
        const uint last_attention_row = min(
            attention_rows - 1, attention_row_start + QueryTile - 1);
        const uint group_kv_length = min(
            active_length,
            prefix + last_attention_row / HeadsPerKv + 1);
        const uint active_key_splits = split_decode
            ? min(
                  requested_key_splits,
                  max(1u, (active_length + 1023) / 1024))
            : 1;
        if (split_decode &&
            split_index >= active_key_splits) {
          return;
        }
        const uint key_tiles =
            (active_length + KeyTile - 1) / KeyTile;
        const uint tiles_per_split =
            (key_tiles + active_key_splits - 1) / active_key_splits;
        const uint first_key_tile = split_decode
            ? split_index * tiles_per_split
            : 0;
        const uint last_key_tile = split_decode
            ? min(key_tiles, first_key_tile + tiles_per_split)
            : key_tiles;
        const uint first_key = first_key_tile * KeyTile;
        const uint last_key = min(
            group_kv_length, last_key_tile * KeyTile);

        for (uint index = tid; index < QueryTile * HeadDim; index += 128) {
          const uint row = index / HeadDim;
          const uint dimension = index - row * HeadDim;
          const uint attention_row = attention_row_start + row;
          float value = 0.0f;
          if (attention_row < attention_rows) {
            const uint query_token = attention_row / HeadsPerKv;
            const uint query_head =
                kv_head * HeadsPerKv + attention_row % HeadsPerKv;
            value = static_cast<float>(query[
                (query_head * query_count + query_token) * HeadDim +
                dimension]);
          }
          shared_query[index] = value;
          shared_output[index] = 0.0f;
        }
        if (tid < QueryTile) {
          shared_stats[2 * tid] = -INFINITY;
          shared_stats[2 * tid + 1] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key_start = first_key; key_start < last_key;
             key_start += KeyTile) {
          simdgroup_float8x8 score_left =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
          simdgroup_float8x8 score_right =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
          threadgroup Activation* key_stage =
              shared_dequant + simd_id * 8 * 16;
          for (ushort dimension_start = 0; dimension_start < HeadDim;
               dimension_start += 16) {
            simdgroup_float8x8 query_low;
            simdgroup_float8x8 query_high;
            simdgroup_load(
                query_low,
                shared_query + dimension_start,
                HeadDim,
                0,
                false);
            simdgroup_load(
                query_high,
                shared_query + dimension_start + 8,
                HeadDim,
                0,
                false);

#pragma unroll
            for (ushort key_half = 0; key_half < 2; ++key_half) {
              const ushort key_row = lane / 4;
              const ushort dimension_quad = lane & 3;
              const uint token = key_start + simd_id * 16 +
                  key_half * 8 + key_row;
#pragma unroll
              for (ushort element = 0; element < 4; ++element) {
                const uint dimension = dimension_start +
                    dimension_quad * 4 + element;
                key_stage[key_row * 16 + dimension_quad * 4 + element] =
                    token < last_key ? sglang_attention_dequantize_q8(
                        key_cache,
                        key_scales,
                        key_biases,
                        kv_head,
                        token,
                        dimension,
                        capacity) : static_cast<Activation>(0.0f);
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              ActivationMatrix key_low;
              ActivationMatrix key_high;
              simdgroup_load(key_low, key_stage, 16, 0, true);
              simdgroup_load(key_high, key_stage + 8, 16, 0, true);
              simdgroup_barrier(mem_flags::mem_threadgroup);
              if (key_half == 0) {
                simdgroup_multiply_accumulate(
                    score_left, query_low, key_low, score_left);
                simdgroup_multiply_accumulate(
                    score_left, query_high, key_high, score_left);
              } else {
                simdgroup_multiply_accumulate(
                    score_right, query_low, key_low, score_right);
                simdgroup_multiply_accumulate(
                    score_right, query_high, key_high, score_right);
              }
            }
          }

          simdgroup_store(
              score_left,
              shared_scores + simd_id * 16,
              KeyTile,
              0,
              false);
          simdgroup_store(
              score_right,
              shared_scores + simd_id * 16 + 8,
              KeyTile,
              0,
              false);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          const ushort softmax_row = tid / 16;
          const ushort softmax_lane = tid & 15;
          const uint attention_row = attention_row_start + softmax_row;
          const bool row_valid = attention_row < attention_rows;
          const uint query_token = row_valid
              ? attention_row / HeadsPerKv
              : 0;
          const uint causal_limit = row_valid
              ? min(active_length, prefix + query_token + 1)
              : 0;

          float row_max = -INFINITY;
          float scaled_scores[4];
#pragma unroll
          for (ushort index = 0; index < 4; ++index) {
            const ushort key_column = softmax_lane + 16 * index;
            const uint logical_token = key_start + key_column;
            const float score = row_valid && logical_token < causal_limit
                ? shared_scores[softmax_row * KeyTile + key_column] * 0.0625f
                : -INFINITY;
            scaled_scores[index] = score;
            row_max = max(row_max, score);
          }
          row_max = sglang_attention_simd_max_16(row_max);

          const float old_max = shared_stats[2 * softmax_row];
          const float old_sum = shared_stats[2 * softmax_row + 1];
          const bool block_valid = row_max != -INFINITY;
          const float next_max = block_valid ? max(old_max, row_max) : old_max;
          const float old_scale = old_sum == 0.0f
              ? 0.0f
              : (block_valid ? exp(old_max - next_max) : 1.0f);
          float block_sum = 0.0f;
#pragma unroll
          for (ushort index = 0; index < 4; ++index) {
            const ushort key_column = softmax_lane + 16 * index;
            const float probability = scaled_scores[index] == -INFINITY
                ? 0.0f
                : exp(scaled_scores[index] - next_max);
            shared_scores[softmax_row * KeyTile + key_column] = probability;
            block_sum += probability;
          }
          block_sum = sglang_attention_simd_sum_16(block_sum);
          for (ushort dimension = softmax_lane; dimension < HeadDim;
               dimension += 16) {
            shared_output[softmax_row * HeadDim + dimension] *= old_scale;
          }
          if (softmax_lane == 0 && row_valid) {
            shared_stats[2 * softmax_row] = next_max;
            shared_stats[2 * softmax_row + 1] =
                old_sum * old_scale + block_sum;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          simdgroup_float8x8 output_fragments[8];
#pragma unroll
          for (ushort output_block = 0; output_block < 8; ++output_block) {
            simdgroup_load(
                output_fragments[output_block],
                shared_output + simd_id * 64 + output_block * 8,
                HeadDim,
                0,
                false);
          }
          threadgroup Activation* value_stage =
              shared_dequant + simd_id * 8 * 16;
          for (ushort key_block = 0; key_block < KeyTile; key_block += 8) {
            simdgroup_float8x8 probability_fragment;
            simdgroup_load(
                probability_fragment,
                shared_scores + key_block,
                KeyTile,
                0,
                false);
#pragma unroll
            for (ushort output_block = 0; output_block < 8; ++output_block) {
              for (ushort index = lane; index < 8 * 8; index += 32) {
                const ushort key_row = index / 8;
                const ushort output_column = index & 7;
                const uint token = key_start + key_block + key_row;
                const uint dimension = simd_id * 64 +
                    output_block * 8 + output_column;
                value_stage[index] = token < last_key ? sglang_attention_dequantize_q8(
                    value_cache,
                    value_scales,
                    value_biases,
                    kv_head,
                    token,
                    dimension,
                    capacity) : static_cast<Activation>(0.0f);
              }
              simdgroup_barrier(mem_flags::mem_threadgroup);
              ActivationMatrix value_fragment;
              simdgroup_load(
                  value_fragment, value_stage, 8, 0, false);
              simdgroup_barrier(mem_flags::mem_threadgroup);
              simdgroup_multiply_accumulate(
                  output_fragments[output_block],
                  probability_fragment,
                  value_fragment,
                  output_fragments[output_block]);
            }
          }
#pragma unroll
          for (ushort output_block = 0; output_block < 8; ++output_block) {
            simdgroup_store(
                output_fragments[output_block],
                shared_output + simd_id * 64 + output_block * 8,
                HeadDim,
                0,
                false);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (split_decode) {
          for (uint index = tid; index < QueryTile * HeadDim; index += 128) {
            const uint row = index / HeadDim;
            const uint dimension = index - row * HeadDim;
            const uint attention_row = attention_row_start + row;
            if (attention_row < attention_rows) {
              const uint query_token = attention_row / HeadsPerKv;
              const uint query_head = kv_head * HeadsPerKv + attention_row % HeadsPerKv;
              const ulong query_row = ulong(query_head) * query_count + query_token;
              const ulong partial_base =
                  (ulong(split_index) * 24 * query_count + query_row) * (HeadDim + 2);
              output[partial_base + dimension] = shared_output[index];
            }
          }
          if (tid < QueryTile && attention_row_start + tid < attention_rows) {
            const uint attention_row = attention_row_start + tid;
            const uint query_token = attention_row / HeadsPerKv;
            const uint query_head = kv_head * HeadsPerKv + attention_row % HeadsPerKv;
            const ulong query_row = ulong(query_head) * query_count + query_token;
            const ulong partial_base =
                (ulong(split_index) * 24 * query_count + query_row) * (HeadDim + 2);
            output[partial_base + HeadDim] = shared_stats[2 * tid];
            output[partial_base + HeadDim + 1] =
                shared_stats[2 * tid + 1];
          }
          return;
        }

        for (uint index = tid; index < QueryTile * HeadDim; index += 128) {
          const uint row = index / HeadDim;
          const uint dimension = index - row * HeadDim;
          const uint attention_row = attention_row_start + row;
          if (attention_row < attention_rows) {
            const uint query_token = attention_row / HeadsPerKv;
            const uint query_head =
                kv_head * HeadsPerKv + attention_row % HeadsPerKv;
            const float denominator = shared_stats[2 * row + 1];
            output[(query_head * query_count + query_token) * HeadDim +
                   dimension] = denominator == 0.0f
                ? 0.0f
                : shared_output[index] / denominator;
          }
        }
)";

constexpr const char* kTiledQ8AttentionSource = R"(
  constexpr uint QTile = 8, KTile = 32, D = 256, HeadsPerKv = 6;
  threadgroup Activation qtile[QTile * D];
  threadgroup Activation kvtile[KTile * D];
  threadgroup float scores[QTile * KTile];
  threadgroup float accumulated[QTile * D];
  threadgroup float stats[QTile * 2];
  const uint tid = thread_index_in_threadgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint kv_head = threadgroup_position_in_grid.y;
  const uint nq = uint(query_tokens), active = uint(active_cache_length);
  const uint capacity = uint(cache_capacity), prefix = uint(prefix_length);
  const uint splits = uint(key_splits);
  const uint rows = nq * HeadsPerKv;
  const uint query_tiles = (rows + QTile - 1) / QTile;
  const uint query_tile = threadgroup_position_in_grid.x % query_tiles;
  const uint split = threadgroup_position_in_grid.x / query_tiles;
  const uint active_splits = min(splits, max(1u, (active + 1023) / 1024));
  if (split >= active_splits) return;
  const uint row_start = query_tile * QTile;
  const uint tiles_per_split = ((active + KTile - 1) / KTile + active_splits - 1) / active_splits;
  const uint first_key = split * tiles_per_split * KTile;
  const uint causal_end = min(active, prefix + min(rows - 1, row_start + QTile - 1) / HeadsPerKv + 1);
  const uint last_key = min(causal_end, (split + 1) * tiles_per_split * KTile);
  for (uint i = tid; i < QTile * D; i += 128) {
    const uint r = row_start + i / D;
    qtile[i] = r < rows ? query[((kv_head * HeadsPerKv + r % HeadsPerKv) * nq + r / HeadsPerKv) * D + i % D] : Activation(0);
    accumulated[i] = 0.0f;
  }
  if (tid < QTile) { stats[2 * tid] = -INFINITY; stats[2 * tid + 1] = 0.0f; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint key_start = first_key; key_start < last_key; key_start += KTile) {
    for (uint pack = tid; pack < KTile * D / 4; pack += 128) {
      const uint token = key_start + pack / (D / 4), dimension = (pack % (D / 4)) * 4;
      const ulong row = ulong(kv_head) * capacity + token;
      const uint code = key_cache[row * (D / 4) + dimension / 4];
      const float scale = float(key_scales[row * (D / 64) + dimension / 64]);
      const float bias = float(key_biases[row * (D / 64) + dimension / 64]);
      for (uint e = 0; e < 4; ++e)
        kvtile[pack * 4 + e] = token < last_key
            ? Activation(float((code >> (8 * e)) & 255) * scale + bias)
            : Activation(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 score = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint dimension = 0; dimension < D; dimension += 8) {
      ActivationMatrix q, k;
      simdgroup_load(q, qtile + dimension, D, 0, false);
      simdgroup_load(k, kvtile + sg * 8 * D + dimension, D, 0, true);
      simdgroup_multiply_accumulate(score, q, k, score);
    }
    simdgroup_store(score, scores + sg * 8, KTile, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint softmax_row = tid / 16, softmax_lane = tid % 16;
    const uint logical_row = row_start + softmax_row;
    const uint limit = logical_row < rows ? min(active, prefix + logical_row / HeadsPerKv + 1) : 0;
    float values[2], maximum = -INFINITY;
    for (uint e = 0; e < 2; ++e) {
      const uint col = softmax_lane + 16 * e;
      values[e] = key_start + col < limit ? scores[softmax_row * KTile + col] * 0.0625f : -INFINITY;
      maximum = max(maximum, values[e]);
    }
    maximum = sglang_attention_simd_max_16(maximum);
    const float old_max = stats[2 * softmax_row], old_sum = stats[2 * softmax_row + 1];
    const float next_max = maximum == -INFINITY ? old_max : max(old_max, maximum);
    const float rescale = old_sum == 0.0f ? 0.0f : (maximum == -INFINITY ? 1.0f : exp(old_max - next_max));
    float sum = 0.0f;
    for (uint e = 0; e < 2; ++e) {
      const float probability = values[e] == -INFINITY ? 0.0f : exp(values[e] - next_max);
      scores[softmax_row * KTile + softmax_lane + 16 * e] = probability;
      sum += probability;
    }
    sum = sglang_attention_simd_sum_16(sum);
    for (uint dimension = softmax_lane; dimension < D; dimension += 16)
      accumulated[softmax_row * D + dimension] *= rescale;
    if (softmax_lane == 0) {
      stats[2 * softmax_row] = next_max;
      stats[2 * softmax_row + 1] = old_sum * rescale + sum;
    }
    for (uint pack = tid; pack < KTile * D / 4; pack += 128) {
      const uint token = key_start + pack / (D / 4), dimension = (pack % (D / 4)) * 4;
      const ulong row = ulong(kv_head) * capacity + token;
      const uint code = value_cache[row * (D / 4) + dimension / 4];
      const float scale = float(value_scales[row * (D / 64) + dimension / 64]);
      const float bias = float(value_biases[row * (D / 64) + dimension / 64]);
      for (uint e = 0; e < 4; ++e)
        kvtile[pack * 4 + e] = token < last_key
            ? Activation(float((code >> (8 * e)) & 255) * scale + bias)
            : Activation(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 outputs[8];
    for (uint block = 0; block < 8; ++block)
      simdgroup_load(outputs[block], accumulated + sg * 64 + block * 8, D, 0, false);
    for (uint key = 0; key < KTile; key += 8) {
      simdgroup_float8x8 probability;
      simdgroup_load(probability, scores + key, KTile, 0, false);
      for (uint block = 0; block < 8; ++block) {
        ActivationMatrix value;
        simdgroup_load(value, kvtile + key * D + sg * 64 + block * 8, D, 0, false);
        simdgroup_multiply_accumulate(outputs[block], probability, value, outputs[block]);
      }
    }
    for (uint block = 0; block < 8; ++block)
      simdgroup_store(outputs[block], accumulated + sg * 64 + block * 8, D, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint i = tid; i < QTile * D; i += 128) {
    const uint row = row_start + i / D;
    if (row < rows) {
      const uint head = kv_head * HeadsPerKv + row % HeadsPerKv;
      const uint token = row / HeadsPerKv;
      const ulong output_row = ulong(head) * nq + token;
      if (splits > 1) {
        const ulong base = (ulong(split) * 24 * nq + output_row) * (D + 2);
        output[base + i % D] = accumulated[i];
      } else {
        const float denominator = stats[2 * (i / D) + 1];
        output[output_row * D + i % D] = denominator == 0.0f ? 0.0f : accumulated[i] / denominator;
      }
    }
  }
  if (splits > 1 && tid < QTile && row_start + tid < rows) {
    const uint row = row_start + tid;
    const uint head = kv_head * HeadsPerKv + row % HeadsPerKv;
    const ulong output_row = ulong(head) * nq + row / HeadsPerKv;
    const ulong base = (ulong(split) * 24 * nq + output_row) * (D + 2);
    output[base + D] = stats[2 * tid];
    output[base + D + 1] = stats[2 * tid + 1];
  }
)";

constexpr const char* kReduceQ8DecodeSource = R"(
        constexpr uint HeadDim = 256;
        const uint QueryHeads = 24 * uint(query_tokens);
        constexpr uint PartialStride = HeadDim + 2;

        const ushort tid = thread_index_in_threadgroup;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint query_head = threadgroup_position_in_grid.x;
        const uint active_length = uint(active_cache_length);
        const uint requested_key_splits = uint(key_splits);
        const uint active_key_splits = min(
            requested_key_splits,
            max(1u, (active_length + 1023) / 1024));
        threadgroup float split_scales[32];
        threadgroup float denominator;

        if (simd_id == 0) {
          const ulong partial_base =
              (ulong(lane) * QueryHeads + query_head) * PartialStride;
          const float partial_sum = lane < active_key_splits
              ? partials[partial_base + HeadDim + 1]
              : 0.0f;
          const float partial_max = partial_sum == 0.0f
              ? -INFINITY
              : partials[partial_base + HeadDim];
          const float merged_max = simd_max(partial_max);
          const float partial_scale = partial_sum == 0.0f
              ? 0.0f
              : exp(partial_max - merged_max);
          if (lane < active_key_splits) {
            split_scales[lane] = partial_scale;
          }
          const float merged_sum = simd_sum(partial_sum * partial_scale);
          if (lane == 0) {
            denominator = merged_sum;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < HeadDim && query_head < QueryHeads) {
          float value = 0.0f;
          for (uint split = 0; split < active_key_splits; ++split) {
            const ulong partial_base =
                (ulong(split) * QueryHeads + query_head) * PartialStride;
            value += partials[partial_base + tid] * split_scales[split];
          }
          output[query_head * HeadDim + tid] = static_cast<Activation>(
              denominator == 0.0f ? 0.0f : value / denominator);
        }
)";

constexpr const char* kGatedDeltaNormGateSource = R"(
        constexpr int N_READS = 4;
        constexpr int SIMD_SIZE = 32;
        constexpr int SIMD_GROUPS = Threads / SIMD_SIZE;
        threadgroup float local_inv_mean[1];
        threadgroup float local_sums[SIMD_SIZE];

        auto row = threadgroup_position_in_grid.x;
        auto lid = thread_position_in_threadgroup.x;
        auto simd_lane = thread_index_in_simdgroup;
        auto simd_group = simdgroup_index_in_threadgroup;

        constexpr int Iterations =
            (D + Threads * N_READS - 1) / (Threads * N_READS);
        float cached[Iterations][N_READS];
        float acc = 0.0f;
        int iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float value = static_cast<float>(recurrent_out[index]);
              cached[iteration][i] = value;
              acc += value * value;
            }
          }
        }

        acc = simd_sum(acc);
        float inv_mean;
        if (SIMD_GROUPS == 1) {
          inv_mean = metal::precise::rsqrt(acc / D + eps);
        } else {
          if (simd_lane == 0) {
            local_sums[simd_group] = acc;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (simd_group == 0) {
            acc = simd_sum(
                simd_lane < SIMD_GROUPS ? local_sums[simd_lane] : 0.0f);
            if (simd_lane == 0) {
              local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps);
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          inv_mean = local_inv_mean[0];
        }

        iteration = 0;
        for (uint base = lid * N_READS; base < D;
             base += Threads * N_READS, ++iteration) {
          for (int i = 0; i < N_READS; ++i) {
            if (base + i < D) {
              auto index = row * D + base + i;
              float normed =
                  static_cast<float>(weight[base + i]) *
                  (cached[iteration][i] * inv_mean);
              float z_value = static_cast<float>(z[index]);
              float sigmoid_low =
                  1.0f /
                  (1.0f + metal::precise::exp(metal::abs(z_value)));
              float sigmoid_value =
                  (z_value < 0.0f) ? sigmoid_low : 1.0f - sigmoid_low;
              gated_out[index] = static_cast<GateT>(
                  (z_value * sigmoid_value) * normed);
            }
          }
        }
)";

constexpr const char* kAffineQ4BatchOneQmvHeader = R"(
#include <metal_simdgroup>

template <typename T, typename U, int ValuesPerThread>
inline U sglang_q4_load_vector(
    const device T* input,
    thread U* input_values) {
  U sum = 0;
  for (int index = 0; index < ValuesPerThread; index += 4) {
    sum += input[index] + input[index + 1] + input[index + 2] +
        input[index + 3];
    input_values[index] = input[index];
    input_values[index + 1] = input[index + 1] / 16.0f;
    input_values[index + 2] = input[index + 2] / 256.0f;
    input_values[index + 3] = input[index + 3] / 4096.0f;
  }
  return sum;
}

template <typename U, int ValuesPerThread>
inline U sglang_q4_dot(
    const device uchar* weights,
    const thread U* input_values,
    U scale,
    U bias,
    U sum) {
  U accumulator = 0;
  const device ushort* packed =
      reinterpret_cast<const device ushort*>(weights);
  for (int index = 0; index < ValuesPerThread / 4; ++index) {
    accumulator +=
        input_values[4 * index] * (packed[index] & 0x000f) +
        input_values[4 * index + 1] * (packed[index] & 0x00f0) +
        input_values[4 * index + 2] * (packed[index] & 0x0f00) +
        input_values[4 * index + 3] * (packed[index] & 0xf000);
  }
  return scale * accumulator + sum * bias;
}

template <typename T>
inline T sglang_q4_sigmoid(T x) {
#if SGLANG_NATIVE_FP16
  // MLX's half sigmoid uses the half overload of exp. The precise overload
  // promotes its result and changes the intermediate rounding boundaries.
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
#else
  // Fast exp differs from precise exp at only BF16 -6.84375 (0xc0db).
  constexpr ushort FastMismatchInput = 0xc0db;
  constexpr ushort PreciseSigmoidResult = 0x3a8b;
  const ushort pattern = as_type<ushort>(x);
  if (pattern == FastMismatchInput) {
    return as_type<T>(PreciseSigmoidResult);
  }
  auto y = 1 / (1 + metal::exp(metal::abs(x)));
#endif
  return (x < 0) ? y : 1 - y;
}
)";

constexpr const char* kAffineQ4BatchOneQmvSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 4;
        constexpr int ResultsPerSimdgroup = 4;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;

        thread float input_values[ValuesPerThread];
        thread float results[ResultsPerSimdgroup] = {0};

        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* weight_cursor =
            reinterpret_cast<const device uchar*>(w) +
            output_start * WeightRowBytes +
            thread_index_in_simdgroup * PacksPerThread * 4;
        const device Activation* scale_cursor =
            scales + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* bias_cursor =
            biases + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* input_cursor =
            x + thread_index_in_simdgroup * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          const float sum =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input_cursor, input_values);

          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const device uchar* row_weights =
                weight_cursor + row * WeightRowBytes;
            const device Activation* row_scale =
                scale_cursor + row * GroupsPerRow;
            const device Activation* row_bias =
                bias_cursor + row * GroupsPerRow;
            const float scale = row_scale[0];
            const float bias = row_bias[0];
            results[row] += sglang_q4_dot<float, ValuesPerThread>(
                row_weights, input_values, scale, bias, sum);
          }

          weight_cursor += BlockSize / 2;
          scale_cursor += BlockSize / 64;
          bias_cursor += BlockSize / 64;
          input_cursor += BlockSize;
        }

        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          results[row] = simd_sum(results[row]);
          if (thread_index_in_simdgroup == 0) {
            y[output_start + row] = static_cast<Activation>(results[row]);
          }
        }
)";

// Two output rows per SIMDgroup reduce register pressure. The sixteen-value
// input fragments and their accumulation order remain unchanged.
constexpr const char* kAffineQ4BatchThreeSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 4;
        constexpr int ResultsPerSimdgroup = 2;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int OutputFeatures = NConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;

        thread float input0_values[ValuesPerThread];
        thread float input1_values[ValuesPerThread];
        thread float input2_values[ValuesPerThread];
        thread float results0[ResultsPerSimdgroup] = {0};
        thread float results1[ResultsPerSimdgroup] = {0};
        thread float results2[ResultsPerSimdgroup] = {0};

        const int lane = thread_index_in_simdgroup;
        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* weight_cursor =
            reinterpret_cast<const device uchar*>(w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device Activation* scale_cursor =
            scales + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* bias_cursor =
            biases + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* input0_cursor =
            x + lane * ValuesPerThread;
        const device Activation* input1_cursor =
            x + InputFeatures + lane * ValuesPerThread;
        const device Activation* input2_cursor =
            x + 2 * InputFeatures + lane * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          const float sum0 =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input0_cursor, input0_values);
          const float sum1 =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input1_cursor, input1_values);
          const float sum2 =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input2_cursor, input2_values);

#pragma unroll
          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const float scale = static_cast<float>(
                scale_cursor[row * GroupsPerRow]);
            const float bias = static_cast<float>(
                bias_cursor[row * GroupsPerRow]);
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(
                    weight_cursor + row * WeightRowBytes);
            float accumulator0 = 0.0f;
            float accumulator1 = 0.0f;
            float accumulator2 = 0.0f;
#pragma unroll
            for (int index = 0; index < ValuesPerThread / 4; ++index) {
              const ushort code = packed[index];
              accumulator0 +=
                  input0_values[4 * index] * (code & 0x000f) +
                  input0_values[4 * index + 1] * (code & 0x00f0) +
                  input0_values[4 * index + 2] * (code & 0x0f00) +
                  input0_values[4 * index + 3] * (code & 0xf000);
              accumulator1 +=
                  input1_values[4 * index] * (code & 0x000f) +
                  input1_values[4 * index + 1] * (code & 0x00f0) +
                  input1_values[4 * index + 2] * (code & 0x0f00) +
                  input1_values[4 * index + 3] * (code & 0xf000);
              accumulator2 +=
                  input2_values[4 * index] * (code & 0x000f) +
                  input2_values[4 * index + 1] * (code & 0x00f0) +
                  input2_values[4 * index + 2] * (code & 0x0f00) +
                  input2_values[4 * index + 3] * (code & 0xf000);
            }
            results0[row] += scale * accumulator0 + sum0 * bias;
            results1[row] += scale * accumulator1 + sum1 * bias;
            results2[row] += scale * accumulator2 + sum2 * bias;
          }

          weight_cursor += BlockSize / 2;
          scale_cursor += BlockSize / 64;
          bias_cursor += BlockSize / 64;
          input0_cursor += BlockSize;
          input1_cursor += BlockSize;
          input2_cursor += BlockSize;
        }

#pragma unroll
        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          const float result0 = simd_sum(results0[row]);
          const float result1 = simd_sum(results1[row]);
          const float result2 = simd_sum(results2[row]);
          if (lane == 0) {
            y[output_start + row] =
                static_cast<Activation>(result0);
            y[OutputFeatures + output_start + row] =
                static_cast<Activation>(result1);
            y[2 * OutputFeatures + output_start + row] =
                static_cast<Activation>(result2);
          }
        }
)";


constexpr const char* kAffineQ4BatchTwoQmvSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 4;
        constexpr int ResultsPerSimdgroup = 4;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int OutputFeatures = NConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;

        thread float input0_values[ValuesPerThread];
        thread float input1_values[ValuesPerThread];
        thread float results0[ResultsPerSimdgroup] = {0};
        thread float results1[ResultsPerSimdgroup] = {0};

        const int lane = thread_index_in_simdgroup;
        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* weight_cursor =
            reinterpret_cast<const device uchar*>(w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device Activation* scale_cursor =
            scales + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* bias_cursor =
            biases + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* input0_cursor =
            x + lane * ValuesPerThread;
        const device Activation* input1_cursor =
            x + InputFeatures + lane * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          const float sum0 =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input0_cursor, input0_values);
          const float sum1 =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input1_cursor, input1_values);

#pragma unroll
          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const float scale = static_cast<float>(
                scale_cursor[row * GroupsPerRow]);
            const float bias = static_cast<float>(
                bias_cursor[row * GroupsPerRow]);
            const device ushort* packed =
                reinterpret_cast<const device ushort*>(
                    weight_cursor + row * WeightRowBytes);
            float accumulator0 = 0.0f;
            float accumulator1 = 0.0f;
#pragma unroll
            for (int index = 0; index < ValuesPerThread / 4; ++index) {
              const ushort code = packed[index];
              accumulator0 +=
                  input0_values[4 * index] * (code & 0x000f) +
                  input0_values[4 * index + 1] * (code & 0x00f0) +
                  input0_values[4 * index + 2] * (code & 0x0f00) +
                  input0_values[4 * index + 3] * (code & 0xf000);
              accumulator1 +=
                  input1_values[4 * index] * (code & 0x000f) +
                  input1_values[4 * index + 1] * (code & 0x00f0) +
                  input1_values[4 * index + 2] * (code & 0x0f00) +
                  input1_values[4 * index + 3] * (code & 0xf000);
            }
            results0[row] += scale * accumulator0 + sum0 * bias;
            results1[row] += scale * accumulator1 + sum1 * bias;
          }

          weight_cursor += BlockSize / 2;
          scale_cursor += BlockSize / 64;
          bias_cursor += BlockSize / 64;
          input0_cursor += BlockSize;
          input1_cursor += BlockSize;
        }

#pragma unroll
        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          const float result0 = simd_sum(results0[row]);
          const float result1 = simd_sum(results1[row]);
          if (lane == 0) {
            y[output_start + row] =
                static_cast<Activation>(result0);
            y[OutputFeatures + output_start + row] =
                static_cast<Activation>(result1);
          }
        }
)";

constexpr const char* kAffineQ4FusedSwiGluSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 8;
        constexpr int ResultsPerSimdgroup = 4;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;

        thread float input_values[ValuesPerThread];
        thread float gate_results[ResultsPerSimdgroup] = {0};
        thread float up_results[ResultsPerSimdgroup] = {0};

        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* gate_weight_cursor =
            reinterpret_cast<const device uchar*>(gate_w) +
            output_start * WeightRowBytes +
            thread_index_in_simdgroup * PacksPerThread * 4;
        const device uchar* up_weight_cursor =
            reinterpret_cast<const device uchar*>(up_w) +
            output_start * WeightRowBytes +
            thread_index_in_simdgroup * PacksPerThread * 4;
        const device Activation* gate_scale_cursor =
            gate_scales + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* gate_bias_cursor =
            gate_biases + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* up_scale_cursor =
            up_scales + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* up_bias_cursor =
            up_biases + output_start * GroupsPerRow +
            thread_index_in_simdgroup / ScaleStep;
        const device Activation* input_cursor =
            x + thread_index_in_simdgroup * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          const float sum =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input_cursor, input_values);

          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const device uchar* row_weights =
                gate_weight_cursor + row * WeightRowBytes;
            const device Activation* row_scale =
                gate_scale_cursor + row * GroupsPerRow;
            const device Activation* row_bias =
                gate_bias_cursor + row * GroupsPerRow;
            const float scale = row_scale[0];
            const float bias = row_bias[0];
            gate_results[row] += sglang_q4_dot<float, ValuesPerThread>(
                row_weights, input_values, scale, bias, sum);
          }

          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const device uchar* row_weights =
                up_weight_cursor + row * WeightRowBytes;
            const device Activation* row_scale =
                up_scale_cursor + row * GroupsPerRow;
            const device Activation* row_bias =
                up_bias_cursor + row * GroupsPerRow;
            const float scale = row_scale[0];
            const float bias = row_bias[0];
            up_results[row] += sglang_q4_dot<float, ValuesPerThread>(
                row_weights, input_values, scale, bias, sum);
          }

          gate_weight_cursor += BlockSize / 2;
          up_weight_cursor += BlockSize / 2;
          gate_scale_cursor += BlockSize / 64;
          gate_bias_cursor += BlockSize / 64;
          up_scale_cursor += BlockSize / 64;
          up_bias_cursor += BlockSize / 64;
          input_cursor += BlockSize;
        }

        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          gate_results[row] = simd_sum(gate_results[row]);
          up_results[row] = simd_sum(up_results[row]);
        }
        const int row = thread_index_in_simdgroup;
        if (row < ResultsPerSimdgroup) {
          const float gate_result = row == 0 ? gate_results[0]
              : row == 1                 ? gate_results[1]
              : row == 2                 ? gate_results[2]
                                         : gate_results[3];
          const float up_result = row == 0 ? up_results[0]
              : row == 1               ? up_results[1]
              : row == 2               ? up_results[2]
                                       : up_results[3];
          const Activation gate_value = static_cast<Activation>(gate_result);
          const Activation up_value = static_cast<Activation>(up_result);
          const Activation sigmoid_value = sglang_q4_sigmoid(gate_value);
          const Activation silu_value =
              static_cast<Activation>(gate_value * sigmoid_value);
          y[output_start + row] =
              static_cast<Activation>(silu_value * up_value);
        }
)";

constexpr const char* kAffineQ4FusedSwiGluRawParamsSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 8;
        constexpr int ResultsPerSimdgroup = 4;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;
        constexpr int ParameterWordsPerBundle = 8;

        thread float input_values[ValuesPerThread];
        thread float gate_results[ResultsPerSimdgroup] = {0};
        thread float up_results[ResultsPerSimdgroup] = {0};

        const int lane = thread_index_in_simdgroup;
        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* gate_weight_cursor =
            reinterpret_cast<const device uchar*>(gate_w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device uchar* up_weight_cursor =
            reinterpret_cast<const device uchar*>(up_w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device uint* parameter_cursor = params +
            ((output_start / ResultsPerSimdgroup) * GroupsPerRow +
             lane / ScaleStep) * ParameterWordsPerBundle;
        const device Activation* input_cursor =
            x + lane * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          const float sum =
              sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                  input_cursor, input_values);
          const uint4 gate_parameters =
              *reinterpret_cast<const device uint4*>(parameter_cursor);
          const uint4 up_parameters =
              *reinterpret_cast<const device uint4*>(parameter_cursor + 4);

          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const uint code = gate_parameters[row];
            const float scale = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code)));
            const float bias = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code >> 16)));
            gate_results[row] += sglang_q4_dot<float, ValuesPerThread>(
                gate_weight_cursor + row * WeightRowBytes,
                input_values,
                scale,
                bias,
                sum);
          }

          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const uint code = up_parameters[row];
            const float scale = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code)));
            const float bias = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code >> 16)));
            up_results[row] += sglang_q4_dot<float, ValuesPerThread>(
                up_weight_cursor + row * WeightRowBytes,
                input_values,
                scale,
                bias,
                sum);
          }

          gate_weight_cursor += BlockSize / 2;
          up_weight_cursor += BlockSize / 2;
          parameter_cursor +=
              (BlockSize / 64) * ParameterWordsPerBundle;
          input_cursor += BlockSize;
        }

        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          gate_results[row] = simd_sum(gate_results[row]);
          up_results[row] = simd_sum(up_results[row]);
        }
        const int row = lane;
        if (row < ResultsPerSimdgroup) {
          const float gate_result = row == 0 ? gate_results[0]
              : row == 1                 ? gate_results[1]
              : row == 2                 ? gate_results[2]
                                         : gate_results[3];
          const float up_result = row == 0 ? up_results[0]
              : row == 1               ? up_results[1]
              : row == 2               ? up_results[2]
                                       : up_results[3];
          const Activation gate_value = static_cast<Activation>(gate_result);
          const Activation up_value = static_cast<Activation>(up_result);
          const Activation sigmoid_value = sglang_q4_sigmoid(gate_value);
          const Activation silu_value =
              static_cast<Activation>(gate_value * sigmoid_value);
          y[output_start + row] =
              static_cast<Activation>(silu_value * up_value);
        }
)";

constexpr const char* kAffineQ4FusedSwiGluBatchTwoRawParamsSource = R"(
        constexpr int PacksPerThread = 2;
        constexpr int SimdGroups = 8;
        constexpr int ResultsPerSimdgroup = 4;
        constexpr int ValuesPerThread = 8 * PacksPerThread;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int ScaleStep = 64 / ValuesPerThread;
        constexpr int InputFeatures = KConst;
        constexpr int OutputFeatures = NConst;
        constexpr int WeightRowBytes = InputFeatures / 2;
        constexpr int GroupsPerRow = InputFeatures / 64;
        constexpr int ParameterWordsPerBundle = 8;

        thread float2 input_values[ValuesPerThread];
        thread float2 gate_results[ResultsPerSimdgroup];
        thread float2 up_results[ResultsPerSimdgroup];
#pragma unroll
        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          gate_results[row] = float2(0.0f);
          up_results[row] = float2(0.0f);
        }

        const int lane = thread_index_in_simdgroup;
        const int output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simdgroup_index_in_threadgroup * ResultsPerSimdgroup;
        const device uchar* gate_weight_cursor =
            reinterpret_cast<const device uchar*>(gate_w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device uchar* up_weight_cursor =
            reinterpret_cast<const device uchar*>(up_w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 4;
        const device uint* parameter_cursor = params +
            ((output_start / ResultsPerSimdgroup) * GroupsPerRow +
             lane / ScaleStep) * ParameterWordsPerBundle;
        const device Activation* input0_cursor =
            x + lane * ValuesPerThread;
        const device Activation* input1_cursor =
            x + InputFeatures + lane * ValuesPerThread;

        for (int k = 0; k < InputFeatures; k += BlockSize) {
          float2 sum = float2(0.0f);
          if constexpr (ScalarInputs) {
            // Keep MLX's BF16 additions in the affine bias sum. Promoting
            // each operand before addition changes cancellation/rounding.
            thread float input0_values[ValuesPerThread];
            thread float input1_values[ValuesPerThread];
            sum.x = sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                input0_cursor, input0_values);
            sum.y = sglang_q4_load_vector<Activation, float, ValuesPerThread>(
                input1_cursor, input1_values);
#pragma unroll
            for (int index = 0; index < ValuesPerThread; ++index) {
              input_values[index] = float2(
                  input0_values[index], input1_values[index]);
            }
          } else {
#pragma unroll
            for (int index = 0; index < ValuesPerThread; index += 4) {
              const float2 value0 = float2(
                  static_cast<float>(input0_cursor[index]),
                  static_cast<float>(input1_cursor[index]));
              const float2 value1 = float2(
                  static_cast<float>(input0_cursor[index + 1]),
                  static_cast<float>(input1_cursor[index + 1]));
              const float2 value2 = float2(
                  static_cast<float>(input0_cursor[index + 2]),
                  static_cast<float>(input1_cursor[index + 2]));
              const float2 value3 = float2(
                  static_cast<float>(input0_cursor[index + 3]),
                  static_cast<float>(input1_cursor[index + 3]));
              // Preserve each row's BF16 addition chain while keeping the
              // shared float2 projection inputs in their existing registers.
              // This avoids staging two extra scalar activation arrays.
              sum += float2(
                  static_cast<float>(input0_cursor[index] +
                      input0_cursor[index + 1] + input0_cursor[index + 2] +
                      input0_cursor[index + 3]),
                  static_cast<float>(input1_cursor[index] +
                      input1_cursor[index + 1] + input1_cursor[index + 2] +
                      input1_cursor[index + 3]));
              input_values[index] = value0;
              input_values[index + 1] = value1 / 16.0f;
              input_values[index + 2] = value2 / 256.0f;
              input_values[index + 3] = value3 / 4096.0f;
            }
          }
          const uint4 gate_parameters =
              *reinterpret_cast<const device uint4*>(parameter_cursor);
          const uint4 up_parameters =
              *reinterpret_cast<const device uint4*>(parameter_cursor + 4);

#pragma unroll
          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const uint code = gate_parameters[row];
            const float scale = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code)));
            const float bias = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code >> 16)));
            gate_results[row] +=
                sglang_q4_dot<float2, ValuesPerThread>(
                    gate_weight_cursor + row * WeightRowBytes,
                    input_values,
                    float2(scale),
                    float2(bias),
                    sum);
          }

#pragma unroll
          for (int row = 0; row < ResultsPerSimdgroup; ++row) {
            const uint code = up_parameters[row];
            const float scale = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code)));
            const float bias = static_cast<float>(
                as_type<Activation>(static_cast<ushort>(code >> 16)));
            up_results[row] +=
                sglang_q4_dot<float2, ValuesPerThread>(
                    up_weight_cursor + row * WeightRowBytes,
                    input_values,
                    float2(scale),
                    float2(bias),
                    sum);
          }

          gate_weight_cursor += BlockSize / 2;
          up_weight_cursor += BlockSize / 2;
          parameter_cursor +=
              (BlockSize / 64) * ParameterWordsPerBundle;
          input0_cursor += BlockSize;
          input1_cursor += BlockSize;
        }

#pragma unroll
        for (int row = 0; row < ResultsPerSimdgroup; ++row) {
          gate_results[row] = float2(
              simd_sum(gate_results[row].x),
              simd_sum(gate_results[row].y));
          up_results[row] = float2(
              simd_sum(up_results[row].x),
              simd_sum(up_results[row].y));
        }
        const int row = lane;
        if (row < ResultsPerSimdgroup) {
          const float2 gate_result = row == 0 ? gate_results[0]
              : row == 1                   ? gate_results[1]
              : row == 2                   ? gate_results[2]
                                           : gate_results[3];
          const float2 up_result = row == 0 ? up_results[0]
              : row == 1                 ? up_results[1]
              : row == 2                 ? up_results[2]
                                         : up_results[3];
          const Activation gate_value0 =
              static_cast<Activation>(gate_result.x);
          const Activation gate_value1 =
              static_cast<Activation>(gate_result.y);
          const Activation up_value0 = static_cast<Activation>(up_result.x);
          const Activation up_value1 = static_cast<Activation>(up_result.y);
          const Activation sigmoid_value0 = sglang_q4_sigmoid(gate_value0);
          const Activation sigmoid_value1 = sglang_q4_sigmoid(gate_value1);
          const Activation silu_value0 =
              static_cast<Activation>(gate_value0 * sigmoid_value0);
          const Activation silu_value1 =
              static_cast<Activation>(gate_value1 * sigmoid_value1);
          y[output_start + row] =
              static_cast<Activation>(silu_value0 * up_value0);
          y[OutputFeatures + output_start + row] =
              static_cast<Activation>(silu_value1 * up_value1);
        }
)";

constexpr const char* kAffineSmallBatchQmmHeader = R"(
#include <metal_simdgroup>
)";

constexpr const char* kAffineSmallBatchQmmSource = R"(
        constexpr ushort OutputTile = 64;
        constexpr ushort RowTile = 8;
        constexpr ushort KTile = 64;
        constexpr ushort Threads = 128;
        constexpr ushort ValuesPerWord = 32 / Bits;
        constexpr ushort WordsPerRow = KTile / ValuesPerWord;
        constexpr ushort QuantGroupsPerTile = KTile / 64;

        threadgroup Activation staged_x[RowTile * KTile];
        threadgroup Activation staged_w[OutputTile * KTile];
        threadgroup float staged_scales[
            OutputTile * QuantGroupsPerTile];
        threadgroup float staged_biases[
            OutputTile * QuantGroupsPerTile];
        threadgroup float staged_y[RowTile * OutputTile];

        const ushort tid = thread_position_in_threadgroup.x;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.x * OutputTile;
        const uint packed_k = K * Bits / 32;
        const uint groups_per_row = K / 64;

        simdgroup_float8x8 accumulators[2];
        accumulators[0] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        accumulators[1] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);

        for (uint k_start = 0; k_start < K; k_start += KTile) {
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (ushort index = tid; index < RowTile * KTile;
               index += Threads) {
            const ushort row = index / KTile;
            const ushort column = index % KTile;
            staged_x[index] = row < M
                ? x[row * K + k_start + column]
                : static_cast<Activation>(0.0f);
          }

          if (tid < OutputTile * QuantGroupsPerTile) {
            const ushort output = tid / QuantGroupsPerTile;
            const ushort group = tid % QuantGroupsPerTile;
            const uint parameter_index =
                (output_start + output) * groups_per_row +
                k_start / 64 + group;
            staged_scales[tid] = static_cast<float>(scales[parameter_index]);
            staged_biases[tid] = static_cast<float>(biases[parameter_index]);
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          constexpr ushort PackedWords = OutputTile * WordsPerRow;
          for (ushort index = tid; index < PackedWords;
               index += Threads) {
            const ushort output = index / WordsPerRow;
            const ushort word_column = index % WordsPerRow;
            const uint packed = w[
                (output_start + output) * packed_k +
                k_start / ValuesPerWord + word_column];
            const ushort group =
                word_column * ValuesPerWord / 64;
            const ushort parameter = output * QuantGroupsPerTile + group;
            const float scale = staged_scales[parameter];
            const float bias = staged_biases[parameter];
#pragma unroll
            for (ushort value = 0; value < ValuesPerWord; ++value) {
              const uint quantized =
                  (packed >> (value * Bits)) & ((1u << Bits) - 1u);
              staged_w[
                  output * KTile + word_column * ValuesPerWord + value] =
                  static_cast<Activation>(scale * quantized + bias);
            }
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          threadgroup const Activation* input_tile = staged_x;
          threadgroup const Activation* weight_tile =
              staged_w + simd_id * 16 * KTile;
#pragma unroll
          for (ushort k_step = 0; k_step < KTile / 8; ++k_step) {
            ActivationMatrix input_fragment;
            ActivationMatrix weight_fragment;
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(
                input_fragment, input_tile + k_step * 8, KTile, 0, false);
#pragma unroll
            for (ushort output_step = 0; output_step < 2; ++output_step) {
              simdgroup_load(
                  weight_fragment,
                  weight_tile + output_step * 8 * KTile + k_step * 8,
                  KTile,
                  0,
                  true);
              simdgroup_multiply_accumulate(
                  accumulators[output_step],
                  input_fragment,
                  weight_fragment,
                  accumulators[output_step]);
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort output_step = 0; output_step < 2; ++output_step) {
          simdgroup_store(
              accumulators[output_step],
              staged_y + simd_id * 16 + output_step * 8,
              OutputTile,
              0,
              false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort index = tid; index < M * OutputTile; index += Threads) {
          const ushort row = index / OutputTile;
          const ushort column = index % OutputTile;
          y[row * N + output_start + column] =
              static_cast<Activation>(staged_y[row * OutputTile + column]);
        }
)";

constexpr const char* kAffineQ5BatchOneQmvSource = R"(
        constexpr ushort ValuesPerPack = 8;
        constexpr ushort PacksPerThread = 2;
        constexpr ushort ValuesPerThread = ValuesPerPack * PacksPerThread;
        constexpr ushort BlockSize = ValuesPerThread * 32;
        constexpr ushort ScaleStep = 64 / ValuesPerThread;
        constexpr ushort SimdGroups = 4;
        constexpr ushort ResultsPerSimdgroup = 4;
        constexpr uint K = KConst;
        constexpr uint WeightRowBytes = K * 5 / 8;
        constexpr uint GroupsPerRow = K / 64;

        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_group = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simd_group * ResultsPerSimdgroup;

        const device uchar* weight_ptr =
            reinterpret_cast<const device uchar*>(w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 5;
        const device Activation* scale_ptr =
            scales + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* bias_ptr =
            biases + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* input_ptr = x + lane * ValuesPerThread;

        float input_values[ValuesPerThread];
        float results[ResultsPerSimdgroup] = {0.0f};

        for (uint k = 0; k < K; k += BlockSize) {
          float input_sum = 0.0f;
#pragma unroll
          for (ushort index = 0; index < ValuesPerThread; ++index) {
            const float value = static_cast<float>(input_ptr[index]);
            input_values[index] = value;
            input_sum += value;
          }

#pragma unroll
          for (ushort row = 0; row < ResultsPerSimdgroup; ++row) {
            const device uchar* packed =
                weight_ptr + row * WeightRowBytes;
            const float scale =
                static_cast<float>(scale_ptr[row * GroupsPerRow]);
            const float bias =
                static_cast<float>(bias_ptr[row * GroupsPerRow]);
            float quantized_dot = 0.0f;

            const packed_ushort4 words =
                *reinterpret_cast<const device packed_ushort4*>(packed);
            const uint trailing =
                *reinterpret_cast<const device ushort*>(packed + 8);
            const uint window0 =
                static_cast<uint>(words[0]) |
                (static_cast<uint>(words[1]) << 16);
            const uint window1 =
                static_cast<uint>(words[2]) |
                (static_cast<uint>(words[3]) << 16);
            quantized_dot = fma(
                input_values[0],
                static_cast<float>(window0 & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[1],
                static_cast<float>((window0 >> 5) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[2],
                static_cast<float>((window0 >> 10) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[3],
                static_cast<float>((window0 >> 15) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[4],
                static_cast<float>((window0 >> 20) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[5],
                static_cast<float>((window0 >> 25) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[6],
                static_cast<float>(
                    (window0 >> 30) | ((window1 & 0x07u) << 2)),
                quantized_dot);
            quantized_dot = fma(
                input_values[7],
                static_cast<float>((window1 >> 3) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[8],
                static_cast<float>((window1 >> 8) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[9],
                static_cast<float>((window1 >> 13) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[10],
                static_cast<float>((window1 >> 18) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[11],
                static_cast<float>((window1 >> 23) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[12],
                static_cast<float>(
                    (window1 >> 28) | ((trailing & 0x01u) << 4)),
                quantized_dot);
            quantized_dot = fma(
                input_values[13],
                static_cast<float>((trailing >> 1) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[14],
                static_cast<float>((trailing >> 6) & 0x1fu),
                quantized_dot);
            quantized_dot = fma(
                input_values[15],
                static_cast<float>(trailing >> 11),
                quantized_dot);
            results[row] += scale * quantized_dot + bias * input_sum;
          }

          weight_ptr += BlockSize * 5 / 8;
          scale_ptr += BlockSize / 64;
          bias_ptr += BlockSize / 64;
          input_ptr += BlockSize;
        }

#pragma unroll
        for (ushort row = 0; row < ResultsPerSimdgroup; ++row) {
          const float result = simd_sum(results[row]);
          if (lane == 0) {
            y[output_start + row] = static_cast<Activation>(result);
          }
        }
)";

constexpr const char* kAffineQ5BatchTwoQmvSource = R"(
        constexpr ushort ValuesPerPack = 8;
        constexpr ushort PacksPerThread = 2;
        constexpr ushort ValuesPerThread = ValuesPerPack * PacksPerThread;
        constexpr ushort BlockSize = ValuesPerThread * 32;
        constexpr ushort ScaleStep = 64 / ValuesPerThread;
        constexpr ushort SimdGroups = 4;
        constexpr ushort ResultsPerSimdgroup = 4;
        constexpr uint K = KConst;
        constexpr uint N = NConst;
        constexpr uint WeightRowBytes = K * 5 / 8;
        constexpr uint GroupsPerRow = K / 64;

        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_group = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.y *
                (SimdGroups * ResultsPerSimdgroup) +
            simd_group * ResultsPerSimdgroup;

        const device uchar* weight_ptr =
            reinterpret_cast<const device uchar*>(w) +
            output_start * WeightRowBytes + lane * PacksPerThread * 5;
        const device Activation* scale_ptr =
            scales + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* bias_ptr =
            biases + output_start * GroupsPerRow + lane / ScaleStep;
        const device Activation* input0_ptr = x + lane * ValuesPerThread;
        const device Activation* input1_ptr =
            x + K + lane * ValuesPerThread;

        float2 input_values[ValuesPerThread];
        float2 results[ResultsPerSimdgroup];
#pragma unroll
        for (ushort row = 0; row < ResultsPerSimdgroup; ++row) {
          results[row] = float2(0.0f);
        }

        for (uint k = 0; k < K; k += BlockSize) {
          float2 input_sum = float2(0.0f);
#pragma unroll
          for (ushort index = 0; index < ValuesPerThread; ++index) {
            const float2 value = float2(
                static_cast<float>(input0_ptr[index]),
                static_cast<float>(input1_ptr[index]));
            input_values[index] = value;
            input_sum += value;
          }

#pragma unroll
          for (ushort row = 0; row < ResultsPerSimdgroup; ++row) {
            const device uchar* packed =
                weight_ptr + row * WeightRowBytes;
            const float scale =
                static_cast<float>(scale_ptr[row * GroupsPerRow]);
            const float bias =
                static_cast<float>(bias_ptr[row * GroupsPerRow]);
            const packed_ushort4 words =
                *reinterpret_cast<const device packed_ushort4*>(packed);
            const uint trailing =
                *reinterpret_cast<const device ushort*>(packed + 8);
            const uint window0 =
                static_cast<uint>(words[0]) |
                (static_cast<uint>(words[1]) << 16);
            const uint window1 =
                static_cast<uint>(words[2]) |
                (static_cast<uint>(words[3]) << 16);
            const uint quantized[ValuesPerThread] = {
                window0 & 0x1fu,
                (window0 >> 5) & 0x1fu,
                (window0 >> 10) & 0x1fu,
                (window0 >> 15) & 0x1fu,
                (window0 >> 20) & 0x1fu,
                (window0 >> 25) & 0x1fu,
                (window0 >> 30) | ((window1 & 0x07u) << 2),
                (window1 >> 3) & 0x1fu,
                (window1 >> 8) & 0x1fu,
                (window1 >> 13) & 0x1fu,
                (window1 >> 18) & 0x1fu,
                (window1 >> 23) & 0x1fu,
                (window1 >> 28) | ((trailing & 0x01u) << 4),
                (trailing >> 1) & 0x1fu,
                (trailing >> 6) & 0x1fu,
                trailing >> 11,
            };
            float2 quantized_dot = float2(0.0f);
#pragma unroll
            for (ushort index = 0; index < ValuesPerThread; ++index) {
              quantized_dot = fma(
                  input_values[index],
                  float2(static_cast<float>(quantized[index])),
                  quantized_dot);
            }
            results[row] += scale * quantized_dot + bias * input_sum;
          }

          weight_ptr += BlockSize * 5 / 8;
          scale_ptr += BlockSize / 64;
          bias_ptr += BlockSize / 64;
          input0_ptr += BlockSize;
          input1_ptr += BlockSize;
        }

#pragma unroll
        for (ushort row = 0; row < ResultsPerSimdgroup; ++row) {
          const float result0 = simd_sum(results[row].x);
          const float result1 = simd_sum(results[row].y);
          if (lane == 0) {
            y[output_start + row] = static_cast<Activation>(result0);
            y[N + output_start + row] = static_cast<Activation>(result1);
          }
        }
)";

constexpr const char* kAffineM8KsplitQmmSource = R"(
        constexpr ushort Rows = 8;
        constexpr ushort OutputTile = 32;
        constexpr ushort KTile = 32;
        constexpr ushort KStep = 8;
        constexpr ushort SimdGroups = 16;
        constexpr uint K = KConst;
        constexpr uint PackedBytes = K * Bits / 8;
        constexpr uint QuantGroups = K / 64;
        constexpr uint KChunk = K / SimdGroups;

        constexpr uint StagedElements =
            SimdGroups * KTile * OutputTile;
        // Weight staging and cross-SIMDgroup reduction have disjoint
        // lifetimes. Reuse the 32 KiB staging allocation for FP32 partials.
        threadgroup float storage[StagedElements / 2];
        threadgroup Activation* staged_w =
            reinterpret_cast<threadgroup Activation*>(storage);
        threadgroup float* partial = storage;

        const ushort tid = thread_position_in_threadgroup.x;
        const ushort lane = thread_index_in_simdgroup;
        const ushort simd_id = simdgroup_index_in_threadgroup;
        const uint output_start =
            threadgroup_position_in_grid.y * OutputTile;
        const uint k_begin = simd_id * KChunk;
        const uint k_end = k_begin + KChunk;

        ActivationMatrix input_fragment;
        ActivationMatrix weight_fragment;
        simdgroup_float8x8 accumulators[4];
#pragma unroll
        for (ushort output_step = 0; output_step < 4; ++output_step) {
          accumulators[output_step] =
              make_filled_simdgroup_matrix<float, 8>(0.0f);
        }

        const ushort output_column = lane;
        for (uint k_start = k_begin; k_start < k_end; k_start += KTile) {
#pragma unroll
          for (ushort pack_in_tile = 0; pack_in_tile < 4; ++pack_in_tile) {
            const uint k_base = k_start + pack_in_tile * 8;
            const uint output = output_start + output_column;
            const device uchar* packed =
                reinterpret_cast<const device uchar*>(w) +
                output * PackedBytes + k_base * Bits / 8;
            const uint parameter = output * QuantGroups + k_base / 64;
            const float scale = static_cast<float>(scales[parameter]);
            const float bias = static_cast<float>(biases[parameter]);
            threadgroup Activation* destination =
                staged_w + simd_id * KTile * OutputTile +
                pack_in_tile * 8 * OutputTile + output_column;
            if constexpr (Bits == 4) {
              const uint packed_word =
                  *reinterpret_cast<const device uint*>(packed);
#pragma unroll
              for (ushort value = 0; value < 8; ++value) {
                const uint quantized =
                    (packed_word >> (value * 4)) & 0xfu;
                destination[value * OutputTile] =
                    static_cast<Activation>(scale * quantized + bias);
              }
            } else {
              const uint byte0 = packed[0];
              const uint byte1 = packed[1];
              const uint byte2 = packed[2];
              const uint byte3 = packed[3];
              const uint byte4 = packed[4];
              const uint quantized[8] = {
                  byte0 & 0x1fu,
                  (byte0 >> 5) | ((byte1 & 0x03u) << 3),
                  (byte1 >> 2) & 0x1fu,
                  (byte1 >> 7) | ((byte2 & 0x0fu) << 1),
                  (byte2 >> 4) | ((byte3 & 0x01u) << 4),
                  (byte3 >> 1) & 0x1fu,
                  (byte3 >> 6) | ((byte4 & 0x07u) << 2),
                  byte4 >> 3,
              };
#pragma unroll
              for (ushort value = 0; value < 8; ++value) {
                destination[value * OutputTile] =
                    static_cast<Activation>(scale * quantized[value] + bias);
              }
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
          for (ushort k_step = 0; k_step < KTile / KStep; ++k_step) {
            simdgroup_load(
                input_fragment, x + k_start + k_step * KStep, K);
#pragma unroll
            for (ushort output_step = 0; output_step < 4; ++output_step) {
              simdgroup_load(
                  weight_fragment,
                  staged_w + simd_id * KTile * OutputTile +
                      k_step * KStep * OutputTile +
                      output_step * 8,
                  OutputTile);
              simdgroup_multiply_accumulate(
                  accumulators[output_step],
                  input_fragment,
                  weight_fragment,
                  accumulators[output_step]);
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (ushort output_step = 0; output_step < 4; ++output_step) {
          simdgroup_store(
              accumulators[output_step],
              partial + simd_id * Rows * OutputTile + output_step * 8,
              OutputTile);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (ushort offset = tid; offset < Rows * OutputTile;
             offset += SimdGroups * 32) {
          float value = 0.0f;
#pragma unroll
          for (ushort group = 0; group < SimdGroups; ++group) {
            value += partial[group * Rows * OutputTile + offset];
          }
          const ushort row = offset / OutputTile;
          const ushort column = offset % OutputTile;
          y[row * N_size + output_start + column] =
              static_cast<Activation>(value);
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
  static const auto kernel = activation_metal_kernel(
      "sglang_gated_delta_step",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out"},
      kGatedDeltaSource,
      kGatedDeltaHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_tape_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_gated_delta_step_tape",
      {"q", "k", "v", "g", "beta", "state_in", "T"},
      {"y", "state_out", "delta_out"},
      kGatedDeltaSource,
      kGatedDeltaTapeHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_commit_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_gated_delta_commit",
      {"keys", "decay", "delta", "state_in", "T", "token_count"},
      {"state_out"},
      kGatedDeltaCommitSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& causal_conv_decode_silu_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_causal_conv_decode_silu",
      {"state", "qkv", "weight"},
      {"conv_out", "next_state"},
      kCausalConvDecodeSiluSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& causal_conv_two_token_silu_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_causal_conv_two_token_silu",
      {"state", "qkv", "weight"},
      {"conv_out", "next_state"},
      kCausalConvTwoTokenSiluSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& residual_rms_norm_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_residual_rms_norm",
      {"x", "residual", "weight", "eps"},
      {"residual_out", "norm_out"},
      kResidualRmsNormSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_qk_norm_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_gated_delta_qk_norm",
      {"q", "k", "q_scale", "k_scale", "eps"},
      {"q_out", "k_out"},
      kGatedDeltaQkNormSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& full_attn_qk_norm_rope_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_full_attn_qk_norm_rope",
      {"qg", "k", "q_weight", "k_weight", "eps", "log2_base", "rope_offset"},
      {"q_out", "k_out"},
      kFullAttnQkNormRopeSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& fixed_prefill_attention_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_fixed_prefill_attention",
      {"query",
       "key_cache",
       "value_cache",
       "query_tokens",
       "prefix_length",
       "cache_capacity",
       "active_cache_length"},
      {"output"},
      kFixedPrefillAttentionSource,
      kFixedPrefillAttentionHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& fixed_q8_attention_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_fixed_q8_attention",
      {"query",
       "key_cache",
       "key_scales",
       "key_biases",
       "value_cache",
       "value_scales",
       "value_biases",
       "query_tokens",
       "prefix_length",
       "cache_capacity",
       "active_cache_length",
       "key_splits"},
      {"output"},
      kFixedQ8AttentionSource,
      kFixedPrefillAttentionHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& tiled_q8_attention_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_tiled_q8_attention",
      {"query",
       "key_cache",
       "key_scales",
       "key_biases",
       "value_cache",
       "value_scales",
       "value_biases",
       "query_tokens",
       "prefix_length",
       "cache_capacity",
       "active_cache_length",
       "key_splits"},
      {"output"},
      kTiledQ8AttentionSource,
      kFixedPrefillAttentionHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& reduce_q8_decode_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_reduce_q8_decode",
      {"partials", "active_cache_length", "key_splits", "query_tokens"},
      {"output"},
      kReduceQ8DecodeSource,
      kFixedPrefillAttentionHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& gated_delta_norm_gate_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_gated_delta_norm_gate",
      {"recurrent_out", "z", "weight", "eps"},
      {"gated_out"},
      kGatedDeltaNormGateSource);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q4_batch_one_qmv_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_batch_one_qmv",
      {"w", "scales", "biases", "x"},
      {"y"},
      kAffineQ4BatchOneQmvSource,
      kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q4_batch_two_qmv_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_batch_two_qmv",
      {"w", "scales", "biases", "x"},
      {"y"},
      kAffineQ4BatchTwoQmvSource,
      kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q4_batch_three_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_batch_three",
      {"w", "scales", "biases", "x"}, {"y"},
      kAffineQ4BatchThreeSource, kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q4_fused_swiglu_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_fused_swiglu",
      {"gate_w",
       "gate_scales",
       "gate_biases",
       "up_w",
       "up_scales",
       "up_biases",
       "x"},
      {"y"},
      kAffineQ4FusedSwiGluSource,
      kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction&
affine_q4_fused_swiglu_raw_params_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_fused_swiglu_raw_params",
      {"gate_w", "up_w", "params", "x"},
      {"y"},
      kAffineQ4FusedSwiGluRawParamsSource,
      kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction&
affine_q4_fused_swiglu_batch_two_raw_params_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q4_fused_swiglu_batch_two_raw_params",
      {"gate_w", "up_w", "params", "x"},
      {"y"},
      kAffineQ4FusedSwiGluBatchTwoRawParamsSource,
      kAffineQ4BatchOneQmvHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_small_batch_qmm_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_small_batch_qmm",
      {"w", "scales", "biases", "x", "K", "N", "M"},
      {"y"},
      kAffineSmallBatchQmmSource,
      kAffineSmallBatchQmmHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_m8_ksplit_qmm_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_m8_ksplit_qmm",
      {"w", "scales", "biases", "x", "N_size"},
      {"y"},
      kAffineM8KsplitQmmSource,
      kAffineSmallBatchQmmHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q5_batch_one_qmv_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q5_batch_one_qmv",
      {"w", "scales", "biases", "x"},
      {"y"},
      kAffineQ5BatchOneQmvSource,
      kAffineSmallBatchQmmHeader);
  return kernel;
}

const mx::fast::CustomKernelFunction& affine_q5_batch_two_qmv_metal() {
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q5_batch_two_qmv",
      {"w", "scales", "biases", "x"},
      {"y"},
      kAffineQ5BatchTwoQmvSource,
      kAffineSmallBatchQmmHeader);
  return kernel;
}

} // namespace

std::pair<array, array> align_mtp_committed_history(
    const array& target_hidden,
    const array& token_ids,
    const array& previous_hidden,
    bool has_previous_hidden,
    const array& final_norm,
    float rms_norm_eps,
    bool post_norm_hidden) {
  if (target_hidden.ndim() != 3 || target_hidden.shape()[0] != 1 ||
      token_ids.ndim() != 2 || token_ids.shape()[0] != 1 ||
      token_ids.dtype() != mx::int32 ||
      token_ids.shape()[1] != target_hidden.shape()[1] ||
      target_hidden.shape()[1] < 1 || target_hidden.shape()[2] < 1) {
    throw std::runtime_error("invalid committed MTP history inputs");
  }
  const int token_count = static_cast<int>(token_ids.shape()[1]);
  const int hidden_size = static_cast<int>(target_hidden.shape()[2]);
  array target_prefix = slice(
      target_hidden,
      {0, 0, 0},
      {1, token_count - 1, hidden_size});
  array aligned_hidden = target_prefix;
  array aligned_tokens = slice(
      token_ids,
      {0, 1},
      {1, token_count});
  if (has_previous_hidden) {
    array previous = previous_hidden;
    if (previous.ndim() == 2 && previous.shape() == mx::Shape{1, hidden_size}) {
      previous = expand_dims(previous, 1);
    }
    if (previous.ndim() != 3 ||
        previous.shape() != mx::Shape{1, 1, hidden_size} ||
        previous.dtype() != target_hidden.dtype()) {
      throw std::runtime_error("invalid previous hidden state for MTP history");
    }
    aligned_hidden = token_count == 1
        ? previous
        : concatenate({previous, target_prefix}, 1);
    aligned_tokens = token_ids;
  }
  if (post_norm_hidden && aligned_hidden.shape()[1] > 0) {
    if (final_norm.ndim() != 1 || final_norm.shape()[0] != hidden_size) {
      throw std::runtime_error("invalid final norm for committed MTP history");
    }
    aligned_hidden = mx::fast::rms_norm(
        aligned_hidden, final_norm, rms_norm_eps);
  }
  return {aligned_hidden, aligned_tokens};
}

bool prepare_fused_q4_raw_decode_parameters(
    QLinear& gate, const QLinear& up) {
  if (!gate.valid || !up.valid || gate.bits != 4 || up.bits != 4 ||
      gate.group_size != 64 || up.group_size != 64 ||
      gate.scales.ndim() != 2 || gate.biases.ndim() != 2 ||
      up.scales.ndim() != 2 || up.biases.ndim() != 2 ||
      gate.scales.dtype() != activation_dtype() ||
      gate.biases.dtype() != activation_dtype() ||
      up.scales.dtype() != activation_dtype() ||
      up.biases.dtype() != activation_dtype() ||
      gate.scales.shape() != gate.biases.shape() ||
      gate.scales.shape() != up.scales.shape() ||
      gate.scales.shape() != up.biases.shape()) {
    return false;
  }
  const int output_features = gate.scales.shape()[0];
  const int groups_per_row = gate.scales.shape()[1];
  if (output_features <= 0 || output_features % 4 != 0 ||
      groups_per_row <= 0 || groups_per_row % 8 != 0) {
    return false;
  }

  mx::eval(gate.scales, gate.biases, up.scales, up.biases);
  const auto* gate_scales = gate.scales.data<std::uint16_t>();
  const auto* gate_biases = gate.biases.data<std::uint16_t>();
  const auto* up_scales = up.scales.data<std::uint16_t>();
  const auto* up_biases = up.biases.data<std::uint16_t>();
  const std::uint64_t pair_count = gate.scales.size();
  if (pair_count >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::size_t>::max() / 2) ||
      pair_count >
          static_cast<std::uint64_t>(
              std::numeric_limits<mx::ShapeElem>::max() / 2)) {
    return false;
  }
  std::vector<std::uint32_t> packed(
      static_cast<std::size_t>(pair_count) * 2);

  std::size_t destination = 0;
  for (int output_start = 0; output_start < output_features;
       output_start += 4) {
    for (int group = 0; group < groups_per_row; ++group) {
      for (int row = 0; row < 4; ++row) {
        const std::size_t source =
            static_cast<std::size_t>(output_start + row) * groups_per_row +
            group;
        packed[destination++] =
            static_cast<std::uint32_t>(gate_scales[source]) |
            (static_cast<std::uint32_t>(gate_biases[source]) << 16);
      }
      for (int row = 0; row < 4; ++row) {
        const std::size_t source =
            static_cast<std::size_t>(output_start + row) * groups_per_row +
            group;
        packed[destination++] =
            static_cast<std::uint32_t>(up_scales[source]) |
            (static_cast<std::uint32_t>(up_biases[source]) << 16);
      }
    }
  }
  if (destination != packed.size()) {
    throw std::runtime_error("fused Q4 parameter count mismatch");
  }
  gate.fused_q4_decode_params = array(
      packed.data(),
      {static_cast<mx::ShapeElem>(packed.size())},
      mx::uint32);
  mx::eval(gate.fused_q4_decode_params);
  gate.fused_q4_decode_params_valid = true;
  return true;
}

std::size_t release_fused_q4_raw_decode_parameters(QLinear& gate) {
  if (!gate.fused_q4_decode_params_valid) {
    if (gate.fused_q4_decode_params.ndim() != 0) {
      throw std::runtime_error("inconsistent raw fused Q4 decode parameters");
    }
    return 0;
  }
  if (gate.fused_q4_decode_params.ndim() != 1 ||
      gate.fused_q4_decode_params.dtype() != mx::uint32) {
    throw std::runtime_error("invalid raw fused Q4 decode parameters");
  }
  const std::size_t released_bytes = gate.fused_q4_decode_params.nbytes();
  gate.fused_q4_decode_params = array(0);
  gate.fused_q4_decode_params_valid = false;
  return released_bytes;
}

int dspark_select_verify_draft_tokens(
    const float* confidence,
    int count,
    float full_to_short_cost_ratio) {
  constexpr int kDraftTokens = 7;
  if (confidence == nullptr || count != kDraftTokens ||
      !std::isfinite(full_to_short_cost_ratio) ||
      full_to_short_cost_ratio <= 0.0f) {
    throw std::runtime_error("invalid DSpark confidence budget inputs");
  }
  double survival = 1.0;
  double short_expected_width = 1.0;
  double full_expected_width = 1.0;
  for (int index = 0; index < count; ++index) {
    const float value = confidence[index];
    if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
      throw std::runtime_error("invalid DSpark confidence budget inputs");
    }
    survival *= static_cast<double>(value);
    full_expected_width += survival;
    if (index == 0) {
      short_expected_width += survival;
    }
  }
  return full_expected_width >
          short_expected_width * full_to_short_cost_ratio
      ? kDraftTokens
      : 1;
}

std::pair<array, array> causal_conv_decode_silu(
    const array& state,
    const array& qkv,
    const array& weight) {
  int B = static_cast<int>(qkv.shape()[0]);
  int K = static_cast<int>(state.shape()[1]) + 1;
  int D = static_cast<int>(qkv.shape()[2]);
  auto outs = causal_conv_decode_silu_metal()(
      {state, qkv, weight},
      {{B, 1, D}, {B, K - 1, D}},
      {qkv.dtype(), state.dtype()},
      {D, B, 1},
      {std::min(D, 256), 1, 1},
      {
          {"InT", mx::fast::TemplateArg{qkv.dtype()}},
          {"K", mx::fast::TemplateArg{K}},
          {"D", mx::fast::TemplateArg{D}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> causal_conv_two_token_silu(
    const array& state,
    const array& qkv,
    const array& weight) {
  if (state.ndim() != 3 || qkv.ndim() != 3 || weight.ndim() != 3 ||
      qkv.shape()[1] != 2 || state.shape()[0] != qkv.shape()[0] ||
      state.shape()[2] != qkv.shape()[2] ||
      weight.shape()[0] != qkv.shape()[2] ||
      weight.shape()[1] != state.shape()[1] + 1 || weight.shape()[2] != 1 ||
      state.dtype() != qkv.dtype() || weight.dtype() != qkv.dtype()) {
    throw std::runtime_error("invalid two-token causal convolution inputs");
  }
  const int B = static_cast<int>(qkv.shape()[0]);
  const int K = static_cast<int>(state.shape()[1]) + 1;
  const int D = static_cast<int>(qkv.shape()[2]);
  if (B <= 0 || K < 2 || D <= 0) {
    throw std::runtime_error("invalid two-token causal convolution shape");
  }
  auto outs = causal_conv_two_token_silu_metal()(
      {state, qkv, weight},
      {{B, 2, D}, {B, K - 1, D}},
      {qkv.dtype(), state.dtype()},
      {D, B, 1},
      {std::min(D, 256), 1, 1},
      {
          {"InT", mx::fast::TemplateArg{qkv.dtype()}},
          {"K", mx::fast::TemplateArg{K}},
          {"D", mx::fast::TemplateArg{D}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> residual_rms_norm(
    const array& x,
    const array& residual,
    const array& weight,
    float eps) {
  if (x.shape() != residual.shape() || x.ndim() == 0 || weight.ndim() != 1 ||
      x.shape().back() != weight.shape()[0] || x.dtype() != residual.dtype() ||
      x.dtype() != weight.dtype()) {
    throw std::runtime_error("invalid residual RMSNorm inputs");
  }
  const int D = static_cast<int>(x.shape().back());
  const int rows = static_cast<int>(x.size() / static_cast<size_t>(D));
  int threads = 1024;
  if (D <= 4096) {
    const int threads_needed = (D + 3) / 4;
    threads = ((threads_needed + 31) / 32) * 32;
  }
  auto outs = residual_rms_norm_metal()(
      {x, residual, weight, array(eps)},
      {x.shape(), x.shape()},
      {x.dtype(), x.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{x.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> normalize_gated_delta_qk(
    const array& q,
    const array& k,
    float q_scale,
    float k_scale,
    float eps) {
  if (q.shape() != k.shape() || q.ndim() == 0 || q.dtype() != k.dtype()) {
    throw std::runtime_error("invalid gated-delta q/k normalization inputs");
  }
  const int D = static_cast<int>(q.shape().back());
  if (D <= 0 || D > 4096) {
    throw std::runtime_error("gated-delta q/k normalization width out of range");
  }
  const int rows = static_cast<int>(q.size() / static_cast<size_t>(D));
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  auto outs = gated_delta_qk_norm_metal()(
      {q, k, array(q_scale), array(k_scale), array(eps)},
      {q.shape(), k.shape()},
      {mx::float32, mx::float32},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{q.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

std::pair<array, array> full_attn_qk_norm_rope(
    const array& qg,
    const array& k,
    const array& q_weight,
    const array& k_weight,
    float eps,
    float rope_theta,
    int rope_dims,
    int rope_offset) {
  if (qg.ndim() != 4 || k.ndim() != 4 || qg.shape()[0] != k.shape()[0] ||
      qg.shape()[1] != 1 || k.shape()[1] != 1 ||
      qg.shape().back() != 2 * k.shape().back() ||
      q_weight.ndim() != 1 || k_weight.ndim() != 1 ||
      q_weight.shape()[0] != k.shape().back() ||
      k_weight.shape()[0] != k.shape().back() || qg.dtype() != k.dtype() ||
      qg.dtype() != q_weight.dtype() || qg.dtype() != k_weight.dtype()) {
    throw std::runtime_error("invalid full-attention q/k norm/RoPE inputs");
  }
  const int B = static_cast<int>(qg.shape()[0]);
  const int Nq = static_cast<int>(qg.shape()[2]);
  const int Nk = static_cast<int>(k.shape()[2]);
  const int D = static_cast<int>(k.shape().back());
  if (D <= 0 || D > 4096 || rope_dims <= 0 || rope_dims > D ||
      rope_dims % 2 != 0) {
    throw std::runtime_error("full-attention q/k norm/RoPE width out of range");
  }
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  const int rows = B * (Nq + Nk);
  auto outs = full_attn_qk_norm_rope_metal()(
      {qg,
       k,
       q_weight,
       k_weight,
       array(eps),
       array(std::log2(rope_theta)),
       array(rope_offset)},
      {{B, Nq, 1, D}, {B, Nk, 1, D}},
      {qg.dtype(), k.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"InT", mx::fast::TemplateArg{qg.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"RopeDims", mx::fast::TemplateArg{rope_dims}},
          {"Nq", mx::fast::TemplateArg{Nq}},
          {"Nk", mx::fast::TemplateArg{Nk}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return {outs[0], outs[1]};
}

array gated_delta_norm_gate(
    const array& recurrent_out,
    const array& z,
    const array& weight,
    float eps) {
  if (recurrent_out.shape() != z.shape() || recurrent_out.ndim() == 0 ||
      weight.ndim() != 1 || recurrent_out.shape().back() != weight.shape()[0] ||
      recurrent_out.dtype() != mx::float32 || z.dtype() != weight.dtype()) {
    throw std::runtime_error("invalid gated-delta norm/gate inputs");
  }
  const int D = static_cast<int>(recurrent_out.shape().back());
  if (D <= 0 || D > 4096) {
    throw std::runtime_error("gated-delta norm/gate width out of range");
  }
  const int rows =
      static_cast<int>(recurrent_out.size() / static_cast<size_t>(D));
  const int threads_needed = (D + 3) / 4;
  const int threads = ((threads_needed + 31) / 32) * 32;
  auto outs = gated_delta_norm_gate_metal()(
      {recurrent_out, z, weight, array(eps)},
      {z.shape()},
      {z.dtype()},
      {rows * threads, 1, 1},
      {threads, 1, 1},
      {
          {"GateT", mx::fast::TemplateArg{z.dtype()}},
          {"D", mx::fast::TemplateArg{D}},
          {"Threads", mx::fast::TemplateArg{threads}},
      },
      std::nullopt,
      false,
      {});
  return outs[0];
}

array silu(const array& x) {
  return x * sigmoid(x);
}

array softplus(const array& x) {
  return logaddexp(x, array(0.0f));
}

array dspark_yarn_rope(const array& x, int offset) {
  constexpr int kHeadDim = 128;
  constexpr int kFrequencyCount = kHeadDim / 2;
  constexpr double kBase = 10000000.0;
  constexpr double kFactor = 32.0;
  constexpr double kOriginalPositions = 8192.0;
  constexpr double kBetaFast = 32.0;
  constexpr double kBetaSlow = 1.0;
  if (x.ndim() == 0 || x.shape().back() != kHeadDim || offset < 0) {
    throw std::runtime_error("invalid DSpark YaRN RoPE input");
  }

  static const array frequencies = [] {
    const auto correction_dimension = [](double rotations) {
      return kHeadDim *
          std::log(kOriginalPositions / (rotations * 2.0 * std::numbers::pi)) /
          (2.0 * std::log(kBase));
    };
    const int low = std::max(
        static_cast<int>(std::floor(correction_dimension(kBetaFast))), 0);
    const int high = std::min(
        static_cast<int>(std::ceil(correction_dimension(kBetaSlow))),
        kHeadDim - 1);
    std::vector<float> values(kFrequencyCount);
    for (int index = 0; index < kFrequencyCount; ++index) {
      const double position_frequency =
          std::pow(kBase, (2.0 * index) / kHeadDim);
      const double extrapolated = 1.0 / position_frequency;
      const double interpolated = extrapolated / kFactor;
      const double ramp = std::clamp(
          (static_cast<double>(index) - low) /
              static_cast<double>(high - low),
          0.0,
          1.0);
      const double extrapolation_mask = 1.0 - ramp;
      const double inverse_frequency =
          interpolated * (1.0 - extrapolation_mask) +
          extrapolated * extrapolation_mask;
      values[static_cast<size_t>(index)] =
          static_cast<float>(1.0 / inverse_frequency);
    }
    array result(values.begin(), {kFrequencyCount}, mx::float32);
    eval(result);
    return result;
  }();
  static const float attention_factor =
      static_cast<float>(0.1 * std::log(kFactor) + 1.0);
  array rotated = mx::fast::rope(
      x,
      kHeadDim,
      /*traditional=*/false,
      std::nullopt,
      1.0f,
      offset,
      frequencies);
  return rotated * array(attention_factor, rotated.dtype());
}

array dspark_confidence(
    const array& hidden,
    const array& markov_embeddings,
    const array& weight,
    const array& bias) {
  if (hidden.ndim() != 3 || markov_embeddings.ndim() != 3 ||
      hidden.shape()[0] != markov_embeddings.shape()[0] ||
      hidden.shape()[1] != markov_embeddings.shape()[1] ||
      hidden.dtype() != activation_dtype() ||
      markov_embeddings.dtype() != activation_dtype() ||
      weight.dtype() != activation_dtype() || bias.dtype() != activation_dtype() ||
      weight.shape() != mx::Shape{
          1, hidden.shape()[2] + markov_embeddings.shape()[2]} ||
      bias.shape() != mx::Shape{1}) {
    throw std::runtime_error("invalid DSpark confidence inputs");
  }
  array features = concatenate({hidden, markov_embeddings}, -1);
  array raw = mx::matmul(features, transpose(weight)) + bias;
  return squeeze(sigmoid(astype(raw, mx::float32)), -1);
}

array QLinear::operator()(const array& x) const {
  if (!valid) {
    throw std::runtime_error("QLinear used before load");
  }
  if (w.dtype() == activation_dtype()) {
    if (x.ndim() == 0 || x.dtype() != activation_dtype() || w.ndim() != 2 ||
        x.shape().back() != w.shape()[1]) {
      throw std::runtime_error("invalid dense QLinear inputs");
    }
    return matmul(x, transpose(w, {1, 0}));
  }
  if (w.dtype() != mx::uint32) {
    throw std::runtime_error("unsupported QLinear weight dtype");
  }
  if (native_q4_batch_one_qmv_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 1 &&
      x.dtype() == activation_dtype() && w.ndim() == 2 && scales.ndim() == 2 &&
      biases.ndim() == 2 && scales.dtype() == activation_dtype() &&
      biases.dtype() == activation_dtype() && group_size == 64 && bits == 4) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (input_features > 0 && output_features > 0 &&
        input_features % 512 == 0 && output_features % 16 == 0 &&
        w.shape()[1] * 8 == input_features &&
        scales.shape() == mx::Shape{output_features, input_features / 64} &&
        biases.shape() == scales.shape()) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmv q4 rows=1 K=%d N=%d geometry=4x4x2-fixed\n",
            input_features,
            output_features);
      }
      return affine_q4_qmv_batch_one(*this, x);
    }
  }
  if (native_q4_batch_two_qmv_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 2 &&
      x.dtype() == activation_dtype() && w.ndim() == 2 && scales.ndim() == 2 &&
      biases.ndim() == 2 && scales.dtype() == activation_dtype() &&
      biases.dtype() == activation_dtype() && group_size == 64 && bits == 4) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (input_features > 0 && output_features > 0 &&
        input_features % 512 == 0 && output_features % 16 == 0 &&
        w.shape()[1] * 8 == input_features &&
        scales.shape() == mx::Shape{output_features, input_features / 64} &&
        biases.shape() == scales.shape()) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmv q4 rows=2 K=%d N=%d geometry=4x4x2-shared\n",
            input_features,
            output_features);
      }
      return affine_q4_qmv_batch_two(*this, x);
    }
  }
  if (native_q4_batch_three_qmv_enabled() &&
      supports_q4_batch_three(*this, x)) {
    return affine_q4_qmv_batch_three(*this, x);
  }
  if (native_small_batch_qmm_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] >= 6 && x.shape()[1] <= 8) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (native_m8_ksplit_qmm_enabled() && x.shape()[1] == 8 &&
        x.dtype() == activation_dtype() && w.dtype() == mx::uint32 &&
        scales.dtype() == activation_dtype() && biases.dtype() == activation_dtype() &&
        group_size == 64 && (bits == 4 || bits == 5) &&
        input_features % 512 == 0 &&
        output_features % 32 == 0) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmm m8_ksplit rows=8 K=%d N=%d bits=%d\n",
            input_features,
            output_features,
            bits);
      }
      return affine_qmm_m8_ksplit(*this, x);
    }
    if (x.dtype() == activation_dtype() && w.dtype() == mx::uint32 &&
        scales.dtype() == activation_dtype() && biases.dtype() == activation_dtype() &&
        group_size == 64 && (bits == 2 || bits == 4) &&
        input_features % 64 == 0 && output_features % 64 == 0 &&
        output_features >= 6144) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmm custom rows=%d K=%d N=%d bits=%d\n",
            static_cast<int>(x.shape()[1]),
            input_features,
            output_features,
            bits);
      }
      return affine_qmm_small_batch(*this, x);
    }
    if (native_qmm_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_qmm fallback rows=%d K=%d N=%d bits=%d\n",
          static_cast<int>(x.shape()[1]),
          input_features,
          output_features,
          bits);
    }
  }
  if (native_q5_batch_one_qmv_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 1 &&
      x.dtype() == activation_dtype() && w.ndim() == 2 &&
      scales.ndim() == 2 && biases.ndim() == 2 &&
      scales.dtype() == activation_dtype() && biases.dtype() == activation_dtype() &&
      group_size == 64 && bits == 5) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (input_features > 0 && output_features > 0 &&
        input_features % 512 == 0 && output_features % 16 == 0 &&
        w.shape()[1] * 32 == input_features * 5 &&
        scales.shape() == mx::Shape{output_features, input_features / 64} &&
        biases.shape() == scales.shape()) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmv q5 rows=1 K=%d N=%d geometry=4x4x2-fixed\n",
            input_features,
            output_features);
      }
      return affine_q5_qmv_batch_one(*this, x);
    }
  }
  if (native_q5_batch_two_qmv_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 2 &&
      x.dtype() == activation_dtype() && w.ndim() == 2 &&
      scales.ndim() == 2 && biases.ndim() == 2 &&
      scales.dtype() == activation_dtype() && biases.dtype() == activation_dtype() &&
      group_size == 64 && bits == 5) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (input_features > 0 && output_features > 0 &&
        input_features % 512 == 0 && output_features % 16 == 0 &&
        w.shape()[1] * 32 == input_features * 5 &&
        scales.shape() ==
            mx::Shape{output_features, input_features / 64} &&
        biases.shape() == scales.shape()) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(
            stderr,
            "qwen38_qmv q5 rows=2 K=%d N=%d geometry=4x4x2-shared\n",
            input_features,
            output_features);
      }
      return affine_q5_qmv_batch_two(*this, x);
    }
  }
  if (native_q5_multirow_qmv_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 3 &&
      x.dtype() == activation_dtype() && w.ndim() == 2 && w.dtype() == mx::uint32 &&
      scales.ndim() == 2 && biases.ndim() == 2 &&
      scales.dtype() == activation_dtype() && biases.dtype() == activation_dtype() &&
      group_size == 64 && bits == 5) {
    const int input_features = static_cast<int>(x.shape()[2]);
    const int output_features = static_cast<int>(w.shape()[0]);
    if (input_features > 0 && output_features > 0 &&
        input_features % 512 == 0 && output_features % 16 == 0 &&
        w.shape()[1] * 32 == input_features * 5 &&
        scales.shape() == mx::Shape{output_features, input_features / 64} &&
        biases.shape() == scales.shape()) {
      if (native_qmm_trace_enabled()) {
        std::fprintf(stderr, "qwen38_qmv q5 rows=%d K=%d N=%d shared\n",
                     static_cast<int>(x.shape()[1]), input_features,
                     output_features);
      }
      return affine_q5_qmv_multirow(*this, x);
    }
  }
  return quantized_matmul(
      x, w, scales, biases, /*transpose=*/true, group_size, bits, "affine");
}

array quantized_embedding_rows(
    const QLinear& embedding, const array& tokens) {
  if (!embedding.valid || embedding.w.ndim() != 2 ||
      embedding.scales.ndim() != 2 || embedding.biases.ndim() != 2 ||
      embedding.w.dtype() != mx::uint32 ||
      embedding.scales.dtype() != activation_dtype() ||
      embedding.biases.dtype() != activation_dtype() ||
      embedding.w.shape()[0] != embedding.scales.shape()[0] ||
      embedding.scales.shape() != embedding.biases.shape() ||
      embedding.group_size <= 0 || embedding.bits <= 0 ||
      (tokens.dtype() != mx::int32 && tokens.dtype() != mx::uint32)) {
    throw std::runtime_error("invalid quantized embedding inputs");
  }
  const int hidden_size =
      embedding.scales.shape()[1] * embedding.group_size;
  if (hidden_size <= 0 ||
      embedding.w.shape()[1] * 32 != hidden_size * embedding.bits) {
    throw std::runtime_error("unsupported quantized embedding shape");
  }
  return mx::dequantize(
      take(embedding.w, tokens, 0),
      take(embedding.scales, tokens, 0),
      take(embedding.biases, tokens, 0),
      embedding.group_size,
      embedding.bits,
      "affine",
      std::nullopt,
      activation_dtype());
}

array fixed_prefill_attention(
    const array& queries,
    const array& key_cache,
    const array& value_cache,
    int prefix_length,
    int active_cache_length) {
  constexpr int kQueryHeads = 24;
  constexpr int kKeyValueHeads = 4;
  constexpr int kHeadDimension = 256;
  constexpr int kHeadsPerKeyValue = kQueryHeads / kKeyValueHeads;
  constexpr int kQueryTile = 8;
  constexpr int kKeyTile = 64;
  constexpr int kThreads = 128;
  if (queries.ndim() != 4 || key_cache.ndim() != 4 ||
      value_cache.ndim() != 4 || queries.dtype() != activation_dtype() ||
      key_cache.dtype() != activation_dtype() ||
      value_cache.dtype() != activation_dtype() ||
      queries.shape()[0] != 1 || queries.shape()[1] != kQueryHeads ||
      queries.shape()[3] != kHeadDimension ||
      key_cache.shape()[0] != 1 ||
      key_cache.shape()[1] != kKeyValueHeads ||
      key_cache.shape()[3] != kHeadDimension ||
      key_cache.shape() != value_cache.shape()) {
    throw std::runtime_error("invalid fixed prefill attention inputs");
  }
  const int query_tokens = queries.shape()[2];
  const int cache_capacity = key_cache.shape()[2];
  if (query_tokens < 1 || query_tokens > 1024 || prefix_length < 0 ||
      active_cache_length != prefix_length + query_tokens ||
      active_cache_length > cache_capacity ||
      cache_capacity % kKeyTile != 0) {
    throw std::runtime_error("unsupported fixed prefill attention shape");
  }
  const int attention_rows = query_tokens * kHeadsPerKeyValue;
  const int query_tiles =
      (attention_rows + kQueryTile - 1) / kQueryTile;
  auto outputs = fixed_prefill_attention_metal()(
      {queries,
       key_cache,
       value_cache,
       array(query_tokens, mx::int32),
       array(prefix_length, mx::int32),
       array(cache_capacity, mx::int32),
       array(active_cache_length, mx::int32)},
      {queries.shape()},
      {queries.dtype()},
      {query_tiles * kThreads, kKeyValueHeads, 1},
      {kThreads, 1, 1},
      {},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array fixed_q8_attention(
    const array& queries,
    const array& key_cache,
    const array& key_scales,
    const array& key_biases,
    const array& value_cache,
    const array& value_scales,
    const array& value_biases,
    int prefix_length,
    int active_cache_length) {
  constexpr int kQueryHeads = 24;
  constexpr int kKeyValueHeads = 4;
  constexpr int kHeadDimension = 256;
  constexpr int kPackedDimension = kHeadDimension / 4;
  constexpr int kParameterDimension = kHeadDimension / 64;
  constexpr int kHeadsPerKeyValue = kQueryHeads / kKeyValueHeads;
  constexpr int kQueryTile = 8;
  constexpr int kKeyTile = 64;
  constexpr int kThreads = 128;
  if (queries.ndim() != 4 || key_cache.ndim() != 4 ||
      key_scales.ndim() != 4 || key_biases.ndim() != 4 ||
      value_cache.ndim() != 4 || value_scales.ndim() != 4 ||
      value_biases.ndim() != 4 || queries.dtype() != activation_dtype() ||
      key_cache.dtype() != mx::uint32 ||
      value_cache.dtype() != mx::uint32 ||
      key_scales.dtype() != activation_dtype() ||
      key_biases.dtype() != activation_dtype() ||
      value_scales.dtype() != activation_dtype() ||
      value_biases.dtype() != activation_dtype() || queries.shape()[0] != 1 ||
      queries.shape()[1] != kQueryHeads ||
      queries.shape()[3] != kHeadDimension || key_cache.shape()[0] != 1 ||
      key_cache.shape()[1] != kKeyValueHeads ||
      key_cache.shape()[3] != kPackedDimension ||
      key_cache.shape() != value_cache.shape() ||
      key_scales.shape()[0] != 1 ||
      key_scales.shape()[1] != kKeyValueHeads ||
      key_scales.shape()[3] != kParameterDimension ||
      key_scales.shape() != key_biases.shape() ||
      key_scales.shape() != value_scales.shape() ||
      key_scales.shape() != value_biases.shape() ||
      key_cache.shape()[2] != key_scales.shape()[2]) {
    throw std::runtime_error("invalid fixed Q8 attention inputs");
  }
  const int query_tokens = queries.shape()[2];
  const int cache_capacity = key_cache.shape()[2];
  if (query_tokens < 1 || query_tokens > 1024 || prefix_length < 0 ||
      active_cache_length != prefix_length + query_tokens ||
      active_cache_length > cache_capacity ||
      cache_capacity % kKeyTile != 0) {
    throw std::runtime_error("unsupported fixed Q8 attention shape");
  }
  const char* segmented = std::getenv("SGLANG_MLX_NATIVE_Q8_SEGMENTED_MATMUL");
  constexpr int kSegmentTokens = 1024;
  if (segmented != nullptr && std::string_view(segmented) == "1" &&
      query_tokens <= 8 && active_cache_length >= 4096 &&
      cache_capacity % kSegmentTokens == 0) {
    const int extent = ((active_cache_length + kSegmentTokens - 1) /
                        kSegmentTokens) * kSegmentTokens;
    const int rows = query_tokens * kHeadsPerKeyValue;
    const int segments = extent / kSegmentTokens;
    const auto active = [extent](const array& cache) {
      return mx::slice(cache, {0, 0, 0, 0},
          {1, kKeyValueHeads, extent, cache.shape()[3]});
    };
    const array positions = mx::arange(extent, mx::int32);
    const array valid = reshape(mx::less(positions, array(active_cache_length)),
                                {1, 1, extent, 1});
    // An abandoned speculative suffix may contain NaN coefficients. Remove
    // those values before either product, including the zero-probability PV.
    const auto coefficients = [&](const array& parameter) {
      return mx::where(valid, astype(active(parameter), mx::float32), array(0.0f));
    };
    const array grouped_queries = reshape(transpose(
        reshape(queries, {1, kKeyValueHeads, kHeadsPerKeyValue,
                          query_tokens, kHeadDimension}), {0, 1, 3, 2, 4}),
        {1, kKeyValueHeads, rows, kHeadDimension});
    const array scores = quantized_matmul(astype(grouped_queries, mx::float32),
        active(key_cache), coefficients(key_scales), coefficients(key_biases),
        true, 64, 8, "affine");
    std::vector<int> limits(static_cast<std::size_t>(rows));
    for (int row = 0; row < rows; ++row)
      limits[row] = prefix_length + row / kHeadsPerKeyValue;
    const array causal = mx::less_equal(
        reshape(positions, {1, 1, 1, extent}),
        array(limits.data(), {1, 1, rows, 1}, mx::int32));
    const array probabilities = mx::softmax(mx::where(causal, scores * 0.0625f,
        array(-std::numeric_limits<float>::infinity())), -1, true);
    const array partitioned = transpose(
        reshape(probabilities, {1, kKeyValueHeads, rows, segments, kSegmentTokens}),
        {0, 1, 3, 2, 4});
    // Independent sequence segments expose enough PV workgroups for the GPU.
    // Both softmax and the final segment reduction retain FP32 arithmetic.
    const mx::Shape payload_shape{
        1, kKeyValueHeads, segments, kSegmentTokens, kPackedDimension};
    const mx::Shape parameter_shape{
        1, kKeyValueHeads, segments, kSegmentTokens, kParameterDimension};
    const array partial = quantized_matmul(partitioned,
        reshape(active(value_cache), payload_shape),
        reshape(coefficients(value_scales), parameter_shape),
        reshape(coefficients(value_biases), parameter_shape),
        false, 64, 8, "affine");
    const array output = reshape(transpose(
        reshape(sum(partial, 2), {1, kKeyValueHeads, query_tokens,
                                 kHeadsPerKeyValue, kHeadDimension}),
        {0, 1, 3, 2, 4}), queries.shape());
    return astype(output, queries.dtype());
  }

  const int attention_rows = query_tokens * kHeadsPerKeyValue;
  const int query_tiles =
      (attention_rows + kQueryTile - 1) / kQueryTile;
  const int requested_key_splits = std::min(
      32, std::max(1, (cache_capacity + 4095) / 4096));
  const bool split_decode = requested_key_splits > 1 &&
      (query_tokens == 1 ||
       ((native_q8_split_verify_enabled() || native_q8_tiled_attention_enabled()) && query_tokens <= 8));
  const auto& kernel = native_q8_tiled_attention_enabled()
      ? tiled_q8_attention_metal() : fixed_q8_attention_metal();
  auto outputs = kernel(
      {queries,
       key_cache,
       key_scales,
       key_biases,
       value_cache,
       value_scales,
       value_biases,
       array(query_tokens, mx::int32),
       array(prefix_length, mx::int32),
       array(cache_capacity, mx::int32),
       array(active_cache_length, mx::int32),
       array(split_decode ? requested_key_splits : 1, mx::int32)},
      {split_decode
           ? mx::Shape{requested_key_splits, kQueryHeads, query_tokens, kHeadDimension + 2}
           : queries.shape()},
      {mx::float32},
      {query_tiles * (split_decode ? requested_key_splits : 1) * kThreads,
       kKeyValueHeads,
       1},
      {kThreads, 1, 1},
      {},
      std::nullopt,
      false,
      {});
  if (split_decode) {
    auto reduced = reduce_q8_decode_metal()(
        {outputs[0],
         array(active_cache_length, mx::int32),
         array(requested_key_splits, mx::int32),
         array(query_tokens, mx::int32)},
        {queries.shape()},
        {queries.dtype()},
        {kQueryHeads * query_tokens * kHeadDimension, 1, 1},
        {kHeadDimension, 1, 1},
        {},
        std::nullopt,
        false,
        {});
    return reduced[0];
  }
  return astype(outputs[0], queries.dtype());
}

array affine_qmm_small_batch(const QLinear& linear, const array& x) {
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] < 2 || x.shape()[1] > 8 ||
      x.dtype() != activation_dtype() || linear.w.dtype() != mx::uint32 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      (linear.bits != 2 && linear.bits != 4)) {
    throw std::runtime_error("invalid small-batch affine QMM inputs");
  }
  const int rows = static_cast<int>(x.shape()[1]);
  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features % 64 != 0 || output_features % 64 != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported small-batch affine QMM shape");
  }

  auto outputs = affine_small_batch_qmm_metal()(
      {linear.w,
       linear.scales,
       linear.biases,
       x,
       array(input_features, mx::int32),
       array(output_features, mx::int32),
       array(rows, mx::int32)},
      {{1, rows, output_features}},
      {x.dtype()},
      {output_features / 64 * 128, 1, 1},
      {128, 1, 1},
      {{"Bits", mx::fast::TemplateArg{linear.bits}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_qmm_m8_ksplit(const QLinear& linear, const array& x) {
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 8 || x.dtype() != activation_dtype() ||
      linear.w.dtype() != mx::uint32 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      (linear.bits != 4 && linear.bits != 5)) {
    throw std::runtime_error("invalid M8 K-split affine QMM inputs");
  }
  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features % 512 != 0 || output_features % 32 != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() != mx::Shape{
          output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported M8 K-split affine QMM shape");
  }

  auto outputs = affine_m8_ksplit_qmm_metal()(
      {linear.w,
       linear.scales,
       linear.biases,
       x,
       array(output_features, mx::int32)},
      {{1, 8, output_features}},
      {x.dtype()},
      {512, output_features / 32, 1},
      {512, 1, 1},
      {
          {"KConst", mx::fast::TemplateArg{input_features}},
          {"Bits", mx::fast::TemplateArg{linear.bits}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q4_qmv_batch_one(const QLinear& linear, const array& x) {
  constexpr int kSimdGroups = 4;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 1 || x.dtype() != activation_dtype() ||
      linear.w.ndim() != 2 || linear.w.dtype() != mx::uint32 ||
      linear.scales.ndim() != 2 || linear.biases.ndim() != 2 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      linear.bits != 4) {
    throw std::runtime_error("invalid batch-one affine Q4 QMV inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      linear.w.shape()[1] * 8 != input_features ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported batch-one affine Q4 QMV shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  auto outputs = affine_q4_batch_one_qmv_metal()(
      {linear.w, linear.scales, linear.biases, x},
      {{1, 1, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {{"KConst", mx::fast::TemplateArg{input_features}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q4_qmv_batch_two(const QLinear& linear, const array& x) {
  constexpr int kSimdGroups = 4;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 2 || x.dtype() != activation_dtype() ||
      linear.w.ndim() != 2 || linear.w.dtype() != mx::uint32 ||
      linear.scales.ndim() != 2 || linear.biases.ndim() != 2 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      linear.bits != 4) {
    throw std::runtime_error("invalid batch-two affine Q4 QMV inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      linear.w.shape()[1] * 8 != input_features ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported batch-two affine Q4 QMV shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  auto outputs = affine_q4_batch_two_qmv_metal()(
      {linear.w, linear.scales, linear.biases, x},
      {{1, 2, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {
          {"KConst", mx::fast::TemplateArg{input_features}},
          {"NConst", mx::fast::TemplateArg{output_features}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q4_qmv_batch_three(const QLinear& linear, const array& x) {
  if (!supports_q4_batch_three(linear, x)) {
    throw std::runtime_error("unsupported batch-three affine Q4 QMV inputs");
  }
  const int k = x.shape()[2];
  const int n = linear.w.shape()[0];
  return affine_q4_batch_three_metal()(
      {linear.w, linear.scales, linear.biases, x},
      {{1, 3, n}}, {x.dtype()}, {128, n / 8, 1}, {128, 1, 1},
      {{"KConst", mx::fast::TemplateArg{k}},
       {"NConst", mx::fast::TemplateArg{n}}},
      std::nullopt, false, {})[0];
}

array affine_q4_fused_swiglu_batch_one(
    const QLinear& gate, const QLinear& up, const array& x) {
  constexpr int kSimdGroups = 8;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!gate.valid || !up.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 1 || x.dtype() != activation_dtype() ||
      gate.w.ndim() != 2 || up.w.ndim() != 2 ||
      gate.w.dtype() != mx::uint32 || up.w.dtype() != mx::uint32 ||
      gate.scales.ndim() != 2 || gate.biases.ndim() != 2 ||
      up.scales.ndim() != 2 || up.biases.ndim() != 2 ||
      gate.scales.dtype() != activation_dtype() ||
      gate.biases.dtype() != activation_dtype() ||
      up.scales.dtype() != activation_dtype() ||
      up.biases.dtype() != activation_dtype() || gate.group_size != 64 ||
      up.group_size != 64 || gate.bits != 4 || up.bits != 4) {
    throw std::runtime_error("invalid batch-one affine Q4 fused SwiGLU inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(gate.w.shape()[0]);
  const mx::Shape expected_weight_shape{
      output_features, input_features / 8};
  const mx::Shape expected_parameter_shape{
      output_features, input_features / 64};
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      gate.w.shape() != expected_weight_shape ||
      up.w.shape() != expected_weight_shape ||
      gate.scales.shape() != expected_parameter_shape ||
      gate.biases.shape() != expected_parameter_shape ||
      up.scales.shape() != expected_parameter_shape ||
      up.biases.shape() != expected_parameter_shape) {
    throw std::runtime_error("unsupported batch-one affine Q4 fused SwiGLU shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  if (gate.fused_q4_decode_params_valid) {
    const std::size_t expected_parameter_words =
        gate.scales.size() * 2;
    if (gate.fused_q4_decode_params.ndim() != 1 ||
        gate.fused_q4_decode_params.dtype() != mx::uint32 ||
        gate.fused_q4_decode_params.size() != expected_parameter_words) {
      throw std::runtime_error("invalid raw fused Q4 decode parameters");
    }
    auto outputs = affine_q4_fused_swiglu_raw_params_metal()(
        {gate.w, up.w, gate.fused_q4_decode_params, x},
        {{1, 1, output_features}},
        {x.dtype()},
        {kThreads, output_features / kOutputTile, 1},
        {kThreads, 1, 1},
        {{"KConst", mx::fast::TemplateArg{input_features}}},
        std::nullopt,
        false,
        {});
    return outputs[0];
  }
  auto outputs = affine_q4_fused_swiglu_metal()(
      {gate.w,
       gate.scales,
       gate.biases,
       up.w,
       up.scales,
       up.biases,
       x},
      {{1, 1, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {{"KConst", mx::fast::TemplateArg{input_features}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q4_fused_swiglu_batch_two(
    const QLinear& gate, const QLinear& up, const array& x) {
  constexpr int kSimdGroups = 8;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!gate.valid || !up.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 2 || x.dtype() != activation_dtype() ||
      gate.w.ndim() != 2 || up.w.ndim() != 2 ||
      gate.w.dtype() != mx::uint32 || up.w.dtype() != mx::uint32 ||
      gate.scales.ndim() != 2 || gate.biases.ndim() != 2 ||
      up.scales.ndim() != 2 || up.biases.ndim() != 2 ||
      gate.scales.dtype() != activation_dtype() ||
      gate.biases.dtype() != activation_dtype() ||
      up.scales.dtype() != activation_dtype() ||
      up.biases.dtype() != activation_dtype() || gate.group_size != 64 ||
      up.group_size != 64 || gate.bits != 4 || up.bits != 4 ||
      !gate.fused_q4_decode_params_valid) {
    throw std::runtime_error("invalid batch-two affine Q4 fused SwiGLU inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(gate.w.shape()[0]);
  const mx::Shape expected_weight_shape{
      output_features, input_features / 8};
  const mx::Shape expected_parameter_shape{
      output_features, input_features / 64};
  const std::size_t expected_parameter_words = gate.scales.size() * 2;
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      gate.w.shape() != expected_weight_shape ||
      up.w.shape() != expected_weight_shape ||
      gate.scales.shape() != expected_parameter_shape ||
      gate.biases.shape() != expected_parameter_shape ||
      up.scales.shape() != expected_parameter_shape ||
      up.biases.shape() != expected_parameter_shape ||
      gate.fused_q4_decode_params.ndim() != 1 ||
      gate.fused_q4_decode_params.dtype() != mx::uint32 ||
      gate.fused_q4_decode_params.size() != expected_parameter_words) {
    throw std::runtime_error(
        "unsupported batch-two affine Q4 fused SwiGLU shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  auto outputs = affine_q4_fused_swiglu_batch_two_raw_params_metal()(
      {gate.w, up.w, gate.fused_q4_decode_params, x},
      {{1, 2, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {
          {"KConst", mx::fast::TemplateArg{input_features}},
          {"NConst", mx::fast::TemplateArg{output_features}},
          {"ScalarInputs", mx::fast::TemplateArg{
               native_q4_fused_swiglu_batch_two_scalar_inputs_enabled()}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q5_qmv_batch_one(const QLinear& linear, const array& x) {
  constexpr int kSimdGroups = 4;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 1 || x.dtype() != activation_dtype() ||
      linear.w.ndim() != 2 || linear.w.dtype() != mx::uint32 ||
      linear.scales.ndim() != 2 || linear.biases.ndim() != 2 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      linear.bits != 5) {
    throw std::runtime_error("invalid batch-one affine Q5 QMV inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported batch-one affine Q5 QMV shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  auto outputs = affine_q5_batch_one_qmv_metal()(
      {linear.w, linear.scales, linear.biases, x},
      {{1, 1, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {{"KConst", mx::fast::TemplateArg{input_features}}},
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q5_qmv_batch_two(const QLinear& linear, const array& x) {
  constexpr int kSimdGroups = 4;
  constexpr int kResultsPerSimdgroup = 4;
  constexpr int kPacksPerThread = 2;
  constexpr int kOutputTile = kSimdGroups * kResultsPerSimdgroup;
  constexpr int kBlockSize = kPacksPerThread * 8 * 32;
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 2 || x.dtype() != activation_dtype() ||
      linear.w.ndim() != 2 || linear.w.dtype() != mx::uint32 ||
      linear.scales.ndim() != 2 || linear.biases.ndim() != 2 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      linear.bits != 5) {
    throw std::runtime_error("invalid batch-two affine Q5 QMV inputs");
  }

  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  if (input_features <= 0 || output_features <= 0 ||
      input_features % kBlockSize != 0 ||
      output_features % kOutputTile != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() !=
          mx::Shape{output_features, input_features / linear.group_size} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported batch-two affine Q5 QMV shape");
  }

  constexpr int kThreads = kSimdGroups * 32;
  auto outputs = affine_q5_batch_two_qmv_metal()(
      {linear.w, linear.scales, linear.biases, x},
      {{1, 2, output_features}},
      {x.dtype()},
      {kThreads, output_features / kOutputTile, 1},
      {kThreads, 1, 1},
      {
          {"KConst", mx::fast::TemplateArg{input_features}},
          {"NConst", mx::fast::TemplateArg{output_features}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

array affine_q5_qmv_multirow(const QLinear& linear, const array& x) {
  if (!linear.valid || x.ndim() != 3 || x.shape()[0] != 1 ||
      x.shape()[1] != 3 || x.dtype() != activation_dtype() ||
      linear.w.ndim() != 2 || linear.w.dtype() != mx::uint32 ||
      linear.scales.ndim() != 2 || linear.biases.ndim() != 2 ||
      linear.scales.dtype() != activation_dtype() ||
      linear.biases.dtype() != activation_dtype() || linear.group_size != 64 ||
      linear.bits != 5) {
    throw std::runtime_error("invalid multirow affine Q5 QMV inputs");
  }
  const int input_features = static_cast<int>(x.shape()[2]);
  const int output_features = static_cast<int>(linear.w.shape()[0]);
  const int rows = static_cast<int>(x.shape()[1]);
  if (input_features <= 0 || output_features <= 0 ||
      input_features % 512 != 0 || output_features % 16 != 0 ||
      linear.w.shape()[1] * 32 != input_features * linear.bits ||
      linear.scales.shape() != mx::Shape{output_features, input_features / 64} ||
      linear.biases.shape() != linear.scales.shape()) {
    throw std::runtime_error("unsupported multirow affine Q5 QMV shape");
  }
  static const auto kernel = activation_metal_kernel(
      "sglang_affine_q5_multirow_qmv",
      {"w", "scales", "biases", "x"},
      {"y"},
      R"(
        constexpr int ValuesPerThread = 16;
        constexpr int BlockSize = ValuesPerThread * 32;
        constexpr int Results = 4;
        constexpr int WeightRowBytes = KConst * 5 / 8;
        constexpr int GroupsPerRow = KConst / 64;
        const int lane = thread_index_in_simdgroup;
        const int output_start = threadgroup_position_in_grid.y * 16 +
            simdgroup_index_in_threadgroup * Results;
        using Batch = float3;
        thread Batch inputs[ValuesPerThread];
        thread Batch results[Results];
#pragma unroll
        for (int row = 0; row < Results; ++row) {
          results[row] = Batch(0.0f);
        }
        for (int k = 0; k < KConst; k += BlockSize) {
          Batch input_sum = Batch(0.0f);
#pragma unroll
          for (int index = 0; index < ValuesPerThread; ++index) {
            Batch value;
#pragma unroll
            for (int batch = 0; batch < Rows; ++batch) {
              value[batch] = static_cast<float>(
                  x[batch * KConst + k + lane * ValuesPerThread + index]);
            }
            inputs[index] = value;
            input_sum += value;
          }
#pragma unroll
          for (int row = 0; row < Results; ++row) {
            const int parameter = (output_start + row) * GroupsPerRow +
                k / 64 + lane / 4;
            const float scale = static_cast<float>(scales[parameter]);
            const float bias = static_cast<float>(biases[parameter]);
            const device uchar* packed =
                reinterpret_cast<const device uchar*>(w) +
                (output_start + row) * WeightRowBytes +
                (k + lane * ValuesPerThread) * 5 / 8;
            const packed_ushort4 words =
                *reinterpret_cast<const device packed_ushort4*>(packed);
            const uint trailing =
                *reinterpret_cast<const device ushort*>(packed + 8);
            const uint window0 =
                static_cast<uint>(words[0]) |
                (static_cast<uint>(words[1]) << 16);
            const uint window1 =
                static_cast<uint>(words[2]) |
                (static_cast<uint>(words[3]) << 16);
            const uint codes[ValuesPerThread] = {
                window0 & 0x1fu, (window0 >> 5) & 0x1fu,
                (window0 >> 10) & 0x1fu, (window0 >> 15) & 0x1fu,
                (window0 >> 20) & 0x1fu, (window0 >> 25) & 0x1fu,
                (window0 >> 30) | ((window1 & 0x07u) << 2),
                (window1 >> 3) & 0x1fu, (window1 >> 8) & 0x1fu,
                (window1 >> 13) & 0x1fu, (window1 >> 18) & 0x1fu,
                (window1 >> 23) & 0x1fu,
                (window1 >> 28) | ((trailing & 0x01u) << 4),
                (trailing >> 1) & 0x1fu,
                (trailing >> 6) & 0x1fu, trailing >> 11,
            };
            Batch accumulator = Batch(0.0f);
#pragma unroll
            for (int index = 0; index < ValuesPerThread; ++index) {
              accumulator = fma(
                  inputs[index], Batch(static_cast<float>(codes[index])),
                  accumulator);
            }
            results[row] += scale * accumulator + bias * input_sum;
          }
        }
#pragma unroll
        for (int batch = 0; batch < Rows; ++batch) {
#pragma unroll
          for (int row = 0; row < Results; ++row) {
            const float result = simd_sum(results[row][batch]);
            if (lane == 0) {
              y[batch * NConst + output_start + row] =
                  static_cast<Activation>(result);
            }
          }
        }
      )");
  auto outputs = kernel(
      {linear.w, linear.scales, linear.biases, x},
      {{1, rows, output_features}},
      {x.dtype()},
      {128, output_features / 16, 1},
      {128, 1, 1},
      {
          {"KConst", mx::fast::TemplateArg{input_features}},
          {"NConst", mx::fast::TemplateArg{output_features}},
          {"Rows", mx::fast::TemplateArg{rows}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
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

std::vector<array> gated_delta_update_outputs(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state,
    bool capture_delta) {
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
  std::vector<mx::Shape> output_shapes = {
      {B, T, Hv, Dv}, state.shape()};
  std::vector<mx::Dtype> output_dtypes = {q.dtype(), state.dtype()};
  if (capture_delta) {
    output_shapes.push_back({B, T, Hv, Dv});
    output_dtypes.push_back(mx::float32);
  }
  const auto& kernel =
      capture_delta ? gated_delta_tape_metal() : gated_delta_metal();
  return kernel(
      {q, k, v, g, beta, state, array(T, mx::int32)},
      output_shapes,
      output_dtypes,
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
}

std::pair<array, array> gated_delta_update(
    const array& q,
    const array& k,
    const array& v,
    const array& g,
    const array& beta,
    const array& state) {
  auto outputs = gated_delta_update_outputs(
      q, k, v, g, beta, state, false);
  return {outputs[0], outputs[1]};
}

array gated_delta_commit(
    const array& keys,
    const array& decay,
    const array& delta,
    const array& state,
    int token_count) {
  if (keys.ndim() != 4 || decay.ndim() != 3 || delta.ndim() != 4 ||
      state.ndim() != 4 || keys.shape()[0] != delta.shape()[0] ||
      keys.shape()[0] != state.shape()[0] ||
      keys.shape()[0] != decay.shape()[0] ||
      keys.shape()[1] != delta.shape()[1] ||
      keys.shape()[1] != decay.shape()[1] ||
      delta.shape()[2] != state.shape()[1] ||
      delta.shape()[2] != decay.shape()[2] ||
      delta.shape()[3] != state.shape()[2] ||
      keys.shape()[3] != state.shape()[3] ||
      state.shape()[1] % keys.shape()[2] != 0 ||
      keys.shape()[3] % 32 != 0 || token_count <= 0 ||
      token_count > keys.shape()[1] || decay.dtype() != mx::float32 ||
      delta.dtype() != mx::float32 || state.dtype() != mx::float32) {
    throw std::runtime_error("invalid gated-delta commit tape");
  }
  const int batch = static_cast<int>(keys.shape()[0]);
  const int tape_length = static_cast<int>(keys.shape()[1]);
  const int key_heads = static_cast<int>(keys.shape()[2]);
  const int key_dim = static_cast<int>(keys.shape()[3]);
  const int value_heads = static_cast<int>(state.shape()[1]);
  const int value_dim = static_cast<int>(state.shape()[2]);
  auto outputs = gated_delta_commit_metal()(
      {keys,
       decay,
       delta,
       state,
       array(tape_length, mx::int32),
       array(token_count, mx::int32)},
      {state.shape()},
      {state.dtype()},
      {32, value_dim, batch * value_heads},
      {32, 4, 1},
      {
          {"Dk", mx::fast::TemplateArg{key_dim}},
          {"Dv", mx::fast::TemplateArg{value_dim}},
          {"Hk", mx::fast::TemplateArg{key_heads}},
          {"Hv", mx::fast::TemplateArg{value_heads}},
      },
      std::nullopt,
      false,
      {});
  return outputs[0];
}

int attention_cache_growth_capacity(
    int current_capacity, int needed_capacity, int reserve_capacity) {
  if (current_capacity < 0 || needed_capacity <= current_capacity ||
      needed_capacity <= 0 || reserve_capacity < 0) {
    throw std::runtime_error("invalid attention cache growth request");
  }
  int capacity = std::max(256, current_capacity);
  if (reserve_capacity >= needed_capacity) {
    capacity = std::max(capacity, reserve_capacity);
  }
  while (capacity < needed_capacity) {
    if (capacity > std::numeric_limits<int>::max() / 2) {
      throw std::runtime_error("attention cache capacity overflow");
    }
    capacity *= 2;
  }
  return capacity;
}

bool attention_cache_append_requires_large_growth(
    int cache_length, int cache_capacity, int appended_tokens) {
  if (cache_length < 0 || cache_capacity < 0 ||
      cache_length > cache_capacity || appended_tokens <= 0) {
    throw std::runtime_error("invalid attention cache extent");
  }
  // Below 64K, replacement and a full prefill chunk coexist within the
  // qualified Metal margin. Serialize only the measured 64K-to-128K boundary
  // so ordinary and smaller-context prefill retains its established cadence.
  constexpr int kMinimumSerializedCapacity = 65536;
  return cache_capacity >= kMinimumSerializedCapacity &&
      appended_tokens > cache_capacity - cache_length;
}

int post_growth_prefill_chunk_size(
    int target_cache_length,
    int target_cache_capacity,
    int mtp_cache_length,
    int mtp_cache_capacity,
    int requested_tokens,
    int configured_max_tokens) {
  constexpr int kGrowthBoundaryCapacity = 65536;
  constexpr int kMaximumConfiguredTokens = 1024;
  if (target_cache_length < 0 || target_cache_capacity < 0 ||
      target_cache_length > target_cache_capacity || mtp_cache_length < 0 ||
      mtp_cache_capacity < 0 || mtp_cache_length > mtp_cache_capacity ||
      requested_tokens <= 0 || configured_max_tokens < 0 ||
      configured_max_tokens > kMaximumConfiguredTokens) {
    throw std::runtime_error("invalid post-growth prefill chunk request");
  }
  if (configured_max_tokens == 0 ||
      target_cache_length <= kGrowthBoundaryCapacity ||
      target_cache_capacity <= kGrowthBoundaryCapacity ||
      mtp_cache_length <= kGrowthBoundaryCapacity ||
      mtp_cache_capacity <= kGrowthBoundaryCapacity) {
    return requested_tokens;
  }
  return std::min(requested_tokens, configured_max_tokens);
}

int serialized_attention_cache_growth_chunk_size(
    int cache_length, int cache_capacity, int requested_tokens) {
  if (!attention_cache_append_requires_large_growth(
          cache_length, cache_capacity, requested_tokens)) {
    return requested_tokens;
  }
  const int available = cache_capacity - cache_length;
  return std::max(1, available);
}

Engine::Engine(MlxQwen38Config cfg, const std::string& model_dir)
    : cfg_(configure_mlx_runtime(cfg)) {
  if (cfg_.hidden_size <= 0 || cfg_.num_hidden_layers <= 0) {
    throw std::runtime_error("invalid Qwen3.8 config");
  }
  target_only_prefill_chunk_size_ = native_target_only_prefill_chunk_size();
  attention_cache_reserve_capacity_ =
      native_attention_cache_reserve_capacity();
  attention_cache_bits_ = native_attention_cache_bits();
  serialize_attention_cache_growth_ =
      native_serialize_attention_cache_growth_enabled();
  evict_q4_raw_params_at_mtp_growth_ =
      native_evict_q4_raw_params_at_mtp_growth_enabled();
  post_growth_mtp_prefill_chunk_size_ =
      native_post_growth_mtp_prefill_chunk_size();
  if (evict_q4_raw_params_at_mtp_growth_ &&
      (!serialize_attention_cache_growth_ ||
       !native_q4_fused_swiglu_enabled() ||
       !native_q4_fused_raw_params_enabled())) {
    throw std::runtime_error(
        "MTP-growth raw-parameter eviction requires serialized growth and "
        "raw fused Q4 SwiGLU");
  }
  if (post_growth_mtp_prefill_chunk_size_ > 0 &&
      (!serialize_attention_cache_growth_ ||
       !evict_q4_raw_params_at_mtp_growth_)) {
    throw std::runtime_error(
        "post-growth MTP prefill cap requires serialized growth and "
        "MTP-growth raw-parameter eviction");
  }
  quantized_embedding_enabled_ = native_quantized_embedding_enabled();
  append_only_attention_snapshot_enabled_ =
      native_append_only_attention_snapshot_enabled();
  mtp_prompt_cache_enabled_ = native_mtp_prompt_cache_enabled();
  if (mtp_prompt_cache_enabled_ && !append_only_attention_snapshot_enabled_) {
    throw std::runtime_error("MTP prompt caching requires append-only snapshots");
  }
  if (attention_cache_bits_ == 8 &&
      !append_only_attention_snapshot_enabled_) {
    throw std::runtime_error(
        "affine-Q8 attention cache requires append-only snapshots");
  }
  layers_.resize(static_cast<size_t>(cfg_.num_hidden_layers));
  load_weights(model_dir);
  sampling_enabled_ = native_sampling_enabled();
  if (sampling_enabled_) {
    sampling_seed_ = native_sampling_seed();
  }
  max_reasoning_tokens_ = native_reasoning_token_limit();
  reset();
}

void Engine::reset() {
  if (native_state_trace_enabled()) {
    std::fprintf(
        stderr, "qwen38_native reset history=%zu pending=%d\n",
        token_history_.size(), request_boundary_pending_ ? 1 : 0);
  }
  for (auto& layer : layers_) {
    if (layer.is_linear) {
      layer.linear.has_state = false;
    } else {
      layer.attn.offset = 0;
      layer.attn.cache_length = 0;
      if (layer.attn.cache_bits != 16) {
        layer.attn.keys = array(0);
        layer.attn.key_scales = array(0);
        layer.attn.key_biases = array(0);
        layer.attn.values = array(0);
        layer.attn.value_scales = array(0);
        layer.attn.value_biases = array(0);
        layer.attn.cache_capacity = 0;
        layer.attn.cache_bits = 16;
      }
    }
  }
  if (sampling_enabled_) {
    mx::random::seed(sampling_seed_);
  }
  reset_decode_pipeline();
  request_boundary_pending_ = false;
  token_history_.clear();
  selected_reasoning_tokens_ = 0;
  reasoning_open_ = false;
  reasoning_cap_selected_ = false;
  prompt_snapshot_valid_ = false;
  prompt_snapshot_history_.clear();
  mtp_prompt_snapshot_.clear();
  mtp_prompt_hidden_ = array(0);
  mtp_prompt_cache_length_ = 0;
  snap_.clear();
  if (mtp_valid_) {
    mtp_reset();
  }
  if (dflash_valid_) {
    dflash_reset();
  }
  if (dspark_valid_) {
    dspark_reset();
  }
}

void Engine::begin_request() {
  if (native_state_trace_enabled()) {
    std::fprintf(
        stderr, "qwen38_native begin_request history=%zu\n",
        token_history_.size());
  }
  request_boundary_pending_ = true;
}

void Engine::reset_decode_pipeline() {
  pending_tok_ = array(0);
  last_hidden_ = array(0);
  last_emitted_ = -1;
  decode_scheduled_ = false;
  last_emitted_in_state_ = false;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
}

void Engine::record_processed_token(int32_t token) {
  token_history_.push_back(token);
  last_emitted_in_state_ = true;
}

QLinear Engine::load_qlinear(
    const std::unordered_map<std::string, array>& weights,
    const std::string& prefix) {
  QLinear q;
  q.w = require(weights, prefix + ".weight");
  if (q.w.dtype() == activation_dtype()) {
    if (q.w.ndim() != 2) {
      throw std::runtime_error("invalid dense tensor for " + prefix);
    }
    q.valid = true;
    return q;
  }
  q.scales = require(weights, prefix + ".scales");
  q.biases = require(weights, prefix + ".biases");
  q.group_size = cfg_.quant_group_size;

  const auto& weight_shape = q.w.shape();
  const auto& scale_shape = q.scales.shape();
  if (weight_shape.size() < 2 || scale_shape.size() < 2 ||
      q.biases.shape() != scale_shape) {
    throw std::runtime_error("invalid affine tensors for " + prefix);
  }
  const int64_t packed_bits = static_cast<int64_t>(weight_shape.back()) * 32;
  const int64_t input_features =
      static_cast<int64_t>(scale_shape.back()) * q.group_size;
  if (input_features <= 0 || packed_bits % input_features != 0) {
    throw std::runtime_error("cannot infer affine bit width for " + prefix);
  }
  q.bits = static_cast<int>(packed_bits / input_features);
  if (q.bits < 2 || q.bits > 8 || q.bits == 7) {
    throw std::runtime_error(
        "unsupported affine bit width " + std::to_string(q.bits) + " for " +
        prefix);
  }
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
    auto loaded = load_activation_safetensors(model_dir + "/" + name);
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

  const char* const out_proj_override =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_OUT_PROJ_OVERRIDE_PATH");
  const char* const linear_attn_override =
      std::getenv("SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_PATH");
  const bool has_out_proj_override =
      out_proj_override != nullptr && *out_proj_override != '\0';
  const bool has_linear_attn_override =
      linear_attn_override != nullptr && *linear_attn_override != '\0';
  if (has_out_proj_override && has_linear_attn_override) {
    throw std::runtime_error(
        "linear-attention override paths are mutually exclusive");
  }
  if (has_out_proj_override || has_linear_attn_override) {
    const LinearAttnOverrideScope scope = has_linear_attn_override
        ? linear_attn_override_scope()
        : LinearAttnOverrideScope::kOutProjection;
    const char* const override_path =
        has_linear_attn_override ? linear_attn_override : out_proj_override;
    const size_t replaced =
        apply_linear_attn_override(weights, override_path, scope);
    const size_t linear_layers = static_cast<size_t>(
        cfg_.num_hidden_layers -
        cfg_.num_hidden_layers / cfg_.full_attention_interval);
    const size_t projections_per_layer = linear_attn_projection_count(scope);
    const size_t expected = linear_layers * projections_per_layer * 3;
    if (replaced != expected) {
      throw std::runtime_error(
          "linear-attention override replaced " + std::to_string(replaced) +
          " tensors; expected " + std::to_string(expected));
    }
  }

  embed_tokens_ = load_qlinear(weights, "language_model.model.embed_tokens");
  if (!quantized_embedding_enabled_) {
    embed_table_ = mx::dequantize(
        embed_tokens_.w,
        embed_tokens_.scales,
        embed_tokens_.biases,
        embed_tokens_.group_size,
        embed_tokens_.bits,
        "affine",
        std::nullopt,
        activation_dtype());
    eval(embed_table_);
  }
  lm_head_ = load_qlinear(weights, "language_model.lm_head");
  final_norm_ = require(weights, "language_model.model.norm.weight");

  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    DecoderLayer layer;
    layer.is_linear = ((i + 1) % cfg_.full_attention_interval) != 0;
    layer.input_norm = require(weights, layer_key(i, ".input_layernorm.weight"));
    layer.post_norm = require(weights, layer_key(i, ".post_attention_layernorm.weight"));
    layer.gate_proj = load_qlinear(weights, layer_key(i, ".mlp.gate_proj"));
    layer.up_proj = load_qlinear(weights, layer_key(i, ".mlp.up_proj"));
    if (native_q4_fused_swiglu_enabled() &&
        native_q4_fused_raw_params_enabled() &&
        layer.gate_proj.bits == 4 && layer.up_proj.bits == 4 &&
        !prepare_fused_q4_raw_decode_parameters(
            layer.gate_proj, layer.up_proj)) {
      throw std::runtime_error(
          "cannot prepare raw fused Q4 decode parameters for layer " +
          std::to_string(i));
    }
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

std::size_t Engine::release_target_fused_q4_raw_decode_parameters() {
  std::size_t released_bytes = 0;
  std::size_t released_layers = 0;
  for (DecoderLayer& layer : layers_) {
    if (layer.gate_proj.bits != 4 || layer.up_proj.bits != 4) {
      continue;
    }
    const std::size_t layer_bytes =
        release_fused_q4_raw_decode_parameters(layer.gate_proj);
    if (layer_bytes == 0) {
      throw std::runtime_error("missing target raw fused Q4 decode parameters");
    }
    if (released_bytes >
        std::numeric_limits<std::size_t>::max() - layer_bytes) {
      throw std::runtime_error("target raw fused Q4 parameter size overflow");
    }
    released_bytes += layer_bytes;
    ++released_layers;
  }
  if (released_layers == 0) {
    throw std::runtime_error("target has no raw fused Q4 decode parameters");
  }
  return released_bytes;
}

std::size_t Engine::restore_target_fused_q4_raw_decode_parameters() {
  std::size_t restored_bytes = 0;
  std::size_t restored_layers = 0;
  for (DecoderLayer& layer : layers_) {
    if (layer.gate_proj.bits != 4 || layer.up_proj.bits != 4) {
      continue;
    }
    if (layer.gate_proj.fused_q4_decode_params_valid ||
        !prepare_fused_q4_raw_decode_parameters(
            layer.gate_proj, layer.up_proj)) {
      throw std::runtime_error(
          "cannot restore target raw fused Q4 decode parameters");
    }
    const std::size_t layer_bytes =
        layer.gate_proj.fused_q4_decode_params.nbytes();
    if (restored_bytes >
        std::numeric_limits<std::size_t>::max() - layer_bytes) {
      throw std::runtime_error("target raw fused Q4 parameter size overflow");
    }
    restored_bytes += layer_bytes;
    ++restored_layers;
  }
  if (restored_layers == 0) {
    throw std::runtime_error("target has no raw fused Q4 decode parameters");
  }
  return restored_bytes;
}

array Engine::embed(const array& tokens) const {
  if (quantized_embedding_enabled_) {
    return quantized_embedding_rows(embed_tokens_, tokens);
  }
  return take(embed_table_, tokens, 0);
}

array Engine::logits(const array& hidden) const {
  array n = mx::fast::rms_norm(hidden, final_norm_, cfg_.rms_norm_eps);
  return lm_head_(n);
}

array Engine::mlp(const DecoderLayer& layer, const array& x) const {
  if (native_q4_fused_swiglu_enabled() &&
      native_q4_fused_swiglu_batch_two_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 2 &&
      x.dtype() == activation_dtype() && layer.gate_proj.valid &&
      layer.up_proj.valid && layer.gate_proj.w.dtype() == mx::uint32 &&
      layer.up_proj.w.dtype() == mx::uint32 && layer.gate_proj.bits == 4 &&
      layer.up_proj.bits == 4 && layer.gate_proj.group_size == 64 &&
      layer.up_proj.group_size == 64 &&
      layer.gate_proj.fused_q4_decode_params_valid) {
    if (native_qmm_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_qmv q4-fused-swiglu rows=2 K=%d N=%d "
          "geometry=8x4x2-shared\n",
          static_cast<int>(x.shape()[2]),
          static_cast<int>(layer.gate_proj.w.shape()[0]));
    }
    return layer.down_proj(affine_q4_fused_swiglu_batch_two(
        layer.gate_proj, layer.up_proj, x));
  }
  if (native_q4_fused_swiglu_enabled() && x.ndim() == 3 &&
      x.shape()[0] == 1 && x.shape()[1] == 1 &&
      x.dtype() == activation_dtype() && layer.gate_proj.valid &&
      layer.up_proj.valid && layer.gate_proj.w.dtype() == mx::uint32 &&
      layer.up_proj.w.dtype() == mx::uint32 && layer.gate_proj.bits == 4 &&
      layer.up_proj.bits == 4 && layer.gate_proj.group_size == 64 &&
      layer.up_proj.group_size == 64) {
    return layer.down_proj(affine_q4_fused_swiglu_batch_one(
        layer.gate_proj, layer.up_proj, x));
  }
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

  if (L == 1) {
    auto normalized = full_attn_qk_norm_rope(
        qg,
        keys,
        attn.q_norm,
        attn.k_norm,
        cfg_.rms_norm_eps,
        cfg_.rope_theta,
        rope_dims,
        attn.offset);
    queries = normalized.first;
    keys = normalized.second;
  } else {
    queries = mx::fast::rms_norm(queries, attn.q_norm, cfg_.rms_norm_eps);
    keys = mx::fast::rms_norm(keys, attn.k_norm, cfg_.rms_norm_eps);
    queries = transpose(queries, {0, 2, 1, 3});
    keys = transpose(keys, {0, 2, 1, 3});

    queries = mx::fast::rope(
        queries,
        rope_dims,
        /*traditional=*/false,
        cfg_.rope_theta,
        1.0f,
        attn.offset);
    keys = mx::fast::rope(
        keys,
        rope_dims,
        /*traditional=*/false,
        cfg_.rope_theta,
        1.0f,
        attn.offset);
  }
  values = transpose(values, {0, 2, 1, 3});

  const int prefix_length = attn.cache_length;
  const int needed = prefix_length + L;
  constexpr std::size_t kLongContextMinimumTokens = 8192;
  const bool long_context =
      token_history_.size() > kLongContextMinimumTokens;
  const int reserve_capacity =
      long_context ? attention_cache_reserve_capacity_ : 0;
  const bool use_q8_cache =
      attn.cache_bits == 8 ||
      (attention_cache_bits_ == 8 && long_context &&
       &attn != &mtp_layer_.attn);

  if (use_q8_cache) {
    constexpr int kQ8Bits = 8;
    constexpr int kQ8GroupSize = 64;
    constexpr int kPackedValuesPerWord = 4;
    if (B != 1 || n_q != 24 || n_kv != 4 || hd != 256 || L > 1024) {
      throw std::runtime_error("unsupported affine-Q8 attention geometry");
    }

    if (attn.cache_bits == 16) {
      array dense_keys = keys;
      array dense_values = values;
      if (prefix_length > 0) {
        dense_keys = concatenate(
            {slice(
                 attn.keys,
                 {0, 0, 0, 0},
                 {B, n_kv, prefix_length, hd}),
             keys},
            2);
        dense_values = concatenate(
            {slice(
                 attn.values,
                 {0, 0, 0, 0},
                 {B, n_kv, prefix_length, hd}),
             values},
            2);
      }
      std::vector<array> quantized_keys =
          mx::quantize(dense_keys, kQ8GroupSize, kQ8Bits, "affine");
      std::vector<array> quantized_values =
          mx::quantize(dense_values, kQ8GroupSize, kQ8Bits, "affine");
      if (quantized_keys.size() != 3 || quantized_values.size() != 3) {
        throw std::runtime_error("invalid affine-Q8 cache quantization");
      }
      const int capacity = attention_cache_growth_capacity(
          0, needed, reserve_capacity);
      array new_keys = zeros(
          {B, n_kv, capacity, hd / kPackedValuesPerWord}, mx::uint32);
      array new_key_scales = zeros(
          {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
      array new_key_biases = zeros(
          {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
      array new_values = zeros(
          {B, n_kv, capacity, hd / kPackedValuesPerWord}, mx::uint32);
      array new_value_scales = zeros(
          {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
      array new_value_biases = zeros(
          {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
      new_keys = slice_update(
          new_keys,
          quantized_keys[0],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kPackedValuesPerWord});
      new_key_scales = slice_update(
          new_key_scales,
          quantized_keys[1],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      new_key_biases = slice_update(
          new_key_biases,
          quantized_keys[2],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      new_values = slice_update(
          new_values,
          quantized_values[0],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kPackedValuesPerWord});
      new_value_scales = slice_update(
          new_value_scales,
          quantized_values[1],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      new_value_biases = slice_update(
          new_value_biases,
          quantized_values[2],
          {0, 0, 0, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      eval(
          new_keys,
          new_key_scales,
          new_key_biases,
          new_values,
          new_value_scales,
          new_value_biases);
      attn.keys = std::move(new_keys);
      attn.key_scales = std::move(new_key_scales);
      attn.key_biases = std::move(new_key_biases);
      attn.values = std::move(new_values);
      attn.value_scales = std::move(new_value_scales);
      attn.value_biases = std::move(new_value_biases);
      attn.cache_capacity = capacity;
      attn.cache_bits = kQ8Bits;
    } else {
      if (attn.cache_bits != kQ8Bits || attn.keys.ndim() != 4 ||
          attn.key_scales.ndim() != 4 || attn.key_biases.ndim() != 4 ||
          attn.values.ndim() != 4 || attn.value_scales.ndim() != 4 ||
          attn.value_biases.ndim() != 4 ||
          attn.keys.dtype() != mx::uint32 ||
          attn.values.dtype() != mx::uint32 ||
          attn.key_scales.dtype() != activation_dtype() ||
          attn.key_biases.dtype() != activation_dtype() ||
          attn.value_scales.dtype() != activation_dtype() ||
          attn.value_biases.dtype() != activation_dtype()) {
        throw std::runtime_error("invalid affine-Q8 attention cache");
      }
      if (attn.cache_capacity < needed) {
        const int capacity = attention_cache_growth_capacity(
            attn.cache_capacity, needed, reserve_capacity);
        array new_keys = zeros(
            {B, n_kv, capacity, hd / kPackedValuesPerWord}, mx::uint32);
        array new_key_scales = zeros(
            {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
        array new_key_biases = zeros(
            {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
        array new_values = zeros(
            {B, n_kv, capacity, hd / kPackedValuesPerWord}, mx::uint32);
        array new_value_scales = zeros(
            {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
        array new_value_biases = zeros(
            {B, n_kv, capacity, hd / kQ8GroupSize}, activation_dtype());
        if (prefix_length > 0) {
          new_keys = slice_update(
              new_keys,
              slice(
                  attn.keys,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kPackedValuesPerWord}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kPackedValuesPerWord});
          new_key_scales = slice_update(
              new_key_scales,
              slice(
                  attn.key_scales,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kQ8GroupSize}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kQ8GroupSize});
          new_key_biases = slice_update(
              new_key_biases,
              slice(
                  attn.key_biases,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kQ8GroupSize}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kQ8GroupSize});
          new_values = slice_update(
              new_values,
              slice(
                  attn.values,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kPackedValuesPerWord}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kPackedValuesPerWord});
          new_value_scales = slice_update(
              new_value_scales,
              slice(
                  attn.value_scales,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kQ8GroupSize}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kQ8GroupSize});
          new_value_biases = slice_update(
              new_value_biases,
              slice(
                  attn.value_biases,
                  {0, 0, 0, 0},
                  {B, n_kv, prefix_length, hd / kQ8GroupSize}),
              {0, 0, 0, 0},
              {B, n_kv, prefix_length, hd / kQ8GroupSize});
          eval(
              new_keys,
              new_key_scales,
              new_key_biases,
              new_values,
              new_value_scales,
              new_value_biases);
        }
        attn.keys = std::move(new_keys);
        attn.key_scales = std::move(new_key_scales);
        attn.key_biases = std::move(new_key_biases);
        attn.values = std::move(new_values);
        attn.value_scales = std::move(new_value_scales);
        attn.value_biases = std::move(new_value_biases);
        attn.cache_capacity = capacity;
      }
      std::vector<array> quantized_keys =
          mx::quantize(keys, kQ8GroupSize, kQ8Bits, "affine");
      std::vector<array> quantized_values =
          mx::quantize(values, kQ8GroupSize, kQ8Bits, "affine");
      if (quantized_keys.size() != 3 || quantized_values.size() != 3) {
        throw std::runtime_error("invalid affine-Q8 cache quantization");
      }
      attn.keys = slice_update(
          attn.keys,
          quantized_keys[0],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kPackedValuesPerWord});
      attn.key_scales = slice_update(
          attn.key_scales,
          quantized_keys[1],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      attn.key_biases = slice_update(
          attn.key_biases,
          quantized_keys[2],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      attn.values = slice_update(
          attn.values,
          quantized_values[0],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kPackedValuesPerWord});
      attn.value_scales = slice_update(
          attn.value_scales,
          quantized_values[1],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
      attn.value_biases = slice_update(
          attn.value_biases,
          quantized_values[2],
          {0, 0, prefix_length, 0},
          {B, n_kv, needed, hd / kQ8GroupSize});
    }
  } else {
    if (attn.cache_bits != 16) {
      throw std::runtime_error("cannot restore BF16 attention cache mode");
    }
    if (attn.cache_capacity < needed) {
      const int capacity = attention_cache_growth_capacity(
          attn.cache_capacity, needed, reserve_capacity);
      array new_keys = zeros({B, n_kv, capacity, hd}, keys.dtype());
      array new_values = zeros({B, n_kv, capacity, hd}, values.dtype());
      if (prefix_length > 0) {
        auto active_keys = slice(
            attn.keys, {0, 0, 0, 0}, {B, n_kv, prefix_length, hd});
        auto active_values = slice(
            attn.values, {0, 0, 0, 0}, {B, n_kv, prefix_length, hd});
        new_keys = slice_update(
            new_keys,
            active_keys,
            {0, 0, 0, 0},
            {B, n_kv, prefix_length, hd});
        new_values = slice_update(
            new_values,
            active_values,
            {0, 0, 0, 0},
            {B, n_kv, prefix_length, hd});
        eval(new_keys, new_values);
      }
      attn.keys = std::move(new_keys);
      attn.values = std::move(new_values);
      attn.cache_capacity = capacity;
    }
    attn.keys = slice_update(
        attn.keys,
        keys,
        {0, 0, prefix_length, 0},
        {B, n_kv, needed, hd});
    attn.values = slice_update(
        attn.values,
        values,
        {0, 0, prefix_length, 0},
        {B, n_kv, needed, hd});
  }
  attn.cache_length = needed;
  attn.offset += L;

  array output(0);
  constexpr int kFixedAttentionMinimumActiveTokens = 8192;
  if (attn.cache_bits == 8) {
    output = fixed_q8_attention(
        queries,
        attn.keys,
        attn.key_scales,
        attn.key_biases,
        attn.values,
        attn.value_scales,
        attn.value_biases,
        prefix_length,
        needed);
  } else if (native_fixed_prefill_attention_enabled() &&
      needed > kFixedAttentionMinimumActiveTokens && L > 1 && B == 1 &&
      n_q == 24 && n_kv == 4 && hd == 256 && L <= 1024) {
    output = fixed_prefill_attention(
        queries, attn.keys, attn.values, prefix_length, needed);
  } else {
    keys = slice(attn.keys, {0, 0, 0, 0}, {B, n_kv, needed, hd});
    values = slice(attn.values, {0, 0, 0, 0}, {B, n_kv, needed, hd});
    std::string mask_mode = (L > 1) ? "causal" : "";
    output = mx::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        1.0f / std::sqrt(static_cast<float>(hd)),
        mask_mode);
  }
  output = reshape(transpose(output, {0, 2, 1, 3}), {B, L, n_q * hd});
  return attn.o_proj(output * sigmoid(gate));
}

array Engine::gated_delta(
    LinearAttn& lin, const array& x, LinearCommitTape* commit_tape) {
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
  if (commit_tape != nullptr) {
    commit_tape->conv_tokens = qkv;
    commit_tape->valid = false;
  }
  array z = reshape(lin.in_proj_z(x), {B, S, hv, dv});
  array b = lin.in_proj_b(x);
  array a = lin.in_proj_a(x);

  if (!lin.has_state) {
    lin.conv_state = zeros({B, ksz - 1, conv_dim}, x.dtype());
    lin.rec_state = zeros({B, hv, dv, dk}, mx::float32);
    lin.has_state = true;
  }
  array conv_out = qkv;
  if (S == 1 && ksz > 1) {
    auto conv = causal_conv_decode_silu(lin.conv_state, qkv, lin.conv1d);
    conv_out = conv.first;
    lin.conv_state = conv.second;
  } else if (S == 2 && ksz > 1 && native_two_token_causal_conv_enabled()) {
    auto conv = causal_conv_two_token_silu(lin.conv_state, qkv, lin.conv1d);
    conv_out = conv.first;
    lin.conv_state = conv.second;
  } else {
    array conv_input = concatenate({lin.conv_state, qkv}, 1);
    lin.conv_state = slice(
        conv_input, {0, S, 0}, {B, S + (ksz - 1), conv_dim});
    conv_out = silu(mx::conv1d(conv_input, lin.conv1d, 1, 0, 1, conv_dim));
  }

  auto qkv_split = split(conv_out, mx::Shape{key_dim, 2 * key_dim}, -1);
  array q = reshape(qkv_split[0], {B, S, hk, dk});
  array k = reshape(qkv_split[1], {B, S, hk, dk});
  array v = reshape(qkv_split[2], {B, S, hv, dv});

  float inv = 1.0f / std::sqrt(static_cast<float>(dk));
  const bool fuse_norms = S == 1 ||
      (S <= 8 && native_verify_fused_norms_enabled());
  if (fuse_norms) {
    auto normalized = normalize_gated_delta_qk(q, k, inv * inv, inv, 1e-6f);
    q = normalized.first;
    k = normalized.second;
  } else {
    q = (inv * inv) * mx::fast::rms_norm(q, std::nullopt, 1e-6f);
    k = inv * mx::fast::rms_norm(k, std::nullopt, 1e-6f);
  }

  array beta = sigmoid(b);
  array g = compiled_compute_g()({lin.A_log, a, lin.dt_bias})[0];
  auto updated = gated_delta_update_outputs(
      q, k, v, g, beta, lin.rec_state, commit_tape != nullptr);
  lin.rec_state = updated[1];
  array out = updated[0];
  if (commit_tape != nullptr) {
    commit_tape->keys = k;
    commit_tape->decay = g;
    commit_tape->delta = updated[2];
    commit_tape->valid = true;
  }
  array gated = fuse_norms
      ? gated_delta_norm_gate(out, z, lin.norm, cfg_.rms_norm_eps)
      : astype(
            silu(astype(z, mx::float32)) *
                astype(
                    mx::fast::rms_norm(out, lin.norm, cfg_.rms_norm_eps),
                    mx::float32),
            x.dtype());
  return lin.out_proj(reshape(gated, {B, S, value_dim}));
}

array Engine::forward_hidden(const array& tokens) {
  return forward_hidden_impl(tokens, nullptr, nullptr);
}

TargetForward Engine::forward_hidden_captured(
    const array& tokens, bool capture_commit_tape) {
  TargetForward result;
  result.captured.reserve(5);
  if (capture_commit_tape) {
    result.linear_tapes.resize(layers_.size());
  }
  result.hidden = forward_hidden_impl(
      tokens,
      &result.captured,
      capture_commit_tape ? &result.linear_tapes : nullptr);
  if (result.captured.size() != 5) {
    throw std::runtime_error(
        "speculative draft capture requires layers 5, 19, 33, 47, and 61");
  }
  return result;
}

array Engine::forward_hidden_impl(
    const array& tokens,
    std::vector<array>* captured,
    std::vector<LinearCommitTape>* commit_tapes) {
  array h = embed(tokens);
  const int verify_tokens = tokens.shape()[1];
  const bool submit_verify = verify_tokens >= 2 && verify_tokens <= 8 &&
      native_async_verify_enabled();
  const auto capture = [&captured, &h, submit_verify](size_t layer_index) {
    // Let Metal execute the completed prefix while the host constructs the
    // remaining verifier graph. Async evaluation retains the array dependencies
    // and never waits for completion or changes the speculative commit boundary.
    if (submit_verify && (layer_index + 1) % 4 == 0) {
      async_eval(h);
    }
    if (captured != nullptr &&
        (layer_index == 5 || layer_index == 19 || layer_index == 33 ||
         layer_index == 47 || layer_index == 61)) {
      captured->push_back(h);
    }
  };
  const int token_count = tokens.shape()[1];
  const bool fuse_norms = token_count == 1 ||
      (token_count <= 8 && native_verify_fused_norms_enabled());
  if (!fuse_norms) {
    for (size_t i = 0; i < layers_.size(); ++i) {
      auto& layer = layers_[i];
      array n = mx::fast::rms_norm(h, layer.input_norm, cfg_.rms_norm_eps);
      LinearCommitTape* const commit_tape =
          commit_tapes != nullptr && layer.is_linear
          ? &(*commit_tapes)[i]
          : nullptr;
      array r = layer.is_linear
          ? gated_delta(layer.linear, n, commit_tape)
          : full_attn(layer.attn, n);
      h = h + r;
      array n2 = mx::fast::rms_norm(h, layer.post_norm, cfg_.rms_norm_eps);
      h = h + mlp(layer, n2);
      capture(i);
    }
    return h;
  }

  array n = mx::fast::rms_norm(
      h, layers_.front().input_norm, cfg_.rms_norm_eps);
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    LinearCommitTape* const commit_tape =
        commit_tapes != nullptr && layer.is_linear
        ? &(*commit_tapes)[i]
        : nullptr;
    array r = layer.is_linear
        ? gated_delta(layer.linear, n, commit_tape)
        : full_attn(layer.attn, n);
    auto post_norm =
        residual_rms_norm(h, r, layer.post_norm, cfg_.rms_norm_eps);
    h = post_norm.first;
    array m = mlp(layer, post_norm.second);
    if (i + 1 == layers_.size()) {
      h = h + m;
    } else {
      auto next_norm = residual_rms_norm(
          h, m, layers_[i + 1].input_norm, cfg_.rms_norm_eps);
      h = next_norm.first;
      n = next_norm.second;
    }
    capture(i);
  }
  return h;
}

array sampling_topk_indices(const array& token_logits, int k) {
  if (token_logits.ndim() == 0 || k < 1 || k > token_logits.shape().back())
    throw std::runtime_error("invalid top-k selection shape or count");
  const auto shape = token_logits.shape();
  const int vocab = shape.back();
  int block = 0;
  if (const char* value = std::getenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK")) {
    const std::string_view text(value);
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), block);
    if (error != std::errc() || end != text.data() + text.size())
      throw std::runtime_error("invalid hierarchical top-k block");
  }
  if (block != 0 && block != 256 && block != 512 && block != 1024 && block != 2048)
    throw std::runtime_error("invalid hierarchical top-k block");
  if (block == 0 || vocab <= block || vocab % block != 0 || k > block) {
    const array partitioned = mx::argpartition(token_logits, vocab - k, -1);
    mx::Shape start(shape.size(), 0), end(shape);
    start.back() = vocab - k;
    return slice(partitioned, start, end);
  }
  // Every global top-k value appears in its block's top-k. Partition the
  // smaller candidate set without changing full-vocabulary normalization.
  // Equal-score boundary tokens can have different valid partition indices.
  const int blocks = vocab / block;
  const int rows = static_cast<int>(token_logits.size()) / vocab;
  const auto grouped = reshape(token_logits, {rows, blocks, block});
  const auto partitioned = mx::argpartition(grouped, block - k, -1);
  const auto local_ids = slice(partitioned, {0, 0, block - k}, {rows, blocks, block});
  const auto local_logits = reshape(mx::take_along_axis(grouped, local_ids, -1), {rows, blocks * k});
  const auto global_ids = reshape(local_ids + reshape(mx::arange(blocks, mx::int32) * block, {1, blocks, 1}), {rows, blocks * k});
  const auto selected = slice(mx::argpartition(local_logits, blocks * k - k, -1),
      {0, blocks * k - k}, {rows, blocks * k});
  auto output_shape = shape;
  output_shape.back() = k;
  return reshape(mx::take_along_axis(global_ids, selected, -1), output_shape);
}

array Engine::select_token(const array& hidden) {
  last_hidden_ = last_token(hidden);
  if (reasoning_open_) {
    if (max_reasoning_tokens_ > 0 && !reasoning_cap_selected_ &&
        selected_reasoning_tokens_ >= max_reasoning_tokens_) {
      reasoning_cap_selected_ = true;
      reasoning_open_ = false;
      return array({248069}, mx::int32);
    }
    ++selected_reasoning_tokens_;
  }
  array token_logits = logits(last_hidden_);
  if (!sampling_enabled_) {
    return mx::argmax(token_logits, -1);
  }

  // Qwen3.8's model sampling contract is temperature 1.0, top-k 20, top-p
  // 0.95. Keep the vocabulary partition and normalization on Metal, retain
  // the model's full-vocabulary nucleus threshold, then run Gumbel-max over
  // the surviving top-k candidates. This preserves the native graph's
  // asynchronous two-token pipeline and transfers no logits to the host.
  constexpr int kTopK = 20;
  constexpr float kTopP = 0.95f;
  const auto shape = token_logits.shape();
  const int vocab = static_cast<int>(shape[1]);
  if (vocab < kTopK) {
    throw std::runtime_error("native sampling vocabulary is smaller than top-k");
  }

  array candidate_ids = sampling_topk_indices(token_logits, kTopK);
  array candidate_logits = astype(
      mx::take_along_axis(token_logits, candidate_ids, -1), mx::float32);
  array order = mx::argsort(-candidate_logits, -1);
  candidate_ids = mx::take_along_axis(candidate_ids, order, -1);
  candidate_logits = mx::take_along_axis(candidate_logits, order, -1);

  array candidate_probs = mx::exp(
      candidate_logits -
      mx::logsumexp(astype(token_logits, mx::float32), -1, true));
  array cumulative = mx::cumsum(candidate_probs, -1);
  array keep = mx::less_equal(
      cumulative - candidate_probs, array(kTopP, mx::float32));
  array filtered = mx::where(
      keep,
      candidate_logits,
      array(-std::numeric_limits<float>::infinity(), mx::float32));
  array noise = mx::random::gumbel(filtered.shape(), mx::float32);
  array selected_rank = mx::argmax(filtered + noise, -1);
  return squeeze(
      mx::take_along_axis(candidate_ids, expand_dims(selected_rank, -1), -1),
      -1);
}

array Engine::sampling_probabilities(const array& token_logits) {
  constexpr int kTopK = 20;
  constexpr float kTopP = 0.95f;
  const auto shape = token_logits.shape();
  const int vocab = static_cast<int>(shape.back());
  if (vocab < kTopK) {
    throw std::runtime_error("native sampling vocabulary is smaller than top-k");
  }

  array candidate_ids = sampling_topk_indices(token_logits, kTopK);
  array candidate_logits = astype(
      mx::take_along_axis(token_logits, candidate_ids, -1), mx::float32);
  array order = mx::argsort(-candidate_logits, -1);
  candidate_ids = mx::take_along_axis(candidate_ids, order, -1);
  candidate_logits = mx::take_along_axis(candidate_logits, order, -1);

  array candidate_probs = mx::exp(
      candidate_logits -
      mx::logsumexp(astype(token_logits, mx::float32), -1, true));
  array cumulative = mx::cumsum(candidate_probs, -1);
  array keep = mx::less_equal(
      cumulative - candidate_probs, array(kTopP, mx::float32));
  array filtered = mx::where(
      keep,
      candidate_logits,
      array(-std::numeric_limits<float>::infinity(), mx::float32));
  array support_probs = mx::softmax(filtered, -1, true);
  return mx::put_along_axis(
      mx::zeros(token_logits.shape(), mx::float32),
      candidate_ids,
      support_probs,
      -1);
}

void Engine::snapshot() {
  capture_snapshot(snap_);
}

void Engine::capture_snapshot(std::vector<LayerSnap>& destination) const {
  destination.resize(layers_.size());
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto& layer = layers_[i];
    auto& s = destination[i];
    s.is_linear = layer.is_linear;
    if (layer.is_linear) {
      s.has_state = layer.linear.has_state;
      s.conv = layer.linear.conv_state;
      s.rec = layer.linear.rec_state;
    } else {
      s.offset = layer.attn.offset;
      s.cache_length = layer.attn.cache_length;
      s.cache_bits = layer.attn.cache_bits;
      if (append_only_attention_snapshot_enabled_) {
        s.cache_capacity = 0;
        s.keys = array(0);
        s.key_scales = array(0);
        s.key_biases = array(0);
        s.values = array(0);
        s.value_scales = array(0);
        s.value_biases = array(0);
      } else {
        s.cache_capacity = layer.attn.cache_capacity;
        s.keys = layer.attn.keys;
        s.key_scales = layer.attn.key_scales;
        s.key_biases = layer.attn.key_biases;
        s.values = layer.attn.values;
        s.value_scales = layer.attn.value_scales;
        s.value_biases = layer.attn.value_biases;
      }
    }
  }
}

void Engine::restore() {
  restore_snapshot(snap_);
}

void Engine::restore_snapshot(const std::vector<LayerSnap>& source) {
  if (source.size() != layers_.size()) {
    throw std::runtime_error("invalid target snapshot layer count");
  }
  for (size_t i = 0; i < layers_.size(); ++i) {
    auto& layer = layers_[i];
    const auto& s = source[i];
    if (layer.is_linear) {
      layer.linear.has_state = s.has_state;
      layer.linear.conv_state = s.conv;
      layer.linear.rec_state = s.rec;
    } else {
      if (append_only_attention_snapshot_enabled_) {
        const bool bf16_storage_valid =
            layer.attn.cache_bits == 16 && layer.attn.keys.ndim() == 4 &&
            layer.attn.values.ndim() == 4 &&
            layer.attn.keys.dtype() == activation_dtype() &&
            layer.attn.values.dtype() == activation_dtype() &&
            layer.attn.keys.shape() == layer.attn.values.shape() &&
            layer.attn.keys.shape()[0] == 1 &&
            layer.attn.keys.shape()[1] == cfg_.num_key_value_heads &&
            layer.attn.keys.shape()[2] == layer.attn.cache_capacity &&
            layer.attn.keys.shape()[3] == cfg_.head_dim;
        const mx::Shape q8_parameter_shape{
            1,
            cfg_.num_key_value_heads,
            layer.attn.cache_capacity,
            cfg_.head_dim / 64};
        const bool q8_storage_valid =
            layer.attn.cache_bits == 8 && layer.attn.keys.ndim() == 4 &&
            layer.attn.values.ndim() == 4 &&
            layer.attn.keys.dtype() == mx::uint32 &&
            layer.attn.values.dtype() == mx::uint32 &&
            layer.attn.keys.shape() == layer.attn.values.shape() &&
            layer.attn.keys.shape()[0] == 1 &&
            layer.attn.keys.shape()[1] == cfg_.num_key_value_heads &&
            layer.attn.keys.shape()[2] == layer.attn.cache_capacity &&
            layer.attn.keys.shape()[3] == cfg_.head_dim / 4 &&
            layer.attn.key_scales.ndim() == 4 &&
            layer.attn.key_biases.ndim() == 4 &&
            layer.attn.value_scales.ndim() == 4 &&
            layer.attn.value_biases.ndim() == 4 &&
            layer.attn.key_scales.dtype() == activation_dtype() &&
            layer.attn.key_biases.dtype() == activation_dtype() &&
            layer.attn.value_scales.dtype() == activation_dtype() &&
            layer.attn.value_biases.dtype() == activation_dtype() &&
            layer.attn.key_scales.shape() == q8_parameter_shape &&
            layer.attn.key_biases.shape() == q8_parameter_shape &&
            layer.attn.value_scales.shape() == q8_parameter_shape &&
            layer.attn.value_biases.shape() == q8_parameter_shape;
        if (layer.attn.offset < s.offset ||
            layer.attn.cache_length < s.cache_length ||
            layer.attn.cache_capacity < s.cache_length ||
            layer.attn.cache_bits != s.cache_bits ||
            (!bf16_storage_valid && !q8_storage_valid)) {
          throw std::runtime_error(
              "attention cache storage cannot restore snapshot prefix");
        }
      } else {
        layer.attn.cache_capacity = s.cache_capacity;
        layer.attn.cache_bits = s.cache_bits;
        layer.attn.keys = s.keys;
        layer.attn.key_scales = s.key_scales;
        layer.attn.key_biases = s.key_biases;
        layer.attn.values = s.values;
        layer.attn.value_scales = s.value_scales;
        layer.attn.value_biases = s.value_biases;
      }
      layer.attn.offset = s.offset;
      layer.attn.cache_length = s.cache_length;
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

std::uint64_t attention_cache_digest(const FullAttn& cache) {
  return attention_cache_digest(cache, true);
}

std::uint64_t attention_cache_digest(const FullAttn& cache, bool include_capacity) {
  if (cache.cache_length < 0 || cache.offset < 0 ||
      cache.cache_capacity < cache.cache_length) {
    throw std::runtime_error("invalid attention cache digest metadata");
  }
  std::uint64_t digest = UINT64_C(14695981039346656037);
  const auto mix = [&digest](unsigned char byte) {
    digest = (digest ^ byte) * UINT64_C(1099511628211);
  };
  const auto mix_integer = [&mix](int value) {
    const auto bits = static_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
      mix(static_cast<unsigned char>(bits >> shift));
    }
  };
  mix_integer(cache.offset);
  mix_integer(cache.cache_length);
  if (include_capacity) mix_integer(cache.cache_capacity);
  if (cache.cache_bits != 8 && cache.cache_bits != 16) {
    throw std::runtime_error("unsupported attention cache digest format");
  }
  if (cache.cache_bits == 8) {
    // Separate packed payloads from the legacy dense digest domain.
    mix_integer(cache.cache_bits);
    if (cache.cache_length == 0) return digest;
    if (cache.keys.ndim() != 4 || cache.keys.shape()[0] <= 0 ||
        cache.keys.shape()[1] <= 0 || cache.keys.shape()[3] <= 0 ||
        cache.keys.shape()[3] % 16 != 0 ||
        cache.keys.shape()[2] != cache.cache_capacity ||
        cache.keys.shape() != cache.values.shape() ||
        cache.keys.dtype() != mx::uint32 || cache.values.dtype() != mx::uint32) {
      throw std::runtime_error("invalid Q8 attention cache digest payloads");
    }
    auto parameter_shape = cache.keys.shape();
    parameter_shape[3] /= 16;  // Four packed Q8 values per word, group size 64.
    for (const auto& parameter : {cache.key_scales, cache.key_biases,
                                 cache.value_scales, cache.value_biases}) {
      if (parameter.shape() != parameter_shape ||
          parameter.dtype() != activation_dtype()) {
        throw std::runtime_error("invalid Q8 attention cache digest coefficients");
      }
    }
    for (const auto& source : {cache.keys, cache.key_scales, cache.key_biases,
                              cache.values, cache.value_scales, cache.value_biases}) {
      auto active_shape = source.shape();
      active_shape[2] = cache.cache_length;
      for (int dimension : include_capacity ? source.shape() : active_shape)
        mix_integer(dimension);
      const auto active = mx::contiguous(slice(source, {0, 0, 0, 0}, active_shape));
      eval(active);
      const auto* bytes = active.data<unsigned char>();
      for (std::size_t i = 0; i < active.nbytes(); ++i) mix(bytes[i]);
    }
    return digest;
  }

  if (cache.cache_length == 0) {
    return digest;
  }
  if (cache.keys.ndim() != 4 || cache.keys.shape() != cache.values.shape() ||
      cache.keys.shape()[2] != cache.cache_capacity ||
      cache.keys.dtype() != activation_dtype() || cache.values.dtype() != activation_dtype()) {
    throw std::runtime_error("invalid attention cache digest arrays");
  }
  auto active_shape = cache.keys.shape();
  active_shape[2] = cache.cache_length;
  for (int dimension : include_capacity ? cache.keys.shape() : active_shape) {
    mix_integer(dimension);
  }
  for (const auto& source : {cache.keys, cache.values}) {
    const auto active = mx::contiguous(slice(source, {0, 0, 0, 0}, active_shape));
    eval(active);
    const auto* bytes = reinterpret_cast<const unsigned char*>(
        active.data<std::uint16_t>());
    for (std::size_t i = 0; i < active.nbytes(); ++i) {
      mix(bytes[i]);
    }
  }
  return digest;
}

std::uint64_t Engine::mtp_history_digest() const {
  return mtp_history_digest(true);
}

std::uint64_t Engine::mtp_history_digest(bool include_capacity) const {
  if (!mtp_valid_ || !mtp_committed_history_enabled_ || mtp_cycle_pending_ ||
      mtp_layer_.attn.cache_length != target_sequence_length() - 1 ||
      mtp_layer_.attn.offset != mtp_layer_.attn.cache_length) {
    throw std::runtime_error("MTP history digest requires committed state");
  }
  return attention_cache_digest(mtp_layer_.attn, include_capacity);
}

std::uint64_t Engine::target_state_digest() const {
  std::uint64_t digest = UINT64_C(14695981039346656037);
  const auto mix = [&digest](unsigned char byte) {
    digest = (digest ^ byte) * UINT64_C(1099511628211);
  };
  const auto mix_integer = [&mix](std::uint64_t value) {
    for (int shift = 0; shift < 64; shift += 8) {
      mix(static_cast<unsigned char>(value >> shift));
    }
  };
  const auto mix_array = [&](const array& value) {
    const auto compact = mx::contiguous(value);
    eval(compact);
    mix_integer(compact.ndim());
    for (int dimension : compact.shape()) mix_integer(dimension);
    mix_integer(compact.nbytes());
    const auto* bytes = compact.data<unsigned char>();
    for (std::size_t i = 0; i < compact.nbytes(); ++i) mix(bytes[i]);
  };
  for (const auto& layer : layers_) {
    mix(layer.is_linear);
    if (layer.is_linear) {
      mix(layer.linear.has_state);
      if (layer.linear.has_state) {
        mix_array(layer.linear.conv_state);
        mix_array(layer.linear.rec_state);
      }
    } else {
      mix_integer(attention_cache_digest(layer.attn, false));
    }
  }
  mix_array(last_hidden_);
  return digest;
}

int Engine::target_sequence_length() const {
  int sequence_length = -1;
  for (const DecoderLayer& layer : layers_) {
    if (layer.is_linear) {
      continue;
    }
    if (sequence_length < 0) {
      sequence_length = layer.attn.offset;
    } else if (layer.attn.offset != sequence_length) {
      throw std::runtime_error("inconsistent target attention offsets");
    }
  }
  if (sequence_length < 0) {
    throw std::runtime_error("target has no full-attention layer");
  }
  return sequence_length;
}

int Engine::serialized_attention_cache_chunk_size(
    int requested_tokens, bool include_mtp) const {
  if (!serialize_attention_cache_growth_) {
    return requested_tokens;
  }
  int chunk_size = requested_tokens;
  for (const DecoderLayer& layer : layers_) {
    if (!layer.is_linear) {
      chunk_size = std::min(
          chunk_size,
          serialized_attention_cache_growth_chunk_size(
              layer.attn.cache_length,
              layer.attn.cache_capacity,
              requested_tokens));
    }
  }
  if (include_mtp && mtp_valid_) {
    chunk_size = std::min(
        chunk_size,
        serialized_attention_cache_growth_chunk_size(
            mtp_layer_.attn.cache_length,
            mtp_layer_.attn.cache_capacity,
            requested_tokens));
  }
  return chunk_size;
}

int Engine::bounded_post_growth_mtp_prefill_chunk_size(
    int requested_tokens, bool include_mtp) const {
  if (post_growth_mtp_prefill_chunk_size_ == 0 || !include_mtp ||
      !mtp_valid_) {
    return requested_tokens;
  }
  int target_capacity = -1;
  for (const DecoderLayer& layer : layers_) {
    if (layer.is_linear) {
      continue;
    }
    if (target_capacity < 0) {
      target_capacity = layer.attn.cache_capacity;
    } else if (layer.attn.cache_capacity != target_capacity) {
      throw std::runtime_error("inconsistent target attention capacities");
    }
  }
  if (target_capacity < 0) {
    throw std::runtime_error("target has no full-attention layer");
  }
  return post_growth_prefill_chunk_size(
      target_sequence_length(),
      target_capacity,
      mtp_layer_.attn.cache_length,
      mtp_layer_.attn.cache_capacity,
      requested_tokens,
      post_growth_mtp_prefill_chunk_size_);
}

array Engine::mtp_seed_hidden() const {
  if (!native_mtp_post_norm_seed_enabled()) {
    return last_hidden_;
  }
  return mx::fast::rms_norm(
      last_hidden_, final_norm_, cfg_.rms_norm_eps);
}

void Engine::mtp_reset() {
  mtp_layer_.attn.cache_length = 0;
  mtp_layer_.attn.offset = target_sequence_length();
  mtp_cycle_snapshot_ = LayerSnap{};
  mtp_cycle_previous_hidden_ = array(0);
  mtp_cycle_pending_ = false;
}

void Engine::mtp_append_history(
    const array& target_hidden, const array& token_ids) {
  if (target_hidden.ndim() != 3 || target_hidden.shape()[0] != 1 ||
      token_ids.ndim() != 2 || token_ids.shape()[0] != 1 ||
      target_hidden.shape()[1] != token_ids.shape()[1] ||
      target_hidden.shape()[1] < 1 || token_ids.dtype() != mx::int32) {
    throw std::runtime_error("invalid aligned MTP history append");
  }
  if (mtp_layer_.attn.offset != mtp_layer_.attn.cache_length) {
    throw std::runtime_error("noncontiguous committed MTP history");
  }
  const int previous_length = mtp_layer_.attn.cache_length;
  const int token_count = static_cast<int>(token_ids.shape()[1]);
  // Committed history consumes only K/V. Evaluating the decoder output also
  // executes its attention read, output projection, MLP, and final norm even
  // though no caller uses that result. Keep the identical cache dependency
  // graph and let MLX discard those unused output-only branches.
  (void)mtp_forward(embed(token_ids), target_hidden);
  eval(mtp_layer_.attn.keys, mtp_layer_.attn.values);
  if (mtp_layer_.attn.cache_length != previous_length + token_count ||
      mtp_layer_.attn.offset != mtp_layer_.attn.cache_length) {
    throw std::runtime_error("invalid committed MTP history length");
  }
}

void Engine::mtp_append_prompt_history(
    const array& target_hidden,
    const int32_t* tokens,
    int token_count,
    const array& previous_hidden,
    bool has_previous_hidden) {
  if (token_count < 1) {
    throw std::runtime_error("MTP prompt history requires a token");
  }
  const int current_target_length = target_sequence_length();
  const int previous_target_length = current_target_length - token_count;
  const int expected_previous_mtp_length =
      std::max(0, previous_target_length - 1);
  if (previous_target_length < 0 ||
      has_previous_hidden != (previous_target_length > 0) ||
      mtp_layer_.attn.cache_length != expected_previous_mtp_length ||
      mtp_layer_.attn.offset != expected_previous_mtp_length) {
    throw std::runtime_error("invalid MTP prompt history state");
  }
  array token_ids(tokens, {1, token_count}, mx::int32);
  auto [aligned_hidden, aligned_tokens] = align_mtp_committed_history(
      target_hidden,
      token_ids,
      previous_hidden,
      has_previous_hidden,
      final_norm_,
      cfg_.rms_norm_eps,
      native_mtp_post_norm_seed_enabled());
  if (aligned_tokens.shape()[1] > 0) {
    mtp_append_history(aligned_hidden, aligned_tokens);
  }
  if (mtp_layer_.attn.cache_length != current_target_length - 1 ||
      mtp_layer_.attn.offset != current_target_length - 1) {
    throw std::runtime_error("MTP prompt history is not one token behind");
  }
  if (native_spec_trace_enabled()) {
    std::fprintf(
        stderr,
        "qwen38_mtp prompt_history target=%d mtp=%d appended=%d\n",
        current_target_length,
        mtp_layer_.attn.cache_length,
        static_cast<int>(aligned_tokens.shape()[1]));
  }
}

void Engine::mtp_begin_committed_cycle() {
  if (!mtp_committed_history_enabled_) {
    throw std::runtime_error("committed MTP history is disabled");
  }
  if (mtp_cycle_pending_) {
    throw std::runtime_error("MTP committed cycle already pending");
  }
  const int target_length = target_sequence_length();
  if (target_length < 1 ||
      mtp_layer_.attn.cache_length != target_length - 1 ||
      mtp_layer_.attn.offset != target_length - 1 ||
      last_hidden_.ndim() < 1) {
    throw std::runtime_error("invalid MTP committed cycle state");
  }
  mtp_cycle_snapshot_.is_linear = false;
  mtp_cycle_snapshot_.keys = mtp_layer_.attn.keys;
  mtp_cycle_snapshot_.values = mtp_layer_.attn.values;
  mtp_cycle_snapshot_.offset = mtp_layer_.attn.offset;
  mtp_cycle_snapshot_.cache_length = mtp_layer_.attn.cache_length;
  mtp_cycle_snapshot_.cache_capacity = mtp_layer_.attn.cache_capacity;
  mtp_cycle_previous_hidden_ = last_hidden_;
  mtp_cycle_pending_ = true;
  if (native_spec_trace_enabled()) {
    std::fprintf(
        stderr,
        "qwen38_mtp cycle_begin target=%d mtp=%d\n",
        target_length,
        mtp_layer_.attn.cache_length);
  }
}

void Engine::mtp_commit_cycle(
    const int32_t* tokens,
    int token_count,
    const array& committed_hidden) {
  if (!mtp_cycle_pending_ || token_count < 1 ||
      committed_hidden.ndim() != 3 || committed_hidden.shape()[0] != 1 ||
      committed_hidden.shape()[1] < token_count ||
      committed_hidden.shape()[2] != cfg_.hidden_size) {
    throw std::runtime_error("invalid committed MTP cycle result");
  }
  mtp_layer_.attn.keys = mtp_cycle_snapshot_.keys;
  mtp_layer_.attn.values = mtp_cycle_snapshot_.values;
  mtp_layer_.attn.offset = mtp_cycle_snapshot_.offset;
  mtp_layer_.attn.cache_length = mtp_cycle_snapshot_.cache_length;
  mtp_layer_.attn.cache_capacity = mtp_cycle_snapshot_.cache_capacity;

  array token_ids(tokens, {1, token_count}, mx::int32);
  array hidden_prefix = slice(
      committed_hidden,
      {0, 0, 0},
      {1, token_count, cfg_.hidden_size});
  auto [aligned_hidden, aligned_tokens] = align_mtp_committed_history(
      hidden_prefix,
      token_ids,
      mtp_cycle_previous_hidden_,
      /*has_previous_hidden=*/true,
      final_norm_,
      cfg_.rms_norm_eps,
      native_mtp_post_norm_seed_enabled());
  mtp_append_history(aligned_hidden, aligned_tokens);

  const int target_length = target_sequence_length();
  if (mtp_layer_.attn.cache_length != target_length - 1 ||
      mtp_layer_.attn.offset != target_length - 1) {
    throw std::runtime_error("committed MTP cache is not one token behind");
  }
  if (native_spec_trace_enabled()) {
    std::fprintf(
        stderr,
        "qwen38_mtp cycle_commit target=%d mtp=%d appended=%d\n",
        target_length,
        mtp_layer_.attn.cache_length,
        token_count);
  }
  mtp_cycle_snapshot_ = LayerSnap{};
  mtp_cycle_previous_hidden_ = array(0);
  mtp_cycle_pending_ = false;
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
  array hid = mtp_seed_hidden();
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

void Engine::load_dflash2(
    const std::unordered_map<std::string, array>& weights) {
  constexpr size_t kDenseTensorCount = 81;
  constexpr size_t kAffineTensorCount = 175;
  constexpr int kHiddenSize = 5120;
  constexpr int kIntermediateSize = 17408;
  constexpr int kDraftLayers = 5;
  constexpr int kDraftHeads = 32;
  constexpr int kDraftKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSelectorRank = 256;
  constexpr int kDynamicWidth = 1280;
  const bool affine = weights.size() == kAffineTensorCount;
  if (!affine && weights.size() != kDenseTensorCount) {
    throw std::runtime_error(
        "DFlash2 checkpoint contains " + std::to_string(weights.size()) +
        " tensors; expected 81 dense or 175 affine tensors");
  }
  if (cfg_.hidden_size != kHiddenSize || cfg_.num_hidden_layers != 64 ||
      cfg_.vocab_size != 248320) {
    throw std::runtime_error(
        "Qwen3.8-27B DFlash2 requires target shape 64x5120x248320");
  }
  dflash_selector_temperature_ = native_dflash_selector_temperature();
  dflash_mean_q_threshold_ = native_dflash_mean_q_threshold();
  if (native_spec_trace_enabled()) {
    std::fprintf(
        stderr,
        "qwen38_dflash selector_temperature=%.6f mean_q_threshold=%.6f\n",
        dflash_selector_temperature_,
        dflash_mean_q_threshold_);
  }

  const auto dense = [this, &weights](
                         const std::string& name,
                         const mx::Shape& shape) -> array {
    array value = require(weights, name);
    if (value.shape() != shape || value.dtype() != activation_dtype()) {
      throw std::runtime_error("invalid DFlash2 tensor " + name);
    }
    return value;
  };
  const auto linear = [this, &weights, &dense, affine](
                          const std::string& prefix,
                          int output_features,
                          int input_features) -> QLinear {
    if (!affine) {
      QLinear value;
      value.w = dense(
          prefix + ".weight", {output_features, input_features});
      value.valid = true;
      return value;
    }
    QLinear value = load_qlinear(weights, prefix);
    if (value.bits != 4 || value.group_size != 64 ||
        value.w.dtype() != mx::uint32 ||
        value.scales.dtype() != activation_dtype() ||
        value.biases.dtype() != activation_dtype() ||
        value.w.shape()[0] != output_features ||
        value.scales.shape()[0] != output_features ||
        value.scales.shape().back() * value.group_size != input_features) {
      throw std::runtime_error("invalid DFlash2 affine linear " + prefix);
    }
    return value;
  };

  dflash_fc_ = linear("fc", kHiddenSize, kDraftLayers * kHiddenSize);
  dflash_hidden_norm_ =
      dense("hidden_norm.weight", {kHiddenSize});
  dflash_norm_ = dense("norm.weight", {kHiddenSize});
  dflash_selector_hidden_ = linear(
      "candidate_selector.hidden_projection", kSelectorRank, kHiddenSize);
  dflash_predecessor_ = dense(
      "candidate_selector.predecessor_codebook",
      {cfg_.vocab_size, kSelectorRank});
  dflash_successor_ = dense(
      "candidate_selector.successor_codebook",
      {cfg_.vocab_size, kSelectorRank});

  dflash_layers_.clear();
  dflash_layers_.resize(kDraftLayers);
  for (int index = 0; index < kDraftLayers; ++index) {
    DFlashLayer& layer = dflash_layers_[static_cast<size_t>(index)];
    const std::string prefix = "layers." + std::to_string(index);
    layer.input_norm =
        dense(prefix + ".input_layernorm.weight", {kHiddenSize});
    layer.post_norm =
        dense(prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    layer.gate_proj =
        linear(prefix + ".mlp.gate_proj", kIntermediateSize, kHiddenSize);
    layer.up_proj =
        linear(prefix + ".mlp.up_proj", kIntermediateSize, kHiddenSize);
    layer.down_proj =
        linear(prefix + ".mlp.down_proj", kHiddenSize, kIntermediateSize);
    layer.attn.q_proj =
        linear(prefix + ".self_attn.q_proj", kDraftHeads * kHeadDim, kHiddenSize);
    layer.attn.k_proj = linear(
        prefix + ".self_attn.k_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.v_proj = linear(
        prefix + ".self_attn.v_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.o_proj = linear(
        prefix + ".self_attn.o_proj", kHiddenSize, kDraftHeads * kHeadDim);
    layer.attn.q_norm =
        dense(prefix + ".self_attn.q_norm.weight", {kHeadDim});
    layer.attn.k_norm =
        dense(prefix + ".self_attn.k_norm.weight", {kHeadDim});
    layer.attention_conv.base_kernel = dense(
        prefix + ".attention_conv.base_kernel", {2, 2, kHiddenSize});
    layer.attention_conv.kernel_projection = linear(
        prefix + ".attention_conv.kernel_projection",
        kDynamicWidth,
        kHiddenSize);
    layer.mlp_conv.base_kernel = dense(
        prefix + ".mlp_conv.base_kernel", {2, 2, kHiddenSize});
    layer.mlp_conv.kernel_projection = linear(
        prefix + ".mlp_conv.kernel_projection", kDynamicWidth, kHiddenSize);
  }

  mtp_valid_ = false;
  mtp_committed_history_enabled_ = false;
  mtp_cycle_pending_ = false;
  dflash_valid_ = true;
  dspark_valid_ = false;
  dflash_reset();
}

void Engine::load_dspark(
    const std::unordered_map<std::string, array>& weights) {
  constexpr size_t kDenseTensorCount = 62;
  constexpr size_t kAffineTensorCount = 136;
  constexpr int kHiddenSize = 5120;
  constexpr int kIntermediateSize = 17408;
  constexpr int kDraftLayers = 5;
  constexpr int kDraftTokens = 7;
  constexpr int kDraftHeads = 32;
  constexpr int kDraftKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kMarkovRank = 256;
  const bool affine = weights.size() == kAffineTensorCount;
  if (!affine && weights.size() != kDenseTensorCount) {
    throw std::runtime_error(
        "DSpark checkpoint contains " + std::to_string(weights.size()) +
        " tensors; expected 62 dense or 136 affine tensors");
  }
  if (cfg_.hidden_size != kHiddenSize || cfg_.num_hidden_layers != 64 ||
      cfg_.vocab_size != 248320) {
    throw std::runtime_error(
        "Qwen3.8-27B DSpark requires target shape 64x5120x248320");
  }

  const auto dense = [this, &weights](
                         const std::string& name,
                         const mx::Shape& shape) -> array {
    array value = require(weights, name);
    if (value.shape() != shape || value.dtype() != activation_dtype()) {
      throw std::runtime_error("invalid DSpark tensor " + name);
    }
    return value;
  };
  const auto linear = [this, &weights, &dense, affine](
                          const std::string& prefix,
                          int output_features,
                          int input_features) -> QLinear {
    if (!affine) {
      QLinear value;
      value.w = dense(
          prefix + ".weight", {output_features, input_features});
      value.valid = true;
      return value;
    }
    QLinear value = load_qlinear(weights, prefix);
    if (value.bits != 4 || value.group_size != 64 ||
        value.w.dtype() != mx::uint32 ||
        value.scales.dtype() != activation_dtype() ||
        value.biases.dtype() != activation_dtype() ||
        value.w.shape()[0] != output_features ||
        value.scales.shape()[0] != output_features ||
        value.scales.shape().back() * value.group_size != input_features) {
      throw std::runtime_error("invalid DSpark affine linear " + prefix);
    }
    return value;
  };

  dspark_fc_ = linear("fc", kHiddenSize, kDraftLayers * kHiddenSize);
  dspark_hidden_norm_ = dense("hidden_norm.weight", {kHiddenSize});
  dspark_norm_ = dense("norm.weight", {kHiddenSize});
  dspark_markov_w1_ = dense(
      "markov_head.markov_w1.weight", {cfg_.vocab_size, kMarkovRank});
  dspark_markov_w2_ =
      linear("markov_head.markov_w2", cfg_.vocab_size, kMarkovRank);
  dspark_confidence_weight_ = dense(
      "confidence_head.proj.weight", {1, kHiddenSize + kMarkovRank});
  dspark_confidence_bias_ = dense("confidence_head.proj.bias", {1});
  dspark_verify_draft_tokens_ = native_dspark_verify_draft_tokens();
  dspark_confidence_cost_ratio_ = native_dspark_confidence_cost_ratio();
  dspark_bypass_refills_ = native_dspark_bypass_refills();
  if (dspark_confidence_cost_ratio_ > 0.0f &&
      dspark_verify_draft_tokens_ != kDraftTokens) {
    throw std::runtime_error(
        "DSpark confidence budgeting requires seven proposal tokens");
  }
  if (dspark_confidence_cost_ratio_ > 0.0f && !sampling_enabled_) {
    throw std::runtime_error(
        "DSpark confidence budgeting requires native sampling");
  }
  if (dspark_bypass_refills_ > 0 &&
      dspark_confidence_cost_ratio_ <= 0.0f) {
    throw std::runtime_error(
        "DSpark bypass refills require confidence budgeting");
  }

  dspark_layers_.clear();
  dspark_layers_.resize(kDraftLayers);
  for (int index = 0; index < kDraftLayers; ++index) {
    DSparkLayer& layer = dspark_layers_[static_cast<size_t>(index)];
    const std::string prefix = "layers." + std::to_string(index);
    layer.input_norm =
        dense(prefix + ".input_layernorm.weight", {kHiddenSize});
    layer.post_norm =
        dense(prefix + ".post_attention_layernorm.weight", {kHiddenSize});
    layer.gate_proj =
        linear(prefix + ".mlp.gate_proj", kIntermediateSize, kHiddenSize);
    layer.up_proj =
        linear(prefix + ".mlp.up_proj", kIntermediateSize, kHiddenSize);
    layer.down_proj =
        linear(prefix + ".mlp.down_proj", kHiddenSize, kIntermediateSize);
    layer.attn.q_proj = linear(
        prefix + ".self_attn.q_proj", kDraftHeads * kHeadDim, kHiddenSize);
    layer.attn.k_proj = linear(
        prefix + ".self_attn.k_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.v_proj = linear(
        prefix + ".self_attn.v_proj", kDraftKvHeads * kHeadDim, kHiddenSize);
    layer.attn.o_proj = linear(
        prefix + ".self_attn.o_proj", kHiddenSize, kDraftHeads * kHeadDim);
    layer.attn.q_norm =
        dense(prefix + ".self_attn.q_norm.weight", {kHeadDim});
    layer.attn.k_norm =
        dense(prefix + ".self_attn.k_norm.weight", {kHeadDim});
  }

  mtp_valid_ = false;
  mtp_committed_history_enabled_ = false;
  mtp_cycle_pending_ = false;
  dflash_valid_ = false;
  dspark_valid_ = true;
  dspark_reset();
}

void Engine::dspark_reset() {
  dspark_context_offset_ = 0;
  dspark_bypass_remaining_ = 0;
  for (DSparkLayer& layer : dspark_layers_) {
    layer.attn.keys = array(0);
    layer.attn.values = array(0);
    layer.attn.cache_length = 0;
    layer.attn.cache_capacity = 0;
  }
}

void Engine::dspark_append_layer_context(
    DSparkAttention& attn,
    const array& projected,
    int position_offset) {
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  const int batch = static_cast<int>(projected.shape()[0]);
  const int length = static_cast<int>(projected.shape()[1]);
  if (length <= 0) {
    return;
  }
  if (attn.cache_length != position_offset) {
    throw std::runtime_error("noncontiguous DSpark context append");
  }

  array keys = reshape(
      attn.k_proj(projected), {batch, length, kKvHeads, kHeadDim});
  keys = mx::fast::rms_norm(keys, attn.k_norm, cfg_.rms_norm_eps);
  keys = dspark_yarn_rope(transpose(keys, {0, 2, 1, 3}), position_offset);
  array values = transpose(
      reshape(
          attn.v_proj(projected),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  const int needed = attn.cache_length + length;
  if (attn.cache_capacity < needed) {
    int capacity = std::max(256, attn.cache_capacity);
    while (capacity < needed) {
      capacity *= 2;
    }
    array new_keys =
        zeros({batch, kKvHeads, capacity, kHeadDim}, keys.dtype());
    array new_values =
        zeros({batch, kKvHeads, capacity, kHeadDim}, values.dtype());
    if (attn.cache_length > 0) {
      array active_keys = slice(
          attn.keys,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      array active_values = slice(
          attn.values,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      new_keys = slice_update(
          new_keys,
          active_keys,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      new_values = slice_update(
          new_values,
          active_values,
          {0, 0, 0, 0},
          {batch, kKvHeads, attn.cache_length, kHeadDim});
      eval(new_keys, new_values);
    }
    attn.keys = new_keys;
    attn.values = new_values;
    attn.cache_capacity = capacity;
  }
  attn.keys = slice_update(
      attn.keys,
      keys,
      {0, 0, attn.cache_length, 0},
      {batch, kKvHeads, needed, kHeadDim});
  attn.values = slice_update(
      attn.values,
      values,
      {0, 0, attn.cache_length, 0},
      {batch, kKvHeads, needed, kHeadDim});
  attn.cache_length = needed;
  async_eval(attn.keys, attn.values);
}

void Engine::dspark_append_context(
    const std::vector<array>& captured, int token_count) {
  if (captured.size() != 5 || token_count <= 0) {
    throw std::runtime_error("invalid DSpark target context capture");
  }
  const int available = static_cast<int>(captured.front().shape()[1]);
  if (token_count > available) {
    throw std::runtime_error("DSpark committed context exceeds target capture");
  }
  std::vector<array> selected;
  selected.reserve(captured.size());
  for (const array& value : captured) {
    if (value.ndim() != 3 || value.shape()[0] != 1 ||
        value.shape()[1] != available || value.shape()[2] != cfg_.hidden_size) {
      throw std::runtime_error("inconsistent DSpark target capture shape");
    }
    selected.push_back(slice(
        value,
        {0, 0, 0},
        {1, token_count, cfg_.hidden_size}));
  }
  array projected = mx::fast::rms_norm(
      dspark_fc_(concatenate(selected, -1)),
      dspark_hidden_norm_,
      cfg_.rms_norm_eps);
  for (DSparkLayer& layer : dspark_layers_) {
    dspark_append_layer_context(
        layer.attn, projected, dspark_context_offset_);
  }
  dspark_context_offset_ += token_count;
}

void Engine::dflash_reset() {
  dflash_context_offset_ = 0;
  for (DFlashLayer& layer : dflash_layers_) {
    layer.attn.keys = array(0);
    layer.attn.values = array(0);
    layer.attn.positions = array(0);
    layer.attn.cache_length = 0;
  }
}

void Engine::dflash_append_layer_context(
    DFlashAttention& attn,
    const array& projected,
    int position_offset) {
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSinkSize = 64;
  constexpr int kWindowSize = 2048;
  const int batch = static_cast<int>(projected.shape()[0]);
  const int length = static_cast<int>(projected.shape()[1]);
  if (length <= 0) {
    return;
  }

  array keys = reshape(
      attn.k_proj(projected), {batch, length, kKvHeads, kHeadDim});
  keys = mx::fast::rms_norm(keys, attn.k_norm, 1e-6f);
  keys = transpose(keys, {0, 2, 1, 3});
  keys = mx::fast::rope(
      keys,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      position_offset);
  array values = transpose(
      reshape(
          attn.v_proj(projected),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});
  array positions = mx::arange(
      position_offset, position_offset + length, mx::int32);

  if (attn.cache_length == 0) {
    attn.keys = keys;
    attn.values = values;
    attn.positions = positions;
    attn.cache_length = length;
  } else {
    attn.keys = concatenate({attn.keys, keys}, 2);
    attn.values = concatenate({attn.values, values}, 2);
    attn.positions = concatenate({attn.positions, positions}, 0);
    attn.cache_length += length;
  }

  const int maximum = kSinkSize + kWindowSize;
  if (attn.cache_length > maximum) {
    const int tail_start = attn.cache_length - kWindowSize;
    array sink_keys = slice(
        attn.keys, {0, 0, 0, 0}, {batch, kKvHeads, kSinkSize, kHeadDim});
    array tail_keys = slice(
        attn.keys,
        {0, 0, tail_start, 0},
        {batch, kKvHeads, attn.cache_length, kHeadDim});
    array sink_values = slice(
        attn.values, {0, 0, 0, 0}, {batch, kKvHeads, kSinkSize, kHeadDim});
    array tail_values = slice(
        attn.values,
        {0, 0, tail_start, 0},
        {batch, kKvHeads, attn.cache_length, kHeadDim});
    array sink_positions =
        slice(attn.positions, {0}, {kSinkSize});
    array tail_positions = slice(
        attn.positions, {tail_start}, {attn.cache_length});
    attn.keys = concatenate({sink_keys, tail_keys}, 2);
    attn.values = concatenate({sink_values, tail_values}, 2);
    attn.positions = concatenate({sink_positions, tail_positions}, 0);
    attn.cache_length = maximum;
  }
  async_eval(attn.keys, attn.values, attn.positions);
}

void Engine::dflash_append_context(
    const std::vector<array>& captured, int token_count) {
  constexpr int kSinkSize = 64;
  constexpr int kWindowSize = 2048;
  if (captured.size() != 5 || token_count <= 0) {
    throw std::runtime_error("invalid DFlash2 target context capture");
  }
  const int available = static_cast<int>(captured.front().shape()[1]);
  if (token_count > available) {
    throw std::runtime_error("DFlash2 committed context exceeds target capture");
  }
  for (const array& value : captured) {
    if (value.ndim() != 3 || value.shape()[0] != 1 ||
        value.shape()[1] != available || value.shape()[2] != cfg_.hidden_size) {
      throw std::runtime_error("inconsistent DFlash2 target capture shape");
    }
  }

  std::vector<std::pair<int, int>> spans;
  const bool empty = dflash_layers_.front().attn.cache_length == 0;
  if (empty && token_count > kSinkSize + kWindowSize) {
    spans.emplace_back(0, kSinkSize);
    spans.emplace_back(token_count - kWindowSize, token_count);
  } else if (!empty && token_count > kWindowSize) {
    spans.emplace_back(token_count - kWindowSize, token_count);
  } else {
    spans.emplace_back(0, token_count);
  }

  for (const auto& [start, end] : spans) {
    std::vector<array> selected;
    selected.reserve(captured.size());
    for (const array& value : captured) {
      selected.push_back(slice(
          value,
          {0, start, 0},
          {1, end, cfg_.hidden_size}));
    }
    array projected = mx::fast::rms_norm(
        dflash_fc_(concatenate(selected, -1)),
        dflash_hidden_norm_,
        cfg_.rms_norm_eps);
    for (DFlashLayer& layer : dflash_layers_) {
      dflash_append_layer_context(
          layer.attn, projected, dflash_context_offset_ + start);
    }
  }
  dflash_context_offset_ += token_count;
}

void Engine::draft_append_context(
    const std::vector<array>& captured, int token_count) {
  if (dspark_valid_) {
    dspark_append_context(captured, token_count);
    return;
  }
  if (dflash_valid_) {
    dflash_append_context(captured, token_count);
    return;
  }
  if (mtp_valid_) {
    return;
  }
  throw std::runtime_error("speculative draft context is unavailable");
}

array Engine::dflash_grouped_convolve(
    const array& hidden, const array& dynamic, const array& base) const {
  constexpr int kGroupSize = 16;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  const int hidden_size = static_cast<int>(hidden.shape()[2]);
  const int groups = hidden_size / kGroupSize;
  array blocks = reshape(
      hidden, {batch, length, groups, kGroupSize});
  array output = mx::zeros(blocks.shape(), hidden.dtype());
  for (int offset = 0; offset < 2; ++offset) {
    array values = blocks;
    if (offset != 0) {
      array leading =
          mx::zeros({batch, offset, groups, kGroupSize}, hidden.dtype());
      array preceding = slice(
          blocks,
          {0, 0, 0, 0},
          {batch, length - offset, groups, kGroupSize});
      values = concatenate({leading, preceding}, 1);
    }
    array kernel = reshape(
        astype(take(base, offset, 0), hidden.dtype()),
        {1, 1, groups, kGroupSize});
    array dynamic_offset = expand_dims(take(dynamic, offset, 2), -1);
    output = output + (kernel + dynamic_offset) * values;
  }
  return reshape(output, hidden.shape());
}

std::pair<array, array> Engine::dflash_conv_prepare(
    const DFlashDynamicConv& conv, const array& hidden) const {
  constexpr int kGroupSize = 16;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  const int groups = static_cast<int>(hidden.shape()[2]) / kGroupSize;
  array dynamic = reshape(
      conv.kernel_projection(hidden), {batch, length, 2, 2, groups});
  array prepared = dflash_grouped_convolve(
      hidden, take(dynamic, 0, 2), take(conv.base_kernel, 0, 0));
  return {prepared, take(dynamic, 1, 2)};
}

array Engine::dflash_conv_finish(
    const DFlashDynamicConv& conv,
    const array& hidden,
    const array& dynamic) const {
  return dflash_grouped_convolve(
      hidden, dynamic, take(conv.base_kernel, 1, 0));
}

array Engine::dflash_attention(
    DFlashAttention& attn, const array& hidden) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  constexpr int kSlidingWindow = 2048;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  if (attn.cache_length <= 0) {
    throw std::runtime_error("DFlash2 draft context cache is empty");
  }

  array queries = reshape(
      attn.q_proj(hidden), {batch, length, kHeads, kHeadDim});
  queries = mx::fast::rms_norm(queries, attn.q_norm, 1e-6f);
  queries = transpose(queries, {0, 2, 1, 3});
  queries = mx::fast::rope(
      queries,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      dflash_context_offset_);

  array noise_keys = reshape(
      attn.k_proj(hidden), {batch, length, kKvHeads, kHeadDim});
  noise_keys = mx::fast::rms_norm(noise_keys, attn.k_norm, 1e-6f);
  noise_keys = transpose(noise_keys, {0, 2, 1, 3});
  noise_keys = mx::fast::rope(
      noise_keys,
      kHeadDim,
      /*traditional=*/false,
      10000000.0f,
      1.0f,
      dflash_context_offset_);
  array noise_values = transpose(
      reshape(
          attn.v_proj(hidden),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  array keys = concatenate({attn.keys, noise_keys}, 2);
  array values = concatenate({attn.values, noise_values}, 2);
  array block_positions = mx::arange(
      dflash_context_offset_,
      dflash_context_offset_ + length,
      mx::int32);
  array key_positions = concatenate({attn.positions, block_positions}, 0);
  array query_grid = expand_dims(block_positions, 1);
  array key_grid = expand_dims(key_positions, 0);
  array context_mask = mx::logical_and(
      mx::less(key_grid, array(dflash_context_offset_, mx::int32)),
      mx::less(query_grid - key_grid, array(kSlidingWindow, mx::int32)));
  array block_mask =
      mx::greater_equal(key_grid, array(dflash_context_offset_, mx::int32));
  array mask = mx::logical_or(context_mask, block_mask);

  array output = mx::fast::scaled_dot_product_attention(
      queries,
      keys,
      values,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)),
      "",
      mask);
  output = reshape(
      transpose(output, {0, 2, 1, 3}),
      {batch, length, kHeads * kHeadDim});
  return attn.o_proj(output);
}

array Engine::dflash_forward(int32_t anchor) {
  constexpr int kBlockSize = 8;
  constexpr int32_t kMaskToken = 248070;
  int32_t tokens[kBlockSize];
  tokens[0] = anchor;
  std::fill(tokens + 1, tokens + kBlockSize, kMaskToken);
  array ids(tokens, {1, kBlockSize}, mx::int32);
  array hidden = embed(ids);
  for (DFlashLayer& layer : dflash_layers_) {
    array residual = hidden;
    auto attention_prepared = dflash_conv_prepare(
        layer.attention_conv,
        mx::fast::rms_norm(
            hidden, layer.input_norm, cfg_.rms_norm_eps));
    array attention_output = dflash_attention(
        layer.attn, attention_prepared.first);
    hidden = residual + dflash_conv_finish(
        layer.attention_conv,
        attention_output,
        attention_prepared.second);

    residual = hidden;
    auto mlp_prepared = dflash_conv_prepare(
        layer.mlp_conv,
        mx::fast::rms_norm(
            hidden, layer.post_norm, cfg_.rms_norm_eps));
    array mlp_output = layer.down_proj(
        silu(layer.gate_proj(mlp_prepared.first)) *
        layer.up_proj(mlp_prepared.first));
    hidden = residual + dflash_conv_finish(
        layer.mlp_conv, mlp_output, mlp_prepared.second);
  }
  hidden = mx::fast::rms_norm(
      hidden, dflash_norm_, cfg_.rms_norm_eps);
  return slice(
      hidden, {0, 1, 0}, {1, kBlockSize, cfg_.hidden_size});
}

array Engine::dspark_attention(
    DSparkAttention& attn, const array& hidden) {
  constexpr int kHeads = 32;
  constexpr int kKvHeads = 8;
  constexpr int kHeadDim = 128;
  const int batch = static_cast<int>(hidden.shape()[0]);
  const int length = static_cast<int>(hidden.shape()[1]);
  if (attn.cache_length <= 0 ||
      attn.cache_length != dspark_context_offset_ ||
      attn.cache_capacity < attn.cache_length) {
    throw std::runtime_error("invalid DSpark draft context cache");
  }

  array queries = reshape(
      attn.q_proj(hidden), {batch, length, kHeads, kHeadDim});
  queries = mx::fast::rms_norm(queries, attn.q_norm, cfg_.rms_norm_eps);
  queries = dspark_yarn_rope(
      transpose(queries, {0, 2, 1, 3}), dspark_context_offset_);

  array noise_keys = reshape(
      attn.k_proj(hidden), {batch, length, kKvHeads, kHeadDim});
  noise_keys =
      mx::fast::rms_norm(noise_keys, attn.k_norm, cfg_.rms_norm_eps);
  noise_keys = dspark_yarn_rope(
      transpose(noise_keys, {0, 2, 1, 3}), dspark_context_offset_);
  array noise_values = transpose(
      reshape(
          attn.v_proj(hidden),
          {batch, length, kKvHeads, kHeadDim}),
      {0, 2, 1, 3});

  array context_keys = slice(
      attn.keys,
      {0, 0, 0, 0},
      {batch, kKvHeads, attn.cache_length, kHeadDim});
  array context_values = slice(
      attn.values,
      {0, 0, 0, 0},
      {batch, kKvHeads, attn.cache_length, kHeadDim});
  array keys = concatenate({context_keys, noise_keys}, 2);
  array values = concatenate({context_values, noise_values}, 2);
  array output = mx::fast::scaled_dot_product_attention(
      queries,
      keys,
      values,
      1.0f / std::sqrt(static_cast<float>(kHeadDim)));
  output = reshape(
      transpose(output, {0, 2, 1, 3}),
      {batch, length, kHeads * kHeadDim});
  return attn.o_proj(output);
}

array Engine::dspark_forward(int32_t anchor) {
  constexpr int kBlockSize = 7;
  constexpr int32_t kMaskToken = 248070;
  int32_t tokens[kBlockSize];
  tokens[0] = anchor;
  std::fill(tokens + 1, tokens + kBlockSize, kMaskToken);
  array ids(tokens, {1, kBlockSize}, mx::int32);
  array hidden = embed(ids);
  for (DSparkLayer& layer : dspark_layers_) {
    array residual = hidden;
    array normalized = mx::fast::rms_norm(
        hidden, layer.input_norm, cfg_.rms_norm_eps);
    hidden = residual + dspark_attention(layer.attn, normalized);

    residual = hidden;
    normalized = mx::fast::rms_norm(
        hidden, layer.post_norm, cfg_.rms_norm_eps);
    hidden = residual + layer.down_proj(
        silu(layer.gate_proj(normalized)) * layer.up_proj(normalized));
  }
  return mx::fast::rms_norm(hidden, dspark_norm_, cfg_.rms_norm_eps);
}

std::pair<array, array> Engine::dspark_propose(
    const array& hidden, int32_t anchor, bool sampled) {
  constexpr int kBlockSize = 7;
  constexpr int kMarkovRank = 256;
  if (hidden.shape() != mx::Shape{1, kBlockSize, cfg_.hidden_size}) {
    throw std::runtime_error("invalid DSpark proposal hidden state");
  }
  array base_logits = lm_head_(hidden);
  array previous(&anchor, {1}, mx::int32);
  std::vector<array> path;
  std::vector<array> probabilities;
  path.reserve(kBlockSize);
  probabilities.reserve(kBlockSize);
  for (int position = 0; position < kBlockSize; ++position) {
    array base_row = reshape(
        slice(
            base_logits,
            {0, position, 0},
            {1, position + 1, cfg_.vocab_size}),
        {1, cfg_.vocab_size});
    array markov_embedding = reshape(
        take(dspark_markov_w1_, previous, 0), {1, kMarkovRank});
    array corrected = astype(base_row, mx::float32) +
        astype(dspark_markov_w2_(markov_embedding), mx::float32);
    if (sampled) {
      array probs = mx::softmax(corrected, -1, true);
      previous = astype(
          mx::random::categorical(mx::log(probs), -1), mx::int32);
      probabilities.push_back(probs);
    } else {
      previous = astype(mx::argmax(corrected, -1), mx::int32);
    }
    path.push_back(previous);
  }
  return {
      astype(mx::stack(path, 1), mx::int32),
      sampled ? mx::stack(probabilities, 1) : array(0)};
}

std::tuple<array, array, array> Engine::dflash_select(
    const array& hidden, const array& draft_logits, int32_t anchor) {
  constexpr int kTopK = 16;
  constexpr int kSelectorRank = 256;
  const int length = static_cast<int>(hidden.shape()[1]);
  const int vocab = static_cast<int>(draft_logits.shape()[2]);
  array partitioned = mx::argpartition(draft_logits, vocab - kTopK, -1);
  array candidates = slice(
      partitioned, {0, 0, vocab - kTopK}, {1, length, vocab});
  array unary = mx::take_along_axis(draft_logits, candidates, -1);
  array projected = dflash_selector_hidden_(hidden);
  array predecessor(&anchor, {1}, mx::int32);
  std::vector<array> path;
  std::vector<array> probabilities;
  path.reserve(static_cast<size_t>(length));
  probabilities.reserve(static_cast<size_t>(length));
  for (int position = 0; position < length; ++position) {
    array ids = reshape(
        slice(
            candidates,
            {0, position, 0},
            {1, position + 1, kTopK}),
        {1, kTopK});
    array unary_row = reshape(
        slice(
            unary,
            {0, position, 0},
            {1, position + 1, kTopK}),
        {1, kTopK});
    array hidden_row = reshape(
        slice(
            projected,
            {0, position, 0},
            {1, position + 1, kSelectorRank}),
        {1, kSelectorRank});
    array predecessor_embedding = take(
        dflash_predecessor_, predecessor, 0);
    array successor_embedding = take(dflash_successor_, ids, 0);
    array edges = sum(
        expand_dims(predecessor_embedding * hidden_row, 1) *
            successor_embedding,
        -1);
    array scores = astype(unary_row, mx::float32) +
        astype(edges, mx::float32);
    if (dflash_selector_temperature_ != 1.0f) {
      scores = scores /
          array(dflash_selector_temperature_, mx::float32);
    }
    array probs = mx::softmax(scores, -1, true);
    array selected = mx::random::categorical(mx::log(probs), -1);
    predecessor = squeeze(
        mx::take_along_axis(ids, expand_dims(selected, -1), -1), -1);
    path.push_back(predecessor);
    probabilities.push_back(probs);
  }
  return {
      astype(mx::stack(path, 1), mx::int32),
      astype(candidates, mx::int32),
      mx::stack(probabilities, 1)};
}

void Engine::draft_commit_verified_prefix(
    const TargetForward& verified, int token_count) {
  if (token_count <= 0 || verified.hidden.ndim() != 3 ||
      token_count > verified.hidden.shape()[1] ||
      verified.linear_tapes.size() != layers_.size() ||
      snap_.size() != layers_.size()) {
    throw std::runtime_error("invalid speculative verified-prefix commit");
  }

  std::vector<array> committed_states;
  committed_states.reserve(layers_.size() * 2);
  for (size_t index = 0; index < layers_.size(); ++index) {
    DecoderLayer& layer = layers_[index];
    const LayerSnap& snapshot_state = snap_[index];
    if (layer.is_linear) {
      const LinearCommitTape& tape = verified.linear_tapes[index];
      if (!snapshot_state.has_state || !tape.valid ||
          snapshot_state.conv.ndim() != 3 ||
          tape.conv_tokens.ndim() != 3 ||
          snapshot_state.conv.shape()[0] != tape.conv_tokens.shape()[0] ||
          snapshot_state.conv.shape()[2] != tape.conv_tokens.shape()[2] ||
          token_count > tape.conv_tokens.shape()[1]) {
        throw std::runtime_error("invalid speculative linear commit tape");
      }
      const int batch = static_cast<int>(snapshot_state.conv.shape()[0]);
      const int window = static_cast<int>(snapshot_state.conv.shape()[1]);
      const int width = static_cast<int>(snapshot_state.conv.shape()[2]);
      array committed_tokens = slice(
          tape.conv_tokens,
          {0, 0, 0},
          {batch, token_count, width});
      array combined = concatenate(
          {snapshot_state.conv, committed_tokens}, 1);
      layer.linear.conv_state = slice(
          combined,
          {0, token_count, 0},
          {batch, token_count + window, width});
      layer.linear.rec_state = gated_delta_commit(
          tape.keys,
          tape.decay,
          tape.delta,
          snapshot_state.rec,
          token_count);
      layer.linear.has_state = true;
      committed_states.push_back(layer.linear.conv_state);
      committed_states.push_back(layer.linear.rec_state);
      continue;
    }

    const int committed_length = snapshot_state.cache_length + token_count;
    if (layer.attn.cache_length < committed_length ||
        layer.attn.offset < snapshot_state.offset + token_count) {
      throw std::runtime_error("invalid speculative attention commit state");
    }
    layer.attn.cache_length = committed_length;
    layer.attn.offset = snapshot_state.offset + token_count;
  }

  const int hidden_size = static_cast<int>(verified.hidden.shape()[2]);
  last_hidden_ = reshape(
      slice(
          verified.hidden,
          {0, token_count - 1, 0},
          {1, token_count, hidden_size}),
      {1, hidden_size});
  draft_append_context(verified.captured, token_count);
  async_eval(std::move(committed_states));
}

int Engine::verify_speculative_block(
    int32_t token,
    const array& draft_tokens,
    const array& proposal_indices,
    const array& proposal_probs,
    const array& confidence,
    bool dense_proposal,
    bool greedy,
    float confidence_cost_ratio,
    float sparse_mean_q_threshold,
    const char* trace_tag) {
  constexpr int kMaxDraftTokens = 7;
  const bool trace = native_spec_trace_enabled();
  const auto started = std::chrono::steady_clock::now();
  if (draft_tokens.ndim() != 2 || draft_tokens.shape()[0] != 1 ||
      draft_tokens.shape()[1] < 1 ||
      draft_tokens.shape()[1] > kMaxDraftTokens ||
      draft_tokens.dtype() != mx::int32) {
    throw std::runtime_error("invalid speculative draft token block");
  }
  const int proposal_token_count = static_cast<int>(draft_tokens.shape()[1]);
  const bool has_confidence = confidence.ndim() != 0;
  if (!std::isfinite(confidence_cost_ratio) || confidence_cost_ratio < 0.0f ||
      (confidence_cost_ratio > 0.0f &&
       (!has_confidence || proposal_token_count != kMaxDraftTokens ||
        !dense_proposal || greedy))) {
    throw std::runtime_error("invalid speculative confidence budget");
  }
  if (!std::isfinite(sparse_mean_q_threshold) ||
      sparse_mean_q_threshold < 0.0f || sparse_mean_q_threshold > 1.0f ||
      (sparse_mean_q_threshold > 0.0f &&
       (dense_proposal || greedy ||
        proposal_token_count != kMaxDraftTokens))) {
    throw std::runtime_error("invalid sparse speculative q budget");
  }
  if (has_confidence &&
      (confidence.shape() != mx::Shape{1, proposal_token_count} ||
       confidence.dtype() != mx::float32)) {
    throw std::runtime_error("invalid speculative confidence block");
  }
  if (greedy) {
    if (has_confidence) {
      eval(draft_tokens, confidence);
    } else {
      eval(draft_tokens);
    }
  } else if (dense_proposal) {
    if (proposal_probs.shape() !=
            mx::Shape{1, proposal_token_count, cfg_.vocab_size} ||
        proposal_probs.dtype() != mx::float32) {
      throw std::runtime_error("invalid dense speculative proposal");
    }
    if (has_confidence) {
      eval(draft_tokens, proposal_probs, confidence);
    } else {
      eval(draft_tokens, proposal_probs);
    }
  } else {
    if (proposal_indices.shape() != proposal_probs.shape() ||
        proposal_indices.ndim() != 3 || proposal_indices.shape()[0] != 1 ||
        proposal_indices.shape()[1] != proposal_token_count ||
        proposal_indices.dtype() != mx::int32 ||
        proposal_probs.dtype() != mx::float32) {
      throw std::runtime_error("invalid sparse speculative proposal");
    }
    if (has_confidence) {
      eval(draft_tokens, proposal_indices, proposal_probs, confidence);
    } else {
      eval(draft_tokens, proposal_indices, proposal_probs);
    }
  }
  const auto draft_done = std::chrono::steady_clock::now();
  float sparse_mean_q = 0.0f;
  if ((trace || sparse_mean_q_threshold > 0.0f) && !dense_proposal &&
      !greedy) {
    const int support = static_cast<int>(proposal_probs.shape()[2]);
    const int mean_positions = std::min(proposal_token_count, 6);
    const int32_t* const drafted = draft_tokens.data<int32_t>();
    const int32_t* const indices = proposal_indices.data<int32_t>();
    const float* const probabilities = proposal_probs.data<float>();
    if (trace) {
      std::fprintf(stderr, "qwen38_%s selected_q=", trace_tag);
    }
    for (int position = 0; position < proposal_token_count; ++position) {
      float selected_q = 0.0f;
      for (int index = 0; index < support; ++index) {
        const int offset = position * support + index;
        if (indices[offset] == drafted[position]) {
          selected_q = probabilities[offset];
          break;
        }
      }
      if (position < mean_positions) {
        sparse_mean_q += selected_q;
      }
      if (trace) {
        std::fprintf(
            stderr, "%s%.6f", position == 0 ? "" : ",", selected_q);
      }
    }
    sparse_mean_q /= static_cast<float>(mean_positions);
    if (trace) {
      std::fprintf(stderr, " mean_q6=%.6f\n", sparse_mean_q);
    }
  }
  if (trace && has_confidence) {
    const float* const values = confidence.data<float>();
    std::fprintf(stderr, "qwen38_%s confidence=", trace_tag);
    for (int index = 0; index < proposal_token_count; ++index) {
      std::fprintf(stderr, "%s%.6f", index == 0 ? "" : ",", values[index]);
    }
    std::fprintf(stderr, "\n");
  }
  int draft_token_count = proposal_token_count;
  if (confidence_cost_ratio > 0.0f) {
    draft_token_count = dspark_select_verify_draft_tokens(
        confidence.data<float>(), proposal_token_count, confidence_cost_ratio);
    if (trace) {
      std::fprintf(
          stderr,
          "qwen38_%s confidence_budget drafts=%d cost_ratio=%.6f\n",
          trace_tag,
          draft_token_count,
          confidence_cost_ratio);
    }
  } else if (sparse_mean_q_threshold > 0.0f &&
             sparse_mean_q < sparse_mean_q_threshold) {
    draft_token_count = 1;
    if (trace) {
      std::fprintf(
          stderr,
          "qwen38_%s q_budget drafts=1 mean_q6=%.6f threshold=%.6f\n",
          trace_tag,
          sparse_mean_q,
          sparse_mean_q_threshold);
    }
  }
  array verified_draft_tokens = draft_tokens;
  array verified_proposal_indices = proposal_indices;
  array verified_proposal_probs = proposal_probs;
  if (draft_token_count < proposal_token_count) {
    verified_draft_tokens = slice(
        draft_tokens, {0, 0}, {1, draft_token_count});
    if (dense_proposal) {
      verified_proposal_probs = slice(
          proposal_probs,
          {0, 0, 0},
          {1, draft_token_count, cfg_.vocab_size});
    } else if (!greedy) {
      const int support = static_cast<int>(proposal_probs.shape()[2]);
      verified_proposal_indices = slice(
          proposal_indices, {0, 0, 0}, {1, draft_token_count, support});
      verified_proposal_probs = slice(
          proposal_probs, {0, 0, 0}, {1, draft_token_count, support});
    }
    eval(verified_draft_tokens);
  }
  const int32_t* const drafted = verified_draft_tokens.data<int32_t>();
  int32_t input[kMaxDraftTokens + 1];
  input[0] = token;
  for (int index = 0; index < draft_token_count; ++index) {
    input[index + 1] = drafted[index];
  }

  snapshot();
  const bool tape_commit = native_dflash_tape_commit_enabled();
  TargetForward verified = forward_hidden_captured(
      array(input, {1, draft_token_count + 1}, mx::int32), tape_commit);
  array committed_hidden = verified.hidden;
  last_hidden_ = last_token(verified.hidden);
  array target_logits = logits(verified.hidden);
  int accepted = 0;
  array next(0);
  auto verify_done = draft_done;
  if (greedy) {
    array target_tokens = astype(mx::argmax(target_logits, -1), mx::int32);
    array target_rows = slice(
        target_tokens, {0, 0}, {1, draft_token_count});
    array accepted_flags = astype(
        mx::equal(target_rows, verified_draft_tokens), mx::int32);
    array accepted_array = sum(mx::cumprod(accepted_flags, -1), -1);
    eval(target_tokens, accepted_array);
    verify_done = std::chrono::steady_clock::now();
    accepted = std::clamp(
        accepted_array.item<int32_t>(), 0, draft_token_count);
    next = reshape(
        slice(
            target_tokens,
            {0, accepted},
            {1, accepted + 1}),
        {1});
  } else {
    array target_probs = sampling_probabilities(target_logits);
    array proposal_column = expand_dims(verified_draft_tokens, -1);
    array target_rows = slice(
        target_probs,
        {0, 0, 0},
        {1, draft_token_count, cfg_.vocab_size});
    array p = squeeze(
        mx::take_along_axis(target_rows, proposal_column, -1), -1);
    array q = dense_proposal
        ? squeeze(
              mx::take_along_axis(
                  verified_proposal_probs, proposal_column, -1),
              -1)
        : sum(
              verified_proposal_probs *
                  mx::equal(verified_proposal_indices, proposal_column),
              -1);
    array accepted_flags = astype(
        mx::less(
            mx::random::uniform(q.shape(), mx::float32) * q,
            p),
        mx::int32);
    array accepted_array = sum(mx::cumprod(accepted_flags, -1), -1);
    if (trace) {
      eval(p, q, accepted_array);
      const float* const target_values = p.data<float>();
      const float* const proposal_values = q.data<float>();
      std::fprintf(stderr, "qwen38_%s target_p=", trace_tag);
      for (int position = 0; position < draft_token_count; ++position) {
        std::fprintf(
            stderr,
            "%s%.6f",
            position == 0 ? "" : ",",
            target_values[position]);
      }
      std::fprintf(stderr, " accept_probability=");
      for (int position = 0; position < draft_token_count; ++position) {
        const float proposal = proposal_values[position];
        const float probability = proposal > 0.0f
            ? std::min(1.0f, target_values[position] / proposal)
            : 1.0f;
        std::fprintf(
            stderr,
            "%s%.6f",
            position == 0 ? "" : ",",
            probability);
      }
      std::fprintf(stderr, "\n");
    } else {
      eval(accepted_array);
    }
    verify_done = std::chrono::steady_clock::now();
    accepted = std::clamp(
        accepted_array.item<int32_t>(), 0, draft_token_count);

    array next_probs(0);
    if (accepted == draft_token_count) {
      next_probs = reshape(
          slice(
              target_probs,
              {0, draft_token_count, 0},
              {1, draft_token_count + 1, cfg_.vocab_size}),
          {1, cfg_.vocab_size});
    } else {
      array target_row = reshape(
          slice(
              target_probs,
              {0, accepted, 0},
              {1, accepted + 1, cfg_.vocab_size}),
          {1, cfg_.vocab_size});
      array residual(0);
      if (dense_proposal) {
        array proposal_row = reshape(
            slice(
                verified_proposal_probs,
                {0, accepted, 0},
                {1, accepted + 1, cfg_.vocab_size}),
            {1, cfg_.vocab_size});
        residual = target_row - proposal_row;
      } else {
        const int support =
            static_cast<int>(verified_proposal_probs.shape()[2]);
        array indices = reshape(
            slice(
                verified_proposal_indices,
                {0, accepted, 0},
                {1, accepted + 1, support}),
            {1, support});
        array proposal_values = reshape(
            slice(
                verified_proposal_probs,
                {0, accepted, 0},
                {1, accepted + 1, support}),
            {1, support});
        array residual_values =
            mx::take_along_axis(target_row, indices, -1) - proposal_values;
        residual = mx::put_along_axis(
            target_row, indices, residual_values, -1);
      }
      residual = mx::maximum(residual, array(0.0f, mx::float32));
      array total = sum(residual, -1, true);
      next_probs = mx::where(
          mx::greater(total, array(0.0f, mx::float32)),
          residual / mx::maximum(total, array(1e-30f, mx::float32)),
          target_row);
    }
    next = astype(
        mx::random::categorical(mx::log(next_probs), -1), mx::int32);
  }
  eval(next);
  const auto sample_done = std::chrono::steady_clock::now();

  if (accepted < draft_token_count) {
    if (tape_commit) {
      draft_commit_verified_prefix(verified, accepted + 1);
    } else {
      restore();
      TargetForward committed = forward_hidden_captured(
          array(input, {1, accepted + 1}, mx::int32));
      committed_hidden = committed.hidden;
      last_hidden_ = last_token(committed.hidden);
      draft_append_context(committed.captured, accepted + 1);
    }
  } else {
    draft_append_context(verified.captured, draft_token_count + 1);
  }
  if (mtp_cycle_pending_) {
    mtp_commit_cycle(input, accepted + 1, committed_hidden);
  }

  for (int index = 0; index < accepted; ++index) {
    spec_buf_[index] = drafted[index];
  }
  spec_buf_[accepted] = next.item<int32_t>();
  spec_buf_n_ = accepted + 1;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;

  if (reasoning_open_) {
    for (int index = 0; index < spec_buf_n_; ++index) {
      if (spec_buf_[index] == 248069) {
        reasoning_open_ = false;
        break;
      }
      ++selected_reasoning_tokens_;
    }
  }
  if (trace) {
    mx::synchronize();
    const auto finished = std::chrono::steady_clock::now();
    const auto elapsed_ms = [](auto begin, auto end) {
      return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    std::fprintf(
        stderr,
        "qwen38_%s accepted=%d width=%d drafts=%d draft_ms=%.3f "
        "verify_ms=%.3f "
        "sample_ms=%.3f commit_ms=%.3f total_ms=%.3f\n",
        trace_tag,
        accepted,
        spec_buf_n_,
        draft_token_count,
        elapsed_ms(started, draft_done),
        elapsed_ms(draft_done, verify_done),
        elapsed_ms(verify_done, sample_done),
        elapsed_ms(sample_done, finished),
        elapsed_ms(started, finished));
  }
  return draft_token_count;
}

int32_t Engine::target_only_spec_refill(int32_t token) {
  if (mtp_valid_ && mtp_committed_history_enabled_) {
    mtp_begin_committed_cycle();
  }
  int32_t input[1] = {token};
  TargetForward target = forward_hidden_captured(
      array(input, {1, 1}, mx::int32));
  last_hidden_ = last_token(target.hidden);
  draft_append_context(target.captured, 1);
  if (mtp_cycle_pending_) {
    mtp_commit_cycle(input, 1, target.hidden);
  }
  array next = select_token(target.hidden);
  eval(next);
  spec_buf_[0] = next.item<int32_t>();
  spec_buf_n_ = 1;
  return spec_buf_[0];
}

float Engine::dspark_anchor_score(int32_t token) {
  array target_hidden = mx::fast::rms_norm(
      expand_dims(last_hidden_, 1), final_norm_, cfg_.rms_norm_eps);
  array anchor(&token, {1, 1}, mx::int32);
  array score = dspark_confidence(
      target_hidden,
      take(dspark_markov_w1_, anchor, 0),
      dspark_confidence_weight_,
      dspark_confidence_bias_);
  eval(score);
  return score.item<float>();
}

void Engine::dflash_spec_refill(int32_t token) {
  constexpr int kDraftTokens = 7;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  if (reasoning_open_ && max_reasoning_tokens_ > 0 &&
      selected_reasoning_tokens_ + kDraftTokens + 1 >=
          max_reasoning_tokens_) {
    target_only_spec_refill(token);
    return;
  }

  array draft_hidden = dflash_forward(token);
  array draft_logits = lm_head_(draft_hidden);
  auto [draft_tokens, draft_indices, draft_probs] =
      dflash_select(draft_hidden, draft_logits, token);
  verify_speculative_block(
      token,
      draft_tokens,
      draft_indices,
      draft_probs,
      array(0),
      /*dense_proposal=*/false,
      /*greedy=*/false,
      /*confidence_cost_ratio=*/0.0f,
      dflash_mean_q_threshold_,
      "dflash");
}

void Engine::dspark_spec_refill(int32_t token) {
  constexpr int kMaxDraftTokens = 7;
  const int draft_token_count = dspark_verify_draft_tokens_;
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  if (reasoning_open_ && max_reasoning_tokens_ > 0 &&
      selected_reasoning_tokens_ + draft_token_count + 1 >=
          max_reasoning_tokens_) {
    target_only_spec_refill(token);
    return;
  }
  if (dspark_bypass_remaining_ > 0) {
    --dspark_bypass_remaining_;
    if (native_spec_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_dspark bypass remaining=%d\n",
          dspark_bypass_remaining_);
    }
    if (target_only_spec_refill(token) == 248069) {
      reasoning_open_ = false;
    }
    return;
  }

  const bool sampled = sampling_enabled_;
  const bool confidence_budget = dspark_confidence_cost_ratio_ > 0.0f;
  const bool trace = native_spec_trace_enabled();
  if (trace) {
    const float anchor_score = dspark_anchor_score(token);
    std::fprintf(
        stderr,
        "qwen38_dspark anchor_score=%.6f\n",
        anchor_score);
  }
  array draft_hidden = dspark_forward(token);
  auto [draft_tokens, draft_probs] =
      dspark_propose(draft_hidden, token, sampled);
  // Keep the seven-position proposal fixed while profiling target verifier
  // prefixes, so acceptance and target geometry are the only changed inputs.
  if (!confidence_budget && draft_token_count < kMaxDraftTokens) {
    draft_tokens = slice(
        draft_tokens, {0, 0}, {1, draft_token_count});
    if (sampled) {
      draft_probs = slice(
          draft_probs,
          {0, 0, 0},
          {1, draft_token_count, cfg_.vocab_size});
    }
    draft_hidden = slice(
        draft_hidden,
        {0, 0, 0},
        {1, draft_token_count, cfg_.hidden_size});
  }
  array confidence(0);
  if (native_spec_trace_enabled() || confidence_budget) {
    const int confidence_token_count =
        static_cast<int>(draft_tokens.shape()[1]);
    array anchor(&token, {1, 1}, mx::int32);
    array previous = anchor;
    if (confidence_token_count > 1) {
      previous = concatenate(
          {
              anchor,
              slice(
                  draft_tokens,
                  {0, 0},
                  {1, confidence_token_count - 1}),
          },
          1);
    }
    array markov_embeddings = take(dspark_markov_w1_, previous, 0);
    confidence = dspark_confidence(
        draft_hidden,
        markov_embeddings,
        dspark_confidence_weight_,
        dspark_confidence_bias_);
  }
  const int selected_draft_tokens = verify_speculative_block(
      token,
      draft_tokens,
      array(0),
      draft_probs,
      confidence,
      /*dense_proposal=*/sampled,
      /*greedy=*/!sampled,
      dspark_confidence_cost_ratio_,
      /*sparse_mean_q_threshold=*/0.0f,
      "dspark");
  if (confidence_budget && dspark_bypass_refills_ > 0 &&
      selected_draft_tokens == 1) {
    dspark_bypass_remaining_ = dspark_bypass_refills_;
  }
}

void Engine::spec_refill(int32_t token) {
  if (spec_buf_pos_ < spec_buf_n_ && token == last_emitted_) {
    return;
  }
  if (dspark_valid_) {
    dspark_spec_refill(token);
    return;
  }
  if (dflash_valid_) {
    dflash_spec_refill(token);
    return;
  }
  spec_buf_n_ = 0;
  spec_buf_pos_ = 0;
  decode_scheduled_ = false;
  const int n_draft = mtp_block_ - 1;
  if (sampling_enabled_) {
    if (reasoning_open_ && max_reasoning_tokens_ > 0 &&
        selected_reasoning_tokens_ + n_draft + 1 >=
            max_reasoning_tokens_) {
      target_only_spec_refill(token);
      return;
    }

    if (mtp_committed_history_enabled_) {
      mtp_begin_committed_cycle();
    } else {
      mtp_reset();
    }
    array hidden = mtp_seed_hidden();
    if (hidden.ndim() == 1) {
      hidden = reshape(hidden, {1, 1, hidden.shape()[0]});
    } else if (hidden.ndim() == 2) {
      hidden = expand_dims(hidden, 1);
    }
    int32_t proposed = token;
    std::vector<array> path;
    std::vector<array> probabilities;
    path.reserve(static_cast<size_t>(n_draft));
    probabilities.reserve(static_cast<size_t>(n_draft));
    for (int index = 0; index < n_draft; ++index) {
      array ids(&proposed, {1, 1}, mx::int32);
      hidden = mtp_forward(embed(ids), hidden);
      array proposal = sampling_probabilities(lm_head_(hidden));
      array next = astype(
          mx::random::categorical(mx::log(proposal), -1), mx::int32);
      eval(next);
      proposed = next.item<int32_t>();
      path.push_back(reshape(next, {1}));
      probabilities.push_back(
          reshape(proposal, {1, cfg_.vocab_size}));
    }
    verify_speculative_block(
        token,
        astype(mx::stack(path, 1), mx::int32),
        array(0),
        mx::stack(probabilities, 1),
        array(0),
        /*dense_proposal=*/true,
        /*greedy=*/false,
        /*confidence_cost_ratio=*/0.0f,
        /*sparse_mean_q_threshold=*/0.0f,
        "mtp");
    return;
  }
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
    auto loaded = load_activation_safetensors(mtp_dir + "/" + name);
    for (auto& kv : loaded.first) {
      weights.emplace(kv.first, std::move(kv.second));
    }
  }
  closedir(dir);
  if (weights.empty()) {
    throw std::runtime_error("no MTP safetensors in " + mtp_dir);
  }
  if (weights.contains("markov_head.markov_w1.weight")) {
    load_dspark(weights);
    return;
  }
  if (weights.contains("candidate_selector.predecessor_codebook")) {
    load_dflash2(weights);
    return;
  }
  std::string mtp_prefix;
  if (!weights.contains("fc.weight")) {
    if (!weights.contains("mtp.fc.weight")) {
      throw std::runtime_error("MTP checkpoint is missing fc.weight");
    }
    mtp_prefix = "mtp.";
  }
  mtp_fc_ = load_qlinear(weights, mtp_prefix + "fc");
  mtp_pre_emb_ =
      require(weights, mtp_prefix + "pre_fc_norm_embedding.weight");
  mtp_pre_hid_ = require(weights, mtp_prefix + "pre_fc_norm_hidden.weight");
  mtp_norm_ = require(weights, mtp_prefix + "norm.weight");
  mtp_layer_.is_linear = false;
  mtp_layer_.input_norm =
      require(weights, mtp_prefix + "layers.0.input_layernorm.weight");
  mtp_layer_.post_norm = require(
      weights, mtp_prefix + "layers.0.post_attention_layernorm.weight");
  mtp_layer_.gate_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.mlp.gate_proj");
  mtp_layer_.up_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.mlp.up_proj");
  mtp_layer_.down_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.mlp.down_proj");
  mtp_layer_.attn.q_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.self_attn.q_proj");
  mtp_layer_.attn.k_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.self_attn.k_proj");
  mtp_layer_.attn.v_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.self_attn.v_proj");
  mtp_layer_.attn.o_proj =
      load_qlinear(weights, mtp_prefix + "layers.0.self_attn.o_proj");
  mtp_layer_.attn.q_norm =
      require(weights, mtp_prefix + "layers.0.self_attn.q_norm.weight");
  mtp_layer_.attn.k_norm =
      require(weights, mtp_prefix + "layers.0.self_attn.k_norm.weight");
  mtp_block_ = native_mtp_block_size();
  mtp_committed_history_enabled_ = native_mtp_committed_history_enabled();
  if (mtp_committed_history_enabled_ && !sampling_enabled_) {
    throw std::runtime_error(
        "committed MTP history requires native sampling");
  }
  dflash_valid_ = false;
  dspark_valid_ = false;
  mtp_valid_ = true;
  mtp_reset();
}

int32_t Engine::emit_scheduled() {
  last_emitted_ = pending_tok_.item<int32_t>();
  if (last_emitted_ == 248069) {
    reasoning_open_ = false;
  }
  return last_emitted_;
}

bool Engine::can_restore_mtp_prompt() const {
  if (!mtp_prompt_cache_enabled_ || !mtp_valid_ ||
      !mtp_committed_history_enabled_ || dflash_valid_ || dspark_valid_ ||
      mtp_cycle_pending_ || mtp_prompt_snapshot_.size() != layers_.size() ||
      mtp_prompt_hidden_.ndim() != 2 ||
      mtp_prompt_cache_length_ !=
          static_cast<int>(prompt_snapshot_history_.size()) - 1 ||
      mtp_layer_.attn.cache_length < mtp_prompt_cache_length_ ||
      mtp_layer_.attn.offset < mtp_prompt_cache_length_) {
    return false;
  }
  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto& layer = layers_[i];
    const auto& saved = mtp_prompt_snapshot_[i];
    // A changed cache representation needs a fresh prefill. Recurrent
    // snapshots alone cannot recover the original attention prefix bits.
    if (!layer.is_linear &&
        (layer.attn.cache_bits != saved.cache_bits ||
         layer.attn.cache_length < saved.cache_length)) {
      return false;
    }
  }
  return true;
}

int32_t Engine::prefill(const int32_t* tokens, int n, bool schedule_decode) {
  last_prefill_cached_tokens_ = 0;
  if (n <= 0) {
    throw std::runtime_error("prefill requires at least one token");
  }
  if (has_mtp() && schedule_decode) {
    throw std::runtime_error(
        "speculative prefill requires schedule_decode=false");
  }

  array reused_prompt_hidden(0);
  const int32_t* new_tokens = tokens;
  int new_token_count = n;
  const bool starts_request =
      request_boundary_pending_ || token_history_.empty();
  if (request_boundary_pending_) {
    const std::size_t history_compare_count =
        std::min(token_history_.size(), static_cast<std::size_t>(n));
    const auto mismatch = std::mismatch(
        token_history_.begin(),
        token_history_.begin() + history_compare_count,
        tokens,
        tokens + history_compare_count);
    const std::size_t common_prefix =
        static_cast<std::size_t>(mismatch.first - token_history_.begin());
    const bool can_reuse_current =
        !has_mtp() && token_history_.size() < size_t(n) &&
        std::equal(token_history_.begin(), token_history_.end(), tokens);
    const std::size_t snapshot_compare_count =
        std::min(prompt_snapshot_history_.size(), static_cast<std::size_t>(n));
    const auto snapshot_mismatch = std::mismatch(
        prompt_snapshot_history_.begin(),
        prompt_snapshot_history_.begin() + snapshot_compare_count,
        tokens,
        tokens + snapshot_compare_count);
    const std::size_t snapshot_common = static_cast<std::size_t>(
        snapshot_mismatch.first - prompt_snapshot_history_.begin());
    const bool reuse_mtp_prompt = has_mtp() && can_restore_mtp_prompt();
    const bool can_reuse_snapshot =
        !can_reuse_current && (!has_mtp() || reuse_mtp_prompt) &&
        prompt_snapshot_valid_ &&
        (prompt_snapshot_history_.size() < size_t(n) ||
         (reuse_mtp_prompt && prompt_snapshot_history_.size() == size_t(n))) &&
        std::equal(
            prompt_snapshot_history_.begin(), prompt_snapshot_history_.end(),
            tokens);
    if (native_state_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_native prefill history=%zu input=%d common=%zu "
          "snapshot=%zu snapshot_common=%zu reuse_current=%d "
          "reuse_snapshot=%d mtp=%d\n",
          token_history_.size(), n, common_prefix,
          prompt_snapshot_history_.size(), snapshot_common,
          can_reuse_current ? 1 : 0, can_reuse_snapshot ? 1 : 0,
          has_mtp() ? 1 : 0);
    }
    if (can_reuse_current) {
      last_prefill_cached_tokens_ = static_cast<int>(token_history_.size());
      new_tokens += token_history_.size();
      new_token_count -= static_cast<int>(token_history_.size());
      reset_decode_pipeline();
      request_boundary_pending_ = false;
    } else if (can_reuse_snapshot) {
      if (reuse_mtp_prompt) {
        restore_snapshot(mtp_prompt_snapshot_);
        // Committed MTP keys/values are append-only too. Keep their current
        // allocation while returning to the saved one-token-behind prefix.
        mtp_layer_.attn.cache_length = mtp_prompt_cache_length_;
        mtp_layer_.attn.offset = mtp_prompt_cache_length_;
      } else {
        restore();
      }
      token_history_ = prompt_snapshot_history_;
      last_prefill_cached_tokens_ = static_cast<int>(token_history_.size());
      new_tokens += token_history_.size();
      new_token_count -= static_cast<int>(token_history_.size());
      reset_decode_pipeline();
      if (reuse_mtp_prompt) {
        last_hidden_ = mtp_prompt_hidden_;
        if (new_token_count == 0) {
          // The saved final hidden row is sufficient to recompute logits.
          // Keep the target and MTP caches at their restored boundaries.
          reused_prompt_hidden = expand_dims(mtp_prompt_hidden_, 1);
        }
        // The restored layers now own these states. Do not retain another
        // recurrent-state generation while extending a long prompt.
        mtp_prompt_snapshot_.clear();
        mtp_prompt_hidden_ = array(0);
        prompt_snapshot_valid_ = false;
        snap_.clear();
        mx::random::seed(sampling_seed_);
      }
      request_boundary_pending_ = false;
    } else {
      reset();
    }
  }

  if (starts_request) {
    reasoning_open_ = false;
    for (int index = 0; index < n; ++index) {
      if (tokens[index] == 248068) {
        reasoning_open_ = true;
      } else if (tokens[index] == 248069) {
        reasoning_open_ = false;
      }
    }
    selected_reasoning_tokens_ = 0;
    reasoning_cap_selected_ = false;
    if (native_state_trace_enabled()) {
      std::fprintf(
          stderr,
          "qwen38_native request_state input=%d reasoning_open=%d limit=%d\n",
          n, reasoning_open_ ? 1 : 0, max_reasoning_tokens_);
    }
  }

  token_history_.insert(
      token_history_.end(), new_tokens, new_tokens + new_token_count);
  array hidden = std::move(reused_prompt_hidden);
  const bool capture_draft_context = dflash_valid_ || dspark_valid_;
  const bool capture_mtp_history =
      mtp_valid_ && mtp_committed_history_enabled_;
  array previous_mtp_hidden = last_hidden_;
  bool has_previous_mtp_hidden =
      capture_mtp_history && target_sequence_length() > 0;
  constexpr int kDraftPrefillChunkSize = 2048;
  constexpr int kMtpCommittedPrefillChunkSize = 1024;
  const int internal_chunk_size = capture_draft_context
      ? kDraftPrefillChunkSize
      : (capture_mtp_history
             ? kMtpCommittedPrefillChunkSize
             : (!has_mtp() ? target_only_prefill_chunk_size_ : 0));
  if (internal_chunk_size > 0) {
    for (int offset = 0; offset < new_token_count;) {
      int chunk_size = std::min(
          internal_chunk_size, new_token_count - offset);
      const int serialized_chunk_size = serialized_attention_cache_chunk_size(
          chunk_size, capture_mtp_history);
      if (serialized_chunk_size != chunk_size) {
        chunk_size = serialized_chunk_size;
        if (native_state_trace_enabled()) {
          std::fprintf(
              stderr,
              "qwen38_native serialize_cache_growth offset=%d remaining=%d\n",
              offset,
              new_token_count - offset);
        }
      }
      const int post_growth_chunk_size =
          bounded_post_growth_mtp_prefill_chunk_size(
              chunk_size, capture_mtp_history);
      if (post_growth_chunk_size != chunk_size) {
        if (native_state_trace_enabled()) {
          std::fprintf(
              stderr,
              "qwen38_native cap_post_growth_prefill offset=%d remaining=%d "
              "requested=%d selected=%d target_length=%d mtp_length=%d "
              "mtp_capacity=%d\n",
              offset,
              new_token_count - offset,
              chunk_size,
              post_growth_chunk_size,
              target_sequence_length(),
              mtp_layer_.attn.cache_length,
              mtp_layer_.attn.cache_capacity);
        }
        chunk_size = post_growth_chunk_size;
      }
      const int mtp_append_tokens = capture_mtp_history
          ? (has_previous_mtp_hidden ? chunk_size : chunk_size - 1)
          : 0;
      const bool evict_raw_params_for_mtp_growth =
          evict_q4_raw_params_at_mtp_growth_ && mtp_append_tokens > 0 &&
          attention_cache_append_requires_large_growth(
              mtp_layer_.attn.cache_length,
              mtp_layer_.attn.cache_capacity,
              mtp_append_tokens);
      array chunk_ids(
          new_tokens + offset, {1, chunk_size}, mx::int32);
      std::size_t released_raw_parameter_bytes = 0;
      if (capture_draft_context) {
        TargetForward target = forward_hidden_captured(chunk_ids);
        hidden = target.hidden;
        draft_append_context(target.captured, chunk_size);
      } else {
        hidden = forward_hidden(chunk_ids);
        if (capture_mtp_history) {
          if (evict_raw_params_for_mtp_growth) {
            // Materialize the target result while its selected decode
            // parameters are still resident. MTP is the only later consumer,
            // so those parameters can then leave the working set while the
            // large MTP cache replacement is active.
            eval(hidden, previous_mtp_hidden);
            mx::synchronize();
            hidden.detach();
            previous_mtp_hidden.detach();
            const std::size_t active_before = mx::get_active_memory();
            released_raw_parameter_bytes =
                release_target_fused_q4_raw_decode_parameters();
            mx::clear_cache();
            if (native_state_trace_enabled()) {
              std::fprintf(
                  stderr,
                  "qwen38_native evict_raw_q4_params mtp_length=%d "
                  "mtp_capacity=%d append=%d bytes=%zu active_before=%zu "
                  "active_after=%zu\n",
                  mtp_layer_.attn.cache_length,
                  mtp_layer_.attn.cache_capacity,
                  mtp_append_tokens,
                  released_raw_parameter_bytes,
                  active_before,
                  mx::get_active_memory());
            }
          }
          mtp_append_prompt_history(
              hidden,
              new_tokens + offset,
              chunk_size,
              previous_mtp_hidden,
              has_previous_mtp_hidden);
          previous_mtp_hidden = last_token(hidden);
          has_previous_mtp_hidden = true;
        }
      }
      mx::synchronize();
      if (released_raw_parameter_bytes > 0) {
        // MTP replacement and its source graph are complete, so its retired
        // cache can be reclaimed before rebuilding the exact decode stream.
        mx::clear_cache();
        const std::size_t active_before = mx::get_active_memory();
        const std::size_t restored_raw_parameter_bytes =
            restore_target_fused_q4_raw_decode_parameters();
        if (restored_raw_parameter_bytes != released_raw_parameter_bytes) {
          throw std::runtime_error(
              "restored target raw fused Q4 parameter size mismatch");
        }
        // Every rebuilt buffer must be complete before target prefill resumes.
        // Otherwise the next chunk can overlap the final host-to-device copies
        // and retain both their staging storage and its activation workspace.
        mx::synchronize();
        mx::clear_cache();
        if (native_state_trace_enabled()) {
          std::fprintf(
              stderr,
              "qwen38_native restore_raw_q4_params bytes=%zu "
              "active_before=%zu active_after=%zu\n",
              restored_raw_parameter_bytes,
              active_before,
              mx::get_active_memory());
        }
      }
      offset += chunk_size;
    }
  } else {
    array ids(new_tokens, {1, new_token_count}, mx::int32);
    hidden = forward_hidden(ids);
    if (capture_mtp_history) {
      mtp_append_prompt_history(
          hidden,
          new_tokens,
          new_token_count,
          previous_mtp_hidden,
          has_previous_mtp_hidden);
    }
  }
  if (!has_mtp()) {
    snapshot();
    prompt_snapshot_history_ = token_history_;
    prompt_snapshot_valid_ = true;
  } else if (mtp_prompt_cache_enabled_ && capture_mtp_history &&
             !capture_draft_context) {
    // Speculative verification overwrites snap_; request-prefix state needs
    // its own recurrent snapshots, but never a duplicate target/MTP KV pool.
    capture_snapshot(mtp_prompt_snapshot_);
    mtp_prompt_hidden_ = mx::copy(last_token(hidden));
    eval(mtp_prompt_hidden_);
    mtp_prompt_hidden_.detach();
    mtp_prompt_cache_length_ = mtp_layer_.attn.cache_length;
    prompt_snapshot_history_ = token_history_;
    prompt_snapshot_valid_ = true;
  }
  array first = select_token(hidden);
  async_eval(first);
  if (!schedule_decode) {
    pending_tok_ = first;
    decode_scheduled_ = false;
    last_emitted_in_state_ = false;
    return emit_scheduled();
  }
  array following = select_token(forward_hidden(reshape(first, {1, 1})));
  async_eval(following);
  pending_tok_ = first;
  int32_t out = emit_scheduled();
  record_processed_token(out);
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

int32_t Engine::decode(int32_t token) {
  if (has_mtp()) {
    if (spec_buf_pos_ >= spec_buf_n_ || token != last_emitted_) {
      spec_refill(token);
    }
    last_emitted_ = spec_buf_[spec_buf_pos_++];
    decode_scheduled_ = false;
    return last_emitted_;
  }
  if (decode_scheduled_ && token == last_emitted_) {
    array following =
        select_token(forward_hidden(reshape(pending_tok_, {1, 1})));
    async_eval(following);
    int32_t out = emit_scheduled();
    pending_tok_ = following;
    record_processed_token(out);
    return out;
  }
  if (!last_emitted_in_state_ || token != last_emitted_) {
    token_history_.push_back(token);
  }
  array ids = (token == last_emitted_) ? reshape(pending_tok_, {1, 1})
                                       : array(&token, {1, 1}, mx::int32);
  array next = select_token(forward_hidden(ids));
  async_eval(next);
  array following = select_token(forward_hidden(reshape(next, {1, 1})));
  async_eval(following);
  pending_tok_ = next;
  int32_t out = emit_scheduled();
  record_processed_token(out);
  pending_tok_ = following;
  decode_scheduled_ = true;
  return out;
}

} // namespace mlx_qwen38
} // namespace sglang
