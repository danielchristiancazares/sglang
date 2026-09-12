#include "sglang/native/qwen35_checkpoint.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using sglang::native::kQwen35TargetTensorBytes;
using sglang::native::kQwen35TargetTensorCount;
using sglang::native::open_qwen35_target_checkpoint;
using sglang::native::qwen35_checkpoint_error_code_name;
using sglang::native::Qwen35CheckpointOpenResult;
using sglang::native::Qwen35CheckpointTensor;
using sglang::native::SafetensorsDType;
using sglang::native::SafetensorsTensorInfo;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {   \
      return false;                                                            \
    }                                                                          \
  } while (false)

[[nodiscard]] bool
RealCheckpointIsFullyCrossValidated(const std::filesystem::path &root) {
  if (!std::filesystem::exists(root / L"config.json")) {
    std::printf("[  SKIPPED ] real target checkpoint not present: %ls\n",
                root.c_str());
    return true;
  }
  Qwen35CheckpointOpenResult result = open_qwen35_target_checkpoint(root);
  if (!result.ok()) {
    const std::string_view code =
        qwen35_checkpoint_error_code_name(result.error.code);
    std::printf("checkpoint error=%.*s system=%lu detail=%llu actual=%llu "
                "required=%llu path=%s member=%s\n",
                static_cast<int>(code.size()), code.data(),
                static_cast<unsigned long>(result.error.system_error),
                static_cast<unsigned long long>(result.error.detail),
                static_cast<unsigned long long>(result.error.actual),
                static_cast<unsigned long long>(result.error.required),
                result.error.path.c_str(), result.error.member.c_str());
  }
  CHECK(result.ok());
  CHECK(result.checkpoint.shards().size() == 10U);
  CHECK(result.checkpoint.tensors().size() == kQwen35TargetTensorCount);
  CHECK(result.checkpoint.total_tensor_bytes() == kQwen35TargetTensorBytes);
  CHECK(result.checkpoint.total_parameters() == 18164649200ULL);

  const auto &config = result.checkpoint.config();
  CHECK(config.vocabulary_size == 248320U);
  CHECK(config.hidden_size == 5120U);
  CHECK(config.intermediate_size == 17408U);
  CHECK(config.layers == 64U);
  CHECK(config.full_attention_layers == 16U);
  CHECK(config.gdn_layers == 48U);
  CHECK(config.full_attention_layer_ids ==
        (std::array<uint32_t, 16>{3U, 7U, 11U, 15U, 19U, 23U, 27U, 31U, 35U,
                                  39U, 43U, 47U, 51U, 55U, 59U, 63U}));
  CHECK(config.mtp_layers == 1U);
  CHECK(config.max_position_embeddings == 262144ULL);

  const auto &quantization = result.checkpoint.quantization();
  CHECK(quantization.quantized_layers == 401U);
  CHECK(quantization.nvfp4_group_size == 16U);
  CHECK(quantization.fp8_kv_cache);
  CHECK(quantization.nvfp4_projection_names.size() == 401U);

  const Qwen35CheckpointTensor *lm_head =
      result.checkpoint.find_tensor("lm_head.weight");
  CHECK(lm_head != nullptr);
  if (lm_head == nullptr) {
    return false;
  }
  CHECK(lm_head->shard_index == 0U);
  const SafetensorsTensorInfo *lm_head_info =
      result.checkpoint.tensor_info(*lm_head);
  CHECK(lm_head_info != nullptr);
  if (lm_head_info == nullptr) {
    return false;
  }
  CHECK(lm_head_info->dtype == SafetensorsDType::kUInt8);
  CHECK(lm_head_info->shape == (std::vector<uint64_t>{248320U, 2560U}));
  const auto bytes = result.checkpoint.tensor_bytes(*lm_head);
  CHECK(bytes.has_value());
  if (!bytes.has_value()) {
    return false;
  }
  CHECK(bytes->size() == 635699200ULL);

  const Qwen35CheckpointTensor *last =
      result.checkpoint.find_tensor("mtp.layers.0.self_attn.v_proj.weight");
  CHECK(last != nullptr);
  CHECK(result.checkpoint.find_tensor("not.a.tensor") == nullptr);
  return true;
}

[[nodiscard]] bool MissingRootFailsClosed() {
  const Qwen35CheckpointOpenResult result = open_qwen35_target_checkpoint(
      LR"(C:\this-path-must-not-exist\qwen35-checkpoint)");
  CHECK(!result.ok());
  CHECK(result.error.code ==
        sglang::native::Qwen35CheckpointErrorCode::kFileOpenFailed);
  return true;
}

} // namespace

int wmain(int argc, wchar_t **argv) {
  const std::filesystem::path root =
      argc > 1
          ? std::filesystem::path(argv[1])
          : std::filesystem::path(
                LR"(C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4)");
  const std::array tests{
      std::pair{"missing root fails closed", &MissingRootFailsClosed},
  };
  size_t passed = 0;
  for (const auto &[name, test] : tests) {
    if (!test()) {
      std::printf("[  FAILED  ] %s\n", name);
      return 1;
    }
    ++passed;
    std::printf("[       OK ] %s\n", name);
  }
  if (!RealCheckpointIsFullyCrossValidated(root)) {
    std::printf("[  FAILED  ] real checkpoint is fully cross-validated\n");
    return 1;
  }
  ++passed;
  std::printf("[       OK ] real checkpoint is fully cross-validated\n");
  std::printf("[  PASSED  ] %zu tests\n", passed);
  return 0;
}
