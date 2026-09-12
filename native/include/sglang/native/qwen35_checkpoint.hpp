#ifndef SGLANG_NATIVE_QWEN35_CHECKPOINT_HPP_
#define SGLANG_NATIVE_QWEN35_CHECKPOINT_HPP_

#include "sglang/native/safetensors_file.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::native {

struct Qwen35CheckpointOpenResult;

inline constexpr uint32_t kQwen35CheckpointManifestVersion = 1;
inline constexpr size_t kQwen35TargetShardCount = 10;
inline constexpr size_t kQwen35TargetTensorCount = 2402;
inline constexpr uint64_t kQwen35TargetTensorBytes = 18765214312ULL;
inline constexpr uint64_t kQwen35TargetTotalParameters = 18164649200ULL;
inline constexpr uint32_t kQwen35TargetQuantizedLayers = 401;
inline constexpr uint32_t kQwen35TargetSelectedProjectionBases = 208;
inline constexpr uint32_t kQwen35TargetSelectedTensorCount = 832;
inline constexpr uint32_t kQwen35TargetFullAttentionLayers = 16;

enum class Qwen35CheckpointErrorCode : uint32_t {
  kOk = 0,
  kInvalidRoot,
  kFileOpenFailed,
  kFileSizeFailed,
  kFileReadFailed,
  kFileTooLarge,
  kJsonInvalid,
  kSchemaInvalid,
  kValueMismatch,
  kHashMismatch,
  kIndexInvalid,
  kQuantizationInvalid,
  kManifestInvalid,
  kShardInvalid,
  kTensorMissing,
  kTensorUnexpected,
  kTensorShardMismatch,
  kTensorCountMismatch,
  kTensorByteCountMismatch,
  kAllocationFailed,
  kInternalFailure,
};

struct Qwen35CheckpointError final {
  Qwen35CheckpointErrorCode code{Qwen35CheckpointErrorCode::kOk};
  uint32_t system_error{0};
  uint64_t detail{0};
  uint64_t actual{0};
  uint64_t required{0};
  std::string path;
  std::string member;
};

struct Qwen35TargetConfig final {
  uint32_t vocabulary_size{0};
  uint32_t hidden_size{0};
  uint32_t intermediate_size{0};
  uint32_t layers{0};
  uint32_t full_attention_layers{0};
  uint32_t gdn_layers{0};
  uint32_t attention_heads{0};
  uint32_t attention_kv_heads{0};
  uint32_t attention_head_dimension{0};
  uint32_t gdn_key_heads{0};
  uint32_t gdn_value_heads{0};
  uint32_t gdn_key_dimension{0};
  uint32_t gdn_value_dimension{0};
  uint32_t convolution_kernel_size{0};
  uint32_t mtp_layers{0};
  uint64_t max_position_embeddings{0};
  std::array<uint32_t, kQwen35TargetFullAttentionLayers>
      full_attention_layer_ids{};
};

struct Qwen35QuantizationConfig final {
  uint32_t quantized_layers{0};
  uint32_t nvfp4_group_size{0};
  bool fp8_kv_cache{false};
  std::vector<std::string> nvfp4_projection_names;
};

struct Qwen35CheckpointShard final {
  std::string name;
  uint64_t file_bytes{0};
  uint64_t tensor_bytes{0};
  uint32_t tensor_count{0};
  std::string sha256;
  SafetensorsFile file;
};

struct Qwen35CheckpointTensor final {
  std::string name;
  uint32_t shard_index{0};
  uint32_t tensor_index{0};
};

class Qwen35CheckpointBuilder;

class Qwen35Checkpoint final {
public:
  Qwen35Checkpoint() noexcept = default;
  ~Qwen35Checkpoint() noexcept = default;

  Qwen35Checkpoint(const Qwen35Checkpoint &) = delete;
  Qwen35Checkpoint &operator=(const Qwen35Checkpoint &) = delete;
  Qwen35Checkpoint(Qwen35Checkpoint &&) noexcept = default;
  Qwen35Checkpoint &operator=(Qwen35Checkpoint &&) noexcept = default;

  [[nodiscard]] uint32_t manifest_version() const noexcept {
    return manifest_version_;
  }
  [[nodiscard]] const std::filesystem::path &root() const noexcept {
    return root_;
  }
  [[nodiscard]] const Qwen35TargetConfig &config() const noexcept {
    return config_;
  }
  [[nodiscard]] const Qwen35QuantizationConfig &quantization() const noexcept {
    return quantization_;
  }
  [[nodiscard]] std::span<const Qwen35CheckpointShard> shards() const noexcept {
    return shards_;
  }
  [[nodiscard]] std::span<const Qwen35CheckpointTensor>
  tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] uint64_t total_parameters() const noexcept {
    return total_parameters_;
  }
  [[nodiscard]] uint64_t total_tensor_bytes() const noexcept {
    return total_tensor_bytes_;
  }
  [[nodiscard]] const Qwen35CheckpointTensor *
  find_tensor(std::string_view name) const noexcept;
  [[nodiscard]] const SafetensorsTensorInfo *
  tensor_info(const Qwen35CheckpointTensor &tensor) const noexcept;
  [[nodiscard]] std::optional<std::span<const std::byte>>
  tensor_bytes(const Qwen35CheckpointTensor &tensor) const noexcept;

private:
  friend struct Qwen35CheckpointOpenResult;
  friend Qwen35CheckpointOpenResult
  open_qwen35_target_checkpoint(const std::filesystem::path &root) noexcept;
  friend class Qwen35CheckpointBuilder;

  uint32_t manifest_version_{0};
  std::filesystem::path root_;
  Qwen35TargetConfig config_;
  Qwen35QuantizationConfig quantization_;
  uint64_t total_parameters_{0};
  uint64_t total_tensor_bytes_{0};
  std::vector<Qwen35CheckpointShard> shards_;
  std::vector<Qwen35CheckpointTensor> tensors_;
};

struct Qwen35CheckpointOpenResult final {
  Qwen35Checkpoint checkpoint;
  Qwen35CheckpointError error;

  [[nodiscard]] bool ok() const noexcept {
    return error.code == Qwen35CheckpointErrorCode::kOk &&
           checkpoint.manifest_version() == kQwen35CheckpointManifestVersion;
  }
};

[[nodiscard]] std::string_view
qwen35_checkpoint_error_code_name(Qwen35CheckpointErrorCode code) noexcept;

// Opens and cross-validates the target config, index, quantization config,
// provenance manifest, all ten shard hashes, and every indexed tensor. No CUDA
// allocation occurs. Returned tensor byte spans alias the retained read-only
// shard mappings and remain valid only while the checkpoint remains alive.
[[nodiscard]] Qwen35CheckpointOpenResult
open_qwen35_target_checkpoint(const std::filesystem::path &root) noexcept;

} // namespace sglang::native

#endif // SGLANG_NATIVE_QWEN35_CHECKPOINT_HPP_
