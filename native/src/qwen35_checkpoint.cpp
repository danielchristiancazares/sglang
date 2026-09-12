#include "sglang/native/qwen35_checkpoint.hpp"

#include "sglang/native/json_value.hpp"
#include "sglang/native/sha256.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

namespace sglang::native {
namespace {

constexpr uint64_t kMaximumJsonBytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::string_view kConfigName = "config.json";
constexpr std::string_view kIndexName = "model.safetensors.index.json";
constexpr std::string_view kQuantName = "hf_quant_config.json";
constexpr std::string_view kManifestName = "selective-nvfp4-manifest.json";
constexpr std::string_view kExpectedConfigHash =
    "88cb667373c8556f13fef4813cc9ba32c15a061d7867a6ca2b04cc082fa03ce1";
constexpr std::string_view kExpectedIndexHash =
    "f694aa7216ee4adf9895326ea706e40dd2426c83372f45c05416b48230aaa4ae";
constexpr std::string_view kExpectedQuantHash =
    "302d028778a8da5954458de596d48fff8b5beadfc36ba14f4808ed669b71999b";
constexpr uint32_t kTargetVocabularySize = 248320U;
constexpr uint32_t kTargetLayers = 64U;
constexpr uint32_t kTargetFullAttentionLayers = 16U;
constexpr uint32_t kTargetGdnLayers = 48U;
constexpr uint32_t kTargetAttentionHeads = 24U;
constexpr uint32_t kTargetKvHeads = 4U;
constexpr uint32_t kTargetHeadDimension = 256U;
constexpr uint32_t kTargetGdnKeyHeads = 16U;
constexpr uint32_t kTargetGdnValueHeads = 48U;
constexpr uint32_t kTargetGdnKeyDimension = 128U;
constexpr uint32_t kTargetGdnValueDimension = 128U;
constexpr uint32_t kTargetConvolutionKernel = 4U;
constexpr std::array<uint32_t, kTargetFullAttentionLayers>
    kTargetFullAttentionLayerIds{3,  7,  11, 15, 19, 23, 27, 31,
                                 35, 39, 43, 47, 51, 55, 59, 63};

struct LoadedJson final {
  std::vector<std::byte> bytes;
  std::string hash;
  NativeJsonValue value;
};

struct ShardExpectation final {
  std::string name;
  uint64_t file_bytes{0};
  uint64_t tensor_bytes{0};
  uint32_t tensor_count{0};
  std::string sha256;
};

[[nodiscard]] std::filesystem::path path_from_utf8(std::string_view text) {
  const auto *begin = reinterpret_cast<const char8_t *>(text.data());
  return std::filesystem::path(
      std::u8string_view(begin, static_cast<size_t>(text.size())));
}

} // namespace

class Qwen35CheckpointBuilder final {
public:
  static uint64_t &total_parameters(Qwen35Checkpoint &checkpoint) noexcept {
    return checkpoint.total_parameters_;
  }

  static uint64_t &total_tensor_bytes(Qwen35Checkpoint &checkpoint) noexcept {
    return checkpoint.total_tensor_bytes_;
  }

  static std::vector<Qwen35CheckpointTensor> &
  tensors(Qwen35Checkpoint &checkpoint) noexcept {
    return checkpoint.tensors_;
  }

  static const Qwen35QuantizationConfig &
  quantization(const Qwen35Checkpoint &checkpoint) noexcept {
    return checkpoint.quantization_;
  }
};

namespace {

[[nodiscard]] Qwen35CheckpointError
make_error(Qwen35CheckpointErrorCode code, std::string_view path = {},
           std::string_view member = {}, uint64_t detail = 0,
           uint64_t actual = 0, uint64_t required = 0,
           uint32_t system_error = 0) {
  return Qwen35CheckpointError{code,
                               system_error,
                               detail,
                               actual,
                               required,
                               std::string(path),
                               std::string(member)};
}

[[nodiscard]] std::string utf8_path(const std::filesystem::path &path) {
  const std::u8string encoded = path.u8string();
  return std::string(reinterpret_cast<const char *>(encoded.data()),
                     encoded.size());
}

[[nodiscard]] bool string_equals(const NativeJsonValue *value,
                                 std::string_view expected) noexcept {
  const std::string *text = value == nullptr ? nullptr : value->string_value();
  return text != nullptr && *text == expected;
}

[[nodiscard]] bool bool_equals(const NativeJsonValue *value,
                               bool expected) noexcept {
  const bool *actual = value == nullptr ? nullptr : value->bool_value();
  return actual != nullptr && *actual == expected;
}

[[nodiscard]] bool uint64_value(const NativeJsonValue *value,
                                uint64_t &output) noexcept {
  const int64_t *actual = value == nullptr ? nullptr : value->int_value();
  if (actual == nullptr || *actual < 0) {
    return false;
  }
  output = static_cast<uint64_t>(*actual);
  return true;
}

[[nodiscard]] bool uint32_value(const NativeJsonValue *value,
                                uint32_t &output) noexcept {
  uint64_t parsed = 0;
  if (!uint64_value(value, parsed) ||
      parsed > std::numeric_limits<uint32_t>::max()) {
    return false;
  }
  output = static_cast<uint32_t>(parsed);
  return true;
}

[[nodiscard]] Qwen35CheckpointError
load_json_file(const std::filesystem::path &path, LoadedJson &loaded) {
  const std::string display = utf8_path(path);
  HANDLE file =
      CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    return make_error(Qwen35CheckpointErrorCode::kFileOpenFailed, display, {},
                      0, 0, 0, GetLastError());
  }
  LARGE_INTEGER file_size{};
  if (GetFileSizeEx(file, &file_size) == 0) {
    const uint32_t error = GetLastError();
    static_cast<void>(CloseHandle(file));
    return make_error(Qwen35CheckpointErrorCode::kFileSizeFailed, display, {},
                      0, 0, 0, error);
  }
  if (file_size.QuadPart < 0 ||
      static_cast<uint64_t>(file_size.QuadPart) > kMaximumJsonBytes) {
    static_cast<void>(CloseHandle(file));
    return make_error(
        Qwen35CheckpointErrorCode::kFileTooLarge, display, {}, 0,
        file_size.QuadPart < 0 ? 0U : static_cast<uint64_t>(file_size.QuadPart),
        kMaximumJsonBytes);
  }
  loaded.bytes.resize(static_cast<size_t>(file_size.QuadPart));
  size_t offset = 0;
  while (offset < loaded.bytes.size()) {
    const size_t remaining = loaded.bytes.size() - offset;
    const DWORD request = static_cast<DWORD>(
        std::min<size_t>(remaining, std::numeric_limits<DWORD>::max()));
    DWORD read = 0;
    if (ReadFile(file, loaded.bytes.data() + offset, request, &read, nullptr) ==
            0 ||
        read == 0U) {
      const uint32_t error = GetLastError();
      static_cast<void>(CloseHandle(file));
      return make_error(Qwen35CheckpointErrorCode::kFileReadFailed, display, {},
                        offset, read, request, error);
    }
    offset += read;
  }
  LARGE_INTEGER final_size{};
  const bool final_size_ok = GetFileSizeEx(file, &final_size) != 0;
  const uint32_t final_size_error = final_size_ok ? 0U : GetLastError();
  static_cast<void>(CloseHandle(file));
  if (!final_size_ok) {
    return make_error(Qwen35CheckpointErrorCode::kFileSizeFailed, display, {},
                      0, 0, 0, final_size_error);
  }
  if (final_size.QuadPart != file_size.QuadPart) {
    return make_error(Qwen35CheckpointErrorCode::kFileReadFailed, display, {},
                      0, static_cast<uint64_t>(final_size.QuadPart),
                      static_cast<uint64_t>(file_size.QuadPart));
  }
  loaded.hash = sha256_hex(sha256(loaded.bytes));
  const auto *characters = reinterpret_cast<const char *>(loaded.bytes.data());
  NativeJsonParseResult parsed =
      parse_native_json(std::string_view(characters, loaded.bytes.size()));
  if (!parsed.ok()) {
    return make_error(Qwen35CheckpointErrorCode::kJsonInvalid, display, {},
                      parsed.error.byte_offset,
                      static_cast<uint64_t>(parsed.error.code));
  }
  loaded.value = std::move(parsed.value);
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

[[nodiscard]] Qwen35CheckpointError
parse_target_config(const LoadedJson &json, Qwen35TargetConfig &config) {
  const NativeJsonValue &root = json.value;
  const NativeJsonValue *architectures = root.find("architectures");
  const NativeJsonValue::Array *architecture_values =
      architectures == nullptr ? nullptr : architectures->array_value();
  if (architecture_values == nullptr || architecture_values->size() != 1U ||
      !string_equals(&(*architecture_values)[0],
                     "Qwen3_5ForConditionalGeneration") ||
      !string_equals(root.find("dtype"), "bfloat16") ||
      !string_equals(root.find("model_type"), "qwen3_5") ||
      !bool_equals(root.find("language_model_only"), false)) {
    return make_error(Qwen35CheckpointErrorCode::kValueMismatch, kConfigName,
                      "root");
  }
  const NativeJsonValue *text_value = root.find("text_config");
  if (text_value == nullptr || !text_value->is_object()) {
    return make_error(Qwen35CheckpointErrorCode::kSchemaInvalid, kConfigName,
                      "text_config");
  }
  const NativeJsonValue &text = *text_value;
  const struct Field final {
    std::string_view name;
    uint32_t *destination;
    uint32_t expected;
  } fields[]{
      {"vocab_size", &config.vocabulary_size, kTargetVocabularySize},
      {"hidden_size", &config.hidden_size, 5120U},
      {"intermediate_size", &config.intermediate_size, 17408U},
      {"num_hidden_layers", &config.layers, kTargetLayers},
      {"num_attention_heads", &config.attention_heads, kTargetAttentionHeads},
      {"num_key_value_heads", &config.attention_kv_heads, kTargetKvHeads},
      {"head_dim", &config.attention_head_dimension, kTargetHeadDimension},
      {"linear_num_key_heads", &config.gdn_key_heads, kTargetGdnKeyHeads},
      {"linear_num_value_heads", &config.gdn_value_heads, kTargetGdnValueHeads},
      {"linear_key_head_dim", &config.gdn_key_dimension,
       kTargetGdnKeyDimension},
      {"linear_value_head_dim", &config.gdn_value_dimension,
       kTargetGdnValueDimension},
      {"linear_conv_kernel_dim", &config.convolution_kernel_size,
       kTargetConvolutionKernel},
      {"mtp_num_hidden_layers", &config.mtp_layers, 1U},
  };
  for (const Field &field : fields) {
    if (!uint32_value(text.find(field.name), *field.destination) ||
        *field.destination != field.expected) {
      return make_error(Qwen35CheckpointErrorCode::kValueMismatch, kConfigName,
                        field.name, 0, *field.destination, field.expected);
    }
  }
  if (!uint64_value(text.find("max_position_embeddings"),
                    config.max_position_embeddings) ||
      config.max_position_embeddings != 262144ULL ||
      !string_equals(text.find("dtype"), "bfloat16") ||
      !string_equals(text.find("model_type"), "qwen3_5_text") ||
      !string_equals(text.find("mamba_ssm_dtype"), "float32") ||
      !bool_equals(text.find("mtp_use_dedicated_embeddings"), false)) {
    return make_error(Qwen35CheckpointErrorCode::kValueMismatch, kConfigName,
                      "text_config");
  }

  const NativeJsonValue *layer_types_value = text.find("layer_types");
  const NativeJsonValue::Array *layer_types =
      layer_types_value == nullptr ? nullptr : layer_types_value->array_value();
  if (layer_types == nullptr || layer_types->size() != kTargetLayers) {
    return make_error(
        Qwen35CheckpointErrorCode::kValueMismatch, kConfigName, "layer_types",
        0, layer_types == nullptr ? 0U : layer_types->size(), kTargetLayers);
  }
  uint32_t full_count = 0;
  uint32_t gdn_count = 0;
  for (uint32_t layer = 0; layer < layer_types->size(); ++layer) {
    const std::string *type = (*layer_types)[layer].string_value();
    if (type == nullptr) {
      return make_error(Qwen35CheckpointErrorCode::kSchemaInvalid, kConfigName,
                        "layer_types", layer);
    }
    if (*type == "full_attention") {
      if (full_count >= config.full_attention_layer_ids.size()) {
        return make_error(Qwen35CheckpointErrorCode::kValueMismatch,
                          kConfigName, "layer_types", layer);
      }
      config.full_attention_layer_ids[full_count++] = layer;
    } else if (*type == "linear_attention") {
      ++gdn_count;
    } else {
      return make_error(Qwen35CheckpointErrorCode::kValueMismatch, kConfigName,
                        "layer_types", layer);
    }
  }
  config.full_attention_layers = full_count;
  config.gdn_layers = gdn_count;
  if (full_count != kTargetFullAttentionLayers ||
      gdn_count != kTargetGdnLayers ||
      config.full_attention_layer_ids != kTargetFullAttentionLayerIds) {
    return make_error(Qwen35CheckpointErrorCode::kValueMismatch, kConfigName,
                      "layer_types");
  }
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

[[nodiscard]] Qwen35CheckpointError
parse_quantization(const LoadedJson &json,
                   Qwen35QuantizationConfig &quantization) {
  const NativeJsonValue &root = json.value;
  const NativeJsonValue *producer = root.find("producer");
  if (producer == nullptr || !producer->is_object() ||
      !string_equals(producer->find("name"), "modelopt") ||
      !string_equals(producer->find("version"), "0.47.0.dev0")) {
    return make_error(Qwen35CheckpointErrorCode::kQuantizationInvalid,
                      kQuantName, "producer");
  }
  const NativeJsonValue *quant = root.find("quantization");
  if (quant == nullptr || !quant->is_object() ||
      !string_equals(quant->find("quant_algo"), "MIXED_PRECISION") ||
      !string_equals(quant->find("kv_cache_quant_algo"), "FP8")) {
    return make_error(Qwen35CheckpointErrorCode::kQuantizationInvalid,
                      kQuantName, "quantization");
  }
  const NativeJsonValue *layers_value = quant->find("quantized_layers");
  const NativeJsonValue::Object *layers =
      layers_value == nullptr ? nullptr : layers_value->object_value();
  if (layers == nullptr || layers->size() != kQwen35TargetQuantizedLayers) {
    return make_error(Qwen35CheckpointErrorCode::kQuantizationInvalid,
                      kQuantName, "quantized_layers", 0,
                      layers == nullptr ? 0U : layers->size(),
                      kQwen35TargetQuantizedLayers);
  }
  quantization.quantized_layers = static_cast<uint32_t>(layers->size());
  quantization.nvfp4_group_size = 16U;
  quantization.fp8_kv_cache = true;
  quantization.nvfp4_projection_names.reserve(layers->size());
  for (const auto &[name, descriptor] : *layers) {
    uint32_t group_size = 0;
    if (!descriptor.is_object() ||
        !string_equals(descriptor.find("quant_algo"), "NVFP4") ||
        !uint32_value(descriptor.find("group_size"), group_size) ||
        group_size != quantization.nvfp4_group_size) {
      return make_error(Qwen35CheckpointErrorCode::kQuantizationInvalid,
                        kQuantName, name);
    }
    quantization.nvfp4_projection_names.push_back(name);
  }
  const NativeJsonValue *exclude_value = quant->find("exclude_modules");
  const NativeJsonValue::Array *exclude =
      exclude_value == nullptr ? nullptr : exclude_value->array_value();
  if (exclude == nullptr || exclude->size() != 2U ||
      !string_equals(&(*exclude)[0], "mtp*") ||
      !string_equals(&(*exclude)[1], "mtp.layers.0*")) {
    return make_error(Qwen35CheckpointErrorCode::kQuantizationInvalid,
                      kQuantName, "exclude_modules");
  }
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

[[nodiscard]] Qwen35CheckpointError
parse_manifest(const LoadedJson &json,
               std::vector<ShardExpectation> &expectations) {
  const NativeJsonValue &root = json.value;
  uint32_t projection_bases = 0;
  uint32_t selected_tensors = 0;
  uint32_t output_tensors = 0;
  if (!uint32_value(root.find("selected_projection_bases"), projection_bases) ||
      projection_bases != kQwen35TargetSelectedProjectionBases ||
      !uint32_value(root.find("selected_tensor_count"), selected_tensors) ||
      selected_tensors != kQwen35TargetSelectedTensorCount ||
      !uint32_value(root.find("output_tensor_count"), output_tensors) ||
      output_tensors != kQwen35TargetTensorCount ||
      !bool_equals(root.find("source_checkpoints_mutated"), false) ||
      !string_equals(root.find("config_sha256"), kExpectedConfigHash) ||
      !string_equals(root.find("index_sha256"), kExpectedIndexHash) ||
      !string_equals(root.find("hf_quant_config_sha256"), kExpectedQuantHash)) {
    return make_error(Qwen35CheckpointErrorCode::kManifestInvalid,
                      kManifestName, "root");
  }
  const NativeJsonValue *shards_value = root.find("shards");
  const NativeJsonValue *output_value = root.find("output_shards");
  const NativeJsonValue::Object *shards =
      shards_value == nullptr ? nullptr : shards_value->object_value();
  const NativeJsonValue::Array *outputs =
      output_value == nullptr ? nullptr : output_value->array_value();
  if (shards == nullptr || outputs == nullptr ||
      shards->size() != kQwen35TargetShardCount ||
      outputs->size() != kQwen35TargetShardCount) {
    return make_error(Qwen35CheckpointErrorCode::kManifestInvalid,
                      kManifestName, "shards");
  }
  expectations.reserve(outputs->size());
  for (const NativeJsonValue &output : *outputs) {
    const std::string *name = output.find("name") == nullptr
                                  ? nullptr
                                  : output.find("name")->string_value();
    const std::string *hash = output.find("sha256") == nullptr
                                  ? nullptr
                                  : output.find("sha256")->string_value();
    uint64_t size = 0;
    if (name == nullptr || hash == nullptr || hash->size() != 64U ||
        !uint64_value(output.find("size"), size)) {
      return make_error(Qwen35CheckpointErrorCode::kManifestInvalid,
                        kManifestName, "output_shards");
    }
    const auto shard = shards->find(*name);
    uint32_t tensor_count = 0;
    uint64_t tensor_bytes = 0;
    if (shard == shards->end() ||
        !uint32_value(shard->second.find("tensor_count"), tensor_count) ||
        !uint64_value(shard->second.find("tensor_bytes"), tensor_bytes)) {
      return make_error(Qwen35CheckpointErrorCode::kManifestInvalid,
                        kManifestName, *name);
    }
    expectations.push_back(
        ShardExpectation{*name, size, tensor_bytes, tensor_count, *hash});
  }
  std::sort(
      expectations.begin(), expectations.end(),
      [](const ShardExpectation &left, const ShardExpectation &right) noexcept {
        return left.name < right.name;
      });
  for (size_t index = 1; index < expectations.size(); ++index) {
    if (expectations[index - 1U].name == expectations[index].name) {
      return make_error(Qwen35CheckpointErrorCode::kManifestInvalid,
                        kManifestName, expectations[index].name);
    }
  }
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

[[nodiscard]] Qwen35CheckpointError
parse_index(const LoadedJson &json,
            const std::vector<ShardExpectation> &expectations,
            Qwen35Checkpoint &checkpoint) {
  const NativeJsonValue &root = json.value;
  const NativeJsonValue *metadata = root.find("metadata");
  uint64_t &total_parameters =
      Qwen35CheckpointBuilder::total_parameters(checkpoint);
  uint64_t &total_tensor_bytes =
      Qwen35CheckpointBuilder::total_tensor_bytes(checkpoint);
  std::vector<Qwen35CheckpointTensor> &tensors =
      Qwen35CheckpointBuilder::tensors(checkpoint);
  if (metadata == nullptr || !metadata->is_object() ||
      !uint64_value(metadata->find("total_parameters"), total_parameters) ||
      total_parameters != kQwen35TargetTotalParameters ||
      !uint64_value(metadata->find("total_size"), total_tensor_bytes) ||
      total_tensor_bytes != kQwen35TargetTensorBytes) {
    return make_error(Qwen35CheckpointErrorCode::kIndexInvalid, kIndexName,
                      "metadata");
  }
  const NativeJsonValue *map_value = root.find("weight_map");
  const NativeJsonValue::Object *weight_map =
      map_value == nullptr ? nullptr : map_value->object_value();
  if (weight_map == nullptr || weight_map->size() != kQwen35TargetTensorCount) {
    return make_error(Qwen35CheckpointErrorCode::kIndexInvalid, kIndexName,
                      "weight_map", 0,
                      weight_map == nullptr ? 0U : weight_map->size(),
                      kQwen35TargetTensorCount);
  }
  tensors.reserve(weight_map->size());
  for (const auto &[tensor_name, shard_value] : *weight_map) {
    const std::string *shard_name = shard_value.string_value();
    if (shard_name == nullptr) {
      return make_error(Qwen35CheckpointErrorCode::kIndexInvalid, kIndexName,
                        tensor_name);
    }
    const auto shard = std::lower_bound(
        expectations.begin(), expectations.end(), *shard_name,
        [](const ShardExpectation &entry, std::string_view wanted) noexcept {
          return entry.name < wanted;
        });
    if (shard == expectations.end() || shard->name != *shard_name) {
      return make_error(Qwen35CheckpointErrorCode::kIndexInvalid, kIndexName,
                        tensor_name);
    }
    tensors.push_back(Qwen35CheckpointTensor{
        tensor_name, static_cast<uint32_t>(shard - expectations.begin()), 0U});
  }
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

[[nodiscard]] Qwen35CheckpointError
validate_quantized_tensor_families(const Qwen35Checkpoint &checkpoint) {
  for (const std::string &base :
       Qwen35CheckpointBuilder::quantization(checkpoint)
           .nvfp4_projection_names) {
    for (const std::string_view suffix : std::array<std::string_view, 4>{
             ".input_scale", ".weight", ".weight_scale", ".weight_scale_2"}) {
      const std::string name = base + std::string(suffix);
      if (checkpoint.find_tensor(name) == nullptr) {
        return make_error(Qwen35CheckpointErrorCode::kTensorMissing, {}, name);
      }
    }
  }
  return make_error(Qwen35CheckpointErrorCode::kOk);
}

} // namespace

const Qwen35CheckpointTensor *
Qwen35Checkpoint::find_tensor(std::string_view name) const noexcept {
  const auto found = std::lower_bound(
      tensors_.begin(), tensors_.end(), name,
      [](const Qwen35CheckpointTensor &tensor,
         std::string_view wanted) noexcept { return tensor.name < wanted; });
  return found != tensors_.end() && found->name == name ? &*found : nullptr;
}

const SafetensorsTensorInfo *Qwen35Checkpoint::tensor_info(
    const Qwen35CheckpointTensor &tensor) const noexcept {
  if (tensor.shard_index >= shards_.size()) {
    return nullptr;
  }
  const auto shard_tensors = shards_[tensor.shard_index].file.tensors();
  if (tensor.tensor_index >= shard_tensors.size()) {
    return nullptr;
  }
  const SafetensorsTensorInfo &info = shard_tensors[tensor.tensor_index];
  return info.name == tensor.name ? &info : nullptr;
}

std::optional<std::span<const std::byte>> Qwen35Checkpoint::tensor_bytes(
    const Qwen35CheckpointTensor &tensor) const noexcept {
  const SafetensorsTensorInfo *info = tensor_info(tensor);
  return info == nullptr ? std::nullopt
                         : shards_[tensor.shard_index].file.tensor_bytes(*info);
}

Qwen35CheckpointOpenResult
open_qwen35_target_checkpoint(const std::filesystem::path &root) noexcept {
  Qwen35CheckpointOpenResult result;
  if (root.empty()) {
    result.error = make_error(Qwen35CheckpointErrorCode::kInvalidRoot);
    return result;
  }
  try {
    result.checkpoint.root_ = root;
    LoadedJson config_json;
    LoadedJson index_json;
    LoadedJson quant_json;
    LoadedJson manifest_json;
    for (const auto &[name, destination] :
         std::array<std::pair<std::string_view, LoadedJson *>, 4>{
             std::pair{kConfigName, &config_json},
             std::pair{kIndexName, &index_json},
             std::pair{kQuantName, &quant_json},
             std::pair{kManifestName, &manifest_json}}) {
      result.error = load_json_file(root / path_from_utf8(name), *destination);
      if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
        return result;
      }
    }
    if (config_json.hash != kExpectedConfigHash ||
        index_json.hash != kExpectedIndexHash ||
        quant_json.hash != kExpectedQuantHash) {
      result.error =
          make_error(Qwen35CheckpointErrorCode::kHashMismatch, utf8_path(root));
      return result;
    }
    result.error = parse_target_config(config_json, result.checkpoint.config_);
    if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
      return result;
    }
    result.error =
        parse_quantization(quant_json, result.checkpoint.quantization_);
    if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
      return result;
    }
    std::vector<ShardExpectation> expectations;
    result.error = parse_manifest(manifest_json, expectations);
    if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
      return result;
    }
    result.error = parse_index(index_json, expectations, result.checkpoint);
    if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
      return result;
    }

    result.checkpoint.shards_.reserve(expectations.size());
    uint64_t observed_tensor_bytes = 0;
    size_t observed_tensor_count = 0;
    for (const ShardExpectation &expected : expectations) {
      SafetensorsOpenResult opened = open_safetensors_file(
          (root / path_from_utf8(expected.name)).native());
      if (!opened.ok()) {
        result.error = make_error(
            Qwen35CheckpointErrorCode::kShardInvalid, expected.name, {},
            static_cast<uint64_t>(opened.error.code), opened.error.actual,
            opened.error.required, opened.error.system_error);
        return result;
      }
      if (opened.file.file_size() != expected.file_bytes ||
          opened.file.data_size() != expected.tensor_bytes ||
          opened.file.tensors().size() != expected.tensor_count) {
        result.error =
            make_error(Qwen35CheckpointErrorCode::kShardInvalid, expected.name);
        return result;
      }
      const std::string actual_hash =
          sha256_hex(sha256(opened.file.file_bytes()));
      if (actual_hash != expected.sha256) {
        result.error = make_error(Qwen35CheckpointErrorCode::kHashMismatch,
                                  expected.name, "sha256");
        return result;
      }
      observed_tensor_bytes += opened.file.data_size();
      observed_tensor_count += opened.file.tensors().size();
      result.checkpoint.shards_.push_back(Qwen35CheckpointShard{
          expected.name, expected.file_bytes, expected.tensor_bytes,
          expected.tensor_count, expected.sha256, std::move(opened.file)});
    }
    if (observed_tensor_bytes != kQwen35TargetTensorBytes) {
      result.error =
          make_error(Qwen35CheckpointErrorCode::kTensorByteCountMismatch, {},
                     {}, 0, observed_tensor_bytes, kQwen35TargetTensorBytes);
      return result;
    }
    if (observed_tensor_count != kQwen35TargetTensorCount) {
      result.error =
          make_error(Qwen35CheckpointErrorCode::kTensorCountMismatch, {}, {}, 0,
                     observed_tensor_count, kQwen35TargetTensorCount);
      return result;
    }

    std::vector<bool> indexed(observed_tensor_count, false);
    std::vector<size_t> shard_bases(result.checkpoint.shards_.size(), 0U);
    for (size_t shard = 1; shard < shard_bases.size(); ++shard) {
      shard_bases[shard] =
          shard_bases[shard - 1U] +
          result.checkpoint.shards_[shard - 1U].file.tensors().size();
    }
    for (Qwen35CheckpointTensor &tensor : result.checkpoint.tensors_) {
      const SafetensorsFile &shard =
          result.checkpoint.shards_[tensor.shard_index].file;
      const SafetensorsTensorInfo *info = shard.find(tensor.name);
      if (info == nullptr) {
        result.error = make_error(
            Qwen35CheckpointErrorCode::kTensorMissing,
            result.checkpoint.shards_[tensor.shard_index].name, tensor.name);
        return result;
      }
      const auto shard_tensors = shard.tensors();
      tensor.tensor_index = static_cast<uint32_t>(info - shard_tensors.data());
      indexed[shard_bases[tensor.shard_index] + tensor.tensor_index] = true;
    }
    if (std::find(indexed.begin(), indexed.end(), false) != indexed.end()) {
      result.error = make_error(Qwen35CheckpointErrorCode::kTensorUnexpected);
      return result;
    }
    result.error = validate_quantized_tensor_families(result.checkpoint);
    if (result.error.code != Qwen35CheckpointErrorCode::kOk) {
      return result;
    }
    result.checkpoint.manifest_version_ = kQwen35CheckpointManifestVersion;
  } catch (const std::bad_alloc &) {
    result.error = make_error(Qwen35CheckpointErrorCode::kAllocationFailed);
  } catch (...) {
    result.error = make_error(Qwen35CheckpointErrorCode::kInternalFailure);
  }
  return result;
}

std::string_view
qwen35_checkpoint_error_code_name(Qwen35CheckpointErrorCode code) noexcept {
  switch (code) {
  case Qwen35CheckpointErrorCode::kOk:
    return "ok";
  case Qwen35CheckpointErrorCode::kInvalidRoot:
    return "invalid_root";
  case Qwen35CheckpointErrorCode::kFileOpenFailed:
    return "file_open_failed";
  case Qwen35CheckpointErrorCode::kFileSizeFailed:
    return "file_size_failed";
  case Qwen35CheckpointErrorCode::kFileReadFailed:
    return "file_read_failed";
  case Qwen35CheckpointErrorCode::kFileTooLarge:
    return "file_too_large";
  case Qwen35CheckpointErrorCode::kJsonInvalid:
    return "json_invalid";
  case Qwen35CheckpointErrorCode::kSchemaInvalid:
    return "schema_invalid";
  case Qwen35CheckpointErrorCode::kValueMismatch:
    return "value_mismatch";
  case Qwen35CheckpointErrorCode::kHashMismatch:
    return "hash_mismatch";
  case Qwen35CheckpointErrorCode::kIndexInvalid:
    return "index_invalid";
  case Qwen35CheckpointErrorCode::kQuantizationInvalid:
    return "quantization_invalid";
  case Qwen35CheckpointErrorCode::kManifestInvalid:
    return "manifest_invalid";
  case Qwen35CheckpointErrorCode::kShardInvalid:
    return "shard_invalid";
  case Qwen35CheckpointErrorCode::kTensorMissing:
    return "tensor_missing";
  case Qwen35CheckpointErrorCode::kTensorUnexpected:
    return "tensor_unexpected";
  case Qwen35CheckpointErrorCode::kTensorShardMismatch:
    return "tensor_shard_mismatch";
  case Qwen35CheckpointErrorCode::kTensorCountMismatch:
    return "tensor_count_mismatch";
  case Qwen35CheckpointErrorCode::kTensorByteCountMismatch:
    return "tensor_byte_count_mismatch";
  case Qwen35CheckpointErrorCode::kAllocationFailed:
    return "allocation_failed";
  case Qwen35CheckpointErrorCode::kInternalFailure:
    return "internal_failure";
  default:
    return "invalid_qwen35_checkpoint_error";
  }
}

} // namespace sglang::native
