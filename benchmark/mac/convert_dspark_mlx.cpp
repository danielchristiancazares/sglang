#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mlx/io.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"

namespace {

namespace mx = mlx::core;
using mx::array;

constexpr int kGroupSize = 64;
constexpr int kBits = 4;
constexpr std::size_t kDSparkTensorCount = 62;
constexpr std::size_t kDSparkQuantizedMatrixCount = 37;
constexpr std::string_view kSourceRevision =
    "b9a5dbdf03bc999c6c73c426b19c2d9041cea393";
constexpr std::string_view kSourceSha256 =
    "2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7";

bool should_quantize(std::string_view name, const array &weight) {
  if (weight.ndim() != 2 || weight.shape().back() % kGroupSize != 0) {
    return false;
  }
  return name == "fc.weight" || name == "markov_head.markov_w2.weight" ||
         (name.starts_with("layers.") && name.ends_with(".weight"));
}

void require_tensor(const std::unordered_map<std::string, array> &weights,
                    const std::string &name, const mx::Shape &shape) {
  const auto found = weights.find(name);
  if (found == weights.end() || found->second.shape() != shape ||
      found->second.dtype() != mx::bfloat16) {
    throw std::runtime_error("invalid DSpark tensor " + name);
  }
}

void require_dspark_checkpoint(
    const std::unordered_map<std::string, array> &weights) {
  if (weights.size() != kDSparkTensorCount) {
    throw std::runtime_error(
        "DSpark source contains " + std::to_string(weights.size()) +
        " tensors; expected " + std::to_string(kDSparkTensorCount));
  }

  require_tensor(weights, "confidence_head.proj.bias", {1});
  require_tensor(weights, "confidence_head.proj.weight", {1, 5376});
  require_tensor(weights, "fc.weight", {5120, 25600});
  require_tensor(weights, "hidden_norm.weight", {5120});
  require_tensor(weights, "markov_head.markov_w1.weight", {248320, 256});
  require_tensor(weights, "markov_head.markov_w2.weight", {248320, 256});
  require_tensor(weights, "norm.weight", {5120});

  for (int index = 0; index < 5; ++index) {
    const std::string prefix = "layers." + std::to_string(index) + ".";
    require_tensor(weights, prefix + "input_layernorm.weight", {5120});
    require_tensor(weights, prefix + "mlp.down_proj.weight", {5120, 17408});
    require_tensor(weights, prefix + "mlp.gate_proj.weight", {17408, 5120});
    require_tensor(weights, prefix + "mlp.up_proj.weight", {17408, 5120});
    require_tensor(weights, prefix + "post_attention_layernorm.weight", {5120});
    require_tensor(weights, prefix + "self_attn.k_norm.weight", {128});
    require_tensor(weights, prefix + "self_attn.k_proj.weight", {1024, 5120});
    require_tensor(weights, prefix + "self_attn.o_proj.weight", {5120, 4096});
    require_tensor(weights, prefix + "self_attn.q_norm.weight", {128});
    require_tensor(weights, prefix + "self_attn.q_proj.weight", {4096, 5120});
    require_tensor(weights, prefix + "self_attn.v_proj.weight", {1024, 5120});
  }
}

std::size_t
convert_weights(const std::unordered_map<std::string, array> &source,
                std::unordered_map<std::string, array> &destination) {
  std::size_t quantized_count = 0;
  destination.reserve(source.size() + 2 * kDSparkQuantizedMatrixCount);
  for (const auto &[name, weight] : source) {
    if (!should_quantize(name, weight)) {
      destination.emplace(name, weight);
      continue;
    }
    std::vector<array> quantized =
        mx::quantize(weight, kGroupSize, kBits, "affine");
    if (quantized.size() != 3) {
      throw std::runtime_error("MLX affine quantization returned " +
                               std::to_string(quantized.size()) +
                               " tensors for " + name);
    }
    mx::eval(quantized);
    const std::string prefix = name.substr(0, name.size() - 7);
    destination.emplace(name, std::move(quantized[0]));
    destination.emplace(prefix + ".scales", std::move(quantized[1]));
    destination.emplace(prefix + ".biases", std::move(quantized[2]));
    ++quantized_count;
    std::cout << "quantized " << quantized_count << "/"
              << kDSparkQuantizedMatrixCount << " " << name << "\n";
  }
  if (quantized_count != kDSparkQuantizedMatrixCount) {
    throw std::runtime_error(
        "DSpark conversion quantized " + std::to_string(quantized_count) +
        " matrices; expected " + std::to_string(kDSparkQuantizedMatrixCount));
  }
  return quantized_count;
}

void convert_file(const std::filesystem::path &source_path,
                  const std::filesystem::path &destination_path) {
  if (source_path == destination_path) {
    throw std::runtime_error("source and destination paths must differ");
  }
  if (source_path.extension() != ".safetensors" ||
      destination_path.extension() != ".safetensors") {
    throw std::runtime_error("source and destination must use .safetensors");
  }
  if (std::filesystem::exists(destination_path)) {
    throw std::runtime_error("destination already exists: " +
                             destination_path.string());
  }
  const std::filesystem::path partial_path =
      destination_path.string() + ".part.safetensors";
  if (std::filesystem::exists(partial_path)) {
    throw std::runtime_error("partial destination already exists: " +
                             partial_path.string());
  }

  auto [source, metadata] = mx::load_safetensors(source_path.string());
  require_dspark_checkpoint(source);
  std::unordered_map<std::string, array> destination;
  const std::size_t quantized_count = convert_weights(source, destination);
  metadata["sglang.format"] = "dspark-mlx-affine";
  metadata["sglang.quantization"] = "affine-w4-gs64";
  metadata["sglang.quantized_matrices"] = std::to_string(quantized_count);
  metadata["sglang.source_revision"] = kSourceRevision;
  metadata["sglang.source_sha256"] = kSourceSha256;
  mx::save_safetensors(partial_path.string(), destination, metadata);

  auto [verified, verified_metadata] =
      mx::load_safetensors(partial_path.string());
  const std::size_t expected_tensors =
      kDSparkTensorCount + 2 * kDSparkQuantizedMatrixCount;
  if (verified.size() != expected_tensors ||
      verified_metadata.at("sglang.format") != "dspark-mlx-affine" ||
      verified_metadata.at("sglang.quantization") != "affine-w4-gs64" ||
      verified_metadata.at("sglang.quantized_matrices") != "37" ||
      verified_metadata.at("sglang.source_revision") != kSourceRevision ||
      verified_metadata.at("sglang.source_sha256") != kSourceSha256) {
    throw std::runtime_error("saved DSpark checkpoint verification failed");
  }
  for (const auto &[name, expected] : destination) {
    const auto found = verified.find(name);
    if (found == verified.end() || found->second.shape() != expected.shape() ||
        found->second.dtype() != expected.dtype()) {
      throw std::runtime_error("saved DSpark tensor mismatch: " + name);
    }
  }
  std::filesystem::rename(partial_path, destination_path);
  std::cout << "saved " << destination_path << " with " << verified.size()
            << " tensors\n";
}

void self_test() {
  const array matrix =
      mx::astype(mx::reshape(mx::arange(256), {4, kGroupSize}), mx::bfloat16) /
      array(32.0f, mx::bfloat16);
  constexpr std::array<std::string_view, 4> quantized_names = {
      "fc.weight",
      "layers.0.self_attn.q_proj.weight",
      "layers.4.mlp.down_proj.weight",
      "markov_head.markov_w2.weight",
  };
  for (const std::string_view name : quantized_names) {
    if (!should_quantize(name, matrix)) {
      throw std::runtime_error("self-test failed to select " +
                               std::string(name));
    }
  }
  constexpr std::array<std::string_view, 3> retained_names = {
      "confidence_head.proj.weight",
      "markov_head.markov_w1.weight",
      "norm.weight",
  };
  for (const std::string_view name : retained_names) {
    if (should_quantize(name, matrix)) {
      throw std::runtime_error("self-test unexpectedly selected " +
                               std::string(name));
    }
  }

  std::vector<array> quantized =
      mx::quantize(matrix, kGroupSize, kBits, "affine");
  const array restored =
      mx::dequantize(quantized[0], quantized[1], quantized[2], kGroupSize,
                     kBits, "affine", std::nullopt, mx::bfloat16);
  const array error = mx::max(mx::abs(mx::astype(restored, mx::float32) -
                                      mx::astype(matrix, mx::float32)));
  mx::eval(error);
  if (error.item<float>() > 0.3f) {
    throw std::runtime_error("self-test dequantization error is too large");
  }
  std::cout << "DSpark MLX converter self-test passed\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
      self_test();
      return 0;
    }
    if (argc != 3) {
      std::cerr << "usage: convert_dspark_mlx SOURCE.safetensors "
                   "DESTINATION.safetensors\n"
                   "       convert_dspark_mlx --self-test\n";
      return 2;
    }
    convert_file(argv[1], argv[2]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DSpark conversion failed: " << error.what() << "\n";
    return 1;
  }
}
