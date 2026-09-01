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
constexpr std::size_t kDFlash2TensorCount = 81;
constexpr std::size_t kDFlash2QuantizedMatrixCount = 47;

bool should_quantize(std::string_view name, const array &weight) {
  return name.ends_with(".weight") && weight.ndim() == 2 &&
         weight.shape().back() % kGroupSize == 0;
}

void require_dflash2_checkpoint(
    const std::unordered_map<std::string, array> &weights) {
  constexpr std::string_view required[] = {
      "candidate_selector.hidden_projection.weight",
      "candidate_selector.predecessor_codebook",
      "candidate_selector.successor_codebook",
      "fc.weight",
      "hidden_norm.weight",
      "layers.0.attention_conv.base_kernel",
      "layers.4.self_attn.v_proj.weight",
      "norm.weight",
  };
  for (const std::string_view name : required) {
    if (!weights.contains(std::string(name))) {
      throw std::runtime_error("source is missing DFlash2 tensor " +
                               std::string(name));
    }
  }
  if (weights.size() != kDFlash2TensorCount) {
    throw std::runtime_error(
        "DFlash2 source contains " + std::to_string(weights.size()) +
        " tensors; expected " + std::to_string(kDFlash2TensorCount));
  }
}

std::size_t
convert_weights(const std::unordered_map<std::string, array> &source,
                std::unordered_map<std::string, array> &destination) {
  std::size_t quantized_count = 0;
  destination.reserve(source.size() + 2 * kDFlash2QuantizedMatrixCount);
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
              << kDFlash2QuantizedMatrixCount << " " << name << "\n";
  }
  if (quantized_count != kDFlash2QuantizedMatrixCount) {
    throw std::runtime_error(
        "DFlash2 conversion quantized " + std::to_string(quantized_count) +
        " matrices; expected " + std::to_string(kDFlash2QuantizedMatrixCount));
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
  require_dflash2_checkpoint(source);
  std::unordered_map<std::string, array> destination;
  const std::size_t quantized_count = convert_weights(source, destination);
  metadata["sglang.format"] = "dflash2-mlx-affine";
  metadata["sglang.quantization"] = "affine-w4-gs64";
  metadata["sglang.quantized_matrices"] = std::to_string(quantized_count);
  metadata["sglang.source_sha256"] =
      "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c";
  mx::save_safetensors(partial_path.string(), destination, metadata);

  auto [verified, verified_metadata] =
      mx::load_safetensors(partial_path.string());
  const std::size_t expected_tensors =
      kDFlash2TensorCount + 2 * kDFlash2QuantizedMatrixCount;
  if (verified.size() != expected_tensors ||
      verified_metadata.at("sglang.format") != "dflash2-mlx-affine" ||
      verified_metadata.at("sglang.quantization") != "affine-w4-gs64" ||
      verified_metadata.at("sglang.quantized_matrices") != "47" ||
      verified_metadata.at("sglang.source_sha256") !=
          "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c") {
    throw std::runtime_error("saved DFlash2 checkpoint verification failed");
  }
  for (const auto &[name, expected] : destination) {
    const auto found = verified.find(name);
    if (found == verified.end() || found->second.shape() != expected.shape() ||
        found->second.dtype() != expected.dtype()) {
      throw std::runtime_error("saved DFlash2 tensor mismatch: " + name);
    }
  }
  std::filesystem::rename(partial_path, destination_path);
  std::cout << "saved " << destination_path << " with " << verified.size()
            << " tensors\n";
}

void self_test() {
  std::unordered_map<std::string, array> source;
  const array matrix =
      mx::astype(mx::reshape(mx::arange(256), {4, kGroupSize}), mx::bfloat16) /
      array(32.0f, mx::bfloat16);
  constexpr std::string_view required_matrices[] = {
      "candidate_selector.hidden_projection.weight",
      "fc.weight",
      "layers.4.self_attn.v_proj.weight",
  };
  for (const std::string_view name : required_matrices) {
    source.emplace(std::string(name), matrix);
  }
  for (std::size_t index = 0;
       index < kDFlash2QuantizedMatrixCount - std::size(required_matrices);
       ++index) {
    source.emplace("matrix." + std::to_string(index) + ".weight", matrix);
  }
  constexpr std::string_view required_retained[] = {
      "candidate_selector.predecessor_codebook",
      "candidate_selector.successor_codebook",
      "hidden_norm.weight",
      "layers.0.attention_conv.base_kernel",
      "norm.weight",
  };
  const array retained = mx::zeros({1}, mx::bfloat16);
  for (const std::string_view name : required_retained) {
    source.emplace(std::string(name), retained);
  }
  constexpr std::size_t retained_count =
      kDFlash2TensorCount - kDFlash2QuantizedMatrixCount;
  for (std::size_t index = 0;
       index < retained_count - std::size(required_retained); ++index) {
    source.emplace("retained." + std::to_string(index), retained);
  }

  require_dflash2_checkpoint(source);
  std::unordered_map<std::string, array> destination;
  const std::size_t count = convert_weights(source, destination);
  if (count != kDFlash2QuantizedMatrixCount ||
      destination.size() !=
          kDFlash2TensorCount + 2 * kDFlash2QuantizedMatrixCount) {
    throw std::runtime_error("self-test tensor count mismatch");
  }
  const array restored =
      mx::dequantize(destination.at("fc.weight"), destination.at("fc.scales"),
                     destination.at("fc.biases"), kGroupSize, kBits, "affine",
                     std::nullopt, mx::bfloat16);
  const array expected = source.at("fc.weight");
  const array error = mx::max(mx::abs(mx::astype(restored, mx::float32) -
                                      mx::astype(expected, mx::float32)));
  mx::eval(error);
  if (error.item<float>() > 0.3f) {
    throw std::runtime_error("self-test dequantization error is too large");
  }
  std::cout << "DFlash2 MLX converter self-test passed\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
      self_test();
      return 0;
    }
    if (argc != 3) {
      std::cerr << "usage: convert_dflash2_mlx SOURCE.safetensors "
                   "DESTINATION.safetensors\n"
                   "       convert_dflash2_mlx --self-test\n";
      return 2;
    }
    convert_file(argv[1], argv[2]);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DFlash2 conversion failed: " << error.what() << "\n";
    return 1;
  }
}
