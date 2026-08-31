#include <charconv>
#include <chrono>
#include <cstdint>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mlx/stream.h"
#include "qwen38_c_api.h"

namespace {

template <typename Function>
Function LoadSymbol(void* library, const char* name) {
  dlerror();
  void* symbol = dlsym(library, name);
  if (const char* error = dlerror()) {
    throw std::runtime_error(std::string("cannot load ") + name + ": " + error);
  }
  return reinterpret_cast<Function>(symbol);
}

int ParsePositive(std::string_view text, const char* label) {
  int value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value <= 0) {
    throw std::runtime_error(std::string(label) + " must be a positive integer");
  }
  return value;
}

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read " + path.string());
  }
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

std::uint64_t UpdateDigest(std::uint64_t digest, std::int32_t token) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto bits = static_cast<std::uint32_t>(token);
  for (int shift = 0; shift < 32; shift += 8) {
    digest ^= (bits >> shift) & 0xffU;
    digest *= kPrime;
  }
  return digest;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr
        << "usage: bench_qwen38_native LIBRARY MODEL_DIR PROMPT_TOKENS "
           "WARMUP_TOKENS OUTPUT_TOKENS\n";
    return 2;
  }

  try {
    const int prompt_tokens = ParsePositive(argv[3], "PROMPT_TOKENS");
    const int warmup_tokens = ParsePositive(argv[4], "WARMUP_TOKENS");
    const int output_tokens = ParsePositive(argv[5], "OUTPUT_TOKENS");
    // MLX's process-lifetime compile cache retains primitives implemented by
    // the engine dylib. Keep that dylib loaded until process teardown so their
    // destructors never refer to unloaded code.
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
      throw std::runtime_error(std::string("cannot load library: ") + dlerror());
    }

    using ConfigFromJson = int (*)(
        const char*, MlxQwen38Config*, char*, int);
    using Load = MlxQwen38Engine* (*)(
        const MlxQwen38Config*, const char*, char*, int);
    using Prefill = int (*)(
        MlxQwen38Engine*, const std::int32_t*, int, int, std::int32_t*, char*, int);
    using Decode = int (*)(
        MlxQwen38Engine*, std::int32_t, std::int32_t*, char*, int);
    using Free = void (*)(MlxQwen38Engine*);

    const auto config_from_json =
        LoadSymbol<ConfigFromJson>(library, "mlx_qwen38_config_from_json");
    const auto load = LoadSymbol<Load>(library, "mlx_qwen38_load");
    const auto prefill = LoadSymbol<Prefill>(library, "mlx_qwen38_prefill");
    const auto decode = LoadSymbol<Decode>(library, "mlx_qwen38_decode");
    const auto free_engine = LoadSymbol<Free>(library, "mlx_qwen38_free");

    const std::filesystem::path model_dir(argv[2]);
    const std::string config_text = ReadFile(model_dir / "config.json");
    MlxQwen38Config config{};
    char error[1024]{};
    if (config_from_json(config_text.c_str(), &config, error, sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    if (config.vocab_size <= 2048) {
      throw std::runtime_error("model vocabulary is too small for benchmark tokens");
    }

    std::unique_ptr<MlxQwen38Engine, Free> engine(
        load(&config, model_dir.c_str(), error, sizeof(error)), free_engine);
    if (!engine) {
      throw std::runtime_error(error);
    }

    std::vector<std::int32_t> prompt(static_cast<std::size_t>(prompt_tokens));
    const std::uint32_t token_range =
        static_cast<std::uint32_t>(config.vocab_size - 1024);
    for (int i = 0; i < prompt_tokens; ++i) {
      prompt[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
          1024U + (static_cast<std::uint32_t>(i) * 7919U + 17U) % token_range);
    }

    std::int32_t token = 0;
    if (prefill(
            engine.get(),
            prompt.data(),
            prompt_tokens,
            1,
            &token,
            error,
            sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    for (int i = 0; i < warmup_tokens; ++i) {
      if (decode(engine.get(), token, &token, error, sizeof(error)) != 0) {
        throw std::runtime_error(error);
      }
    }
    mlx::core::synchronize();

    constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
    std::uint64_t digest = kFnvOffset;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < output_tokens; ++i) {
      if (decode(engine.get(), token, &token, error, sizeof(error)) != 0) {
        throw std::runtime_error(error);
      }
      digest = UpdateDigest(digest, token);
    }
    mlx::core::synchronize();
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();

    std::cout << std::fixed << std::setprecision(9)
              << "prompt_tokens=" << prompt_tokens << '\n'
              << "warmup_tokens=" << warmup_tokens << '\n'
              << "output_tokens=" << output_tokens << '\n'
              << "seconds=" << seconds << '\n'
              << "tokens_per_second=" << output_tokens / seconds << '\n'
              << "token_digest_fnv1a64=" << std::hex << std::setw(16)
              << std::setfill('0') << digest << std::dec << '\n'
              << "last_token=" << token << '\n';
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
