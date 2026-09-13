#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

#include "mlx/memory.h"
#include "mlx/stream.h"
#include "qwen38_c_api.h"

namespace {

using Clock = std::chrono::steady_clock;

void ReportStage(const char* stage, Clock::time_point started) {
  std::cerr << std::fixed << std::setprecision(3)
            << "stage=" << stage
            << " elapsed_seconds="
            << std::chrono::duration<double>(Clock::now() - started).count()
            << " active_bytes=" << mlx::core::get_active_memory()
            << " cache_bytes=" << mlx::core::get_cache_memory()
            << " peak_bytes=" << mlx::core::get_peak_memory()
            << " memory_limit_bytes=" << mlx::core::get_memory_limit()
            << '\n';
}

template <typename Function>
Function LoadSymbol(void* library, const char* name) {
  dlerror();
  void* symbol = dlsym(library, name);
  if (const char* error = dlerror()) {
    throw std::runtime_error(std::string("cannot load ") + name + ": " + error);
  }
  return reinterpret_cast<Function>(symbol);
}

int ParseTokenCount(std::string_view text, const char* label, int minimum = 1) {
  int value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value < minimum) {
    throw std::runtime_error(
        std::string(label) +
        (minimum == 0 ? " must be a nonnegative integer"
                      : " must be a positive integer"));
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

std::vector<std::int32_t> ReadPromptIds(
    const std::filesystem::path& path, int count, int vocab_size) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot read prompt IDs from " + path.string());
  }
  std::vector<std::int32_t> tokens;
  tokens.reserve(static_cast<std::size_t>(count));
  std::string field;
  while (input >> field) {
    std::int32_t token = 0;
    const auto [end, error] =
        std::from_chars(field.data(), field.data() + field.size(), token);
    if (error != std::errc{} || end != field.data() + field.size() ||
        token < 0 || token >= vocab_size) {
      throw std::runtime_error("invalid token in prompt ID file");
    }
    if (tokens.size() == static_cast<std::size_t>(count)) {
      throw std::runtime_error("prompt ID file exceeds PROMPT_TOKENS");
    }
    tokens.push_back(token);
  }
  if (!input.eof() || tokens.size() != static_cast<std::size_t>(count)) {
    throw std::runtime_error("prompt ID file does not match PROMPT_TOKENS");
  }
  return tokens;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6 && argc != 7) {
    std::cerr
        << "usage: bench_qwen38_native LIBRARY MODEL_DIR PROMPT_TOKENS "
           "WARMUP_TOKENS OUTPUT_TOKENS [MTP_DIR]\n"
           "SGLANG_MLX_BENCH_PROMPT_IDS: optional whitespace-separated token file\n"
           "SGLANG_MLX_BENCH_OUTPUT_IDS: optional file for prefill, warmup and "
           "timed output token IDs\n";
    return 2;
  }

  try {
    const auto process_started = Clock::now();
    const int prompt_tokens = ParseTokenCount(argv[3], "PROMPT_TOKENS");
    const int warmup_tokens = ParseTokenCount(argv[4], "WARMUP_TOKENS", 0);
    const int output_tokens = ParseTokenCount(argv[5], "OUTPUT_TOKENS");
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
    using LoadMtp = int (*)(
        MlxQwen38Engine*, const char*, char*, int);
    using HasMtp = int (*)(MlxQwen38Engine*);
    using LastSpecWidth = int (*)(MlxQwen38Engine*);
    using HistoryDigest = int (*)(
        MlxQwen38Engine*, std::uint64_t*, char*, int);
    using Free = void (*)(MlxQwen38Engine*);

    const auto config_from_json =
        LoadSymbol<ConfigFromJson>(library, "mlx_qwen38_config_from_json");
    const auto load = LoadSymbol<Load>(library, "mlx_qwen38_load");
    const auto prefill = LoadSymbol<Prefill>(library, "mlx_qwen38_prefill");
    const auto decode = LoadSymbol<Decode>(library, "mlx_qwen38_decode");
    const auto load_mtp = LoadSymbol<LoadMtp>(library, "mlx_qwen38_load_mtp");
    const auto has_mtp = LoadSymbol<HasMtp>(library, "mlx_qwen38_has_mtp");
    const auto last_spec_width =
        LoadSymbol<LastSpecWidth>(library, "mlx_qwen38_last_spec_width");
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

    const char* const prompt_path = std::getenv("SGLANG_MLX_BENCH_PROMPT_IDS");
    std::vector<std::int32_t> prompt;
    if (prompt_path != nullptr) {
      prompt = ReadPromptIds(prompt_path, prompt_tokens, config.vocab_size);
    } else {
      prompt.resize(static_cast<std::size_t>(prompt_tokens));
      const auto token_range =
          static_cast<std::uint32_t>(config.vocab_size - 1024);
      for (int i = 0; i < prompt_tokens; ++i) {
        prompt[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(
            1024U + (static_cast<std::uint32_t>(i) * 7919U + 17U) % token_range);
      }
    }
    std::uint64_t prompt_digest = UINT64_C(14695981039346656037);
    for (const auto token : prompt) {
      prompt_digest = UpdateDigest(prompt_digest, token);
    }
    const char* const output_path = std::getenv("SGLANG_MLX_BENCH_OUTPUT_IDS");
    std::ofstream output_file;
    std::vector<std::int32_t> output_ids;
    if (output_path != nullptr) {
      if (prompt_path != nullptr &&
          std::filesystem::exists(output_path) &&
          std::filesystem::equivalent(prompt_path, output_path)) {
        throw std::runtime_error("output ID file must differ from prompt ID file");
      }
      output_file.open(output_path);
      if (!output_file) {
        throw std::runtime_error("cannot write output ID file");
      }
      output_ids.reserve(
          1 + static_cast<std::size_t>(warmup_tokens) +
          static_cast<std::size_t>(output_tokens));
    }

    std::unique_ptr<MlxQwen38Engine, Free> engine(
        load(&config, model_dir.c_str(), error, sizeof(error)), free_engine);
    if (!engine) {
      throw std::runtime_error(error);
    }
    if (argc == 7 &&
        load_mtp(engine.get(), argv[6], error, sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    const bool mtp_enabled = has_mtp(engine.get()) != 0;
    HistoryDigest history_digest = nullptr;
    const char* check_history = std::getenv("SGLANG_MLX_NATIVE_CHECK_MTP_HISTORY");
    if (check_history != nullptr && std::string_view(check_history) == "1") {
      history_digest = LoadSymbol<HistoryDigest>(
          library, "mlx_qwen38_mtp_history_digest");
    }
    const auto report_history = [&](const char* stage) {
      if (history_digest == nullptr) {
        return;
      }
      std::uint64_t digest = 0;
      if (history_digest(engine.get(), &digest, error, sizeof(error)) != 0) {
        throw std::runtime_error(error);
      }
      std::cerr << "stage=" << stage << " mtp_history_digest=" << std::hex
                << digest << std::dec << '\n';
    };
    ReportStage("loaded", process_started);

    std::int32_t token = 0;
    if (prefill(
            engine.get(),
            prompt.data(),
            prompt_tokens,
            mtp_enabled ? 0 : 1,
            &token,
            error,
            sizeof(error)) != 0) {
      throw std::runtime_error(error);
    }
    if (output_path != nullptr) {
      output_ids.push_back(token);
    }
    ReportStage("prefilled", process_started);
    report_history("prefilled");
    int buffered_tokens = 0;
    for (int i = 0; i < warmup_tokens; ++i) {
      const bool refilling = mtp_enabled && buffered_tokens == 0;
      if (decode(engine.get(), token, &token, error, sizeof(error)) != 0) {
        throw std::runtime_error(error);
      }
      if (refilling) {
        const int width = last_spec_width(engine.get());
        if (width <= 0) {
          throw std::runtime_error("MTP refill produced an empty token block");
        }
        buffered_tokens = width - 1;
      } else if (mtp_enabled) {
        --buffered_tokens;
      }
      if (output_path != nullptr) {
        output_ids.push_back(token);
      }
    }
    mlx::core::synchronize();
    ReportStage("warmed", process_started);
    report_history("warmed");

    constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
    std::uint64_t digest = kFnvOffset;
    std::uint64_t spec_width_sum = 0;
    int spec_refills = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < output_tokens; ++i) {
      const bool refilling = mtp_enabled && buffered_tokens == 0;
      if (decode(engine.get(), token, &token, error, sizeof(error)) != 0) {
        throw std::runtime_error(error);
      }
      if (refilling) {
        const int width = last_spec_width(engine.get());
        if (width <= 0) {
          throw std::runtime_error("MTP refill produced an empty token block");
        }
        buffered_tokens = width - 1;
        spec_width_sum += static_cast<std::uint64_t>(width);
        ++spec_refills;
      } else if (mtp_enabled) {
        --buffered_tokens;
      }
      digest = UpdateDigest(digest, token);
      if (output_path != nullptr) {
        output_ids.push_back(token);
      }
    }
    mlx::core::synchronize();
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    ReportStage("decoded", process_started);
    report_history("decoded");
    if (output_path != nullptr) {
      for (const auto id : output_ids) {
        output_file << id << '\n';
      }
      output_file.close();
      if (!output_file) {
        throw std::runtime_error("cannot finish output ID file");
      }
    }

    std::cout << std::fixed << std::setprecision(9)
              << "prompt_tokens=" << prompt_tokens << '\n'
              << "prompt_source=" << (prompt_path == nullptr ? "synthetic" : "file")
              << '\n'
              << "prompt_digest_fnv1a64=" << std::hex << prompt_digest << std::dec
              << '\n'
              << "warmup_tokens=" << warmup_tokens << '\n'
              << "output_tokens=" << output_tokens << '\n'
              << "seconds=" << seconds << '\n'
              << "tokens_per_second=" << output_tokens / seconds << '\n'
              << "mtp_enabled=" << (mtp_enabled ? 1 : 0) << '\n'
              << "spec_refills=" << spec_refills << '\n'
              << "mean_spec_width="
              << (spec_refills == 0
                      ? 0.0
                      : static_cast<double>(spec_width_sum) / spec_refills)
              << '\n'
              << "token_digest_fnv1a64=" << std::hex << std::setw(16)
              << std::setfill('0') << digest << std::dec << '\n'
              << "last_token=" << token << '\n';
  } catch (const std::exception& error) {
    std::cerr << "benchmark failed: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
