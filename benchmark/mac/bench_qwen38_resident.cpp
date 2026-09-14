#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "mlx/memory.h"
#include "mlx/stream.h"
#include "qwen38_engine.h"

namespace native = sglang::mlx_qwen38;
using Clock = std::chrono::steady_clock;

int Positive(const char* argument) {
  const std::string_view text(argument);
  int value = 0;
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size() || value <= 0)
    throw std::runtime_error("expected positive integer");
  return value;
}

std::uint64_t Digest(const std::vector<std::int32_t>& ids, std::size_t start) {
  std::uint64_t digest = UINT64_C(14695981039346656037);
  for (std::size_t i = start; i < ids.size(); ++i) {
    const auto id = static_cast<std::uint32_t>(ids[i]);
    for (int shift = 0; shift < 32; shift += 8) {
      digest ^= (id >> shift) & 0xff;
      digest *= UINT64_C(1099511628211);
    }
  }
  return digest;
}

int main(int argc, char** argv) {
  try {
    if (argc != 9) throw std::runtime_error(
        "usage: resident MODEL MTP PROMPT_IDS SAMPLES WARMUP OUTPUT ENV CANDIDATE");
    const int samples = Positive(argv[4]);
    const int warmup = Positive(argv[5]);
    const int output = Positive(argv[6]);
    const std::string_view option(argv[7]);
    if (!option.starts_with("SGLANG_MLX_NATIVE_") || samples > 100 ||
        warmup > 4096 || output > 16384) throw std::runtime_error("unsupported benchmark options");
    std::ifstream config_file(std::string(argv[1]) + "/config.json");
    std::ostringstream config_text;
    config_text << config_file.rdbuf();
    MlxQwen38Config config{};
    char error[1024]{};
    if (!config_file || mlx_qwen38_config_from_json(config_text.str().c_str(),
          &config, error, sizeof(error)) != 0) throw std::runtime_error(error);
    std::ifstream prompt_file(argv[3]);
    if (!prompt_file) throw std::runtime_error("cannot open prompt IDs");
    std::vector<std::int32_t> prompt;
    std::int64_t id;
    while (prompt_file >> id) {
      if (id < 0 || id >= config.vocab_size) throw std::runtime_error("invalid prompt token");
      prompt.push_back(static_cast<std::int32_t>(id));
    }
    if (!prompt_file.eof() || prompt.empty() || prompt.size() + warmup + output > 131072)
      throw std::runtime_error("prompt is invalid or exceeds the target context budget");
    if (setenv("SGLANG_MLX_NATIVE_MTP_PROMPT_CACHE", "1", 1) != 0 ||
        setenv("SGLANG_MLX_NATIVE_APPEND_ONLY_ATTN_SNAPSHOT", "1", 1) != 0 ||
        setenv(argv[7], "0", 1) != 0) throw std::runtime_error("cannot configure benchmark");
    native::Engine engine(config, argv[1]);
    engine.load_mtp(argv[2]);
    auto started = Clock::now();
    const auto cold_first = engine.prefill(prompt.data(), static_cast<int>(prompt.size()), false);
    mlx::core::synchronize();
    std::cout << std::fixed << std::setprecision(9)
              << "cold_prefill_seconds=" << std::chrono::duration<double>(Clock::now() - started).count()
              << " prompt_tokens=" << prompt.size() << " cold_first=" << cold_first << std::endl;
    std::vector<std::int32_t> reference;
    std::uint64_t reference_target = 0, reference_mtp = 0;
    for (int sample = 0; sample < samples; ++sample) {
      const bool candidate = sample % 2 != 0;
      if (setenv(argv[7], candidate ? argv[8] : "0", 1) != 0)
        throw std::runtime_error("cannot switch benchmark arm");
      mlx::core::synchronize();
      engine.begin_request();
      started = Clock::now();
      auto token = engine.prefill(prompt.data(), static_cast<int>(prompt.size()), false);
      mlx::core::synchronize();
      const double prefill_seconds = std::chrono::duration<double>(Clock::now() - started).count();
      if (engine.last_prefill_cached_tokens() != static_cast<int>(prompt.size()) || token != cold_first)
        throw std::runtime_error("identical prompt reuse changed the first sampled token");
      std::vector<std::int32_t> ids{token};
      ids.reserve(static_cast<std::size_t>(warmup + output + 1));
      int buffered = 0;
      int refills = 0;
      int widths = 0;
      const auto decode = [&](bool measured) {
        const bool refill = buffered == 0;
        token = engine.decode(token);
        if (refill) {
          const int width = engine.last_spec_width();
          if (width <= 0) throw std::runtime_error("empty speculative refill");
          buffered = width - 1;
          if (measured) { ++refills; widths += width; }
        } else --buffered;
        ids.push_back(token);
      };
      for (int i = 0; i < warmup; ++i) decode(false);
      mlx::core::synchronize();
      started = Clock::now();
      for (int i = 0; i < output; ++i) decode(true);
      mlx::core::synchronize();
      const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
const auto target_state = engine.target_state_digest();
const auto mtp_state = engine.mtp_history_digest(false);
if (sample == 0) {
  reference = ids;
  reference_target = target_state;
  reference_mtp = mtp_state;
}
if (reference != ids || target_state != reference_target || mtp_state != reference_mtp)
  throw std::runtime_error("candidate changed sampled IDs or final target/MTP state");
      std::cout << "sample=" << sample << " candidate=" << candidate
                << " cached_tokens=" << engine.last_prefill_cached_tokens()
                << " prefill_seconds=" << prefill_seconds << " output_tokens=" << output
                << " seconds=" << seconds << " tokens_per_second=" << output / seconds
                << " refills=" << refills << " mean_width=" << static_cast<double>(widths) / refills
                << " digest=" << std::hex << Digest(ids, static_cast<std::size_t>(warmup + 1)) << std::dec
                << " target_state=" << std::hex << target_state << " mtp_state=" << mtp_state << std::dec
                << " exact_ids=" << (reference == ids)
                << " peak_bytes=" << mlx::core::get_peak_memory() << std::endl;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
