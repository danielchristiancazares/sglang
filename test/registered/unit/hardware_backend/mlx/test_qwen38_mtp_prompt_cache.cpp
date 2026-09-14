#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/random.h"
#include "qwen38_engine.h"

namespace {
using Engine = sglang::mlx_qwen38::Engine;
using Clock = std::chrono::steady_clock;

struct Result {
  std::uint64_t history;
  std::uint64_t target;
  std::vector<std::int32_t> tokens;
  std::uint64_t final_history = 0;
  std::uint64_t final_target = 0;
};

Result Finish(Engine& engine, std::int32_t token) {
  Result result{
      engine.mtp_history_digest(false), engine.target_state_digest(), {token}};
  for (int i = 1; i < 64; ++i) {
    token = engine.decode(token);
    result.tokens.push_back(token);
  }
  result.final_history = engine.mtp_history_digest(false);
  result.final_target = engine.target_state_digest();
  return result;
}

void CheckEqual(const Result& expected, const Result& actual,
                const char* scenario) {
  if (expected.history != actual.history || expected.target != actual.target ||
      expected.tokens != actual.tokens ||
      expected.final_history != actual.final_history ||
      expected.final_target != actual.final_target) {
    std::cerr << scenario << " expected_mtp=" << std::hex << expected.history
              << " actual_mtp=" << actual.history
              << " expected_target=" << expected.target
              << " actual_target=" << actual.target << std::dec
              << " equal_tokens=" << (expected.tokens == actual.tokens) << '\n';
    throw std::runtime_error(std::string(scenario) +
                             " changed state or tokens");
  }
  std::cout << scenario << " exact_mtp_state=" << std::hex << actual.history
            << " exact_target_state=" << actual.target << std::dec
            << " exact_tokens=" << actual.tokens.size() << '\n';
}

void Disturb(Engine& engine, std::int32_t token) {
  // Advance speculative cycles, leaving the
  // target recurrence and draft KV beyond their saved prompt boundary.
  for (int i = 0; i < 32; ++i) token = engine.decode(token);
}

std::int32_t Prefill(Engine& engine, const std::vector<std::int32_t>& tokens) {
  return engine.prefill(tokens.data(), static_cast<int>(tokens.size()), false);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: test_qwen38_mtp_prompt_cache MODEL_DIR MTP_DIR\n";
    return 2;
  }
  try {
    std::ifstream file(std::string(argv[1]) + "/config.json");
    std::ostringstream json;
    json << file.rdbuf();
    MlxQwen38Config config{};
    char error[1024]{};
    if (!file || mlx_qwen38_config_from_json(json.str().c_str(), &config, error,
                                             sizeof(error)) != 0) {
      throw std::runtime_error(std::string("model config: ") + error);
    }
    if (setenv("SGLANG_MLX_NATIVE_ASYNC_VERIFY", "0", 1) != 0)
      throw std::runtime_error("cannot select synchronous reference");
    Engine engine(config, argv[1]);
    engine.load_mtp(argv[2]);
    std::vector<std::int32_t> prefix(128), suffix(64);
    for (int i = 0; i < 128; ++i) prefix[i] = 1000 + (i * 17) % 1000;
    for (int i = 0; i < 64; ++i) suffix[i] = 1100 + (i * 19) % 1000;
    auto complete = prefix;
    complete.insert(complete.end(), suffix.begin(), suffix.end());

    // The reference performs the same two prefill chunks consecutively.
    // Reseed at the request boundary just as an uncached request would.
    Prefill(engine, prefix);
    mlx::core::random::seed(42);
    const auto expected = Finish(engine, Prefill(engine, suffix));

    if (setenv("SGLANG_MLX_NATIVE_ASYNC_VERIFY", "1", 1) != 0)
      throw std::runtime_error("cannot select asynchronous verifier");
    engine.reset();
    Disturb(engine, Prefill(engine, prefix));
    engine.begin_request();
    const auto start = Clock::now();
    auto next = Prefill(engine, complete);
    if (engine.last_prefill_cached_tokens() != 128) {
      throw std::runtime_error("matching MTP prompt was not reused");
    }
    std::cout << "cached_suffix_seconds="
              << std::chrono::duration<double>(Clock::now() - start).count()
              << '\n';
    CheckEqual(expected, Finish(engine, next), "after_speculation");

    // A second continuation must use the replacement prompt snapshot, not
    // the earlier 128-token boundary or the verifier's temporary snapshot.
    complete.insert(complete.end(), suffix.begin(), suffix.end());
    engine.begin_request();
    next = Prefill(engine, complete);
    if (engine.last_prefill_cached_tokens() != 192) {
      throw std::runtime_error("replacement prompt was not reused");
    }
    const auto second = Finish(engine, next);
    if (setenv("SGLANG_MLX_NATIVE_ASYNC_VERIFY", "0", 1) != 0)
      throw std::runtime_error("cannot select synchronous reference");
    engine.reset();
    Prefill(engine, prefix);
    Prefill(engine, suffix);
    mlx::core::random::seed(42);
    CheckEqual(Finish(engine, Prefill(engine, suffix)), second,
               "second_continuation");

    // A different prefix must discard both target and MTP saved state.
    ++complete[0];
    engine.begin_request();
    next = Prefill(engine, complete);
    if (engine.last_prefill_cached_tokens() != 0) {
      throw std::runtime_error("mismatching prompt reused state");
    }
    const auto mismatch = Finish(engine, next);
    engine.reset();
    CheckEqual(Finish(engine, Prefill(engine, complete)), mismatch,
               "mismatching_prefix");

    // An identical request recomputes logits from the saved last hidden
    // row. Repeated restores must preserve target and MTP state as well
    // as reseeding the same sampled output after speculative work.
    engine.begin_request();
    next = Prefill(engine, complete);
    if (engine.last_prefill_cached_tokens() != static_cast<int>(complete.size())) {
      throw std::runtime_error("identical prompt did not reuse its full prefix");
    }
    CheckEqual(mismatch, Finish(engine, next), "empty_suffix");
    for (int repeat = 0; repeat < 3; ++repeat) {
      if (setenv("SGLANG_MLX_NATIVE_ASYNC_VERIFY", repeat % 2 == 0 ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot alternate verifier submissions");
      engine.begin_request();
      next = Prefill(engine, complete);
      if (engine.last_prefill_cached_tokens() != static_cast<int>(complete.size())) {
        throw std::runtime_error("repeated identical prompt lost its snapshot");
      }
      CheckEqual(mismatch, Finish(engine, next), "repeated_empty_suffix");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
