#include <charconv>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

constexpr std::string_view kQ5TargetSnapshot =
    "models--maglun--Qwen3.8-27B-MLX-Mixed-4.95bpw/snapshots/"
    "596b8067f7cf429007bb668874ffee7e917c8340";
constexpr std::string_view kQ4TargetSnapshot =
    "models--mlx-community--Qwen3.8-27B-4bit/snapshots/"
    "3e6447f082e89cc7f0bc6e5441afd38dfce760ff";
constexpr std::string_view kMtpSnapshot =
    "models--Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed/snapshots/"
    "123db8bcc7101455b00d9aad36c0e760c6e7de02";

struct Options {
  std::string profile = "q5";
  fs::path repository;
  fs::path python;
  fs::path mlx_prefix;
  fs::path model;
  fs::path mtp;
  std::optional<fs::path> websocket_frontend;
  std::string backend_port = "30001";
  std::string served_model_name;
  std::string host = "127.0.0.1";
  std::string port = "30000";
  bool print_config = false;
  bool mtp_prompt_cache = false;
};

[[noreturn]] void ThrowUsage(std::string_view message) {
  throw std::runtime_error(std::string(message) +
                           "; run with --help for usage");
}

void PrintUsage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n\n"
      << "Start the native Apple-Silicon Qwen3.8-27B SGLang lane.\n"
      << "The default checkpoint is the pinned mixed-Q5 model.\n"
      << "The launcher owns the measured 131,072-token profile and leaves "
         "Qwen's\n"
      << "reasoning depth uncapped so request-side reasoning_effort=xhigh "
         "can run.\n\n"
      << "Options:\n"
      << "  --profile q4|q5    Select checkpoint, API ID and kernel policy "
         "(default: q5)\n"
      << "  --mtp-prompt-cache  Reuse matching MTP prompt prefixes; for q4\n"
      << "                     this also enables MTP (default q4 is target-only)\n"
      << "  --repository PATH  SGLang checkout; defaults to the nearest "
         "parent\n"
      << "  --python PATH      Python executable; defaults to checkout .venv "
         "or\n"
      << "                     $HOME/sglang/.venv/bin/python\n"
      << "  --model PATH       Target checkpoint; defaults to the pinned "
         "snapshot\n"
      << "  --mtp PATH         MTP checkpoint; defaults to the pinned snapshot\n"
      << "  --served-model-name NAME  API model ID "
         "(default: qwen3.8-27b-<profile>)\n"
      << "  --host ADDRESS     Listen address (default: 127.0.0.1)\n"
      << "  --port PORT        Listen port (default: 30000)\n"
      << "  --websocket-frontend PATH  Native sglang_http executable; enables "
         "Responses WebSockets\n"
      << "  --backend-port PORT  Private HTTP port with WebSockets "
         "(default: 30001)\n"
      << "  --print-config     Validate and print the resolved launch, then exit\n"
      << "  --help             Show this help\n";
}

std::string RequireValue(int argc, char** argv, int& index,
                         std::string_view option) {
  if (index + 1 >= argc) {
    ThrowUsage(std::string(option) + " requires a value");
  }
  ++index;
  return argv[index];
}

std::optional<fs::path> EnvironmentPath(const char* name) {
  const char* const value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::nullopt;
  }
  return fs::path(value);
}

bool IsRepository(const fs::path& path) {
  std::error_code error;
  return fs::is_regular_file(path / "python/sglang/__init__.py", error) &&
         !error;
}

std::optional<fs::path> FindRepository(fs::path path) {
  std::error_code error;
  path = fs::absolute(path, error);
  if (error) {
    return std::nullopt;
  }
  for (;;) {
    if (IsRepository(path)) {
      return fs::weakly_canonical(path);
    }
    const fs::path parent = path.parent_path();
    if (parent == path) {
      return std::nullopt;
    }
    path = parent;
  }
}

fs::path CanonicalExistingDirectory(const fs::path& path,
                                    std::string_view description) {
  std::error_code error;
  if (!fs::is_directory(path, error) || error) {
    throw std::runtime_error(std::string(description) +
                             " is not a directory: " + path.string());
  }
  return fs::canonical(path);
}

fs::path CanonicalExecutable(const fs::path& path,
                             std::string_view description = "Python executable") {
  std::error_code error;
  const fs::path absolute = fs::absolute(path, error);
  if (error || !fs::is_regular_file(absolute, error) || error ||
      ::access(absolute.c_str(), X_OK) != 0) {
    throw std::runtime_error(std::string(description) + " is missing or not executable: " +
                             path.string());
  }
  // Preserve the virtual-environment entry path. Resolving its interpreter
  // symlink would bypass pyvenv.cfg discovery and can silently select the base
  // environment instead of SGLang's checked-in dependencies.
  return absolute.lexically_normal();
}

void RequireRegularFile(const fs::path& path, std::string_view description) {
  std::error_code error;
  if (!fs::is_regular_file(path, error) || error) {
    throw std::runtime_error(std::string(description) + " is missing: " +
                             path.string());
  }
}

fs::path ResolveMlxPrefix(const fs::path& python) {
  const auto validate = [](const fs::path& prefix) {
    RequireRegularFile(prefix / "include/mlx/array.h", "MLX C++ headers");
    RequireRegularFile(prefix / "lib/libmlx.dylib", "MLX native library");
    return fs::canonical(prefix);
  };
  if (const auto override_path = EnvironmentPath("MLX_PREFIX")) {
    return validate(*override_path);
  }

  // Preserve the venv executable path, as ParseOptions does for Python.
  // Its lib/python*/site-packages directory owns the matching MLX install.
  const fs::path library_root = python.parent_path().parent_path() / "lib";
  std::optional<fs::path> selected;
  std::error_code error;
  const fs::directory_iterator entries(library_root, error);
  if (!error) {
    for (const auto& entry : entries) {
      if (!entry.path().filename().string().starts_with("python")) {
        continue;
      }
      const fs::path prefix = entry.path() / "site-packages/mlx";
      if (!fs::is_regular_file(prefix / "include/mlx/array.h", error) ||
          !fs::is_regular_file(prefix / "lib/libmlx.dylib", error)) {
        continue;
      }
      if (selected.has_value()) {
        throw std::runtime_error(
            "multiple MLX installations below " + library_root.string() +
            "; set MLX_PREFIX to the selected Python environment's MLX directory");
      }
      selected = prefix;
    }
  }
  if (!selected.has_value()) {
    throw std::runtime_error(
        "cannot locate MLX headers and library for " + python.string() +
        "; set MLX_PREFIX to the installed MLX directory");
  }
  return validate(*selected);
}

fs::path DefaultSnapshot(std::string_view suffix) {
  const auto home = EnvironmentPath("HOME");
  if (!home.has_value()) {
    throw std::runtime_error(
        "HOME is unset; pass explicit --model and --mtp paths");
  }
  return *home / ".cache/huggingface/hub" / suffix;
}

void ValidatePort(std::string_view value) {
  int port = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), port);
  if (error != std::errc() || end != value.data() + value.size() || port < 1 ||
      port > 65535) {
    ThrowUsage("--port must be an integer from 1 through 65535");
  }
}

Options ParseOptions(int argc, char** argv) {
  std::optional<fs::path> repository;
  std::optional<fs::path> python;
  std::optional<fs::path> model;
  std::optional<fs::path> mtp;
  std::optional<std::string> served_model_name;
  Options options;

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      PrintUsage(std::cout, argv[0]);
      std::exit(EXIT_SUCCESS);
    }
    if (argument == "--print-config") {
      options.print_config = true;
    } else if (argument == "--mtp-prompt-cache") {
      options.mtp_prompt_cache = true;
    } else if (argument == "--profile") {
      options.profile = RequireValue(argc, argv, index, argument);
    } else if (argument == "--repository") {
      repository = RequireValue(argc, argv, index, argument);
    } else if (argument == "--python") {
      python = RequireValue(argc, argv, index, argument);
    } else if (argument == "--model") {
      model = RequireValue(argc, argv, index, argument);
    } else if (argument == "--mtp") {
      mtp = RequireValue(argc, argv, index, argument);
    } else if (argument == "--served-model-name") {
      served_model_name = RequireValue(argc, argv, index, argument);
    } else if (argument == "--host") {
      options.host = RequireValue(argc, argv, index, argument);
    } else if (argument == "--port") {
      options.port = RequireValue(argc, argv, index, argument);
    } else if (argument == "--websocket-frontend") {
      options.websocket_frontend = RequireValue(argc, argv, index, argument);
    } else if (argument == "--backend-port") {
      options.backend_port = RequireValue(argc, argv, index, argument);
    } else {
      ThrowUsage("unknown option: " + std::string(argument));
    }
  }

  if (options.profile != "q4" && options.profile != "q5") {
    ThrowUsage("--profile must be q4 or q5");
  }
  options.served_model_name =
      served_model_name.value_or("qwen3.8-27b-" + options.profile);
  if (options.served_model_name.empty()) {
    ThrowUsage("--served-model-name must not be empty");
  }
  if (options.host.empty()) {
    ThrowUsage("--host must not be empty");
  }
  ValidatePort(options.port);
  ValidatePort(options.backend_port);
  if (options.websocket_frontend) {
    options.websocket_frontend = CanonicalExecutable(*options.websocket_frontend,
                                                     "WebSocket frontend");
    if (options.port == options.backend_port) {
      ThrowUsage("--port and --backend-port must differ with WebSockets");
    }
  }

  if (!repository.has_value()) {
    repository = EnvironmentPath("SGLANG_QWEN38_REPOSITORY");
  }
  if (!repository.has_value()) {
    repository = FindRepository(fs::current_path());
  }
  if (!repository.has_value()) {
    throw std::runtime_error(
        "cannot find the SGLang checkout; run from it or pass --repository");
  }
  options.repository =
      CanonicalExistingDirectory(*repository, "SGLang repository");
  if (!IsRepository(options.repository)) {
    throw std::runtime_error("not an SGLang checkout: " +
                             options.repository.string());
  }

  if (!python.has_value()) {
    python = EnvironmentPath("SGLANG_QWEN38_PYTHON");
  }
  if (!python.has_value()) {
    const fs::path checkout_python = options.repository / ".venv/bin/python";
    std::error_code error;
    if (fs::is_regular_file(checkout_python, error) && !error) {
      python = checkout_python;
    }
  }
  if (!python.has_value()) {
    const auto home = EnvironmentPath("HOME");
    if (home.has_value()) {
      python = *home / "sglang/.venv/bin/python";
    }
  }
  if (!python.has_value()) {
    throw std::runtime_error(
        "cannot find the checked-in Python environment; pass --python");
  }
  options.python = CanonicalExecutable(*python);
  options.mlx_prefix = ResolveMlxPrefix(options.python);

  if (!model.has_value()) {
    model = EnvironmentPath("SGLANG_QWEN38_MODEL");
  }
  if (!model.has_value()) {
    model = DefaultSnapshot(options.profile == "q4" ? kQ4TargetSnapshot
                                                    : kQ5TargetSnapshot);
  }
  if (!mtp.has_value()) {
    mtp = EnvironmentPath("SGLANG_QWEN38_MTP");
  }
  if (!mtp.has_value()) {
    mtp = DefaultSnapshot(kMtpSnapshot);
  }
  options.model = CanonicalExistingDirectory(*model, "target checkpoint");
  options.mtp = CanonicalExistingDirectory(*mtp, "MTP checkpoint");

  RequireRegularFile(options.repository /
                         "python/sglang/srt/hardware_backend/mlx/native/"
                         "libqwen38_engine.dylib",
                     "native Qwen3.8 engine");
  RequireRegularFile(options.model / "config.json", "target config");
  RequireRegularFile(options.model / "model.safetensors.index.json",
                     "target tensor index");
  RequireRegularFile(options.model / "tokenizer.json", "target tokenizer");
  RequireRegularFile(options.mtp / "config.json", "MTP config");
  RequireRegularFile(options.mtp / "mtp.safetensors", "MTP tensors");
  RequireRegularFile(options.mtp / "mtplx_runtime.json", "MTP runtime manifest");
  return options;
}

std::vector<std::pair<std::string, std::string>> ProductionEnvironment(
    const Options& options) {
  const bool q4_mtp = options.profile == "q4" && options.mtp_prompt_cache;
  std::vector<std::pair<std::string, std::string>> environment{
      {"PYTHONPATH", (options.repository / "python").string()},
      {"MLX_PREFIX", options.mlx_prefix.string()},
      {"SGLANG_USE_MLX", "1"},
      {"SGLANG_USE_MLX_NATIVE_GRAPH", "1"},
      {"SGLANG_MLX_CLEAR_CACHE_STEPS", "0"},
      {"SGLANG_MLX_CACHE_LIMIT_GB", q4_mtp ? "1" : "0"},
      {"MLX_SDPA_BLOCKS", "64"},
      {"MLX_MAX_MB_PER_BUFFER", "256"},
      {"MLX_MAX_OPS_PER_BUFFER", options.profile == "q5" ? "200" : "100"},
      {"MLX_METAL_FAST_SYNCH", "1"},
      {"SGLANG_MLX_NATIVE_TARGET_ONLY_PREFILL_CHUNK_SIZE", "1024"},
      {"SGLANG_MLX_NATIVE_SAMPLING", "1"},
      {"SGLANG_MLX_NATIVE_SAMPLING_SEED", "42"},
      {"SGLANG_MLX_NATIVE_Q5_BATCH_ONE_QMV", "1"},
      {"SGLANG_MLX_NATIVE_Q5_BATCH_TWO_QMV", "1"},
      {"SGLANG_MLX_NATIVE_Q4_BATCH_ONE_QMV", "1"},
      {"SGLANG_MLX_NATIVE_Q4_BATCH_TWO_QMV", "1"},
      {"SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU", "1"},
      {"SGLANG_MLX_NATIVE_Q4_FUSED_RAW_PARAMS", "1"},
      {"SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU_BATCH_TWO", "1"},
      {"SGLANG_MLX_NATIVE_Q4_FUSED_SWIGLU_BATCH_TWO_SCALAR_INPUTS",
       options.profile == "q4" && !q4_mtp ? "1" : "0"},
      {"SGLANG_MLX_NATIVE_QUANTIZED_EMBEDDING", "1"},
      {"SGLANG_MLX_NATIVE_FIXED_PREFILL_ATTENTION", "1"},
      {"SGLANG_MLX_NATIVE_ATTN_CACHE_BITS", "8"},
      {"SGLANG_MLX_NATIVE_ATTN_CACHE_RESERVE", "0"},
      {"SGLANG_MLX_NATIVE_APPEND_ONLY_ATTN_SNAPSHOT", "1"},
      {"SGLANG_MLX_NATIVE_SERIALIZE_ATTN_CACHE_GROWTH", "1"},
      {"SGLANG_MLX_NATIVE_EVICT_Q4_RAW_PARAMS_AT_MTP_GROWTH", "1"},
      {"SGLANG_MLX_NATIVE_POST_GROWTH_MTP_PREFILL_CHUNK_SIZE", "256"},
      {"SGLANG_MLX_NATIVE_MTP_POST_NORM_SEED", "1"},
      {"SGLANG_MLX_NATIVE_MTP_BLOCK_SIZE", "2"},
      {"SGLANG_MLX_NATIVE_ASYNC_VERIFY", q4_mtp ? "1" : "0"},
      {"SGLANG_MLX_NATIVE_VERIFY_FUSED_NORMS", q4_mtp ? "1" : "0"},
      {"SGLANG_MLX_NATIVE_DFLASH_TAPE_COMMIT", q4_mtp ? "1" : "0"},
      {"SGLANG_MLX_NATIVE_Q8_SPLIT_VERIFY", q4_mtp ? "1" : "0"},
      {"SGLANG_MLX_NATIVE_TWO_TOKEN_CAUSAL_CONV", "1"},
      {"SGLANG_MLX_NATIVE_MTP_COMMITTED_HISTORY", "1"},
      {"SGLANG_MLX_NATIVE_MTP_PROMPT_CACHE", options.mtp_prompt_cache ? "1" : "0"},
  };
  if (options.profile == "q5" || options.mtp_prompt_cache) {
    environment.emplace_back("SGLANG_MLX_MTP_DIR", options.mtp.string());
  }
  return environment;
}

const std::vector<std::string>& ClearedEnvironment() {
  static const std::vector<std::string> names{
      "SGLANG_RUST_SERVER",
      "SGLANG_RUST_BUILD_MODE",
      "SGLANG_MLX_NATIVE_MAX_REASONING_TOKENS",
      "SGLANG_MLX_NATIVE_TRACE_STATE",
      "SGLANG_MLX_NATIVE_TRACE_SPEC",
      "SGLANG_MLX_NATIVE_TRACE_QMM",
      "SGLANG_MLX_NATIVE_TWO_TOKEN_CAUSAL_CONV",
      "SGLANG_MLX_NATIVE_Q4_PREPARED_INPUTS",
      "SGLANG_MLX_NATIVE_Q8_VECTOR_ATTENTION",
      "SGLANG_MLX_NATIVE_Q8_TILED_ATTENTION",
      "SGLANG_MLX_NATIVE_Q8_DEQUANT_SDPA",
      "SGLANG_MLX_NATIVE_Q8_SEGMENTED_MATMUL",
      "SGLANG_MLX_NATIVE_SMALL_BATCH_QMM",
      "SGLANG_MLX_NATIVE_M8_KSPLIT_QMM",
      "SGLANG_MLX_NATIVE_Q5_MULTIROW_QMV",
      "SGLANG_MLX_NATIVE_Q4_BATCH_THREE_QMV",
      "SGLANG_MLX_NATIVE_MTP_KV_ONLY_HISTORY",
      "SGLANG_MLX_MTP_DIR",
      "SGLANG_MLX_NATIVE_CHECK_MTP_HISTORY",
      "SGLANG_MLX_NATIVE_DFLASH_TAPE_COMMIT",
      "SGLANG_MLX_NATIVE_DFLASH_SELECTOR_TEMPERATURE",
      "SGLANG_MLX_NATIVE_DFLASH_MEAN_Q_THRESHOLD",
      "SGLANG_MLX_NATIVE_DSPARK_BYPASS_REFILLS",
      "SGLANG_MLX_NATIVE_DSPARK_CONFIDENCE_COST_RATIO",
      "SGLANG_MLX_NATIVE_DSPARK_VERIFY_DRAFT_TOKENS",
      "SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_PATH",
      "SGLANG_MLX_NATIVE_LINEAR_ATTN_OVERRIDE_SCOPE",
      "SGLANG_MLX_NATIVE_LINEAR_OUT_PROJ_OVERRIDE_PATH",
      "SGLANG_MLX_QUANTIZED_PREFILL_QUERY_TILE",
      "SGLANG_MLX_USE_CUSTOM_ROPE",
      "SGLANG_MLX_FUSE_SWIGLU",
  };
  return names;
}

std::vector<std::string> ServerArguments(const Options& options) {
  std::vector<std::string> arguments{
      options.python.string(),
      "-m",
      "sglang.launch_server",
      "--model-path",
      options.model.string(),
      "--served-model-name",
      options.served_model_name,
      "--language-model-only",
      "--context-length",
      "131072",
      "--max-total-tokens",
      "131072",
      "--max-running-requests",
      "1",
      "--max-mamba-cache-size",
      "5",
      "--chunked-prefill-size",
      "131072",
      "--max-prefill-tokens",
      "131072",
      "--page-size",
      "1",
      "--disable-radix-cache",
      "--mlx-enable-sampling",
      "--sampling-defaults",
      "model",
      "--random-seed",
      "42",
      "--reasoning-parser",
      "qwen3",
      "--tool-call-parser",
      "qwen3_coder",
      "--incremental-streaming-output",
      "--stream-interval",
      "4",
      "--scheduler-recv-interval",
      "4",
      "--sleep-on-idle",
      "--watchdog-timeout",
      "14400",
      "--cuda-graph-backend-decode",
      "disabled",
      "--cuda-graph-backend-prefill",
      "disabled",
      "--host",
      options.websocket_frontend ? "127.0.0.1" : options.host,
      "--port",
      options.websocket_frontend ? options.backend_port : options.port,
  };
  if (options.websocket_frontend) {
    std::vector<std::string> frontend{
        options.websocket_frontend->string(), "--host", options.host,
        "--port", options.port, "--upstream-port", options.backend_port, "--"};
    frontend.insert(frontend.end(), arguments.begin(), arguments.end());
    return frontend;
  }
  return arguments;
}

std::string ShellQuote(std::string_view value) {
  std::string quoted("'");
  for (const char character : value) {
    if (character == '\'') {
      quoted += "'\\''";
    } else {
      quoted += character;
    }
  }
  quoted += '\'';
  return quoted;
}

void PrintConfiguration(
    std::ostream& output,
    const std::vector<std::pair<std::string, std::string>>& environment,
    const std::vector<std::string>& arguments) {
  output << "cd " << ShellQuote(fs::current_path().string()) << "\nexec env";
  for (const std::string& name : ClearedEnvironment()) {
    output << " -u " << ShellQuote(name);
  }
  for (const auto& [name, value] : environment) {
    output << " " << name << "=" << ShellQuote(value);
  }
  for (const std::string& argument : arguments) {
    output << " " << ShellQuote(argument);
  }
  output << '\n';
}

void ApplyEnvironment(
    const std::vector<std::pair<std::string, std::string>>& environment) {
  for (const std::string& name : ClearedEnvironment()) {
    if (::unsetenv(name.c_str()) != 0) {
      throw std::runtime_error("cannot clear " + name + ": " +
                               std::strerror(errno));
    }
  }
  for (const auto& [name, value] : environment) {
    if (::setenv(name.c_str(), value.c_str(), 1) != 0) {
      throw std::runtime_error("cannot set " + name + ": " +
                               std::strerror(errno));
    }
  }
}

[[noreturn]] void Execute(std::vector<std::string> arguments) {
  std::vector<char*> raw_arguments;
  raw_arguments.reserve(arguments.size() + 1);
  for (std::string& argument : arguments) {
    raw_arguments.push_back(argument.data());
  }
  raw_arguments.push_back(nullptr);
  ::execv(arguments.front().c_str(), raw_arguments.data());
  throw std::runtime_error("cannot execute " + arguments.front() + ": " +
                           std::strerror(errno));
}

}  // namespace

int main(int argc, char** argv) {
#ifndef __APPLE__
  std::cerr << "This launcher supports Apple Silicon only.\n";
  return EXIT_FAILURE;
#else
  try {
    const Options options = ParseOptions(argc, argv);
    fs::current_path(options.repository);
    const auto environment = ProductionEnvironment(options);
    auto arguments = ServerArguments(options);
    PrintConfiguration(options.print_config ? std::cout : std::cerr,
                       environment, arguments);
    if (options.print_config) {
      return EXIT_SUCCESS;
    }
    ApplyEnvironment(environment);
    Execute(std::move(arguments));
  } catch (const std::exception& error) {
    std::cerr << "serve_qwen38_27b_q5_mlx: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
#endif
}
