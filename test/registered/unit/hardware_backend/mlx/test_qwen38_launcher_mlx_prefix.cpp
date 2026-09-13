#include <fstream>

#define main qwen38_launcher_main
#include "../../../../../scripts/serve_qwen38_27b_q5_mlx.cpp"
#undef main

namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    char pattern[] = "/private/tmp/qwen38-launcher-XXXXXX";
    const char* path = ::mkdtemp(pattern);
    if (path == nullptr) throw std::runtime_error("mkdtemp failed");
    path_ = path;
  }
  ~TemporaryDirectory() { fs::remove_all(path_); }
  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

void Touch(const fs::path& path) {
  fs::create_directories(path.parent_path());
  std::ofstream file(path);
  if (!file) throw std::runtime_error("cannot create fixture");
}

void InstallMlx(const fs::path& prefix) {
  Touch(prefix / "include/mlx/array.h");
  Touch(prefix / "lib/libmlx.dylib");
}

void Require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

void RequireFailure(const fs::path& python, std::string_view expected) {
  try {
    (void)ResolveMlxPrefix(python);
  } catch (const std::runtime_error& error) {
    Require(std::string_view(error.what()).find(expected) != std::string_view::npos,
            "unexpected MLX discovery failure");
    return;
  }
  throw std::runtime_error("invalid MLX installation accepted");
}

}  // namespace

int main() {
  const auto previous = EnvironmentPath("MLX_PREFIX");
  try {
    ::unsetenv("MLX_PREFIX");
    TemporaryDirectory fixture;
    const auto environment = fixture.path() / "venv with spaces";
    const auto python = environment / "bin/python";
    RequireFailure(python, "cannot locate MLX");
    const auto prefix = environment / "lib/python3.11/site-packages/mlx";
    Touch(prefix / "include/mlx/array.h");
    RequireFailure(python, "cannot locate MLX");
    InstallMlx(prefix);
    Require(ResolveMlxPrefix(python) == fs::canonical(prefix),
            "did not select the venv's MLX package");

    const auto second = environment / "lib/python3.13/site-packages/mlx";
    InstallMlx(second);
    RequireFailure(python, "multiple MLX installations");
    const auto override_path = fixture.path() / "explicit MLX";
    InstallMlx(override_path);
    Require(::setenv("MLX_PREFIX", override_path.c_str(), 1) == 0,
            "setenv failed");
    Require(ResolveMlxPrefix(python) == fs::canonical(override_path),
            "explicit MLX_PREFIX was not honored");
    Options options;
    options.mlx_prefix = ResolveMlxPrefix(python);
    bool forwarded = false;
    bool ops_per_buffer = false;
    bool two_token_convolution = false;
    bool q5_mtp = false;
    for (const auto& [name, value] : ProductionEnvironment(options)) {
      if (name == "MLX_PREFIX") forwarded = value == options.mlx_prefix.string();
      if (name == "MLX_MAX_OPS_PER_BUFFER") ops_per_buffer = value == "200";
      if (name == "SGLANG_MLX_NATIVE_TWO_TOKEN_CAUSAL_CONV") {
        two_token_convolution = value == "1";
      }
      if (name == "SGLANG_MLX_MTP_DIR") q5_mtp = true;
    }
    Require(forwarded, "launcher did not forward MLX_PREFIX to native rebuilds");
    Require(ops_per_buffer, "Q5 launcher did not select the qualified op batch");
    Require(two_token_convolution,
            "Q5 launcher did not select two-token convolution");
    Require(q5_mtp, "Q5 launcher did not select MTP");
    Options q4_options;
    q4_options.profile = "q4";
    bool q4_ops_per_buffer = false;
    bool q4_two_token_convolution = false;
    bool q4_mtp = false;
    for (const auto& [name, value] : ProductionEnvironment(q4_options)) {
      if (name == "MLX_MAX_OPS_PER_BUFFER") {
        q4_ops_per_buffer = value == "100";
      }
      if (name == "SGLANG_MLX_NATIVE_TWO_TOKEN_CAUSAL_CONV") {
        q4_two_token_convolution = value == "1";
      }
      if (name == "SGLANG_MLX_MTP_DIR") q4_mtp = true;
    }
    Require(q4_ops_per_buffer, "Q4 launcher changed its op batch");
    Require(q4_two_token_convolution,
            "Q4 launcher did not select two-token convolution");
    Require(!q4_mtp, "Q4 launcher did not select target-only mode");
    q4_options.mtp_prompt_cache = true;
    bool q4_cached_mtp = false;
    bool q4_prompt_cache = false;
    for (const auto& [name, value] : ProductionEnvironment(q4_options)) {
      if (name == "SGLANG_MLX_MTP_DIR") q4_cached_mtp = true;
      if (name == "SGLANG_MLX_NATIVE_MTP_PROMPT_CACHE") {
        q4_prompt_cache = value == "1";
      }
    }
    Require(q4_cached_mtp, "Q4 prompt cache did not select MTP");
    Require(q4_prompt_cache, "Q4 prompt cache was not enabled");
    fs::remove(override_path / "lib/libmlx.dylib");
    RequireFailure(python, "MLX native library");
    if (previous) ::setenv("MLX_PREFIX", previous->c_str(), 1);
    else ::unsetenv("MLX_PREFIX");
    std::cout << "Qwen launcher MLX environment discovery passed\n";
    return 0;
  } catch (const std::exception& error) {
    if (previous) ::setenv("MLX_PREFIX", previous->c_str(), 1);
    else ::unsetenv("MLX_PREFIX");
    std::cerr << error.what() << '\n';
    return 1;
  }
}
