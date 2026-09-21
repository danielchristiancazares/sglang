#include <sglang/windows_ttft/launcher.hpp>

#include "../../../benchmark/native/test/test_support.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
using namespace sglang::windows_ttft;

void write(const fs::path &path, std::string_view bytes) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  output.close();
  REQUIRE(static_cast<bool>(output));
}

std::string read(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  REQUIRE(static_cast<bool>(input));
  return std::string(std::istreambuf_iterator<char>{input}, {});
}

struct Fixture {
  fs::path root;
  Options options;
  Fixture() {
    root = fs::current_path() / ("ttft-fixture-" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()));
    REQUIRE(fs::create_directory(root));
    options.command = "prepare";
    options.repo = root / "repo";
    options.target = root / "target checkpoint";
    options.draft = root / text_path("draft ü");
    options.runtime_tag = "test-only-sm120-cuda13.3-driver-fixture";
    write(options.repo / "python/sglang/source.cpp", "// first source version\n");
    write(options.repo / "python/pyproject.toml", "[project]\nname='fixture'\n");
    write(options.repo / "native/windows_ttft/launcher.cpp", "// policy fixture\n");
    write(options.repo / "scripts/windows/initialize_cuda_build_env.ps1", "initializer-fixture");
    write(options.repo / "scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1", "launcher-fixture");
    write(options.repo / ".venv/pyvenv.cfg", "runtime-fixture");
    write(options.repo / ".venv/Scripts/sglang.exe", "not-an-executable");
    write(options.repo / ".venv/Scripts/python.exe", "not-an-executable");
    write(options.repo / ".venv/Lib/site-packages/torch-1.dist-info/METADATA",
          "Name: torch\nVersion: fixture\n");
    for (const auto &checkpoint : {options.target, options.draft}) {
      write(checkpoint / "config.json", R"({"architectures":["fixture"]})");
      write(checkpoint / "weights.safetensors", "immutable-weight-fixture");
      write(checkpoint / "tokenizer.json", R"({"fixture":"unchanged"})");
    }
    write(options.target / "generation_config.json",
          R"({"temperature":0.6,"top_p":0.95,"nested":{"keep":"exact"}})");
    write(options.target / "selective-nvfp4-manifest.json", R"({"source":"immutable"})");
  }
  ~Fixture() {
    std::error_code ignored;
    fs::remove_all(root, ignored);
  }
};

std::string argument(const Plan &plan, const std::string &name) {
  const auto found = std::find(plan.arguments.begin(), plan.arguments.end(), name);
  REQUIRE(found != plan.arguments.end());
  REQUIRE(found + 1 != plan.arguments.end());
  return *(found + 1);
}

void PlanIsReadOnlyAndPreservesSelectedContract() {
  Fixture f;
  const auto plan = make_plan(f.options, {});
  REQUIRE(!fs::exists(plan.state_root));
  REQUIRE(argument(plan, "--model-path") == path_text(plan.view_root / "target"));
  REQUIRE(argument(plan, "--speculative-draft-model-path") == path_text(plan.view_root / "draft"));
  const std::vector<std::pair<std::string, std::string>> selected{
      {"--host", "127.0.0.1"}, {"--port", "30000"}, {"--served-model-name", "qwen3.8-27b"},
      {"--context-length", "200000"}, {"--max-total-tokens", "200000"},
      {"--max-mamba-cache-size", "5"}, {"--max-running-requests", "1"},
      {"--chunked-prefill-size", "4096"}, {"--page-size", "64"},
      {"--mamba-ssm-dtype", "float32"}, {"--mamba-radix-cache-strategy", "extra_buffer_lazy"},
      {"--speculative-algorithm", "DSPARK"}, {"--speculative-dspark-block-size", "7"},
      {"--speculative-draft-model-quantization", "fp8"},
      {"--kv-cache-dtype", "fp8_e4m3"}, {"--speculative-draft-kv-cache-dtype", "fp8_e4m3"},
      {"--prefill-attention-backend", "flashinfer"}, {"--decode-attention-backend", "trtllm_mha"},
      {"--reasoning-parser", "qwen3"}, {"--tool-call-parser", "qwen3_coder"},
      {"--cuda-graph-backend-decode", "full"}, {"--cuda-graph-max-bs-decode", "1"},
      {"--cuda-graph-backend-prefill", "disabled"}, {"--fp4-gemm-backend", "hybrid_marlin"},
      {"--sampling-defaults", "model"}, {"--flashinfer-autotune-skip-ops", "fp8_gemm"}};
  for (const auto &[key, value] : selected) REQUIRE(argument(plan, key) == value);
  for (const auto *flag : {"--language-model-only", "--incremental-streaming-output",
                           "--enable-linear-replayssm-spec", "--enable-cache-report"}) {
    REQUIRE(std::find(plan.arguments.begin(), plan.arguments.end(), flag) != plan.arguments.end());
  }
  for (const auto *flag : {"--random-seed", "--enable-torch-compile", "--speculative-adaptive"}) {
    REQUIRE(std::find(plan.arguments.begin(), plan.arguments.end(), flag) == plan.arguments.end());
  }
  REQUIRE(plan.environment.at("SGLANG_OPT_FLASHINFER_PAGE_ALIGNED_PREFILL") == "0");
  REQUIRE(plan.environment.at("SGLANG_FLASHINFER_AUTOTUNE_EXTEND") == "0");
  REQUIRE(plan.environment.at("SGLANG_FLASHINFER_AUTOTUNE_CACHE") == "1");
  REQUIRE(plan.environment.at("SGLANG_SIMULATE_ACC_LEN") == "-1");
  REQUIRE(make_plan(f.options, {}).fingerprint == plan.fingerprint);
}

void StableViewCopiesMetadataAndNeverChangesCheckpoint() {
  Fixture f;
  const auto original = read(f.options.target / "generation_config.json");
  const auto plan = make_plan(f.options, {});
  WorkspaceLock lock(plan.state_root);
  prepare_views(plan);
  const auto target = plan.view_root / "target";
  const auto draft = plan.view_root / "draft";
  auto generation = Json::parse(original);
  generation.as_object()["temperature"] = 1.0;
  REQUIRE(Json::parse(read(target / "generation_config.json")) == generation);
  REQUIRE(read(f.options.target / "generation_config.json") == original);
  REQUIRE(read(target / "selective-nvfp4-manifest.json") ==
          read(f.options.target / "selective-nvfp4-manifest.json"));
  REQUIRE(read(draft / "config.json") == read(f.options.draft / "config.json"));
  REQUIRE(fs::equivalent(target / "weights.safetensors", f.options.target / "weights.safetensors"));
  REQUIRE(fs::equivalent(draft / "weights.safetensors", f.options.draft / "weights.safetensors"));
  REQUIRE(!fs::equivalent(target / "config.json", f.options.target / "config.json"));
  const auto prior_identity = file_identity(target / "generation_config.json");
  const auto prior_time = fs::last_write_time(target / "generation_config.json");
  prepare_views(plan);
  REQUIRE(file_identity(target / "generation_config.json") == prior_identity);
  REQUIRE(fs::last_write_time(target / "generation_config.json") == prior_time);
  REQUIRE(make_plan(f.options, {}).fingerprint == plan.fingerprint);
}

void CorruptedViewAndChangedSourceFailWithoutRepair() {
  Fixture f;
  const auto plan = make_plan(f.options, {});
  WorkspaceLock lock(plan.state_root);
  prepare_views(plan);
  const auto file = plan.view_root / "target/config.json";
  write(file, R"({"corrupted":true})");
  REQUIRE(native_test::throws([&] { prepare_views(plan); }));
  REQUIRE(read(file) == R"({"corrupted":true})");
  REQUIRE(read(f.options.target / "config.json") == R"({"architectures":["fixture"]})");
  write(f.options.target / "config.json", R"({"new_source":true})");
  REQUIRE(native_test::throws([&] { prepare_views(plan); }));
  REQUIRE(read(file) == R"({"corrupted":true})");
}

void FingerprintsInvalidateInputsAndIsolateExtend() {
  Fixture f;
  const auto baseline = make_plan(f.options, {});
  f.options.autotune_extend = true;
  const auto extend = make_plan(f.options, {});
  REQUIRE(extend.fingerprint != baseline.fingerprint);
  REQUIRE(extend.view_root != baseline.view_root);
  REQUIRE(extend.environment.at("SGLANG_FLASHINFER_AUTOTUNE_EXTEND") == "1");
  f.options.autotune_extend = false;
  f.options.runtime_tag += "-new-driver";
  REQUIRE(make_plan(f.options, {}).fingerprint != baseline.fingerprint);
  f.options.runtime_tag = baseline.options.runtime_tag;
  write(f.options.repo / "python/sglang/source.cpp", "// second source version\n");
  const auto source_changed = make_plan(f.options, {});
  REQUIRE(source_changed.fingerprint != baseline.fingerprint);
  write(f.options.repo / ".venv/Lib/site-packages/torch-1.dist-info/METADATA", "new build metadata");
  const auto package_changed = make_plan(f.options, {});
  REQUIRE(package_changed.fingerprint != source_changed.fingerprint);
  // Replace a shard with equal bytes/size/mtime: file identity still changes.
  const auto weight = f.options.target / "weights.safetensors";
  const auto modified = fs::last_write_time(weight);
  fs::rename(weight, f.options.target / "old-weight");
  write(weight, "immutable-weight-fixture");
  fs::last_write_time(weight, modified);
  fs::remove(f.options.target / "old-weight");
  REQUIRE(make_plan(f.options, {}).fingerprint != package_changed.fingerprint);
}

void RuntimeEditableSourceMustBeCovered() {
  Fixture f;
  const auto dependency = fs::canonical(f.root) / "external";
  write(dependency / "kernel.cu", "// external source fixture\n");
  std::string url = "file://";
#ifdef _WIN32
  url += "/";
#endif
  url += path_text(dependency);
  write(f.options.repo / ".venv/Lib/site-packages/torch-1.dist-info/direct_url.json",
        Json(Json::object{{"url", url}, {"dir_info", Json::object{{"editable", true}}}}).dump());
  REQUIRE(native_test::throws([&] { static_cast<void>(make_plan(f.options, {})); }));
  f.options.dependency_roots.push_back(dependency);
  const auto baseline = make_plan(f.options, {});
  write(dependency / "kernel.cu", "// changed external source fixture\n");
  REQUIRE(make_plan(f.options, {}).fingerprint != baseline.fingerprint);
}

void LockIsExclusiveAndReleasedAfterFailure() {
  Fixture f;
  const auto plan = make_plan(f.options, {});
  {
    WorkspaceLock first(plan.state_root);
    REQUIRE(native_test::throws([&] { WorkspaceLock second(plan.state_root); }));
  }
  {
    WorkspaceLock next(plan.state_root);
    REQUIRE(fs::exists(plan.state_root / "workspace.lock"));
  }
  WorkspaceLock after_release(plan.state_root);
}

void InvalidOptionsAndOverridesCannotBecomeCandidates() {
  const std::vector<std::vector<std::string_view>> invalid{
      {}, {"serve"}, {"launch", "--target", "a", "--draft", "b", "--runtime-tag", "x"},
      {"plan", "--target", "a", "--draft", "b"},
      {"plan", "--target", "a", "--draft", "b", "--runtime-tag", "x", "--page-aligned-prefill"},
      {"plan", "--target", "a", "--draft", "b", "--runtime-tag", "x", "--autotune-extend",
       "--autotune-extend"}};
  for (const auto &args : invalid) {
    REQUIRE(native_test::throws([&] { static_cast<void>(parse_arguments(args)); }));
  }
  Options options;
  REQUIRE(native_test::throws([&] {
    static_cast<void>(selected_environment(options, {{"SGLANG_SIMULATE_ACC_LEN", "3"}}));
  }));
  REQUIRE(native_test::throws([&] {
    static_cast<void>(selected_environment(options, {{"SGLANG_UNKNOWN_EXPERIMENT", "1"}}));
  }));
  const auto selected = selected_environment(
      options, {{"SGLANG_OPT_FLASHINFER_PAGE_ALIGNED_PREFILL", "1"}, {"UNRELATED", "retain"}});
  REQUIRE(selected.at("SGLANG_OPT_FLASHINFER_PAGE_ALIGNED_PREFILL") == "0");
  REQUIRE(!selected.contains("UNRELATED"));
}

void UnicodeAndWindowsArgumentBoundaries() {
  for (const std::string text : {"", "ASCII", "C:\\path with spaces\\", "ü 😀 中", "a\"b"}) {
    REQUIRE(wide_to_utf8(utf8_to_wide(text)) == text);
    REQUIRE(path_text(text_path(text)) == text || text.find('\\') != std::string::npos);
  }
  REQUIRE(quote_windows_argument(L"") == L"\"\"");
  REQUIRE(quote_windows_argument(L"plain") == L"\"plain\"");
  REQUIRE(quote_windows_argument(L"C:\\space name\\") == L"\"C:\\space name\\\\\"");
  REQUIRE(quote_windows_argument(L"x\\\"z") == L"\"x\\\\\\\"z\"");
  REQUIRE(quote_windows_argument(L"$() & ^ %PATH%") == L"\"$() & ^ %PATH%\"");
  REQUIRE(native_test::throws([] { static_cast<void>(utf8_to_wide("\xC0\x80")); }));
  REQUIRE(native_test::throws([] {
    static_cast<void>(quote_windows_argument(std::wstring_view(L"a\0b", 3)));
  }));
}

void UnsafeStateAndPartialPublicationAreRejected() {
  Fixture f;
  auto options = f.options;
  options.target = options.repo; // cannot overlap writable view state
  REQUIRE(native_test::throws([&] { static_cast<void>(make_plan(options, {})); }));
  const auto plan = make_plan(f.options, {});
  WorkspaceLock lock(plan.state_root);
  fs::create_directory(plan.view_root);
  write(plan.view_root / "manifest.json", "{}");
  REQUIRE(native_test::throws([&] { prepare_views(plan); }));
  REQUIRE(read(plan.view_root / "manifest.json") == "{}");
  REQUIRE(!fs::exists(plan.view_root / "target"));
#ifndef _WIN32
  const auto linked = plan.state_root / "linked-state";
  fs::create_directory_symlink(f.options.target, linked);
  REQUIRE(native_test::throws([&] { WorkspaceLock rejected(linked); }));
#endif
}
} // namespace

int main() {
  const std::array tests{
      std::pair{"PlanIsReadOnlyAndPreservesSelectedContract", PlanIsReadOnlyAndPreservesSelectedContract},
      std::pair{"StableViewCopiesMetadataAndNeverChangesCheckpoint", StableViewCopiesMetadataAndNeverChangesCheckpoint},
      std::pair{"CorruptedViewAndChangedSourceFailWithoutRepair", CorruptedViewAndChangedSourceFailWithoutRepair},
      std::pair{"FingerprintsInvalidateInputsAndIsolateExtend", FingerprintsInvalidateInputsAndIsolateExtend},
      std::pair{"RuntimeEditableSourceMustBeCovered", RuntimeEditableSourceMustBeCovered},
      std::pair{"LockIsExclusiveAndReleasedAfterFailure", LockIsExclusiveAndReleasedAfterFailure},
      std::pair{"InvalidOptionsAndOverridesCannotBecomeCandidates", InvalidOptionsAndOverridesCannotBecomeCandidates},
      std::pair{"UnicodeAndWindowsArgumentBoundaries", UnicodeAndWindowsArgumentBoundaries},
      std::pair{"UnsafeStateAndPartialPublicationAreRejected", UnsafeStateAndPartialPublicationAreRejected}};
  return native_test::run("windows_ttft_test", tests);
}
