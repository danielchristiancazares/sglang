#include <sglang/windows_ttft/launcher.hpp>
#include <sglang/benchmark/sha256.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace sglang::windows_ttft {
namespace {
using Object = Json::object;
using Array = Json::array;
constexpr std::uintmax_t kMetadataLimit = 64U * 1024U * 1024U;

std::string file_mtime(const fs::path &path) {
  // libc++ can use a 128-bit file-clock representation; Windows uses a
  // different epoch. Preserve native ticks losslessly without narrowing.
  auto ticks = fs::last_write_time(path).time_since_epoch().count();
  const bool negative = ticks < 0;
  std::string result;
  do {
    const auto digit = ticks % 10;
    result += static_cast<char>('0' + static_cast<int>(digit < 0 ? -digit : digit));
    ticks /= 10;
  } while (ticks != 0);
  if (negative) result += '-';
  std::reverse(result.begin(), result.end());
  return result;
}

std::string read_file(const fs::path &path) {
  if (fs::file_size(path) > kMetadataLimit) {
    throw std::runtime_error("metadata exceeds 64 MiB: " + path_text(path));
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path_text(path));
  std::string result(std::istreambuf_iterator<char>{input}, {});
  if (input.bad()) throw std::runtime_error("read failed: " + path_text(path));
  return result;
}

std::string hash_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot read " + path_text(path));
  benchmark::Sha256 hash;
  std::array<char, 65536> buffer{};
  while (input.read(buffer.data(), static_cast<std::streamsize>(buffer.size())) ||
         input.gcount() != 0) {
    hash.update(std::string_view(buffer.data(),
                                 static_cast<std::size_t>(input.gcount())));
  }
  if (!input.eof()) throw std::runtime_error("read failed: " + path_text(path));
  return hash.final_hex();
}

bool within(const fs::path &path, const fs::path &parent) {
  const auto relative = path.lexically_relative(parent);
  return !relative.empty() && !relative.is_absolute() &&
         *relative.begin() != "..";
}

bool source_file(const fs::path &path) {
  static const std::set<std::string> extensions{
      ".py", ".pyi", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh",
      ".cu", ".inc", ".in", ".cmake", ".toml", ".json", ".ps1", ".yaml",
      ".yml", ".jinja", ".jinja2", ".ptx"};
  return path.filename() == "CMakeLists.txt" ||
         extensions.contains(path.extension().string());
}

bool ignored_directory(const fs::path &path) {
  static const std::set<std::string> names{
      ".git", ".cache", "__pycache__", ".venv", "build", "dist", ".pytest_cache"};
  return names.contains(path.filename().string());
}

std::vector<fs::path> files_below(const fs::path &root, bool sources_only) {
  if (!fs::is_directory(root) || is_link(root)) {
    throw std::runtime_error("expected a non-linked directory: " + path_text(root));
  }
  std::vector<fs::path> files;
  for (fs::recursive_directory_iterator it(root), end; it != end; ++it) {
    const auto &entry = *it;
    if (entry.is_directory() && ignored_directory(entry.path())) {
      it.disable_recursion_pending();
      continue;
    }
    if (is_link(entry.path())) {
      throw std::runtime_error("linked input is not supported: " +
                               path_text(entry.path()));
    }
    if (entry.is_directory()) continue;
    if (!entry.is_regular_file()) {
      throw std::runtime_error("non-regular input: " + path_text(entry.path()));
    }
    if (!sources_only || source_file(entry.path())) files.push_back(entry.path());
    if (files.size() > 200000) throw std::runtime_error("input file limit exceeded");
  }
  std::sort(files.begin(), files.end());
  return files;
}

Json content_record(const fs::path &path, const fs::path &root) {
  return Object{{"path", path_text(path.lexically_relative(root))},
                {"size", fs::file_size(path)}, {"sha256", hash_file(path)}};
}

Json source_inventory(const fs::path &root) {
  Array entries;
  for (const auto &file : files_below(root, true)) {
    entries.push_back(content_record(file, root));
  }
  if (entries.empty()) throw std::runtime_error("no sources in " + path_text(root));
  return Object{{"root", path_text(root)}, {"files", entries.size()},
                {"sha256", benchmark::sha256_hex(Json(entries).dump())}};
}

Json checkpoint_inventory(const fs::path &root, bool target) {
  Array entries;
  std::size_t shards = 0;
  for (const auto &file : files_below(root, false)) {
    const auto relative = file.lexically_relative(root);
    if (file.extension() == ".safetensors") {
      ++shards;
      entries.emplace_back(Object{
          {"path", path_text(relative)}, {"size", fs::file_size(file)},
          {"file_id", file_identity(file)},
          {"mtime", file_mtime(file)}});
    } else {
      if (fs::file_size(file) > kMetadataLimit) {
        throw std::runtime_error("only safetensors may be hard-linked: " +
                                 path_text(file));
      }
      entries.push_back(content_record(file, root));
    }
  }
  if (shards == 0 || !fs::is_regular_file(root / "config.json")) {
    throw std::runtime_error("checkpoint requires config.json and safetensors");
  }
  const auto config = Json::parse(read_file(root / "config.json"));
  if (!config.is_object()) throw std::runtime_error("config.json must be an object");
  if (target) {
    const auto generation = Json::parse(read_file(root / "generation_config.json"));
    const auto provenance =
        Json::parse(read_file(root / "selective-nvfp4-manifest.json"));
    if (!generation.is_object() || !provenance.is_object()) {
      throw std::runtime_error("target generation/provenance must be JSON objects");
    }
  }
  return Object{{"root", path_text(root)}, {"files", std::move(entries)}};
}

fs::path local_file_url(const std::string &url) {
  if (!url.starts_with("file:///")) {
    throw std::runtime_error("editable dependency must have a local file URL");
  }
  std::string decoded;
  const auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (std::size_t i = 7; i < url.size(); ++i) {
    if (url[i] != '%') {
      decoded += url[i];
    } else {
      if (i + 2 >= url.size() || hex(url[i + 1]) < 0 || hex(url[i + 2]) < 0) {
        throw std::runtime_error("invalid editable file URL");
      }
      const auto value = static_cast<char>(hex(url[i + 1]) * 16 + hex(url[i + 2]));
      if (value == '\0') throw std::runtime_error("NUL in editable file URL");
      decoded += value;
      i += 2;
    }
  }
#ifdef _WIN32
  if (decoded.size() >= 3 && decoded[0] == '/' && decoded[2] == ':') {
    decoded.erase(0, 1);
  }
#endif
  return fs::canonical(text_path(decoded));
}

Json runtime_inventory(const Options &options) {
  const auto venv = options.repo / ".venv";
  const auto site = venv / "Lib/site-packages";
  Array entries;
  for (const auto *name : {"pyvenv.cfg", "Scripts/sglang.exe", "Scripts/python.exe"}) {
    const auto file = venv / name;
    if (!fs::is_regular_file(file) || is_link(file)) {
      throw std::runtime_error("missing/non-regular native runtime: " + path_text(file));
    }
    entries.push_back(content_record(file, venv));
  }
  std::vector<fs::path> metadata;
  if (!fs::is_directory(site) || is_link(site)) {
    throw std::runtime_error("native .venv/Lib/site-packages is unavailable");
  }
  for (const auto &entry : fs::directory_iterator(site)) {
    if (entry.path().extension() == ".dist-info" && entry.is_directory()) {
      const auto direct = entry.path() / "direct_url.json";
      if (fs::is_regular_file(direct)) {
        const auto url = Json::parse(read_file(direct));
        const auto *info = url.find("dir_info");
        const auto *editable = info == nullptr ? nullptr : info->find("editable");
        if (editable != nullptr && editable->as_bool()) {
          const auto root = local_file_url(url.at("url").as_string());
          const auto covered = [&](const fs::path &parent) { return within(root, parent); };
          if (!within(root, options.repo / "python") &&
              !std::any_of(options.dependency_roots.begin(),
                           options.dependency_roots.end(), covered)) {
            throw std::runtime_error(
                "editable source is not fingerprinted; add --dependency-root " +
                path_text(root));
          }
        }
      }
      auto files = files_below(entry.path(), false);
      metadata.insert(metadata.end(), files.begin(), files.end());
    } else if (entry.is_regular_file() &&
               (entry.path().extension() == ".pth" ||
                entry.path().extension() == ".egg-link" ||
                entry.path().filename().string().starts_with("__editable__"))) {
      if (is_link(entry.path())) throw std::runtime_error("linked runtime metadata");
      metadata.push_back(entry.path());
    }
  }
  if (metadata.empty()) throw std::runtime_error("installed package metadata is missing");
  std::sort(metadata.begin(), metadata.end());
  for (const auto &file : metadata) entries.push_back(content_record(file, venv));
  return Object{{"files", entries.size()},
                {"sha256", benchmark::sha256_hex(Json(entries).dump())}};
}

std::vector<std::string> serve_arguments(const fs::path &target, const fs::path &draft) {
  return {
      "serve", "--model-path", path_text(target),
      "--served-model-name", "qwen3.8-27b", "--host", "127.0.0.1", "--port", "30000",
      "--language-model-only", "--reasoning-parser", "qwen3",
      "--tool-call-parser", "qwen3_coder", "--attention-backend", "triton",
      "--prefill-attention-backend", "flashinfer", "--decode-attention-backend", "trtllm_mha",
      "--sampling-backend", "flashinfer", "--sampling-defaults", "model",
      "--fp8-gemm-backend", "flashinfer_cutlass", "--fp4-gemm-backend", "hybrid_marlin",
      "--kv-cache-dtype", "fp8_e4m3", "--page-size", "64", "--disable-custom-all-reduce",
      "--cuda-graph-backend-decode", "full", "--cuda-graph-max-bs-decode", "1",
      "--cuda-graph-backend-prefill", "disabled", "--mamba-radix-cache-strategy",
      "extra_buffer_lazy", "--mamba-ssm-dtype", "float32",
      "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton",
      "--linear-attn-verify-backend", "triton", "--max-mamba-cache-size", "5",
      "--context-length", "200000", "--mem-fraction-static", "0.98",
      "--max-total-tokens", "200000", "--chunked-prefill-size", "4096",
      "--triton-attention-num-kv-splits", "16", "--max-running-requests", "1",
      "--scheduler-recv-interval", "4", "--stream-interval", "4",
      "--incremental-streaming-output", "--flashinfer-autotune-skip-ops", "fp8_gemm",
      "--speculative-algorithm", "DSPARK", "--speculative-draft-model-path", path_text(draft),
      "--speculative-dspark-block-size", "7", "--speculative-num-steps", "1",
      "--speculative-eagle-topk", "1", "--speculative-draft-model-quantization", "fp8",
      "--speculative-draft-attention-backend", "triton", "--speculative-attention-mode", "decode",
      "--enable-linear-replayssm-spec", "--speculative-draft-kv-cache-dtype", "fp8_e4m3",
      "--enable-cache-report"};
}

void write_file(const fs::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  output.close();
  if (!output) throw std::runtime_error("write failed: " + path_text(path));
}

void validate_view(const Plan &plan, const fs::path &root) {
  if (is_link(root) || !fs::is_directory(root)) {
    throw std::runtime_error("sampling view is not a plain directory");
  }
  if (is_link(root / "manifest.json") ||
      Json::parse(read_file(root / "manifest.json")) != plan.manifest) {
    throw std::runtime_error("sampling view manifest mismatch; refusing reuse");
  }
  for (const auto *role : {"target", "draft"}) {
    const auto &inventory = plan.manifest.at(role);
    const auto source = text_path(inventory.at("root").as_string());
    const auto destination = root / role;
    const auto &entries = inventory.at("files").as_array();
    if (files_below(destination, false).size() != entries.size()) {
      throw std::runtime_error("sampling view file set changed");
    }
    for (const auto &entry : entries) {
      const auto relative = text_path(entry.at("path").as_string());
      const auto file = destination / relative;
      if (!fs::is_regular_file(file) || is_link(file)) {
        throw std::runtime_error("sampling view file is missing or linked");
      }
      if (entry.contains("file_id")) {
        if (!fs::equivalent(file, source / relative) ||
            file_identity(file) != entry.at("file_id").as_string() ||
            fs::file_size(file) != static_cast<std::uintmax_t>(entry.at("size").as_int()) ||
            file_mtime(file) != entry.at("mtime").as_string()) {
          throw std::runtime_error("sampling view tensor identity changed");
        }
      } else if (std::string_view(role) == "target" && relative == "generation_config.json") {
        auto expected = Json::parse(read_file(source / relative));
        expected.as_object()["temperature"] = 1.0;
        if (read_file(file) != expected.dump() + "\n") {
          throw std::runtime_error("sampling view temperature/configuration changed");
        }
      } else if (hash_file(file) != entry.at("sha256").as_string()) {
        throw std::runtime_error("sampling view metadata changed");
      }
      if (!entry.contains("file_id") && fs::equivalent(file, source / relative)) {
        throw std::runtime_error("sampling view metadata must be an independent copy");
      }
    }
  }
}
} // namespace

std::string path_text(const fs::path &path) {
  const auto text = path.generic_u8string();
  return std::string(text.begin(), text.end());
}

fs::path text_path(std::string_view text) {
  static_cast<void>(benchmark::utf8_code_point_count(text));
  if (text.find('\0') != std::string_view::npos) throw std::invalid_argument("NUL in path");
  return fs::path(std::u8string(text.begin(), text.end()));
}

std::string_view help() noexcept {
  return R"(usage: serve_windows_ttft plan|prepare|serve --target PATH --draft PATH
       --runtime-tag TAG [--repo PATH] [--dependency-root PATH ...]
       [--autotune-extend]

Opt-in native-Windows DSpark-v2 TTFT lane. Production defaults are unchanged.
  plan      read-only fingerprint, resolved arguments and environment (JSON)
  prepare   atomically create/revalidate stable sampling views; no process launch
  serve     prepare, then run the existing .venv SGLang executable in foreground

Target requires the selected attention-NVFP4 checkpoint and provenance manifest.
Draft must be the matching DSpark-v2 checkpoint. Tensor shards are hard-linked;
the checkout and both checkpoints must reside on the same filesystem/volume.
Only the target view's temperature is changed, to 1.0; source files stay intact.
Views persist under REPO/.sglang-ttft for autotune reuse across restarts.
Control and large-EXTEND have distinct identities. Raw page-aligned prefill is
disabled: the September 20 Windows screen rejected its short-prompt regression.

--runtime-tag is REQUIRED: identify the GPU, driver, CUDA/MSVC and runtime
binary build. Change it after any uncaptured binary/toolchain change.
Source files and installed metadata are hashed; tensor identity/size/mtime
are recorded, not full tensor checksums. Checkpoints must remain immutable.
Supply --dependency-root for every external editable source or local patch.

Before serve, run the existing initialize_cuda_build_env.ps1 -MaxJobs 2,
verify the editable install, and establish exclusive GPU/process ownership.
No dependency repair or GPU-owner discovery is performed. Port 30000 must
be free. Unknown inherited SGLANG_* overrides are rejected. Ctrl+C requests
shutdown; after 30 seconds only this launch's owned job is terminated.
)";
}

Options parse_arguments(std::span<const std::string_view> args) {
  if (args.empty()) throw std::invalid_argument("explicit plan, prepare or serve required");
  Options options;
  options.command = args[0];
  if (options.command != "plan" && options.command != "prepare" && options.command != "serve") {
    throw std::invalid_argument("unknown command");
  }
  std::set<std::string_view> seen;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const auto key = args[i];
    if (key != "--dependency-root" && !seen.insert(key).second) {
      throw std::invalid_argument("duplicate option: " + std::string(key));
    }
    if (key == "--autotune-extend") options.autotune_extend = true;
    else {
      if (i + 1 == args.size() || args[i + 1].empty() || args[i + 1].starts_with("--")) {
        throw std::invalid_argument("missing value: " + std::string(key));
      }
      const auto value = args[++i];
      if (key == "--repo") options.repo = text_path(value);
      else if (key == "--target") options.target = text_path(value);
      else if (key == "--draft") options.draft = text_path(value);
      else if (key == "--runtime-tag") options.runtime_tag = value;
      else if (key == "--dependency-root") options.dependency_roots.push_back(text_path(value));
      else throw std::invalid_argument("unknown option: " + std::string(key));
    }
  }
  if (options.target.empty() || options.draft.empty() || options.runtime_tag.empty()) {
    throw std::invalid_argument("--target, --draft and --runtime-tag are required");
  }
  return options;
}

Environment selected_environment(const Options &o, const Environment &inherited) {
  Environment owned{
      {"SGLANG_OPT_SPARSE_TOP_P_RENORM", "1"},
      {"SGLANG_DSPARK_TRUNCATED_DRAFT_SAMPLING", "0"},
      {"SGLANG_DSPARK_STATIC_GRAPH_KV_COMMIT", "1"},
      {"SGLANG_FLASHINFER_AUTOTUNE_EXTEND", o.autotune_extend ? "1" : "0"},
      {"SGLANG_OPT_FLASHINFER_PAGE_ALIGNED_PREFILL", "0"},
      {"SGLANG_FLASHINFER_WORKSPACE_SIZE", "134217728"},
      {"SGLANG_FLASHINFER_AUTOTUNE_CACHE", "1"},
      {"SGLANG_SIMULATE_ACC_LEN", "-1"}};
  const std::set<std::string> retained{
      "CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "MAX_JOBS", "DISTUTILS_USE_SDK",
      "TORCH_CUDA_ARCH_LIST", "FLASHINFER_CUDA_ARCH_LIST", "SGLANG_CACHE_DIR"};
  for (const auto &[key, value] : inherited) {
    if (key == "SGLANG_SIMULATE_ACC_LEN" && value != "-1" && !value.empty()) {
      throw std::runtime_error("inherited accepted-length simulation must be disabled");
    }
    if (key.starts_with("SGLANG_") && !owned.contains(key) && !retained.contains(key)) {
      throw std::runtime_error("unsupported inherited override: " + key);
    }
    if (retained.contains(key)) owned.emplace(key, value);
  }
  return owned;
}

Plan make_plan(Options o, const Environment &inherited) {
  if (o.runtime_tag.empty() || o.runtime_tag.size() > 256 ||
      o.runtime_tag.find_first_of("\r\n") != std::string::npos) {
    throw std::invalid_argument("a nonempty, single-line runtime tag is required");
  }
  if (o.target.empty() || o.draft.empty()) throw std::invalid_argument("checkpoint paths required");
  o.repo = fs::canonical(o.repo);
  o.target = fs::canonical(o.target);
  o.draft = fs::canonical(o.draft);
  for (auto &root : o.dependency_roots) root = fs::canonical(root);
  std::sort(o.dependency_roots.begin(), o.dependency_roots.end());
  o.dependency_roots.erase(std::unique(o.dependency_roots.begin(), o.dependency_roots.end()),
                           o.dependency_roots.end());
  const auto state = o.repo / ".sglang-ttft";
  if (is_link(state) || within(o.target, state) || within(o.draft, state) ||
      within(state, o.target) || within(state, o.draft)) {
    throw std::runtime_error("sampling state must be separate from checkpoints");
  }
  Plan plan;
  plan.options = o;
  plan.state_root = state;
  plan.executable = o.repo / ".venv/Scripts/sglang.exe";
  plan.environment = selected_environment(o, inherited);
  Array sources;
  sources.push_back(source_inventory(o.repo / "python/sglang"));
  sources.push_back(source_inventory(o.repo / "native/windows_ttft"));
  for (const auto *relative : {"native/src", "native/include"}) {
    if (fs::is_directory(o.repo / relative)) sources.push_back(source_inventory(o.repo / relative));
  }
  for (const auto &root : o.dependency_roots) sources.push_back(source_inventory(root));
  for (const auto *relative :
       {"scripts/windows/initialize_cuda_build_env.ps1",
        "scripts/windows/serve_qwen38_27b_nvfp4_5090.ps1", "python/pyproject.toml"}) {
    sources.push_back(content_record(o.repo / relative, o.repo));
  }
  Array arguments;
  for (const auto &arg : serve_arguments("$TARGET_VIEW", "$DRAFT_VIEW")) arguments.emplace_back(arg);
  Object environment;
  for (const auto &[key, value] : plan.environment) environment.emplace(key, value);
  plan.manifest = Object{
      {"schema", "native-windows-ttft-view-v1"},
      {"runtime_tag", o.runtime_tag}, {"target_temperature", 1.0},
      {"page_aligned_prefill", false}, {"autotune_extend", o.autotune_extend},
      {"target", checkpoint_inventory(o.target, true)},
      {"draft", checkpoint_inventory(o.draft, false)},
      {"sources", std::move(sources)}, {"runtime_metadata", runtime_inventory(o)},
      {"environment", std::move(environment)}, {"arguments", std::move(arguments)}};
  plan.fingerprint = benchmark::sha256_hex(plan.manifest.dump());
  plan.view_root = state / plan.fingerprint;
  plan.arguments = serve_arguments(plan.view_root / "target", plan.view_root / "draft");
  return plan;
}

Json receipt(const Plan &plan) {
  Array args;
  Object environment;
  for (const auto &arg : plan.arguments) args.emplace_back(arg);
  for (const auto &[key, value] : plan.environment) environment.emplace(key, value);
  return Object{
      {"schema", "native-windows-ttft-launch-v1"}, {"fingerprint", plan.fingerprint},
      {"command", plan.options.command}, {"executable", path_text(plan.executable)},
      {"view_root", path_text(plan.view_root)}, {"arguments", std::move(args)},
      {"environment", std::move(environment)}, {"runtime_tag", plan.options.runtime_tag},
      {"page_aligned_prefill", false},
      {"autotune_extend", plan.options.autotune_extend},
      {"qualification", "opt-in; Windows runtime/performance qualification required"}};
}

void prepare_views(const Plan &plan) {
  // Re-read before reuse as well as after copying: a plan is not a lease on inputs.
  if (make_plan(plan.options, plan.environment).fingerprint != plan.fingerprint) {
    throw std::runtime_error("inputs changed after plan; no view published");
  }
  if (fs::exists(plan.view_root) || is_link(plan.view_root)) {
    validate_view(plan, plan.view_root);
    return;
  }
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto stage = plan.state_root / (".staging-" + std::to_string(nonce));
  if (!fs::create_directory(stage)) throw std::runtime_error("staging directory already exists");
  try {
    for (const auto *role : {"target", "draft"}) {
      const auto &inventory = plan.manifest.at(role);
      const auto source = text_path(inventory.at("root").as_string());
      for (const auto &entry : inventory.at("files").as_array()) {
        const auto relative = text_path(entry.at("path").as_string());
        const auto destination = stage / role / relative;
        fs::create_directories(destination.parent_path());
        if (entry.contains("file_id")) {
          std::error_code error;
          fs::create_hard_link(source / relative, destination, error);
          if (error) throw std::runtime_error(
              "tensor hard-link failed (same volume required): " + error.message());
        } else if (std::string_view(role) == "target" && relative == "generation_config.json") {
          auto generation = Json::parse(read_file(source / relative));
          generation.as_object()["temperature"] = 1.0;
          write_file(destination, generation.dump() + "\n");
        } else {
          fs::copy_file(source / relative, destination, fs::copy_options::none);
        }
      }
    }
    write_file(stage / "manifest.json", plan.manifest.dump() + "\n");
    if (make_plan(plan.options, plan.environment).fingerprint != plan.fingerprint) {
      throw std::runtime_error("inputs changed during preparation; no view published");
    }
    validate_view(plan, stage);
    fs::rename(stage, plan.view_root);
  } catch (...) {
    // Only remove the private directory this invocation just created.
    std::error_code cleanup_error;
    fs::remove_all(stage, cleanup_error);
    throw;
  }
}
} // namespace sglang::windows_ttft
