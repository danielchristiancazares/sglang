// SPDX-License-Identifier: Apache-2.0
// Native adapters for installed Transformers and FlashInfer. No Python source
// is authored, generated, copied, patched, or evaluated by this runtime.
#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <filesystem>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {
class InputFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
class CheckpointFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class NonEmptyString final {
    std::string value_;
public:
    explicit NonEmptyString(std::string value) : value_(std::move(value)) {
        if (value_.empty()) {
            throw InputFailure("A nonempty string is required.");
        }
    }
    const std::string& text() const noexcept { return value_; }
};

class ModelDirectory final {
    std::filesystem::path path_;
public:
    explicit ModelDirectory(std::filesystem::path path) : path_(std::filesystem::canonical(path)) {
        for (const auto* name : {"config.json", "generation_config.json", "tokenizer.json",
                               "model.safetensors.index.json"}) {
            if (!std::filesystem::is_regular_file(path_ / name)) {
                throw CheckpointFailure("The checkpoint is missing " + std::string(name));
            }
        }
    }
    std::string native_path() const { return path_.string(); }
};

class LibraryScope final {
    py::object scope_;
public:
    explicit LibraryScope(py::object scope) : scope_(std::move(scope)) {
        scope_.attr("__enter__")();
    }
    LibraryScope(const LibraryScope&) = delete;
    LibraryScope& operator=(const LibraryScope&) = delete;
    ~LibraryScope() {
        try {
            scope_.attr("__exit__")(py::none(), py::none(), py::none());
        } catch (py::error_already_set& error) {
            error.discard_as_unraisable("native DiffusionGemma library scope");
        }
    }
};

// Owns mapped, immutable safetensors shards. Lookup failures are reported before
// a weight is admitted to a native expert bank or a Transformers module.
class CheckpointReader final {
    py::dict index_;
    std::map<std::string, py::object> shards_;
public:
    CheckpointReader(const ModelDirectory& directory, py::dict index) : index_(std::move(index)) {
        std::set<std::string> files;
        for (const auto& item : index_) {
            files.insert(py::cast<std::string>(item.second));
        }
        for (const auto& file : files) {
            const auto path = std::filesystem::path(directory.native_path()) / file;
            if (path.parent_path() != std::filesystem::path(directory.native_path())) {
                throw CheckpointFailure("A checkpoint shard path escapes the model directory.");
            }
            shards_.emplace(file, py::module_::import("safetensors").attr("safe_open")(
                path.string(), "framework"_a="pt", "device"_a="cpu"));
        }
    }

    py::object tensor(const NonEmptyString& name) const {
        if (!index_.contains(py::str(name.text()))) {
            throw CheckpointFailure("Missing checkpoint tensor: " + name.text());
        }
        return shards_.at(index_[py::str(name.text())].cast<std::string>()).attr("get_tensor")(name.text());
    }

    std::vector<NonEmptyString> names() const {
        std::vector<NonEmptyString> result;
        result.reserve(index_.size());
        for (const auto& item : index_) {
            result.emplace_back(py::cast<std::string>(item.first));
        }
        return result;
    }
};

// The callback is a native C++ implementation of the installed nn.Module ABI.
// Packed checkpoint bytes stay on the GPU. The native owner groups selected
// rows and invokes the installed NVFP4 GEMM kernels for each participating
// expert. This eager compatibility path does not use fused-MoE JIT kernels.
class Nvfp4Experts final {
    py::module_ torch_;
    py::object weights_13_;
    py::object weights_2_;
    py::list scales_;
    py::object kernel_;
    py::object quantize_;
    py::object gelu_;

    static py::object stack_projection(const CheckpointReader& reader, int layer,
                                       const char* projection, const char* suffix,
                                       const py::module_& torch) {
        py::list experts;
        for (int expert = 0; expert < 128; ++expert) {
            experts.append(reader.tensor(NonEmptyString("model.decoder.layers." +
                std::to_string(layer) + ".experts." + std::to_string(expert) + "." + projection + "." + suffix)));
        }
        return torch.attr("stack")(experts).attr("to")("cuda");
    }

    static py::object swizzle(py::object scale, const py::module_& torch) {
        const auto sizes = scale.attr("shape").cast<std::vector<int>>();
        const int rows = (sizes.at(1) + 127) / 128 * 128;
        const int columns = (sizes.at(2) + 3) / 4 * 4;
        auto padded = torch.attr("zeros")(py::make_tuple(sizes.at(0), rows, columns),
            "dtype"_a=torch.attr("float8_e4m3fn"), "device"_a="cuda");
        padded.attr("narrow")(1, 0, sizes.at(1)).attr("narrow")(2, 0, sizes.at(2)).attr("copy_")(scale);
        return padded.attr("reshape")(sizes.at(0), rows / 128, 4, 32, columns / 4, 4)
            .attr("permute")(0, 1, 4, 3, 2, 5).attr("contiguous")()
            .attr("reshape")(sizes.at(0), rows, columns).attr("view")(torch.attr("int32"));
    }

    Nvfp4Experts(py::module_ torch, py::object weights13, py::object weights2, py::list scales)
        : torch_(std::move(torch)), weights_13_(std::move(weights13)), weights_2_(std::move(weights2)),
          scales_(std::move(scales)), kernel_(py::module_::import("flashinfer").attr("mm_fp4")),
          quantize_(py::module_::import("flashinfer").attr("fp4_quantize")),
          gelu_(py::module_::import("torch.nn.functional").attr("gelu")) {}

public:
    void verify(const CheckpointReader& reader) const {
        LibraryScope inference(torch_.attr("inference_mode")());
        torch_.attr("manual_seed")(42);
        auto hidden = torch_.attr("randn")(py::make_tuple(16, 2816),
            "dtype"_a=torch_.attr("bfloat16"), "device"_a="cuda");
        std::vector<std::vector<int>> routes;
        std::set<int> participating;
        for (int token = 0; token < 16; ++token) {
            std::vector<int> row;
            for (int slot = 0; slot < 8; ++slot) {
                const int expert = (token * 7 + slot * 3) % 128;
                row.push_back(expert);
                participating.insert(expert);
            }
            routes.push_back(std::move(row));
        }
        auto indices = torch_.attr("tensor")(routes, "dtype"_a=torch_.attr("int64"), "device"_a="cuda");
        auto probabilities = torch_.attr("randn")(py::make_tuple(16, 8),
            "dtype"_a=torch_.attr("float32"), "device"_a="cuda").attr("softmax")(1);
        std::cout << "Running isolated native NVFP4 expert routing..." << std::endl;
        auto actual = forward(hidden, indices, probabilities).attr("float")();
        auto reference = torch_.attr("zeros_like")(actual);
        const py::object quantize = py::module_::import("flashinfer").attr("fp4_quantize");
        const py::object gemm = py::module_::import("flashinfer").attr("mm_fp4");
        auto functional = py::module_::import("torch.nn.functional");
        auto quantized_input = quantize(hidden, scales_[0]);
        std::cout << "Comparing with independent expert GEMMs and tanh-GELU..." << std::endl;
        for (const int expert : participating) {
            const auto projection = [&](const char* name, py::object input, py::object activation_scale) {
                const auto prefix = std::string("model.decoder.layers.0.experts.") + std::to_string(expert) + "." + name + ".";
                auto weight = reader.tensor(NonEmptyString(prefix + "weight")).attr("to")("cuda");
                auto scale = reader.tensor(NonEmptyString(prefix + "weight_scale")).attr("to")("cuda");
                if (std::string_view(name) == "down_proj") {
                    weight = functional.attr("pad")(weight, py::make_tuple(0, 32));
                    scale = functional.attr("pad")(scale, py::make_tuple(0, 4));
                } else {
                    weight = functional.attr("pad")(weight, py::make_tuple(0, 0, 0, 64));
                    scale = functional.attr("pad")(scale, py::make_tuple(0, 0, 0, 64));
                }
                auto global = reader.tensor(NonEmptyString(prefix + "weight_scale_2")).attr("to")("cuda").attr("float")();
                return gemm(input[py::int_(0)], weight.attr("t")(), input[py::int_(1)],
                    swizzle(scale.attr("unsqueeze")(0), torch_).attr("squeeze")(0).attr("view")(torch_.attr("uint8")).attr("t")(),
                    "alpha"_a=torch_.attr("div")(global, activation_scale),
                    "out_dtype"_a=torch_.attr("bfloat16"), "backend"_a="cutlass", "enable_pdl"_a=false);
            };
            auto gate = projection("gate_proj", quantized_input, scales_[0]);
            auto up = projection("up_proj", quantized_input, scales_[0]);
            auto intermediate = torch_.attr("mul")(functional.attr("gelu")(gate, "approximate"_a="tanh"), up);
            auto down = projection("down_proj", quantize(intermediate, scales_[3]), scales_[3]);
            auto route_weight = torch_.attr("mul")(torch_.attr("eq")(indices, expert), probabilities)
                .attr("sum")(1).attr("unsqueeze")(1);
            reference.attr("add_")(torch_.attr("mul")(down.attr("float")(), route_weight));
        }
        if (!torch_.attr("isfinite")(actual).attr("all")().attr("item")().cast<bool>()) {
            throw CheckpointFailure("Native NVFP4 experts produced nonfinite values.");
        }
        auto difference = torch_.attr("sub")(actual, reference);
        const auto relative = torch_.attr("div")(difference.attr("norm")(), reference.attr("norm")()).attr("item")().cast<double>();
        std::cout << "MoE relative L2 error: " << relative << std::endl;
        if (!std::isfinite(relative) || relative > 0.03) {
            throw CheckpointFailure("Native NVFP4 expert parity exceeded the 3% relative-L2 admission bound.");
        }
    }

    static std::shared_ptr<Nvfp4Experts> load(const CheckpointReader& reader, int layer) {
        auto torch = py::module_::import("torch");
        const py::object pad = py::module_::import("torch.nn.functional").attr("pad");
        auto gate = stack_projection(reader, layer, "gate_proj", "weight", torch);
        auto up = stack_projection(reader, layer, "up_proj", "weight", torch);
        auto down = stack_projection(reader, layer, "down_proj", "weight", torch);
        if (gate.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 704, 1408} ||
            up.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 704, 1408} ||
            down.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 2816, 352}) {
            throw CheckpointFailure("Unexpected packed expert dimensions at layer " + std::to_string(layer));
        }
        auto gate_scale = stack_projection(reader, layer, "gate_proj", "weight_scale", torch);
        auto up_scale = stack_projection(reader, layer, "up_proj", "weight_scale", torch);
        auto down_scale = stack_projection(reader, layer, "down_proj", "weight_scale", torch);
        if (!gate.attr("dtype").is(torch.attr("uint8")) ||
            !up.attr("dtype").is(torch.attr("uint8")) ||
            !down.attr("dtype").is(torch.attr("uint8")) ||
            !gate_scale.attr("dtype").is(torch.attr("float8_e4m3fn")) ||
            !up_scale.attr("dtype").is(torch.attr("float8_e4m3fn")) ||
            !down_scale.attr("dtype").is(torch.attr("float8_e4m3fn")) ||
            gate_scale.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 704, 176} ||
            up_scale.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 704, 176} ||
            down_scale.attr("shape").cast<std::vector<int>>() != std::vector<int>{128, 2816, 44}) {
            throw CheckpointFailure("Malformed NVFP4 storage or block scales at layer " + std::to_string(layer));
        }
        auto gate_global = stack_projection(reader, layer, "gate_proj", "weight_scale_2", torch).attr("float")().attr("reshape")(-1);
        auto up_global = stack_projection(reader, layer, "up_proj", "weight_scale_2", torch).attr("float")().attr("reshape")(-1);
        auto down_global = stack_projection(reader, layer, "down_proj", "weight_scale_2", torch).attr("float")().attr("reshape")(-1);
        if (!torch.attr("equal")(gate_global, up_global).cast<bool>()) {
            throw CheckpointFailure("Combined gate/up GEMM requires matching global weight scales at layer " + std::to_string(layer));
        }
        auto gate_input = stack_projection(reader, layer, "gate_proj", "input_scale", torch).attr("float")();
        auto up_input = stack_projection(reader, layer, "up_proj", "input_scale", torch).attr("float")();
        auto down_input = stack_projection(reader, layer, "down_proj", "input_scale", torch).attr("float")();
        auto input13 = torch.attr("maximum")(gate_input.attr("max")(), up_input.attr("max")());
        auto input2 = down_input.attr("max")();
        for (const auto& scale : py::make_tuple(gate_global, up_global, down_global, gate_input, up_input, down_input)) {
            if (!torch.attr("logical_and")(torch.attr("isfinite")(scale), torch.attr("gt")(scale, 0))
                    .attr("all")().attr("item")().cast<bool>()) {
                throw CheckpointFailure("NVFP4 global and activation scales must be finite and positive.");
            }
        }
        // Pad each gate/up half independently, preserving the activation split.
        // MoE GEMMs require 128-aligned intermediate width: 704 -> 768.
        auto weights13 = torch.attr("cat")(py::make_tuple(pad(up, py::make_tuple(0, 0, 0, 64)),
            pad(gate, py::make_tuple(0, 0, 0, 64))), "dim"_a=1).attr("contiguous")().attr("view")(torch.attr("int64"));
        auto weights2 = pad(down, py::make_tuple(0, 32)).attr("contiguous")().attr("view")(torch.attr("int64"));
        auto scales13 = torch.attr("cat")(py::make_tuple(pad(up_scale, py::make_tuple(0, 0, 0, 64)),
            pad(gate_scale, py::make_tuple(0, 0, 0, 64))), "dim"_a=1);
        auto scales2 = pad(down_scale, py::make_tuple(0, 4));
        py::list scales;
        scales.append(input13.attr("reciprocal")());
        scales.append(swizzle(scales13, torch));
        scales.append(torch.attr("mul")(input13, gate_global));
        scales.append(input2.attr("reciprocal")());
        scales.append(swizzle(scales2, torch));
        scales.append(torch.attr("mul")(input2, down_global));
        return std::shared_ptr<Nvfp4Experts>(new Nvfp4Experts(std::move(torch), std::move(weights13),
            std::move(weights2), std::move(scales)));
    }

    py::object forward(py::object hidden, py::object indices, py::object probabilities) const {
        if (hidden.attr("ndim").cast<int>() != 2 || hidden.attr("shape")[py::int_(1)].cast<int>() != 2816) {
            throw InputFailure("NVFP4 experts require [tokens, 2816] hidden states.");
        }
        const auto selected = indices.attr("tolist")().cast<std::vector<std::vector<int>>>();
        if (selected.size() != hidden.attr("shape")[py::int_(0)].cast<std::size_t>() ||
            probabilities.attr("shape").cast<std::vector<int>>() != indices.attr("shape").cast<std::vector<int>>()) {
            throw InputFailure("Expert routing dimensions do not match the hidden states.");
        }
        std::map<int, std::vector<std::pair<std::int64_t, std::int64_t>>> groups;
        for (std::size_t token = 0; token < selected.size(); ++token) {
            if (selected[token].size() != 8) {
                throw InputFailure("DiffusionGemma requires eight expert routes per token.");
            }
            std::set<int> unique;
            for (std::size_t slot = 0; slot < 8; ++slot) {
                const auto expert = selected[token][slot];
                if (expert < 0 || expert >= 128 || !unique.insert(expert).second) {
                    throw InputFailure("A routed expert is out of range or repeated.");
                }
                groups[expert].emplace_back(static_cast<std::int64_t>(token),
                                            static_cast<std::int64_t>(token * 8 + slot));
            }
        }
        auto output = torch_.attr("zeros_like")(hidden, "dtype"_a=torch_.attr("float32"));
        auto flat_probabilities = probabilities.attr("reshape")(-1);
        for (const auto& [expert, routes] : groups) {
            std::vector<std::int64_t> token_positions;
            std::vector<std::int64_t> probability_positions;
            for (const auto& [token, position] : routes) {
                token_positions.push_back(token);
                probability_positions.push_back(position);
            }
            auto positions = torch_.attr("tensor")(token_positions, "dtype"_a=torch_.attr("int64"), "device"_a="cuda");
            auto probability_indices = torch_.attr("tensor")(probability_positions, "dtype"_a=torch_.attr("int64"), "device"_a="cuda");
            auto input = hidden.attr("index_select")(0, positions);
            const auto multiply = [&](py::object activation, py::object weight, py::object block_scale,
                                      py::object input_scale, py::object alpha) {
                auto quantized = quantize_(activation.attr("contiguous")(), input_scale);
                return kernel_(quantized[py::int_(0)], weight.attr("view")(torch_.attr("uint8")).attr("t")(),
                    quantized[py::int_(1)], block_scale.attr("view")(torch_.attr("uint8")).attr("t")(),
                    "alpha"_a=alpha, "out_dtype"_a=torch_.attr("bfloat16"), "backend"_a="cutlass", "enable_pdl"_a=false);
            };
            auto up_gate = multiply(input, weights_13_[py::int_(expert)], scales_[1][py::int_(expert)],
                                    scales_[0], scales_[2][py::int_(expert)]);
            auto intermediate = torch_.attr("mul")(up_gate.attr("narrow")(1, 0, 768),
                gelu_(up_gate.attr("narrow")(1, 768, 768), "approximate"_a="tanh"));
            auto down = multiply(intermediate, weights_2_[py::int_(expert)], scales_[4][py::int_(expert)],
                                 scales_[3], scales_[5][py::int_(expert)]);
            auto weighted = torch_.attr("mul")(down.attr("float")(),
                flat_probabilities.attr("index_select")(0, probability_indices).attr("unsqueeze")(1));
            output.attr("index_add_")(0, positions, weighted);
        }
        return output.attr("to")(hidden.attr("dtype"));
    }

    static py::object module(std::shared_ptr<Nvfp4Experts> bank) {
        const py::object base = py::module_::import("torch.nn").attr("Module");
        py::dict attributes;
        attributes["__module__"] = "sglang_native_diffusiongemma";
        auto type = py::module_::import("builtins").attr("type")("NativeNvfp4Experts", py::make_tuple(base), attributes);
        type.attr("__init__") = py::cpp_function([base](py::object self) { base.attr("__init__")(self); }, py::is_method(type));
        type.attr("forward") = py::cpp_function([bank=std::move(bank)](
                py::object, py::object hidden, py::object indices, py::object probabilities) {
            return bank->forward(std::move(hidden), std::move(indices), std::move(probabilities));
        }, py::is_method(type));
        return type();
    }
};

struct Inspect final {};
struct VerifyMoe final {};
struct Serve final {};
class WorkerBootstrap final {
    std::uint32_t parent_pid_;
    std::uintptr_t pipe_handle_;
    WorkerBootstrap(std::uint32_t parent, std::uintptr_t pipe) : parent_pid_(parent), pipe_handle_(pipe) {}
public:
    static WorkerBootstrap parse(const NonEmptyString& command) {
        const std::regex format(R"(^from multiprocessing\.spawn import spawn_main; spawn_main\(parent_pid=([0-9]+), pipe_handle=([0-9]+)\)$)");
        std::smatch arguments;
        if (!std::regex_match(command.text(), arguments, format)) {
            throw InputFailure("Unsupported native worker bootstrap command.");
        }
        const auto parent = std::stoull(arguments[1].str());
        const auto pipe = std::stoull(arguments[2].str());
        if (parent == 0 || parent > std::numeric_limits<std::uint32_t>::max() || pipe == 0 ||
            pipe > std::numeric_limits<std::uintptr_t>::max()) {
            throw InputFailure("The native worker requires a valid parent process and inherited pipe.");
        }
        return WorkerBootstrap(static_cast<std::uint32_t>(parent), static_cast<std::uintptr_t>(pipe));
    }
    void run() const {
        py::module_::import("multiprocessing.spawn").attr("spawn_main")(
            "parent_pid"_a=parent_pid_, "pipe_handle"_a=pipe_handle_);
    }
};
struct Prompt final { NonEmptyString text; };
using Operation = std::variant<Inspect, VerifyMoe, Serve, WorkerBootstrap, Prompt>;

// All dynamic library objects remain inside this boundary owner. Its public
// interface receives validated requests and returns complete text results.
class InstalledRuntime final {
    py::module_ torch_ = py::module_::import("torch");
    py::module_ transformers_ = py::module_::import("transformers");
    ModelDirectory directory_;

    py::object read_json(const char* filename) const {
        const auto path = py::module_::import("pathlib").attr("Path")(directory_.native_path(), filename);
        return py::module_::import("json").attr("loads")(path.attr("read_text")("encoding"_a="utf-8"));
    }

    void validate_checkpoint() const {
        const auto config = read_json("config.json").cast<py::dict>();
        if (config["architectures"].cast<std::vector<std::string>>() !=
                std::vector<std::string>{"DiffusionGemmaForBlockDiffusion"}) {
            throw CheckpointFailure("Expected DiffusionGemmaForBlockDiffusion architecture.");
        }
        const auto text = config["text_config"].cast<py::dict>();
        for (const auto& [key, expected] : std::map<std::string, int>{
                {"hidden_size", 2816}, {"num_hidden_layers", 30}, {"num_experts", 128},
                {"moe_intermediate_size", 704}, {"top_k_experts", 8}, {"vocab_size", 262144}}) {
            if (text[py::str(key)].cast<int>() != expected) {
                throw CheckpointFailure("Unsupported DiffusionGemma dimension: " + key);
            }
        }
        const auto quant = config["quantization_config"].cast<py::dict>();
        if (quant["quant_algo"].cast<std::string>() != "NVFP4" ||
            quant["quant_method"].cast<std::string>() != "modelopt") {
            throw CheckpointFailure("The native trial requires the NVIDIA ModelOpt NVFP4 checkpoint.");
        }
    }

public:
    explicit InstalledRuntime(ModelDirectory directory) : directory_(std::move(directory)) {
        validate_checkpoint();
    }

    void inspect() const {
        std::cout << "Checkpoint: " << directory_.native_path() << '\n'
                  << "Torch: " << torch_.attr("__version__").cast<std::string>() << '\n'
                  << "Transformers: " << transformers_.attr("__version__").cast<std::string>() << '\n'
                  << "Diffusion model: "
                  << transformers_.attr("DiffusionGemmaForBlockDiffusion").attr("__name__").cast<std::string>()
                  << '\n'
                  << "FlashInfer: " << py::module_::import("flashinfer").attr("__version__").cast<std::string>()
                  << '\n';
    }

    void verify_moe() const {
        CheckpointReader reader(directory_, read_json("model.safetensors.index.json")["weight_map"].cast<py::dict>());
        Nvfp4Experts::load(reader, 0)->verify(reader);
    }

    py::object load_model() const {
        LibraryScope inference(torch_.attr("inference_mode")());
        auto config = transformers_.attr("AutoConfig").attr("from_pretrained")(
            directory_.native_path(), "local_files_only"_a=true);
        config.attr("_attn_implementation") = "sdpa";
        auto model = [&] {
            const auto initialization = py::module_::import("transformers.initialization");
            LibraryScope no_initialization(initialization.attr("no_init_weights")());
            LibraryScope empty(torch_.attr("device")("meta"));
            LibraryScope safe_creation(initialization.attr("meta_device_safe_creation_ops")());
            return transformers_.attr("DiffusionGemmaForBlockDiffusion")(config);
        }();
        CheckpointReader reader(directory_, read_json("model.safetensors.index.json")["weight_map"].cast<py::dict>());
        std::cout << "Loading language tensors and preserving shared encoder/decoder weights..." << std::endl;
        for (const auto& name : reader.names()) {
            if (name.text().find(".experts.") != std::string::npos ||
                name.text().starts_with("model.encoder.vision_tower.") ||
                name.text().starts_with("model.encoder.embed_vision.")) {
                continue;
            }
            const auto separator = name.text().rfind('.');
            if (separator == std::string::npos) {
                throw CheckpointFailure("Unqualified language tensor name: " + name.text());
            }
            auto module = model.attr("get_submodule")(name.text().substr(0, separator));
            const auto leaf = name.text().substr(separator + 1);
            auto tensor = reader.tensor(name).attr("to")("cuda");
            if (py::getattr(module, leaf.c_str()).attr("shape").cast<std::vector<int>>() !=
                    tensor.attr("shape").cast<std::vector<int>>()) {
                throw CheckpointFailure("Unexpected language tensor dimensions: " + name.text());
            }
            if (module.attr("_parameters").cast<py::dict>().contains(py::str(leaf))) {
                py::setattr(module, leaf.c_str(), torch_.attr("nn").attr("Parameter")(tensor, "requires_grad"_a=false));
            } else if (module.attr("_buffers").cast<py::dict>().contains(py::str(leaf))) {
                py::setattr(module, leaf.c_str(), tensor);
            } else {
                throw CheckpointFailure("Unexpected language checkpoint tensor: " + name.text());
            }
        }
        model.attr("tie_weights")();
        const py::object rotary_type = py::module_::import("transformers.models.diffusion_gemma.modeling_diffusion_gemma")
            .attr("DiffusionGemmaTextRotaryEmbedding");
        for (const auto& stack : py::make_tuple(model.attr("model").attr("encoder").attr("language_model"),
                                               model.attr("model").attr("decoder"))) {
            stack.attr("rotary_emb") = rotary_type(config.attr("text_config"), "device"_a="cuda");
            stack.attr("embed_tokens").attr("embed_scale") = torch_.attr("tensor")(
                stack.attr("embed_tokens").attr("scalar_embed_scale"), "dtype"_a=torch_.attr("float32"), "device"_a="cuda");
        }
        for (int layer = 0; layer < 30; ++layer) {
            auto experts = Nvfp4Experts::module(Nvfp4Experts::load(reader, layer));
            model.attr("model").attr("decoder").attr("layers")[py::int_(layer)].attr("experts") = experts;
            model.attr("model").attr("encoder").attr("language_model").attr("layers")[py::int_(layer)].attr("experts") = experts;
            std::cout << "Loaded NVFP4 expert layer " << layer + 1 << "/30" << std::endl;
        }
        for (const auto& item : model.attr("named_parameters")()) {
            const auto parameter = py::reinterpret_borrow<py::tuple>(item);
            if (parameter[1].attr("is_meta").cast<bool>()) {
                const auto name = parameter[0].cast<std::string>();
                if (!name.starts_with("model.encoder.vision_tower.") && !name.starts_with("model.encoder.embed_vision.")) {
                    throw CheckpointFailure("Language parameter was not loaded: " + name);
                }
            }
        }
        model.attr("eval")();
        model.attr("generation_config") = py::module_::import("transformers.models.diffusion_gemma.generation_diffusion_gemma")
            .attr("DiffusionGemmaGenerationConfig").attr("from_pretrained")(directory_.native_path(), "local_files_only"_a=true);
        // This entry point exposes text generation. Remove the unused meta
        // vision modules so SGLang's device/parameter inspection sees only
        // materialized language parameters.
        model.attr("model").attr("encoder").attr("vision_tower") = torch_.attr("nn").attr("Identity")();
        model.attr("model").attr("encoder").attr("embed_vision") = torch_.attr("nn").attr("Identity")();
        return model;
    }

    NonEmptyString prompt(const Prompt& request) const {
        LibraryScope inference(torch_.attr("inference_mode")());
        auto tokenizer = transformers_.attr("AutoTokenizer").attr("from_pretrained")(
            directory_.native_path(), "local_files_only"_a=true);
        py::list messages;
        messages.append(py::dict("role"_a="user", "content"_a=request.text.text()));
        auto input = tokenizer.attr("apply_chat_template")(messages, "tokenize"_a=true,
            "add_generation_prompt"_a=true, "enable_thinking"_a=false,
            "return_dict"_a=false, "return_tensors"_a="pt");
        if (input.attr("shape")[py::int_(1)].cast<int>() > 1792) {
            throw InputFailure("The prompt exceeds this trial's 1792-token input limit.");
        }
        auto model = load_model();
        input = input.attr("to")("cuda");
        std::cout << "Generating with the checkpoint's entropy-bound denoising schedule..." << std::endl;
        const py::object output = model.attr("generate")("input_ids"_a=input, "generation_config"_a=model.attr("generation_config"),
            "max_new_tokens"_a=256).attr("sequences");
        auto tokens = output[py::int_(0)].attr("narrow")(0, input.attr("shape")[py::int_(1)],
            output.attr("shape")[py::int_(1)].cast<int>() - input.attr("shape")[py::int_(1)].cast<int>());
        py::list visible_tokens;
        for (const auto token : tokens.attr("tolist")().cast<std::vector<std::int64_t>>()) {
            if (token == 1 || token == 106 || token == 50) {
                break;
            }
            visible_tokens.append(token);
        }
        const auto raw_text = tokenizer.attr("decode")(visible_tokens, "skip_special_tokens"_a=false);
        const auto detector = py::module_::import("sglang.srt.parser.reasoning_parser")
            .attr("Gemma4Detector")("stream_reasoning"_a=false);
        const auto parsed = detector.attr("detect_and_parse")(raw_text);
        if (py::len(parsed.attr("normal_text")) > 0) {
            return NonEmptyString(parsed.attr("normal_text").cast<std::string>());
        }
        if (py::len(parsed.attr("reasoning_text")) > 0) {
            return NonEmptyString(parsed.attr("reasoning_text").cast<std::string>());
        }
        throw CheckpointFailure("DiffusionGemma completed without visible response text.");
    }
};

class TokenSequence final {
    std::vector<std::int64_t> tokens_;
public:
    explicit TokenSequence(std::vector<std::int64_t> tokens) : tokens_(std::move(tokens)) {
        if (tokens_.empty()) {
            throw InputFailure("A token sequence must contain at least one token.");
        }
        for (const auto token : tokens_) {
            if (token < 0 || token >= 262144) {
                throw InputFailure("A token lies outside the DiffusionGemma vocabulary.");
            }
        }
    }
    const std::vector<std::int64_t>& values() const noexcept { return tokens_; }
};

class GenerationBudget final {
    int count_;
public:
    explicit GenerationBudget(int count) : count_(count) {
        if (count < 1 || count > 1024) {
            throw InputFailure("This DiffusionGemma trial accepts 1 through 1024 output tokens.");
        }
    }
    int count() const noexcept { return count_; }
};

class NativeGenerator final {
    py::object model_;
public:
    explicit NativeGenerator(py::object loaded_model) : model_(std::move(loaded_model)) {}

    TokenSequence generate(const TokenSequence& prompt, GenerationBudget budget) const {
        const auto torch = py::module_::import("torch");
        LibraryScope inference(torch.attr("inference_mode")());
        auto input = torch.attr("tensor")(prompt.values(), "dtype"_a=torch.attr("int64"), "device"_a="cuda")
            .attr("unsqueeze")(0);
        const py::object output = model_.attr("generate")("input_ids"_a=input,
            "generation_config"_a=model_.attr("generation_config"), "max_new_tokens"_a=budget.count()).attr("sequences");
        auto tokens = output[py::int_(0)].attr("narrow")(0, prompt.values().size(),
            std::min(static_cast<std::size_t>(budget.count()),
                output.attr("shape")[py::int_(1)].cast<std::size_t>() - prompt.values().size()));
        const auto generated = tokens.attr("tolist")().cast<std::vector<std::int64_t>>();
        for (std::size_t index = 0; index < generated.size(); ++index) {
            switch (generated[index]) {
            case 1:
            case 106:
            case 50:
                return TokenSequence(std::vector<std::int64_t>(generated.begin(), generated.begin() + index + 1));
            default:
                break;
            }
        }
        return TokenSequence(generated);
    }
};

struct ReadyForPrompt final {};
struct DeliveringReply final {
    py::object scheduler_request;
    TokenSequence reply;
    std::size_t cursor;
};

// SGLang's single-request scheduler asks for bounded output slices. The native
// owner retains one completed diffusion reply until all slices have been handed
// off. Holding the exact scheduler request prevents a reused public request ID
// from exposing a cancelled request's retained reply to a later request.
class SequentialReplies final {
    using Phase = std::variant<ReadyForPrompt, DeliveringReply>;
    Phase phase_ = ReadyForPrompt{};
    NativeGenerator generator_;
public:
    explicit SequentialReplies(NativeGenerator generator) : generator_(std::move(generator)) {}

    TokenSequence take(py::object request, const TokenSequence& prompt,
                       GenerationBudget budget, std::size_t capacity) {
        if (capacity < 1 || capacity > 256) {
            throw InputFailure("A scheduled diffusion slice must contain 1 through 256 positions.");
        }
        auto ready = std::visit([&](const auto& phase) -> DeliveringReply {
            using State = std::decay_t<decltype(phase)>;
            if constexpr (std::is_same_v<State, ReadyForPrompt>) {
                return {request, generator_.generate(prompt, budget), 0};
            } else {
                if (phase.scheduler_request.is(request)) {
                    return phase;
                }
                return {request, generator_.generate(prompt, budget), 0};
            }
        }, phase_);
        const auto end = std::min(ready.cursor + capacity, ready.reply.values().size());
        TokenSequence slice(std::vector<std::int64_t>(ready.reply.values().begin() + ready.cursor,
                                                     ready.reply.values().begin() + end));
        if (end == ready.reply.values().size()) {
            phase_ = ReadyForPrompt{};
        } else {
            ready.cursor = end;
            phase_ = std::move(ready);
        }
        return slice;
    }
};

py::object native_callable_type(const char* name) {
    py::dict attributes;
    attributes["__module__"] = "sglang_native_diffusiongemma";
    return py::module_::import("builtins").attr("type")(name,
        py::make_tuple(py::module_::import("builtins").attr("object")), attributes);
}

void install_native_serving(const NonEmptyString& executable) {
    const auto builtins = py::module_::import("builtins");
    py::module_::import("multiprocessing").attr("set_executable")(
        executable.text());
    auto native_module = py::module_::import("types").attr("ModuleType")("sglang_native_diffusiongemma");
    py::module_::import("sys").attr("modules")["sglang_native_diffusiongemma"] = native_module;
    const py::object base = py::module_::import("torch.nn").attr("Module");
    py::dict attributes;
    attributes["__module__"] = "sglang_native_diffusiongemma";
    auto type = builtins.attr("type")("DiffusionGemmaForBlockDiffusion", py::make_tuple(base), attributes);

    // A callable instance supplies an explicit inspect.Signature, as required
    // by SGLang's model runner. Python source is unnecessary for this ABI.
    auto forward_type = native_callable_type("NativeDiffusionForward");
    auto inspect = py::module_::import("inspect");
    const py::object parameter = inspect.attr("Parameter");
    py::list parameters;
    for (const char* name : {"input_ids", "positions", "forward_batch"}) {
        parameters.append(parameter(name, parameter.attr("POSITIONAL_OR_KEYWORD")));
    }
    forward_type.attr("__signature__") = inspect.attr("Signature")(parameters);
    forward_type.attr("__call__") = py::cpp_function([](py::object, py::args, py::kwargs) -> py::object {
        throw InputFailure("DiffusionGemma must run through its native block-generation adapter.");
    }, py::is_method(forward_type));

    type.attr("__init__") = py::cpp_function([base, forward_type](py::object self, py::kwargs arguments) {
        base.attr("__init__")(self);
        if (!arguments.contains("config") || !arguments.contains("quant_config") || arguments.size() != 2) {
            throw InputFailure("The native DiffusionGemma model requires config and quant_config.");
        }
        self.attr("config") = arguments["config"];
        self.attr("forward") = forward_type();
    }, py::is_method(type));
    type.attr("load_weights") = py::cpp_function([](py::object self, py::object) {
        InstalledRuntime runtime(ModelDirectory(self.attr("config").attr("_name_or_path").cast<std::string>()));
        auto model = runtime.load_model();
        self.attr("hf_model") = model;
        auto replies = std::make_shared<SequentialReplies>(NativeGenerator(model));
        auto callable = native_callable_type("NativeReplySlices");
        callable.attr("__call__") = py::cpp_function([replies](py::object, py::object request, int capacity) {
            auto slice = replies->take(request,
                TokenSequence(request.attr("origin_input_ids").cast<std::vector<std::int64_t>>()),
                GenerationBudget(request.attr("sampling_params").attr("max_new_tokens").cast<int>()),
                static_cast<std::size_t>(capacity));
            return slice.values();
        }, py::is_method(callable));
        self.attr("native_reply_slice") = callable();
    }, py::is_method(type));
    type.attr("get_input_embeddings") = py::cpp_function([](py::object self) {
        return self.attr("hf_model").attr("get_input_embeddings")();
    }, py::is_method(type));
    native_module.attr("DiffusionGemmaForBlockDiffusion") = type;
    py::module_::import("sglang.srt.models.registry").attr("ModelRegistry").attr("models")
        ["DiffusionGemmaForBlockDiffusion"] = type;
    const py::object server_args = py::module_::import("sglang.srt.server_args").attr("ServerArgs");
    server_args.attr("LANGUAGE_MODEL_ONLY_ARCHITECTURES") =
        server_args.attr("LANGUAGE_MODEL_ONLY_ARCHITECTURES").attr("__add__")(
            py::make_tuple("DiffusionGemmaForBlockDiffusion"));

    const py::object dllm_config = py::module_::import("sglang.srt.dllm.config").attr("DllmConfig");
    const py::object original_config = dllm_config.attr("from_server_args");
    dllm_config.attr("from_server_args") = builtins.attr("staticmethod")(py::cpp_function(
        [dllm_config, original_config](py::object arguments) -> py::object {
            auto view = py::module_::import("sglang.srt.arg_groups.overrides").attr("resolving_view")(arguments);
            if (view.attr("dllm_algorithm").is_none()) {
                return original_config(arguments);
            }
            if (view.attr("dllm_algorithm").cast<std::string>() != "NativeDiffusionGemma") {
                return original_config(arguments);
            }
            if (view.attr("max_running_requests").cast<int>() != 1) {
                throw InputFailure("Native DiffusionGemma requires one running request.");
            }
            return dllm_config("algorithm"_a="NativeDiffusionGemma", "algorithm_config"_a=py::dict(),
                "block_size"_a=256, "mask_id"_a=-1, "max_running_requests"_a=1,
                "first_done_first_out_mode"_a=false);
        }));

    auto algorithm = native_callable_type("NativeDiffusionGemma");
    algorithm.attr("__init__") = py::cpp_function([](py::object, py::object) {}, py::is_method(algorithm));
    algorithm.attr("fdfo") = builtins.attr("property")(py::cpp_function([](py::object) {
        return py::bool_(false);
    }));
    py::module_::import("sglang.srt.dllm.algorithm").attr("algo_name_to_cls")["NativeDiffusionGemma"] = algorithm;

    const py::object worker = py::module_::import("sglang.srt.managers.tp_worker").attr("TpModelWorker");
    const py::object original_forward = worker.attr("_forward_batch_generation_dllm");
    worker.attr("_forward_batch_generation_dllm") = py::cpp_function(
        [original_forward, type](py::object self, py::object forward_batch, py::object batch) -> py::object {
            const py::object model = self.attr("model_runner").attr("model");
            if (!py::isinstance(model, type)) {
                return original_forward(self, forward_batch, batch);
            }
            if (batch.is_none()) {
                throw InputFailure("Native DiffusionGemma requires a scheduler-owned request batch.");
            }
            auto requests = batch.attr("reqs").cast<py::list>();
            if (requests.size() != 1) {
                throw InputFailure("Native DiffusionGemma requires a single-request batch.");
            }
            const auto ids = forward_batch.attr("input_ids").attr("tolist")().cast<std::vector<std::int64_t>>();
            const auto capacity = static_cast<int>(std::count(ids.begin(), ids.end(), -1));
            py::list outputs;
            if (capacity > 0) {
                auto request = requests[0];
                // The checkpoint's complete EOS set also includes token 50.
                // SGLang retains the user's additional stop IDs.
                if (request.attr("sampling_params").attr("stop_token_ids").is_none()) {
                    request.attr("sampling_params").attr("stop_token_ids") = py::set(py::make_tuple(1, 106, 50));
                } else {
                    request.attr("sampling_params").attr("stop_token_ids").attr("update")(py::make_tuple(1, 106, 50));
                }
                const auto tokens = model.attr("native_reply_slice")(request, capacity);
                const auto torch = py::module_::import("torch");
                outputs.append(torch.attr("tensor")(tokens, "dtype"_a=torch.attr("int64"), "device"_a="cpu"));
            }
            auto logits = py::module_::import("sglang.srt.layers.logits_processor").attr("LogitsProcessorOutput")(
                "next_token_logits"_a=py::none());
            return py::module_::import("sglang.srt.managers.scheduler").attr("GenerationBatchResult")(
                "logits_output"_a=logits, "next_token_ids"_a=outputs, "can_run_cuda_graph"_a=false);
        }, py::is_method(worker));

    // The synchronous diffusion result path does not invoke SGLang's ordinary
    // reasoning-accounting owner. Feed that owner each native CPU token slice
    // before the existing result handler serializes usage and streamed output.
    const py::object scheduler = py::module_::import("sglang.srt.managers.scheduler").attr("Scheduler");
    const py::object original_result = scheduler.attr("process_batch_result_dllm");
    scheduler.attr("process_batch_result_dllm") = py::cpp_function(
        [original_result, type](py::object self, py::object batch, py::object result) {
            if (py::isinstance(self.attr("tp_worker").attr("model_runner").attr("model"), type)) {
                const auto outputs = result.attr("next_token_ids").cast<py::list>();
                for (std::size_t index = 0; index < outputs.size(); ++index) {
                    self.attr("batch_result_processor").attr("_maybe_update_reasoning_tokens")(
                        batch.attr("reqs")[py::int_(index)], outputs[index].attr("tolist")());
                }
            }
            original_result(self, batch, result);
        }, py::is_method(scheduler));

    const py::object manager = py::module_::import("sglang.srt.managers.tokenizer_manager").attr("TokenizerManager");
    const py::object original_validate = manager.attr("_validate_one_request");
    manager.attr("_validate_one_request") = py::cpp_function(
        [original_validate](py::object self, py::object request, py::object input_ids) {
            try {
                TokenSequence(input_ids.cast<std::vector<std::int64_t>>());
            } catch (const InputFailure& error) {
                throw py::value_error(error.what());
            } catch (const py::cast_error&) {
                throw py::value_error("Native DiffusionGemma requires integer vocabulary token IDs.");
            }
            const auto parameters = request.attr("sampling_params").cast<py::dict>();
            const std::set<std::string> supported_parameters{
                "max_new_tokens", "stop", "stop_token_ids", "stop_regex", "temperature", "top_p",
                "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty",
                "min_new_tokens", "n", "beam_width", "json_schema", "regex", "ebnf", "structural_tag",
                "ignore_eos", "skip_special_tokens", "spaces_between_special_tokens", "no_stop_trim",
                "stream_interval", "logit_bias", "sampling_seed", "custom_params"};
            for (const auto& item : parameters) {
                if (!supported_parameters.contains(py::cast<std::string>(item.first))) {
                    throw py::value_error("Unknown native DiffusionGemma sampling parameter: " +
                        py::cast<std::string>(item.first));
                }
            }
            if (parameters.contains("max_new_tokens") && !parameters["max_new_tokens"].is_none()) {
                if (!PyLong_CheckExact(parameters["max_new_tokens"].ptr())) {
                    throw py::value_error("Native DiffusionGemma requires an integer output-token budget.");
                }
                try {
                    GenerationBudget(parameters["max_new_tokens"].cast<int>());
                } catch (const InputFailure& error) {
                    throw py::value_error(error.what());
                } catch (const py::cast_error&) {
                    throw py::value_error("This DiffusionGemma trial accepts 1 through 1024 output tokens.");
                }
            } else {
                parameters["max_new_tokens"] = 256;
            }
            // SGLang reserves this request-ID prefix for its own one-token
            // health probes. They exercise the actual diffusion schedule;
            // ordinary requests still reject autoregressive temperature=0.
            if (request.attr("rid").cast<std::string>().starts_with("HEALTH_CHECK_") &&
                input_ids.cast<std::vector<std::int64_t>>() == std::vector<std::int64_t>{0} &&
                parameters["max_new_tokens"].cast<int>() == 1) {
                parameters["temperature"] = 1.0;
            }
            original_validate(self, request, input_ids);
            if (py::len(input_ids) + parameters["max_new_tokens"].cast<std::size_t>() > 2048) {
                throw py::value_error("Native DiffusionGemma accepts at most 2048 prompt plus output tokens.");
            }
            for (const auto& [name, neutral] : std::map<std::string, double>{
                    {"temperature", 1.0}, {"top_p", 1.0}, {"top_k", -1.0}, {"min_p", 0.0},
                    {"presence_penalty", 0.0}, {"frequency_penalty", 0.0}, {"repetition_penalty", 1.0},
                    {"min_new_tokens", 0.0}, {"n", 1.0}}) {
                if (parameters.contains(py::str(name)) && !parameters[py::str(name)].is_none()) {
                    if (!PyFloat_CheckExact(parameters[py::str(name)].ptr()) &&
                        !PyLong_CheckExact(parameters[py::str(name)].ptr())) {
                        throw py::value_error("Native DiffusionGemma requires a numeric sampling parameter: " + name);
                    }
                    try {
                        if (parameters[py::str(name)].cast<double>() != neutral) {
                            throw py::value_error("Native DiffusionGemma uses its checkpoint denoising schedule; unsupported sampling override: " + name);
                        }
                    } catch (const py::cast_error&) {
                        throw py::value_error("Native DiffusionGemma sampling parameter is outside the numeric range: " + name);
                    }
                }
            }
            for (const char* name : {"json_schema", "regex", "ebnf", "structural_tag", "logit_bias", "sampling_seed",
                                     "beam_width", "custom_params"}) {
                if (parameters.contains(name) && !parameters[name].is_none()) {
                    throw py::value_error(std::string("Native DiffusionGemma does not support ") + name);
                }
            }
            if (parameters.contains("ignore_eos") && !parameters["ignore_eos"].is_none()) {
                if (!PyBool_Check(parameters["ignore_eos"].ptr())) {
                    throw py::value_error("Native DiffusionGemma requires a JSON boolean for ignore_eos.");
                }
                if (parameters["ignore_eos"].cast<bool>()) {
                    throw py::value_error("Native DiffusionGemma requires checkpoint EOS stopping.");
                }
            }
            if (request.attr("return_logprob").cast<bool>()) {
                throw py::value_error("Native DiffusionGemma does not provide autoregressive log probabilities.");
            }
        }, py::is_method(manager));

    const py::object chat = py::module_::import("sglang.srt.entrypoints.openai.serving_chat").attr("OpenAIServingChat");
    const py::object original_chat_init = chat.attr("__init__");
    chat.attr("__init__") = py::cpp_function([original_chat_init](
            py::object self, py::object tokenizer_manager, py::object template_manager) {
        original_chat_init(self, tokenizer_manager, template_manager);
        // DiffusionGemma uses Gemma4's channel delimiters. Keep those tokens
        // until the existing reasoning/tool parsers have consumed them.
        if (tokenizer_manager.attr("model_config").attr("hf_config").attr("model_type")
                .cast<std::string>() == "diffusion_gemma") {
            self.attr("is_gemma4") = true;
        }
    }, py::is_method(chat), py::arg("tokenizer_manager"), py::arg("template_manager"));
}

void serve_native(const NonEmptyString& executable) {
    install_native_serving(executable);
    py::list arguments;
    for (const char* argument : {"sglang", "serve", "--model-path",
            "C:\\Users\\Daniel\\models\\diffusiongemma-26B-A4B-it-NVFP4",
            "--served-model-name", "diffusiongemma-26b-a4b-nvfp4", "--host", "127.0.0.1", "--port", "30001",
            "--context-length", "2048", "--max-total-tokens", "4096", "--max-running-requests", "1",
            "--mem-fraction-static", "0.80", "--language-model-only", "--dllm-algorithm", "NativeDiffusionGemma",
            "--skip-server-warmup", "--disable-flashinfer-autotune",
            "--no-dllm-fdfo", "--disable-radix-cache", "--disable-overlap-schedule",
            "--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled",
            "--chunked-prefill-size", "256", "--attention-backend", "triton",
            "--moe-runner-backend", "flashinfer_cutlass", "--reasoning-parser", "gemma4", "--tool-call-parser", "gemma4"}) {
        arguments.append(argument);
    }
    py::module_::import("sys").attr("argv") = arguments;
    py::module_::import("sglang.cli.main").attr("main")();
}

void configure_paths() {
    const auto repo = std::filesystem::path(DG_REPO_ROOT);
    py::module_::import("site").attr("addsitedir")((repo / ".venv/Lib/site-packages").string());
    py::module_::import("sys").attr("path").attr("insert")(0, (repo / "python").string());
    const py::object environment = py::module_::import("os").attr("environ");
    environment["FLASHINFER_NVFP4_4OVER6"] = "0";
    environment["SGLANG_HEALTH_CHECK_TIMEOUT"] = "90";
    // The installed Windows JIT derives these library paths from sys.executable.
    // An embedded native launcher lives outside the virtual environment; supply
    // the real installed libraries through the linker boundary instead.
    const auto libraries = (repo / ".venv/Lib/site-packages/torch/lib").string() + ";" +
                           (repo / ".venv/Lib/site-packages/tvm_ffi/lib").string();
    if (environment.attr("__contains__")("LIB").cast<bool>()) {
        environment["LIB"] = libraries + ";" + environment["LIB"].cast<std::string>();
    } else {
        environment["LIB"] = libraries;
    }
    const auto modules = (repo / ".venv/Lib/site-packages").string() + ";" + (repo / "python").string();
    if (environment.attr("__contains__")("PYTHONPATH").cast<bool>()) {
        environment["PYTHONPATH"] = modules + ";" + environment["PYTHONPATH"].cast<std::string>();
    } else {
        environment["PYTHONPATH"] = modules;
    }
    py::module_::import("faulthandler").attr("enable")();
}

Operation parse_operation(int argc, wchar_t** argv) {
    if (argc == 2 && std::wstring_view(argv[1]) == L"inspect") {
        return Inspect{};
    }
    if (argc == 2 && std::wstring_view(argv[1]) == L"verify-moe") {
        return VerifyMoe{};
    }
    if (argc == 2 && std::wstring_view(argv[1]) == L"serve") {
        return Serve{};
    }
    if (argc == 4 && std::wstring_view(argv[1]) == L"-c" &&
        std::wstring_view(argv[3]) == L"--multiprocessing-fork") {
        return WorkerBootstrap::parse(NonEmptyString(py::cast(std::wstring(argv[2])).cast<std::string>()));
    }
    if (argc == 5 && std::wstring_view(argv[1]) == L"-B" && std::wstring_view(argv[2]) == L"-c" &&
        std::wstring_view(argv[4]) == L"--multiprocessing-fork") {
        return WorkerBootstrap::parse(NonEmptyString(py::cast(std::wstring(argv[3])).cast<std::string>()));
    }
    if (argc == 3 && std::wstring_view(argv[1]) == L"prompt") {
        return Prompt{NonEmptyString(py::cast(std::wstring(argv[2])).cast<std::string>())};
    }
    throw InputFailure("Usage: diffusiongemma inspect | diffusiongemma verify-moe | diffusiongemma serve | diffusiongemma prompt \"your prompt\"");
}
} // namespace

extern "C" __declspec(dllexport) int diffusiongemma_main(int argc, wchar_t** argv) {
    PyConfig configuration;
    PyConfig_InitPythonConfig(&configuration);
    configuration.parse_argv = 0;
    configuration.write_bytecode = 0;
    const auto python_base = std::filesystem::path(DG_PYTHON_BASE).wstring();
    if (PyStatus_Exception(PyConfig_SetString(&configuration, &configuration.home, python_base.c_str()))) {
        PyConfig_Clear(&configuration);
        std::cerr << "Cannot configure CPython's base directory.\n";
        return 1;
    }
    try {
        py::scoped_interpreter interpreter(&configuration, 0, nullptr, false);
        try {
            configure_paths();
            const NonEmptyString executable(std::filesystem::canonical(argv[0]).string());
            py::module_::import("sys").attr("executable") =
                (std::filesystem::path(DG_PYTHON_BASE) / "python.exe").string();
            const auto operation = parse_operation(argc, argv);
            InstalledRuntime runtime(ModelDirectory(
                L"C:\\Users\\Daniel\\models\\diffusiongemma-26B-A4B-it-NVFP4"));
            std::visit([&](const auto& command) {
                using Command = std::decay_t<decltype(command)>;
                if constexpr (std::is_same_v<Command, Inspect>) {
                    runtime.inspect();
                } else if constexpr (std::is_same_v<Command, VerifyMoe>) {
                    runtime.verify_moe();
                } else if constexpr (std::is_same_v<Command, Serve>) {
                    serve_native(executable);
                } else if constexpr (std::is_same_v<Command, WorkerBootstrap>) {
                    py::list worker_arguments;
                    worker_arguments.append("-c");
                    worker_arguments.append("--multiprocessing-fork");
                    py::module_::import("sys").attr("argv") = worker_arguments;
                    install_native_serving(executable);
                    command.run();
                } else {
                    std::cout << runtime.prompt(command).text() << std::endl;
                }
            }, operation);
            return 0;
        } catch (const py::error_already_set& error) {
            if (error.matches(PyExc_SystemExit)) {
                if (error.value().attr("code").is_none()) {
                    return 0;
                }
                if (py::isinstance<py::int_>(error.value().attr("code"))) {
                    return error.value().attr("code").cast<int>();
                }
                std::cerr << py::str(error.value().attr("code")).cast<std::string>() << '\n';
                return 1;
            }
            std::cerr << "Installed runtime failed: " << error.what() << '\n';
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "DiffusionGemma failed: " << error.what() << '\n';
        return 1;
    }
}
