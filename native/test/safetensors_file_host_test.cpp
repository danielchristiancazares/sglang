#include "sglang/native/safetensors_file.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using sglang::native::open_safetensors_file;
using sglang::native::safetensors_dtype_name;
using sglang::native::safetensors_error_code_name;
using sglang::native::SafetensorsDType;
using sglang::native::SafetensorsErrorCode;
using sglang::native::SafetensorsFile;
using sglang::native::SafetensorsOpenResult;
using sglang::native::SafetensorsTensorInfo;

[[nodiscard]] bool record_check(bool passed, const char *expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) {   \
      return false;                                                            \
    }                                                                          \
  } while (false)

class TemporaryFile final {
public:
  explicit TemporaryFile(std::wstring_view suffix) {
    std::array<wchar_t, MAX_PATH + 1U> directory{};
    const DWORD length =
        GetTempPathW(static_cast<DWORD>(directory.size()), directory.data());
    if (length == 0U || length >= directory.size()) {
      return;
    }
    path_ = directory.data();
    path_ += L"sglang-native-safetensors-";
    path_ += std::to_wstring(GetCurrentProcessId());
    path_ += L"-";
    path_ += std::to_wstring(GetTickCount64());
    path_ += suffix;
  }

  ~TemporaryFile() noexcept {
    if (!path_.empty()) {
      static_cast<void>(DeleteFileW(path_.c_str()));
    }
  }

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  [[nodiscard]] const std::wstring &path() const noexcept { return path_; }

private:
  std::wstring path_;
};

[[nodiscard]] bool write_shard(const std::filesystem::path &path,
                               std::string header,
                               std::span<const std::byte> data) {
  while ((header.size() % 8U) != 0U) {
    header.push_back(' ');
  }
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  CHECK(stream.good());
  const uint64_t header_size = header.size();
  std::array<char, sizeof(uint64_t)> prefix{};
  for (uint32_t index = 0; index < prefix.size(); ++index) {
    prefix[index] = static_cast<char>((header_size >> (index * 8U)) & 0xffU);
  }
  stream.write(prefix.data(), static_cast<std::streamsize>(prefix.size()));
  stream.write(header.data(), static_cast<std::streamsize>(header.size()));
  stream.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size()));
  stream.close();
  CHECK(stream.good());
  return true;
}

[[nodiscard]] bool ValidSyntheticShardMapsZeroCopy() {
  TemporaryFile temporary(L"-valid.safetensors");
  CHECK(!temporary.path().empty());
  const std::string header =
      R"({"z_float":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"__metadata__":{"format":"pt"},"a_bfloat":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}})";
  const std::array<std::byte, 12> data{
      std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x3f},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x40},
      std::byte{0x80}, std::byte{0x3f}, std::byte{0x00}, std::byte{0x40}};
  CHECK(write_shard(temporary.path(), header, data));

  SafetensorsOpenResult result = open_safetensors_file(temporary.path());
  CHECK(result.ok());
  CHECK(result.file.is_open());
  CHECK(result.file.tensors().size() == 2U);
  CHECK(result.file.tensors()[0].name == "a_bfloat");
  CHECK(result.file.tensors()[1].name == "z_float");

  const SafetensorsTensorInfo *floating = result.file.find("z_float");
  CHECK(floating != nullptr);
  if (floating == nullptr) {
    return false;
  }
  CHECK(floating->dtype == SafetensorsDType::kFloat32);
  CHECK(floating->shape == (std::vector<uint64_t>{2U}));
  CHECK(floating->data_begin == 0U);
  CHECK(floating->data_end == 8U);
  const auto floating_bytes = result.file.tensor_bytes(*floating);
  CHECK(floating_bytes.has_value());
  if (!floating_bytes.has_value()) {
    return false;
  }
  CHECK(floating_bytes->size() == 8U);
  CHECK((*floating_bytes)[2] == std::byte{0x80});

  const auto bfloat_bytes = result.file.tensor_bytes("a_bfloat");
  CHECK(bfloat_bytes.has_value());
  if (!bfloat_bytes.has_value()) {
    return false;
  }
  CHECK(bfloat_bytes->size() == 4U);
  CHECK((*bfloat_bytes)[0] == std::byte{0x80});
  CHECK(!result.file.tensor_bytes("missing").has_value());

  SafetensorsFile moved = std::move(result.file);
  CHECK(moved.is_open());
  CHECK(!result.file.is_open());
  CHECK(moved.tensor_bytes("z_float").has_value());
  return true;
}

[[nodiscard]] bool Utf8TensorNamesAreValidated() {
  TemporaryFile valid(L"-utf8-valid.safetensors");
  CHECK(!valid.path().empty());
  const std::string valid_header =
      "{\"tensor_\\u03b1\":{\"dtype\":\"U8\",\"shape\":[1],"
      "\"data_offsets\":[0,1]}}";
  const std::array<std::byte, 1> data{std::byte{0x2a}};
  CHECK(write_shard(valid.path(), valid_header, data));
  SafetensorsOpenResult valid_result = open_safetensors_file(valid.path());
  CHECK(valid_result.ok());
  CHECK(valid_result.file.find("tensor_\xce\xb1") != nullptr);

  TemporaryFile malformed(L"-utf8-malformed.safetensors");
  CHECK(!malformed.path().empty());
  std::string malformed_header = "{\"tensor_";
  malformed_header.push_back(static_cast<char>(0xc0U));
  malformed_header.push_back(static_cast<char>(0x80U));
  malformed_header +=
      "\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
  CHECK(write_shard(malformed.path(), malformed_header, data));
  const SafetensorsOpenResult malformed_result =
      open_safetensors_file(malformed.path());
  CHECK(!malformed_result.ok());
  CHECK(malformed_result.error.code ==
        SafetensorsErrorCode::kTensorNameInvalid);
  return true;
}

[[nodiscard]] bool PackedSubByteDTypesUseCeilingByteSize() {
  TemporaryFile temporary(L"-packed.safetensors");
  CHECK(!temporary.path().empty());
  const std::string header =
      R"({"packed4":{"dtype":"U4","shape":[3],"data_offsets":[0,2]},"packed6":{"dtype":"F6_E2M3","shape":[3],"data_offsets":[2,5]}})";
  const std::array<std::byte, 5> data{};
  CHECK(write_shard(temporary.path(), header, data));
  SafetensorsOpenResult result = open_safetensors_file(temporary.path());
  CHECK(result.ok());
  const SafetensorsTensorInfo *packed4 = result.file.find("packed4");
  const SafetensorsTensorInfo *packed6 = result.file.find("packed6");
  CHECK(packed4 != nullptr);
  CHECK(packed6 != nullptr);
  if (packed4 == nullptr || packed6 == nullptr) {
    return false;
  }
  CHECK(packed4->byte_size() == 2U);
  CHECK(packed6->byte_size() == 3U);
  return true;
}

[[nodiscard]] bool MalformedTensorSizeFailsClosed() {
  TemporaryFile temporary(L"-size.safetensors");
  CHECK(!temporary.path().empty());
  const std::string header =
      R"({"bad":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})";
  const std::array<std::byte, 4> data{};
  CHECK(write_shard(temporary.path(), header, data));

  const SafetensorsOpenResult result = open_safetensors_file(temporary.path());
  CHECK(!result.ok());
  CHECK(!result.file.is_open());
  CHECK(result.error.code == SafetensorsErrorCode::kTensorByteSizeMismatch);
  CHECK(result.error.actual == 4U);
  CHECK(result.error.required == 8U);
  return true;
}

[[nodiscard]] bool OverlapAndHoleFailClosed() {
  TemporaryFile overlap(L"-overlap.safetensors");
  CHECK(!overlap.path().empty());
  const std::string overlap_header =
      R"({"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}})";
  const std::array<std::byte, 3> overlap_data{};
  CHECK(write_shard(overlap.path(), overlap_header, overlap_data));
  const SafetensorsOpenResult overlap_result =
      open_safetensors_file(overlap.path());
  CHECK(overlap_result.error.code == SafetensorsErrorCode::kDataRangeOverlap);

  TemporaryFile hole(L"-hole.safetensors");
  CHECK(!hole.path().empty());
  const std::string hole_header =
      R"({"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}})";
  const std::array<std::byte, 3> hole_data{};
  CHECK(write_shard(hole.path(), hole_header, hole_data));
  const SafetensorsOpenResult hole_result = open_safetensors_file(hole.path());
  CHECK(hole_result.error.code == SafetensorsErrorCode::kDataRangeHole);
  return true;
}

[[nodiscard]] std::filesystem::path
target_checkpoint_root(std::wstring_view requested_root) {
  return requested_root.empty()
             ? std::filesystem::path(
                   LR"(C:\Users\Daniel\models\Qwen3.8-27B-NVFP4-RadixArk-AttnNVFP4)")
             : std::filesystem::path(requested_root);
}

[[nodiscard]] bool
RealTargetLmHeadShardMatchesCheckpoint(std::wstring_view requested_root) {
  const std::filesystem::path path = target_checkpoint_root(requested_root) /
                                     L"model-00001-of-00010.safetensors";
  if (!std::filesystem::exists(path)) {
    std::printf("[  SKIPPED ] real target shard not present: %ls\n",
                path.c_str());
    return true;
  }

  SafetensorsOpenResult result = open_safetensors_file(path.native());
  if (!result.ok()) {
    std::printf(
        "real shard error=%.*s system=%lu offset=%llu actual=%llu "
        "required=%llu\n",
        static_cast<int>(safetensors_error_code_name(result.error.code).size()),
        safetensors_error_code_name(result.error.code).data(),
        static_cast<unsigned long>(result.error.system_error),
        static_cast<unsigned long long>(result.error.byte_offset),
        static_cast<unsigned long long>(result.error.actual),
        static_cast<unsigned long long>(result.error.required));
  }
  CHECK(result.ok());
  CHECK(result.file.file_size() == 715161976ULL);
  CHECK(result.file.header_size() == 360ULL);
  CHECK(result.file.data_offset() == 368ULL);
  CHECK(result.file.data_size() == 715161608ULL);
  CHECK(result.file.tensors().size() == 4U);

  const SafetensorsTensorInfo *input_scale =
      result.file.find("lm_head.input_scale");
  CHECK(input_scale != nullptr);
  if (input_scale == nullptr) {
    return false;
  }
  CHECK(input_scale->dtype == SafetensorsDType::kFloat32);
  CHECK(input_scale->shape.empty());
  CHECK(input_scale->data_begin == 0U && input_scale->data_end == 4U);

  const SafetensorsTensorInfo *weight_scale =
      result.file.find("lm_head.weight_scale");
  CHECK(weight_scale != nullptr);
  if (weight_scale == nullptr) {
    return false;
  }
  CHECK(weight_scale->dtype == SafetensorsDType::kFloat8E4M3);
  CHECK(weight_scale->shape == (std::vector<uint64_t>{248320U, 320U}));
  CHECK(weight_scale->data_begin == 8ULL);
  CHECK(weight_scale->data_end == 79462408ULL);
  CHECK(weight_scale->byte_size() == 79462400ULL);

  const SafetensorsTensorInfo *weight = result.file.find("lm_head.weight");
  CHECK(weight != nullptr);
  if (weight == nullptr) {
    return false;
  }
  CHECK(weight->dtype == SafetensorsDType::kUInt8);
  CHECK(weight->shape == (std::vector<uint64_t>{248320U, 2560U}));
  CHECK(weight->data_begin == 79462408ULL);
  CHECK(weight->data_end == 715161608ULL);
  CHECK(weight->byte_size() == 635699200ULL);
  const auto weight_bytes = result.file.tensor_bytes(*weight);
  CHECK(weight_bytes.has_value());
  if (!weight_bytes.has_value()) {
    return false;
  }
  CHECK(weight_bytes->size() == 635699200ULL);
  CHECK(safetensors_dtype_name(weight->dtype) == "U8");
  return true;
}

[[nodiscard]] bool
RealTargetCheckpointAllShardsValidate(std::wstring_view requested_root) {
  struct ExpectedShard final {
    std::wstring_view name;
    uint64_t file_bytes;
    uint64_t tensor_bytes;
    size_t tensor_count;
  };
  constexpr std::array expected{
      ExpectedShard{L"model-00001-of-00010.safetensors", 715161976ULL,
                    715161608ULL, 4U},
      ExpectedShard{L"model-00002-of-00010.safetensors", 2542796952ULL,
                    2542796800ULL, 1U},
      ExpectedShard{L"model-00003-of-00010.safetensors", 2144250468ULL,
                    2144210924ULL, 317U},
      ExpectedShard{L"model-00004-of-00010.safetensors", 2105267072ULL,
                    2105227248ULL, 319U},
      ExpectedShard{L"model-00005-of-00010.safetensors", 2145745552ULL,
                    2145706608ULL, 313U},
      ExpectedShard{L"model-00006-of-00010.safetensors", 2139992440ULL,
                    2139951616ULL, 327U},
      ExpectedShard{L"model-00007-of-00010.safetensors", 2144260752ULL,
                    2144221160ULL, 317U},
      ExpectedShard{L"model-00008-of-00010.safetensors", 2105267072ULL,
                    2105227248ULL, 319U},
      ExpectedShard{L"model-00009-of-00010.safetensors", 1978234324ULL,
                    1978180156ULL, 472U},
      ExpectedShard{L"model-00010-of-00010.safetensors", 744532384ULL,
                    744530944ULL, 13U},
  };
  const std::filesystem::path root = target_checkpoint_root(requested_root);
  if (!std::filesystem::exists(root / expected[0].name)) {
    std::printf("[  SKIPPED ] real target checkpoint not present: %ls\n",
                root.c_str());
    return true;
  }

  uint64_t total_tensor_bytes = 0;
  size_t total_tensor_count = 0;
  for (const ExpectedShard &expected_shard : expected) {
    const std::filesystem::path path = root / expected_shard.name;
    SafetensorsOpenResult result = open_safetensors_file(path.native());
    if (!result.ok()) {
      std::printf("shard %ls error=%.*s system=%lu offset=%llu actual=%llu "
                  "required=%llu\n",
                  path.c_str(),
                  static_cast<int>(
                      safetensors_error_code_name(result.error.code).size()),
                  safetensors_error_code_name(result.error.code).data(),
                  static_cast<unsigned long>(result.error.system_error),
                  static_cast<unsigned long long>(result.error.byte_offset),
                  static_cast<unsigned long long>(result.error.actual),
                  static_cast<unsigned long long>(result.error.required));
    }
    CHECK(result.ok());
    CHECK(result.file.file_size() == expected_shard.file_bytes);
    CHECK(result.file.data_size() == expected_shard.tensor_bytes);
    CHECK(result.file.tensors().size() == expected_shard.tensor_count);
    total_tensor_bytes += result.file.data_size();
    total_tensor_count += result.file.tensors().size();
  }
  CHECK(total_tensor_bytes == 18765214312ULL);
  CHECK(total_tensor_count == 2402U);
  return true;
}

static_assert(!std::is_copy_constructible_v<SafetensorsFile>);
static_assert(!std::is_copy_assignable_v<SafetensorsFile>);
static_assert(std::is_nothrow_move_constructible_v<SafetensorsFile>);
static_assert(std::is_nothrow_move_assignable_v<SafetensorsFile>);

} // namespace

int wmain(int argc, wchar_t **argv) {
  const std::wstring_view target_root = argc > 1 ? argv[1] : L"";
  const std::array tests{
      std::pair{"valid synthetic shard maps zero-copy",
                &ValidSyntheticShardMapsZeroCopy},
      std::pair{"UTF-8 tensor names are validated",
                &Utf8TensorNamesAreValidated},
      std::pair{"packed sub-byte dtypes use ceiling byte size",
                &PackedSubByteDTypesUseCeilingByteSize},
      std::pair{"malformed tensor size fails closed",
                &MalformedTensorSizeFailsClosed},
      std::pair{"overlap and hole fail closed", &OverlapAndHoleFailClosed},
  };

  size_t passed = 0;
  for (const auto &[name, test] : tests) {
    if (!test()) {
      std::printf("[  FAILED  ] %s\n", name);
      return 1;
    }
    ++passed;
    std::printf("[       OK ] %s\n", name);
  }
  if (!RealTargetLmHeadShardMatchesCheckpoint(target_root)) {
    std::printf("[  FAILED  ] real target LM-head shard matches checkpoint\n");
    return 1;
  }
  ++passed;
  std::printf("[       OK ] real target LM-head shard matches checkpoint\n");
  if (!RealTargetCheckpointAllShardsValidate(target_root)) {
    std::printf("[  FAILED  ] all real target shards validate\n");
    return 1;
  }
  ++passed;
  std::printf("[       OK ] all real target shards validate\n");
  std::printf("[  PASSED  ] %zu tests\n", passed);
  return 0;
}
