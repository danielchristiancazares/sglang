#ifndef SGLANG_NATIVE_SAFETENSORS_FILE_HPP_
#define SGLANG_NATIVE_SAFETENSORS_FILE_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sglang::native {

struct SafetensorsOpenResult;

inline constexpr uint64_t kSafetensorsMaximumHeaderBytes =
    64ULL * 1024ULL * 1024ULL;

enum class SafetensorsDType : uint32_t {
  kInvalid = 0,
  kBool,
  kUInt4,
  kInt4,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kUInt32,
  kInt32,
  kUInt64,
  kInt64,
  kFloat4,
  kFloat6E2M3,
  kFloat6E3M2,
  kFloat8E4M3,
  kFloat8E5M2,
  kFloat8E8M0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
  kComplex64,
  kComplex128,
};

enum class SafetensorsErrorCode : uint32_t {
  kOk = 0,
  kInvalidPath,
  kOpenFailed,
  kFileSizeFailed,
  kFileTooSmall,
  kFileTooLarge,
  kMappingFailed,
  kViewFailed,
  kHeaderLengthInvalid,
  kHeaderTooLarge,
  kJsonSyntax,
  kJsonDuplicateKey,
  kMetadataInvalid,
  kTensorNameInvalid,
  kTensorDescriptorInvalid,
  kDTypeUnsupported,
  kShapeInvalid,
  kDataOffsetsInvalid,
  kIntegerOverflow,
  kTensorByteSizeMismatch,
  kDataRangeOverlap,
  kDataRangeHole,
  kDataNotFullyCovered,
  kFileChanged,
  kAllocationFailed,
  kInternalFailure,
};

struct SafetensorsError final {
  SafetensorsErrorCode code{SafetensorsErrorCode::kOk};
  uint32_t system_error{0};
  uint64_t byte_offset{0};
  uint64_t actual{0};
  uint64_t required{0};
};

struct SafetensorsTensorInfo final {
  std::string name;
  SafetensorsDType dtype{SafetensorsDType::kInvalid};
  std::vector<uint64_t> shape;
  uint64_t data_begin{0};
  uint64_t data_end{0};

  [[nodiscard]] uint64_t byte_size() const noexcept {
    return data_end - data_begin;
  }
};

class SafetensorsFile final {
public:
  SafetensorsFile() noexcept = default;
  ~SafetensorsFile() noexcept;

  SafetensorsFile(const SafetensorsFile &) = delete;
  SafetensorsFile &operator=(const SafetensorsFile &) = delete;
  SafetensorsFile(SafetensorsFile &&other) noexcept;
  SafetensorsFile &operator=(SafetensorsFile &&other) noexcept;

  [[nodiscard]] bool is_open() const noexcept;
  [[nodiscard]] uint64_t file_size() const noexcept { return file_size_; }
  [[nodiscard]] uint64_t header_size() const noexcept { return header_size_; }
  [[nodiscard]] uint64_t data_offset() const noexcept { return data_offset_; }
  [[nodiscard]] uint64_t data_size() const noexcept {
    return data_offset_ <= file_size_ ? file_size_ - data_offset_ : 0U;
  }
  [[nodiscard]] std::span<const SafetensorsTensorInfo>
  tensors() const noexcept {
    return tensors_;
  }
  [[nodiscard]] std::span<const std::byte> file_bytes() const noexcept {
    return is_open() ? std::span<const std::byte>(
                           mapping_, static_cast<size_t>(file_size_))
                     : std::span<const std::byte>();
  }
  [[nodiscard]] const SafetensorsTensorInfo *
  find(std::string_view name) const noexcept;
  [[nodiscard]] std::optional<std::span<const std::byte>>
  tensor_bytes(const SafetensorsTensorInfo &tensor) const noexcept;
  [[nodiscard]] std::optional<std::span<const std::byte>>
  tensor_bytes(std::string_view name) const noexcept;

private:
  friend struct SafetensorsOpenResult;
  friend SafetensorsOpenResult
  open_safetensors_file(std::wstring_view path) noexcept;

  void close() noexcept;

  void *file_handle_{nullptr};
  void *mapping_handle_{nullptr};
  const std::byte *mapping_{nullptr};
  uint64_t file_size_{0};
  uint64_t header_size_{0};
  uint64_t data_offset_{0};
  std::vector<SafetensorsTensorInfo> tensors_;
};

struct SafetensorsOpenResult final {
  SafetensorsFile file;
  SafetensorsError error;

  [[nodiscard]] bool ok() const noexcept {
    return error.code == SafetensorsErrorCode::kOk && file.is_open();
  }
};

[[nodiscard]] std::string_view
safetensors_dtype_name(SafetensorsDType dtype) noexcept;
[[nodiscard]] uint32_t safetensors_dtype_bits(SafetensorsDType dtype) noexcept;
[[nodiscard]] std::string_view
safetensors_error_code_name(SafetensorsErrorCode code) noexcept;

// Maps one shard read-only and validates the complete safetensors envelope.
// Tensor ranges are checked for exact dtype/shape byte sizes and must form one
// contiguous, non-overlapping partition of the data section. Tensor lookup is
// a binary search over name-sorted metadata; returned byte spans alias the
// mapping and remain valid only while the SafetensorsFile remains alive.
[[nodiscard]] SafetensorsOpenResult
open_safetensors_file(std::wstring_view path) noexcept;

} // namespace sglang::native

#endif // SGLANG_NATIVE_SAFETENSORS_FILE_HPP_
