#include "sglang/native/safetensors_file.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#include <algorithm>
#include <charconv>
#include <limits>
#include <new>
#include <system_error>
#include <utility>

namespace sglang::native {
namespace {

constexpr uint64_t kSafetensorsPrefixBytes = sizeof(uint64_t);
constexpr uint64_t kMaximumRank = 32;

[[nodiscard]] constexpr SafetensorsError
make_error(SafetensorsErrorCode code, uint64_t byte_offset = 0,
           uint64_t actual = 0, uint64_t required = 0,
           uint32_t system_error = 0) noexcept {
  return SafetensorsError{code, system_error, byte_offset, actual, required};
}

[[nodiscard]] constexpr uint64_t load_le_u64(const std::byte *source) noexcept {
  uint64_t value = 0;
  for (uint32_t index = 0; index < 8U; ++index) {
    value |= static_cast<uint64_t>(std::to_integer<uint8_t>(source[index]))
             << (index * 8U);
  }
  return value;
}

[[nodiscard]] constexpr bool checked_add(uint64_t left, uint64_t right,
                                         uint64_t &result) noexcept {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_multiply(uint64_t left, uint64_t right,
                                              uint64_t &result) noexcept {
  if (left != 0U && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] constexpr bool is_json_whitespace(char value) noexcept {
  return value == ' ' || value == '\t' || value == '\r' || value == '\n';
}

[[nodiscard]] constexpr bool is_hex(char value) noexcept {
  return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f') ||
         (value >= 'A' && value <= 'F');
}

[[nodiscard]] constexpr uint32_t hex_value(char value) noexcept {
  if (value >= '0' && value <= '9') {
    return static_cast<uint32_t>(value - '0');
  }
  if (value >= 'a' && value <= 'f') {
    return static_cast<uint32_t>(value - 'a' + 10);
  }
  return static_cast<uint32_t>(value - 'A' + 10);
}

[[nodiscard]] bool append_utf8(std::string &output, uint32_t scalar) {
  if (scalar <= 0x7fU) {
    output.push_back(static_cast<char>(scalar));
    return true;
  }
  if (scalar <= 0x7ffU) {
    output.push_back(static_cast<char>(0xc0U | (scalar >> 6U)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  if (scalar >= 0xd800U && scalar <= 0xdfffU) {
    return false;
  }
  if (scalar <= 0xffffU) {
    output.push_back(static_cast<char>(0xe0U | (scalar >> 12U)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  if (scalar <= 0x10ffffU) {
    output.push_back(static_cast<char>(0xf0U | (scalar >> 18U)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 12U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | ((scalar >> 6U) & 0x3fU)));
    output.push_back(static_cast<char>(0x80U | (scalar & 0x3fU)));
    return true;
  }
  return false;
}

[[nodiscard]] SafetensorsDType parse_dtype(std::string_view name) noexcept {
  if (name == "BOOL") {
    return SafetensorsDType::kBool;
  }
  if (name == "U4") {
    return SafetensorsDType::kUInt4;
  }
  if (name == "I4") {
    return SafetensorsDType::kInt4;
  }
  if (name == "U8") {
    return SafetensorsDType::kUInt8;
  }
  if (name == "I8") {
    return SafetensorsDType::kInt8;
  }
  if (name == "U16") {
    return SafetensorsDType::kUInt16;
  }
  if (name == "I16") {
    return SafetensorsDType::kInt16;
  }
  if (name == "U32") {
    return SafetensorsDType::kUInt32;
  }
  if (name == "I32") {
    return SafetensorsDType::kInt32;
  }
  if (name == "U64") {
    return SafetensorsDType::kUInt64;
  }
  if (name == "I64") {
    return SafetensorsDType::kInt64;
  }
  if (name == "F4") {
    return SafetensorsDType::kFloat4;
  }
  if (name == "F6_E2M3") {
    return SafetensorsDType::kFloat6E2M3;
  }
  if (name == "F6_E3M2") {
    return SafetensorsDType::kFloat6E3M2;
  }
  if (name == "F8_E4M3" || name == "F8_E4M3FN") {
    return SafetensorsDType::kFloat8E4M3;
  }
  if (name == "F8_E5M2") {
    return SafetensorsDType::kFloat8E5M2;
  }
  if (name == "F8_E8M0") {
    return SafetensorsDType::kFloat8E8M0;
  }
  if (name == "F16") {
    return SafetensorsDType::kFloat16;
  }
  if (name == "BF16") {
    return SafetensorsDType::kBFloat16;
  }
  if (name == "F32") {
    return SafetensorsDType::kFloat32;
  }
  if (name == "F64") {
    return SafetensorsDType::kFloat64;
  }
  if (name == "C64") {
    return SafetensorsDType::kComplex64;
  }
  if (name == "C128") {
    return SafetensorsDType::kComplex128;
  }
  return SafetensorsDType::kInvalid;
}

class HeaderParser final {
public:
  HeaderParser(std::string_view source,
               std::vector<SafetensorsTensorInfo> &tensors)
      : source_(source), tensors_(tensors) {}

  [[nodiscard]] SafetensorsError parse() {
    skip_whitespace();
    if (!consume('{')) {
      return syntax_error();
    }
    skip_whitespace();
    if (consume('}')) {
      skip_whitespace();
      return at_end() ? make_error(SafetensorsErrorCode::kOk) : syntax_error();
    }

    for (;;) {
      std::string name;
      if (!parse_string(name) || name.empty()) {
        return make_error(SafetensorsErrorCode::kTensorNameInvalid, position_);
      }
      skip_whitespace();
      if (!consume(':')) {
        return syntax_error();
      }
      skip_whitespace();

      if (name == "__metadata__") {
        if (metadata_seen_) {
          return make_error(SafetensorsErrorCode::kJsonDuplicateKey, position_);
        }
        metadata_seen_ = true;
        if (!skip_metadata_object()) {
          return make_error(SafetensorsErrorCode::kMetadataInvalid, position_);
        }
      } else {
        SafetensorsTensorInfo tensor;
        tensor.name = std::move(name);
        const SafetensorsError status = parse_tensor(tensor);
        if (status.code != SafetensorsErrorCode::kOk) {
          return status;
        }
        tensors_.push_back(std::move(tensor));
      }

      skip_whitespace();
      if (consume('}')) {
        break;
      }
      if (!consume(',')) {
        return syntax_error();
      }
      skip_whitespace();
    }
    skip_whitespace();
    return at_end() ? make_error(SafetensorsErrorCode::kOk) : syntax_error();
  }

private:
  [[nodiscard]] SafetensorsError syntax_error() const noexcept {
    return make_error(SafetensorsErrorCode::kJsonSyntax, position_);
  }

  [[nodiscard]] bool at_end() const noexcept {
    return position_ == source_.size();
  }

  void skip_whitespace() noexcept {
    while (!at_end() && is_json_whitespace(source_[position_])) {
      ++position_;
    }
  }

  [[nodiscard]] bool consume(char expected) noexcept {
    if (at_end() || source_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  [[nodiscard]] bool parse_hex4(uint32_t &value) noexcept {
    if (source_.size() - position_ < 4U) {
      return false;
    }
    value = 0;
    for (uint32_t index = 0; index < 4U; ++index) {
      const char digit = source_[position_++];
      if (!is_hex(digit)) {
        return false;
      }
      value = (value << 4U) | hex_value(digit);
    }
    return true;
  }

  [[nodiscard]] bool parse_string(std::string &output) {
    if (!consume('"')) {
      return false;
    }
    output.clear();
    while (!at_end()) {
      const unsigned char value =
          static_cast<unsigned char>(source_[position_++]);
      if (value == '"') {
        return true;
      }
      if (value < 0x20U) {
        return false;
      }
      if (value != '\\') {
        if (value >= 0x80U) {
          size_t sequence_bytes = 0;
          uint32_t scalar = 0;
          uint32_t minimum = 0;
          if (value >= 0xc2U && value <= 0xdfU) {
            sequence_bytes = 2U;
            scalar = value & 0x1fU;
            minimum = 0x80U;
          } else if (value >= 0xe0U && value <= 0xefU) {
            sequence_bytes = 3U;
            scalar = value & 0x0fU;
            minimum = 0x800U;
          } else if (value >= 0xf0U && value <= 0xf4U) {
            sequence_bytes = 4U;
            scalar = value & 0x07U;
            minimum = 0x10000U;
          } else {
            return false;
          }
          if (source_.size() - (position_ - 1U) < sequence_bytes) {
            return false;
          }
          for (size_t index = 1U; index < sequence_bytes; ++index) {
            const unsigned char continuation =
                static_cast<unsigned char>(source_[position_++]);
            if ((continuation & 0xc0U) != 0x80U) {
              return false;
            }
            scalar = (scalar << 6U) | (continuation & 0x3fU);
          }
          if (scalar < minimum || scalar > 0x10ffffU ||
              (scalar >= 0xd800U && scalar <= 0xdfffU)) {
            return false;
          }
          output.append(
              source_.substr(position_ - sequence_bytes, sequence_bytes));
          continue;
        }
        output.push_back(static_cast<char>(value));
        continue;
      }
      if (at_end()) {
        return false;
      }
      const char escape = source_[position_++];
      switch (escape) {
      case '"':
      case '\\':
      case '/':
        output.push_back(escape);
        break;
      case 'b':
        output.push_back('\b');
        break;
      case 'f':
        output.push_back('\f');
        break;
      case 'n':
        output.push_back('\n');
        break;
      case 'r':
        output.push_back('\r');
        break;
      case 't':
        output.push_back('\t');
        break;
      case 'u': {
        uint32_t scalar = 0;
        if (!parse_hex4(scalar)) {
          return false;
        }
        if (scalar >= 0xd800U && scalar <= 0xdbffU) {
          if (source_.size() - position_ < 6U || source_[position_] != '\\' ||
              source_[position_ + 1U] != 'u') {
            return false;
          }
          position_ += 2U;
          uint32_t low = 0;
          if (!parse_hex4(low) || low < 0xdc00U || low > 0xdfffU) {
            return false;
          }
          scalar = 0x10000U + ((scalar - 0xd800U) << 10U) + (low - 0xdc00U);
        }
        if (!append_utf8(output, scalar)) {
          return false;
        }
        break;
      }
      default:
        return false;
      }
    }
    return false;
  }

  [[nodiscard]] bool parse_uint64(uint64_t &value) noexcept {
    if (at_end() || source_[position_] < '0' || source_[position_] > '9') {
      return false;
    }
    const size_t begin = position_;
    if (source_[position_] == '0') {
      ++position_;
      if (!at_end() && source_[position_] >= '0' && source_[position_] <= '9') {
        return false;
      }
    } else {
      while (!at_end() && source_[position_] >= '0' &&
             source_[position_] <= '9') {
        ++position_;
      }
    }
    const char *first = source_.data() + begin;
    const char *last = source_.data() + position_;
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last;
  }

  [[nodiscard]] bool skip_metadata_object() {
    if (!consume('{')) {
      return false;
    }
    skip_whitespace();
    if (consume('}')) {
      return true;
    }
    std::vector<std::string> keys;
    for (;;) {
      std::string key;
      std::string value;
      if (!parse_string(key) ||
          std::find(keys.begin(), keys.end(), key) != keys.end()) {
        return false;
      }
      keys.push_back(std::move(key));
      skip_whitespace();
      if (!consume(':')) {
        return false;
      }
      skip_whitespace();
      if (!parse_string(value)) {
        return false;
      }
      skip_whitespace();
      if (consume('}')) {
        return true;
      }
      if (!consume(',')) {
        return false;
      }
      skip_whitespace();
    }
  }

  [[nodiscard]] bool parse_shape(std::vector<uint64_t> &shape) {
    if (!consume('[')) {
      return false;
    }
    skip_whitespace();
    if (consume(']')) {
      return true;
    }
    for (;;) {
      uint64_t dimension = 0;
      if (!parse_uint64(dimension)) {
        return false;
      }
      shape.push_back(dimension);
      if (shape.size() > kMaximumRank) {
        return false;
      }
      skip_whitespace();
      if (consume(']')) {
        return true;
      }
      if (!consume(',')) {
        return false;
      }
      skip_whitespace();
    }
  }

  [[nodiscard]] bool parse_offsets(uint64_t &begin, uint64_t &end) noexcept {
    if (!consume('[')) {
      return false;
    }
    skip_whitespace();
    if (!parse_uint64(begin)) {
      return false;
    }
    skip_whitespace();
    if (!consume(',')) {
      return false;
    }
    skip_whitespace();
    if (!parse_uint64(end)) {
      return false;
    }
    skip_whitespace();
    return consume(']');
  }

  [[nodiscard]] SafetensorsError parse_tensor(SafetensorsTensorInfo &tensor) {
    if (!consume('{')) {
      return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                        position_);
    }
    bool dtype_seen = false;
    bool shape_seen = false;
    bool offsets_seen = false;
    skip_whitespace();
    if (consume('}')) {
      return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                        position_);
    }
    for (;;) {
      std::string key;
      if (!parse_string(key)) {
        return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                          position_);
      }
      skip_whitespace();
      if (!consume(':')) {
        return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                          position_);
      }
      skip_whitespace();
      if (key == "dtype") {
        if (dtype_seen) {
          return make_error(SafetensorsErrorCode::kJsonDuplicateKey, position_);
        }
        dtype_seen = true;
        std::string dtype_name;
        if (!parse_string(dtype_name)) {
          return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                            position_);
        }
        tensor.dtype = parse_dtype(dtype_name);
        if (tensor.dtype == SafetensorsDType::kInvalid) {
          return make_error(SafetensorsErrorCode::kDTypeUnsupported, position_);
        }
      } else if (key == "shape") {
        if (shape_seen) {
          return make_error(SafetensorsErrorCode::kJsonDuplicateKey, position_);
        }
        shape_seen = true;
        if (!parse_shape(tensor.shape)) {
          return make_error(SafetensorsErrorCode::kShapeInvalid, position_);
        }
      } else if (key == "data_offsets") {
        if (offsets_seen) {
          return make_error(SafetensorsErrorCode::kJsonDuplicateKey, position_);
        }
        offsets_seen = true;
        if (!parse_offsets(tensor.data_begin, tensor.data_end)) {
          return make_error(SafetensorsErrorCode::kDataOffsetsInvalid,
                            position_);
        }
      } else {
        return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                          position_);
      }
      skip_whitespace();
      if (consume('}')) {
        break;
      }
      if (!consume(',')) {
        return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                          position_);
      }
      skip_whitespace();
    }
    if (!dtype_seen || !shape_seen || !offsets_seen) {
      return make_error(SafetensorsErrorCode::kTensorDescriptorInvalid,
                        position_);
    }
    return make_error(SafetensorsErrorCode::kOk);
  }

  std::string_view source_;
  std::vector<SafetensorsTensorInfo> &tensors_;
  size_t position_{0};
  bool metadata_seen_{false};
};

[[nodiscard]] SafetensorsError
validate_tensors(std::vector<SafetensorsTensorInfo> &tensors,
                 uint64_t data_size) {
  std::sort(tensors.begin(), tensors.end(),
            [](const SafetensorsTensorInfo &left,
               const SafetensorsTensorInfo &right) noexcept {
              return left.name < right.name;
            });
  for (size_t index = 1; index < tensors.size(); ++index) {
    if (tensors[index - 1U].name == tensors[index].name) {
      return make_error(SafetensorsErrorCode::kJsonDuplicateKey);
    }
  }

  for (const SafetensorsTensorInfo &tensor : tensors) {
    if (tensor.data_begin > tensor.data_end || tensor.data_end > data_size) {
      return make_error(SafetensorsErrorCode::kDataOffsetsInvalid, 0,
                        tensor.data_end, data_size);
    }
    uint64_t elements = 1;
    for (const uint64_t dimension : tensor.shape) {
      if (!checked_multiply(elements, dimension, elements)) {
        return make_error(SafetensorsErrorCode::kIntegerOverflow);
      }
    }
    const uint32_t bits = safetensors_dtype_bits(tensor.dtype);
    uint64_t total_bits = 0;
    if (bits == 0U ||
        !checked_multiply(elements, static_cast<uint64_t>(bits), total_bits)) {
      return make_error(SafetensorsErrorCode::kIntegerOverflow);
    }
    uint64_t expected_bytes = 0;
    if (!checked_add(total_bits, 7U, expected_bytes)) {
      return make_error(SafetensorsErrorCode::kIntegerOverflow);
    }
    expected_bytes /= 8U;
    if (tensor.byte_size() != expected_bytes) {
      return make_error(SafetensorsErrorCode::kTensorByteSizeMismatch, 0,
                        tensor.byte_size(), expected_bytes);
    }
  }

  std::vector<const SafetensorsTensorInfo *> by_offset;
  by_offset.reserve(tensors.size());
  for (const SafetensorsTensorInfo &tensor : tensors) {
    by_offset.push_back(&tensor);
  }
  std::sort(by_offset.begin(), by_offset.end(),
            [](const SafetensorsTensorInfo *left,
               const SafetensorsTensorInfo *right) noexcept {
              if (left->data_begin != right->data_begin) {
                return left->data_begin < right->data_begin;
              }
              return left->data_end < right->data_end;
            });
  uint64_t cursor = 0;
  for (const SafetensorsTensorInfo *tensor : by_offset) {
    if (tensor->data_begin < cursor) {
      return make_error(SafetensorsErrorCode::kDataRangeOverlap, 0,
                        tensor->data_begin, cursor);
    }
    if (tensor->data_begin > cursor) {
      return make_error(SafetensorsErrorCode::kDataRangeHole, 0,
                        tensor->data_begin, cursor);
    }
    cursor = tensor->data_end;
  }
  if (cursor != data_size) {
    return make_error(SafetensorsErrorCode::kDataNotFullyCovered, 0, cursor,
                      data_size);
  }
  return make_error(SafetensorsErrorCode::kOk);
}

} // namespace

SafetensorsFile::~SafetensorsFile() noexcept { close(); }

SafetensorsFile::SafetensorsFile(SafetensorsFile &&other) noexcept
    : file_handle_(std::exchange(other.file_handle_, nullptr)),
      mapping_handle_(std::exchange(other.mapping_handle_, nullptr)),
      mapping_(std::exchange(other.mapping_, nullptr)),
      file_size_(std::exchange(other.file_size_, 0)),
      header_size_(std::exchange(other.header_size_, 0)),
      data_offset_(std::exchange(other.data_offset_, 0)),
      tensors_(std::move(other.tensors_)) {}

SafetensorsFile &SafetensorsFile::operator=(SafetensorsFile &&other) noexcept {
  if (this != &other) {
    close();
    file_handle_ = std::exchange(other.file_handle_, nullptr);
    mapping_handle_ = std::exchange(other.mapping_handle_, nullptr);
    mapping_ = std::exchange(other.mapping_, nullptr);
    file_size_ = std::exchange(other.file_size_, 0);
    header_size_ = std::exchange(other.header_size_, 0);
    data_offset_ = std::exchange(other.data_offset_, 0);
    tensors_ = std::move(other.tensors_);
  }
  return *this;
}

bool SafetensorsFile::is_open() const noexcept {
  return file_handle_ != nullptr && mapping_handle_ != nullptr &&
         mapping_ != nullptr;
}

const SafetensorsTensorInfo *
SafetensorsFile::find(std::string_view name) const noexcept {
  const auto found = std::lower_bound(
      tensors_.begin(), tensors_.end(), name,
      [](const SafetensorsTensorInfo &tensor,
         std::string_view wanted) noexcept { return tensor.name < wanted; });
  return found != tensors_.end() && found->name == name ? &*found : nullptr;
}

std::optional<std::span<const std::byte>> SafetensorsFile::tensor_bytes(
    const SafetensorsTensorInfo &tensor) const noexcept {
  if (!is_open() || tensor.data_begin > tensor.data_end ||
      tensor.data_end > data_size() ||
      tensor.byte_size() >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return std::nullopt;
  }
  const auto belongs = std::lower_bound(
      tensors_.begin(), tensors_.end(), tensor.name,
      [](const SafetensorsTensorInfo &entry, std::string_view wanted) noexcept {
        return entry.name < wanted;
      });
  if (belongs == tensors_.end() || &*belongs != &tensor) {
    return std::nullopt;
  }
  return std::span<const std::byte>(mapping_ + data_offset_ + tensor.data_begin,
                                    static_cast<size_t>(tensor.byte_size()));
}

std::optional<std::span<const std::byte>>
SafetensorsFile::tensor_bytes(std::string_view name) const noexcept {
  const SafetensorsTensorInfo *tensor = find(name);
  return tensor == nullptr ? std::nullopt : tensor_bytes(*tensor);
}

void SafetensorsFile::close() noexcept {
  tensors_.clear();
  if (mapping_ != nullptr) {
    static_cast<void>(UnmapViewOfFile(mapping_));
  }
  if (mapping_handle_ != nullptr) {
    static_cast<void>(CloseHandle(static_cast<HANDLE>(mapping_handle_)));
  }
  if (file_handle_ != nullptr) {
    static_cast<void>(CloseHandle(static_cast<HANDLE>(file_handle_)));
  }
  mapping_ = nullptr;
  mapping_handle_ = nullptr;
  file_handle_ = nullptr;
  file_size_ = 0;
  header_size_ = 0;
  data_offset_ = 0;
}

SafetensorsOpenResult open_safetensors_file(std::wstring_view path) noexcept {
  SafetensorsOpenResult result;
  if (path.empty() || path.find(L'\0') != std::wstring_view::npos) {
    result.error = make_error(SafetensorsErrorCode::kInvalidPath);
    return result;
  }

  try {
    const std::wstring null_terminated(path);
    HANDLE file =
        CreateFileW(null_terminated.c_str(), GENERIC_READ, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      result.error = make_error(SafetensorsErrorCode::kOpenFailed, 0, 0, 0,
                                GetLastError());
      return result;
    }
    result.file.file_handle_ = file;

    LARGE_INTEGER file_size{};
    if (GetFileSizeEx(file, &file_size) == 0) {
      result.error = make_error(SafetensorsErrorCode::kFileSizeFailed, 0, 0, 0,
                                GetLastError());
      result.file.close();
      return result;
    }
    if (file_size.QuadPart < 0) {
      result.error = make_error(SafetensorsErrorCode::kFileTooLarge);
      result.file.close();
      return result;
    }
    result.file.file_size_ = static_cast<uint64_t>(file_size.QuadPart);
    if (result.file.file_size_ < kSafetensorsPrefixBytes + 2U) {
      result.error =
          make_error(SafetensorsErrorCode::kFileTooSmall, 0,
                     result.file.file_size_, kSafetensorsPrefixBytes + 2U);
      result.file.close();
      return result;
    }
    if (result.file.file_size_ >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      result.error = make_error(SafetensorsErrorCode::kFileTooLarge);
      result.file.close();
      return result;
    }

    HANDLE mapping =
        CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
      result.error = make_error(SafetensorsErrorCode::kMappingFailed, 0, 0, 0,
                                GetLastError());
      result.file.close();
      return result;
    }
    result.file.mapping_handle_ = mapping;
    const void *view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (view == nullptr) {
      result.error = make_error(SafetensorsErrorCode::kViewFailed, 0, 0, 0,
                                GetLastError());
      result.file.close();
      return result;
    }
    result.file.mapping_ = static_cast<const std::byte *>(view);
    result.file.header_size_ = load_le_u64(result.file.mapping_);
    if (result.file.header_size_ < 2U) {
      result.error = make_error(SafetensorsErrorCode::kHeaderLengthInvalid, 0,
                                result.file.header_size_, 2U);
      result.file.close();
      return result;
    }
    if (result.file.header_size_ > kSafetensorsMaximumHeaderBytes) {
      result.error =
          make_error(SafetensorsErrorCode::kHeaderTooLarge, 0,
                     result.file.header_size_, kSafetensorsMaximumHeaderBytes);
      result.file.close();
      return result;
    }
    if (!checked_add(kSafetensorsPrefixBytes, result.file.header_size_,
                     result.file.data_offset_) ||
        result.file.data_offset_ > result.file.file_size_) {
      result.error =
          make_error(SafetensorsErrorCode::kHeaderLengthInvalid, 0,
                     result.file.header_size_, result.file.file_size_);
      result.file.close();
      return result;
    }

    LARGE_INTEGER mapped_file_size{};
    if (GetFileSizeEx(file, &mapped_file_size) == 0) {
      result.error = make_error(SafetensorsErrorCode::kFileSizeFailed, 0, 0, 0,
                                GetLastError());
      result.file.close();
      return result;
    }
    if (mapped_file_size.QuadPart < 0 ||
        static_cast<uint64_t>(mapped_file_size.QuadPart) !=
            result.file.file_size_) {
      result.error =
          make_error(SafetensorsErrorCode::kFileChanged, 0,
                     mapped_file_size.QuadPart < 0
                         ? 0U
                         : static_cast<uint64_t>(mapped_file_size.QuadPart),
                     result.file.file_size_);
      result.file.close();
      return result;
    }

    const std::string_view header(
        reinterpret_cast<const char *>(result.file.mapping_ +
                                       kSafetensorsPrefixBytes),
        static_cast<size_t>(result.file.header_size_));
    HeaderParser parser(header, result.file.tensors_);
    result.error = parser.parse();
    if (result.error.code == SafetensorsErrorCode::kOk) {
      result.error =
          validate_tensors(result.file.tensors_, result.file.data_size());
    }
    if (result.error.code != SafetensorsErrorCode::kOk) {
      result.file.close();
    }
  } catch (const std::bad_alloc &) {
    result.error = make_error(SafetensorsErrorCode::kAllocationFailed);
    result.file.close();
  } catch (...) {
    result.error = make_error(SafetensorsErrorCode::kInternalFailure);
    result.file.close();
  }
  return result;
}

std::string_view safetensors_dtype_name(SafetensorsDType dtype) noexcept {
  switch (dtype) {
  case SafetensorsDType::kBool:
    return "BOOL";
  case SafetensorsDType::kUInt4:
    return "U4";
  case SafetensorsDType::kInt4:
    return "I4";
  case SafetensorsDType::kUInt8:
    return "U8";
  case SafetensorsDType::kInt8:
    return "I8";
  case SafetensorsDType::kUInt16:
    return "U16";
  case SafetensorsDType::kInt16:
    return "I16";
  case SafetensorsDType::kUInt32:
    return "U32";
  case SafetensorsDType::kInt32:
    return "I32";
  case SafetensorsDType::kUInt64:
    return "U64";
  case SafetensorsDType::kInt64:
    return "I64";
  case SafetensorsDType::kFloat4:
    return "F4";
  case SafetensorsDType::kFloat6E2M3:
    return "F6_E2M3";
  case SafetensorsDType::kFloat6E3M2:
    return "F6_E3M2";
  case SafetensorsDType::kFloat8E4M3:
    return "F8_E4M3";
  case SafetensorsDType::kFloat8E5M2:
    return "F8_E5M2";
  case SafetensorsDType::kFloat8E8M0:
    return "F8_E8M0";
  case SafetensorsDType::kFloat16:
    return "F16";
  case SafetensorsDType::kBFloat16:
    return "BF16";
  case SafetensorsDType::kFloat32:
    return "F32";
  case SafetensorsDType::kFloat64:
    return "F64";
  case SafetensorsDType::kComplex64:
    return "C64";
  case SafetensorsDType::kComplex128:
    return "C128";
  default:
    return "invalid";
  }
}

uint32_t safetensors_dtype_bits(SafetensorsDType dtype) noexcept {
  switch (dtype) {
  case SafetensorsDType::kBool:
  case SafetensorsDType::kUInt8:
  case SafetensorsDType::kInt8:
  case SafetensorsDType::kFloat8E4M3:
  case SafetensorsDType::kFloat8E5M2:
  case SafetensorsDType::kFloat8E8M0:
    return 8U;
  case SafetensorsDType::kUInt4:
  case SafetensorsDType::kInt4:
  case SafetensorsDType::kFloat4:
    return 4U;
  case SafetensorsDType::kFloat6E2M3:
  case SafetensorsDType::kFloat6E3M2:
    return 6U;
  case SafetensorsDType::kUInt16:
  case SafetensorsDType::kInt16:
  case SafetensorsDType::kFloat16:
  case SafetensorsDType::kBFloat16:
    return 16U;
  case SafetensorsDType::kUInt32:
  case SafetensorsDType::kInt32:
  case SafetensorsDType::kFloat32:
    return 32U;
  case SafetensorsDType::kUInt64:
  case SafetensorsDType::kInt64:
  case SafetensorsDType::kFloat64:
  case SafetensorsDType::kComplex64:
    return 64U;
  case SafetensorsDType::kComplex128:
    return 128U;
  default:
    return 0U;
  }
}

std::string_view
safetensors_error_code_name(SafetensorsErrorCode code) noexcept {
  switch (code) {
  case SafetensorsErrorCode::kOk:
    return "ok";
  case SafetensorsErrorCode::kInvalidPath:
    return "invalid_path";
  case SafetensorsErrorCode::kOpenFailed:
    return "open_failed";
  case SafetensorsErrorCode::kFileSizeFailed:
    return "file_size_failed";
  case SafetensorsErrorCode::kFileTooSmall:
    return "file_too_small";
  case SafetensorsErrorCode::kFileTooLarge:
    return "file_too_large";
  case SafetensorsErrorCode::kMappingFailed:
    return "mapping_failed";
  case SafetensorsErrorCode::kViewFailed:
    return "view_failed";
  case SafetensorsErrorCode::kHeaderLengthInvalid:
    return "header_length_invalid";
  case SafetensorsErrorCode::kHeaderTooLarge:
    return "header_too_large";
  case SafetensorsErrorCode::kJsonSyntax:
    return "json_syntax";
  case SafetensorsErrorCode::kJsonDuplicateKey:
    return "json_duplicate_key";
  case SafetensorsErrorCode::kMetadataInvalid:
    return "metadata_invalid";
  case SafetensorsErrorCode::kTensorNameInvalid:
    return "tensor_name_invalid";
  case SafetensorsErrorCode::kTensorDescriptorInvalid:
    return "tensor_descriptor_invalid";
  case SafetensorsErrorCode::kDTypeUnsupported:
    return "dtype_unsupported";
  case SafetensorsErrorCode::kShapeInvalid:
    return "shape_invalid";
  case SafetensorsErrorCode::kDataOffsetsInvalid:
    return "data_offsets_invalid";
  case SafetensorsErrorCode::kIntegerOverflow:
    return "integer_overflow";
  case SafetensorsErrorCode::kTensorByteSizeMismatch:
    return "tensor_byte_size_mismatch";
  case SafetensorsErrorCode::kDataRangeOverlap:
    return "data_range_overlap";
  case SafetensorsErrorCode::kDataRangeHole:
    return "data_range_hole";
  case SafetensorsErrorCode::kDataNotFullyCovered:
    return "data_not_fully_covered";
  case SafetensorsErrorCode::kFileChanged:
    return "file_changed";
  case SafetensorsErrorCode::kAllocationFailed:
    return "allocation_failed";
  case SafetensorsErrorCode::kInternalFailure:
    return "internal_failure";
  default:
    return "invalid_safetensors_error";
  }
}

} // namespace sglang::native
