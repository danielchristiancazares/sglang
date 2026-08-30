#include "sglang/native/tensor_view.hpp"

#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

namespace sglang::native {
namespace {

constexpr uint32_t kNoDimension = std::numeric_limits<uint32_t>::max();
constexpr uint64_t kSignedDomainMaximum =
    static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

[[nodiscard]] constexpr uint64_t signed_bits(int64_t value) noexcept {
  return static_cast<uint64_t>(value);
}

[[nodiscard]] constexpr bool checked_add(uint64_t left, uint64_t right,
                                         uint64_t* result) noexcept {
  if (left > std::numeric_limits<uint64_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

[[nodiscard]] constexpr bool checked_multiply(uint64_t left, uint64_t right,
                                              uint64_t* result) noexcept {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

[[nodiscard]] constexpr TensorValidationError make_error(
    TensorValidationCode code, TensorValidationField field,
    uint32_t dimension, uint64_t actual, uint64_t required) noexcept {
  return TensorValidationError{code, field, dimension, 0, actual, required};
}

[[nodiscard]] constexpr bool is_known_dtype(SglNativeDType dtype) noexcept {
  return dtype >= SGL_NATIVE_DTYPE_BOOL8 &&
         dtype <= SGL_NATIVE_DTYPE_NVFP4_E2M1;
}

[[nodiscard]] constexpr bool is_known_device(
    SglNativeDeviceKind device) noexcept {
  return device >= SGL_NATIVE_DEVICE_CPU &&
         device <= SGL_NATIVE_DEVICE_CUDA_HOST;
}

template <typename Pointer>
struct ValidationResult final {
  SglNativeTensorMetadataV1 metadata;
  Pointer allocation_base;
  TensorValidationError error;
  bool valid;
};

template <typename Pointer>
[[nodiscard]] constexpr ValidationResult<Pointer> failed(
    TensorValidationError error) noexcept {
  return ValidationResult<Pointer>{make_tensor_metadata_v1(), nullptr, error,
                                   false};
}

template <typename RawView, typename Pointer>
[[nodiscard]] ValidationResult<Pointer> validate_impl(
    const RawView* view, bool mutable_access) noexcept {
  if (view == nullptr) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kNullView, TensorValidationField::kView,
        kNoDimension, 0, 1));
  }

  const SglNativeTensorMetadataV1 metadata = view->metadata;
  Pointer const allocation_base = view->allocation_base;

  if (metadata.struct_size != SGL_NATIVE_TENSOR_METADATA_V1_SIZE) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kMetadataSizeMismatch,
        TensorValidationField::kStructSize, kNoDimension,
        metadata.struct_size, SGL_NATIVE_TENSOR_METADATA_V1_SIZE));
  }
  if (metadata.abi_major != SGL_NATIVE_TENSOR_ABI_MAJOR ||
      metadata.abi_minor != SGL_NATIVE_TENSOR_ABI_MINOR) {
    const uint64_t actual =
        (static_cast<uint64_t>(metadata.abi_major) << 16U) |
        metadata.abi_minor;
    const uint64_t required =
        (static_cast<uint64_t>(SGL_NATIVE_TENSOR_ABI_MAJOR) << 16U) |
        SGL_NATIVE_TENSOR_ABI_MINOR;
    return failed<Pointer>(make_error(
        TensorValidationCode::kAbiVersionMismatch,
        TensorValidationField::kAbiVersion, kNoDimension, actual, required));
  }
  for (uint32_t index = 0; index < 2; ++index) {
    if (metadata.reserved[index] != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kReservedFieldNonZero,
          TensorValidationField::kReserved, index, metadata.reserved[index],
          0));
    }
  }
  if (!is_known_dtype(metadata.dtype)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kUnknownDType, TensorValidationField::kDType,
        kNoDimension, metadata.dtype, 0));
  }
  if (!is_known_device(metadata.device_kind)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kUnknownDevice,
        TensorValidationField::kDeviceKind, kNoDimension,
        metadata.device_kind, 0));
  }

  const bool ordinal_valid =
      (metadata.device_kind == SGL_NATIVE_DEVICE_CUDA &&
       metadata.device_ordinal >= 0) ||
      ((metadata.device_kind == SGL_NATIVE_DEVICE_CPU ||
        metadata.device_kind == SGL_NATIVE_DEVICE_CUDA_HOST) &&
       metadata.device_ordinal == 0);
  if (!ordinal_valid) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kInvalidDeviceOrdinal,
        TensorValidationField::kDeviceOrdinal, kNoDimension,
        signed_bits(metadata.device_ordinal), 0));
  }
  if (metadata.rank > SGL_NATIVE_TENSOR_MAX_RANK) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kRankOutOfRange, TensorValidationField::kRank,
        kNoDimension, metadata.rank, SGL_NATIVE_TENSOR_MAX_RANK));
  }
  for (uint32_t dimension = metadata.rank;
       dimension < SGL_NATIVE_TENSOR_MAX_RANK; ++dimension) {
    if (metadata.extents[dimension] != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kUnusedDimensionNonZero,
          TensorValidationField::kExtent, dimension,
          signed_bits(metadata.extents[dimension]), 0));
    }
    if (metadata.strides[dimension] != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kUnusedDimensionNonZero,
          TensorValidationField::kStride, dimension,
          signed_bits(metadata.strides[dimension]), 0));
    }
  }
  bool empty = false;
  uint32_t negative_stride_dimension = kNoDimension;
  for (uint32_t dimension = 0; dimension < metadata.rank; ++dimension) {
    if (metadata.extents[dimension] < 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kNegativeExtent,
          TensorValidationField::kExtent, dimension,
          signed_bits(metadata.extents[dimension]), 0));
    }
    empty = empty || metadata.extents[dimension] == 0;
    if (negative_stride_dimension == kNoDimension &&
        metadata.strides[dimension] < 0) {
      negative_stride_dimension = dimension;
    }
  }
  if (negative_stride_dimension != kNoDimension) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kNegativeStride,
        TensorValidationField::kStride, negative_stride_dimension,
        signed_bits(metadata.strides[negative_stride_dimension]), 0));
  }
  if (metadata.allocation_bytes > kSignedDomainMaximum) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kValueOutOfDomain,
        TensorValidationField::kAllocationBytes, kNoDimension,
        metadata.allocation_bytes, kSignedDomainMaximum));
  }
  if (metadata.storage_offset_elements > kSignedDomainMaximum) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kValueOutOfDomain,
        TensorValidationField::kStorageOffsetElements, kNoDimension,
        metadata.storage_offset_elements, kSignedDomainMaximum));
  }

  uint64_t capacity_bits = 0;
  if (!checked_multiply(metadata.allocation_bytes, uint64_t{8},
                        &capacity_bits)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kArithmeticOverflow,
        TensorValidationField::kAllocationBytes, kNoDimension, 0, 0));
  }
  const DType dtype = static_cast<DType>(metadata.dtype);
  const uint64_t element_bits = dtype_element_bits(dtype);
  uint64_t origin_bits = 0;
  if (!checked_multiply(metadata.storage_offset_elements, element_bits,
                        &origin_bits)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kArithmeticOverflow,
        TensorValidationField::kStorageOffsetElements, kNoDimension, 0, 0));
  }
  if (origin_bits % 8 != 0) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kMisalignedStorageOffset,
        TensorValidationField::kStorageOffsetElements, kNoDimension,
        origin_bits % 8, 0));
  }

  if (!empty && allocation_base == nullptr) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kNullAllocation,
        TensorValidationField::kAllocationBase, kNoDimension, 0, 1));
  }
  if (empty && metadata.allocation_bytes == 0) {
    if (allocation_base != nullptr) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kInvalidEmptyView,
          TensorValidationField::kAllocationBase, kNoDimension, 0, 0));
    }
    if (metadata.storage_offset_elements != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kInvalidEmptyView,
          TensorValidationField::kStorageOffsetElements, kNoDimension,
          metadata.storage_offset_elements, 0));
    }
  } else if (empty && allocation_base == nullptr) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kNullAllocation,
        TensorValidationField::kAllocationBase, kNoDimension, 0, 1));
  }

  if (allocation_base != nullptr) {
    const uintptr_t base_integer =
        reinterpret_cast<uintptr_t>(allocation_base);
    const uint64_t natural_alignment = dtype_natural_alignment(dtype);
    if (base_integer % natural_alignment != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kMisalignedAllocationBase,
          TensorValidationField::kAllocationBase, kNoDimension, 0,
          natural_alignment));
    }
    const uint64_t origin_bytes = origin_bits / 8;
    if (origin_bytes % natural_alignment != 0) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kMisalignedStorageOffset,
          TensorValidationField::kStorageOffsetElements, kNoDimension,
          origin_bytes % natural_alignment, 0));
    }
    if (metadata.allocation_bytes >
        static_cast<uint64_t>(std::numeric_limits<uintptr_t>::max()) ||
        base_integer > std::numeric_limits<uintptr_t>::max() -
                           static_cast<uintptr_t>(metadata.allocation_bytes)) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kPointerRangeOverflow,
          TensorValidationField::kAllocationBase, kNoDimension, 0, 0));
    }
  }

  if (empty) {
    if (origin_bits > capacity_bits) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kOutOfBounds, TensorValidationField::kSpan,
          kNoDimension, origin_bits, capacity_bits));
    }
    return ValidationResult<Pointer>{metadata, allocation_base, {}, true};
  }

  uint64_t max_offset_elements = 0;
  for (uint32_t dimension = 0; dimension < metadata.rank; ++dimension) {
    const uint64_t extent = static_cast<uint64_t>(metadata.extents[dimension]);
    const uint64_t stride = static_cast<uint64_t>(metadata.strides[dimension]);
    uint64_t contribution = 0;
    if (!checked_multiply(extent - 1, stride, &contribution)) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kArithmeticOverflow,
          TensorValidationField::kStride, dimension, 0, 0));
    }
    if (!checked_add(max_offset_elements, contribution,
                     &max_offset_elements)) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kArithmeticOverflow,
          TensorValidationField::kSpan, dimension, 0, 0));
    }
  }

  uint64_t occupied_elements = 0;
  if (!checked_add(max_offset_elements, 1, &occupied_elements)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kArithmeticOverflow,
        TensorValidationField::kSpan, kNoDimension, 0, 0));
  }
  uint64_t occupied_bits = 0;
  if (!checked_multiply(occupied_elements, element_bits, &occupied_bits)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kArithmeticOverflow,
        TensorValidationField::kSpan, kNoDimension, 0, 0));
  }
  uint64_t occupied_end_bits = 0;
  if (!checked_add(origin_bits, occupied_bits, &occupied_end_bits)) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kArithmeticOverflow,
        TensorValidationField::kSpan, kNoDimension, 0, 0));
  }
  if (occupied_end_bits > capacity_bits) {
    return failed<Pointer>(make_error(
        TensorValidationCode::kOutOfBounds, TensorValidationField::kSpan,
        kNoDimension, occupied_end_bits, capacity_bits));
  }

  if (mutable_access) {
    if (element_bits < 8) {
      return failed<Pointer>(make_error(
          TensorValidationCode::kMutableSubByteUnsupported,
          TensorValidationField::kMutableLayout, kNoDimension, element_bits,
          8));
    }

    struct MutableDimension final {
      uint64_t stride;
      uint64_t extent;
      uint32_t dimension;
    };
    MutableDimension active[SGL_NATIVE_TENSOR_MAX_RANK]{};
    uint32_t active_count = 0;
    for (uint32_t dimension = 0; dimension < metadata.rank; ++dimension) {
      const uint64_t extent =
          static_cast<uint64_t>(metadata.extents[dimension]);
      if (extent > 1) {
        active[active_count] = MutableDimension{
            static_cast<uint64_t>(metadata.strides[dimension]), extent,
            dimension};
        ++active_count;
      }
    }
    for (uint32_t index = 1; index < active_count; ++index) {
      const MutableDimension entry = active[index];
      uint32_t position = index;
      while (position > 0 &&
             (entry.stride < active[position - 1].stride ||
              (entry.stride == active[position - 1].stride &&
               entry.dimension < active[position - 1].dimension))) {
        active[position] = active[position - 1];
        --position;
      }
      active[position] = entry;
    }

    uint64_t required_span = 1;
    for (uint32_t index = 0; index < active_count; ++index) {
      const MutableDimension entry = active[index];
      if (entry.stride < required_span) {
        return failed<Pointer>(make_error(
            TensorValidationCode::kMutableOverlap,
            TensorValidationField::kMutableLayout, entry.dimension,
            entry.stride, required_span));
      }
      uint64_t contribution = 0;
      if (!checked_multiply(entry.extent - 1, entry.stride,
                            &contribution) ||
          !checked_add(required_span, contribution, &required_span)) {
        return failed<Pointer>(make_error(
            TensorValidationCode::kArithmeticOverflow,
            TensorValidationField::kMutableLayout, entry.dimension, 0, 0));
      }
    }
  }

  return ValidationResult<Pointer>{metadata, allocation_base, {}, true};
}

class BoundedWriter final {
 public:
  explicit BoundedWriter(std::span<char> destination) noexcept
      : destination_(destination) {}

  void append(std::string_view text) noexcept {
    for (const char character : text) {
      append(character);
    }
  }

  void append(char character) noexcept {
    if (!destination_.empty() && required_ + 1 < destination_.size()) {
      destination_[required_] = character;
    }
    ++required_;
  }

  void append_unsigned(uint64_t value) noexcept {
    char digits[20]{};
    std::size_t count = 0;
    do {
      digits[count] = static_cast<char>('0' + value % 10);
      ++count;
      value /= 10;
    } while (value != 0);
    while (count != 0) {
      --count;
      append(digits[count]);
    }
  }

  void append_signed_bits(uint64_t bits) noexcept {
    constexpr uint64_t kSignBit = uint64_t{1} << 63U;
    if ((bits & kSignBit) != 0) {
      append('-');
      append_unsigned(~bits + 1);
      return;
    }
    append_unsigned(bits);
  }

  [[nodiscard]] std::size_t finish() noexcept {
    if (!destination_.empty()) {
      const std::size_t terminator =
          required_ < destination_.size() ? required_
                                          : destination_.size() - 1;
      destination_[terminator] = '\0';
    }
    return required_;
  }

 private:
  std::span<char> destination_;
  std::size_t required_ = 0;
};

[[nodiscard]] constexpr bool uses_signed_actual(
    TensorValidationCode code) noexcept {
  return code == TensorValidationCode::kInvalidDeviceOrdinal ||
         code == TensorValidationCode::kUnusedDimensionNonZero ||
         code == TensorValidationCode::kNegativeExtent ||
         code == TensorValidationCode::kNegativeStride;
}

}  // namespace

ValidationOutcome<ValidatedConstTensorView> validate(
    const SglNativeConstTensorViewV1* view) noexcept {
  using Outcome = ValidationOutcome<ValidatedConstTensorView>;
  const auto result = validate_impl<SglNativeConstTensorViewV1, const void*>(
      view, false);
  if (!result.valid) {
    return Outcome::from_error(result.error);
  }
  return Outcome::from_value(
      ValidatedConstTensorView(result.metadata, result.allocation_base));
}

ValidationOutcome<ValidatedMutableTensorView> validate(
    const SglNativeMutableTensorViewV1* view) noexcept {
  using Outcome = ValidationOutcome<ValidatedMutableTensorView>;
  const auto result = validate_impl<SglNativeMutableTensorViewV1, void*>(
      view, true);
  if (!result.valid) {
    return Outcome::from_error(result.error);
  }
  return Outcome::from_value(
      ValidatedMutableTensorView(result.metadata, result.allocation_base));
}

std::string_view tensor_validation_code_name(
    TensorValidationCode code) noexcept {
  switch (code) {
    case TensorValidationCode::kOk:
      return "ok";
    case TensorValidationCode::kNullView:
      return "null_view";
    case TensorValidationCode::kMetadataSizeMismatch:
      return "metadata_size_mismatch";
    case TensorValidationCode::kAbiVersionMismatch:
      return "abi_version_mismatch";
    case TensorValidationCode::kReservedFieldNonZero:
      return "reserved_field_non_zero";
    case TensorValidationCode::kUnknownDType:
      return "unknown_dtype";
    case TensorValidationCode::kUnknownDevice:
      return "unknown_device";
    case TensorValidationCode::kInvalidDeviceOrdinal:
      return "invalid_device_ordinal";
    case TensorValidationCode::kRankOutOfRange:
      return "rank_out_of_range";
    case TensorValidationCode::kUnusedDimensionNonZero:
      return "unused_dimension_non_zero";
    case TensorValidationCode::kNegativeExtent:
      return "negative_extent";
    case TensorValidationCode::kNegativeStride:
      return "negative_stride";
    case TensorValidationCode::kValueOutOfDomain:
      return "value_out_of_domain";
    case TensorValidationCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case TensorValidationCode::kNullAllocation:
      return "null_allocation";
    case TensorValidationCode::kInvalidEmptyView:
      return "invalid_empty_view";
    case TensorValidationCode::kMisalignedAllocationBase:
      return "misaligned_allocation_base";
    case TensorValidationCode::kMisalignedStorageOffset:
      return "misaligned_storage_offset";
    case TensorValidationCode::kPointerRangeOverflow:
      return "pointer_range_overflow";
    case TensorValidationCode::kOutOfBounds:
      return "out_of_bounds";
    case TensorValidationCode::kMutableOverlap:
      return "mutable_overlap";
    case TensorValidationCode::kDTypeMismatch:
      return "dtype_mismatch";
    case TensorValidationCode::kRankMismatch:
      return "rank_mismatch";
    case TensorValidationCode::kMutableSubByteUnsupported:
      return "mutable_sub_byte_unsupported";
    default:
      return "invalid_validation_code";
  }
}

std::string_view tensor_validation_field_name(
    TensorValidationField field) noexcept {
  switch (field) {
    case TensorValidationField::kNone:
      return "none";
    case TensorValidationField::kView:
      return "view";
    case TensorValidationField::kStructSize:
      return "struct_size";
    case TensorValidationField::kAbiVersion:
      return "abi_version";
    case TensorValidationField::kReserved:
      return "reserved";
    case TensorValidationField::kDType:
      return "dtype";
    case TensorValidationField::kDeviceKind:
      return "device_kind";
    case TensorValidationField::kDeviceOrdinal:
      return "device_ordinal";
    case TensorValidationField::kRank:
      return "rank";
    case TensorValidationField::kExtent:
      return "extent";
    case TensorValidationField::kStride:
      return "stride";
    case TensorValidationField::kAllocationBytes:
      return "allocation_bytes";
    case TensorValidationField::kStorageOffsetElements:
      return "storage_offset_elements";
    case TensorValidationField::kAllocationBase:
      return "allocation_base";
    case TensorValidationField::kSpan:
      return "span";
    case TensorValidationField::kMutableLayout:
      return "mutable_layout";
    case TensorValidationField::kNarrowDType:
      return "narrow_dtype";
    case TensorValidationField::kNarrowRank:
      return "narrow_rank";
    default:
      return "invalid_validation_field";
  }
}

std::size_t format_tensor_validation_error(
    const TensorValidationError& error,
    std::span<char> destination) noexcept {
  BoundedWriter writer(destination);
  writer.append("tensor_validation_error code=");
  writer.append(tensor_validation_code_name(error.code));
  writer.append(" field=");
  writer.append(tensor_validation_field_name(error.field));
  writer.append(" dimension=");
  if (error.dimension == kNoDimension) {
    writer.append("none");
  } else {
    writer.append_unsigned(error.dimension);
  }
  writer.append(" actual=");
  if (uses_signed_actual(error.code)) {
    writer.append_signed_bits(error.actual);
  } else {
    writer.append_unsigned(error.actual);
  }
  writer.append(" required=");
  writer.append_unsigned(error.required);
  return writer.finish();
}

}  // namespace sglang::native
