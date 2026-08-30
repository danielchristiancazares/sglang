#ifndef SGLANG_NATIVE_TENSOR_VIEW_HPP_
#define SGLANG_NATIVE_TENSOR_VIEW_HPP_

#include "sglang/native/tensor_view.h"

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#if !defined(_WIN64)
#error "The native tensor-view ABI v1 requires Windows x64."
#endif

namespace sglang::native {

enum class DType : uint32_t {
  kInvalid = SGL_NATIVE_DTYPE_INVALID,
  kBool8 = SGL_NATIVE_DTYPE_BOOL8,
  kUInt8 = SGL_NATIVE_DTYPE_UINT8,
  kInt8 = SGL_NATIVE_DTYPE_INT8,
  kUInt16 = SGL_NATIVE_DTYPE_UINT16,
  kInt16 = SGL_NATIVE_DTYPE_INT16,
  kUInt32 = SGL_NATIVE_DTYPE_UINT32,
  kInt32 = SGL_NATIVE_DTYPE_INT32,
  kUInt64 = SGL_NATIVE_DTYPE_UINT64,
  kInt64 = SGL_NATIVE_DTYPE_INT64,
  kFloat16 = SGL_NATIVE_DTYPE_FLOAT16,
  kBFloat16 = SGL_NATIVE_DTYPE_BFLOAT16,
  kFloat32 = SGL_NATIVE_DTYPE_FLOAT32,
  kFloat8E4M3Fn = SGL_NATIVE_DTYPE_FLOAT8_E4M3FN,
  kFloat8E5M2 = SGL_NATIVE_DTYPE_FLOAT8_E5M2,
  kNvFp4E2M1 = SGL_NATIVE_DTYPE_NVFP4_E2M1,
};

enum class DeviceKind : uint32_t {
  kInvalid = SGL_NATIVE_DEVICE_INVALID,
  kCpu = SGL_NATIVE_DEVICE_CPU,
  kCuda = SGL_NATIVE_DEVICE_CUDA,
  kCudaHost = SGL_NATIVE_DEVICE_CUDA_HOST,
};

enum class TensorAccess : uint8_t {
  kReadOnly = 1,
  kReadWrite = 2,
};

enum class TensorValidationCode : uint32_t {
  kOk = 0,
  kNullView = 1,
  kMetadataSizeMismatch = 2,
  kAbiVersionMismatch = 3,
  kReservedFieldNonZero = 4,
  kUnknownDType = 5,
  kUnknownDevice = 6,
  kInvalidDeviceOrdinal = 7,
  kRankOutOfRange = 8,
  kUnusedDimensionNonZero = 9,
  kNegativeExtent = 10,
  kNegativeStride = 11,
  kValueOutOfDomain = 12,
  kArithmeticOverflow = 13,
  kNullAllocation = 14,
  kInvalidEmptyView = 15,
  kMisalignedAllocationBase = 16,
  kMisalignedStorageOffset = 17,
  kPointerRangeOverflow = 18,
  kOutOfBounds = 19,
  kMutableOverlap = 20,
  kDTypeMismatch = 21,
  kRankMismatch = 22,
  kMutableSubByteUnsupported = 23,
};

enum class TensorValidationField : uint32_t {
  kNone = 0,
  kView = 1,
  kStructSize = 2,
  kAbiVersion = 3,
  kReserved = 4,
  kDType = 5,
  kDeviceKind = 6,
  kDeviceOrdinal = 7,
  kRank = 8,
  kExtent = 9,
  kStride = 10,
  kAllocationBytes = 11,
  kStorageOffsetElements = 12,
  kAllocationBase = 13,
  kSpan = 14,
  kMutableLayout = 15,
  kNarrowDType = 16,
  kNarrowRank = 17,
};

[[nodiscard]] constexpr uint32_t dtype_element_bits(DType dtype) noexcept {
  switch (dtype) {
    case DType::kBool8:
    case DType::kUInt8:
    case DType::kInt8:
    case DType::kFloat8E4M3Fn:
    case DType::kFloat8E5M2:
      return 8;
    case DType::kUInt16:
    case DType::kInt16:
    case DType::kFloat16:
    case DType::kBFloat16:
      return 16;
    case DType::kUInt32:
    case DType::kInt32:
    case DType::kFloat32:
      return 32;
    case DType::kUInt64:
    case DType::kInt64:
      return 64;
    case DType::kNvFp4E2M1:
      return 4;
    default:
      return 0;
  }
}

[[nodiscard]] constexpr uint32_t dtype_natural_alignment(DType dtype) noexcept {
  switch (dtype) {
    case DType::kBool8:
    case DType::kUInt8:
    case DType::kInt8:
    case DType::kFloat8E4M3Fn:
    case DType::kFloat8E5M2:
    case DType::kNvFp4E2M1:
      return 1;
    case DType::kUInt16:
    case DType::kInt16:
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kUInt32:
    case DType::kInt32:
    case DType::kFloat32:
      return 4;
    case DType::kUInt64:
    case DType::kInt64:
      return 8;
    default:
      return 0;
  }
}

[[nodiscard]] constexpr SglNativeTensorMetadataV1 make_tensor_metadata_v1()
    noexcept {
  return SglNativeTensorMetadataV1{
      SGL_NATIVE_TENSOR_METADATA_V1_SIZE,
      static_cast<uint16_t>(SGL_NATIVE_TENSOR_ABI_MAJOR),
      static_cast<uint16_t>(SGL_NATIVE_TENSOR_ABI_MINOR),
      SGL_NATIVE_DTYPE_INVALID,
      SGL_NATIVE_DEVICE_INVALID,
      0,
      0,
      0,
      0,
      {},
      {},
      {}};
}

#if defined(_MSC_VER)
#pragma pack(push, 8)
#endif

struct TensorValidationError final {
  TensorValidationCode code;
  TensorValidationField field;
  uint32_t dimension;
  uint32_t reserved;
  uint64_t actual;
  uint64_t required;
};

template <typename T>
class ValidationOutcome;

template <TensorAccess Access>
class ValidatedTensorView;

template <DType D, uint32_t Rank, TensorAccess Access>
class TypedTensorView;

using ValidatedConstTensorView =
    ValidatedTensorView<TensorAccess::kReadOnly>;
using ValidatedMutableTensorView =
    ValidatedTensorView<TensorAccess::kReadWrite>;

template <DType D, uint32_t Rank>
using TypedConstTensorView =
    TypedTensorView<D, Rank, TensorAccess::kReadOnly>;

template <DType D, uint32_t Rank>
using TypedMutableTensorView =
    TypedTensorView<D, Rank, TensorAccess::kReadWrite>;

[[nodiscard]] ValidationOutcome<ValidatedConstTensorView> validate(
    const SglNativeConstTensorViewV1* view) noexcept;
[[nodiscard]] ValidationOutcome<ValidatedMutableTensorView> validate(
    const SglNativeMutableTensorViewV1* view) noexcept;

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] ValidationOutcome<TypedTensorView<D, Rank, Access>> narrow(
    ValidatedTensorView<Access> view) noexcept;

template <typename T>
class ValidationOutcome final {
  static_assert(std::is_trivially_copyable_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);

 public:
  ValidationOutcome() = delete;
  ValidationOutcome(const ValidationOutcome&) = delete;
  ValidationOutcome& operator=(const ValidationOutcome&) = delete;
  ValidationOutcome(ValidationOutcome&&) = default;
  ValidationOutcome& operator=(ValidationOutcome&&) = delete;
  ~ValidationOutcome() = default;

  template <typename OnValid, typename OnError,
            typename ValidResult = std::invoke_result_t<OnValid, T>,
            typename ErrorResult =
                std::invoke_result_t<OnError, TensorValidationError>>
    requires std::same_as<ValidResult, ErrorResult> &&
             (!std::is_reference_v<ValidResult>)
  ValidResult match(OnValid&& on_valid, OnError&& on_error) &&
      noexcept(noexcept(std::invoke(std::forward<OnValid>(on_valid),
                                    std::declval<T>())) &&
               noexcept(std::invoke(std::forward<OnError>(on_error),
                                    std::declval<TensorValidationError>()))) {
    if (has_value_) {
      return std::invoke(std::forward<OnValid>(on_valid),
                         std::move(storage_.value));
    }
    return std::invoke(std::forward<OnError>(on_error),
                       TensorValidationError{storage_.error});
  }

 private:
  struct ValueTag final {};
  struct ErrorTag final {};

  union Storage {
    T value;
    TensorValidationError error;

    constexpr Storage(ValueTag, T input) noexcept : value(input) {}
    constexpr Storage(ErrorTag, TensorValidationError input) noexcept
        : error(input) {}
    ~Storage() = default;
  };

  constexpr ValidationOutcome(ValueTag tag, T value) noexcept
      : storage_(tag, value), has_value_(true) {}
  constexpr ValidationOutcome(ErrorTag tag, TensorValidationError error)
      noexcept
      : storage_(tag, error), has_value_(false) {}

  [[nodiscard]] static constexpr ValidationOutcome from_value(T value)
      noexcept {
    return ValidationOutcome(ValueTag{}, value);
  }
  [[nodiscard]] static constexpr ValidationOutcome from_error(
      TensorValidationError error) noexcept {
    return ValidationOutcome(ErrorTag{}, error);
  }

  Storage storage_;
  bool has_value_;

  friend ValidationOutcome<ValidatedConstTensorView> validate(
      const SglNativeConstTensorViewV1*) noexcept;
  friend ValidationOutcome<ValidatedMutableTensorView> validate(
      const SglNativeMutableTensorViewV1*) noexcept;
  template <DType D, uint32_t Rank, TensorAccess Access>
  friend ValidationOutcome<TypedTensorView<D, Rank, Access>> narrow(
      ValidatedTensorView<Access>) noexcept;
};

namespace detail {

[[nodiscard]] constexpr bool metadata_is_empty(
    const SglNativeTensorMetadataV1& metadata) noexcept {
  if (metadata.rank == 0) {
    return false;
  }
  for (uint32_t dimension = 0; dimension < metadata.rank; ++dimension) {
    if (metadata.extents[dimension] == 0) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] constexpr bool metadata_is_row_major_contiguous(
    const SglNativeTensorMetadataV1& metadata) noexcept {
  if (metadata_is_empty(metadata) || metadata.rank == 0) {
    return true;
  }
  uint64_t expected_stride = 1;
  for (uint32_t dimension = metadata.rank; dimension > 0; --dimension) {
    const uint32_t index = dimension - 1;
    const uint64_t extent = static_cast<uint64_t>(metadata.extents[index]);
    if (extent > 1 &&
        static_cast<uint64_t>(metadata.strides[index]) != expected_stride) {
      return false;
    }
    const uint64_t factor = extent == 0 ? 1 : extent;
    if (factor != 0 &&
        expected_stride > std::numeric_limits<uint64_t>::max() / factor) {
      return false;
    }
    expected_stride *= factor;
  }
  return true;
}

}  // namespace detail

template <TensorAccess Access>
class alignas(8) ValidatedTensorView final {
  static_assert(Access == TensorAccess::kReadOnly ||
                Access == TensorAccess::kReadWrite);

 public:
  using AllocationBase =
      std::conditional_t<Access == TensorAccess::kReadOnly, const void*, void*>;

  [[nodiscard]] constexpr DType dtype() const noexcept {
    return static_cast<DType>(metadata_.dtype);
  }
  [[nodiscard]] constexpr DeviceKind device_kind() const noexcept {
    return static_cast<DeviceKind>(metadata_.device_kind);
  }
  [[nodiscard]] constexpr int32_t device_ordinal() const noexcept {
    return metadata_.device_ordinal;
  }
  [[nodiscard]] constexpr uint32_t rank() const noexcept {
    return metadata_.rank;
  }
  [[nodiscard]] constexpr std::span<const int64_t> extents() const noexcept {
    return {metadata_.extents, metadata_.rank};
  }
  [[nodiscard]] constexpr std::span<const int64_t> strides() const noexcept {
    return {metadata_.strides, metadata_.rank};
  }
  [[nodiscard]] constexpr uint64_t allocation_bytes() const noexcept {
    return metadata_.allocation_bytes;
  }
  [[nodiscard]] constexpr uint64_t storage_offset_elements() const noexcept {
    return metadata_.storage_offset_elements;
  }
  [[nodiscard]] constexpr bool is_empty() const noexcept {
    return detail::metadata_is_empty(metadata_);
  }
  [[nodiscard]] constexpr bool is_row_major_contiguous() const noexcept {
    return detail::metadata_is_row_major_contiguous(metadata_);
  }

 private:
  constexpr ValidatedTensorView(SglNativeTensorMetadataV1 metadata,
                                AllocationBase allocation_base) noexcept
      : metadata_(metadata), allocation_base_(allocation_base) {}

  SglNativeTensorMetadataV1 metadata_;
  AllocationBase allocation_base_;

  friend ValidationOutcome<ValidatedConstTensorView> validate(
      const SglNativeConstTensorViewV1*) noexcept;
  friend ValidationOutcome<ValidatedMutableTensorView> validate(
      const SglNativeMutableTensorViewV1*) noexcept;
  template <DType D, uint32_t Rank, TensorAccess OtherAccess>
  friend ValidationOutcome<TypedTensorView<D, Rank, OtherAccess>> narrow(
      ValidatedTensorView<OtherAccess>) noexcept;
};

template <DType D, uint32_t Rank, TensorAccess Access>
class alignas(8) TypedTensorView final {
  static_assert(dtype_element_bits(D) != 0);
  static_assert(Rank <= SGL_NATIVE_TENSOR_MAX_RANK);
  static_assert(Access == TensorAccess::kReadOnly ||
                Access == TensorAccess::kReadWrite);

 public:
  static constexpr DType dtype_v = D;
  static constexpr uint32_t rank_v = Rank;
  static constexpr TensorAccess access_v = Access;
  using AllocationBase =
      std::conditional_t<Access == TensorAccess::kReadOnly, const void*, void*>;

  [[nodiscard]] constexpr DType dtype() const noexcept { return D; }
  [[nodiscard]] constexpr DeviceKind device_kind() const noexcept {
    return static_cast<DeviceKind>(metadata_.device_kind);
  }
  [[nodiscard]] constexpr int32_t device_ordinal() const noexcept {
    return metadata_.device_ordinal;
  }
  [[nodiscard]] constexpr uint32_t rank() const noexcept { return Rank; }
  [[nodiscard]] constexpr std::span<const int64_t, Rank> extents()
      const noexcept {
    return std::span<const int64_t, Rank>(metadata_.extents, Rank);
  }
  [[nodiscard]] constexpr std::span<const int64_t, Rank> strides()
      const noexcept {
    return std::span<const int64_t, Rank>(metadata_.strides, Rank);
  }
  [[nodiscard]] constexpr uint64_t allocation_bytes() const noexcept {
    return metadata_.allocation_bytes;
  }
  [[nodiscard]] constexpr uint64_t storage_offset_elements() const noexcept {
    return metadata_.storage_offset_elements;
  }
  [[nodiscard]] constexpr bool is_empty() const noexcept {
    return detail::metadata_is_empty(metadata_);
  }
  [[nodiscard]] constexpr bool is_row_major_contiguous() const noexcept {
    return detail::metadata_is_row_major_contiguous(metadata_);
  }

  [[nodiscard]] const std::byte* data_bytes() const noexcept
    requires(Access == TensorAccess::kReadOnly)
  {
    if (allocation_base_ == nullptr) {
      return nullptr;
    }
    const uint64_t origin_bytes =
        metadata_.storage_offset_elements * dtype_element_bits(D) / 8;
    return static_cast<const std::byte*>(allocation_base_) + origin_bytes;
  }

  [[nodiscard]] std::byte* data_bytes() const noexcept
    requires(Access == TensorAccess::kReadWrite)
  {
    if (allocation_base_ == nullptr) {
      return nullptr;
    }
    const uint64_t origin_bytes =
        metadata_.storage_offset_elements * dtype_element_bits(D) / 8;
    return static_cast<std::byte*>(allocation_base_) + origin_bytes;
  }

 private:
  constexpr TypedTensorView(SglNativeTensorMetadataV1 metadata,
                            AllocationBase allocation_base) noexcept
      : metadata_(metadata), allocation_base_(allocation_base) {}

  SglNativeTensorMetadataV1 metadata_;
  AllocationBase allocation_base_;

  template <DType OtherD, uint32_t OtherRank, TensorAccess OtherAccess>
  friend ValidationOutcome<
      TypedTensorView<OtherD, OtherRank, OtherAccess>>
  narrow(ValidatedTensorView<OtherAccess>) noexcept;
};

#if defined(_MSC_VER)
#pragma pack(pop)
#endif

template <DType D, uint32_t Rank, TensorAccess Access>
[[nodiscard]] ValidationOutcome<TypedTensorView<D, Rank, Access>> narrow(
    ValidatedTensorView<Access> view) noexcept {
  static_assert(dtype_element_bits(D) != 0,
                "Typed witnesses require an ABI-v1 dtype.");
  static_assert(Rank <= SGL_NATIVE_TENSOR_MAX_RANK,
                "Typed witnesses require rank at most eight.");
  using Outcome = ValidationOutcome<TypedTensorView<D, Rank, Access>>;
  if (view.metadata_.dtype != static_cast<uint32_t>(D)) {
    return Outcome::from_error(TensorValidationError{
        TensorValidationCode::kDTypeMismatch,
        TensorValidationField::kNarrowDType,
        std::numeric_limits<uint32_t>::max(),
        0,
        view.metadata_.dtype,
        static_cast<uint32_t>(D)});
  }
  if (view.metadata_.rank != Rank) {
    return Outcome::from_error(TensorValidationError{
        TensorValidationCode::kRankMismatch,
        TensorValidationField::kNarrowRank,
        std::numeric_limits<uint32_t>::max(),
        0,
        view.metadata_.rank,
        Rank});
  }
  return Outcome::from_value(
      TypedTensorView<D, Rank, Access>(view.metadata_, view.allocation_base_));
}

[[nodiscard]] std::string_view tensor_validation_code_name(
    TensorValidationCode code) noexcept;
[[nodiscard]] std::string_view tensor_validation_field_name(
    TensorValidationField field) noexcept;
[[nodiscard]] std::size_t format_tensor_validation_error(
    const TensorValidationError& error,
    std::span<char> destination) noexcept;

static_assert(sizeof(void*) == 8);
static_assert(sizeof(uint8_t) == 1);
static_assert(sizeof(uint16_t) == 2);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(uint64_t) == 8);
static_assert(sizeof(int32_t) == 4);
static_assert(sizeof(int64_t) == 8);
static_assert(std::endian::native == std::endian::little);

static_assert(sizeof(SglNativeTensorMetadataV1) ==
              SGL_NATIVE_TENSOR_METADATA_V1_SIZE);
static_assert(alignof(SglNativeTensorMetadataV1) == 8);
static_assert(offsetof(SglNativeTensorMetadataV1, struct_size) == 0);
static_assert(offsetof(SglNativeTensorMetadataV1, abi_major) == 4);
static_assert(offsetof(SglNativeTensorMetadataV1, abi_minor) == 6);
static_assert(offsetof(SglNativeTensorMetadataV1, dtype) == 8);
static_assert(offsetof(SglNativeTensorMetadataV1, device_kind) == 12);
static_assert(offsetof(SglNativeTensorMetadataV1, device_ordinal) == 16);
static_assert(offsetof(SglNativeTensorMetadataV1, rank) == 20);
static_assert(offsetof(SglNativeTensorMetadataV1, allocation_bytes) == 24);
static_assert(offsetof(SglNativeTensorMetadataV1, storage_offset_elements) ==
              32);
static_assert(offsetof(SglNativeTensorMetadataV1, extents) == 40);
static_assert(offsetof(SglNativeTensorMetadataV1, strides) == 104);
static_assert(offsetof(SglNativeTensorMetadataV1, reserved) == 168);

static_assert(sizeof(SglNativeConstTensorViewV1) ==
              SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(sizeof(SglNativeMutableTensorViewV1) ==
              SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(alignof(SglNativeConstTensorViewV1) == 8);
static_assert(alignof(SglNativeMutableTensorViewV1) == 8);
static_assert(offsetof(SglNativeConstTensorViewV1, metadata) == 0);
static_assert(offsetof(SglNativeConstTensorViewV1, allocation_base) == 184);
static_assert(offsetof(SglNativeMutableTensorViewV1, metadata) == 0);
static_assert(offsetof(SglNativeMutableTensorViewV1, allocation_base) == 184);
static_assert(std::is_standard_layout_v<SglNativeTensorMetadataV1>);
static_assert(std::is_trivially_copyable_v<SglNativeTensorMetadataV1>);
static_assert(std::is_standard_layout_v<SglNativeConstTensorViewV1>);
static_assert(std::is_trivially_copyable_v<SglNativeConstTensorViewV1>);
static_assert(std::is_standard_layout_v<SglNativeMutableTensorViewV1>);
static_assert(std::is_trivially_copyable_v<SglNativeMutableTensorViewV1>);

static_assert(sizeof(TensorValidationError) == 32);
static_assert(alignof(TensorValidationError) == 8);
static_assert(offsetof(TensorValidationError, code) == 0);
static_assert(offsetof(TensorValidationError, field) == 4);
static_assert(offsetof(TensorValidationError, dimension) == 8);
static_assert(offsetof(TensorValidationError, reserved) == 12);
static_assert(offsetof(TensorValidationError, actual) == 16);
static_assert(offsetof(TensorValidationError, required) == 24);
static_assert(std::is_standard_layout_v<TensorValidationError>);
static_assert(std::is_trivially_copyable_v<TensorValidationError>);

#define SGL_NATIVE_ASSERT_DTYPE(c_name, cpp_name, value)                  \
  static_assert(static_cast<uint32_t>(c_name) == value);                 \
  static_assert(static_cast<uint32_t>(DType::cpp_name) == value);        \
  static_assert(static_cast<uint32_t>(DType::cpp_name) ==                \
                static_cast<uint32_t>(c_name))
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_INVALID, kInvalid, 0);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_BOOL8, kBool8, 1);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_UINT8, kUInt8, 2);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_INT8, kInt8, 3);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_UINT16, kUInt16, 4);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_INT16, kInt16, 5);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_UINT32, kUInt32, 6);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_INT32, kInt32, 7);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_UINT64, kUInt64, 8);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_INT64, kInt64, 9);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_FLOAT16, kFloat16, 10);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_BFLOAT16, kBFloat16, 11);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_FLOAT32, kFloat32, 12);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_FLOAT8_E4M3FN, kFloat8E4M3Fn, 13);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_FLOAT8_E5M2, kFloat8E5M2, 14);
SGL_NATIVE_ASSERT_DTYPE(SGL_NATIVE_DTYPE_NVFP4_E2M1, kNvFp4E2M1, 15);
#undef SGL_NATIVE_ASSERT_DTYPE

static_assert(static_cast<uint32_t>(DeviceKind::kInvalid) ==
              SGL_NATIVE_DEVICE_INVALID);
static_assert(static_cast<uint32_t>(DeviceKind::kCpu) ==
              SGL_NATIVE_DEVICE_CPU);
static_assert(static_cast<uint32_t>(DeviceKind::kCuda) ==
              SGL_NATIVE_DEVICE_CUDA);
static_assert(static_cast<uint32_t>(DeviceKind::kCudaHost) ==
              SGL_NATIVE_DEVICE_CUDA_HOST);

using StaticValidatedConst = ValidatedConstTensorView;
using StaticValidatedMutable = ValidatedMutableTensorView;
using StaticTypedConst = TypedConstTensorView<DType::kUInt8, 1>;
using StaticTypedMutable = TypedMutableTensorView<DType::kUInt8, 1>;

static_assert(sizeof(StaticValidatedConst) == SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(sizeof(StaticValidatedMutable) == SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(sizeof(StaticTypedConst) == SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(sizeof(StaticTypedMutable) == SGL_NATIVE_TENSOR_VIEW_V1_SIZE);
static_assert(alignof(StaticValidatedConst) == 8);
static_assert(alignof(StaticValidatedMutable) == 8);
static_assert(alignof(StaticTypedConst) == 8);
static_assert(alignof(StaticTypedMutable) == 8);
static_assert(std::is_standard_layout_v<StaticValidatedConst>);
static_assert(std::is_standard_layout_v<StaticValidatedMutable>);
static_assert(std::is_standard_layout_v<StaticTypedConst>);
static_assert(std::is_standard_layout_v<StaticTypedMutable>);
static_assert(std::is_trivially_copyable_v<StaticValidatedConst>);
static_assert(std::is_trivially_copyable_v<StaticValidatedMutable>);
static_assert(std::is_trivially_copyable_v<StaticTypedConst>);
static_assert(std::is_trivially_copyable_v<StaticTypedMutable>);
static_assert(!std::is_constructible_v<StaticValidatedConst,
                                       SglNativeTensorMetadataV1,
                                       const void*>);
static_assert(!std::is_constructible_v<StaticValidatedMutable,
                                       SglNativeTensorMetadataV1, void*>);
static_assert(!std::is_constructible_v<StaticTypedConst,
                                       SglNativeTensorMetadataV1,
                                       const void*>);
static_assert(!std::is_constructible_v<StaticTypedMutable,
                                       SglNativeTensorMetadataV1, void*>);
static_assert(std::same_as<
              decltype(std::declval<const StaticTypedConst&>().data_bytes()),
              const std::byte*>);
static_assert(std::same_as<
              decltype(std::declval<const StaticTypedMutable&>().data_bytes()),
              std::byte*>);
static_assert(noexcept(validate(
    static_cast<const SglNativeConstTensorViewV1*>(nullptr))));
static_assert(noexcept(validate(
    static_cast<const SglNativeMutableTensorViewV1*>(nullptr))));
static_assert(noexcept(narrow<DType::kUInt8, 1>(
    std::declval<ValidatedConstTensorView>())));
static_assert(noexcept(narrow<DType::kUInt8, 1>(
    std::declval<ValidatedMutableTensorView>())));

}  // namespace sglang::native

#endif  // SGLANG_NATIVE_TENSOR_VIEW_HPP_
