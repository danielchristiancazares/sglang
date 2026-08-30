#include "sglang/native/tensor_view.hpp"

#define NOMINMAX
#pragma pack(push, 8)
#include <Windows.h>
#pragma pack(pop)

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace {
std::atomic<uint64_t> g_allocation_count{0};
}  // namespace

void* operator new(std::size_t size) {
  g_allocation_count.fetch_add(1, std::memory_order_relaxed);
  if (void* pointer = std::malloc(size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  return ::operator new(size);
}

void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

namespace {

using sglang::native::DType;
using sglang::native::DeviceKind;
using sglang::native::TensorAccess;
using sglang::native::TensorValidationCode;
using sglang::native::TensorValidationError;
using sglang::native::TensorValidationField;
using sglang::native::ValidatedConstTensorView;
using sglang::native::ValidatedMutableTensorView;
using sglang::native::format_tensor_validation_error;
using sglang::native::make_tensor_metadata_v1;
using sglang::native::narrow;
using sglang::native::tensor_validation_code_name;
using sglang::native::tensor_validation_field_name;
using sglang::native::validate;
constexpr uint32_t kNoDimension = std::numeric_limits<uint32_t>::max();
alignas(64) std::array<std::byte, 4096> g_storage{};

struct Probe final {
  bool valid;
  TensorValidationError error;
  uint32_t rank;
  bool empty;
  bool contiguous;
};

template <typename Outcome>
[[nodiscard]] Probe probe(Outcome outcome) noexcept {
  return std::move(outcome).match(
      [](auto witness) noexcept -> Probe {
        return Probe{true, {}, witness.rank(), witness.is_empty(),
                     witness.is_row_major_contiguous()};
      },
      [](TensorValidationError error) noexcept -> Probe {
        return Probe{false, error, 0, false, false};
      });
}
[[nodiscard]] SglNativeTensorMetadataV1 metadata(
    SglNativeDType dtype = SGL_NATIVE_DTYPE_UINT8, uint32_t rank = 1,
    uint64_t allocation_bytes = 1) noexcept {
  auto value = make_tensor_metadata_v1();
  value.dtype = dtype;
  value.device_kind = SGL_NATIVE_DEVICE_CPU;
  value.rank = rank;
  value.allocation_bytes = allocation_bytes;
  if (rank != 0 && rank <= SGL_NATIVE_TENSOR_MAX_RANK) {
    value.extents[0] = 1;
    value.strides[0] = 1;
  }
  return value;
}
[[nodiscard]] SglNativeConstTensorViewV1 const_view(
    SglNativeTensorMetadataV1 value,
    const void* base = g_storage.data()) noexcept {
  return SglNativeConstTensorViewV1{value, base};
}
[[nodiscard]] SglNativeMutableTensorViewV1 mutable_view(
    SglNativeTensorMetadataV1 value,
    void* base = g_storage.data()) noexcept {
  return SglNativeMutableTensorViewV1{value, base};
}
[[nodiscard]] Probe validate_const(
    SglNativeTensorMetadataV1 value,
    const void* base = g_storage.data()) noexcept {
  const auto raw = const_view(value, base);
  return probe(validate(&raw));
}
[[nodiscard]] Probe validate_mutable(
    SglNativeTensorMetadataV1 value,
    void* base = g_storage.data()) noexcept {
  const auto raw = mutable_view(value, base);
  return probe(validate(&raw));
}
[[nodiscard]] bool expect_code(const Probe& result,
                               TensorValidationCode code) noexcept {
  return !result.valid && result.error.code == code;
}
[[nodiscard]] bool expect_error(
    const Probe& result, TensorValidationCode code,
    TensorValidationField field, uint32_t dimension, uint64_t actual,
    uint64_t required) noexcept {
  return expect_code(result, code) && result.error.field == field &&
         result.error.dimension == dimension && result.error.reserved == 0 &&
         result.error.actual == actual && result.error.required == required;
}

[[nodiscard]] bool record_check(bool passed, const char* expression, int line) noexcept {
  if (!passed) std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  return passed;
}
#define CHECK(condition)                                                       \
  if (!record_check(static_cast<bool>(condition), #condition, __LINE__))       \
    return false

[[nodiscard]] bool AbiLayoutDefaultPack() {
  CHECK(sizeof(SglNativeTensorMetadataV1) == 184);
  CHECK(alignof(SglNativeTensorMetadataV1) == 8);
  CHECK(sizeof(SglNativeConstTensorViewV1) == 192);
  CHECK(offsetof(SglNativeTensorMetadataV1, reserved) == 168);
  CHECK(offsetof(SglNativeConstTensorViewV1, allocation_base) == 184);
  CHECK(std::is_standard_layout_v<ValidatedConstTensorView>);
  CHECK(std::is_trivially_copyable_v<ValidatedMutableTensorView>);
  return true;
}
[[nodiscard]] bool AbiLayoutAdversePack() {
  CHECK(SGL_NATIVE_TENSOR_METADATA_V1_SIZE == sizeof(SglNativeTensorMetadataV1));
  CHECK(SGL_NATIVE_TENSOR_VIEW_V1_SIZE == sizeof(SglNativeMutableTensorViewV1));
  CHECK(alignof(SglNativeMutableTensorViewV1) == 8);
  return true;
}
[[nodiscard]] bool NumericIdentifiersAreStable() {
  constexpr std::array<std::string_view, 24> code_names{
      "ok", "null_view", "metadata_size_mismatch", "abi_version_mismatch",
      "reserved_field_non_zero", "unknown_dtype", "unknown_device", "invalid_device_ordinal",
      "rank_out_of_range", "unused_dimension_non_zero", "negative_extent", "negative_stride",
      "value_out_of_domain", "arithmetic_overflow", "null_allocation", "invalid_empty_view",
      "misaligned_allocation_base", "misaligned_storage_offset", "pointer_range_overflow", "out_of_bounds",
      "mutable_overlap", "dtype_mismatch", "rank_mismatch", "mutable_sub_byte_unsupported"};
  constexpr std::array<std::string_view, 18> field_names{
      "none", "view", "struct_size", "abi_version", "reserved", "dtype",
      "device_kind", "device_ordinal", "rank", "extent", "stride", "allocation_bytes",
      "storage_offset_elements", "allocation_base", "span", "mutable_layout", "narrow_dtype", "narrow_rank"};
  CHECK(static_cast<uint32_t>(DType::kInvalid) == 0);
  CHECK(static_cast<uint32_t>(DType::kNvFp4E2M1) == 15);
  CHECK(static_cast<uint32_t>(DeviceKind::kCpu) == 1);
  CHECK(static_cast<uint32_t>(TensorValidationCode::kMutableSubByteUnsupported) ==
        23);
  CHECK(static_cast<uint32_t>(TensorValidationField::kNarrowRank) == 17);
  for (uint32_t id = 0; id < code_names.size(); ++id)
    CHECK(tensor_validation_code_name(static_cast<TensorValidationCode>(id)) == code_names[id]);
  for (uint32_t id = 0; id < field_names.size(); ++id)
    CHECK(tensor_validation_field_name(static_cast<TensorValidationField>(id)) == field_names[id]);
  for (uint32_t id = 1; id <= 15; ++id) {
    auto value = metadata(id, 1, 8);
    CHECK(validate_const(value).valid);
  }
  for (uint32_t id = 1; id <= 3; ++id) {
    auto value = metadata();
    value.device_kind = id;
    value.device_ordinal = id == SGL_NATIVE_DEVICE_CUDA ? 7 : 0;
    CHECK(validate_const(value).valid);
  }
  return true;
}
[[nodiscard]] bool CanonicalMetadataInitializer() {
  constexpr auto canonical = make_tensor_metadata_v1();
  constexpr SglNativeTensorMetadataV1 expected{
      184, 1, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}};
  CHECK(std::memcmp(&canonical, &expected, sizeof(canonical)) == 0);
  return true;
}
[[nodiscard]] bool PlatformContractIsExplicit() {
  CHECK(sizeof(void*) == 8);
  CHECK(std::endian::native == std::endian::little);
  CHECK(sizeof(uintptr_t) == sizeof(void*));
  return true;
}
[[nodiscard]] bool RankZeroScalarExactFit() {
  auto value = metadata(SGL_NATIVE_DTYPE_FLOAT32, 0, 4);
  const Probe result = validate_const(value);
  CHECK(result.valid && result.rank == 0 && !result.empty);
  CHECK(result.contiguous);
  return true;
}
[[nodiscard]] bool RankEightContiguous() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 8, 256);
  int64_t stride = 128;
  for (uint32_t dimension = 0; dimension < 8; ++dimension) {
    value.extents[dimension] = 2;
    value.strides[dimension] = stride;
    stride /= 2;
  }
  const Probe result = validate_const(value);
  CHECK(result.valid && result.rank == 8 && result.contiguous);
  return true;
}
[[nodiscard]] bool EmptyNullAllocation() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 1, 0);
  value.extents[0] = 0;
  const Probe result = validate_const(value, nullptr);
  CHECK(result.valid && result.empty && result.contiguous);
  return true;
}
[[nodiscard]] bool EmptyRetainedAllocationAtEnd() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 2, 8);
  value.extents[0] = 0;
  value.extents[1] = 5;
  value.strides[0] = 5;
  value.strides[1] = 1;
  value.storage_offset_elements = 8;
  const Probe result = validate_const(value);
  CHECK(result.valid && result.empty);
  return true;
}
[[nodiscard]] SglNativeTensorMetadataV1 transpose_metadata() noexcept {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 2, 6);
  value.extents[0] = 2;
  value.extents[1] = 3;
  value.strides[0] = 1;
  value.strides[1] = 2;
  return value;
}
[[nodiscard]] bool ReadonlyTranspose() {
  const Probe result = validate_const(transpose_metadata());
  CHECK(result.valid && !result.contiguous);
  return true;
}
[[nodiscard]] bool MutableTranspose() {
  const Probe result = validate_mutable(transpose_metadata());
  CHECK(result.valid && !result.contiguous);
  return true;
}
[[nodiscard]] SglNativeTensorMetadataV1 broadcast_metadata() noexcept {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 2, 3);
  value.extents[0] = 2;
  value.extents[1] = 3;
  value.strides[0] = 0;
  value.strides[1] = 1;
  return value;
}
[[nodiscard]] bool ReadonlyZeroStrideBroadcast() {
  CHECK(validate_const(broadcast_metadata()).valid);
  return true;
}
[[nodiscard]] bool MutableZeroStrideBroadcastRejected() {
  const Probe result = validate_mutable(broadcast_metadata());
  CHECK(expect_code(result, TensorValidationCode::kMutableOverlap));
  CHECK(result.error.dimension == 0 && result.error.required == 1);
  return true;
}
[[nodiscard]] bool PaddedMutableLayout() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 2, 7);
  value.extents[0] = 2;
  value.extents[1] = 3;
  value.strides[0] = 4;
  value.strides[1] = 1;
  CHECK(validate_mutable(value).valid);
  return true;
}
[[nodiscard]] bool NvFp4EvenOffsetAndOddElementCount() {
  auto value = metadata(SGL_NATIVE_DTYPE_NVFP4_E2M1, 1, 3);
  value.extents[0] = 3;
  value.storage_offset_elements = 2;
  CHECK(validate_const(value).valid);
  return true;
}
[[nodiscard]] bool NvFp4OddOffsetRejected() {
  auto value = metadata(SGL_NATIVE_DTYPE_NVFP4_E2M1, 1, 1);
  value.storage_offset_elements = 1;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kMisalignedStorageOffset));
  return true;
}
[[nodiscard]] bool MutableNvFp4Rejected() {
  auto value = metadata(SGL_NATIVE_DTYPE_NVFP4_E2M1, 1, 1);
  CHECK(expect_code(validate_mutable(value),
                    TensorValidationCode::kMutableSubByteUnsupported));
  return true;
}
[[nodiscard]] bool AllocationExactEnd() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT16, 1, 8);
  value.extents[0] = 3;
  value.storage_offset_elements = 1;
  CHECK(validate_const(value).valid);
  return true;
}
[[nodiscard]] bool AllocationOneBitShort() {
  auto value = metadata(SGL_NATIVE_DTYPE_NVFP4_E2M1, 1, 1);
  value.extents[0] = 3;
  const Probe result = validate_const(value);
  CHECK(expect_code(result, TensorValidationCode::kOutOfBounds));
  CHECK(result.error.actual == 12 && result.error.required == 8);
  return true;
}
[[nodiscard]] bool UnknownDTypeAndDevice() {
  auto value = metadata(16);
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kUnknownDType));
  value = metadata();
  value.device_kind = 4;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kUnknownDevice));
  return true;
}
[[nodiscard]] bool InvalidDeviceOrdinals() {
  auto value = metadata();
  value.device_ordinal = 1;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kInvalidDeviceOrdinal));
  value.device_kind = SGL_NATIVE_DEVICE_CUDA;
  value.device_ordinal = -1;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kInvalidDeviceOrdinal));
  value.device_kind = SGL_NATIVE_DEVICE_CUDA_HOST;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kInvalidDeviceOrdinal));
  return true;
}
[[nodiscard]] bool UnusedDimensionTailRejected() {
  auto value = metadata();
  value.extents[1] = 2;
  Probe result = validate_const(value);
  CHECK(expect_code(result, TensorValidationCode::kUnusedDimensionNonZero));
  CHECK(result.error.field == TensorValidationField::kExtent);
  value.extents[1] = 0;
  value.strides[1] = 2;
  result = validate_const(value);
  CHECK(result.error.field == TensorValidationField::kStride);
  return true;
}
[[nodiscard]] bool NegativeExtentAndStrideRejected() {
  auto value = metadata();
  value.extents[0] = -1;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kNegativeExtent));
  value.extents[0] = 1;
  value.strides[0] = -1;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kNegativeStride));
  return true;
}
[[nodiscard]] bool CapacityAndOffsetDomainRejected() {
  auto value = metadata();
  value.allocation_bytes = uint64_t{1} << 63U;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kValueOutOfDomain));
  value = metadata();
  value.storage_offset_elements = uint64_t{1} << 63U;
  CHECK(expect_code(validate_const(value),
                    TensorValidationCode::kValueOutOfDomain));
  return true;
}
[[nodiscard]] bool ArithmeticOverflowExtentStride() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 1, 1);
  value.extents[0] = std::numeric_limits<int64_t>::max();
  value.strides[0] = 3;
  const Probe result = validate_const(value);
  CHECK(expect_code(result, TensorValidationCode::kArithmeticOverflow));
  CHECK(result.error.field == TensorValidationField::kStride);
  return true;
}
[[nodiscard]] bool PointerRangeOverflow() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 1, 2);
  const void* base = reinterpret_cast<const void*>(
      std::numeric_limits<uintptr_t>::max());
  CHECK(expect_code(validate_const(value, base),
                    TensorValidationCode::kPointerRangeOverflow));
  return true;
}
[[nodiscard]] bool MutableIrregularOverlapRejected() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 2, 5);
  value.extents[0] = 2;
  value.extents[1] = 2;
  value.strides[0] = 2;
  value.strides[1] = 2;
  CHECK(expect_code(validate_mutable(value),
                    TensorValidationCode::kMutableOverlap));
  return true;
}
[[nodiscard]] bool DTypeAndRankNarrowing() {
  auto value = metadata(SGL_NATIVE_DTYPE_UINT16, 1, 6);
  value.extents[0] = 2;
  value.storage_offset_elements = 1;
  const auto raw = const_view(value);
  return std::move(validate(&raw)).match(
      [](ValidatedConstTensorView validated) noexcept -> bool {
        const bool typed_ok = std::move(narrow<DType::kUInt16, 1>(validated))
                                  .match(
                                      [](auto typed) noexcept -> bool {
                                        return typed.data_bytes() ==
                                               g_storage.data() + 2;
                                      },
                                      [](TensorValidationError) noexcept {
                                        return false;
                                      });
        const Probe dtype = probe(narrow<DType::kUInt8, 1>(validated));
        const Probe rank = probe(narrow<DType::kUInt16, 2>(validated));
        return typed_ok &&
               expect_code(dtype, TensorValidationCode::kDTypeMismatch) &&
               expect_code(rank, TensorValidationCode::kRankMismatch);
      },
      [](TensorValidationError) noexcept { return false; });
}
[[nodiscard]] bool OutcomeErrorUsesValueCategory() {
  struct RvalueErrorHandler final {
    [[nodiscard]] bool operator()(TensorValidationError&& error) const
        noexcept {
      return error.code == TensorValidationCode::kNullView &&
             error.field == TensorValidationField::kView;
    }
    bool operator()(const TensorValidationError&) const = delete;
  };
  auto outcome =
      validate(static_cast<const SglNativeConstTensorViewV1*>(nullptr));
  CHECK(noexcept(std::move(outcome).match(
      [](ValidatedConstTensorView) noexcept { return false; },
      RvalueErrorHandler{})));
  return std::move(outcome).match(
      [](ValidatedConstTensorView) noexcept { return false; },
      RvalueErrorHandler{});
}
[[nodiscard]] bool ValidationDoesNotAllocate() {
  const auto raw = const_view(transpose_metadata());
  const uint64_t before = g_allocation_count.load(std::memory_order_relaxed);
  for (uint32_t iteration = 0; iteration < 10000; ++iteration) {
    if (!probe(validate(&raw)).valid) {
      return false;
    }
  }
  CHECK(g_allocation_count.load(std::memory_order_relaxed) == before);
  return true;
}
[[nodiscard]] bool ValidationDoesNotDereference() {
  SYSTEM_INFO info{};
  GetSystemInfo(&info);
  void* guarded = VirtualAlloc(nullptr, info.dwPageSize,
                               MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS);
  CHECK(guarded != nullptr);
  auto value = metadata(SGL_NATIVE_DTYPE_UINT8, 1, 1);
  const auto raw = const_view(value, guarded);
  const Probe result = probe(validate(&raw));
  const BOOL released = VirtualFree(guarded, 0, MEM_RELEASE);
  CHECK(result.valid && released != FALSE);
  return true;
}
[[nodiscard]] bool ErrorsRedactAllocationAddress() {
  const TensorValidationError error{TensorValidationCode::kNegativeExtent,
                                    TensorValidationField::kExtent, 2, 0,
                                    std::numeric_limits<uint64_t>::max(), 0};
  constexpr std::string_view expected =
      "tensor_validation_error code=negative_extent field=extent dimension=2 "
      "actual=-1 required=0";
  std::array<char, 160> buffer{};
  CHECK(format_tensor_validation_error(error, buffer) == expected.size());
  CHECK(std::string_view(buffer.data()) == expected);
  for (std::size_t size = 0; size <= expected.size() + 1; ++size) {
    buffer.fill('x');
    CHECK(format_tensor_validation_error(error,
                                         std::span<char>(buffer.data(), size)) ==
          expected.size());
    if (size != 0) {
      CHECK(buffer[size - 1] == '\0' || size > expected.size());
    }
  }
  CHECK(tensor_validation_code_name(static_cast<TensorValidationCode>(99)) ==
        "invalid_validation_code");
  CHECK(tensor_validation_field_name(static_cast<TensorValidationField>(99)) ==
        "invalid_validation_field");
  return true;
}
[[nodiscard]] bool ConcurrentValidationIsDeterministic() {
  const auto raw = const_view(transpose_metadata());
  std::array<std::atomic<uint64_t>, 8> digests{};
  std::array<std::thread, 8> workers{};
  for (uint32_t worker = 0; worker < workers.size(); ++worker) {
    workers[worker] = std::thread([&raw, &digests, worker]() {
      uint64_t digest = 0;
      for (uint32_t iteration = 0; iteration < 20000; ++iteration) {
        const Probe result = probe(validate(&raw));
        digest += result.valid && result.rank == 2 && !result.contiguous;
      }
      digests[worker].store(digest, std::memory_order_relaxed);
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& digest : digests) {
    CHECK(digest.load(std::memory_order_relaxed) == 20000);
  }
  return true;
}
[[nodiscard]] uint64_t next_random(uint64_t* state) noexcept {
  uint64_t value = *state;
  value ^= value << 13U;
  value ^= value >> 7U;
  value ^= value << 17U;
  *state = value;
  return value;
}
[[nodiscard]] bool MalformedDescriptorPropertyRun() {
  constexpr uint64_t kSeed = 0x53474c54454e534fULL;
  constexpr uint32_t kIterations = 1000000;
  uint64_t state = kSeed;
  const uint64_t allocations_before =
      g_allocation_count.load(std::memory_order_relaxed);
  for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
    auto value = metadata(SGL_NATIVE_DTYPE_UINT8,
                          1 + static_cast<uint32_t>(next_random(&state) % 4),
                          1);
    uint64_t reference_end = 1;
    for (uint32_t dimension = 0; dimension < value.rank; ++dimension) {
      value.extents[dimension] = 1 + next_random(&state) % 8;
      value.strides[dimension] = next_random(&state) % 9;
      reference_end += static_cast<uint64_t>(value.extents[dimension] - 1) *
                       static_cast<uint64_t>(value.strides[dimension]);
    }
    value.allocation_bytes = reference_end - 1;
    const void* base = g_storage.data();
    TensorValidationCode expected = TensorValidationCode::kOutOfBounds;
    switch (next_random(&state) % 16) {
      case 0: value.struct_size = 183; expected = TensorValidationCode::kMetadataSizeMismatch; break;
      case 1: value.abi_minor = 1; expected = TensorValidationCode::kAbiVersionMismatch; break;
      case 2: value.reserved[0] = 1; expected = TensorValidationCode::kReservedFieldNonZero; break;
      case 3: value.dtype = 999; expected = TensorValidationCode::kUnknownDType; break;
      case 4: value.device_kind = 99; expected = TensorValidationCode::kUnknownDevice; break;
      case 5: value.device_ordinal = -1; expected = TensorValidationCode::kInvalidDeviceOrdinal; break;
      case 6: value.rank = 9; expected = TensorValidationCode::kRankOutOfRange; break;
      case 7: value.extents[value.rank] = -3; expected = TensorValidationCode::kUnusedDimensionNonZero; break;
      case 8: value.extents[0] = -1; expected = TensorValidationCode::kNegativeExtent; break;
      case 9: value.strides[0] = -1; expected = TensorValidationCode::kNegativeStride; break;
      case 10: break;
      case 11: base = nullptr; expected = TensorValidationCode::kNullAllocation; break;
      case 12: value.storage_offset_elements = 1; break;
      case 13: value.dtype = SGL_NATIVE_DTYPE_NVFP4_E2M1; value.storage_offset_elements = 1; expected = TensorValidationCode::kMisalignedStorageOffset; break;
      case 14: value.allocation_bytes = uint64_t{1} << 63U; expected = TensorValidationCode::kValueOutOfDomain; break;
      case 15:
        value.rank = 1;
        for (uint32_t dimension = 0; dimension < SGL_NATIVE_TENSOR_MAX_RANK;
             ++dimension) {
          value.extents[dimension] = 0;
          value.strides[dimension] = 0;
        }
        value.allocation_bytes = 0;
        expected = TensorValidationCode::kInvalidEmptyView;
        break;
      default: return false;
    }
    const auto raw = const_view(value, base);
    const Probe first = probe(validate(&raw));
    const Probe second = probe(validate(&raw));
    if (!expect_code(first, expected) || !expect_code(second, expected) ||
        first.error.field != second.error.field ||
        first.error.dimension != second.error.dimension ||
        first.error.actual != second.error.actual ||
        first.error.required != second.error.required) {
      std::printf("seed=%llu iteration=%u expected=%u actual=%u rank=%u "
                  "allocation=%llu offset=%llu\n",
                  static_cast<unsigned long long>(kSeed), iteration,
                  static_cast<uint32_t>(expected),
                  static_cast<uint32_t>(first.error.code), value.rank,
                  static_cast<unsigned long long>(value.allocation_bytes),
                  static_cast<unsigned long long>(value.storage_offset_elements));
      return false;
    }
  }
  CHECK(g_allocation_count.load(std::memory_order_relaxed) ==
        allocations_before);
  return true;
}
[[nodiscard]] bool AllErrorCodesReachable() {
  using Code = TensorValidationCode;
  using Field = TensorValidationField;
  auto value = metadata();
#define EXPECT_ERROR(result, code, field, dimension, actual, required)        \
  CHECK(expect_error(result, code, field, dimension, actual, required))
#define MUTATION_ERROR(statement, code, field, dimension, actual, required)   \
  do {                                                                        \
    value = metadata(); statement;                                            \
    EXPECT_ERROR(validate_const(value), code, field, dimension, actual,       \
                 required);                                                   \
  } while (false)
  EXPECT_ERROR(probe(validate(static_cast<const SglNativeConstTensorViewV1*>(nullptr))),
               Code::kNullView, Field::kView, kNoDimension, 0, 1);
  MUTATION_ERROR(value.struct_size = 0, Code::kMetadataSizeMismatch,
                 Field::kStructSize, kNoDimension, 0, 184);
  MUTATION_ERROR(value.abi_major = 2, Code::kAbiVersionMismatch,
                 Field::kAbiVersion, kNoDimension, 0x20000, 0x10000);
  MUTATION_ERROR(value.reserved[0] = 1, Code::kReservedFieldNonZero,
                 Field::kReserved, 0, 1, 0);
  MUTATION_ERROR(value.dtype = 0, Code::kUnknownDType,
                 Field::kDType, kNoDimension, 0, 0);
  MUTATION_ERROR(value.device_kind = 0, Code::kUnknownDevice,
                 Field::kDeviceKind, kNoDimension, 0, 0);
  MUTATION_ERROR(value.device_ordinal = 1, Code::kInvalidDeviceOrdinal,
                 Field::kDeviceOrdinal, kNoDimension, 1, 0);
  MUTATION_ERROR(value.rank = 9, Code::kRankOutOfRange,
                 Field::kRank, kNoDimension, 9, 8);
  MUTATION_ERROR(value.extents[1] = 1, Code::kUnusedDimensionNonZero,
                 Field::kExtent, 1, 1, 0);
  MUTATION_ERROR(value.extents[0] = -1, Code::kNegativeExtent,
                 Field::kExtent, 0, UINT64_MAX, 0);
  MUTATION_ERROR(value.strides[0] = -1, Code::kNegativeStride,
                 Field::kStride, 0, UINT64_MAX, 0);
  MUTATION_ERROR(value.allocation_bytes = uint64_t{1} << 63U,
                 Code::kValueOutOfDomain, Field::kAllocationBytes,
                 kNoDimension, uint64_t{1} << 63U, INT64_MAX);
  MUTATION_ERROR(value.allocation_bytes = INT64_MAX, Code::kArithmeticOverflow,
                 Field::kAllocationBytes, kNoDimension, 0, 0);
  EXPECT_ERROR(validate_const(metadata(), nullptr), Code::kNullAllocation,
               Field::kAllocationBase, kNoDimension, 0, 1);
  value = metadata(SGL_NATIVE_DTYPE_UINT8, 1, 0);
  value.extents[0] = 0;
  EXPECT_ERROR(validate_const(value, g_storage.data()), Code::kInvalidEmptyView,
               Field::kAllocationBase, kNoDimension, 0, 0);
  EXPECT_ERROR(validate_const(metadata(SGL_NATIVE_DTYPE_UINT16),
                              reinterpret_cast<const void*>(uintptr_t{1})),
               Code::kMisalignedAllocationBase, Field::kAllocationBase,
               kNoDimension, 0, 2);
  MUTATION_ERROR(value.dtype = SGL_NATIVE_DTYPE_NVFP4_E2M1;
                     value.storage_offset_elements = 1,
                 Code::kMisalignedStorageOffset,
                 Field::kStorageOffsetElements, kNoDimension, 4, 0);
  EXPECT_ERROR(validate_const(metadata(SGL_NATIVE_DTYPE_UINT8, 1, 2),
                              reinterpret_cast<const void*>(UINTPTR_MAX)),
               Code::kPointerRangeOverflow, Field::kAllocationBase,
               kNoDimension, 0, 0);
  MUTATION_ERROR(value.allocation_bytes = 0, Code::kOutOfBounds,
                 Field::kSpan, kNoDimension, 8, 0);
  EXPECT_ERROR(validate_mutable(broadcast_metadata()), Code::kMutableOverlap,
               Field::kMutableLayout, 0, 0, 1);
  const auto raw = const_view(metadata());
  CHECK(std::move(validate(&raw)).match(
      [&](auto witness) {
        return expect_error(probe(narrow<DType::kUInt16, 1>(witness)),
                            Code::kDTypeMismatch, Field::kNarrowDType,
                            kNoDimension, 2, 4) &&
               expect_error(probe(narrow<DType::kUInt8, 2>(witness)),
                            Code::kRankMismatch, Field::kNarrowRank,
                            kNoDimension, 1, 2);
      },
      [](TensorValidationError) { return false; }));
  EXPECT_ERROR(validate_mutable(metadata(SGL_NATIVE_DTYPE_NVFP4_E2M1)),
               Code::kMutableSubByteUnsupported, Field::kMutableLayout,
               kNoDimension, 4, 8);
#undef MUTATION_ERROR
#undef EXPECT_ERROR
  return true;
}

struct TestCase final {
  const char* name;
  bool (*function)();
};

constexpr TestCase kTests[] = {
    {"AbiLayoutDefaultPack", AbiLayoutDefaultPack}, {"AbiLayoutAdversePack", AbiLayoutAdversePack},
    {"NumericIdentifiersAreStable", NumericIdentifiersAreStable}, {"CanonicalMetadataInitializer", CanonicalMetadataInitializer},
    {"PlatformContractIsExplicit", PlatformContractIsExplicit}, {"RankZeroScalarExactFit", RankZeroScalarExactFit},
    {"RankEightContiguous", RankEightContiguous}, {"EmptyNullAllocation", EmptyNullAllocation},
    {"EmptyRetainedAllocationAtEnd", EmptyRetainedAllocationAtEnd}, {"ReadonlyTranspose", ReadonlyTranspose},
    {"MutableTranspose", MutableTranspose}, {"ReadonlyZeroStrideBroadcast", ReadonlyZeroStrideBroadcast},
    {"MutableZeroStrideBroadcastRejected", MutableZeroStrideBroadcastRejected}, {"PaddedMutableLayout", PaddedMutableLayout},
    {"NvFp4EvenOffsetAndOddElementCount", NvFp4EvenOffsetAndOddElementCount}, {"NvFp4OddOffsetRejected", NvFp4OddOffsetRejected},
    {"MutableNvFp4Rejected", MutableNvFp4Rejected}, {"AllocationExactEnd", AllocationExactEnd},
    {"AllocationOneBitShort", AllocationOneBitShort}, {"UnknownDTypeAndDevice", UnknownDTypeAndDevice},
    {"InvalidDeviceOrdinals", InvalidDeviceOrdinals}, {"UnusedDimensionTailRejected", UnusedDimensionTailRejected},
    {"NegativeExtentAndStrideRejected", NegativeExtentAndStrideRejected}, {"CapacityAndOffsetDomainRejected", CapacityAndOffsetDomainRejected},
    {"ArithmeticOverflowExtentStride", ArithmeticOverflowExtentStride}, {"PointerRangeOverflow", PointerRangeOverflow},
    {"MutableIrregularOverlapRejected", MutableIrregularOverlapRejected}, {"DTypeAndRankNarrowing", DTypeAndRankNarrowing},
    {"OutcomeErrorUsesValueCategory", OutcomeErrorUsesValueCategory},
    {"ValidationDoesNotAllocate", ValidationDoesNotAllocate}, {"ValidationDoesNotDereference", ValidationDoesNotDereference},
    {"ErrorsRedactAllocationAddress", ErrorsRedactAllocationAddress}, {"ConcurrentValidationIsDeterministic", ConcurrentValidationIsDeterministic},
    {"MalformedDescriptorPropertyRun", MalformedDescriptorPropertyRun}, {"AllErrorCodesReachable", AllErrorCodesReachable},
};

}  // namespace

int main() {
  uint32_t passed = 0;
  for (const auto& test : kTests) {
    std::printf("[ RUN      ] %s\n", test.name);
    if (!test.function()) {
      std::printf("[  FAILED  ] %s\n", test.name);
      return 1;
    }
    ++passed;
    std::printf("[       OK ] %s\n", test.name);
  }
  std::printf("[  PASSED  ] %u tests\n", passed);
  return 0;
}
