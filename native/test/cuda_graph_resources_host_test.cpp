#include "sglang/native/cuda_graph_resources.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using sglang::native::CudaExecutionContext;
using sglang::native::CudaGraphExecutable;
using sglang::native::CudaStream;
using sglang::native::GraphArenaLease;
using sglang::native::GraphMemoryArena;
using sglang::native::GraphMemorySlice;
using sglang::native::NativeRuntimeCode;
using sglang::native::NativeRuntimeError;
using sglang::native::NativeRuntimeOperation;
using sglang::native::Result;
using sglang::native::native_runtime_code_name;
using sglang::native::native_runtime_operation_name;

[[nodiscard]] bool record_check(bool passed, const char* expression,
                                int line) noexcept {
  if (!passed) {
    std::printf("%s:%d: check failed: %s\n", __FILE__, line, expression);
  }
  return passed;
}

#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!record_check(static_cast<bool>(condition), #condition, __LINE__)) { \
      return false;                                                         \
    }                                                                       \
  } while (false)

struct MoveProbe final {
  int* destruction_count;
  bool owns;

  explicit MoveProbe(int* count) noexcept
      : destruction_count(count), owns(true) {}
  MoveProbe(const MoveProbe&) = delete;
  MoveProbe& operator=(const MoveProbe&) = delete;
  MoveProbe(MoveProbe&& other) noexcept
      : destruction_count(other.destruction_count),
        owns(std::exchange(other.owns, false)) {}
  MoveProbe& operator=(MoveProbe&&) = delete;
  ~MoveProbe() noexcept {
    if (owns) {
      ++*destruction_count;
    }
  }
};

[[nodiscard]] bool ResultOwnsExactlyOneAlternative() {
  int destructions = 0;
  {
    auto result =
        Result<MoveProbe, NativeRuntimeError>::success(MoveProbe(&destructions));
    CHECK(result.has_value());
    const bool matched = std::move(result).match(
        [](MoveProbe&& value) noexcept {
          return value.owns && value.destruction_count != nullptr;
        },
        [](NativeRuntimeError&&) noexcept { return false; });
    CHECK(matched);
  }
  CHECK(destructions == 1);

  const NativeRuntimeError expected{
      NativeRuntimeCode::kInvalidArgument,
      NativeRuntimeOperation::kReserve,
      0,
      0,
      3,
      1};
  auto failure =
      Result<MoveProbe, NativeRuntimeError>::failure(expected);
  CHECK(!failure.has_value());
  return std::move(failure).match(
      [](MoveProbe&&) noexcept { return false; },
      [expected](NativeRuntimeError&& error) noexcept {
        return error.code == expected.code &&
               error.operation == expected.operation &&
               error.actual == expected.actual &&
               error.required == expected.required;
      });
}

[[nodiscard]] bool RuntimeIdentifiersAreStable() {
  constexpr std::array<std::string_view, 14> code_names{
      "ok",
      "invalid_argument",
      "invalid_state",
      "host_allocation_failed",
      "device_mismatch",
      "arithmetic_overflow",
      "out_of_capacity",
      "already_sealed",
      "not_sealed",
      "resource_busy",
      "foreign_slice",
      "owner_metadata_conflict",
      "cuda_runtime_failure",
      "tensor_validation_failure"};
  constexpr std::array<std::string_view, 24> operation_names{
      "none",
      "get_device",
      "stream_create",
      "stream_synchronize",
      "stream_destroy",
      "device_allocate",
      "device_free",
      "reserve",
      "seal",
      "acquire_lease",
      "bind_tensor",
      "graph_instantiate",
      "event_create",
      "graph_launch",
      "event_record",
      "event_synchronize",
      "graph_destroy",
      "event_destroy",
      "stream_get_flags",
      "validate_linear_rejection_sampling",
      "launch_linear_rejection_sampling",
      "validate_linear_verify_rng",
      "launch_seeded_linear_verify_rng",
      "launch_stateful_linear_verify_rng"};

  for (uint32_t index = 0; index < code_names.size(); ++index) {
    CHECK(native_runtime_code_name(
              static_cast<NativeRuntimeCode>(index)) == code_names[index]);
  }
  for (uint32_t index = 0; index < operation_names.size(); ++index) {
    CHECK(native_runtime_operation_name(
              static_cast<NativeRuntimeOperation>(index)) ==
          operation_names[index]);
  }
  CHECK(native_runtime_code_name(static_cast<NativeRuntimeCode>(99)) ==
        "invalid_runtime_code");
  CHECK(native_runtime_operation_name(
            static_cast<NativeRuntimeOperation>(99)) ==
        "invalid_runtime_operation");
  return true;
}

static_assert(sizeof(NativeRuntimeError) == 32);
static_assert(std::is_trivially_copyable_v<NativeRuntimeError>);
static_assert(std::is_nothrow_move_constructible_v<CudaStream>);
static_assert(std::is_nothrow_move_constructible_v<CudaExecutionContext>);
static_assert(std::is_nothrow_move_constructible_v<GraphMemoryArena>);
static_assert(std::is_nothrow_move_constructible_v<GraphMemorySlice>);
static_assert(std::is_nothrow_move_constructible_v<GraphArenaLease>);
static_assert(std::is_nothrow_move_constructible_v<CudaGraphExecutable>);
static_assert(!std::is_copy_constructible_v<CudaStream>);
static_assert(!std::is_copy_constructible_v<GraphMemoryArena>);
static_assert(!std::is_copy_constructible_v<CudaGraphExecutable>);
static_assert(std::is_copy_constructible_v<CudaExecutionContext>);
static_assert(std::is_copy_constructible_v<GraphMemorySlice>);
static_assert(std::is_copy_constructible_v<GraphArenaLease>);

}  // namespace

int main() {
  if (!ResultOwnsExactlyOneAlternative() ||
      !RuntimeIdentifiersAreStable()) {
    return 1;
  }
  std::printf("[  PASSED  ] 2 tests\n");
  return 0;
}
