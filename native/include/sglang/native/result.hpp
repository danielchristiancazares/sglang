#ifndef SGLANG_NATIVE_RESULT_HPP_
#define SGLANG_NATIVE_RESULT_HPP_

#include <concepts>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

namespace sglang::native {

template <typename T, typename E>
class [[nodiscard]] Result final {
  static_assert(!std::is_reference_v<T>);
  static_assert(!std::is_reference_v<E>);
  static_assert(std::is_nothrow_move_constructible_v<T>);
  static_assert(std::is_nothrow_destructible_v<T>);
  static_assert(std::is_nothrow_move_constructible_v<E>);
  static_assert(std::is_nothrow_destructible_v<E>);

 public:
  Result() = delete;
  Result(const Result&) = delete;
  Result& operator=(const Result&) = delete;
  Result& operator=(Result&&) = delete;

  Result(Result&& other) noexcept : has_value_(other.has_value_) {
    if (has_value_) {
      std::construct_at(&storage_.value, std::move(other.storage_.value));
    } else {
      std::construct_at(&storage_.error, std::move(other.storage_.error));
    }
  }

  ~Result() noexcept {
    if (has_value_) {
      std::destroy_at(&storage_.value);
    } else {
      std::destroy_at(&storage_.error);
    }
  }

  [[nodiscard]] static Result success(T&& value) noexcept {
    return Result(ValueTag{}, std::move(value));
  }

  [[nodiscard]] static Result failure(E error) noexcept {
    return Result(ErrorTag{}, std::move(error));
  }

  [[nodiscard]] constexpr bool has_value() const noexcept {
    return has_value_;
  }

  template <
      typename OnValue, typename OnError,
      typename ValueResult = std::invoke_result_t<OnValue, T&&>,
      typename ErrorResult = std::invoke_result_t<OnError, E&&>>
    requires std::same_as<ValueResult, ErrorResult> &&
             (!std::is_reference_v<ValueResult>)
  ValueResult match(OnValue&& on_value, OnError&& on_error) &&
      noexcept(noexcept(std::invoke(std::forward<OnValue>(on_value),
                                    std::declval<T&&>())) &&
               noexcept(std::invoke(std::forward<OnError>(on_error),
                                    std::declval<E&&>()))) {
    if (has_value_) {
      return std::invoke(std::forward<OnValue>(on_value),
                         std::move(storage_.value));
    }
    return std::invoke(std::forward<OnError>(on_error),
                       std::move(storage_.error));
  }

 private:
  struct ValueTag final {};
  struct ErrorTag final {};

  union Storage {
    char empty;
    T value;
    E error;

    constexpr Storage() noexcept : empty(0) {}
    explicit Storage(ValueTag, T&& input) noexcept
        : value(std::move(input)) {}
    explicit Storage(ErrorTag, E input) noexcept
        : error(std::move(input)) {}
    ~Storage() noexcept {}
  };

  explicit Result(ValueTag tag, T&& value) noexcept
      : storage_(tag, std::move(value)), has_value_(true) {}

  explicit Result(ErrorTag tag, E error) noexcept
      : storage_(tag, std::move(error)), has_value_(false) {}

  Storage storage_;
  bool has_value_;
};

}  // namespace sglang::native

#endif  // SGLANG_NATIVE_RESULT_HPP_
