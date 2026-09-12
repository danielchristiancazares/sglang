#ifndef SGLANG_NATIVE_SHA256_HPP_
#define SGLANG_NATIVE_SHA256_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace sglang::native {

using Sha256Digest = std::array<uint8_t, 32>;

class Sha256 final {
public:
  Sha256() noexcept;

  [[nodiscard]] bool update(std::span<const std::byte> input) noexcept;
  [[nodiscard]] bool update(std::string_view input) noexcept;
  [[nodiscard]] Sha256Digest finalize() noexcept;
  [[nodiscard]] bool finalized() const noexcept { return finalized_; }

private:
  void transform(const std::byte *block) noexcept;

  std::array<uint32_t, 8> state_{};
  std::array<std::byte, 64> buffer_{};
  uint64_t total_size_{0};
  size_t buffer_size_{0};
  bool finalized_{false};
  Sha256Digest digest_{};
};

[[nodiscard]] std::string sha256_hex(const Sha256Digest &digest);
[[nodiscard]] Sha256Digest sha256(std::span<const std::byte> input) noexcept;
[[nodiscard]] Sha256Digest sha256(std::string_view input) noexcept;

} // namespace sglang::native

#endif // SGLANG_NATIVE_SHA256_HPP_
