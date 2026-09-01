#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include <sglang/benchmark/config.hpp>

namespace sglang::benchmark {

using Sha256Digest = std::array<std::uint8_t, 32>;

class Sha256 {
public:
  Sha256() noexcept;

  Sha256 &update(std::span<const std::byte> input);
  Sha256 &update(std::string_view input);

  // Finalization is idempotent. Updating a finalized hash throws logic_error.
  [[nodiscard]] Sha256Digest finalize();
  [[nodiscard]] std::string final_hex();

private:
  void transform(const std::byte *block) noexcept;

  std::array<std::uint32_t, 8> state_{};
  std::array<std::byte, 64> buffer_{};
  std::uint64_t total_size_ = 0;
  std::size_t buffer_size_ = 0;
  bool finalized_ = false;
  Sha256Digest digest_{};
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::byte> input);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(std::span<const std::byte> input);
[[nodiscard]] std::string sha256_hex(std::string_view input);

} // namespace sglang::benchmark
