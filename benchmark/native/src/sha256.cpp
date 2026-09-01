#include "sglang/benchmark/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sglang::benchmark {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

constexpr std::uint64_t kMaximumMessageBytes =
    std::numeric_limits<std::uint64_t>::max() / 8U;

std::string digest_to_hex(const Sha256Digest &digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output;
  output.resize(digest.size() * 2);
  for (std::size_t i = 0; i < digest.size(); ++i) {
    output[i * 2] = kHex[digest[i] >> 4U];
    output[i * 2 + 1] = kHex[digest[i] & 0x0fU];
  }
  return output;
}

} // namespace

Sha256::Sha256() noexcept
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::transform(const std::byte *block) noexcept {
  std::array<std::uint32_t, 64> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    const std::size_t offset = i * 4;
    words[i] = (std::to_integer<std::uint32_t>(block[offset]) << 24U) |
               (std::to_integer<std::uint32_t>(block[offset + 1]) << 16U) |
               (std::to_integer<std::uint32_t>(block[offset + 2]) << 8U) |
               std::to_integer<std::uint32_t>(block[offset + 3]);
  }
  for (std::size_t i = 16; i < words.size(); ++i) {
    const std::uint32_t sigma0 = std::rotr(words[i - 15], 7) ^
                                 std::rotr(words[i - 15], 18) ^
                                 (words[i - 15] >> 3U);
    const std::uint32_t sigma1 = std::rotr(words[i - 2], 17) ^
                                 std::rotr(words[i - 2], 19) ^
                                 (words[i - 2] >> 10U);
    words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];

  for (std::size_t i = 0; i < words.size(); ++i) {
    const std::uint32_t sum1 =
        std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const std::uint32_t choose = (e & f) ^ (~e & g);
    const std::uint32_t temporary1 =
        h + sum1 + choose + kRoundConstants[i] + words[i];
    const std::uint32_t sum0 =
        std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temporary2 = sum0 + majority;

    h = g;
    g = f;
    f = e;
    e = d + temporary1;
    d = c;
    c = b;
    b = a;
    a = temporary1 + temporary2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

Sha256 &Sha256::update(std::span<const std::byte> input) {
  if (finalized_) {
    throw std::logic_error("cannot update a finalized SHA-256 hash");
  }
  if (input.size() > kMaximumMessageBytes - total_size_) {
    throw std::length_error("SHA-256 message exceeds the 64-bit length field");
  }
  total_size_ += static_cast<std::uint64_t>(input.size());

  std::size_t position = 0;
  if (buffer_size_ != 0) {
    const std::size_t copied =
        std::min(buffer_.size() - buffer_size_, input.size());
    std::copy_n(input.data(), copied, buffer_.data() + buffer_size_);
    buffer_size_ += copied;
    position += copied;
    if (buffer_size_ == buffer_.size()) {
      transform(buffer_.data());
      buffer_size_ = 0;
    }
  }

  while (input.size() - position >= buffer_.size()) {
    transform(input.data() + position);
    position += buffer_.size();
  }
  if (position != input.size()) {
    buffer_size_ = input.size() - position;
    std::copy_n(input.data() + position, buffer_size_, buffer_.data());
  }
  return *this;
}

Sha256 &Sha256::update(std::string_view input) {
  const auto *data = reinterpret_cast<const std::byte *>(input.data());
  return update(std::span<const std::byte>(data, input.size()));
}

Sha256Digest Sha256::finalize() {
  if (finalized_) {
    return digest_;
  }

  const std::uint64_t bit_length = total_size_ * 8U;
  buffer_[buffer_size_++] = std::byte{0x80};
  if (buffer_size_ > 56) {
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
              buffer_.end(), std::byte{0});
    transform(buffer_.data());
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
            buffer_.begin() + 56, std::byte{0});
  for (std::size_t i = 0; i < 8; ++i) {
    buffer_[56 + i] = static_cast<std::byte>(
        bit_length >> (56U - static_cast<unsigned>(i) * 8U));
  }
  transform(buffer_.data());

  for (std::size_t i = 0; i < state_.size(); ++i) {
    digest_[i * 4] = static_cast<std::uint8_t>(state_[i] >> 24U);
    digest_[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16U);
    digest_[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8U);
    digest_[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
  }
  buffer_size_ = 0;
  finalized_ = true;
  return digest_;
}

std::string Sha256::final_hex() { return digest_to_hex(finalize()); }

Sha256Digest sha256(std::span<const std::byte> input) {
  Sha256 hash;
  hash.update(input);
  return hash.finalize();
}

Sha256Digest sha256(std::string_view input) {
  Sha256 hash;
  hash.update(input);
  return hash.finalize();
}

std::string sha256_hex(std::span<const std::byte> input) {
  return digest_to_hex(sha256(input));
}

std::string sha256_hex(std::string_view input) {
  return digest_to_hex(sha256(input));
}

} // namespace sglang::benchmark
