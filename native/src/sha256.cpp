#include "sglang/native/sha256.hpp"

#include <algorithm>
#include <bit>
#include <limits>

namespace sglang::native {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants{
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

constexpr uint64_t kMaximumMessageBytes =
    std::numeric_limits<uint64_t>::max() / 8U;

} // namespace

Sha256::Sha256() noexcept
    : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
             0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

void Sha256::transform(const std::byte *block) noexcept {
  std::array<uint32_t, 64> words{};
  for (size_t index = 0; index < 16U; ++index) {
    const size_t offset = index * 4U;
    words[index] = (std::to_integer<uint32_t>(block[offset]) << 24U) |
                   (std::to_integer<uint32_t>(block[offset + 1U]) << 16U) |
                   (std::to_integer<uint32_t>(block[offset + 2U]) << 8U) |
                   std::to_integer<uint32_t>(block[offset + 3U]);
  }
  for (size_t index = 16U; index < words.size(); ++index) {
    const uint32_t sigma0 = std::rotr(words[index - 15U], 7) ^
                            std::rotr(words[index - 15U], 18) ^
                            (words[index - 15U] >> 3U);
    const uint32_t sigma1 = std::rotr(words[index - 2U], 17) ^
                            std::rotr(words[index - 2U], 19) ^
                            (words[index - 2U] >> 10U);
    words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
  }

  uint32_t a = state_[0];
  uint32_t b = state_[1];
  uint32_t c = state_[2];
  uint32_t d = state_[3];
  uint32_t e = state_[4];
  uint32_t f = state_[5];
  uint32_t g = state_[6];
  uint32_t h = state_[7];
  for (size_t index = 0; index < words.size(); ++index) {
    const uint32_t sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
    const uint32_t choose = (e & f) ^ (~e & g);
    const uint32_t temporary1 =
        h + sum1 + choose + kRoundConstants[index] + words[index];
    const uint32_t sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
    const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t temporary2 = sum0 + majority;
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

bool Sha256::update(std::span<const std::byte> input) noexcept {
  if (finalized_ || input.size() > kMaximumMessageBytes - total_size_) {
    return false;
  }
  total_size_ += static_cast<uint64_t>(input.size());
  size_t position = 0;
  if (buffer_size_ != 0U) {
    const size_t copied = std::min(buffer_.size() - buffer_size_, input.size());
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
  return true;
}

bool Sha256::update(std::string_view input) noexcept {
  return update(std::span<const std::byte>(
      reinterpret_cast<const std::byte *>(input.data()), input.size()));
}

Sha256Digest Sha256::finalize() noexcept {
  if (finalized_) {
    return digest_;
  }
  const uint64_t bit_length = total_size_ * 8U;
  buffer_[buffer_size_++] = std::byte{0x80};
  if (buffer_size_ > 56U) {
    std::fill(buffer_.begin() + static_cast<ptrdiff_t>(buffer_size_),
              buffer_.end(), std::byte{0});
    transform(buffer_.data());
    buffer_size_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<ptrdiff_t>(buffer_size_),
            buffer_.begin() + 56, std::byte{0});
  for (size_t index = 0; index < 8U; ++index) {
    buffer_[56U + index] = static_cast<std::byte>(
        bit_length >> (56U - static_cast<unsigned>(index) * 8U));
  }
  transform(buffer_.data());
  for (size_t index = 0; index < state_.size(); ++index) {
    digest_[index * 4U] = static_cast<uint8_t>(state_[index] >> 24U);
    digest_[index * 4U + 1U] = static_cast<uint8_t>(state_[index] >> 16U);
    digest_[index * 4U + 2U] = static_cast<uint8_t>(state_[index] >> 8U);
    digest_[index * 4U + 3U] = static_cast<uint8_t>(state_[index]);
  }
  buffer_size_ = 0;
  finalized_ = true;
  return digest_;
}

std::string sha256_hex(const Sha256Digest &digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output(digest.size() * 2U, '\0');
  for (size_t index = 0; index < digest.size(); ++index) {
    output[index * 2U] = kHex[digest[index] >> 4U];
    output[index * 2U + 1U] = kHex[digest[index] & 0x0fU];
  }
  return output;
}

Sha256Digest sha256(std::span<const std::byte> input) noexcept {
  Sha256 hash;
  static_cast<void>(hash.update(input));
  return hash.finalize();
}

Sha256Digest sha256(std::string_view input) noexcept {
  Sha256 hash;
  static_cast<void>(hash.update(input));
  return hash.finalize();
}

} // namespace sglang::native
