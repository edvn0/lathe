#include "core/sha256.hxx"

#include <openssl/evp.h>

#include <array>
#include <string_view>

auto sha256_hex(std::span<std::byte const> data) -> std::string {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    unsigned int digest_size = 0;

    if (EVP_Digest(data.data(), data.size(), digest.data(), &digest_size, EVP_sha256(), nullptr) != 1) {
        return {};
    }

    constexpr std::string_view digits = "0123456789abcdef";
    std::string hex;
    hex.reserve(static_cast<std::size_t>(digest_size) * 2U);

    for (unsigned int i = 0; i < digest_size; ++i) {
        hex.push_back(digits[digest[i] >> 4U]);
        hex.push_back(digits[digest[i] & 0xFU]);
    }

    return hex;
}
