#include "core/sha256.hxx"

#include <openssl/evp.h>

#include <array>
#include <string_view>

auto sha256_hex(std::span<std::byte const> data) -> std::string {
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
    std::size_t digest_size = digest.size();

    if (EVP_Q_digest(nullptr, "SHA256", nullptr, data.data(), data.size(), digest.data(), &digest_size) != 1) {
        return {};
    }

    constexpr std::string_view digits = "0123456789abcdef";

    std::string hex;
    hex.reserve(digest_size * 2);

    for (std::size_t i = 0; i < digest_size; ++i) {
        hex.push_back(digits[digest[i] >> 4]);
        hex.push_back(digits[digest[i] & 0x0f]);
    }

    return hex;
}
