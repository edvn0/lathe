#include "core/sha256.hxx"

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <array>
#include <string_view>

namespace {

    // OpenSSL keeps error state per thread, allocated the first time a thread uses it. By default OpenSSL registers a
    // cleanup with atexit() that runs before the static destructors of a thread pool created earlier, so the pool's threads
    // end after the cleanup and their state is never freed (LeakSanitizer reports it). Without the atexit cleanup each
    // thread frees its own state when it ends, and everything else stays reachable until the process exits.
    auto initialise_openssl() noexcept -> bool {
        return OPENSSL_init_crypto(OPENSSL_INIT_NO_ATEXIT, nullptr) == 1;
    }

}

auto sha256_hex(std::span<std::byte const> data) -> std::string {
    static auto const initialised = initialise_openssl();

    if (!initialised) {
        return {};
    }

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
