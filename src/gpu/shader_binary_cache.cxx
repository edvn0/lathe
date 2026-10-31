#include "gpu/shader_binary_cache.hxx"

#include <format>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/sha256.hxx"

auto ShaderBinaryCache::create(std::shared_ptr<CacheStore> store, std::array<std::uint8_t, VK_UUID_SIZE> binary_uuid,
                               std::uint32_t binary_version)
        -> std::expected<ShaderBinaryCache, ShaderBinaryCacheError> {
    if (store == nullptr) {
        return std::unexpected(ShaderBinaryCacheError{.type = ShaderBinaryCacheErrorType::invalid_argument});
    }

    auto declared = store->declare(CacheNamespaceSpec{
            .name = std::string{namespace_name},
            .version = 1,
            .kind = CacheKind::inline_blob,
    });

    if (!declared) {
        return std::unexpected(ShaderBinaryCacheError{.type = ShaderBinaryCacheErrorType::store_failed,
                                                      .cause = std::move(declared.error())});
    }

    ShaderBinaryCache cache;
    cache.store_ = std::move(store);

    for (auto const byte: binary_uuid) {
        cache.driver_ += std::format("{:02x}", byte);
    }

    cache.driver_ += std::format("_v{}", binary_version);

    return cache;
}

auto ShaderBinaryCache::full_key(std::string_view cache_key, std::span<std::uint32_t const> spirv) const
        -> std::string {
    return std::format("{}|{}|{}", driver_, cache_key, sha256_hex(std::as_bytes(spirv)));
}

auto ShaderBinaryCache::find(std::string_view cache_key, std::span<std::uint32_t const> spirv) const
        -> std::optional<std::vector<std::byte>> {
    if (store_ == nullptr) {
        return std::nullopt;
    }

    auto binary = store_->get_blob(namespace_name, full_key(cache_key, spirv));

    if (binary && binary->empty()) {
        binary.reset();
    }

    debug("ShaderBinaryCache: {} '{}'", binary ? "hit" : "miss", cache_key);

    return binary;
}

auto ShaderBinaryCache::store(std::string_view cache_key, std::span<std::uint32_t const> spirv,
                              std::span<std::byte const> binary) const -> void {
    if (store_ == nullptr || binary.empty()) {
        return;
    }

    if (auto const stored = store_->put_blob(namespace_name, full_key(cache_key, spirv), binary); !stored) {
        warn("ShaderBinaryCache: {}", describe(stored.error()));
    }
}

auto ShaderBinaryCache::discard(std::string_view cache_key, std::span<std::uint32_t const> spirv) const -> void {
    if (store_ != nullptr) {
        store_->erase(namespace_name, full_key(cache_key, spirv));
    }
}
