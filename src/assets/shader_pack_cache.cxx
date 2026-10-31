#include "assets/shader_pack_cache.hxx"

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "storage/caches.hxx"

auto ShaderPackCache::global() -> ShaderPackCache { return ShaderPackCache{caches::store()}; }

ShaderPackCache::ShaderPackCache(std::shared_ptr<CacheStore> store) : store_(std::move(store)) {
    if (store_ == nullptr) {
        return;
    }

    auto const declared = store_->declare(CacheNamespaceSpec{
            .name = std::string{namespace_name},
            .version = 1,
            .kind = CacheKind::file,
            .extension = ".lsp",
    });

    if (!declared) {
        warn("ShaderPackCache: {}", describe(declared.error()));
        store_.reset();
    }
}

auto ShaderPackCache::find(std::string_view key) const -> std::optional<std::filesystem::path> {
    if (store_ == nullptr) {
        return std::nullopt;
    }

    // Packs of checkouts that were deleted go before they can be found.
    static_cast<void>(store_->prune_missing_sources(namespace_name));

    return store_->find_file(namespace_name, key);
}

auto ShaderPackCache::begin(std::string_view key) const -> std::optional<PendingCacheFile> {
    if (store_ == nullptr) {
        return std::nullopt;
    }

    auto pending = store_->begin_file(namespace_name, key, "shaders");

    if (!pending) {
        warn("ShaderPackCache: {}", describe(pending.error()));

        return std::nullopt;
    }

    return std::move(*pending);
}

auto ShaderPackCache::commit(PendingCacheFile &&pending, std::string_view source) const -> bool {
    if (store_ == nullptr) {
        return false;
    }

    auto const committed = store_->commit(std::move(pending), CacheEntryInfo{.source = std::string{source}});

    if (!committed) {
        warn("ShaderPackCache: {}", describe(committed.error()));
    }

    return committed.has_value();
}
