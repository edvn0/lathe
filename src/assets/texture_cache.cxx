#include "assets/texture_cache.hxx"

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "storage/caches.hxx"

namespace {
    auto declare_namespace(std::shared_ptr<CacheStore> const &store) -> bool {
        auto const declared = store->declare(CacheNamespaceSpec{
                .name = std::string{TextureCache::namespace_name},
                .version = texture_cache_version,
                .kind = CacheKind::file,
                .extension = ".ktx2",
        });

        if (!declared) {
            warn("TextureCache: {}", describe(declared.error()));
        }

        return declared.has_value();
    }
}

auto TextureCache::global() -> TextureCache { return TextureCache{caches::store()}; }

TextureCache::TextureCache(std::shared_ptr<CacheStore> store) : store_(std::move(store)) {
    if (store_ != nullptr && !declare_namespace(store_)) {
        store_.reset();
    }
}

TextureCache::TextureCache(std::filesystem::path const &directory) {
    if (auto store = CacheStore::open(directory)) {
        store_ = *std::move(store);

        if (!declare_namespace(store_)) {
            store_.reset();
        }
    } else {
        warn("TextureCache: {}", describe(store.error()));
    }
}

auto TextureCache::find(std::string_view key) const -> std::optional<std::filesystem::path> {
    return store_ != nullptr ? store_->find_file(namespace_name, key) : std::nullopt;
}

auto TextureCache::begin(std::string_view key, std::string_view label) const -> std::optional<PendingCacheFile> {
    if (store_ == nullptr) {
        return std::nullopt;
    }

    auto pending = store_->begin_file(namespace_name, key, label);

    if (!pending) {
        warn("TextureCache: {}", describe(pending.error()));

        return std::nullopt;
    }

    return std::move(*pending);
}

auto TextureCache::commit(PendingCacheFile &&pending, std::string_view source) const -> bool {
    if (store_ == nullptr) {
        return false;
    }

    auto const committed = store_->commit(std::move(pending), CacheEntryInfo{.source = std::string{source}});

    if (!committed) {
        warn("TextureCache: {}", describe(committed.error()));
    }

    return committed.has_value();
}

auto TextureCache::discard(std::string_view key) const -> void {
    if (store_ != nullptr) {
        store_->erase(namespace_name, key);
    }
}

auto TextureCache::prune_missing_sources() const -> CacheReclaimed {
    return store_ != nullptr ? store_->prune_missing_sources(namespace_name) : CacheReclaimed{};
}
