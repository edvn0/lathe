#include "storage/download_cache.hxx"

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "storage/caches.hxx"

namespace {
    // Moves an earlier version's file into the store; across file systems that is a copy.
    auto adopt(std::filesystem::path const &legacy, std::filesystem::path const &destination) -> void {
        std::error_code error;

        if (legacy.empty() || std::filesystem::exists(destination, error) || !std::filesystem::exists(legacy, error)) {
            return;
        }

        std::filesystem::rename(legacy, destination, error);

        if (!error) {
            info("DownloadCache: moved '{}' into the cache store", legacy.string());
            return;
        }

        std::filesystem::copy_file(legacy, destination, error);

        if (error) {
            warn("DownloadCache: could not adopt '{}': {}", legacy.string(), error.message());
            std::filesystem::remove(destination, error);
            return;
        }

        std::filesystem::remove(legacy, error);
        info("DownloadCache: copied '{}' into the cache store", legacy.string());
    }
}

auto DownloadCache::global() -> DownloadCache { return DownloadCache{caches::store()}; }

DownloadCache::DownloadCache(std::shared_ptr<CacheStore> store) : store_(std::move(store)) {
    if (store_ == nullptr) {
        return;
    }

    auto const declared = store_->declare(CacheNamespaceSpec{
            .name = std::string{namespace_name},
            .version = 1,
            .kind = CacheKind::file,
            .evictable = false,
    });

    if (!declared) {
        warn("DownloadCache: {}", describe(declared.error()));
        store_.reset();
    }
}

auto DownloadCache::path(std::string_view key, std::string_view label, std::string_view extension,
                         std::filesystem::path const &legacy) const -> std::filesystem::path {
    if (store_ == nullptr) {
        return {};
    }

    auto reserved = store_->reserve_file(namespace_name, key, label, extension);

    if (!reserved) {
        warn("DownloadCache: {}", describe(reserved.error()));

        return {};
    }

    adopt(legacy, *reserved);

    return *std::move(reserved);
}

auto DownloadCache::find(std::string_view key, std::string_view label, std::string_view extension,
                         std::filesystem::path const &legacy) const -> std::optional<std::filesystem::path> {
    if (store_ == nullptr) {
        return std::nullopt;
    }

    if (auto const found = store_->find_file(namespace_name, key)) {
        return found;
    }

    // Not recorded yet: reserve only when there is a file to adopt, so lookups of things never downloaded leave no entry.
    std::error_code error;

    if (legacy.empty() || !std::filesystem::exists(legacy, error)) {
        return std::nullopt;
    }

    auto const adopted = path(key, label, extension, legacy);

    if (adopted.empty() || !std::filesystem::exists(adopted, error)) {
        return std::nullopt;
    }

    return adopted;
}
