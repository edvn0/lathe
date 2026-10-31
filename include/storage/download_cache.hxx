#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "storage/cache_store.hxx"

// Files fetched from the network (model galleries, map textures, what scripts download), kept in the cache store so they
// are listed, counted and cleared with everything else. The budget never evicts them: they cost a download to get back.
//
// The HTTP client writes and verifies these files itself, so the service hands out the path to write to rather than a
// pending file.
class DownloadCache {
public:
    static constexpr std::string_view namespace_name = "downloads";

    // On the process-wide store (caches::store()).
    [[nodiscard]] static auto global() -> DownloadCache;

    DownloadCache() = default;
    explicit DownloadCache(std::shared_ptr<CacheStore> store);

    [[nodiscard]] auto enabled() const noexcept -> bool { return store_ != nullptr; }

    // Where to write the download for 'key' (for example "gallery/ToyCar" or a SHA-256). 'label' and 'extension' only
    // shape the file name. 'legacy' is where earlier versions kept it: a file there is moved in, so nothing is fetched
    // twice. Empty when caching is off.
    [[nodiscard]] auto path(std::string_view key, std::string_view label, std::string_view extension,
                            std::filesystem::path const &legacy = {}) const -> std::filesystem::path;

    // The file for 'key' if it is there (adopting 'legacy' as path() does).
    [[nodiscard]] auto find(std::string_view key, std::string_view label, std::string_view extension,
                            std::filesystem::path const &legacy = {}) const -> std::optional<std::filesystem::path>;

private:
    std::shared_ptr<CacheStore> store_;
};
