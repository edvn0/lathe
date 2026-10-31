#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "storage/cache_store.hxx"

// Version of the cooked-texture format and encoder settings. Bump it when what a given key produces changes; every
// cached texture of an older version is then dropped.
inline constexpr std::uint32_t texture_cache_version = 3;

// The service the texture pipeline caches its KTX2 files through. A copy is cheap and shares the store. Without a store
// (caching is off) everything misses and nothing is kept.
class TextureCache {
public:
    static constexpr std::string_view namespace_name = "ktx2";

    // On the process-wide store (caches::store()).
    [[nodiscard]] static auto global() -> TextureCache;

    TextureCache() = default;
    explicit TextureCache(std::shared_ptr<CacheStore> store);
    // On a store of its own in 'directory', for tools and tests.
    explicit TextureCache(std::filesystem::path const &directory);

    [[nodiscard]] auto enabled() const noexcept -> bool { return store_ != nullptr; }

    // The cooked file for 'key', if there is one.
    [[nodiscard]] auto find(std::string_view key) const -> std::optional<std::filesystem::path>;

    // Where to write the cooked file for 'key'; 'label' is only for a readable file name.
    [[nodiscard]] auto begin(std::string_view key, std::string_view label) const -> std::optional<PendingCacheFile>;

    // Installs a file written to begin()'s path. 'source' is the file it was cooked from, so the entry can be pruned
    // when that is gone.
    auto commit(PendingCacheFile &&pending, std::string_view source) const -> bool;

    // Waits while another process or thread is producing the file for 'key', then holds the right to. Look again after
    // taking it: whoever held it before you has probably finished the job. Empty when caching is off or the wait was
    // given up (shutdown, a producer that never finished).
    [[nodiscard]] auto lock(std::string_view key) const -> CacheLock;

    // Forgets 'key', after its file turned out to be unreadable.
    auto discard(std::string_view key) const -> void;

    // Drops cooked textures whose source file no longer exists.
    auto prune_missing_sources() const -> CacheReclaimed;

private:
    std::shared_ptr<CacheStore> store_;
};
