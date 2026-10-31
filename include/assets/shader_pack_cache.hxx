#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>

#include "storage/cache_store.hxx"

// Shader packs a dev run records for itself (one per checkout) so the next run skips compiling. The service the engine
// start-up caches them through; a pack beside the data (shaders.lsp) or named with --shader-pack never comes here.
class ShaderPackCache {
public:
    static constexpr std::string_view namespace_name = "shader_packs";

    [[nodiscard]] static auto global() -> ShaderPackCache;

    ShaderPackCache() = default;
    explicit ShaderPackCache(std::shared_ptr<CacheStore> store);

    // The pack recorded for 'key' (the checkout's engine root).
    [[nodiscard]] auto find(std::string_view key) const -> std::optional<std::filesystem::path>;

    // Where to write the pack for 'key'.
    [[nodiscard]] auto begin(std::string_view key) const -> std::optional<PendingCacheFile>;

    // Installs a pack written to begin()'s path. 'source' is the shader directory it was compiled from, so the pack can
    // be pruned once that checkout is gone.
    auto commit(PendingCacheFile &&pending, std::string_view source) const -> bool;

private:
    std::shared_ptr<CacheStore> store_;
};
