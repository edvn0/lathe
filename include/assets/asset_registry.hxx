#pragma once

#include <algorithm>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "assets/material.hxx"
#include "assets/model.hxx"
#include "core/logger.hxx"
#include "gpu/image.hxx"
#include "scene/script_handle.hxx"

// Human-readable names for asset handles, for the editor's pickers. Separate from the *Storage classes, which
// don't know about names.
template<typename HandleT>
class NamedAssetTable {
public:
    struct Entry {
        std::string name;
        HandleT handle;
    };

    // Returns false and logs a warning if the name is taken; never overwrites.
    auto register_asset(std::string name, HandleT handle) -> bool {
        if (find(name).valid()) {
            warn("AssetRegistry: name '{}' is already registered, ignoring", name);
            return false;
        }

        auto const position = std::ranges::upper_bound(entries_, name, {}, &Entry::name);
        entries_.insert(position, Entry{.name = std::move(name), .handle = handle});
        return true;
    }

    auto unregister(HandleT handle) -> void {
        std::erase_if(entries_, [handle](Entry const &entry) { return entry.handle == handle; });
    }

    [[nodiscard]]
    auto find(std::string_view name) const noexcept -> HandleT {
        auto const it = std::ranges::find(entries_, name, &Entry::name);
        return it != entries_.end() ? it->handle : HandleT{};
    }

    // Empty if `handle` has no name.
    [[nodiscard]]
    auto name_of(HandleT handle) const noexcept -> std::string_view {
        auto const it = std::ranges::find(entries_, handle, &Entry::handle);
        return it != entries_.end() ? std::string_view{it->name} : std::string_view{};
    }

    // Sorted by name.
    [[nodiscard]]
    auto entries() const noexcept -> std::span<Entry const> {
        return entries_;
    }

private:
    std::vector<Entry> entries_;
};

// One NamedAssetTable per asset kind. Lookups are linear, which is fine at editor scale.
class AssetRegistry {
public:
    [[nodiscard]] auto models() noexcept -> NamedAssetTable<ModelHandle> & { return models_; }
    [[nodiscard]] auto models() const noexcept -> NamedAssetTable<ModelHandle> const & { return models_; }

    [[nodiscard]] auto materials() noexcept -> NamedAssetTable<MaterialHandle> & { return materials_; }
    [[nodiscard]] auto materials() const noexcept -> NamedAssetTable<MaterialHandle> const & { return materials_; }

    [[nodiscard]] auto scripts() noexcept -> NamedAssetTable<ScriptHandle> & { return scripts_; }
    [[nodiscard]] auto scripts() const noexcept -> NamedAssetTable<ScriptHandle> const & { return scripts_; }

    [[nodiscard]] auto textures() noexcept -> NamedAssetTable<ImageHandle> & { return textures_; }
    [[nodiscard]] auto textures() const noexcept -> NamedAssetTable<ImageHandle> const & { return textures_; }

private:
    NamedAssetTable<ModelHandle> models_;
    NamedAssetTable<MaterialHandle> materials_;
    NamedAssetTable<ScriptHandle> scripts_;
    NamedAssetTable<ImageHandle> textures_;
};
