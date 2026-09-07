#pragma once

#include <algorithm>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "assets/material.hxx" // MaterialHandle
#include "assets/model.hxx" // ModelHandle
#include "core/logger.hxx"
#include "gpu/image.hxx" // ImageHandle
#include "scene/script_handle.hxx" // ScriptHandle

// One human-readable name per asset handle, so editor UI (the Inspector's
// component pickers, the Assets browser panel -- see application.cxx) can
// let a user pick "the grass material" instead of typing a raw
// index/generation pair. Deliberately separate from the *Storage classes
// themselves (ModelStorage, MaterialStorage, ...): those own the actual
// asset data and know nothing about names -- this is a thin, storage-
// agnostic naming layer on top, populated by whoever loads/creates a
// reusable asset (see AssetRegistry below and Renderer::assets()).
template<typename HandleT>
class NamedAssetTable {
public:
    struct Entry {
        std::string name;
        HandleT handle;
    };

    // Rejects (returns false, logs a warning) a name already in use --
    // callers own recovering from that (e.g. falling back to a fuller path
    // string); the table never silently overwrites one asset's name with
    // another's.
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

    // Empty string_view if `handle` isn't registered under any name.
    [[nodiscard]]
    auto name_of(HandleT handle) const noexcept -> std::string_view {
        auto const it = std::ranges::find(entries_, handle, &Entry::handle);
        return it != entries_.end() ? std::string_view{it->name} : std::string_view{};
    }

    // Always sorted by name (register_asset inserts in place) -- ready for
    // UI iteration with no separate sort step.
    [[nodiscard]]
    auto entries() const noexcept -> std::span<Entry const> {
        return entries_;
    }

private:
    std::vector<Entry> entries_;
};

// Aggregates one NamedAssetTable per asset kind this engine has. Owned by
// Renderer (see Renderer::assets()) -- lookups are a linear scan over a
// handful to a few hundred entries, which is fine for editor-scale usage
// and avoids needing a std::hash<Handle<T,N>> specialization that doesn't
// exist today.
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
