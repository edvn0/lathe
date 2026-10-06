#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

class Paths;

enum class PathRoot : std::uint8_t {
    data,        // Read-only game/engine data: shaders, models, textures, scenes, fonts.
    cache,       // Regenerable derived data (KTX2, shader binaries).
    state,       // Per-user state: imgui.ini, saves.
    screenshots, // User-visible output.
};

// A path under one of the Paths roots. It has no public constructor from strings, so the only way to obtain one is
// through Paths (or by joining onto one already held), which keeps every file access anchored to a known root.
template<PathRoot Root>
class RootedPath {
public:
    // The absolute location on disk. Use this to open the file.
    [[nodiscard]] auto absolute() const noexcept -> std::filesystem::path const & { return absolute_; }

    // The root-relative, forward-slash form. Stable across installs, so it is what asset IDs are derived from.
    [[nodiscard]] auto logical() const -> std::string { return logical_.generic_string(); }

    // Appends a relative sub-path. Empty if `child` is absolute or escapes the root.
    [[nodiscard]] auto join(std::string_view child) const -> std::optional<RootedPath>;

    auto operator==(RootedPath const &other) const noexcept -> bool { return logical_ == other.logical_; }

private:
    friend class Paths;

    RootedPath(std::filesystem::path root, std::filesystem::path logical)
        : absolute_(root / logical), logical_(std::move(logical)), root_(std::move(root)) {}

    std::filesystem::path absolute_;
    std::filesystem::path logical_;
    std::filesystem::path root_;
};

using DataPath = RootedPath<PathRoot::data>;
using CachePath = RootedPath<PathRoot::cache>;
using StatePath = RootedPath<PathRoot::state>;
using ScreenshotPath = RootedPath<PathRoot::screenshots>;

// A file the engine loads as an asset: either shipped content under the data root, or a file the user picked from
// elsewhere on disk (the editor's file browser, a scene referencing one, a CLI argument). It converts implicitly from
// DataPath only; external files go through the named AssetPath::external, so every escape from the data root is
// explicit and greppable.
class AssetPath {
public:
    AssetPath(DataPath path) : absolute_(path.absolute()), key_(path.logical()), external_(false) {} // NOLINT

    // `path` made absolute. Empty if it is empty.
    [[nodiscard]] static auto external(std::filesystem::path const &path) -> std::optional<AssetPath>;

    // Reads a path stored in a scene file or typed by the user: relative resolves under the data root of
    // Paths::current(), absolute is external. Empty if it is empty or escapes the data root.
    [[nodiscard]] static auto from_serialised(std::string_view text) -> std::optional<AssetPath>;

    // A file the user picked (file browser, drag and drop, CLI). Under the data root it is shipped content, keyed
    // relative to it; anywhere else it is external. A relative `path` is taken relative to the data root.
    [[nodiscard]] static auto from_user(std::filesystem::path const &path) -> std::optional<AssetPath>;

    // Stand-in for a path that could not be resolved. It doesn't exist, so loading it fails like any missing file.
    [[nodiscard]] static auto missing() -> AssetPath;

    // A file next to this one, as a glTF references its textures. Stays under the data root when this does and the
    // result doesn't climb out of it; otherwise it is external. Empty if `relative` is empty.
    [[nodiscard]] auto sibling(std::filesystem::path const &relative) const -> std::optional<AssetPath>;

    [[nodiscard]] auto absolute() const noexcept -> std::filesystem::path const & { return absolute_; }

    // What asset IDs hash: root-relative for data assets, so it is stable across installs; absolute for external ones.
    [[nodiscard]] auto key() const noexcept -> std::string const & { return key_; }
    [[nodiscard]] auto is_external() const noexcept -> bool { return external_; }

    auto operator==(AssetPath const &other) const noexcept -> bool {
        return external_ == other.external_ && key_ == other.key_;
    }

private:
    AssetPath(std::filesystem::path absolute, std::string key) : absolute_(std::move(absolute)), key_(std::move(key)),
                                                                 external_(true) {}

    std::filesystem::path absolute_;
    std::string key_;
    bool external_;
};

struct PathsOptions {
    std::optional<std::filesystem::path> data_dir; // --data-dir
};

class Paths {
public:
    // Data root, in order: options.data_dir; `<exe>/data` or `<exe>/../share/lathe` when it holds a game.toml (an
    // installed game); the working directory (a development checkout). In development the other roots stay under the
    // working directory, as before; when installed they follow the XDG base directories.
    [[nodiscard]] static auto resolve(PathsOptions const &options) -> Paths;

    // The process-wide instance. Set once from main before anything touches the disk; defaults to resolve({}).
    static auto set_current(Paths paths) -> void;
    [[nodiscard]] static auto current() -> Paths const &;

    [[nodiscard]] auto installed() const noexcept -> bool { return installed_; }
    [[nodiscard]] auto data_root() const noexcept -> std::filesystem::path const & { return data_; }

    [[nodiscard]] auto data(std::string_view relative) const -> std::optional<DataPath>;
    [[nodiscard]] auto cache(std::string_view relative) const -> std::optional<CachePath>;
    [[nodiscard]] auto state(std::string_view relative) const -> std::optional<StatePath>;
    [[nodiscard]] auto screenshot(std::string_view relative) const -> std::optional<ScreenshotPath>;

private:
    Paths() = default;

    std::filesystem::path data_;
    std::filesystem::path cache_;
    std::filesystem::path state_;
    std::filesystem::path screenshots_;
    bool installed_ = false;
};

namespace paths_detail {
    // Lexically normalised `relative`, or empty if it is absolute or climbs out of its root.
    [[nodiscard]] auto sanitise(std::string_view relative) -> std::optional<std::filesystem::path>;
} // namespace paths_detail

template<PathRoot Root>
auto RootedPath<Root>::join(std::string_view child) const -> std::optional<RootedPath> {
    auto const combined = paths_detail::sanitise((logical_ / std::filesystem::path{child}).generic_string());

    if (!combined) {
        return std::nullopt;
    }

    return RootedPath{root_, *combined};
}

// For literal, known-good relative paths ("assets/shaders/x.slang"); aborts if `relative` is absolute or escapes.
[[nodiscard]] auto data_path(std::string_view relative) -> DataPath;
[[nodiscard]] auto cache_path(std::string_view relative) -> CachePath;
[[nodiscard]] auto state_path(std::string_view relative) -> StatePath;
[[nodiscard]] auto screenshot_path(std::string_view relative) -> ScreenshotPath;
