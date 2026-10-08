#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class Paths;

namespace paths_detail {
    extern std::atomic<bool> access_recording;

    auto note_data_access(std::string const &logical) -> void;
}

enum class PathRoot : std::uint8_t {
    data,
    cache,
    state,
    screenshots,
};

template<PathRoot Root>
class RootedPath {
public:
    [[nodiscard]] auto absolute() const -> std::filesystem::path const & {
        if constexpr (Root == PathRoot::data) {
            if (paths_detail::access_recording.load(std::memory_order_relaxed)) {
                paths_detail::note_data_access(logical_.generic_string());
            }
        }

        return absolute_;
    }

    [[nodiscard]] auto logical() const -> std::string { return logical_.generic_string(); }

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

class AssetPath {
public:
    AssetPath(DataPath path) : absolute_(path.absolute()), key_(path.logical()), external_(false) {} // NOLINT

    [[nodiscard]] static auto external(std::filesystem::path const &path) -> std::optional<AssetPath>;

    [[nodiscard]] static auto from_serialised(std::string_view text) -> std::optional<AssetPath>;

    [[nodiscard]] static auto from_user(std::filesystem::path const &path) -> std::optional<AssetPath>;

    [[nodiscard]] static auto missing() -> AssetPath;

    [[nodiscard]] auto sibling(std::filesystem::path const &relative) const -> std::optional<AssetPath>;

    [[nodiscard]] auto absolute() const -> std::filesystem::path const & {
        if (!external_ && paths_detail::access_recording.load(std::memory_order_relaxed)) {
            paths_detail::note_data_access(key_);
        }

        return absolute_;
    }

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
    std::optional<std::filesystem::path> data_dir;
};

class Paths {
public:
    [[nodiscard]] static auto resolve(PathsOptions const &options) -> Paths;

    static auto set_current(Paths paths) -> void;
    [[nodiscard]] static auto current() -> Paths const &;

    static auto start_access_recording() -> void;
    [[nodiscard]] static auto finish_access_recording() -> std::vector<std::string>;

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
    [[nodiscard]] auto sanitise(std::string_view relative) -> std::optional<std::filesystem::path>;
}

template<PathRoot Root>
auto RootedPath<Root>::join(std::string_view child) const -> std::optional<RootedPath> {
    auto const combined = paths_detail::sanitise((logical_ / std::filesystem::path{child}).generic_string());

    if (!combined) {
        return std::nullopt;
    }

    return RootedPath{root_, *combined};
}

[[nodiscard]] auto data_path(std::string_view relative) -> DataPath;
[[nodiscard]] auto cache_path(std::string_view relative) -> CachePath;
[[nodiscard]] auto state_path(std::string_view relative) -> StatePath;
[[nodiscard]] auto screenshot_path(std::string_view relative) -> ScreenshotPath;
