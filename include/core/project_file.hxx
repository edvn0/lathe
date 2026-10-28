#pragma once

#include "core/game_manifest.hxx"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// lathe.project, the marker at the root of a project. Its directory is the project's data root. It carries the same
// keys as a packaged game's game.toml (which is its cooked subset), but 'game' may be left out.
struct ProjectFile {
    static constexpr auto file_name = "lathe.project";

    GameManifest manifest;
    std::filesystem::path root;

    // 'path' is a lathe.project or a directory holding one.
    [[nodiscard]] static auto load(std::filesystem::path const &path) -> std::expected<ProjectFile, std::string>;

    // The nearest directory at or above 'start' that holds a lathe.project.
    [[nodiscard]] static auto find_root(std::filesystem::path const &start) -> std::optional<std::filesystem::path>;

    [[nodiscard]] auto save() const -> std::expected<void, std::string>;
};

// The projects opened lately, newest first, kept one path per line in a user-wide file.
class RecentProjects {
public:
    static constexpr std::size_t capacity = 10;

    explicit RecentProjects(std::filesystem::path file) : file_(std::move(file)) {}

    // The default location, in the user's state directory.
    [[nodiscard]] static auto user_default() -> RecentProjects;

    // The remembered project roots that still hold a lathe.project.
    [[nodiscard]] auto list() const -> std::vector<std::filesystem::path>;

    auto remember(std::filesystem::path const &root) const -> void;

private:
    std::filesystem::path file_;
};
