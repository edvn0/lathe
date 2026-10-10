#pragma once

#include <expected>
#include <filesystem>
#include <string>

struct GameManifest {
    std::string name;
    std::string title;
    std::string game;
    std::string version;
    std::string entry;
    // A cooked scene (.lbf) relative to the data directory, opened before play starts; empty to use the game's own.
    std::string scene;

    [[nodiscard]] static auto load(std::filesystem::path const &path) -> std::expected<GameManifest, std::string>;

    [[nodiscard]] auto to_text() const -> std::string;
};
