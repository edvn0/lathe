#pragma once

#include <expected>
#include <filesystem>
#include <string>

// data/game.toml of an installed game: its presence is what turns the engine into a player. A flat file of
// `key = "value"` lines (a TOML subset); unknown keys are ignored so newer packagers can add fields.
struct GameManifest {
    std::string name;    // Package and directory name, e.g. "chess".
    std::string title;   // Window title.
    std::string game;    // The engine game to run (--game).
    std::string version;

    [[nodiscard]] static auto load(std::filesystem::path const &path) -> std::expected<GameManifest, std::string>;

    [[nodiscard]] auto to_text() const -> std::string;
};
