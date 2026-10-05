#include <array>
#include <memory>
#include <span>
#include <string_view>

#include "app/game.hxx"
#include "basic_game.hxx"
#include "core/logger.hxx"
#include "punt_game.hxx"

namespace {
    constexpr std::array<std::string_view, 2> names{"basic", "punt"};
} // namespace

// The first name is the default; main.cxx rejects any other name at parse time.
auto game_names() -> std::span<std::string_view const> { return names; }

// Called once by the engine's main.cxx at startup, with one of game_names().
auto create_game(std::string_view name) -> std::unique_ptr<IGame> {
    if (name == "punt") {
        info("Starting the 'punt' game");
        return std::make_unique<PuntGame>();
    }

    return std::make_unique<BasicGame>();
}
