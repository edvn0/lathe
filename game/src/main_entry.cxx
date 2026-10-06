#include <array>
#include <memory>
#include <span>
#include <string_view>

#include "app/game.hxx"
#include "app/lua_game.hxx"
#include "basic_game.hxx"
#include "chess_game.hxx"
#include "chess_lua.hxx"
#include "core/logger.hxx"
#include "punt_game.hxx"

namespace {
    constexpr std::array<std::string_view, 4> names{"basic", "punt", "chess", "lua"};
} // namespace

// The first name is the default; main.cxx rejects any other name at parse time.
auto game_names() -> std::span<std::string_view const> { return names; }

// Called once by the engine's main.cxx at startup, with one of game_names().
auto create_game(std::string_view name) -> std::unique_ptr<IGame> {
    if (name == "punt") {
        info("Starting the 'punt' game");
        return std::make_unique<PuntGame>();
    }

    if (name == "lua") {
        info("Starting a Lua game");
        auto game = std::make_unique<LuaGame>();

        game->add_native_module("native.chess", &luaopen_lathe_chess);

        return game;
    }

    if (name == "chess") {
        info("Starting the 'chess' game");
        return std::make_unique<ChessGame>();
    }

    return std::make_unique<BasicGame>();
}
