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
#include "material_wall_game.hxx"
#include "meshlet_gallery_game.hxx"
#include "shadow_city_game.hxx"
#include "meshlet_showcase_game.hxx"
#include "moving_game.hxx"
#include "net_lua.hxx"
#include "punt_game.hxx"
#include "shadow_caster_game.hxx"

namespace {
    constexpr std::array<std::string_view, 11> names{"basic", "punt", "chess", "lua", "moving", "meshlets", "gallery", "lights", "materials", "shadows", "shadowcity"};
}

auto game_names() -> std::span<std::string_view const> { return names; }

auto create_game(std::string_view name) -> std::unique_ptr<IGame> {
    if (name == "punt") {
        info("Starting the 'punt' game");
        return std::make_unique<PuntGame>();
    }

    if (name == "meshlets") {
        info("Starting the 'meshlets' showcase");
        return std::make_unique<MeshletShowcaseGame>();
    }

    if (name == "lights") {
        info("Starting the 'lights' bloom demo");
        auto game = std::make_unique<LuaGame>();

        game->set_default_entry("assets/scripts/lights/main.lua");

        return game;
    }

    if (name == "materials") {
        info("Starting the 'materials' wall");
        return std::make_unique<MaterialWallGame>();
    }

    if (name == "shadows") {
        info("Starting the 'shadows' caster test scene");
        return std::make_unique<ShadowCasterGame>();
    }

    if (name == "shadowcity") {
        info("Starting the 'shadowcity' stress scene");
        return std::make_unique<ShadowCityGame>();
    }

    if (name == "gallery") {
        info("Starting the 'gallery' showcase");
        return std::make_unique<MeshletGalleryGame>();
    }

    if (name == "moving") {
        info("Starting the 'moving' game");
        return std::make_unique<MovingGame>();
    }

    if (name == "lua") {
        info("Starting a Lua game");
        auto game = std::make_unique<LuaGame>();

        game->add_native_module("native.chess", &luaopen_lathe_chess);
        game->add_native_module("native.net", &luaopen_lathe_net);

        return game;
    }

    if (name == "chess") {
        info("Starting the 'chess' game");
        return std::make_unique<ChessGame>();
    }

    return std::make_unique<BasicGame>();
}
