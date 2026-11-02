#include <array>
#include <memory>
#include <span>
#include <string_view>

#include "game/game.hxx"
#include "luagame/script_system.hxx"
#include "basic_game.hxx"
#include "chess_game.hxx"
#include "empty_game.hxx"
#include "glass_game.hxx"
#include "globe_game.hxx"
#include "core/logger.hxx"
#include "material_wall_game.hxx"
#include "meshlet_gallery_game.hxx"
#include "shadow_city_game.hxx"
#include "meshlet_showcase_game.hxx"
#include "moving_game.hxx"
#include "punt_game.hxx"
#include "water_game.hxx"
#include "shadow_caster_game.hxx"

namespace {
    constexpr std::array<std::string_view, 13> names{"empty", "basic", "punt", "chess", "moving", "meshlets", "gallery", "materials", "shadows", "shadowcity", "glass", "water", "globe"};
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

    if (name == "empty") {
        info("Starting an empty project");
        return std::make_unique<EmptyGame>();
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

    if (name == "glass") {
        info("Starting the 'glass' transparency gallery");
        return std::make_unique<GlassGame>();
    }

    if (name == "globe") {
        info("Starting the 'globe' Earth");
        return std::make_unique<GlobeGame>();
    }

    if (name == "water") {
        info("Starting the 'water' pool and sea");
        return std::make_unique<WaterGame>();
    }

    if (name == "gallery") {
        info("Starting the 'gallery' showcase");
        return std::make_unique<MeshletGalleryGame>();
    }

    if (name == "moving") {
        info("Starting the 'moving' game");
        return std::make_unique<MovingGame>();
    }

    if (name == "chess") {
        info("Starting the 'chess' game");
        return std::make_unique<ChessGame>();
    }

    return std::make_unique<BasicGame>();
}
