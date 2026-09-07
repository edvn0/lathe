#pragma once

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"
#include "assets/material.hxx"
#include "assets/material_storage.hxx" // MaterialCreateInfo
#include "assets/model.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"
#include "terrain/terrain_mesh.hxx"

struct GrassParams {
    float field_size = 100.0F;
    float spacing = 0.5F;
    float blotch_scale = 20.0F;
    float blotch_threshold = 0.5F;
    float blotch_softness = 0.15F;
    std::uint32_t blotch_seed = 1337U;
};

class BasicGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

    [[nodiscard]] auto terrain_create_info(Renderer &renderer) -> std::optional<TerrainWorldCreateInfo> override;

    auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void override;

private:
    auto shoot_bullet(Scene &scene, std::size_t n = 1) -> void;


    auto rebuild_grass_field(Scene &scene) -> void;

    entt::entity player_entity_{entt::null};
    PlayerController player_controller_;
    PlayerCamera player_camera_;

    ModelHandle cube_model_{};
    glm::vec3 cube_half_extents_{0.5F};

    MaterialHandle grass_material_{};

    MaterialCreateInfo grass_material_info_{};

    entt::entity grass_field_entity_{entt::null};
    GrassParams grass_field_params_{};
    std::uint32_t grass_field_blade_count_ = 0U;

    TerrainParams terrain_params_{};

    MaterialHandle terrain_material_{};
    float terrain_ground_y_ = 0.0F;
};
