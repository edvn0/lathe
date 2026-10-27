#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include <entt/entt.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "game/game.hxx"
#include "assets/material.hxx"
#include "assets/model.hxx"
#include "net/http.hxx"

// Twenty different meshes in one grid, shown as meshlets and pulled apart together. Seventeen are downloaded
// concurrently on start-up (Khronos glTF sample assets, pinned by commit and sha256); three ship with the engine.
// Keys: 1 plain, 2 meshlet colours, 3 exploded, 4 exploded with normal cones, 5 shattered, 6 the finale: everything
// standing on a debug grid under an HDR sky, with point and spot lights.
class MeshletGalleryGame final : public IGame {
public:
    MeshletGalleryGame();

    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto subscribe_input(InputContext const &ctx) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void;

    auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void override {
        clone_editor_into_runtime<Components::DebugMeshletCones>(editor_scene, runtime_scene);
    }

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Slot {
        std::string name;
        std::filesystem::path path;
        glm::vec3 position{0.0F};
        float pitch_degrees = 0.0F;
        float yaw_degrees = 0.0F;

        std::future<std::expected<void, HttpError>> download;
        bool loaded = false;
        bool failed = false;

        entt::entity entity{entt::null};
        std::vector<MaterialHandle> variants;

        // Set when the model is placed, so the layout can move it between the wall and the floor.
        float scale = 1.0F;
        glm::vec3 centre{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        float below_centre = 0.0F;
    };

    struct Light {
        entt::entity entity{entt::null};
        float intensity = 0.0F;
        bool finale = false;
        float orbit_radius = 0.0F;
        float orbit_height = 0.0F;
        float orbit_phase = 0.0F;
        float orbit_speed = 0.0F;
    };

    enum class Look : std::uint8_t { plain, colours, exploded, cones, shattered };

    auto place(Scene &scene, Slot &slot) -> void;
    auto set_look(Look look) -> void;
    auto set_finale(Scene &scene, bool finale) -> void;
    auto layout(Scene &scene) -> void;
    [[nodiscard]] auto floor_position(std::size_t index) const -> glm::vec3;
    auto apply_materials(bool colours, float extrude, float shrink) -> void;

    HttpClient http_;
    Renderer *renderer_ = nullptr;
    EngineModels engine_models_{};

    std::vector<Slot> slots_;
    std::vector<Light> lights_;
    Scene const *bound_scene_ = nullptr;
    entt::entity floor_{entt::null};
    bool finale_ = false;
    float finale_time_ = 0.0F;
    std::size_t downloads_total_ = 0;
    std::size_t downloads_done_ = 0;

    Look look_ = Look::plain;
    float extrude_ = 0.0F;
    float shrink_ = 0.0F;
    float target_extrude_ = 0.0F;
    float target_shrink_ = 0.0F;
    bool colours_ = false;
    bool cones_ = false;
    float applied_extrude_ = -1.0F;
    float applied_shrink_ = -1.0F;
    bool applied_colours_ = false;
};
