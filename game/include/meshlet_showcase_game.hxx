#pragma once

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <entt/entt.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "game/game.hxx"
#include "assets/material.hxx"
#include "assets/model.hxx"

// A studio scene for explaining meshlet rendering: dense meshes drawn through the task/mesh shader path, shown plain,
// as meshlet colours, and pulled apart ("exploded") by the per-material meshlet extrude/shrink.
// Keys 1-7 pick a shot; screenshots are F12 (see --inject-keys).
class MeshletShowcaseGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto subscribe_input(InputContext const &ctx) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void;

    auto clone_into_runtime(Scene const &editor_scene, Scene &runtime_scene) -> void override {
        clone_editor_into_runtime<Components::DebugMeshletCones>(editor_scene, runtime_scene);
    }

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    enum class Look : std::uint8_t { plain, colours, exploded };

    static constexpr std::size_t model_count = 2;
    static constexpr std::size_t look_count = 3;
    static constexpr std::size_t prop_count = model_count * look_count;

    struct MaterialSet {
        std::vector<MaterialHandle> sources;
        std::array<std::vector<MaterialHandle>, look_count> variants;
    };

    struct Placement {
        bool visible = false;
        glm::vec3 position{0.0F};
        float scale = 1.0F;
    };

    struct Shot {
        std::string_view title;
        std::string_view caption;
        std::array<Placement, prop_count> props;
        glm::vec3 target{0.0F};
        float distance = 5.0F;
        float height = 0.3F;
        float yaw_degrees = 20.0F;
        float yaw_speed = 6.0F;
        float fov_degrees = 38.0F;
        float extrude = 0.0F;
        float shrink = 0.0F;
        bool cones = false;
    };

    static constexpr std::size_t shot_count = 9;

    auto bind_to(Scene &scene) -> void;
    auto apply_shot(Scene &scene) -> void;
    auto set_exploded(float extrude, float shrink) -> void;

    [[nodiscard]] static auto prop_index(std::size_t model, Look look) -> std::size_t {
        return model * look_count + static_cast<std::size_t>(look);
    }

    Renderer *renderer_ = nullptr;
    Scene const *bound_scene_ = nullptr;

    struct Fit {
        float scale = 1.0F;
        glm::vec3 centre{0.0F};
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    };

    std::array<MaterialSet, model_count> materials_;
    std::array<Fit, model_count> fits_;
    std::array<entt::entity, prop_count> props_{};
    std::array<Shot, shot_count> shots_;

    std::size_t shot_index_ = 0;
    float time_ = 0.0F;
    float yaw_degrees_ = 0.0F;

    float extrude_ = 0.0F;
    float shrink_ = 0.0F;
    float applied_extrude_ = -1.0F;
    float applied_shrink_ = -1.0F;

    glm::vec3 eye_{0.0F, 0.5F, 5.0F};
    glm::vec3 look_at_{0.0F};
    float fov_degrees_ = 38.0F;
};
