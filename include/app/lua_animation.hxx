#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "animation/batch.hxx"
#include "animation/clip_state_table.hxx"
#include "assets/model.hxx"
#include "assets/model_skin.hxx"

class Scene;
struct Renderer;

// Skinned, animated characters for Lua games. A rig is a skinned model with its locomotion clips found by name (Idle,
// Walk, Run, Jump, Death; the first clip is the idle for models that name none) and one state machine; an actor is one
// character drawn with it. A script places each actor and tells it how it is moving; update() advances every actor's
// state machine and writes the draw instances, in the same way for every rig.
class LuaAnimationSystem {
public:
    struct RigOptions {
        // The height the model is scaled to, in metres, and how it is measured: the model's own height, the y of its
        // feet and the yaw that turns it to face +Z. The defaults fit animated_human.glb.
        float height = 1.75F;
        float model_height = 5.535F;
        float model_feet_y = -0.015F;
        float yaw_offset = 3.14159265F;

        // Speeds the walk and run clips cover the ground at, so feet do not slide.
        float walk_speed = 1.4F;
        float run_speed = 4.5F;
    };

    struct Actor {
        glm::vec3 position{0.0F};
        float yaw = 0.0F;
        Animation::AnimInputs inputs;
        bool active = true;
    };

    LuaAnimationSystem();
    ~LuaAnimationSystem();
    LuaAnimationSystem(LuaAnimationSystem const &) = delete;
    auto operator=(LuaAnimationSystem const &) -> LuaAnimationSystem & = delete;

    // The rig index; one rig per model, so asking again for a model returns its rig.
    auto create_rig(Scene &scene, ModelHandle model, RigOptions const &options) -> std::uint32_t;

    // True once the model has loaded and has a skeleton and an idle clip.
    [[nodiscard]] auto ready(Renderer const &renderer, std::uint32_t rig) -> bool;

    [[nodiscard]] auto rig_count() const noexcept -> std::size_t;

    auto add_actor(std::uint32_t rig) -> std::uint32_t;
    [[nodiscard]] auto actor(std::uint32_t rig, std::uint32_t index) -> Actor *;

    // The state machine's current state for an actor ("idle", "walk", ...), or empty before its first update.
    [[nodiscard]] auto state_name(std::uint32_t rig, std::uint32_t index) const -> std::string_view;

    auto update(Scene &scene, Renderer &renderer, float delta_time) -> void;

private:
    struct Rig;

    auto prepare(Renderer const &renderer, Rig &rig) -> bool;

    std::vector<std::unique_ptr<Rig>> rigs_;
};
