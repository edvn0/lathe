#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

#include "animation/batch.hxx"
#include "animation/clip_state_table.hxx"
#include "animation/humanoid.hxx"
#include "app/game.hxx"
#include "assets/model_skin.hxx"
#include "physics/character_body.hxx"
#include "physics/fixed_stepper.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"

// A character-movement and crowd-animation test bed: a kinematic player (WASD, Shift to run, Space to jump, C to lie
// down) and a slider-controlled crowd of wandering NPCs, all posed by the CPU animation batch.
//
// Two ways to draw the characters, switched in the panel (or started with LATHE_MOVING_SKINNED=1):
//  - Rigid rig: each body part of the procedural humanoid is a primitive mesh.
//  - Skinned model: assets/models/animated_human.glb (Quaternius, CC0), GPU-skinned. Its Idle/Walk/Run/Jump/Death
//    clips feed the same locomotion state machine; the palettes of the characters near the camera go to the
//    renderer every frame.
//
// Rigid rig: each body part of the humanoid rig is one InstancedModel entity holding one transform per character, so the whole
// crowd costs 14 draws however many characters there are. Character 0 is the player, 1.. are NPCs.
class MovingGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

private:
    struct Npc {
        glm::vec2 position{0.0F};
        float heading{0.0F}; // radians about +Y; 0 faces +Z
        float speed{0.0F};
        float time_left{0.0F}; // until the next heading/gait change
    };

    // Finds the part entities by name and (re)builds the player and animation state for a newly seen scene.
    auto bind_to(Scene &scene) -> void;
    auto set_crowd_size(std::size_t count) -> void;
    auto update_crowd(float delta_time, glm::vec3 const &camera_position) -> void;
    // (Re)creates the animation batch for the active mode and the current crowd size.
    auto rebuild_batch() -> void;
    auto load_skinned_model(Scene &scene, Renderer &renderer) -> void;
    [[nodiscard]] auto active_machine() const -> Animation::AnimStateMachine const &;
    // Fills the skinned entity with the characters that are close to and in front of the camera.
    auto compose_skinned(Scene &scene, glm::vec3 const &player_position) -> void;

    Scene const *bound_scene_ = nullptr;
    Renderer *renderer_ = nullptr;

    // Skinned model mode. The clips/table/machine reference each other and `skin_data_`, so they live as long as it.
    bool skinned_mode_ = false;
    bool batch_skinned_ = false; // which mode batch_ was built for
    ModelHandle skinned_model_{};
    std::shared_ptr<ModelAnimationData const> skin_data_;
    std::vector<std::unique_ptr<Animation::KeyframeClip>> skin_clips_;
    std::unique_ptr<Animation::LocomotionStateTable> skin_table_;
    std::unique_ptr<Animation::AnimStateMachine> skin_machine_;
    entt::entity skinned_entity_{entt::null};
    std::int32_t max_skinned_ = 256;
    float skin_distance_ = 100.0F;
    std::size_t skinned_drawn_ = 0;
    std::vector<std::pair<float, std::uint32_t>> skin_candidates_;
    std::vector<glm::mat4> skin_transforms_;
    std::vector<std::uint32_t> skin_offsets_;

    std::unique_ptr<Animation::Humanoid::Rig> rig_;
    std::unique_ptr<Animation::AnimStateMachine> machine_;
    std::unique_ptr<Animation::AnimationBatch> batch_;
    // Per part: the bind-space joint matrix times the part's local offset/scale, so a part's world transform is
    // root * palette[joint] * rest.
    std::vector<glm::mat4> part_rest_;
    std::array<entt::entity, Animation::Humanoid::JointCount> part_entities_{};

    std::optional<CharacterBody> body_;
    FixedStepper stepper_;
    PlayerController controller_;
    PlayerCamera camera_;

    float facing_yaw_ = 0.0F; // radians; the player's body, which turns towards the movement direction
    float step_alpha_ = 0.0F;
    bool prone_ = false;
    bool sprint_ = false;
    bool jump_held_ = false;

    std::vector<Npc> npcs_;
    std::vector<Animation::AnimInputs> inputs_; // index 0 is the player
    std::uint32_t crowd_target_ = 200;
    std::uint32_t crowd_size_ = 0;

    float animation_ms_ = 0.0F;
    float compose_ms_ = 0.0F;
    std::array<std::size_t, 3> lod_counts_{};
    std::uint32_t frames_since_update_ = 0;

    // Optional CPU timing report (LATHE_MOVING_BENCH=<frames>): averages after a 30-frame warm-up, printed once.
    struct Bench {
        std::uint32_t frames = 0;
        std::uint32_t target = 0;
        double animation_ms = 0.0;
        double compose_ms = 0.0;
        double skin_jobs = 0.0;
        double skin_vertices = 0.0;
        double scratch_bytes = 0.0;
        double fallbacks = 0.0;
        double drawn = 0.0;
        std::uint32_t seen = 0;
    } bench_;
    auto record_bench() -> void;
};
