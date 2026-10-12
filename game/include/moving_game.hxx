#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <numbers>
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
#include "net/http.hxx"
#include "physics/character_body.hxx"
#include "physics/fixed_stepper.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"

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
        float heading{0.0F};
        float speed{0.0F};
        float time_left{0.0F};
    };

    auto bind_to(Scene &scene) -> void;
    auto set_crowd_size(std::size_t count) -> void;
    auto update_crowd(float delta_time, glm::vec3 const &camera_position) -> void;
    auto rebuild_batch() -> void;
    auto load_skinned_model(Scene &scene, Renderer &renderer) -> void;
    auto request_brainstem() -> void;
    auto poll_brainstem(Scene &scene, Renderer &renderer) -> void;
    [[nodiscard]] auto active_machine() const -> Animation::AnimStateMachine const &;
    auto compose_skinned(Scene &scene, glm::vec3 const &player_position) -> void;

    Scene const *bound_scene_ = nullptr;
    Renderer *renderer_ = nullptr;

    bool skinned_mode_ = true;
    bool batch_skinned_ = false;
    ModelHandle skinned_model_{};
    bool use_brainstem_ = false;
    float skinned_height_ = 5.535F;
    float skinned_feet_y_ = -0.015F;
    float skinned_yaw_offset_ = std::numbers::pi_v<float>;
    HttpClient http_client_;
    std::future<std::expected<void, HttpError>> brainstem_download_;
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
    std::vector<glm::mat4> part_rest_;
    std::array<entt::entity, Animation::Humanoid::JointCount> part_entities_{};

    std::optional<CharacterBody> body_;
    FixedStepper stepper_;
    PlayerController controller_;
    PlayerCamera camera_;

    float facing_yaw_ = 0.0F;
    float step_alpha_ = 0.0F;
    bool prone_ = false;
    bool sprint_ = false;
    bool jump_held_ = false;

    std::vector<Npc> npcs_;
    std::vector<Animation::AnimInputs> inputs_;
    std::uint32_t crowd_target_ = 200;
    std::uint32_t crowd_size_ = 0;

    float animation_ms_ = 0.0F;
    float compose_ms_ = 0.0F;
    std::array<std::size_t, 3> lod_counts_{};
    std::uint32_t frames_since_update_ = 0;

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
