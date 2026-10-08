#pragma once

#include <cstdint>
#include <filesystem>

#include <entt/entt.hpp>
#include <glm/vec3.hpp>

#include "app/game.hxx"
#include "core/paths.hxx"
#include "core/transform.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"

class PuntGame final : public IGame {
public:
    auto on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void override;
    auto on_update(Scene &scene, float delta_time) -> void override;

    auto on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void override;
    auto on_key_released(Scene &scene, KeyReleasedEvent const &event) -> void override;
    auto on_mouse_moved(Scene &scene, MouseMovedEvent const &event) -> void override;
    auto on_mouse_button_pressed(Scene &scene, MouseButtonPressedEvent const &event) -> void override;

    auto on_ui(Scene &scene, Renderer &renderer) -> void override;

    [[nodiscard]] auto camera(Scene const &scene, float aspect_ratio) const -> CameraParams override;

    [[nodiscard]] auto benchmark_camera_path() const -> std::vector<CameraKeyframe> override;

private:
    enum class Phase : std::uint8_t {
        briefing,
        playing,
        holed,
        failed,
    };

    auto bind_to(Scene &scene) -> void;
    auto restart(Scene &scene) -> void;

    auto try_punt(Scene &scene) -> void;

    [[nodiscard]] auto round_live() const noexcept -> bool {
        return phase_ == Phase::briefing || phase_ == Phase::playing;
    }

    static constexpr std::uint32_t punt_allowance = 6;
    static constexpr float round_seconds = 90.0F;

    static constexpr float punt_range = 1.9F;
    static constexpr float punt_interval_seconds = 0.4F;

    static constexpr float punt_speed = 11.0F;
    static constexpr float punt_lift = 3.2F;

    static constexpr float ball_resting_speed = 0.35F;
    static constexpr float settle_seconds = 1.0F;

    std::filesystem::path scene_file_{data_path("assets/scenes/punt.lbf").absolute()};

    Scene const *bound_scene_ = nullptr;

    entt::entity player_entity_{entt::null};
    entt::entity ball_entity_{entt::null};

    Components::Transform player_spawn_{};
    Components::Transform ball_spawn_{};

    glm::vec3 hole_centre_{0.0F};
    glm::vec3 hole_half_extents_{0.0F};

    Phase phase_ = Phase::briefing;
    std::uint32_t punts_used_ = 0;
    float time_left_ = round_seconds;
    float round_seconds_taken_ = 0.0F;
    float punt_cooldown_ = 0.0F;

    glm::vec3 previous_ball_position_{0.0F};
    float ball_speed_ = 0.0F;
    float ball_still_for_ = 0.0F;

    bool punt_requested_ = false;
    bool restart_requested_ = false;

    std::uint32_t frames_since_update_ = 0;

    PlayerController player_controller_;
    PlayerCamera player_camera_;
};
