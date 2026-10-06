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

// "Punt!": walk up to the ball, kick it down the course and drop it into the hole before the punts or the clock run
// out. One round, a win and a loss, and R to go again.
//
// The course itself is not built by this class on every run: on_populate() loads assets/scenes/punt.lbf, and only
// falls back to build_punt_level() and saves the result when the file isn't there yet. Everything the rules need is
// found by entity name afterwards (punt::player_entity_name and friends), so the course stays editable in the editor.
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
        briefing, // the round hasn't started; the clock is still
        playing,
        holed,
        failed,
    };

    // Finds the player, ball and hole by name in `scene` and restarts the round. Called when a scene first shows up
    // in on_update(), which is every time the editor starts play.
    auto bind_to(Scene &scene) -> void;
    auto restart(Scene &scene) -> void;

    // Kicks the ball away from the player, along where the camera is looking. Ignored unless the player is close
    // enough and the round is still live.
    auto try_punt(Scene &scene) -> void;

    [[nodiscard]] auto round_live() const noexcept -> bool {
        return phase_ == Phase::briefing || phase_ == Phase::playing;
    }

    static constexpr std::uint32_t punt_allowance = 6;
    static constexpr float round_seconds = 90.0F;

    // How close the player has to be to the ball's centre to kick it, and how long between kicks.
    static constexpr float punt_range = 1.9F;
    static constexpr float punt_interval_seconds = 0.4F;

    // The kick, as the velocity it gives a ball of punt::ball_mass.
    static constexpr float punt_speed = 11.0F;
    static constexpr float punt_lift = 3.2F;

    // A ball slower than this for settle_seconds has stopped, which is what ends a round with no punts left.
    static constexpr float ball_resting_speed = 0.35F;
    static constexpr float settle_seconds = 1.0F;

    std::filesystem::path scene_file_{data_path("assets/scenes/punt.lbf").absolute()};

    // The scene bind_to() last ran against, so play() is noticed without the engine telling us.
    Scene const *bound_scene_ = nullptr;

    entt::entity player_entity_{entt::null};
    entt::entity ball_entity_{entt::null};

    // Where the course put them, which is where restart() puts them back.
    Components::Transform player_spawn_{};
    Components::Transform ball_spawn_{};

    // The hole's trigger volume, in world space.
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

    // Set by the input callbacks, consumed by the next on_update(), which is where the scene is known.
    bool punt_requested_ = false;
    bool restart_requested_ = false;

    // on_ui() is called whether or not the editor is playing, and isn't told which; the HUD belongs to the round, so
    // it only draws on frames just after an on_update().
    std::uint32_t frames_since_update_ = 0;

    PlayerController player_controller_;
    PlayerCamera player_camera_;
};
