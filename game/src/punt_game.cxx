#include "punt_game.hxx"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "physics/physics_world.hxx"
#include "punt_level.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"
#include "serialisation/scene_serialisation.hxx"

namespace {

    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    [[nodiscard]] auto find_entity(entt::registry const &registry, std::string_view name) -> entt::entity {
        for (auto const &&[entity, meta]: registry.view<Components::Meta const>().each()) {
            if (meta.name == name) {
                return entity;
            }
        }

        // A course saved with runtime-generated names comes back this way; see SceneEntityFlags::generated_name.
        for (auto const &&[entity, meta]: registry.view<Components::GeneratedMeta const>().each()) {
            if (meta.name == name) {
                return entity;
            }
        }

        return entt::null;
    }

    [[nodiscard]] auto inside_box(glm::vec3 const &point, glm::vec3 const &centre, glm::vec3 const &half_extents)
            -> bool {
        auto const offset = glm::abs(point - centre);
        return offset.x <= half_extents.x && offset.y <= half_extents.y && offset.z <= half_extents.z;
    }

    [[nodiscard]] auto format_clock(float seconds) -> std::string {
        auto const whole = static_cast<int>(std::ceil(std::max(0.0F, seconds)));
        return std::format("{}:{:02}", whole / 60, whole % 60);
    }

} // namespace

auto PuntGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    // A later run (or a Ctrl+R) picks up whatever the editor last saved to the course file, so the level is data
    // rather than code as soon as it exists.
    if (std::filesystem::exists(scene_file_)) {
        auto const loaded = load_scene(scene, renderer, engine_models, scene_file_);

        if (loaded) {
            info("[PuntGame] Loaded the course from '{}' in {:.2f} s", scene_file_.string(), loaded->seconds);

            for (auto const &warning: loaded->instantiate.warnings) {
                warn("[PuntGame] {}", warning);
            }

            bound_scene_ = nullptr; // rebind on the next on_update()
            return;
        }

        error("[PuntGame] Could not load '{}': {}; rebuilding the course", scene_file_.string(),
              describe(loaded.error()));
    }

    build_punt_level(scene, renderer, engine_models);
    bound_scene_ = nullptr;

    std::error_code directory_error;
    std::filesystem::create_directories(scene_file_.parent_path(), directory_error);

    if (auto const saved = save_scene(scene, renderer, engine_models, scene_file_)) {
        info("[PuntGame] Wrote the course to '{}' ({} bytes); edit it in the editor and it loads from there next time",
             scene_file_.string(), saved->file_size);
    } else {
        error("[PuntGame] Could not write '{}': {}; the course stays code-only this run", scene_file_.string(),
              describe(saved.error()));
    }
}

auto PuntGame::bind_to(Scene &scene) -> void {
    bound_scene_ = &scene;

    auto const &registry = scene.get_registry();

    player_entity_ = find_entity(registry, punt::player_entity_name);
    ball_entity_ = find_entity(registry, punt::ball_entity_name);
    auto const hole_entity = find_entity(registry, punt::hole_entity_name);

    if (player_entity_ == entt::null || ball_entity_ == entt::null) {
        error("[PuntGame] The course has no '{}' or no '{}' entity; there is no round to play",
              punt::player_entity_name, punt::ball_entity_name);
    }

    auto const transform_of = [&registry](entt::entity entity) -> Components::Transform const * {
        return registry.valid(entity) ? registry.try_get<Components::Transform>(entity) : nullptr;
    };

    // Wherever the course puts them now is where a restart puts them back.
    if (auto const *transform = transform_of(player_entity_)) {
        player_spawn_ = *transform;
    }
    if (auto const *transform = transform_of(ball_entity_)) {
        ball_spawn_ = *transform;
    }

    if (auto const *transform = transform_of(hole_entity)) {
        hole_centre_ = transform->position;
        hole_half_extents_ = glm::abs(transform->scale);
    } else {
        warn("[PuntGame] The course has no '{}' marker; falling back to the built-in hole position",
             punt::hole_entity_name);
        hole_centre_ =
                glm::vec3{punt::hole_centre.x, (punt::floor_top_y + punt::pit_floor_y) * 0.5F, punt::hole_centre.y};
        hole_half_extents_ =
                glm::vec3{punt::hole_half_size, (punt::floor_top_y - punt::pit_floor_y) * 0.5F, punt::hole_half_size};
    }

    // Facing down the course from the tee.
    player_controller_ = PlayerController{PlayerControllerCreateInfo{.yaw_degrees = 0.0F}};
    player_camera_ = PlayerCamera{PlayerCameraCreateInfo{}};

    phase_ = Phase::briefing;
    punts_used_ = 0;
    time_left_ = round_seconds;
    round_seconds_taken_ = 0.0F;
    punt_cooldown_ = 0.0F;
    ball_still_for_ = 0.0F;
    ball_speed_ = 0.0F;
    previous_ball_position_ = ball_spawn_.position;
    punt_requested_ = false;
    restart_requested_ = false;
}

auto PuntGame::restart(Scene &scene) -> void {
    auto &registry = scene.get_registry();
    auto &physics = *scene.physics_world;

    // The component is what the renderer draws from; the body is what the next step writes back into it.
    registry.get<Components::Transform>(player_entity_) = player_spawn_;
    registry.get<Components::Transform>(ball_entity_) = ball_spawn_;
    physics.set_transform(registry, player_entity_, player_spawn_);
    physics.set_transform(registry, ball_entity_, ball_spawn_);

    player_controller_ = PlayerController{PlayerControllerCreateInfo{.yaw_degrees = 0.0F}};

    phase_ = Phase::briefing;
    punts_used_ = 0;
    time_left_ = round_seconds;
    round_seconds_taken_ = 0.0F;
    punt_cooldown_ = 0.0F;
    ball_still_for_ = 0.0F;
    ball_speed_ = 0.0F;
    previous_ball_position_ = ball_spawn_.position;
}

auto PuntGame::try_punt(Scene &scene) -> void {
    if (!round_live() || punt_cooldown_ > 0.0F || punts_used_ >= punt_allowance) {
        return;
    }

    auto &registry = scene.get_registry();

    auto const &player_position = registry.get<Components::Transform>(player_entity_).position;
    auto const &ball_position = registry.get<Components::Transform>(ball_entity_).position;

    if (glm::distance(player_position, ball_position) > punt_range) {
        return;
    }

    // Along the camera, flattened: the pitch aims the camera, not the kick, which always has the same arc.
    auto direction = player_camera_.forward();
    direction.y = 0.0F;

    if (glm::dot(direction, direction) < 1e-6F) {
        direction = player_controller_.forward();
    }

    direction = glm::normalize(direction);

    auto const impulse = ((direction * punt_speed) + (world_up * punt_lift)) * punt::ball_mass;
    scene.physics_world->apply_impulse(registry, ball_entity_, impulse);

    ++punts_used_;
    punt_cooldown_ = punt_interval_seconds;
    ball_still_for_ = 0.0F;

    if (phase_ == Phase::briefing) {
        phase_ = Phase::playing;
    }
}

auto PuntGame::on_update(Scene &scene, float delta_time) -> void {
    frames_since_update_ = 0;

    if (bound_scene_ != &scene) {
        bind_to(scene);
    }

    auto &registry = scene.get_registry();

    if (scene.physics_world == nullptr || !registry.valid(player_entity_) || !registry.valid(ball_entity_) ||
        !registry.all_of<Components::Transform, Components::RigidBody>(player_entity_) ||
        !registry.all_of<Components::Transform>(ball_entity_)) {
        return;
    }

    auto &physics = *scene.physics_world;

    if (std::exchange(restart_requested_, false)) {
        restart(scene);
    }

    // ---- The player.

    auto const &player_transform = registry.get<Components::Transform>(player_entity_);
    auto const &player_body = registry.get<Components::RigidBody>(player_entity_);

    auto const capsule_half_height = player_body.capsule_height * 0.5F;

    if (player_controller_.consumes_jump() &&
        physics.is_grounded(registry, player_entity_, capsule_half_height, player_body.capsule_radius)) {
        constexpr float jump_velocity = 5.5F;
        physics.jump(registry, player_entity_, jump_velocity);
    }

    auto const desired_velocity = player_controller_.desired_horizontal_velocity();
    physics.set_velocity(registry, player_entity_, desired_velocity);

    // Walked into the shaft, or off the course somehow: back to the tee, at no cost but the time.
    if (player_transform.position.y < punt::floor_top_y - 1.0F) {
        registry.get<Components::Transform>(player_entity_) = player_spawn_;
        physics.set_transform(registry, player_entity_, player_spawn_);
    }

    // ---- The ball.

    auto const &ball_transform = registry.get<Components::Transform>(ball_entity_);

    ball_speed_ =
            delta_time > 0.0F ? glm::distance(ball_transform.position, previous_ball_position_) / delta_time : 0.0F;
    previous_ball_position_ = ball_transform.position;

    if (ball_transform.position.y < punt::pit_floor_y - 20.0F) {
        warn("[PuntGame] The ball left the course; returning it to the tee");
        registry.get<Components::Transform>(ball_entity_) = ball_spawn_;
        physics.set_transform(registry, ball_entity_, ball_spawn_);
    }

    // ---- The round.

    punt_cooldown_ = std::max(0.0F, punt_cooldown_ - delta_time);

    if (std::exchange(punt_requested_, false)) {
        try_punt(scene);
    }

    if (phase_ == Phase::playing) {
        time_left_ = std::max(0.0F, time_left_ - delta_time);
        round_seconds_taken_ += delta_time;
    }

    if (round_live()) {
        if (inside_box(ball_transform.position, hole_centre_, hole_half_extents_)) {
            phase_ = Phase::holed;
            info("[PuntGame] Holed in {} punt(s), {:.1f} s", punts_used_, round_seconds_taken_);
        } else if (phase_ == Phase::playing && time_left_ <= 0.0F) {
            phase_ = Phase::failed;
        } else if (punts_used_ >= punt_allowance) {
            // Out of punts isn't a loss until the ball has actually stopped; the last one may still go in.
            ball_still_for_ = ball_speed_ < ball_resting_speed ? ball_still_for_ + delta_time : 0.0F;

            if (ball_still_for_ >= settle_seconds) {
                phase_ = Phase::failed;
            }
        }
    }

    // ---- The camera.

    auto const speed_factor = player_controller_.move_speed() > 0.0F
                                      ? glm::length(desired_velocity) / player_controller_.move_speed()
                                      : 0.0F;

    auto const occlusion_query = [&physics, player = player_entity_](glm::vec3 const &origin,
                                                                     glm::vec3 const &direction,
                                                                     float max_distance) -> std::optional<float> {
        auto const hit = physics.raycast(origin, direction, max_distance);

        if (hit && hit->entity != player) {
            return hit->distance;
        }

        return std::nullopt;
    };

    player_camera_.update(player_transform.position, player_controller_.yaw_degrees(),
                          player_controller_.pitch_degrees(), speed_factor, delta_time, occlusion_query);
}

auto PuntGame::on_key_pressed(Scene & /*scene*/, KeyPressedEvent const &event) -> void {
    player_controller_.on_key_pressed(event.key);

    if (event.key == GLFW_KEY_E) {
        punt_requested_ = true;
    }

    // Plain R only: Ctrl+R is the editor's repopulate.
    if (event.key == GLFW_KEY_R && event.modifiers == 0) {
        restart_requested_ = true;
    }

    if (event.key == GLFW_KEY_LEFT_SHIFT) {
        player_controller_.set_sprinting(true);
    }
}

auto PuntGame::on_key_released(Scene & /*scene*/, KeyReleasedEvent const &event) -> void {
    player_controller_.on_key_released(event.key);

    if (event.key == GLFW_KEY_LEFT_SHIFT) {
        player_controller_.set_sprinting(false);
    }
}

auto PuntGame::on_mouse_moved(Scene & /*scene*/, MouseMovedEvent const &event) -> void {
    player_controller_.on_mouse_moved(static_cast<float>(event.delta_x), static_cast<float>(event.delta_y),
                                      /*look_enabled=*/true);
}

auto PuntGame::on_mouse_button_pressed(Scene & /*scene*/, MouseButtonPressedEvent const &event) -> void {
    if (event.button == GLFW_MOUSE_BUTTON_LEFT) {
        punt_requested_ = true;
    }
}

auto PuntGame::on_ui(Scene &scene, Renderer & /*renderer*/) -> void {
    // See frames_since_update_: on_ui() runs in the editor too, where there is no round.
    if (frames_since_update_ > 2) {
        return;
    }

    ++frames_since_update_;

    auto const &registry = scene.get_registry();
    auto const playable = registry.valid(player_entity_) && registry.valid(ball_entity_);

    gui::widget("Punt!", [&] {
        if (!playable) {
            ImGui::TextColored(ImVec4{1.0F, 0.4F, 0.4F, 1.0F}, "The course has no player or ball.");
            return;
        }

        ImGui::Text("Punts  %u / %u", punts_used_, punt_allowance);
        ImGui::SameLine();
        ImGui::Text("   Time  %s", format_clock(time_left_).c_str());

        ImGui::Separator();

        switch (phase_) {
            case Phase::briefing:
                ImGui::TextWrapped("Get the ball down the course and into the hole by the flag.");
                ImGui::TextWrapped("WASD to walk, mouse to look, Space to jump. Stand next to the ball and press E "
                                   "(or click) to punt it. The clock starts on your first punt.");
                break;

            case Phase::playing: {
                auto const in_range =
                        glm::distance(registry.get<Components::Transform>(player_entity_).position,
                                      registry.get<Components::Transform>(ball_entity_).position) <= punt_range;

                if (punts_used_ >= punt_allowance) {
                    ImGui::TextColored(ImVec4{1.0F, 0.75F, 0.3F, 1.0F}, "Last punt is away.");
                } else if (in_range) {
                    ImGui::TextColored(ImVec4{0.5F, 1.0F, 0.5F, 1.0F}, "In range -- press E to punt.");
                } else {
                    ImGui::TextDisabled("Walk up to the ball to punt it.");
                }
                break;
            }

            case Phase::holed:
                ImGui::TextColored(ImVec4{0.4F, 1.0F, 0.5F, 1.0F}, "HOLED IT");
                ImGui::Text("%u punt(s) in %.1f s.", punts_used_, round_seconds_taken_);
                break;

            case Phase::failed:
                ImGui::TextColored(ImVec4{1.0F, 0.45F, 0.4F, 1.0F}, "ROUND OVER");
                ImGui::TextUnformatted(time_left_ <= 0.0F ? "The clock ran out." : "Out of punts.");
                break;
        }

        if (!round_live()) {
            ImGui::Separator();
            ImGui::TextUnformatted("Press R to play again.");

            if (ImGui::Button("Play again")) {
                restart_requested_ = true;
            }
        }
    });
}

auto PuntGame::camera(Scene const & /*scene*/, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = player_camera_.view(),
            .projection = player_camera_.projection(aspect_ratio),
            .near_clip = player_camera_.near_clip(),
            .far_clip = player_camera_.far_clip(),
            .vertical_fov_radians = glm::radians(player_camera_.field_of_view_degrees()),
    };
}

auto PuntGame::benchmark_camera_path() const -> std::vector<CameraKeyframe> {
    // Tee to hole and back, low enough to see the obstacles against the walls.
    return {
            // The tee, from the side, with the player and the ball in frame.
            {.position = {-14.0F, 2.0F, 5.5F}, .target = {-13.0F, 0.4F, 0.0F}},
            {.position = {-8.0F, 2.0F, 6.0F}, .target = {0.0F, 0.5F, 0.0F}},
            {.position = {0.0F, 2.0F, -8.0F}, .target = {8.0F, 0.5F, 0.0F}},
            {.position = {9.0F, 2.5F, 5.0F}, .target = {14.0F, 0.0F, 0.0F}},
            {.position = {14.0F, 4.0F, 6.0F}, .target = {14.0F, -2.0F, 0.0F}},
            {.position = {18.0F, 8.0F, -10.0F}, .target = {0.0F, 0.0F, 0.0F}},
            {.position = {0.0F, 18.0F, 16.0F}, .target = {0.0F, 0.0F, 0.0F}},
    };
}
