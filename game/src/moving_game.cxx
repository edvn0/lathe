#include "moving_game.hxx"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numbers>
#include <optional>
#include <random>
#include <string>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <imgui.h>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "core/random.hxx"
#include "core/thread_pool.hxx"
#include "physics/physics_world.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    using Animation::Humanoid::Joint;

    constexpr float player_radius = 0.3F;
    constexpr float player_height = 1.8F;
    constexpr glm::vec3 player_spawn{0.0F, 0.5F, 0.0F};

    constexpr std::uint32_t max_crowd = 5000;
    constexpr float crowd_half_area = 60.0F;
    constexpr float full_lod_distance = 25.0F;
    constexpr float reduced_lod_distance = 70.0F;

    constexpr float cube_half_extent = 0.5F;

    constexpr glm::vec3 world_up{0.0F, 1.0F, 0.0F};

    constexpr std::string_view skinned_model_path = "assets/models/animated_human.glb";
    constexpr float skinned_feet_y = -0.015F;
    constexpr float skinned_height = 5.535F;
    constexpr float skinned_yaw_offset = std::numbers::pi_v<float>;
    constexpr float skinned_walk_speed = 1.4F;
    constexpr float skinned_run_speed = 4.5F;

    enum class Shape : std::uint8_t { cube, capsule, sphere };

    struct PartSpec {
        Joint joint;
        Shape shape;
        glm::vec3 centre;
        glm::vec3 size;
    };

    constexpr std::array<PartSpec, Joint::JointCount> parts{{
            {Joint::Pelvis, Shape::cube, {0.0F, 0.0F, 0.0F}, {0.34F, 0.16F, 0.2F}},
            {Joint::Spine, Shape::cube, {0.0F, 0.1F, 0.0F}, {0.3F, 0.22F, 0.18F}},
            {Joint::Chest, Shape::cube, {0.0F, 0.15F, 0.0F}, {0.38F, 0.32F, 0.22F}},
            {Joint::Head, Shape::sphere, {0.0F, 0.1F, 0.0F}, {0.22F, 0.24F, 0.24F}},
            {Joint::UpperArmL, Shape::capsule, {0.0F, -0.14F, 0.0F}, {0.09F, 0.26F, 0.09F}},
            {Joint::ForearmL, Shape::capsule, {0.0F, -0.14F, 0.0F}, {0.08F, 0.26F, 0.08F}},
            {Joint::UpperArmR, Shape::capsule, {0.0F, -0.14F, 0.0F}, {0.09F, 0.26F, 0.09F}},
            {Joint::ForearmR, Shape::capsule, {0.0F, -0.14F, 0.0F}, {0.08F, 0.26F, 0.08F}},
            {Joint::ThighL, Shape::capsule, {0.0F, -0.225F, 0.0F}, {0.13F, 0.42F, 0.13F}},
            {Joint::ShinL, Shape::capsule, {0.0F, -0.225F, 0.0F}, {0.11F, 0.42F, 0.11F}},
            {Joint::FootL, Shape::cube, {0.0F, -0.02F, 0.07F}, {0.1F, 0.07F, 0.26F}},
            {Joint::ThighR, Shape::capsule, {0.0F, -0.225F, 0.0F}, {0.13F, 0.42F, 0.13F}},
            {Joint::ShinR, Shape::capsule, {0.0F, -0.225F, 0.0F}, {0.11F, 0.42F, 0.11F}},
            {Joint::FootR, Shape::cube, {0.0F, -0.02F, 0.07F}, {0.1F, 0.07F, 0.26F}},
    }};

    [[nodiscard]] auto part_scale(PartSpec const &part) -> glm::vec3 {
        switch (part.shape) {
            case Shape::capsule:
                return {part.size.x, (part.size.y + part.size.x * 0.5F) * 0.5F, part.size.z};
            case Shape::sphere:
                return part.size;
            case Shape::cube:
                break;
        }
        return part.size;
    }

    [[nodiscard]] auto part_name(std::size_t index) -> std::string { return "moving_part_" + std::to_string(index); }

    [[nodiscard]] auto wrap_angle(float angle) -> float {
        constexpr float two_pi = 2.0F * std::numbers::pi_v<float>;
        angle = std::fmod(angle + std::numbers::pi_v<float>, two_pi);
        if (angle < 0.0F) {
            angle += two_pi;
        }
        return angle - std::numbers::pi_v<float>;
    }

    [[nodiscard]] auto root_matrix(glm::vec3 const &position, float yaw) -> glm::mat4 {
        return glm::rotate(glm::translate(glm::mat4{1.0F}, position), yaw, world_up);
    }

    [[nodiscard]] auto lod_for(float distance) -> Animation::Lod {
        if (distance < full_lod_distance) {
            return Animation::Lod::Full;
        }
        return distance < reduced_lod_distance ? Animation::Lod::Reduced : Animation::Lod::Hold;
    }

    using Clock = std::chrono::steady_clock;

    [[nodiscard]] auto milliseconds_since(Clock::time_point start) -> float {
        return std::chrono::duration<float, std::milli>(Clock::now() - start).count();
    }

}

auto MovingGame::active_machine() const -> Animation::AnimStateMachine const & {
    return batch_skinned_ ? *skin_machine_ : *machine_;
}

auto MovingGame::load_skinned_model(Scene &scene, Renderer &renderer) -> void {
    skinned_entity_ = entt::null;

    if (!skinned_model_.valid()) {
        auto loaded = renderer.load_model(data_path(skinned_model_path));
        if (!loaded) {
            warn("[MovingGame] Could not load '{}': {}; skinned mode is unavailable", skinned_model_path,
                 describe(loaded.error()));
            return;
        }
        skinned_model_ = *loaded;
    }

    if (!skin_data_) {
        skin_data_ = renderer.model_animation(skinned_model_);
        if (!skin_data_) {
            warn("[MovingGame] '{}' has no skin; skinned mode is unavailable", skinned_model_path);
            return;
        }

        auto const make = [&](std::string_view name) -> Animation::Clip const * {
            auto const *imported = skin_data_->find_clip(name);
            if (imported == nullptr) {
                warn("[MovingGame] '{}' has no '{}' clip", skinned_model_path, name);
                return nullptr;
            }
            skin_clips_.push_back(imported->make_clip());
            return skin_clips_.back().get();
        };

        Animation::LocomotionClipSet set;
        set.idle = make("Idle");
        set.walk = make("Walk");
        set.run = make("Run");
        set.jump = make("Jump");
        set.lie_down = make("Death");
        if (set.idle == nullptr) {
            skin_data_.reset();
            skin_clips_.clear();
            return;
        }
        set.walk_stride = set.walk != nullptr ? skinned_walk_speed * set.walk->duration() : 0.0F;
        set.run_stride = set.run != nullptr ? skinned_run_speed * set.run->duration() : 0.0F;

        skin_table_ = std::make_unique<Animation::LocomotionStateTable>(set);
        skin_machine_ = std::make_unique<Animation::AnimStateMachine>(skin_table_->table());
    }

    auto const entity = GeneratedEntity{&scene, "{}", "moving_skinned"};
    entity.emplace<Components::InstancedModel>(Components::InstancedModel{.model = skinned_model_});
    skinned_entity_ = scene.find_entity("moving_skinned");
}

auto MovingGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;
    skinned_mode_ = true;

    if (auto could_wait = renderer.wait_idle(); !could_wait.has_value()) {
        return;
    }

    scene.get_registry().clear();
    scene.physics_settings = PhysicsWorldSettings{};
    scene.environment = new_scene_environment();
    scene.environment.sun.elevation_degrees = 50.0F;
    scene.environment.sun.azimuth_degrees = 150.0F;

    auto const add_static_box = [&](std::string const &name, glm::vec3 const &centre, glm::vec3 const &half_extents,
                                    glm::quat const &rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F}) {
        auto const entity = GeneratedEntity{&scene, "{}", name};
        entity.emplace<Components::Transform>(Components::Transform{
                .position = centre,
                .rotation = rotation,
                .scale = half_extents / cube_half_extent,
        });
        entity.emplace<Components::Model>(Components::Model{.model = engine_models.cube});
        entity.emplace<Components::RigidBody>(Components::RigidBody{.half_extents = half_extents, .is_static = true});
    };

    add_static_box("floor", {0.0F, -0.5F, 0.0F}, {80.0F, 0.5F, 80.0F});

    for (int i = 0; i < 4; ++i) {
        auto const height = 0.25F * static_cast<float>(i + 1);
        add_static_box("step_" + std::to_string(i), {6.0F + 1.5F * static_cast<float>(i), height * 0.5F, 0.0F},
                       {0.75F, height * 0.5F, 3.0F});
    }

    add_static_box("block_low", {-6.0F, 0.5F, 4.0F}, {1.0F, 0.5F, 1.0F});
    add_static_box("block_high", {-6.0F, 0.55F, -4.0F}, {1.5F, 0.55F, 1.5F});

    add_static_box("ramp_walkable", {0.0F, 1.2F, -10.0F}, {4.0F, 0.2F, 3.0F},
                   glm::angleAxis(glm::radians(20.0F), glm::vec3{0.0F, 0.0F, 1.0F}));
    add_static_box("ramp_steep", {-12.0F, 2.4F, -10.0F}, {3.0F, 0.2F, 3.0F},
                   glm::angleAxis(glm::radians(65.0F), glm::vec3{0.0F, 0.0F, 1.0F}));

    load_skinned_model(scene, renderer);

    for (std::size_t i = 0; i < parts.size(); ++i) {
        auto const entity = GeneratedEntity{&scene, "{}", part_name(i)};
        entity.emplace<Components::InstancedModel>(Components::InstancedModel{
                .model = parts[i].shape == Shape::capsule  ? engine_models.capsule
                         : parts[i].shape == Shape::sphere ? engine_models.sphere
                                                           : engine_models.cube,
                .transforms = {glm::translate(glm::mat4{1.0F}, player_spawn) *
                               glm::scale(glm::mat4{1.0F}, part_scale(parts[i]))},
        });
    }
}

auto MovingGame::bind_to(Scene &scene) -> void {
    bound_scene_ = &scene;

    for (std::size_t i = 0; i < parts.size(); ++i) {
        part_entities_[i] = scene.find_entity(part_name(i));
    }

    if (!rig_) {
        rig_ = Animation::Humanoid::make_rig();
        machine_ = std::make_unique<Animation::AnimStateMachine>(Animation::Humanoid::make_state_table(*rig_));

        auto bind_models = std::vector<glm::mat4>(rig_->skeleton.joint_count());
        Animation::compute_model_matrices(rig_->skeleton, rig_->skeleton.bind_pose().view(), bind_models);

        part_rest_.clear();
        for (auto const &part: parts) {
            part_rest_.push_back(bind_models[part.joint] * glm::translate(glm::mat4{1.0F}, part.centre) *
                                 glm::scale(glm::mat4{1.0F}, part_scale(part)));
        }
    }

    body_.emplace(player_spawn, player_radius, player_height);
    stepper_ = FixedStepper{};
    controller_ = PlayerController{PlayerControllerCreateInfo{.yaw_degrees = 90.0F}};
    camera_ = PlayerCamera{PlayerCameraCreateInfo{}};
    facing_yaw_ = 0.0F;
    prone_ = false;
    sprint_ = false;
    jump_held_ = false;

    skinned_entity_ = scene.find_entity("moving_skinned");
    batch_.reset();
    crowd_size_ = crowd_target_;
    set_crowd_size(crowd_size_);
}

auto MovingGame::set_crowd_size(std::size_t count) -> void {
    auto random = make_random_engine(0x6d6f76U);
    std::uniform_real_distribution<float> unit{0.0F, 1.0F};

    npcs_.resize(count);
    for (auto &npc: npcs_) {
        npc.position = {(unit(random) * 2.0F - 1.0F) * crowd_half_area, (unit(random) * 2.0F - 1.0F) * crowd_half_area};
        npc.heading = unit(random) * 2.0F * std::numbers::pi_v<float>;
        npc.time_left = unit(random) * 4.0F;
    }

    inputs_.assign(count + 1, Animation::AnimInputs{});

    rebuild_batch();
}

auto MovingGame::rebuild_batch() -> void {
    batch_skinned_ = skinned_mode_ && skin_machine_ != nullptr && skin_data_ != nullptr;
    auto const &skeleton = batch_skinned_ ? skin_data_->skeleton : rig_->skeleton;
    batch_ = std::make_unique<Animation::AnimationBatch>(skeleton, active_machine(), inputs_.size(), &thread_pool());
}

auto MovingGame::update_crowd(float delta_time, glm::vec3 const &camera_position) -> void {
    thread_local auto random = make_random_engine(0x6e7063U);
    std::uniform_real_distribution<float> unit{0.0F, 1.0F};

    constexpr std::array<float, 4> gaits{0.0F, 1.4F, 1.4F, 4.5F};

    lod_counts_ = {};

    for (std::size_t i = 0; i < npcs_.size(); ++i) {
        auto &npc = npcs_[i];

        npc.time_left -= delta_time;
        if (npc.time_left <= 0.0F) {
            npc.time_left = 2.0F + unit(random) * 5.0F;
            npc.speed = gaits[static_cast<std::size_t>(unit(random) * 4.0F) % gaits.size()];
            npc.heading += (unit(random) - 0.5F) * 2.5F;
        }

        if (std::abs(npc.position.x) > crowd_half_area || std::abs(npc.position.y) > crowd_half_area) {
            auto const to_centre = std::atan2(-npc.position.x, -npc.position.y);
            npc.heading += glm::clamp(wrap_angle(to_centre - npc.heading), -3.0F * delta_time, 3.0F * delta_time);
        }

        npc.position += glm::vec2{std::sin(npc.heading), std::cos(npc.heading)} * (npc.speed * delta_time);

        auto const distance = glm::distance(glm::vec3{npc.position.x, 0.0F, npc.position.y}, camera_position);

        auto &input = inputs_[i + 1];
        input.horizontal_speed = npc.speed;
        input.lod = lod_for(distance);
        ++lod_counts_[static_cast<std::size_t>(input.lod)];
    }
}

auto MovingGame::on_update(Scene &scene, float delta_time) -> void {
    frames_since_update_ = 0;

    if (bound_scene_ != &scene) {
        bind_to(scene);
    }

    if (scene.physics_world == nullptr || !body_) {
        return;
    }

    if (skinned_mode_ != batch_skinned_ && (skinned_mode_ || batch_skinned_)) {
        auto const wanted = skinned_mode_ && skin_machine_ != nullptr && skin_data_ != nullptr;
        if (wanted != batch_skinned_) {
            rebuild_batch();
        }
    }

    if (crowd_target_ != crowd_size_) {
        crowd_size_ = crowd_target_;
        set_crowd_size(crowd_size_);
    }

    auto &registry = scene.get_registry();
    auto &physics = *scene.physics_world;

    auto const params = body_->params;
    glm::vec3 direction{0.0F};
    if (!prone_ && controller_.is_moving()) {
        direction = glm::normalize(controller_.desired_horizontal_velocity());
    }

    auto jump_pressed = controller_.consumes_jump() && !prone_;

    step_alpha_ = stepper_.advance(delta_time, [&](float step_dt) {
        body_->step(physics,
                    CharacterInput{
                            .desired_velocity = direction * (sprint_ ? params.run_speed : params.walk_speed),
                            .jump_pressed = std::exchange(jump_pressed, false),
                            .jump_held = jump_held_ && !prone_,
                    },
                    step_dt);

        if (body_->position().y < -20.0F) {
            body_->teleport(player_spawn);
        }
    });

    auto const player_position = body_->interpolated_position(step_alpha_);
    auto const velocity = body_->velocity();
    auto const horizontal_speed = glm::length(glm::vec2{velocity.x, velocity.z});

    if (horizontal_speed > 0.2F) {
        constexpr float turn_rate = 12.0F;
        auto const target = std::atan2(velocity.x, velocity.z);
        auto const max_turn = turn_rate * delta_time;
        facing_yaw_ = wrap_angle(facing_yaw_ + glm::clamp(wrap_angle(target - facing_yaw_), -max_turn, max_turn));
    }

    inputs_[0] = Animation::AnimInputs{
            .horizontal_speed = horizontal_speed,
            .grounded = body_->grounded(),
            .vertical_velocity = velocity.y,
            .prone_requested = prone_,
            .lod = Animation::Lod::Full,
    };

    auto const occlusion_query = [&physics](glm::vec3 const &origin, glm::vec3 const &dir,
                                            float max_distance) -> std::optional<float> {
        if (auto const hit = physics.raycast(origin, dir, max_distance)) {
            return hit->distance;
        }
        return std::nullopt;
    };

    camera_.update(player_position + glm::vec3{0.0F, player_height * 0.5F, 0.0F}, controller_.yaw_degrees(),
                   controller_.pitch_degrees(), params.walk_speed > 0.0F ? horizontal_speed / params.walk_speed : 0.0F,
                   delta_time, occlusion_query);

    update_crowd(delta_time, camera_.position());

    auto const animation_start = Clock::now();
    batch_->update(inputs_, delta_time);
    animation_ms_ = milliseconds_since(animation_start);

    auto const compose_start = Clock::now();

    std::array<std::vector<glm::mat4> *, Joint::JointCount> outputs{};
    for (std::size_t p = 0; p < parts.size(); ++p) {
        if (!registry.valid(part_entities_[p]) || !registry.all_of<Components::InstancedModel>(part_entities_[p])) {
            return;
        }
        auto &instanced = registry.get<Components::InstancedModel>(part_entities_[p]);
        instanced.transforms.resize(batch_skinned_ ? 0 : inputs_.size());
        outputs[p] = &instanced.transforms;
    }

    if (batch_skinned_) {
        for (std::size_t p = 0; p < parts.size(); ++p) {
            registry.get<Components::InstancedModel>(part_entities_[p]).touch();
        }
        compose_skinned(scene, player_position);
        compose_ms_ = milliseconds_since(compose_start);
        record_bench();
        return;
    }
    skinned_drawn_ = 0;
    if (registry.valid(skinned_entity_) && registry.all_of<Components::InstancedModel>(skinned_entity_)) {
        auto &skinned = registry.get<Components::InstancedModel>(skinned_entity_);
        skinned.transforms.clear();
        skinned.palette_offsets.clear();
    }

    for (std::size_t c = 0; c < inputs_.size(); ++c) {
        auto const root = c == 0 ? root_matrix(player_position, facing_yaw_)
                                 : root_matrix({npcs_[c - 1].position.x, 0.0F, npcs_[c - 1].position.y},
                                               npcs_[c - 1].heading);
        auto const palette = batch_->palette(c);

        for (std::size_t p = 0; p < parts.size(); ++p) {
            (*outputs[p])[c] = root * palette[parts[p].joint] * part_rest_[p];
        }
    }

    for (std::size_t p = 0; p < parts.size(); ++p) {
        registry.get<Components::InstancedModel>(part_entities_[p]).touch();
    }

    compose_ms_ = milliseconds_since(compose_start);
    record_bench();
}

auto MovingGame::record_bench() -> void {
    constexpr std::uint32_t warmup = 30;
    if (bench_.target == 0 || bench_.frames >= bench_.target) {
        return;
    }
    if (++bench_.seen <= warmup) {
        return;
    }
    ++bench_.frames;
    bench_.animation_ms += static_cast<double>(animation_ms_);
    bench_.compose_ms += static_cast<double>(compose_ms_);
    bench_.drawn += static_cast<double>(skinned_drawn_);
    if (renderer_ != nullptr) {
        auto const &stats = renderer_->last_frame_stats();
        bench_.skin_jobs += stats.skin_job_count;
        bench_.skin_vertices += stats.skinned_vertex_count;
        bench_.scratch_bytes += static_cast<double>(stats.skin_scratch_bytes_used);
        bench_.fallbacks += stats.skin_fallback_instance_count;
    }
    if (bench_.frames == bench_.target) {
        auto const n = static_cast<double>(bench_.frames);
        std::fprintf(stderr,
                     "BENCH crowd=%u skinned=%d frames=%u animation_ms=%.3f compose_ms=%.3f drawn=%.0f skin_jobs=%.0f "
                     "skin_vertices=%.0f scratch_MiB=%.2f fallbacks=%.0f\n",
                     crowd_size_, batch_skinned_ ? 1 : 0, bench_.frames, bench_.animation_ms / n, bench_.compose_ms / n,
                     bench_.drawn / n, bench_.skin_jobs / n, bench_.skin_vertices / n,
                     bench_.scratch_bytes / n / (1024.0 * 1024.0), bench_.fallbacks / n);
    }
}

auto MovingGame::compose_skinned(Scene &scene, glm::vec3 const &player_position) -> void {
    auto &registry = scene.get_registry();
    skinned_drawn_ = 0;
    if (!registry.valid(skinned_entity_) || !registry.all_of<Components::InstancedModel>(skinned_entity_) ||
        renderer_ == nullptr) {
        return;
    }
    auto &skinned = registry.get<Components::InstancedModel>(skinned_entity_);
    skinned.transforms.clear();
    skinned.palette_offsets.clear();

    static_cast<void>(renderer_->set_skin_palette({}));

    auto const view = camera_.view();
    auto const forward = glm::vec3{view[0][2], view[1][2], view[2][2]};
    auto const camera_position = camera_.position();

    skin_candidates_.clear();
    for (std::uint32_t c = 1; c < inputs_.size(); ++c) {
        auto const position = glm::vec3{npcs_[c - 1].position.x, 0.0F, npcs_[c - 1].position.y};
        auto const to_character = position - camera_position;
        auto const distance = glm::length(to_character);
        if (distance > skin_distance_ || glm::dot(forward, to_character) < -2.0F) {
            continue;
        }
        skin_candidates_.emplace_back(distance, c);
    }
    auto const limit = static_cast<std::size_t>(std::max(max_skinned_ - 1, 0));
    if (skin_candidates_.size() > limit) {
        std::nth_element(skin_candidates_.begin(), skin_candidates_.begin() + static_cast<std::ptrdiff_t>(limit),
                         skin_candidates_.end());
        skin_candidates_.resize(limit);
    }

    auto const fit = glm::translate(glm::scale(glm::mat4{1.0F}, glm::vec3{player_height / skinned_height}),
                                    {0.0F, -skinned_feet_y, 0.0F});

    auto const add = [&](std::uint32_t c) {
        auto const root = c == 0 ? root_matrix(player_position, facing_yaw_)
                                 : root_matrix({npcs_[c - 1].position.x, 0.0F, npcs_[c - 1].position.y},
                                               npcs_[c - 1].heading);
        auto offset = renderer_->append_skin_palette(batch_->palette(c));
        if (!offset) {
            return false;
        }
        skinned.transforms.push_back(root * glm::rotate(glm::mat4{1.0F}, skinned_yaw_offset, world_up) * fit);
        skinned.palette_offsets.push_back(*offset);
        return true;
    };

    if (add(0)) {
        for (auto const &[distance, c]: skin_candidates_) {
            if (!add(c)) {
                break;
            }
        }
    }
    skinned_drawn_ = skinned.transforms.size();
    skinned.touch();
}

auto MovingGame::on_key_pressed(Scene & , KeyPressedEvent const &event) -> void {
    controller_.on_key_pressed(event.key);

    if (event.key == GLFW_KEY_SPACE) {
        jump_held_ = true;
    }
    if (event.key == GLFW_KEY_C) {
        prone_ = !prone_;
    }
    if (event.key == GLFW_KEY_LEFT_SHIFT) {
        sprint_ = true;
    }
}

auto MovingGame::on_key_released(Scene & , KeyReleasedEvent const &event) -> void {
    controller_.on_key_released(event.key);

    if (event.key == GLFW_KEY_SPACE) {
        jump_held_ = false;
    }
    if (event.key == GLFW_KEY_LEFT_SHIFT) {
        sprint_ = false;
    }
}

auto MovingGame::on_mouse_moved(Scene & , MouseMovedEvent const &event) -> void {
    controller_.on_mouse_moved(static_cast<float>(event.delta_x), static_cast<float>(event.delta_y),
                               true);
}

auto MovingGame::on_ui(Scene & , Renderer & ) -> void {
    if (frames_since_update_ > 2 || !body_ || !batch_) {
        return;
    }
    ++frames_since_update_;

    gui::widget("Moving", [&] {
        ImGui::TextUnformatted("WASD move, Shift run, Space jump, C lie down / get up");

        ImGui::Separator();
        auto const skin_available = skin_machine_ != nullptr && skin_data_ != nullptr;
        ImGui::BeginDisabled(!skin_available);
        if (ImGui::RadioButton("Rigid rig", !skinned_mode_)) {
            skinned_mode_ = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Skinned model", skinned_mode_)) {
            skinned_mode_ = true;
        }
        ImGui::EndDisabled();
        if (!skin_available) {
            ImGui::TextUnformatted("Skinned model unavailable (see assets/models/README.md: animated_human.glb)");
        } else if (batch_skinned_) {
            ImGui::SliderInt("Max skinned characters", &max_skinned_, 1, 2000);
            ImGui::SliderFloat("Skin distance", &skin_distance_, 10.0F, 300.0F);
            ImGui::Text("Skinned drawn %zu of %zu", skinned_drawn_, inputs_.size());
            if (renderer_ != nullptr) {
                auto const &stats = renderer_->last_frame_stats();
                ImGui::Text("Skin jobs %u  vertices %u  scratch %.2f MiB  rest-pose fallbacks %u", stats.skin_job_count,
                            stats.skinned_vertex_count,
                            static_cast<double>(stats.skin_scratch_bytes_used) / (1024.0 * 1024.0),
                            stats.skin_fallback_instance_count);
            }
        }

        ImGui::Separator();
        auto crowd = static_cast<int>(crowd_target_);
        if (ImGui::SliderInt("NPCs", &crowd, 0, static_cast<int>(max_crowd))) {
            crowd_target_ = static_cast<std::uint32_t>(crowd);
        }

        ImGui::Text("Animation update  %.3f ms", static_cast<double>(animation_ms_));
        ImGui::Text("Transform compose %.3f ms", static_cast<double>(compose_ms_));
        if (!batch_skinned_) {
            ImGui::Text("Instanced transforms uploaded  %zu (%zu parts x %zu characters)",
                        parts.size() * inputs_.size(), parts.size(), inputs_.size());
        }
        ImGui::Text("LOD  full %zu  reduced %zu  hold %zu", lod_counts_[0], lod_counts_[1], lod_counts_[2]);

        ImGui::Separator();
        auto const &state = batch_->state(0);
        auto const &definition = active_machine().definition(state.current);
        ImGui::Text("Player state  %.*s%s", static_cast<int>(definition.name.size()), definition.name.data(),
                    prone_ ? " (prone requested)" : "");
        ImGui::Text("Speed %.2f m/s   grounded %s   vertical %.2f m/s",
                    static_cast<double>(glm::length(glm::vec2{body_->velocity().x, body_->velocity().z})),
                    body_->grounded() ? "yes" : "no", static_cast<double>(body_->velocity().y));

        ImGui::Separator();
        auto &p = body_->params;
        ImGui::SliderFloat("Walk speed", &p.walk_speed, 0.5F, 10.0F);
        ImGui::SliderFloat("Run speed", &p.run_speed, 1.0F, 15.0F);
        ImGui::SliderFloat("Ground accel", &p.ground_accel, 5.0F, 150.0F);
        ImGui::SliderFloat("Ground decel", &p.ground_decel, 5.0F, 150.0F);
        ImGui::SliderFloat("Air accel", &p.air_accel, 0.0F, 80.0F);
        ImGui::SliderFloat("Gravity", &p.gravity, 5.0F, 60.0F);
        ImGui::SliderFloat("Jump height", &p.jump_height, 0.2F, 3.0F);
        ImGui::SliderFloat("Fall gravity x", &p.fall_gravity_multiplier, 1.0F, 4.0F);
        ImGui::SliderFloat("Jump cut x", &p.jump_cut_multiplier, 1.0F, 8.0F);
        ImGui::SliderFloat("Coyote time", &p.coyote_time, 0.0F, 0.4F);
        ImGui::SliderFloat("Jump buffer", &p.jump_buffer_time, 0.0F, 0.4F);
        ImGui::SliderFloat("Max slope", &p.max_slope_degrees, 10.0F, 80.0F);
        ImGui::SliderFloat("Step height", &p.step_height, 0.0F, 0.8F);

        if (ImGui::Button("Defaults")) {
            p = MovementParams{};
        }
    });
}

auto MovingGame::camera(Scene const & , float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = camera_.view(),
            .projection = camera_.projection(aspect_ratio),
            .near_clip = camera_.near_clip(),
            .far_clip = camera_.far_clip(),
            .vertical_fov_radians = glm::radians(camera_.field_of_view_degrees()),
    };
}
