#include "meshlet_showcase_game.hxx"

#include <algorithm>
#include <cmath>
#include <format>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/common.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    constexpr std::array<char const *, 2> model_files{"assets/models/damaged_helmet.glb",
                                                      "assets/models/scattering_skull.glb"};

    constexpr std::array<char const *, 2> model_names{"helmet", "skull"};
    constexpr std::array<char const *, 3> look_names{"plain", "colours", "exploded"};

    // glTF rest orientation of each model, so it stands upright in the studio.
    auto const model_rotations = std::array<glm::quat, 2>{
            glm::angleAxis(glm::radians(90.0F), glm::vec3{1.0F, 0.0F, 0.0F}),
            glm::angleAxis(glm::radians(-90.0F), glm::vec3{0.0F, 1.0F, 0.0F}),
    };

    constexpr float fitted_size = 2.0F;
    constexpr float parked_y = -500.0F;
    constexpr float blend_rate = 3.0F;
}

auto MeshletShowcaseGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;
    bound_scene_ = nullptr;

    scene.environment.source = EnvironmentSource::flat_ambient;
    scene.environment.draw_skybox = false;
    scene.environment.ambient_intensity = 0.35F;
    scene.environment.exposure_ev = 0.0F;
    scene.environment.sun.azimuth_degrees = 145.0F;
    scene.environment.sun.elevation_degrees = 32.0F;
    scene.environment.sun.intensity = 4.0F;

    for (std::size_t m = 0; m < model_count; ++m) {
        auto loaded = renderer.load_model(data_path(model_files[m]));

        auto model = engine_models.cube;
        if (loaded) {
            model = *loaded;
        } else {
            error("[MeshletShowcase] Could not load '{}': {}", model_files[m], describe(loaded.error()));
        }

        if (auto const bounds = renderer.model_submesh_bounds(model); bounds && !bounds->empty()) {
            auto low = bounds->front().first;
            auto high = bounds->front().second;
            for (auto const &[lo, hi]: *bounds) {
                low = glm::min(low, lo);
                high = glm::max(high, hi);
            }
            auto const extent = high - low;
            fits_[m].scale = fitted_size / std::max({extent.x, extent.y, extent.z});
            fits_[m].centre = (low + high) * 0.5F;
        }
        fits_[m].rotation = model_rotations[m];

        auto &set = materials_[m];
        set.sources = renderer.model_materials(model);

        for (std::size_t look = 0; look < look_count; ++look) {
            for (auto const source: set.sources) {
                if (look == static_cast<std::size_t>(Look::plain)) {
                    set.variants[look].push_back(source);
                    continue;
                }

                auto copy = renderer.duplicate_material(source, std::format("{}_{}", model_names[m], look_names[look]));
                if (!copy) {
                    error("[MeshletShowcase] Could not duplicate a material: {}", describe(copy.error()));
                    set.variants[look].push_back(source);
                    continue;
                }

                if (auto const *info = renderer.material_storage().create_info(*copy)) {
                    auto variant = *info;
                    variant.debug_meshlet_colours = true;
                    static_cast<void>(renderer.update_material(*copy, variant));
                }
                set.variants[look].push_back(*copy);
            }
        }

        for (std::size_t look = 0; look < look_count; ++look) {
            auto const prop = Entity{&scene, "{}_{}", model_names[m], look_names[look]};
            prop.emplace<Components::Transform>(Components::Transform{
                    .position = glm::vec3{0.0F, parked_y, 0.0F},
                    .scale = glm::vec3{fits_[m].scale},
            });
            prop.emplace<Components::Model>(Components::Model{.model = model});

            if (look != static_cast<std::size_t>(Look::plain)) {
                prop.emplace<Components::DebugMeshletCones>(Components::DebugMeshletCones{.enabled = false});
            }

            if (look != static_cast<std::size_t>(Look::plain)) {
                Components::MaterialOverride override_materials;
                for (std::size_t i = 0; i < set.sources.size(); ++i) {
                    override_materials.slots.push_back(
                            MaterialSlotOverride{.source = set.sources[i], .material = set.variants[look][i]});
                }
                prop.emplace<Components::MaterialOverride>(std::move(override_materials));
            }
        }
    }

    constexpr std::array<glm::vec3, 3> rim_colours{glm::vec3{0.1F, 0.8F, 1.0F}, glm::vec3{1.0F, 0.2F, 0.7F},
                                                   glm::vec3{1.0F, 0.75F, 0.3F}};
    constexpr std::array<glm::vec3, 3> rim_positions{glm::vec3{-7.0F, 3.0F, -5.0F}, glm::vec3{7.0F, 2.5F, -5.0F},
                                                     glm::vec3{0.0F, 6.0F, 6.0F}};

    for (std::size_t i = 0; i < rim_colours.size(); ++i) {
        auto const light = Entity{&scene, "rim_light_{}", i};
        light.emplace<Components::Transform>(Components::Transform{.position = rim_positions[i]});
        light.emplace<Components::PointLight>(Components::PointLight{
                .colour = rim_colours[i],
                .intensity = i == 2 ? 30.0F : 90.0F,
                .range = 24.0F,
        });
    }

    // Left to right on screen: the camera looks down -z with +x to its left.
    auto const place = [](Shot &shot, std::size_t model, Look look, float x, float scale = 1.0F) {
        shot.props[prop_index(model, look)] = Placement{.visible = true, .position = {-x, 0.0F, 0.0F}, .scale = scale};
    };

    constexpr std::size_t helmet = 0;
    constexpr std::size_t skull = 1;

    shots_[0] = Shot{.title = "Plain", .caption = "A dense mesh, shaded normally", .target = {0.0F, 0.0F, 0.0F},
                     .distance = 4.4F, .height = 0.05F, .yaw_degrees = 18.0F, .yaw_speed = 0.0F};
    place(shots_[0], helmet, Look::plain, 0.0F);

    shots_[1] = Shot{.title = "Meshlets", .caption = "Each colour is one meshlet: up to 64 vertices, 124 triangles",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 4.4F, .height = 0.05F, .yaw_degrees = 18.0F, .yaw_speed = 0.0F};
    place(shots_[1], helmet, Look::colours, 0.0F);

    shots_[2] = Shot{.title = "Exploded", .caption = "Meshlets pushed out along their normal cone and shrunk",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 5.6F, .height = 0.05F, .yaw_degrees = 18.0F, .yaw_speed = 0.0F,
                     .extrude = 0.7F, .shrink = 0.12F};
    place(shots_[2], helmet, Look::exploded, 0.0F);

    shots_[3] = Shot{.title = "Side by side", .caption = "Plain, meshlets, exploded",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 9.8F, .height = 0.2F, .yaw_degrees = 0.0F,
                     .yaw_speed = 0.0F, .fov_degrees = 34.0F, .extrude = 0.7F, .shrink = 0.12F};
    place(shots_[3], helmet, Look::plain, -3.3F);
    place(shots_[3], helmet, Look::colours, 0.0F);
    place(shots_[3], helmet, Look::exploded, 3.3F);

    shots_[4] = Shot{.title = "Skull", .caption = "Millions of triangles, thousands of meshlets",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 7.5F, .height = 0.2F, .yaw_degrees = 0.0F,
                     .yaw_speed = 0.0F, .fov_degrees = 34.0F, .extrude = 3.0F, .shrink = 0.2F};
    place(shots_[4], skull, Look::colours, -2.1F);
    place(shots_[4], skull, Look::exploded, 2.1F);

    shots_[5] = Shot{.title = "Close-up", .caption = "Meshlets are small, local clusters of triangles",
                     .target = {0.0F, 0.05F, 0.0F}, .distance = 2.6F, .height = 0.1F, .yaw_degrees = 15.0F,
                     .yaw_speed = 0.0F, .fov_degrees = 32.0F, .extrude = 0.3F, .shrink = 0.22F};
    place(shots_[5], helmet, Look::exploded, 0.0F);

    shots_[6] = Shot{.title = "Shattered", .caption = "Extrude pushed to the limit",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 7.5F, .height = 0.05F, .yaw_degrees = 18.0F, .yaw_speed = 0.0F,
                     .extrude = 1.8F, .shrink = 0.35F};
    place(shots_[6], helmet, Look::exploded, 0.0F);

    shots_[7] = Shot{.title = "Normal cones", .caption = "Each meshlet's cone of surface normals",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 4.4F, .height = 0.05F, .yaw_degrees = 18.0F,
                     .yaw_speed = 0.0F, .cones = true};
    place(shots_[7], helmet, Look::colours, 0.0F);

    shots_[8] = Shot{.title = "Cones, exploded", .caption = "Meshlets slide out along their cone axis",
                     .target = {0.0F, 0.0F, 0.0F}, .distance = 6.0F, .height = 0.05F, .yaw_degrees = 18.0F,
                     .yaw_speed = 0.0F, .extrude = 0.8F, .shrink = 0.12F, .cones = true};
    place(shots_[8], helmet, Look::exploded, 0.0F);

    shot_index_ = 0;
}

auto MeshletShowcaseGame::bind_to(Scene &scene) -> void {
    bound_scene_ = &scene;

    for (std::size_t m = 0; m < model_count; ++m) {
        for (std::size_t look = 0; look < look_count; ++look) {
            props_[(m * look_count) + look] =
                    scene.find_entity(std::format("{}_{}", model_names[m], look_names[look]));
        }
    }

    auto const &shot = shots_[shot_index_];
    extrude_ = shot.extrude;
    shrink_ = shot.shrink;
    applied_extrude_ = -1.0F;
    applied_shrink_ = -1.0F;
    yaw_degrees_ = shot.yaw_degrees;
    apply_shot(scene);
}

auto MeshletShowcaseGame::apply_shot(Scene &scene) -> void {
    auto &registry = scene.get_registry();
    auto const &shot = shots_[shot_index_];

    for (std::size_t m = 0; m < model_count; ++m) {
        for (std::size_t look = 0; look < look_count; ++look) {
            auto const index = (m * look_count) + look;
            auto const entity = props_[index];

            if (!registry.valid(entity)) {
                continue;
            }

            auto &transform = registry.get<Components::Transform>(entity);
            auto const &placement = shot.props[index];
            transform.scale = glm::vec3{fits_[m].scale * placement.scale};
            transform.rotation = fits_[m].rotation;
            transform.position = placement.visible
                                         ? placement.position - fits_[m].rotation * (fits_[m].centre * transform.scale.x)
                                         : glm::vec3{0.0F, parked_y, 0.0F};
        }
    }

    for (auto const entity: props_) {
        if (auto *cones = registry.try_get<Components::DebugMeshletCones>(entity)) {
            cones->enabled = shot.cones;
        }
    }

    info("[MeshletShowcase] Shot {}: {}", shot_index_ + 1, shot.title);
}

auto MeshletShowcaseGame::set_exploded(float extrude, float shrink) -> void {
    if (renderer_ == nullptr || (std::abs(extrude - applied_extrude_) < 1e-4F && std::abs(shrink - applied_shrink_) < 1e-4F)) {
        return;
    }

    applied_extrude_ = extrude;
    applied_shrink_ = shrink;

    for (auto const &set: materials_) {
        for (auto const handle: set.variants[static_cast<std::size_t>(Look::exploded)]) {
            if (auto const *current = renderer_->material_storage().create_info(handle)) {
                auto next = *current;
                next.meshlet_extrude = extrude;
                next.meshlet_shrink = shrink;
                static_cast<void>(renderer_->update_material(handle, next));
            }
        }
    }
}

auto MeshletShowcaseGame::on_update(Scene &scene, float delta_time) -> void {
    if (bound_scene_ != &scene) {
        bind_to(scene);
    }

    auto const &shot = shots_[shot_index_];
    auto const blend = 1.0F - std::exp(-blend_rate * delta_time);

    extrude_ += (shot.extrude - extrude_) * blend;
    shrink_ += (shot.shrink - shrink_) * blend;
    set_exploded(extrude_, shrink_);

    for (auto const entity: props_) {
        if (auto *cones = scene.get_registry().try_get<Components::DebugMeshletCones>(entity)) {
            cones->extrude = extrude_;
            cones->shrink = shrink_;
        }
    }

    time_ += delta_time;
    yaw_degrees_ += shot.yaw_speed * delta_time;

    auto const yaw = glm::radians(yaw_degrees_);
    auto const eye_offset = glm::vec3{std::sin(yaw) * shot.distance, shot.height * shot.distance,
                                      std::cos(yaw) * shot.distance};
    eye_ = shot.target + eye_offset;
    look_at_ = shot.target;
    fov_degrees_ = shot.fov_degrees;
}

auto MeshletShowcaseGame::on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void {
    if (event.key >= GLFW_KEY_1 && event.key < GLFW_KEY_1 + static_cast<std::int32_t>(shot_count)) {
        shot_index_ = static_cast<std::size_t>(event.key - GLFW_KEY_1);
        yaw_degrees_ = shots_[shot_index_].yaw_degrees;
        apply_shot(scene);
    }
}

auto MeshletShowcaseGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = glm::lookAtLH(eye_, look_at_, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov_degrees_), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(fov_degrees_),
    };
}
