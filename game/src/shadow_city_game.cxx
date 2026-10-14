#include "shadow_city_game.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string_view>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {
    struct ModelSource {
        std::string_view file;
        bool cached;
        float pitch_degrees;
        float yaw_degrees;
    };

    // Local models, then the gallery's downloads: twenty meshes in all.
    constexpr std::array<ModelSource, 20> sources{{
            {"assets/models/damaged_helmet.glb", false, 90.0F, 0.0F},
            {"assets/models/scattering_skull.glb", false, 0.0F, -150.0F},
            {"assets/models/chess/white_knight.glb", false, 0.0F, 0.0F},
            {"AnisotropyBarnLamp", true, 0.0F, 180.0F},
            {"AntiqueCamera", true, 0.0F, 180.0F},
            {"Avocado", true, 0.0F, 180.0F},
            {"BarramundiFish", true, 0.0F, 180.0F},
            {"BoomBox", true, 0.0F, 180.0F},
            {"CesiumMilkTruck", true, 0.0F, 180.0F},
            {"Corset", true, 0.0F, 180.0F},
            {"DragonAttenuation", true, 0.0F, 180.0F},
            {"Duck", true, 0.0F, 180.0F},
            {"GlamVelvetSofa", true, 0.0F, 180.0F},
            {"IridescenceLamp", true, 0.0F, 180.0F},
            {"Lantern", true, 0.0F, 180.0F},
            {"MaterialsVariantsShoe", true, 0.0F, 180.0F},
            {"MosquitoInAmber", true, 0.0F, 180.0F},
            {"SheenChair", true, 0.0F, 180.0F},
            {"ToyCar", true, 0.0F, 180.0F},
            {"WaterBottle", true, 0.0F, 180.0F},
    }};

    constexpr std::uint32_t copies_per_model = 50;
    constexpr std::uint32_t grid_columns = 40;
    constexpr float cell = 3.4F;
    constexpr float base_size = 2.0F;

    constexpr std::uint32_t spot_columns = 6;
    constexpr std::uint32_t spot_rows = 4;
    constexpr std::uint32_t point_count = 10;

    // A small deterministic generator: the layout is the same every run.
    auto next_random(std::uint32_t &state) -> float {
        state = (state * 1664525U) + 1013904223U;
        return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
    }
}

auto ShadowCityGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;

    scene.environment.source = EnvironmentSource::flat_ambient;
    scene.environment.draw_skybox = false;
    scene.environment.ambient_intensity = 0.04F;
    scene.environment.sun.intensity = 0.0F;
    scene.environment.exposure_ev = 0.0F;

    auto bloom = renderer.bloom_settings();
    bloom.enabled = false;
    renderer.set_bloom_settings(bloom);

    auto shadows = renderer.local_shadow_settings();
    shadows.maximum_lights = 32;
    shadows.minimum_screen_radius_pixels = 4.0F;
    renderer.set_local_shadow_settings(shadows);

    auto &images = renderer.image_storage();
    auto &samplers = renderer.sampler_storage();

    float const half_width = static_cast<float>(grid_columns) * cell * 0.5F;
    constexpr std::uint32_t total = copies_per_model * static_cast<std::uint32_t>(sources.size());
    constexpr std::uint32_t grid_rows = (total + grid_columns - 1) / grid_columns;
    float const half_depth = static_cast<float>(grid_rows) * cell * 0.5F;

    if (auto floor_material = renderer.create_material(MaterialCreateInfo{
                .base_colour_factor = glm::vec4{0.3F, 0.3F, 0.32F, 1.0F},
                .metallic_factor = 0.0F,
                .roughness_factor = 0.85F,
                .base_colour_texture = images.white(),
                .normal_texture = images.flat_normal(),
                .metallic_roughness_texture = images.white(),
                .occlusion_texture = images.occlusion(),
                .emissive_texture = images.emissive(),
                .sampler = samplers.linear_repeat(),
        });
        floor_material) {
        auto const floor = Entity{&scene, "city_floor"};
        floor.emplace<Components::Transform>(Components::Transform{
                .position = {0.0F, -0.25F, 0.0F}, .scale = {half_width + 20.0F, 0.25F, half_depth + 20.0F}});
        floor.emplace<Components::Model>(Components::Model{.model = engine_models.cube});
        floor.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = *floor_material});
    } else {
        error("[ShadowCity] Could not create the floor material: {}", describe(floor_material.error()));
    }

    // Load every mesh once; each is scaled to a common size and stood on the floor.
    struct Loaded {
        ModelHandle model;
        glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
        glm::vec3 centre{0.0F};
        float scale = 1.0F;
        float below_centre = 0.0F;
        bool valid = false;
    };
    std::array<Loaded, sources.size()> models{};

    for (std::size_t i = 0; i < sources.size(); ++i) {
        auto const &source = sources[i];
        auto const path = source.cached ? cache_path(std::format("downloads/gallery/{}.glb", source.file)).absolute()
                                        : data_path(source.file).absolute();
        auto const asset = AssetPath::external(path);
        auto loaded = asset ? renderer.load_model(*asset) : std::unexpected(RendererError{});
        if (!loaded) {
            warn("[ShadowCity] Skipping '{}': {}", source.file, describe(loaded.error()));
            continue;
        }

        auto &entry = models[i];
        entry.model = *loaded;
        entry.rotation = glm::angleAxis(glm::radians(source.yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}) *
                         glm::angleAxis(glm::radians(source.pitch_degrees), glm::vec3{1.0F, 0.0F, 0.0F});
        entry.below_centre = 0.5F * base_size;

        if (auto const bounds = renderer.model_submesh_bounds(entry.model); bounds && !bounds->empty()) {
            auto low = bounds->front().first;
            auto high = bounds->front().second;
            for (auto const &[lo, hi]: *bounds) {
                low = glm::min(low, lo);
                high = glm::max(high, hi);
            }
            auto const extent = high - low;
            entry.scale = base_size / std::max({extent.x, extent.y, extent.z, 1e-4F});
            entry.centre = (low + high) * 0.5F;

            float lowest = 0.0F;
            for (int corner = 0; corner < 8; ++corner) {
                auto const point = glm::vec3{(corner & 1) != 0 ? high.x : low.x, (corner & 2) != 0 ? high.y : low.y,
                                             (corner & 4) != 0 ? high.z : low.z};
                lowest = std::min(lowest, (entry.rotation * ((point - entry.centre) * entry.scale)).y);
            }
            entry.below_centre = -lowest;
        }
        entry.valid = true;
    }

    // Shuffle which mesh sits in which grid cell, so the kinds are mixed instead of banded.
    std::array<std::uint32_t, total> order{};
    for (std::uint32_t i = 0; i < total; ++i) {
        order[i] = i / copies_per_model;
    }
    std::uint32_t seed = 0x1234ABCDU;
    for (std::uint32_t i = total - 1; i > 0; --i) {
        std::swap(order[i], order[static_cast<std::uint32_t>(next_random(seed) * static_cast<float>(i + 1))]);
    }

    std::uint32_t placed = 0;
    for (std::uint32_t i = 0; i < total; ++i) {
        auto const &entry = models[order[i]];
        if (!entry.valid) {
            continue;
        }

        auto const column = i % grid_columns;
        auto const row = i / grid_columns;
        auto const jitter_x = (next_random(seed) - 0.5F) * 0.8F;
        auto const jitter_z = (next_random(seed) - 0.5F) * 0.8F;
        auto const size = 0.8F + (0.7F * next_random(seed));
        auto const yaw = glm::angleAxis(glm::radians(360.0F * next_random(seed)), glm::vec3{0.0F, 1.0F, 0.0F});

        auto const rotation = yaw * entry.rotation;
        auto const scale = entry.scale * size;
        auto const stand = glm::vec3{(static_cast<float>(column) + 0.5F) * cell - half_width + jitter_x,
                                     entry.below_centre * size,
                                     (static_cast<float>(row) + 0.5F) * cell - half_depth + jitter_z};

        auto const object = Entity{&scene, "city_{}", i};
        object.emplace<Components::Transform>(Components::Transform{
                .position = stand - (rotation * (entry.centre * scale)),
                .rotation = rotation,
                .scale = glm::vec3{scale},
        });
        object.emplace<Components::Model>(Components::Model{.model = entry.model});
        ++placed;
    }

    // Spot lights from above over the whole grid, and point lights low between the rows: more than the atlas holds.
    constexpr std::array<glm::vec3, 6> palette{{{1.0F, 0.8F, 0.6F},
                                                {0.6F, 0.8F, 1.0F},
                                                {1.0F, 0.55F, 0.45F},
                                                {0.6F, 1.0F, 0.7F},
                                                {0.9F, 0.65F, 1.0F},
                                                {1.0F, 0.95F, 0.7F}}};

    for (std::uint32_t row = 0; row < spot_rows; ++row) {
        for (std::uint32_t column = 0; column < spot_columns; ++column) {
            auto const index = (row * spot_columns) + column;
            auto const x = ((static_cast<float>(column) + 0.5F) / static_cast<float>(spot_columns) - 0.5F) * 2.0F *
                           half_width * 0.9F;
            auto const z = ((static_cast<float>(row) + 0.5F) / static_cast<float>(spot_rows) - 0.5F) * 2.0F *
                           half_depth * 0.9F;

            auto const light = Entity{&scene, "city_spot_{}", index};
            auto const aim = glm::normalize(glm::vec3{0.15F * std::sin(static_cast<float>(index)), -1.0F,
                                                      0.15F * std::cos(static_cast<float>(index))});
            light.emplace<Components::Transform>(Components::Transform{
                    .position = {x, 9.0F, z},
                    .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F}, aim),
            });
            light.emplace<Components::SpotLight>(Components::SpotLight{
                    .colour = palette[index % palette.size()],
                    .intensity = 1100.0F,
                    .range = 28.0F,
                    .inner_cone_degrees = 28.0F,
                    .outer_cone_degrees = 52.0F,
                    .casts_shadows = true,
            });
        }
    }

    for (std::uint32_t i = 0; i < point_count; ++i) {
        auto const t = (static_cast<float>(i) + 0.5F) / static_cast<float>(point_count);
        auto const light = Entity{&scene, "city_point_{}", i};
        light.emplace<Components::Transform>(Components::Transform{
                .position = {(t - 0.5F) * 2.0F * half_width * 0.8F, 3.2F,
                             (std::fmod(t * 7.0F, 1.0F) - 0.5F) * half_depth * 1.6F}});
        light.emplace<Components::PointLight>(Components::PointLight{.colour = palette[(i + 3) % palette.size()],
                                                                     .intensity = 80.0F,
                                                                     .range = 14.0F,
                                                                     .casts_shadows = true});
    }

    info("[ShadowCity] {} casters from {} meshes, {} spot and {} point shadow lights", placed, sources.size(),
         spot_columns * spot_rows, point_count);
    info("[ShadowCity] Keys: 1-3 shots, P pause orbit, C GPU culling, V caching, B bloom");
}

auto ShadowCityGame::on_update(Scene &, float) -> void {
    if (orbiting_) {
        ++frame_;
    }

    if (renderer_ != nullptr && ++updates_ % 60 == 0) {
        auto const &stats = renderer_->local_shadow_stats();
        info("[ShadowCity] lights requested={} shadowed={} tiles={} cached={} cells={} culled={}",
             stats.requested_lights, stats.shadowed_lights, stats.rendered_tiles, stats.cached_lights,
             stats.atlas_cells_used, stats.culled ? 1 : 0);
    }
}

auto ShadowCityGame::on_key_pressed(Scene &, KeyPressedEvent const &event) -> void {
    if (renderer_ == nullptr) {
        return;
    }

    auto settings = renderer_->local_shadow_settings();

    switch (event.key) {
        case GLFW_KEY_1:
            shot_ = 0;
            break;
        case GLFW_KEY_2:
            shot_ = 1;
            break;
        case GLFW_KEY_3:
            shot_ = 2;
            break;
        case GLFW_KEY_P:
            orbiting_ = !orbiting_;
            break;
        case GLFW_KEY_C:
            settings.gpu_culling = !settings.gpu_culling;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCity] GPU culling {}", settings.gpu_culling ? "on" : "off");
            break;
        case GLFW_KEY_V:
            settings.caching = !settings.caching;
            renderer_->set_local_shadow_settings(settings);
            info("[ShadowCity] Caching {}", settings.caching ? "on" : "off");
            break;
        case GLFW_KEY_B: {
            bloom_ = !bloom_;
            auto bloom = renderer_->bloom_settings();
            bloom.enabled = bloom_;
            renderer_->set_bloom_settings(bloom);
            info("[ShadowCity] Bloom {}", bloom_ ? "on" : "off");
            break;
        }
        default:
            break;
    }
}

auto ShadowCityGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    // Shot 1 circles the grid high up, shot 2 walks between the objects, shot 3 looks down the long axis.
    auto const angle = static_cast<float>(frame_) * 0.004F;
    glm::vec3 eye{};
    glm::vec3 target{};

    switch (shot_) {
        case 1:
            eye = {std::cos(angle * 2.0F) * 18.0F, 3.0F, std::sin(angle * 2.0F) * 10.0F};
            target = {eye.x + std::sin(angle * 2.0F) * 6.0F, 1.0F, eye.z - std::cos(angle * 2.0F) * 4.0F};
            break;
        case 2:
            eye = {-70.0F, 9.0F, 0.0F};
            target = {0.0F, 0.5F, 0.0F};
            break;
        default:
            eye = {std::cos(angle) * 48.0F, 24.0F, std::sin(angle) * 34.0F};
            target = {0.0F, 0.0F, 0.0F};
            break;
    }

    return CameraParams{
            .view = glm::lookAtLH(eye, target, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(55.0F), aspect_ratio, 0.1F, 400.0F),
            .near_clip = 0.1F,
            .far_clip = 400.0F,
            .vertical_fov_radians = glm::radians(55.0F),
    };
}
