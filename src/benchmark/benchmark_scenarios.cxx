#include "benchmark/benchmark_scenarios.hxx"

#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

#include "game/shadow_city.hxx"
#include "assets/material_storage.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/random.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    constexpr std::uint32_t random_stream_base = 0xBE7C'0000U;

    [[nodiscard]] auto orbit_path(glm::vec3 const &centre, float radius, float low, float high,
                                  std::uint32_t count) -> std::vector<CameraKeyframe> {
        std::vector<CameraKeyframe> keyframes;
        keyframes.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            auto const angle = static_cast<float>(i) / static_cast<float>(count) * 2.0F * std::numbers::pi_v<float>;
            auto const height = (i % 2 == 0) ? high : low;
            keyframes.push_back(CameraKeyframe{
                    .position = centre + glm::vec3{radius * std::cos(angle), height, radius * std::sin(angle)},
                    .target = centre,
            });
        }
        return keyframes;
    }

    class MaterialSet {
    public:
        explicit MaterialSet(Renderer &renderer) : renderer_(&renderer) {}
        ~MaterialSet() {
            for (auto const material: created_) {
                renderer_->release_material(material);
            }
        }

        auto keep(MaterialHandle material, std::vector<MaterialHandle> &owned) -> MaterialHandle {
            if (auto const found = std::ranges::find(created_, material); found != created_.end()) {
                created_.erase(found);
                owned.push_back(material);
            }
            return material;
        }

        MaterialSet(MaterialSet const &) = delete;
        MaterialSet(MaterialSet &&) = delete;
        auto operator=(MaterialSet const &) -> MaterialSet & = delete;
        auto operator=(MaterialSet &&) -> MaterialSet & = delete;

        auto make(glm::vec4 const &colour, AlphaMode alpha_mode = AlphaMode::opaque, float roughness = 0.6F,
                  std::uint32_t max_shadow_cascade = shadow_cascade_count - 1) -> MaterialHandle {
            auto &images = renderer_->image_storage();
            auto &samplers = renderer_->sampler_storage();

            auto material = renderer_->create_material(MaterialCreateInfo{
                    .base_colour_factor = colour,
                    .metallic_factor = 0.0F,
                    .roughness_factor = roughness,
                    .base_colour_texture = images.white(),
                    .normal_texture = images.flat_normal(),
                    .metallic_roughness_texture = images.metallic_roughness(),
                    .occlusion_texture = images.occlusion(),
                    .emissive_texture = images.emissive(),
                    .sampler = samplers.linear_repeat(),
                    .alpha_mode = alpha_mode,
                    .max_shadow_cascade = max_shadow_cascade,
            });

            if (!material) {
                error("benchmark scenario: could not create a material: {}", describe(material.error()));
                return MaterialHandle{};
            }

            created_.push_back(*material);
            return *material;
        }

    private:
        Renderer *renderer_;
        std::vector<MaterialHandle> created_;
    };

    auto spawn(Scene &scene, std::string const &name, ModelHandle model, Components::Transform const &transform,
               MaterialHandle material) -> void {
        auto entity = GeneratedEntity{&scene, "{}", name};
        entity.emplace<Components::Transform>(transform);
        entity.emplace<Components::Model>(Components::Model{.model = model});
        if (material.valid()) {
            entity.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = material});
        }
    }

    auto spawn_ground(BenchmarkScenarioContext const &context, MaterialSet &materials, float size) -> void {
        spawn(context.scene, "bench_ground", context.engine_models.cube,
              Components::Transform{.position = glm::vec3{0.0F, -0.1F, 0.0F}, .scale = glm::vec3{size, 0.2F, size}},
              materials.make(glm::vec4{0.45F, 0.45F, 0.47F, 1.0F}, AlphaMode::opaque, 0.9F));
    }

    [[nodiscard]] auto palette(MaterialSet &materials, std::uint32_t count) -> std::vector<MaterialHandle> {
        std::vector<MaterialHandle> result;
        result.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            auto const hue = static_cast<float>(i) / static_cast<float>(count);
            auto const colour = glm::vec3{0.5F + 0.5F * std::cos(6.2831853F * (hue + 0.00F)),
                                          0.5F + 0.5F * std::cos(6.2831853F * (hue + 0.33F)),
                                          0.5F + 0.5F * std::cos(6.2831853F * (hue + 0.67F))};
            result.push_back(materials.make(glm::vec4{colour * 0.8F + 0.1F, 1.0F}));
        }
        return result;
    }

    [[nodiscard]] auto grid_side(std::uint32_t count) -> std::uint32_t {
        return std::max(1U, static_cast<std::uint32_t>(std::ceil(std::sqrt(static_cast<double>(count)))));
    }

    constexpr float draw_call_spacing = 2.5F;

    auto populate_draw_calls(BenchmarkScenarioContext const &context) -> void {
        MaterialSet materials{context.renderer};
        auto const colours = palette(materials, 16);

        auto const side = grid_side(context.load);
        auto const extent = static_cast<float>(side) * draw_call_spacing;
        spawn_ground(context, materials, extent + 20.0F);

        auto engine = make_random_engine(random_stream_base + 1);
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};

        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto const x = static_cast<float>(i % side) * draw_call_spacing - extent * 0.5F;
            auto const row = i / side;
            auto const z = static_cast<float>(row) * draw_call_spacing - extent * 0.5F;
            auto const scale = 0.6F + 0.6F * unit(engine);
            auto const yaw = unit(engine) * 2.0F * std::numbers::pi_v<float>;

            spawn(context.scene, std::format("bench_object_{}", i),
                  i % 2 == 0 ? context.engine_models.cube : context.engine_models.sphere,
                  Components::Transform{
                          .position = glm::vec3{x, scale * 0.5F, z},
                          .rotation = glm::angleAxis(yaw, glm::vec3{0.0F, 1.0F, 0.0F}),
                          .scale = glm::vec3{scale},
                  },
                  colours[i % colours.size()]);
        }
    }

    [[nodiscard]] auto draw_calls_path(std::uint32_t load) -> std::vector<CameraKeyframe> {
        auto const extent = static_cast<float>(grid_side(load)) * draw_call_spacing;
        return orbit_path(glm::vec3{0.0F}, extent * 0.55F + 8.0F, 4.0F, extent * 0.35F + 6.0F, 8);
    }

    constexpr float instance_spacing = 1.6F;

    auto populate_instances(BenchmarkScenarioContext const &context, bool cast_shadows) -> void {
        MaterialSet materials{context.renderer};

        auto const side = grid_side(context.load);
        auto const extent = static_cast<float>(side) * instance_spacing;
        spawn_ground(context, materials, extent + 20.0F);

        auto engine = make_random_engine(random_stream_base + 2);
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};

        std::vector<glm::mat4> transforms;
        transforms.reserve(context.load);
        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto const x = static_cast<float>(i % side) * instance_spacing - extent * 0.5F;
            auto const row = i / side;
            auto const z = static_cast<float>(row) * instance_spacing - extent * 0.5F;
            auto const scale = 0.5F + 0.5F * unit(engine);
            auto const lift = unit(engine) * 2.0F;
            auto const yaw = unit(engine) * 2.0F * std::numbers::pi_v<float>;

            transforms.push_back(Components::Transform{
                    .position = glm::vec3{x, scale * 0.5F + lift, z},
                    .rotation = glm::angleAxis(yaw, glm::vec3{0.0F, 1.0F, 0.0F}),
                    .scale = glm::vec3{scale},
            }
                                         .matrix());
        }

        auto const material = materials.make(glm::vec4{0.75F, 0.55F, 0.3F, 1.0F}, AlphaMode::opaque, 0.6F,
                                             cast_shadows ? shadow_cascade_count - 1 : GpuMaterial::no_shadow_cascade);

        auto const field = GeneratedEntity{&context.scene, "bench_instances"};
        field.emplace<Components::InstancedModel>(Components::InstancedModel{
                .model = context.engine_models.cube,
                .material_override = materials.keep(material, context.owned_materials),
                .transforms = std::move(transforms),
        });
    }

    auto populate_instancing(BenchmarkScenarioContext const &context) -> void { populate_instances(context, true); }

    auto populate_instancing_no_shadows(BenchmarkScenarioContext const &context) -> void {
        populate_instances(context, false);
    }

    [[nodiscard]] auto instancing_path(std::uint32_t load) -> std::vector<CameraKeyframe> {
        auto const extent = static_cast<float>(grid_side(load)) * instance_spacing;
        return orbit_path(glm::vec3{0.0F}, extent * 0.55F + 8.0F, 5.0F, extent * 0.3F + 8.0F, 8);
    }

    // A big field of shadow casters, one entity each, with a number of shadow-casting point and spot lights scattered
    // through its middle. The casters are entities rather than an InstancedModel on purpose: an InstancedModel is a
    // resident GPU-LOD set whose instances are picked by camera distance, so its lights are always redrawn and
    // caching has nothing to show. The three variants differ only in the local shadow optimisations: all of them on,
    // culling without caching, and neither (every tile draws every caster, every frame).
    constexpr std::uint32_t shadow_field_objects = 8000;
    constexpr float shadow_light_area = 60.0F;

    auto populate_shadow_lights(BenchmarkScenarioContext const &context, bool culling, bool caching,
                                bool spheres = false) -> void {
        MaterialSet materials{context.renderer};

        auto const side = grid_side(shadow_field_objects);
        auto const extent = static_cast<float>(side) * instance_spacing;
        spawn_ground(context, materials, extent + 20.0F);

        auto const box_material = materials.make(glm::vec4{0.75F, 0.55F, 0.3F, 1.0F}, AlphaMode::opaque, 0.6F);

        auto field_engine = make_random_engine(random_stream_base + 2);
        std::uniform_real_distribution<float> field_unit{0.0F, 1.0F};

        for (std::uint32_t i = 0; i < shadow_field_objects; ++i) {
            auto const x = static_cast<float>(i % side) * instance_spacing - extent * 0.5F;
            auto const z = static_cast<float>(i / side) * instance_spacing - extent * 0.5F;
            auto const scale = 0.5F + 0.5F * field_unit(field_engine);
            auto const lift = field_unit(field_engine) * 2.0F;
            auto const yaw = field_unit(field_engine) * 2.0F * std::numbers::pi_v<float>;

            spawn(context.scene, std::format("bench_shadow_caster_{}", i),
                  spheres ? context.engine_models.sphere : context.engine_models.cube,
                  Components::Transform{
                          .position = glm::vec3{x, scale * 0.5F + lift, z},
                          .rotation = glm::angleAxis(yaw, glm::vec3{0.0F, 1.0F, 0.0F}),
                          .scale = glm::vec3{scale},
                  },
                  box_material);
        }

        auto settings = context.renderer.local_shadow_settings();
        settings.maximum_lights = std::max(context.load, 1U);
        settings.gpu_culling = culling;
        settings.caching = caching;
        context.renderer.set_local_shadow_settings(settings);

        auto engine = make_random_engine(random_stream_base + 4);
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};

        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto light = GeneratedEntity{&context.scene, "bench_shadow_light_{}", i};
            auto const position = glm::vec3{(unit(engine) - 0.5F) * shadow_light_area, 3.0F + unit(engine) * 6.0F,
                                            (unit(engine) - 0.5F) * shadow_light_area};
            auto const colour = glm::vec3{0.4F + 0.6F * unit(engine), 0.4F + 0.6F * unit(engine),
                                          0.4F + 0.6F * unit(engine)};

            if (i % 3 == 2) {
                light.emplace<Components::Transform>(Components::Transform{.position = position});
                light.emplace<Components::PointLight>(Components::PointLight{
                        .colour = colour, .intensity = 120.0F, .range = 14.0F, .casts_shadows = true});
            } else {
                // Straight down, so the cones overlap the casters below.
                light.emplace<Components::Transform>(Components::Transform{
                        .position = position,
                        .rotation = glm::angleAxis(unit(engine) * 0.4F, glm::vec3{1.0F, 0.0F, 0.0F})});
                light.emplace<Components::SpotLight>(Components::SpotLight{
                        .colour = colour,
                        .intensity = 400.0F,
                        .range = 18.0F,
                        .inner_cone_degrees = 25.0F,
                        .outer_cone_degrees = 45.0F,
                        .casts_shadows = true,
                });
            }
        }
    }

    [[nodiscard]] auto shadow_lights_path(std::uint32_t) -> std::vector<CameraKeyframe> {
        return orbit_path(glm::vec3{0.0F, 1.0F, 0.0F}, 42.0F, 6.0F, 24.0F, 8);
    }

    [[nodiscard]] auto shadow_city_path(std::uint32_t) -> std::vector<CameraKeyframe> {
        return orbit_path(glm::vec3{0.0F, 0.0F, 0.0F}, 60.0F, 12.0F, 28.0F, 8);
    }

    auto populate_shadow_city_scenario(BenchmarkScenarioContext const &context, bool culling, bool caching) -> void {
        auto const points = context.load / 3;
        auto const placed = populate_shadow_city(
                context.scene, context.renderer, context.engine_models,
                ShadowCityOptions{.spot_lights = context.load - points,
                                  .point_lights = points,
                                  .gpu_culling = culling,
                                  .caching = caching,
                                  .owned_materials = &context.owned_materials});
        if (placed == 0) {
            warn("[Benchmark] shadow_city placed no casters: run --game=gallery once to fetch the models");
        }
    }

    constexpr float grass_spacing = 0.5F;

    auto populate_grass(BenchmarkScenarioContext const &context) -> void {
        MaterialSet materials{context.renderer};

        auto const side = grid_side(context.load);
        auto const extent = static_cast<float>(side) * grass_spacing;
        spawn_ground(context, materials, extent + 20.0F);

        auto &images = context.renderer.image_storage();
        auto &samplers = context.renderer.sampler_storage();

        auto const grass = grass_materials(context.renderer, context.engine_models,
                                           MaterialCreateInfo{
                                                   .base_colour_factor = glm::vec4{0.25F, 0.55F, 0.18F, 1.0F},
                                                   .base_colour_texture = images.white(),
                                                   .normal_texture = images.flat_normal(),
                                                   .metallic_roughness_texture = images.metallic_roughness(),
                                                   .occlusion_texture = images.occlusion(),
                                                   .emissive_texture = images.emissive(),
                                                   .sampler = samplers.linear_repeat(),
                                                   .wind_strength = 0.28F,
                                                   .max_shadow_cascade = GpuMaterial::no_shadow_cascade,
                                           },
                                           {}, {});

        if (!grass) {
            error("benchmark scenario: could not create the grass materials: {}", describe(grass.error()));
            return;
        }

        context.owned_materials.push_back(grass->blades);
        context.owned_materials.push_back(grass->cards);

        auto engine = make_random_engine(random_stream_base + 5);
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};

        std::vector<glm::mat4> transforms;
        transforms.reserve(context.load);
        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto const jitter_x = (unit(engine) - 0.5F) * grass_spacing * 0.8F;
            auto const jitter_z = (unit(engine) - 0.5F) * grass_spacing * 0.8F;
            auto const x = (static_cast<float>(i % side) + 0.5F) * grass_spacing - extent * 0.5F + jitter_x;
            auto const z = (static_cast<float>(i / side) + 0.5F) * grass_spacing - extent * 0.5F + jitter_z;
            auto const yaw = unit(engine) * 2.0F * std::numbers::pi_v<float>;

            transforms.push_back(Components::Transform{
                    .position = glm::vec3{x, 0.0F, z},
                    .rotation = glm::angleAxis(yaw, glm::vec3{0.0F, 1.0F, 0.0F}),
                    .scale = glm::vec3{0.85F + 0.3F * unit(engine)},
            }
                                         .matrix());
        }

        auto const field = GeneratedEntity{&context.scene, "bench_grass"};
        field.emplace<Components::InstancedModel>(Components::InstancedModel{
                .model = context.engine_models.grass_clump,
                .material_override = grass->blades,
                .transforms = std::move(transforms),
        });
    }

    [[nodiscard]] auto grass_path(std::uint32_t load) -> std::vector<CameraKeyframe> {
        auto const extent = static_cast<float>(grid_side(load)) * grass_spacing;
        return orbit_path(glm::vec3{0.0F}, std::max(extent * 0.3F, 6.0F), 1.7F, 5.0F, 8);
    }

    constexpr float light_field_size = 160.0F;

    auto populate_lights(BenchmarkScenarioContext const &context) -> void {
        MaterialSet materials{context.renderer};
        spawn_ground(context, materials, light_field_size + 20.0F);

        auto const box_material = materials.make(glm::vec4{0.7F, 0.7F, 0.72F, 1.0F}, AlphaMode::opaque, 0.5F);
        constexpr std::uint32_t boxes_per_side = 20;
        constexpr auto box_spacing = light_field_size / static_cast<float>(boxes_per_side);

        for (std::uint32_t i = 0; i < boxes_per_side * boxes_per_side; ++i) {
            auto const x = (static_cast<float>(i % boxes_per_side) + 0.5F) * box_spacing - light_field_size * 0.5F;
            auto const row = i / boxes_per_side;
            auto const z = (static_cast<float>(row) + 0.5F) * box_spacing - light_field_size * 0.5F;
            auto const height = 1.0F + static_cast<float>(i % 3);
            spawn(context.scene, std::format("bench_box_{}", i), context.engine_models.cube,
                  Components::Transform{.position = glm::vec3{x, height * 0.5F, z},
                                        .scale = glm::vec3{2.0F, height, 2.0F}},
                  box_material);
        }

        auto engine = make_random_engine(random_stream_base + 3);
        std::uniform_real_distribution<float> unit{0.0F, 1.0F};

        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto light = GeneratedEntity{&context.scene, "bench_light_{}", i};
            light.emplace<Components::Transform>(Components::Transform{
                    .position = glm::vec3{(unit(engine) - 0.5F) * light_field_size, 0.5F + unit(engine) * 4.0F,
                                          (unit(engine) - 0.5F) * light_field_size},
            });
            light.emplace<Components::PointLight>(Components::PointLight{
                    .colour = glm::vec3{0.3F + 0.7F * unit(engine), 0.3F + 0.7F * unit(engine),
                                        0.3F + 0.7F * unit(engine)},
                    .intensity = 4.0F + 8.0F * unit(engine),
                    .range = 6.0F + 6.0F * unit(engine),
            });
        }
    }

    [[nodiscard]] auto lights_path(std::uint32_t ) -> std::vector<CameraKeyframe> {
        return orbit_path(glm::vec3{0.0F}, light_field_size * 0.35F, 3.0F, 30.0F, 8);
    }

    auto populate_overdraw(BenchmarkScenarioContext const &context) -> void {
        MaterialSet materials{context.renderer};
        spawn_ground(context, materials, 80.0F);

        spawn(context.scene, "bench_backdrop", context.engine_models.cube,
              Components::Transform{.position = glm::vec3{0.0F, 10.0F, -40.0F},
                                    .scale = glm::vec3{120.0F, 40.0F, 1.0F}},
              materials.make(glm::vec4{0.3F, 0.35F, 0.4F, 1.0F}));

        auto const layer_material = materials.make(glm::vec4{0.9F, 0.6F, 0.4F, 0.08F}, AlphaMode::blend, 0.4F);

        for (std::uint32_t i = 0; i < context.load; ++i) {
            auto const depth = 3.0F + 24.0F * static_cast<float>(i) / static_cast<float>(std::max(context.load, 1U));
            auto const size = 4.0F + depth * 2.0F;
            spawn(context.scene, std::format("bench_layer_{}", i), context.engine_models.cube,
                  Components::Transform{.position = glm::vec3{0.0F, 2.0F, -depth},
                                        .scale = glm::vec3{size * 1.8F, size, 0.02F}},
                  layer_material);
        }
    }

    [[nodiscard]] auto overdraw_path(std::uint32_t ) -> std::vector<CameraKeyframe> {
        return {
                {.position = {0.0F, 2.0F, 0.5F}, .target = {0.0F, 2.0F, -30.0F}},
                {.position = {0.6F, 2.3F, 0.5F}, .target = {0.4F, 2.1F, -30.0F}},
                {.position = {0.0F, 2.6F, 0.5F}, .target = {0.0F, 2.2F, -30.0F}},
                {.position = {-0.6F, 2.3F, 0.5F}, .target = {-0.4F, 2.1F, -30.0F}},
        };
    }

}

auto builtin_benchmark_scenarios(bool game_has_benchmark_path) -> std::vector<BenchmarkScenario> {
    std::vector<BenchmarkScenario> scenarios;

    if (game_has_benchmark_path) {
        scenarios.push_back(BenchmarkScenario{
                .info = {.name = "game"},
                .description = "The game's own scene and camera path: the realistic mix.",
                .source = BenchmarkSceneSource::game,
        });
        scenarios.push_back(BenchmarkScenario{
                .info = {.name = "game_resolution",
                         .load_axis = "render_scale_percent",
                         .default_loads = {50, 100, 150}},
                .description = "The game scene at several render resolutions. Cost that follows resolution is GPU "
                               "pixel work; cost that doesn't is CPU or per-draw.",
                .source = BenchmarkSceneSource::game,
                .load_target = BenchmarkLoadTarget::render_scale,
        });
    }

    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "draw_calls", .load_axis = "objects", .default_loads = {1000, 4000, 16000}},
            .description = "One entity per object (cubes and spheres, 16 materials): per-object CPU submission, "
                           "culling and draw count.",
            .populate = populate_draw_calls,
            .camera_path = draw_calls_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "instancing", .load_axis = "instances", .default_loads = {10000, 50000, 200000}},
            .description = "One instanced model (cubes) with many instances: GPU culling and per-instance cost, "
                           "shadows included, without per-object CPU cost.",
            .populate = populate_instancing,
            .camera_path = instancing_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "instancing_no_shadows",
                     .load_axis = "instances",
                     .default_loads = {10000, 50000, 200000}},
            .description = "The instancing field with shadow casting off: subtract from instancing for the shadow "
                           "passes' share.",
            .populate = populate_instancing_no_shadows,
            .camera_path = instancing_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_lights", .load_axis = "shadowed_lights", .default_loads = {4, 16, 32}},
            .description = "A field of 8000 shadow-casting objects with shadow-casting point and spot lights in its "
                           "middle, GPU culling and caching on: the shipped configuration of local light shadows.",
            .populate = [](BenchmarkScenarioContext const &context) { populate_shadow_lights(context, true, true); },
            .camera_path = shadow_lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_lights_cull_only",
                     .load_axis = "shadowed_lights",
                     .default_loads = {4, 16, 32}},
            .description = "shadow_lights with caching off: every shadowed light redraws each frame, but through the GPU "
                           "culling. Compare with shadow_lights for the cache, with shadow_lights_baseline for culling.",
            .populate = [](BenchmarkScenarioContext const &context) { populate_shadow_lights(context, true, false); },
            .camera_path = shadow_lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_lights_spheres", .load_axis = "shadowed_lights", .default_loads = {4, 16, 32}},
            .description = "shadow_lights with spheres as casters: far more triangles per shadow tile, so redrawing a "
                           "tile costs real GPU time and caching has something to save.",
            .populate = [](BenchmarkScenarioContext const &context) {
                populate_shadow_lights(context, true, true, true);
            },
            .camera_path = shadow_lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_lights_spheres_cull_only",
                     .load_axis = "shadowed_lights",
                     .default_loads = {4, 16, 32}},
            .description = "shadow_lights_spheres with caching off, for the cost caching saves on heavy casters.",
            .populate = [](BenchmarkScenarioContext const &context) {
                populate_shadow_lights(context, true, false, true);
            },
            .camera_path = shadow_lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_lights_baseline",
                     .load_axis = "shadowed_lights",
                     .default_loads = {4, 16, 32}},
            .description = "shadow_lights with culling and caching both off: every tile draws every caster, every "
                           "frame. The cost the other two scenarios remove.",
            .populate = [](BenchmarkScenarioContext const &context) { populate_shadow_lights(context, false, false); },
            .camera_path = shadow_lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_city", .load_axis = "shadowed_lights", .default_loads = {8, 24, 32}},
            .description = "A thousand real meshes (twenty models, fifty copies each, a few million triangles per tile "
                           "redraw) under shadow-casting spot and point lights, GPU culling and caching on. Needs the "
                           "gallery downloads (run --game=gallery once).",
            .populate = [](BenchmarkScenarioContext const &context) {
                populate_shadow_city_scenario(context, true, true);
            },
            .camera_path = shadow_city_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_city_cull_only", .load_axis = "shadowed_lights", .default_loads = {8, 24, 32}},
            .description = "shadow_city with caching off: the cost caching saves on heavy casters.",
            .populate = [](BenchmarkScenarioContext const &context) {
                populate_shadow_city_scenario(context, true, false);
            },
            .camera_path = shadow_city_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "shadow_city_baseline", .load_axis = "shadowed_lights", .default_loads = {8, 24, 32}},
            .description = "shadow_city with culling and caching both off.",
            .populate = [](BenchmarkScenarioContext const &context) {
                populate_shadow_city_scenario(context, false, false);
            },
            .camera_path = shadow_city_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "grass", .load_axis = "clumps", .default_loads = {20000, 60000, 180000}},
            .description = "A field of engine grass clumps, as in the game: blade geometry near the camera, "
                           "alpha-to-coverage cards far away, wind on, no shadows.",
            .populate = populate_grass,
            .camera_path = grass_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "lights", .load_axis = "point_lights", .default_loads = {64, 512, 4096}},
            .description = "A fixed field of 400 boxes lit by many point lights: light culling, clustering and "
                           "shading cost per light.",
            .populate = populate_lights,
            .camera_path = lights_path,
    });
    scenarios.push_back(BenchmarkScenario{
            .info = {.name = "overdraw", .load_axis = "layers", .default_loads = {4, 16, 64}},
            .description = "Full-screen alpha-blended layers in front of the camera: fill rate and blending.",
            .populate = populate_overdraw,
            .camera_path = overdraw_path,
    });

    return scenarios;
}

auto scenario_infos(std::vector<BenchmarkScenario> const &scenarios) -> std::vector<BenchmarkScenarioInfo> {
    std::vector<BenchmarkScenarioInfo> infos;
    infos.reserve(scenarios.size());
    for (auto const &scenario: scenarios) {
        infos.push_back(scenario.info);
    }
    return infos;
}
