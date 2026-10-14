#include "meshlet_gallery_game.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <format>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace {

    constexpr std::string_view sample_assets_commit = "5bad5aaa0bbb5d0f9cdc934e626f27d0df1e79b8";

    struct Download {
        std::string_view name;
        std::string_view sha256;
    };

    // Models/<Name>/glTF-Binary/<Name>.glb in KhronosGroup/glTF-Sample-Assets; licences vary per model (see its
    // README), so they are fetched on demand and never committed.
    constexpr std::array<Download, 17> downloads{{
            {"AnisotropyBarnLamp", "b961a243fde26ec8e3e5b339f09222097a1fe84cbf6ce7cd988ec2d2491fd66f"},
            {"AntiqueCamera", "8540140ee5bccccc9455d5ee712f532c43c2932b32d65d0bb609404af5a81555"},
            {"Avocado", "85ae91e60ed519f53a7e2096508837fedd446dcc450f8e05678b7a204679bd12"},
            {"BarramundiFish", "2355533a51a7499366044b065133324b0300c73a24c4ebeda2e48fc75b4c8665"},
            {"BoomBox", "84c32a396684d0b329bbe60428f5cb89c5f447356ccf359b0374856653eca331"},
            {"CesiumMilkTruck", "2e7600185bbcfe771f0a69a82ebc70d214d75380f31d079891548538f8f5aa3a"},
            {"Corset", "6b4270364f03352db03818c0659f637a52005b9d99576ca973f7cc30bcafec9e"},
            {"DragonAttenuation", "dbac78de3fefb33ba744cfa703936892b6e2fc47b58d88acc52f4be39e5464be"},
            {"Duck", "65bf938f54d6073e619e76e007820bbf980cdc3dc0daec0d94830ffc4ae54ab5"},
            {"GlamVelvetSofa", "67202c74a1a33377771f162dc7fad612a6c9bd51ee15124c488e9851d9ac5266"},
            {"IridescenceLamp", "267c128a3def801c340cea4aa79109fc32aa9c1872f4bdcacd1045f03f209297"},
            {"Lantern", "2d7ea69e1f4b7f5b9f00854978b20f41b9effa392272c50f4c98b5a8db3152b2"},
            {"MaterialsVariantsShoe", "e1d7cb190382111e5a5b37b51e9a7f007f7eb2ab1b6185e0188e8d0a0d1265a7"},
            {"MosquitoInAmber", "1c0b49000650e8a8c00d69bc8d64e46d33527871a47d74e4f2278887325cc35f"},
            {"SheenChair", "438b0c98b8d08a1ac79d0b75af7bd445e2e71a4743a9d62d60350e299a43f1e4"},
            {"ToyCar", "9631414db26edc9a4e6dcca7bdedac03c3aa6ba7926b35c6a6125e7d235a5605"},
            {"WaterBottle", "074341e0c9b991244ea88fed77221754dfa33b53ee13523fa5f3c10a28a30160"},
    }};

    constexpr int columns = 5;
    constexpr float spacing_x = 3.4F;
    constexpr float spacing_y = 3.3F;
    constexpr float fitted_size = 2.2F;
    constexpr float blend_rate = 3.0F;
    constexpr float parked_y = -500.0F;
    constexpr float floor_slab_y = -0.5F;

    auto download_options() -> HttpClientOptions {
        HttpClientOptions options;
        options.worker_count = 8;
        options.timeout = std::chrono::seconds{120};
        return options;
    }

}

MeshletGalleryGame::MeshletGalleryGame() : http_(download_options()) {}

auto MeshletGalleryGame::on_populate(Scene &scene, Renderer &renderer, EngineModels const &engine_models) -> void {
    renderer_ = &renderer;
    engine_models_ = engine_models;
    slots_.clear();
    downloads_total_ = 0;
    downloads_done_ = 0;

    scene.environment.source = EnvironmentSource::flat_ambient;
    scene.environment.draw_skybox = false;
    scene.environment.ambient_intensity = 0.35F;
    scene.environment.sun.azimuth_degrees = 145.0F;
    scene.environment.sun.elevation_degrees = 32.0F;
    scene.environment.sun.intensity = 4.0F;

    lights_.clear();

    constexpr std::array<glm::vec3, 3> rim_colours{glm::vec3{0.1F, 0.8F, 1.0F}, glm::vec3{1.0F, 0.2F, 0.7F},
                                                   glm::vec3{1.0F, 0.75F, 0.3F}};
    constexpr std::array<glm::vec3, 3> rim_positions{glm::vec3{-12.0F, 5.0F, 6.0F}, glm::vec3{12.0F, 5.0F, 6.0F},
                                                     glm::vec3{0.0F, 9.0F, 12.0F}};
    for (std::size_t i = 0; i < rim_colours.size(); ++i) {
        auto const light = Entity{&scene, "rim_light_{}", i};
        light.emplace<Components::Transform>(Components::Transform{.position = rim_positions[i]});
        light.emplace<Components::PointLight>(
                Components::PointLight{.colour = rim_colours[i], .intensity = 250.0F, .range = 50.0F});
        lights_.push_back(Light{.entity = light, .intensity = 250.0F});
    }

    // Finale lights: coloured point lights drifting round the floor and spot lights from above. They start dark.
    constexpr std::array<glm::vec3, 4> finale_colours{glm::vec3{1.0F, 0.25F, 0.2F}, glm::vec3{0.2F, 0.6F, 1.0F},
                                                      glm::vec3{0.3F, 1.0F, 0.4F}, glm::vec3{1.0F, 0.8F, 0.25F}};
    for (std::size_t i = 0; i < finale_colours.size(); ++i) {
        auto const light = Entity{&scene, "finale_point_{}", i};
        light.emplace<Components::Transform>(Components::Transform{});
        light.emplace<Components::PointLight>(
                Components::PointLight{.colour = finale_colours[i], .intensity = 0.0F, .range = 12.0F});
        lights_.push_back(Light{.entity = light,
                                .intensity = 70.0F,
                                .finale = true,
                                .orbit_radius = 7.0F + (2.0F * static_cast<float>(i % 2)),
                                .orbit_height = 1.6F + (0.7F * static_cast<float>(i)),
                                .orbit_phase = static_cast<float>(i) * 1.5708F,
                                .orbit_speed = (i % 2 == 0 ? 0.35F : -0.28F)});
    }

    constexpr std::array<glm::vec3, 3> spot_positions{glm::vec3{-6.5F, 11.0F, -2.0F}, glm::vec3{0.0F, 12.0F, 2.0F},
                                                      glm::vec3{6.5F, 11.0F, -2.0F}};
    constexpr std::array<glm::vec3, 3> spot_colours{glm::vec3{1.0F, 0.85F, 0.7F}, glm::vec3{1.0F, 1.0F, 1.0F},
                                                    glm::vec3{0.7F, 0.85F, 1.0F}};
    for (std::size_t i = 0; i < spot_positions.size(); ++i) {
        auto const light = Entity{&scene, "finale_spot_{}", i};
        auto const aim = glm::normalize(glm::vec3{0.0F, 1.2F, 0.0F} - spot_positions[i]);
        light.emplace<Components::Transform>(Components::Transform{
                .position = spot_positions[i],
                .rotation = glm::rotation(glm::vec3{0.0F, -1.0F, 0.0F}, aim),
        });
        light.emplace<Components::SpotLight>(Components::SpotLight{.colour = spot_colours[i],
                                                                   .intensity = 0.0F,
                                                                   .range = 30.0F,
                                                                   .inner_cone_degrees = 14.0F,
                                                                   .outer_cone_degrees = 28.0F});
        lights_.push_back(Light{.entity = light, .intensity = 900.0F, .finale = true});
    }

    // The finale floor: a dark, glossy slab whose material draws the debug grid. Parked out of sight until it's wanted.
    {
        auto &images = renderer.image_storage();
        auto &samplers = renderer.sampler_storage();

        auto const floor_material = renderer.create_material(
                MaterialCreateInfo{.base_colour_factor = glm::vec4{0.045F, 0.05F, 0.06F, 1.0F},
                                   .metallic_factor = 0.0F,
                                   .roughness_factor = 0.3F,
                                   .base_colour_texture = images.white(),
                                   .normal_texture = images.flat_normal(),
                                   .metallic_roughness_texture = images.metallic_roughness(),
                                   .occlusion_texture = images.occlusion(),
                                   .emissive_texture = images.emissive(),
                                   .sampler = samplers.linear_repeat(),
                                   .grid = true},
                "gallery_floor");

        auto const floor = Entity{&scene, "gallery_floor"};
        floor.emplace<Components::Transform>(Components::Transform{
                .position = glm::vec3{0.0F, parked_y, 0.0F}, .scale = glm::vec3{12000.0F, 1.0F, 12000.0F}});
        floor.emplace<Components::Model>(Components::Model{.model = engine_models.cube});
        if (floor_material) {
            floor.emplace<Components::MaterialOverride>(Components::MaterialOverride{.material = *floor_material});
        } else {
            error("[Gallery] Could not create the floor material: {}", describe(floor_material.error()));
        }

    }

    // Local models first, then the downloads, in grid order (left to right, top to bottom).
    struct LocalModel {
        std::string_view name;
        std::string_view file;
        float pitch;
        float yaw;
    };
    constexpr std::array<LocalModel, 3> locals{{
            {"DamagedHelmet", "assets/models/damaged_helmet.glb", 90.0F, 0.0F},
            {"ScatteringSkull", "assets/models/scattering_skull.glb", 0.0F, -150.0F},
            {"ChessKnight", "assets/models/chess/white_knight.glb", 0.0F, 0.0F},
    }};

    for (auto const &local: locals) {
        slots_.push_back(Slot{.name = std::string{local.name},
                              .path = data_path(local.file).absolute(),
                              .pitch_degrees = local.pitch,
                              .yaw_degrees = local.yaw});
    }

    for (auto const &item: downloads) {
        auto const destination = cache_path(std::format("downloads/gallery/{}.glb", item.name)).absolute();

        std::error_code directory_error;
        std::filesystem::create_directories(destination.parent_path(), directory_error);

        // Plain glTF is right-handed with +Z as its front; the loader mirrors it into the engine's left-handed space, which
        // sends the front to -Z, so turn it round to face the camera at +Z.
        slots_.push_back(Slot{.name = std::string{item.name}, .path = destination, .yaw_degrees = 180.0F});
        slots_.back().download = http_.get_file_async({
                .uri = std::format("https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{}/Models/{}/"
                                   "glTF-Binary/{}.glb",
                                   sample_assets_commit, item.name, item.name),
                .destination = destination,
                .sha256 = std::string{item.sha256},
        });
        ++downloads_total_;
    }

    for (std::size_t i = 0; i < slots_.size(); ++i) {
        auto const column = static_cast<int>(i) % columns;
        auto const row = static_cast<int>(i) / columns;
        slots_[i].position = glm::vec3{(static_cast<float>(columns - 1) * 0.5F - static_cast<float>(column)) * spacing_x,
                                       (1.5F - static_cast<float>(row)) * spacing_y, 0.0F};
    }

    info("[Gallery] Fetching {} models concurrently", downloads_total_);

    // The local models are ready now.
    for (auto &slot: slots_) {
        if (!slot.download.valid()) {
            place(scene, slot);
        }
    }
}

auto MeshletGalleryGame::place(Scene &scene, Slot &slot) -> void {
    auto const asset = AssetPath::external(slot.path);
    auto loaded = asset ? renderer_->load_model(*asset) : std::unexpected(RendererError{});

    if (!loaded) {
        error("[Gallery] Could not load '{}': {}", slot.name, describe(loaded.error()));
        slot.failed = true;
        return;
    }

    auto const model = *loaded;
    slot.loaded = true;

    float scale = 1.0F;
    glm::vec3 centre{0.0F};
    if (auto const bounds = renderer_->model_submesh_bounds(model); bounds && !bounds->empty()) {
        auto low = bounds->front().first;
        auto high = bounds->front().second;
        for (auto const &[lo, hi]: *bounds) {
            low = glm::min(low, lo);
            high = glm::max(high, hi);
        }
        auto const extent = high - low;
        scale = fitted_size / std::max({extent.x, extent.y, extent.z, 1e-4F});
        centre = (low + high) * 0.5F;
    }

    auto const rotation = glm::angleAxis(glm::radians(slot.yaw_degrees), glm::vec3{0.0F, 1.0F, 0.0F}) *
                          glm::angleAxis(glm::radians(slot.pitch_degrees), glm::vec3{1.0F, 0.0F, 0.0F});

    auto const sources = renderer_->model_materials(model);
    Components::MaterialOverride override_materials;

    for (auto const source: sources) {
        auto copy = renderer_->duplicate_material(source, std::format("{}_gallery_{}", slot.name, slot.variants.size()));
        if (!copy) {
            continue;
        }
        slot.variants.push_back(*copy);
        override_materials.slots.push_back(MaterialSlotOverride{.source = source, .material = *copy});
    }

    slot.scale = scale;
    slot.centre = centre;
    slot.rotation = rotation;

    // How far the lowest point sits below the model's centre once rotated and scaled, for standing on the floor.
    slot.below_centre = 0.5F * fitted_size;
    if (auto const bounds = renderer_->model_submesh_bounds(model); bounds && !bounds->empty()) {
        float lowest = 0.0F;
        for (auto const &[lo, hi]: *bounds) {
            for (int corner = 0; corner < 8; ++corner) {
                auto const point = glm::vec3{(corner & 1) != 0 ? hi.x : lo.x, (corner & 2) != 0 ? hi.y : lo.y,
                                             (corner & 4) != 0 ? hi.z : lo.z};
                lowest = std::min(lowest, (rotation * ((point - centre) * scale)).y);
            }
        }
        slot.below_centre = -lowest;
    }

    auto const entity = Entity{&scene, "{}", slot.name};
    entity.emplace<Components::Transform>(Components::Transform{
            .position = slot.position - rotation * (centre * scale),
            .rotation = rotation,
            .scale = glm::vec3{scale},
    });
    entity.emplace<Components::Model>(Components::Model{.model = model});
    entity.emplace<Components::MaterialOverride>(std::move(override_materials));
    entity.emplace<Components::DebugMeshletCones>(Components::DebugMeshletCones{.enabled = cones_});
    slot.entity = entity;

    // The meshlet count decides whether the mesh/task shader path (and so the extrude) applies to it.
    auto const meshlets = renderer_->model_meshlet_cones(model).size();
    info("[Gallery] {}: {} meshlets{}", slot.name, meshlets, meshlets < 32 ? " (below 32: drawn the classic way)" : "");

    applied_extrude_ = -1.0F;
    applied_shrink_ = -1.0F;
    layout(scene);
}

auto MeshletGalleryGame::floor_position(std::size_t index) const -> glm::vec3 {
    auto const column = static_cast<int>(index) % columns;
    auto const row = static_cast<int>(index) / columns;
    return glm::vec3{(static_cast<float>(column) - 2.0F) * 4.4F, 0.0F, (static_cast<float>(row) - 1.5F) * 4.4F};
}

auto MeshletGalleryGame::layout(Scene &scene) -> void {
    auto &registry = scene.get_registry();

    for (std::size_t i = 0; i < slots_.size(); ++i) {
        auto const &slot = slots_[i];
        if (!slot.loaded || !registry.valid(slot.entity)) {
            continue;
        }

        auto const target = finale_ ? floor_position(i) + glm::vec3{0.0F, slot.below_centre, 0.0F} : slot.position;
        registry.get<Components::Transform>(slot.entity).position = target - slot.rotation * (slot.centre * slot.scale);
    }
}

auto MeshletGalleryGame::set_finale(Scene &scene, bool finale) -> void {
    finale_ = finale;
    auto &registry = scene.get_registry();

    if (finale) {
        scene.environment.source = EnvironmentSource::hdr_image;
        scene.environment.hdr_source =
                data_path("assets/environments/belfast_sunset_puresky_512.ktx2").absolute().string();
        scene.environment.draw_skybox = true;
        scene.environment.ambient_intensity = 0.15F;
        scene.environment.sun.azimuth_degrees = 235.0F;
        scene.environment.sun.elevation_degrees = 12.0F;
        scene.environment.sun.intensity = 2.5F;
        scene.environment.exposure_ev = -0.8F;
        // Fade the floor into the sky at the horizon instead of ending in a visible edge.
        scene.environment.fog.enabled = true;
        // Matches the sky just above the horizon in this cubemap; sampling the environment for the fog colour gave a
        // visibly lighter band on the floor side of the horizon.
        scene.environment.fog.from_environment = false;
        scene.environment.fog.colour = glm::vec3{0.13F, 0.135F, 0.2F};
        scene.environment.fog_sky = false;
        scene.environment.fog.extinction = 0.012F;
    } else {
        scene.environment.exposure_ev = 0.0F;
        scene.environment.fog.enabled = false;
        scene.environment.fog_sky = false;
        scene.environment.source = EnvironmentSource::flat_ambient;
        scene.environment.draw_skybox = false;
        scene.environment.ambient_intensity = 0.35F;
        scene.environment.sun.azimuth_degrees = 145.0F;
        scene.environment.sun.elevation_degrees = 32.0F;
        scene.environment.sun.intensity = 4.0F;
    }

    if (registry.valid(floor_)) {
        registry.get<Components::Transform>(floor_).position =
                glm::vec3{0.0F, finale ? floor_slab_y - 0.5F : parked_y, 0.0F};
    }

    for (auto const &light: lights_) {
        if (!registry.valid(light.entity)) {
            continue;
        }

        auto const on = light.finale == finale;
        if (auto *point = registry.try_get<Components::PointLight>(light.entity)) {
            point->intensity = on ? light.intensity : 0.0F;
        }
        if (auto *spot = registry.try_get<Components::SpotLight>(light.entity)) {
            spot->intensity = on ? light.intensity : 0.0F;
        }
    }

    layout(scene);
}

auto MeshletGalleryGame::apply_materials(bool colours, float extrude, float shrink) -> void {
    for (auto const &slot: slots_) {
        for (auto const handle: slot.variants) {
            if (auto const *current = renderer_->material_storage().create_info(handle)) {
                auto next = *current;
                next.debug_meshlet_colours = colours;
                next.meshlet_extrude = extrude;
                next.meshlet_shrink = shrink;
                static_cast<void>(renderer_->update_material(handle, next));
            }
        }
    }
}

auto MeshletGalleryGame::set_look(Look look) -> void {
    look_ = look;
    colours_ = look != Look::plain;
    cones_ = look == Look::cones;

    switch (look) {
        case Look::plain:
        case Look::colours:
            target_extrude_ = 0.0F;
            target_shrink_ = 0.0F;
            break;
        case Look::exploded:
        case Look::cones:
            target_extrude_ = 0.8F;
            target_shrink_ = 0.12F;
            break;
        case Look::shattered:
            target_extrude_ = 2.4F;
            target_shrink_ = 0.3F;
            break;
    }
}

auto MeshletGalleryGame::on_update(Scene &scene, float delta_time) -> void {
    if (bound_scene_ != &scene) {
        // The play-mode scene is a clone of the one built in on_populate: find everything again by name.
        bound_scene_ = &scene;
        floor_ = scene.find_entity("gallery_floor");

        for (auto &slot: slots_) {
            if (slot.loaded) {
                slot.entity = scene.find_entity(slot.name);
            }
        }

        std::size_t rim = 0;
        std::size_t point = 0;
        std::size_t spot = 0;
        for (auto &light: lights_) {
            if (!light.finale) {
                light.entity = scene.find_entity(std::format("rim_light_{}", rim++));
            } else if (light.orbit_radius > 0.0F) {
                light.entity = scene.find_entity(std::format("finale_point_{}", point++));
            } else {
                light.entity = scene.find_entity(std::format("finale_spot_{}", spot++));
            }
        }
    }

    finale_time_ += delta_time;
    if (finale_) {
        auto &finale_registry = scene.get_registry();
        for (auto const &light: lights_) {
            if (light.finale && light.orbit_radius > 0.0F && finale_registry.valid(light.entity)) {
                auto const angle = light.orbit_phase + (light.orbit_speed * finale_time_);
                finale_registry.get<Components::Transform>(light.entity).position =
                        glm::vec3{std::cos(angle) * light.orbit_radius, light.orbit_height,
                                  std::sin(angle) * light.orbit_radius * 0.6F};
            }
        }
    }

    for (auto &slot: slots_) {
        if (!slot.download.valid() ||
            slot.download.wait_for(std::chrono::seconds{0}) != std::future_status::ready) {
            continue;
        }

        auto const result = slot.download.get();
        ++downloads_done_;

        if (!result) {
            warn("[Gallery] Could not fetch {}: {}", slot.name, describe(result.error()));
            slot.failed = true;
            continue;
        }

        info("[Gallery] Downloaded {} ({}/{})", slot.name, downloads_done_, downloads_total_);
        place(scene, slot);
    }

    auto const blend = 1.0F - std::exp(-blend_rate * delta_time);
    extrude_ += (target_extrude_ - extrude_) * blend;
    shrink_ += (target_shrink_ - shrink_) * blend;

    if (std::abs(extrude_ - applied_extrude_) > 1e-4F || std::abs(shrink_ - applied_shrink_) > 1e-4F ||
        colours_ != applied_colours_) {
        apply_materials(colours_, extrude_, shrink_);
        applied_extrude_ = extrude_;
        applied_shrink_ = shrink_;
        applied_colours_ = colours_;
    }

    auto &registry = scene.get_registry();
    for (auto const &slot: slots_) {
        if (auto *cones = registry.valid(slot.entity) ? registry.try_get<Components::DebugMeshletCones>(slot.entity)
                                                      : nullptr) {
            cones->enabled = cones_;
            cones->extrude = extrude_;
            cones->shrink = shrink_;
        }
    }
}

auto MeshletGalleryGame::on_key_pressed(Scene &scene, KeyPressedEvent const &event) -> void {
    switch (event.key) {
        case GLFW_KEY_1: set_look(Look::plain); break;
        case GLFW_KEY_2: set_look(Look::colours); break;
        case GLFW_KEY_3: set_look(Look::exploded); break;
        case GLFW_KEY_4: set_look(Look::cones); break;
        case GLFW_KEY_5: set_look(Look::shattered); break;
        case GLFW_KEY_6: set_finale(scene, !finale_); break;
        default: break;
    }
}

auto MeshletGalleryGame::camera(Scene const &, float aspect_ratio) const -> CameraParams {
    if (finale_) {
        constexpr float finale_fov = 40.0F;
        auto const angle = glm::radians(-18.0F + (5.0F * finale_time_));
        auto const eye = glm::vec3{std::sin(angle) * 19.0F, 11.5F, std::cos(angle) * 19.0F};
        return CameraParams{
                .view = glm::lookAtLH(eye, glm::vec3{0.0F, 1.4F, 0.0F}, glm::vec3{0.0F, 1.0F, 0.0F}),
                .projection = glm::perspectiveLH_ZO(glm::radians(finale_fov), aspect_ratio, 0.1F, 6000.0F),
                .near_clip = 0.1F,
                .far_clip = 6000.0F,
                .vertical_fov_radians = glm::radians(finale_fov),
        };
    }

    constexpr float fov_degrees = 26.0F;
    constexpr float distance = 31.0F;
    return CameraParams{
            .view = glm::lookAtLH(glm::vec3{0.0F, 0.0F, distance}, glm::vec3{0.0F}, glm::vec3{0.0F, 1.0F, 0.0F}),
            .projection = glm::perspectiveLH_ZO(glm::radians(fov_degrees), aspect_ratio, 0.1F, 200.0F),
            .near_clip = 0.1F,
            .far_clip = 200.0F,
            .vertical_fov_radians = glm::radians(fov_degrees),
    };
}
