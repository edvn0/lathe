#include "app/application.hxx"
#include "core/paths.hxx"

#include <csignal>
#include <memory>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <limits>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/vec2.hpp>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <entt/entt.hpp>

#include "assets/model_formats.hxx"
#include "core/allocator.hxx"
#include "core/config.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "gpu/context.hxx"
#include "implot.h"
#include "imgui_internal.h"
#include "app/pass_timeline.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/environment_panel.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/imgui_widget.hxx"
#include "rendering/toast.hxx"
#include "scene/components.hxx"
#include "scene/editor_camera.hxx"
#include "scene/selection_context.hxx"
#if LATHE_TRACK_MEMORY
#include "core/memory_tracking_ui.hxx"
#endif
#include "assets/file_change_watcher.hxx"
#include "gpu/renderdoc.hxx"
#include "gpu/swapchain.hxx"
#include "physics/physics.hxx"
#include "physics/physics_world.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/screenshot.hxx"
#include "app/asset_picker.hxx"
#include "app/surface_pick.hxx"
#include "app/text_placement.hxx"

namespace {

    // The models the scene uses, for the pickers to list first.
    auto models_in_use(entt::registry &registry) -> std::vector<ModelHandle> {
        auto used = std::vector<ModelHandle>{};
        for (auto const &&[entity, model]: registry.view<Components::Model>().each()) {
            if (model.model.valid() && std::ranges::find(used, model.model) == used.end()) {
                used.push_back(model.model);
            }
        }
        return used;
    }


    [[nodiscard]] auto entity_display_name(entt::registry const &registry, entt::entity entity) -> std::string {
        char const *raw = nullptr;
        if (auto const *generated = registry.try_get<Components::GeneratedMeta>(entity)) {
            raw = generated->name.c_str();
        } else if (auto const *meta = registry.try_get<Components::Meta>(entity)) {
            raw = meta->name.c_str();
        }
        if (raw != nullptr && raw[0] != '\0') {
            return raw;
        }
        return std::format("Entity {}", entt::to_integral(entity));
    }

    constexpr auto draw_point_light = [](Components::PointLight &point_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &point_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &point_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &point_light.range, 0.5F, 100.0F);
        changed |= ImGui::Checkbox("Casts shadows", &point_light.casts_shadows);
        return changed;
    };

    constexpr auto draw_spot_light = [](Components::SpotLight &spot_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &spot_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &spot_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &spot_light.range, 0.5F, 100.0F);
        changed |= ImGui::SliderFloat("Inner cone", &spot_light.inner_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        changed |= ImGui::SliderFloat("Outer cone", &spot_light.outer_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        changed |= ImGui::Checkbox("Casts shadows", &spot_light.casts_shadows);
        return changed;
    };

    constexpr auto draw_decal = [](Components::Decal &decal) -> bool {
        bool changed = false;
        changed |= ImGui::SliderFloat("Opacity", &decal.opacity, 0.0F, 1.0F);
        changed |= ImGui::SliderFloat("Full strength to", &decal.angle_fade_start_degrees, 0.0F, 90.0F, "%.0f deg");
        changed |= ImGui::SliderFloat("Fades out by", &decal.angle_fade_end_degrees, 0.0F, 90.0F, "%.0f deg");
        changed |= ImGui::DragInt("Sort order", &decal.sort_order);
        return changed;
    };

    constexpr auto draw_particle_emitter = [](Components::ParticleEmitter &emitter) -> bool {
        bool changed = false;

        auto count = static_cast<int>(emitter.count);
        if (ImGui::DragInt("Count", &count, 16.0F, 1, static_cast<int>(Components::ParticleEmitter::max_count))) {
            emitter.count = static_cast<std::uint32_t>(count);
            changed = true;
        }
        changed |= ImGui::Checkbox("Emitting", &emitter.emitting);
        changed |= ImGui::DragFloat("Rate", &emitter.rate, 1.0F, 0.0F, 1.0e6F, "%.0f / s");
        changed |= ImGui::DragFloat("Lifetime", &emitter.lifetime, 0.05F, 0.01F, 3600.0F, "%.2f s");
        changed |= ImGui::DragFloat3("Gravity", &emitter.gravity.x, 0.1F);

        constexpr std::array<char const *, 5> shapes{"Point", "Sphere", "Box", "Cone", "Text outline"};
        auto shape = static_cast<int>(emitter.shape);
        if (ImGui::Combo("Shape", &shape, shapes.data(), static_cast<int>(shapes.size()))) {
            emitter.shape = static_cast<Components::ParticleShape>(shape);
            changed = true;
        }
        changed |= ImGui::DragFloat("Shape size", &emitter.shape_size, 0.01F, 0.0F, 1.0e4F);
        changed |= ImGui::SliderFloat("Cone", &emitter.cone_degrees, 0.0F, 180.0F, "%.1f deg");
        changed |= ImGui::DragFloat("Speed", &emitter.speed, 0.05F, -1.0e4F, 1.0e4F);
        changed |= ImGui::SliderFloat("Speed variance", &emitter.speed_variance, 0.0F, 1.0F);
        changed |= ImGui::DragFloat("Size start", &emitter.size_start, 0.005F, 0.0F, 1.0e3F);
        changed |= ImGui::DragFloat("Size end", &emitter.size_end, 0.005F, 0.0F, 1.0e3F);
        changed |= ImGui::ColorEdit4("Colour start", &emitter.colour_start.x);
        changed |= ImGui::ColorEdit4("Colour end", &emitter.colour_end.x);
        return changed;
    };

    constexpr auto draw_water = [](Components::Water &water) -> bool {
        bool changed = false;

        constexpr std::array<char const *, 2> bodies{"Ocean", "Lake / pool"};
        auto body = static_cast<int>(water.body);
        if (ImGui::Combo("Body", &body, bodies.data(), static_cast<int>(bodies.size()))) {
            water.body = static_cast<Components::WaterBody>(body);
            changed = true;
        }
        changed |= ImGui::Checkbox("Visible", &water.visible);
        changed |= ImGui::DragFloat2(water.body == Components::WaterBody::ocean ? "Water zone" : "Extent", &water.extent.x,
                                     0.1F, 0.1F, 1.0e5F, "%.1f m");
        changed |= ImGui::ColorEdit3("Deep colour", &water.colour.x);
        changed |= ImGui::SliderFloat("Roughness", &water.roughness, 0.0F, 1.0F);
        changed |= ImGui::DragFloat3("Absorption", &water.absorption.x, 0.005F, 0.0F, 10.0F, "%.3f / m");
        changed |= ImGui::DragFloat("Shore fade", &water.shore_fade, 0.01F, 0.0F, 10.0F, "%.2f m");
        changed |= ImGui::DragFloat("Wave scale", &water.wave_scale, 0.005F, 0.001F, 10.0F, "%.3f / m");
        changed |= ImGui::DragFloat("Wave speed", &water.wave_speed, 0.01F, 0.0F, 10.0F);
        changed |= ImGui::DragFloat("Wave height", &water.wave_height, 0.005F, 0.0F, 20.0F, "%.3f m");
        changed |= ImGui::DragFloat("Normal strength", &water.normal_strength, 0.01F, 0.0F, 10.0F);
        changed |= ImGui::DragFloat("Refraction", &water.refraction_strength, 0.001F, 0.0F, 0.5F);
        changed |= ImGui::SliderFloat("Foam", &water.foam, 0.0F, 2.0F);
        changed |= ImGui::SliderFloat("Whitecaps", &water.whitecaps, 0.0F, 2.0F);
        changed |= ImGui::SliderFloat("Caustics", &water.caustics, 0.0F, 2.0F);

        if (ImGui::Button("Calm pool")) {
            water = Components::Water::pool(water.body == Components::WaterBody::lake ? water.extent
                                                                                      : glm::vec2{10.0F, 10.0F});
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Rough ocean")) {
            water = Components::Water::ocean();
            changed = true;
        }
        return changed;
    };

    constexpr auto draw_planet = [](Components::Planet &planet) -> bool {
        bool changed = false;
        auto radius_km = static_cast<float>(planet.radius / 1000.0);
        if (ImGui::DragFloat("Radius", &radius_km, 1.0F, 1.0F, 1.0e6F, "%.0f km")) {
            planet.radius = static_cast<double>(radius_km) * 1000.0;
            changed = true;
        }
        changed |= ImGui::Checkbox("Visible", &planet.visible);
        changed |= ImGui::SliderFloat("Exaggeration", &planet.exaggeration, 0.0F, 20.0F, "%.1f x");
        changed |= ImGui::DragFloat("Sea level", &planet.sea_level, 1.0F, -10000.0F, 10000.0F, "%.0f m");
        auto level = static_cast<int>(planet.max_level);
        if (ImGui::SliderInt("Max level", &level, 0, 15)) {
            planet.max_level = static_cast<std::uint8_t>(level);
            changed = true;
        }
        constexpr std::array<char const *, 2> colours{"Image", "From elevation"};
        auto colour = static_cast<int>(planet.colour);
        if (ImGui::Combo("Colour", &colour, colours.data(), static_cast<int>(colours.size()))) {
            planet.colour = static_cast<Components::PlanetColour>(colour);
            changed = true;
        }
        changed |= ImGui::Checkbox("Ocean", &planet.ocean);
        changed |= ImGui::Checkbox("Atmosphere", &planet.atmosphere);
        auto atmosphere_km = planet.atmosphere_height / 1000.0F;
        if (ImGui::DragFloat("Atmosphere height", &atmosphere_km, 0.5F, 1.0F, 1000.0F, "%.0f km")) {
            planet.atmosphere_height = atmosphere_km * 1000.0F;
            changed = true;
        }
        changed |= ImGui::SliderFloat("Atmosphere intensity", &planet.atmosphere_intensity, 0.0F, 5.0F);
        changed |= ImGui::Checkbox("LOD colours", &planet.debug_lod);
        ImGui::TextDisabled("Elevation: %s", planet.elevation_path.empty() ? "(none)" : planet.elevation_path.c_str());
        return changed;
    };

    constexpr auto draw_text = [](Components::Text &text) -> bool {
        bool changed = false;

        // A std::string edited in place: ImGui asks for a bigger buffer through the callback as the text grows.
        changed |= ImGui::InputTextMultiline(
                "Text", text.text.data(), text.text.capacity() + 1, ImVec2{-FLT_MIN, 3.0F * ImGui::GetTextLineHeightWithSpacing()},
                ImGuiInputTextFlags_CallbackResize,
                [](ImGuiInputTextCallbackData *data) -> int {
                    if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
                        auto *const value = static_cast<std::string *>(data->UserData);
                        value->resize(static_cast<std::size_t>(data->BufTextLen));
                        data->Buf = value->data();
                    }
                    return 0;
                },
                &text.text);

        auto font = static_cast<int>(text.font);
        if (ImGui::DragInt("Font", &font, 0.1F, 0, 1000)) {
            text.font = static_cast<std::uint32_t>(std::max(font, 0));
            changed = true;
        }
        changed |= ImGui::DragFloat("Size", &text.size, 0.01F, 0.001F, 1.0e4F, "%.3f m");
        changed |= ImGui::ColorEdit4("Colour", &text.colour.x, ImGuiColorEditFlags_Float);
        changed |= ImGui::DragFloat("Emissive", &text.emissive, 0.05F, 0.0F, 1.0e3F, "%.2f x");
        changed |= ImGui::Checkbox("Centred", &text.centred);
        changed |= ImGui::Checkbox("Visible", &text.visible);
        return changed;
    };

    constexpr auto draw_transform = [](Components::Transform &transform) -> bool {
        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &transform.position.x, 0.1F);

        auto euler_degrees = glm::degrees(glm::eulerAngles(transform.rotation));
        if (ImGui::DragFloat3("Rotation", &euler_degrees.x, 0.5F, 0.0F, 0.0F, "%.1f deg")) {
            transform.rotation = glm::quat(glm::radians(euler_degrees));
            changed = true;
        }

        changed |= ImGui::DragFloat3("Scale", &transform.scale.x, 0.01F, 0.001F, 1000.0F);
        return changed;
    };

    constexpr auto draw_lifetime = [](Components::Lifetime &lifetime) -> bool {
        return ImGui::DragFloat("Remaining seconds", &lifetime.remaining_seconds, 0.05F, 0.0F, 3600.0F);
    };

    constexpr auto draw_rigid_body = [](Components::RigidBody &body) -> bool {
        bool changed = false;

        bool const generated_shape =
                body.shape == Components::BodyShape::heightfield || body.shape == Components::BodyShape::compound;

        if (generated_shape) {
            ImGui::TextDisabled("Shape: %s (generated, not editable)",
                                body.shape == Components::BodyShape::heightfield ? "Heightfield" : "Compound");
        } else {
            constexpr std::array selectable_shapes{Components::BodyShape::box, Components::BodyShape::capsule,
                                                   Components::BodyShape::sphere};
            constexpr std::array<char const *, 3> shape_names{"Box", "Capsule", "Sphere"};

            auto const selected = std::ranges::find(selectable_shapes, body.shape);
            int shape_index =
                    selected == selectable_shapes.end() ? 0 : static_cast<int>(selected - selectable_shapes.begin());

            if (ImGui::Combo("Shape", &shape_index, shape_names.data(), static_cast<int>(shape_names.size()))) {
                body.shape = selectable_shapes[static_cast<std::size_t>(shape_index)];
                changed = true;
            }

            switch (body.shape) {
                case Components::BodyShape::capsule:
                    changed |= ImGui::DragFloat("Capsule radius", &body.capsule_radius, 0.05F, 0.01F, 1000.0F);
                    changed |= ImGui::DragFloat("Capsule height", &body.capsule_height, 0.05F, 0.01F, 1000.0F);
                    break;
                case Components::BodyShape::sphere:
                    changed |= ImGui::DragFloat("Sphere radius", &body.sphere_radius, 0.05F, 0.01F, 1000.0F);
                    break;
                default:
                    changed |= ImGui::DragFloat3("Half extents", &body.half_extents.x, 0.05F, 0.01F, 1000.0F);
                    break;
            }
        }

        changed |= ImGui::Checkbox("Static", &body.is_static);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Lock rotation", &body.lock_rotation);

        ImGui::BeginDisabled(body.is_static);
        changed |= ImGui::DragFloat("Mass", &body.mass, 0.1F, 0.01F, 10000.0F);
        ImGui::EndDisabled();

        changed |= ImGui::SliderFloat("Restitution", &body.restitution, 0.0F, 1.0F);
        changed |= ImGui::DragFloat3("Initial velocity", &body.velocity.x, 0.1F);

        return changed;
    };

    template<typename... Cs>
    auto copy_components(entt::registry &registry, entt::entity source, entt::entity dest) -> void {
        (
                [&] {
                    if constexpr (std::is_empty_v<Cs>) {
                        if (registry.all_of<Cs>(source)) {
                            registry.emplace<Cs>(dest);
                        }
                    } else if (auto const *component = registry.try_get<Cs>(source)) {
                        registry.emplace<Cs>(dest, *component);
                    }
                }(),
                ...);
    }

    template<std::size_t N>
    auto copy_to_buffer(std::array<char, N> &buffer, std::string_view text) -> void {
        auto const length = std::min(text.size(), N - 1);
        std::copy_n(text.begin(), length, buffer.begin());
        buffer[length] = '\0';
    }

    constexpr std::string_view bullets_group_label = "Bullets";

    [[nodiscard]] auto collect_hierarchy_nodes(entt::registry const &registry) -> std::vector<HierarchyModel::Node> {
        std::vector<HierarchyModel::Node> nodes;
        std::unordered_set<entt::entity> listed;

        auto const add = [&](entt::entity entity) {
            if (!listed.insert(entity).second) {
                return;
            }

            auto const *parent = registry.try_get<Components::Parent>(entity);
            nodes.push_back(HierarchyModel::Node{
                    .entity = entity,
                    .parent = parent != nullptr ? parent->entity : entt::entity{entt::null},
                    .name = entity_display_name(registry, entity),
                    .group = registry.all_of<Components::BulletTag>(entity) ? 0U : HierarchyModel::no_group,
            });
        };

        for (auto const entity: registry.view<Components::Transform const, Components::GeneratedMeta const>()) {
            add(entity);
        }
        for (auto const entity: registry.view<Components::Transform const, Components::Meta const>()) {
            add(entity);
        }

        return nodes;
    }

    auto rename_entity(entt::registry &registry, entt::entity entity, std::string_view name) -> void {
        auto const is_space = [](unsigned char c) { return std::isspace(c) != 0; };
        while (!name.empty() && is_space(static_cast<unsigned char>(name.front()))) {
            name.remove_prefix(1);
        }
        while (!name.empty() && is_space(static_cast<unsigned char>(name.back()))) {
            name.remove_suffix(1);
        }

        if (name.empty() || !registry.valid(entity)) {
            return;
        }

        if (registry.all_of<Components::GeneratedMeta>(entity)) {
            registry.patch<Components::GeneratedMeta>(
                    entity, [&](Components::GeneratedMeta &meta) { meta.name = std::string{name}; });
        } else {
            registry.emplace_or_replace<Components::Meta>(entity, Components::Meta{.name = FlyString{name}});
        }
    }

    [[nodiscard]] auto selection_roots(entt::registry const &registry, std::span<entt::entity const> entities)
            -> std::vector<entt::entity> {
        constexpr std::size_t max_parent_depth = 1024;

        std::vector<entt::entity> roots;
        roots.reserve(entities.size());

        for (auto const entity: entities) {
            if (!registry.valid(entity)) {
                continue;
            }

            bool has_selected_ancestor = false;
            std::size_t depth = 0;
            for (auto const *parent = registry.try_get<Components::Parent>(entity);
                 parent != nullptr && registry.valid(parent->entity) && depth++ < max_parent_depth;
                 parent = registry.try_get<Components::Parent>(parent->entity)) {
                if (std::ranges::find(entities, parent->entity) != entities.end()) {
                    has_selected_ancestor = true;
                    break;
                }
            }

            if (!has_selected_ancestor) {
                roots.push_back(entity);
            }
        }

        return roots;
    }

    auto set_transform_from_matrix(entt::registry &registry, entt::entity entity, glm::mat4 const &matrix) -> void {
        auto const translation = glm::vec3{matrix[3]};
        glm::vec3 const scale{glm::length(glm::vec3{matrix[0]}), glm::length(glm::vec3{matrix[1]}),
                              glm::length(glm::vec3{matrix[2]})};
        glm::mat3 const rotation_matrix{glm::vec3{matrix[0]} / scale.x, glm::vec3{matrix[1]} / scale.y,
                                        glm::vec3{matrix[2]} / scale.z};
        auto const rotation = glm::quat_cast(rotation_matrix);

        registry.patch<Components::Transform>(entity, [&](Components::Transform &transform) {
            transform.position = translation;
            transform.rotation = rotation;
            transform.scale = scale;
        });
    }

    [[nodiscard]] auto model_file_filters() -> std::vector<gui::FileBrowser::Filter> {
        return {
                {.label = "Models (*.gltf, *.glb, *.fbx, *.obj, *.dae, *.ply, *.stl, *.usd*)",
                 .extensions = {model_file_extensions.begin(), model_file_extensions.end()}},
                {.label = "All files", .extensions = {}},
        };
    }

    using gui::widget;
    using gui::panel;

    [[nodiscard]] auto material_deletion_label(std::string_view name) -> std::string {
        return std::format("material:{}", name);
    }

    [[nodiscard]] auto is_material_pending_deletion(std::span<Application::PendingDeletion const> pending_deletions,
                                                    std::string_view name) -> bool {
        auto const label = material_deletion_label(name);
        return std::ranges::any_of(pending_deletions,
                                   [&](Application::PendingDeletion const &pending) { return pending.label == label; });
    }
}

auto Application::add_pass_timings(std::span<frame_graph::PassTiming const> passes) -> void {
    timing_x += 1.0F;

    for (auto const &pass: passes) {
        if (std::ranges::none_of(timing_series,
                                 [&](TimingSeries const &series) { return series.id == pass.name_id; })) {
            auto buffer = timing_series.empty() ? ScrollingBuffer{600} : timing_series.back().buffer;
            timing_series.push_back(TimingSeries{.id = pass.name_id, .label = pass.label, .buffer = std::move(buffer)});
        }
    }

    auto running_total = 0.0F;
    for (auto &series: timing_series) {
        auto const found = std::ranges::find_if(
                passes, [&](frame_graph::PassTiming const &pass) { return pass.name_id == series.id; });
        auto const sample = found != passes.end() ? found->milliseconds.value_or(0.0F) : 0.0F;
        // Smooth each pass before stacking, so one slow frame nudges the graph instead of spiking it.
        series.smoothed = series.primed ? series.smoothed + (timing_smoothing * (sample - series.smoothed)) : sample;
        series.primed = true;
        running_total += series.smoothed;
        series.buffer.add_point(timing_x, running_total);
    }
}

Application::Application(VulkanContext &ctx) noexcept : Runtime(ctx) {
    event_subscriptions.push_back(event_bus.subscribe<KeyPressedEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    event_subscriptions.push_back(event_bus.subscribe<KeyReleasedEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    event_subscriptions.push_back(event_bus.subscribe<MouseMovedEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    event_subscriptions.push_back(event_bus.subscribe<MouseScrolledEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    event_subscriptions.push_back(event_bus.subscribe<MouseButtonPressedEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    event_subscriptions.push_back(event_bus.subscribe<MouseButtonReleasedEvent>([this](auto const &ev) { return handle_event(ev); }, 0));
    stage_registry.bind_debug_renderer(debug_renderer.get());
    // The editor ray-casts against models' triangles to pick and to place things on surfaces.
    renderer->model_streamer().collect_pick_geometry(true);
}

Application::~Application() = default;

auto Application::draw_event_metrics() -> void {
    if (ImGui::Button("Reset")) {
        event_bus.reset_metrics();
    }

    if (ImGui::BeginTable("Events", 7, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        for (auto const *column: {"Event", "Subs", "Published", "Consumed", "Calls", "Avg us", "Max us"}) {
            ImGui::TableSetupColumn(column);
        }
        ImGui::TableHeadersRow();

        for (auto const &m: event_bus.metrics()) {
            auto const avg_us = m.handler_calls == 0 ? 0.0 : static_cast<double>(m.total_ns) / 1000.0 / static_cast<double>(m.handler_calls);

            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(m.name.data(), m.name.data() + m.name.size());
            ImGui::TableNextColumn();
            ImGui::Text("%zu", m.subscribers);
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(m.published));
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(m.consumed));
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(m.handler_calls));
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", avg_us);
            ImGui::TableNextColumn();
            ImGui::Text("%.2f", static_cast<double>(m.max_ns) / 1000.0);
        }

        ImGui::EndTable();
    }
}

auto Application::dock_second_viewport(ImGuiID dockspace_id) -> void {
    auto const *const first = ImGui::FindWindowByName("Viewport");
    auto const *const second = ImGui::FindWindowByName("Viewport 2");

    auto const second_docked = second != nullptr && second->DockId != 0 && ImGui::DockBuilderGetNode(second->DockId);
    if (first == nullptr || first->DockId == 0 || second_docked) {
        return;
    }

    // The first viewport's node splits in two; the viewport keeps the left half and the new one takes the right.
    auto left = first->DockId;
    auto const right = ImGui::DockBuilderSplitNode(left, ImGuiDir_Right, 0.5F, nullptr, &left);

    ImGui::DockBuilderDockWindow("Viewport 2", right);
    for (auto const id: {left, right}) {
        if (auto *const node = ImGui::DockBuilderGetNode(id)) {
            node->LocalFlags |= ImGuiDockNodeFlags_HiddenTabBar;
        }
    }

    ImGui::DockBuilderFinish(dockspace_id);
}

auto Application::draw_viewport(std::size_t index, char const *title, std::uint32_t frame_index) -> void {
    auto &pane = viewports[index];
    pane.hovered = false;

    widget(title, [&] {
        auto &camera = pane.camera;

        pane.hovered = ImGui::IsWindowHovered();
        pane.screen_pos = ImGui::GetCursorScreenPos();
        pane.content_size = ImGui::GetContentRegionAvail();
        pane.seen_frame = ImGui::GetFrameCount();

        if (pane.hovered) {
            active_viewport = index;
        }

        if (index == 0) {
            viewport_screen_pos = pane.screen_pos;
            viewport_content_size = pane.content_size;
            viewport_seen_frame = pane.seen_frame;
        }

        auto const is_active = active_viewport == index;

        auto const target = renderer->view_viewport_target(pane.view, frame_index);
        bool const has_room = pane.content_size.x > 0.0F && pane.content_size.y > 0.0F;
        if (target.valid() && has_room) {
            ImGui::Image(gui::linear_source_texture_id(target.index), pane.content_size);
        }

        if (index != 0 && pane.view != Renderer::primary_view) {
            auto const extent = renderer->view_extent(pane.view);
            auto const aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
            renderer->set_view_camera(pane.view,
                                      Renderer::CameraMatrices{
                                              .view = camera.view(),
                                              .projection = camera.projection(aspect),
                                              .near_clip = camera.near_clip(),
                                              .far_clip = camera.far_clip(),
                                              .vertical_fov_radians = glm::radians(camera.field_of_view_degrees()),
                                              .aspect_ratio = aspect,
                                              .time = elapsed_time,
                                      });
        }

        if (has_room) {
            // The view mode of this viewport, over its top left corner.
            ImGui::SetCursorScreenPos(ImVec2{pane.screen_pos.x + 8.0F, pane.screen_pos.y + 8.0F});
            ImGui::SetNextItemWidth(140.0F);
            auto const current_mode = renderer->view_mode(pane.view);
            auto const current_name = std::ranges::find(view_mode_names, current_mode, &decltype(view_mode_names)::value_type::first)->second;
            if (ImGui::BeginCombo(std::format("##view_mode_{}", index).c_str(), std::string{current_name}.c_str())) {
                for (auto const &[mode, name]: view_mode_names) {
                    if (ImGui::Selectable(std::string{name}.c_str(), mode == current_mode)) {
                        renderer->set_view_mode(pane.view, mode);
                    }
                }
                ImGui::EndCombo();
            }

            if (index != 0) {
                ImGui::SameLine();
                if (ImGui::Button("Close")) {
                    second_viewport_open = false;
                }
            }
        }

        if (!is_playing && has_room) {
            auto &registry = active_scene()->get_registry();

            auto const selected_entity = selection_context().primary();

            // One gizmo at a time: in the viewport the pointer is in (or was last in).
            if (is_active && selected_entity != entt::null && registry.valid(selected_entity) &&
                registry.all_of<Components::Transform>(selected_entity)) {
                ImGuizmo::SetOrthographic(false);
                ImGuizmo::SetDrawlist();
                ImGuizmo::SetRect(pane.screen_pos.x, pane.screen_pos.y, pane.content_size.x, pane.content_size.y);

                auto const aspect = pane.content_size.y > 0.0F ? pane.content_size.x / pane.content_size.y : 1.0F;
                auto const flip_z = glm::scale(glm::mat4(1.0F), glm::vec3(1.0F, 1.0F, -1.0F));
                auto const view = flip_z * camera.view();
                auto const projection = camera.projection(aspect) * flip_z;

                auto const previous_matrix = registry.get<Components::Transform>(selected_entity).matrix();
                auto matrix = previous_matrix;

                if (ImGuizmo::Manipulate(glm::value_ptr(view), glm::value_ptr(projection), gizmo_operation, gizmo_mode,
                                         glm::value_ptr(matrix))) {
                    auto const delta = matrix * glm::inverse(previous_matrix);
                    auto const snapshot = selection_context().snapshot();

                    for (auto const entity: selection_roots(registry, snapshot.entities)) {
                        if (entity == selected_entity) {
                            set_transform_from_matrix(registry, entity, matrix);
                        } else if (registry.all_of<Components::Transform>(entity)) {
                            set_transform_from_matrix(registry, entity,
                                                      delta * registry.get<Components::Transform>(entity).matrix());
                        }
                    }

                    renderer->mark_dynamic_shadow_casters_dirty();
                }
            }

            // Click (not drag) in the viewport to pick; ctrl toggles, clicking empty space clears. With the text tool
            // active the mouse previews and places text on a surface instead. Only the viewport under the pointer.
            auto const pointer_on_scene = is_active && pane.hovered && !ImGui::IsAnyItemHovered() &&
                                          !ImGuizmo::IsOver() && !ImGuizmo::IsUsing();
            auto const clicked = pointer_on_scene && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
                                 ImGui::GetMouseDragDelta(ImGuiMouseButton_Left).x == 0.0F &&
                                 ImGui::GetMouseDragDelta(ImGuiMouseButton_Left).y == 0.0F;

            if (clicked || (pointer_on_scene && text_placement.active())) {
                auto const aspect = pane.content_size.x / pane.content_size.y;
                auto const mouse = ImGui::GetIO().MousePos;
                auto const ndc_x = (((mouse.x - pane.screen_pos.x) / pane.content_size.x) * 2.0F) - 1.0F;
                auto const ndc_y = 1.0F - (((mouse.y - pane.screen_pos.y) / pane.content_size.y) * 2.0F);

                // Camera-space ray through the pixel (LH, +z forward), recovered from the projection's focal terms.
                auto const projection = camera.projection(aspect);
                auto const camera_ray = glm::vec3{ndc_x / projection[0][0], ndc_y / projection[1][1], 1.0F};
                auto const to_world = glm::inverse(camera.view());
                auto const direction = glm::normalize(glm::vec3{to_world * glm::vec4{camera_ray, 0.0F}});
                auto &selection = selection_context();

                if (text_placement.active()) {
                    if (!ImGui::GetIO().WantTextInput && ImGui::IsKeyPressed(ImGuiKey_R, false) &&
                        !ImGui::GetIO().KeyCtrl) {
                        text_placement.step_roll(ImGui::GetIO().KeyShift);
                    }

                    auto const placed = text_placement.update(
                            *active_scene(), *renderer, *debug_renderer,
                            {.origin = camera.position(),
                             .direction = direction,
                             .camera_forward = glm::normalize(glm::vec3{to_world * glm::vec4{0.0F, 0.0F, 1.0F, 0.0F}}),
                             .projector = {.view = camera.view(),
                                           .focal = {projection[0][0], projection[1][1]},
                                           .origin = {pane.screen_pos.x, pane.screen_pos.y},
                                           .size = {pane.content_size.x, pane.content_size.y}},
                             .cursor = {mouse.x, mouse.y},
                             .click = clicked,
                             .snap_to_grid = ImGui::GetIO().KeyCtrl,
                             .bypass_snap = ImGui::GetIO().KeyAlt});

                    if (placed != entt::null) {
                        selection.select(placed);
                    }
                } else {
                    auto const hit = pick_entity(*active_scene(), *renderer, camera.position(), direction);

                    if (hit != entt::null) {
                        if (ImGui::GetIO().KeyCtrl) {
                            selection.toggle(hit);
                        } else {
                            selection.select(hit);
                        }
                    } else if (!ImGui::GetIO().KeyCtrl) {
                        selection.clear();
                    }
                }
            }
        }
    });
}

// Runs once a frame after the renderer's own resize: the second viewport's view exists while the viewport is split
// (and not playing, which stays single-view), and follows the size of its window.
auto Application::update_views() -> void {
    auto &pane = viewports[1];
    auto const wanted = second_viewport_open && !is_playing;

    if (!wanted) {
        if (pane.view != Renderer::primary_view) {
            renderer->destroy_view(pane.view);
            pane.view = Renderer::primary_view;
        }
        return;
    }

    if (pane.seen_frame < 0 || pane.content_size.x < 1.0F || pane.content_size.y < 1.0F) {
        return;
    }

    auto const scale = renderer->render_scale();
    auto const scaled = [scale](float value) {
        return static_cast<std::uint32_t>(std::clamp(std::round(value * scale), 1.0F, 16384.0F));
    };
    auto const target = VkExtent2D{.width = scaled(pane.content_size.x), .height = scaled(pane.content_size.y)};

    if (pane.view == Renderer::primary_view) {
        auto created = renderer->create_view(target);

        if (!created) {
            error("Could not create the second viewport: {}", describe(created.error()));
            second_viewport_open = false;
            return;
        }

        pane.view = *created;
        pane.pending_extent = target;
        pane.pending_since = std::chrono::steady_clock::now();

        // The debug camera starts where the editor camera is, and keeps its own pose from then on.
        if (!second_viewport_camera_seeded) {
            pane.camera = viewports[0].camera;
            second_viewport_camera_seeded = true;
        }
        return;
    }

    // A window being dragged changes size every frame; the view follows once it has held still for a moment.
    if (target.width != pane.pending_extent.width || target.height != pane.pending_extent.height) {
        pane.pending_extent = target;
        pane.pending_since = std::chrono::steady_clock::now();
    }

    constexpr auto settle_time = std::chrono::milliseconds{150};
    auto const current = renderer->view_extent(pane.view);
    if ((pane.pending_extent.width != current.width || pane.pending_extent.height != current.height) &&
        std::chrono::steady_clock::now() - pane.pending_since >= settle_time) {
        if (auto const resized = renderer->resize_view(pane.view, pane.pending_extent); !resized) {
            error("Could not resize the second viewport: {}", describe(resized.error()));
        }
    }
}

auto Application::on_ui(std::uint32_t frame_index) -> void {
    if (is_playing && play_fullscreen) {
        Runtime::on_ui(frame_index);
        return;
    }

    handle_window_resize();
    draw_main_menu_bar();

    auto const *main_viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(main_viewport->WorkPos);
    ImGui::SetNextWindowSize(main_viewport->WorkSize);
    ImGui::SetNextWindowViewport(main_viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("##dockspace_host", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                         ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                         ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground);
    ImGui::PopStyleVar(3);

    ImGuiID const dockspace_id = ImGui::GetID("MainDockSpace");

    if (reset_layout_requested) {
        ImGui::DockBuilderRemoveNode(dockspace_id);
        panels = PanelVisibility{};
        reset_layout_requested = false;
    }

    if (ImGui::DockBuilderGetNode(dockspace_id) == nullptr) {
        ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspace_id, main_viewport->WorkSize);

        ImGuiID center = dockspace_id;
        ImGuiID const left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.20F, nullptr, &center);
        ImGuiID const right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.25F, nullptr, &center);
        ImGuiID const bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.28F, nullptr, &center);

        ImGui::DockBuilderDockWindow("Viewport", center);
        if (auto *const node = ImGui::DockBuilderGetNode(center)) {
            node->LocalFlags |= ImGuiDockNodeFlags_HiddenTabBar;
        }
        ImGui::DockBuilderDockWindow("Hierarchy", left);
        ImGui::DockBuilderDockWindow("Inspector", right);
        ImGui::DockBuilderDockWindow("Console", bottom);

        ImGui::DockBuilderFinish(dockspace_id);
    }

    if (split_viewport_requested) {
        split_viewport_requested = false;
        dock_second_viewport(dockspace_id);
    }

    ImGui::DockSpace(dockspace_id, ImVec2(0.0F, 0.0F), ImGuiDockNodeFlags_PassthruCentralNode);
    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    draw_viewport(0, "Viewport", frame_index);
    auto const second_shown = second_viewport_open && !is_playing;
    if (second_shown) {
        draw_viewport(1, "Viewport 2", frame_index);
    }
    viewport_hovered = viewports[0].hovered || (second_shown && viewports[1].hovered);
    ImGui::PopStyleVar();

    if (game && game_hooks_enabled && panels.game) {
        game->on_ui(*active_scene(), *renderer);
    }
#if LATHE_TRACK_MEMORY
    panel("Memory", panels.memory, [] { on_memory_ui(); });
#endif
    panel("Events", panels.events, [&] { draw_event_metrics(); });
    panel("Console", panels.console, [&] { terminal_widget.draw(); });

    panel("Script", panels.script, [&] {
        auto const world = is_playing ? std::optional<ScriptWorld>{}
                                      : std::optional<ScriptWorld>{ScriptWorld{
                                                .registry = &source_scene->get_registry(),
                                                .hierarchy_revision = source_scene->hierarchy_revision(),
                                        }};
        if (script_widget.draw(world).transforms_written > 0) {
            renderer->mark_dynamic_shadow_casters_dirty();
        }
    });

    auto const draw_material_fields = [&](MaterialCreateInfo &info) -> bool {
        bool changed = false;

        changed |= ImGui::ColorEdit4("Base colour", &info.base_colour_factor.x);
        changed |= ImGui::ColorEdit3("Emissive", &info.emissive_factor.x);
        changed |= ImGui::DragFloat("Emissive strength", &info.emissive_strength, 0.05F, 0.0F, 100.0F);
        changed |= ImGui::SliderFloat("Metallic", &info.metallic_factor, 0.0F, 1.0F);
        changed |= ImGui::SliderFloat("Roughness", &info.roughness_factor, 0.0F, 1.0F);
        changed |= ImGui::DragFloat("Normal scale", &info.normal_scale, 0.01F, 0.0F, 4.0F);
        changed |= ImGui::SliderFloat("Occlusion strength", &info.occlusion_strength, 0.0F, 1.0F);
        changed |= ImGui::DragFloat("Wind strength", &info.wind_strength, 0.01F, 0.0F, 4.0F, "%.2f",
                                    ImGuiSliderFlags_AlwaysClamp);

        int alpha_mode_index = static_cast<int>(info.alpha_mode);
        constexpr std::array<char const *, 3> alpha_mode_names{"Opaque", "Mask", "Blend"};
        if (ImGui::Combo("Alpha mode", &alpha_mode_index, alpha_mode_names.data(),
                         static_cast<int>(alpha_mode_names.size()))) {
            info.alpha_mode = static_cast<AlphaMode>(alpha_mode_index);
            changed = true;
        }
        if (info.alpha_mode == AlphaMode::mask) {
            changed |= ImGui::SliderFloat("Alpha cutoff", &info.alpha_cutoff, 0.0F, 1.0F);
            changed |= ImGui::Checkbox("Alpha to coverage", &info.alpha_to_coverage);
        }

        changed |= ImGui::Checkbox("Double-sided", &info.double_sided);
        changed |= ImGui::Checkbox("Debug grid", &info.grid);

        if (info.far_material.valid()) {
            auto const far_name = renderer->assets().materials().name_of(info.far_material);
            ImGui::TextDisabled("From LOD %u: %s", info.far_material_lod,
                                far_name.empty() ? "(unnamed material)" : std::string{far_name}.c_str());
        }

        bool casts_shadows = info.max_shadow_cascade != GpuMaterial::no_shadow_cascade;
        if (ImGui::Checkbox("Casts shadows", &casts_shadows)) {
            info.max_shadow_cascade = casts_shadows ? shadow_cascade_count - 1 : GpuMaterial::no_shadow_cascade;
            changed = true;
        }

        changed |= ImGui::Checkbox("Debug meshlet colours", &info.debug_meshlet_colours);
        changed |= ImGui::SliderFloat("Meshlet extrude", &info.meshlet_extrude, 0.0F, 4.0F);
        changed |= ImGui::SliderFloat("Meshlet shrink", &info.meshlet_shrink, 0.0F, 0.9F);

        auto const texture_picker = [&](char const *label, ImageHandle &slot, ImageHandle default_handle) {
            auto const &textures = renderer->assets().textures();
            bool const is_default = slot == default_handle;
            auto const current_name = textures.name_of(slot);
            std::string const preview =
                    is_default ? "(default)" : (current_name.empty() ? "(unnamed)" : std::string(current_name));

            if (!ImGui::BeginCombo(label, preview.c_str())) {
                return;
            }

            if (ImGui::Selectable("(default)", is_default)) {
                slot = default_handle;
                changed = true;
            }
            for (auto const &entry: textures.entries()) {
                if (entry.name.empty()) {
                    continue; // ImGui rejects an item without a label
                }
                bool const is_selected = entry.handle == slot;
                if (ImGui::Selectable(entry.name.c_str(), is_selected)) {
                    slot = entry.handle;
                    changed = true;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        };

        auto &images = renderer->image_storage();
        texture_picker("Base colour tex", info.base_colour_texture, images.white());
        texture_picker("Normal tex", info.normal_texture, images.flat_normal());
        texture_picker("Metallic/roughness tex", info.metallic_roughness_texture, images.metallic_roughness());
        texture_picker("Occlusion tex", info.occlusion_texture, images.occlusion());
        texture_picker("Emissive tex", info.emissive_texture, images.emissive());

        return changed;
    };

    panel("Assets", panels.assets, [&] {
        auto &assets = renderer->assets();

        auto scan_directory = [](std::filesystem::path const &root, std::span<std::string_view const> extensions) {
            std::vector<std::filesystem::path> found;
            std::error_code ec;

            if (!std::filesystem::exists(root, ec)) {
                return found;
            }

            for (auto const &entry: std::filesystem::recursive_directory_iterator(root, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }

                auto extension = entry.path().extension().string();
                std::ranges::transform(extension, extension.begin(),
                                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                if (std::ranges::find(extensions, extension) != extensions.end()) {
                    found.push_back(entry.path());
                }
            }

            return found;
        };

        auto const draw_bullet_entry = [](auto const &entry) {
            ImGui::BulletText("%s%s", entry.name.c_str(), entry.handle.valid() ? "" : " (invalid)");
        };

        auto draw_file_backed_section = [&]<typename HandleT>(char const *label, std::filesystem::path const &root,
                                                              std::span<std::string_view const> extensions,
                                                              NamedAssetTable<HandleT> &table, auto &&load,
                                                              auto &&draw_entry) {
            if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                return;
            }

            for (auto const &entry: table.entries()) {
                draw_entry(entry);
            }

            for (auto const &path: scan_directory(root, extensions)) {
                auto const filename = path.filename().string();
                if (table.find(filename).valid()) {
                    continue;
                }

                ImGui::PushID(path.string().c_str());
                if (ImGui::Button("Load")) {
                    load(path, filename);
                }
                ImGui::PopID();
                ImGui::SameLine();
                ImGui::TextDisabled("%s (not loaded)", path.string().c_str());
            }
        };

        static constexpr auto const &model_extensions = model_file_extensions;
        draw_file_backed_section(
                "Models", data_path("assets/models").absolute(), model_extensions, assets.models(),
                [&](std::filesystem::path const &path, std::string const &name) {
                    static_cast<void>(
                            renderer->model_streamer().request(*renderer, AssetPath::from_user(path).value_or(AssetPath::missing()),
                                                      engine_models.cube, FlyString{name}));
                },
                draw_bullet_entry);

        auto const draw_texture_entry = [](auto const &entry) {
            if (!entry.handle.valid()) {
                ImGui::BulletText("%s (invalid)", entry.name.c_str());
                return;
            }

            float const thumb_size = ImGui::GetFontSize() * 2.0F;
            ImVec2 const row_start = ImGui::GetCursorPos();
            ImGui::Image(ImTextureID{entry.handle.index}, ImVec2(thumb_size, thumb_size));
            ImGui::SameLine();
            ImGui::SetCursorPosY(row_start.y + (thumb_size - ImGui::GetTextLineHeight()) * 0.5F);
            ImGui::TextUnformatted(entry.name.c_str());
        };

        static constexpr std::array<std::string_view, 3> texture_extensions{".png", ".jpg", ".jpeg"};
        draw_file_backed_section(
                "Textures", data_path("assets/textures").absolute(), texture_extensions, assets.textures(),
                [&](std::filesystem::path const &path, std::string const &name) {
                    static_cast<void>(renderer->request_texture(AssetPath::from_user(path).value_or(AssetPath::missing()), TextureRole::colour,
                                                                renderer->image_storage().white(), name));
                },
                draw_texture_entry);

        auto draw_named_only_section = [&](char const *label, auto &table) {
            if (!ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen)) {
                return;
            }

            if (table.entries().empty()) {
                ImGui::TextDisabled("Nothing registered yet.");
                return;
            }

            for (auto const &entry: table.entries()) {
                ImGui::BulletText("%s%s", entry.name.c_str(), entry.handle.valid() ? "" : " (invalid)");
            }
        };

        if (ImGui::CollapsingHeader("Materials", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("New Material")) {
                new_material_info = MaterialCreateInfo{
                        .base_colour_texture = renderer->image_storage().white(),
                        .normal_texture = renderer->image_storage().flat_normal(),
                        .metallic_roughness_texture = renderer->image_storage().metallic_roughness(),
                        .occlusion_texture = renderer->image_storage().occlusion(),
                        .emissive_texture = renderer->image_storage().emissive(),
                        .sampler = renderer->sampler_storage().linear_repeat(),
                };
                new_material_name.clear();
                ImGui::OpenPopup("new_material_popup");
            }

            if (ImGui::BeginPopup("new_material_popup")) {
                std::array<char, 64> name_buf{};
                auto const copy_len = std::min(new_material_name.size(), name_buf.size() - 1);
                std::copy_n(new_material_name.begin(), copy_len, name_buf.begin());
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 16.0F);
                if (ImGui::InputTextWithHint("Name", "material name", name_buf.data(), name_buf.size())) {
                    new_material_name.assign(name_buf.data());
                }

                ImGui::Separator();
                draw_material_fields(new_material_info);
                ImGui::Separator();

                bool const name_taken =
                        !new_material_name.empty() && assets.materials().find(new_material_name).valid();
                if (name_taken) {
                    ImGui::TextColored(ImVec4(0.95F, 0.45F, 0.35F, 1.0F), "That name is already registered.");
                }

                ImGui::BeginDisabled(new_material_name.empty() || name_taken);
                if (ImGui::Button("Create")) {
                    auto const created = renderer->create_material(new_material_info, new_material_name);
                    if (!created) {
                        warn("Assets panel: failed to create material '{}': {}", new_material_name,
                             describe(created.error()));
                    }
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            auto &materials = assets.materials();
            bool any_shown = false;

            for (auto const &entry: materials.entries()) {
                if (is_material_pending_deletion(pending_deletions, entry.name)) {
                    continue;
                }
                any_shown = true;

                ImGui::PushID(entry.name.c_str());

                if (!entry.handle.valid()) {
                    ImGui::BulletText("%s (invalid)", entry.name.c_str());
                    ImGui::PopID();
                    continue;
                }

                bool const is_default = entry.handle == renderer->default_material();
                if (ImGui::TreeNode(entry.name.c_str())) {
                    if (auto const *info = renderer->material_storage().create_info(entry.handle)) {
                        auto edited = *info;
                        if (draw_material_fields(edited)) {
                            static_cast<void>(renderer->update_material(entry.handle, edited));
                        }
                    } else {
                        ImGui::TextDisabled("(material data unavailable)");
                    }

                    if (is_default) {
                        ImGui::TextDisabled("The default material can't be deleted.");
                    } else if (ImGui::Button("Delete")) {
                        pending_deletions.push_back(PendingDeletion{
                                .label = material_deletion_label(entry.name),
                                .delete_at = elapsed_time + deletion_grace_seconds,
                                .commit =
                                        [renderer = renderer.get(), handle = entry.handle, name = entry.name] {
                                            if (auto const result = renderer->destroy_material(handle); !result) {
                                                warn("Assets panel: failed to delete material '{}': {}", name,
                                                     describe(result.error()));
                                            }
                                        },
                        });
                    }

                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            if (!any_shown) {
                ImGui::TextDisabled("Nothing registered yet.");
            }

            static constexpr std::string_view material_prefix = "material:";
            bool has_material_deletions = std::ranges::any_of(pending_deletions, [&](PendingDeletion const &pending) {
                return pending.label.starts_with(material_prefix);
            });

            if (has_material_deletions && ImGui::TreeNode("Recently deleted")) {
                std::optional<std::size_t> restore_index;
                std::optional<std::size_t> commit_now_index;

                for (std::size_t index = 0; index < pending_deletions.size(); ++index) {
                    auto const &pending = pending_deletions[index];
                    if (!pending.label.starts_with(material_prefix)) {
                        continue;
                    }

                    ImGui::PushID(static_cast<int>(index));
                    auto const name = pending.label.substr(material_prefix.size());
                    auto const seconds_left = std::max(0.0F, pending.delete_at - elapsed_time);
                    ImGui::Text("%s -- deleting in %.0fs", name.c_str(), seconds_left);
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Restore")) {
                        restore_index = index;
                    }
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Delete now")) {
                        commit_now_index = index;
                    }
                    ImGui::PopID();
                }

                if (restore_index) {
                    pending_deletions.erase(pending_deletions.begin() + static_cast<std::ptrdiff_t>(*restore_index));
                } else if (commit_now_index) {
                    pending_deletions[*commit_now_index].commit();
                    pending_deletions.erase(pending_deletions.begin() + static_cast<std::ptrdiff_t>(*commit_now_index));
                }

                ImGui::TreePop();
            }
        }

        draw_named_only_section("Scripts", assets.scripts());
    });

    panel("Hierarchy", panels.hierarchy, [&] {
        auto const &style = ImGui::GetStyle();

        if (text_placement.active()) {
            ImGui::SeparatorText("Place text");
            {
                bool placing = true;
                if (ImGui::Checkbox("Place text on surfaces (T)", &placing)) {
                    text_placement.set_active(placing);
                }

                auto &settings = text_placement.settings;
                std::array<char, 256> buffer{};
                copy_to_buffer(buffer, settings.text);
                if (ImGui::InputText("Text##placement", buffer.data(), buffer.size())) {
                    settings.text = buffer.data();
                }
                ImGui::DragFloat("Size##placement", &settings.size, 0.01F, 0.05F, 10.0F);
                ImGui::SliderFloat("Roll", &settings.roll_degrees, -180.0F, 180.0F, "%.0f deg");
                ImGui::Checkbox("Align to nearest edge", &settings.align_to_edges);
                ImGui::Checkbox("Snap to vertices, edges and centres", &settings.snap_to_geometry);
                ImGui::DragFloat("Grid (Ctrl)", &settings.grid_step, 0.01F, 0.01F, 10.0F, "%.2f m");
                ImGui::DragFloat("Angle step", &settings.angle_step, 0.5F, 1.0F, 90.0F, "%.0f deg");
                if (settings.snap_to_geometry) {
                    ImGui::DragFloat("Snap radius (px)", &settings.snap_radius, 0.5F, 4.0F, 64.0F);
                }
                ImGui::TextDisabled("Click a model to place it. R turns it (Shift+R back). Ctrl snaps to the grid, Alt skips snapping. Esc to stop.");
            }
        }

        ImGui::SeparatorText("Entities");

        auto &registry = active_scene()->get_registry();

        {
            float const icon_sz = ImGui::GetTextLineHeight();
            float const frame_h = ImGui::GetFrameHeight();
            ImVec2 const row_start = ImGui::GetCursorPos();

            ImGui::SetCursorPos(ImVec2(row_start.x + style.FramePadding.x, row_start.y + (frame_h - icon_sz) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(gui::EditorIcon::search), ImVec2(icon_sz, icon_sz), ImVec2(0, 0),
                               ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(0.55F, 0.55F, 0.60F, 1.0F));

            ImGui::SetCursorPos(
                    ImVec2(row_start.x + icon_sz + style.FramePadding.x + style.ItemInnerSpacing.x, row_start.y));
            ImGui::SetNextItemWidth(-1.0F);

            std::array<char, 128> search_buf{};
            auto const copy_len = std::min(hierarchy_search.size(), search_buf.size() - 1);
            std::copy_n(hierarchy_search.begin(), copy_len, search_buf.begin());
            if (ImGui::InputTextWithHint("##hierarchy_search", "Search entities...", search_buf.data(),
                                         search_buf.size())) {
                hierarchy_search.assign(search_buf.data());
            }
        }

        struct EntityVisual {
            gui::EditorIcon icon;
            ImVec4 tint;
        };

        auto const visual_for = [&](entt::entity entity) -> EntityVisual {
            if (registry.all_of<Components::PointLight>(entity)) {
                return {.icon = gui::EditorIcon::point_light, .tint = ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::SpotLight>(entity)) {
                return {.icon = gui::EditorIcon::spot_light, .tint = ImVec4(1.00F, 0.84F, 0.35F, 1.0F)};
            }
            if (registry.all_of<Components::PlayerTag>(entity)) {
                return {.icon = gui::EditorIcon::player, .tint = ImVec4(0.47F, 0.86F, 0.55F, 1.0F)};
            }
            if (registry.all_of<Components::BulletTag>(entity)) {
                return {.icon = gui::EditorIcon::bullet, .tint = ImVec4(1.00F, 0.53F, 0.38F, 1.0F)};
            }
            if (registry.all_of<Components::Model>(entity) || registry.all_of<Components::InstancedModel>(entity)) {
                return {.icon = gui::EditorIcon::mesh, .tint = ImVec4(0.88F, 0.64F, 0.37F, 1.0F)};
            }
            if (registry.any_of<Components::Script, Components::Behaviour>(entity)) {
                return {.icon = gui::EditorIcon::script, .tint = ImVec4(0.42F, 0.70F, 1.00F, 1.0F)};
            }
            return {.icon = gui::EditorIcon::empty, .tint = ImVec4(0.60F, 0.60F, 0.64F, 1.0F)};
        };

        if (auto const *scene = active_scene();
            scene != hierarchy_model_scene || scene->hierarchy_revision() != hierarchy_model_revision) {
            hierarchy_model_scene = scene;
            hierarchy_model_revision = scene->hierarchy_revision();
            hierarchy_model.rebuild(collect_hierarchy_nodes(registry), {std::string{bullets_group_label}});
        }
        hierarchy_model.set_filter(hierarchy_search);

        enum class HierarchyAction : std::uint8_t { none, add_child, duplicate, remove, reparent };
        HierarchyAction pending_action = HierarchyAction::none;
        entt::entity action_target = entt::null;
        entt::entity drag_reparent_source = entt::null;

        auto &selection = selection_context();
        selection.retain_if([&](entt::entity entity) { return registry.valid(entity); });

        auto const rows = hierarchy_model.rows();
        entt::entity clicked_entity = entt::null;
        bool open_context_menu = false;

        auto const begin_rename = [&](entt::entity entity) {
            renaming_entity = entity;
            copy_to_buffer(rename_buffer, entity_display_name(registry, entity));
            rename_needs_focus = true;
        };

        float const row_height = ImGui::GetFrameHeight();
        float const icon_size = ImGui::GetTextLineHeight();

        auto const draw_alternate_background = [&](std::size_t row_index) {
            if (row_index % 2 == 1) {
                ImVec2 const row_screen_pos = ImGui::GetCursorScreenPos();
                ImGui::GetWindowDrawList()->AddRectFilled(
                        row_screen_pos,
                        ImVec2(row_screen_pos.x + ImGui::GetContentRegionAvail().x, row_screen_pos.y + row_height),
                        ImGui::GetColorU32(ImGuiCol_TableRowBgAlt));
            }
        };

        auto const draw_group_row = [&](HierarchyModel::Row const &row) {
            ImGui::PushID("hierarchy_group");
            ImGui::PushID(static_cast<int>(row.group));

            ImVec2 const row_pos = ImGui::GetCursorPos();

            ImGui::SetNextItemOpen(row.expanded, ImGuiCond_Always);
            bool const open = ImGui::TreeNodeEx(" ##group_node", ImGuiTreeNodeFlags_SpanAvailWidth |
                                                                         ImGuiTreeNodeFlags_FramePadding |
                                                                         ImGuiTreeNodeFlags_NoTreePushOnOpen);
            if (open != row.expanded) {
                hierarchy_model.set_group_expanded(row.group, open);
            }
            ImVec2 const after_row_pos = ImGui::GetCursorPos();

            float const label_x = row_pos.x + ImGui::GetTreeNodeToLabelSpacing();
            ImGui::SetCursorPos(ImVec2(label_x, row_pos.y + (row_height - icon_size) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(gui::EditorIcon::folder), ImVec2(icon_size, icon_size),
                               ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), ImVec4(0.95F, 0.80F, 0.45F, 1.0F));
            ImGui::SetCursorPos(ImVec2(label_x + icon_size + style.ItemInnerSpacing.x,
                                       row_pos.y + (row_height - ImGui::GetTextLineHeight()) * 0.5F));
            auto const label = hierarchy_model.group_label(row.group);
            ImGui::Text("%.*s (%zu)", static_cast<int>(label.size()), label.data(),
                        hierarchy_model.group_size(row.group));

            ImGui::SetCursorPos(after_row_pos);
            ImGui::PopID();
            ImGui::PopID();
        };

        auto const draw_entity_row = [&](HierarchyModel::Row const &row, std::size_t row_index) {
            auto const entity = row.entity;
            auto const name = entity_display_name(registry, entity);

            auto visual = visual_for(entity);
            if (row.has_children && visual.icon == gui::EditorIcon::empty) {
                visual = {.icon = gui::EditorIcon::folder, .tint = ImVec4(0.95F, 0.80F, 0.45F, 1.0F)};
            }

            ImGui::PushID(static_cast<int>(entity));

            ImVec2 const row_pos = ImGui::GetCursorPos();
            float const avail_width = ImGui::GetContentRegionAvail().x;

            bool const is_selected = selection.contains(entity);
            float label_x = row_pos.x + style.FramePadding.x;
            ImVec2 next_row_pos;

            ImGui::SetNextItemSelectionUserData(static_cast<ImGuiSelectionUserData>(row_index));

            if (row.has_children) {
                ImGuiTreeNodeFlags const flags = ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_OpenOnArrow |
                                                 ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                                 (is_selected ? ImGuiTreeNodeFlags_Selected : ImGuiTreeNodeFlags_None);
                ImGui::SetNextItemOpen(row.expanded, ImGuiCond_Always);
                bool const open = ImGui::TreeNodeEx(" ##node", flags);
                if (open != row.expanded) {
                    hierarchy_model.set_expanded(entity, open);
                }
                next_row_pos = ImGui::GetCursorPos();
                label_x = row_pos.x + ImGui::GetTreeNodeToLabelSpacing();
            } else {
                static_cast<void>(ImGui::Selectable("##row", is_selected, ImGuiSelectableFlags_None,
                                                    ImVec2(avail_width, row_height)));
                next_row_pos = ImGui::GetCursorPos();
            }

            if (ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
                clicked_entity = entity;
            }
            if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                begin_rename(entity);
            }

            if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                hierarchy_context_entity = entity;
                open_context_menu = true;
            }

            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_None)) {
                ImGui::SetDragDropPayload("HIERARCHY_ENTITY", &entity, sizeof(entity));
                if (auto const count = selection.size(); is_selected && count > 1) {
                    ImGui::Text("%zu entities", count);
                } else {
                    ImGui::TextUnformatted(name.c_str());
                }
                ImGui::EndDragDropSource();
            }

            if (ImGui::BeginDragDropTarget()) {
                if (auto const *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                    entt::entity dropped{};
                    std::memcpy(&dropped, payload->Data, sizeof(dropped));
                    pending_action = HierarchyAction::reparent;
                    action_target = entity;
                    drag_reparent_source = dropped;
                }
                ImGui::EndDragDropTarget();
            }

            ImGui::SetCursorPos(ImVec2(label_x, row_pos.y + (row_height - icon_size) * 0.5F));
            ImGui::ImageWithBg(editor_icons->texture(visual.icon), ImVec2(icon_size, icon_size), ImVec2(0, 0),
                               ImVec2(1, 1), ImVec4(0, 0, 0, 0), visual.tint);

            float const text_x = label_x + icon_size + style.ItemInnerSpacing.x;
            if (renaming_entity == entity) {
                ImGui::SetCursorPos(ImVec2(text_x, row_pos.y));
                ImGui::SetNextItemWidth(-FLT_MIN);
                if (rename_needs_focus) {
                    ImGui::SetKeyboardFocusHere();
                    rename_needs_focus = false;
                }

                bool const submitted =
                        ImGui::InputText("##rename", rename_buffer.data(), rename_buffer.size(),
                                         ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll);
                if (submitted || (ImGui::IsItemDeactivated() && !ImGui::IsKeyPressed(ImGuiKey_Escape))) {
                    rename_entity(registry, entity, rename_buffer.data());
                    renaming_entity = entt::null;
                } else if (ImGui::IsItemDeactivated()) {
                    renaming_entity = entt::null;
                }
            } else {
                ImGui::SetCursorPos(ImVec2(text_x, row_pos.y + (row_height - ImGui::GetTextLineHeight()) * 0.5F));
                ImGui::TextUnformatted(name.c_str());
            }

            ImGui::SetCursorPos(next_row_pos);
            ImGui::PopID();
        };

        auto const total_count = hierarchy_model.node_count();
        if (auto const selected_count = selection.size(); selected_count > 1) {
            ImGui::TextDisabled("%zu %s, %zu selected", total_count, total_count == 1 ? "entity" : "entities",
                                selected_count);
        } else {
            ImGui::TextDisabled("%zu %s", total_count, total_count == 1 ? "entity" : "entities");
        }

        auto const apply_selection_requests = [&](ImGuiMultiSelectIO const *io) {
            if (io == nullptr || io->Requests.empty()) {
                return;
            }
            if (renaming_entity != entt::null && !ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                return;
            }

            selection.modify([&](SelectionContext::Transaction &transaction) {
                for (auto const &request: io->Requests) {
                    if (request.Type == ImGuiSelectionRequestType_SetAll) {
                        transaction.clear();
                        if (!request.Selected) {
                            continue;
                        }
                        for (auto const entity: hierarchy_model.matching_entities()) {
                            transaction.set(entity, true);
                        }
                    } else if (request.Type == ImGuiSelectionRequestType_SetRange) {
                        auto const first = std::max<ImGuiSelectionUserData>(
                                std::min(request.RangeFirstItem, request.RangeLastItem), 0);
                        auto const last = std::min<ImGuiSelectionUserData>(
                                std::max(request.RangeFirstItem, request.RangeLastItem),
                                static_cast<ImGuiSelectionUserData>(rows.size()) - 1);
                        for (auto row = first; row <= last; ++row) {
                            if (auto const entity = rows[static_cast<std::size_t>(row)].entity; entity != entt::null) {
                                transaction.set(entity, request.Selected);
                            }
                        }
                    }
                }
            });
        };

        constexpr ImGuiMultiSelectFlags multi_select_flags = ImGuiMultiSelectFlags_ClearOnEscape |
                                                             ImGuiMultiSelectFlags_ClearOnClickVoid |
                                                             ImGuiMultiSelectFlags_BoxSelect1d;
        auto *const multi_select_io = ImGui::BeginMultiSelect(multi_select_flags, static_cast<int>(selection.size()),
                                                              static_cast<int>(rows.size()));
        apply_selection_requests(multi_select_io);

        float const rows_x = ImGui::GetCursorPosX();

        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(rows.size()));

        if (multi_select_io->RangeSrcItem != -1) {
            clipper.IncludeItemByIndex(static_cast<int>(multi_select_io->RangeSrcItem));
        }
        if (renaming_entity != entt::null) {
            if (auto const renaming_row = hierarchy_model.row_of(renaming_entity); renaming_row >= 0) {
                clipper.IncludeItemByIndex(static_cast<int>(renaming_row));
            }
        }

        while (clipper.Step()) {
            for (auto row_index = clipper.DisplayStart; row_index < clipper.DisplayEnd; ++row_index) {
                auto const index = static_cast<std::size_t>(row_index);
                auto const &row = rows[index];

                ImGui::SetCursorPosX(rows_x + (style.IndentSpacing * static_cast<float>(row.depth)));
                draw_alternate_background(index);

                if (row.entity == entt::null) {
                    draw_group_row(row);
                } else {
                    draw_entity_row(row, index);
                }
            }
        }

        ImGui::Dummy(ImVec2(0.0F, 0.0F));

        if (ImGui::BeginDragDropTarget()) {
            if (auto const *payload = ImGui::AcceptDragDropPayload("HIERARCHY_ENTITY")) {
                entt::entity dropped{};
                std::memcpy(&dropped, payload->Data, sizeof(dropped));
                pending_action = HierarchyAction::reparent;
                action_target = entt::null;
                drag_reparent_source = dropped;
            }
            ImGui::EndDragDropTarget();
        }

        apply_selection_requests(ImGui::EndMultiSelect());
        if (clicked_entity != entt::null) {
            selection.modify(
                    [&](SelectionContext::Transaction &transaction) { transaction.set_primary(clicked_entity); });
        }

        if (open_context_menu) {
            ImGui::OpenPopup("entity_context");
        }

        if (ImGui::BeginPopup("entity_context")) {
            auto const entity = hierarchy_context_entity;

            if (!registry.valid(entity)) {
                ImGui::CloseCurrentPopup();
            } else {
                auto const count = selection.size();

                if (ImGui::MenuItem("Add Child Entity")) {
                    pending_action = HierarchyAction::add_child;
                    action_target = entity;
                }
                if (ImGui::MenuItem("Rename", "F2")) {
                    begin_rename(entity);
                }
                if (ImGui::MenuItem(count > 1 ? "Duplicate Selected" : "Duplicate", "Ctrl+D")) {
                    pending_action = HierarchyAction::duplicate;
                }
                ImGui::Separator();
                if (ImGui::MenuItem(count > 1 ? "Delete Selected" : "Delete", "Del")) {
                    pending_action = HierarchyAction::remove;
                }
            }
            ImGui::EndPopup();
        }

        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput) {
            if (auto const primary = selection.primary();
                ImGui::IsKeyPressed(ImGuiKey_F2, false) && primary != entt::null) {
                begin_rename(primary);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) && !selection.empty()) {
                pending_action = HierarchyAction::remove;
            }
            if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D) && !selection.empty()) {
                pending_action = HierarchyAction::duplicate;
            }
        }

        if (ImGui::BeginPopupContextWindow("hierarchy_bg_context",
                                           ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
            if (ImGui::MenuItem("Create Empty Entity")) {
                pending_action = HierarchyAction::add_child;
                action_target = entt::null;
            }
            ImGui::EndPopup();
        }

        auto const selected_roots = [&] { return selection_roots(registry, selection.snapshot().entities); };

        switch (pending_action) {
            case HierarchyAction::add_child: {
                auto new_entity = GeneratedEntity{active_scene(), "Entity"};
                new_entity.emplace<Components::Transform>();
                if (action_target != entt::null && registry.valid(action_target)) {
                    new_entity.emplace<Components::Parent>(Components::Parent{.entity = action_target});
                }
                selection.select(new_entity);
                break;
            }
            case HierarchyAction::duplicate: {
                std::function<entt::entity(entt::entity, entt::entity)> duplicate_subtree =
                        [&](entt::entity source, entt::entity parent) -> entt::entity {
                    auto const clone = registry.create();

                    copy_components<Components::Transform, Components::Model, Components::InstancedModel,
                                    Components::RigidBody, Components::MaterialOverride, Components::PlayerTag,
                                    Components::Lifetime, Components::PointLight, Components::SpotLight,
                                    Components::Decal, Components::ParticleEmitter, Components::Water, Components::Planet, Components::Text, Components::Script, Components::Behaviour,
                                    Components::BulletTag, Components::Static>(
                            registry, source, clone);

                    if (auto const *model = registry.try_get<Components::Model>(clone);
                        model != nullptr && registry.all_of<Components::StreamedModelTag>(source)) {
                        renderer->retain_model(model->model);
                        registry.emplace<Components::StreamedModelTag>(clone);
                    }

                    registry.emplace<Components::GeneratedMeta>(
                            clone,
                            Components::GeneratedMeta{.name = entity_display_name(registry, source) + " (Copy)"});

                    if (parent != entt::null) {
                        registry.emplace<Components::Parent>(clone, Components::Parent{.entity = parent});
                    }

                    for (auto const child: hierarchy_model.children_of(source)) {
                        duplicate_subtree(child, clone);
                    }

                    return clone;
                };

                std::vector<entt::entity> clones;
                for (auto const source: selected_roots()) {
                    auto const *source_parent = registry.try_get<Components::Parent>(source);
                    clones.push_back(
                            duplicate_subtree(source, source_parent != nullptr ? source_parent->entity : entt::null));
                }
                selection.assign(clones);
                break;
            }
            case HierarchyAction::remove: {
                std::function<void(entt::entity)> delete_subtree = [&](entt::entity target) {
                    if (!registry.valid(target)) {
                        return;
                    }
                    for (auto const child: hierarchy_model.children_of(target)) {
                        delete_subtree(child);
                    }

                    registry.destroy(target);
                };

                for (auto const target: selected_roots()) {
                    delete_subtree(target);
                }
                selection.clear();
                break;
            }
            case HierarchyAction::reparent: {
                auto const sources = selection.contains(drag_reparent_source)
                                             ? selected_roots()
                                             : std::vector<entt::entity>{drag_reparent_source};

                for (auto const source: sources) {
                    if (!registry.valid(source) || source == action_target) {
                        continue;
                    }

                    bool creates_cycle = false;
                    for (auto walk = action_target; walk != entt::null && registry.valid(walk);) {
                        if (walk == source) {
                            creates_cycle = true;
                            break;
                        }
                        auto const *walk_parent = registry.try_get<Components::Parent>(walk);
                        walk = walk_parent != nullptr ? walk_parent->entity : entt::null;
                    }

                    if (creates_cycle) {
                        continue;
                    }

                    if (action_target == entt::null) {
                        registry.remove<Components::Parent>(source);
                    } else {
                        registry.emplace_or_replace<Components::Parent>(source,
                                                                        Components::Parent{.entity = action_target});
                    }
                }
                break;
            }
            case HierarchyAction::none:
                break;
        }

        if (renaming_entity != entt::null && !registry.valid(renaming_entity)) {
            renaming_entity = entt::null;
        }
    });

    panel("Inspector", panels.inspector, [&] {
        auto &registry = active_scene()->get_registry();
        auto const selected_entity = selection_context().primary();

        if (selected_entity == entt::null || !registry.valid(selected_entity)) {
            ImGui::TextDisabled("No entity selected");
            return;
        }

        if (auto const count = selection_context().size(); count > 1) {
            ImGui::TextDisabled("%zu entities selected -- editing the last one", count);
        }

        auto const name_id = ImGui::GetID("##entity_name");
        if (ImGui::GetActiveID() != name_id) {
            if (inspector_name_dirty) {
                rename_entity(registry, inspector_name_entity, inspector_name_buffer.data());
                inspector_name_dirty = false;
            }
            inspector_name_entity = selected_entity;
            copy_to_buffer(inspector_name_buffer, entity_display_name(registry, selected_entity));
        }
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputText("##entity_name", inspector_name_buffer.data(), inspector_name_buffer.size())) {
            inspector_name_dirty = true;
        }
        ImGui::Separator();

        auto section = [&]<typename T>(char const *label, auto &&draw_fields) {
            if (!registry.all_of<T>(selected_entity)) {
                return;
            }

            bool open = true;
            ImGui::PushID(label);
            if (ImGui::CollapsingHeader(label, &open, ImGuiTreeNodeFlags_DefaultOpen)) {
                if (draw_fields(registry.get<T>(selected_entity))) {
                    registry.patch<T>(selected_entity);

                    if constexpr (std::is_same_v<T, Components::Transform>) {
                        renderer->mark_dynamic_shadow_casters_dirty();
                    }
                }
            }
            ImGui::PopID();

            if (!open) {
                if constexpr (std::is_same_v<T, Components::Model>) {
                    registry.remove<Components::StreamedModelTag>(selected_entity);
                }
                registry.remove<T>(selected_entity);
            }
        };

        std::vector<MaterialHandle> created_materials;

        auto const make_material_copy = [&](MaterialHandle source) -> std::optional<MaterialHandle> {
            auto copy = renderer->duplicate_material(source.valid() ? source : renderer->default_material());
            if (!copy) {
                warn("Inspector: failed to create a material: {}", describe(copy.error()));
                return std::nullopt;
            }
            created_materials.push_back(*copy);
            return *copy;
        };

        auto const material_picker = [&](char const *label, MaterialHandle current, char const *none_label,
                                         MaterialHandle copy_source) -> std::optional<MaterialHandle> {
            auto const &materials = renderer->assets().materials();
            auto const current_name = materials.name_of(current);
            std::string const preview = !current.valid()       ? none_label
                                        : current_name.empty() ? "(own material)"
                                                               : std::string(current_name);

            std::optional<MaterialHandle> picked;
            if (!ImGui::BeginCombo(label, preview.c_str())) {
                return picked;
            }

            if (ImGui::Selectable(none_label, !current.valid()) && current.valid()) {
                picked = MaterialHandle{};
            }
            if (ImGui::Selectable("New material (copy)")) {
                picked = make_material_copy(current.valid() ? current : copy_source);
            }
            ImGui::Separator();

            for (auto const &entry: materials.entries()) {
                bool const is_selected = entry.handle == current;
                if (!entry.handle.valid() || entry.name.empty() ||
                    (!is_selected && is_material_pending_deletion(pending_deletions, entry.name))) {
                    continue;
                }
                if (ImGui::Selectable(entry.name.c_str(), is_selected) && !is_selected) {
                    picked = entry.handle;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
            return picked;
        };

        auto const draw_material_editor = [&](MaterialHandle handle) -> std::optional<MaterialHandle> {
            auto const *info = renderer->material_storage().create_info(handle);
            if (info == nullptr) {
                ImGui::TextDisabled("(material data unavailable)");
                return std::nullopt;
            }

            auto &materials = renderer->assets().materials();
            auto const name = materials.name_of(handle);
            bool const shared = !name.empty() || renderer->material_storage().ref_count(handle) > 1;

            if (!name.empty()) {
                ImGui::TextDisabled("Asset \"%.*s\" -- edits show everywhere it's used.", static_cast<int>(name.size()),
                                    name.data());
            } else if (shared) {
                ImGui::TextDisabled("Unnamed, also used elsewhere.");
            } else {
                ImGui::TextDisabled("Unnamed, owned by this entity.");
            }

            std::optional<MaterialHandle> replacement;
            if (shared) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Make unique")) {
                    replacement = make_material_copy(handle);
                }
            }
            if (name.empty()) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Save as asset...")) {
                    save_material_name.clear();
                    ImGui::OpenPopup("save_material_asset");
                }
            }

            if (ImGui::BeginPopup("save_material_asset")) {
                std::array<char, 64> name_buffer{};
                copy_to_buffer(name_buffer, save_material_name);
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 16.0F);
                if (ImGui::InputTextWithHint("Name", "material name", name_buffer.data(), name_buffer.size())) {
                    save_material_name.assign(name_buffer.data());
                }

                bool const name_taken = !save_material_name.empty() && materials.find(save_material_name).valid();
                if (name_taken) {
                    ImGui::TextColored(ImVec4(0.95F, 0.45F, 0.35F, 1.0F), "That name is already registered.");
                }

                ImGui::BeginDisabled(save_material_name.empty() || name_taken);
                if (ImGui::Button("Save")) {
                    static_cast<void>(renderer->register_material_name(handle, save_material_name));
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndDisabled();
                ImGui::SameLine();
                if (ImGui::Button("Cancel")) {
                    ImGui::CloseCurrentPopup();
                }
                ImGui::EndPopup();
            }

            auto edited = *info;
            if (draw_material_fields(edited)) {
                static_cast<void>(renderer->update_material(handle, edited));
            }

            return replacement;
        };

        auto const draw_material_overrides = [&](entt::entity entity, ModelHandle model) {
            auto const *existing = registry.try_get<Components::MaterialOverride>(entity);
            auto edited = existing != nullptr ? *existing : Components::MaterialOverride{};
            bool changed = false;

            auto const set_slot = [&](MaterialHandle source, MaterialHandle material) {
                std::erase_if(edited.slots, [&](MaterialSlotOverride const &slot) { return slot.source == source; });
                if (material.valid()) {
                    edited.slots.push_back(MaterialSlotOverride{.source = source, .material = material});
                }
                changed = true;
            };

            bool const loading = renderer->model_streamer().state(model) == ModelRequestState::loading;
            auto const sources = loading ? std::vector<MaterialHandle>{} : renderer->model_materials(model);
            if (loading) {
                ImGui::TextDisabled("Per-slot materials appear once the model has loaded.");
            }

            ImGui::PushID("all_submeshes");
            if (auto const picked = material_picker("All submeshes", edited.material, "(none)",
                                                    sources.empty() ? MaterialHandle{} : sources.front())) {
                edited.material = *picked;
                changed = true;
            }
            if (edited.material.valid() && ImGui::TreeNode("Edit")) {
                if (auto const replacement = draw_material_editor(edited.material)) {
                    edited.material = *replacement;
                    changed = true;
                }
                ImGui::TreePop();
            }
            ImGui::PopID();

            auto const slot_material = [&](MaterialHandle source) {
                auto const it = std::ranges::find(edited.slots, source, &MaterialSlotOverride::source);
                return it != edited.slots.end() ? it->material : MaterialHandle{};
            };

            for (std::size_t index = 0; index < sources.size(); ++index) {
                auto const source = sources[index];

                ImGui::PushID(static_cast<int>(index));

                auto const replacement = edited.replacement_for(source);
                if (auto const *shown =
                            renderer->material_storage().create_info(replacement.valid() ? replacement : source)) {
                    auto const &colour = shown->base_colour_factor;
                    ImGui::ColorButton("##swatch", ImVec4(colour.r, colour.g, colour.b, colour.a),
                                       ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_AlphaPreview);
                    ImGui::SameLine();
                }

                auto const label = std::format("Slot {}", index);
                ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.0F);
                if (auto const picked =
                            material_picker(label.c_str(), slot_material(source), "(model material)", source)) {
                    set_slot(source, *picked);
                }

                ImGui::SameLine();
                if (slot_material(source).valid()) {
                    if (ImGui::SmallButton("Revert")) {
                        set_slot(source, MaterialHandle{});
                    }
                } else if (ImGui::SmallButton("Override")) {
                    if (auto const copy = make_material_copy(source)) {
                        set_slot(source, *copy);
                        ImGui::SetNextItemOpen(true);
                    }
                }

                if (auto const current = slot_material(source); current.valid() && ImGui::TreeNode("Edit")) {
                    if (auto const copy = draw_material_editor(current)) {
                        set_slot(source, *copy);
                    }
                    ImGui::TreePop();
                }

                ImGui::PopID();
            }

            auto const stale = std::ranges::count_if(edited.slots, [&](MaterialSlotOverride const &slot) {
                return std::ranges::find(sources, slot.source) == sources.end();
            });
            if (model.valid() && !loading && stale > 0) {
                ImGui::TextDisabled("%d override(s) for materials this model doesn't have.", static_cast<int>(stale));
                ImGui::SameLine();
                if (ImGui::SmallButton("Remove")) {
                    std::erase_if(edited.slots, [&](MaterialSlotOverride const &slot) {
                        return std::ranges::find(sources, slot.source) == sources.end();
                    });
                    changed = true;
                }
            }

            if (changed) {
                active_scene()->set_material_override(entity, std::move(edited));
            }

            for (auto const handle: created_materials) {
                renderer->release_material(handle);
            }
            created_materials.clear();
        };

        section.operator()<Components::Transform>("Transform", draw_transform);
        section.operator()<Components::PointLight>("Point Light", draw_point_light);
        section.operator()<Components::SpotLight>("Spot Light", draw_spot_light);
        section.operator()<Components::Decal>("Decal", draw_decal);
        section.operator()<Components::ParticleEmitter>("Particle Emitter", draw_particle_emitter);
        section.operator()<Components::Water>("Water", draw_water);
        section.operator()<Components::Planet>("Planet", draw_planet);
        section.operator()<Components::Text>("Text", draw_text);
        section.operator()<Components::RigidBody>("Rigid Body", draw_rigid_body);
        section.operator()<Components::Lifetime>("Lifetime", draw_lifetime);

        section.operator()<Components::Model>("Model", [&](Components::Model &model) {
            ImGui::Text("Handle: index %u, generation %u (%s)", model.model.index, model.model.generation,
                        model.model.valid() ? "valid" : "invalid");

            if (auto const bounds = renderer->model_bounds(model.model)) {
                auto const &[min, max] = *bounds;
                ImGui::Text("Bounds (model space): min (%.2f, %.2f, %.2f)", min.x, min.y, min.z);
                ImGui::Text("                      max (%.2f, %.2f, %.2f)", max.x, max.y, max.z);
            } else {
                ImGui::TextDisabled("Bounds unavailable");
            }

            if (auto const submesh_bounds = renderer->model_submesh_bounds(model.model)) {
                ImGui::Text("Submeshes: %u", static_cast<std::uint32_t>(submesh_bounds->size()));
            }

            if (auto const lights = renderer->model_lights(model.model); !lights.empty()) {
                ImGui::Text("Embedded lights: %u", static_cast<std::uint32_t>(lights.size()));
            }

            if (auto const state = renderer->model_streamer().state(model.model); state == ModelRequestState::loading) {
                ImGui::TextDisabled("Loading -- showing the placeholder model");
            } else if (state == ModelRequestState::failed) {
                auto const reason = renderer->model_streamer().failure_reason(model.model);
                ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F),
                                   "Load failed: %.*s -- showing the placeholder model",
                                   static_cast<int>(reason.size()), reason.data());
            }

            auto &models = renderer->assets().models();
            auto const current_name = models.name_of(model.model);

            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                auto const used = models_in_use(registry);
                if (auto const picked = ui::draw_asset_choices(models.entries(), std::span{used}, model.model);
                    picked && *picked != model.model) {
                    renderer->retain_model(*picked);
                    set_entity_model(registry, selected_entity, *picked);
                }
                ImGui::EndCombo();
            }

            ImGui::SameLine();
            ImGui::BeginDisabled(model_browser.is_open());
            if (ImGui::Button("Browse...##model")) {
                model_browse_target = ModelBrowseTarget::inspector;
                model_browse_entity = selected_entity;
                model_browser.open("Change Model", model_file_filters());
            }
            ImGui::EndDisabled();

            ImGui::SeparatorText("Materials");
            draw_material_overrides(selected_entity, model.model);

            return false;
        });
        section.operator()<Components::MaterialOverride>("Material Override", [&](Components::MaterialOverride &mat) {
            if (registry.all_of<Components::Model>(selected_entity)) {
                ImGui::TextDisabled("%zu slot override(s)%s -- edit them under Model > Materials.", mat.slots.size(),
                                    mat.material.valid() ? ", all submeshes overridden" : "");
            } else {
                draw_material_overrides(selected_entity, ModelHandle{});
            }
            return false;
        });
        section.operator()<Components::Script>("Script", [&](Components::Script &script) {
            ImGui::Text("Handle: index %u, generation %u (%s)", script.script.index, script.script.generation,
                        script.script.valid() ? "valid" : "invalid");

            auto &scripts = renderer->assets().scripts();
            if (scripts.entries().empty()) {
                ImGui::TextDisabled("No named scripts registered yet.");
                return false;
            }

            bool changed = false;
            auto const current_name = scripts.name_of(script.script);
            if (ImGui::BeginCombo("Asset", current_name.empty() ? "(unnamed)" : std::string(current_name).c_str())) {
                for (auto const &entry: scripts.entries()) {
                    if (entry.name.empty()) {
                        continue; // ImGui rejects an item without a label
                    }
                    bool const is_selected = entry.handle == script.script;
                    if (ImGui::Selectable(entry.name.c_str(), is_selected) && entry.handle != script.script) {
                        script.script = entry.handle;
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            return changed;
        });
        section.operator()<Components::Behaviour>("Behaviour", [&](Components::Behaviour &script) {
            bool changed = false;
            auto const names = game->behaviour_names();

            if (ImGui::BeginCombo("Script", script.script.empty() ? "(none)" : script.script.c_str())) {
                for (auto const &name: names) {
                    bool const is_selected = name == script.script;
                    if (ImGui::Selectable(name.c_str(), is_selected) && !is_selected) {
                        script.script = name;
                        script.values.clear();
                        changed = true;
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::SameLine();
            if (ImGui::SmallButton("Reload")) {
                game->reload_behaviours();
            }

            auto const *const schema = game->behaviour_schema(script.script);

            if (schema == nullptr) {
                ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F), "No behaviour named '%s'.",
                                   script.script.c_str());
                return changed;
            }

            if (!schema->error.empty()) {
                ImGui::TextColored(ImVec4(1.0F, 0.45F, 0.40F, 1.0F), "%s", schema->error.c_str());
                return changed;
            }

            if (schema->properties.empty()) {
                ImGui::TextDisabled("This script declares no properties.");
            }

            auto const asset_combo = [&](char const *label, std::string &current, auto const &table,
                                         std::vector<std::string> extra) {
                bool picked = false;
                if (ImGui::BeginCombo(label, current.empty() ? "(none)" : current.c_str())) {
                    if (ImGui::Selectable("(none)", current.empty()) && !current.empty()) {
                        current.clear();
                        picked = true;
                    }
                    for (auto const &name: extra) {
                        if (ImGui::Selectable(name.c_str(), name == current) && name != current) {
                            current = name;
                            picked = true;
                        }
                    }
                    for (auto const &entry: table.entries()) {
                        if (entry.name.empty()) {
                            continue; // ImGui rejects an item without a label
                        }
                        if (ImGui::Selectable(entry.name.c_str(), entry.name == current) && entry.name != current) {
                            current = entry.name;
                            picked = true;
                        }
                    }
                    ImGui::EndCombo();
                }
                return picked;
            };

            for (auto const &desc: schema->properties) {
                ImGui::PushID(desc.name.c_str());

                auto const *const stored = script.find(desc.name);
                auto value = stored != nullptr ? script_value_sanitised(desc, stored->value) : desc.default_value;
                bool edited = false;
                auto const label = desc.name.c_str();
                auto const speed = desc.step > 0.0 ? desc.step
                                   : desc.has_range ? (desc.max - desc.min) / 200.0
                                                    : 0.05;
                auto const low = desc.has_range ? desc.min : -1.0e9;
                auto const high = desc.has_range ? desc.max : 1.0e9;

                switch (desc.type) {
                    case ScriptPropertyType::boolean:
                        edited = ImGui::Checkbox(label, &std::get<bool>(value));
                        break;
                    case ScriptPropertyType::integer: {
                        auto &integer = std::get<std::int64_t>(value);
                        auto const minimum = static_cast<std::int64_t>(low);
                        auto const maximum = static_cast<std::int64_t>(high);
                        edited = ImGui::DragScalar(label, ImGuiDataType_S64, &integer, static_cast<float>(std::max(speed, 0.1)),
                                                   &minimum, &maximum);
                        break;
                    }
                    case ScriptPropertyType::number:
                        edited = ImGui::DragScalar(label, ImGuiDataType_Double, &std::get<double>(value),
                                                   static_cast<float>(speed), &low, &high, "%.3f");
                        break;
                    case ScriptPropertyType::string: {
                        std::array<char, 256> text{};
                        std::snprintf(text.data(), text.size(), "%s", std::get<std::string>(value).c_str());
                        if (ImGui::InputText(label, text.data(), text.size())) {
                            value = std::string{text.data()};
                            edited = true;
                        }
                        break;
                    }
                    case ScriptPropertyType::vec3:
                        edited = ImGui::DragFloat3(label, &std::get<glm::vec3>(value).x, static_cast<float>(speed),
                                                   static_cast<float>(low), static_cast<float>(high));
                        break;
                    case ScriptPropertyType::colour:
                        edited = ImGui::ColorEdit4(label, &std::get<glm::vec4>(value).x,
                                                   ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
                        break;
                    case ScriptPropertyType::choice: {
                        auto &current = std::get<std::string>(value);
                        if (ImGui::BeginCombo(label, current.c_str())) {
                            for (auto const &option: desc.choices) {
                                if (ImGui::Selectable(option.c_str(), option == current) && option != current) {
                                    current = option;
                                    edited = true;
                                }
                            }
                            ImGui::EndCombo();
                        }
                        break;
                    }
                    case ScriptPropertyType::entity: {
                        auto &target = std::get<entt::entity>(value);
                        auto const target_name = [&]() -> std::string {
                            if (target == entt::null || !registry.valid(target)) {
                                return "(none)";
                            }
                            auto const *const meta = registry.try_get<Components::Meta>(target);
                            return meta != nullptr ? std::string{meta->name.view()} : "(unnamed)";
                        }();
                        if (ImGui::BeginCombo(label, target_name.c_str())) {
                            if (ImGui::Selectable("(none)", target == entt::null) && target != entt::null) {
                                target = entt::null;
                                edited = true;
                            }
                            for (auto const other: registry.view<Components::Meta>()) {
                                auto const name = std::string{registry.get<Components::Meta>(other).name.view()};
                                ImGui::PushID(static_cast<int>(entt::to_integral(other)));
                                if (ImGui::Selectable(name.c_str(), other == target) && other != target) {
                                    target = other;
                                    edited = true;
                                }
                                ImGui::PopID();
                            }
                            ImGui::EndCombo();
                        }
                        break;
                    }
                    case ScriptPropertyType::model:
                        edited = asset_combo(label, std::get<std::string>(value), renderer->assets().models(), {});
                        break;
                    case ScriptPropertyType::material:
                        edited = asset_combo(label, std::get<std::string>(value), renderer->assets().materials(), {});
                        break;
                    case ScriptPropertyType::texture:
                        edited = asset_combo(label, std::get<std::string>(value), renderer->assets().textures(),
                                             {"scene_colour", "scene_depth"});
                        break;
                }

                if (edited) {
                    script.set(desc.name, script_value_sanitised(desc, std::move(value)));
                    changed = true;
                }

                if (script.find(desc.name) != nullptr) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton("Reset")) {
                        script.reset(desc.name);
                        changed = true;
                    }
                }

                if (!desc.tooltip.empty() && ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", desc.tooltip.c_str());
                }

                ImGui::PopID();
            }

            return changed;
        });
        auto tag_section = [&]<typename T>(char const *label) {
            if (!registry.all_of<T>(selected_entity)) {
                return;
            }

            bool open = true;
            ImGui::PushID(label);
            if (ImGui::CollapsingHeader(label, &open, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextDisabled("Marker component -- no fields.");
            }
            ImGui::PopID();

            if (!open) {
                registry.remove<T>(selected_entity);
            }
        };

        tag_section.operator()<Components::PlayerTag>("Player Tag");
        tag_section.operator()<Components::BulletTag>("Bullet Tag");
        tag_section.operator()<Components::Static>("Static");

        ImGui::Separator();

        if (ImGui::Button("Add Component")) {
            ImGui::OpenPopup("inspector_add_component");
        }

        if (ImGui::BeginPopup("inspector_add_component")) {
            bool const has_transform = registry.all_of<Components::Transform>(selected_entity);

            if (!has_transform) {
                if (ImGui::MenuItem("Transform")) {
                    registry.emplace<Components::Transform>(selected_entity);
                }
            } else {
                if (!registry.all_of<Components::PointLight>(selected_entity) && ImGui::MenuItem("Point Light")) {
                    registry.emplace<Components::PointLight>(selected_entity);
                }
                if (!registry.all_of<Components::SpotLight>(selected_entity) && ImGui::MenuItem("Spot Light")) {
                    registry.emplace<Components::SpotLight>(selected_entity);
                }
                if (!registry.all_of<Components::ParticleEmitter>(selected_entity) &&
                    ImGui::MenuItem("Particle Emitter")) {
                    registry.emplace<Components::ParticleEmitter>(selected_entity);
                }
                if (!registry.all_of<Components::Water>(selected_entity) && ImGui::MenuItem("Water")) {
                    registry.emplace<Components::Water>(selected_entity, Components::Water::pool());
                }
                if (!registry.all_of<Components::Text>(selected_entity) && ImGui::MenuItem("Text")) {
                    registry.emplace<Components::Text>(selected_entity, Components::Text{.text = "Text"});
                }
                if (!registry.all_of<Components::RigidBody>(selected_entity) && ImGui::MenuItem("Rigid Body")) {
                    registry.emplace<Components::RigidBody>(selected_entity);
                }

                if (!registry.all_of<Components::Model>(selected_entity) &&
                    !renderer->assets().models().entries().empty() && ImGui::BeginMenu("Model")) {
                    auto const used = models_in_use(registry);
                    if (auto const picked = ui::draw_asset_choices(renderer->assets().models().entries(),
                                                                   std::span{used}, ModelHandle{})) {
                        renderer->retain_model(*picked);
                        registry.emplace<Components::Model>(selected_entity, Components::Model{.model = *picked});
                        registry.emplace<Components::StreamedModelTag>(selected_entity);
                    }
                    ImGui::EndMenu();
                }
            }

            if (!registry.all_of<Components::Lifetime>(selected_entity) && ImGui::MenuItem("Lifetime")) {
                registry.emplace<Components::Lifetime>(selected_entity,
                                                       Components::Lifetime{.remaining_seconds = 5.0F});
            }

            if (!registry.all_of<Components::MaterialOverride>(selected_entity) &&
                ImGui::MenuItem("Material Override")) {
                registry.emplace<Components::MaterialOverride>(selected_entity);
            }

            if (!registry.all_of<Components::Script>(selected_entity) &&
                !renderer->assets().scripts().entries().empty() && ImGui::BeginMenu("Script")) {
                for (auto const &entry: renderer->assets().scripts().entries()) {
                    if (entry.name.empty()) {
                        continue; // ImGui rejects an item without a label
                    }
                    if (ImGui::MenuItem(entry.name.c_str())) {
                        registry.emplace<Components::Script>(selected_entity,
                                                             Components::Script{.script = entry.handle});
                    }
                }
                ImGui::EndMenu();
            }

            if (!registry.all_of<Components::Behaviour>(selected_entity) && !game->behaviour_names().empty() &&
                ImGui::BeginMenu("Behaviour")) {
                for (auto const &name: game->behaviour_names()) {
                    if (ImGui::MenuItem(name.c_str())) {
                        registry.emplace<Components::Behaviour>(selected_entity, Components::Behaviour{.script = name});
                    }
                }
                ImGui::EndMenu();
            }

            ImGui::EndPopup();
        }
    });

    panel("Scene stats", panels.scene_stats, [&] {
        auto const &stats = renderer->last_frame_stats();
        renderer->request_pipeline_stats();
        auto const &pipeline_stats = renderer->last_frame_pipeline_stats();

        constexpr auto fmt = [](std::uint64_t count) {
            static thread_local std::array<char, 64> buf{};
            if (count >= 1'000'000'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fB", static_cast<double>(count) / 1e9F);
            } else if (count >= 1'000'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fM", static_cast<double>(count) / 1e6F);
            } else if (count >= 1'000) {
                std::snprintf(buf.data(), buf.size(), "%.2fK", static_cast<double>(count) / 1e3F);
            } else {
                std::snprintf(buf.data(), buf.size(), "%llu", static_cast<unsigned long long>(count));
            }
            return buf.data();
        };

        if (pipeline_stats.valid) {
            if (!pipeline_stats.mesh_stats_valid) {
                ImGui::Text("Triangles rendered (post-clip): %s (%llu)", fmt(pipeline_stats.clipped_primitive_count),
                            static_cast<unsigned long long>(pipeline_stats.clipped_primitive_count));
            } else {
                ImGui::Text("Task shader invocations: %s (%llu)", fmt(pipeline_stats.task_shader_invocation_count),
                            static_cast<unsigned long long>(pipeline_stats.task_shader_invocation_count));
                ImGui::Text("Mesh shader invocations: %s (%llu)", fmt(pipeline_stats.mesh_shader_invocation_count),
                            static_cast<unsigned long long>(pipeline_stats.mesh_shader_invocation_count));
            }
            ImGui::Text("Fragment shader invocations: %s (%llu)", fmt(pipeline_stats.fragment_shader_invocation_count),
                        static_cast<unsigned long long>(pipeline_stats.fragment_shader_invocation_count));
        } else {
            ImGui::TextDisabled("Pipeline stats not yet available");
        }

        ImGui::Text("Triangles submitted (pre-cull): %s (%u)", fmt(stats.submitted_triangle_count),
                    stats.submitted_triangle_count);
        ImGui::Text("Draw calls: %u  (opaque %u / double-sided %u / mask %u / blend %u)", stats.indirect_command_count,
                    stats.opaque_indirect_count, stats.double_sided_indirect_count, stats.mask_indirect_count,
                    stats.blend_indirect_count);
        ImGui::Text("Instances submitted: %s (%u)", fmt(stats.submitted_instance_count),
                    stats.submitted_instance_count);

        auto const culled_percent = stats.submitted_instance_count != 0
                                            ? 100.0F * static_cast<float>(stats.visible_instance_count) /
                                                      static_cast<float>(stats.submitted_instance_count)
                                            : 0.0F;
        ImGui::Text("Instances visible (post-cull): %s (%u, %.1f%%)", fmt(stats.visible_instance_count),
                    stats.visible_instance_count, culled_percent);

        if (stats.occlusion_stats_valid) {
            auto const frustum_culled = stats.submitted_instance_count > stats.frustum_visible_instance_count
                                                ? stats.submitted_instance_count - stats.frustum_visible_instance_count
                                                : 0U;
            auto const occluded_percent = stats.frustum_visible_instance_count != 0
                                                  ? 100.0F * static_cast<float>(stats.occluded_instance_count) /
                                                            static_cast<float>(stats.frustum_visible_instance_count)
                                                  : 0.0F;

            ImGui::Text("Instances frustum-culled: %s (%u)", fmt(frustum_culled), frustum_culled);
            ImGui::Text("Instances occluded (Hi-Z): %s (%u, %.1f%% of frustum-visible)",
                        fmt(stats.occluded_instance_count), stats.occluded_instance_count, occluded_percent);
            ImGui::Text("Phase 1 / phase 2 instances: %u / %u (%u deferred by phase 1)", stats.early_instance_count,
                        stats.late_instance_count, stats.occlusion_candidate_count);

            if (stats.meshlet_occlusion_stats_valid) {
                ImGui::Text("Meshlets occluded: %s (%u final, %u deferred by phase 1)",
                            fmt(stats.occluded_meshlet_count), stats.occluded_meshlet_count,
                            stats.deferred_meshlet_count);
            }
        } else {
            ImGui::TextDisabled("Occlusion culling inactive");
        }

        ImGui::Text("Model / mesh submissions: %u / %u", stats.model_submission_count, stats.mesh_submission_count);
        ImGui::Text("Lights: %u point / %u spot", stats.point_light_count, stats.spot_light_count);
    });

    panel("Frame timings", panels.frame_timings, [&] {
        if (ImPlot::BeginPlot("Stage timings (cumulative ms)", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Frame", "ms", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, timing_x - 600.0, timing_x, ImGuiCond_Always);

            for (auto index = std::size_t{0}; index < timing_series.size(); ++index) {
                auto const &buf = timing_series[index].buffer;

                if (buf.data.empty()) {
                    continue;
                }

                ImPlotSpec spec;
                spec.Offset = buf.offset;
                spec.Stride = sizeof(ImVec2);
                spec.FillAlpha = 0.35F;

                auto const &label = timing_series[index].label;

                if (index == 0) {
                    ImPlot::PlotShaded(label.c_str(), &buf.data[0].x, &buf.data[0].y, static_cast<int>(buf.data.size()),
                                       0.0, spec);
                } else {
                    auto const &prev = timing_series[index - 1].buffer;

                    ImPlotSpec prev_spec;
                    prev_spec.Offset = prev.offset;
                    prev_spec.Stride = sizeof(ImVec2);

                    ImPlot::PlotShaded(label.c_str(), &buf.data[0].x, &buf.data[0].y, &prev.data[0].y,
                                       static_cast<int>(buf.data.size()), prev_spec);
                }
            }

            ImPlot::EndPlot();
        }

        if (auto const graph_timings = renderer->frame_graph_timings();
            !graph_timings.empty() &&
            ImGui::BeginTable("Frame graph", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Pass");
            ImGui::TableSetupColumn("Id");
            ImGui::TableSetupColumn("Queue");
            ImGui::TableSetupColumn("GPU (ms)");
            ImGui::TableHeadersRow();

            for (auto const &timing: graph_timings) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.label.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.name_id.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.queue == frame_graph::LogicalQueue::graphics ? "graphics" : "compute");
                ImGui::TableNextColumn();
                if (timing.milliseconds) {
                    ImGui::Text("%.3f", static_cast<double>(*timing.milliseconds));
                } else {
                    ImGui::TextUnformatted("-");
                }
            }

            ImGui::EndTable();
        }

        auto const &overlay_timings = renderer->last_frame_timings().overlays;

        if (!overlay_timings.empty() &&
            ImGui::BeginTable("Overlay timings", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Overlay");
            ImGui::TableSetupColumn("Stage");
            ImGui::TableSetupColumn("Prepare (ms)");
            ImGui::TableSetupColumn("Record (ms)");
            ImGui::TableHeadersRow();

            for (auto const &timing: overlay_timings) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.name.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(timing.stage == OverlayStage::scene ? "scene" : "ui");
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", static_cast<double>(timing.prepare_milliseconds));
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", static_cast<double>(timing.record_milliseconds));
            }

            ImGui::EndTable();
        }
    });

    panel("Frame graph", panels.frame_graph, [&] {
        if (ImGui::BeginTabBar("##frame_graph_tabs")) {
            if (ImGui::BeginTabItem("Graph")) {
                frame_graph_editor.draw(*renderer, stage_registry);
                ImGui::EndTabItem();
            }
            if (ImGui::BeginTabItem("Timeline")) {
                gui::draw_pass_timeline(*renderer);
                ImGui::EndTabItem();
            }
            ImGui::EndTabBar();
        }
    });

    auto const draw_rendering_settings = [&] {
        auto const draw_stage = [&](char const *id) {
            auto const *stage = stage_registry.find(FlyString{id});
            ImGui::SeparatorText(stage->title.data());
            stage_registry.draw_settings(stage->id, *renderer);
        };

        draw_stage("msaa");

        draw_stage("debug_overlays");

        draw_stage("occlusion_culling");
        draw_stage("clustered_lighting");

        draw_stage("shadows");
        draw_stage("ambient_occlusion");
        draw_stage("bloom");
    };

    auto const draw_lighting_settings = [&] {
        ImGui::SeparatorText("Punctual lights");
        auto &registry = active_scene()->get_registry();

        std::vector<entt::entity> light_rows;
        for (auto const entity: registry.view<Components::Transform, Components::PointLight>()) {
            light_rows.push_back(entity);
        }
        auto const point_light_count = light_rows.size();
        for (auto const entity: registry.view<Components::Transform, Components::SpotLight>()) {
            light_rows.push_back(entity);
        }

        ImGui::TextDisabled("%zu point, %zu spot", point_light_count, light_rows.size() - point_light_count);

        auto &selection = selection_context();
        auto const selected = selection.primary();

        constexpr std::size_t visible_light_rows = 10;
        auto const list_height = static_cast<float>(std::clamp<std::size_t>(light_rows.size(), 1, visible_light_rows)) *
                                         ImGui::GetTextLineHeightWithSpacing() +
                                 (ImGui::GetStyle().WindowPadding.y * 2.0F);

        if (ImGui::BeginChild("##punctual_lights", ImVec2{0.0F, list_height}, ImGuiChildFlags_Borders)) {
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(light_rows.size()));

            while (clipper.Step()) {
                for (auto row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                    auto const row_index = static_cast<std::size_t>(row);
                    auto const entity = light_rows[row_index];
                    auto const label = std::format("{} ({})", entity_display_name(registry, entity),
                                                   row_index < point_light_count ? "point" : "spot");

                    ImGui::PushID(static_cast<int>(entt::to_integral(entity)));
                    if (ImGui::Selectable(label.c_str(), entity == selected)) {
                        selection.select(entity);
                    }
                    ImGui::PopID();
                }
            }
        }
        ImGui::EndChild();

        if (registry.valid(selected)) {
            auto const edit_light = [&]<typename LightT>(LightT &punctual_light, auto &&draw_light) {
                auto &transform = registry.get<Components::Transform>(selected);

                ImGui::TextUnformatted(entity_display_name(registry, selected).c_str());
                bool changed = ImGui::DragFloat3("Position", &transform.position.x, 0.1F);
                changed |= draw_light(punctual_light);

                if (changed) {
                    registry.patch<LightT>(selected);
                }
            };

            if (registry.all_of<Components::Transform>(selected)) {
                if (auto *point_light = registry.try_get<Components::PointLight>(selected)) {
                    edit_light(*point_light, draw_point_light);
                } else if (auto *spot_light = registry.try_get<Components::SpotLight>(selected)) {
                    edit_light(*spot_light, draw_spot_light);
                }
            }
        }
    };

    auto const draw_environment_settings = [&] {
        (void) gui::draw_environment_panel(active_scene()->environment, renderer->environment_system(), model_browser,
                                           browsing_environment);
    };

    // Project-wide configuration (rendering, lights, environment) lives in one large window rather than in docked
    // panels. It is a normal window, not a modal, so the viewport stays live and the environment's file browser works.
    if (panels.project_settings) {
        auto const *const viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowSize(ImVec2{viewport->Size.x * 0.6F, viewport->Size.y * 0.75F}, ImGuiCond_Appearing);
        ImGui::SetNextWindowPos(viewport->GetCenter(), ImGuiCond_Appearing, ImVec2{0.5F, 0.5F});

        if (ImGui::Begin("Project Settings", &panels.project_settings, ImGuiWindowFlags_NoDocking)) {
            constexpr std::array<char const *, 3> pages{"Rendering", "Lighting", "Environment"};

            if (ImGui::BeginChild("##project_settings_nav", ImVec2{170.0F, 0.0F}, ImGuiChildFlags_Borders)) {
                for (auto const [index, name]: std::views::enumerate(pages)) {
                    if (ImGui::Selectable(name, project_settings_page == static_cast<int>(index))) {
                        project_settings_page = static_cast<int>(index);
                    }
                }
            }
            ImGui::EndChild();

            ImGui::SameLine();

            if (ImGui::BeginChild("##project_settings_page", ImVec2{0.0F, 0.0F})) {
                switch (project_settings_page) {
                    case 0:
                        draw_rendering_settings();
                        break;
                    case 1:
                        draw_lighting_settings();
                        break;
                    default:
                        draw_environment_settings();
                        break;
                }
            }
            ImGui::EndChild();
        }
        ImGui::End();
    }

    draw_scene_file_ui();

    if (auto const picked = model_browser.draw(editor_icons.get()); picked && browsing_environment) {
        browsing_environment = false;
        active_scene()->environment.hdr_source = gui::path_to_utf8(*picked);
    } else if (picked) {
        pending_model_pick = *picked;
    }

    if (auto const picked = std::exchange(pending_model_pick, std::nullopt)) {
        switch (model_browse_target) {
            case ModelBrowseTarget::spawn_entity:
                spawn_streamed_model(*picked);
                break;

            case ModelBrowseTarget::inspector: {
                auto &registry = active_scene()->get_registry();
                if (registry.valid(model_browse_entity) && registry.all_of<Components::Model>(model_browse_entity)) {
                    auto const model = renderer->model_streamer().request(
                            *renderer, AssetPath::from_user(*picked).value_or(AssetPath::missing()), engine_models.cube, FlyString{gui::path_to_utf8(picked->filename())});
                    set_entity_model(registry, model_browse_entity, model);
                }
                break;
            }
        }
        model_browse_entity = entt::null;
    }

    // Toasts go in the viewport while it is showing, so they are where the work is.
    if (viewport_seen_frame == ImGui::GetFrameCount() && viewport_content_size.x > 0.0F &&
        viewport_content_size.y > 0.0F) {
        gui::set_toast_anchor(viewport_screen_pos.x, viewport_screen_pos.y, viewport_content_size.x,
                              viewport_content_size.y);
    } else {
        gui::clear_toast_anchor();
    }

    gui::render_toasts();
}

auto Application::request_import_model() -> void {
    model_browse_target = ModelBrowseTarget::spawn_entity;
    model_browser.open("Load Model", model_file_filters());
}

auto Application::spawn_streamed_model(std::filesystem::path const &path) -> void {
    auto file_name = gui::path_to_utf8(path.filename());
    auto const model = renderer->model_streamer().request(*renderer, AssetPath::from_user(path).value_or(AssetPath::missing()),
                                                       engine_models.cube, FlyString{file_name});

    auto entity = Entity{active_scene(), gui::path_to_utf8(path.stem())};
    entity.emplace<Components::Transform>();
    entity.emplace<Components::Model>(Components::Model{.model = model});
    entity.emplace<Components::StreamedModelTag>();
    selection_context().select(entity);

    model_loads.push_back(StreamedModelLoad{
            .scene = active_scene(),
            .entity = entity,
            .model = model,
            .file_name = std::move(file_name),
    });

    while (model_loads.size() > max_listed_model_loads) {
        auto const settled = std::ranges::find_if(model_loads, &StreamedModelLoad::settled);
        if (settled == model_loads.end()) {
            break;
        }
        model_loads.erase(settled);
    }
}

auto Application::update_model_loads() -> void {
    auto &streamer = renderer->model_streamer();

    for (auto &load: model_loads) {
        if (load.settled) {
            continue;
        }

        auto const state = streamer.state(load.model);
        if (state == ModelRequestState::loading) {
            continue;
        }

        load.settled = true;

        if (state == ModelRequestState::failed) {
            load.status = std::format("Failed to load '{}': {}. Showing the placeholder cube.", load.file_name,
                                      streamer.failure_reason(load.model));
            gui::toast_error("Model failed to load", load.status);
            continue;
        }

        if (load.model == engine_models.cube) {
            load.status = std::format("Could not reserve a model slot for '{}'. Showing the placeholder cube.",
                                      load.file_name);
            gui::toast_warn("Model not loaded", load.status);
            continue;
        }

        load.status = std::format("Loaded '{}'", load.file_name);
        gui::toast_success("Model loaded", load.status);

        if (load.scene != active_scene()) {
            continue;
        }
        auto &registry = load.scene->get_registry();
        if (!registry.valid(load.entity) || registry.all_of<Components::RigidBody>(load.entity)) {
            continue;
        }
        auto const *component = registry.try_get<Components::Model>(load.entity);
        if (component == nullptr || component->model != load.model) {
            continue;
        }
        if (auto const submesh_bounds = renderer->model_submesh_bounds(load.model)) {
            registry.emplace<Components::RigidBody>(load.entity,
                                                    Components::RigidBody::from_submesh_boxes(*submesh_bounds));
        }
    }
}

auto Application::set_entity_model(entt::registry &registry, entt::entity entity, ModelHandle model) -> void {
    auto const previous = registry.get<Components::Model>(entity).model;
    bool const owned_previous = registry.all_of<Components::StreamedModelTag>(entity);

    registry.patch<Components::Model>(entity, [&](Components::Model &component) { component.model = model; });
    registry.emplace_or_replace<Components::StreamedModelTag>(entity);

    if (owned_previous) {
        renderer->release_model(previous);
    }

    renderer->mark_dynamic_shadow_casters_dirty();
}

auto Application::on_play_starting() -> void {
    selection_context().clear();
    renaming_entity = entt::null;
    inspector_name_dirty = false;
}

auto Application::on_play_stopping() -> void {
    selection_context().clear();
    renaming_entity = entt::null;
    inspector_name_dirty = false;

    std::erase_if(model_loads, [&](StreamedModelLoad const &load) { return load.scene == runtime_scene.get(); });
}

auto Application::create_ui_resources() -> void {
    editor_icons = std::make_unique<gui::EditorIcons>(*renderer);
    renderer->set_editor_grid(true);
    remove_window_decorations();
}

auto Application::on_source_scene_replaced() -> void { mark_source_scene_clean(); }

auto Application::on_scene_opening() -> void {
    selection_context().clear();
    model_loads.clear();
}

auto Application::on_shutdown() -> void {
    scene_save_job.reset();

    Runtime::on_shutdown();
}

auto Application::editing_camera(float aspect_ratio) const -> CameraParams {
    return CameraParams{
            .view = viewports[0].camera.view(),
            .projection = viewports[0].camera.projection(aspect_ratio),
            .near_clip = viewports[0].camera.near_clip(),
            .far_clip = viewports[0].camera.far_clip(),
            .vertical_fov_radians = glm::radians(viewports[0].camera.field_of_view_degrees()),
    };
}

auto Application::camera_pivot() const -> glm::vec3 {
    auto const &camera = active_camera();
    auto const &registry = active_scene()->get_registry();
    auto const selected = selection_context().primary();

    if (selected != entt::null && registry.valid(selected) && registry.all_of<Components::Transform>(selected)) {
        return registry.get<Components::Transform>(selected).position;
    }

    return camera.position() + (camera.forward() * camera_pivot_distance);
}

auto Application::focus_selection() -> void {
    auto &camera = active_camera();
    auto const &registry = active_scene()->get_registry();
    auto const selected = selection_context().primary();

    if (selected == entt::null || !registry.valid(selected) || !registry.all_of<Components::Transform>(selected)) {
        return;
    }

    auto const &transform = registry.get<Components::Transform>(selected);
    auto const extent = std::max({std::abs(transform.scale.x), std::abs(transform.scale.y), std::abs(transform.scale.z)});

    camera_pivot_distance = std::max(3.0F, extent * 3.0F);
    camera.look_at(transform.position - (camera.forward() * camera_pivot_distance), transform.position);
}

auto Application::pre_update(float delta_time) -> void {
    for (auto &pane: viewports) {
        pane.camera.update(std::min(delta_time, 0.1F));
    }
}

auto Application::update(float delta_time) -> void {
    ZoneScopedNC("ApplicationUpdate", tracy::Color::Firebrick);

    last_delta_time = delta_time;

    update_model_loads();
    update_scene_save();
    update_package_jobs();

    // Packaging pauses the simulation; rendering and the modal progress popup keep running.
    if (packaging_active()) {
        update_scene_load();
        return;
    }

    std::erase_if(pending_deletions, [this](PendingDeletion &pending) {
        if (elapsed_time < pending.delete_at) {
            return false;
        }
        pending.commit();
        return true;
    });

    Runtime::update(delta_time);
}

auto Application::handle_event(KeyPressedEvent ev) -> bool {
    if (packaging_active()) {
        return true;
    }

    if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL && !scene_load_job.has_value()) {
        renderer->queue_render_thread_event([this] {
            if (is_playing) {
                stop();
            }

            game->on_populate(*source_scene, *renderer, engine_models);
            scene_path.clear();
            scene_pack.reset();
            mark_source_scene_clean();
        });
    }
    if (ev.key == GLFW_KEY_S && (ev.modifiers & GLFW_MOD_CONTROL) != 0) {
        start_save_scene((ev.modifiers & GLFW_MOD_SHIFT) != 0 ? std::filesystem::path{} : scene_path);
    }
    if (ev.key == GLFW_KEY_N && ev.modifiers == GLFW_MOD_CONTROL) {
        request_switch_game("empty");
    }
    if (ev.key == GLFW_KEY_O && ev.modifiers == GLFW_MOD_CONTROL && !scene_browser.is_open()) {
        scene_browser.open("Open Scene", {{.label = "Lathe scenes (*.lbf)", .extensions = {".lbf"}}});
    }

    if (!is_playing) {
        if (!ImGui::GetIO().WantCaptureKeyboard) {
            switch (ev.key) {
                case GLFW_KEY_1:
                    gizmo_operation = ImGuizmo::TRANSLATE;
                    break;
                case GLFW_KEY_2:
                    gizmo_operation = ImGuizmo::ROTATE;
                    break;
                case GLFW_KEY_3:
                    gizmo_operation = ImGuizmo::SCALE;
                    break;
                case GLFW_KEY_4:
                    gizmo_mode = gizmo_mode == ImGuizmo::WORLD ? ImGuizmo::LOCAL : ImGuizmo::WORLD;
                    break;
                case GLFW_KEY_F:
                    if (ev.modifiers == 0) {
                        focus_selection();
                    }
                    break;
                case GLFW_KEY_T:
                    text_placement.set_active(!text_placement.active());
                    break;
                case GLFW_KEY_ESCAPE:
                    text_placement.set_active(false);
                    break;
                default:
                    break;
            }
        }

        // F1..F8 pick the view mode of the viewport the pointer is in.
        auto const function_key = ev.key - GLFW_KEY_F1;
        if (function_key >= 0 && std::cmp_less(function_key, view_mode_names.size()) &&
            !ImGui::GetIO().WantTextInput) {
            renderer->set_view_mode(viewports[active_viewport].view,
                                    view_mode_names[static_cast<std::size_t>(function_key)].first);
            return true;
        }

        active_camera().on_key_pressed(ev.key);
    }

    return false;
}

auto Application::handle_event(KeyReleasedEvent ev) -> bool {
    if (!is_playing) {
        // Both, so a key let go after the pointer moved to the other viewport does not leave the first one moving.
        for (auto &pane: viewports) {
            pane.camera.on_key_released(ev.key);
        }
    }

    return false;
}

auto Application::handle_event(MouseMovedEvent ev) -> bool {
    if (!is_playing) {
        auto &camera = active_camera();
        auto const dx = static_cast<float>(ev.delta_x);
        auto const dy = static_cast<float>(ev.delta_y);

        if (camera_drag == CameraDrag::orbit) {
            camera.orbit(dx, dy, camera_pivot());
        } else if (camera_drag == CameraDrag::pan) {
            camera.pan(dx, dy, glm::length(camera_pivot() - camera.position()));
        } else {
            camera.on_mouse_moved(dx, dy, mouse_dragging);
        }
    }

    return false;
}

auto Application::handle_event(MouseScrolledEvent ev) -> bool {
    if (packaging_active()) {
        return true;
    }

    auto &camera = active_camera();

    if (!is_playing && ImGui::GetIO().KeyCtrl) {
        auto const distance = glm::length(camera_pivot() - camera.position());

        camera.dolly(static_cast<float>(ev.delta_y) * std::max(distance, 1.0F) * 0.1F);
    } else {
        camera.on_mouse_scrolled(static_cast<float>(ev.delta_y));
    }

    return false;
}

auto Application::handle_event(MouseButtonPressedEvent ev) -> bool {
    if (packaging_active()) {
        return true;
    }

    if (!is_playing && ev.button == GLFW_MOUSE_BUTTON_LEFT && viewport_hovered && !ImGuizmo::IsOver() &&
        !ImGuizmo::IsUsing()) {
        if ((ev.modifiers & GLFW_MOD_ALT) != 0) {
            camera_drag = CameraDrag::orbit;
        } else if ((ev.modifiers & GLFW_MOD_CONTROL) != 0) {
            camera_drag = CameraDrag::pan;
        }
    }

    return false;
}

auto Application::handle_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_LEFT) {
        camera_drag = CameraDrag::none;
    }

    return false;
}
