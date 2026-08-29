#include "application.hxx"

#include <csignal>
#include <memory>
#include <random>
#include <ranges>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <future>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include <entt/entt.hpp>

#include "allocator.hxx"
#include "components.hxx"
#include "config.hxx"
#include "context.hxx"
#include "debug_renderer.hxx"
#include "demo_scene.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "imgui_renderer.hxx"
#include "implot.h"
#include "logger.hxx"
#include "memory_tracking_ui.hxx"
#include "physics.hxx"
#include "physics_world.hxx"
#include "player_camera.hxx"
#include "player_controller.hxx"
#include "renderdoc.hxx"
#include "renderer.hxx"
#include "renderer_application_policy.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "swapchain.hxx"

namespace {

    constexpr auto draw_point_light = [](Components::PointLight &point_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &point_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &point_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &point_light.range, 0.5F, 100.0F);
        return changed;
    };

    constexpr auto draw_spot_light = [](Components::SpotLight &spot_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &spot_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &spot_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &spot_light.range, 0.5F, 100.0F);
        changed |= ImGui::SliderFloat("Inner cone", &spot_light.inner_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        changed |= ImGui::SliderFloat("Outer cone", &spot_light.outer_cone_degrees, 0.0F, 89.0F, "%.1f deg");
        return changed;
    };

    constexpr auto draw_rows = [](auto &index, entt::registry &registry, auto &&view, auto &&draw_light) {
        for (auto [entity, transform, light, meta]: view.each()) {
            ImGui::PushID(static_cast<int>(index++));
            if (ImGui::TreeNode(meta.name.c_str())) {
                bool changed = ImGui::DragFloat3("Position", &transform.position.x, 0.1F);
                changed |= draw_light(light);

                if (changed) {
                    using LightT = std::decay_t<decltype(light)>;
                    registry.patch<LightT>(entity);
                }

                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    };

    constexpr auto widget = [](const std::string_view name, auto &&f) -> bool {
        if (!ImGui::Begin(name.data())) {
            ImGui::End();
            return false;
        }

        f();

        ImGui::End();
        return false;
    };
} // namespace


Application::Application(VulkanContext &ctx) noexcept :
    context(ctx), renderer(std::make_unique<Renderer>(context)),
    debug_renderer(std::make_unique<debug_draw::DebugRenderer>(*renderer)) {
    timing_buffers.fill(ScrollingBuffer{600});
}

Application::~Application() { shader_watcher_.stop(); }


auto Application::on_ui() -> void {
    on_memory_ui();

    widget("Simulation", [&] {
        if (is_playing) {
            if (ImGui::Button("Stop")) {
                stop();
            }
        } else {
            if (ImGui::Button("Play")) {
                play();
            }
        }
    });

    widget("Scene stats", [&] {
        auto const &stats = renderer->last_frame_stats();
        auto const &pipeline_stats = renderer->last_frame_pipeline_stats();

        // Helper lambda to get a formatted string in-line
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

        auto &&[assembled_vertex_count, assembled_primitive_count, clipped_primitive_count,
                fragment_shader_invocation_count, valid] = pipeline_stats;

        if (valid) {
            ImGui::Text("Triangles assembled (post-cull): %s (%llu)", fmt(pipeline_stats.assembled_primitive_count),
                        static_cast<unsigned long long>(pipeline_stats.assembled_primitive_count));
            ImGui::Text("Triangles rendered (post-clip): %s (%llu)", fmt(pipeline_stats.clipped_primitive_count),
                        static_cast<unsigned long long>(pipeline_stats.clipped_primitive_count));
            ImGui::Text("Vertices assembled (post-cull): %s (%llu)", fmt(pipeline_stats.assembled_vertex_count),
                        static_cast<unsigned long long>(pipeline_stats.assembled_vertex_count));
            ImGui::Text("Fragment shader invocations: %s (%llu)", fmt(pipeline_stats.fragment_shader_invocation_count),
                        static_cast<unsigned long long>(pipeline_stats.fragment_shader_invocation_count));
        } else {
            ImGui::TextDisabled("Pipeline stats not yet available");
        }

        ImGui::Text("Triangles submitted (pre-cull): %s (%u)", fmt(stats.submitted_triangle_count),
                    stats.submitted_triangle_count);
        ImGui::Text("Draw calls: %u  (opaque %u / mask %u / blend %u)", stats.indirect_command_count,
                    stats.opaque_indirect_count, stats.mask_indirect_count, stats.blend_indirect_count);
        ImGui::Text("Instances submitted: %s (%u)", fmt(stats.submitted_instance_count),
                    stats.submitted_instance_count);
        ImGui::Text("Model / mesh submissions: %u / %u", stats.model_submission_count, stats.mesh_submission_count);
        ImGui::Text("Lights: %u point / %u spot", stats.point_light_count, stats.spot_light_count);
    });

    widget("Frame timings", [&] {
        if (ImPlot::BeginPlot("Stage timings (cumulative ms)", ImVec2(-1, 250))) {
            ImPlot::SetupAxes("Frame", "ms", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisLimits(ImAxis_X1, timing_x - 600.0, timing_x, ImGuiCond_Always);

            constexpr auto first_stage = static_cast<std::uint32_t>(RenderStage::Culling);

            for (std::uint32_t stage = first_stage; stage < stage_count; ++stage) {
                auto const &buf = timing_buffers[stage];

                if (buf.data.empty()) {
                    continue;
                }

                ImPlotSpec spec;
                spec.Offset = buf.offset;
                spec.Stride = sizeof(ImVec2);
                spec.FillAlpha = 0.35F;

                if (stage == first_stage) {
                    ImPlot::PlotShaded(to_string(static_cast<RenderStage>(stage)).data(), &buf.data[0].x,
                                       &buf.data[0].y, static_cast<int>(buf.data.size()), 0.0, spec);
                } else {
                    auto const &prev = timing_buffers[stage - 1];

                    ImPlotSpec prev_spec;
                    prev_spec.Offset = prev.offset;
                    prev_spec.Stride = sizeof(ImVec2);

                    ImPlot::PlotShaded(to_string(static_cast<RenderStage>(stage)).data(), &buf.data[0].x,
                                       &buf.data[0].y, &prev.data[0].y, static_cast<int>(buf.data.size()), prev_spec);
                }
            }

            ImPlot::EndPlot();
        }
    });

    widget("Lighting", [&] {
        ImGui::SeparatorText("Debug");
        bool draw_light_icons = renderer->debug_draw_light_icons();
        if (ImGui::Checkbox("Draw light icons", &draw_light_icons)) {
            renderer->set_debug_draw_light_icons(draw_light_icons);
        }

        auto light = renderer->directional_light();
        auto shadows = renderer->shadow_settings();
        bool dirty = false;

        dirty |= ImGui::SliderFloat("Azimuth", &light_azimuth_degrees, -180.0F, 180.0F, "%.1f deg");
        // Clamped so the light never goes horizontal -- that
        // degenerates the ortho depth range in shadow_cascades.cxx.
        dirty |= ImGui::SliderFloat("Elevation", &light_elevation_degrees, 5.0F, 89.0F, "%.1f deg");
        dirty |= ImGui::ColorEdit3("Colour", &light.colour.x);
        dirty |= ImGui::SliderFloat("Intensity", &light.intensity, 0.0F, 10.0F);

        float ambient_intensity = renderer->ambient_intensity();
        if (ImGui::SliderFloat("Ambient intensity", &ambient_intensity, 0.0F, 1.0F)) {
            renderer->set_ambient_intensity(ambient_intensity);
        }

        ImGui::SeparatorText("Shadows");
        dirty |= ImGui::SliderFloat("Split lambda", &shadows.cascades.split_lambda, 0.0F, 1.0F);
        dirty |= ImGui::SliderFloat("Shadow distance", &shadows.cascades.shadow_distance, 20.0F, 500.0F);
        dirty |= ImGui::SliderFloat("PCF radius", &shadows.pcf_radius_texels, 0.5F, 4.0F);
        dirty |= ImGui::SliderFloat("Normal offset", &shadows.normal_offset_texels, 0.0F, 8.0F);
        dirty |= ImGui::SliderFloat("Depth bias", &shadows.depth_bias_world, 0.0F, 0.5F);
        dirty |= ImGui::SliderFloat("Bias slope", &shadows.depth_bias_slope, -8.0F, 0.0F);
        dirty |= ImGui::Checkbox("Cascade tint", &shadows.debug_cascade_tint);

        if (dirty) {
            auto const azimuth = glm::radians(light_azimuth_degrees);
            auto const elevation = glm::radians(light_elevation_degrees);

            light.direction = glm::normalize(glm::vec3{
                    std::cos(elevation) * std::cos(azimuth),
                    std::sin(elevation),
                    std::cos(elevation) * std::sin(azimuth),
            });

            renderer->set_directional_light(light);
            renderer->set_shadow_settings(shadows);
        }

        ImGui::SeparatorText("Punctual lights");
        auto &registry = active_scene()->get_registry();
        std::size_t index = 0;
        draw_rows(index, registry,
                  registry.view<Components::Transform, Components::PointLight, Components::GeneratedMeta>(),
                  draw_point_light);
        draw_rows(index, registry, registry.view<Components::Transform, Components::PointLight, Components::Meta>(),
                  draw_point_light);
        draw_rows(index, registry, registry.view<Components::Transform, Components::SpotLight, Components::Meta>(),
                  draw_spot_light);
        draw_rows(index, registry,
                  registry.view<Components::Transform, Components::SpotLight, Components::GeneratedMeta>(),
                  draw_spot_light);
    });
}

auto Application::play() -> void {
    runtime_scene = std::make_unique<Scene>(*renderer);
    runtime_scene->physics_settings = editor_scene->physics_settings;
    clone_registry<Components::Transform, Components::Model, Components::RigidBody, Components::MaterialOverride,
                   Components::PlayerTag, Components::Lifetime, Components::PointLight, Components::SpotLight,
                   Components::GeneratedMeta, Components::Meta>(editor_scene->get_registry(),
                                                                runtime_scene->get_registry());

    // is_playing flips first so active_scene() already resolves to
    // runtime_scene for on_scene_start()/attach_debug_renderer() below.
    is_playing = true;
    active_scene()->on_scene_start();
    active_scene()->attach_debug_renderer(*debug_renderer);

    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    has_last_mouse_position = false;
}

auto Application::stop() -> void {
    active_scene()->detach_debug_renderer();
    debug_renderer->clear_lines();
    active_scene()->on_scene_stop();
    // Flips active_scene() over to editor_scene immediately -- nothing below
    // this line touches the runtime scene, so runtime_scene.reset() is safe.
    is_playing = false;
    runtime_scene.reset();

    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
}

auto Application::update(float delta_time) -> void {
    ZoneScopedNC("ApplicationUpdate", tracy::Color::Firebrick);

    if (!is_playing) {
        return;
    }

    active_scene()->step(delta_time);

    systems::player_movement(active_scene()->get_registry(), *active_scene()->physics_world, player_entity,
                             player_controller, player_camera, delta_time);
    systems::lifetime(active_scene()->get_registry(), *active_scene()->physics_world, delta_time);
}
auto Application::on_startup() -> void {

    std::array const shader_directories{
            std::filesystem::path{"assets/shaders"},
    };
    if (!shader_watcher_.start(*renderer, shader_directories)) {
        error("Shader hot-reload watcher failed to start -- shaders will not live-reload this run");
    }
    imgui_renderer = std::make_unique<gui::ImGuiRenderer>(
            *renderer, gui::FontChoice{
                               .font_path = "assets/fonts/GoogleSansCode-Regular.ttf",
                               .size = 12,
                       });
    renderer->queue_render_thread_event([this] {
        auto models = create_engine_models(*renderer);

        if (!models) {
            error("Fatal: could not create built-in engine models: {}", describe(models.error()));

            context.running.store(false, std::memory_order_release);
            glfwPostEmptyEvent();

            return;
        }

        engine_models = *models;

        build_demo_scene(*this);
    });
}

auto Application::on_event(KeyPressedEvent ev) -> bool {
    if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL) {
        renderer->queue_render_thread_event([this] { build_demo_scene(*this); });
    }
    if (ev.key == GLFW_KEY_F12) {
        renderer->request_screenshot();
    }

    if (is_playing) {
        player_controller.on_key_pressed(ev.key);
    } else {
        camera.on_key_pressed(ev.key);
    }

    return true;
}
auto Application::on_event(KeyReleasedEvent ev) -> bool {
    if (is_playing) {
        player_controller.on_key_released(ev.key);
    } else {
        camera.on_key_released(ev.key);
    }

    return true;
}
auto Application::on_event(MouseMovedEvent ev) -> bool {
    if (is_playing) {
        player_controller.on_mouse_moved(static_cast<float>(ev.delta_x), static_cast<float>(ev.delta_y),
                                         /*look_enabled=*/true);
    } else {
        camera.on_mouse_moved(static_cast<float>(ev.delta_x), static_cast<float>(ev.delta_y), mouse_dragging);
    }

    return true;
}
auto Application::on_event(MouseScrolledEvent ev) -> bool {
    camera.on_mouse_scrolled(static_cast<float>(ev.delta_y));

    return true;
}
auto Application::on_event(MouseButtonPressedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = true;
    }

    if (ev.button == GLFW_MOUSE_BUTTON_LEFT) {
        if (ev.modifiers & GLFW_MOD_CONTROL) {
            shoot_bullet(12);
        } else {
            shoot_bullet();
        }
    }

    return true;
}
auto Application::on_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = false;
    }

    return true;
}

[[nodiscard]] auto Application::cursor_ray() const -> std::pair<glm::vec3, glm::vec3> {
    int window_width = 0;
    int window_height = 0;
    glfwGetWindowSize(context.window, &window_width, &window_height);

    auto const framebuffer_width = context.framebuffer_width.load(std::memory_order_relaxed);
    auto const framebuffer_height = context.framebuffer_height.load(std::memory_order_relaxed);

    if (window_width <= 0 || window_height <= 0 || framebuffer_width <= 0 || framebuffer_height <= 0) {
        return {camera.position(), camera.forward()};
    }

    auto const ndc_x = (2.0F * static_cast<float>(last_mouse_x)) / static_cast<float>(window_width) - 1.0F;
    auto const ndc_y = 1.0F - (2.0F * static_cast<float>(last_mouse_y)) / static_cast<float>(window_height);
    auto const aspect_ratio = static_cast<float>(framebuffer_width) / static_cast<float>(framebuffer_height);

    auto view_projection =
            is_playing ? player_camera.view_projection(aspect_ratio) : this->camera.view_projection(aspect_ratio);
    auto const inverse_view_projection = glm::inverse(std::move(view_projection));

    auto const unproject = [&](float ndc_z) {
        auto const clip_space_point = inverse_view_projection * glm::vec4{ndc_x, ndc_y, ndc_z, 1.0F};
        return glm::vec3{clip_space_point} / clip_space_point.w;
    };

    auto const ray_origin = unproject(0.0F);
    auto const ray_direction = glm::normalize(unproject(1.0F) - ray_origin);

    return {ray_origin, ray_direction};
}
auto Application::shoot_bullet(std::size_t n) -> void {
    if (!is_playing || !active_scene()->physics_world) {
        return;
    }

    if (!active_scene()->get_registry().valid(player_entity) ||
        !active_scene()->get_registry().all_of<Components::Transform>(player_entity)) {
        return;
    }

    constexpr auto bullet_half_extent = 0.15F;
    constexpr auto bullet_speed = 40.0F;
    constexpr auto bullet_mass = 0.2F;
    constexpr auto bullet_lifetime_seconds = 3.0F;
    constexpr auto max_aim_distance = 1000.0F;
    constexpr auto player_eye_height = 1.5F; // Height offset from player base position

    auto const &position = ReadOnlyEntity{active_scene(), this->player_entity}.get<Components::Transform>().position;
    auto const player_position = position; // Replace with your player position accessor
    auto const muzzle_position = player_position + glm::vec3{0.0F, player_eye_height, 0.0F};

    auto const cam_origin = player_camera.position();
    auto const cam_forward = player_camera.forward();

    glm::vec3 target_point = cam_origin + (cam_forward * max_aim_distance);

    if (auto const hit = active_scene()->physics_world->raycast(cam_origin, cam_forward, max_aim_distance)) {
        target_point = hit->point;
    }

    auto const bullet_direction = glm::normalize(target_point - muzzle_position);

    auto const transform = Components::Transform{
            .position = muzzle_position + bullet_direction * (bullet_half_extent + 0.2F),
            .scale = glm::vec3{bullet_half_extent} / cube_half_extents,
    };
    auto const rigid_body = Components::RigidBody{
            .velocity = bullet_direction * bullet_speed,
            .half_extents = glm::vec3{bullet_half_extent},
            .restitution = 0.3F,
            .mass = bullet_mass,
    };

    for (auto i = 0U; i < n; ++i) {
        auto const entity = GeneratedEntity{active_scene(), "bullet_{}", static_cast<std::uint32_t>(i)};
        entity.emplace<Components::Transform>(transform);
        entity.emplace<Components::Model>(Components::Model{.model = cube_model});
        entity.emplace<Components::RigidBody>(rigid_body);
        entity.emplace<Components::Lifetime>(bullet_lifetime_seconds);

        active_scene()->physics_world->add_body(active_scene()->get_registry(), entity, transform, rigid_body);
    }
}
