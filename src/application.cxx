#include "application.hxx"

#include <csignal>
#include <memory>
#include <volk.h>

#include <GLFW/glfw3.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <glm/vec2.hpp>
#include <string_view>
#include <type_traits>
#include <vector>

#include <entt/entt.hpp>

#include "allocator.hxx"
#include "components.hxx"
#include "config.hxx"
#include "context.hxx"
#include "debug_renderer.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "imgui_renderer.hxx"
#include "implot.h"
#include "logger.hxx"
#if MINGW_VULKAN_TRACK_MEMORY
#include "memory_tracking_ui.hxx"
#endif
#include "physics.hxx"
#include "physics_world.hxx"
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

        // Dispatched on f's arity rather than overloading widget() itself: a generic
        // lambda has one signature, so which of these f can accept is a compile-time
        // property of f, not something callers select between at the call site.
        if constexpr (std::is_invocable_v<decltype(f), glm::vec2, glm::vec2>) {
            auto const size = ImGui::GetWindowSize();
            auto const position = ImGui::GetWindowPos();
            f(glm::vec2{size.x, size.y}, glm::vec2{position.x, position.y});
        } else if constexpr (std::is_invocable_v<decltype(f), glm::vec2>) {
            auto const size = ImGui::GetWindowSize();
            f(glm::vec2{size.x, size.y});
        } else {
            f();
        }

        ImGui::End();
        return false;
    };
} // namespace


Application::Application(VulkanContext &ctx) noexcept :
    context(ctx), renderer(std::make_unique<Renderer>(context)),
    debug_renderer(std::make_unique<debug_draw::DebugRenderer>(*renderer)) {
    timing_buffers.fill(ScrollingBuffer{600});
}

Application::~Application() {
    // Not a strict safety requirement the way it is for
    // texture_streamer_/model_streamer_ (a chunk-generation task only
    // captures a shared_ptr<TerrainField const>, never Renderer storages,
    // so it can't outlive-and-touch anything this destructor frees) --
    // this just avoids leaving background work running for content nobody
    // will use once shutdown has been decided.
    if (terrain) {
        terrain->wait_all();
    }

    shader_watcher_.stop();
}


auto Application::on_ui() -> void {
#if MINGW_VULKAN_TRACK_MEMORY
    widget("Memory", [] { on_memory_ui(); });
#endif
    widget("Console", [&] { terminal_widget.draw(); });

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

        bool draw_physics_debug = debug_renderer->physics_debug_enabled();
        if (ImGui::Checkbox("Draw physics colliders", &draw_physics_debug)) {
            debug_renderer->set_physics_debug_enabled(draw_physics_debug);
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

        ImGui::SeparatorText("Fog");
        auto fog = renderer->fog_settings();
        bool fog_dirty = false;
        fog_dirty |= ImGui::Checkbox("Enabled", &fog.enabled);
        fog_dirty |= ImGui::ColorEdit3("Fog colour", &fog.colour.x);
        fog_dirty |= ImGui::SliderFloat("Fog extinction", &fog.extinction, 0.0F, 0.02F, "%.4f");
        fog_dirty |= ImGui::SliderFloat("Fog inscattering", &fog.inscattering, 0.0F, 2.0F);
        if (fog_dirty) {
            renderer->set_fog_settings(fog);
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

    // PhysicsWorld is recreated wholesale per run (on_scene_start() above
    // just made a fresh one) -- every terrain collider handle TerrainWorld
    // held from a previous run belonged to a PhysicsWorld that's already
    // gone. GPU slots are untouched; colliders rebind lazily under
    // TerrainWorld's normal per-frame budget.
    if (terrain) {
        terrain->on_physics_world_changed(active_scene()->physics_world.get());
    }

    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    has_last_mouse_position = false;
}

auto Application::stop() -> void {
    // Before on_scene_stop() tears down the runtime PhysicsWorld, so
    // TerrainWorld never holds a handle into an instance that's already
    // been destroyed.
    if (terrain) {
        terrain->on_physics_world_changed(nullptr);
    }

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

    // Residency selection only (no GPU work -- see TerrainWorld::update) --
    // runs regardless of is_playing, so terrain streams in the editor too.
    // Keyed off the player entity's own Transform rather than the follow
    // camera: PlayerCamera springs and gets pulled by wall occlusion, which
    // would otherwise jitter which chunks are considered resident.
    if (terrain) {
        auto camera_xz = glm::vec2{camera.position().x, camera.position().z};

        if (is_playing) {
            auto &registry = active_scene()->get_registry();
            auto view = registry.view<Components::Transform const, Components::PlayerTag const>();

            for (auto &&[entity, transform]: view.each()) {
                camera_xz = glm::vec2{transform.position.x, transform.position.z};
                break;
            }
        }

        terrain->update(camera_xz);
    }

    if (!is_playing) {
        return;
    }

    active_scene()->step(delta_time);

    game->on_update(*active_scene(), delta_time);
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

        game->on_populate(*editor_scene, *renderer, engine_models);

        if (auto terrain_info = game->terrain_create_info(*renderer)) {
            // create_engine_models() above already went through this same
            // one-time-submit path internally (via create_model_from_cpu_data),
            // so nesting another one here -- still on the render thread,
            // still serialized through queue_render_thread_event -- is safe.
            renderer->context().one_time_submit([this, info = *terrain_info](VkCommandBuffer command_buffer) {
                auto world = TerrainWorld::create(*renderer, command_buffer, info);

                if (!world) {
                    error("Fatal: could not create TerrainWorld: {}", world.error().message);

                    context.running.store(false, std::memory_order_release);
                    glfwPostEmptyEvent();

                    return;
                }

                terrain = std::make_unique<TerrainWorld>(std::move(*world));
            });
        }
    });
}

auto Application::on_event(KeyPressedEvent ev) -> bool {
    if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL) {
        renderer->queue_render_thread_event([this] { game->on_populate(*editor_scene, *renderer, engine_models); });
    }
    if (ev.key == GLFW_KEY_F12) {
        renderer->request_screenshot();
    }

    if (is_playing) {
        game->on_key_pressed(*active_scene(), ev);
    } else {
        camera.on_key_pressed(ev.key);
    }

    return true;
}
auto Application::on_event(KeyReleasedEvent ev) -> bool {
    if (is_playing) {
        game->on_key_released(*active_scene(), ev);
    } else {
        camera.on_key_released(ev.key);
    }

    return true;
}
auto Application::on_event(MouseMovedEvent ev) -> bool {
    if (is_playing) {
        game->on_mouse_moved(*active_scene(), ev);
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

    if (is_playing) {
        game->on_mouse_button_pressed(*active_scene(), ev);
    }

    return true;
}
auto Application::on_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = false;
    }

    if (is_playing) {
        game->on_mouse_button_released(*active_scene(), ev);
    }

    return true;
}
