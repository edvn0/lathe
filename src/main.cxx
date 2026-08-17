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
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "imgui_renderer.hxx"
#include "implot.h"
#include "light.hxx"
#include "logger.hxx"
#include "physics.hxx"
#include "physics_world.hxx"
#include "renderdoc.hxx"
#include "renderer.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "swapchain.hxx"

namespace {

    auto direction_to_rotation(glm::vec3 const &direction) -> glm::quat {
        constexpr auto local_forward = glm::vec3{0.0F, -1.0F, 0.0F};
        auto const dot = glm::dot(local_forward, direction);

        if (dot > 0.9999F) {
            return glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        }
        if (dot < -0.9999F) {
            return glm::angleAxis(glm::pi<float>(), glm::vec3{1.0F, 0.0F, 0.0F});
        }

        auto const axis = glm::normalize(glm::cross(local_forward, direction));
        return glm::angleAxis(std::acos(dot), axis);
    }

    constexpr auto draw_point_light = [](PointLight &point_light) -> bool {
        bool changed = false;
        changed |= ImGui::ColorEdit3("Colour", &point_light.colour.x);
        changed |= ImGui::SliderFloat("Intensity", &point_light.intensity, 0.0F, 200.0F);
        changed |= ImGui::SliderFloat("Range", &point_light.range, 0.5F, 100.0F);
        return changed;
    };

    constexpr auto draw_spot_light = [](SpotLight &spot_light) -> bool {
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

    template<typename T>
    concept HasDepth = requires(T t) {
        { t.depth };
    };

    template<typename T>
    concept HasHeight = requires(T t) {
        { t.height };
    };

    template<typename T>
    concept HasWidth = requires(T t) {
        { t.width };
    };

    constexpr auto compare = []<typename A, typename B>(const A &a, const B &b) -> bool {
        if constexpr (HasDepth<A> && HasDepth<B>) {
            return a.width == b.width && a.height == b.height && a.depth == b.depth;
        } else if constexpr ((HasHeight<A> && !HasDepth<A>) && (HasHeight<B> && !HasDepth<B>) ) {
            return a.width == b.width && a.height == b.height;
        } else if constexpr ((HasWidth<A> && !HasHeight<A>) && (HasWidth<B> && !HasHeight<B>) ) {
            return a.width == b.width;
        } else {
            return false;
        }
    };

    struct ScrollingBuffer {
        std::int32_t max_size;
        std::int32_t offset = 0;
        std::vector<ImVec2> data;

        explicit ScrollingBuffer(const std::int32_t m = 600U) : max_size(m) {
            data.reserve(static_cast<std::size_t>(max_size));
        }

        [[gnu::always_inline]]
        constexpr auto size() -> decltype(auto) {
            return data.size();
        }

        auto add_point(float x, float y) -> void {

            if (std::cmp_less(size(), max_size)) {
                data.emplace_back(x, y);
            } else {
                data[static_cast<std::size_t>(offset)] = ImVec2(x, y);
                offset = (offset + 1) % max_size;
            }
        }
    };

#ifndef NDEBUG
    constexpr bool enable_validation = true;

    constexpr std::array validation_layers{
            "VK_LAYER_KHRONOS_validation",
    };
#else
    constexpr bool enable_validation = false;

    constexpr std::array<const char *, 0> validation_layers{};
#endif

    struct KeyPressedEvent {
        std::int32_t key{};
        std::int32_t modifiers{};
    };

    struct KeyReleasedEvent {
        std::int32_t key{};
        std::int32_t modifiers{};
    };

    // Raw cursor delta in pixels since the previous callback -- not an
    // absolute position. Application decides whether it counts as a
    // "look" based on whether a drag is currently active.
    struct MouseMovedEvent {
        double delta_x{};
        double delta_y{};
    };

    struct MouseScrolledEvent {
        double delta_x{};
        double delta_y{};
    };

    struct MouseButtonPressedEvent {
        std::int32_t button{};
        std::int32_t modifiers{};
    };

    struct MouseButtonReleasedEvent {
        std::int32_t button{};
        std::int32_t modifiers{};
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

    // Attach to an entity to force its render material regardless of what
    // its ModelHandle's submeshes normally use -- e.g. making the floor red.
    struct MaterialOverride {
        MaterialHandle material{};
    };

    struct Application {
        explicit Application(VulkanContext &ctx) noexcept :
            context(ctx), renderer(std::make_unique<Renderer>(context)) {
            timing_buffers.fill(ScrollingBuffer{600});
        }
        ~Application() { shader_watcher_.stop(); }

        void on_ui() {
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

                auto &&[assembled_vertex_count, assembled_primitive_count, clipped_primitive_count,
                        fragment_shader_invocation_count, valid] = pipeline_stats;
                if (valid) {
                    ImGui::Text("Triangles assembled (post-cull): %llu",
                                static_cast<unsigned long long>(pipeline_stats.assembled_primitive_count));
                    ImGui::Text("Triangles rendered (post-clip): %llu",
                                static_cast<unsigned long long>(pipeline_stats.clipped_primitive_count));
                    ImGui::Text("Vertices assembled (post-cull): %llu",
                                static_cast<unsigned long long>(pipeline_stats.assembled_vertex_count));
                    ImGui::Text("Fragment shader invocations: %llu",
                                static_cast<unsigned long long>(pipeline_stats.fragment_shader_invocation_count));
                } else {
                    ImGui::TextDisabled("Pipeline stats not yet available");
                }

                ImGui::Text("Triangles submitted (pre-cull): %u", stats.submitted_triangle_count);
                ImGui::Text("Draw calls: %u  (opaque %u / mask %u / blend %u)", stats.indirect_command_count,
                            stats.opaque_indirect_count, stats.mask_indirect_count, stats.blend_indirect_count);
                ImGui::Text("Instances submitted: %u", stats.submitted_instance_count);
                ImGui::Text("Model / mesh submissions: %u / %u", stats.model_submission_count,
                            stats.mesh_submission_count);
                ImGui::Text("Lights: %u point / %u spot", stats.point_light_count, stats.spot_light_count);
            });

            widget("Frame timings", [&] {
                bool frustum_culling_enabled = renderer->frustum_culling_enabled();

                // A/B check: with the camera stationary, toggling this must
                // not change the rendered image at all -- only the depth
                // prepass and forward pass timings. The shadow pass is
                // never affected (see Renderer::frustum_culling_enabled).
                if (ImGui::Checkbox("GPU frustum culling", &frustum_culling_enabled)) {
                    renderer->set_frustum_culling_enabled(frustum_culling_enabled);
                }

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
                                               &buf.data[0].y, &prev.data[0].y, static_cast<int>(buf.data.size()),
                                               prev_spec);
                        }

                        ImPlot::PlotLine(to_string(static_cast<RenderStage>(stage)).data(), &buf.data[0].x,
                                         &buf.data[0].y, static_cast<int>(buf.data.size()), spec);
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
                auto &registry = active_scene->get_registry();
                std::size_t index = 0;
                draw_rows(index, registry, registry.view<Components::Transform, PointLight, GeneratedMeta>(),
                          draw_point_light);
                draw_rows(index, registry, registry.view<Components::Transform, PointLight, Meta>(), draw_point_light);
                draw_rows(index, registry, registry.view<Components::Transform, SpotLight, Meta>(), draw_spot_light);
                draw_rows(index, registry, registry.view<Components::Transform, SpotLight, GeneratedMeta>(),
                          draw_spot_light);
            });
        }

        VulkanContext &context;
        std::unique_ptr<Renderer> renderer;
        std::unique_ptr<gui::ImGuiRenderer> imgui_renderer;
        ShaderHotReloadWatcher shader_watcher_;

        std::unique_ptr<Scene> editor_scene = std::make_unique<Scene>();
        std::unique_ptr<Scene> runtime_scene;
        Scene *active_scene = editor_scene.get();
        bool is_playing = false;


        auto play() -> void {
            runtime_scene = std::make_unique<Scene>();
            runtime_scene->physics_settings = editor_scene->physics_settings;
            clone_registry<Components::Transform, ModelHandle, RigidBody, MaterialOverride, PointLight, SpotLight>(
                    editor_scene->get_registry(), runtime_scene->get_registry());

            active_scene = runtime_scene.get();
            is_playing = true;
            active_scene->on_scene_start();
        }

        auto stop() -> void {
            active_scene->on_scene_stop();
            is_playing = false;
            active_scene = editor_scene.get();
            runtime_scene.reset();
        }

        EngineModels engine_models{};

        // Set by recreate_entities() -- reused by shoot_bullet() so
        // projectiles are built from the same model/collision-shape
        // relationship as the grid cubes and floor.
        ModelHandle cube_model{};
        glm::vec3 cube_half_extents{0.5F};

        ModelHandle house_model{};
        ModelHandle tree_model{};

        // Wind-swaying grass material -- created once in recreate_entities()
        // and shared as a MaterialOverride across every grass clump entity.
        MaterialHandle grass_material{};

        // Seconds since startup -- forwarded to UBO.time each frame so the
        // wind shader (wind.slang) has something to animate against.
        float elapsed_time = 0.0F;

        std::array<ScrollingBuffer, stage_count> timing_buffers;
        float timing_x = 0.0F;

        EditorCamera camera;

        float light_azimuth_degrees = 30.0F;
        float light_elevation_degrees = 55.0F;

        bool mouse_dragging = false;

        double last_mouse_x = 0.0;
        double last_mouse_y = 0.0;
        bool has_last_mouse_position = false;

        auto update(float delta_time) -> void {
            if (!is_playing || !active_scene) {
                return;
            }

            active_scene->step(delta_time);

            systems::lifetime(active_scene->get_registry(), *active_scene->physics_world, delta_time);
        }

        auto on_startup() -> void {

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

                this->recreate_entities();
            });
        }

        auto on_event(KeyPressedEvent ev) -> bool {
            if (ev.key == GLFW_KEY_R && ev.modifiers == GLFW_MOD_CONTROL) {
                renderer->queue_render_thread_event([this] { this->recreate_entities(); });
            }
            if (ev.key == GLFW_KEY_F12) {
                renderer->request_screenshot();
            }
            camera.on_key_pressed(ev.key);
            return true;
        }

        auto on_event(KeyReleasedEvent ev) -> bool {
            camera.on_key_released(ev.key);

            return true;
        }

        auto on_event(MouseMovedEvent ev) -> bool {
            camera.on_mouse_moved(static_cast<float>(ev.delta_x), static_cast<float>(ev.delta_y), mouse_dragging);

            return true;
        }

        auto on_event(MouseScrolledEvent ev) -> bool {
            camera.on_mouse_scrolled(static_cast<float>(ev.delta_y));

            return true;
        }

        auto on_event(MouseButtonPressedEvent ev) -> bool {
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

        // Unprojects the current cursor position into a world-space ray,
        // using the same view/projection the camera renders with. The
        // cursor isn't locked to screen center outside of a right-drag
        // look, so aiming from camera.forward() alone would visibly miss
        // wherever the mouse actually is.
        [[nodiscard]] auto cursor_ray() const -> std::pair<glm::vec3, glm::vec3> {
            int window_width = 0;
            int window_height = 0;
            glfwGetWindowSize(context.window, &window_width, &window_height);

            auto const framebuffer_width = context.framebuffer_width.load(std::memory_order_relaxed);
            auto const framebuffer_height = context.framebuffer_height.load(std::memory_order_relaxed);

            if (window_width <= 0 || window_height <= 0 || framebuffer_width <= 0 || framebuffer_height <= 0) {
                return {camera.position(), camera.forward()};
            }

            // NDC fraction comes from the mouse position in window
            // coordinates (what GLFW reports); the projection's aspect
            // ratio comes from the framebuffer (what's actually rendered).
            // They differ only under HiDPI content scaling.
            auto const ndc_x = (2.0F * static_cast<float>(last_mouse_x)) / static_cast<float>(window_width) - 1.0F;
            auto const ndc_y = 1.0F - (2.0F * static_cast<float>(last_mouse_y)) / static_cast<float>(window_height);
            auto const aspect_ratio = static_cast<float>(framebuffer_width) / static_cast<float>(framebuffer_height);

            auto const inverse_view_projection = glm::inverse(camera.view_projection(aspect_ratio));

            auto const unproject = [&](float ndc_z) {
                auto const clip_space_point = inverse_view_projection * glm::vec4{ndc_x, ndc_y, ndc_z, 1.0F};
                return glm::vec3{clip_space_point} / clip_space_point.w;
            };

            auto const ray_origin = unproject(0.0F);
            auto const ray_direction = glm::normalize(unproject(1.0F) - ray_origin);

            return {ray_origin, ray_direction};
        }


        auto shoot_bullet(std::size_t n = 1) -> void {
            if (!is_playing || !active_scene || !active_scene->physics_world) {
                return;
            }

            constexpr auto bullet_half_extent = 0.15F;
            constexpr auto bullet_speed = 40.0F;
            constexpr auto bullet_mass = 0.2F;
            constexpr auto bullet_lifetime_seconds = 3.0F; // Expire after 3 seconds

            auto const [ray_origin, ray_direction] = cursor_ray();

            auto const transform = Components::Transform{
                    .position = ray_origin + ray_direction * (bullet_half_extent + 0.2F),
                    .scale = glm::vec3{bullet_half_extent} / cube_half_extents,
            };
            auto const rigid_body = RigidBody{
                    .velocity = ray_direction * bullet_speed,
                    .half_extents = glm::vec3{bullet_half_extent},
                    .restitution = 0.3F,
                    .mass = bullet_mass,
            };

            for (auto i = 0U; i < n; ++i) {
                auto const entity = GeneratedEntity{active_scene, "bullet_{}", static_cast<std::uint32_t>(i)};
                entity.emplace<Components::Transform>(transform);
                entity.emplace<ModelHandle>(cube_model);
                entity.emplace<RigidBody>(rigid_body);

                entity.emplace<Components::Lifetime>(bullet_lifetime_seconds);

                active_scene->physics_world->add_body(entity, transform, rigid_body);
            }
        }

        auto on_event(MouseButtonReleasedEvent ev) -> bool {
            if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
                mouse_dragging = false;
            }

            return true;
        }

        auto recreate_entities() -> void {

            if (auto could_wait = renderer->wait_idle(); !could_wait.has_value()) {
                info("{}", describe(could_wait.error()));
                return;
            }

            editor_scene->get_registry().clear();


            auto const load_or_fallback = [&r = renderer, &default_model = engine_models.cube,
                                           s = editor_scene.get()](std::filesystem::path const &path,
                                                                   glm::mat4 const &instance_transform =
                                                                           glm::mat4{1.0F}) -> ModelHandle {
                auto model = r->load_model(path);

                if (model) {
                    auto const rotation_scale = glm::mat3{instance_transform};

                    for (auto &&[index, light]: r->model_lights(model.value()) | std::views::enumerate) {
                        auto const light_entity =
                                GeneratedEntity{s, "model_light_{}", static_cast<std::uint32_t>(index)};

                        auto const world_position = glm::vec3{instance_transform * glm::vec4{light.position, 1.0F}};
                        auto const world_direction = glm::normalize(rotation_scale * light.direction);

                        light_entity.emplace<Components::Transform>(Components::Transform{
                                .position = world_position,
                                .rotation = direction_to_rotation(world_direction),
                        });

                        if (light.type == ModelLightType::point) {
                            light_entity.emplace<PointLight>(PointLight{
                                    .colour = light.colour,
                                    .intensity = light.intensity,
                                    .range = light.range,
                            });
                        } else {
                            light_entity.emplace<SpotLight>(SpotLight{
                                    .colour = light.colour,
                                    .intensity = light.intensity,
                                    .range = light.range,
                                    .inner_cone_degrees = light.inner_cone_degrees,
                                    .outer_cone_degrees = light.outer_cone_degrees,
                            });
                        }
                    }

                    info("Loaded model '{}', with {} lights", path.string(), r->model_lights(model.value()).size());
                    return model.value();
                }

                error("Could not load model '{}': {}", path.string(), describe(model.error()));
                warn("Falling back to engine cube for '{}'", path.string());
                return default_model;
            };

            auto const helmet_model = load_or_fallback("assets/models/damaged_helmet/DamagedHelmet.gltf");
            cube_model = load_or_fallback("assets/models/test_cube.glb");

            info("BEFORE");
            constexpr auto house_position = glm::vec3{40.0F, 10.0F, 10.0F};
            auto const house_transform = glm::translate(glm::mat4{1.0F}, house_position);

            house_model = load_or_fallback("assets/models/scene.glb", house_transform);

            auto e = Entity{editor_scene.get(), "house"};
            e.emplace<Components::Transform>(Components::Transform{.position = house_position});
            e.emplace<ModelHandle>(house_model);
            e.emplace<RigidBody>(RigidBody::from_model_bounds(renderer->model_bounds(house_model).value()));
            info("AFTER");

            tree_model = load_or_fallback("assets/models/tree.glb");

            auto tree_entity = Entity{editor_scene.get(), "tree"};
            tree_entity.emplace<Components::Transform>(
                    Components::Transform{.position = glm::vec3{20.0F, 0.0F, 20.0F}});
            tree_entity.emplace<ModelHandle>(tree_model);

            std::random_device r;
            std::seed_seq seed{r(), r(), r(), r(), r(), r(), r(), r()};
            std::mt19937 eng(seed);
            std::uniform_real_distribution<float> urd(-20, 20);

            const auto count = 10;
            for (auto i = 0; i < count; i++) {
                for (auto j = 0; j < count; j++) {
                    for (auto k = 0; k < count; k++) {
                        auto entity = GeneratedEntity{editor_scene.get(), "helmet_{}_{}_{}", i, j, k};
                        entity.emplace<Components::Transform>(Components::Transform{
                                .position =
                                        glm::vec3{
                                                5 * urd(eng),
                                                5 * urd(eng),
                                                5 * urd(eng),
                                        },
                        });
                        entity.emplace<ModelHandle>(helmet_model);
                    }
                }
            }

            // A grid of physics-driven cubes dropped from above the ground
            // plane -- gravity, then Bullet collision against each other and
            // the floor, is stepped in update_physics() every frame. Spacing
            // is derived from the model's actual bounds (rather than an
            // assumed half-extent) so the collision shapes match what's
            // rendered, and neighbours start apart so they only overlap once
            // gravity pulls them into each other.
            constexpr auto physics_grid = 12;

            auto const cube_bounds = renderer->model_bounds(cube_model);
            cube_half_extents =
                    cube_bounds.has_value() ? (cube_bounds->second - cube_bounds->first) * 0.5F : glm::vec3{0.5F};
            auto const cube_diameter = std::max({cube_half_extents.x, cube_half_extents.y, cube_half_extents.z}) * 2.0F;
            auto const spacing = cube_diameter * 1.3F;

            for (auto i = 0; i < physics_grid; i++) {
                for (auto j = 0; j < physics_grid; j++) {
                    for (auto k = 0; k < physics_grid; k++) {
                        auto const entity = GeneratedEntity{editor_scene.get(), "physics_cube_{}_{}_{}", i, j, k};
                        auto const position = glm::vec3{
                                static_cast<float>(i - physics_grid / 2) * spacing,
                                5.0F + static_cast<float>(j) * spacing,
                                static_cast<float>(k - physics_grid / 2) * spacing,
                        };
                        entity.emplace<Components::Transform>(Components::Transform{.position = position});
                        entity.emplace<ModelHandle>(cube_model);
                        entity.emplace<RigidBody>(RigidBody{.half_extents = cube_half_extents});
                    }
                }
            }

            // A visible floor slab, sized/scaled from the same cube model so
            // its rendered surface matches its collider exactly -- reusing
            // an ordinary static RigidBody rather than a bespoke ground
            // plane means PhysicsWorld needs no special-case handling for
            // it, and it renders through the same submit_scene() path as
            // everything else.
            constexpr auto floor_half_extents = glm::vec3{40.0F, 0.5F, 40.0F};

            auto const floor_entity = GeneratedEntity{editor_scene.get(), "floor"};
            floor_entity.emplace<Components::Transform>(Components::Transform{
                    .position = glm::vec3{0.0F, editor_scene->physics_settings.ground_y - floor_half_extents.y, 0.0F},
                    .scale = floor_half_extents / cube_half_extents,
            });
            floor_entity.emplace<ModelHandle>(cube_model);
            floor_entity.emplace<RigidBody>(RigidBody{.half_extents = floor_half_extents, .is_static = true});

            // Texture fields must be filled with real dummy handles (not
            // left default) -- a default-constructed ImageHandle carries
            // invalid_image_index (UINT32_MAX), and to_gpu_material() copies
            // that straight into the bindless texture index with no
            // validity check, so leaving it unset means an out-of-bounds
            // descriptor read on the GPU (device lost).
            auto &images = renderer->image_storage();
            auto &samplers = renderer->sampler_storage();

            auto const floor_material = renderer->create_material(MaterialCreateInfo{
                    .base_colour_factor = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
                    .base_colour_texture = images.white(),
                    .normal_texture = images.flat_normal(),
                    .metallic_roughness_texture = images.metallic_roughness(),
                    .occlusion_texture = images.occlusion(),
                    .emissive_texture = images.emissive(),
                    .sampler = samplers.linear_repeat(),
            });

            if (floor_material) {
                floor_entity.emplace<MaterialOverride>(MaterialOverride{*floor_material});
            } else {
                error("Could not create floor material override: {}", describe(floor_material.error()));
            }

            // A handful of coloured point lights hovering over the physics
            // cube grid, and one spot light angled down at it -- demo
            // content exercising the punctual-light path end to end
            // (ECS component -> submit_scene -> lights SSBO -> BRDF).
            {
                constexpr std::array<glm::vec3, 4> point_light_colours{
                        glm::vec3{1.0F, 0.35F, 0.25F},
                        glm::vec3{0.25F, 0.55F, 1.0F},
                        glm::vec3{0.35F, 1.0F, 0.4F},
                        glm::vec3{1.0F, 0.85F, 0.25F},
                };

                for (std::size_t i = 0; i < point_light_colours.size(); ++i) {
                    auto const angle = (static_cast<float>(i) / static_cast<float>(point_light_colours.size())) *
                                       6.2831853F;
                    auto const radius = spacing * static_cast<float>(physics_grid) * 0.5F;

                    auto const light_entity = GeneratedEntity{editor_scene.get(), "point_light_{}", i};
                    light_entity.emplace<Components::Transform>(Components::Transform{
                            .position = glm::vec3{radius * std::cos(angle), 12.0F, radius * std::sin(angle)},
                    });
                    light_entity.emplace<PointLight>(PointLight{
                            .colour = point_light_colours[i],
                            .intensity = 25.0F,
                            .range = 20.0F,
                    });
                }

                auto const spot_entity = GeneratedEntity{editor_scene.get(), "spot_light"};
                spot_entity.emplace<Components::Transform>(Components::Transform{
                        .position = glm::vec3{0.0F, 15.0F, 0.0F},
                        .rotation = glm::angleAxis(glm::radians(30.0F), glm::vec3{1.0F, 0.0F, 0.0F}),
                });
                spot_entity.emplace<SpotLight>(SpotLight{
                        .colour = glm::vec3{0.9F, 0.95F, 1.0F},
                        .intensity = 60.0F,
                        .range = 30.0F,
                        .inner_cone_degrees = 15.0F,
                        .outer_cone_degrees = 25.0F,
                });
            }

            // A dense field of wind-swaying grass clumps covering a 100x100
            // area centred on the ground plane. Positions are jittered on a
            // fine grid rather than drawn from a fully uniform distribution
            // so density stays even (uniform-random over this area visibly
            // clumps and gaps); each clump gets a MaterialOverride pointing
            // at the same grass_material, whose wind_strength > 0 is what
            // the vertex shader (wind.slang) keys the sway off of -- every
            // other material in the scene defaults to wind_strength = 0.
            auto const grass_material_result = renderer->create_material(MaterialCreateInfo{
                    .base_colour_factor = glm::vec4{0.25F, 0.55F, 0.18F, 1.0F},
                    .base_colour_texture = images.white(),
                    .normal_texture = images.flat_normal(),
                    .metallic_roughness_texture = images.metallic_roughness(),
                    .occlusion_texture = images.occlusion(),
                    .emissive_texture = images.emissive(),
                    .sampler = samplers.linear_repeat(),
                    .wind_strength = 0.28F,
                    // Individual blades are sub-pixel by cascade 2 already
                    // (see shadow_cascade_resolutions), so the ~40k-instance
                    // grass batch is skipped entirely in the farthest two
                    // cascades -- see Renderer::prepare_frame's shadow
                    // batch partition.
                    .max_shadow_cascade = 1,
            });

            if (!grass_material_result) {
                error("Could not create grass material: {}", describe(grass_material_result.error()));
            } else {
                grass_material = *grass_material_result;

                constexpr auto grass_field_size = 100.0F;
                constexpr auto grass_spacing = 0.5F;
                constexpr auto grass_cells = static_cast<int>(grass_field_size / grass_spacing);

                std::uniform_real_distribution<float> jitter(-grass_spacing * 0.4F, grass_spacing * 0.4F);
                std::uniform_real_distribution<float> yaw(0.0F, 6.2831853F);
                std::uniform_real_distribution<float> height_scale(0.7F, 1.3F);
                std::uniform_real_distribution<float> width_scale(0.8F, 1.2F);

                for (auto cell_x = 0; cell_x < grass_cells; ++cell_x) {
                    for (auto cell_z = 0; cell_z < grass_cells; ++cell_z) {
                        auto const x = (static_cast<float>(cell_x) + 0.5F) * grass_spacing - grass_field_size * 0.5F +
                                       jitter(eng);
                        auto const z = (static_cast<float>(cell_z) + 0.5F) * grass_spacing - grass_field_size * 0.5F +
                                       jitter(eng);

                        auto const grass_entity = GeneratedEntity{editor_scene.get(), "grass_{}_{}", cell_x, cell_z};
                        grass_entity.emplace<Components::Transform>(Components::Transform{
                                .position = glm::vec3{x, editor_scene->physics_settings.ground_y, z},
                                .rotation = glm::angleAxis(yaw(eng), glm::vec3{0.0F, 1.0F, 0.0F}),
                                .scale = glm::vec3{width_scale(eng), height_scale(eng), width_scale(eng)},
                        });
                        grass_entity.emplace<ModelHandle>(engine_models.grass_clump);
                        grass_entity.emplace<MaterialOverride>(MaterialOverride{grass_material});
                    }
                }
            }
        }
    };

    auto vk_result_name(VkResult result) noexcept -> std::string_view {
        switch (result) {
            case VK_SUCCESS:
                return "VK_SUCCESS";

            case VK_NOT_READY:
                return "VK_NOT_READY";

            case VK_TIMEOUT:
                return "VK_TIMEOUT";

            case VK_EVENT_SET:
                return "VK_EVENT_SET";

            case VK_EVENT_RESET:
                return "VK_EVENT_RESET";

            case VK_INCOMPLETE:
                return "VK_INCOMPLETE";

            case VK_ERROR_OUT_OF_HOST_MEMORY:
                return "VK_ERROR_OUT_OF_HOST_MEMORY";

            case VK_ERROR_OUT_OF_DEVICE_MEMORY:
                return "VK_ERROR_OUT_OF_DEVICE_MEMORY";

            case VK_ERROR_INITIALIZATION_FAILED:
                return "VK_ERROR_INITIALIZATION_FAILED";

            case VK_ERROR_DEVICE_LOST:
                return "VK_ERROR_DEVICE_LOST";

            case VK_ERROR_MEMORY_MAP_FAILED:
                return "VK_ERROR_MEMORY_MAP_FAILED";

            case VK_ERROR_LAYER_NOT_PRESENT:
                return "VK_ERROR_LAYER_NOT_PRESENT";

            case VK_ERROR_EXTENSION_NOT_PRESENT:
                return "VK_ERROR_EXTENSION_NOT_PRESENT";

            case VK_ERROR_FEATURE_NOT_PRESENT:
                return "VK_ERROR_FEATURE_NOT_PRESENT";

            case VK_ERROR_INCOMPATIBLE_DRIVER:
                return "VK_ERROR_INCOMPATIBLE_DRIVER";

            case VK_ERROR_TOO_MANY_OBJECTS:
                return "VK_ERROR_TOO_MANY_OBJECTS";

            case VK_ERROR_FORMAT_NOT_SUPPORTED:
                return "VK_ERROR_FORMAT_NOT_SUPPORTED";

            case VK_ERROR_FRAGMENTED_POOL:
                return "VK_ERROR_FRAGMENTED_POOL";

            case VK_ERROR_SURFACE_LOST_KHR:
                return "VK_ERROR_SURFACE_LOST_KHR";

            case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
                return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";

            default:
                return "VK_UNKNOWN_RESULT";
        }
    }

    auto report_vk_error(std::string_view operation, VkResult result) noexcept -> void {
        error("{} failed: {} ({})", operation, vk_result_name(result), static_cast<int>(result));
    }

    auto validation_layer_available() noexcept -> bool {
        std::uint32_t layer_count = 0;

        auto result = vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateInstanceLayerProperties(count)", result);

            return false;
        }

        std::vector<VkLayerProperties> layers(layer_count);

        result = vkEnumerateInstanceLayerProperties(&layer_count, layers.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateInstanceLayerProperties(list)", result);

            return false;
        }

        return std::ranges::any_of(layers, [](VkLayerProperties const &layer) {
            return std::string_view{layer.layerName} == validation_layers.front();
        });
    }

    VKAPI_ATTR auto VKAPI_CALL vulkan_debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                     VkDebugUtilsMessageTypeFlagsEXT,
                                                     VkDebugUtilsMessengerCallbackDataEXT const *callback_data,
                                                     void *) noexcept -> VkBool32 {
        auto const *message = callback_data != nullptr && callback_data->pMessage != nullptr
                                      ? callback_data->pMessage
                                      : "<no validation message>";

        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
            error("Vulkan validation: {}", message);
        } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
            warn("Vulkan validation: {}", message);
        } else {
            debug("Vulkan validation: {}", message);
        }

        return VK_FALSE;
    }

    auto make_debug_messenger_create_info() noexcept -> VkDebugUtilsMessengerCreateInfoEXT {
        return VkDebugUtilsMessengerCreateInfoEXT{
                .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
                .pNext = nullptr,
                .flags = 0,
                .messageSeverity =
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
                .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                .pfnUserCallback = vulkan_debug_callback,
                .pUserData = nullptr,
        };
    }

    auto glfw_error_callback(int error_code, char const *description) noexcept -> void {
        error("GLFW error {}: {}", error_code, description != nullptr ? description : "<no description>");
    }

    auto initialize_glfw(VulkanContext &context) noexcept -> bool {
        glfwSetErrorCallback(glfw_error_callback);


        auto renderdoc = renderdoc_init();

        info("Was created via renderdoc: {}", renderdoc.is_active());

#if defined(__linux__)
        // RenderDoc does not support capturing Wayland surfaces, so force X11 when active.
        glfwInitHint(GLFW_PLATFORM, renderdoc.is_active() ? GLFW_PLATFORM_X11 : GLFW_ANY_PLATFORM);
        warn("Chosing: {} as platform.", renderdoc.is_active() ? "X11" : "auto-detected");
#endif

        if (glfwInit() != GLFW_TRUE) {
            error("glfwInit failed");
            return false;
        }

        glfwSetMonitorCallback([](GLFWmonitor *monitor, int event) {
            auto const *name = glfwGetMonitorName(monitor);

            if (event == GLFW_CONNECTED) {
                info("Monitor connected: {}", name != nullptr ? name : "<unknown>");
            } else if (event == GLFW_DISCONNECTED) {
                info("Monitor disconnected: {}", name != nullptr ? name : "<unknown>");
            }
        });

        if (glfwVulkanSupported() != GLFW_TRUE) {
            error("GLFW could not find a Vulkan loader "
                  "or required platform support");

            return false;
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        std::int32_t monitor_count{};
        glfwGetMonitors(&monitor_count);
        auto monitors = glfwGetMonitors(&monitor_count);
        std::span<GLFWmonitor *> ms{monitors, static_cast<std::uint32_t>(monitor_count)};
        GLFWmonitor *selected = nullptr;
        std::int32_t max_width = std::numeric_limits<std::int32_t>::min();
        debug("Evaluating monitors...");
        for (auto &m: ms) {
            auto name = glfwGetMonitorName(m);
            auto mode = glfwGetVideoMode(m);
            debug("\t{} - {}", name, mode->width);
            if (mode->width > max_width) {
                selected = m;
                max_width = mode->width;
            }
        }
        debug("Done.");

        if (selected == nullptr) {
            error("Could not find a monitor");
            return false;
        }


        auto monitor = selected;
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
        context.window = glfwCreateWindow(mode->width, mode->height, "VK", monitor, NULL);

        if (context.window == nullptr) {
            error("glfwCreateWindow failed");
            return false;
        }

        info("GLFW window created");

        return true;
    }

    auto create_instance(VulkanContext &context) noexcept -> bool {
        auto const volk_result = volkInitialize();

        if (volk_result != VK_SUCCESS) {
            report_vk_error("volkInitialize", volk_result);

            return false;
        }

        auto const loader_version = volkGetInstanceVersion();

        info("Vulkan loader version: {}.{}.{}", VK_VERSION_MAJOR(loader_version), VK_VERSION_MINOR(loader_version),
             VK_VERSION_PATCH(loader_version));

        if (loader_version < VK_API_VERSION_1_3) {
            error("Vulkan 1.3 is required for "
                  "core synchronization2");

            return false;
        }

        constexpr auto requested_version = VK_API_VERSION_1_4;

        std::uint32_t extension_count = 0;

        auto const *required_extensions = glfwGetRequiredInstanceExtensions(&extension_count);

        if (required_extensions == nullptr || extension_count == 0) {
            error("glfwGetRequiredInstanceExtensions "
                  "returned no extensions");

            return false;
        }

        std::vector<char const *> instance_extensions(required_extensions, required_extensions + extension_count);

        auto validation_enabled = enable_validation;
        info("Validation is {}", validation_enabled ? "on" : "off");

        if (validation_enabled && !validation_layer_available()) {
            warn("{} is unavailable; validation "
                 "is disabled",
                 validation_layers.front());

            validation_enabled = false;
        }

        if (validation_enabled) {
            instance_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        auto const debug_create_info = make_debug_messenger_create_info();

        VkApplicationInfo const application_info{
                .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                .pNext = nullptr,
                .pApplicationName = "glfw-vulkan-test",
                .applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
                .pEngineName = "none",
                .engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0),
                .apiVersion = requested_version,
        };

        VkInstanceCreateInfo const create_info{
                .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                .pNext = validation_enabled ? &debug_create_info : nullptr,
                .flags = 0,
                .pApplicationInfo = &application_info,
                .enabledLayerCount = validation_enabled ? static_cast<std::uint32_t>(validation_layers.size()) : 0,
                .ppEnabledLayerNames = validation_enabled ? validation_layers.data() : nullptr,
                .enabledExtensionCount = static_cast<std::uint32_t>(instance_extensions.size()),
                .ppEnabledExtensionNames = instance_extensions.data(),
        };

        auto const result = vkCreateInstance(&create_info, nullptr, &context.instance);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateInstance", result);

            return false;
        }

        volkLoadInstance(context.instance);

        if (validation_enabled) {
            auto const debug_result = vkCreateDebugUtilsMessengerEXT(context.instance, &debug_create_info, nullptr,
                                                                     &context.debug_messenger);

            if (debug_result != VK_SUCCESS) {
                report_vk_error("vkCreateDebugUtilsMessengerEXT", debug_result);

                return false;
            }

            info("Vulkan validation enabled");
        }

        info("Vulkan instance created");

        return true;
    }

    auto create_surface(VulkanContext &context) noexcept -> bool {
        auto const result = glfwCreateWindowSurface(context.instance, context.window, nullptr, &context.surface);

        if (result != VK_SUCCESS) {
            report_vk_error("glfwCreateWindowSurface", result);

            return false;
        }

        info("GLFW Vulkan surface created");

        return true;
    }

    auto find_queue_families(VkPhysicalDevice physical_device, VkSurfaceKHR surface) noexcept -> QueueFamilies {
        std::uint32_t queue_family_count = 0;

        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);

        std::vector<VkQueueFamilyProperties> properties(queue_family_count);

        vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, properties.data());

        QueueFamilies result{};

        for (std::uint32_t index = 0; index < queue_family_count; ++index) {
            auto const supports_graphics = (properties[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;

            if (supports_graphics) {
                result.graphics = index;
            }

            VkBool32 supports_present = VK_FALSE;

            auto const present_result =
                    vkGetPhysicalDeviceSurfaceSupportKHR(physical_device, index, surface, &supports_present);

            if (present_result != VK_SUCCESS) {
                report_vk_error("vkGetPhysicalDeviceSurfaceSupportKHR", present_result);

                continue;
            }

            if (supports_present == VK_TRUE) {
                result.present = index;
            }

            if (result.complete()) {
                break;
            }
        }

        return result;
    }

    auto supports_device_extension(VkPhysicalDevice physical_device, std::string_view required_extension) noexcept
            -> bool {
        std::uint32_t extension_count = 0;

        auto result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateDeviceExtensionProperties(count)", result);

            return false;
        }

        std::vector<VkExtensionProperties> extensions(extension_count);

        result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, extensions.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumerateDeviceExtensionProperties(list)", result);

            return false;
        }

        return std::ranges::any_of(extensions, [required_extension](VkExtensionProperties const &extension) {
            return required_extension == extension.extensionName;
        });
    }

    auto rate_device_type(VkPhysicalDeviceType type) noexcept -> int {
        switch (type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                return 500;

            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                return 400;

            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                return 300;

            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                return 200;

            default:
                return 100;
        }
    }

    auto select_physical_device(VulkanContext &context) noexcept -> bool {
        std::uint32_t physical_device_count = 0;

        auto result = vkEnumeratePhysicalDevices(context.instance, &physical_device_count, nullptr);

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumeratePhysicalDevices(count)", result);

            return false;
        }

        if (physical_device_count == 0) {
            error("No Vulkan physical devices found");

            return false;
        }

        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);

        result = vkEnumeratePhysicalDevices(context.instance, &physical_device_count, physical_devices.data());

        if (result != VK_SUCCESS) {
            report_vk_error("vkEnumeratePhysicalDevices(list)", result);

            return false;
        }

        auto best_score = -1;

        for (auto const physical_device: physical_devices) {
            VkPhysicalDeviceProperties properties{};

            vkGetPhysicalDeviceProperties(physical_device, &properties);

            if (properties.apiVersion < VK_API_VERSION_1_3) {
                continue;
            }

            VkPhysicalDeviceVulkan12Features vulkan12_features{};
            vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
            vulkan12_features.pNext = nullptr;

            VkPhysicalDeviceVulkan13Features vulkan13_features{};
            vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
            vulkan13_features.pNext = &vulkan12_features;

            VkPhysicalDeviceFeatures2 features2{
                    .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                    .pNext = &vulkan13_features,
                    .features = {},
            };

            vkGetPhysicalDeviceFeatures2(physical_device, &features2);

            if (vulkan12_features.bufferDeviceAddress != VK_TRUE || vulkan13_features.synchronization2 != VK_TRUE ||
                vulkan13_features.dynamicRendering != VK_TRUE) {
                continue;
            }

            if (!supports_device_extension(physical_device, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
                continue;
            }

            auto const queue_families = find_queue_families(physical_device, context.surface);

            if (!queue_families.complete()) {
                continue;
            }

            auto const score = rate_device_type(properties.deviceType);

            if (score > best_score) {
                best_score = score;

                context.physical_device = physical_device;

                context.queue_families = queue_families;
            }
        }

        if (context.physical_device == VK_NULL_HANDLE) {
            error("No suitable Vulkan physical "
                  "device found");

            return false;
        }

        VkPhysicalDeviceProperties properties{};

        vkGetPhysicalDeviceProperties(context.physical_device, &properties);

        info("Selected physical device: {}", properties.deviceName);

        return true;
    }

    auto create_device(VulkanContext &context) noexcept -> bool {
        constexpr float queue_priority = 1.0F;

        std::array<std::uint32_t, 2> queue_family_indices{
                context.queue_families.graphics,
                context.queue_families.present,
        };

        auto const unique_end = std::unique(queue_family_indices.begin(), queue_family_indices.end());

        auto const queue_count = static_cast<std::size_t>(unique_end - queue_family_indices.begin());

        std::array<VkDeviceQueueCreateInfo, 2> queue_create_infos{};

        for (std::size_t index = 0; index < queue_count; ++index) {
            queue_create_infos[index] = VkDeviceQueueCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .queueFamilyIndex = queue_family_indices[index],
                    .queueCount = 1,
                    .pQueuePriorities = &queue_priority,
            };
        }

        constexpr std::array device_extensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_EXT_MESH_SHADER_EXTENSION_NAME,
        };

        VkPhysicalDeviceVulkan11Features vulkan11_features{};
        vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11_features.pNext = nullptr;
        vulkan11_features.shaderDrawParameters = VK_TRUE;

        VkPhysicalDeviceVulkan12Features vulkan12_features{};
        vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12_features.pNext = &vulkan11_features;
        vulkan12_features.bufferDeviceAddress = VK_TRUE;
        vulkan12_features.scalarBlockLayout = VK_TRUE;
        vulkan12_features.runtimeDescriptorArray = VK_TRUE;
        vulkan12_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        vulkan12_features.hostQueryReset = VK_TRUE;
        vulkan12_features.shaderFloat16 = VK_TRUE;

        VkPhysicalDeviceVulkan13Features vulkan13_features{};
        vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13_features.pNext = &vulkan12_features;
        vulkan13_features.synchronization2 = VK_TRUE;
        vulkan13_features.dynamicRendering = VK_TRUE;
        vulkan13_features.maintenance4 = VK_TRUE;
        vulkan13_features.shaderDemoteToHelperInvocation = VK_TRUE;

        VkPhysicalDeviceVulkan14Features vulkan14_features{};
        vulkan14_features.pNext = &vulkan13_features;
        vulkan14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
        vulkan14_features.maintenance5 = VK_TRUE;

        VkPhysicalDeviceMeshShaderFeaturesEXT mesh_shader_features{};
        mesh_shader_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
        mesh_shader_features.pNext = &vulkan14_features;
        mesh_shader_features.taskShader = VK_TRUE;
        mesh_shader_features.meshShader = VK_TRUE;

        VkPhysicalDeviceFeatures enabled_features{};
        enabled_features.multiDrawIndirect = VK_TRUE;
        enabled_features.samplerAnisotropy = VK_TRUE;
        enabled_features.fillModeNonSolid = VK_TRUE;
        enabled_features.wideLines = VK_TRUE;
        enabled_features.pipelineStatisticsQuery = VK_TRUE;
        enabled_features.shaderInt16 = VK_TRUE;

        VkDeviceCreateInfo const create_info{
                .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                .pNext = &mesh_shader_features,
                .flags = 0,
                .queueCreateInfoCount = static_cast<std::uint32_t>(queue_count),
                .pQueueCreateInfos = queue_create_infos.data(),
                .enabledLayerCount = 0,
                .ppEnabledLayerNames = nullptr,
                .enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size()),
                .ppEnabledExtensionNames = device_extensions.data(),
                .pEnabledFeatures = &enabled_features,
        };

        auto const result = vkCreateDevice(context.physical_device, &create_info, nullptr, &context.device);

        if (result != VK_SUCCESS) {
            report_vk_error("vkCreateDevice", result);

            return false;
        }

        volkLoadDevice(context.device);
        vkGetDeviceQueue(context.device, context.queue_families.graphics, 0, &context.graphics_queue);
        vkGetDeviceQueue(context.device, context.queue_families.present, 0, &context.present_queue);

        if (context.graphics_queue == VK_NULL_HANDLE || context.present_queue == VK_NULL_HANDLE) {
            error("One or more Vulkan queues are null");

            return false;
        }

        info("Logical Vulkan device created");

        VkCommandPoolCreateInfo command_pool_create_info{};
        command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_create_info.queueFamilyIndex = context.queue_families.graphics;
        command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(context.device, &command_pool_create_info, nullptr, &context.one_time_pool) !=
            VK_SUCCESS) {
            error("Could not allocate a one time command pool;");
            return false;
        }

        VkCommandBufferAllocateInfo command_buffer_allocate_info{};
        command_buffer_allocate_info.commandPool = context.one_time_pool;
        command_buffer_allocate_info.commandBufferCount =
                static_cast<std::uint32_t>(context.one_time_command_buffers.size());
        command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        if (vkAllocateCommandBuffers(context.device, &command_buffer_allocate_info,
                                     context.one_time_command_buffers.data()) != VK_SUCCESS) {
            error("Could not allocate the command buffers");
            return false;
        }

        return true;
    }

    auto create_allocator(VulkanContext &context) noexcept -> bool {
        VmaAllocatorCreateInfo create_info{

                .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                         VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT |
                         VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT | VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT,
                .physicalDevice = context.physical_device,
                .device = context.device,
                .preferredLargeHeapBlockSize = 0,
                .pAllocationCallbacks = nullptr,
                .pDeviceMemoryCallbacks = nullptr,
                .pHeapSizeLimit = nullptr,
                .pVulkanFunctions = nullptr,
                .instance = context.instance,
                .vulkanApiVersion = VK_API_VERSION_1_4,
                .pTypeExternalMemoryHandleTypes = nullptr,
        };

        VmaVulkanFunctions allocator_functions{};

        auto const import_result = vmaImportVulkanFunctionsFromVolk(&create_info, &allocator_functions);

        if (import_result != VK_SUCCESS) {
            error("Could not initialise VMA");

            return false;
        }

        create_info.pVulkanFunctions = &allocator_functions;

        auto const result = vmaCreateAllocator(&create_info, &context.allocator);

        if (result != VK_SUCCESS) {
            error("Could not create Vulkan allocator");

            return false;
        }

        return true;
    }

    auto create_swapchain(VulkanContext &context) noexcept -> bool {
        int framebuffer_width = 0;
        int framebuffer_height = 0;

        glfwGetFramebufferSize(context.window, &framebuffer_width, &framebuffer_height);

        if (framebuffer_width <= 0 || framebuffer_height <= 0) {
            error("Cannot create a swapchain for "
                  "a zero-sized framebuffer");

            return false;
        }

        context.framebuffer_width.store(framebuffer_width, std::memory_order_relaxed);

        context.framebuffer_height.store(framebuffer_height, std::memory_order_relaxed);

        return context.swapchain.initialize(SwapchainCreateInfo{
                .physical_device = context.physical_device,
                .device = context.device,
                .surface = context.surface,
                .graphics_queue = context.graphics_queue,
                .present_queue = context.present_queue,
                .graphics_queue_family = context.queue_families.graphics,
                .present_queue_family = context.queue_families.present,
                .framebuffer_extent =
                        VkExtent2D{
                                .width = static_cast<std::uint32_t>(framebuffer_width),
                                .height = static_cast<std::uint32_t>(framebuffer_height),
                        },
                .vsync = true,
        });
    }

    auto initialize_vulkan(VulkanContext &context) noexcept -> bool {
        return initialize_glfw(context) && create_instance(context) && create_surface(context) &&
               select_physical_device(context) && create_device(context) && create_allocator(context) &&
               create_swapchain(context);
    }

    auto submit_scene(Application &application) -> std::expected<void, RendererError> {
        auto &registry = application.active_scene->get_registry();
        auto view = registry.view<Components::Transform const, ModelHandle const>();


        for (auto [entity, transform, model]: view.each()) {
            auto const *override_component = registry.try_get<MaterialOverride const>(entity);
            auto const material_override =
                    override_component != nullptr ? override_component->material : MaterialHandle{};

            auto result = application.renderer->submit_model(model, transform.matrix(), material_override);

            if (!result) {
                error("Could not submit scene object (model index {}): {}", model.index, describe(result.error()));
            }
        }

        auto point_light_view = registry.view<Components::Transform const, PointLight const>();

        for (auto [entity, transform, light]: point_light_view.each()) {
            auto result = application.renderer->submit_point_light(Renderer::PointLight{
                    .position = transform.position,
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
            });

            if (!result) {
                error("Could not submit point light: {}", describe(result.error()));
            }
        }

        auto spot_light_view = registry.view<Components::Transform const, SpotLight const>();

        for (auto [entity, transform, light]: spot_light_view.each()) {
            auto const direction = transform.rotation * glm::vec3{0.0F, -1.0F, 0.0F};

            auto result = application.renderer->submit_spot_light(Renderer::SpotLight{
                    .position = transform.position,
                    .direction = direction,
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
                    .inner_cone_degrees = light.inner_cone_degrees,
                    .outer_cone_degrees = light.outer_cone_degrees,
            });

            if (!result) {
                error("Could not submit spot light: {}", describe(result.error()));
            }
        }

        if (application.active_scene->lights_dirty) {
            application.renderer->mark_lights_dirty();
            application.active_scene->lights_dirty = false;
        }

        return {};
    }

    auto initialize_application(VulkanContext &context, Application &application) noexcept -> bool {
        auto renderer_result = application.renderer->initialize(RendererCreateInfo{
                .extent = context.swapchain.extent(),
                .frames_in_flight = context.swapchain.frame_count(),
                .geometry_capacity = 256UZ * 1024UZ * 1024UZ,
                .material_capacity = 4096,
                .mesh_capacity = 4096,
                .model_capacity = 1024,
                .pipeline_capacity = 1024,
                .swapchain_format = context.swapchain.format(),
                .samples = VK_SAMPLE_COUNT_4_BIT,
                .maximum_draw_count = 65536,
                .maximum_submission_count = 65536,
        });

        if (!renderer_result) {
            error("Could not initialize renderer: {}", describe(renderer_result.error()));

            return false;
        }

        return true;
    }

    auto draw(VulkanContext &context, Application &application) noexcept -> bool {
        auto frame = context.swapchain.begin_frame();

        if (!frame) {
            switch (frame.error().kind) {
                case SwapchainBeginFrameError::Kind::recreated:
                    return true;

                case SwapchainBeginFrameError::Kind::device_lost:
                    error("The Vulkan device was lost");

                    return false;

                case SwapchainBeginFrameError::Kind::fatal_error:
                    if (frame.error().context.has_value()) {
                        error("Could not begin swapchain frame: {}", describe(*frame.error().context));
                    } else {
                        error("Could not begin swapchain frame");
                    }

                    return false;
            }

            return false;
        }

        // From here on, begin_frame() has already signalled
        // frame.image_available (a binary semaphore) and put the frame's
        // command buffer into the recording state. Both must be retired
        // through end_frame() -- no matter what happens below -- or that
        // semaphore is left signalled with nothing to consume it. Signalling
        // a binary semaphore again before it's been waited on is invalid
        // Vulkan usage, and doing so on this frame slot's *next*
        // vkAcquireNextImageKHR is exactly the kind of thing that can stall
        // a later wait on some drivers rather than fail cleanly -- which
        // would show up later and non-deterministically, including inside a
        // vkDeviceWaitIdle at shutdown, since that's the first point
        // everything is forced to actually finish.
        auto frame_ok = true;

        auto submit_result = submit_scene(application);

        if (!submit_result) {
            error("Could not submit scene: {}", describe(submit_result.error()));

            frame_ok = false;
        }

        if (frame_ok) {
            application.imgui_renderer->begin_frame(gui::ImGuiFramebuffer{frame->extent, frame->format});

            {
                application.on_ui();
            }

            application.imgui_renderer->end_frame();

            auto aspect = application.renderer->aspect(frame->frame_index);
            auto prepare_result = application.renderer->prepare_frame(
                    frame->command_buffer,
                    {
                            .view = application.camera.view(),
                            .projection = application.camera.projection(aspect),
                            .near_clip = application.camera.near_clip(),
                            .far_clip = application.camera.far_clip(),
                            .vertical_fov_radians = glm::radians(application.camera.field_of_view_degrees()),
                            .aspect_ratio = aspect,
                            .time = application.elapsed_time,
                    },
                    frame->frame_index);

            if (!prepare_result) {
                error("Could not prepare renderer frame: {}", describe(prepare_result.error()));

                frame_ok = false;
            }
        }

        if (frame_ok) {
            if (auto const &timings = application.renderer->last_frame_timings(); timings.valid) {
                application.timing_x += 1.0F;

                float running_total = 0.0F;

                for (auto stage = static_cast<std::uint32_t>(RenderStage::Culling); stage < stage_count; ++stage) {
                    running_total += timings.milliseconds[stage];
                    application.timing_buffers[stage].add_point(application.timing_x, running_total);
                }
            }

            auto record_result =
                    application.renderer->record_frame(frame->command_buffer,
                                                       SwapchainImage{
                                                               .image = frame->image,
                                                               .view = frame->image_view,
                                                               .format = frame->format,
                                                               .extent = frame->extent,
                                                       },
                                                       frame->frame_index, [&](VkCommandBuffer c) {
                                                           application.imgui_renderer->render(c, frame->frame_index);
                                                       });

            if (!record_result) {
                error("Could not record renderer frame: {}", describe(record_result.error()));

                frame_ok = false;
            }
        }

        // Always retire the frame we began, whether or not its content ended
        // up being usable -- this is what keeps image_available's
        // signal/wait balanced and in_flight's state honest, regardless of
        // which step above failed.
        //
        // Caveat: this assumes prepare_frame()/record_frame() never return
        // failure while frame->command_buffer still has an open dynamic
        // rendering scope (vkCmdBeginRendering without a matching
        // vkCmdEndRendering) -- ending a command buffer with one open is
        // itself invalid. If that assumption doesn't hold on the renderer
        // side, this needs a matching fix there too.
        auto const end_result = context.swapchain.end_frame(*frame);

        if (!frame_ok) {
            return false;
        }

        switch (end_result) {
            case SwapchainFrameResult::success:
            case SwapchainFrameResult::recreated:
                return true;

            case SwapchainFrameResult::device_lost:
                error("The Vulkan device was lost");

                return false;

            case SwapchainFrameResult::fatal_error:
                error("Could not submit or present frame");

                return false;
        }

        return false;
    }

    auto request_resize_if_needed(VulkanContext &context, int width, int height) noexcept -> void {
        if (!context.framebuffer_dirty.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        context.swapchain.request_recreate(VkExtent2D{
                .width = static_cast<std::uint32_t>(width),
                .height = static_cast<std::uint32_t>(height),
        });
    }

    struct WindowData {
        VulkanContext *ctx;
        Application *app;
    };

    auto framebuffer_size_callback(GLFWwindow *window, int width, int height) noexcept -> void {
        auto *context = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->ctx;

        if (context == nullptr) {
            return;
        }

        // This callback and the render loop both run on the main thread
        // now, so there's no concurrent reader/writer to lock against --
        // a plain relaxed store is enough.
        context->framebuffer_width.store(width, std::memory_order_relaxed);
        context->framebuffer_height.store(height, std::memory_order_relaxed);
        context->framebuffer_dirty.store(true, std::memory_order_relaxed);
    }

    auto event_callback(GLFWwindow *window, int key, int, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (action == GLFW_PRESS && app->on_event(KeyPressedEvent{key, mods})) {
        }

        if (action == GLFW_RELEASE && app->on_event(KeyReleasedEvent{key, mods})) {
        }
    }

    auto mouse_button_callback(GLFWwindow *window, int button, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Capture + hide the cursor for the duration of a right-drag so it
        // can't leave the window mid-look; GLFW_CURSOR_DISABLED also
        // switches to unbounded virtual cursor movement, which is what we
        // want for a free-look camera rather than clamping at the screen
        // edge. This is specific to the right button; every button's
        // press/release still gets forwarded to on_event() below.
        if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        }

        if (action == GLFW_PRESS) {
            app->on_event(MouseButtonPressedEvent{button, mods});
        } else if (action == GLFW_RELEASE) {
            app->on_event(MouseButtonReleasedEvent{button, mods});
        }
    }

    auto cursor_position_callback(GLFWwindow *window, double x_position, double y_position) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Swallow the first callback after (re)acquiring a cursor position:
        // without a previous sample the delta would be "distance from
        // wherever the cursor happened to be", producing a large spurious
        // look jump on the first drag frame.
        if (!app->has_last_mouse_position) {
            app->last_mouse_x = x_position;
            app->last_mouse_y = y_position;
            app->has_last_mouse_position = true;

            return;
        }

        auto const delta_x = x_position - app->last_mouse_x;
        auto const delta_y = y_position - app->last_mouse_y;

        app->last_mouse_x = x_position;
        app->last_mouse_y = y_position;

        app->on_event(MouseMovedEvent{delta_x, delta_y});
    }

    auto scroll_callback(GLFWwindow *window, double x_offset, double y_offset) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app != nullptr) {
            app->on_event(MouseScrolledEvent{x_offset, y_offset});
        }
    }


    auto install_window_callbacks(VulkanContext &context, Application &app) noexcept -> void {
        static WindowData wd{};
        wd.app = &app;
        wd.ctx = &context;
        glfwSetWindowUserPointer(context.window, &wd);

        glfwSetFramebufferSizeCallback(context.window, framebuffer_size_callback);
        glfwSetKeyCallback(context.window, event_callback);
        glfwSetMouseButtonCallback(context.window, mouse_button_callback);
        glfwSetCursorPosCallback(context.window, cursor_position_callback);
        glfwSetScrollCallback(context.window, scroll_callback);
    }

    auto destroy_context(VulkanContext &context) noexcept -> void {
        context.swapchain.destroy();

        if (context.one_time_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(context.device, context.one_time_pool, nullptr);

            context.one_time_pool = VK_NULL_HANDLE;
            context.one_time_command_buffers.fill(VK_NULL_HANDLE);
        }

        if (context.allocator != VK_NULL_HANDLE) {
            vmaDestroyAllocator(context.allocator);

            context.allocator = VK_NULL_HANDLE;
        }

        if (context.device != VK_NULL_HANDLE) {
            vkDestroyDevice(context.device, nullptr);

            context.device = VK_NULL_HANDLE;
        }

        if (context.surface != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(context.instance, context.surface, nullptr);

            context.surface = VK_NULL_HANDLE;
        }

        if (context.debug_messenger != VK_NULL_HANDLE) {
            vkDestroyDebugUtilsMessengerEXT(context.instance, context.debug_messenger, nullptr);

            context.debug_messenger = VK_NULL_HANDLE;
        }

        if (context.instance != VK_NULL_HANDLE) {
            vkDestroyInstance(context.instance, nullptr);

            context.instance = VK_NULL_HANDLE;
        }

        if (context.window != nullptr) {
            glfwDestroyWindow(context.window);

            context.window = nullptr;
        }

        glfwTerminate();
    }

    // vkDeviceWaitIdle has no timeout parameter -- a genuinely non-responding
    // GPU hangs it forever with no way to interrupt the wait from this
    // thread. Running it as an async task and giving up on *waiting for it*
    // after a bound at least lets the process exit instead of hanging. Once
    // we've given up, continuing on to call more Vulkan destroy functions
    // against a device the driver may still be touching is itself unsafe, so
    // this terminates immediately rather than attempting further cleanup.
    auto wait_idle_bounded(VkDevice device, std::string_view label) noexcept -> VkResult {
        constexpr auto timeout = std::chrono::seconds{3};

        auto future = std::async(std::launch::async, [device] { return vkDeviceWaitIdle(device); });

        if (future.wait_for(timeout) != std::future_status::ready) {
            error("{}: vkDeviceWaitIdle did not return within {} -- the GPU is not responding. Exiting immediately.",
                  label, timeout);

            std::_Exit(EXIT_FAILURE);
        }

        return future.get();
    }

    auto destroy_application(VulkanContext &context, Application &application) noexcept -> void {
        if (context.device != VK_NULL_HANDLE) {
            auto const result = wait_idle_bounded(context.device, "destroy_application");

            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                report_vk_error("vkDeviceWaitIdle(application destroy)", result);
            }
        }

        application.imgui_renderer.reset();
        application.renderer->destroy();

        destroy_context(context);
    }
} // namespace

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    glfwPostEmptyEvent();
}


auto main() -> int {
    info("Starting GLFW Vulkan test at {}", std::filesystem::current_path().string());

    std::signal(SIGINT, ctrl_c_handler);

    VulkanContext context{};

    if (!initialize_vulkan(context)) {
        error("Vulkan initialization failed");

        destroy_context(context);

        return EXIT_FAILURE;
    }


    Application application{context};
    install_window_callbacks(context, application);

    if (!initialize_application(context, application)) {
        destroy_application(context, application);

        return EXIT_FAILURE;
    }

    application.on_startup();

    info("Initialization complete; close the window to exit");

    auto renderer_extent = context.swapchain.extent();
    auto last_frame_time = std::chrono::steady_clock::now();
    auto exit_code = EXIT_SUCCESS;

    while (g_running.load(std::memory_order_acquire) && context.running.load(std::memory_order_acquire) &&
           glfwWindowShouldClose(context.window) != GLFW_TRUE) {

        auto const width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const height = context.framebuffer_height.load(std::memory_order_relaxed);

        // Minimized / zero-sized framebuffer: nothing to render, so block
        // for the next event instead of busy-looping. Everywhere else we
        // poll (non-blocking), since we want to keep rendering every
        // iteration rather than waiting for input.
        if (width <= 0 || height <= 0) {
            glfwWaitEvents();
        } else {
            glfwPollEvents();
        }

        if (glfwWindowShouldClose(context.window) == GLFW_TRUE) {
            break;
        }

        application.renderer->drain_event_queue();

        auto const current_width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const current_height = context.framebuffer_height.load(std::memory_order_relaxed);

        if (current_width <= 0 || current_height <= 0) {
            last_frame_time = std::chrono::steady_clock::now();
            continue;
        }

        auto const now = std::chrono::steady_clock::now();
        auto const delta_time = std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;

        application.elapsed_time += delta_time;

        application.camera.update(std::min(delta_time, 0.1F));
        application.update(delta_time);

        request_resize_if_needed(context, current_width, current_height);

        if (!draw(context, application)) {
            exit_code = EXIT_FAILURE;
            break;
        }

        auto const swapchain_extent = context.swapchain.extent();

        if (compare(swapchain_extent, renderer_extent)) {
            auto resize_result = application.renderer->resize(swapchain_extent);

            if (!resize_result) {
                error("Could not resize renderer: {}", describe(resize_result.error()));

                exit_code = EXIT_FAILURE;
                break;
            }

            renderer_extent = swapchain_extent;
        }
    }

    context.running.store(false, std::memory_order_release);

    destroy_application(context, application);

    if (exit_code == EXIT_SUCCESS) {
        info("Application exited successfully");
    }

    return exit_code;
}
