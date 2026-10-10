#include <fstream>
#include <charconv>
#include <csignal>
#include <memory>
#include <print>
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
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <fstream>
#include <format>
#include <future>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/random.hpp>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <entt/entt.hpp>

#include "app/application.hxx"
#include "core/game_manifest.hxx"
#include "core/paths.hxx"
#include "core/resources.hxx"
#include "serialisation/resource_pack.hxx"
#include "app/benchmark.hxx"
#include "app/benchmark_compare.hxx"
#include "app/benchmark_driver.hxx"
#include "app/fatal_dialog.hxx"
#include "app/frame_clock.hxx"
#include "app/game.hxx"
#include "assets/shader_hot_reload_watcher.hxx"
#include "assets/shader_bake.hxx"
#include "assets/shader_pack.hxx"
#include "core/allocator.hxx"
#include "core/config.hxx"
#include "core/command_line.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/random.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "gpu/context.hxx"
#include "gpu/device_wait.hxx"
#include "gpu/queue_selection.hxx"
#include "gpu/swapchain.hxx"
#include "imgui.h"
#include "implot.h"
#include "maths/aabb.hxx"
#include "physics/physics.hxx"
#include "physics/physics_world.hxx"
#include "rendering/cluster_grid.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/engine_models.hxx"
#include "rendering/entity.hxx"
#include "rendering/imgui_renderer.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "rendering/screenshot.hxx"
#include "scene/components.hxx"
#include "scene/editor_camera.hxx"
#include "vulkan_bootstrap.hxx"

namespace {

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

    constexpr auto compare = []<typename A, typename B>(A const &a, B const &b) -> bool {
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

    auto submit_scene(Application &application) -> std::expected<void, RendererError> {
        ZoneScopedNC("SubmitScene", tracy::Color::RoyalBlue);

        auto &registry = application.active_scene()->get_registry();

        application.renderer->set_environment(application.active_scene()->environment);

        auto view = registry.view<Components::Transform const, Components::Model const>();

        auto const draw_model_bounds = application.debug_renderer->model_bounds_debug_enabled();
        constexpr auto model_bounds_debug_colour = glm::vec3{0.2F, 1.0F, 0.4F};

        for (auto [entity, transform, model]: view.each()) {
            auto const *override_component = registry.try_get<Components::MaterialOverride const>(entity);
            auto const material_override =
                    override_component != nullptr ? override_component->material : MaterialHandle{};
            auto const slot_overrides = override_component != nullptr
                                                ? std::span<MaterialSlotOverride const>{override_component->slots}
                                                : std::span<MaterialSlotOverride const>{};

            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto result = application.renderer->submit_model(model.model, world_transform, material_override,
                                                             slot_overrides, registry.all_of<Components::Outlined>(entity));

            if (!result) {
                error("Could not submit scene object (model index {}): {}", model.model.index,
                      describe(result.error()));
            }

            if (draw_model_bounds) {
                auto const submesh_bounds = application.renderer->model_submesh_bounds(model.model);

                if (submesh_bounds) {
                    for (auto const &[min, max]: *submesh_bounds) {
                        auto const [world_min, world_max] = maths::transform_aabb(world_transform, min, max);
                        application.debug_renderer->add_aabb(world_min, world_max, model_bounds_debug_colour);
                    }
                }
            }
        }

        auto instanced_model_view = registry.view<Components::InstancedModel const>();

        for (auto [entity, instanced]: instanced_model_view.each()) {
            auto result = application.renderer->submit_model_instances(
                    instanced.model, instanced.transforms, instanced.material_override, instanced.revision,
                    instanced.palette_offsets);

            if (!result) {
                error("Could not submit instanced model (model index {}, {} instances): {}", instanced.model.index,
                      instanced.transforms.size(), describe(result.error()));
            }
        }

        auto point_light_view = registry.view<Components::Transform const, Components::PointLight const>();

        for (auto [entity, transform, light]: point_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);

            auto result = application.renderer->submit_point_light(Renderer::PointLight{
                    .position = glm::vec3{world_transform[3]},
                    .colour = light.colour,
                    .intensity = light.intensity,
                    .range = light.range,
            });

            if (!result) {
                error("Could not submit point light: {}", describe(result.error()));
            }
        }

        auto spot_light_view = registry.view<Components::Transform const, Components::SpotLight const>();

        for (auto [entity, transform, light]: spot_light_view.each()) {
            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto const direction = glm::normalize(glm::mat3{world_transform} * glm::vec3{0.0F, -1.0F, 0.0F});

            auto result = application.renderer->submit_spot_light(Renderer::SpotLight{
                    .position = glm::vec3{world_transform[3]},
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

        if (auto *const terrain = application.active_terrain(); terrain != nullptr) {
            terrain->submit(*application.renderer);
        }

        return {};
    }

    auto initialize_application(VulkanContext &context, Application &application) noexcept -> bool {
        auto renderer_result = application.renderer->initialize(RendererCreateInfo{
                .extent = context.swapchain.extent(),
                .geometry_capacity = 256UZ * 1024UZ * 1024UZ,
                .material_capacity = 4096,
                .mesh_capacity = 4096,
                .model_capacity = 1024,
                .pipeline_capacity = 1024,
                .swapchain_format = context.swapchain.format(),
                .samples = VK_SAMPLE_COUNT_4_BIT,
        });

        if (!renderer_result) {
            error("Could not initialize renderer: {}", describe(renderer_result.error()));
            return false;
        }

        return true;
    }

    auto to_begin_error(QueueSetError queue_error) noexcept -> SwapchainBeginFrameError {
        return SwapchainBeginFrameError{
                .kind = queue_error.kind == QueueSetError::Kind::device_lost
                                ? SwapchainBeginFrameError::Kind::device_lost
                                : SwapchainBeginFrameError::Kind::fatal_error,
                .context = std::move(queue_error.context),
        };
    }

    auto begin_gpu_frame(VulkanContext &context,
                         FrameClock &clock) noexcept -> std::expected<SwapchainFrame, SwapchainBeginFrameError> {
        auto const slot = context.swapchain.current_slot();

        {
            auto const waiting = clock.scope(CpuPhase::slot_wait);
            if (auto begun = context.queue_set.begin_slot(slot); !begun) {
                return std::unexpected(to_begin_error(std::move(begun.error())));
            }
        }

        auto frame = [&] {
            auto const acquiring = clock.scope(CpuPhase::acquire);
            return context.swapchain.acquire(slot);
        }();
        if (!frame) {
            return std::unexpected(std::move(frame.error()));
        }

        auto buffer = context.queue_set.command_buffer(frame_graph::LogicalQueue::graphics);
        if (!buffer) {
            return std::unexpected(to_begin_error(std::move(buffer.error())));
        }

        frame->command_buffer = *buffer;

        return *frame;
    }

    auto end_gpu_frame(VulkanContext &context, SwapchainFrame const &frame, std::span<SubmitBatch const> planned,
                       FrameClock &clock) noexcept -> SwapchainFrameResult {
        auto submitting = std::optional<FrameClock::Scope>{};
        submitting.emplace(clock, CpuPhase::submit);

        if (auto const ended = vkEndCommandBuffer(frame.command_buffer); ended != VK_SUCCESS) {
            error("vkEndCommandBuffer failed with VkResult {}", static_cast<int>(ended));

            return ended == VK_ERROR_DEVICE_LOST ? SwapchainFrameResult::device_lost
                                                 : SwapchainFrameResult::fatal_error;
        }

        std::vector<SubmitBatch> batches{planned.begin(), planned.end()};
        if (batches.empty()) {
            batches.push_back(SubmitBatch{
                    .queue = frame_graph::LogicalQueue::graphics,
                    .command_buffer = frame.command_buffer,
                    .signal_index = 0,
                    .waits_swapchain_acquire = true,
                    .signals_render_finished = true,
            });
        }

        auto const submitted = context.queue_set.submit(batches, context.swapchain.image_available(frame.frame_index),
                                                        context.swapchain.render_finished(frame.image_index));
        if (!submitted) {
            return submitted.error().kind == QueueSetError::Kind::device_lost ? SwapchainFrameResult::device_lost
                                                                              : SwapchainFrameResult::fatal_error;
        }

        submitting.reset();
        auto const presenting = clock.scope(CpuPhase::present);
        return context.swapchain.present(frame);
    }

    auto draw(VulkanContext &context, Application &application, FrameClock &clock) noexcept -> bool {
        ZoneScopedNC("Draw", tracy::Color::RoyalBlue);

        auto frame = begin_gpu_frame(context, clock);

        if (!frame) {
            switch (frame.error().kind) {
                case SwapchainBeginFrameError::Kind::recreated:
                    return true;

                case SwapchainBeginFrameError::Kind::device_lost:
                    error("The Vulkan device was lost");
                    context.device_lost.store(true, std::memory_order_release);

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

        auto frame_ok = true;
        {
            auto const submitting = clock.scope(CpuPhase::scene_submit);

            if (auto *const terrain = application.active_terrain(); terrain != nullptr) {
                terrain->process_ready(*application.renderer, frame->command_buffer,
                                       application.active_scene()->physics_world.get());
            }

            auto submit_result = submit_scene(application);
            if (!submit_result) {
                error("Could not submit scene: {}", describe(submit_result.error()));
                frame_ok = false;
            }
        }

        auto const active_aspect = application.renderer->aspect(frame->frame_index);
        auto const active_camera =
                application.is_playing
                        ? application.game->camera(*application.active_scene(), active_aspect)
                        : CameraParams{
                                  .view = application.camera.view(),
                                  .projection = application.camera.projection(active_aspect),
                                  .near_clip = application.camera.near_clip(),
                                  .far_clip = application.camera.far_clip(),
                                  .vertical_fov_radians = glm::radians(application.camera.field_of_view_degrees()),
                          };

        if (frame_ok) {
            {
                auto const ui = clock.scope(CpuPhase::ui);
                application.imgui_renderer->begin_frame(gui::ImGuiFramebuffer{frame->extent, frame->format});
                application.on_ui(frame->frame_index);
                application.imgui_renderer->end_frame();
            }

            auto const preparing = clock.scope(CpuPhase::prepare);
            auto prepare_result = application.renderer->prepare_frame(
                    frame->command_buffer,
                    {
                            .view = active_camera.view,
                            .projection = active_camera.projection,
                            .near_clip = active_camera.near_clip,
                            .far_clip = active_camera.far_clip,
                            .vertical_fov_radians = active_camera.vertical_fov_radians,
                            .aspect_ratio = active_aspect,
                            .time = application.elapsed_time,
                    },
                    frame->frame_index);

            if (!prepare_result) {
                error("Could not prepare renderer frame: {}", describe(prepare_result.error()));

                frame_ok = false;
            }
        }

        if (frame_ok) {
            auto const recording = clock.scope(CpuPhase::record);
            auto record_result = application.renderer->record_frame(FrameRecordInfo{
                    .command_buffer = frame->command_buffer,
                    .swapchain_image =
                            SwapchainImage{
                                    .image = frame->image,
                                    .view = frame->image_view,
                                    .format = frame->format,
                                    .extent = frame->extent,
                            },
                    .frame_index = frame->frame_index,
                    .composite_target = application.is_playing && application.play_fullscreen
                                                ? CompositeTarget::swapchain
                                                : CompositeTarget::viewport_panel,
            });

            if (!record_result) {
                error("Could not record renderer frame: {}", describe(record_result.error()));

                frame_ok = false;
            }
        }

        auto const end_result = end_gpu_frame(context, *frame, application.renderer->submit_batches(), clock);
        clock.mark_present();

        if (!frame_ok) {
            return false;
        }

        switch (end_result) {
            case SwapchainFrameResult::success:
            case SwapchainFrameResult::recreated:
                return true;

            case SwapchainFrameResult::device_lost:
                error("The Vulkan device was lost");
                context.device_lost.store(true, std::memory_order_release);

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

        context->framebuffer_width.store(width, std::memory_order_relaxed);
        context->framebuffer_height.store(height, std::memory_order_relaxed);
        context->framebuffer_dirty.store(true, std::memory_order_relaxed);
    }

    auto imgui_wants_keyboard() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
    }

    auto imgui_wants_mouse() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
    }

    auto viewport_has_mouse(Application const &app) -> bool {
        return app.viewport_hovered || app.mouse_dragging || app.game_mouse_captured;
    }

    auto key_callback(GLFWwindow *window, int key, int, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (action == GLFW_PRESS && !imgui_wants_keyboard()) {
            app->on_event(KeyPressedEvent{.key = key, .modifiers = mods});
        }
        if (action == GLFW_RELEASE) {
            app->on_event(KeyReleasedEvent{.key = key, .modifiers = mods});
        }
    }

    auto mouse_button_callback(GLFWwindow *window, int button, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (action == GLFW_RELEASE) {
            if (!app->is_playing && button == GLFW_MOUSE_BUTTON_RIGHT && app->mouse_dragging) {
                app->release_mouse();
            }

            app->on_event(MouseButtonReleasedEvent{.button = button, .modifiers = mods});
            return;
        }

        if (action != GLFW_PRESS || (imgui_wants_mouse() && !viewport_has_mouse(*app))) {
            return;
        }

        if (!app->is_playing) {
            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                app->capture_mouse();
            }
        } else if (!app->play_fullscreen && app->viewport_hovered && !app->game_mouse_captured &&
                   !app->game->wants_cursor()) {
            app->game_mouse_captured = true;
            app->capture_mouse();
        }

        app->on_event(MouseButtonPressedEvent{.button = button, .modifiers = mods});
    }

    auto cursor_position_callback(GLFWwindow *window, double x_position, double y_position) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

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

        app->on_event(MouseMovedEvent{.delta_x = delta_x, .delta_y = delta_y});
    }

    auto scroll_callback(GLFWwindow *window, double x_offset, double y_offset) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr || (imgui_wants_mouse() && !viewport_has_mouse(*app))) {
            return;
        }

        app->on_event(MouseScrolledEvent{.delta_x = x_offset, .delta_y = y_offset});
    }

    auto focus_callback(GLFWwindow *window, int focused) noexcept -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // The editor leaves play when the window loses focus; an installed game (--player) has no editor to fall back
        // into, so it carries on, which an online game needs to keep its connection and turn.
        if (focused == GLFW_FALSE && app->is_playing && !app->player_mode) {
            app->stop();
        }
    }

    auto drop_callback(GLFWwindow *window, int path_count, char const **paths) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr || path_count <= 0) {
            return;
        }

        std::vector<std::filesystem::path> dropped;
        dropped.reserve(static_cast<std::size_t>(path_count));

        for (auto const *path: std::span{paths, static_cast<std::size_t>(path_count)}) {
            dropped.push_back(gui::utf8_to_path(path));
        }

        app->on_files_dropped(dropped);
    }

    auto install_window_callbacks(VulkanContext &context, Application &app) noexcept -> void {
        static WindowData wd{};
        wd.app = &app;
        wd.ctx = &context;
        glfwSetWindowUserPointer(context.window, &wd);

        glfwSetFramebufferSizeCallback(context.window, framebuffer_size_callback);
        glfwSetKeyCallback(context.window, key_callback);
        glfwSetMouseButtonCallback(context.window, mouse_button_callback);
        glfwSetCursorPosCallback(context.window, cursor_position_callback);
        glfwSetScrollCallback(context.window, scroll_callback);
        glfwSetWindowFocusCallback(context.window, focus_callback);
        glfwSetDropCallback(context.window, drop_callback);
    }

    auto wait_idle_or_exit(VkDevice device, std::string_view label) noexcept -> VkResult {
        auto const result = wait_idle_bounded(device, label);

        if (result == VK_TIMEOUT) {
            std::_Exit(EXIT_FAILURE);
        }

        return result;
    }

    auto report_device_lost(VulkanContext &context) noexcept -> void {
        error("The GPU device was lost or stopped responding; the application cannot continue.");

        if (context.window != nullptr) {
            glfwHideWindow(context.window);
        }

        auto const shown = show_fatal_dialog(
                "Graphics device lost",
                "The GPU stopped responding, so the renderer cannot continue.\n\n"
                "Please restart the application. Work since the last save could not be recovered.\n\n"
                "If this keeps happening, update your graphics driver and check the log for details.");

        if (!shown) {
            error("No dialog tool is available to show the device-lost notice; see the log above.");
        }
    }

    auto destroy_application(VulkanContext &context, Application &application) noexcept -> void {
        if (context.device != VK_NULL_HANDLE) {
            auto const result = wait_idle_or_exit(context.device, "destroy_application");

            if (result != VK_SUCCESS && result != VK_ERROR_DEVICE_LOST) {
                report_vk_error("vkDeviceWaitIdle(application destroy)", result);
            }
        }
        application.scene_save_job.reset();
        application.scene_load_job.reset();
        application.scene_pack.reset();

        application.debug_renderer.reset();
        application.imgui_renderer.reset();
        application.renderer->destroy();

        context.destroy();
    }
}

auto game_names() -> std::span<std::string_view const>;
auto create_game(std::string_view name) -> std::unique_ptr<IGame>;

namespace {

    constexpr std::array<std::pair<std::string_view, ScreenType>, 4> screen_type_choices{{
            {"windowed", ScreenType::windowed},
            {"fullscreen", ScreenType::fullscreen},
            {"borderless", ScreenType::borderless},
            {"headless", ScreenType::headless},
    }};

    [[nodiscard]] auto game_help() -> std::string {
        std::string names;
        for (auto const name: game_names()) {
            names += names.empty() ? "" : ", ";
            names += name;
        }
        return std::format("Which game to run: {} (default {})", names, game_names().front());
    }

    struct EngineArguments {
        explicit EngineArguments(CommandLine &cli) {
            auto packaging = cli.group("Packaging");
            packaging.value("--shader-pack", "FILE.lsp", "Precompiled shaders to use (default: shaders.lsp in the data directory)", shader_pack);
            packaging.value("--bake-shaders", "FILE.lsp", "Compile every shader under assets/shaders (plus assets/shaders/variants.txt) into a shader pack, then exit; needs no GPU or window", bake_shaders);
            packaging.value("--record-shaders", "FILE.lsp", "Write every shader this run compiles to a shader pack, then exit", record_shaders);
            packaging.value("--record-resources", "FILE.lbf", "Write the editor font and icons this run loaded to a resource pack, then exit", record_resources);
            packaging.value("--record-assets", "FILE.txt", "Write the data files this run opened to a list, then exit", record_assets);
            packaging.value("--screenshot-frame", "N", "Save a screenshot on frame N (see the screenshots directory)", screenshot_frame);
            packaging.value("--inject-keys", "FRAME:KEY,...", "Press GLFW key codes on given frames, e.g. 120:257 presses Enter on frame 120", inject_keys);
            packaging.value("--exit-after-frames", "N", "Exit after N frames (0: run until closed)", exit_after_frames);
            cli.group("Paths").value("--data-dir", "DIR", "Game data directory (default: an installed game's data/, else the working directory)", data_dir);
            auto display = cli.group("Display");
            display.choice<ScreenType>("--screen-type", "Window mode (default fullscreen)", screen_type_choices,
                                       screen_type);

            auto diagnostics = cli.group("Diagnostics");
            diagnostics.flag("--frame-graph-dump",
                             "Log the compiled frame graph whenever it changes", dump_frame_graph);
            diagnostics.value("--frame-graph-dot", "FILE.dot",
                              "Write the compiled frame graph as Graphviz whenever it changes (render with `dot "
                              "-Tsvg`).",
                              frame_graph_dot);

            auto game_group = cli.group("Game");
            game_group.value("--script", "FILE.lua", "Entry script of a Lua game (--game lua), relative to the data directory", script);
            game_group.flag("--player", "Run as an installed game would: fullscreen play, no editor", player);
            game_group.option("--game", "NAME", game_help(),
                              [this](std::string_view text) -> std::expected<void, std::string> {
                                  auto const names = game_names();

                                  if (std::ranges::find(names, text) == names.end()) {
                                      std::string expected;
                                      for (auto const name: names) {
                                          expected += expected.empty() ? "" : "|";
                                          expected += name;
                                      }
                                      return std::unexpected(std::format("'{}' (expected {})", text, expected));
                                  }

                                  game = std::string{text};
                                  return {};
                              });

            auto scene = cli.group("Scene");
            scene.value("--scene", "FILE.lbf", "Open a saved scene in place of the game's", open_scene);
            scene.value("--save-scene", "FILE.lbf", "Cook the (opened) scene into a self-contained .lbf", save_scene);
        }

        EngineArguments(EngineArguments const &) = delete;
        auto operator=(EngineArguments const &) -> EngineArguments & = delete;

        std::optional<ScreenType> screen_type;
        bool dump_frame_graph = false;
        std::string frame_graph_dot;
        std::string game;
        std::string data_dir;
        std::string script;
        bool player = false;
        std::string shader_pack;
        std::string record_shaders;
        std::string bake_shaders;
        std::string record_assets;
        std::string record_resources;
        std::uint32_t exit_after_frames = 0;
        std::uint32_t screenshot_frame = 0;
        std::string inject_keys;
        std::optional<std::filesystem::path> open_scene;
        std::optional<std::filesystem::path> save_scene;
    };

}

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    glfwPostEmptyEvent();
}

auto main(int argc, char **argv) -> int {
    CommandLine cli{"lathe", "Lathe engine"};
    EngineArguments engine{cli};
    PresentationArguments presentation{cli};
    BenchmarkArguments benchmark_arguments{cli};
    BenchmarkCompareArguments compare_arguments{cli};

    auto const parsed = cli.parse(argc, argv);
    if (!parsed) {
        std::println(stderr, "lathe: {}\nTry --help.", parsed.error());
        return EXIT_FAILURE;
    }
    if (*parsed == CommandLine::Outcome::help) {
        std::fputs(cli.help_text().c_str(), stdout);
        return EXIT_SUCCESS;
    }

    if (auto const compare_options = compare_arguments.options()) {
        return run_benchmark_compare(*compare_options);
    }

    Paths::set_current(Paths::resolve({.data_dir = engine.data_dir.empty() ? std::nullopt : std::optional{std::filesystem::path{engine.data_dir}}}));
    std::optional<GameManifest> manifest;

    if (auto const manifest_path = Paths::current().data_root() / "game.toml"; std::filesystem::exists(manifest_path)) {
        if (auto loaded = GameManifest::load(manifest_path)) {
            manifest = std::move(*loaded);
        } else {
            error("Ignoring game manifest: {}", loaded.error());
        }
    }

    auto const player_mode = manifest.has_value() || engine.player;

    if (!engine.record_assets.empty()) {
        Paths::start_access_recording();
    }

    if (!engine.record_resources.empty()) {
        start_resource_recording();
    } else if (auto const resource_pack_path = Paths::current().data_root() / "resources.lbf";
               std::filesystem::exists(resource_pack_path)) {
        if (auto pack = ResourcePack::open(resource_pack_path)) {
            info("Using {} bundled resources from '{}'", (*pack)->size(), resource_pack_path.string());
            install_resource_provider(*std::move(pack));
        } else {
            error("Ignoring resource pack: {}", describe(pack.error()));
        }
    }

    if (!engine.bake_shaders.empty()) {
        auto const baked = renderer::bake_shaders(Renderer::compiler(), engine.bake_shaders);

        if (!baked) {
            error("{}", baked.error());
            return EXIT_FAILURE;
        }

        info("Baked {} entry points and {} variants to '{}'", baked->entry_points, baked->variants,
             engine.bake_shaders);
        return EXIT_SUCCESS;
    }

    if (!engine.record_shaders.empty()) {
        renderer::start_shader_recording();
    } else {
        auto const pack_path = engine.shader_pack.empty() ? Paths::current().data_root() / "shaders.lsp"
                                                          : std::filesystem::path{engine.shader_pack};

        if (std::filesystem::exists(pack_path)) {
            if (auto pack = renderer::ShaderPack::load(pack_path)) {
                if (pack->matches_sources(Paths::current().data_root() / "assets" / "shaders")) {
                    info("Using {} precompiled shaders from '{}'", pack->size(), pack_path.string());
                    renderer::install_shader_pack(std::make_shared<renderer::ShaderPack const>(std::move(*pack)));
                } else {
                    warn("Ignoring shader pack '{}': the shader sources changed since it was recorded; "
                         "re-record it with --record-shaders",
                         pack_path.string());
                }
            } else {
                error("Ignoring shader pack: {}", pack.error());
            }
        }
    }

    Renderer::prefetch_shaders();

    info("Starting GLFW Vulkan test, data directory {}", Paths::current().data_root().string());

    std::signal(SIGINT, ctrl_c_handler);

    auto const screen_type = engine.screen_type.value_or(ScreenType::fullscreen);

    auto benchmark_options = benchmark_arguments.options();
    if (!benchmark_options) {
        error("Invalid benchmark arguments: {}", benchmark_options.error());
        return EXIT_FAILURE;
    }

    auto const dump_frame_graph = engine.dump_frame_graph;

    if (*benchmark_options) {
        set_fixed_random_seed((*benchmark_options)->seed);
    }

    VulkanContext context{};
    context.vsync = presentation.vsync.value_or(!benchmark_options->has_value());
    context.swapchain_image_count = presentation.swapchain_images.value_or(0U);

    auto const chosen_present_mode = presentation.present_mode.has_value() ? presentation.present_mode
                                     : (benchmark_options->has_value() && !context.vsync)
                                             ? std::optional{PresentModeChoice::immediate}
                                             : std::nullopt;
    if (chosen_present_mode) {
        switch (*chosen_present_mode) {
            case PresentModeChoice::immediate:
                context.present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                break;
            case PresentModeChoice::mailbox:
                context.present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            case PresentModeChoice::fifo:
                context.present_mode = VK_PRESENT_MODE_FIFO_KHR;
                break;
            case PresentModeChoice::fifo_relaxed:
                context.present_mode = VK_PRESENT_MODE_FIFO_RELAXED_KHR;
                break;
        }
    }

    context.sync_validation = true;

    if (!initialize_vulkan(context, screen_type)) {
        error("Vulkan initialization failed");

        context.destroy();

        return EXIT_FAILURE;
    }

    Application application{context};
    application.player_mode = player_mode;
    application.game_name = !engine.game.empty() ? engine.game
                            : manifest           ? manifest->game
                                                 : std::string{game_names().front()};
    application.script_entry = !engine.script.empty() ? engine.script : manifest ? manifest->entry : std::string{};
    application.game = create_game(application.game_name);

    application.game->attach_host(GameHost{
            .player_mode = player_mode,
            .request_exit = [window = context.window] { glfwSetWindowShouldClose(window, GLFW_TRUE); },
            .script_entry = !engine.script.empty() ? engine.script : manifest ? manifest->entry : std::string{},
            .effects = &application.effect_system,
    });

    if (manifest && !manifest->title.empty()) {
        glfwSetWindowTitle(context.window, manifest->title.c_str());
    }
    install_window_callbacks(context, application);

    if (!initialize_application(context, application)) {
        destroy_application(context, application);
        return EXIT_FAILURE;
    }

    application.renderer->set_async_candidates(Renderer::async_light_clustering | Renderer::async_occlusion |
                                               Renderer::async_gtao);
    application.renderer->set_frame_graph_dump(dump_frame_graph);
    application.renderer->set_frame_graph_dot(engine.frame_graph_dot);

    application.on_startup();

    // A packaged game ships its own cooked scene; it has to be open before play starts, because opening stops play.
    auto const manifest_scene = manifest && !manifest->scene.empty() && !engine.open_scene
                                        ? std::optional{Paths::current().data_root() / manifest->scene}
                                        : std::nullopt;

    if (manifest_scene) {
        application.renderer->queue_render_thread_event([&application, path = *manifest_scene] {
            application.play_when_loaded = true;
            application.request_open_scene(path);
        });
    } else if (player_mode) {
        application.renderer->queue_render_thread_event([&application] {
            application.play_fullscreen = true;
            application.play();
        });
    }

    if (engine.open_scene) {
        application.renderer->queue_render_thread_event(
                [&application, path = *engine.open_scene] { application.request_open_scene(path); });
    }
    if (engine.save_scene) {
        application.renderer->queue_render_thread_event(
                [&application, path = *engine.save_scene] { application.start_save_scene(path); });
    }

    std::optional<BenchmarkDriver> benchmark;
    if (*benchmark_options) {
        auto driver = BenchmarkDriver::create(std::move(**benchmark_options), application);

        if (!driver) {
            error("{}", driver.error());
            destroy_application(context, application);
            return EXIT_FAILURE;
        }

        ImGui::GetIO().IniFilename = nullptr;

        info("Benchmark: {} measured frames per run, seed {}, writing {}{}", driver->options().frame_count,
             driver->options().seed, driver->options().output_path.string(),
             context.vsync ? " (vsync on: displayed intervals are capped at the refresh rate)" : "");
        info("Benchmark: presenting with {} swapchain images", context.swapchain.image_count());
        benchmark.emplace(std::move(*driver));
    }

    FrameClock frame_clock;

    info("Initialization complete; close the window to exit");

    auto renderer_extent = context.swapchain.extent();
    constexpr auto resize_settle_time = std::chrono::milliseconds{150};
    auto pending_extent = renderer_extent;
    auto pending_since = std::chrono::steady_clock::now();
    auto last_frame_time = std::chrono::steady_clock::now();
    auto exit_code = EXIT_SUCCESS;

    std::uint32_t frames_run = 0;

    std::vector<std::pair<std::uint32_t, int>> injected_keys;

    for (auto const item: CommandLine::split(engine.inject_keys, ',')) {
        auto const colon = item.find(':');

        if (colon == std::string_view::npos) {
            continue;
        }

        std::uint32_t frame = 0;
        int key = 0;

        std::from_chars(item.data(), item.data() + colon, frame);
        std::from_chars(item.data() + colon + 1, item.data() + item.size(), key);

        injected_keys.emplace_back(frame, key);
    }

    while (g_running.load(std::memory_order_acquire) && context.running.load(std::memory_order_acquire) &&
           glfwWindowShouldClose(context.window) != GLFW_TRUE) {
        ZoneScopedNC("MainLoop", tracy::Color::Gray);

        auto const frame_number = frames_run++;

        if (engine.exit_after_frames != 0 && frame_number >= engine.exit_after_frames) {
            break;
        }

        if (engine.screenshot_frame != 0 && frame_number == engine.screenshot_frame) {
            application.request_screenshot();
        }

        for (auto const &[frame, key]: injected_keys) {
            if (frame == frame_number) {
                application.on_event(KeyPressedEvent{.key = key, .modifiers = 0});
            }
        }

        frame_clock.begin_frame();

        auto const width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const height = context.framebuffer_height.load(std::memory_order_relaxed);

        {
            auto const polling = frame_clock.scope(CpuPhase::events);

            if (width <= 0 || height <= 0) {
                glfwWaitEvents();
            } else {
                glfwPollEvents();
            }
        }

        if (glfwWindowShouldClose(context.window) == GLFW_TRUE) {
            break;
        }

        {
            auto const draining = frame_clock.scope(CpuPhase::events);
            application.renderer->drain_event_queue();
        }

        auto const current_width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const current_height = context.framebuffer_height.load(std::memory_order_relaxed);

        if (current_width <= 0 || current_height <= 0) {
            last_frame_time = std::chrono::steady_clock::now();
            continue;
        }

        auto const now = std::chrono::steady_clock::now();
        auto const delta_time =
                benchmark ? benchmark_timestep : std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;

        {
            auto const updating = frame_clock.scope(CpuPhase::update);

            application.elapsed_time += delta_time;
            application.camera.update(std::min(delta_time, 0.1F));

            if (benchmark) {
                benchmark->begin_frame(application);
            }

            application.update(delta_time);
        }

        request_resize_if_needed(context, current_width, current_height);

        if (!draw(context, application, frame_clock)) {
            exit_code = EXIT_FAILURE;
            break;
        }

        FrameMark;

        if (auto const &timings = application.renderer->last_frame_timings();
            timings.valid && application.can_start_recording_statistics()) {
            application.add_pass_timings(timings.passes);
        }

        if (benchmark) {
            auto const status = benchmark->end_frame(
                    application, context, frame_clock.finish_frame(),
                    BenchmarkRenderSize{.width = renderer_extent.width, .height = renderer_extent.height});

            if (status == BenchmarkDriver::Status::failed) {
                exit_code = EXIT_FAILURE;
                break;
            }
            if (status == BenchmarkDriver::Status::finished) {
                break;
            }
        }

        auto const desired_render_extent = [&]() -> VkExtent2D {
            if (application.is_playing && application.play_fullscreen) {
                return context.swapchain.extent();
            }

            auto const &size = application.viewport_content_size;
            if (size.x <= 0.0F || size.y <= 0.0F) {
                return renderer_extent;
            }

            return VkExtent2D{
                    .width = static_cast<std::uint32_t>(size.x),
                    .height = static_cast<std::uint32_t>(size.y),
            };
        }();

        auto target_render_extent = desired_render_extent;

        if (benchmark) {
            auto const size = benchmark->render_size(
                    BenchmarkRenderSize{.width = desired_render_extent.width, .height = desired_render_extent.height});
            target_render_extent = VkExtent2D{.width = size.width, .height = size.height};
        }

        if (!compare(target_render_extent, pending_extent)) {
            pending_extent = target_render_extent;
            pending_since = std::chrono::steady_clock::now();
        }

        auto const settled = benchmark.has_value() ||
                             std::chrono::steady_clock::now() - pending_since >= resize_settle_time;

        if (settled && !compare(target_render_extent, renderer_extent)) {
            auto resize_result = application.renderer->resize(target_render_extent);

            if (!resize_result) {
                error("Could not resize renderer: {}", describe(resize_result.error()));

                exit_code = EXIT_FAILURE;
                break;
            }

            info("Resizing renderer to {}x{}", target_render_extent.width, target_render_extent.height);
            renderer_extent = target_render_extent;
        }
    }

    context.running.store(false, std::memory_order_release);

    if (!engine.record_shaders.empty()) {
        if (auto const saved = renderer::finish_shader_recording(engine.record_shaders)) {
            info("Recorded {} shaders to '{}'", *saved, engine.record_shaders);
        } else {
            error("Could not write the shader pack: {}", saved.error());
            exit_code = EXIT_FAILURE;
        }
    }

    if (!engine.record_resources.empty()) {
        auto resources = finish_resource_recording();
        auto const count = resources.size();

        if (auto const written = ResourcePack::write(std::move(resources), engine.record_resources)) {
            info("Recorded {} resources to '{}'", count, engine.record_resources);
        } else {
            error("Could not write the resource pack: {}", describe(written.error()));
            exit_code = EXIT_FAILURE;
        }
    }

    if (!engine.record_assets.empty()) {
        std::ofstream list{engine.record_assets, std::ios::trunc};
        auto const files = Paths::finish_access_recording();

        for (auto const &file: files) {
            list << file << '\n';
        }

        if (list) {
            info("Recorded {} data files to '{}'", files.size(), engine.record_assets);
        } else {
            error("Could not write '{}'", engine.record_assets);
            exit_code = EXIT_FAILURE;
        }
    }

    if (context.device_lost.load(std::memory_order_acquire)) {
        exit_code = EXIT_FAILURE;
        report_device_lost(context);
    }

    if (benchmark && !benchmark->finished()) {
        error("Benchmark interrupted before it finished; results of unfinished runs were not written");
        exit_code = EXIT_FAILURE;
    }

    destroy_application(context, application);

    if (exit_code == EXIT_SUCCESS) {
        info("Application exited successfully");
    }

    return exit_code;
}
