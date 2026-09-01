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
#include "application.hxx"
#include "components.hxx"
#include "config.hxx"
#include "context.hxx"
#include "debug_renderer.hxx"
#include "editor_camera.hxx"
#include "engine_models.hxx"
#include "entity.hxx"
#include "error_describe.hxx"
#include "game.hxx"
#include "glm/gtc/type_ptr.hpp"
#include "imgui.h"
#include "imgui_renderer.hxx"
#include "implot.h"
#include "logger.hxx"
#include "physics.hxx"
#include "physics_world.hxx"
#include "renderer.hxx"
#include "renderer_application_policy.hxx"
#include "scene.hxx"
#include "shader_hot_reload_watcher.hxx"
#include "swapchain.hxx"
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


    auto submit_scene(Application &application) -> std::expected<void, RendererError> {
        ZoneScopedNC("SubmitScene", tracy::Color::RoyalBlue);

        auto &registry = application.active_scene()->get_registry();

        auto view = registry.view<Components::Transform const, Components::Model const>();

        for (auto [entity, transform, model]: view.each()) {
            auto const *override_component = registry.try_get<Components::MaterialOverride const>(entity);
            auto const material_override =
                    override_component != nullptr ? override_component->material : MaterialHandle{};

            auto const world_transform = systems::get_world_transform(registry, entity, transform);
            auto result = application.renderer->submit_model(model.model, world_transform, material_override);

            if (!result) {
                error("Could not submit scene object (model index {}): {}", model.model.index,
                      describe(result.error()));
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

        if (application.terrain) {
            application.terrain->submit(*application.renderer);
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

    auto draw(VulkanContext &context, Application &application) noexcept -> bool {
        ZoneScopedNC("Draw", tracy::Color::RoyalBlue);

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

        if (application.terrain) {
            // Before submit_scene(): puts the vertex rewrite copy before any
            // draw recorded into this command buffer, so the only hazard
            // direction is write-then-read -- which GeometryArena::write's
            // existing barrier already covers (see the terrain streaming
            // plan). Also gives a chunk zero-frame residency latency
            // instead of one.
            application.terrain->process_ready(*application.renderer, frame->command_buffer,
                                               application.active_scene()->physics_world.get());
        }

        auto frame_ok = true;
        auto submit_result = submit_scene(application);
        if (!submit_result) {
            error("Could not submit scene: {}", describe(submit_result.error()));
            frame_ok = false;
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
            application.imgui_renderer->begin_frame(gui::ImGuiFramebuffer{frame->extent, frame->format});
            {
                application.on_ui();
            }
            application.imgui_renderer->end_frame();

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
            if (auto const &timings = application.renderer->last_frame_timings();
                timings.valid && application.can_start_recording_statistics()) {
                application.timing_x += 1.0F;

                float running_total = 0.0F;

                for (auto stage = static_cast<std::uint32_t>(RenderStage::Culling); stage < stage_count; ++stage) {
                    running_total += timings.milliseconds[stage];
                    application.timing_buffers[stage].add_point(application.timing_x, running_total);
                }
            }


            auto record_result = application.renderer->record_frame<ApplicationOverlayPolicy>(
                    frame->command_buffer,
                    SwapchainImage{
                            .image = frame->image,
                            .view = frame->image_view,
                            .format = frame->format,
                            .extent = frame->extent,
                    },
                    frame->frame_index, application, active_camera.projection * active_camera.view);

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

    // Callbacks are installed (install_window_callbacks) before ImGuiRenderer
    // constructs its context (Application::on_startup(), later the same
    // frame before the first glfwPollEvents()), so today nothing can invoke
    // these before ImGui exists. Guarding here just means a future reordering
    // -- an early glfwPollEvents(), or callback installation moving earlier
    // -- can't turn into a null-context read of ImGui::GetIO().
    auto imgui_wants_keyboard() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureKeyboard;
    }

    auto imgui_wants_mouse() -> bool {
        return ImGui::GetCurrentContext() != nullptr && ImGui::GetIO().WantCaptureMouse;
    }

    auto key_callback(GLFWwindow *window, int key, int, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr || imgui_wants_keyboard()) {
            return;
        }

        if (action == GLFW_PRESS) {
            app->on_event(KeyPressedEvent{key, mods});
        }
        if (action == GLFW_RELEASE) {
            app->on_event(KeyReleasedEvent{key, mods});
        }
    }

    auto mouse_button_callback(GLFWwindow *window, int button, int action, int mods) -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        if (imgui_wants_mouse()) {
            return;
        }

        if (!app->is_playing) {
            if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_RIGHT) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            } else if (action == GLFW_RELEASE && button == GLFW_MOUSE_BUTTON_RIGHT) {
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }
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

        if (app == nullptr || imgui_wants_mouse()) {
            return;
        }

        app->on_event(MouseScrolledEvent{x_offset, y_offset});
    }

    auto focus_callback(GLFWwindow *window, int focused) noexcept -> void {
        auto *app = static_cast<WindowData *>(glfwGetWindowUserPointer(window))->app;

        if (app == nullptr) {
            return;
        }

        // Alt-tabbing (or any focus loss) while captured would otherwise leave
        // the cursor disabled and the game's input handling still integrating
        // whatever stray deltas the OS/compositor delivers to an unfocused window --
        // behaviour that varies between X11 and Wayland (see the platform
        // hint in initialize_glfw). Exiting play mode on focus loss sidesteps
        // that entirely rather than trying to special-case each platform.
        if (focused == GLFW_FALSE && app->is_playing) {
            app->stop();
        }
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
        application.debug_renderer.reset();
        application.imgui_renderer.reset();
        application.renderer->destroy();

        context.destroy();
    }
} // namespace

static std::atomic<bool> g_running{true};
static auto ctrl_c_handler(int) -> void {
    g_running.store(false, std::memory_order_relaxed);
    glfwPostEmptyEvent();
}

// Implemented by whichever game is linked into this executable (see
// game/src/main_entry.cxx) -- this is the one place the engine names
// game-specific content.
auto create_game() -> std::unique_ptr<IGame>;

auto main(int argc, char **argv) -> int {
    info("Starting GLFW Vulkan test at {}", std::filesystem::current_path().string());

    std::signal(SIGINT, ctrl_c_handler);

    auto const screen_type = parse_screen_type(argc, argv);

    VulkanContext context{};

    if (!initialize_vulkan(context, screen_type)) {
        error("Vulkan initialization failed");

        context.destroy();

        return EXIT_FAILURE;
    }


    Application application{context};
    application.game = create_game();
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
        ZoneScopedNC("MainLoop", tracy::Color::Gray);

        auto const width = context.framebuffer_width.load(std::memory_order_relaxed);
        auto const height = context.framebuffer_height.load(std::memory_order_relaxed);

        // Minimized / zero-sized framebuffer: nothing to render, so block
        // for the next event instead of busy-looping. Everywhere else we
        // poll (non-blocking), since we want to keep rendering every
        // iteration rather than waiting for input.
        //
        // This is the only glfwPollEvents()/glfwWaitEvents() call in the
        // loop, and it runs before any per-frame work below -- window
        // callbacks (focus_callback in particular, which can call
        // Application::stop()) therefore only ever fire between frames,
        // never while a frame is mid-flight. Keep it that way: a second
        // poll call added elsewhere in this loop would let a callback run
        // mid-frame instead.
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

        FrameMark;

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
