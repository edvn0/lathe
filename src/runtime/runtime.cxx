#include "runtime/runtime.hxx"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <array>
#include <format>
#include <cstdint>
#include <glm/vec2.hpp>
#include <utility>

#include <entt/entt.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "imgui.h"
#include "physics/physics_world.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/file_browser.hxx"
#include "rendering/screenshot.hxx"
#include "rendering/toast.hxx"
#include "scene/components.hxx"

Runtime::Runtime(VulkanContext &ctx) noexcept :
    context(ctx), renderer(std::make_unique<Renderer>(context)),
    debug_renderer(std::make_unique<debug_draw::DebugRenderer>(*renderer)),
    text_renderer(std::make_unique<TextRenderer>()) {}

Runtime::~Runtime() {
    if (terrain) {
        terrain->wait_all();
    }

    shader_watcher_.stop();

    // A Lua game's finalizers release effects through effect_system, which is declared after `game` and so would be
    // destroyed first.
    game.reset();
}

auto Runtime::play() -> void {
    on_play_starting();

    runtime_scene = std::make_unique<Scene>(*renderer);
    runtime_scene->physics_settings = source_scene->physics_settings;
    runtime_scene->environment = source_scene->environment;
    game->clone_into_runtime(*source_scene, *runtime_scene);

    is_playing = true;
    active_scene()->on_scene_start();
    active_scene()->attach_debug_renderer(*debug_renderer);

    if (terrain) {
        terrain->on_physics_world_changed(active_scene()->physics_world.get());
    }

    game_mouse_captured = false;

    if (play_fullscreen) {
        if (!game->wants_cursor()) {
            capture_mouse();
        }
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoKeyboard;
    }
}

auto Runtime::stop() -> void {
    on_play_stopping();

    if (terrain) {
        terrain->on_physics_world_changed(nullptr);
    }

    game->on_play_stop(*active_scene());
    active_scene()->detach_debug_renderer();
    debug_renderer->clear_lines();
    active_scene()->on_scene_stop();
    is_playing = false;
    game_mouse_captured = false;

    runtime_scene.reset();

    release_mouse();
    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoKeyboard;
}

auto Runtime::open_scene(std::filesystem::path path) -> void {
    if (scene_load_job.has_value()) {
        scene_status = "A scene is already opening; wait for it to finish.";
        return;
    }

    if (is_playing) {
        stop();
    }

    on_scene_opening();

    scene_status = std::format("Opening '{}'...", gui::path_to_utf8(path.filename()));
    scene_load_job = SceneLoadJob::start(std::move(path));
}

auto Runtime::update_scene_load() -> void {
    if (!scene_load_job.has_value()) {
        return;
    }

    auto const path = scene_load_job->path();
    auto finished = scene_load_job->step(*source_scene, *renderer, engine_models);

    if (!finished.has_value()) {
        if (scene_load_job->phase() == SceneLoadJob::Phase::streaming) {
            auto const total = scene_load_job->models_total();
            scene_status = std::format("Opening '{}': streaming models {}/{}", gui::path_to_utf8(path.filename()),
                                       total - scene_load_job->models_remaining(*renderer), total);
        }
        return;
    }

    scene_load_job.reset();

    if (*finished) {
        scene_path = path;
        scene_pack = std::move((*finished)->pack);
        on_source_scene_replaced();
        if (play_when_loaded) {
            play_when_loaded = false;
            play_fullscreen = true;
            play();
        }

        scene_status = std::format("Opened '{}' in {:.2f}s{}", gui::path_to_utf8(path.filename()),
                                   (*finished)->seconds,
                                   (*finished)->instantiate.warnings.empty() ? "" : " with warnings; see the console");
    } else {
        error("Opening '{}' failed: {}", path.string(), describe(finished->error()));
        scene_status = std::format("Opening '{}' failed; see the console.", gui::path_to_utf8(path.filename()));
    }
}

auto Runtime::on_shutdown() -> void {
    scene_load_job.reset();
    scene_pack.reset();
}

auto Runtime::capture_mouse() -> void {
    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    has_last_mouse_position = false;

    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouse;
}

auto Runtime::release_mouse() -> void {
    glfwSetInputMode(context.window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    has_last_mouse_position = false;

    auto &io = ImGui::GetIO();
    io.ConfigFlags &= ~ImGuiConfigFlags_NoMouse;

    double x = 0.0;
    double y = 0.0;
    glfwGetCursorPos(context.window, &x, &y);
    io.AddMousePosEvent(static_cast<float>(x), static_cast<float>(y));
}

auto Runtime::cursor_over_game() const -> CursorPositionEvent {
    glm::dvec2 uv{-1.0};

    if (play_fullscreen) {
        double x = 0.0;
        double y = 0.0;
        int width = 0;
        int height = 0;
        glfwGetCursorPos(context.window, &x, &y);
        glfwGetWindowSize(context.window, &width, &height);

        if (width > 0 && height > 0) {
            uv = glm::dvec2{x / width, y / height};
        }
    } else if (viewport_content_size.x > 0.0F && viewport_content_size.y > 0.0F) {
        auto const mouse = ImGui::GetIO().MousePos;
        uv = glm::dvec2{(mouse.x - viewport_screen_pos.x) / viewport_content_size.x,
                        (mouse.y - viewport_screen_pos.y) / viewport_content_size.y};
    }

    return CursorPositionEvent{
            .ndc_x = (uv.x * 2.0) - 1.0,
            .ndc_y = 1.0 - (uv.y * 2.0),
            .inside = uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0,
    };
}

auto Runtime::update(float delta_time) -> void {
    ZoneScopedNC("RuntimeUpdate", tracy::Color::Firebrick);

    last_delta_time = delta_time;

    update_scene_load();

    if (auto *const streaming_terrain = active_terrain(); streaming_terrain != nullptr) {
        auto const observer = observer_position();
        auto camera_xz = glm::vec2{observer.x, observer.z};

        if (is_playing) {
            auto &registry = active_scene()->get_registry();
            auto view = registry.view<Components::Transform const, Components::PlayerTag const>();

            for (auto &&[entity, transform]: view.each()) {
                camera_xz = glm::vec2{transform.position.x, transform.position.z};
                break;
            }
        }

        streaming_terrain->update(camera_xz);
    }

    if (!is_playing) {
        return;
    }

    active_scene()->step(delta_time);

    if (game->wants_cursor()) {
        game->on_cursor_position(*active_scene(), cursor_over_game());
    }

    game->on_update(*active_scene(), delta_time);
    systems::lifetime(active_scene()->get_registry(), *active_scene()->physics_world, delta_time);
}

auto Runtime::on_ui(std::uint32_t frame_index) -> void {
    (void) frame_index;

    if (is_playing && play_fullscreen && player_mode && game && game_hooks_enabled) {
        game->on_ui(*active_scene(), *renderer);
    }

    gui::clear_toast_anchor();
    gui::render_toasts();
}

auto Runtime::register_overlays() -> void {
    auto add = [this](OverlayDesc desc) {
        auto const name = desc.name;

        if (auto registration = renderer->register_overlay(std::move(desc)); registration) {
            overlays.push_back(std::move(*registration));
        } else {
            error("Could not register the '{}' overlay: {}", name, describe(registration.error()));
        }
    };

    add(OverlayDesc{
            .name = "Debug lines",
            .stage = OverlayStage::scene,
            .order = 0,
            .prepare = {},
            .record = [this](OverlayRecordContext const &overlay_context) { debug_renderer->record(overlay_context); },
    });

    add(OverlayDesc{
            .name = "ImGui",
            .stage = OverlayStage::ui,
            .order = 0,
            .prepare = {},
            .record =
                    [this](OverlayRecordContext const &overlay_context) {
                        imgui_renderer->record(overlay_context);
                    },
    });
}

auto Runtime::on_startup() -> void {
    std::array const shader_directories{
            data_path("assets/shaders").absolute(),
    };
    if (!player_mode && !shader_watcher_.start(renderer->shader_change_queue(), shader_directories)) {
        error("Shader hot-reload watcher failed to start -- shaders will not live-reload this run");
    }
    imgui_renderer = std::make_unique<gui::ImGuiRenderer>(
            *renderer, gui::FontChoice{
                               .font_path = "assets/fonts/GoogleSansCode-Regular.ttf",
                               .size = 12,
                       });
    create_ui_resources();
    register_overlays();
    renderer->queue_render_thread_event([this] {
        auto models = create_engine_models(*renderer);

        if (!models) {
            error("Fatal: could not create built-in engine models: {}", describe(models.error()));

            context.running.store(false, std::memory_order_release);
            glfwPostEmptyEvent();

            return;
        }

        engine_models = *models;

        if (auto const created = particle_system.create(renderer->game_gpu()); !created) {
            warn("Could not set up particles: {}", describe(created.error()));
        }

        effect_system.create(renderer->game_gpu());

        if (ray_tracing_poc_enabled) {
            if (!context.ray_query_supported) {
                warn("--rt-poc needs VK_KHR_ray_query, which this device does not support");
            } else if (auto const created = ray_tracing_poc.create(*renderer); !created) {
                warn("Could not set up the ray tracing PoC: {}", describe(created.error()));
            }
        }

        particle_system.set_text_source(text_renderer.get());

        if (auto const created = text_renderer->create(*renderer, renderer->game_gpu()); !created) {
            warn("Could not set up text rendering: {}", describe(created.error()));
        }

        renderer->set_game_graph_hook([this](GameGraph &graph) {
            // The text renderer goes first: it opens the frame's upload buffers, and emitters on text add to them.
            text_renderer->declare(graph, active_scene() != nullptr ? &active_scene()->get_registry() : nullptr);
            if (auto const *scene = active_scene(); scene != nullptr) {
                particle_system.declare(graph, scene->get_registry(), &renderer->material_storage(), last_delta_time);
            }
            ray_tracing_poc.declare(graph, frame_view, frame_projection, renderer->directional_light().direction);
            if (game && game_hooks_enabled) {
                effect_system.declare(graph);
                game->on_frame_graph(graph, last_delta_time);
            }
        });

        game->on_populate(*source_scene, *renderer, engine_models);
        on_source_scene_replaced();

        create_terrain();
    });
}

auto Runtime::create_terrain() -> void {
    auto terrain_info = game->terrain_create_info(*renderer);

    if (!terrain_info) {
        return;
    }

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

auto Runtime::switch_game(std::string name) -> void {
    renderer->queue_render_thread_event([this, name = std::move(name)] {
        if (scene_load_job.has_value()) {
            scene_status = "A scene is opening; wait for it to finish before loading a game.";
            return;
        }

        auto next = game_factory ? game_factory(name) : nullptr;

        if (next == nullptr) {
            scene_status = std::format("Unknown game '{}'.", name);
            return;
        }

        if (is_playing) {
            stop();
        }

        on_scene_opening();

        if (auto const idle = renderer->wait_idle(); !idle.has_value()) {
            warn("{}", describe(idle.error()));
        }

        if (terrain) {
            terrain->wait_all();
            terrain.reset();
        }

        // The old game's finalizers release effects through effect_system, so it goes before anything new binds.
        game.reset();
        game = std::move(next);
        game_name = name;

        game->attach_host(GameHost{
                .player_mode = player_mode,
                .request_exit = [window = context.window] { glfwSetWindowShouldClose(window, GLFW_TRUE); },
                .script_entry = script_entry,
                .effects = &effect_system,
                .text = text_renderer.get(),
        });

        scene_path.clear();
        scene_pack.reset();
        game->on_populate(*source_scene, *renderer, engine_models);
        on_source_scene_replaced();
        create_terrain();

        scene_status = std::format("Loaded game '{}'.", name);
    });
}

auto Runtime::request_screenshot() -> void {
    renderer->request_screenshot(screenshot_viewport_only ? ScreenshotSource::viewport : ScreenshotSource::window);
}

auto Runtime::on_event(KeyPressedEvent ev) -> bool {
    if (ev.key == GLFW_KEY_F12) {
        request_screenshot();
    }
    if (ev.key >= GLFW_KEY_F1 && ev.key < GLFW_KEY_F1 + static_cast<std::int32_t>(view_mode_names.size())) {
        renderer->set_view_mode(view_mode_names[static_cast<std::size_t>(ev.key - GLFW_KEY_F1)].first);
    }

    if (is_playing) {
        if (ev.key == GLFW_KEY_ESCAPE) {
            if (game_mouse_captured) {
                game_mouse_captured = false;
                release_mouse();
                return true;
            }

            if (!player_mode) {
                stop();
                return true;
            }
        }

        game->on_key_pressed(*active_scene(), ev);
    }

    return true;
}

auto Runtime::on_event(KeyReleasedEvent ev) -> bool {
    if (is_playing) {
        game->on_key_released(*active_scene(), ev);
    }

    return true;
}

auto Runtime::on_event(MouseMovedEvent ev) -> bool {
    if (is_playing && (play_fullscreen || game_mouse_captured)) {
        game->on_mouse_moved(*active_scene(), ev);
    }

    return true;
}

auto Runtime::on_event(MouseScrolledEvent ev) -> bool {
    (void) ev;

    return true;
}

auto Runtime::on_event(MouseButtonPressedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = true;
    }

    if (is_playing) {
        game->on_mouse_button_pressed(*active_scene(), ev);
    }

    return true;
}

auto Runtime::on_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = false;
    }

    if (is_playing) {
        game->on_mouse_button_released(*active_scene(), ev);
    }

    return true;
}
