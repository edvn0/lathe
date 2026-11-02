#include "runtime/runtime.hxx"
#include "runtime/engine_cvars.hxx"

#include <volk.h>

#include <GLFW/glfw3.h>

#include <array>
#include <algorithm>
#include <format>
#include <cstdint>
#include <glm/vec2.hpp>
#include <utility>

#include <entt/entt.hpp>

#include "core/error_describe.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "imgui.h"
#include "gpu/performance_tier.hxx"
#include "physics/physics_world.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/file_browser.hxx"
#include "rendering/screenshot.hxx"
#include "rendering/toast.hxx"
#include "scene/components.hxx"

Runtime::Runtime(VulkanContext &ctx) noexcept :
    context(ctx), renderer(std::make_unique<Renderer>(context)),
    debug_renderer(std::make_unique<debug_draw::DebugRenderer>(*renderer)),
    text_renderer(std::make_unique<TextRenderer>()) {
    event_subscriptions.push_back(event_bus.subscribe<KeyPressedEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
    event_subscriptions.push_back(event_bus.subscribe<KeyReleasedEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
    event_subscriptions.push_back(event_bus.subscribe<MouseMovedEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
    event_subscriptions.push_back(event_bus.subscribe<MouseScrolledEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
    event_subscriptions.push_back(event_bus.subscribe<MouseButtonPressedEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
    event_subscriptions.push_back(event_bus.subscribe<MouseButtonReleasedEvent>([this](auto const &ev) { return handle_event(ev); }, 100));
}

auto Runtime::subscribe_game_input() -> void {
    auto const context_for_input = [this] {
        return InputContext{
                .bus = event_bus,
                .scene = [this] { return active_scene(); },
                .playing = [this] { return is_playing; },
                .mouse_look = [this] { return play_fullscreen || game_mouse_captured; },
        };
    };

    game->subscribe_input(context_for_input());

    if (scripts) {
        scripts->subscribe_input(context_for_input());
    }
}

auto Runtime::setup_game(std::string_view name) const -> std::optional<GameSetup> {
    if (std::ranges::find(available_games, name) == available_games.end() || !game_factory) {
        return std::nullopt;
    }

    auto setup = GameSetup{.game = game_factory(name), .scripts = std::make_unique<ScriptSystem>()};

    for (auto const &[module_name, opener]: native_script_modules) {
        setup.scripts->add_native_module(module_name, opener);
    }

    return setup;
}

auto Runtime::repopulate() -> void {
    renderer->queue_render_thread_event([this] {
        if (is_playing) {
            stop();
        }

        populate_game();
        scene_path.clear();
        scene_pack.reset();
        on_source_scene_replaced();
    });
}

auto Runtime::apply_scripts(std::vector<std::string> paths) -> void {
    script_paths = std::move(paths);

    if (scripts) {
        scripts->set_scripts(script_paths);
    }

    repopulate();
}

auto Runtime::populate_game() -> void {
    game->on_populate(*source_scene, *renderer, engine_models);

    if (scripts) {
        scripts->on_populate(*source_scene, renderer.get(), &engine_models);
    }
}

auto Runtime::wants_cursor() const -> bool {
    return game->wants_cursor() || (scripts && scripts->wants_cursor());
}

auto Runtime::game_camera(float aspect_ratio) const -> CameraParams {
    if (scripts && scripts->has_camera()) {
        return scripts->camera(*active_scene(), aspect_ratio);
    }

    return game->camera(*active_scene(), aspect_ratio);
}

auto Runtime::behaviour_names() -> std::span<std::string const> {
    return scripts ? scripts->behaviour_names() : game->behaviour_names();
}

auto Runtime::behaviour_schema(std::string const &name) -> ScriptSchema const * {
    return scripts ? scripts->behaviour_schema(name) : game->behaviour_schema(name);
}

auto Runtime::reload_behaviours() -> void {
    if (scripts) {
        scripts->reload_behaviours();
    } else {
        game->reload_behaviours();
    }
}

Runtime::~Runtime() {
    if (terrain) {
        terrain->wait_all();
    }

    shader_watcher_.stop();

    // A script's finalizers release effects through effect_system, which is declared after `game` and so would be
    // destroyed first.
    scripts.reset();
    game.reset();
}

auto Runtime::play() -> void {
    on_play_starting();

    runtime_scene = std::make_unique<Scene>(*renderer);
    runtime_scene->physics_settings = source_scene->physics_settings;
    runtime_scene->environment = source_scene->environment;
    if (scripts) {
        scripts->clone_into_runtime(*source_scene, *runtime_scene);
    } else {
        game->clone_into_runtime(*source_scene, *runtime_scene);
    }

    is_playing = true;
    active_scene()->on_scene_start();
    active_scene()->attach_debug_renderer(*debug_renderer);

    if (terrain) {
        terrain->on_physics_world_changed(active_scene()->physics_world.get());
    }

    game_mouse_captured = false;

    if (play_fullscreen) {
        if (!wants_cursor()) {
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

    if (scripts) {
        scripts->on_play_stop(*active_scene());
    }
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

    // A script saved in an editor reloads: the scene is rebuilt from the game and the scripts (not while one is opening).
    if (scripts && !scene_load_job.has_value() && scripts->poll_changes()) {
        info("[lua] a script changed on disk, reloading");
        repopulate();
    }

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

    if (scripts && scripts->wants_cursor()) {
        scripts->on_cursor_position(*active_scene(), cursor_over_game());
    }

    game->on_update(*active_scene(), delta_time);

    if (scripts) {
        scripts->on_update(*active_scene(), delta_time);
    }
    systems::lifetime(active_scene()->get_registry(), *active_scene()->physics_world, delta_time);
}

auto Runtime::on_ui(std::uint32_t frame_index) -> void {
    (void) frame_index;

    if (is_playing && play_fullscreen && player_mode && game && game_hooks_enabled) {
        game->on_ui(*active_scene(), *renderer);

        if (scripts) {
            scripts->on_ui(*active_scene(), *renderer);
        }
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
    auto const shader_directories = Paths::current().data_directories("assets/shaders");
    if (!player_mode && !shader_watcher_.start(renderer->shader_change_queue(), shader_directories)) {
        error("Shader hot-reload watcher failed to start -- shaders will not live-reload this run");
    }
    imgui_renderer = std::make_unique<gui::ImGuiRenderer>(
            *renderer, gui::FontChoice{
                               .font_path = "assets/fonts/GoogleSansCode-Regular.ttf",
                               .size = cvars::ui_font_size.get(),
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

        if (auto const created = planet_system.create(renderer->game_gpu()); !created) {
            warn("Could not set up planet atmospheres: {}", describe(created.error()));
        }

        // The GPU class that "auto" quality settings follow: detected, unless the project names one.
        auto const detected_tier = performance_tier::detect(context.device_type, context.device_local_bytes);
        auto const tier = performance_tier::parse(render_config.performance_tier).value_or(detected_tier);
        auto const tier_profile = performance_tier::profile(tier);
        info("Performance tier: {} ({}, detected {})", performance_tier::name(tier),
             render_config.performance_tier == "auto" ? "auto" : "set by the project", performance_tier::name(detected_tier));

        if (render_config.ray_traced_shadows && context.ray_query_supported) {
            if (auto const created = renderer->ray_traced_shadows().create(*renderer); !created) {
                warn("Could not set up ray traced shadows: {}", describe(created.error()));
            } else {
                auto const &resolution = render_config.ray_traced_shadow_resolution;
                renderer->ray_traced_shadows().settings().resolution_shift =
                        resolution == "quarter" ? 2U
                        : resolution == "half"  ? 1U
                        : resolution == "full"  ? 0U
                                                : tier_profile.rt_shadow_resolution_shift;
            }
        }

        if (gi_config.enabled) {
            if (!context.ray_query_supported) {
                info("Probe GI off: this device does not support VK_KHR_ray_query");
            } else if (auto const created = renderer->probe_volume().create(*renderer); !created) {
                warn("Could not set up the probe volume: {}", describe(created.error()));
            } else {
                // The coarser cascade: same probe counts at four times the spacing, updated every few frames.
                auto &far = renderer->probe_volume_far();
                auto &near_settings = renderer->probe_volume().settings();
                far.settings() = near_settings;
                far.settings().spacing = near_settings.spacing * 4.0F;
                far.settings().normal_bias = near_settings.normal_bias * 4.0F;
                far.settings().view_bias = near_settings.view_bias * 4.0F;
                far.settings().cascade = 1;
                far.settings().update_interval = gi_config.far_update_interval != 0 ? gi_config.far_update_interval
                                                                                    : tier_profile.far_cascade_update_interval;
                far.settings().debug_view = 0;
                if (!gi_config.far_cascade) {
                    // Left uncreated: the cascade is simply not there.
                } else if (auto const far_created = far.create(*renderer); !far_created) {
                    warn("Could not set up the far probe cascade: {}", describe(far_created.error()));
                }
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
            if (auto const *scene = active_scene(); scene != nullptr) {
                planet_system.declare(graph, scene->get_registry(), frame_view, frame_projection, frame_origin,
                                      renderer->directional_light().direction,
                                      renderer->directional_light().intensity);
            }
            if (game && game_hooks_enabled) {
                effect_system.declare(graph);
                game->on_frame_graph(graph, last_delta_time);
            }
        });

        populate_game();
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

        auto next = setup_game(name);

        if (!next) {
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
        scripts.reset();
        game.reset();
        game = std::move(next->game);
        scripts = std::move(next->scripts);
        subscribe_game_input();
        game_name = name;

        auto const host = GameHost{
                .player_mode = player_mode,
                .request_exit = [window = context.window] { glfwSetWindowShouldClose(window, GLFW_TRUE); },
                .scripts = script_paths,
                .effects = &effect_system,
                .text = text_renderer.get(),
        };

        game->attach_host(host);

        if (scripts) {
            scripts->attach_host(host);
        }

        scene_path.clear();
        scene_pack.reset();
        populate_game();
        on_source_scene_replaced();
        create_terrain();

        scene_status = std::format("Loaded game '{}'.", name);
    });
}

auto Runtime::switch_project(std::filesystem::path root) -> void {
    renderer->queue_render_thread_event([this, root = std::move(root)] {
        if (scene_load_job.has_value()) {
            scene_status = "A scene is opening; wait for it to finish before opening a project.";
            return;
        }

        auto loaded = ProjectFile::load(root);

        if (!loaded) {
            error("Could not open the project: {}", loaded.error());
            scene_status = std::format("Could not open the project: {}", loaded.error());
            return;
        }

        if (is_playing) {
            stop();
        }

        if (auto const idle = renderer->wait_idle(); !idle.has_value()) {
            warn("{}", describe(idle.error()));
        }

        // Everything that reads Paths from now on sees the new project; the engine's resources do not move.
        Paths::set_current(Paths::resolve({.project_root = loaded->root, .engine_dir = Paths::current().engine_root()}));
        project = std::move(*loaded);
        RecentProjects::user_default().remember(project->root);
        info("Project '{}' at {}", project->manifest.name, project->root.string());

        shader_watcher_.stop();

        if (!player_mode && !shader_watcher_.start(renderer->shader_change_queue(),
                                                    Paths::current().data_directories("assets/shaders"))) {
            error("Shader hot-reload watcher failed to restart");
        }

        if (imgui_renderer) {
            imgui_renderer->use_project_layout();
        }

        auto const &title = project->manifest.title.empty() ? project->manifest.name : project->manifest.title;
        glfwSetWindowTitle(context.window, title.c_str());

        script_paths = project->manifest.scripts;
        switch_game(project->manifest.game.empty() ? std::string{available_games.front()} : project->manifest.game);

        // Queued after the game switch, which replaces the source scene the project's scene then loads into.
        if (!project->manifest.scene.empty()) {
            renderer->queue_render_thread_event([this, scene = project->root / project->manifest.scene] {
                open_scene(scene);
            });
        }
    });
}

auto Runtime::request_screenshot() -> void {
    renderer->request_screenshot(screenshot_viewport_only ? ScreenshotSource::viewport : ScreenshotSource::window);
}

auto Runtime::handle_event(KeyPressedEvent ev) -> bool {
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
    }

    // Everything else goes on to the game (its handlers run later): consuming it here left a playing game with no keys.
    return false;
}

auto Runtime::handle_event(KeyReleasedEvent ev) -> bool {
    (void) ev;
    return false;
}

auto Runtime::handle_event(MouseMovedEvent ev) -> bool {
    (void) ev;
    return false;
}

auto Runtime::handle_event(MouseScrolledEvent ev) -> bool {
    (void) ev;

    return false;
}

auto Runtime::handle_event(MouseButtonPressedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = true;
    }

    return false;
}

auto Runtime::handle_event(MouseButtonReleasedEvent ev) -> bool {
    if (ev.button == GLFW_MOUSE_BUTTON_RIGHT) {
        mouse_dragging = false;
    }

    return false;
}
