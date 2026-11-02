// lathe-lua: runs a game script against a scene with no window, no renderer and no GPU. The same ScriptSystem the engine
// runs, on a Scene with HeadlessSceneResources: scene, entity, physics and the other libraries that need no GPU work;
// the ones that do (assets.load_model, effects, ...) report that they are unavailable. Exit status 0 when the script ran
// its frames without an error, 1 when it raised one, 2 on a bad command line. Meant for testing scripts and their logic.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "core/command_line.hxx"
#include "core/logger.hxx"
#include "core/paths.hxx"
#include "luagame/script_system.hxx"
#include "runtime/engine_main.hxx"
#include "rendering/scene.hxx"
#include "rendering/scene_resources.hxx"

auto main(int argc, char **argv) -> int {
    std::string script;
    std::string data_dir;
    std::string mounts;
    std::uint32_t frames = 1;
    float step = 1.0F / 60.0F;
    bool simulate_physics = true;

    CommandLine cli{"lathe-lua", "Runs a game script (Lua) without the engine: no window, renderer or GPU."};

    cli.positional("SCRIPT", "The entry script, relative to the data directory (default assets/scripts/main.lua)", script);

    auto run = cli.group("Run");
    run.value("--frames", "N", "Frames to run, calling on_update each (1)", frames, {.min = 0U, .max = 100000000U});
    run.value("--dt", "SECONDS", "Seconds each frame advances (1/60)", step, {.min = 0.0F, .max = 10.0F});
    run.flag("--no-physics", "Do not step the physics world", [&simulate_physics] { simulate_physics = false; });

    auto paths = cli.group("Paths");
    paths.value("--data-dir", "DIR", "Game data directory (default the working directory)", data_dir);
    paths.value("--mount", "DIR,DIR", "Extra content directories laid out like the data directory (see lathe.project)", mounts);

    auto const outcome = cli.parse(argc, argv);

    if (!outcome) {
        std::fprintf(stderr, "lathe-lua: %s\nTry --help.\n", outcome.error().c_str());
        return 2;
    }

    if (*outcome == CommandLine::Outcome::help) {
        std::fputs(cli.help_text().c_str(), stdout);
        return 0;
    }

    PathsOptions options;

    if (!data_dir.empty()) {
        options.data_dir = std::filesystem::path{data_dir};
    }

    for (auto const piece: CommandLine::split(mounts, ',')) {
        if (!piece.empty()) {
            options.mounts.emplace_back(std::string{piece});
        }
    }

    Paths::set_current(Paths::resolve(options));

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    std::atomic<bool> quit{false};

    info("lathe-lua: no render context. Scripts run, but assets, materials, render settings, animation, effects and text do "
         "nothing here (each call is reported once and returns a placeholder); use the engine to see a game.");

    ScriptSystem scripts;

    for (auto const &[name, opener]: native_script_modules()) {
        scripts.add_native_module(name, opener);
    }

    scripts.attach_host(GameHost{
            .player_mode = true,
            .request_exit = [&quit] { quit = true; },
            .script_entry = script,
    });
    scripts.on_populate(scene, nullptr, nullptr);

    for (std::uint32_t frame = 0; frame < frames && !quit && scripts.ok(); ++frame) {
        if (simulate_physics) {
            scene.step(step);
        }

        scripts.on_update(scene, step);
    }

    scripts.on_play_stop(scene);

    if (!scripts.ok()) {
        std::fprintf(stderr, "lathe-lua: %s\n", scripts.error_text().c_str());
        return 1;
    }

    return 0;
}
