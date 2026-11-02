#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "core/paths.hxx"
#include "luagame/script_system.hxx"
#include "rendering/scene.hxx"
#include "rendering/scene_resources.hxx"

namespace {
    // A data directory holding one script, made current for the test and put back after.
    struct ScriptDirectory {
        std::filesystem::path root;
        Paths previous = Paths::current();

        explicit ScriptDirectory(char const *name, std::string const &source) {
            root = std::filesystem::temp_directory_path() / name;
            std::filesystem::remove_all(root);
            std::filesystem::create_directories(root / "assets" / "scripts");
            std::ofstream{root / "assets" / "scripts" / "main.lua"} << source;
            Paths::set_current(Paths::resolve({.data_dir = root}));
        }

        ~ScriptDirectory() { Paths::set_current(previous); }
    };
}

TEST_CASE("a script runs against a scene with no renderer") {
    ScriptDirectory const directory{"lathe_headless_ok", R"(
        local ticks = 0
        function on_populate()
            scene.spawn("one"):set_position(1, 2, 3)
        end
        function on_update(dt)
            ticks = ticks + 1
            local x, y, z = scene.find("one"):position()
            assert(x == 1 and y == 2 and z == 3, "the entity moved")
            if ticks == 3 then game.quit() end
        end
    )"};

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    auto quit = false;
    ScriptSystem scripts;

    scripts.attach_host(GameHost{.request_exit = [&quit] { quit = true; }});
    scripts.on_populate(scene, nullptr, nullptr);

    REQUIRE(scripts.ok());

    for (auto frame = 0; frame < 10 && !quit; ++frame) {
        scene.step(0.016F);
        scripts.on_update(scene, 0.016F);
    }

    CHECK(quit);
    CHECK(scripts.ok());
    CHECK((scene.find_entity("one") != entt::null));
}

TEST_CASE("a script error is reported and stops the run") {
    ScriptDirectory const directory{"lathe_headless_error", R"(
        function on_update(dt) error("boom") end
    )"};

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    ScriptSystem scripts;

    scripts.attach_host(GameHost{});
    scripts.on_populate(scene, nullptr, nullptr);
    scripts.on_update(scene, 0.016F);

    CHECK_FALSE(scripts.ok());
    CHECK(scripts.error_text().find("boom") != std::string::npos);
}

TEST_CASE("a missing entry script is an error, not a crash") {
    ScriptDirectory const directory{"lathe_headless_missing", ""};

    std::filesystem::remove(directory.root / "assets" / "scripts" / "main.lua");

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    ScriptSystem scripts;

    scripts.attach_host(GameHost{});
    scripts.on_populate(scene, nullptr, nullptr);

    CHECK_FALSE(scripts.ok());
    CHECK(scripts.error_text().find("not found") != std::string::npos);
}
