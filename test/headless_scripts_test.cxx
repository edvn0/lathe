#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

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

    scripts.attach_host(GameHost{.request_exit = [&quit] { quit = true; }, .scripts = {"assets/scripts/main.lua"}});
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

    scripts.attach_host(GameHost{.scripts = {"assets/scripts/main.lua"}});
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

    scripts.attach_host(GameHost{.scripts = {"assets/scripts/main.lua"}});
    scripts.on_populate(scene, nullptr, nullptr);

    CHECK_FALSE(scripts.ok());
    CHECK(scripts.error_text().find("not found") != std::string::npos);
}

TEST_CASE("several scripts run side by side, each with its own globals") {
    ScriptDirectory const directory{"lathe_headless_two", ""};

    std::ofstream{directory.root / "assets" / "scripts" / "a.lua"} << R"(
        count = 0
        function on_update(dt) count = count + 1; scene.find("a_" .. count) end
        function on_populate() scene.spawn("from_a") end
    )";
    std::ofstream{directory.root / "assets" / "scripts" / "b.lua"} << R"(
        count = 100
        function on_update(dt) count = count + 1 end
        function on_populate() scene.spawn("from_b") assert(count == 100, "b saw a's globals") end
    )";

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    ScriptSystem scripts;

    scripts.attach_host(GameHost{.scripts = {"assets/scripts/a.lua", "assets/scripts/b.lua"}});
    scripts.on_populate(scene, nullptr, nullptr);
    scripts.on_update(scene, 0.016F);

    CHECK(scripts.ok());
    CHECK((scene.find_entity("from_a") != entt::null));
    CHECK((scene.find_entity("from_b") != entt::null));
}

TEST_CASE("a script saved on disk is noticed") {
    ScriptDirectory const directory{"lathe_headless_watch", "function on_update(dt) end"};

    HeadlessSceneResources resources;
    Scene scene{resources};
    scene.on_scene_start();

    ScriptSystem scripts;

    scripts.attach_host(GameHost{.scripts = {"assets/scripts/main.lua"}});
    scripts.on_populate(scene, nullptr, nullptr);

    CHECK_FALSE(scripts.poll_changes());

    std::ofstream{directory.root / "assets" / "scripts" / "main.lua"} << "function on_update(dt) end -- edited";

    auto noticed = false;

    for (auto attempt = 0; attempt < 40 && !noticed; ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
        noticed = scripts.poll_changes();
    }

    CHECK(noticed);
}
