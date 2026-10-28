#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>

#include "core/paths.hxx"
#include "core/project_file.hxx"

namespace {
    auto scratch_directory(char const *name) -> std::filesystem::path {
        auto const directory = std::filesystem::temp_directory_path() / name;

        std::filesystem::remove_all(directory);
        std::filesystem::create_directories(directory);

        return directory;
    }
}

TEST_CASE("a project file round-trips and may leave out the game") {
    auto const root = scratch_directory("lathe_project_roundtrip");
    ProjectFile project{.manifest = {.name = "demo", .title = "Demo", .scene = "assets/scenes/a.lbf"}, .root = root};

    REQUIRE(project.save().has_value());

    auto const loaded = ProjectFile::load(root);

    REQUIRE(loaded.has_value());
    CHECK(loaded->manifest.name == "demo");
    CHECK(loaded->manifest.scene == "assets/scenes/a.lbf");
    CHECK(loaded->manifest.game.empty());
    CHECK(loaded->root == root);
}

TEST_CASE("find_root walks up to the nearest lathe.project") {
    auto const root = scratch_directory("lathe_project_find");
    auto const nested = root / "assets" / "scenes";

    std::filesystem::create_directories(nested);
    std::ofstream{root / ProjectFile::file_name} << "name = \"x\"\n";

    CHECK(ProjectFile::find_root(nested) == root);
    CHECK_FALSE(ProjectFile::find_root(std::filesystem::temp_directory_path() / "lathe_no_such_project").has_value());
}

TEST_CASE("a project roots the data at the project and keeps cache and state under .lathe") {
    auto const root = scratch_directory("lathe_project_paths");
    auto const paths = Paths::resolve({.project_root = root});

    CHECK(paths.in_project());
    CHECK_FALSE(paths.installed());
    CHECK(paths.data_root() == root);
    CHECK(paths.cache("a")->absolute() == root / ".lathe" / "cache" / "a");
    CHECK(paths.state("imgui.ini")->absolute() == root / ".lathe" / "state" / "imgui.ini");
}

TEST_CASE("a project's content shadows the engine's, which stays reachable") {
    auto const project = scratch_directory("lathe_mounts_project");
    auto const engine = scratch_directory("lathe_mounts_engine");

    std::filesystem::create_directories(engine / "assets" / "shaders");
    std::filesystem::create_directories(project / "assets" / "shaders");
    std::ofstream{engine / "assets" / "shared.txt"} << "engine";
    std::ofstream{engine / "assets" / "engine_only.txt"} << "engine";
    std::ofstream{project / "assets" / "shared.txt"} << "project";

    auto const paths = Paths::resolve({.project_root = project, .engine_dir = engine});

    CHECK(paths.engine_root() == engine);
    CHECK(paths.data("assets/shared.txt")->absolute() == project / "assets" / "shared.txt");
    CHECK(paths.data("assets/engine_only.txt")->absolute() == engine / "assets" / "engine_only.txt");
    CHECK(paths.data("assets/new.txt")->absolute() == project / "assets" / "new.txt");
    CHECK(paths.engine("assets/shared.txt")->absolute() == engine / "assets" / "shared.txt");
    CHECK(paths.data_directories("assets/shaders") == std::vector{project / "assets" / "shaders", engine / "assets" / "shaders"});
    CHECK(paths.relative_to_roots(engine / "assets" / "shared.txt") == std::filesystem::path{"assets/shared.txt"});
}
