#include <doctest/doctest.h>

#include "app/game_packager.hxx"
#include "core/game_manifest.hxx"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

namespace {

    namespace fs = std::filesystem;

    struct TempDir {
        fs::path path = fs::temp_directory_path() / ("lathe-packager-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

        TempDir() { fs::create_directories(path); }
        ~TempDir() {
            std::error_code ignored;
            fs::remove_all(path, ignored);
        }
        TempDir(TempDir const &) = delete;
        auto operator=(TempDir const &) -> TempDir & = delete;
    };

    auto write(fs::path const &path, std::string const &text) -> void {
        fs::create_directories(path.parent_path());
        std::ofstream{path, std::ios::binary} << text;
    }

    auto slurp(fs::path const &path) -> std::string {
        std::ifstream in{path, std::ios::binary};
        return {std::istreambuf_iterator<char>{in}, {}};
    }

    auto wait_for(GamePackageJob &job) -> void {
        while (!job.ready()) {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    }

}

TEST_SUITE("unit") {
    TEST_CASE("GameManifest: scene round-trips through its text form") {
        TempDir dir;
        GameManifest manifest{.name = "demo", .title = "Demo", .game = "lua", .version = "1.2.3", .entry = "a.lua", .scene = "scenes/main.lbf"};

        write(dir.path / "game.toml", manifest.to_text());
        auto const loaded = GameManifest::load(dir.path / "game.toml");

        REQUIRE(loaded.has_value());
        CHECK(loaded->scene == "scenes/main.lbf");
        CHECK(loaded->entry == "a.lua");
        CHECK(loaded->game == "lua");
    }

    TEST_CASE("write_tar: emits ustar headers, content and padding") {
        TempDir dir;
        write(dir.path / "pkg" / "a.txt", "hello");
        write(dir.path / "pkg" / "sub" / "b.bin", std::string(600, 'x'));

        REQUIRE(write_tar(dir.path / "pkg", dir.path / "pkg.tar").has_value());

        auto const tar = slurp(dir.path / "pkg.tar");

        // 2 headers + 1 + 2 data blocks + 2 end blocks.
        CHECK(tar.size() == 512U * 7U);
        CHECK(tar.substr(0, 9) == "pkg/a.txt");
        CHECK(tar.substr(257, 5) == "ustar");
        CHECK(tar.substr(512, 5) == "hello");
        CHECK(tar.substr(1024, 13) == "pkg/sub/b.bin");
        // Size field of the second entry: 600 bytes = 0o1130.
        CHECK(tar.substr(1024 + 124, 11) == "00000001130");

        unsigned checksum = 0;
        for (std::size_t i = 0; i < 512; ++i) {
            checksum += (i >= 148 && i < 156) ? 32U : static_cast<unsigned char>(tar[i]);
        }
        CHECK(std::stoul(tar.substr(148, 6), nullptr, 8) == checksum);
    }

    TEST_CASE("GamePackageJob: builds the installed-game layout") {
        TempDir dir;
        write(dir.path / "data" / "assets" / "scripts" / "x.lua", "return {}");
        write(dir.path / "data" / "shaders.lsp", "pack");
        write(dir.path / "bin" / "lathe", "exe");
        write(dir.path / "bin" / "libslang.so", "lib");
        write(dir.path / "staging.lbf", "scene");

        PackageOptions options;
        options.manifest = GameManifest{.name = "demo", .title = "Demo", .game = "lua", .version = "1", .entry = "assets/scripts/x.lua", .scene = "scenes/main.lbf"};
        options.output_dir = dir.path / "out";
        options.cooked_scene = dir.path / "staging.lbf";
        options.data_root = dir.path / "data";
        options.executable = dir.path / "bin" / "lathe";
        options.archive = true;

        fs::create_directories(options.output_dir);

        auto job = GamePackageJob::start(options);
        wait_for(job);
        auto const result = job.take();

        REQUIRE_MESSAGE(result.has_value(), result.error());

        auto const root = dir.path / "out" / "demo";
        CHECK(slurp(root / "demo") == "exe");
        CHECK(slurp(root / "libslang.so") == "lib");
        CHECK(slurp(root / "data" / "assets" / "scripts" / "x.lua") == "return {}");
        CHECK(slurp(root / "data" / "shaders.lsp") == "pack");
        CHECK(slurp(root / "data" / "scenes" / "main.lbf") == "scene");
        CHECK_FALSE(fs::exists(dir.path / "staging.lbf"));
        CHECK_FALSE(fs::exists(dir.path / "out" / "demo.partial"));
        CHECK(fs::exists(dir.path / "out" / "demo.tar"));

        auto const manifest = GameManifest::load(root / "data" / "game.toml");
        REQUIRE(manifest.has_value());
        CHECK(manifest->scene == "scenes/main.lbf");

        // A second run replaces the first.
        write(dir.path / "staging.lbf", "scene2");
        options.cooked_scene = dir.path / "staging.lbf";
        auto again = GamePackageJob::start(options);
        wait_for(again);
        CHECK(again.take().has_value());
        CHECK(slurp(root / "data" / "scenes" / "main.lbf") == "scene2");
    }

    TEST_CASE("GamePackageJob: refuses to replace a directory that is not a package") {
        TempDir dir;
        write(dir.path / "data" / "shaders.lsp", "pack");
        write(dir.path / "bin" / "lathe", "exe");
        write(dir.path / "staging.lbf", "scene");
        write(dir.path / "out" / "demo" / "precious.txt", "keep");

        PackageOptions options;
        options.manifest = GameManifest{.name = "demo", .game = "lua", .scene = "scenes/main.lbf"};
        options.output_dir = dir.path / "out";
        options.cooked_scene = dir.path / "staging.lbf";
        options.data_root = dir.path / "data";
        options.executable = dir.path / "bin" / "lathe";

        auto job = GamePackageJob::start(std::move(options));
        wait_for(job);

        CHECK_FALSE(job.take().has_value());
        CHECK(slurp(dir.path / "out" / "demo" / "precious.txt") == "keep");
    }

    TEST_CASE("GamePackageJob: rejects names that escape the output directory") {
        TempDir dir;

        PackageOptions options;
        options.manifest = GameManifest{.name = "../evil", .game = "lua"};
        options.output_dir = dir.path;

        auto job = GamePackageJob::start(std::move(options));
        wait_for(job);

        CHECK_FALSE(job.take().has_value());
    }
}
