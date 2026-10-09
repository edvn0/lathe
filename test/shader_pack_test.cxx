#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "assets/shader_pack.hxx"

namespace {
    auto write_file(std::filesystem::path const &path, std::string const &contents) -> void {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream{path, std::ios::binary} << contents;
    }

    struct TempDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() / "lathe_shader_pack_test";

        TempDirectory() {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }

        ~TempDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };
}

TEST_CASE("shader pack: source hash tracks shader sources") {
    TempDirectory const temp;
    auto const shaders = temp.path / "shaders";

    CHECK_FALSE(renderer::hash_shader_sources(shaders).has_value());

    write_file(shaders / "a.slang", "float a;");
    write_file(shaders / "sub" / "b.slang", "float b;");
    write_file(shaders / "notes.txt", "ignored");

    auto const original = renderer::hash_shader_sources(shaders);
    REQUIRE(original.has_value());
    CHECK(*original != 0);
    CHECK(renderer::hash_shader_sources(shaders) == original);

    write_file(shaders / "notes.txt", "still ignored");
    CHECK(renderer::hash_shader_sources(shaders) == original);

    write_file(shaders / "sub" / "b.slang", "float b2;");
    CHECK(renderer::hash_shader_sources(shaders) != original);
}

TEST_CASE("shader pack: staleness check") {
    TempDirectory const temp;
    auto const shaders = temp.path / "shaders";
    write_file(shaders / "a.slang", "float a;");

    renderer::ShaderPack pack;

    SUBCASE("an unhashed pack is never stale") { CHECK(pack.matches_sources(shaders)); }

    SUBCASE("a pack matches the sources it was recorded from") {
        pack.set_source_hash(*renderer::hash_shader_sources(shaders));
        CHECK(pack.matches_sources(shaders));
    }

    SUBCASE("a pack is stale once a source changes") {
        pack.set_source_hash(*renderer::hash_shader_sources(shaders));
        write_file(shaders / "a.slang", "float changed;");
        CHECK_FALSE(pack.matches_sources(shaders));
    }

    SUBCASE("a pack without sources on disk is trusted") {
        pack.set_source_hash(*renderer::hash_shader_sources(shaders));
        CHECK(pack.matches_sources(temp.path / "missing"));
    }
}

TEST_CASE("shader pack: the source hash survives save and load") {
    TempDirectory const temp;
    auto const file = temp.path / "shaders.lsp";

    renderer::ShaderPack pack;
    pack.set_source_hash(0x1234'5678'9abc'def0ULL);
    pack.add("key", renderer::CompiledShader{.stage = renderer::ShaderStage::compute,
                                              .entry_point = FlyString{"main_cs"},
                                              .spirv = {0x07230203U, 1U}});
    REQUIRE(pack.save(file).has_value());

    auto const loaded = renderer::ShaderPack::load(file);
    REQUIRE(loaded.has_value());
    CHECK(loaded->source_hash() == 0x1234'5678'9abc'def0ULL);
    CHECK(loaded->find("key").has_value());
}
