#include <doctest/doctest.h>

#include <unistd.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>

#include "assets/shader_bake.hxx"
#include "assets/shader_pack.hxx"
#include "core/paths.hxx"

namespace {
    auto write_file(std::filesystem::path const &path, std::string const &contents) -> void {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream{path, std::ios::binary} << contents;
    }

    struct TempDataRoot {
        std::filesystem::path path = std::filesystem::temp_directory_path() / std::format("lathe_bake_test_{}", getpid());

        TempDataRoot() {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }

        ~TempDataRoot() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };

    constexpr auto compute_shader = R"(
[shader("compute")]
[numthreads(1, 1, 1)]
void main_cs(uint3 id : SV_DispatchThreadID) {}
)";

    constexpr auto graphics_shader = R"(
struct VsOut { float4 position : SV_Position; };

[shader("vertex")]
VsOut main_vs(uint vertex : SV_VertexID) {
    VsOut output;
#ifdef TINT
    output.position = float4(1, 0, 0, 1);
#else
    output.position = float4(0, 0, 0, 1);
#endif
    return output;
}

[shader("fragment")]
float4 main_fs(VsOut input) : SV_Target { return float4(1, 1, 1, 1); }
)";
}

TEST_CASE("shader variants: parsing") {
    auto const variants = renderer::parse_shader_variants(
            "# comment\n"
            "\n"
            "forward.slang main_fs OUTLINE_MASK=1  # trailing comment\n"
            "sub/other.slang main_cs FAST QUALITY=high\n");

    REQUIRE(variants.has_value());
    REQUIRE(variants->size() == 2);

    CHECK((*variants)[0].file == "forward.slang");
    CHECK((*variants)[0].entry == "main_fs");
    REQUIRE((*variants)[0].defines.size() == 1);
    CHECK((*variants)[0].defines[0].name == "OUTLINE_MASK");
    CHECK((*variants)[0].defines[0].value == "1");

    REQUIRE((*variants)[1].defines.size() == 2);
    CHECK((*variants)[1].defines[0].name == "FAST");
    CHECK((*variants)[1].defines[0].value == "1");
    CHECK((*variants)[1].defines[1].value == "high");
}

TEST_CASE("shader variants: a line without an entry point is an error") {
    CHECK_FALSE(renderer::parse_shader_variants("only_a_file.slang\n").has_value());
}

TEST_CASE("shader bake: compiles every entry point of every shader, plus variants") {
    auto compiler = renderer::SlangCompiler::create();

    if (!compiler) {
        MESSAGE("Slang is not available next to the test executable; skipping");
        return;
    }

    TempDataRoot const root;
    write_file(root.path / "assets" / "shaders" / "compute.slang", compute_shader);
    // A game's own shader, in a subdirectory: found without being registered anywhere.
    write_file(root.path / "assets" / "shaders" / "mygame" / "graphics.slang", graphics_shader);
    write_file(root.path / "assets" / "shaders" / "variants.txt", "mygame/graphics.slang main_vs TINT=1\n");

    Paths::set_current(Paths::resolve({.data_dir = root.path}));

    auto const output = root.path / "shaders.lsp";
    auto const result = renderer::bake_shaders(*compiler, output);

    REQUIRE(result.has_value());
    CHECK(result->entry_points == 3);
    CHECK(result->variants == 1);

    auto const pack = renderer::ShaderPack::load(output);
    REQUIRE(pack.has_value());
    CHECK(pack->size() == 4);
    CHECK(pack->matches_sources(root.path / "assets" / "shaders"));

    auto const key = [](std::string const &file, char const *entry, renderer::ShaderStage stage,
                        std::vector<renderer::ShaderDefine> defines = {}) {
        return renderer::shader_request_key(renderer::ShaderCompileRequest{
                .source_path = *Paths::current().data("assets/shaders/" + file),
                .entry_point = FlyString{entry},
                .stage = stage,
                .defines = std::move(defines),
        });
    };

    // The keys must match the requests the renderer makes at runtime (default flags, same logical path).
    CHECK(pack->find(key("compute.slang", "main_cs", renderer::ShaderStage::compute)).has_value());
    CHECK(pack->find(key("mygame/graphics.slang", "main_vs", renderer::ShaderStage::vertex)).has_value());
    CHECK(pack->find(key("mygame/graphics.slang", "main_fs", renderer::ShaderStage::fragment)).has_value());
    CHECK(pack->find(key("mygame/graphics.slang", "main_vs", renderer::ShaderStage::vertex,
                         {renderer::ShaderDefine{.name = "TINT", .value = "1"}}))
                  .has_value());
}

TEST_CASE("shader bake: a variant of an unknown entry point fails the bake") {
    auto compiler = renderer::SlangCompiler::create();

    if (!compiler) {
        MESSAGE("Slang is not available next to the test executable; skipping");
        return;
    }

    TempDataRoot const root;
    write_file(root.path / "assets" / "shaders" / "compute.slang", compute_shader);
    write_file(root.path / "assets" / "shaders" / "variants.txt", "compute.slang missing_entry X=1\n");

    Paths::set_current(Paths::resolve({.data_dir = root.path}));

    auto const output = root.path / "shaders.lsp";
    auto const result = renderer::bake_shaders(*compiler, output);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().find("missing_entry") != std::string::npos);
    CHECK_FALSE(std::filesystem::exists(output));
}
