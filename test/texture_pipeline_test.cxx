#include <doctest/doctest.h>

#include "texture_pipeline.hxx"

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <format>
#include <string_view>

#include <ktx.h>

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {

    // A fresh directory per test case avoids cross-test interference and
    // stale results from a previous run.
    auto make_temp_cache_dir(std::string_view label) -> std::filesystem::path {
        auto const dir = std::filesystem::temp_directory_path() /
                         std::format("texture_pipeline_test_{}_{}", label,
                                     std::chrono::steady_clock::now().time_since_epoch().count());

        std::filesystem::create_directories(dir);

        return dir;
    }

    // Independently verifies the on-disk .ktx2 cache file this pipeline run
    // produced is itself a well-formed, loadable, Basis-supercompressed KTX2
    // container -- i.e. the atomic temp-file-then-rename write path actually
    // produced something libktx can read back and transcode, not just bytes
    // this process happens to interpret correctly from memory.
    auto verify_cache_file_round_trips(std::filesystem::path const &cache_dir, ktx_transcode_fmt_e target) -> void {
        std::vector<std::filesystem::path> ktx2_files;

        for (auto const &entry: std::filesystem::directory_iterator{cache_dir}) {
            if (entry.path().extension() == ".ktx2") {
                ktx2_files.push_back(entry.path());
            }
        }

        REQUIRE(ktx2_files.size() == 1);

        ktxTexture2 *raw = nullptr;
        auto const create_result = ktxTexture2_CreateFromNamedFile(ktx2_files.front().string().c_str(),
                                                                    KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw);

        REQUIRE(create_result == KTX_SUCCESS);
        REQUIRE(raw != nullptr);

        auto const transcode_result = ktxTexture2_TranscodeBasis(raw, target, 0);
        CHECK(transcode_result == KTX_SUCCESS);

        ktxTexture_Destroy(ktxTexture(raw));
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("load_compressed_texture: colour role transcodes to BC7 sRGB, mips, and caches") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/dirt/dirt_diff_1k.jpg";
        auto const cache_dir = make_temp_cache_dir("colour");

        auto first = load_compressed_texture(source, TextureRole::colour, cache_dir);
        REQUIRE(first.has_value());

        CHECK(first->format == VK_FORMAT_BC7_SRGB_BLOCK);
        CHECK(first->width > 0);
        CHECK(first->height > 0);
        REQUIRE_FALSE(first->mips.empty());
        CHECK(first->mips.front().width == first->width);
        CHECK(first->mips.front().height == first->height);

        auto const expected_mip_count =
                static_cast<std::size_t>(std::bit_width(std::max(first->width, first->height)));
        CHECK(first->mips.size() == expected_mip_count);

        verify_cache_file_round_trips(cache_dir, KTX_TTF_BC7_RGBA);

        // Second call should be a cache hit: identical transcoded output.
        auto second = load_compressed_texture(source, TextureRole::colour, cache_dir);
        REQUIRE(second.has_value());

        CHECK(second->format == first->format);
        CHECK(second->mips.size() == first->mips.size());
        CHECK(second->data == first->data);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: normal_map role transcodes to BC5") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/dirt/dirt_nor_gl_1k_zip.exr";
        auto const cache_dir = make_temp_cache_dir("normal");

        auto result = load_compressed_texture(source, TextureRole::normal_map, cache_dir);
        REQUIRE(result.has_value());

        CHECK(result->format == VK_FORMAT_BC5_UNORM_BLOCK);
        CHECK(result->width > 0);
        CHECK(result->height > 0);

        verify_cache_file_round_trips(cache_dir, KTX_TTF_BC5_RG);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: generic role transcodes to BC7 UNORM") {
        // dirt_rough_1k.exr uses DWAA compression, which this project's
        // vendored tinyexr build does not decode (a pre-existing gap,
        // unrelated to this pipeline) -- use a different EXR asset that
        // decodes cleanly to exercise the generic-role encode path instead.
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/dirt/dirt_nor_gl_1k_zip.exr";
        auto const cache_dir = make_temp_cache_dir("generic");

        auto result = load_compressed_texture(source, TextureRole::generic, cache_dir);
        REQUIRE(result.has_value());

        CHECK(result->format == VK_FORMAT_BC7_UNORM_BLOCK);

        verify_cache_file_round_trips(cache_dir, KTX_TTF_BC7_RGBA);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: missing source file fails cleanly") {
        auto const cache_dir = make_temp_cache_dir("missing");

        auto result =
                load_compressed_texture("assets/textures/does_not_exist.png", TextureRole::colour, cache_dir);

        CHECK_FALSE(result.has_value());
        CHECK(result.error().type == TexturePipelineErrorType::source_not_found);

        std::filesystem::remove_all(cache_dir);
    }
}
