#include <doctest/doctest.h>

#include "assets/texture_pipeline.hxx"
#include "core/paths.hxx"

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

    auto make_temp_cache_dir(std::string_view label) -> std::filesystem::path {
        auto const dir = std::filesystem::temp_directory_path() /
                         std::format("texture_pipeline_test_{}_{}", label,
                                     std::chrono::steady_clock::now().time_since_epoch().count());

        std::filesystem::create_directories(dir);

        return dir;
    }

    auto verify_cache_file_round_trips(std::filesystem::path const &cache_dir, VkFormat expected_format) -> void {
        std::vector<std::filesystem::path> ktx2_files;

        for (auto const &entry: std::filesystem::recursive_directory_iterator{cache_dir}) {
            if (entry.path().extension() == ".ktx2") {
                ktx2_files.push_back(entry.path());
            }
        }

        REQUIRE(ktx2_files.size() == 1);

        ktxTexture2 *raw = nullptr;
        // Without the load-image-data bit libktx leaves the file's supercompression as it is on disk.
        auto const create_result = ktxTexture2_CreateFromNamedFile(ktx2_files.front().string().c_str(),
                                                                   KTX_TEXTURE_CREATE_NO_FLAGS, &raw);

        REQUIRE(create_result == KTX_SUCCESS);
        REQUIRE(raw != nullptr);

        CHECK(raw->vkFormat == static_cast<ktx_uint32_t>(expected_format));
        CHECK(raw->supercompressionScheme == KTX_SS_ZSTD);

        ktxTexture_Destroy(ktxTexture(raw));
    }

}

TEST_SUITE("unit") {
    TEST_CASE("load_compressed_texture: colour role transcodes to BC7 sRGB, mips, and caches") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_albedo.jpg";
        auto const cache_dir = make_temp_cache_dir("colour");

        auto first = load_compressed_texture(AssetPath::external(source).value(), TextureRole::colour, TextureCache{cache_dir});
        REQUIRE(first.has_value());

        CHECK(first->format == VK_FORMAT_BC7_SRGB_BLOCK);
        CHECK(first->width > 0);
        CHECK(first->height > 0);
        REQUIRE_FALSE(first->mips.empty());
        CHECK(first->mips.front().width == first->width);
        CHECK(first->mips.front().height == first->height);

        auto const expected_mip_count = static_cast<std::size_t>(std::bit_width(std::max(first->width, first->height)));
        CHECK(first->mips.size() == expected_mip_count);

        verify_cache_file_round_trips(cache_dir, VK_FORMAT_BC7_SRGB_BLOCK);

        auto second = load_compressed_texture(AssetPath::external(source).value(), TextureRole::colour, TextureCache{cache_dir});
        REQUIRE(second.has_value());

        CHECK(second->format == first->format);
        CHECK(second->mips.size() == first->mips.size());
        CHECK(second->data == first->data);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: reads a cache file written without supercompression") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_albedo.jpg";
        auto const cache_dir = make_temp_cache_dir("legacy");
        auto const path = AssetPath::external(source).value();

        auto first = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir});
        REQUIRE(first.has_value());

        // libktx inflates on load, so writing the texture back leaves the plain layout older caches have.
        for (auto const &entry: std::filesystem::recursive_directory_iterator{cache_dir}) {
            if (entry.path().extension() != ".ktx2") {
                continue;
            }

            ktxTexture2 *raw = nullptr;
            REQUIRE(ktxTexture2_CreateFromNamedFile(entry.path().string().c_str(),
                                                    KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &raw) == KTX_SUCCESS);
            REQUIRE(ktxTexture2_WriteToNamedFile(raw, entry.path().string().c_str()) == KTX_SUCCESS);
            CHECK(raw->supercompressionScheme == KTX_SS_NONE);
            ktxTexture_Destroy(ktxTexture(raw));
        }

        auto second = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir});
        REQUIRE(second.has_value());
        CHECK(second->data == first->data);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: a maximum extent drops the top mips, on a miss and on a hit") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_albedo.jpg";
        auto const cache_dir = make_temp_cache_dir("max_extent");
        auto const path = AssetPath::external(source).value();

        auto full = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir});
        REQUIRE(full.has_value());
        REQUIRE(std::max(full->width, full->height) > 128);

        set_texture_max_extent(128);

        // The cache already holds the full chain, so this is a hit that reads from a lower level.
        auto hit = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir});

        // A different cache directory forces a fresh encode, which is capped as it is returned.
        auto const miss_dir = make_temp_cache_dir("max_extent_miss");
        auto miss = load_compressed_texture(path, TextureRole::colour, TextureCache{miss_dir});

        set_texture_max_extent(0);

        for (auto const *capped: {&hit, &miss}) {
            REQUIRE(capped->has_value());
            CHECK(std::max((*capped)->width, (*capped)->height) <= 128);
            CHECK_FALSE(validate_compressed_texture(**capped).has_value());

            auto const skipped = full->mips.size() - (*capped)->mips.size();
            auto const &kept = full->mips[skipped];
            CHECK((*capped)->mips.front().width == kept.width);
            auto const &first = (*capped)->mips.front();
            CHECK(std::equal((*capped)->data.begin() + first.byte_offset,
                             (*capped)->data.begin() + first.byte_offset + first.byte_length,
                             full->data.begin() + kept.byte_offset));
        }

        std::filesystem::remove_all(cache_dir);
        std::filesystem::remove_all(miss_dir);
    }

    TEST_CASE("load_compressed_texture: cache hit publishes a preview that matches the full mip tail") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_albedo.jpg";
        auto const cache_dir = make_temp_cache_dir("preview");

        auto const path = AssetPath::external(source).value();
        auto const miss_preview = std::make_shared<TexturePreviewSlot>();
        auto full = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir}, nullptr, miss_preview);
        REQUIRE(full.has_value());
        CHECK_FALSE(miss_preview->take().has_value());

        auto const hit_preview = std::make_shared<TexturePreviewSlot>();
        auto cached = load_compressed_texture(path, TextureRole::colour, TextureCache{cache_dir}, nullptr, hit_preview);
        REQUIRE(cached.has_value());
        CHECK(cached->data == full->data);

        auto preview = hit_preview->take();

        if (std::max(full->width, full->height) > 256) {
            REQUIRE(preview.has_value());
            CHECK(std::max(preview->width, preview->height) <= 256);
            CHECK_FALSE(validate_compressed_texture(*preview).has_value());

            auto const skipped = full->mips.size() - preview->mips.size();
            auto const &full_level = full->mips[skipped];
            auto const &preview_level = preview->mips.front();

            CHECK(preview_level.width == full_level.width);
            CHECK(std::equal(preview->data.begin() + preview_level.byte_offset,
                             preview->data.begin() + preview_level.byte_offset + preview_level.byte_length,
                             full->data.begin() + full_level.byte_offset));
        } else {
            CHECK_FALSE(preview.has_value());
        }

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: normal_map role transcodes to BC5") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_normal.jpg";
        auto const cache_dir = make_temp_cache_dir("normal");

        auto result = load_compressed_texture(AssetPath::external(source).value(), TextureRole::normal_map, TextureCache{cache_dir});
        REQUIRE(result.has_value());

        CHECK(result->format == VK_FORMAT_BC5_UNORM_BLOCK);
        CHECK(result->width > 0);
        CHECK(result->height > 0);

        verify_cache_file_round_trips(cache_dir, VK_FORMAT_BC5_UNORM_BLOCK);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: generic role transcodes to BC7 UNORM") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/textures/terrain/terrain_normal.jpg";
        auto const cache_dir = make_temp_cache_dir("generic");

        auto result = load_compressed_texture(AssetPath::external(source).value(), TextureRole::generic, TextureCache{cache_dir});
        REQUIRE(result.has_value());

        CHECK(result->format == VK_FORMAT_BC7_UNORM_BLOCK);

        verify_cache_file_round_trips(cache_dir, VK_FORMAT_BC7_UNORM_BLOCK);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("load_compressed_texture: missing source file fails cleanly") {
        auto const cache_dir = make_temp_cache_dir("missing");

        auto result = load_compressed_texture(data_path("assets/textures/does_not_exist.png"), TextureRole::colour, TextureCache{cache_dir});

        CHECK_FALSE(result.has_value());
        CHECK(result.error().type == TexturePipelineErrorType::source_not_found);

        std::filesystem::remove_all(cache_dir);
    }
}
