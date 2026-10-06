#include <doctest/doctest.h>

#include <cstring>
#include <limits>
#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

#include <filesystem>

#include "gpu/sampler_storage.hxx"
#include "core/paths.hxx"
#include "serialisation/asset_id.hxx"
#include "serialisation/asset_pack.hxx"
#include "serialisation/byte_stream.hxx"
#include "serialisation/cooked_environment.hxx"
#include "serialisation/lbf_container.hxx"
#include "serialisation/scene_codec.hxx"
#include "serialisation/scene_serialisation.hxx"

namespace {

    auto custom_environment() -> SceneEnvironment {
        SceneEnvironment environment;
        environment.source = EnvironmentSource::hdr_image;
        environment.hdr_source = "assets/environments/belfast_sunset_puresky_512.ktx2";
        environment.ambient_intensity = 0.3F;
        environment.rotation_degrees = 42.0F;
        environment.exposure_ev = -1.5F;
        environment.diffuse_intensity = 0.8F;
        environment.specular_intensity = 1.25F;
        environment.specular_occlusion = 0.5F;
        environment.sky_intensity = 2.0F;
        environment.draw_skybox = false;
        environment.fog_sky = true;
        environment.sun_drives_directional_light = false;
        environment.multi_scatter = false;
        environment.hdr_cube_size = 1024;

        environment.sun.azimuth_degrees = -120.0F;
        environment.sun.elevation_degrees = 12.5F;
        environment.sun.turbidity = 6.0F;
        environment.sun.ground_albedo = glm::vec3{0.1F, 0.2F, 0.3F};
        environment.sun.angular_radius_degrees = 0.5F;
        environment.sun.colour = glm::vec3{1.0F, 0.5F, 0.25F};
        environment.sun.intensity = 5.0F;
        environment.sun.derive_colour_from_sky = false;

        environment.fog.enabled = true;
        environment.fog.colour = glm::vec3{0.2F, 0.3F, 0.4F};
        environment.fog.extinction = 0.01F;
        environment.fog.inscattering = 1.5F;
        environment.fog.from_environment = true;

        return environment;
    }

    // `payload` without its section of `type`, as a build from before that section existed would have written it.
    auto without_section(std::span<std::byte const> payload, std::uint32_t type) -> std::vector<std::byte> {
        std::vector<std::byte> result;

        std::size_t cursor = 0;

        while (cursor + 16 <= payload.size()) {
            std::uint32_t section_type = 0;
            std::uint64_t size = 0;

            std::memcpy(&section_type, payload.data() + cursor, sizeof(section_type));
            std::memcpy(&size, payload.data() + cursor + 8, sizeof(size));

            auto const total = 16 + static_cast<std::size_t>(size);

            if (section_type != type) {
                result.insert(result.end(), payload.begin() + static_cast<std::ptrdiff_t>(cursor),
                              payload.begin() + static_cast<std::ptrdiff_t>(cursor + total));
            }

            cursor += total;
        }

        return result;
    }

} // namespace

TEST_CASE("Environment section round-trips every field") {
    SceneDescription scene;
    scene.environment = custom_environment();
    scene.environment_id = asset_id_from_key(environment_asset_key(AssetPath::from_serialised(scene.environment.hdr_source).value()));

    auto const payload = encode_scene(scene);
    auto const decoded = decode_scene(payload);

    REQUIRE(decoded.has_value());
    CHECK(decoded->environment == scene.environment);
    CHECK(decoded->environment_id == scene.environment_id);

    // Same input, same bytes: the dirty check compares encodings.
    CHECK(encode_scene(*decoded) == payload);
}

TEST_CASE("A scene without an environment section loads as flat ambient, as scenes always looked") {
    SceneDescription scene;
    scene.environment = custom_environment();

    auto const stripped = without_section(encode_scene(scene), scene_section::environment);

    SceneDecodeReport report;
    auto const decoded = decode_scene(stripped, scene_chunk_version, &report);

    REQUIRE(decoded.has_value());
    CHECK(report.skipped_sections == 0);
    CHECK(decoded->environment == SceneEnvironment{});
    CHECK(decoded->environment.source == EnvironmentSource::flat_ambient);
    CHECK(decoded->environment.ambient_intensity == doctest::Approx(0.15F));
}

TEST_CASE("A new scene starts with the procedural sky, a decoded one with flat ambient") {
    CHECK(new_scene_environment().source == EnvironmentSource::procedural_sky);
    CHECK(SceneEnvironment{}.source == EnvironmentSource::flat_ambient);
}

TEST_CASE("An environment from a newer section version is skipped, not misread") {
    SceneDescription scene;
    scene.environment = custom_environment();

    auto payload = without_section(encode_scene(scene), scene_section::environment);

    ByteWriter newer;
    newer.write(scene_section::environment);
    newer.write(static_cast<std::uint16_t>(environment_section_version + 1));
    newer.write(std::uint16_t{0});
    newer.write(std::uint64_t{4});
    newer.write(std::uint32_t{0xDEADBEEF});
    payload.insert(payload.end(), newer.bytes().begin(), newer.bytes().end());

    SceneDecodeReport report;
    auto const decoded = decode_scene(payload, scene_chunk_version, &report);

    REQUIRE(decoded.has_value());
    CHECK(report.skipped_sections == 1);
    CHECK(decoded->environment == SceneEnvironment{});
}

TEST_CASE("Environment values that would poison the renderer are refused") {
    constexpr auto nan = std::numeric_limits<float>::quiet_NaN();

    auto const rejected = [](auto &&spoil) {
        SceneDescription scene;
        scene.environment = custom_environment();
        spoil(scene.environment);

        CHECK_FALSE(validate_scene(scene).has_value());
        // Through the codec too, which is the path a file takes.
        CHECK_FALSE(decode_scene(encode_scene(scene)).has_value());
    };

    {
        SceneDescription valid;
        valid.environment = custom_environment();
        CHECK(validate_scene(valid).has_value());
    }

    rejected([&](SceneEnvironment &environment) { environment.exposure_ev = nan; });
    rejected([](SceneEnvironment &environment) { environment.exposure_ev = 40.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.turbidity = 1.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.turbidity = 11.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.ground_albedo.x = 2.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.elevation_degrees = 120.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.intensity = -1.0F; });
    rejected([](SceneEnvironment &environment) { environment.sun.angular_radius_degrees = 0.0F; });
    rejected([](SceneEnvironment &environment) { environment.fog.extinction = -0.1F; });
    rejected([](SceneEnvironment &environment) { environment.diffuse_intensity = -1.0F; });
    rejected([](SceneEnvironment &environment) { environment.hdr_cube_size = 300; });
    rejected([](SceneEnvironment &environment) { environment.hdr_source.clear(); });
}

TEST_CASE("The scene fingerprint sees environment edits") {
    SceneDescription scene;
    scene.environment = custom_environment();

    auto const base = scene_fingerprint(scene);

    auto edited = scene;
    edited.environment.exposure_ev += 0.5F;
    CHECK(scene_fingerprint(edited) != base);

    auto moved_sun = scene;
    moved_sun.environment.sun.azimuth_degrees += 5.0F;
    CHECK(scene_fingerprint(moved_sun) != base);

    auto other_fog = scene;
    other_fog.environment.fog.enabled = false;
    CHECK(scene_fingerprint(other_fog) != base);

    CHECK(scene_fingerprint(scene) == base);
}

TEST_CASE("Cooked environments round-trip and refuse hostile payloads") {
    HdrImage equirect;
    equirect.width = 8;
    equirect.height = 4;
    equirect.layers = 1;
    equirect.pixels.resize(8U * 4U * 4U);

    for (std::size_t index = 0; index < equirect.pixels.size(); ++index) {
        equirect.pixels[index] = static_cast<std::uint16_t>((index * 2654435761U) >> 7U);
    }

    auto const payload = encode_cooked_environment(equirect);
    auto const decoded = decode_cooked_environment(payload);

    REQUIRE(decoded.has_value());
    CHECK(decoded->width == 8);
    CHECK(decoded->height == 4);
    CHECK(decoded->layers == 1);
    CHECK(decoded->pixels == equirect.pixels);

    HdrImage cube;
    cube.width = 4;
    cube.height = 4;
    cube.layers = 6;
    cube.pixels.assign(4U * 4U * 6U * 4U, 0x3C00);

    auto const cube_decoded = decode_cooked_environment(encode_cooked_environment(cube));
    REQUIRE(cube_decoded.has_value());
    CHECK(cube_decoded->layers == 6);
    CHECK(cube_decoded->pixels == cube.pixels);

    // Truncated and padded payloads.
    auto truncated = payload;
    truncated.pop_back();
    CHECK_FALSE(decode_cooked_environment(truncated).has_value());

    auto padded = payload;
    padded.push_back(std::byte{0});
    CHECK_FALSE(decode_cooked_environment(padded).has_value());

    CHECK_FALSE(decode_cooked_environment({}).has_value());
    CHECK_FALSE(decode_cooked_environment(payload, cooked_environment_version + 1).has_value());

    auto const with_header = [&](std::uint32_t width, std::uint32_t height, std::uint32_t format, std::uint32_t layers) {
        ByteWriter writer;
        writer.write(width);
        writer.write(height);
        writer.write(format);
        writer.write(layers);
        return writer.take();
    };

    // Sizes that a payload this small could never hold: refused before anything is allocated.
    CHECK_FALSE(decode_cooked_environment(with_header(16384, 8192, cooked_environment_format, 1)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(8, 4, 37, 1)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(8, 8, cooked_environment_format, 1)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(32768, 16384, cooked_environment_format, 1)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(3, 3, cooked_environment_format, 6)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(4096, 4096, cooked_environment_format, 6)).has_value());
    CHECK_FALSE(decode_cooked_environment(with_header(4, 4, cooked_environment_format, 3)).has_value());
}

TEST_CASE("cook_assets cooks an environment into a pack that loads back, and copies it on the next save") {
    auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/environments/belfast_sunset_puresky_512.ktx2";

    SamplerStorage sampler_storage;
    LbfWriter writer{LbfFileKind::asset_pack};

    auto const report = cook_assets(AssetCookRequest{.environments = {AssetPath::external(source).value()}}, sampler_storage, writer);

    CHECK(report.failures.empty());
    CHECK(report.environments_cooked == 1);

    auto reader = LbfReader::from_memory(*writer.finish());
    REQUIRE(reader.has_value());

    auto const pack = AssetPack::from_reader(std::move(*reader));
    auto const id = asset_id_from_key(environment_asset_key(AssetPath::external(source).value()));

    REQUIRE(pack->has_environment(id));

    auto const loaded = pack->load_environment(id);
    auto const reference = load_hdr_image(source.string());

    REQUIRE(loaded.has_value());
    REQUIRE(reference.has_value());
    CHECK(loaded->layers == 6);
    CHECK(loaded->width == 512);
    CHECK(loaded->pixels == reference->pixels);

    LbfWriter second{LbfFileKind::asset_pack};
    auto const copied =
            cook_assets(AssetCookRequest{.environments = {AssetPath::external(source).value()}}, sampler_storage, second, AssetCookOptions{.source_packs = {pack}});

    CHECK(copied.environments_copied == 1);
    CHECK(copied.environments_cooked == 0);

    // A missing file is a reported failure, not a crash, and the scene keeps its source path.
    LbfWriter missing{LbfFileKind::asset_pack};
    auto const failed = cook_assets(AssetCookRequest{.environments = {data_path("no/such/environment.hdr")}}, sampler_storage, missing);
    CHECK(failed.failures.size() == 1);
}
