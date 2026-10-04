#include <doctest/doctest.h>

#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <numeric>

#include "assets/load_model.hxx"
#include "assets/meshlet.hxx"
#include "assets/primitive_meshes.hxx"
#include "gpu/sampler_storage.hxx"
#include "serialisation/asset_pack.hxx"
#include "serialisation/byte_stream.hxx"
#include "serialisation/checksum.hxx"
#include "serialisation/cooked_model.hxx"
#include "serialisation/cooked_texture.hxx"
#include "serialisation/lbf_container.hxx"
#include "serialisation/scene_codec.hxx"
#include "serialisation/scene_serialisation.hxx"

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {

    auto bytes_of(std::string_view text) -> std::vector<std::byte> {
        auto const view = std::as_bytes(std::span<char const>{text.data(), text.size()});
        return {view.begin(), view.end()};
    }

    // Highly compressible, so zstd kicks in.
    auto repetitive_payload(std::size_t size) -> std::vector<std::byte> {
        std::vector<std::byte> payload(size);

        for (std::size_t index = 0; index < size; ++index) {
            payload[index] = static_cast<std::byte>(index % 7);
        }

        return payload;
    }

    // A finalized single-primitive model, built the way load_model_cpu() would.
    auto finalized_cube() -> ModelCpuData {
        auto cpu_data = to_model_cpu_data(*make_sphere_mesh());

        for (auto &mesh: cpu_data.meshes) {
            for (auto &primitive: mesh.primitives) {
                primitive.reduced_indices = generate_mesh_lods(primitive.vertices, primitive.indices);
                prepare_primitive_gpu_data(primitive);
            }
        }

        return cpu_data;
    }

    // The payload of one SCEN section, and the version it was written with.
    struct EncodedSection {
        std::uint16_t version = 0;
        std::span<std::byte const> payload;
    };

    auto find_section(std::span<std::byte const> scene, std::uint32_t type) -> EncodedSection {
        ByteReader reader{scene};

        while (reader.remaining() > 0) {
            auto const section_type = reader.read<std::uint32_t>();
            auto const version = reader.read<std::uint16_t>();
            static_cast<void>(reader.read<std::uint16_t>());
            auto const size = reader.read<std::uint64_t>();
            auto const offset = scene.size() - reader.remaining();

            if (reader.failed() || size > reader.remaining()) {
                break;
            }
            if (section_type == type) {
                return EncodedSection{.version = version, .payload = scene.subspan(offset, size)};
            }

            std::vector<std::byte> skipped(size);
            reader.read_span(std::span<std::byte>{skipped});
        }

        return {};
    }

    // A grass-like field: yaw-only rotation, uniform scale, translation on a jittered grid.
    auto grass_like_transforms(std::size_t count) -> std::vector<glm::mat4> {
        std::vector<glm::mat4> transforms;
        transforms.reserve(count);

        for (std::size_t index = 0; index < count; ++index) {
            auto const yaw = static_cast<float>(index) * 0.731F;
            auto const scale = 0.6F + (0.01F * static_cast<float>(index % 40));
            auto const column = index % 25;
            auto const row = index / 25;
            auto transform = glm::translate(glm::mat4{1.0F}, glm::vec3{static_cast<float>(column) * 0.5F, 0.1F,
                                                                         static_cast<float>(row) * 0.5F});
            transform = glm::rotate(transform, yaw, glm::vec3{0.0F, 1.0F, 0.0F});
            transforms.push_back(glm::scale(transform, glm::vec3{scale}));
        }

        return transforms;
    }

    auto check_matrices_close(glm::mat4 const &actual, glm::mat4 const &expected) -> void {
        for (glm::length_t column = 0; column < 4; ++column) {
            for (glm::length_t row = 0; row < 4; ++row) {
                CHECK(actual[column][row] == doctest::Approx(expected[column][row]).epsilon(1e-5));
            }
        }
    }

    // Rewrites the first table-of-contents entry and re-seals the table's checksum, so the file is internally
    // consistent and only the entry's own values are hostile.
    template<typename Patch>
    auto with_patched_entry(std::vector<std::byte> file, Patch &&patch) -> std::vector<std::byte> {
        LbfFileHeader header{};
        std::memcpy(&header, file.data(), sizeof(header));

        LbfChunkEntry entry{};
        std::memcpy(&entry, file.data() + header.toc_offset, sizeof(entry));
        patch(entry);
        std::memcpy(file.data() + header.toc_offset, &entry, sizeof(entry));

        header.toc_checksum = xxh64(std::span<std::byte const>{file}.subspan(header.toc_offset, header.toc_size));
        std::memcpy(file.data(), &header, sizeof(header));
        return file;
    }

    auto single_chunk_file(bool compress) -> std::vector<std::byte> {
        LbfWriter writer{LbfFileKind::asset_pack};
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::model,
                                       .id = 1,
                                       .version = 1,
                                       .payload = repetitive_payload(4096),
                                       .compress = compress});
        return *writer.finish(LbfWriteOptions{.parallel = false});
    }

    auto valid_texture() -> CompressedTexture {
        return CompressedTexture{
                .format = VK_FORMAT_BC7_SRGB_BLOCK,
                .width = 8,
                .height = 4,
                .mips = {{.width = 8, .height = 4, .byte_offset = 0, .byte_length = 32},
                         {.width = 4, .height = 2, .byte_offset = 32, .byte_length = 16}},
                .data = repetitive_payload(48),
                .debug_name = FlyString{"valid"},
        };
    }

    // A scene with one entity carrying one of every component, which each test then spoils in one way.
    auto populated_scene() -> SceneDescription {
        SceneDescription scene;
        scene.models.push_back(SceneAssetRef{.source = "engine://cube"});
        scene.materials.emplace_back();
        scene.entities.push_back(SceneEntity{.name = "e", .transform = Components::Transform{}});
        scene.model_components.push_back(SceneModelComponent{.entity = 0, .model = 0});
        scene.point_lights.push_back(ScenePointLightComponent{.entity = 0});
        scene.spot_lights.push_back(SceneSpotLightComponent{.entity = 0});
        scene.rigid_bodies.push_back(SceneRigidBodyComponent{.entity = 0});
        scene.lifetimes.push_back(SceneLifetimeComponent{.entity = 0, .remaining_seconds = 1.0F});
        return scene;
    }

} // namespace

TEST_SUITE("unit") {
    TEST_CASE("xxh64 matches the reference implementation") {
        CHECK(xxh64(std::string_view{}) == 0xEF46DB3751D8E999ULL);
        CHECK(xxh64(std::string_view{"abc"}) == 0x44BC2CF5AD770999ULL);

        std::vector<std::byte> sequence(100);
        for (std::size_t index = 0; index < sequence.size(); ++index) {
            sequence[index] = static_cast<std::byte>(index);
        }

        CHECK(xxh64(sequence) == 0x6AC1E58032166597ULL);
        CHECK(xxh64(sequence, 7) == 0x80653E7E9B887CDDULL);
    }

    TEST_CASE("ByteReader latches failure on overrun and on oversized counts") {
        ByteWriter writer;
        writer.write(std::uint32_t{42});
        writer.write_string("hello");
        writer.write(std::uint32_t{1'000'000}); // a bogus array count

        auto const bytes = writer.take();
        ByteReader reader{bytes};

        CHECK(reader.read<std::uint32_t>() == 42);
        CHECK(reader.read_string() == "hello");

        std::vector<std::uint64_t> values;
        CHECK_FALSE(reader.read_array(values));
        CHECK(reader.failed());
        CHECK(values.empty());
        CHECK(reader.read<std::uint32_t>() == 0);
    }

    TEST_CASE("LBF container: chunks round-trip, compressed and raw") {
        LbfWriter writer{LbfFileKind::asset_pack};

        writer.add_chunk(
                LbfChunkInput{.type = lbf_chunk::texture, .id = 7, .version = 3, .payload = repetitive_payload(4096)});
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::model, .id = 2, .payload = bytes_of("tiny")});
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::model, .id = 1, .payload = {}});

        auto file = writer.finish(LbfWriteOptions{.parallel = false});
        REQUIRE(file.has_value());

        auto reader = LbfReader::from_memory(std::move(*file));
        REQUIRE(reader.has_value());

        CHECK(reader->header().kind == LbfFileKind::asset_pack);
        CHECK(reader->chunks().size() == 3);

        auto const *texture = reader->find(lbf_chunk::texture, 7);
        REQUIRE(texture != nullptr);
        CHECK(texture->compression == LbfCompression::zstd);
        CHECK(texture->version == 3);
        CHECK(texture->offset % lbf_payload_alignment == 0);
        CHECK(texture->stored_size < texture->raw_size);

        auto payload = reader->read_chunk(*texture);
        REQUIRE(payload.has_value());
        CHECK(*payload == repetitive_payload(4096));

        // Too small to be worth compressing.
        auto const *model = reader->find(lbf_chunk::model, 2);
        REQUIRE(model != nullptr);
        CHECK(model->compression == LbfCompression::none);
        CHECK(*reader->read_chunk(*model) == bytes_of("tiny"));

        auto const *empty = reader->find(lbf_chunk::model, 1);
        REQUIRE(empty != nullptr);
        CHECK(reader->read_chunk(*empty)->empty());

        CHECK(reader->find(lbf_chunk::model, 3) == nullptr);
    }

    TEST_CASE("LBF container: corruption is detected") {
        LbfWriter writer{LbfFileKind::scene};
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::scene, .payload = repetitive_payload(1024)});

        auto file = writer.finish(LbfWriteOptions{.parallel = false});
        REQUIRE(file.has_value());

        SUBCASE("flipped payload byte fails the chunk checksum") {
            auto corrupt = *file;
            corrupt[lbf_payload_alignment + 2] ^= std::byte{0xFF};

            auto reader = LbfReader::from_memory(std::move(corrupt));
            REQUIRE(reader.has_value());

            auto payload = reader->read_chunk(*reader->find(lbf_chunk::scene));
            REQUIRE_FALSE(payload.has_value());
            CHECK(payload.error().type == LbfErrorType::checksum_mismatch);
        }

        SUBCASE("truncation is rejected up front") {
            auto truncated = *file;
            truncated.resize(truncated.size() - 1);

            CHECK_FALSE(LbfReader::from_memory(std::move(truncated)).has_value());
        }

        SUBCASE("bad magic") {
            auto corrupt = *file;
            corrupt[0] = std::byte{'X'};

            auto reader = LbfReader::from_memory(std::move(corrupt));
            REQUIRE_FALSE(reader.has_value());
            CHECK(reader.error().type == LbfErrorType::not_an_lbf_file);
        }

        SUBCASE("a newer major version is refused") {
            auto newer = *file;
            LbfFileHeader header{};
            std::memcpy(&header, newer.data(), sizeof(header));
            header.version_major = lbf_version_major + 1;
            std::memcpy(newer.data(), &header, sizeof(header));

            auto reader = LbfReader::from_memory(std::move(newer));
            REQUIRE_FALSE(reader.has_value());
            CHECK(reader.error().type == LbfErrorType::unsupported_version);
        }
    }

    TEST_CASE("LBF container: stored chunks copy verbatim between files") {
        LbfWriter first{LbfFileKind::asset_pack};
        first.add_chunk(
                LbfChunkInput{.type = lbf_chunk::texture, .id = 9, .version = 1, .payload = repetitive_payload(2048)});

        auto source = LbfReader::from_memory(*first.finish(LbfWriteOptions{.parallel = false}));
        REQUIRE(source.has_value());

        auto const &entry = *source->find(lbf_chunk::texture, 9);

        LbfWriter second{LbfFileKind::scene};
        second.add_stored_chunk(entry, *source->read_stored_chunk(entry));

        auto copy = LbfReader::from_memory(*second.finish(LbfWriteOptions{.parallel = false}));
        REQUIRE(copy.has_value());

        auto const *copied = copy->find(lbf_chunk::texture, 9);
        REQUIRE(copied != nullptr);
        CHECK(copied->checksum == entry.checksum);
        CHECK(*copy->read_chunk(*copied) == repetitive_payload(2048));
    }

    TEST_CASE("Asset ids are stable and distinguish texture roles") {
        auto const colour = asset_id_from_key(texture_asset_key("assets/textures/a.png", TextureRole::colour));
        auto const normal = asset_id_from_key(texture_asset_key("assets/textures/a.png", TextureRole::normal_map));

        CHECK(colour.valid());
        CHECK(colour != normal);
        CHECK(colour == asset_id_from_key(texture_asset_key("assets/textures/./a.png", TextureRole::colour)));
        CHECK(asset_id_from_key("model:x") == asset_id_from_key("model:x"));
    }

    TEST_CASE("Cooked texture round-trips") {
        CompressedTexture texture{
                .format = VK_FORMAT_BC7_SRGB_BLOCK,
                .width = 8,
                .height = 4,
                .mips = {{.width = 8, .height = 4, .byte_offset = 0, .byte_length = 32},
                         {.width = 4, .height = 2, .byte_offset = 32, .byte_length = 16}},
                .data = repetitive_payload(48),
                .debug_name = FlyString{"bricks"},
        };

        auto const payload = encode_cooked_texture(texture, TextureRole::colour);
        auto decoded = decode_cooked_texture(payload);
        REQUIRE(decoded.has_value());

        CHECK(decoded->role == TextureRole::colour);
        CHECK(decoded->texture.format == texture.format);
        CHECK(decoded->texture.width == 8);
        CHECK(decoded->texture.mips.size() == 2);
        CHECK(decoded->texture.mips[1].byte_offset == 32);
        CHECK(decoded->texture.data == texture.data);
        CHECK(decoded->texture.debug_name == "bricks");

        CHECK_FALSE(decode_cooked_texture(payload, cooked_texture_version + 1).has_value());
        CHECK_FALSE(decode_cooked_texture(std::span{payload}.first(payload.size() - 1)).has_value());
    }

    TEST_CASE("Cooked model round-trips geometry, LODs and meshlets losslessly") {
        auto cpu_data = finalized_cube();
        cpu_data.materials.push_back(ModelCpuMaterial{.base_colour_factor = glm::vec4{0.5F}, .base_colour_image = 0});
        cpu_data.image_sources.push_back(
                ModelCpuImageSource{.slot = ModelTextureSlot::base_colour, .debug_name = FlyString{"albedo"}});
        cpu_data.meshes[0].primitives[0].material_index = 0;
        cpu_data.lights.push_back(ModelCpuLight{.type = ModelLightType::spot, .intensity = 3.0F});

        std::array const images{CookedImageRef{.texture = AssetId{.value = 77},
                                               .slot = ModelTextureSlot::base_colour,
                                               .debug_name = FlyString{"albedo"}}};
        std::array const samplers{DefaultSampler::nearest_clamp};

        auto payload = encode_cooked_model(cpu_data, images, samplers);
        REQUIRE(payload.has_value());

        auto decoded = decode_cooked_model(*payload);
        REQUIRE(decoded.has_value());

        auto const &original = cpu_data.meshes[0].primitives[0];
        auto const &primitive = decoded->cpu_data.meshes[0].primitives[0];

        CHECK(primitive.vertices.empty());
        REQUIRE(primitive.compressed_vertices.size() == original.compressed_vertices.size());
        CHECK(std::memcmp(primitive.compressed_vertices.data(), original.compressed_vertices.data(),
                          original.compressed_vertices.size() * sizeof(CompressedModelVertex)) == 0);
        CHECK(primitive.indices.size() == original.indices.size());
        CHECK(primitive.material_index == std::optional<std::uint32_t>{0});
        REQUIRE(primitive.bounds.has_value());
        CHECK(primitive.bounds->first.x < 0.0F);
        CHECK(primitive.bounds->second.x > 0.0F);

        for (std::uint32_t level = 0; level < lod_count; ++level) {
            CHECK(primitive.meshlets[level].has_value() == original.meshlets[level].has_value());

            if (primitive.meshlets[level].has_value()) {
                CHECK(primitive.meshlets[level]->topology.data == original.meshlets[level]->topology.data);
                CHECK(primitive.meshlets[level]->meshlets.size() == original.meshlets[level]->meshlets.size());
                CHECK(primitive.meshlets[level]->topology.meshlets.size() == original.meshlets[level]->meshlets.size());
            }
        }

        CHECK(decoded->images.size() == 1);
        CHECK(decoded->images[0].texture.value == 77);
        CHECK(decoded->material_samplers[0] == DefaultSampler::nearest_clamp);
        CHECK(decoded->cpu_data.materials[0].base_colour_image == std::optional<std::size_t>{0});
        CHECK(decoded->cpu_data.lights[0].type == ModelLightType::spot);
        CHECK(decoded->cpu_data.nodes.size() == cpu_data.nodes.size());
        CHECK(decoded->cpu_data.bounds.has_value());

        CHECK_FALSE(decode_cooked_model(*payload, cooked_model_version + 1).has_value());
        CHECK_FALSE(decode_cooked_model(std::span{*payload}.first(payload->size() / 2)).has_value());
    }

    TEST_CASE("Cooked model refuses unfinalized primitives") {
        auto cpu_data = to_model_cpu_data(*make_sphere_mesh());
        cpu_data.meshes[0].primitives[0].compressed_vertices.clear();

        CHECK_FALSE(encode_cooked_model(cpu_data, {}, {}).has_value());
    }

    TEST_CASE("AssetPack hands out models whose textures load from the same pack") {
        auto cpu_data = finalized_cube();
        cpu_data.materials.push_back(ModelCpuMaterial{.base_colour_image = 0});
        cpu_data.image_sources.push_back(ModelCpuImageSource{.slot = ModelTextureSlot::base_colour});

        AssetId const model_id{.value = 11};
        AssetId const texture_id{.value = 12};

        std::array const images{CookedImageRef{.texture = texture_id}};
        std::array const samplers{DefaultSampler::linear_clamp};

        CompressedTexture const texture{
                .format = VK_FORMAT_BC5_UNORM_BLOCK,
                .width = 4,
                .height = 4,
                .mips = {{.width = 4, .height = 4, .byte_offset = 0, .byte_length = 16}},
                .data = repetitive_payload(16),
        };

        LbfWriter writer{LbfFileKind::asset_pack};
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::model,
                                       .id = model_id.value,
                                       .version = cooked_model_version,
                                       .payload = *encode_cooked_model(cpu_data, images, samplers)});
        writer.add_chunk(LbfChunkInput{.type = lbf_chunk::texture,
                                       .id = texture_id.value,
                                       .version = cooked_texture_version,
                                       .payload = encode_cooked_texture(texture, TextureRole::normal_map)});

        auto reader = LbfReader::from_memory(*writer.finish(LbfWriteOptions{.parallel = false}));
        REQUIRE(reader.has_value());

        auto const pack = AssetPack::from_reader(std::move(*reader));
        SamplerStorage const sampler_storage;

        CHECK(pack->has_model(model_id));
        CHECK(pack->has_texture(texture_id));

        auto model = pack->load_model(model_id, sampler_storage);
        REQUIRE(model.has_value());
        CHECK(model->materials[0].sampler == sampler_storage.linear_clamp());
        REQUIRE(model->image_sources.size() == 1);
        REQUIRE(static_cast<bool>(model->image_sources[0].cooked));
        CHECK(model->image_sources[0].cache_key == AssetPack::texture_cache_key(texture_id));

        auto loaded = model->image_sources[0].cooked();
        REQUIRE(loaded.has_value());
        CHECK(loaded->format == VK_FORMAT_BC5_UNORM_BLOCK);
        CHECK(loaded->data == texture.data);

        CHECK_FALSE(pack->load_model(AssetId{.value = 99}, sampler_storage).has_value());
    }

    TEST_CASE("Scene codec round-trips every section") {
        SceneDescription scene;
        scene.physics_settings.gravity = glm::vec3{0.0F, -3.0F, 0.0F};
        scene.models.push_back(SceneAssetRef{.id = asset_id_from_key("model:a.gltf"), .source = "a.gltf"});
        scene.textures.push_back(
                SceneTextureRef{.id = AssetId{.value = 5}, .source = "t.png", .role = TextureRole::generic});

        SceneMaterial material{.name = "red", .base_colour_factor = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F}};
        material.textures[scene_material_texture::metallic_roughness] = 0;
        material.alpha_mode = AlphaMode::mask;
        material.alpha_to_coverage = true;
        material.double_sided = true;
        material.far_material = 1;
        material.far_material_lod = 3;
        scene.materials.push_back(material);
        scene.materials.push_back(SceneMaterial{.name = "red_far"});

        scene.entities.push_back(SceneEntity{
                .name = "root", .transform = Components::Transform{.position = glm::vec3{1.0F, 2.0F, 3.0F}}});
        scene.entities.push_back(SceneEntity{.name = "child", .parent = 0, .flags = SceneEntityFlags::generated_name});
        scene.entities.push_back(SceneEntity{.name = "player", .flags = SceneEntityFlags::player});

        scene.model_components.push_back(SceneModelComponent{.entity = 0, .model = 0});
        scene.material_overrides.push_back(SceneMaterialOverrideComponent{
                .entity = 0, .material = 0, .slots = {{.source_slot = 2, .material = 0}}});
        scene.instanced_models.push_back(SceneInstancedModelComponent{
                .entity = 1, .model = 0, .transforms = {glm::mat4{1.0F}, glm::mat4{2.0F}}});
        scene.point_lights.push_back(ScenePointLightComponent{.entity = 1, .light = {.intensity = 4.0F}});
        scene.spot_lights.push_back(SceneSpotLightComponent{.entity = 2, .light = {.outer_cone_degrees = 45.0F}});
        scene.rigid_bodies.push_back(SceneRigidBodyComponent{
                .entity = 2, .body = Components::RigidBody::from_submesh_boxes({{glm::vec3{0.0F}, glm::vec3{1.0F}}})});
        scene.scripts.push_back(SceneScriptComponent{.entity = 2, .script = "player_controller"});
        scene.lifetimes.push_back(SceneLifetimeComponent{.entity = 1, .remaining_seconds = 2.5F});

        auto const payload = encode_scene(scene);
        auto decoded = decode_scene(payload);
        REQUIRE(decoded.has_value());

        CHECK(decoded->physics_settings.gravity.y == doctest::Approx(-3.0F));
        CHECK(decoded->models[0].id == scene.models[0].id);
        CHECK(decoded->textures[0].role == TextureRole::generic);
        CHECK(decoded->materials[0].name == "red");
        CHECK(decoded->materials[0].alpha_mode == AlphaMode::mask);
        CHECK(decoded->materials[0].textures[scene_material_texture::metallic_roughness] == 0);
        CHECK(decoded->materials[0].textures[scene_material_texture::base_colour] == scene_no_index);
        CHECK(decoded->materials[0].alpha_to_coverage);
        CHECK(decoded->materials[0].double_sided);
        CHECK(decoded->materials[0].far_material == 1);
        CHECK(decoded->materials[0].far_material_lod == 3);
        CHECK(decoded->materials[1].far_material == scene_no_index);
        CHECK_FALSE(decoded->materials[1].double_sided);
        REQUIRE(decoded->entities.size() == 3);
        CHECK(decoded->entities[0].transform->position.z == doctest::Approx(3.0F));
        CHECK_FALSE(decoded->entities[1].transform.has_value());
        CHECK(decoded->entities[1].parent == 0);
        CHECK(has_flag(decoded->entities[2].flags, SceneEntityFlags::player));
        CHECK(decoded->material_overrides[0].slots[0].source_slot == 2);
        CHECK(decoded->instanced_models[0].transforms[1][0][0] == doctest::Approx(2.0F));
        CHECK(decoded->point_lights[0].light.intensity == doctest::Approx(4.0F));
        CHECK(decoded->spot_lights[0].light.outer_cone_degrees == doctest::Approx(45.0F));
        CHECK(decoded->rigid_bodies[0].body.shape == Components::BodyShape::compound);
        REQUIRE(decoded->rigid_bodies[0].body.compound_boxes != nullptr);
        CHECK(decoded->rigid_bodies[0].body.compound_boxes->size() == 1);
        CHECK(decoded->scripts[0].script == "player_controller");
        CHECK(decoded->lifetimes[0].remaining_seconds == doctest::Approx(2.5F));

        // Same input, same bytes: the dirty check compares encodings.
        CHECK(encode_scene(*decoded) == payload);
    }

    TEST_CASE("Instance transforms that decompose are stored as TRS columns") {
        SceneDescription scene;
        scene.models.push_back(SceneAssetRef{.id = asset_id_from_key("model:grass.gltf"), .source = "grass.gltf"});
        scene.entities.push_back(SceneEntity{.name = "grass"});
        scene.entities.push_back(SceneEntity{.name = "mirrored"});

        auto const transforms = grass_like_transforms(500);
        scene.instanced_models.push_back(
                SceneInstancedModelComponent{.entity = 0, .model = 0, .transforms = transforms});

        auto const mirrored = std::vector{glm::scale(glm::mat4{1.0F}, glm::vec3{-1.0F, 1.0F, 1.0F}),
                                          glm::scale(glm::rotate(glm::mat4{1.0F}, 0.4F, glm::vec3{1.0F, 0.0F, 0.0F}),
                                                     glm::vec3{2.0F, -3.0F, 0.5F})};
        scene.instanced_models.push_back(
                SceneInstancedModelComponent{.entity = 1, .model = 0, .transforms = mirrored});

        auto const payload = encode_scene(scene);

        auto const section = find_section(payload, scene_section::instanced_models);
        CHECK(section.version == instanced_models_section_version);
        // Component count, then per component: entity, model, material, count, encoding and 10 floats per instance.
        CHECK(section.payload.size() == 4 + (17 + (40 * transforms.size())) + (17 + (40 * mirrored.size())));

        auto const decoded = decode_scene(payload);
        REQUIRE(decoded.has_value());
        REQUIRE(decoded->instanced_models.size() == 2);
        REQUIRE(decoded->instanced_models[0].transforms.size() == transforms.size());
        REQUIRE(decoded->instanced_models[1].transforms.size() == mirrored.size());

        for (std::size_t index = 0; index < transforms.size(); ++index) {
            check_matrices_close(decoded->instanced_models[0].transforms[index], transforms[index]);
        }
        for (std::size_t index = 0; index < mirrored.size(); ++index) {
            check_matrices_close(decoded->instanced_models[1].transforms[index], mirrored[index]);
        }
    }

    TEST_CASE("A sheared instance keeps its component's transforms as exact matrices") {
        SceneDescription scene;
        scene.models.push_back(SceneAssetRef{.id = asset_id_from_key("model:a.gltf"), .source = "a.gltf"});
        scene.entities.push_back(SceneEntity{.name = "sheared"});

        auto transforms = grass_like_transforms(8);
        auto sheared = glm::mat4{1.0F};
        sheared[1][0] = 0.5F;
        transforms.push_back(sheared);
        scene.instanced_models.push_back(
                SceneInstancedModelComponent{.entity = 0, .model = 0, .transforms = transforms});

        auto const payload = encode_scene(scene);
        auto const section = find_section(payload, scene_section::instanced_models);
        CHECK(section.payload.size() == 4 + 17 + (64 * transforms.size()));

        auto const decoded = decode_scene(payload);
        REQUIRE(decoded.has_value());
        CHECK(decoded->instanced_models[0].transforms == transforms);
    }

    TEST_CASE("Version 1 instanced model sections still decode") {
        SceneDescription scene;
        scene.models.push_back(SceneAssetRef{.id = asset_id_from_key("model:a.gltf"), .source = "a.gltf"});
        scene.entities.push_back(SceneEntity{.name = "old"});

        auto payload = encode_scene(scene);

        // v1 layout: interleaved column-major matrices.
        auto const transforms = grass_like_transforms(3);
        ByteWriter section;
        section.write(std::uint32_t{1});
        section.write(std::uint32_t{0}); // entity
        section.write(std::uint32_t{0}); // model
        section.write(scene_no_index); // material
        section.write(static_cast<std::uint32_t>(transforms.size()));
        for (auto const &transform: transforms) {
            section.write_span(std::span<float const>{&transform[0][0], 16});
        }

        // A second instanced_models section replaces the empty current-version one when decoded.
        ByteWriter framed;
        framed.write(scene_section::instanced_models);
        framed.write(std::uint16_t{1});
        framed.write(std::uint16_t{0});
        framed.write(static_cast<std::uint64_t>(section.size()));
        framed.write_span(section.bytes());
        payload.insert(payload.end(), framed.bytes().begin(), framed.bytes().end());

        auto const decoded = decode_scene(payload);
        REQUIRE(decoded.has_value());
        REQUIRE(decoded->instanced_models.size() == 1);
        CHECK(decoded->instanced_models[0].transforms == transforms);
    }

    TEST_CASE("Version 1 material sections still decode, single-sided and without a far material") {
        SceneDescription scene;
        auto payload = encode_scene(scene);

        // v1 layout: everything up to and including the texture indices.
        ByteWriter section;
        section.write(std::uint32_t{1});
        section.write_string("old");
        for (auto const value: {1.0F, 1.0F, 1.0F, 1.0F, 0.0F, 0.0F, 0.0F}) {
            section.write(value); // base colour, emissive
        }
        for (auto const value: {1.0F, 0.0F, 0.5F, 1.0F, 1.0F, 0.5F, 0.3F}) {
            section.write(value); // emissive strength .. alpha cutoff, wind
        }
        section.write(std::uint32_t{shadow_cascade_count - 1});
        section.write(std::to_underlying(AlphaMode::mask));
        section.write(std::uint8_t{0}); // sampler
        section.write(std::uint8_t{0}); // debug meshlet colours
        for (auto index = 0; index < 5; ++index) {
            section.write(scene_no_index);
        }

        ByteWriter framed;
        framed.write(scene_section::materials);
        framed.write(std::uint16_t{1});
        framed.write(std::uint16_t{0});
        framed.write(static_cast<std::uint64_t>(section.size()));
        framed.write_span(section.bytes());
        payload.insert(payload.end(), framed.bytes().begin(), framed.bytes().end());

        auto const decoded = decode_scene(payload);
        REQUIRE(decoded.has_value());
        REQUIRE(decoded->materials.size() == 1);
        CHECK(decoded->materials[0].name == "old");
        CHECK(decoded->materials[0].alpha_mode == AlphaMode::mask);
        CHECK(decoded->materials[0].wind_strength == doctest::Approx(0.3F));
        CHECK_FALSE(decoded->materials[0].double_sided);
        CHECK_FALSE(decoded->materials[0].alpha_to_coverage);
        CHECK(decoded->materials[0].far_material == scene_no_index);
    }

    TEST_CASE("Scene codec skips sections from a newer engine and rejects bad references") {
        SceneDescription scene;
        scene.entities.push_back(SceneEntity{.name = "only"});

        auto payload = encode_scene(scene);

        // Append a section type this build has never heard of.
        ByteWriter extra;
        extra.write(std::uint32_t{999});
        extra.write(std::uint16_t{1});
        extra.write(std::uint16_t{0});
        extra.write(std::uint64_t{4});
        extra.write(std::uint32_t{0xDEADBEEF});
        payload.insert(payload.end(), extra.bytes().begin(), extra.bytes().end());

        SceneDecodeReport report;
        auto decoded = decode_scene(payload, scene_chunk_version, &report);
        REQUIRE(decoded.has_value());
        CHECK(report.skipped_sections == 1);
        CHECK(decoded->entities.size() == 1);

        CHECK_FALSE(decode_scene(payload, scene_chunk_version + 1).has_value());

        SceneDescription cyclic;
        cyclic.entities.push_back(SceneEntity{.name = "a", .parent = 1});
        cyclic.entities.push_back(SceneEntity{.name = "b", .parent = 0});
        CHECK_FALSE(validate_scene(cyclic).has_value());

        SceneDescription dangling;
        dangling.model_components.push_back(SceneModelComponent{.entity = 0, .model = 0});
        CHECK_FALSE(validate_scene(dangling).has_value());
    }

    TEST_CASE("Scene fingerprint ignores entity order but sees edits") {
        auto make_scene = [](bool reversed) {
            SceneDescription scene;
            scene.models.push_back(SceneAssetRef{.id = AssetId{.value = 1}, .source = "a.gltf"});
            scene.models.push_back(SceneAssetRef{.id = AssetId{.value = 2}, .source = "b.gltf"});

            std::vector<SceneEntity> entities{
                    SceneEntity{.name = "first", .transform = Components::Transform{.position = glm::vec3{1.0F}}},
                    SceneEntity{.name = "second", .transform = Components::Transform{.position = glm::vec3{2.0F}}},
            };

            std::uint32_t first = 0;
            std::uint32_t second = 1;

            if (reversed) {
                std::ranges::reverse(entities);
                std::ranges::reverse(scene.models);
                std::swap(first, second);
            }

            scene.entities = std::move(entities);
            // "first" always gets a.gltf, wherever both sit in their tables.
            scene.model_components.push_back(SceneModelComponent{.entity = first, .model = reversed ? 1U : 0U});
            scene.point_lights.push_back(ScenePointLightComponent{.entity = second});
            return scene;
        };

        auto const forward = make_scene(false);
        auto const backward = make_scene(true);

        CHECK(scene_fingerprint(forward) == scene_fingerprint(backward));

        auto edited = forward;
        edited.entities[0].transform->position.x += 0.01F;
        CHECK(scene_fingerprint(edited) != scene_fingerprint(forward));

        auto renamed = forward;
        renamed.entities[1].name = "other";
        CHECK(scene_fingerprint(renamed) != scene_fingerprint(forward));
    }

    TEST_CASE("cook_assets cooks a real glTF into a pack that loads back") {
        auto const source = std::filesystem::path{TEST_ASSETS_DIR} / "assets/models/test_cube.glb";
        auto const cache_dir = std::filesystem::temp_directory_path() / "lathe_lbf_test_cache";
        std::filesystem::remove_all(cache_dir);

        SamplerStorage sampler_storage;
        LbfWriter writer{LbfFileKind::asset_pack};

        auto const report = cook_assets(AssetCookRequest{.models = {source}}, sampler_storage, writer,
                                        AssetCookOptions{.texture_cache_directory = cache_dir});

        CHECK(report.failures.empty());
        CHECK(report.models_cooked == 1);

        auto reader = LbfReader::from_memory(*writer.finish());
        REQUIRE(reader.has_value());

        auto const pack = AssetPack::from_reader(std::move(*reader));
        auto const id = asset_id_from_key(model_asset_key(source));
        REQUIRE(pack->has_model(id));

        auto model = pack->load_model(id, sampler_storage);
        REQUIRE(model.has_value());
        REQUIRE_FALSE(model->meshes.empty());

        auto reference = load_model_cpu(source, sampler_storage);
        REQUIRE(reference.has_value());
        REQUIRE(model->meshes.size() == reference->meshes.size());

        auto const &cooked = model->meshes[0].primitives[0];
        auto const &original = reference->meshes[0].primitives[0];
        CHECK(cooked.compressed_vertices.size() == original.compressed_vertices.size());
        CHECK(cooked.indices.size() == original.indices.size());
        CHECK(cooked.meshlets[0]->meshlets.size() == original.meshlets[0]->meshlets.size());

        // Every texture the model references was cooked into the pack and decodes.
        CHECK(report.textures_cooked == model->image_sources.size());

        for (auto const &image: model->image_sources) {
            REQUIRE(static_cast<bool>(image.cooked));
            auto texture = image.cooked();
            REQUIRE(texture.has_value());
            CHECK_FALSE(texture->mips.empty());
        }

        // A second cook reuses every chunk from the first pack instead of cooking again.
        LbfWriter second{LbfFileKind::asset_pack};
        auto const copied = cook_assets(AssetCookRequest{.models = {source}}, sampler_storage, second,
                                        AssetCookOptions{.source_packs = {pack}, .texture_cache_directory = cache_dir});

        CHECK(copied.models_copied == 1);
        CHECK(copied.models_cooked == 0);
        CHECK(copied.textures_cooked == 0);
        CHECK(copied.textures_copied == report.textures_cooked);

        std::filesystem::remove_all(cache_dir);
    }

    TEST_CASE("Xxh64Stream matches one-shot xxh64 for any split") {
        auto const data = repetitive_payload(1000);

        for (std::size_t split: {0UL, 1UL, 31UL, 32UL, 33UL, 500UL, 999UL, 1000UL}) {
            Xxh64Stream stream;
            stream.update(std::span{data}.first(split));
            stream.update(std::span{data}.subspan(split));
            CHECK(stream.digest() == xxh64(data));
        }

        CHECK(Xxh64Stream{}.digest() == xxh64(std::span<std::byte const>{}));
    }

    TEST_CASE("LBF files stream multi-megabyte chunks from disk") {
        auto const path = std::filesystem::temp_directory_path() / "lathe_lbf_stream_test.lbf";

        // Several 1 MiB read blocks each; one compressible, one not.
        auto const compressible = repetitive_payload(3 * 1024 * 1024 + 17);
        std::vector<std::byte> incompressible(2 * 1024 * 1024 + 5);
        std::uint64_t state = 0x12345678;
        for (auto &byte: incompressible) {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            byte = static_cast<std::byte>(state >> 56U);
        }

        {
            LbfWriter writer{LbfFileKind::asset_pack};
            writer.add_chunk(LbfChunkInput{.type = lbf_chunk::texture, .id = 1, .payload = compressible});
            writer.add_chunk(LbfChunkInput{.type = lbf_chunk::texture, .id = 2, .payload = incompressible});

            auto written = writer.write_file(path);
            REQUIRE(written.has_value());
            CHECK(*written == std::filesystem::file_size(path));
        }

        auto reader = LbfReader::open(path);
        REQUIRE(reader.has_value());

        auto const *first = reader->find(lbf_chunk::texture, 1);
        auto const *second = reader->find(lbf_chunk::texture, 2);
        REQUIRE(first != nullptr);
        REQUIRE(second != nullptr);
        CHECK(first->compression == LbfCompression::zstd);
        CHECK(second->compression == LbfCompression::none);

        CHECK(*reader->read_chunk(*first) == compressible);
        CHECK(*reader->read_chunk(*second) == incompressible);

        // Corrupt one byte deep inside the compressed chunk: the streamed checksum catches it.
        {
            std::fstream file{path, std::ios::binary | std::ios::in | std::ios::out};
            file.seekp(static_cast<std::streamoff>(first->offset + first->stored_size / 2));
            char const flipped = 0x5A;
            file.write(&flipped, 1);
        }

        auto corrupted = LbfReader::open(path);
        REQUIRE(corrupted.has_value());
        CHECK_FALSE(corrupted->read_chunk(*corrupted->find(lbf_chunk::texture, 1)).has_value());

        std::filesystem::remove(path);
    }

    TEST_CASE("LBF container: a hostile table of contents is refused before anything is allocated") {
        constexpr auto terabyte = std::uint64_t{1} << 40U;

        SUBCASE("a chunk claiming a terabyte decompressed size") {
            // Valid checksums throughout: only raw_size lies. Before the cap this aborted the process on allocation.
            auto file = with_patched_entry(single_chunk_file(true),
                                           [&](LbfChunkEntry &entry) { entry.raw_size = terabyte; });

            CHECK_FALSE(LbfReader::from_memory(std::move(file)).has_value());
        }

        SUBCASE("a plausible raw_size that disagrees with the zstd frame") {
            auto file =
                    with_patched_entry(single_chunk_file(true), [](LbfChunkEntry &entry) { entry.raw_size = 1000; });

            auto reader = LbfReader::from_memory(std::move(file));
            REQUIRE(reader.has_value());
            CHECK_FALSE(reader->read_chunk(reader->chunks().front()).has_value());
        }

        SUBCASE("the same lie through the streamed file path") {
            auto const path = std::filesystem::temp_directory_path() / "lathe_lbf_raw_size_lie.lbf";
            auto const file =
                    with_patched_entry(single_chunk_file(true), [](LbfChunkEntry &entry) { entry.raw_size = 1000; });
            {
                std::ofstream out{path, std::ios::binary};
                out.write(reinterpret_cast<char const *>(file.data()), static_cast<std::streamsize>(file.size()));
            }

            auto reader = LbfReader::open(path);
            REQUIRE(reader.has_value());
            CHECK_FALSE(reader->read_chunk(reader->chunks().front()).has_value());
            std::filesystem::remove(path);
        }

        SUBCASE("an unknown compression scheme") {
            auto file = with_patched_entry(single_chunk_file(true), [](LbfChunkEntry &entry) {
                entry.compression = static_cast<LbfCompression>(9);
            });

            CHECK_FALSE(LbfReader::from_memory(std::move(file)).has_value());
        }

        SUBCASE("a chunk overlapping the file header") {
            auto file = with_patched_entry(single_chunk_file(false), [](LbfChunkEntry &entry) { entry.offset = 0; });

            CHECK_FALSE(LbfReader::from_memory(std::move(file)).has_value());
        }

        SUBCASE("a header that promises a huge table in a tiny file") {
            // Used to make open() allocate toc_size bytes before comparing file_size with the real size.
            auto const path = std::filesystem::temp_directory_path() / "lathe_lbf_huge_toc.lbf";
            LbfFileHeader header{};
            header.chunk_count = 1U << 30U;
            header.toc_offset = sizeof(LbfFileHeader);
            header.toc_size = std::uint64_t{header.chunk_count} * sizeof(LbfChunkEntry);
            header.file_size = terabyte;
            {
                std::ofstream out{path, std::ios::binary};
                out.write(reinterpret_cast<char const *>(&header), sizeof(header));
            }

            auto reader = LbfReader::open(path);
            REQUIRE_FALSE(reader.has_value());
            CHECK(reader.error().type == LbfErrorType::corrupt_header);
            std::filesystem::remove(path);
        }
    }

    TEST_CASE("Cooked texture: descriptions the GPU could not copy safely are refused") {
        REQUIRE_FALSE(validate_compressed_texture(valid_texture()).has_value());

        auto const rejected = [](auto &&spoil) {
            auto texture = valid_texture();
            spoil(texture);

            auto const decoded = decode_cooked_texture(encode_cooked_texture(texture, TextureRole::colour));
            CHECK(validate_compressed_texture(texture).has_value());
            CHECK_FALSE(decoded.has_value());
        };

        SUBCASE("a format outside the supported block formats") {
            rejected([](CompressedTexture &texture) { texture.format = VK_FORMAT_R8G8B8A8_UNORM; });
            rejected([](CompressedTexture &texture) { texture.format = VK_FORMAT_ASTC_4x4_UNORM_BLOCK; });

            // An arbitrary u32 in the file isn't a VkFormat at all; it is refused before it is cast.
            auto payload = encode_cooked_texture(valid_texture(), TextureRole::colour);
            auto const bogus = std::uint32_t{0xFFFFFF91U};
            std::memcpy(payload.data(), &bogus, sizeof(bogus));
            CHECK_FALSE(decode_cooked_texture(payload).has_value());
        }

        SUBCASE("zero or oversized extents") {
            rejected([](CompressedTexture &texture) { texture.width = 0; });
            rejected([](CompressedTexture &texture) { texture.width = max_compressed_texture_extent + 1; });
        }

        SUBCASE("a mip that reads past the data") {
            rejected([](CompressedTexture &texture) { texture.mips[1].byte_offset = 40; });
            rejected([](CompressedTexture &texture) { texture.mips[1].byte_offset = 0xFFFFFFF0U; });
        }

        SUBCASE("a mip whose size does not match its format and extent") {
            rejected([](CompressedTexture &texture) { texture.mips[0].byte_length = 31; });
            rejected([](CompressedTexture &texture) { texture.mips[1].width = 5; });
        }

        SUBCASE("more mips than the extent has levels") {
            rejected([](CompressedTexture &texture) {
                texture.mips.resize(5,
                                    CompressedMipLevel{.width = 1, .height = 1, .byte_offset = 0, .byte_length = 16});
            });
        }
    }

    TEST_CASE("Scene codec: values that would poison the renderer or physics are refused") {
        REQUIRE(validate_scene(populated_scene()).has_value());

        auto const rejected = [](auto &&spoil) {
            auto scene = populated_scene();
            spoil(scene);

            CHECK_FALSE(validate_scene(scene).has_value());
            // Through the codec too, which is the path a file takes.
            CHECK_FALSE(decode_scene(encode_scene(scene)).has_value());
        };

        constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
        constexpr auto inf = std::numeric_limits<float>::infinity();

        SUBCASE("transforms") {
            rejected([&](SceneDescription &scene) { scene.entities[0].transform->position.x = nan; });
            rejected([&](SceneDescription &scene) { scene.entities[0].transform->scale.y = inf; });
            rejected([&](SceneDescription &scene) { scene.entities[0].transform->rotation = glm::quat{0, 0, 0, 0}; });
        }

        SUBCASE("lights") {
            rejected([&](SceneDescription &scene) { scene.point_lights[0].light.intensity = nan; });
            rejected([&](SceneDescription &scene) { scene.point_lights[0].light.range = -1.0F; });
            rejected([&](SceneDescription &scene) { scene.spot_lights[0].light.outer_cone_degrees = inf; });
        }

        SUBCASE("rigid bodies") {
            rejected([&](SceneDescription &scene) { scene.rigid_bodies[0].body.mass = -1.0F; });
            rejected([&](SceneDescription &scene) { scene.rigid_bodies[0].body.half_extents.x = -0.5F; });
            rejected([&](SceneDescription &scene) { scene.rigid_bodies[0].body.velocity.z = nan; });
        }

        SUBCASE("materials") {
            rejected([&](SceneDescription &scene) { scene.materials[0].roughness_factor = nan; });
            rejected([](SceneDescription &scene) { scene.materials[0].max_shadow_cascade = shadow_cascade_count; });
            rejected([](SceneDescription &scene) { scene.materials[0].max_shadow_cascade = 0xFFFFFFFEU; });
            rejected([](SceneDescription &scene) { scene.materials[0].far_material = 0; });
            rejected([](SceneDescription &scene) {
                scene.materials[0].far_material = static_cast<std::uint32_t>(scene.materials.size());
            });
        }

        SUBCASE("lifetimes and gravity") {
            rejected([&](SceneDescription &scene) { scene.lifetimes[0].remaining_seconds = nan; });
            rejected([&](SceneDescription &scene) { scene.physics_settings.gravity.y = inf; });
        }

        SUBCASE("a component listed twice for one entity") {
            // instantiate_scene() emplaces each component, and EnTT asserts on a second emplace.
            rejected([](SceneDescription &scene) { scene.model_components.push_back(scene.model_components[0]); });
            rejected([](SceneDescription &scene) { scene.point_lights.push_back(scene.point_lights[0]); });
            rejected([](SceneDescription &scene) { scene.rigid_bodies.push_back(scene.rigid_bodies[0]); });
            rejected([](SceneDescription &scene) { scene.lifetimes.push_back(scene.lifetimes[0]); });
        }

        SUBCASE("a material that never casts shadows is still valid") {
            auto scene = populated_scene();
            scene.materials[0].max_shadow_cascade = GpuMaterial::no_shadow_cascade;
            CHECK(validate_scene(scene).has_value());
        }
    }

    TEST_CASE("Cooked model: geometry the GPU would read out of bounds is refused") {
        // The encoder doesn't validate, so a valid model corrupted before encoding stands in for a hostile file.
        auto const decode_after = [](auto &&spoil) {
            auto cpu_data = finalized_cube();
            spoil(cpu_data);

            auto const payload = encode_cooked_model(cpu_data, {}, {});
            REQUIRE(payload.has_value());
            return decode_cooked_model(*payload);
        };

        CHECK(decode_after([](ModelCpuData &) {}).has_value());

        SUBCASE("an index past the vertex count") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            auto &primitive = data.meshes[0].primitives[0];
                            primitive.indices[0] = static_cast<std::uint32_t>(primitive.compressed_vertices.size());
                        }).has_value());
        }

        SUBCASE("a meshlet vertex list pointing past the vertex buffer") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            auto &build = *data.meshes[0].primitives[0].meshlets[0];
                            build.topology.data[build.meshlets[0].vertex_offset] = 0x00FFFFFFU;
                        }).has_value());
        }

        SUBCASE("a packed triangle indexing past its meshlet's vertices") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            auto &build = *data.meshes[0].primitives[0].meshlets[0];
                            build.topology.data[build.meshlets[0].triangle_offset] = 0x00FFFFFFU;
                        }).has_value());
        }

        SUBCASE("meshlets larger than the mesh shader's output arrays") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            data.meshes[0].primitives[0].meshlets[0]->meshlets[0].vertex_count =
                                    meshlet_max_vertices + 1;
                        }).has_value());
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            data.meshes[0].primitives[0].meshlets[0]->meshlets[0].triangle_count =
                                    meshlet_max_triangles + 1;
                        }).has_value());
        }

        SUBCASE("non-finite values") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            data.meshes[0].primitives[0].meshlets[0]->meshlets[0].radius =
                                    std::numeric_limits<float>::quiet_NaN();
                        }).has_value());
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            data.nodes[0].local_transform[3][0] = std::numeric_limits<float>::infinity();
                        }).has_value());
        }

        SUBCASE("a node graph with a cycle or a shared child") {
            CHECK_FALSE(decode_after([](ModelCpuData &data) {
                            data.nodes[0].children = {0};
                            data.scene_roots = {0};
                        }).has_value());
            CHECK_FALSE(decode_after([](ModelCpuData &data) { data.scene_roots = {0, 0}; }).has_value());
        }
    }

    TEST_CASE("Cooked model: counts that outrun the payload are refused before allocating") {
        ByteWriter writer;

        // Bounds, then zero images and materials, then a node count no payload this small could hold.
        for (int component = 0; component < 6; ++component) {
            writer.write(0.0F);
        }

        writer.write(std::uint32_t{0}); // images
        writer.write(std::uint32_t{0}); // materials
        writer.write(std::uint32_t{0x7FFFFFFFU}); // nodes: 2^31 * sizeof(ModelNode) if believed

        CHECK_FALSE(decode_cooked_model(writer.bytes()).has_value());
    }
}
