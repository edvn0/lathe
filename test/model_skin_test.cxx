#include <doctest/doctest.h>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/tools.hpp>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>

#include "animation/humanoid.hxx"
#include "animation/joint_remap.hxx"
#include "assets/load_model.hxx"
#include "assets/model_skin_import.hxx"
#include "serialisation/cooked_model.hxx"

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {
    struct Loaded {
        fastgltf::Asset asset;
        GltfSkinImport skin;
    };

    auto load_fixture() -> std::optional<Loaded> {
        auto const path = std::filesystem::path{TEST_ASSETS_DIR} / "test/data/animated_human.glb";
        auto data = fastgltf::GltfDataBuffer::FromPath(path);
        if (!data) {
            return std::nullopt;
        }
        fastgltf::Parser parser;
        auto asset = parser.loadGltf(data.get(), path.parent_path(), fastgltf::Options::LoadExternalBuffers);
        if (asset.error() != fastgltf::Error::None) {
            return std::nullopt;
        }
        auto skin = import_gltf_skin(asset.get(), true);
        if (!skin || !skin->has_value()) {
            return std::nullopt;
        }
        return Loaded{std::move(asset.get()), std::move(**skin)};
    }

    auto fixture_cpu_data(Loaded const &loaded) -> ModelCpuData {
        auto const &asset = loaded.asset;
        auto const &primitive = asset.meshes[0].primitives[0];
        auto const positions = [&] {
            auto const &accessor = asset.accessors[primitive.findAttribute("POSITION")->accessorIndex];
            std::vector<glm::vec3> values(accessor.count);
            fastgltf::copyFromAccessor<glm::vec3>(asset, accessor, values.data());
            return values;
        }();
        ModelCpuPrimitive raw;
        for (auto const &p: positions) {
            raw.vertices.push_back(ModelVertex{.position = glm::vec3{p.x, p.y, -p.z}});
        }
        auto const &index_accessor = asset.accessors[*primitive.indicesAccessor];
        raw.indices.resize(index_accessor.count);
        fastgltf::copyFromAccessor<std::uint32_t>(asset, index_accessor, raw.indices.data());
        for (std::size_t i = 0; i + 2 < raw.indices.size(); i += 3) {
            std::swap(raw.indices[i + 1], raw.indices[i + 2]);
        }
        auto skin = read_skin_vertices(asset, primitive, loaded.skin.joint_remap, raw.vertices.size());
        REQUIRE(skin.has_value());
        raw.skin = std::move(*skin);
        REQUIRE(generate_tangents(raw.vertices, raw.indices, &raw.skin).has_value());
        raw.reduced_indices = generate_mesh_lods(raw.vertices, raw.indices);
        prepare_primitive_gpu_data(raw);

        ModelCpuData data;
        data.meshes.push_back(ModelCpuMesh{});
        data.meshes[0].primitives.push_back(std::move(raw));
        data.animation = loaded.skin.data;
        return data;
    }
}

TEST_SUITE("model skin import") {
    TEST_CASE("skeleton is topologically ordered with 48 joints") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const &skeleton = loaded->skin.data->skeleton;
        CHECK(skeleton.joint_count() == 48);
        for (std::size_t i = 0; i < skeleton.joint_count(); ++i) {
            CHECK(skeleton.parents()[i] < static_cast<std::int32_t>(i));
        }
        CHECK(skeleton.find_joint("Hips").has_value());
        CHECK(skeleton.find_joint("LeftUpLeg").has_value());
    }

    TEST_CASE("bind pose palette is the identity") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const &skeleton = loaded->skin.data->skeleton;
        std::vector<glm::mat4> palette(skeleton.joint_count());
        Animation::compute_skinning_palette(skeleton, skeleton.bind_pose().view(), palette);
        for (auto const &matrix: palette) {
            for (int c = 0; c < 4; ++c) {
                for (int r = 0; r < 4; ++r) {
                    CHECK(matrix[c][r] == doctest::Approx(c == r ? 1.0F : 0.0F).epsilon(1e-3).scale(1.0));
                }
            }
        }
    }

    TEST_CASE("clips are imported by name and sample") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const &data = *loaded->skin.data;
        CHECK(data.clips.size() >= 8);
        for (auto const *name: {"Idle", "Walk", "Run", "Jump"}) {
            auto const *clip = data.find_clip(name);
            REQUIRE(clip != nullptr);
            CHECK(clip->duration > 0.0F);
            auto const instance = clip->make_clip();
            Animation::Pose pose{data.skeleton.joint_count()};
            instance->sample(0.5F, pose.view());
            for (std::size_t i = 0; i < pose.size(); ++i) {
                CHECK(std::isfinite(pose.joint(i).translation.x));
                CHECK(glm::length(pose.joint(i).rotation) == doctest::Approx(1.0F).epsilon(1e-3));
            }
        }
    }

    TEST_CASE("vertex weights are normalised with joints in range") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const &primitive = loaded->asset.meshes[0].primitives[0];
        auto const count = loaded->asset.accessors[primitive.findAttribute("POSITION")->accessorIndex].count;
        auto skin = read_skin_vertices(loaded->asset, primitive, loaded->skin.joint_remap, count);
        REQUIRE(skin.has_value());
        CHECK(skin->size() == count);
        for (auto const &vertex: *skin) {
            std::uint32_t sum = 0;
            for (std::size_t k = 0; k < 4; ++k) {
                sum += vertex.weights[k];
                CHECK(vertex.joints[k] < 48);
            }
            CHECK(sum == 65535);
        }
    }

    TEST_CASE("weight quantisation sums exactly and handles degenerate input") {
        auto const a = quantise_skin_weights({0.3F, 0.3F, 0.3F, 0.0F});
        CHECK(std::accumulate(a.begin(), a.end(), 0U) == 65535);
        auto const zero = quantise_skin_weights({0.0F, 0.0F, 0.0F, 0.0F});
        CHECK(zero[0] == 65535);
        auto const scaled = quantise_skin_weights({2.0F, 2.0F, 0.0F, 0.0F});
        CHECK(scaled[0] == scaled[1] + (scaled[0] - scaled[1]));
        CHECK(scaled[0] + scaled[1] == 65535);
    }

    TEST_CASE("cooked model round-trips skin data and v2 payloads still load") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const cpu_data = fixture_cpu_data(*loaded);

        auto payload = encode_cooked_model(cpu_data, {}, {});
        REQUIRE(payload.has_value());
        auto decoded = decode_cooked_model(*payload);
        REQUIRE(decoded.has_value());

        auto const &original = cpu_data.meshes[0].primitives[0];
        auto const &copy = decoded->cpu_data.meshes[0].primitives[0];
        CHECK(copy.skin == original.skin);
        CHECK(copy.skin.size() == copy.compressed_vertices.size());

        auto const &a = cpu_data.animation->skeleton;
        REQUIRE(decoded->cpu_data.animation != nullptr);
        auto const &b = decoded->cpu_data.animation->skeleton;
        CHECK(b.joint_count() == 48);
        CHECK(std::ranges::equal(a.names(), b.names()));
        CHECK(std::ranges::equal(a.parents(), b.parents()));
        CHECK(std::ranges::equal(a.inverse_bind(), b.inverse_bind()));
        REQUIRE(decoded->cpu_data.animation->clips.size() == cpu_data.animation->clips.size());
        auto const &clip_a = cpu_data.animation->clips[3];
        auto const &clip_b = decoded->cpu_data.animation->clips[3];
        CHECK(clip_a.name == clip_b.name);
        CHECK(clip_a.duration == clip_b.duration);
        REQUIRE(clip_a.rotations.size() == clip_b.rotations.size());
        CHECK(clip_a.rotations[0].times == clip_b.rotations[0].times);
        CHECK(clip_a.rotations[0].values == clip_b.rotations[0].values);

        auto unskinned = cpu_data;
        unskinned.meshes[0].primitives[0].skin.clear();
        unskinned.animation = nullptr;
        auto old_payload = encode_cooked_model(unskinned, {}, {});
        REQUIRE(old_payload.has_value());
        old_payload->resize(old_payload->size() - 2);
        auto old_decoded = decode_cooked_model(*old_payload, 2);
        REQUIRE(old_decoded.has_value());
        CHECK(old_decoded->cpu_data.meshes[0].primitives[0].skin.empty());
        CHECK(old_decoded->cpu_data.animation == nullptr);

        CHECK_FALSE(decode_cooked_model(std::span{*payload}.first(payload->size() - 5)).has_value());
    }
}

TEST_SUITE("joint remap") {
    TEST_CASE("canonical names unify Mixamo and rig names") {
        CHECK(Animation::canonical_joint_name("mixamorig:LeftUpLeg") == Animation::canonical_joint_name("thigh_l"));
        CHECK(Animation::canonical_joint_name("RightForeArm") == Animation::canonical_joint_name("forearm_r"));
        CHECK(Animation::canonical_joint_name("Hips") == "pelvis");
    }

    TEST_CASE("48-joint source maps onto the 14-joint humanoid") {
        auto const loaded = load_fixture();
        REQUIRE(loaded.has_value());
        auto const remap = Animation::remap_to_humanoid(loaded->skin.data->skeleton);
        CHECK(remap.mapped_count == Animation::Humanoid::JointCount);
        auto const &skeleton = loaded->skin.data->skeleton;
        auto const source = [&](Animation::Humanoid::Joint joint) {
            return skeleton.names()[static_cast<std::size_t>(remap.target_to_source[joint])];
        };
        CHECK(source(Animation::Humanoid::Pelvis) == "Hips");
        CHECK(source(Animation::Humanoid::Chest) == "Spine2");
        CHECK(source(Animation::Humanoid::ThighL) == "LeftUpLeg");
        CHECK(source(Animation::Humanoid::ShinR) == "RightLeg");
        CHECK(source(Animation::Humanoid::ForearmL) == "LeftForeArm");
        CHECK(source(Animation::Humanoid::UpperArmR) == "RightArm");
        CHECK(source(Animation::Humanoid::FootR) == "RightFoot");
        CHECK(std::ranges::count(remap.source_to_target, Animation::no_joint) == 48 - 14);
    }
}
