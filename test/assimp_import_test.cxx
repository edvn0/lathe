#include <doctest/doctest.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>

#include "assets/load_model.hxx"
#include "assets/model_formats.hxx"
#include "gpu/sampler_storage.hxx"

namespace {

    struct TemporaryDirectory {
        TemporaryDirectory() : path{std::filesystem::temp_directory_path() / "lathe_assimp_import_test"} {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }

        ~TemporaryDirectory() { std::filesystem::remove_all(path); }

        TemporaryDirectory(TemporaryDirectory const &) = delete;
        auto operator=(TemporaryDirectory const &) -> TemporaryDirectory & = delete;

        std::filesystem::path path;
    };

    auto write_file(std::filesystem::path const &path, std::string const &contents) -> void {
        std::ofstream file{path, std::ios::binary};
        file << contents;
    }

    // A unit quad in the XY plane at z = 2, facing +Z, with a red diffuse material.
    constexpr auto quad_obj = R"(mtllib quad.mtl
o quad
v 0 0 2
v 1 0 2
v 1 1 2
v 0 1 2
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl red
f 1/1/1 2/2/1 3/3/1 4/4/1
)";

    constexpr auto quad_mtl = R"(newmtl red
Kd 1 0 0
Ns 100
)";

    // The same quad as quad_obj, authored as USD ASCII.
    constexpr auto quad_usda = R"(#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root"
{
    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(0, 0, 2), (1, 0, 2), (1, 1, 2), (0, 1, 2)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
    }
}
)";

} // namespace

TEST_SUITE("assimp import") {
    TEST_CASE("extension helpers route glTF and Assimp formats") {
        CHECK(is_gltf_model("a.GLB"));
        CHECK(is_gltf_model("a.gltf"));
        CHECK_FALSE(is_gltf_model("a.fbx"));

        CHECK(is_assimp_model("a.fbx"));
        CHECK(is_assimp_model("dir/b.OBJ"));
        CHECK(is_assimp_model("c.dae"));
        CHECK(is_assimp_model("scene.USDZ"));
        CHECK(is_assimp_model("scene.usda"));
        CHECK(is_assimp_model("scene.usdc"));
        CHECK(is_assimp_model("scene.usd"));
        CHECK_FALSE(is_assimp_model("c.glb"));

        CHECK(is_supported_model("d.stl"));
        CHECK_FALSE(is_supported_model("d.txt"));
        CHECK_FALSE(is_supported_model("noextension"));
    }

    TEST_CASE("an OBJ loads with the glTF path's handedness and UV conventions") {
        TemporaryDirectory const directory;
        write_file(directory.path / "quad.obj", quad_obj);
        write_file(directory.path / "quad.mtl", quad_mtl);

        SamplerStorage sampler_storage;
        auto model = load_model_cpu(AssetPath::external(directory.path / "quad.obj").value(), sampler_storage);
        REQUIRE(model.has_value());

        REQUIRE(model->meshes.size() == 1);
        REQUIRE(model->meshes[0].primitives.size() == 1);
        REQUIRE(model->materials.size() >= 1);

        auto const &primitive = model->meshes[0].primitives[0];
        CHECK(primitive.indices.size() == 6);
        CHECK(primitive.material_index.has_value());

        // Z is negated to match the engine's convention, and the +Z normal follows.
        for (auto const &vertex: primitive.vertices) {
            CHECK(vertex.position.z == doctest::Approx(-2.0F));
            CHECK(vertex.normal.z == doctest::Approx(-1.0F));
            CHECK(vertex.texcoord.y == doctest::Approx(1.0F - (vertex.position.y > 0.5F ? 1.0F : 0.0F)));
        }

        // Mirroring Z flips the winding; the triangles must still face the way their normals point.
        for (std::size_t triangle = 0; triangle + 2 < primitive.indices.size(); triangle += 3) {
            auto const &a = primitive.vertices[primitive.indices[triangle]];
            auto const &b = primitive.vertices[primitive.indices[triangle + 1]];
            auto const &c = primitive.vertices[primitive.indices[triangle + 2]];

            CHECK(glm::dot(glm::cross(b.position - a.position, c.position - a.position), a.normal) > 0.0F);
        }

        // Assimp's OBJ importer also adds its own grey "default" material, so go through the primitive's index.
        REQUIRE(*primitive.material_index < model->materials.size());
        auto const &material = model->materials[*primitive.material_index];
        CHECK(material.base_colour_factor.x == doctest::Approx(1.0F));
        CHECK(material.base_colour_factor.y == doctest::Approx(0.0F));
        CHECK(material.metallic_factor == doctest::Approx(0.0F));
        CHECK(material.roughness_factor < 1.0F);

        REQUIRE(model->scene_roots.size() == 1);
        CHECK_FALSE(model->nodes.empty());
    }

    TEST_CASE("a USD quad loads as two triangles with Z mirrored") {
        TemporaryDirectory const directory;
        write_file(directory.path / "quad.usda", quad_usda);

        SamplerStorage sampler_storage;
        auto model = load_model_cpu(AssetPath::external(directory.path / "quad.usda").value(), sampler_storage);
        REQUIRE(model.has_value());

        std::size_t index_count = 0;

        for (auto const &mesh: model->meshes) {
            for (auto const &primitive: mesh.primitives) {
                index_count += primitive.indices.size();

                for (auto const &vertex: primitive.vertices) {
                    CHECK(vertex.position.z == doctest::Approx(-2.0F));
                }
            }
        }

        CHECK(index_count == 6);
        CHECK_FALSE(model->nodes.empty());
    }

    TEST_CASE("a missing file reports file_not_found") {
        SamplerStorage sampler_storage;
        auto const missing = std::filesystem::temp_directory_path() / "lathe_assimp_import_test_missing.obj";
        auto model = load_model_cpu(AssetPath::external(missing).value(), sampler_storage);

        REQUIRE_FALSE(model.has_value());
        CHECK(model.error().type == ModelLoadErrorType::file_not_found);
    }

    TEST_CASE("garbage content reports parse_error") {
        TemporaryDirectory const directory;
        write_file(directory.path / "broken.fbx", "this is not an fbx file");

        SamplerStorage sampler_storage;
        auto model = load_model_cpu(AssetPath::external(directory.path / "broken.fbx").value(), sampler_storage);

        REQUIRE_FALSE(model.has_value());
        CHECK(model.error().type == ModelLoadErrorType::parse_error);
    }

    TEST_CASE("garbage USD content reports parse_error") {
        TemporaryDirectory const directory;
        write_file(directory.path / "broken.usda", "#usda 1.0\ndef Mesh {{{ not usd");

        SamplerStorage sampler_storage;
        auto model = load_model_cpu(AssetPath::external(directory.path / "broken.usda").value(), sampler_storage);

        REQUIRE_FALSE(model.has_value());
        CHECK(model.error().type == ModelLoadErrorType::parse_error);
    }
}
