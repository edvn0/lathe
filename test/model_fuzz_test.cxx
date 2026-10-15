#include <doctest/doctest.h>

// Linux only: each load runs in a forked child, so a crash, abort or hang in a loader (ours or Assimp's) fails the one
// input that caused it instead of taking the whole test run down.
#if defined(__linux__)

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "assets/load_model.hxx"
#include "gpu/sampler_storage.hxx"

#ifndef TEST_ASSETS_DIR
#error "TEST_ASSETS_DIR must be defined by the build"
#endif

namespace {

    enum class Outcome : std::uint8_t {
        loaded,
        rejected,
        crashed,
        timed_out,
    };

    auto describe(Outcome outcome) -> std::string_view {
        switch (outcome) {
            case Outcome::loaded:
                return "loaded";
            case Outcome::rejected:
                return "rejected";
            case Outcome::crashed:
                return "crashed";
            case Outcome::timed_out:
                return "timed out";
        }

        return "unknown";
    }

    constexpr int exit_loaded = 10;
    constexpr int exit_rejected = 11;
    constexpr unsigned int load_timeout_seconds = 30;

    // Loads `file` in a child process. Anything but a clean "loaded" or "rejected" exit (a signal, an ASan/UBSan
    // report, an assert, the alarm) is a crash or a hang.
    auto load_isolated(std::filesystem::path const &file) -> Outcome {
        std::fflush(nullptr);

        auto const child = fork();

        if (child < 0) {
            return Outcome::crashed;
        }

        if (child == 0) {
            alarm(load_timeout_seconds);

            SamplerStorage sampler_storage;
            auto const path = AssetPath::external(file);

            if (!path.has_value()) {
                _exit(exit_rejected);
            }

            auto const model = load_model_cpu(*path, sampler_storage);

            _exit(model.has_value() ? exit_loaded : exit_rejected);
        }

        int status = 0;
        waitpid(child, &status, 0);

        if (WIFEXITED(status)) {
            switch (WEXITSTATUS(status)) {
                case exit_loaded:
                    return Outcome::loaded;
                case exit_rejected:
                    return Outcome::rejected;
                default:
                    return Outcome::crashed;
            }
        }

        return WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM ? Outcome::timed_out : Outcome::crashed;
    }

    struct TemporaryDirectory {
        TemporaryDirectory() : path{std::filesystem::temp_directory_path() / "lathe_model_fuzz_test"} {
            std::filesystem::remove_all(path);
            std::filesystem::create_directories(path);
        }

        ~TemporaryDirectory() { std::filesystem::remove_all(path); }

        TemporaryDirectory(TemporaryDirectory const &) = delete;
        auto operator=(TemporaryDirectory const &) -> TemporaryDirectory & = delete;

        std::filesystem::path path;
    };

    auto write_file(std::filesystem::path const &path, std::span<std::byte const> bytes) -> void {
        std::ofstream file{path, std::ios::binary};
        file.write(reinterpret_cast<char const *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }

    auto write_file(std::filesystem::path const &path, std::string_view text) -> void {
        write_file(path, std::as_bytes(std::span{text.data(), text.size()}));
    }

    auto read_file(std::filesystem::path const &path) -> std::vector<std::byte> {
        std::ifstream file{path, std::ios::binary};
        std::vector<char> raw{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
        return std::vector<std::byte>{reinterpret_cast<std::byte const *>(raw.data()),
                                      reinterpret_cast<std::byte const *>(raw.data()) + raw.size()};
    }

    auto base64(std::span<std::byte const> bytes) -> std::string {
        constexpr std::string_view alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string encoded;

        for (std::size_t offset = 0; offset < bytes.size(); offset += 3) {
            auto const remaining = std::min<std::size_t>(3, bytes.size() - offset);
            std::uint32_t group = 0;

            for (std::size_t i = 0; i < 3; ++i) {
                group = (group << 8U) | (i < remaining ? std::to_integer<std::uint32_t>(bytes[offset + i]) : 0U);
            }

            encoded.push_back(alphabet[(group >> 18U) & 63U]);
            encoded.push_back(alphabet[(group >> 12U) & 63U]);
            encoded.push_back(remaining > 1 ? alphabet[(group >> 6U) & 63U] : '=');
            encoded.push_back(remaining > 2 ? alphabet[group & 63U] : '=');
        }

        return encoded;
    }

    // ---- glTF builders -----------------------------------------------------------------------------------------

    auto gltf_with_nodes(std::string const &nodes_json, std::string const &scene_nodes_json = "[0]") -> std::string {
        return R"({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":)" + scene_nodes_json +
               R"(}],"nodes":)" + nodes_json + "}";
    }

    // One triangle (positions, then `index_count` uint16 indices) in a data-URI buffer.
    auto triangle_gltf(std::array<std::uint16_t, 3> indices, std::size_t index_count = 3) -> std::string {
        constexpr std::array<float, 9> positions{0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};

        std::vector<std::byte> buffer(sizeof(positions) + index_count * sizeof(std::uint16_t));
        std::memcpy(buffer.data(), positions.data(), sizeof(positions));
        std::memcpy(buffer.data() + sizeof(positions), indices.data(), index_count * sizeof(std::uint16_t));

        return std::format(
                R"({{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],)"
                R"("meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],)"
                R"("buffers":[{{"byteLength":{},"uri":"data:application/octet-stream;base64,{}"}}],)"
                R"("bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":36,"target":34962}},)"
                R"({{"buffer":0,"byteOffset":36,"byteLength":{},"target":34963}}],)"
                R"("accessors":[{{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3",)"
                R"("min":[0,0,0],"max":[1,1,0]}},)"
                R"({{"bufferView":1,"componentType":5123,"count":{},"type":"SCALAR"}}]}})",
                buffer.size(), base64(buffer), index_count * sizeof(std::uint16_t), index_count);
    }

    // ---- Assimp-format samples ---------------------------------------------------------------------------------

    constexpr std::string_view quad_obj = R"(mtllib quad.mtl
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

    constexpr std::string_view quad_mtl = "newmtl red\nKd 1 0 0\nNs 100\n";

    constexpr std::string_view quad_usda = R"(#usda 1.0
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
    }
}
)";

    constexpr std::string_view triangle_dae = R"(<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_geometries>
    <geometry id="g" name="g">
      <mesh>
        <source id="p">
          <float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array>
          <technique_common>
            <accessor source="#pa" count="3" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
        <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
)";

    constexpr std::string_view triangle_ply = R"(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
3 0 1 2
)";

    constexpr std::string_view triangle_stl = R"(solid t
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid t
)";

    // quad_obj with its mtllib pointing back at the OBJ itself.
    auto self_referencing_obj() -> std::string {
        std::string text{quad_obj};
        text.replace(0, text.find('\n'), "mtllib self_mtl.obj");
        return text;
    }

    // ---- Mutation ----------------------------------------------------------------------------------------------

    auto below(std::mt19937_64 &rng, std::size_t bound) -> std::size_t { return bound == 0 ? 0 : rng() % bound; }

    auto mutate(std::vector<std::byte> bytes, std::mt19937_64 &rng) -> std::vector<std::byte> {
        auto const rounds = 1 + below(rng, 4);

        for (std::size_t round = 0; round < rounds && !bytes.empty(); ++round) {
            auto const at = below(rng, bytes.size());
            auto const length = std::min<std::size_t>(1 + below(rng, 64), bytes.size() - at);

            switch (below(rng, 6)) {
                case 0:
                    bytes[at] ^= static_cast<std::byte>(1U << below(rng, 8));
                    break;
                case 1:
                    bytes[at] = static_cast<std::byte>(below(rng, 2) != 0 ? 0xFFU : 0x00U);
                    break;
                case 2:
                    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(at),
                                bytes.begin() + static_cast<std::ptrdiff_t>(at + length));
                    break;
                case 3: {
                    std::vector<std::byte> copy{bytes.begin() + static_cast<std::ptrdiff_t>(at),
                                                bytes.begin() + static_cast<std::ptrdiff_t>(at + length)};
                    bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at), copy.begin(), copy.end());
                    break;
                }
                case 4: {
                    std::vector<std::byte> noise(length);
                    std::ranges::generate(noise, [&] { return static_cast<std::byte>(rng() & 0xFFU); });
                    bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at), noise.begin(), noise.end());
                    break;
                }
                default:
                    bytes.resize(at);
                    break;
            }
        }

        return bytes;
    }

    auto env_unsigned(char const *name, std::uint64_t fallback) -> std::uint64_t {
        if (auto const *value = std::getenv(name); value != nullptr && *value != '\0') {
            return std::strtoull(value, nullptr, 10);
        }

        return fallback;
    }

    struct Seed {
        std::string name;
        std::string extension;
        std::vector<std::byte> bytes;
    };

    auto text_seed(std::string name, std::string extension, std::string_view text) -> Seed {
        auto const bytes = std::as_bytes(std::span{text.data(), text.size()});
        return Seed{std::move(name), std::move(extension), std::vector<std::byte>{bytes.begin(), bytes.end()}};
    }

} // namespace

TEST_SUITE("model fuzz") {
    TEST_CASE("the harness tells a clean load from a rejection") {
        TemporaryDirectory const directory;

        write_file(directory.path / "valid.gltf", gltf_with_nodes(R"([{"children":[1]},{}])"));
        CHECK(describe(load_isolated(directory.path / "valid.gltf")) == describe(Outcome::loaded));

        write_file(directory.path / "garbage.gltf", "not json at all");
        CHECK(describe(load_isolated(directory.path / "garbage.gltf")) == describe(Outcome::rejected));

        CHECK(describe(load_isolated(directory.path / "absent.gltf")) == describe(Outcome::rejected));
    }

    TEST_CASE("glTF node hierarchies that are not finite trees are rejected") {
        TemporaryDirectory const directory;

        struct Case {
            std::string name;
            std::string document;
        };

        std::string deep_chain = "[";
        constexpr std::size_t deep_count = 5000;
        for (std::size_t i = 0; i < deep_count; ++i) {
            deep_chain += i + 1 < deep_count ? std::format(R"({{"children":[{}]}},)", i + 1) : "{}";
        }
        deep_chain += "]";

        std::string shallow_chain = "[";
        constexpr std::size_t shallow_count = 100;
        for (std::size_t i = 0; i < shallow_count; ++i) {
            shallow_chain += i + 1 < shallow_count ? std::format(R"({{"children":[{}]}},)", i + 1) : "{}";
        }
        shallow_chain += "]";

        std::array<Case, 8> const cases{{
                {"node is its own child", gltf_with_nodes(R"([{"children":[0]}])")},
                {"two node cycle", gltf_with_nodes(R"([{"children":[1]},{"children":[0]}])")},
                {"cycle below the root", gltf_with_nodes(R"([{"children":[1]},{"children":[2]},{"children":[1]}])")},
                {"cycle unreachable from the scene", gltf_with_nodes(R"([{},{"children":[2]},{"children":[1]}])")},
                {"child index out of range", gltf_with_nodes(R"([{"children":[7]}])")},
                {"scene root out of range", gltf_with_nodes(R"([{}])", "[3]")},
                {"node with two parents", gltf_with_nodes(R"([{"children":[1,2]},{"children":[2]},{}])")},
                {"hierarchy deeper than the limit", gltf_with_nodes(deep_chain)},
        }};

        for (auto const &test: cases) {
            auto const file = directory.path / "case.gltf";
            write_file(file, test.document);

            auto const outcome = load_isolated(file);
            CHECK_MESSAGE(describe(outcome) == describe(Outcome::rejected), test.name, ": ", describe(outcome));
        }

        write_file(directory.path / "tree.gltf", gltf_with_nodes(shallow_chain));
        auto const control = load_isolated(directory.path / "tree.gltf");
        CHECK_MESSAGE(describe(control) == describe(Outcome::loaded), "a legal deep chain: ", describe(control));
    }

    TEST_CASE("glTF primitives whose indices do not fit their vertices are rejected") {
        TemporaryDirectory const directory;

        write_file(directory.path / "valid.gltf", triangle_gltf({0, 1, 2}));
        auto const control = load_isolated(directory.path / "valid.gltf");
        REQUIRE_MESSAGE(describe(control) == describe(Outcome::loaded), "control triangle: ", describe(control));

        struct Case {
            std::string name;
            std::string document;
        };

        std::array<Case, 4> const cases{{
                {"index past the last vertex", triangle_gltf({0, 1, 3})},
                {"index at the uint16 maximum", triangle_gltf({0, 1, 0xFFFF})},
                {"index count not a multiple of three", triangle_gltf({0, 1, 2}, 2)},
                {"no indices", triangle_gltf({0, 1, 2}, 0)},
        }};

        for (auto const &test: cases) {
            auto const file = directory.path / "case.gltf";
            write_file(file, test.document);

            auto const outcome = load_isolated(file);
            CHECK_MESSAGE(describe(outcome) == describe(Outcome::rejected), test.name, ": ", describe(outcome));
        }
    }

    TEST_CASE("malformed files in the Assimp formats are rejected") {
        TemporaryDirectory const directory;

        std::string stl_header(80, ' ');
        std::string truncated_binary_stl = stl_header;
        truncated_binary_stl.append("\xE8\x03\x00\x00", 4); // claims 1000 triangles, carries none

        struct Case {
            std::string file;
            std::string contents;
        };

        std::array<Case, 8> const cases{{
                {"empty.obj", ""},
                {"vertices_only.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\n"},
                {"face_out_of_range.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 99\n"},
                {"junk.fbx", std::string{"Kaydara FBX Binary  \0\x1a\0", 23} + std::string(64, '\xFF')},
                {"not_a_zip.usdz", std::string(256, 'A')},
                {"unterminated.usda", "#usda 1.0\n(\n    defaultPrim = \"Root\"\ndef Xform \"Root\" {"},
                {"truncated.stl", truncated_binary_stl},
                {"header_only.ply", "ply\nformat ascii 1.0\nelement vertex 1000000\nproperty float x\nend_header\n"},
        }};

        for (auto const &test: cases) {
            auto const file = directory.path / test.file;
            write_file(file, test.contents);

            auto const outcome = load_isolated(file);
            CHECK_MESSAGE(describe(outcome) == describe(Outcome::rejected), test.file, ": ", describe(outcome));
        }
    }

    // Assimp and tinyusdz decide for themselves what a cyclic file means: they may break the cycle and load what is
    // left, or refuse it. Either is fine; recursing until the stack runs out, or never returning, is not.
    TEST_CASE("cyclic references in the Assimp formats terminate") {
        TemporaryDirectory const directory;

        struct Case {
            std::string file;
            std::string contents;
        };

        std::array<Case, 6> const cases{{
                {"self_mtl.obj", self_referencing_obj()},
                {"instance_loop.dae", R"(<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_nodes>
    <node id="a"><instance_node url="#b"/></node>
    <node id="b"><instance_node url="#a"/></node>
  </library_nodes>
  <library_visual_scenes>
    <visual_scene id="s"><node id="n"><instance_node url="#a"/></node></visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
)"},
                {"self_instance.dae", R"(<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_nodes><node id="a"><instance_node url="#a"/></node></library_nodes>
  <library_visual_scenes>
    <visual_scene id="s"><node id="n"><instance_node url="#a"/></node></visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>
)"},
                {"self_sublayer.usda", "#usda 1.0\n(\n    defaultPrim = \"A\"\n    subLayers = [@./self_sublayer.usda@]\n)\n"
                                       "def Xform \"A\"\n{\n}\n"},
                {"ref_a.usda", "#usda 1.0\n(\n    defaultPrim = \"A\"\n)\n"
                               "def Xform \"A\" (\n    references = @./ref_b.usda@\n)\n{\n}\n"},
                {"ref_b.usda", "#usda 1.0\n(\n    defaultPrim = \"B\"\n)\n"
                               "def Xform \"B\" (\n    references = @./ref_a.usda@\n)\n{\n}\n"},
        }};

        for (auto const &test: cases) {
            write_file(directory.path / test.file, test.contents);
        }

        for (auto const &test: cases) {
            auto const outcome = load_isolated(directory.path / test.file);
            CHECK_MESSAGE((outcome == Outcome::loaded || outcome == Outcome::rejected), test.file, ": ",
                          describe(outcome));
        }
    }

    // Deterministic mutation fuzzing: LATHE_FUZZ_SEED picks the stream and LATHE_FUZZ_ITERATIONS the length, so a
    // failure reported here reproduces exactly.
    TEST_CASE("mutated files never crash or hang a loader") {
        TemporaryDirectory const directory;
        auto const base_seed = env_unsigned("LATHE_FUZZ_SEED", 0x1A7E);
        auto const iterations = env_unsigned("LATHE_FUZZ_ITERATIONS", 60);

        write_file(directory.path / "quad.mtl", quad_mtl);

        std::vector<Seed> seeds;
        seeds.push_back(text_seed("obj", ".obj", quad_obj));
        seeds.push_back(text_seed("usda", ".usda", quad_usda));
        seeds.push_back(text_seed("gltf", ".gltf", triangle_gltf({0, 1, 2})));
        seeds.push_back(text_seed("dae", ".dae", triangle_dae));
        seeds.push_back(text_seed("ply", ".ply", triangle_ply));
        seeds.push_back(text_seed("stl", ".stl", triangle_stl));

        auto const glb = std::filesystem::path{TEST_ASSETS_DIR} / "assets/models/test_cube.glb";
        REQUIRE(std::filesystem::exists(glb));
        seeds.push_back(Seed{"glb", ".glb", read_file(glb)});

        for (auto const &seed: seeds) {
            INFO("seed file: ", seed.name);

            auto const file = directory.path / ("fuzz" + seed.extension);
            write_file(file, seed.bytes);

            auto const control = load_isolated(file);
            REQUIRE_MESSAGE(describe(control) == describe(Outcome::loaded), seed.name, " seed does not load unmutated");

            std::mt19937_64 rng{base_seed ^ std::hash<std::string>{}(seed.name)};
            std::size_t failures = 0;

            for (std::uint64_t iteration = 0; iteration < iterations && failures < 3; ++iteration) {
                auto const mutated = mutate(seed.bytes, rng);
                write_file(file, mutated);

                auto const outcome = load_isolated(file);

                if (outcome == Outcome::crashed || outcome == Outcome::timed_out) {
                    ++failures;

                    auto const saved = std::filesystem::temp_directory_path() /
                                       std::format("lathe_fuzz_{}_{}{}", seed.name, iteration, seed.extension);
                    write_file(saved, mutated);

                    FAIL_CHECK(seed.name, " iteration ", iteration, " (LATHE_FUZZ_SEED=", base_seed, "): ",
                               describe(outcome), "; input kept at ", saved.string());
                }
            }
        }
    }
}

#endif // defined(__linux__)
