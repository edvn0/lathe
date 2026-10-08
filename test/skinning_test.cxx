#include <doctest/doctest.h>

#include "gpu/skinning.hxx"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/epsilon.hpp>

namespace {
    auto sample_vertex() -> ModelVertex {
        return {
                .position = {1.0F, 2.0F, -0.5F},
                .normal = glm::normalize(glm::vec3{0.3F, 0.8F, 0.5F}),
                .tangent = {glm::normalize(glm::vec3{0.8F, -0.3F, 0.0F}), -1.0F},
                .texcoord = {0.25F, 0.75F},
        };
    }

    auto near(glm::vec3 a, glm::vec3 b, float eps) -> bool { return glm::all(glm::epsilonEqual(a, b, eps)); }
}

TEST_CASE("compress/decompress round trip stays within tolerance") {
    auto const vertex = sample_vertex();
    auto const back = decompress_vertex(compress_vertex(vertex));
    CHECK(near(back.position, vertex.position, 1e-3F));
    CHECK(near(back.normal, vertex.normal, 1e-3F));
    CHECK(near(glm::vec3{back.tangent}, glm::vec3{vertex.tangent}, 1e-3F));
    CHECK(back.tangent.w == -1.0F);
    CHECK(near({back.texcoord, 0.0F}, {vertex.texcoord, 0.0F}, 1e-3F));
}

TEST_CASE("identity palette reproduces the rest pose") {
    std::array palette{glm::mat4{1.0F}, glm::mat4{1.0F}};
    auto const rest = sample_vertex();
    auto const out = skin_vertex(rest, pack_skin_vertex({0, 1, 0, 0}, {128, 127, 0, 0}), palette);
    CHECK(near(out.position, rest.position, 1e-5F));
    CHECK(near(out.normal, rest.normal, 1e-5F));
    CHECK(near(glm::vec3{out.tangent}, glm::vec3{rest.tangent}, 1e-5F));
}

TEST_CASE("two equal weights blend to the midpoint") {
    std::array palette{glm::translate(glm::mat4{1.0F}, {2.0F, 0.0F, 0.0F}),
                       glm::translate(glm::mat4{1.0F}, {0.0F, 4.0F, 0.0F})};
    auto const rest = sample_vertex();
    auto const out = skin_vertex(rest, pack_skin_vertex({0, 1, 0, 0}, {100, 100, 0, 0}), palette);
    CHECK(near(out.position, rest.position + glm::vec3{1.0F, 2.0F, 0.0F}, 1e-5F));
}

TEST_CASE("weights not summing to one are renormalised") {
    std::array palette{glm::translate(glm::mat4{1.0F}, {2.0F, 0.0F, 0.0F}),
                       glm::translate(glm::mat4{1.0F}, {0.0F, 4.0F, 0.0F})};
    auto const rest = sample_vertex();
    auto const half = skin_vertex(rest, pack_skin_vertex({0, 1, 0, 0}, {10, 10, 0, 0}), palette);
    CHECK(near(half.position, rest.position + glm::vec3{1.0F, 2.0F, 0.0F}, 1e-5F));

    auto const single = skin_vertex(rest, pack_skin_vertex({1, 0, 0, 0}, {50, 0, 0, 0}), palette);
    CHECK(near(single.position, rest.position + glm::vec3{0.0F, 4.0F, 0.0F}, 1e-5F));

    auto const none = skin_vertex(rest, pack_skin_vertex({1, 0, 0, 0}, {0, 0, 0, 0}), palette);
    CHECK(near(none.position, rest.position, 1e-5F));
}

TEST_CASE("normals use the inverse-transpose under non-uniform scale") {
    std::array palette{glm::scale(glm::mat4{1.0F}, {4.0F, 1.0F, 1.0F})};
    ModelVertex rest = sample_vertex();
    rest.normal = glm::normalize(glm::vec3{1.0F, 1.0F, 0.0F});
    rest.tangent = {glm::normalize(glm::vec3{1.0F, -1.0F, 0.0F}), 1.0F};
    auto const out = skin_vertex(rest, pack_skin_vertex({0, 0, 0, 0}, {255, 0, 0, 0}), palette);

    CHECK(near(out.normal, glm::normalize(glm::vec3{0.25F, 1.0F, 0.0F}), 1e-5F));
    CHECK(glm::dot(out.normal, glm::vec3{out.tangent}) == doctest::Approx(0.0F).epsilon(1e-5).scale(1.0));
}

TEST_CASE("skin chunk table is the exclusive prefix sum of per-job chunks") {
    std::array<GpuSkinJob, 3> jobs{};
    jobs[0].vertex_count = 1;
    jobs[1].vertex_count = 0;
    jobs[2].vertex_count = skin_chunk_size + 1;
    CHECK(skin_chunk_table(jobs) == std::vector<std::uint32_t>{0, 1, 1, 3});
}

TEST_CASE("pack_gpu_skin_vertex requantises weights to sum to 255") {
    SkinVertex const vertex{.joints = {3, 7, 0, 255}, .weights = {20000, 20000, 20000, 5535}};
    auto const packed = pack_gpu_skin_vertex(vertex);
    REQUIRE(packed.has_value());
    CHECK(packed->joints == 0xFF000703U);
    std::uint32_t sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += (packed->weights >> (8 * i)) & 0xFFU;
    }
    CHECK(sum == 255);

    CHECK_FALSE(pack_gpu_skin_vertex({.joints = {256, 0, 0, 0}, .weights = {65535, 0, 0, 0}}).has_value());
}
