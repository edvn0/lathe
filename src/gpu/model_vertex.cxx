#include "gpu/model_vertex.hxx"

#include <algorithm>
#include <cmath>

#include <glm/gtc/packing.hpp>
#include <glm/geometric.hpp>

#include "core/thread_pool.hxx"

namespace {
    constexpr auto pack_sign_into_snorm2x16(glm::vec2 value, bool sign_bit) -> glm::uint32 {
        auto const packed = glm::packSnorm2x16(value);
        return (packed & ~glm::uint32{1}) | (sign_bit ? 1U : 0U);
    }
} // namespace

auto encode_octahedral(glm::vec3 direction) -> glm::vec2 {
    auto const l1_norm = std::abs(direction.x) + std::abs(direction.y) + std::abs(direction.z);
    auto encoded = glm::vec2{direction.x, direction.y} / l1_norm;

    if (direction.z < 0.0F) {
        auto const wrapped = glm::vec2{1.0F} - glm::abs(glm::vec2{encoded.y, encoded.x});
        encoded = glm::vec2{
                encoded.x >= 0.0F ? wrapped.x : -wrapped.x,
                encoded.y >= 0.0F ? wrapped.y : -wrapped.y,
        };
    }

    return encoded;
}

auto decode_octahedral(glm::vec2 encoded) -> glm::vec3 {
    auto direction = glm::vec3{encoded.x, encoded.y, 1.0F - std::abs(encoded.x) - std::abs(encoded.y)};
    auto const t = std::max(-direction.z, 0.0F);

    direction.x += direction.x >= 0.0F ? -t : t;
    direction.y += direction.y >= 0.0F ? -t : t;

    return glm::normalize(direction);
}

auto compress_vertex(ModelVertex const &vertex) -> CompressedModelVertex {
    CompressedModelVertex compressed{};

    compressed.normal_oct = glm::packSnorm2x16(encode_octahedral(glm::normalize(vertex.normal)));

    auto const tangent_direction = glm::vec3{vertex.tangent};
    compressed.tangent_oct =
            pack_sign_into_snorm2x16(encode_octahedral(glm::normalize(tangent_direction)), vertex.tangent.w < 0.0F);

    compressed.position_x = glm::packHalf1x16(vertex.position.x);
    compressed.position_y = glm::packHalf1x16(vertex.position.y);
    compressed.position_z = glm::packHalf1x16(vertex.position.z);

    compressed.texcoord_u = glm::packHalf1x16(vertex.texcoord.x);
    compressed.texcoord_v = glm::packHalf1x16(vertex.texcoord.y);

    return compressed;
}

auto compress_vertices(std::span<ModelVertex const> vertices) -> std::vector<CompressedModelVertex> {
    std::vector<CompressedModelVertex> compressed(vertices.size());

    auto &pool = thread_pool();

    auto blocks = pool.submit_blocks(std::size_t{0}, vertices.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            compressed[i] = compress_vertex(vertices[i]);
        }
    });
    blocks.wait();

    return compressed;
}
