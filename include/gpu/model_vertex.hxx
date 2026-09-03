#pragma once

#include <volk.h>

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// GPU vertex formats and the pipeline vertex-input layout that matches
// them. Split out of load_model.hxx (glTF CPU-side loading) since
// Pipeline::create_graphics' default vertex-input state (see
// default_vertex_description() below) is a gpu-layer concern independent
// of how a ModelVertex gets populated.
struct ModelVertex {
    glm::vec3 position{};
    glm::vec3 normal{};
    glm::vec4 tangent{};
    glm::vec2 texcoord{};
};

constexpr auto default_vertex_description() {
    std::array<VkVertexInputAttributeDescription, 4> attributes{};
    attributes[0] = {
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(ModelVertex, position),
    };
    attributes[1] = {
            .location = 1,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32_SFLOAT,
            .offset = offsetof(ModelVertex, normal),
    };
    attributes[2] = {
            .location = 2,
            .binding = 0,
            .format = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset = offsetof(ModelVertex, tangent),
    };
    attributes[3] = {
            .location = 3,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = offsetof(ModelVertex, texcoord),
    };
    std::array<VkVertexInputBindingDescription, 1> bindings{};
    bindings[0] = {
            .binding = 0,
            .stride = sizeof(ModelVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };
    return std::pair{attributes, bindings};
}

auto encode_octahedral(glm::vec3 direction) -> glm::vec2;
auto decode_octahedral(glm::vec2 encoded) -> glm::vec3;

#pragma pack(push, 1)
struct CompressedModelVertex {
    glm::uint32 normal_oct{}; // packSnorm2x16(encode_octahedral(normal))
    glm::uint32 tangent_oct{}; // packSnorm2x16(encode_octahedral(tangent.xyz));
                               // LSB of the packed value doubles as the
                               // handedness sign (tangent.w < 0 ? 1 : 0)
    glm::uint16 position_x{}, position_y{}, position_z{}; // half floats
    glm::uint16 texcoord_u{}, texcoord_v{}; // half floats
    glm::uint16 pad_{};
};
#pragma pack(pop)
static_assert(sizeof(CompressedModelVertex) == 20);

auto compress_vertex(ModelVertex const &vertex) -> CompressedModelVertex;
auto compress_vertices(std::span<ModelVertex const> vertices) -> std::vector<CompressedModelVertex>;
