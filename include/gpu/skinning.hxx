#pragma once

#include "assets/model_skin.hxx"
#include "gpu/model_vertex.hxx"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <numeric>
#include <optional>

#include <glm/geometric.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/packing.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

// CPU reference for assets/shaders/skin.slang, plus the host-side layouts the shader reads. The shader must stay
// bit-for-bit comparable to skin_vertex() within the compression tolerance.

// Per-vertex skinning influences, parallel to the rest vertices (the GPU form of SkinVertex in assets/model_skin.hxx).
// Mirrors GpuSkinVertex in skin.slang. joints: 4 x u8 (byte i = influence i); weights: 4 x unorm8, same packing.
// Skinning renormalises the weights, so they need not sum to 255; unused influences are weight 0.
struct GpuSkinVertex {
    std::uint32_t joints{};
    std::uint32_t weights{};
};
static_assert(sizeof(GpuSkinVertex) == 8);

// One mesh instance to skin. Mirrors GpuSkinJob in skin.slang. The addresses point at CompressedModelVertex arrays
// (rest, out) and a GpuSkinVertex array (skin) of vertex_count entries. palette[palette_offset + joint] is
// joint_world * inverse_bind, in the same space as the rest vertices.
struct GpuSkinJob {
    std::uint64_t rest_vertex_addr{};
    std::uint64_t skin_addr{};
    std::uint64_t out_addr{};
    std::uint32_t vertex_count{};
    std::uint32_t palette_offset{};
};
static_assert(sizeof(GpuSkinJob) == 32);

// Dispatch scheme: one workgroup of skin_chunk_size threads per chunk of one job's vertices; dispatch.x * dispatch.y
// covers `total` chunks with x <= skin_dispatch_width. job_first_chunk[j] is the exclusive prefix sum of the jobs'
// chunk counts (job_count + 1 entries); the shader binary-searches it for its group's job.
constexpr std::uint32_t skin_chunk_size = 64;
constexpr std::uint32_t skin_dispatch_width = 65535;

[[nodiscard]] inline auto skin_chunk_table(std::span<GpuSkinJob const> jobs) -> std::vector<std::uint32_t> {
    std::vector<std::uint32_t> table;
    table.reserve(jobs.size() + 1);
    std::uint32_t total = 0;
    for (auto const &job: jobs) {
        table.push_back(total);
        total += (job.vertex_count + skin_chunk_size - 1) / skin_chunk_size;
    }
    table.push_back(total);
    return table;
}

[[nodiscard]] inline auto pack_skin_vertex(std::array<std::uint8_t, 4> joints, std::array<std::uint8_t, 4> weights)
        -> GpuSkinVertex {
    GpuSkinVertex packed;
    for (std::size_t i = 0; i < 4; ++i) {
        packed.joints |= std::uint32_t{joints[i]} << (8 * i);
        packed.weights |= std::uint32_t{weights[i]} << (8 * i);
    }
    return packed;
}

// Importer form (u16 joints, unorm16 weights summing to 65535) to the GPU form. nullopt if a joint index needs more
// than 8 bits. Weights are requantised with largest-remainder rounding so they sum to exactly 255 (all-zero stays zero).
[[nodiscard]] inline auto pack_gpu_skin_vertex(SkinVertex const &vertex) -> std::optional<GpuSkinVertex> {
    std::array<std::uint8_t, 4> joints{};
    std::array<std::uint8_t, 4> weights{};
    std::array<std::uint32_t, 4> remainder{};
    std::uint32_t sum16 = 0;
    std::uint32_t sum8 = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        if (vertex.joints[i] > 255) {
            return std::nullopt;
        }
        joints[i] = static_cast<std::uint8_t>(vertex.joints[i]);
        sum16 += vertex.weights[i];
        auto const scaled = std::uint32_t{vertex.weights[i]} * 255U;
        weights[i] = static_cast<std::uint8_t>(scaled / 65535U);
        remainder[i] = scaled % 65535U;
        sum8 += weights[i];
    }
    if (sum16 == 0) {
        return pack_skin_vertex(joints, weights);
    }
    while (sum8 < 255) {
        auto const best = static_cast<std::size_t>(std::ranges::max_element(remainder) - remainder.begin());
        ++weights[best];
        remainder[best] = 0;
        ++sum8;
    }
    return pack_skin_vertex(joints, weights);
}

[[nodiscard]] inline auto decompress_vertex(CompressedModelVertex const &vertex) -> ModelVertex {
    ModelVertex result;
    result.position = {glm::unpackHalf1x16(vertex.position_x), glm::unpackHalf1x16(vertex.position_y),
                       glm::unpackHalf1x16(vertex.position_z)};
    result.texcoord = {glm::unpackHalf1x16(vertex.texcoord_u), glm::unpackHalf1x16(vertex.texcoord_v)};
    result.normal = decode_octahedral(glm::unpackSnorm2x16(vertex.normal_oct));
    auto const tangent = decode_octahedral(glm::unpackSnorm2x16(vertex.tangent_oct & ~glm::uint32{1}));
    result.tangent = glm::vec4{tangent, (vertex.tangent_oct & 1U) != 0 ? -1.0F : 1.0F};
    return result;
}

// Skins one vertex. Normal and tangent use the blended matrix's inverse-transpose and plain linear part
// respectively, so non-uniform scale keeps the normal perpendicular to the surface.
[[nodiscard]] inline auto skin_vertex(ModelVertex const &rest, GpuSkinVertex const &skin,
                                      std::span<glm::mat4 const> palette) -> ModelVertex {
    std::array<float, 4> weights{};
    float total = 0.0F;
    for (std::size_t i = 0; i < 4; ++i) {
        weights[i] = static_cast<float>((skin.weights >> (8 * i)) & 0xFFU) / 255.0F;
        total += weights[i];
    }

    // No influence at all: leave the vertex at rest rather than collapsing it to the origin.
    glm::mat4 blended{1.0F};
    if (total > 0.0F) {
        blended = glm::mat4{0.0F};
        for (std::size_t i = 0; i < 4; ++i) {
            blended += (weights[i] / total) * palette[(skin.joints >> (8 * i)) & 0xFFU];
        }
    }

    glm::mat3 const linear{blended};
    // Cofactor matrix == det * inverse-transpose; only its direction matters, plus det's sign for mirrored poses.
    glm::mat3 const cofactor{glm::cross(linear[1], linear[2]), glm::cross(linear[2], linear[0]),
                             glm::cross(linear[0], linear[1])};
    auto const determinant = glm::dot(linear[0], cofactor[0]);
    bool const invertible = std::abs(determinant) > 1e-12F;

    ModelVertex result = rest;
    result.position = glm::vec3{blended * glm::vec4{rest.position, 1.0F}};
    auto const normal = invertible ? cofactor * rest.normal * (determinant < 0.0F ? -1.0F : 1.0F) : linear * rest.normal;
    result.normal = glm::normalize(normal);
    result.tangent = glm::vec4{glm::normalize(linear * glm::vec3{rest.tangent}), rest.tangent.w};
    return result;
}

[[nodiscard]] inline auto skin_compressed_vertex(CompressedModelVertex const &rest, GpuSkinVertex const &skin,
                                                 std::span<glm::mat4 const> palette) -> CompressedModelVertex {
    return compress_vertex(skin_vertex(decompress_vertex(rest), skin, palette));
}
