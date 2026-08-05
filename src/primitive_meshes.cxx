#include "primitive_meshes.hxx"

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <numbers>

namespace {

    struct CubeFace {
        glm::vec3 normal;
        glm::vec3 tangent;
    };

    // tangent x bitangent(=cross(normal, tangent)) == normal for each face,
    // which keeps winding (and outward-facing culling) consistent across faces.
    constexpr std::array<CubeFace, 6> cube_faces{{
            {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
            {{-1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
            {{0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
            {{0.0F, -1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
            {{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},
            {{0.0F, 0.0F, -1.0F}, {1.0F, 0.0F, 0.0F}},
    }};

    constexpr std::array<glm::vec2, 4> corner_signs{{{-0.5F, -0.5F}, {0.5F, -0.5F}, {0.5F, 0.5F}, {-0.5F, 0.5F}}};
    constexpr std::array<glm::vec2, 4> corner_uvs{{{0.0F, 0.0F}, {1.0F, 0.0F}, {1.0F, 1.0F}, {0.0F, 1.0F}}};

} // namespace

auto make_cube_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError> {
    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;

    vertices.reserve(cube_faces.size() * 4);
    indices.reserve(cube_faces.size() * 6);

    for (auto const &face: cube_faces) {
        auto const bitangent = glm::cross(face.normal, face.tangent);
        auto const base_index = static_cast<std::uint32_t>(vertices.size());

        for (std::size_t corner = 0; corner < corner_signs.size(); ++corner) {
            auto const position = face.normal * 0.5F + face.tangent * corner_signs[corner].x +
                                  bitangent * corner_signs[corner].y;

            vertices.push_back(ModelVertex{
                    .position = position,
                    .normal = face.normal,
                    .tangent = glm::vec4{face.tangent, 1.0F},
                    .texcoord = corner_uvs[corner],
            });
        }

        indices.push_back(base_index + 0);
        indices.push_back(base_index + 1);
        indices.push_back(base_index + 2);
        indices.push_back(base_index + 0);
        indices.push_back(base_index + 2);
        indices.push_back(base_index + 3);
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return PrimitiveMeshData{.vertices = std::move(vertices), .indices = std::move(indices)};
}

auto make_sphere_mesh(std::uint32_t rings, std::uint32_t segments) -> std::expected<PrimitiveMeshData, ModelLoadError> {
    rings = std::max(rings, 2U);
    segments = std::max(segments, 3U);

    auto const row_stride = segments + 1;

    std::vector<ModelVertex> vertices;
    vertices.reserve(static_cast<std::size_t>(rings + 1) * row_stride);

    for (std::uint32_t ring = 0; ring <= rings; ++ring) {
        auto const v = static_cast<float>(ring) / static_cast<float>(rings);
        auto const theta = v * std::numbers::pi_v<float>;
        auto const sin_theta = std::sin(theta);
        auto const cos_theta = std::cos(theta);

        for (std::uint32_t segment = 0; segment <= segments; ++segment) {
            auto const u = static_cast<float>(segment) / static_cast<float>(segments);
            auto const phi = u * 2.0F * std::numbers::pi_v<float>;

            glm::vec3 const normal{sin_theta * std::cos(phi), cos_theta, sin_theta * std::sin(phi)};

            vertices.push_back(ModelVertex{
                    .position = normal * 0.5F,
                    .normal = normal,
                    .tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
                    .texcoord = glm::vec2{u, v},
            });
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(rings) * segments * 6);

    for (std::uint32_t ring = 0; ring < rings; ++ring) {
        for (std::uint32_t segment = 0; segment < segments; ++segment) {
            auto const a = ring * row_stride + segment;
            auto const b = a + row_stride;

            indices.push_back(a);
            indices.push_back(a + 1);
            indices.push_back(b + 1);

            indices.push_back(a);
            indices.push_back(b + 1);
            indices.push_back(b);
        }
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return PrimitiveMeshData{.vertices = std::move(vertices), .indices = std::move(indices)};
}

auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError> {
    constexpr auto blade_count = 3U;
    constexpr auto half_width = 0.22F;
    constexpr auto tip_half_width = 0.03F;

    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;

    vertices.reserve(blade_count * 2ULL * 4ULL);
    indices.reserve(blade_count * 2ULL * 6ULL);

    for (std::uint32_t blade = 0; blade < blade_count; ++blade) {
        auto const angle =
                static_cast<float>(blade) * std::numbers::pi_v<float> / static_cast<float>(blade_count);

        auto const across = glm::vec3{std::cos(angle), 0.0F, std::sin(angle)};
        auto const face_normal = glm::vec3{-std::sin(angle), 0.0F, std::cos(angle)};

        std::array<glm::vec3, 4> const corners{{
                across * -half_width,
                across * half_width,
                across * tip_half_width + glm::vec3{0.0F, 1.0F, 0.0F},
                across * -tip_half_width + glm::vec3{0.0F, 1.0F, 0.0F},
        }};

        // Emit the quad with both winding orders (front normal, then back
        // normal with reversed index order) so the blade stays lit and
        // visible under back-face culling regardless of which side the
        // camera ends up on.
        for (auto const winding_forward: {true, false}) {
            auto const normal = winding_forward ? face_normal : -face_normal;
            auto const tangent = glm::vec4{across, 1.0F};
            auto const base_index = static_cast<std::uint32_t>(vertices.size());

            for (std::size_t corner = 0; corner < corners.size(); ++corner) {
                vertices.push_back(ModelVertex{
                        .position = corners[corner],
                        .normal = normal,
                        .tangent = tangent,
                        .texcoord = glm::vec2{corner == 0 || corner == 3 ? 0.0F : 1.0F, corner < 2 ? 1.0F : 0.0F},
                });
            }

            if (winding_forward) {
                indices.push_back(base_index + 0);
                indices.push_back(base_index + 1);
                indices.push_back(base_index + 2);
                indices.push_back(base_index + 0);
                indices.push_back(base_index + 2);
                indices.push_back(base_index + 3);
            } else {
                indices.push_back(base_index + 2);
                indices.push_back(base_index + 1);
                indices.push_back(base_index + 0);
                indices.push_back(base_index + 3);
                indices.push_back(base_index + 2);
                indices.push_back(base_index + 0);
            }
        }
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return PrimitiveMeshData{.vertices = std::move(vertices), .indices = std::move(indices)};
}
