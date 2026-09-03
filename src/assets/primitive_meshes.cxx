#include "assets/primitive_meshes.hxx"

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <random>

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
            auto const position =
                    face.normal * 0.5F + face.tangent * corner_signs[corner].x + bitangent * corner_signs[corner].y;

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
    constexpr auto blade_count = 12U;
    constexpr auto row_count = 4U;

    // Roughly matches your current ~0.8 m spacing between clump instances.
    // The outer blades reach ~0.30 m from the clump origin.
    constexpr auto clump_radius = 0.30F;

    constexpr auto min_height = 0.48F;
    constexpr auto max_height = 0.90F;

    // Full blade width is twice these values.
    constexpr auto min_half_width = 0.012F;
    constexpr auto max_half_width = 0.028F;

    // Horizontal permanent curvature before wind deformation.
    constexpr auto min_lean = 0.015F;
    constexpr auto max_lean = 0.090F;

    constexpr auto min_curve = 0.015F;
    constexpr auto max_curve = 0.075F;

    // Tiny non-zero tip avoids degenerate triangles and works nicely with
    // tangent generation.
    constexpr auto tip_width_factor = 0.035F;

    constexpr auto pi = std::numbers::pi_v<float>;
    constexpr auto two_pi = 2.0F * pi;

    // Golden angle gives much better coverage than independently sampling
    // every blade position from a uniform random distribution.
    constexpr auto golden_angle = pi * (3.0F - 2.2360679774997896964F);

    constexpr std::array<float, row_count> row_heights{
            0.0F,
            0.32F,
            0.68F,
            1.0F,
    };

    std::vector<ModelVertex> vertices;
    std::vector<std::uint32_t> indices;

    // 4 rows * 2 vertices, duplicated for front/back.
    constexpr auto vertices_per_side = row_count * 2U;
    constexpr auto vertices_per_blade = vertices_per_side * 2U;

    // Three rectangular sections = 6 triangles per side.
    constexpr auto triangles_per_side = (row_count - 1U) * 2U;
    constexpr auto triangles_per_blade = triangles_per_side * 2U;

    vertices.reserve(static_cast<std::size_t>(blade_count) * vertices_per_blade);

    indices.reserve(static_cast<std::size_t>(blade_count) * triangles_per_blade * 3U);

    // Fixed seed: this function builds a reusable mesh, so the mesh should be
    // deterministic. Instance rotation/scale/placement can provide the
    // large-scale variation.
    std::mt19937 random_engine{0x47524153U}; // "GRAS"

    std::uniform_real_distribution<float> unit_distribution{0.0F, 1.0F};
    std::uniform_real_distribution<float> signed_distribution{-1.0F, 1.0F};

    auto random_range = [&](float min_value, float max_value) {
        return std::lerp(min_value, max_value, unit_distribution(random_engine));
    };

    auto emit_blade_side =
            [&](std::array<glm::vec3, row_count> const &centres, std::array<float, row_count> const &half_widths,
                std::array<glm::vec3, row_count> const &normals, glm::vec3 const &across, bool front_face) {
                auto const base_index = static_cast<std::uint32_t>(vertices.size());

                for (std::uint32_t row = 0; row < row_count; ++row) {
                    auto const t = row_heights[row];
                    auto const normal = front_face ? normals[row] : -normals[row];

                    // Keep UV.y semantically useful:
                    //
                    //   root -> 0
                    //   tip  -> 1
                    //
                    // This can later directly drive wind, colour gradients,
                    // translucency, etc.
                    vertices.push_back(ModelVertex{
                            .position = centres[row] - across * half_widths[row],
                            .normal = normal,
                            .tangent = glm::vec4{across, 1.0F},
                            .texcoord = glm::vec2{0.0F, t},
                    });

                    vertices.push_back(ModelVertex{
                            .position = centres[row] + across * half_widths[row],
                            .normal = normal,
                            .tangent = glm::vec4{across, 1.0F},
                            .texcoord = glm::vec2{1.0F, t},
                    });
                }

                for (std::uint32_t row = 0; row + 1U < row_count; ++row) {
                    auto const lower_left = base_index + row * 2U + 0U;
                    auto const lower_right = base_index + row * 2U + 1U;
                    auto const upper_left = base_index + (row + 1U) * 2U + 0U;
                    auto const upper_right = base_index + (row + 1U) * 2U + 1U;

                    if (front_face) {
                        indices.push_back(lower_left);
                        indices.push_back(lower_right);
                        indices.push_back(upper_right);

                        indices.push_back(lower_left);
                        indices.push_back(upper_right);
                        indices.push_back(upper_left);
                    } else {
                        indices.push_back(upper_right);
                        indices.push_back(lower_right);
                        indices.push_back(lower_left);

                        indices.push_back(upper_left);
                        indices.push_back(upper_right);
                        indices.push_back(lower_left);
                    }
                }
            };

    for (std::uint32_t blade = 0; blade < blade_count; ++blade) {
        //
        // Position blades using a sunflower/golden-angle distribution.
        //
        // Compared with:
        //
        //     x = random(...)
        //     z = random(...)
        //
        // this avoids accidentally creating empty patches or dense clusters
        // inside this very small reusable mesh.
        //
        auto const radial_fraction = (static_cast<float>(blade) + 0.35F) / static_cast<float>(blade_count);

        auto const radius = clump_radius * std::sqrt(radial_fraction);

        auto const position_angle =
                static_cast<float>(blade) * golden_angle + signed_distribution(random_engine) * 0.20F;

        auto const root_position = glm::vec3{
                std::cos(position_angle) * radius,
                0.0F,
                std::sin(position_angle) * radius,
        };

        //
        // Don't correlate blade facing with its radial position. If we did,
        // the clump would acquire an obvious flower/star appearance.
        //
        auto const yaw = unit_distribution(random_engine) * two_pi;

        auto const across = glm::normalize(glm::vec3{
                std::cos(yaw),
                0.0F,
                std::sin(yaw),
        });

        auto const face_normal = glm::normalize(glm::vec3{
                -across.z,
                0.0F,
                across.x,
        });

        //
        // Slightly shorter blades near the edge help give the clump a
        // natural bunch shape rather than a cylindrical silhouette.
        //
        auto const edge_factor = radius / clump_radius;

        auto height = random_range(min_height, max_height);

        height *= std::lerp(1.0F, 0.82F, edge_factor * edge_factor);

        auto const half_width = random_range(min_half_width, max_half_width);

        //
        // Lean mostly normal to the ribbon plane, with some sideways
        // variation. This creates curved silhouettes without making every
        // blade bend in exactly the same direction.
        //
        auto const bend_angle = random_range(-0.65F, 0.65F);

        auto bend_direction = face_normal * std::cos(bend_angle) + across * std::sin(bend_angle);

        if (signed_distribution(random_engine) < 0.0F) {
            bend_direction = -bend_direction;
        }

        bend_direction = glm::normalize(bend_direction);

        auto const lean_amount = random_range(min_lean, max_lean);

        auto const curve_amount = random_range(min_curve, max_curve);

        std::array<glm::vec3, row_count> centres{};
        std::array<float, row_count> half_widths{};
        std::array<glm::vec3, row_count> normals{};

        for (std::uint32_t row = 0; row < row_count; ++row) {
            auto const t = row_heights[row];

            //
            // Centerline:
            //
            // linear term      -> general lean
            // quadratic term   -> visible curvature towards tip
            //
            auto const horizontal_offset = bend_direction * (lean_amount * t + curve_amount * t * t);

            centres[row] = root_position + glm::vec3{0.0F, height * t, 0.0F} + horizontal_offset;

            //
            // Taper non-linearly. Keeping some width through the lower
            // two-thirds reads better than linearly shrinking the entire
            // blade.
            //
            auto const taper = std::pow(std::max(0.0F, 1.0F - t), 0.72F);

            half_widths[row] = half_width * std::lerp(tip_width_factor, 1.0F, taper);

            //
            // Derivative of the blade centerline:
            //
            // center(t) =
            //     y * t +
            //     dir * (lean*t + curve*t^2)
            //
            auto const centerline_tangent = glm::normalize(glm::vec3{0.0F, height, 0.0F} +
                                                           bend_direction * (lean_amount + 2.0F * curve_amount * t));

            //
            // across x vertical_tangent gives our front-facing geometric
            // normal. Because width changes only along `across`, taper does
            // not alter this surface normal.
            //
            normals[row] = glm::normalize(glm::cross(across, centerline_tangent));
        }

        emit_blade_side(centres, half_widths, normals, across, true);

        emit_blade_side(centres, half_widths, normals, across, false);
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return PrimitiveMeshData{
            .vertices = std::move(vertices),
            .indices = std::move(indices),
    };
}

auto to_model_cpu_data(PrimitiveMeshData mesh) -> ModelCpuData {
    ModelCpuData cpu_data;

    cpu_data.meshes.push_back(ModelCpuMesh{
            .primitives = {ModelCpuPrimitive{
                    .vertices = std::move(mesh.vertices),
                    .indices = std::move(mesh.indices),
                    .material_index = std::nullopt,
            }},
    });

    cpu_data.nodes.push_back(ModelNode{
            .local_transform = glm::mat4{1.0F},
            .mesh_index = 0,
    });

    cpu_data.scene_roots.push_back(0);

    return cpu_data;
}

auto make_capsule_mesh(std::uint32_t segments, std::uint32_t rings)
        -> std::expected<PrimitiveMeshData, ModelLoadError> {
    segments = std::max(segments, 3U);
    rings = std::max(rings, 1U); // Rings per hemisphere

    constexpr float radius = 0.5F;
    constexpr float half_cylinder_height = 0.5F; // Total cylinder height = 1.0F, total capsule height = 2.0F

    auto const row_stride = segments + 1;
    // Total rings = top hemisphere (rings + 1) + bottom hemisphere (rings + 1)
    auto const total_rings = rings * 2 + 1;

    std::vector<ModelVertex> vertices;
    vertices.reserve(static_cast<std::size_t>(total_rings + 1) * row_stride);

    for (std::uint32_t ring = 0; ring <= total_rings; ++ring) {
        float theta = 0.0F;
        float y_offset = 0.0F;

        if (ring <= rings) {
            // Top hemisphere: theta ranges from 0 (top pole) to PI/2 (equator)
            auto const v_hemi = static_cast<float>(ring) / static_cast<float>(rings);
            theta = v_hemi * (std::numbers::pi_v<float> * 0.5F);
            y_offset = half_cylinder_height;
        } else {
            // Bottom hemisphere: theta ranges from PI/2 (equator) to PI (bottom pole)
            // Subtract (rings + 1) to start v_hemi correctly at 0.0F
            auto const v_hemi = static_cast<float>(ring - (rings + 1)) / static_cast<float>(rings);
            theta = (std::numbers::pi_v<float> * 0.5F) + v_hemi * (std::numbers::pi_v<float> * 0.5F);
            y_offset = -half_cylinder_height;
        }

        auto const v = static_cast<float>(ring) / static_cast<float>(total_rings);
        auto const sin_theta = std::sin(theta);
        auto const cos_theta = std::cos(theta);

        for (std::uint32_t segment = 0; segment <= segments; ++segment) {
            auto const u = static_cast<float>(segment) / static_cast<float>(segments);
            auto const phi = u * 2.0F * std::numbers::pi_v<float>;

            // Sphere normal at theta/phi
            glm::vec3 const normal{sin_theta * std::cos(phi), cos_theta, sin_theta * std::sin(phi)};

            // Capsule position = normal * radius + cylindrical height offset
            glm::vec3 const position = normal * radius + glm::vec3{0.0F, y_offset, 0.0F};

            vertices.push_back(ModelVertex{
                    .position = position,
                    .normal = normal,
                    .tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
                    .texcoord = glm::vec2{u, v},
            });
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(total_rings) * segments * 6);

    for (std::uint32_t ring = 0; ring < total_rings; ++ring) {
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
