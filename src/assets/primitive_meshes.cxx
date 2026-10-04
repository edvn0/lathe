#include "assets/primitive_meshes.hxx"

#include <glm/geometric.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <span>

namespace {

    struct CubeFace {
        glm::vec3 normal;
        glm::vec3 tangent;
    };

    // tangent x cross(normal, tangent) == normal for each face, so winding is consistent.
    constexpr std::array<CubeFace, 6> cube_faces{{
            {.normal = {1.0F, 0.0F, 0.0F}, .tangent = {0.0F, 1.0F, 0.0F}},
            {.normal = {-1.0F, 0.0F, 0.0F}, .tangent = {0.0F, 1.0F, 0.0F}},
            {.normal = {0.0F, 1.0F, 0.0F}, .tangent = {1.0F, 0.0F, 0.0F}},
            {.normal = {0.0F, -1.0F, 0.0F}, .tangent = {1.0F, 0.0F, 0.0F}},
            {.normal = {0.0F, 0.0F, 1.0F}, .tangent = {1.0F, 0.0F, 0.0F}},
            {.normal = {0.0F, 0.0F, -1.0F}, .tangent = {1.0F, 0.0F, 0.0F}},
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

namespace {

    // One grass blade's centreline and width profile, before it is cut into a particular LOD's rows.
    struct GrassBlade {
        glm::vec3 root{0.0F};
        glm::vec3 across{1.0F, 0.0F, 0.0F};
        glm::vec3 bend_direction{0.0F, 0.0F, 1.0F};
        float height = 0.7F;
        float half_width = 0.02F;
        float lean = 0.05F;
        float curve = 0.05F;
    };

    // Non-linear taper keeps the lower blade wide.
    constexpr float grass_taper_exponent = 0.72F;

    // Emits `blade` as a strip of quads through `row_heights` (fractions of its height, the first 0) closed by a
    // single tip triangle at the top, one-sided: the material is double-sided. Wider by `width_scale`, and
    // straighter by `curve_scale` for the coarse LOD, whose few blades can't show much bend.
    auto emit_grass_blade(GrassBlade const &blade, std::span<float const> row_heights, float width_scale,
                          float curve_scale, std::vector<ModelVertex> &vertices,
                          std::vector<std::uint32_t> &indices) -> void {
        auto const base_index = static_cast<std::uint32_t>(vertices.size());

        auto const centre = [&](float t) {
            auto const horizontal = blade.bend_direction * (blade.lean * t + blade.curve * curve_scale * t * t);
            return blade.root + glm::vec3{0.0F, blade.height * t, 0.0F} + horizontal;
        };

        // Derivative of centre(t); width only varies along `across`, so the taper doesn't change the normal.
        auto const normal = [&](float t) {
            auto const tangent =
                    glm::normalize(glm::vec3{0.0F, blade.height, 0.0F} +
                                   blade.bend_direction * (blade.lean + 2.0F * blade.curve * curve_scale * t));
            return glm::normalize(glm::cross(blade.across, tangent));
        };

        auto const tangent = glm::vec4{blade.across, 1.0F};

        for (auto const t: row_heights) {
            auto const half_width =
                    blade.half_width * width_scale * std::pow(std::max(0.0F, 1.0F - t), grass_taper_exponent);
            auto const row_centre = centre(t);
            auto const row_normal = normal(t);

            // UV.y runs from 0 at the root to 1 at the tip.
            vertices.push_back(ModelVertex{
                    .position = row_centre - blade.across * half_width,
                    .normal = row_normal,
                    .tangent = tangent,
                    .texcoord = glm::vec2{0.0F, t},
            });
            vertices.push_back(ModelVertex{
                    .position = row_centre + blade.across * half_width,
                    .normal = row_normal,
                    .tangent = tangent,
                    .texcoord = glm::vec2{1.0F, t},
            });
        }

        vertices.push_back(ModelVertex{
                .position = centre(1.0F),
                .normal = normal(1.0F),
                .tangent = tangent,
                .texcoord = glm::vec2{0.5F, 1.0F},
        });

        auto const row_count = static_cast<std::uint32_t>(row_heights.size());

        // Wound so the face whose normal points at the camera is the front face.
        for (std::uint32_t row = 0; row + 1U < row_count; ++row) {
            auto const lower_left = base_index + row * 2U;
            auto const lower_right = lower_left + 1U;
            auto const upper_left = lower_left + 2U;
            auto const upper_right = lower_left + 3U;

            indices.insert(indices.end(), {lower_left, lower_right, upper_right, lower_left, upper_right, upper_left});
        }

        auto const top_left = base_index + (row_count - 1U) * 2U;
        auto const tip = base_index + row_count * 2U;
        indices.insert(indices.end(), {top_left, top_left + 1U, tip});
    }

    // Three vertical cards through the clump's centre, 60 degrees apart, `half_width` either side and `height` tall.
    // UV.x runs across the card (mirrored on alternate cards, so neighbours don't repeat), UV.y from 0 at the top to
    // 1 at the root, matching make_grass_card_texture()'s rows.
    auto emit_grass_cards(float half_width, float height, std::vector<ModelVertex> &vertices,
                          std::vector<std::uint32_t> &indices) -> void {
        constexpr std::uint32_t card_count = 3;

        for (std::uint32_t card = 0; card < card_count; ++card) {
            auto const angle = static_cast<float>(card) * std::numbers::pi_v<float> / static_cast<float>(card_count);
            auto const across = glm::vec3{std::cos(angle), 0.0F, std::sin(angle)};
            auto const normal = glm::vec3{-across.z, 0.0F, across.x};
            auto const tangent = glm::vec4{across, 1.0F};

            auto const mirrored = card % 2U == 1U;
            auto const left_u = mirrored ? 1.0F : 0.0F;
            auto const right_u = 1.0F - left_u;

            auto const base_index = static_cast<std::uint32_t>(vertices.size());
            auto const left = -across * half_width;
            auto const right = across * half_width;
            auto const up = glm::vec3{0.0F, height, 0.0F};

            vertices.push_back(
                    ModelVertex{.position = left, .normal = normal, .tangent = tangent, .texcoord = {left_u, 1.0F}});
            vertices.push_back(
                    ModelVertex{.position = right, .normal = normal, .tangent = tangent, .texcoord = {right_u, 1.0F}});
            vertices.push_back(ModelVertex{
                    .position = right + up, .normal = normal, .tangent = tangent, .texcoord = {right_u, 0.0F}});
            vertices.push_back(ModelVertex{
                    .position = left + up, .normal = normal, .tangent = tangent, .texcoord = {left_u, 0.0F}});

            indices.insert(indices.end(), {base_index, base_index + 1U, base_index + 2U, base_index, base_index + 2U,
                                           base_index + 3U});
        }
    }

} // namespace

auto make_grass_clump_mesh() -> std::expected<PrimitiveMeshData, ModelLoadError> {
    constexpr auto blade_count = 12U;

    // Outer blades reach ~0.30 m from the clump origin.
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

    constexpr auto pi = std::numbers::pi_v<float>;
    constexpr auto two_pi = 2.0F * pi;

    // Golden angle spreads blades more evenly than uniform random sampling.
    constexpr auto golden_angle = pi * (3.0F - 2.2360679774997896964F);

    // LOD0: two quads and a tip triangle per blade. LOD1: one quad and a tip, on every other blade, widened to keep
    // roughly the same coverage.
    constexpr std::array near_rows{0.0F, 0.32F, 0.68F};
    constexpr std::array mid_rows{0.0F, 0.5F};
    constexpr auto mid_blade_stride = 2U;
    constexpr auto mid_width_scale = 2.0F;
    constexpr auto mid_curve_scale = 0.5F;

    // LOD2+: crossed cards spanning the blades' reach (lean included) and most of their height.
    constexpr auto card_half_width = 0.36F;
    constexpr auto card_height = 0.85F;

    // Fixed seed so the mesh is deterministic; instances add the variation.
    std::mt19937 random_engine{0x47524153U};

    std::uniform_real_distribution<float> unit_distribution{0.0F, 1.0F};
    std::uniform_real_distribution<float> signed_distribution{-1.0F, 1.0F};

    auto random_range = [&](float min_value, float max_value) {
        return std::lerp(min_value, max_value, unit_distribution(random_engine));
    };

    std::vector<GrassBlade> blades;
    blades.reserve(blade_count);

    for (std::uint32_t blade = 0; blade < blade_count; ++blade) {
        // Sunflower distribution, avoiding empty patches and clusters.
        auto const radial_fraction = (static_cast<float>(blade) + 0.35F) / static_cast<float>(blade_count);

        auto const radius = clump_radius * std::sqrt(radial_fraction);

        auto const position_angle =
                static_cast<float>(blade) * golden_angle + signed_distribution(random_engine) * 0.20F;

        // Facing independent of radial position, or the clump looks like a star.
        auto const yaw = unit_distribution(random_engine) * two_pi;

        auto const across = glm::normalize(glm::vec3{std::cos(yaw), 0.0F, std::sin(yaw)});
        auto const face_normal = glm::normalize(glm::vec3{-across.z, 0.0F, across.x});

        // Shorter blades near the edge give a rounded silhouette.
        auto const edge_factor = radius / clump_radius;

        auto height = random_range(min_height, max_height);
        height *= std::lerp(1.0F, 0.82F, edge_factor * edge_factor);

        auto const half_width = random_range(min_half_width, max_half_width);

        // Lean mostly normal to the blade plane, with some sideways variation.
        auto const bend_angle = random_range(-0.65F, 0.65F);

        auto bend_direction = face_normal * std::cos(bend_angle) + across * std::sin(bend_angle);

        if (signed_distribution(random_engine) < 0.0F) {
            bend_direction = -bend_direction;
        }

        auto const lean = random_range(min_lean, max_lean);
        auto const curve = random_range(min_curve, max_curve);

        blades.push_back(GrassBlade{
                .root = glm::vec3{std::cos(position_angle) * radius, 0.0F, std::sin(position_angle) * radius},
                .across = across,
                .bend_direction = glm::normalize(bend_direction),
                .height = height,
                .half_width = half_width,
                .lean = lean,
                .curve = curve,
        });
    }

    PrimitiveMeshData mesh;

    for (auto const &blade: blades) {
        emit_grass_blade(blade, near_rows, 1.0F, 1.0F, mesh.vertices, mesh.indices);
    }

    auto &mid_indices = mesh.lod_indices[0].emplace();
    for (std::uint32_t blade = 0; blade < blade_count; blade += mid_blade_stride) {
        emit_grass_blade(blades[blade], mid_rows, mid_width_scale, mid_curve_scale, mesh.vertices, mid_indices);
    }

    static_assert(grass_clump_card_lod == 2);
    emit_grass_cards(card_half_width, card_height, mesh.vertices, mesh.lod_indices[1].emplace());

    // LOD3 reuses the cards.
    return mesh;
}

auto to_model_cpu_data(PrimitiveMeshData mesh) -> ModelCpuData {
    ModelCpuData cpu_data;

    ModelCpuPrimitive primitive{
            .vertices = std::move(mesh.vertices),
            .indices = std::move(mesh.indices),
            .reduced_indices = std::move(mesh.lod_indices),
            .material_index = std::nullopt,
    };

    // Procedural meshes skip finalize_primitive_cpu(), so build the GPU data here.
    prepare_primitive_gpu_data(primitive);

    cpu_data.meshes.push_back(ModelCpuMesh{
            .primitives = {std::move(primitive)},
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
    rings = std::max(rings, 1U);

    constexpr float radius = 0.5F;
    constexpr float half_cylinder_height = 0.5F; // Cylinder height 1, capsule height 2.

    auto const row_stride = segments + 1;
    auto const total_rings = rings * 2 + 1;

    std::vector<ModelVertex> vertices;
    vertices.reserve(static_cast<std::size_t>(total_rings + 1) * row_stride);

    for (std::uint32_t ring = 0; ring <= total_rings; ++ring) {
        float theta = 0.0F;
        float y_offset = 0.0F;

        if (ring <= rings) {
            // Top hemisphere: theta from 0 (pole) to pi/2 (equator).
            auto const v_hemi = static_cast<float>(ring) / static_cast<float>(rings);
            theta = v_hemi * (std::numbers::pi_v<float> * 0.5F);
            y_offset = half_cylinder_height;
        } else {
            // Bottom hemisphere: theta from pi/2 (equator) to pi (pole).
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

            glm::vec3 const normal{sin_theta * std::cos(phi), cos_theta, sin_theta * std::sin(phi)};

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

auto make_ribbon_mesh(std::span<glm::vec3 const> grid, std::uint32_t columns, float uv_scale)
        -> std::expected<PrimitiveMeshData, ModelLoadError> {
    if (columns < 2U || grid.size() < static_cast<std::size_t>(columns) * 2U || grid.size() % columns != 0U) {
        return std::unexpected(ModelLoadError{.type = ModelLoadErrorType::invalid_argument});
    }

    auto const rows = static_cast<std::uint32_t>(grid.size() / columns);
    auto const at = [&](std::uint32_t row, std::uint32_t column) { return grid[row * columns + column]; };

    std::vector<ModelVertex> vertices;
    vertices.reserve(grid.size());

    float along = 0.0F;

    for (std::uint32_t row = 0; row < rows; ++row) {
        auto const centre = at(row, columns / 2U);

        if (row > 0U) {
            along += glm::distance(centre, at(row - 1U, columns / 2U));
        }

        auto const forward = glm::normalize(at(std::min(row + 1U, rows - 1U), columns / 2U) -
                                            at(row > 0U ? row - 1U : 0U, columns / 2U));
        auto const across = at(row, columns - 1U) - at(row, 0U);
        auto const normal = glm::normalize(glm::cross(across, forward));
        auto const width = glm::length(across);

        for (std::uint32_t column = 0; column < columns; ++column) {
            auto const fraction = static_cast<float>(column) / static_cast<float>(columns - 1U);

            vertices.push_back(ModelVertex{
                    .position = at(row, column),
                    .normal = normal,
                    // Placeholder; generate_tangents() overwrites it.
                    .tangent = glm::vec4{1.0F, 0.0F, 0.0F, 1.0F},
                    .texcoord = glm::vec2{fraction * width * uv_scale, along * uv_scale},
            });
        }
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(static_cast<std::size_t>(rows - 1U) * (columns - 1U) * 6U);

    for (std::uint32_t row = 0; row + 1U < rows; ++row) {
        for (std::uint32_t column = 0; column + 1U < columns; ++column) {
            auto const near_first = row * columns + column;
            auto const near_second = near_first + 1U;
            auto const far_first = near_first + columns;
            auto const far_second = far_first + 1U;

            indices.push_back(near_first);
            indices.push_back(near_second);
            indices.push_back(far_second);

            indices.push_back(near_first);
            indices.push_back(far_second);
            indices.push_back(far_first);
        }
    }

    if (auto tangents = generate_tangents(vertices, indices); !tangents) {
        return std::unexpected(tangents.error());
    }

    return PrimitiveMeshData{.vertices = std::move(vertices), .indices = std::move(indices)};
}
