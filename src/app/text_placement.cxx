#include "app/text_placement.hxx"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

#include "app/surface_pick.hxx"
#include "rendering/debug_renderer.hxx"
#include "rendering/entity.hxx"
#include "rendering/renderer.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

namespace text_placement {

    auto surface_frame(glm::vec3 const &position, glm::vec3 const &normal, glm::vec3 const &up,
                       glm::vec3 const &fallback_up, float lift) -> Frame {
        auto const n = glm::normalize(normal);

        // Text faces a viewer looking along its +z, so its back goes to the surface.
        auto const z = -n;

        constexpr auto min_projected = 0.2F;
        auto y = up - (n * glm::dot(up, n));

        if (glm::length(y) < min_projected) {
            y = fallback_up - (n * glm::dot(fallback_up, n));
        }

        if (glm::length(y) < 1e-4F) {
            // Both hints along the normal: any direction across the surface will do.
            y = std::abs(n.x) < 0.9F ? glm::vec3{1.0F, 0.0F, 0.0F} - (n * n.x)
                                     : glm::vec3{0.0F, 1.0F, 0.0F} - (n * n.y);
        }

        y = glm::normalize(y);
        auto const x = glm::cross(y, z);

        return {.position = position + (n * lift), .rotation = glm::normalize(glm::quat_cast(glm::mat3{x, y, z}))};
    }

    auto ScreenProjector::project(glm::vec3 const &world) const -> std::optional<glm::vec2> {
        auto const camera = glm::vec3{view * glm::vec4{world, 1.0F}};

        if (camera.z <= 1e-4F) {
            return std::nullopt;
        }

        auto const ndc = glm::vec2{camera.x * focal.x, camera.y * focal.y} / camera.z;

        return glm::vec2{origin.x + ((ndc.x + 1.0F) * 0.5F * size.x), origin.y + ((1.0F - ndc.y) * 0.5F * size.y)};
    }

    auto snap_to_triangle(std::array<glm::vec3, 3> const &triangle, ScreenProjector const &projector,
                          glm::vec2 const &cursor, float radius) -> std::optional<Snap> {
        // Ordered by preference: a vertex wins over an edge midpoint wins over the centre when within reach.
        std::array<Snap, 7> const candidates{{
                {triangle[0], SnapKind::vertex},
                {triangle[1], SnapKind::vertex},
                {triangle[2], SnapKind::vertex},
                {(triangle[0] + triangle[1]) * 0.5F, SnapKind::edge_midpoint},
                {(triangle[1] + triangle[2]) * 0.5F, SnapKind::edge_midpoint},
                {(triangle[2] + triangle[0]) * 0.5F, SnapKind::edge_midpoint},
                {(triangle[0] + triangle[1] + triangle[2]) / 3.0F, SnapKind::face_centre},
        }};

        std::optional<Snap> best;
        auto best_rank = 3;
        auto best_distance = radius;

        for (auto const &candidate: candidates) {
            auto const screen = projector.project(candidate.position);

            if (!screen) {
                continue;
            }

            auto const distance = glm::length(*screen - cursor);
            auto const rank = static_cast<int>(candidate.kind);

            if (distance > radius) {
                continue;
            }

            // Prefer the finer kind outright; within a kind, the closer one.
            if (rank < best_rank || (rank == best_rank && distance < best_distance)) {
                best = candidate;
                best_rank = rank;
                best_distance = distance;
            }
        }

        return best;
    }

    auto local_transform(glm::mat4 const &parent_world, Frame const &frame) -> Components::Transform {
        auto const world = glm::translate(glm::mat4{1.0F}, frame.position) * glm::mat4_cast(frame.rotation);
        auto const local = glm::inverse(parent_world) * world;

        glm::vec3 const scale{glm::length(glm::vec3{local[0]}), glm::length(glm::vec3{local[1]}),
                              glm::length(glm::vec3{local[2]})};
        glm::mat3 const rotation{glm::vec3{local[0]} / scale.x, glm::vec3{local[1]} / scale.y,
                                 glm::vec3{local[2]} / scale.z};

        return {.position = glm::vec3{local[3]}, .rotation = glm::normalize(glm::quat_cast(rotation)), .scale = scale};
    }

}

auto TextPlacement::update(Scene &scene, Renderer &renderer, debug_draw::DebugRenderer &lines,
                           Pointer const &pointer) -> entt::entity {
    if (!active_) {
        return entt::null;
    }

    auto const hit = pick_surface(scene, renderer, pointer.origin, pointer.direction, PickTargets::models);

    if (!hit) {
        return entt::null;
    }

    auto position = hit->position;
    std::optional<text_placement::Snap> snap;

    if (settings.snap_to_geometry && !pointer.bypass_snap && hit->triangle) {
        snap = text_placement::snap_to_triangle(*hit->triangle, pointer.projector, pointer.cursor,
                                                settings.snap_radius);

        if (snap) {
            position = snap->position;
        }
    }

    auto const frame = text_placement::surface_frame(position, hit->normal, glm::vec3{0.0F, 1.0F, 0.0F},
                                                     pointer.camera_forward, settings.lift);

    // The preview: the hit triangle, the normal, and the text's box.
    constexpr glm::vec3 triangle_colour{1.0F, 0.85F, 0.2F};
    constexpr glm::vec3 box_colour{0.3F, 0.9F, 1.0F};

    if (hit->triangle) {
        auto const &corners = *hit->triangle;
        lines.add_line(corners[0], corners[1], triangle_colour);
        lines.add_line(corners[1], corners[2], triangle_colour);
        lines.add_line(corners[2], corners[0], triangle_colour);
    }

    auto const right = frame.rotation * glm::vec3{1.0F, 0.0F, 0.0F};
    auto const up = frame.rotation * glm::vec3{0.0F, 1.0F, 0.0F};

    auto longest = std::size_t{0};
    auto line_count = std::size_t{1};
    auto current = std::size_t{0};

    for (auto const character: settings.text) {
        if (character == '\n') {
            ++line_count;
            current = 0;
        } else if ((static_cast<unsigned char>(character) & 0xC0U) != 0x80U) {
            longest = std::max(longest, ++current);
        }
    }

    // Glyph advances are not known here; about half an em per character is close enough for a preview.
    auto const half_width = 0.5F * 0.55F * settings.size * static_cast<float>(std::max<std::size_t>(longest, 1));
    auto const half_height = 0.5F * settings.size * static_cast<float>(line_count);
    auto const corner = [&](float sx, float sy) { return frame.position + (right * (sx * half_width)) + (up * (sy * half_height)); };

    lines.add_line(corner(-1, -1), corner(1, -1), box_colour);
    lines.add_line(corner(1, -1), corner(1, 1), box_colour);
    lines.add_line(corner(1, 1), corner(-1, 1), box_colour);
    lines.add_line(corner(-1, 1), corner(-1, -1), box_colour);
    lines.add_line(position, position + (hit->normal * 0.3F), box_colour);

    if (snap) {
        // A small cross in the surface plane, in the colour of what it snapped to, sized to stay the same on screen.
        auto const colour = snap->kind == text_placement::SnapKind::vertex          ? glm::vec3{0.3F, 1.0F, 0.4F}
                            : snap->kind == text_placement::SnapKind::edge_midpoint ? glm::vec3{1.0F, 0.6F, 0.2F}
                                                                                    : glm::vec3{1.0F, 0.3F, 0.9F};
        auto const arm = 0.012F * glm::length(position - pointer.origin);
        auto const across = glm::normalize(glm::cross(hit->normal, std::abs(hit->normal.y) < 0.9F ? glm::vec3{0.0F, 1.0F, 0.0F} : glm::vec3{1.0F, 0.0F, 0.0F}));
        auto const along = glm::cross(hit->normal, across);

        lines.add_line(position - (across * arm), position + (across * arm), colour);
        lines.add_line(position - (along * arm), position + (along * arm), colour);
    }

    if (!pointer.click) {
        return entt::null;
    }

    auto &registry = scene.get_registry();
    auto parent_world = glm::mat4{1.0F};

    if (auto const *parent_transform = registry.try_get<Components::Transform const>(hit->entity)) {
        parent_world = systems::get_world_transform(registry, hit->entity, *parent_transform);
    }

    auto placed = GeneratedEntity{&scene, "Text"};
    placed.emplace<Components::Transform>(text_placement::local_transform(parent_world, frame));
    placed.emplace<Components::Parent>(Components::Parent{.entity = hit->entity});
    placed.emplace<Components::Text>(Components::Text{.text = settings.text, .size = settings.size});

    return placed;
}
