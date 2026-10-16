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

auto TextPlacement::update(Scene &scene, Renderer &renderer, debug_draw::DebugRenderer &lines, Ray const &ray,
                           glm::vec3 const &camera_forward, bool click) -> entt::entity {
    if (!active_) {
        return entt::null;
    }

    auto const hit = pick_surface(scene, renderer, ray.origin, ray.direction, PickTargets::models);

    if (!hit) {
        return entt::null;
    }

    auto const frame = text_placement::surface_frame(hit->position, hit->normal, glm::vec3{0.0F, 1.0F, 0.0F},
                                                     camera_forward, settings.lift);

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
    lines.add_line(hit->position, hit->position + (hit->normal * 0.3F), box_colour);

    if (!click) {
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
