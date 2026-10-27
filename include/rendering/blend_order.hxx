#pragma once

#include <array>
#include <cstdint>

// The order the blended batches are drawn in, kept free of Vulkan so it can be tested on the CPU.
//
// Blended batches are drawn back to front by the distance from the camera to the centre of their bounds. Batches whose
// centres coincide (a sphere inside a sphere) would otherwise tie and come out in whatever order the sort left them, so
// ties go to the larger bounds first (the enclosing one is drawn under what it holds), then to a stable id. A
// double-sided batch is drawn twice, back faces first and then front faces, so its far side is always under its near
// side. Water is drawn before every other blended batch: it composes the scene copy behind it itself (docs/water.md),
// so what is drawn under it first would be painted over.
namespace blend_order {

    // Mirrors blend_faces_* in assets/shaders/scene_types.slang (DrawCommand::blend_faces).
    enum class Faces : std::uint32_t {
        both = 0,
        back = 1,
        front = 2,
    };

    struct Entry {
        float camera_distance = 0.0F;
        float bounds_radius = 0.0F;
        // Identifies the batch independently of where it sits in the batch list (material, mesh, submesh, LOD).
        std::array<std::uint32_t, 4> stable_id{};
        bool water = false;
    };

    // Strict weak order: true when `lhs` is drawn before `rhs`.
    [[nodiscard]] constexpr auto draws_before(Entry const &lhs, Entry const &rhs) noexcept -> bool {
        if (lhs.water != rhs.water) {
            return lhs.water;
        }
        if (lhs.camera_distance != rhs.camera_distance) {
            return lhs.camera_distance > rhs.camera_distance;
        }
        if (lhs.bounds_radius != rhs.bounds_radius) {
            return lhs.bounds_radius > rhs.bounds_radius;
        }
        return lhs.stable_id < rhs.stable_id;
    }

    // The passes a batch is drawn in, in order: both faces at once for a single-sided batch (the fragment shader drops
    // its back faces, as culling would) and back then front for a double-sided one.
    struct Passes {
        std::array<Faces, 2> faces{};
        std::uint32_t count = 0;
    };

    [[nodiscard]] constexpr auto passes_for(bool double_sided, bool split_allowed) noexcept -> Passes {
        if (!double_sided) {
            return Passes{.faces = {Faces::front, Faces::front}, .count = 1};
        }
        if (!split_allowed) {
            return Passes{.faces = {Faces::both, Faces::both}, .count = 1};
        }
        return Passes{.faces = {Faces::back, Faces::front}, .count = 2};
    }

}
