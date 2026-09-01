#include "terrain_quadtree.hxx"

#include <algorithm>
#include <cmath>

namespace {

    [[nodiscard]] auto floor_div(std::int32_t value, std::int32_t divisor) -> std::int32_t {
        // std::int32_t: sufficient range for any chunk index this engine's
        // view distances produce; a plain integer division truncates
        // toward zero, which is wrong for negative values (e.g. -1 / 2
        // should floor to -1, not truncate to 0).
        auto const quotient = value / divisor;
        auto const remainder = value % divisor;
        return (remainder != 0 && ((remainder < 0) != (divisor < 0))) ? quotient - 1 : quotient;
    }

    // Chebyshev (L-infinity) distance from `point` to the axis-aligned box
    // [min, max), 0 when inside. This is what makes residency rings uniform
    // in thickness around the camera rather than circular -- see the
    // terrain streaming plan for why L-infinity was chosen over Euclidean.
    [[nodiscard]] auto distance_to_aabb(glm::vec2 point, glm::vec2 box_min, glm::vec2 box_max) -> float {
        auto const dx = std::max({box_min.x - point.x, 0.0F, point.x - box_max.x});
        auto const dz = std::max({box_min.y - point.y, 0.0F, point.y - box_max.y});
        return std::max(dx, dz);
    }

    [[nodiscard]] auto chunk_bounds(ChunkKey const &key, TerrainLodSettings const &settings)
            -> std::pair<glm::vec2, glm::vec2> {
        auto const span = terrain_chunk_span(settings, key.lod);
        auto const min = glm::vec2{static_cast<float>(key.x) * span, static_cast<float>(key.z) * span};
        return {min, min + glm::vec2{span, span}};
    }

    auto select_node(ChunkKey const &key, glm::vec2 camera_xz, TerrainLodSettings const &settings,
                     ChunkKeySet &split_state, std::vector<ChunkKey> &out_desired) -> void {

        auto const [box_min, box_max] = chunk_bounds(key, settings);
        auto const distance = distance_to_aabb(camera_xz, box_min, box_max);

        if (distance > settings.view_distance) {
            return; // culled -- entirely outside view distance
        }

        if (key.lod == 0) {
            out_desired.push_back(key); // no finer LOD to descend into
            return;
        }

        auto const span = terrain_chunk_span(settings, key.lod);
        auto const threshold = settings.split_factor * span;
        auto const was_split = split_state.contains(key);

        bool split_now{};

        if (distance < threshold * (1.0F - settings.split_hysteresis)) {
            split_now = true;
        } else if (distance > threshold * (1.0F + settings.split_hysteresis)) {
            split_now = false;
        } else {
            split_now = was_split; // inside the deadband: keep last frame's decision
        }

        if (split_now) {
            split_state.insert(key);
        } else {
            split_state.erase(key);
        }

        if (!split_now) {
            out_desired.push_back(key);
            return;
        }

        for (auto const &child: terrain_chunk_children(key)) {
            select_node(child, camera_xz, settings, split_state, out_desired);
        }
    }

} // namespace

auto terrain_chunk_centre(ChunkKey const &key, TerrainLodSettings const &settings) -> glm::vec2 {
    auto const span = terrain_chunk_span(settings, key.lod);
    return glm::vec2{(static_cast<float>(key.x) + 0.5F) * span, (static_cast<float>(key.z) + 0.5F) * span};
}

auto terrain_chunk_children(ChunkKey const &key) -> std::array<ChunkKey, 4> {
    auto const child_lod = static_cast<std::uint8_t>(key.lod - 1);
    auto const cx = key.x * 2;
    auto const cz = key.z * 2;

    return {
            ChunkKey{.x = cx, .z = cz, .lod = child_lod},
            ChunkKey{.x = cx + 1, .z = cz, .lod = child_lod},
            ChunkKey{.x = cx, .z = cz + 1, .lod = child_lod},
            ChunkKey{.x = cx + 1, .z = cz + 1, .lod = child_lod},
    };
}

auto terrain_chunk_parent(ChunkKey const &key) -> ChunkKey {
    return ChunkKey{
            .x = floor_div(key.x, 2),
            .z = floor_div(key.z, 2),
            .lod = static_cast<std::uint8_t>(key.lod + 1),
    };
}

auto select_chunks(glm::vec2 camera_xz, TerrainLodSettings const &settings, ChunkKeySet &split_state,
                   std::vector<ChunkKey> &out_desired) -> void {

    out_desired.clear();

    if (settings.lod_levels == 0) {
        return;
    }

    auto const top_lod = static_cast<std::uint8_t>(settings.lod_levels - 1);
    auto const top_span = terrain_chunk_span(settings, top_lod);

    auto const radius_chunks = static_cast<std::int32_t>(std::ceil(settings.view_distance / top_span)) + 1;

    auto const camera_chunk_x = static_cast<std::int32_t>(std::floor(camera_xz.x / top_span));
    auto const camera_chunk_z = static_cast<std::int32_t>(std::floor(camera_xz.y / top_span));

    for (auto z = camera_chunk_z - radius_chunks; z <= camera_chunk_z + radius_chunks; ++z) {
        for (auto x = camera_chunk_x - radius_chunks; x <= camera_chunk_x + radius_chunks; ++x) {
            select_node(ChunkKey{.x = x, .z = z, .lod = top_lod}, camera_xz, settings, split_state, out_desired);
        }
    }
}
