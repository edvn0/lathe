#include "rendering/local_shadows.hxx"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#include <glm/gtc/matrix_transform.hpp>

namespace local_shadows {

    namespace {

        constexpr float near_plane_fraction = 0.002F;
        constexpr float minimum_near_plane = 0.05F;

        [[nodiscard]] auto cells_for(std::uint32_t resolution) noexcept -> std::uint32_t {
            return std::max(resolution / cell_size, 1U);
        }

        [[nodiscard]] auto near_plane_for(float range) noexcept -> float {
            return std::max(range * near_plane_fraction, minimum_near_plane);
        }

        // Reversed depth like every other shadow view: swapping near and far puts depth 1 at the near plane.
        [[nodiscard]] auto reversed_perspective(float tan_half_angle, float range) noexcept -> glm::mat4 {
            return glm::perspectiveLH_ZO(2.0F * std::atan(tan_half_angle), 1.0F, range, near_plane_for(range));
        }

        [[nodiscard]] auto padded_tan(float tan_half_angle, std::uint32_t resolution) noexcept -> float {
            // One texel of border per side so filtering at a tile edge samples real depth.
            auto const texels = static_cast<float>(resolution);
            return tan_half_angle * (texels + 2.0F) / texels;
        }

        struct FaceBasis {
            glm::vec3 direction;
            glm::vec3 up;
        };

        constexpr std::array<FaceBasis, point_face_count> face_bases{{
                {{1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
                {{-1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}},
                {{0.0F, 1.0F, 0.0F}, {0.0F, 0.0F, -1.0F}},
                {{0.0F, -1.0F, 0.0F}, {0.0F, 0.0F, 1.0F}},
                {{0.0F, 0.0F, 1.0F}, {0.0F, 1.0F, 0.0F}},
                {{0.0F, 0.0F, -1.0F}, {0.0F, 1.0F, 0.0F}},
        }};

    }

    auto Atlas::used_cells() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(
                std::ranges::count_if(owner_, [](std::uint64_t owner) { return owner != 0; }));
    }

    auto Atlas::find(std::uint64_t light_id) const -> Allocation const * {
        auto const it = allocations_.find(light_id);
        return it == allocations_.end() ? nullptr : &it->second;
    }

    auto Atlas::free_cells(std::uint64_t light_id) -> void {
        for (auto &owner: owner_) {
            if (owner == light_id) {
                owner = 0;
            }
        }
    }

    auto Atlas::release(std::uint64_t light_id) -> void {
        free_cells(light_id);
        allocations_.erase(light_id);
    }

    auto Atlas::clear() -> void {
        allocations_.clear();
        owner_.fill(0);
    }

    auto Atlas::place(std::uint64_t light_id, std::uint32_t tile_count, std::uint32_t resolution, Allocation &out)
            -> bool {
        auto const span = cells_for(resolution);
        out = Allocation{.tile_count = tile_count, .resolution = resolution};

        for (std::uint32_t tile = 0; tile < tile_count; ++tile) {
            bool placed = false;

            for (std::uint32_t cy = 0; cy + span <= cells_per_side && !placed; cy += span) {
                for (std::uint32_t cx = 0; cx + span <= cells_per_side && !placed; cx += span) {
                    bool free = true;
                    for (std::uint32_t dy = 0; dy < span && free; ++dy) {
                        for (std::uint32_t dx = 0; dx < span; ++dx) {
                            if (owner_[((cy + dy) * cells_per_side) + cx + dx] != 0) {
                                free = false;
                                break;
                            }
                        }
                    }

                    if (!free) {
                        continue;
                    }

                    for (std::uint32_t dy = 0; dy < span; ++dy) {
                        for (std::uint32_t dx = 0; dx < span; ++dx) {
                            owner_[((cy + dy) * cells_per_side) + cx + dx] = light_id;
                        }
                    }

                    out.tiles[tile] = Tile{.x = cx * cell_size, .y = cy * cell_size, .size = resolution};
                    placed = true;
                }
            }

            if (!placed) {
                free_cells(light_id);
                return false;
            }
        }

        return true;
    }

    auto Atlas::evict_least_recently_used() -> bool {
        auto victim = allocations_.end();

        for (auto it = allocations_.begin(); it != allocations_.end(); ++it) {
            if (it->second.last_used_frame >= frame_) {
                continue;
            }
            if (victim == allocations_.end() || it->second.last_used_frame < victim->second.last_used_frame) {
                victim = it;
            }
        }

        if (victim == allocations_.end()) {
            return false;
        }

        free_cells(victim->first);
        allocations_.erase(victim);
        return true;
    }

    auto Atlas::request(std::uint64_t light_id, std::uint32_t tile_count, std::uint32_t resolution)
            -> Allocation const * {
        if (light_id == 0 || tile_count == 0 || tile_count > point_face_count) {
            return nullptr;
        }

        resolution = std::clamp(std::bit_floor(std::max(resolution, 1U)), minimum_resolution, maximum_resolution);

        if (auto const it = allocations_.find(light_id); it != allocations_.end()) {
            if (it->second.tile_count == tile_count && it->second.resolution == resolution) {
                it->second.last_used_frame = frame_;
                return &it->second;
            }

            release(light_id);
        }

        Allocation placed;
        while (!place(light_id, tile_count, resolution, placed)) {
            if (!evict_least_recently_used()) {
                return nullptr;
            }
        }

        placed.last_used_frame = frame_;
        placed.generation = next_generation_++;

        return &(allocations_[light_id] = placed);
    }

    auto resolution_for_screen_radius(float radius_pixels) noexcept -> std::uint32_t {
        if (radius_pixels >= 320.0F) {
            return 1024;
        }
        if (radius_pixels >= 110.0F) {
            return 512;
        }
        return 256;
    }

    auto spot_texel_world(float outer_cone_radians, std::uint32_t resolution) noexcept -> float {
        auto const half_angle = std::min(outer_cone_radians, maximum_spot_half_angle_radians);
        return 2.0F * padded_tan(std::tan(half_angle), resolution) / static_cast<float>(resolution);
    }

    auto point_texel_world(std::uint32_t resolution) noexcept -> float {
        return 2.0F * padded_tan(1.0F, resolution) / static_cast<float>(resolution);
    }

    auto spot_view_projection(glm::vec3 position, glm::vec3 direction, float outer_cone_radians, float range,
                              std::uint32_t resolution) noexcept -> glm::mat4 {
        auto const forward = glm::normalize(direction);
        auto const up = std::abs(forward.y) > 0.99F ? glm::vec3{0.0F, 0.0F, 1.0F} : glm::vec3{0.0F, 1.0F, 0.0F};
        auto const half_angle = std::min(outer_cone_radians, maximum_spot_half_angle_radians);

        return reversed_perspective(padded_tan(std::tan(half_angle), resolution), range) *
               glm::lookAtLH(position, position + forward, up);
    }

    auto point_face_view_projection(glm::vec3 position, std::uint32_t face, float range,
                                    std::uint32_t resolution) noexcept -> glm::mat4 {
        auto const &basis = face_bases[std::min(face, point_face_count - 1)];

        return reversed_perspective(padded_tan(1.0F, resolution), range) *
               glm::lookAtLH(position, position + basis.direction, basis.up);
    }

    auto make_gpu_tile(glm::mat4 const &view_projection, Tile tile, float texel_world) noexcept -> GpuShadowTile {
        auto const atlas = static_cast<float>(atlas_size);

        return GpuShadowTile{
                .view_projection = view_projection,
                .atlas_rect = glm::vec4{static_cast<float>(tile.x) / atlas, static_cast<float>(tile.y) / atlas,
                                        static_cast<float>(tile.size) / atlas, static_cast<float>(tile.size) / atlas},
                .texel_world = texel_world,
                .tile_texel = 1.0F / static_cast<float>(tile.size),
        };
    }

}
