#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>

#include <glm/glm.hpp>

// Shadow maps for point and spot lights. Every shadowed light owns square tiles of one shared depth atlas: a spot
// light one tile, a point light six (one per cube face). Tiles are power-of-two multiples of a 256-texel cell.
namespace local_shadows {

    inline constexpr std::uint32_t atlas_size = 4096;
    inline constexpr std::uint32_t cell_size = 256;
    inline constexpr std::uint32_t cells_per_side = atlas_size / cell_size;
    inline constexpr std::uint32_t maximum_tiles = cells_per_side * cells_per_side;

    inline constexpr std::uint32_t point_face_count = 6;
    inline constexpr std::uint32_t minimum_resolution = cell_size;
    inline constexpr std::uint32_t maximum_resolution = 1024;

    inline constexpr std::uint32_t no_tile = 0xFFFFFFFFU;

    // Spot cones wider than this are shadowed only inside it (the perspective projection degenerates towards 180).
    inline constexpr float maximum_spot_half_angle_radians = 1.3962634F;

    // Mirrors GpuShadowTile in assets/shaders/scene_types.slang.
    struct alignas(16) GpuShadowTile {
        glm::mat4 view_projection{1.0F};
        // Atlas UV of the tile: xy offset, zw size.
        glm::vec4 atlas_rect{0.0F};
        // World size of one texel at unit distance from the light.
        float texel_world = 0.0F;
        // One texel in tile UV space.
        float tile_texel = 0.0F;
        float depth_bias = 0.0F;
        float _pad = 0.0F;
    };

    static_assert(sizeof(GpuShadowTile) == 96);

    struct Tile {
        std::uint32_t x = 0;
        std::uint32_t y = 0;
        std::uint32_t size = 0;
    };

    struct Allocation {
        std::array<Tile, point_face_count> tiles{};
        std::uint32_t tile_count = 0;
        std::uint32_t resolution = 0;
        std::uint64_t last_used_frame = 0;
        // Bumped whenever the light's tiles move or change size, so cached contents can be told apart.
        std::uint64_t generation = 0;
    };

    // Persistent tile allocator. Lights keep their tiles from frame to frame while they are requested, and an
    // unrequested light's tiles stay put (so its cached depth survives a brief absence) until the space is needed.
    class Atlas {
    public:
        auto begin_frame() noexcept -> void { ++frame_; }

        // Idempotent within a frame. Callers request in priority order; a request evicts only lights that were not
        // requested this frame. Returns nullptr when the atlas has no room left. `light_id` must not be zero.
        [[nodiscard]]
        auto request(std::uint64_t light_id, std::uint32_t tile_count, std::uint32_t resolution) -> Allocation const *;

        auto release(std::uint64_t light_id) -> void;
        auto clear() -> void;

        [[nodiscard]] auto find(std::uint64_t light_id) const -> Allocation const *;
        [[nodiscard]] auto allocation_count() const noexcept -> std::size_t { return allocations_.size(); }
        [[nodiscard]] auto used_cells() const noexcept -> std::uint32_t;
        [[nodiscard]] auto frame() const noexcept -> std::uint64_t { return frame_; }

    private:
        [[nodiscard]] auto place(std::uint64_t light_id, std::uint32_t tile_count, std::uint32_t resolution,
                                 Allocation &out) -> bool;
        auto free_cells(std::uint64_t light_id) -> void;
        [[nodiscard]] auto evict_least_recently_used() -> bool;

        std::unordered_map<std::uint64_t, Allocation> allocations_;
        std::array<std::uint64_t, maximum_tiles> owner_{};
        std::uint64_t frame_ = 0;
        std::uint64_t next_generation_ = 1;
    };

    // Tile edge for a light covering `radius_pixels` on screen, clamped to the supported resolutions.
    [[nodiscard]] auto resolution_for_screen_radius(float radius_pixels) noexcept -> std::uint32_t;

    [[nodiscard]] auto spot_view_projection(glm::vec3 position, glm::vec3 direction, float outer_cone_radians,
                                            float range, std::uint32_t resolution) noexcept -> glm::mat4;

    // Faces are ordered +X, -X, +Y, -Y, +Z, -Z, which the shader's face selection relies on.
    [[nodiscard]] auto point_face_view_projection(glm::vec3 position, std::uint32_t face, float range,
                                                  std::uint32_t resolution) noexcept -> glm::mat4;

    [[nodiscard]] auto spot_texel_world(float outer_cone_radians, std::uint32_t resolution) noexcept -> float;
    [[nodiscard]] auto point_texel_world(std::uint32_t resolution) noexcept -> float;

    [[nodiscard]] auto make_gpu_tile(glm::mat4 const &view_projection, Tile tile, float texel_world) noexcept
            -> GpuShadowTile;

}
