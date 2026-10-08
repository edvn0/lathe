#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "assets/geometry.hxx"
#include "assets/load_model.hxx"
#include "assets/material.hxx"
#include "assets/mesh_sink.hxx"
#include "assets/meshlet.hxx"
#include "assets/model.hxx"
#include "terrain/terrain_chunk.hxx"
#include "terrain/terrain_quadtree.hxx"

struct TerrainSlotPoolError {
    std::string message;
};

struct TerrainSlotHandle {
    std::uint32_t index = std::numeric_limits<std::uint32_t>::max();

    [[nodiscard]] auto valid() const noexcept -> bool { return index != std::numeric_limits<std::uint32_t>::max(); }
};

struct TerrainSlotPoolCreateInfo {
    std::uint8_t lod_levels = 5;
    std::uint32_t slots_per_lod = 48;

    MaterialHandle material{};
    TerrainLodSettings lod_settings{};

    float height_range_min = -2.0F;
    float height_range_max = 2.0F;
};

class TerrainSlotPool {
public:
    TerrainSlotPool() = default;

    [[nodiscard]] static auto create(IMeshSink &mesh_sink, VkCommandBuffer command_buffer,
                                     TerrainSlotPoolCreateInfo const &create_info)
            -> std::expected<TerrainSlotPool, TerrainSlotPoolError>;

    [[nodiscard]] auto acquire(std::uint8_t lod) -> std::optional<TerrainSlotHandle>;

    [[nodiscard]] auto write(IMeshSink &mesh_sink, VkCommandBuffer command_buffer, TerrainSlotHandle handle,
                             TerrainChunkResult const &chunk) -> bool;

    auto release_deferred(TerrainSlotHandle handle) -> void;

    auto tick_retirement() -> void;

    [[nodiscard]] auto mesh(TerrainSlotHandle handle) const -> MeshHandle;

    [[nodiscard]] auto resident_count(std::uint8_t lod) const -> std::uint32_t;
    [[nodiscard]] auto capacity_per_lod() const -> std::uint32_t { return slots_per_lod_; }

private:
    struct SlotRecord {
        MeshHandle mesh{};
        VertexSlice vertices{};

        IndexSlice indices{};
        MeshletSlice meshlets{};
        bool owns_geometry = false;

        std::uint8_t lod = 0;
    };

    struct RetiringSlot {
        std::uint32_t slot_index = 0;
        std::uint32_t frames_remaining = 0;
    };

    std::vector<SlotRecord> slots_{};
    std::vector<std::vector<std::uint32_t>> free_by_lod_{};
    std::vector<RetiringSlot> retiring_{};
    std::uint32_t slots_per_lod_ = 0;
};
