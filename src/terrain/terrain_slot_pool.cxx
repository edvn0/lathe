#include "terrain/terrain_slot_pool.hxx"

#include "core/config.hxx"
#include "core/error_describe.hxx"
#include "core/logger.hxx"

#include <algorithm>

namespace {

    [[nodiscard]] auto slot_bounds(TerrainSlotPoolCreateInfo const &create_info, std::uint8_t lod)
            -> std::pair<glm::vec3, glm::vec3> {

        auto const span = terrain_chunk_span(create_info.lod_settings, lod);
        auto const half_span = span * 0.5F;
        auto const mid_height = (create_info.height_range_min + create_info.height_range_max) * 0.5F;
        auto const skirt_depth = create_info.height_range_max - create_info.height_range_min;

        return {
                glm::vec3{-half_span, create_info.height_range_min - mid_height - skirt_depth, -half_span},
                glm::vec3{half_span, create_info.height_range_max - mid_height, half_span},
        };
    }

} // namespace

auto TerrainSlotPool::create(IMeshSink &mesh_sink, VkCommandBuffer command_buffer,
                             TerrainSlotPoolCreateInfo const &create_info)
        -> std::expected<TerrainSlotPool, TerrainSlotPoolError> {

    TerrainSlotPool pool;
    pool.slots_per_lod_ = create_info.slots_per_lod;
    pool.free_by_lod_.resize(create_info.lod_levels);

    auto const &canonical_indices = terrain_chunk_indices();
    auto index_slice = mesh_sink.geometry_arena().allocate_indices(command_buffer, std::span{canonical_indices});

    if (!index_slice) {
        return std::unexpected(TerrainSlotPoolError{
                .message = std::format("terrain_slot_pool: failed to allocate the placeholder index buffer ({})",
                                       describe(index_slice.error())),
        });
    }

    // Never rendered: slots are only drawn after write(), which replaces all of this. One copy serves every slot.
    std::vector<CompressedModelVertex> const placeholder(terrain_chunk_vertex_count);
    MeshletBuild placeholder_build{
            .topology = build_meshlet_topology(std::span{canonical_indices}, terrain_chunk_vertex_count),
            .meshlets = {},
    };
    placeholder_build.meshlets = compute_meshlet_bounds(placeholder_build.topology, placeholder);

    auto placeholder_meshlets = upload_meshlets(mesh_sink.geometry_arena(), command_buffer, placeholder_build);

    if (!placeholder_meshlets) {
        return std::unexpected(TerrainSlotPoolError{
                .message = std::format("terrain_slot_pool: failed to build/allocate the placeholder meshlets ({})",
                                       describe(placeholder_meshlets.error())),
        });
    }

    for (std::uint8_t lod = 0; lod < create_info.lod_levels; ++lod) {
        auto const [bounds_min, bounds_max] = slot_bounds(create_info, lod);

        for (std::uint32_t slot = 0; slot < create_info.slots_per_lod; ++slot) {
            auto vertex_slice = mesh_sink.geometry_arena().allocate_vertices(command_buffer, std::span{placeholder});

            if (!vertex_slice) {
                return std::unexpected(TerrainSlotPoolError{
                        .message =
                                std::format("terrain_slot_pool: failed to allocate vertex slot (lod={}, slot={}): {}",
                                            lod, slot, describe(vertex_slice.error())),
                });
            }

            MeshGeometry const geometry{
                    .vertices = *vertex_slice,
                    .indices = *index_slice,
                    .meshlets = *placeholder_meshlets,
            };

            SubmeshCreateInfo submesh{
                    .material = create_info.material,
                    .bounds_min = bounds_min,
                    .bounds_max = bounds_max,
            };
            submesh.lods.fill(geometry); // chunks have a single LOD each

            auto mesh = mesh_sink.create_mesh(MeshCreateInfo{.submeshes = std::span{&submesh, 1}});

            if (!mesh) {
                return std::unexpected(TerrainSlotPoolError{
                        .message = std::format("terrain_slot_pool: failed to create mesh (lod={}, slot={}): {}", lod,
                                               slot, describe(mesh.error())),
                });
            }

            auto const slot_index = static_cast<std::uint32_t>(pool.slots_.size());
            pool.slots_.push_back(SlotRecord{
                    .mesh = *mesh,
                    .vertices = *vertex_slice,
                    .indices = *index_slice,
                    .meshlets = *placeholder_meshlets,
                    .owns_geometry = false,
                    .lod = lod,
            });
            pool.free_by_lod_[lod].push_back(slot_index);
        }
    }

    return pool;
}

auto TerrainSlotPool::acquire(std::uint8_t lod) -> std::optional<TerrainSlotHandle> {
    if (lod >= free_by_lod_.size() || free_by_lod_[lod].empty()) {
        return std::nullopt;
    }

    auto const slot_index = free_by_lod_[lod].back();
    free_by_lod_[lod].pop_back();

    return TerrainSlotHandle{.index = slot_index};
}

auto TerrainSlotPool::write(IMeshSink &mesh_sink, VkCommandBuffer command_buffer, TerrainSlotHandle handle,
                            TerrainChunkResult const &chunk) -> bool {

    if (!handle.valid() || handle.index >= slots_.size() || chunk.vertices.size() != terrain_chunk_vertex_count ||
        chunk.indices.empty() || chunk.meshlets.meshlets.empty()) {
        return false;
    }

    auto &slot = slots_[handle.index];
    auto &arena = mesh_sink.geometry_arena();

    auto written = arena.rewrite_slice(command_buffer, slot.vertices.bytes, std::as_bytes(std::span{chunk.vertices}));

    if (!written) {
        error("terrain_slot_pool: rewrite_slice failed for slot {} (lod={}): {}", handle.index, slot.lod,
              describe(written.error()));
        return false;
    }

    auto indices = arena.allocate_indices(command_buffer, std::span{chunk.indices});

    if (!indices) {
        error("terrain_slot_pool: index allocation failed for slot {} (lod={}): {}", handle.index, slot.lod,
              describe(indices.error()));
        return false;
    }

    auto meshlets = upload_meshlets(arena, command_buffer, chunk.meshlets);

    if (!meshlets) {
        error("terrain_slot_pool: meshlet upload failed for slot {} (lod={}): {}", handle.index, slot.lod,
              describe(meshlets.error()));
        arena.retire(indices->bytes);
        return false;
    }

    MeshGeometry const geometry{
            .vertices = slot.vertices,
            .indices = *indices,
            .meshlets = *meshlets,
    };

    if (auto updated = mesh_sink.update_submesh_geometry(slot.mesh, 0, geometry); !updated) {
        error("terrain_slot_pool: failed to repoint slot {} (lod={}): {}", handle.index, slot.lod,
              describe(updated.error()));
        arena.retire(indices->bytes);
        arena.retire(meshlets->descriptors);
        arena.retire(meshlets->data);
        return false;
    }

    // The slot sat in retirement for frames_in_flight frames before acquire(), and retire() defers again, so no
    // frame still reads these.
    if (slot.owns_geometry) {
        arena.retire(slot.indices.bytes);
        arena.retire(slot.meshlets.descriptors);
        arena.retire(slot.meshlets.data);
    }

    slot.indices = *indices;
    slot.meshlets = *meshlets;
    slot.owns_geometry = true;

    return true;
}

auto TerrainSlotPool::release_deferred(TerrainSlotHandle handle) -> void {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return;
    }

    retiring_.push_back(RetiringSlot{.slot_index = handle.index, .frames_remaining = frames_in_flight});
}

auto TerrainSlotPool::tick_retirement() -> void {
    for (auto &retiring: retiring_) {
        if (retiring.frames_remaining > 0) {
            --retiring.frames_remaining;
        }
    }

    std::erase_if(retiring_, [this](RetiringSlot const &retiring) {
        if (retiring.frames_remaining > 0) {
            return false;
        }

        free_by_lod_[slots_[retiring.slot_index].lod].push_back(retiring.slot_index);
        return true;
    });
}

auto TerrainSlotPool::mesh(TerrainSlotHandle handle) const -> MeshHandle {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return MeshHandle{};
    }

    return slots_[handle.index].mesh;
}

auto TerrainSlotPool::resident_count(std::uint8_t lod) const -> std::uint32_t {
    if (lod >= free_by_lod_.size()) {
        return 0;
    }

    return slots_per_lod_ - static_cast<std::uint32_t>(free_by_lod_[lod].size());
}
