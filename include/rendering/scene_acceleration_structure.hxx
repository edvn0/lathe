#pragma once

#include <volk.h>

#include <array>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <glm/mat4x4.hpp>

#include "core/config.hxx"
#include "gpu/buffer.hxx"

class GpuResourceTable;

// The ray tracing view of the scene (docs/probe-gi.md, phase 1): one bottom level acceleration structure (BLAS) per mesh
// submesh, and a top level one (TLAS) rebuilt every frame from the instances the renderer batched.
//
// - BLASes are built lazily, a budgeted number per frame, straight from the geometry arena's resident vertices
//   (R16G16B16A16_SFLOAT positions at stride 20; the fourth half is the texcoord and is ignored by the builder) and
//   indices. They use the coarsest LOD that has indices: tracing quality does not need the raster LOD. Every geometry is
//   flagged opaque, so alpha tested foliage is solid and traversal never calls back (a decision of the GI design).
// - Instances are written on the CPU, per frame in flight, together with a table (RtInstanceData, indexed by
//   instanceCustomIndex) giving a hit the device addresses of its vertices and indices and its material index. The
//   instance transform is the full world matrix, scale included.
// - The build is one engine pass (record_build): pending BLASes, then the TLAS, then a barrier making the TLAS visible to
//   ray queries on this queue. The TLAS storage buffer is a frame graph resource (Use::acceleration_structure_*).
//
// Skinned meshes, terrain and blended geometry are not in the TLAS yet (phase 5 of the plan).
class SceneAccelerationStructure {
public:
    // Mirrors RtInstanceData in assets/shaders/rt_scene.slang. instanceCustomIndex indexes the table.
    struct InstanceData {
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress index_address = 0;
        std::uint32_t material_index = 0;
        std::uint32_t index_count = 0;
        std::uint32_t reserved0 = 0;
        std::uint32_t reserved1 = 0;
    };
    static_assert(sizeof(InstanceData) == 32);

    // What the renderer knows about the trace LOD of one submesh.
    struct Geometry {
        std::uint64_t key = 0;
        VkDeviceSize vertex_offset = 0;
        VkDeviceSize index_offset = 0;
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress index_address = 0;
        std::uint32_t vertex_count = 0;
        std::uint32_t index_count = 0;
        VkIndexType index_type = VK_INDEX_TYPE_UINT32;
    };

    struct Stats {
        std::uint32_t instances = 0;
        std::uint32_t blas_total = 0;
        std::uint32_t blas_built_this_frame = 0;
        std::uint32_t instances_waiting = 0;
        VkDeviceSize blas_bytes = 0;
        VkDeviceSize tlas_bytes = 0;
    };

    static constexpr auto vertex_stride = std::uint32_t{20};
    static constexpr auto position_offset = std::uint32_t{8};

    SceneAccelerationStructure() = default;
    ~SceneAccelerationStructure();

    SceneAccelerationStructure(SceneAccelerationStructure const &) = delete;
    auto operator=(SceneAccelerationStructure const &) -> SceneAccelerationStructure & = delete;

    // False, harmlessly, without VulkanContext::ray_query_supported.
    [[nodiscard]] auto create(VulkanContext &context, GpuResourceTable &resource_table) -> bool;
    auto destroy() noexcept -> void;
    [[nodiscard]] auto ready() const noexcept -> bool { return context_ != nullptr; }

    static auto key_of(std::uint32_t mesh_index, std::uint32_t mesh_generation, std::uint32_t submesh) noexcept
            -> std::uint64_t {
        return (std::uint64_t{mesh_index} << 32U) ^ (std::uint64_t{mesh_generation & 0xFFFFU} << 16U) ^
               std::uint64_t{submesh & 0xFFFFU};
    }

    // Opens the frame: clears the instance list and ages the BLASes. Call once per frame before add().
    auto begin_frame(std::uint32_t frame_index) -> void;

    // One instance of `geometry` per transform. Skipped when the geometry is not usable or its BLAS has to wait for the
    // per frame build budget.
    auto add(Geometry const &geometry, std::uint32_t material_index, std::span<glm::mat4 const *const> transforms)
            -> void;

    // Uploads the instances, sizes the per frame buffers and the TLAS, and rewrites the descriptor when the TLAS changed.
    // Call after the last add() and before the frame graph is built.
    auto end_frame() -> void;

    // Records the BLAS and TLAS builds of this frame into `command_buffer`.
    auto record_build(VkCommandBuffer command_buffer) -> void;

    [[nodiscard]] auto tlas_buffer() const noexcept -> Buffer const & { return frames_[frame_].tlas.storage; }
    [[nodiscard]] auto instance_table_address() const noexcept -> VkDeviceAddress {
        return frames_[frame_].table.valid() ? frames_[frame_].table.device_address : 0;
    }
    // True when the current frame has a TLAS to build and read.
    [[nodiscard]] auto active() const noexcept -> bool { return frames_[frame_].tlas.handle != VK_NULL_HANDLE; }
    [[nodiscard]] auto stats() const noexcept -> Stats { return stats_; }

    // Frames a BLAS may go unused before it is freed.
    static constexpr auto idle_frames = std::uint64_t{600};
    // New BLASes per frame, and the scratch memory they may use together.
    static constexpr auto max_builds_per_frame = std::uint32_t{512};
    static constexpr auto max_scratch_per_frame = VkDeviceSize{64} * 1024 * 1024;

private:
    struct AccelerationStructure {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        Buffer storage;
        VkDeviceAddress address = 0;
        VkDeviceSize size = 0;
    };

    struct Blas {
        AccelerationStructure structure;
        VkDeviceSize vertex_offset = 0;
        VkDeviceSize index_offset = 0;
        std::uint32_t vertex_count = 0;
        std::uint32_t index_count = 0;
        VkDeviceSize scratch_size = 0;
        std::uint64_t last_used = 0;
    };

    struct PendingBuild {
        Blas *blas = nullptr;
        VkDeviceAddress vertex_address = 0;
        VkDeviceAddress index_address = 0;
        VkDeviceSize scratch_offset = 0;
    };

    struct FrameData {
        Buffer instances;
        Buffer table;
        Buffer scratch;
        AccelerationStructure tlas;
        std::uint32_t capacity = 0;
        VkDeviceSize tlas_scratch = 0;
        std::uint32_t instance_count = 0;
    };

    struct Retired {
        AccelerationStructure structure;
        std::uint32_t frames_remaining = frames_in_flight + 1;
    };

    [[nodiscard]] auto create_structure(AccelerationStructure &structure, VkAccelerationStructureTypeKHR type,
                                        VkDeviceSize size, char const *name) -> bool;
    auto destroy_structure(AccelerationStructure &structure) noexcept -> void;
    [[nodiscard]] auto make_blas(Geometry const &geometry) -> Blas *;
    auto retire(Blas &blas) -> void;
    [[nodiscard]] auto align_scratch(VkDeviceSize value) const noexcept -> VkDeviceSize;

    VulkanContext *context_ = nullptr;
    GpuResourceTable *resource_table_ = nullptr;
    VkDeviceSize scratch_alignment_ = 128;

    std::unordered_map<std::uint64_t, Blas> blases_;
    std::vector<Retired> retired_;
    std::array<FrameData, frames_in_flight> frames_{};
    std::uint32_t frame_ = 0;
    std::uint64_t frame_counter_ = 0;

    std::vector<VkAccelerationStructureInstanceKHR> staged_instances_;
    std::vector<InstanceData> staged_table_;
    std::vector<PendingBuild> pending_;
    VkDeviceSize pending_scratch_ = 0;
    std::uint32_t waiting_ = 0;
    bool warned_misaligned_ = false;
    Stats stats_{};
};
