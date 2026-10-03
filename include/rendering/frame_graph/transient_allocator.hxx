#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

#include "gpu/image_storage.hxx"
#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/vk_translate.hxx"

struct VulkanContext;

namespace frame_graph {

    struct TransientAllocationError {
        std::string message;
    };

    // Backs a compiled graph's transient images with memory, per frame slot. The images of one slot are created from
    // blocks of device memory (aliasing images where lifetimes allow, see plan_transients) and kept while the compiled
    // graph and the transient descriptions stay the same, so bindless indices handed to passes stay valid from frame
    // to frame. When either changes (a resize, a pass toggled) the slot's old images and blocks are destroyed: call
    // prepare() only once the slot's earlier work has finished.
    class TransientAllocator {
    public:
        TransientAllocator() = default;
        ~TransientAllocator() { release_all(); }

        TransientAllocator(TransientAllocator const &) = delete;
        auto operator=(TransientAllocator const &) -> TransientAllocator & = delete;
        TransientAllocator(TransientAllocator &&) = delete;
        auto operator=(TransientAllocator &&) -> TransientAllocator & = delete;

        auto initialize(VulkanContext &context, ImageStorage &images, std::uint32_t slot_count) -> void;

        // Makes the slot's transients match `graph`/`compiled`, recreating them only if they changed. Returns whether
        // it did: new images have new bindless slots, which the GPU resource table has to pick up before shaders
        // sample them.
        [[nodiscard]] auto prepare(std::uint32_t slot, GraphDesc const &graph, CompiledGraph const &compiled,
                                   bool alias) -> std::expected<bool, TransientAllocationError>;

        // Destroys one slot's images and blocks (the GPU must be done with them), or all of them.
        auto release(std::uint32_t slot) -> void;
        auto release_all() -> void;

        [[nodiscard]] auto image(std::uint32_t slot, std::uint32_t resource) const noexcept -> Image const *;
        [[nodiscard]] auto handle(std::uint32_t slot, std::uint32_t resource) const noexcept -> ImageHandle;
        [[nodiscard]] auto mip_handle(std::uint32_t slot, std::uint32_t resource, std::uint32_t mip) const noexcept
                -> ImageHandle;

        // The slot's placement and sizes (empty before the first prepare).
        [[nodiscard]] auto plan(std::uint32_t slot) const noexcept -> TransientPlan const &;

        // Bytes of device memory the slot's transients occupy, and what they would take without aliasing.
        [[nodiscard]] auto total_bytes() const noexcept -> std::uint64_t;
        [[nodiscard]] auto unaliased_bytes() const noexcept -> std::uint64_t;

        // Adds the slot's transient images to the handles the executor translates barriers with.
        auto fill(std::uint32_t slot, PhysicalResources &resources) const -> void;

    private:
        struct Entry {
            ImageHolder image;
            std::vector<ImageHolder> mip_slots;
        };

        struct Slot {
            std::uint64_t key = 0; // of the images as created; see prepare()
            bool valid = false;
            std::uint64_t requirements_key = 0;
            bool requirements_valid = false;
            std::vector<MemoryRequirement> requirements; // by resource slot
            std::vector<ImageCreateInfo> infos; // by resource slot
            TransientPlan plan;
            std::vector<VmaAllocation> blocks;
            std::vector<Entry> entries; // by resource slot
        };

        VulkanContext *context_ = nullptr;
        ImageStorage *images_ = nullptr;
        std::vector<Slot> slots_;
    };

} // namespace frame_graph
