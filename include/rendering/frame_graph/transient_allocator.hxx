#pragma once

#include <volk.h>

#include <cstdint>
#include <expected>
#include <string>
#include <optional>
#include <vector>

#include "gpu/image_storage.hxx"
#include "rendering/frame_graph/aliasing.hxx"
#include "rendering/frame_graph/vk_translate.hxx"

struct VulkanContext;

namespace frame_graph {

    struct TransientAllocationError {
        std::string message;
    };

    class TransientAllocator {
    public:
        TransientAllocator() = default;
        ~TransientAllocator() { release_all(); }

        TransientAllocator(TransientAllocator const &) = delete;
        auto operator=(TransientAllocator const &) -> TransientAllocator & = delete;
        TransientAllocator(TransientAllocator &&) = delete;
        auto operator=(TransientAllocator &&) -> TransientAllocator & = delete;

        auto initialize(VulkanContext &context, ImageStorage &images, std::uint32_t slot_count) -> void;

        [[nodiscard]] auto prepare(std::uint32_t slot, GraphDesc const &graph, CompiledGraph const &compiled,
                                   bool alias) -> std::expected<bool, TransientAllocationError>;

        auto release(std::uint32_t slot) -> void;
        auto release_all() -> void;
        // Destroys what reallocations replaced, after the device is idle. Call it where no command buffer of the
        // current frame has been recorded yet.
        auto release_retired() -> void;

        [[nodiscard]] auto image(std::uint32_t slot, std::uint32_t resource) const noexcept -> Image const *;
        [[nodiscard]] auto handle(std::uint32_t slot, std::uint32_t resource) const noexcept -> ImageHandle;
        [[nodiscard]] auto mip_handle(std::uint32_t slot, std::uint32_t resource, std::uint32_t mip) const noexcept
                -> ImageHandle;

        [[nodiscard]] auto plan(std::uint32_t slot) const noexcept -> TransientPlan const &;

        [[nodiscard]] auto total_bytes() const noexcept -> std::uint64_t;
        [[nodiscard]] auto unaliased_bytes() const noexcept -> std::uint64_t;

        auto fill(std::uint32_t slot, PhysicalResources &resources) const -> void;

    private:
        struct Entry {
            ImageHolder image;
            std::vector<ImageHolder> mip_slots;
        };

        // What a reallocation replaced. A command buffer recorded earlier in the same frame (another view's) may have
        // the bindless table bound with these images in it, and destroying them would invalidate it, so they live on
        // until release_retired, which the renderer calls before it records anything for the next frame.
        struct Retired {
            std::vector<Entry> entries;
            std::vector<VmaAllocation> blocks;
        };

        struct Slot {
            std::uint64_t key = 0;
            bool valid = false;
            std::uint64_t requirements_key = 0;
            bool requirements_valid = false;
            std::vector<MemoryRequirement> requirements;
            std::vector<ImageCreateInfo> infos;
            TransientPlan plan;
            std::vector<VmaAllocation> blocks;
            std::vector<Entry> entries;
            std::optional<Retired> retired;
        };

        auto free_retired(Slot &slot) -> void;
        auto retire(std::uint32_t slot) -> void;

        VulkanContext *context_ = nullptr;
        ImageStorage *images_ = nullptr;
        std::vector<Slot> slots_;
    };

}
