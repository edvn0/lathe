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
        };

        VulkanContext *context_ = nullptr;
        ImageStorage *images_ = nullptr;
        std::vector<Slot> slots_;
    };

}
