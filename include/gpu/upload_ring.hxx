#pragma once

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/config.hxx"
#include "gpu/buffer.hxx"
#include "gpu/device_error.hxx"

struct VulkanContext;

// A range of an upload buffer, as a shader or a draw call sees it.
struct UploadSlice {
    VkBuffer buffer = VK_NULL_HANDLE;
    // For a shader that reads it through a pointer.
    VkDeviceAddress address = 0;
    // From the start of `buffer`, for vkCmdBindIndexBuffer and friends.
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;

    [[nodiscard]] auto empty() const noexcept -> bool { return size == 0; }
};

// Host-visible, device-addressable memory the CPU rewrites every frame, one set of blocks per frame in flight.
//
// A frame calls begin_frame(slot) once, then write() as many times as it has data (a renderer that draws twice, or
// uploads several arrays), then flush(slot). The slot is the frame index the frame is recorded with, so the GPU is
// finished with everything the slot held last time and begin_frame may reuse it.
//
// Memory is handed out from blocks. A frame that outgrows its block gets another one instead of a bigger
// replacement, because the commands already recorded this frame still read the first; the next begin_frame folds
// the slot's blocks into one that fits what the last frame needed, so a steady workload ends up with one block per
// slot and no allocation.
class UploadRing {
public:
    struct Info {
        std::string name;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        VkDeviceSize minimum_block = 4096;
        std::uint32_t slots = frames_in_flight;
    };

    UploadRing() = default;
    explicit UploadRing(Info info) : info_{std::move(info)}, slots_(info_.slots) {}

    // Starts the slot's next frame: everything written to it before is free again.
    auto begin_frame(VulkanContext &context, std::uint32_t slot) -> std::expected<void, DeviceError>;

    // Copies `items` into the slot, aligned to `alignment`. The slice stays valid until the slot's next begin_frame.
    template<typename T>
        requires std::is_trivially_copyable_v<T>
    [[nodiscard]] auto write(VulkanContext &context, std::uint32_t slot, std::span<T const> items,
                             VkDeviceSize alignment = 16) -> std::expected<UploadSlice, DeviceError> {
        return write_bytes(context, slot, items.data(), items.size_bytes(), alignment);
    }

    // Makes everything written to the slot since begin_frame visible to the device.
    [[nodiscard]] auto flush(std::uint32_t slot) -> std::expected<void, DeviceError>;

    [[nodiscard]] auto slot_count() const noexcept -> std::uint32_t {
        return static_cast<std::uint32_t>(slots_.size());
    }
    [[nodiscard]] auto block_count(std::uint32_t slot) const noexcept -> std::size_t {
        return slot < slots_.size() ? slots_[slot].blocks.size() : 0;
    }
    // Bytes the slot's blocks hold in all.
    [[nodiscard]] auto capacity(std::uint32_t slot) const noexcept -> VkDeviceSize;

private:
    struct Block {
        Buffer buffer;
        VkDeviceSize cursor = 0;
    };

    struct Slot {
        std::vector<Block> blocks;
        // What the last frame asked for, to size the folded block.
        VkDeviceSize demand = 0;
    };

    [[nodiscard]] auto write_bytes(VulkanContext &context, std::uint32_t slot, void const *data, std::size_t size,
                                   VkDeviceSize alignment) -> std::expected<UploadSlice, DeviceError>;
    [[nodiscard]] auto add_block(VulkanContext &context, Slot &slot, VkDeviceSize at_least)
            -> std::expected<Block *, DeviceError>;

    Info info_;
    std::vector<Slot> slots_;
};
