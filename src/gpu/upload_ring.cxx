#include "gpu/upload_ring.hxx"

#include <algorithm>
#include <bit>
#include <format>

#include "core/human_readable_bytes.hxx"
#include "core/logger.hxx"

namespace {
    [[nodiscard]] auto align_up(VkDeviceSize value, VkDeviceSize alignment) noexcept -> VkDeviceSize {
        return (value + alignment - 1) / alignment * alignment;
    }
} // namespace

auto UploadRing::capacity(std::uint32_t slot) const noexcept -> VkDeviceSize {
    if (slot >= slots_.size()) {
        return 0;
    }

    auto total = VkDeviceSize{0};
    for (auto const &block: slots_[slot].blocks) {
        total += block.buffer.size();
    }
    return total;
}

auto UploadRing::add_block(VulkanContext &context, Slot &slot, VkDeviceSize at_least)
        -> std::expected<Block *, DeviceError> {
    auto const size = std::max(info_.minimum_block, std::bit_ceil(std::max<VkDeviceSize>(at_least, 1)));

    info("[{}] Allocating a {} block", info_.name, human_readable_bytes(size));

    auto created = Buffer::create(context, BufferCreateInfo{
                                                   .size = size,
                                                   .usage = info_.usage,
                                                   .memory = BufferMemory::upload,
                                                   .debug_name = info_.name,
                                           });
    if (!created) {
        return std::unexpected(created.error());
    }

    slot.blocks.push_back(Block{.buffer = std::move(*created)});
    return &slot.blocks.back();
}

auto UploadRing::begin_frame(VulkanContext &context, std::uint32_t slot_index) -> std::expected<void, DeviceError> {
    if (slot_index >= slots_.size()) {
        return std::unexpected(DeviceError::buffer_creation(std::format(
                "upload ring '{}' has {} slots, asked for slot {}", info_.name, slots_.size(), slot_index)));
    }

    auto &slot = slots_[slot_index];

    // What the finished frame wrote is what a frame like it needs; when that spilled over several blocks, replace
    // them with one block that holds it all.
    if (slot.blocks.size() > 1) {
        auto demand = VkDeviceSize{0};
        for (auto const &block: slot.blocks) {
            demand += block.cursor;
        }

        slot.blocks.clear();

        if (auto const added = add_block(context, slot, demand); !added) {
            return std::unexpected(added.error());
        }
    }

    for (auto &block: slot.blocks) {
        block.cursor = 0;
    }

    return {};
}

auto UploadRing::write_bytes(VulkanContext &context, std::uint32_t slot_index, void const *data, std::size_t size,
                             VkDeviceSize alignment) -> std::expected<UploadSlice, DeviceError> {
    if (slot_index >= slots_.size()) {
        return std::unexpected(DeviceError::buffer_creation(std::format(
                "upload ring '{}' has {} slots, asked for slot {}", info_.name, slots_.size(), slot_index)));
    }

    if (size == 0) {
        return UploadSlice{};
    }

    auto &slot = slots_[slot_index];

    auto *block = slot.blocks.empty() ? nullptr : &slot.blocks.back();
    if (block == nullptr || align_up(block->cursor, alignment) + size > block->buffer.size()) {
        auto const added = add_block(context, slot, size + alignment);
        if (!added) {
            return std::unexpected(added.error());
        }
        block = *added;
    }

    auto const offset = align_up(block->cursor, alignment);
    std::memcpy(block->buffer.mapped_data() + offset, data, size);
    block->cursor = offset + size;

    return UploadSlice{
            .buffer = block->buffer.buffer,
            .address = block->buffer.device_address + offset,
            .offset = offset,
            .size = size,
    };
}

auto UploadRing::flush(std::uint32_t slot_index) -> std::expected<void, DeviceError> {
    if (slot_index >= slots_.size()) {
        return {};
    }

    for (auto &block: slots_[slot_index].blocks) {
        if (block.cursor == 0) {
            continue;
        }
        if (auto const flushed = block.buffer.flush(); !flushed) {
            return flushed;
        }
    }

    return {};
}
