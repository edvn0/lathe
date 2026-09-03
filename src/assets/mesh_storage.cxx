#include "assets/mesh_storage.hxx"

#include <utility>

auto MeshStorage::create(MeshStorageCreateInfo const &create_info) -> std::expected<MeshStorage, MeshStorageError> {
    // Slot zero is permanently unused -- see model.hxx's comment on
    // MeshHandle's Sentinel = 0 -- so at least one real slot needs room
    // beyond it.
    if (create_info.capacity < 2) {
        return std::unexpected(MeshStorageError{.type = MeshStorageErrorType::invalid_argument});
    }

    MeshStorage storage;

    storage.slots_ = ObjectPool<MeshSlotData, 0>::create(create_info.capacity);

    static_cast<void>(storage.slots_.allocate());

    return storage;
}

auto MeshStorage::create_mesh(std::vector<Submesh> submeshes) -> std::expected<MeshHandle, MeshStorageError> {
    auto allocation = slots_.allocate();

    if (!allocation) {
        return std::unexpected(MeshStorageError{.type = MeshStorageErrorType::capacity_exceeded});
    }

    auto &[handle, slot] = *allocation;

    slot.submeshes = std::move(submeshes);

    return handle;
}

auto MeshStorage::destroy_mesh(MeshHandle handle) -> std::expected<void, MeshStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(MeshStorageError{.type = MeshStorageErrorType::invalid_handle});
    }

    slot->submeshes.clear();

    static_cast<void>(slots_.release(handle));

    return {};
}

auto MeshStorage::get(MeshHandle handle) noexcept -> MeshSlotData * { return slots_.get(handle); }

auto MeshStorage::get(MeshHandle handle) const noexcept -> MeshSlotData const * { return slots_.get(handle); }

auto MeshStorage::destroy() noexcept -> void { slots_ = ObjectPool<MeshSlotData, 0>{}; }
