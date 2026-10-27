#include "assets/mesh_storage.hxx"

#include <algorithm>
#include <utility>

auto MeshStorage::create(MeshStorageCreateInfo const &create_info) -> std::expected<MeshStorage, MeshStorageError> {
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

    slot.bounding_centre = glm::vec3{0.0F};
    slot.bounding_radius = 0.0F;
    if (!slot.submeshes.empty()) {
        auto low = slot.submeshes.front().bounds_min;
        auto high = slot.submeshes.front().bounds_max;
        for (auto const &submesh: slot.submeshes) {
            low = glm::min(low, submesh.bounds_min);
            high = glm::max(high, submesh.bounds_max);
        }
        slot.bounding_centre = (low + high) * 0.5F;
        slot.bounding_radius = glm::length(high - low) * 0.5F;
    }

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
