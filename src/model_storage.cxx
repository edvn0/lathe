#include "model_storage.hxx"

#include <utility>

auto ModelStorage::create(ModelStorageCreateInfo const &create_info) -> std::expected<ModelStorage, ModelStorageError> {
    // Slot zero is permanently unused -- see model.hxx's comment on
    // ModelHandle's Sentinel = 0 -- so at least one real slot needs room
    // beyond it.
    if (create_info.capacity < 2) {
        return std::unexpected(ModelStorageError{.type = ModelStorageErrorType::invalid_argument});
    }

    ModelStorage storage;

    storage.slots_ = ObjectPool<ModelSlotData, 0>::create(create_info.capacity);

    static_cast<void>(storage.slots_.allocate());

    return storage;
}

auto ModelStorage::create_model(ModelSlotData data) -> std::expected<ModelHandle, ModelStorageError> {
    auto allocation = slots_.allocate();

    if (!allocation) {
        return std::unexpected(ModelStorageError{.type = ModelStorageErrorType::capacity_exceeded});
    }

    auto &[handle, slot] = *allocation;

    slot = std::move(data);

    return handle;
}

auto ModelStorage::get(ModelHandle handle) noexcept -> ModelSlotData * { return slots_.get(handle); }

auto ModelStorage::get(ModelHandle handle) const noexcept -> ModelSlotData const * { return slots_.get(handle); }

auto ModelStorage::destroy() noexcept -> void { slots_ = ObjectPool<ModelSlotData, 0>{}; }
