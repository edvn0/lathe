#include "assets/model_storage.hxx"

#include <utility>

auto ModelStorage::create(ModelStorageCreateInfo const &create_info) -> std::expected<ModelStorage, ModelStorageError> {
    // Slot zero is never used, so at least one more slot is needed.
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
    // A new slot has one owner, whatever ref_count `data` carried (e.g. a copied fallback's).
    slot.ref_count = 1;

    return handle;
}

auto ModelStorage::create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, ModelStorageError> {
    auto const *fallback_slot = slots_.get(fallback);

    if (fallback_slot == nullptr) {
        return std::unexpected(ModelStorageError{.type = ModelStorageErrorType::invalid_handle});
    }

    return create_model(*fallback_slot);
}

auto ModelStorage::upgrade_pending_model(ModelHandle handle, ModelSlotData data)
        -> std::expected<ModelHandle, ModelStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(ModelStorageError{.type = ModelStorageErrorType::invalid_handle});
    }

    // Replacing the data doesn't change who owns the handle.
    auto const preserved_ref_count = slot->ref_count;
    *slot = std::move(data);
    slot->ref_count = preserved_ref_count;

    return handle;
}

auto ModelStorage::release(ModelHandle handle) -> std::expected<void, ModelStorageError> {
    if (slots_.get(handle) == nullptr) {
        return std::unexpected(ModelStorageError{.type = ModelStorageErrorType::invalid_handle});
    }

    static_cast<void>(slots_.release(handle));

    return {};
}

auto ModelStorage::get(ModelHandle handle) noexcept -> ModelSlotData * { return slots_.get(handle); }

auto ModelStorage::get(ModelHandle handle) const noexcept -> ModelSlotData const * { return slots_.get(handle); }

auto ModelStorage::destroy() noexcept -> void { slots_ = ObjectPool<ModelSlotData, 0>{}; }
