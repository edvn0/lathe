#include "assets/model_storage.hxx"

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
    // A freshly allocated slot always starts with exactly one owner,
    // regardless of what ref_count the caller's `data` happened to carry
    // (e.g. create_pending_model() below copies a fallback slot's data
    // wholesale, which would otherwise also copy the fallback's own
    // ref_count onto this unrelated new handle).
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

    // Preserve the existing occupant's ref_count -- this replaces a pending
    // placeholder's data with the real, finished model in place (same
    // handle identity throughout, see this function's header comment), it
    // does not change who owns the handle.
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
