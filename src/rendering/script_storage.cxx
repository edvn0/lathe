#include "rendering/script_storage.hxx"

auto ScriptStorage::create(ScriptStorageCreateInfo const &create_info)
        -> std::expected<ScriptStorage, ScriptStorageError> {
    // Slot zero is permanently unused -- see script_handle.hxx's comment on
    // ScriptHandle's Sentinel = 0 -- so at least one real slot needs room
    // beyond it (same reasoning as MeshStorage::create()).
    if (create_info.capacity < 2) {
        return std::unexpected(ScriptStorageError{.type = ScriptStorageErrorType::invalid_argument});
    }

    ScriptStorage storage;

    storage.slots_ = ObjectPool<ScriptSlotData, 0>::create(create_info.capacity);

    static_cast<void>(storage.slots_.allocate());

    return storage;
}

auto ScriptStorage::get(ScriptHandle handle) noexcept -> IScript * {
    auto *slot = slots_.get(handle);
    return slot != nullptr ? slot->script.get() : nullptr;
}

auto ScriptStorage::destroy(ScriptHandle handle) -> void {
    static_cast<void>(slots_.release(handle));
}
