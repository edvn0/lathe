#include "shader_object_storage.hxx"

#include <utility>

#include "context.hxx"

namespace {
    auto make_error(ShaderObjectStorageErrorType type) noexcept -> ShaderObjectStorageError {
        return ShaderObjectStorageError{
                .type = type,
        };
    }

    auto make_shader_object_error(ShaderObjectError error) noexcept -> ShaderObjectStorageError {
        return ShaderObjectStorageError{
                .type = ShaderObjectStorageErrorType::shader_object_error,
                .cause = ErrorCause{Boxed<ShaderObjectError>{std::move(error)}},
        };
    }
} // namespace

ShaderObjectStorage::~ShaderObjectStorage() { destroy(); }

ShaderObjectStorage::ShaderObjectStorage(ShaderObjectStorage &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), slots_(std::move(other.slots_)),
    global_descriptor_set_layout_(std::exchange(other.global_descriptor_set_layout_, nullptr)),
    debug_name_(std::move(other.debug_name_)) {}

auto ShaderObjectStorage::operator=(ShaderObjectStorage &&other) noexcept -> ShaderObjectStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

    global_descriptor_set_layout_ = std::exchange(other.global_descriptor_set_layout_, nullptr);

    debug_name_ = std::move(other.debug_name_);

    return *this;
}

auto ShaderObjectStorage::create(VulkanContext &context, ShaderObjectStorageCreateInfo const &create_info)
        -> std::expected<ShaderObjectStorage, ShaderObjectStorageError> {
    if (context.device == VK_NULL_HANDLE || create_info.capacity == 0 ||
        create_info.global_descriptor_set_layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_argument));
    }

    ShaderObjectStorage storage;

    storage.context_ = &context;

    storage.debug_name_ = std::string{create_info.debug_name};

    storage.slots_ = ObjectPool<ShaderObjectSet>::create(create_info.capacity);
    storage.global_descriptor_set_layout_ = create_info.global_descriptor_set_layout;

    return storage;
}

auto ShaderObjectStorage::create_linked(ShaderObjectCreateInfo const &create_info)
        -> std::expected<ShaderObjectHandle, ShaderObjectStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_argument));
    }

    // vkCreateShadersEXT (called inside ShaderObjectSet::create_linked) has
    // no shared cache to synchronize -- see the comment on slot_mutex_ --
    // so it runs unlocked; only the free-list bookkeeping below needs it.
    auto shader_object = ShaderObjectSet::create_linked(*context_, create_info, global_descriptor_set_layout());

    if (!shader_object) {
        return std::unexpected(make_shader_object_error(shader_object.error()));
    }

    std::lock_guard<std::mutex> const lock{slot_mutex_};

    auto allocation = slots_.allocate();

    if (!allocation) {
        shader_object->destroy();
        return std::unexpected(make_error(ShaderObjectStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot = std::move(*shader_object);

    return handle;
}

auto ShaderObjectStorage::create_compute(ComputeShaderCreateInfo const &create_info)
        -> std::expected<ShaderObjectHandle, ShaderObjectStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_argument));
    }

    auto shader_object = ShaderObjectSet::create_compute(*context_, create_info, global_descriptor_set_layout());

    if (!shader_object) {
        return std::unexpected(make_shader_object_error(shader_object.error()));
    }

    std::lock_guard<std::mutex> const lock{slot_mutex_};

    auto allocation = slots_.allocate();

    if (!allocation) {
        shader_object->destroy();
        return std::unexpected(make_error(ShaderObjectStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot = std::move(*shader_object);

    return handle;
}

auto ShaderObjectStorage::destroy_shader_object(ShaderObjectHandle handle)
        -> std::expected<void, ShaderObjectStorageError> {
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_handle));
    }

    slot->destroy();

    static_cast<void>(slots_.release(handle));

    return {};
}

auto ShaderObjectStorage::get(ShaderObjectHandle handle) noexcept -> ShaderObjectSet * { return slots_.get(handle); }

auto ShaderObjectStorage::get(ShaderObjectHandle handle) const noexcept -> ShaderObjectSet const * {
    return slots_.get(handle);
}

auto ShaderObjectStorage::destroy() noexcept -> void {
    for (std::uint32_t index = 0; index < slots_.capacity(); ++index) {
        if (!slots_.occupied_at(index)) {
            continue;
        }

        slots_.get_at(index)->destroy();
    }

    slots_ = ObjectPool<ShaderObjectSet>{};

    context_ = nullptr;

    debug_name_.clear();
}
