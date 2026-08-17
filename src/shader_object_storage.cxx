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
    free_head_(std::exchange(other.free_head_, 0)), capacity_(std::exchange(other.capacity_, 0)),
    size_(std::exchange(other.size_, 0)),
    global_descriptor_set_layout_(std::exchange(other.global_descriptor_set_layout_, nullptr)),
    debug_name_(std::move(other.debug_name_)) {}

auto ShaderObjectStorage::operator=(ShaderObjectStorage &&other) noexcept -> ShaderObjectStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

    free_head_ = std::exchange(other.free_head_, 0);

    capacity_ = std::exchange(other.capacity_, 0);

    size_ = std::exchange(other.size_, 0);

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
    storage.capacity_ = create_info.capacity;

    storage.debug_name_ = std::string{create_info.debug_name};

    storage.slots_.resize(create_info.capacity);
    storage.global_descriptor_set_layout_ = create_info.global_descriptor_set_layout;

    /*
     * Index zero is allowed because generation zero is
     * the invalid handle sentinel.
     */
    storage.free_head_ = 0;

    for (std::uint32_t index = 0; index < create_info.capacity; ++index) {
        storage.slots_[index].next_free = index + 1 < create_info.capacity ? index + 1 : create_info.capacity;
    }

    /*
     * capacity_ is used as the end-of-list sentinel,
     * because zero is a valid slot.
     */
    return storage;
}

auto ShaderObjectStorage::create_linked(ShaderObjectCreateInfo const &create_info)
        -> std::expected<ShaderObjectHandle, ShaderObjectStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_argument));
    }

    if (free_head_ >= capacity_) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::capacity_exceeded));
    }

    auto shader_object = ShaderObjectSet::create_linked(*context_, create_info, global_descriptor_set_layout());

    if (!shader_object) {
        return std::unexpected(make_shader_object_error(shader_object.error()));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    free_head_ = slot.next_free;

    slot.shader_object = std::move(*shader_object);

    slot.next_free = capacity_;
    slot.occupied = true;

    ++size_;

    return ShaderObjectHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto ShaderObjectStorage::create_compute(ComputeShaderCreateInfo const &create_info)
        -> std::expected<ShaderObjectHandle, ShaderObjectStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_argument));
    }

    if (free_head_ >= capacity_) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::capacity_exceeded));
    }

    auto shader_object = ShaderObjectSet::create_compute(*context_, create_info, global_descriptor_set_layout());

    if (!shader_object) {
        return std::unexpected(make_shader_object_error(shader_object.error()));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    free_head_ = slot.next_free;

    slot.shader_object = std::move(*shader_object);

    slot.next_free = capacity_;
    slot.occupied = true;

    ++size_;

    return ShaderObjectHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto ShaderObjectStorage::destroy_shader_object(ShaderObjectHandle handle)
        -> std::expected<void, ShaderObjectStorageError> {
    auto *slot = slot_for(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(ShaderObjectStorageErrorType::invalid_handle));
    }

    slot->shader_object.destroy();
    slot->occupied = false;

    ++slot->generation;

    if (slot->generation == 0) {
        slot->generation = 1;
    }

    slot->next_free = free_head_;
    free_head_ = handle.index;

    --size_;

    return {};
}

auto ShaderObjectStorage::get(ShaderObjectHandle handle) noexcept -> ShaderObjectSet * {
    auto *slot = slot_for(handle);

    return slot != nullptr ? &slot->shader_object : nullptr;
}

auto ShaderObjectStorage::get(ShaderObjectHandle handle) const noexcept -> ShaderObjectSet const * {
    auto const *slot = slot_for(handle);

    return slot != nullptr ? &slot->shader_object : nullptr;
}

auto ShaderObjectStorage::destroy() noexcept -> void {
    for (auto &slot: slots_) {
        if (!slot.occupied) {
            continue;
        }

        slot.shader_object.destroy();
        slot.occupied = false;
    }

    slots_.clear();

    context_ = nullptr;
    free_head_ = 0;
    capacity_ = 0;
    size_ = 0;

    debug_name_.clear();
}

auto ShaderObjectStorage::slot_for(ShaderObjectHandle handle) noexcept -> Slot * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto ShaderObjectStorage::slot_for(ShaderObjectHandle handle) const noexcept -> Slot const * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto const &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}
