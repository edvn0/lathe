#include "pipeline_storage.hxx"

#include <utility>

#include "context.hxx"

#include "logger.hxx"

namespace {
    auto make_error(PipelineStorageErrorType type) noexcept -> PipelineStorageError {
        return PipelineStorageError{
                .type = type,
        };
    }

    auto make_pipeline_error(PipelineError error) noexcept -> PipelineStorageError {
        return PipelineStorageError{
                .type = PipelineStorageErrorType::pipeline_error,
                .cause = ErrorCause{Boxed<PipelineError>{std::move(error)}},
        };
    }
} // namespace

PipelineStorage::~PipelineStorage() { destroy(); }

PipelineStorage::PipelineStorage(PipelineStorage &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), slots_(std::move(other.slots_)),
    free_head_(std::exchange(other.free_head_, 0)), capacity_(std::exchange(other.capacity_, 0)),
    size_(std::exchange(other.size_, 0)),
    global_descriptor_set_layout_(std::exchange(other.global_descriptor_set_layout_, nullptr)),
    debug_name_(std::move(other.debug_name_)) {}

auto PipelineStorage::operator=(PipelineStorage &&other) noexcept -> PipelineStorage & {
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

auto PipelineStorage::create(VulkanContext &context, PipelineStorageCreateInfo const &create_info)
        -> std::expected<PipelineStorage, PipelineStorageError> {
    if (context.device == VK_NULL_HANDLE || create_info.capacity == 0 ||
        create_info.global_descriptor_set_layout == VK_NULL_HANDLE) {
        return std::unexpected(make_error(PipelineStorageErrorType::invalid_argument));
    }

    PipelineStorage storage;

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

auto PipelineStorage::create_graphics(GraphicsPipelineCreateInfo const &create_info)
        -> std::expected<PipelineHandle, PipelineStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(PipelineStorageErrorType::invalid_argument));
    }

    if (free_head_ >= capacity_) {
        return std::unexpected(make_error(PipelineStorageErrorType::capacity_exceeded));
    }

    auto pipeline = Pipeline::create_graphics(*context_, create_info, global_descriptor_set_layout());

    if (!pipeline) {
        return std::unexpected(make_pipeline_error(pipeline.error()));
    }

    auto const index = free_head_;
    auto &slot = slots_[index];

    free_head_ = slot.next_free;

    slot.pipeline = std::move(*pipeline);

    slot.next_free = capacity_;
    slot.occupied = true;

    ++size_;

    return PipelineHandle{
            .index = index,
            .generation = slot.generation,
    };
}

auto PipelineStorage::destroy_pipeline(PipelineHandle handle) -> std::expected<void, PipelineStorageError> {
    auto *slot = slot_for(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(PipelineStorageErrorType::invalid_handle));
    }

    slot->pipeline.destroy();
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

auto PipelineStorage::get(PipelineHandle handle) noexcept -> Pipeline * {
    auto *slot = slot_for(handle);

    return slot != nullptr ? &slot->pipeline : nullptr;
}

auto PipelineStorage::get(PipelineHandle handle) const noexcept -> Pipeline const * {
    auto const *slot = slot_for(handle);

    return slot != nullptr ? &slot->pipeline : nullptr;
}

auto PipelineStorage::destroy() noexcept -> void {
    for (auto &slot: slots_) {
        if (!slot.occupied) {
            continue;
        }

        slot.pipeline.destroy();
        slot.occupied = false;
    }

    slots_.clear();

    context_ = nullptr;
    free_head_ = 0;
    capacity_ = 0;
    size_ = 0;

    debug_name_.clear();
}

auto PipelineStorage::slot_for(PipelineHandle handle) noexcept -> Slot * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

auto PipelineStorage::slot_for(PipelineHandle handle) const noexcept -> Slot const * {
    if (!handle.valid() || handle.index >= slots_.size()) {
        return nullptr;
    }

    auto const &slot = slots_[handle.index];

    if (!slot.occupied || slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}
