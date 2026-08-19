#include "pipeline_storage.hxx"

#include <cstring>
#include <fstream>
#include <span>
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

    // Reads and validates a previously-saved pipeline cache blob. Returns an
    // empty vector (never an error) on any failure -- a missing, truncated,
    // corrupt, or stale (different driver/device) cache file just means
    // starting cold, never a reason to fail PipelineStorage::create().
    [[nodiscard]]
    auto load_pipeline_cache_data(VulkanContext const &context, std::filesystem::path const &path)
            -> std::vector<std::byte> {
        if (path.empty()) {
            return {};
        }

        std::error_code error_code;
        if (!std::filesystem::exists(path, error_code) || error_code) {
            return {};
        }

        std::ifstream file{path, std::ios::binary | std::ios::ate};

        if (!file) {
            warn("Pipeline cache file '{}' could not be opened, starting cold", path.string());
            return {};
        }

        auto const size = static_cast<std::size_t>(file.tellg());

        if (size < sizeof(VkPipelineCacheHeaderVersionOne)) {
            warn("Pipeline cache file '{}' is truncated, starting cold", path.string());
            return {};
        }

        std::vector<std::byte> data(size);

        file.seekg(0);
        file.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));

        if (!file) {
            warn("Pipeline cache file '{}' could not be fully read, starting cold", path.string());
            return {};
        }

        VkPipelineCacheHeaderVersionOne header{};
        std::memcpy(&header, data.data(), sizeof(header));

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physical_device, &properties);

        if (header.headerSize < sizeof(VkPipelineCacheHeaderVersionOne) ||
            header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || header.vendorID != properties.vendorID ||
            header.deviceID != properties.deviceID ||
            std::memcmp(header.pipelineCacheUUID, properties.pipelineCacheUUID, VK_UUID_SIZE) != 0) {
            info("Pipeline cache stale (driver/device changed), starting cold");
            return {};
        }

        return data;
    }

    // Writes data to path via a temp-file-then-rename so a crash mid-write
    // never corrupts the previous run's cache file.
    [[nodiscard]]
    auto write_file_atomic(std::filesystem::path const &path, std::span<std::byte const> data) -> bool {
        std::error_code error_code;
        std::filesystem::create_directories(path.parent_path(), error_code);

        auto tmp_path = path;
        tmp_path += ".tmp";

        {
            std::ofstream file{tmp_path, std::ios::binary | std::ios::trunc};

            if (!file) {
                return false;
            }

            file.write(reinterpret_cast<char const *>(data.data()), static_cast<std::streamsize>(data.size()));

            if (!file) {
                return false;
            }
        }

        std::filesystem::rename(tmp_path, path, error_code);

        return !error_code;
    }
} // namespace

PipelineStorage::~PipelineStorage() { destroy(); }

PipelineStorage::PipelineStorage(PipelineStorage &&other) noexcept :
    context_(std::exchange(other.context_, nullptr)), slots_(std::move(other.slots_)),
    free_head_(std::exchange(other.free_head_, 0)), capacity_(std::exchange(other.capacity_, 0)),
    size_(std::exchange(other.size_, 0)), cache_(std::exchange(other.cache_, VK_NULL_HANDLE)),
    cache_file_path_(std::move(other.cache_file_path_)),
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

    cache_ = std::exchange(other.cache_, VK_NULL_HANDLE);

    cache_file_path_ = std::move(other.cache_file_path_);

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
    storage.cache_file_path_ = create_info.cache_file_path;

    /*
     * Index zero is allowed because generation zero is
     * the invalid handle sentinel.
     */
    storage.free_head_ = 0;

    for (std::uint32_t index = 0; index < create_info.capacity; ++index) {
        storage.slots_[index].next_free = index + 1 < create_info.capacity ? index + 1 : create_info.capacity;
    }

    debug("[PipelineStorage::create] loading cache from '{}'", create_info.cache_file_path.string());

    auto const cache_data = load_pipeline_cache_data(context, create_info.cache_file_path);

    debug("[PipelineStorage::create] loaded {} bytes of cache data", cache_data.size());

    VkPipelineCacheCreateInfo const cache_create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .initialDataSize = cache_data.size(),
            .pInitialData = cache_data.empty() ? nullptr : cache_data.data(),
    };

    auto const cache_result = vkCreatePipelineCache(context.device, &cache_create_info, nullptr, &storage.cache_);


    if (cache_result != VK_SUCCESS) {
        return std::unexpected(make_error(PipelineStorageErrorType::pipeline_error));
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


    // Pipeline::create_graphics (including the expensive driver-side
    // compile) is called unlocked so concurrent callers from
    // register_pipelines_parallel's build phase actually run in parallel --
    // only the cheap free-list bookkeeping below is serialized.
    auto pipeline = Pipeline::create_graphics(*context_, create_info, global_descriptor_set_layout(), cache_);


    if (!pipeline) {
        return std::unexpected(make_pipeline_error(pipeline.error()));
    }

    std::lock_guard<std::mutex> const lock{slot_mutex_};

    if (free_head_ >= capacity_) {
        pipeline->destroy();
        return std::unexpected(make_error(PipelineStorageErrorType::capacity_exceeded));
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

auto PipelineStorage::create_compute(ComputePipelineCreateInfo const &create_info)
        -> std::expected<PipelineHandle, PipelineStorageError> {
    if (context_ == nullptr) {
        return std::unexpected(make_error(PipelineStorageErrorType::invalid_argument));
    }


    auto pipeline = Pipeline::create_compute(*context_, create_info, global_descriptor_set_layout(), cache_);


    if (!pipeline) {
        return std::unexpected(make_pipeline_error(pipeline.error()));
    }

    std::lock_guard<std::mutex> const lock{slot_mutex_};

    if (free_head_ >= capacity_) {
        pipeline->destroy();
        return std::unexpected(make_error(PipelineStorageErrorType::capacity_exceeded));
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

auto PipelineStorage::save_cache_to_disk() const -> std::expected<void, PipelineStorageError> {
    if (context_ == nullptr || cache_ == VK_NULL_HANDLE || cache_file_path_.empty()) {
        return {};
    }

    std::size_t size = 0;

    auto result = vkGetPipelineCacheData(context_->device, cache_, &size, nullptr);

    if (result != VK_SUCCESS || size == 0) {
        return std::unexpected(make_error(PipelineStorageErrorType::pipeline_error));
    }

    std::vector<std::byte> data(size);

    result = vkGetPipelineCacheData(context_->device, cache_, &size, data.data());

    if (result != VK_SUCCESS) {
        return std::unexpected(make_error(PipelineStorageErrorType::pipeline_error));
    }

    data.resize(size);

    if (!write_file_atomic(cache_file_path_, data)) {
        return std::unexpected(make_error(PipelineStorageErrorType::pipeline_error));
    }

    return {};
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

    if (context_ != nullptr && cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(context_->device, cache_, nullptr);
    }

    cache_ = VK_NULL_HANDLE;
    cache_file_path_.clear();

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
