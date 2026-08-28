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
    cache_(std::exchange(other.cache_, VK_NULL_HANDLE)), cache_file_path_(std::move(other.cache_file_path_)),
    global_descriptor_set_layout_(std::exchange(other.global_descriptor_set_layout_, nullptr)),
    debug_name_(std::move(other.debug_name_)) {}

auto PipelineStorage::operator=(PipelineStorage &&other) noexcept -> PipelineStorage & {
    if (this == &other) {
        return *this;
    }

    destroy();

    context_ = std::exchange(other.context_, nullptr);

    slots_ = std::move(other.slots_);

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

    storage.debug_name_ = std::string{create_info.debug_name};

    storage.slots_ = ObjectPool<Pipeline>::create(create_info.capacity);
    storage.global_descriptor_set_layout_ = create_info.global_descriptor_set_layout;
    storage.cache_file_path_ = create_info.cache_file_path;

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

    auto allocation = slots_.allocate();

    if (!allocation) {
        pipeline->destroy();
        return std::unexpected(make_error(PipelineStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot = std::move(*pipeline);

    return handle;
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

    auto allocation = slots_.allocate();

    if (!allocation) {
        pipeline->destroy();
        return std::unexpected(make_error(PipelineStorageErrorType::capacity_exceeded));
    }

    auto &[handle, slot] = *allocation;

    slot = std::move(*pipeline);

    return handle;
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
    auto *slot = slots_.get(handle);

    if (slot == nullptr) {
        return std::unexpected(make_error(PipelineStorageErrorType::invalid_handle));
    }

    slot->destroy();

    static_cast<void>(slots_.release(handle));

    return {};
}

auto PipelineStorage::get(PipelineHandle handle) noexcept -> Pipeline * { return slots_.get(handle); }

auto PipelineStorage::get(PipelineHandle handle) const noexcept -> Pipeline const * { return slots_.get(handle); }

auto PipelineStorage::destroy() noexcept -> void {
    for (std::uint32_t index = 0; index < slots_.capacity(); ++index) {
        if (!slots_.occupied_at(index)) {
            continue;
        }

        slots_.get_at(index)->destroy();
    }

    slots_ = ObjectPool<Pipeline>{};

    if (context_ != nullptr && cache_ != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(context_->device, cache_, nullptr);
    }

    cache_ = VK_NULL_HANDLE;
    cache_file_path_.clear();

    context_ = nullptr;

    debug_name_.clear();
}
