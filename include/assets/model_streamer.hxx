#pragma once

#include <volk.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "assets/load_model.hxx"
#include "core/paths.hxx"
#include "assets/model.hxx"
#include "assets/model_load_profile.hxx"
#include "assets/model_sink.hxx"
#include "core/fly_string.hxx"

enum class ModelRequestState : std::uint8_t {
    none,
    loading,
    failed,
};

class ModelStreamer {
public:
    [[nodiscard]]
    auto request(IModelSink &sink, AssetPath source_path, ModelHandle fallback,
                 FlyString debug_name) -> ModelHandle;

    [[nodiscard]]
    auto request_prepared(IModelSink &sink, std::future<std::expected<ModelCpuData, ModelLoadError>> cpu_data,
                          AssetPath source_path, ModelHandle fallback, FlyString debug_name) -> ModelHandle;

    auto forget(ModelHandle handle) -> void;

    [[nodiscard]]
    auto state(ModelHandle handle) const -> ModelRequestState;

    [[nodiscard]]
    auto failure_reason(ModelHandle handle) const -> std::string_view;

    auto process_ready(IModelSink &sink, VkCommandBuffer command_buffer) -> void;

    auto wait_all() -> void;

    [[nodiscard]] auto pending_count() const noexcept -> std::size_t { return pending_.size(); }

private:
    struct PendingRequest {
        ModelHandle handle;
        FlyString debug_name;
        std::future<std::expected<ModelCpuData, ModelLoadError>> future;

        std::optional<ModelPrimitiveFinalization> finalization;

        std::optional<ModelGpuUpload> upload;

        bool installed = false;

        std::shared_ptr<ModelLoadProfile> profile;
        std::chrono::steady_clock::time_point requested_at;

        std::size_t path_hash = 0;
        AssetPath source_path;

        bool prepared = false;
    };

    struct Reservation {
        ModelHandle handle;
        std::size_t path_hash = 0;
        bool final = false;
    };

    [[nodiscard]]
    auto reserve(IModelSink &sink, AssetPath const &source_path, ModelHandle fallback,
                 std::string_view debug_name) -> Reservation;

    struct FailedRequest {
        ModelHandle handle;
        std::string reason;
    };

    auto fail(IModelSink &sink, PendingRequest const &request, std::string reason) -> void;

    std::vector<PendingRequest> pending_;
    std::vector<FailedRequest> failed_;

    std::unordered_map<std::size_t, ModelHandle> path_cache_;
};
