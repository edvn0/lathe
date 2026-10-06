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
    // Installed, or never requested through this streamer.
    none,
    loading,
    // Stays on its fallback model; failure_reason() says why.
    failed,
};

// Loads models on thread_pool() and hands back a handle that renders as `fallback` until the model is
// installed in place. The streamer holds its own reference on the handle until the request settles, so a caller
// may drop the handle mid-load; the model is then destroyed once it installs or fails.
//
// After the background parse, process_ready() on the render thread drives two phases:
//   - finalization: tangents and LOD simplification, one parallel task per primitive;
//   - GPU upload: gpu_upload_items_per_frame materials/primitives per call, so large models spread the cost.
class ModelStreamer {
public:
    // Always profiled; the breakdown is logged once the model installs.
    //
    // A `source_path` that already finished loading (by weakly_canonical() path) returns the installed handle
    // without reloading. Failed requests aren't cached, so they can be retried. The returned handle always carries
    // one reference for the caller, released with IModelSink::release_model().
    [[nodiscard]]
    auto request(IModelSink &sink, AssetPath source_path, ModelHandle fallback,
                 FlyString debug_name) -> ModelHandle;

    // request() for CPU data produced elsewhere, e.g. a cooked model decoded from an asset pack. `cpu_data` must
    // resolve to finalized primitives (compressed vertices and meshlets built), so the finalization phase is skipped
    // and only the budgeted GPU upload runs. `source_path` keys the path cache and is recorded as the model's source,
    // so a later request() for the same file reuses the handle.
    [[nodiscard]]
    auto request_prepared(IModelSink &sink, std::future<std::expected<ModelCpuData, ModelLoadError>> cpu_data,
                          AssetPath source_path, ModelHandle fallback, FlyString debug_name) -> ModelHandle;

    // Drops path_cache_ and failure entries for `handle`. Call when the model is destroyed.
    auto forget(ModelHandle handle) -> void;

    [[nodiscard]]
    auto state(ModelHandle handle) const -> ModelRequestState;

    // Empty unless state(handle) is ModelRequestState::failed.
    [[nodiscard]]
    auto failure_reason(ModelHandle handle) const -> std::string_view;

    // Promotes finished background loads into the GPU-upload phase, steps every upload by up to
    // gpu_upload_items_per_frame, and installs finished models. Call once per frame with a command buffer that is
    // recording for this frame.
    auto process_ready(IModelSink &sink, VkCommandBuffer command_buffer) -> void;

    // Blocks until background jobs finish; partially uploaded requests are dropped, leaving their slots to the
    // Renderer's teardown. Call before destroying the Renderer.
    auto wait_all() -> void;

    // Models still loading or uploading.
    [[nodiscard]] auto pending_count() const noexcept -> std::size_t { return pending_.size(); }

private:
    struct PendingRequest {
        ModelHandle handle;
        FlyString debug_name;
        std::future<std::expected<ModelCpuData, ModelLoadError>> future;

        // Set once `future` resolves.
        std::optional<ModelPrimitiveFinalization> finalization;

        // Set once `finalization` finishes.
        std::optional<ModelGpuUpload> upload;

        // Set once installed. The request stays in `pending_` until its textures finish so the profile covers them.
        bool installed = false;

        std::shared_ptr<ModelLoadProfile> profile;
        std::chrono::steady_clock::time_point requested_at;

        std::size_t path_hash = 0;
        AssetPath source_path;

        // `future` yields finalized primitives; skip straight to the GPU upload.
        bool prepared = false;
    };

    // Shared by request() and request_prepared(): cache lookup and pending-slot reservation. `final` means the
    // handle needs no loading (cached, or the fallback).
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

    // Settles a request that won't install: records why and drops the streamer's reference.
    auto fail(IModelSink &sink, PendingRequest const &request, std::string reason) -> void;

    std::vector<PendingRequest> pending_;
    std::vector<FailedRequest> failed_;

    // Keyed like Renderer::load_model's model_cache_: hash_value() of the weakly_canonical() path.
    std::unordered_map<std::size_t, ModelHandle> path_cache_;
};
