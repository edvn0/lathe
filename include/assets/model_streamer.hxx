#pragma once

#include <volk.h>

#include <chrono>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/model.hxx"
#include "assets/model_load_profile.hxx"
#include "assets/model_sink.hxx"

// Loads models on thread_pool() and hands back a handle that renders as `fallback` until the model is
// installed in place. After the background parse, process_ready() on the render thread drives two phases:
//   - finalization: tangents and LOD simplification, one parallel task per primitive;
//   - GPU upload: gpu_upload_items_per_frame materials/primitives per call, so large models spread the cost.
class ModelStreamer {
public:
    // Always profiled; the breakdown is logged once the model installs.
    //
    // A `source_path` that already finished loading (by weakly_canonical() path) returns the installed handle
    // without reloading. Failed requests aren't cached, so they can be retried.
    [[nodiscard]]
    auto request(IModelSink &sink, std::filesystem::path source_path, ModelHandle fallback, std::string debug_name)
            -> ModelHandle;

    // Drops path_cache_ entries for `handle`. Call when the model is destroyed.
    auto forget(ModelHandle handle) -> void;

    // Promotes finished background loads into the GPU-upload phase, steps every upload by up to
    // gpu_upload_items_per_frame, and installs finished models. Call once per frame with a command buffer that is
    // recording for this frame.
    auto process_ready(IModelSink &sink, VkCommandBuffer command_buffer) -> void;

    // Blocks until background jobs finish; partially uploaded requests are dropped. Call before destroying the
    // Renderer.
    auto wait_all() -> void;

private:
    struct PendingRequest {
        ModelHandle handle;
        std::string debug_name;
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
    };

    std::vector<PendingRequest> pending_;

    // Keyed like Renderer::load_model's model_cache_: hash_value() of the weakly_canonical() path.
    std::unordered_map<std::size_t, ModelHandle> path_cache_;
};
