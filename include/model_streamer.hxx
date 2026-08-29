#pragma once

#include <volk.h>

#include <expected>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

#include "forward.hxx"
#include "load_model.hxx"
#include "model.hxx"

// Kicks off async CPU-side loading (glTF parse, tangent generation, LOD
// simplification) for a model on Renderer::thread_pool(), handing back a
// handle that renders as `fallback` until the background work and GPU
// upload both finish and get installed in place -- mirrors TextureStreamer,
// but the GPU upload half still needs a live command buffer each frame
// (process_ready), since model uploads aren't staged through a separate
// per-frame retirement queue the way texture uploads are.
class ModelStreamer {
public:
    [[nodiscard]]
    auto request(Renderer &renderer, std::filesystem::path source_path, ModelHandle fallback, std::string debug_name)
            -> ModelHandle;

    // Finalizes every request whose background ModelCpuData is ready:
    // records its GPU upload into `command_buffer` and installs it into the
    // still-pending handle in place. Call once per frame -- command_buffer
    // must be a currently-recording command buffer this frame will submit
    // and wait on through the normal frames-in-flight fence discipline.
    auto process_ready(Renderer &renderer, VkCommandBuffer command_buffer) -> void;

    // Blocks until every outstanding background job finishes, without
    // uploading anything. Call before destroying the Renderer (and its
    // storages) this streamer's requests point into.
    auto wait_all() -> void;

private:
    struct PendingRequest {
        ModelHandle handle;
        std::string debug_name;
        std::future<std::expected<ModelCpuData, ModelLoadError>> future;
    };

    std::vector<PendingRequest> pending_;
};
