#pragma once

#include <volk.h>

#include <chrono>
#include <expected>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "assets/load_model.hxx"
#include "assets/model.hxx"
#include "assets/model_load_profile.hxx"
#include "assets/model_sink.hxx"

// Kicks off async CPU-side loading (glTF parse + raw primitive extraction,
// see load_model_cpu_unfinalized()) for a model on thread_pool(), handing
// back a handle that renders as `fallback` until the background work and
// GPU upload both finish and get installed in place -- mirrors
// TextureStreamer, but the GPU upload half still needs a live command
// buffer each frame (process_ready), since model uploads aren't staged
// through a separate per-frame retirement queue the way texture uploads
// are.
//
// The rest of the load happens in two further phases, both driven from
// process_ready() (i.e. the render thread):
//   - Finalization: tangent generation + LOD simplification, one task per
//     primitive, run in parallel across thread_pool() (see
//     start_primitive_finalization()/step_primitive_finalization()).
//   - GPU upload: materials + mesh primitives, paced at
//     gpu_upload_items_per_frame per process_ready() call rather than run
//     to completion in one shot -- a model with many materials/primitives
//     (e.g. Sponza's ~25 materials and dozens of mesh primitives) spreads
//     that cost across several frames instead of spiking a single frame's
//     CPU/GPU work to the size of the whole model.
class ModelStreamer {
public:
    // Always profiled -- see ModelLoadProfile. The overhead is a handful of
    // atomic adds and steady_clock::now() calls per material/primitive/
    // texture, negligible next to the parse/encode/upload work itself, and
    // format_model_load_profile()'s breakdown is logged automatically once
    // the model finishes installing (see process_ready()).
    [[nodiscard]]
    auto request(IModelSink &sink, std::filesystem::path source_path, ModelHandle fallback, std::string debug_name)
            -> ModelHandle;

    // Advances every pending request by up to gpu_upload_items_per_frame
    // worth of GPU-upload work: promotes a request whose background
    // ModelCpuData just finished into its GPU-upload phase, then steps that
    // phase for every request still in it. Installs the finished Model into
    // its still-pending handle in place once a request's upload completes.
    // Call once per frame -- command_buffer must be a currently-recording
    // command buffer this frame will submit and wait on through the normal
    // frames-in-flight fence discipline.
    auto process_ready(IModelSink &sink, VkCommandBuffer command_buffer) -> void;

    // Blocks until every outstanding background job finishes, without
    // advancing anything still mid GPU-upload -- whatever a request's
    // upload had already recorded stays valid (its geometry/material slots
    // get torn down along with the rest of GeometryArena/MaterialStorage
    // right after this call, same as before), the rest is just discarded.
    // Call before destroying the Renderer (and its storages) this
    // streamer's requests point into.
    auto wait_all() -> void;

private:
    struct PendingRequest {
        ModelHandle handle;
        std::string debug_name;
        std::future<std::expected<ModelCpuData, ModelLoadError>> future;

        // Populated once `future` resolves, before `upload` -- see
        // start_primitive_finalization()/step_primitive_finalization().
        // Runs tangent generation and LOD simplification in parallel across
        // thread_pool(), one task per primitive, instead of the single
        // background thread `future` above did its own CPU parse on.
        std::optional<ModelPrimitiveFinalization> finalization;

        // Populated once `finalization` finishes -- see start_model_gpu_upload().
        std::optional<ModelGpuUpload> upload;

        // Set once install_model() succeeds. A request lingers in `pending_`
        // after that, still profiled, until every texture it kicked off
        // finishes (see the texture_count/expected_texture_count check in
        // process_ready()) -- geometry/materials installing doesn't mean
        // this model's textures are done, since those stream in
        // independently on thread_pool() workers.
        bool installed = false;

        std::shared_ptr<ModelLoadProfile> profile;
        std::chrono::steady_clock::time_point requested_at;
    };

    std::vector<PendingRequest> pending_;
};
