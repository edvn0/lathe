#pragma once

#include <expected>
#include <string_view>

#include <volk.h>

#include "assets/load_model.hxx" // ModelCpuData, ModelLoadError, Model
#include "assets/model.hxx" // ModelHandle
#include "core/renderer_error.hxx"

class SamplerStorage;
class ImageStorage;
class TextureStreamer;
struct MaterialStorage;

struct IModelSink {
    [[nodiscard]]
    virtual auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> = 0;

    // Installs a Model finished by step_model_gpu_upload() into `pending` in
    // place. Must run on the render thread.
    [[nodiscard]]
    virtual auto install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> = 0;

    // Bumps `handle`'s ModelSlotData::ref_count -- called by ModelStreamer
    // when its path_cache_ hands the same handle out to a second caller
    // instead of loading a fresh copy, so a later destroy_model() call from
    // either caller doesn't tear the model down while the other still
    // references it.
    virtual auto retain_model(ModelHandle handle) -> void = 0;

    // Registers `handle` under `name` in the sink's AssetRegistry (see
    // asset_registry.hxx), so editor UI can offer it by name later --
    // called by ModelStreamer::process_ready() once a request finishes
    // installing, using the same debug_name the request was made with. A
    // no-op collision (name already taken) is fine here; the model is still
    // usable, just not name-addressable a second way.
    virtual auto register_model_name(ModelHandle handle, std::string_view name) -> void = 0;

    [[nodiscard]]
    virtual auto sampler_storage() noexcept -> SamplerStorage & = 0;

    // Resource providers ModelStreamer needs to drive
    // start_model_gpu_upload()/step_model_gpu_upload() itself, spreading a
    // model's GPU upload across multiple frames instead of doing it all in
    // one process_ready() call.
    [[nodiscard]]
    virtual auto image_storage() noexcept -> ImageStorage & = 0;
    [[nodiscard]]
    virtual auto texture_streamer() noexcept -> TextureStreamer & = 0;
    [[nodiscard]]
    virtual auto material_storage() noexcept -> MaterialStorage & = 0;
    [[nodiscard]]
    virtual auto geometry_arena() noexcept -> GeometryArena & = 0;

protected:
    ~IModelSink() = default;
};
