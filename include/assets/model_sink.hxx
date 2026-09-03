#pragma once

#include <expected>

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
