#pragma once

#include <expected>
#include <string_view>

#include <volk.h>

#include "assets/load_model.hxx"
#include "assets/model.hxx"
#include "core/renderer_error.hxx"

class SamplerStorage;
class ImageStorage;
class TextureStreamer;
struct MaterialStorage;

struct IModelSink {
    [[nodiscard]]
    virtual auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> = 0;

    // Installs a finished Model into `pending` in place. Render thread only.
    [[nodiscard]]
    virtual auto install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> = 0;

    // Adds a reference, for a cache handing the same handle to another caller.
    virtual auto retain_model(ModelHandle handle) -> void = 0;

    // Registers `handle` under `name` in the AssetRegistry. A name collision is ignored.
    virtual auto register_model_name(ModelHandle handle, std::string_view name) -> void = 0;

    [[nodiscard]]
    virtual auto sampler_storage() noexcept -> SamplerStorage & = 0;

    // Resources ModelStreamer needs to run the GPU upload itself across frames.
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
