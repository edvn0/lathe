#pragma once

#include <expected>
#include <filesystem>
#include <string_view>

#include <volk.h>

#include "assets/load_model.hxx"
#include "core/paths.hxx"
#include "assets/model.hxx"
#include "core/renderer_error.hxx"

class SamplerStorage;
class ImageStorage;
class TextureStreamer;
struct MaterialStorage;

struct IModelSink {
    [[nodiscard]]
    virtual auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> = 0;

    [[nodiscard]]
    virtual auto install_model(ModelHandle pending, Model const &model) -> std::expected<void, RendererError> = 0;

    virtual auto retain_model(ModelHandle handle) -> void = 0;

    virtual auto release_model(ModelHandle handle) -> void = 0;

    virtual auto register_model_name(ModelHandle handle, std::string_view name) -> void = 0;

    virtual auto register_model_source(ModelHandle handle, AssetPath const &source) -> void = 0;

    [[nodiscard]]
    virtual auto sampler_storage() noexcept -> SamplerStorage & = 0;

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
