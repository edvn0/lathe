#pragma once

#include <expected>

#include <volk.h>

#include "assets/load_model.hxx" // ModelCpuData, ModelLoadError
#include "assets/model.hxx" // ModelHandle
#include "core/renderer_error.hxx"

struct SamplerStorage;

// Narrow surface of Renderer that ModelStreamer needs to reserve and
// install async model loads. model_streamer.hxx/.cxx depend on this instead
// of the full Renderer -- Renderer owns ModelStreamer (not the other way
// round), so ModelStreamer calling back into it directly would be a cycle.
struct IModelSink {
    [[nodiscard]]
    virtual auto create_pending_model(ModelHandle fallback) -> std::expected<ModelHandle, RendererError> = 0;

    [[nodiscard]]
    virtual auto finish_model_load(ModelHandle pending, ModelCpuData const &cpu_data, VkCommandBuffer command_buffer)
            -> std::expected<void, RendererError> = 0;

    [[nodiscard]]
    virtual auto sampler_storage() noexcept -> SamplerStorage & = 0;

protected:
    ~IModelSink() = default;
};
