#pragma once

#include <expected>

#include "core/forward.hxx"

#include "assets/model.hxx"
#include "core/renderer_error.hxx"

// Built-in primitive models the engine always has available (e.g. as a
// fallback when a real asset fails to load). Created once at startup via
// create_engine_models; unlike load_model, failure here is treated as fatal
// by the caller — these are generated in-process, so a failure means the
// renderer itself is broken, not that an asset is missing.
struct EngineModels {
    ModelHandle cube;
    ModelHandle sphere;
    ModelHandle grass_clump;
    ModelHandle capsule;
};

[[nodiscard]] auto create_engine_models(Renderer &renderer) -> std::expected<EngineModels, RendererError>;
