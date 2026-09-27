#pragma once

#include <expected>

#include "core/forward.hxx"

#include "assets/model.hxx"
#include "core/renderer_error.hxx"

// Built-in primitive models, created once at startup. Failure is fatal: these are generated in-process.
struct EngineModels {
    ModelHandle cube;
    ModelHandle sphere;
    ModelHandle grass_clump;
    ModelHandle capsule;
};

[[nodiscard]] auto create_engine_models(Renderer &renderer) -> std::expected<EngineModels, RendererError>;
