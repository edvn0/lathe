#pragma once

#include <expected>
#include <string_view>

#include "core/forward.hxx"

#include "assets/material_storage.hxx"
#include "assets/model.hxx"
#include "core/renderer_error.hxx"

struct EngineModels {
    ModelHandle cube;
    ModelHandle sphere;
    ModelHandle grass_clump;
    ModelHandle capsule;

    ImageHandle grass_card_texture;
};

[[nodiscard]] auto create_engine_models(Renderer &renderer) -> std::expected<EngineModels, RendererError>;

struct GrassMaterials {
    MaterialHandle blades;
    MaterialHandle cards;
};

[[nodiscard]] auto grass_materials(Renderer &renderer, EngineModels const &engine_models,
                                   MaterialCreateInfo const &look, GrassMaterials existing = {},
                                   std::string_view name = "grass") -> std::expected<GrassMaterials, RendererError>;
