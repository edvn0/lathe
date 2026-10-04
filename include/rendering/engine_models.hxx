#pragma once

#include <expected>
#include <string_view>

#include "core/forward.hxx"

#include "assets/material_storage.hxx"
#include "assets/model.hxx"
#include "core/renderer_error.hxx"

// Built-in primitive models, created once at startup. Failure is fatal: these are generated in-process.
struct EngineModels {
    ModelHandle cube;
    ModelHandle sphere;
    ModelHandle grass_clump;
    ModelHandle capsule;

    // The grass clump's far-LOD card texture (make_grass_card_texture()). Use it through grass_materials().
    ImageHandle grass_card_texture;
};

[[nodiscard]] auto create_engine_models(Renderer &renderer) -> std::expected<EngineModels, RendererError>;

// The two materials the grass clump draws with: its blades (double-sided, opaque) and, from grass_clump_card_lod on,
// its cards (the card texture, alpha-tested with alpha to coverage). `blades` names the second as its far material.
struct GrassMaterials {
    MaterialHandle blades;
    MaterialHandle cards;
};

// Creates the grass clump's materials from `look` (its colour, wind, shadows, ...), or, where `existing` has valid
// handles, updates those in place. Textures, alpha and culling settings in `look` are replaced by what the clump
// needs. New materials are registered as `name` and `name`_cards unless `name` is empty. The caller owns one
// reference to each material; `blades` holds another to `cards`.
[[nodiscard]] auto grass_materials(Renderer &renderer, EngineModels const &engine_models,
                                   MaterialCreateInfo const &look, GrassMaterials existing = {},
                                   std::string_view name = "grass") -> std::expected<GrassMaterials, RendererError>;
