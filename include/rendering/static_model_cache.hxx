#pragma once

#include "assets/material.hxx"
#include "assets/model.hxx"

#include <entt/entt.hpp>

#include <glm/mat4x4.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// The Models that never move (Components::Static, on the entity or any parent), baked into one list of world
// transforms per model and material. The renderer keeps each list on the GPU while its revision is unchanged
// (Renderer::submit_model_instances), so a static scene costs one cheap lookup per batch per frame, not a world
// transform and a submission per entity. The bake is redone when Scene::static_revision changes, or when the entities
// that must be drawn on their own (the editor's selection, for the outline) change.
class StaticModelCache {
public:
    struct Batch {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::vector<glm::mat4> transforms;
        std::uint64_t revision = 0;
    };

    // Brings the bake up to date. `revision` is the registry's Scene::static_revision (unique across scenes, so a
    // different scene is a different revision). `excluded` is sorted: entities left out of the batches however static
    // they are. `has_model` says which models exist; an entity with one that does not is left to be submitted alone.
    auto update(entt::registry const &registry, std::uint64_t revision, std::span<entt::entity const> excluded,
                std::function<bool(ModelHandle)> const &has_model) -> void;

    // Forgets the bake, so every entity is drawn one by one again.
    auto clear() -> void;

    [[nodiscard]] auto batches() const noexcept -> std::span<Batch const> { return batches_; }

    // Whether the entity is part of a batch and so must not also be submitted by itself.
    [[nodiscard]] auto contains(entt::entity entity) const noexcept -> bool {
        auto const index = static_cast<std::size_t>(entt::to_entity(entity));
        return index < batched_.size() && batched_[index] != 0;
    }

    [[nodiscard]] auto entity_count() const noexcept -> std::size_t { return entity_count_; }

private:
    std::vector<Batch> batches_;
    std::vector<std::uint8_t> batched_;
    std::size_t entity_count_ = 0;

    std::uint64_t revision_ = 0;
    std::vector<entt::entity> excluded_;
    bool valid_ = false;
};
