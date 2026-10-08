#pragma once

#include <atomic>
#include <cstdint>

#include <entt/entt.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <optional>
#include <vector>

#include "assets/material.hxx"
#include "assets/model.hxx"
#include "core/transform.hxx"
#include "physics/physics_components.hxx"
#include "scene/script_handle.hxx"

namespace Components {
    struct PointLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
    };

    struct SpotLight {
        glm::vec3 colour{1.0F};
        float intensity = 1.0F;
        float range = 10.0F;
        float inner_cone_degrees = 20.0F;
        float outer_cone_degrees = 30.0F;
    };

    // The Scene holds a reference to every material here. Change an existing override through
    // Scene::set_material_override() so those references stay balanced.
    struct MaterialOverride {
        MaterialHandle material{};
        std::vector<MaterialSlotOverride> slots;

        [[nodiscard]]
        auto replacement_for(MaterialHandle source) const noexcept -> MaterialHandle {
            for (auto const &slot: slots) {
                if (slot.source == source) {
                    return slot.material;
                }
            }
            return material;
        }

        [[nodiscard]]
        auto empty() const noexcept -> bool {
            return !material.valid() && slots.empty();
        }
    };

    struct Model {
        ModelHandle model{};
    };

    // A fresh InstancedModel::revision.
    [[nodiscard]] inline auto next_instanced_model_revision() noexcept -> std::uint64_t {
        static std::atomic<std::uint64_t> counter{0};
        return counter.fetch_add(1, std::memory_order_relaxed) + 1;
    }

    // Many instances of one model with one material_override, owned by a single entity. Transforms are world-space
    // and fixed at spawn; Parent doesn't apply to individual instances.
    //
    // The renderer keeps the transforms on the GPU under `revision` (Renderer::submit_model_instances()), so whoever
    // changes `transforms` after spawning must call touch(). A copy shares the revision, which is fine: it holds the
    // same transforms until one of the two is touched.
    struct InstancedModel {
        ModelHandle model{};
        MaterialHandle material_override{};
        std::vector<glm::mat4> transforms;

        // GPU skinning: per instance, the index of its first matrix in the palette given to
        // Renderer::set_skin_palette() this frame (empty for unskinned models). Not part of `revision`: models
        // with a skin never use the resident path, and the offsets are resubmitted every frame.
        std::vector<std::uint32_t> palette_offsets;
        std::uint64_t revision = next_instanced_model_revision();

        auto touch() noexcept -> void { revision = next_instanced_model_revision(); }
    };

    struct Script {
        ScriptHandle script{};
    };

    struct PlayerTag {};

    // Draws the renderer's selected-object outline (Renderer::outline_settings) around this entity's Model. Add and
    // remove it at runtime; it isn't serialised or cloned into the runtime scene, so it only exists where a game
    // put it. Not applied to InstancedModel.
    struct Outlined {};

    // Bullets spawned by BasicGame::shoot_bullet(). Their names aren't unique, so they need a tag.
    struct BulletTag {};

    // The entity owns one reference on its Model handle (ref-counted through the model cache), which Scene releases
    // when the entity, the Model or the tag goes. Not for entities sharing an unretained handle.
    struct StreamedModelTag {};

    // For on_destroy<Model> and on_destroy<StreamedModelTag>: the model reference to release, if this removal is the
    // one that ends the entity's ownership. entt raises on_destroy before erasing, one pool at a time, so whichever
    // of the pair goes first finds its partner still present and returns the handle; the second finds it gone.
    [[nodiscard]]
    inline auto model_released_by_model_destroy(entt::registry const &registry, entt::entity entity)
            -> std::optional<ModelHandle> {
        if (!registry.all_of<StreamedModelTag>(entity)) {
            return std::nullopt;
        }

        return registry.get<Model>(entity).model;
    }

    [[nodiscard]]
    inline auto model_released_by_tag_destroy(entt::registry const &registry, entt::entity entity)
            -> std::optional<ModelHandle> {
        auto const *model = registry.try_get<Model>(entity);

        if (model == nullptr) {
            return std::nullopt;
        }

        return model->model;
    }
} // namespace Components
