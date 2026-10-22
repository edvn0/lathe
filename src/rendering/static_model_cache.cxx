#include "rendering/static_model_cache.hxx"

#include "physics/physics_components.hxx"
#include "rendering/scene.hxx"
#include "scene/components.hxx"

#include <algorithm>
#include <tuple>
#include <map>
#include <tuple>

#include <tracy/Tracy.hpp>

namespace {
    using GroupKey = std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>;

    [[nodiscard]] auto group_key(ModelHandle model, MaterialHandle material) -> GroupKey {
        return GroupKey{model.index, model.generation, material.index, material.generation};
    }

    // Static on the entity or on a parent above it.
    [[nodiscard]] auto under_static(entt::registry const &registry, entt::storage<Components::Static> const &statics,
                                    entt::entity entity) -> bool {
        for (auto depth = 0; depth < 64 && registry.valid(entity); ++depth) {
            if (statics.contains(entity)) {
                return true;
            }
            auto const *parent = registry.try_get<Components::Parent const>(entity);
            if (parent == nullptr) {
                return false;
            }
            entity = parent->entity;
        }
        return false;
    }
}

auto StaticModelCache::clear() -> void {
    batches_.clear();
    batched_.clear();
    excluded_.clear();
    entity_count_ = 0;
    valid_ = false;
}

auto StaticModelCache::update(entt::registry const &registry, std::uint64_t revision,
                              std::span<entt::entity const> excluded, std::function<bool(ModelHandle)> const &has_model)
        -> void {
    if (valid_ && revision_ == revision && std::ranges::equal(excluded, excluded_)) {
        return;
    }

    ZoneScopedNC("BakeStaticModels", tracy::Color::Teal);

    batches_.clear();
    entity_count_ = 0;
    std::ranges::fill(batched_, std::uint8_t{0});

    auto const *const statics = registry.storage<Components::Static>();
    if (statics != nullptr && !statics->empty()) {
        auto const *const outlines = registry.storage<Components::Outlined>();
        auto const *const overrides = registry.storage<Components::MaterialOverride>();
        auto const *const bodies = registry.storage<Components::RigidBody>();
        auto world_transforms = systems::WorldTransformCache{};
        world_transforms.begin_frame();

        auto groups = std::map<GroupKey, std::size_t>{};

        for (auto const [entity, transform, model]:
             registry.view<Components::Transform const, Components::Model const>().each()) {
            if (!under_static(registry, *statics, entity) || !has_model(model.model) ||
                (outlines != nullptr && outlines->contains(entity)) ||
                (bodies != nullptr && bodies->contains(entity)) ||
                std::ranges::binary_search(excluded, entity)) {
                continue;
            }

            auto material = MaterialHandle{};
            if (overrides != nullptr && overrides->contains(entity)) {
                auto const &override_component = overrides->get(entity);
                if (!override_component.slots.empty()) {
                    continue;
                }
                material = override_component.material;
            }

            auto const [group, inserted] = groups.try_emplace(group_key(model.model, material), batches_.size());
            if (inserted) {
                batches_.push_back(Batch{
                        .model = model.model,
                        .material_override = material,
                        .revision = Components::next_instanced_model_revision(),
                });
            }

            batches_[group->second].transforms.push_back(world_transforms.get(registry, entity, transform));

            auto const index = static_cast<std::size_t>(entt::to_entity(entity));
            if (index >= batched_.size()) {
                batched_.resize(index + 1, std::uint8_t{0});
            }
            batched_[index] = 1;
            ++entity_count_;
        }
    }

    revision_ = revision;
    excluded_.assign(excluded.begin(), excluded.end());
    valid_ = true;
}
