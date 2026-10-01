#include <algorithm>
#include <string>
#include <string_view>

#include "core/transform.hxx"
#include "rendering/entity.hxx"
#include "scripting/script_runtime.hxx"

namespace scripting::detail {
    auto EntityIndex::refresh(ScriptWorld world) -> bool {
        if (built_ && registry_ == world.registry && revision_ == world.hierarchy_revision) {
            return false;
        }

        clear();
        registry_ = world.registry;
        revision_ = world.hierarchy_revision;
        built_ = true;

        if (world.registry == nullptr) {
            return true;
        }

        auto &registry = *world.registry;

        auto const add = [this](std::string_view name, entt::entity entity) {
            if (name.empty()) {
                return;
            }
            auto const [it, inserted] = by_name_.try_emplace(std::string{name}, entity);
            if (!inserted && entt::to_integral(entity) < entt::to_integral(it->second)) {
                it->second = entity;
            }
        };

        // GeneratedMeta wins over Meta when an entity carries both, as in the editor's display name.
        for (auto [entity, meta]: registry.view<Components::GeneratedMeta const>().each()) {
            add(meta.name, entity);
        }
        for (auto [entity, meta]: registry.view<Components::Meta const>().each()) {
            if (!registry.all_of<Components::GeneratedMeta>(entity)) {
                add(meta.name.view(), entity);
            }
        }

        for (auto [entity, parent]: registry.view<Components::Parent const>().each()) {
            if (registry.valid(parent.entity)) {
                children_[parent.entity].push_back(entity);
            }
        }

        for (auto &entry: children_) {
            std::ranges::sort(entry.second, {}, [](entt::entity entity) { return entt::to_integral(entity); });
        }

        names_.reserve(by_name_.size());
        for (auto const &entry: by_name_) {
            names_.push_back(entry.first);
        }
        std::ranges::sort(names_);

        return true;
    }

    auto EntityIndex::find(std::string_view name) const -> entt::entity {
        auto const found = by_name_.find(name);
        if (found == by_name_.end()) {
            return entt::null;
        }
        return found->second;
    }

    auto EntityIndex::children(entt::entity parent) const -> std::span<entt::entity const> {
        auto const found = children_.find(parent);
        if (found == children_.end()) {
            return {};
        }
        return found->second;
    }

    auto EntityIndex::names() const noexcept -> std::span<std::string const> { return names_; }

    auto EntityIndex::clear() -> void {
        registry_ = nullptr;
        revision_ = 0;
        built_ = false;
        by_name_.clear();
        children_.clear();
        names_.clear();
    }
} // namespace scripting::detail
