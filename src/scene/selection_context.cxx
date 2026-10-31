#include "scene/selection_context.hxx"

#include <algorithm>
#include <utility>

auto SelectionContext::Transaction::contains(entt::entity entity) const -> bool {
    return std::ranges::find(entities_, entity) != entities_.end();
}

auto SelectionContext::Transaction::set(entt::entity entity, bool selected) -> void {
    if (entity == entt::null) {
        return;
    }

    auto const it = std::ranges::find(entities_, entity);

    if (selected) {
        if (it == entities_.end()) {
            entities_.push_back(entity);
        }
        primary_ = entity;
        return;
    }

    if (it == entities_.end()) {
        return;
    }

    entities_.erase(it);
    if (primary_ == entity) {
        primary_ = entities_.empty() ? entt::entity{entt::null} : entities_.back();
    }
}

auto SelectionContext::Transaction::toggle(entt::entity entity) -> void { set(entity, !contains(entity)); }

auto SelectionContext::Transaction::clear() -> void {
    entities_.clear();
    primary_ = entt::null;
}

auto SelectionContext::Transaction::set_primary(entt::entity entity) -> void {
    if (contains(entity)) {
        primary_ = entity;
    }
}

auto SelectionContext::snapshot() const -> Snapshot {
    std::scoped_lock const lock{mutex_};

    return Snapshot{
            .entities = entities_,
            .primary = primary_.load(std::memory_order_relaxed),
            .submesh = submesh_,
            .version = version_.load(std::memory_order_relaxed),
    };
}

auto SelectionContext::contains(entt::entity entity) const -> bool {
    std::scoped_lock const lock{mutex_};
    return std::ranges::find(entities_, entity) != entities_.end();
}

auto SelectionContext::size() const -> std::size_t {
    std::scoped_lock const lock{mutex_};
    return entities_.size();
}

auto SelectionContext::select_submesh(entt::entity entity, MaterialHandle material) -> void {
    std::scoped_lock const lock{mutex_};

    auto const before = submesh_;
    submesh_ = SubmeshHighlight{.entity = entity, .material = material};
    publish(entity == entt::null ? std::vector<entt::entity>{} : std::vector<entt::entity>{entity}, entity);

    if (before != submesh_) {
        version_.fetch_add(1, std::memory_order_acq_rel);
    }
}

auto SelectionContext::clear_submesh() -> void {
    std::scoped_lock const lock{mutex_};

    if (submesh_ != SubmeshHighlight{}) {
        submesh_ = {};
        version_.fetch_add(1, std::memory_order_acq_rel);
    }
}

auto SelectionContext::submesh_highlight() const -> SubmeshHighlight {
    std::scoped_lock const lock{mutex_};
    return submesh_;
}

auto SelectionContext::select(entt::entity entity) -> void {
    clear_submesh();
    modify([entity](Transaction &transaction) {
        transaction.clear();
        transaction.set(entity, true);
    });
}

auto SelectionContext::toggle(entt::entity entity) -> void {
    modify([entity](Transaction &transaction) { transaction.toggle(entity); });
}

auto SelectionContext::add(entt::entity entity) -> void {
    modify([entity](Transaction &transaction) { transaction.set(entity, true); });
}

auto SelectionContext::remove(entt::entity entity) -> void {
    modify([entity](Transaction &transaction) { transaction.set(entity, false); });
}

auto SelectionContext::clear() -> void {
    modify([](Transaction &transaction) { transaction.clear(); });
}

auto SelectionContext::assign(std::span<entt::entity const> entities, entt::entity primary) -> void {
    modify([&](Transaction &transaction) {
        transaction.clear();
        for (auto const entity: entities) {
            transaction.set(entity, true);
        }
        if (primary != entt::null) {
            transaction.set_primary(primary);
        }
    });
}

auto SelectionContext::publish(std::vector<entt::entity> entities, entt::entity primary) -> void {
    // The singled-out slot only holds while its entity is all that is selected.
    auto const highlight_holds = submesh_.entity != entt::null && entities.size() == 1 && entities.front() == submesh_.entity;
    auto const highlight = highlight_holds ? submesh_ : SubmeshHighlight{};

    if (entities == entities_ && primary == primary_.load(std::memory_order_relaxed) && highlight == submesh_) {
        return;
    }

    submesh_ = highlight;

    entities_ = std::move(entities);
    primary_.store(primary, std::memory_order_release);
    version_.fetch_add(1, std::memory_order_acq_rel);
}

auto selection_context() -> SelectionContext & {
    static SelectionContext context;
    return context;
}
