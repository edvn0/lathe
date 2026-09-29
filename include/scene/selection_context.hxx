#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include <entt/entity/entity.hpp>

// The editor's entity selection. Every mutation is one atomic step: it runs under a lock and publishes a new
// version, so a reader on any thread sees the selection from before or after it, never half of a multi-entity edit.
//
// Entities keep the order they were selected in. The primary entity (the one the Inspector and gizmo act on) is
// the most recently selected one still in the selection.
class SelectionContext {
public:
    struct Snapshot {
        std::vector<entt::entity> entities;
        entt::entity primary = entt::null;
        std::uint64_t version = 0;
    };

    // Edits applied by modify(); published together when it returns.
    class Transaction {
    public:
        [[nodiscard]] auto contains(entt::entity entity) const -> bool;

        // Selecting makes `entity` primary; deselecting the primary falls back to the latest remaining entity.
        auto set(entt::entity entity, bool selected) -> void;
        auto toggle(entt::entity entity) -> void;
        auto clear() -> void;

        // No-op unless `entity` is selected.
        auto set_primary(entt::entity entity) -> void;

        [[nodiscard]] auto entities() const noexcept -> std::span<entt::entity const> { return entities_; }

    private:
        friend class SelectionContext;

        Transaction(std::vector<entt::entity> &entities, entt::entity &primary) :
            entities_(entities), primary_(primary) {}

        std::vector<entt::entity> &entities_;
        entt::entity &primary_;
    };

    SelectionContext() = default;

    SelectionContext(SelectionContext const &) = delete;
    auto operator=(SelectionContext const &) -> SelectionContext & = delete;
    SelectionContext(SelectionContext &&) = delete;
    auto operator=(SelectionContext &&) -> SelectionContext & = delete;
    ~SelectionContext() = default;

    [[nodiscard]] auto snapshot() const -> Snapshot;

    [[nodiscard]] auto primary() const noexcept -> entt::entity { return primary_.load(std::memory_order_acquire); }

    // Bumped by every mutation that changed something.
    [[nodiscard]] auto version() const noexcept -> std::uint64_t { return version_.load(std::memory_order_acquire); }

    [[nodiscard]] auto contains(entt::entity entity) const -> bool;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto empty() const -> bool { return size() == 0; }

    // Replaces the selection with `entity` alone; entt::null clears it.
    auto select(entt::entity entity) -> void;
    auto toggle(entt::entity entity) -> void;
    auto add(entt::entity entity) -> void;
    auto remove(entt::entity entity) -> void;
    auto clear() -> void;

    // Replaces the selection with `entities`; `primary` defaults to the last of them.
    auto assign(std::span<entt::entity const> entities, entt::entity primary = entt::null) -> void;

    // Runs `edit(Transaction &)` under the lock and publishes its result as one change.
    template<typename F>
    auto modify(F &&edit) -> void {
        std::scoped_lock const lock{mutex_};

        auto entities = entities_;
        auto primary = primary_.load(std::memory_order_relaxed);

        Transaction transaction{entities, primary};
        static_cast<F &&>(edit)(transaction);

        publish(std::move(entities), primary);
    }

    // Drops every entity `keep` rejects, e.g. ones destroyed or from a registry that is no longer active. `keep`
    // runs under the lock, so it must not call back into the context.
    template<typename Predicate>
    auto retain_if(Predicate &&keep) -> void {
        modify([&](Transaction &transaction) {
            auto const current =
                    std::vector<entt::entity>{transaction.entities().begin(), transaction.entities().end()};
            for (auto const entity: current) {
                if (!keep(entity)) {
                    transaction.set(entity, false);
                }
            }
        });
    }

private:
    // Caller holds mutex_.
    auto publish(std::vector<entt::entity> entities, entt::entity primary) -> void;

    mutable std::mutex mutex_;
    std::vector<entt::entity> entities_;
    std::atomic<entt::entity> primary_{entt::entity{entt::null}};
    std::atomic<std::uint64_t> version_{0};
};

// The process-wide editor selection.
[[nodiscard]] auto selection_context() -> SelectionContext &;
