#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

#include <entt/entity/entity.hpp>

class SelectionContext {
public:
    struct Snapshot {
        std::vector<entt::entity> entities;
        entt::entity primary = entt::null;
        std::uint64_t version = 0;
    };

    class Transaction {
    public:
        [[nodiscard]] auto contains(entt::entity entity) const -> bool;

        auto set(entt::entity entity, bool selected) -> void;
        auto toggle(entt::entity entity) -> void;
        auto clear() -> void;

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

    [[nodiscard]] auto version() const noexcept -> std::uint64_t { return version_.load(std::memory_order_acquire); }

    [[nodiscard]] auto contains(entt::entity entity) const -> bool;
    [[nodiscard]] auto size() const -> std::size_t;
    [[nodiscard]] auto empty() const -> bool { return size() == 0; }

    auto select(entt::entity entity) -> void;
    auto toggle(entt::entity entity) -> void;
    auto add(entt::entity entity) -> void;
    auto remove(entt::entity entity) -> void;
    auto clear() -> void;

    auto assign(std::span<entt::entity const> entities, entt::entity primary = entt::null) -> void;

    template<typename F>
    auto modify(F &&edit) -> void {
        std::scoped_lock const lock{mutex_};

        auto entities = entities_;
        auto primary = primary_.load(std::memory_order_relaxed);

        Transaction transaction{entities, primary};
        static_cast<F &&>(edit)(transaction);

        publish(std::move(entities), primary);
    }

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
    auto publish(std::vector<entt::entity> entities, entt::entity primary) -> void;

    mutable std::mutex mutex_;
    std::vector<entt::entity> entities_;
    std::atomic<entt::entity> primary_{entt::entity{entt::null}};
    std::atomic<std::uint64_t> version_{0};
};

[[nodiscard]] auto selection_context() -> SelectionContext &;
