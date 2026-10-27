#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace event_bus_detail {
    inline auto next_event_id() noexcept -> std::size_t {
        static std::size_t counter = 0;

        return counter++;
    }
} // namespace event_bus_detail

// CRTP base of every event. Derived declares `static constexpr std::string_view name`.
template<typename Derived>
struct Event {
    [[nodiscard]] static auto id() noexcept -> std::size_t {
        static auto const value = event_bus_detail::next_event_id();

        return value;
    }

    [[nodiscard]] static constexpr auto event_name() noexcept -> std::string_view { return Derived::name; }
};

struct EventMetrics {
    std::string_view name;

    std::uint64_t published = 0;
    // Published events a handler returned true for, which stopped them reaching later subscribers.
    std::uint64_t consumed = 0;
    std::uint64_t handler_calls = 0;
    std::uint64_t total_ns = 0;
    std::uint64_t max_ns = 0;
    std::size_t subscribers = 0;
};

// Synchronous publish/subscribe. Handlers run lowest priority value first (ties in subscription order); a handler
// returning true consumes the event and later handlers do not see it.
class EventBus {
public:
    class Subscription {
    public:
        Subscription() = default;
        Subscription(Subscription const &) = delete;
        auto operator=(Subscription const &) -> Subscription & = delete;
        Subscription(Subscription &&other) noexcept : bus(other.bus), event(other.event), token(other.token) {
            other.bus = nullptr;
        }
        auto operator=(Subscription &&other) noexcept -> Subscription & {
            if (this != &other) {
                reset();
                bus = std::exchange(other.bus, nullptr);
                event = other.event;
                token = other.token;
            }

            return *this;
        }
        ~Subscription() { reset(); }

        auto reset() noexcept -> void {
            if (bus != nullptr) {
                bus->unsubscribe(event, token);
                bus = nullptr;
            }
        }

    private:
        friend class EventBus;

        Subscription(EventBus *b, std::size_t e, std::uint64_t t) noexcept : bus(b), event(e), token(t) {}

        EventBus *bus = nullptr;
        std::size_t event = 0;
        std::uint64_t token = 0;
    };

    // The returned Subscription unsubscribes on destruction; keep it as long as the handler's captures live.
    template<typename E, typename F>
    [[nodiscard]] auto subscribe(F &&handler, int priority = 0) -> Subscription {
        auto &ch = channel_for<E>();
        auto const token = next_token++;

        auto const at = std::ranges::upper_bound(ch.handlers, priority, {}, &Handler::priority);
        ch.handlers.insert(at, Handler{
                                   .token = token,
                                   .priority = priority,
                                   .fn = [fn = std::forward<F>(handler)](void const *ev) mutable -> bool {
                                       return fn(*static_cast<E const *>(ev));
                                   },
                               });
        ch.metrics.subscribers = ch.handlers.size();

        return {this, E::id(), token};
    }

    // Returns true if a handler consumed the event.
    template<typename E>
    auto publish(E const &event) -> bool {
        auto &ch = channel_for<E>();
        ++ch.metrics.published;

        // Index loop: a handler may subscribe or unsubscribe during dispatch.
        for (std::size_t i = 0; i < ch.handlers.size(); ++i) {
            auto const start = std::chrono::steady_clock::now();
            auto const consumed = ch.handlers[i].fn(&event);
            auto const ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());

            ++ch.metrics.handler_calls;
            ch.metrics.total_ns += ns;
            ch.metrics.max_ns = std::max(ch.metrics.max_ns, ns);

            if (consumed) {
                ++ch.metrics.consumed;

                return true;
            }
        }

        return false;
    }

    // Metrics for every event type seen so far.
    [[nodiscard]] auto metrics() const -> std::vector<EventMetrics> {
        auto out = std::vector<EventMetrics>{};

        for (auto const &ch: channels) {
            if (ch != nullptr) {
                out.push_back(ch->metrics);
            }
        }

        return out;
    }

    auto reset_metrics() -> void {
        for (auto &ch: channels) {
            if (ch != nullptr) {
                auto const name = ch->metrics.name;
                auto const subs = ch->metrics.subscribers;

                ch->metrics = {.name = name, .subscribers = subs};
            }
        }
    }

private:
    struct Handler {
        std::uint64_t token{};
        int priority{};
        std::function<bool(void const *)> fn;
    };

    struct Channel {
        std::vector<Handler> handlers;
        EventMetrics metrics;
    };

    template<typename E>
    auto channel_for() -> Channel & {
        if (channels.size() <= E::id()) {
            channels.resize(E::id() + 1);
        }

        auto &ch = channels[E::id()];

        if (ch == nullptr) {
            ch = std::make_unique<Channel>();
            ch->metrics.name = E::event_name();
        }

        return *ch;
    }

    auto unsubscribe(std::size_t event, std::uint64_t token) noexcept -> void {
        if (event >= channels.size() || channels[event] == nullptr) {
            return;
        }

        auto &ch = *channels[event];

        std::erase_if(ch.handlers, [token](Handler const &h) { return h.token == token; });
        ch.metrics.subscribers = ch.handlers.size();
    }

    std::vector<std::unique_ptr<Channel>> channels;
    std::uint64_t next_token = 1;
};
