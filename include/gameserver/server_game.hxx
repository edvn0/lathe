#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>

#include "core/json.hxx"

namespace gameserver {

    using PlayerId = std::uint32_t;

    struct ActionError {
        std::string code;
        std::string message;
    };

    struct GameOutcome {
        // Empty for a draw or an outcome with no single winner.
        std::optional<PlayerId> winner;

        std::string reason;
    };

    // Pure game logic: no rendering, scene or transport types. One instance per room. Payloads are JSON so the server
    // never interprets game data and any client that can speak JSON can play.
    class IServerGame {
    public:
        virtual ~IServerGame() = default;

        [[nodiscard]] virtual auto player_count() const -> std::size_t = 0;

        // Called once when the room fills. Seat order is join order.
        virtual auto start(std::span<PlayerId const> players) -> void = 0;

        virtual auto apply_action(PlayerId player, JsonValue const &action) -> std::expected<void, ActionError> = 0;

        // The state as `player` is allowed to see it (hidden information stays hidden).
        virtual auto write_state(PlayerId player, JsonWriter &writer) const -> void = 0;

        [[nodiscard]] virtual auto outcome() const -> std::optional<GameOutcome> = 0;
    };

}
