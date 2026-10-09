#pragma once

#include <array>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "chess_engine.hxx"
#include "gameserver/server_game.hxx"

// Chess as a headless IServerGame. Seat 0 plays white, seat 1 plays black.
//
// action: {"from":"e2","to":"e4","promotion":"q"}   promotion is optional (q, r, b, n; default q)
// state:  {"board":[8 rank strings, rank 8 first, FEN letters, '.' empty],"side_to_move":"white","your_side":"black",
//          "status":"playing","in_check":false,"history":["e2e4","e7e8q",...],"legal_moves":["e2e4",...]}
// history is every move played so far, so a client can replay it (or resume mid-game); legal_moves is only for the mover.
class ChessServerGame final : public gameserver::IServerGame {
public:
    [[nodiscard]] static auto create() -> std::unique_ptr<gameserver::IServerGame>;

    [[nodiscard]] auto player_count() const -> std::size_t override { return 2; }

    auto start(std::span<gameserver::PlayerId const> players) -> void override;

    auto apply_action(gameserver::PlayerId player, JsonValue const &action)
            -> std::expected<void, gameserver::ActionError> override;

    auto write_state(gameserver::PlayerId player, JsonWriter &writer) const -> void override;

    [[nodiscard]] auto outcome() const -> std::optional<gameserver::GameOutcome> override;

private:
    [[nodiscard]] auto side_of(gameserver::PlayerId player) const -> std::optional<chess::Side>;

    chess::ChessEngine engine_;

    std::array<gameserver::PlayerId, 2> players_{};

    std::vector<std::string> history_;
};
