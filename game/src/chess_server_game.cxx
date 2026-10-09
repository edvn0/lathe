#include "chess_server_game.hxx"

#include <format>
#include <string>
#include <string_view>
#include <utility>

namespace {
    using gameserver::ActionError;

    auto parse_square(std::string_view text) -> std::optional<chess::Square> {
        if (text.size() != 2 || text[0] < 'a' || text[0] > 'h' || text[1] < '1' || text[1] > '8') {
            return std::nullopt;
        }

        return static_cast<chess::Square>(((text[1] - '1') * 8) + (text[0] - 'a'));
    }

    auto square_name(chess::Square square) -> std::string {
        auto const index = static_cast<int>(square);

        return std::format("{}{}", static_cast<char>('a' + (index & 7)), static_cast<char>('1' + (index >> 3)));
    }

    auto piece_letter(chess::Piece piece) -> char {
        using enum chess::Piece;

        switch (piece) {
            case white_pawn:
                return 'P';
            case white_knight:
                return 'N';
            case white_bishop:
                return 'B';
            case white_rook:
                return 'R';
            case white_queen:
                return 'Q';
            case white_king:
                return 'K';
            case black_pawn:
                return 'p';
            case black_knight:
                return 'n';
            case black_bishop:
                return 'b';
            case black_rook:
                return 'r';
            case black_queen:
                return 'q';
            case black_king:
                return 'k';
            case none:
                break;
        }

        return '.';
    }

    auto side_name(chess::Side side) -> std::string_view { return side == chess::Side::white ? "white" : "black"; }

    auto status_name(chess::GameState state) -> std::string_view {
        using enum chess::GameState;

        switch (state) {
            case playing:
                return "playing";
            case checkmate:
                return "checkmate";
            case stalemate:
                return "stalemate";
            case draw_fifty_move:
                return "draw_fifty_move";
            case draw_insufficient_material:
                return "draw_insufficient_material";
            case draw_repetition:
                return "draw_repetition";
        }

        return "playing";
    }

    auto promotion_letter(chess::PieceType type) -> std::string_view {
        switch (type) {
            case chess::PieceType::rook:
                return "r";
            case chess::PieceType::bishop:
                return "b";
            case chess::PieceType::knight:
                return "n";
            default:
                return "q";
        }
    }

    auto parse_promotion(std::string_view text) -> std::optional<chess::PieceType> {
        if (text.empty() || text == "q") {
            return chess::PieceType::queen;
        }

        if (text == "r") {
            return chess::PieceType::rook;
        }

        if (text == "b") {
            return chess::PieceType::bishop;
        }

        if (text == "n") {
            return chess::PieceType::knight;
        }

        return std::nullopt;
    }
}

auto ChessServerGame::create() -> std::unique_ptr<gameserver::IServerGame> { return std::make_unique<ChessServerGame>(); }

auto ChessServerGame::start(std::span<gameserver::PlayerId const> players) -> void {
    engine_.reset();

    players_ = {players[0], players[1]};

    history_.clear();
}

auto ChessServerGame::side_of(gameserver::PlayerId player) const -> std::optional<chess::Side> {
    if (player == players_[0]) {
        return chess::Side::white;
    }

    if (player == players_[1]) {
        return chess::Side::black;
    }

    return std::nullopt;
}

auto ChessServerGame::apply_action(gameserver::PlayerId player, JsonValue const &action)
        -> std::expected<void, ActionError> {
    auto const side = side_of(player);

    if (!side) {
        return std::unexpected(ActionError{.code = "not_a_player", .message = "you are not seated in this game"});
    }

    if (chess::is_game_over(engine_.game_state())) {
        return std::unexpected(ActionError{.code = "game_over", .message = "the game has ended"});
    }

    if (*side != engine_.side_to_move()) {
        return std::unexpected(ActionError{.code = "not_your_turn", .message = "it is not your turn"});
    }

    auto const from = parse_square(action["from"].as_string());
    auto const to = parse_square(action["to"].as_string());

    if (!from || !to) {
        return std::unexpected(ActionError{.code = "bad_action", .message = "from and to must be squares like e2"});
    }

    auto const promotion = parse_promotion(action["promotion"].as_string());

    if (!promotion) {
        return std::unexpected(ActionError{.code = "bad_action", .message = "promotion must be one of q, r, b, n"});
    }

    auto const result = engine_.update(*from, *to, *promotion);

    if (!result.moved) {
        return std::unexpected(ActionError{.code = "illegal_move", .message = "that move is not legal"});
    }

    history_.push_back(std::format("{}{}{}", square_name(*from), square_name(*to),
                                   result.promotion ? promotion_letter(*promotion) : ""));

    return {};
}

auto ChessServerGame::write_state(gameserver::PlayerId player, JsonWriter &writer) const -> void {
    writer.begin_array("board", true);

    for (int rank = 7; rank >= 0; --rank) {
        std::string row;

        for (int file = 0; file < 8; ++file) {
            row.push_back(piece_letter(engine_.piece_at(static_cast<chess::Square>((rank * 8) + file))));
        }

        writer.value({}, row);
    }

    writer.end_array();

    writer.value("side_to_move", side_name(engine_.side_to_move()));

    if (auto const side = side_of(player)) {
        writer.value("your_side", side_name(*side));
    }

    writer.value("status", status_name(engine_.game_state()));
    writer.value("in_check", engine_.render_state().in_check);

    writer.begin_array("history", true);

    for (auto const &move : history_) {
        writer.value({}, move);
    }

    writer.end_array();

    writer.begin_array("legal_moves", true);

    if (side_of(player) == engine_.side_to_move()) {
        for (auto const move : engine_.legal_moves()) {
            writer.value({}, std::format("{}{}", square_name(move.from), square_name(move.to)));
        }
    }

    writer.end_array();
}

auto ChessServerGame::outcome() const -> std::optional<gameserver::GameOutcome> {
    auto const state = engine_.game_state();

    if (!chess::is_game_over(state)) {
        return std::nullopt;
    }

    gameserver::GameOutcome result{.winner = std::nullopt, .reason = std::string{status_name(state)}};

    if (state == chess::GameState::checkmate) {
        // The side to move is the one that was mated.
        result.winner = engine_.side_to_move() == chess::Side::white ? players_[1] : players_[0];
    }

    return result;
}
