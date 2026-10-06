#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/badge.hxx"

class ChessGame;

namespace chess {

    using Bitboard = std::uint64_t;

    enum class Side : std::uint8_t {
        white,
        black,
    };

    enum class PieceType : std::uint8_t {
        pawn,
        knight,
        bishop,
        rook,
        queen,
        king,
    };

    enum class Piece : std::uint8_t {
        none,

        white_pawn,
        white_knight,
        white_bishop,
        white_rook,
        white_queen,
        white_king,

        black_pawn,
        black_knight,
        black_bishop,
        black_rook,
        black_queen,
        black_king,
    };

    enum class Square : std::uint8_t {
        a1,
        b1,
        c1,
        d1,
        e1,
        f1,
        g1,
        h1,
        a2,
        b2,
        c2,
        d2,
        e2,
        f2,
        g2,
        h2,
        a3,
        b3,
        c3,
        d3,
        e3,
        f3,
        g3,
        h3,
        a4,
        b4,
        c4,
        d4,
        e4,
        f4,
        g4,
        h4,
        a5,
        b5,
        c5,
        d5,
        e5,
        f5,
        g5,
        h5,
        a6,
        b6,
        c6,
        d6,
        e6,
        f6,
        g6,
        h6,
        a7,
        b7,
        c7,
        d7,
        e7,
        f7,
        g7,
        h7,
        a8,
        b8,
        c8,
        d8,
        e8,
        f8,
        g8,
        h8,

        none = 64,
    };

    enum class MoveFlag : std::uint8_t {
        quiet,
        capture,
        double_pawn_push,
        en_passant,
        king_castle,
        queen_castle,
        promotion,
        promotion_capture,
    };

    enum class GameState : std::uint8_t {
        playing,
        checkmate,
        stalemate,
    };

    struct Move {
        Square from = Square::none;
        Square to = Square::none;

        PieceType promotion = PieceType::queen;
        MoveFlag flag = MoveFlag::quiet;

        auto operator==(Move const &) const -> bool = default;
    };

    struct MoveResult {
        bool moved = false;
        bool capture = false;
        bool promotion = false;
        bool check = false;

        GameState game_state = GameState::playing;
    };

    struct RenderPiece {
        Piece piece = Piece::none;
        Square square = Square::none;

        // Stable for the lifetime of a game. The initial position uses IDs 0..31.
        std::uint8_t id = 0;
    };

    struct RenderState {
        std::array<RenderPiece, 32> pieces{};
        std::size_t piece_count = 0;

        Side side_to_move = Side::white;
        GameState game_state = GameState::playing;

        bool in_check = false;
    };

    class ChessEngine {
    public:
        ChessEngine();

        auto reset() -> void;

        [[nodiscard]] auto piece_at(Square square) const noexcept -> Piece;

        [[nodiscard]] auto side_to_move() const noexcept -> Side { return position_.side_to_move; }

        [[nodiscard]] auto game_state() const noexcept -> GameState { return game_state_; }

        [[nodiscard]] auto legal_moves() const -> std::vector<Move>;
        [[nodiscard]] auto legal_moves(Square from) const -> std::vector<Move>;

        [[nodiscard]] auto is_legal(Move move) const -> bool;

        auto update(Move move) -> MoveResult;

        auto update(Square from, Square to, PieceType promotion = PieceType::queen) -> MoveResult;

        [[nodiscard]] auto render_state() const -> RenderState;

        [[nodiscard]] auto perft(Badge<ChessGame> /*badge*/, int depth) const -> std::uint64_t;

    private:
        enum class CastlingRights : std::uint8_t {
            none = 0,

            white_king = 1 << 0,
            white_queen = 1 << 1,

            black_king = 1 << 2,
            black_queen = 1 << 3,
        };

        struct Position {
            std::array<std::array<Bitboard, 6>, 2> pieces{};

            std::array<Bitboard, 2> occupancy{};
            Bitboard occupancy_all = 0;

            std::array<Piece, 64> board{};

            Side side_to_move = Side::white;

            CastlingRights castling_rights =
                    static_cast<CastlingRights>(static_cast<std::uint8_t>(CastlingRights::white_king) |
                                                static_cast<std::uint8_t>(CastlingRights::white_queen) |
                                                static_cast<std::uint8_t>(CastlingRights::black_king) |
                                                static_cast<std::uint8_t>(CastlingRights::black_queen));

            Square en_passant = Square::none;

            std::uint16_t halfmove_clock = 0;
            std::uint16_t fullmove_number = 1;
        };

        struct PieceInstance {
            Piece piece = Piece::none;
            Square square = Square::none;

            std::uint8_t id = 0;

            bool alive = false;
        };

        struct MoveList {
            std::array<Move, 256> moves{};
            std::uint16_t count = 0;

            auto push_back(Move move) noexcept -> void { moves[count++] = move; }

            [[nodiscard]] auto begin() const noexcept { return moves.begin(); }
            [[nodiscard]] auto end() const noexcept { return moves.begin() + count; }
            [[nodiscard]] auto size() const noexcept -> std::size_t { return count; }
        };

        struct UndoState {
            Move move{};
            Piece moving_piece = Piece::none;
            Piece captured_piece = Piece::none;
            Square capture_square = Square::none;

            Side side_to_move = Side::white;
            CastlingRights castling_rights = CastlingRights::none;
            Square en_passant = Square::none;
            std::uint16_t halfmove_clock = 0;
            std::uint16_t fullmove_number = 1;

            std::array<PieceInstance, 2> instances{};
            std::array<std::uint8_t, 2> instance_indices{};
            std::uint8_t instance_count = 0;
        };

        auto generate_pseudo_legal_moves(MoveList &moves) const -> void;
        auto generate_legal_moves(MoveList &moves) const -> void;

        [[nodiscard]] auto square_attacked(Square square, Side by_side) const -> bool;

        [[nodiscard]] auto in_check(Side side) const -> bool;

        auto apply_move(Move move) -> void;
        [[nodiscard]] auto make_move(Move move) -> UndoState;
        auto unmake_move(UndoState const &undo) -> void;

        [[nodiscard]] auto perft_impl(int depth) -> std::uint64_t;

        auto rebuild_bitboards() -> void;

        [[nodiscard]] auto has_castling_right(CastlingRights right) const noexcept -> bool;
        auto clear_castling_right(CastlingRights right) noexcept -> void;

        [[nodiscard]] auto instance_at(Square square) noexcept -> PieceInstance *;
        [[nodiscard]] auto instance_at(Square square) const noexcept -> PieceInstance const *;

        Position position_{};

        // Separates "which sort of chess piece is on this square?" from "which
        // physical mesh/entity represents it?".
        std::array<PieceInstance, 32> instances_{};

        GameState game_state_ = GameState::playing;
    };

} // namespace chess
