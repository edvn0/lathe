#include "chess_engine.hxx"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <ranges>
#include <utility>

namespace {
    constexpr auto square_index(chess::Square square) noexcept -> int { return static_cast<int>(square); }

    constexpr auto valid_square(chess::Square square) noexcept -> bool {
        return square_index(square) >= 0 && square_index(square) < 64;
    }

    constexpr auto file_of(chess::Square square) noexcept -> int { return square_index(square) & 7; }

    constexpr auto rank_of(chess::Square square) noexcept -> int { return square_index(square) >> 3; }

    constexpr auto make_square(int file, int rank) noexcept -> chess::Square {
        return static_cast<chess::Square>((rank * 8) + file);
    }

    constexpr auto bit(chess::Square square) noexcept -> chess::Bitboard {
        return chess::Bitboard{1} << static_cast<unsigned>(square_index(square));
    }

    constexpr auto opposite(chess::Side side) noexcept -> chess::Side {
        return side == chess::Side::white ? chess::Side::black : chess::Side::white;
    }

    constexpr auto side_index(chess::Side side) noexcept -> std::size_t { return static_cast<std::size_t>(side); }

    constexpr auto type_index(chess::PieceType type) noexcept -> std::size_t { return static_cast<std::size_t>(type); }

    constexpr auto piece_side(chess::Piece piece) noexcept -> chess::Side {
        using enum chess::Piece;

        switch (piece) {
            case white_pawn:
            case white_knight:
            case white_bishop:
            case white_rook:
            case white_queen:
            case white_king:
                return chess::Side::white;
            default:
                return chess::Side::black;
        }
    }

    constexpr auto piece_type(chess::Piece piece) noexcept -> chess::PieceType {
        using enum chess::Piece;

        using enum chess::PieceType;

        switch (piece) {
            case white_pawn:
            case black_pawn:
                return pawn;
            case white_knight:
            case black_knight:
                return knight;
            case white_bishop:
            case black_bishop:
                return bishop;
            case white_rook:
            case black_rook:
                return rook;
            case white_queen:
            case black_queen:
                return queen;
            case white_king:
            case black_king:
                return king;
            case none:
                break;
        }

        return pawn;
    }

    constexpr auto make_piece(chess::Side side, chess::PieceType type) noexcept -> chess::Piece {
        constexpr std::array white{
                chess::Piece::white_pawn, chess::Piece::white_knight, chess::Piece::white_bishop,
                chess::Piece::white_rook, chess::Piece::white_queen,  chess::Piece::white_king,
        };

        constexpr std::array black{
                chess::Piece::black_pawn, chess::Piece::black_knight, chess::Piece::black_bishop,
                chess::Piece::black_rook, chess::Piece::black_queen,  chess::Piece::black_king,
        };

        auto const index = type_index(type);

        return side == chess::Side::white ? white[index] : black[index];
    }

    constexpr auto is_capture(chess::MoveFlag flag) noexcept -> bool {
        return flag == chess::MoveFlag::capture || flag == chess::MoveFlag::en_passant ||
               flag == chess::MoveFlag::promotion_capture;
    }

    constexpr auto is_promotion(chess::MoveFlag flag) noexcept -> bool {
        return flag == chess::MoveFlag::promotion || flag == chess::MoveFlag::promotion_capture;
    }

    constexpr std::array promotion_types{
            chess::PieceType::queen,
            chess::PieceType::rook,
            chess::PieceType::bishop,
            chess::PieceType::knight,
    };

    constexpr std::array knight_offsets{
            std::pair{+1, +2}, std::pair{+2, +1}, std::pair{+2, -1}, std::pair{+1, -2},
            std::pair{-1, -2}, std::pair{-2, -1}, std::pair{-2, +1}, std::pair{-1, +2},
    };

    constexpr std::array king_offsets{
            std::pair{-1, -1}, std::pair{0, -1},  std::pair{+1, -1}, std::pair{-1, 0},
            std::pair{+1, 0},  std::pair{-1, +1}, std::pair{0, +1},  std::pair{+1, +1},
    };

    constexpr std::array rook_directions{
            std::pair{+1, 0},
            std::pair{-1, 0},
            std::pair{0, +1},
            std::pair{0, -1},
    };

    constexpr std::array bishop_directions{
            std::pair{+1, +1},
            std::pair{-1, +1},
            std::pair{+1, -1},
            std::pair{-1, -1},
    };

}

namespace chess {
    ChessEngine::ChessEngine() { reset(); }

    auto ChessEngine::reset() -> void {
        using enum PieceType;

        position_ = {};

        position_.board.fill(Piece::none);

        position_.castling_rights = static_cast<CastlingRights>(static_cast<std::uint8_t>(CastlingRights::white_king) |

                                                                static_cast<std::uint8_t>(CastlingRights::white_queen) |

                                                                static_cast<std::uint8_t>(CastlingRights::black_king) |

                                                                static_cast<std::uint8_t>(CastlingRights::black_queen));

        position_.side_to_move = Side::white;

        position_.en_passant = Square::none;

        position_.halfmove_clock = 0;

        position_.fullmove_number = 1;

        instances_ = {};

        constexpr std::array back_rank{
                rook, knight, bishop, queen, king, bishop, knight, rook,
        };

        std::uint8_t next_id = 0;

        auto add_piece = [&](Side side, PieceType type, Square square) mutable {
            auto const piece = make_piece(side, type);

            position_.board[static_cast<std::size_t>(square_index(square))] = piece;

            instances_[next_id] = PieceInstance{
                    .piece = piece,
                    .square = square,
                    .id = next_id,
                    .alive = true,
            };

            ++next_id;
        };

        for (auto const side: {Side::white, Side::black}) {
            auto const back_rank_index = side == Side::white ? 0 : 7;

            auto const pawn_rank_index = side == Side::white ? 1 : 6;

            for (int file = 0; file < 8; ++file) {
                add_piece(side, back_rank[static_cast<std::size_t>(file)], make_square(file, back_rank_index));
            }

            for (int file = 0; file < 8; ++file) {
                add_piece(side, pawn, make_square(file, pawn_rank_index));
            }
        }

        rebuild_bitboards();

        game_state_ = GameState::playing;

        history_.clear();
        history_.push_back(position_key());
    }

    auto ChessEngine::position_key() const noexcept -> std::uint64_t {
        auto const mix = [](std::uint64_t value) noexcept -> std::uint64_t {
            value += 0x9E3779B97F4A7C15ULL;
            value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
            value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;

            return value ^ (value >> 31);
        };

        std::uint64_t key = 0;

        for (std::size_t index = 0; index < position_.board.size(); ++index) {
            if (position_.board[index] != Piece::none) {
                key ^= mix((static_cast<std::uint64_t>(position_.board[index]) << 8) | index);
            }
        }

        key ^= mix(0x100 + static_cast<std::uint64_t>(position_.side_to_move));
        key ^= mix(0x200 + static_cast<std::uint64_t>(position_.castling_rights));

        if (position_.en_passant != Square::none) {
            auto const target = square_index(position_.en_passant);
            auto const file = target & 7;
            auto const pawn = make_piece(position_.side_to_move, PieceType::pawn);
            auto const white = position_.side_to_move == Side::white;

            auto const has_pawn = [&](int square) {
                return square >= 0 && square < 64 && position_.board[static_cast<std::size_t>(square)] == pawn;
            };

            auto const capturable = white ? (file > 0 && has_pawn(target - 9)) || (file < 7 && has_pawn(target - 7))
                                          : (file < 7 && has_pawn(target + 9)) || (file > 0 && has_pawn(target + 7));

            if (capturable) {
                key ^= mix(0x300 + static_cast<std::uint64_t>(target));
            }
        }

        return key;
    }

    auto ChessEngine::insufficient_material() const noexcept -> bool {
        auto const count = [&](Side side, PieceType type) {
            return std::popcount(position_.pieces[static_cast<std::size_t>(side)][static_cast<std::size_t>(type)]);
        };

        for (auto const side: {Side::white, Side::black}) {
            if (count(side, PieceType::pawn) != 0 || count(side, PieceType::rook) != 0 ||
                count(side, PieceType::queen) != 0) {
                return false;
            }
        }

        auto const minors = [&](Side side) { return count(side, PieceType::knight) + count(side, PieceType::bishop); };
        auto const white_minors = minors(Side::white);
        auto const black_minors = minors(Side::black);

        if (white_minors + black_minors <= 1) {
            return true;
        }

        if (white_minors == 1 && black_minors == 1 && count(Side::white, PieceType::bishop) == 1 &&
            count(Side::black, PieceType::bishop) == 1) {
            auto const colour = [&](Side side) {
                auto const square = std::countr_zero(
                        position_.pieces[static_cast<std::size_t>(side)][static_cast<std::size_t>(PieceType::bishop)]);

                return ((square & 7) + (square >> 3)) & 1;
            };

            return colour(Side::white) == colour(Side::black);
        }

        return false;
    }

    auto ChessEngine::piece_at(Square square) const noexcept -> Piece {
        if (!valid_square(square)) {
            return Piece::none;
        }

        return position_.board[static_cast<std::size_t>(square_index(square))];
    }

    auto ChessEngine::has_castling_right(CastlingRights right) const noexcept -> bool {
        auto const rights = static_cast<std::uint8_t>(position_.castling_rights);

        auto const flag = static_cast<std::uint8_t>(right);

        return (rights & flag) != 0;
    }

    auto ChessEngine::clear_castling_right(CastlingRights right) noexcept -> void {
        auto rights = static_cast<std::uint8_t>(position_.castling_rights);

        rights &= static_cast<std::uint8_t>(~static_cast<std::uint8_t>(right));

        position_.castling_rights = static_cast<CastlingRights>(rights);
    }

    auto ChessEngine::instance_at(Square square) noexcept -> PieceInstance * {
        auto const it = std::ranges::find_if(instances_, [square](PieceInstance const &instance) {
            return instance.alive && instance.square == square;
        });

        return it == instances_.end() ? nullptr : &*it;
    }

    auto ChessEngine::instance_at(Square square) const noexcept -> PieceInstance const * {
        auto const it = std::ranges::find_if(instances_, [square](PieceInstance const &instance) {
            return instance.alive && instance.square == square;
        });

        return it == instances_.end() ? nullptr : &*it;
    }

    auto ChessEngine::rebuild_bitboards() -> void {
        for (auto &side: position_.pieces) {
            side.fill(0);
        }

        position_.occupancy.fill(0);

        position_.occupancy_all = 0;

        for (int index = 0; index < 64; ++index) {
            auto const piece = position_.board[static_cast<std::size_t>(index)];

            if (piece == Piece::none) {
                continue;
            }

            auto const square = static_cast<Square>(index);

            auto const side = piece_side(piece);

            auto const type = piece_type(piece);

            position_.pieces[side_index(side)][type_index(type)] |= bit(square);

            position_.occupancy[side_index(side)] |= bit(square);

            position_.occupancy_all |= bit(square);
        }
    }

    auto ChessEngine::square_attacked(Square square, Side by_side) const -> bool {
        auto const target_file = file_of(square);

        auto const target_rank = rank_of(square);

        {
            auto const pawn_rank_delta = by_side == Side::white ? -1 : +1;

            auto const source_rank = target_rank + pawn_rank_delta;

            if (source_rank >= 0 && source_rank < 8) {
                for (auto const file_delta: {-1, +1}) {
                    auto const source_file = target_file + file_delta;

                    if (source_file < 0 || source_file >= 8) {
                        continue;
                    }

                    auto const source = make_square(source_file, source_rank);

                    if (piece_at(source) == make_piece(by_side, PieceType::pawn)) {
                        return true;
                    }
                }
            }
        }

        for (auto const &[dx, dy]: knight_offsets) {
            auto const file = target_file + dx;

            auto const rank = target_rank + dy;

            if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
                continue;
            }

            if (piece_at(make_square(file, rank)) == make_piece(by_side, PieceType::knight)) {
                return true;
            }
        }

        for (auto const &[dx, dy]: king_offsets) {
            auto const file = target_file + dx;

            auto const rank = target_rank + dy;

            if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
                continue;
            }

            if (piece_at(make_square(file, rank)) == make_piece(by_side, PieceType::king)) {
                return true;
            }
        }

        auto const attacked_on_ray = [&](auto const &directions, PieceType first, PieceType second) {
            for (auto const &[dx, dy]: directions) {
                auto file = target_file + dx;

                auto rank = target_rank + dy;

                while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
                    auto const piece = piece_at(make_square(file, rank));

                    if (piece != Piece::none) {
                        if (piece_side(piece) == by_side) {
                            auto const type = piece_type(piece);

                            if (type == first || type == second) {
                                return true;
                            }
                        }

                        break;
                    }

                    file += dx;

                    rank += dy;
                }
            }

            return false;
        };

        if (attacked_on_ray(bishop_directions, PieceType::bishop, PieceType::queen)) {
            return true;
        }

        if (attacked_on_ray(rook_directions, PieceType::rook, PieceType::queen)) {
            return true;
        }

        return false;
    }

    auto ChessEngine::in_check(Side side) const -> bool {
        auto const king = make_piece(side, PieceType::king);

        for (int index = 0; index < 64; ++index) {
            if (position_.board[static_cast<std::size_t>(index)] == king) {
                return square_attacked(static_cast<Square>(index), opposite(side));
            }
        }

        return false;
    }

    auto ChessEngine::generate_pseudo_legal_moves(MoveList &moves) const -> void {
        auto const side = position_.side_to_move;

        auto const enemy = opposite(side);

        auto const add_target = [&](Square from, Square to) {
            auto const target = piece_at(to);

            if (target == Piece::none) {
                moves.push_back(Move{
                        .from = from,
                        .to = to,
                        .flag = MoveFlag::quiet,
                });

                return;
            }

            if (piece_side(target) != side) {
                moves.push_back(Move{
                        .from = from,
                        .to = to,
                        .flag = MoveFlag::capture,
                });
            }
        };

        auto const add_ray = [&](Square from, int dx, int dy) {
            auto file = file_of(from) + dx;

            auto rank = rank_of(from) + dy;

            while (file >= 0 && file < 8 && rank >= 0 && rank < 8) {
                auto const to = make_square(file, rank);

                auto const target = piece_at(to);

                if (target == Piece::none) {
                    moves.push_back(Move{
                            .from = from,
                            .to = to,
                            .flag = MoveFlag::quiet,
                    });
                } else {
                    if (piece_side(target) != side) {
                        moves.push_back(Move{
                                .from = from,
                                .to = to,
                                .flag = MoveFlag::capture,
                        });
                    }

                    break;
                }

                file += dx;

                rank += dy;
            }
        };

        for (int index = 0; index < 64; ++index) {
            auto const from = static_cast<Square>(index);

            auto const piece = piece_at(from);

            if (piece == Piece::none || piece_side(piece) != side) {
                continue;
            }

            auto const type = piece_type(piece);

            switch (type) {
                case PieceType::pawn: {
                    auto const forward = side == Side::white ? +1 : -1;

                    auto const start_rank = side == Side::white ? 1 : 6;

                    auto const promotion_rank = side == Side::white ? 7 : 0;

                    auto const from_file = file_of(from);

                    auto const from_rank = rank_of(from);

                    auto const one_rank = from_rank + forward;

                    if (one_rank >= 0 && one_rank < 8) {
                        auto const one = make_square(from_file, one_rank);

                        if (piece_at(one) == Piece::none) {
                            if (one_rank == promotion_rank) {
                                for (auto const promotion: promotion_types) {
                                    moves.push_back(Move{
                                            .from = from,
                                            .to = one,
                                            .promotion = promotion,
                                            .flag = MoveFlag::promotion,
                                    });
                                }
                            } else {
                                moves.push_back(Move{
                                        .from = from,
                                        .to = one,
                                        .flag = MoveFlag::quiet,
                                });

                                if (from_rank == start_rank) {
                                    auto const two = make_square(from_file, from_rank + (2 * forward));

                                    if (piece_at(two) == Piece::none) {
                                        moves.push_back(Move{
                                                .from = from,
                                                .to = two,
                                                .flag = MoveFlag::double_pawn_push,
                                        });
                                    }
                                }
                            }
                        }
                    }

                    for (auto const dx: {-1, +1}) {
                        auto const file = from_file + dx;

                        auto const rank = from_rank + forward;

                        if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
                            continue;
                        }

                        auto const to = make_square(file, rank);

                        auto const target = piece_at(to);

                        if (target != Piece::none && piece_side(target) == enemy) {
                            if (rank == promotion_rank) {
                                for (auto const promotion: promotion_types) {
                                    moves.push_back(Move{
                                            .from = from,
                                            .to = to,
                                            .promotion = promotion,
                                            .flag = MoveFlag::promotion_capture,
                                    });
                                }
                            } else {
                                moves.push_back(Move{
                                        .from = from,
                                        .to = to,
                                        .flag = MoveFlag::capture,
                                });
                            }

                            continue;
                        }

                        if (to == position_.en_passant) {
                            moves.push_back(Move{
                                    .from = from,
                                    .to = to,
                                    .flag = MoveFlag::en_passant,
                            });
                        }
                    }

                    break;
                }
                case PieceType::knight:
                    for (auto const &[dx, dy]: knight_offsets) {
                        auto const file = file_of(from) + dx;

                        auto const rank = rank_of(from) + dy;

                        if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
                            continue;
                        }

                        add_target(from, make_square(file, rank));
                    }

                    break;
                case PieceType::bishop:
                    for (auto const &[dx, dy]: bishop_directions) {
                        add_ray(from, dx, dy);
                    }

                    break;
                case PieceType::rook:
                    for (auto const &[dx, dy]: rook_directions) {
                        add_ray(from, dx, dy);
                    }

                    break;
                case PieceType::queen:
                    for (auto const &[dx, dy]: bishop_directions) {
                        add_ray(from, dx, dy);
                    }

                    for (auto const &[dx, dy]: rook_directions) {
                        add_ray(from, dx, dy);
                    }

                    break;
                case PieceType::king:
                    for (auto const &[dx, dy]: king_offsets) {
                        auto const file = file_of(from) + dx;

                        auto const rank = rank_of(from) + dy;

                        if (file < 0 || file >= 8 || rank < 0 || rank >= 8) {
                            continue;
                        }

                        add_target(from, make_square(file, rank));
                    }

                    if (side == Side::white && from == Square::e1) {
                        if (has_castling_right(CastlingRights::white_king) &&
                            piece_at(Square::h1) == Piece::white_rook && piece_at(Square::f1) == Piece::none &&
                            piece_at(Square::g1) == Piece::none && !square_attacked(Square::e1, Side::black) &&
                            !square_attacked(Square::f1, Side::black) && !square_attacked(Square::g1, Side::black)) {
                            moves.push_back(Move{
                                    .from = Square::e1,
                                    .to = Square::g1,
                                    .flag = MoveFlag::king_castle,
                            });
                        }

                        if (has_castling_right(CastlingRights::white_queen) &&
                            piece_at(Square::a1) == Piece::white_rook && piece_at(Square::b1) == Piece::none &&
                            piece_at(Square::c1) == Piece::none && piece_at(Square::d1) == Piece::none &&
                            !square_attacked(Square::e1, Side::black) && !square_attacked(Square::d1, Side::black) &&
                            !square_attacked(Square::c1, Side::black)) {
                            moves.push_back(Move{
                                    .from = Square::e1,
                                    .to = Square::c1,
                                    .flag = MoveFlag::queen_castle,
                            });
                        }
                    }

                    if (side == Side::black && from == Square::e8) {
                        if (has_castling_right(CastlingRights::black_king) &&
                            piece_at(Square::h8) == Piece::black_rook && piece_at(Square::f8) == Piece::none &&
                            piece_at(Square::g8) == Piece::none && !square_attacked(Square::e8, Side::white) &&
                            !square_attacked(Square::f8, Side::white) && !square_attacked(Square::g8, Side::white)) {
                            moves.push_back(Move{
                                    .from = Square::e8,
                                    .to = Square::g8,
                                    .flag = MoveFlag::king_castle,
                            });
                        }

                        if (has_castling_right(CastlingRights::black_queen) &&
                            piece_at(Square::a8) == Piece::black_rook && piece_at(Square::b8) == Piece::none &&
                            piece_at(Square::c8) == Piece::none && piece_at(Square::d8) == Piece::none &&
                            !square_attacked(Square::e8, Side::white) && !square_attacked(Square::d8, Side::white) &&
                            !square_attacked(Square::c8, Side::white)) {
                            moves.push_back(Move{
                                    .from = Square::e8,
                                    .to = Square::c8,
                                    .flag = MoveFlag::queen_castle,
                            });
                        }
                    }

                    break;
            }
        }
    }

    auto ChessEngine::generate_legal_moves(MoveList &moves) const -> void {
        MoveList pseudo;
        generate_pseudo_legal_moves(pseudo);

        auto const moving_side = position_.side_to_move;
        auto copy = *this;

        for (auto const move: pseudo) {
            auto const undo = copy.make_move(move);

            if (!copy.in_check(moving_side)) {
                moves.push_back(move);
            }

            copy.unmake_move(undo);
        }
    }

    auto ChessEngine::legal_moves() const -> std::vector<Move> {
        MoveList moves;
        generate_legal_moves(moves);

        return {moves.begin(), moves.end()};
    }

    auto ChessEngine::legal_moves(Square from) const -> std::vector<Move> {
        auto result = legal_moves();

        std::erase_if(result, [from](Move const &move) { return move.from != from; });

        return result;
    }

    auto ChessEngine::is_legal(Move move) const -> bool {
        auto const moves = legal_moves();

        return std::ranges::find(moves, move) != moves.end();
    }

    auto ChessEngine::apply_move(Move move) -> void {
        auto const moving_piece = piece_at(move.from);

        if (moving_piece == Piece::none) {
            return;
        }

        auto const moving_side = piece_side(moving_piece);

        auto const moving_type = piece_type(moving_piece);

        auto *moving_instance = instance_at(move.from);

        auto capture_square = move.to;

        if (move.flag == MoveFlag::en_passant) {
            capture_square = make_square(file_of(move.to), rank_of(move.to) + (moving_side == Side::white ? -1 : +1));
        }

        if (is_capture(move.flag)) {
            if (auto *captured = instance_at(capture_square); captured != nullptr) {
                captured->alive = false;

                captured->square = Square::none;
            }

            position_.board[static_cast<std::size_t>(square_index(capture_square))] = Piece::none;
        }

        switch (capture_square) {
            case Square::a1:
                clear_castling_right(CastlingRights::white_queen);

                break;
            case Square::h1:
                clear_castling_right(CastlingRights::white_king);

                break;
            case Square::a8:
                clear_castling_right(CastlingRights::black_queen);

                break;
            case Square::h8:
                clear_castling_right(CastlingRights::black_king);

                break;
            default:
                break;
        }

        position_.board[static_cast<std::size_t>(square_index(move.from))] = Piece::none;

        auto placed_piece = moving_piece;

        if (is_promotion(move.flag)) {
            placed_piece = make_piece(moving_side, move.promotion);
        }

        position_.board[static_cast<std::size_t>(square_index(move.to))] = placed_piece;

        if (moving_instance != nullptr) {
            moving_instance->square = move.to;

            moving_instance->piece = placed_piece;
        }

        if (move.flag == MoveFlag::king_castle) {
            auto const rook_from = moving_side == Side::white ? Square::h1 : Square::h8;

            auto const rook_to = moving_side == Side::white ? Square::f1 : Square::f8;

            auto const rook = piece_at(rook_from);

            position_.board[static_cast<std::size_t>(square_index(rook_from))] = Piece::none;

            position_.board[static_cast<std::size_t>(square_index(rook_to))] = rook;

            if (auto *rook_instance = instance_at(rook_from); rook_instance != nullptr) {
                rook_instance->square = rook_to;
            }
        }

        if (move.flag == MoveFlag::queen_castle) {
            auto const rook_from = moving_side == Side::white ? Square::a1 : Square::a8;

            auto const rook_to = moving_side == Side::white ? Square::d1 : Square::d8;

            auto const rook = piece_at(rook_from);

            position_.board[static_cast<std::size_t>(square_index(rook_from))] = Piece::none;

            position_.board[static_cast<std::size_t>(square_index(rook_to))] = rook;

            if (auto *rook_instance = instance_at(rook_from); rook_instance != nullptr) {
                rook_instance->square = rook_to;
            }
        }

        if (moving_piece == Piece::white_king) {
            clear_castling_right(CastlingRights::white_king);

            clear_castling_right(CastlingRights::white_queen);
        }

        if (moving_piece == Piece::black_king) {
            clear_castling_right(CastlingRights::black_king);

            clear_castling_right(CastlingRights::black_queen);
        }

        if (moving_piece == Piece::white_rook) {
            if (move.from == Square::a1) {
                clear_castling_right(CastlingRights::white_queen);
            }

            if (move.from == Square::h1) {
                clear_castling_right(CastlingRights::white_king);
            }
        }

        if (moving_piece == Piece::black_rook) {
            if (move.from == Square::a8) {
                clear_castling_right(CastlingRights::black_queen);
            }

            if (move.from == Square::h8) {
                clear_castling_right(CastlingRights::black_king);
            }
        }

        position_.en_passant = Square::none;

        if (move.flag == MoveFlag::double_pawn_push) {
            position_.en_passant = make_square(file_of(move.from), (rank_of(move.from) + rank_of(move.to)) / 2);
        }

        if (moving_type == PieceType::pawn || is_capture(move.flag)) {
            position_.halfmove_clock = 0;
        } else {
            ++position_.halfmove_clock;
        }

        if (moving_side == Side::black) {
            ++position_.fullmove_number;
        }

        position_.side_to_move = opposite(moving_side);

        rebuild_bitboards();
    }

    auto ChessEngine::make_move(Move move) -> UndoState {
        auto const moving_piece = piece_at(move.from);
        auto const moving_side = piece_side(moving_piece);

        auto capture_square = move.to;
        if (move.flag == MoveFlag::en_passant) {
            capture_square = make_square(file_of(move.to), rank_of(move.to) + (moving_side == Side::white ? -1 : +1));
        }

        UndoState undo{
                .move = move,
                .moving_piece = moving_piece,
                .captured_piece = is_capture(move.flag) ? piece_at(capture_square) : Piece::none,
                .capture_square = capture_square,
                .side_to_move = position_.side_to_move,
                .castling_rights = position_.castling_rights,
                .en_passant = position_.en_passant,
                .halfmove_clock = position_.halfmove_clock,
                .fullmove_number = position_.fullmove_number,
        };

        auto save_instance = [&](Square square) {
            auto const *instance = instance_at(square);
            if (instance == nullptr) {
                return;
            }

            auto const index = static_cast<std::size_t>(instance - instances_.data());

            for (std::uint8_t i = 0; i < undo.instance_count; ++i) {
                if (undo.instance_indices[i] == index) {
                    return;
                }
            }

            undo.instance_indices[undo.instance_count] = static_cast<std::uint8_t>(index);
            undo.instances[undo.instance_count] = *instance;
            ++undo.instance_count;
        };

        save_instance(move.from);

        if (is_capture(move.flag)) {
            save_instance(capture_square);
        }

        if (move.flag == MoveFlag::king_castle) {
            save_instance(moving_side == Side::white ? Square::h1 : Square::h8);
        } else if (move.flag == MoveFlag::queen_castle) {
            save_instance(moving_side == Side::white ? Square::a1 : Square::a8);
        }

        apply_move(move);
        return undo;
    }

    auto ChessEngine::unmake_move(UndoState const &undo) -> void {
        auto const move = undo.move;

        position_.board[static_cast<std::size_t>(square_index(move.from))] = undo.moving_piece;
        position_.board[static_cast<std::size_t>(square_index(move.to))] = Piece::none;

        if (undo.captured_piece != Piece::none) {
            position_.board[static_cast<std::size_t>(square_index(undo.capture_square))] = undo.captured_piece;
        }

        if (move.flag == MoveFlag::king_castle) {
            auto const rook_from = undo.side_to_move == Side::white ? Square::h1 : Square::h8;
            auto const rook_to = undo.side_to_move == Side::white ? Square::f1 : Square::f8;

            position_.board[static_cast<std::size_t>(square_index(rook_from))] =
                    make_piece(undo.side_to_move, PieceType::rook);
            position_.board[static_cast<std::size_t>(square_index(rook_to))] = Piece::none;
        } else if (move.flag == MoveFlag::queen_castle) {
            auto const rook_from = undo.side_to_move == Side::white ? Square::a1 : Square::a8;
            auto const rook_to = undo.side_to_move == Side::white ? Square::d1 : Square::d8;

            position_.board[static_cast<std::size_t>(square_index(rook_from))] =
                    make_piece(undo.side_to_move, PieceType::rook);
            position_.board[static_cast<std::size_t>(square_index(rook_to))] = Piece::none;
        }

        position_.side_to_move = undo.side_to_move;
        position_.castling_rights = undo.castling_rights;
        position_.en_passant = undo.en_passant;
        position_.halfmove_clock = undo.halfmove_clock;
        position_.fullmove_number = undo.fullmove_number;

        for (std::uint8_t i = 0; i < undo.instance_count; ++i) {
            instances_[undo.instance_indices[i]] = undo.instances[i];
        }

        rebuild_bitboards();
    }

    auto ChessEngine::update(Move move) -> MoveResult {
        auto const moves = legal_moves();
        if (auto const it = std::ranges::find(moves, move); it == moves.end()) {
            return MoveResult{
                    .game_state = game_state_,
            };
        }

        auto const capture = is_capture(move.flag);
        auto const promotion = is_promotion(move.flag);
        apply_move(move);
        auto const check = in_check(position_.side_to_move);

        history_.push_back(position_key());

        if (auto const replies = legal_moves(); replies.empty()) {
            game_state_ = check ? GameState::checkmate : GameState::stalemate;
        } else if (position_.halfmove_clock >= 100) {
            game_state_ = GameState::draw_fifty_move;
        } else if (insufficient_material()) {
            game_state_ = GameState::draw_insufficient_material;
        } else if (std::ranges::count(history_, history_.back()) >= 3) {
            game_state_ = GameState::draw_repetition;
        } else {
            game_state_ = GameState::playing;
        }

        return MoveResult{
                .moved = true,
                .capture = capture,
                .promotion = promotion,
                .check = check,
                .game_state = game_state_,
        };
    }

    auto ChessEngine::update(Square from, Square to, PieceType promotion) -> MoveResult {
        auto const moves = legal_moves(from);

        auto const it = std::ranges::find_if(moves, [&](Move const &move) {
            if (move.to != to) {
                return false;
            }

            if (is_promotion(move.flag)) {
                return move.promotion == promotion;
            }

            return true;
        });

        if (it == moves.end()) {
            return MoveResult{
                    .game_state = game_state_,
            };
        }

        return update(*it);
    }

    auto ChessEngine::render_state() const -> RenderState {
        RenderState state{
                .side_to_move = position_.side_to_move,
                .game_state = game_state_,
                .in_check = in_check(position_.side_to_move),
        };

        for (auto const &instance: instances_) {
            if (!instance.alive) {
                continue;
            }

            state.pieces[state.piece_count++] = RenderPiece{
                    .piece = instance.piece,
                    .square = instance.square,
                    .id = instance.id,
            };
        }

        return state;
    }

    auto ChessEngine::perft_impl(int depth) -> std::uint64_t {
        if (depth == 0) {
            return 1;
        }

        MoveList moves;
        generate_legal_moves(moves);

        if (depth == 1) {
            return moves.size();
        }

        std::uint64_t nodes = 0;

        for (auto const move: moves) {
            auto const undo = make_move(move);
            nodes += perft_impl(depth - 1);
            unmake_move(undo);
        }

        return nodes;
    }

    auto ChessEngine::perft(Badge<ChessGame> , int depth) const -> std::uint64_t {
        if (depth < 0) {
            return 0;
        }

        auto copy = *this;
        return copy.perft_impl(depth);
    }

}
