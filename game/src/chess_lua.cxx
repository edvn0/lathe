#include "chess_lua.hxx"

#include <array>
#include <new>
#include <optional>
#include <string_view>

#include "chess_engine.hxx"

namespace {
    constexpr char const *engine_meta = "lathe.ChessEngine";

    constexpr std::array<char const *, 13> piece_names{
            "none",        "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
            "black_pawn",  "black_knight", "black_bishop", "black_rook", "black_queen", "black_king",
    };

    constexpr std::array<char const *, 6> state_names{
            "playing", "checkmate", "stalemate", "draw_fifty_move", "draw_insufficient_material", "draw_repetition",
    };

    constexpr std::array<char const *, 6> flag_names{
            "quiet", "capture", "double_pawn_push", "en_passant", "king_castle", "queen_castle",
    };

    auto engine_of(lua_State *state, int index) -> chess::ChessEngine & {
        return *static_cast<chess::ChessEngine *>(luaL_checkudata(state, index, engine_meta));
    }

    auto square_arg(lua_State *state, int index) -> chess::Square {
        auto const value = luaL_checkinteger(state, index);

        if (value < 0 || value > 63) {
            luaL_error(state, "square %d is outside 0..63", static_cast<int>(value));
        }

        return static_cast<chess::Square>(value);
    }

    auto promotion_name(chess::PieceType type) noexcept -> char const * {
        switch (type) {
            case chess::PieceType::knight:
                return "knight";
            case chess::PieceType::bishop:
                return "bishop";
            case chess::PieceType::rook:
                return "rook";
            default:
                return "queen";
        }
    }

    auto promotion_arg(lua_State *state, int index) -> chess::PieceType {
        if (lua_isnoneornil(state, index)) {
            return chess::PieceType::queen;
        }

        auto const name = std::string_view{luaL_checkstring(state, index)};

        if (name == "queen") {
            return chess::PieceType::queen;
        }
        if (name == "rook") {
            return chess::PieceType::rook;
        }
        if (name == "bishop") {
            return chess::PieceType::bishop;
        }
        if (name == "knight") {
            return chess::PieceType::knight;
        }

        luaL_error(state, "'%s' is not a promotion piece (queen, rook, bishop or knight)", name.data());

        return chess::PieceType::queen;
    }

    auto is_promotion(chess::MoveFlag flag) noexcept -> bool {
        return flag == chess::MoveFlag::promotion || flag == chess::MoveFlag::promotion_capture;
    }

    auto flag_name(chess::MoveFlag flag) noexcept -> char const * {
        switch (flag) {
            case chess::MoveFlag::promotion:
                return "promotion";
            case chess::MoveFlag::promotion_capture:
                return "promotion_capture";
            default:
                return flag_names[static_cast<std::size_t>(flag)];
        }
    }

    auto engine_new(lua_State *state) -> int {
        auto *const memory = lua_newuserdatauv(state, sizeof(chess::ChessEngine), 0);

        new (memory) chess::ChessEngine{};

        luaL_setmetatable(state, engine_meta);

        return 1;
    }

    auto engine_gc(lua_State *state) -> int {
        static_cast<chess::ChessEngine *>(lua_touserdata(state, 1))->~ChessEngine();

        return 0;
    }

    auto engine_reset(lua_State *state) -> int {
        engine_of(state, 1).reset();

        return 0;
    }

    auto engine_side_to_move(lua_State *state) -> int {
        lua_pushstring(state, engine_of(state, 1).side_to_move() == chess::Side::white ? "white" : "black");

        return 1;
    }

    auto engine_state(lua_State *state) -> int {
        lua_pushstring(state, state_names[static_cast<std::size_t>(engine_of(state, 1).game_state())]);

        return 1;
    }

    auto engine_in_check(lua_State *state) -> int {
        lua_pushboolean(state, engine_of(state, 1).render_state().in_check ? 1 : 0);

        return 1;
    }

    auto engine_piece_at(lua_State *state) -> int {
        auto &engine = engine_of(state, 1);
        auto const square = square_arg(state, 2);
        auto const piece = engine.piece_at(square);

        if (piece == chess::Piece::none) {
            lua_pushnil(state);
        } else {
            lua_pushstring(state, piece_names[static_cast<std::size_t>(piece)]);
        }

        return 1;
    }

    auto engine_pieces(lua_State *state) -> int {
        auto &engine = engine_of(state, 1);
        auto const render = engine.render_state();

        lua_createtable(state, static_cast<int>(render.piece_count), 0);

        for (std::size_t index = 0; index < render.piece_count; ++index) {
            auto const &piece = render.pieces[index];

            lua_createtable(state, 0, 3);
            lua_pushinteger(state, piece.id);
            lua_setfield(state, -2, "id");
            lua_pushstring(state, piece_names[static_cast<std::size_t>(piece.piece)]);
            lua_setfield(state, -2, "piece");
            lua_pushinteger(state, static_cast<lua_Integer>(piece.square));
            lua_setfield(state, -2, "square");
            lua_rawseti(state, -2, static_cast<lua_Integer>(index) + 1);
        }

        return 1;
    }

    auto engine_moves(lua_State *state) -> int {
        auto &engine = engine_of(state, 1);
        auto const filter = lua_isnoneornil(state, 2) ? std::nullopt : std::optional{square_arg(state, 2)};
        auto const moves = filter ? engine.legal_moves(*filter) : engine.legal_moves();

        lua_createtable(state, static_cast<int>(moves.size()), 0);

        lua_Integer position = 1;

        for (auto const &move: moves) {
            lua_createtable(state, 0, 4);
            lua_pushinteger(state, static_cast<lua_Integer>(move.from));
            lua_setfield(state, -2, "from");
            lua_pushinteger(state, static_cast<lua_Integer>(move.to));
            lua_setfield(state, -2, "to");
            lua_pushstring(state, flag_name(move.flag));
            lua_setfield(state, -2, "flag");

            if (is_promotion(move.flag)) {
                lua_pushstring(state, promotion_name(move.promotion));
                lua_setfield(state, -2, "promotion");
            }

            lua_rawseti(state, -2, position++);
        }

        return 1;
    }

    auto engine_move(lua_State *state) -> int {
        auto &engine = engine_of(state, 1);
        auto const from = square_arg(state, 2);
        auto const to = square_arg(state, 3);
        auto const promotion = promotion_arg(state, 4);

        auto const result = engine.update(from, to, promotion);

        lua_createtable(state, 0, 5);
        lua_pushboolean(state, result.moved ? 1 : 0);
        lua_setfield(state, -2, "moved");
        lua_pushboolean(state, result.capture ? 1 : 0);
        lua_setfield(state, -2, "capture");
        lua_pushboolean(state, result.promotion ? 1 : 0);
        lua_setfield(state, -2, "promotion");
        lua_pushboolean(state, result.check ? 1 : 0);
        lua_setfield(state, -2, "check");
        lua_pushstring(state, state_names[static_cast<std::size_t>(result.game_state)]);
        lua_setfield(state, -2, "state");

        return 1;
    }

    constexpr std::array engine_methods{
            luaL_Reg{"reset", &engine_reset},
            luaL_Reg{"side_to_move", &engine_side_to_move},
            luaL_Reg{"state", &engine_state},
            luaL_Reg{"in_check", &engine_in_check},
            luaL_Reg{"piece_at", &engine_piece_at},
            luaL_Reg{"pieces", &engine_pieces},
            luaL_Reg{"moves", &engine_moves},
            luaL_Reg{"move", &engine_move},
            luaL_Reg{nullptr, nullptr},
    };
}

auto luaopen_lathe_chess(lua_State *state) -> int {
    luaL_newmetatable(state, engine_meta);
    lua_createtable(state, 0, static_cast<int>(engine_methods.size()));
    luaL_setfuncs(state, engine_methods.data(), 0);
    lua_setfield(state, -2, "__index");
    lua_pushcfunction(state, &engine_gc);
    lua_setfield(state, -2, "__gc");
    lua_pushstring(state, engine_meta);
    lua_setfield(state, -2, "__name");
    lua_pop(state, 1);

    lua_createtable(state, 0, 1);
    lua_pushcfunction(state, &engine_new);
    lua_setfield(state, -2, "new");

    return 1;
}
