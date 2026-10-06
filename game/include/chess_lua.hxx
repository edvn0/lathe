#pragma once

#include "scripting/lua_api.hxx"

// require("native.chess"): the rules engine for Lua games.
//
//   local chess = require("native.chess")
//   local engine = chess.new()
//
// Squares are integers 0..63 with a1 = 0, b1 = 1, ... h8 = 63 (file = sq % 8, rank = sq // 8). Pieces are strings
// ("white_pawn" .. "black_king"); a piece type is "queen", "rook", "bishop" or "knight".
//
//   engine:reset()
//   engine:side_to_move()          "white" | "black"
//   engine:state()                 "playing" | "checkmate" | "stalemate" | "draw_fifty_move" |
//                                  "draw_insufficient_material" | "draw_repetition"
//   engine:in_check()              whether the side to move is in check
//   engine:piece_at(square)        the piece, or nil
//   engine:pieces()                { { id = 0..31, piece = "white_rook", square = 0 }, ... } for pieces in play; an
//                                  id is stable for the whole game
//   engine:moves([from])           legal moves, { from, to, flag, promotion = "queen"|... or nil }
//   engine:move(from, to, [promotion])
//                                  plays a legal move: { moved, capture, promotion, check, state }
auto luaopen_lathe_chess(lua_State *state) -> int;
