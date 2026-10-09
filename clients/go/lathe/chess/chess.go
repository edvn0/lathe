// Package chess is the chess layer over package lathe: action and state types for the server's "chess" game.
package chess

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"lathe/client/lathe"
)

const Game = "chess"

// State is the server's view for one player.
type State struct {
	// Board holds 8 rank strings, rank 8 first, in FEN letters with '.' for an empty square.
	Board      []string `json:"board"`
	SideToMove string   `json:"side_to_move"`
	YourSide   string   `json:"your_side"`
	Status     string   `json:"status"`
	InCheck    bool     `json:"in_check"`
	// History is every move played so far ("e2e4", "e7e8q"), enough to replay or resume a game.
	History []string `json:"history"`
	// LegalMoves are "e2e4"-style moves, and only filled in for the side to move.
	LegalMoves []string `json:"legal_moves"`
}

// ParseState decodes the payload of a lathe.State event.
func ParseState(event lathe.State) (State, error) {
	var state State

	err := json.Unmarshal(event.State, &state)

	return state, err
}

// MyTurn reports whether this player may move now.
func (s State) MyTurn() bool { return s.Status == "playing" && s.SideToMove == s.YourSide }

// String draws the board from the player's side, white at the bottom unless they are black.
func (s State) String() string {
	var out strings.Builder

	rows := s.Board
	files := "  a b c d e f g h"

	if s.YourSide == "black" {
		rows = make([]string, len(s.Board))
		for i, row := range s.Board {
			runes := []rune(row)
			for l, r := 0, len(runes)-1; l < r; l, r = l+1, r-1 {
				runes[l], runes[r] = runes[r], runes[l]
			}

			rows[len(s.Board)-1-i] = string(runes)
		}

		files = "  h g f e d c b a"
	}

	for i, row := range rows {
		rank := 8 - i
		if s.YourSide == "black" {
			rank = i + 1
		}

		fmt.Fprintf(&out, "%d", rank)

		for _, square := range row {
			fmt.Fprintf(&out, " %c", square)
		}

		out.WriteByte('\n')
	}

	out.WriteString(files)
	out.WriteByte('\n')

	return out.String()
}

type action struct {
	From      string `json:"from"`
	To        string `json:"to"`
	Promotion string `json:"promotion,omitempty"`
}

// Move plays a move given as "e2e4", with an optional fifth letter for promotion ("e7e8q").
func Move(ctx context.Context, client *lathe.Client, move string) error {
	if len(move) != 4 && len(move) != 5 {
		return fmt.Errorf("chess: move %q is not like e2e4 or e7e8q", move)
	}

	a := action{From: move[0:2], To: move[2:4]}
	if len(move) == 5 {
		a.Promotion = move[4:5]
	}

	return client.Action(ctx, a)
}
