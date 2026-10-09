package lathe

import (
	"encoding/json"
	"fmt"
)

// Event is anything the server (or the connection itself) tells the client. Switch on the concrete type.
type Event interface{ isEvent() }

// Welcome is sent on every connect. Keep Token to resume this player after a dropped connection.
type Welcome struct {
	Player uint32 `json:"player"`
	Token  string `json:"token"`
}

type Joined struct {
	Room   uint32 `json:"room"`
	Game   string `json:"game"`
	Player uint32 `json:"player"`
	Seat   int    `json:"seat"`
}

// State is the game-defined view for this player; decode State with the game's own type (see package chess).
type State struct {
	Room  uint32          `json:"room"`
	State json.RawMessage `json:"state"`
}

type Left struct {
	Room uint32 `json:"room"`
}

type OpponentDisconnected struct {
	Room   uint32 `json:"room"`
	Player uint32 `json:"player"`
}

type OpponentReconnected struct {
	Room   uint32 `json:"room"`
	Player uint32 `json:"player"`
}

// GameOver carries a nil Winner for a draw.
type GameOver struct {
	Room   uint32  `json:"room"`
	Winner *uint32 `json:"winner"`
	Reason string  `json:"reason"`
}

type RoomClosed struct {
	Room   uint32 `json:"room"`
	Reason string `json:"reason"`
}

// ServerError is a request the server rejected; the connection stays usable.
type ServerError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e ServerError) Error() string { return fmt.Sprintf("%s: %s", e.Code, e.Message) }

// Disconnected is raised locally when the connection drops. Call Client.Reconnect to resume.
type Disconnected struct{ Err error }

// Unknown is a message type this client version does not know, passed through verbatim.
type Unknown struct {
	Type string
	Raw  json.RawMessage
}

func (Welcome) isEvent()              {}
func (Joined) isEvent()               {}
func (State) isEvent()                {}
func (Left) isEvent()                 {}
func (OpponentDisconnected) isEvent() {}
func (OpponentReconnected) isEvent()  {}
func (GameOver) isEvent()             {}
func (RoomClosed) isEvent()           {}
func (ServerError) isEvent()          {}
func (Disconnected) isEvent()         {}
func (Unknown) isEvent()              {}

func decode(data []byte) (Event, error) {
	var envelope struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, err
	}

	var event Event

	switch envelope.Type {
	case "welcome":
		event = &Welcome{}
	case "joined":
		event = &Joined{}
	case "state":
		event = &State{}
	case "left":
		event = &Left{}
	case "opponent_disconnected":
		event = &OpponentDisconnected{}
	case "opponent_reconnected":
		event = &OpponentReconnected{}
	case "game_over":
		event = &GameOver{}
	case "room_closed":
		event = &RoomClosed{}
	case "error":
		event = &ServerError{}
	default:
		return Unknown{Type: envelope.Type, Raw: append(json.RawMessage(nil), data...)}, nil
	}

	if err := json.Unmarshal(data, event); err != nil {
		return nil, err
	}

	// Hand out values, not pointers, so callers switch on plain types.
	switch e := event.(type) {
	case *Welcome:
		return *e, nil
	case *Joined:
		return *e, nil
	case *State:
		return *e, nil
	case *Left:
		return *e, nil
	case *OpponentDisconnected:
		return *e, nil
	case *OpponentReconnected:
		return *e, nil
	case *GameOver:
		return *e, nil
	case *RoomClosed:
		return *e, nil
	default:
		return *event.(*ServerError), nil
	}
}
