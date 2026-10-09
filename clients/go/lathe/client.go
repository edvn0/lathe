// Package lathe is a client for lathe-server: JSON over WebSocket, game-agnostic. Games layer their own action and
// state types on top (see package chess).
package lathe

import (
	"context"
	"encoding/json"
	"errors"
	"sync"

	"github.com/coder/websocket"
)

// Client is one player's connection. Events arrive on Events(); requests are fire-and-forget, with the server's answer
// (or a ServerError) coming back as an event.
type Client struct {
	url    string
	events chan Event

	mu      sync.Mutex
	conn    *websocket.Conn
	player  uint32
	token   string
	closing bool
}

// Dial connects and starts delivering events. The first event is a Welcome.
func Dial(ctx context.Context, url string) (*Client, error) {
	client := &Client{url: url, events: make(chan Event, 64)}

	if err := client.connect(ctx); err != nil {
		return nil, err
	}

	return client, nil
}

func (c *Client) connect(ctx context.Context) error {
	conn, _, err := websocket.Dial(ctx, c.url, nil)
	if err != nil {
		return err
	}

	conn.SetReadLimit(1 << 20)

	c.mu.Lock()
	c.conn = conn
	c.closing = false
	c.mu.Unlock()

	go c.read(conn)

	return nil
}

// read pumps one connection. It ends with a Disconnected event unless the client closed it on purpose.
func (c *Client) read(conn *websocket.Conn) {
	for {
		_, data, err := conn.Read(context.Background())
		if err != nil {
			c.mu.Lock()
			closing := c.closing
			c.mu.Unlock()

			if !closing {
				c.events <- Disconnected{Err: err}
			}

			return
		}

		event, err := decode(data)
		if err != nil {
			c.events <- Disconnected{Err: err}
			conn.Close(websocket.StatusInvalidFramePayloadData, "bad message")

			return
		}

		if welcome, ok := event.(Welcome); ok {
			c.mu.Lock()
			c.player, c.token = welcome.Player, welcome.Token
			c.mu.Unlock()
		}

		c.events <- event
	}
}

// Events delivers everything the server sends, in order. It stays open across Reconnect and is never closed.
func (c *Client) Events() <-chan Event { return c.events }

// Player is the id from the latest Welcome, 0 before it arrives.
func (c *Client) Player() uint32 {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.player
}

// Token is the session token from the latest Welcome, for Resume.
func (c *Client) Token() string {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.token
}

func (c *Client) send(ctx context.Context, message any) error {
	data, err := json.Marshal(message)
	if err != nil {
		return err
	}

	c.mu.Lock()
	conn := c.conn
	c.mu.Unlock()

	if conn == nil {
		return errors.New("lathe: not connected")
	}

	return conn.Write(ctx, websocket.MessageText, data)
}

type message map[string]any

// CreateRoom opens a room for game and joins it as the first seat.
func (c *Client) CreateRoom(ctx context.Context, game string) error {
	return c.send(ctx, message{"type": "create_room", "game": game})
}

func (c *Client) JoinRoom(ctx context.Context, room uint32) error {
	return c.send(ctx, message{"type": "join_room", "room": room})
}

// Action sends a game-defined payload; anything json.Marshal accepts.
func (c *Client) Action(ctx context.Context, action any) error {
	return c.send(ctx, message{"type": "action", "action": action})
}

func (c *Client) LeaveRoom(ctx context.Context) error {
	return c.send(ctx, message{"type": "leave_room"})
}

// Resume rebinds this connection to the player that owned token, taking back their seat. Use it on a connection that
// has just connected (Dial a new Client and call Resume, or use Reconnect on the old one).
func (c *Client) Resume(ctx context.Context, token string) error {
	return c.send(ctx, message{"type": "resume", "token": token})
}

// Reconnect redials and resumes this client's session after a Disconnected event. The same Events channel keeps
// delivering, starting with a Welcome and, if the player was in a game, Joined and State.
func (c *Client) Reconnect(ctx context.Context) error {
	token := c.Token()

	if err := c.connect(ctx); err != nil {
		return err
	}

	return c.Resume(ctx, token)
}

// Close ends the connection. It does not close Events.
func (c *Client) Close() error {
	c.mu.Lock()
	c.closing = true
	conn := c.conn
	c.mu.Unlock()

	if conn == nil {
		return nil
	}

	return conn.Close(websocket.StatusNormalClosure, "")
}
