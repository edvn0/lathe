package lathe_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"testing"
	"time"

	"lathe/client/lathe"
	"lathe/client/lathe/chess"
)

// These tests drive a real lathe-server. Point LATHE_SERVER at the binary (build/debug/bin/lathe-server).

func startServer(t *testing.T, extraArgs ...string) string {
	t.Helper()

	binary := os.Getenv("LATHE_SERVER")
	if binary == "" {
		t.Skip("set LATHE_SERVER to the lathe-server binary to run the end-to-end tests")
	}

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	port := listener.Addr().(*net.TCPAddr).Port
	listener.Close()

	args := append([]string{"--port", fmt.Sprint(port)}, extraArgs...)
	server := exec.Command(binary, args...)

	if err := server.Start(); err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		server.Process.Signal(os.Interrupt)
		server.Wait()
	})

	url := fmt.Sprintf("ws://127.0.0.1:%d", port)

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if conn, err := net.Dial("tcp", fmt.Sprintf("127.0.0.1:%d", port)); err == nil {
			conn.Close()
			return url
		}

		time.Sleep(50 * time.Millisecond)
	}

	t.Fatal("server did not start listening")

	return ""
}

func dial(t *testing.T, url string) (*lathe.Client, context.Context) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)

	client, err := lathe.Dial(ctx, url)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { client.Close() })

	return client, ctx
}

// next waits for the next event of type T, skipping others.
func next[T lathe.Event](t *testing.T, client *lathe.Client) T {
	t.Helper()

	timeout := time.After(10 * time.Second)

	for {
		select {
		case event := <-client.Events():
			if typed, ok := event.(T); ok {
				return typed
			}
		case <-timeout:
			var zero T
			t.Fatalf("timed out waiting for %T", zero)
		}
	}
}

func TestScholarsMate(t *testing.T) {
	url := startServer(t)

	white, ctx := dial(t, url)
	black, _ := dial(t, url)

	next[lathe.Welcome](t, white)
	next[lathe.Welcome](t, black)

	if err := white.CreateRoom(ctx, chess.Game); err != nil {
		t.Fatal(err)
	}

	room := next[lathe.Joined](t, white).Room

	if err := black.JoinRoom(ctx, room); err != nil {
		t.Fatal(err)
	}

	state, err := chess.ParseState(next[lathe.State](t, white))
	if err != nil {
		t.Fatal(err)
	}

	if !state.MyTurn() || len(state.LegalMoves) != 20 {
		t.Fatalf("white should open with 20 moves, got %+v", state)
	}

	next[lathe.State](t, black)

	script := []struct {
		mover *lathe.Client
		move  string
	}{
		{white, "e2e4"}, {black, "e7e5"}, {white, "f1c4"}, {black, "b8c6"},
		{white, "d1h5"}, {black, "g8f6"}, {white, "h5f7"},
	}

	for _, step := range script {
		if err := chess.Move(ctx, step.mover, step.move); err != nil {
			t.Fatal(err)
		}

		next[lathe.State](t, white)
		next[lathe.State](t, black)
	}

	for _, client := range []*lathe.Client{white, black} {
		over := next[lathe.GameOver](t, client)

		if over.Winner == nil || *over.Winner != white.Player() || over.Reason != "checkmate" {
			t.Fatalf("expected white to win by checkmate, got %+v", over)
		}
	}
}

func TestIllegalMoveIsRejectedAndTheGameContinues(t *testing.T) {
	url := startServer(t)

	white, ctx := dial(t, url)
	black, _ := dial(t, url)

	next[lathe.Welcome](t, white)
	white.CreateRoom(ctx, chess.Game)
	room := next[lathe.Joined](t, white).Room
	black.JoinRoom(ctx, room)
	next[lathe.State](t, white)

	chess.Move(ctx, white, "e2e5")

	if got := next[lathe.ServerError](t, white); got.Code != "illegal_move" {
		t.Fatalf("got %v", got)
	}

	chess.Move(ctx, white, "e2e4")
	next[lathe.State](t, white)
}

func TestDroppedPlayerResumesTheirSeat(t *testing.T) {
	url := startServer(t)

	white, ctx := dial(t, url)
	black, _ := dial(t, url)

	next[lathe.Welcome](t, white)
	next[lathe.Welcome](t, black)

	white.CreateRoom(ctx, chess.Game)
	room := next[lathe.Joined](t, white).Room
	black.JoinRoom(ctx, room)
	next[lathe.State](t, white)
	next[lathe.State](t, black)

	// Black loses their connection; white keeps playing meanwhile.
	black.Close()

	next[lathe.OpponentDisconnected](t, white)

	chess.Move(ctx, white, "e2e4")
	next[lathe.State](t, white)

	if err := black.Reconnect(ctx); err != nil {
		t.Fatal(err)
	}

	next[lathe.Welcome](t, black)
	joined := next[lathe.Joined](t, black)

	if joined.Room != room || joined.Seat != 1 {
		t.Fatalf("resumed into the wrong seat: %+v", joined)
	}

	state, err := chess.ParseState(next[lathe.State](t, black))
	if err != nil {
		t.Fatal(err)
	}

	if state.YourSide != "black" || !state.MyTurn() {
		t.Fatalf("black should be to move after e4: %+v", state)
	}

	next[lathe.OpponentReconnected](t, white)

	chess.Move(ctx, black, "e7e5")
	next[lathe.State](t, white)
}

func TestSeatExpiresAfterTheGracePeriod(t *testing.T) {
	url := startServer(t, "--reconnect-grace", "1")

	white, ctx := dial(t, url)
	black, _ := dial(t, url)

	next[lathe.Welcome](t, white)
	white.CreateRoom(ctx, chess.Game)
	room := next[lathe.Joined](t, white).Room
	black.JoinRoom(ctx, room)
	next[lathe.State](t, white)

	black.Close()

	next[lathe.OpponentDisconnected](t, white)

	if closed := next[lathe.RoomClosed](t, white); closed.Reason != "player_timeout" {
		t.Fatalf("got %+v", closed)
	}
}
