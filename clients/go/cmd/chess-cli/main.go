// chess-cli plays chess on a lathe-server from the terminal.
//
//	chess-cli -url ws://127.0.0.1:9002 -create        host a room, then wait for an opponent
//	chess-cli -url ws://127.0.0.1:9002 -join 1        join room 1
//
// Type moves like e2e4 (e7e8q to promote). Ctrl-D resigns by leaving. The session token is printed on connect; if the
// connection drops the client resumes on its own.
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"time"

	"lathe/client/lathe"
	"lathe/client/lathe/chess"
)

func main() {
	url := flag.String("url", "ws://127.0.0.1:9002", "lathe-server WebSocket URL")
	create := flag.Bool("create", false, "create a room and wait for an opponent")
	join := flag.Uint("join", 0, "join this room id")
	flag.Parse()

	if *create == (*join != 0) {
		fmt.Fprintln(os.Stderr, "chess-cli: pass exactly one of -create or -join N")
		os.Exit(2)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	client, err := lathe.Dial(ctx, *url)
	if err != nil {
		fmt.Fprintln(os.Stderr, "chess-cli:", err)
		os.Exit(1)
	}
	defer client.Close()

	moves := make(chan string)

	go func() {
		scanner := bufio.NewScanner(os.Stdin)
		for scanner.Scan() {
			if err := scanner.Err(); err != nil {
				fmt.Fprintln(os.Stderr, "chess-cli: error reading input:", err)
				close(moves)
				return
			}
			moves <- strings.TrimSpace(scanner.Text())
		}

		close(moves)
	}()

	var state chess.State

	for {
		select {
		case <-ctx.Done():
			return

		case line, ok := <-moves:
			if !ok {
				client.LeaveRoom(ctx)
				return
			}

			if line == "" {
				continue
			}

			if err := chess.Move(ctx, client, line); err != nil {
				fmt.Println(err)
			}

		case event := <-client.Events():
			switch e := event.(type) {
			case lathe.Welcome:
				fmt.Printf("connected as player %d (token %s)\n", e.Player, e.Token)

				if *create {
					client.CreateRoom(ctx, chess.Game)
				} else {
					client.JoinRoom(ctx, uint32(*join))
				}

				// Only the first welcome starts a game; a resume's welcome is followed by Joined and State.
				*create, *join = false, 0

			case lathe.Joined:
				fmt.Printf("room %d, seat %d (%s)\n", e.Room, e.Seat, [2]string{"white", "black"}[e.Seat%2])

			case lathe.State:
				if state, err = chess.ParseState(e); err != nil {
					fmt.Println("bad state:", err)
					continue
				}

				fmt.Print(state)

				switch {
				case state.MyTurn() && state.InCheck:
					fmt.Println("your move (check):")
				case state.MyTurn():
					fmt.Println("your move:")
				default:
					fmt.Println("waiting for", state.SideToMove)
				}

			case lathe.GameOver:
				switch {
				case e.Winner == nil:
					fmt.Println("draw:", e.Reason)
				case *e.Winner == client.Player():
					fmt.Println("you win:", e.Reason)
				default:
					fmt.Println("you lose:", e.Reason)
				}

				return

			case lathe.OpponentDisconnected:
				fmt.Println("opponent disconnected; holding their seat")
			case lathe.OpponentReconnected:
				fmt.Println("opponent is back")
			case lathe.RoomClosed:
				fmt.Println("room closed:", e.Reason)
				return
			case lathe.Left:
				return
			case lathe.ServerError:
				fmt.Println("server:", e)

			case lathe.Disconnected:
				fmt.Println("connection lost:", e.Err)

				for attempt := 0; ; attempt++ {
					if ctx.Err() != nil {
						return
					}

					time.Sleep(time.Second)

					if client.Reconnect(ctx) == nil {
						break
					}
				}
			}
		}
	}
}
