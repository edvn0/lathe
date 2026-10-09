#!/usr/bin/env bash
# Starts lathe-server and two Go chess-cli players (white and black) in one tmux session.
#
#   scripts/play-chess.sh                 build what is missing, then start
#   scripts/play-chess.sh --engine        black is the engine's own UI instead of chess-cli: pick Play online, Connect,
#                                         then Join room (room 1, already filled in)
#   PORT=9100 scripts/play-chess.sh       use another port
#   BUILD_TYPE=release scripts/play-chess.sh
#
# Type moves like e2e4 in either player pane. Ctrl-b then d detaches; `tmux kill-session -t lathe-chess` stops it all.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_type="${BUILD_TYPE:-debug}"
port="${PORT:-9002}"
session="lathe-chess"

use_engine=0
[[ "${1:-}" == "--engine" ]] && use_engine=1

server="$root/build/$build_type/bin/lathe-server"
engine="$root/build/$build_type/bin/lathe"
cli="$root/build/$build_type/bin/chess-cli"

command -v tmux >/dev/null || { echo "play-chess: tmux is required" >&2; exit 1; }
command -v go >/dev/null || { echo "play-chess: go is required" >&2; exit 1; }

if [[ ! -x "$server" || ( $use_engine == 1 && ! -x "$engine" ) ]]; then
    echo "Building..."
    (cd "$root" && cargo xtask build)
fi

echo "Building chess-cli..."
(cd "$root/clients/go" && go build -o "$cli" ./cmd/chess-cli)

tmux kill-session -t "$session" 2>/dev/null || true

url="ws://127.0.0.1:$port"

# Top pane: the server. Bottom panes: white (creates room 1 on a fresh server) and black (joins it).
tmux new-session -d -s "$session" -x 200 -y 50 "'$server' --port $port; read -rp 'server exited, press enter'"
tmux split-window -v -t "$session" -l 75% \
    "sleep 1; '$cli' -url $url -create; read -rp 'white finished, press enter'"
if [[ $use_engine == 1 ]]; then
    tmux split-window -h -t "$session" \
        "cd '$root/build/$build_type/bin' && ./lathe --player --game lua --script assets/scripts/chess/main.lua; read -rp 'engine exited, press enter'"
else
    tmux split-window -h -t "$session" \
        "sleep 2; '$cli' -url $url -join 1; read -rp 'black finished, press enter'"
fi
tmux select-pane -t "$session.1"

if [[ -n "${TMUX:-}" ]]; then
    tmux switch-client -t "$session"
else
    tmux attach -t "$session"
fi
