#!/usr/bin/env bash
# Headless test of the device-lost path: checkpoints, scene recovery and the automatic restart.
#
# Needs a Vulkan GPU, so it is not part of CTest. test/gpu/device_lost.lua edits the scene and loads a textured model on
# a cold texture cache with game.debug_lose_device() armed, so the first texture batch loses the device. Then the log is
# checked: the last checkpoint per queue, the recovery file, and the restart with it (once: the cooldown stops a loop).
#
# Usage: test/gpu/device_lost_recovery.sh [debug|release]   (default: release; build it first with cargo xtask build)
set -uo pipefail

build=${1:-release}
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
binary="$root/build/$build/bin/lathe"

if [[ ! -x $binary ]]; then
    echo "SKIP: $binary is not built" >&2
    exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
log="$work/run.log"
cache="$work/cache"
mkdir "$cache"

cd "$root"

# An empty PATH hides zenity/kdialog/xmessage (the dialog would block waiting for a click); a private XDG_CACHE_HOME
# gives a cold texture cache and keeps the test away from the real one.
timeout 120 env XDG_CACHE_HOME="$cache" PATH=/nonexistent \
    "$binary" --game=lua --script=test/gpu/device_lost.lua --screen-type=headless >"$log" 2>&1
status=$?

failures=0

expect() {
    local description=$1 pattern=$2 minimum=${3:-1} maximum=${4:-}
    local count
    count=$(grep -c -E -- "$pattern" "$log")

    if ((count < minimum)) || { [[ -n $maximum ]] && ((count > maximum)); }; then
        echo "FAIL: $description (pattern '$pattern' matched $count times)"
        failures=$((failures + 1))
    else
        echo "ok:   $description"
    fi
}

expect "the encoder dispatch is the last checkpoint on the background queue" \
    "Checkpoints, background queue: last reached 'texture_encode: dispatched'"
expect "the device loss is reported" "The GPU device was lost or stopped responding"
expect "the unsaved scene is recovered" "Recovered the unsaved scene to"
expect "the editor restarts with it, once" "Restarting with the recovered scene" 1 1
expect "the restarted editor opens the recovered scene" "load_scene: '.*/recovered/.*\.lbf'"

recovered=$(find "$cache" -path '*/recovered/*.lbf' -size +0c | head -1)

if [[ -n $recovered ]]; then
    echo "ok:   the recovery file exists ($(basename "$recovered"))"
else
    echo "FAIL: no non-empty recovery file under the cache"
    failures=$((failures + 1))
fi

if ((failures != 0)); then
    echo "--- last lines of the log ($status):"
    tail -40 "$log"
    cp "$log" "${TMPDIR:-/tmp}/device_lost_recovery.log"
    echo "full log kept at ${TMPDIR:-/tmp}/device_lost_recovery.log"
    exit 1
fi

echo "PASS"
