#!/usr/bin/env bash
# Runs a spread of benchmark scenarios and prints "<case> <frame checksum>" for each, so two builds can be compared
# with diff: the checksums are equal when prepare_frame produced exactly the same transforms, draw records, indirect
# commands and cull bounds for every measured frame.
#
#   scripts/frame_checksums.sh OUT_DIR [lathe binary]
set -euo pipefail

out=${1:?usage: frame_checksums.sh OUT_DIR [lathe binary]}
binary=${2:-build/release/bin/lathe}

rm -rf "$out"
"$binary" --screen-type=headless --benchmark-suite "$out" \
    --benchmark-scenarios draw_calls,instancing,shadow_lights,shadow_city,lights,overdraw,grass,cpu_submit,cpu_hierarchy \
    --benchmark-sweep draw_calls:4000 --benchmark-sweep instancing:10000 --benchmark-sweep shadow_lights:4 \
    --benchmark-sweep shadow_city:8 --benchmark-sweep lights:64 --benchmark-sweep overdraw:4 \
    --benchmark-sweep grass:20000 --benchmark-sweep cpu_submit:25000 --benchmark-sweep cpu_hierarchy:8 \
    --benchmark-frame-checksum --benchmark-render-size 320x180 --benchmark-repeats 1 --benchmark-frames 90 --benchmark-warmup 20 \
    > "$out.log" 2>&1

python3 - "$out/suite.json" <<'PY'
import json, sys
for case in json.load(open(sys.argv[1]))["cases"]:
    print(case["key"], case["analysis"]["workload"]["frame_checksum"])
PY
