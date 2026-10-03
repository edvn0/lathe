#!/usr/bin/env bash
# Captures the four Phase 0 baselines (docs/frame-graph.md) of a built lathe: for each variant, the benchmark JSON and
# the keyframe screenshots.
#
#   tools/perf/capture_baselines.sh <build_dir> <out_dir> [extra app args...]
#
# e.g.  tools/perf/capture_baselines.sh build/debug perf/baseline
#
# writes <out_dir>/<variant>/{benchmark.json,screenshots/*.png} for the variants occ_off, occ_on, occ_meshlet and
# always_defer. Compare a later capture with tools/perf/compare_screenshots.py --base <a>/<variant>/screenshots
# --head <b>/<variant>/screenshots. Two captures of one build must compare identical.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <build_dir> <out_dir> [extra app args...]" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")" && pwd)
build_dir=$(cd "$1" && pwd)
out_dir=$(realpath -m "$2")
shift 2

declare -A variants=(
  [occ_off]="--occlusion-culling=off"
  [occ_on]="--occlusion-culling=on"
  [occ_meshlet]="--occlusion-culling=on --meshlet-occlusion=on"
  [always_defer]="--occlusion-culling=on --occlusion-test=always_defer"
)

for variant in occ_off occ_on occ_meshlet always_defer; do
  dir="${out_dir}/${variant}"
  rm -rf "${dir}"
  mkdir -p "${dir}"

  # shellcheck disable=SC2086  # the variant flags are meant to split
  "${here}/run_benchmark.sh" "${build_dir}" "${dir}/benchmark.json" --benchmark-screenshots \
    ${variants[${variant}]} "$@"

  mv "${build_dir}/bin/screenshots" "${dir}/screenshots"
  echo "${variant}: $(ls "${dir}/screenshots" | wc -l) screenshots"
done
