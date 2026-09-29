#!/usr/bin/env bash
# Runs one --benchmark pass of a built mingw-vulkan (see
# include/app/benchmark.hxx) and writes its JSON.
#
#   tools/perf/run_benchmark.sh <build_dir> <out.json> [extra app args...]
#
# e.g.  tools/perf/run_benchmark.sh build/linux-native-relwithdebinfo perf/head.json --benchmark-frames=240
#
# Runs from <build_dir>/bin, where the build copies assets/. With no
# display (CI, containers, headless machines) it runs with
# --screen-type=headless: GLFW's null platform and VK_EXT_headless_surface,
# so no X server is involved and any GPU the process can see is usable
# (in a container, pass e.g. --device /dev/dri). With no GPU, Mesa's
# lavapipe does the rendering. BENCHMARK_TIMEOUT (seconds, default 3600)
# bounds the run.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <build_dir> <out.json> [extra app args...]" >&2
  exit 2
fi

build_dir=$(cd "$1" && pwd)
out_json=$(realpath -m "$2")
shift 2

bin_dir="${build_dir}/bin"
if [[ ! -x "${bin_dir}/mingw-vulkan" ]]; then
  echo "error: ${bin_dir}/mingw-vulkan not found -- build the mingw-vulkan target first" >&2
  exit 2
fi

mkdir -p "$(dirname "${out_json}")"

screen_type=windowed
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  screen_type=headless
fi

run=(timeout "${BENCHMARK_TIMEOUT:-3600}" ./mingw-vulkan "--screen-type=${screen_type}" "--benchmark=${out_json}" "$@")

cd "${bin_dir}"
rm -rf screenshots
"${run[@]}"

if [[ ! -s "${out_json}" ]]; then
  echo "error: the benchmark finished without writing ${out_json}" >&2
  exit 1
fi
