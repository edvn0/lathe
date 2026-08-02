#!/usr/bin/env bash
set -euo pipefail

readonly image="cross-build:latest"
readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly build_type="${CMAKE_BUILD_TYPE:-Debug}"
readonly cpm_cache_dir="${HOME}/.cache/CPM"
readonly container_cpm_cache="/cpm-cache"

# windows-mingw: cross-compile to Windows via mingw-w64 (toolchain file lives in-repo)
# linux-native:  build natively inside the container with its own gcc/g++
readonly target="${TARGET:-windows-mingw}"

toolchain_file=""
build_dir=""

case "${target}" in
windows-mingw)
  toolchain_file="/workspace/cmake/toolchains/windows-mingw-x64.cmake"
  build_dir="build/windows-mingw-${build_type,,}"
  ;;
linux-native)
  build_dir="build/linux-native-${build_type,,}"
  ;;
*)
  echo "Unknown TARGET: ${target} (expected windows-mingw or linux-native)" >&2
  exit 1
  ;;
esac

usage() {
  cat <<EOF
Usage:
  TARGET=windows-mingw ./build.sh --configure   (default)
  TARGET=linux-native  ./build.sh --configure
  ./build.sh --build
  ./build.sh --rebuild
  ./build.sh --clean
  ./build.sh --shell
Options:
  --configure  Configure the build with CMake
  --build      Build the existing configuration
  --rebuild    Remove, configure, and build
  --clean      Remove the current target's build directory
  --shell      Open an interactive shell inside the build container

TARGET selects the build:
  windows-mingw  Cross-compile to Windows via mingw-w64 (toolchain file at cmake/toolchains/)
  linux-native   Build natively inside the container
EOF
}

run_container() {
  docker run --rm \
    --mount "type=bind,source=${project_dir},target=/workspace" \
    --mount "type=bind,source=${cpm_cache_dir},target=${container_cpm_cache}" \
    --env "CPM_SOURCE_CACHE=${container_cpm_cache}" \
    --workdir /workspace \
    "${image}" \
    "$@"
}

configure() {
  mkdir -p "${cpm_cache_dir}"

  local cmake_args=(-S . -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE="${build_type}")
  if [[ -n "${toolchain_file}" ]]; then
    cmake_args+=(--toolchain "${toolchain_file}")
  fi

  run_container cmake "${cmake_args[@]}" 2>&1 | sed -u "s#/workspace#${project_dir}#g"
}

build() {
  run_container \
    cmake \
    --build "${build_dir}" \
    --parallel \
    2>&1 | sed -u "s#/workspace#${project_dir}#g"
}

clean() {
  rm -rf "${project_dir:?}/${build_dir}"
}

shell() {
  docker run --rm -it \
    --mount "type=bind,source=${project_dir},target=/workspace" \
    --workdir /workspace \
    "${image}" \
    bash
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
  --configure)
    configure
    ;;
  --build)
    build
    ;;
  --rebuild)
    clean
    configure
    build
    ;;
  --clean)
    clean
    ;;
  --shell)
    shell
    ;;
  --help | -h)
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 1
    ;;
  esac
}

main "$@"