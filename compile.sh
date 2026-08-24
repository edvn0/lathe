#!/usr/bin/env bash
set -euo pipefail

readonly image="cross-build:latest"
readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly build_type="${CMAKE_BUILD_TYPE:-Debug}"
readonly renderdoc_include_path=${RENDERDOC_INCLUDE_PATH:-}
readonly cpm_cache_dir="${HOME}/.cache/CPM"
readonly container_cpm_cache="/cpm-cache"

# windows-mingw: cross-compile to Windows via mingw-w64 (toolchain file lives in-repo)
# linux-native:  build natively inside the container with its own gcc/g++
readonly target="${TARGET:-windows-mingw}"

# perf: perf stat/record (needs to match host kernel, so it always runs on the host)
# callgrind: valgrind --tool=callgrind (also runs on the host, kernel-independent)
readonly profiler="${PROFILER:-perf}"

# fill in your actual binary name/path relative to build_dir
readonly executable_name="${EXECUTABLE_NAME:-mingw-vulkan}"

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
  TARGET=linux-native ./build.sh --profile [-- extra args to the binary]
Options:
  --configure  Configure the build with CMake
  --build      Build the existing configuration
  --rebuild    Remove, configure, and build
  --clean      Remove the current target's build directory
  --shell      Open an interactive shell inside the build container
  --profile    Run the linux-native RelWithDebInfo binary under a profiler on
               the host (perf by default; PROFILER=perf-record or
               PROFILER=callgrind for the others)

TARGET selects the build:
  windows-mingw  Cross-compile to Windows via mingw-w64 (toolchain file at cmake/toolchains/)
  linux-native   Build natively inside the container

PROFILE_BUILD=1 adds -fno-omit-frame-pointer to a linux-native configure,
for cheaper 'perf record --call-graph fp' unwinding instead of dwarf.
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

  if [[ -n "${renderdoc_include_path}" ]]; then
    cmake_args+=("-DRENDERDOC_INCLUDE_PATH=${renderdoc_include_path}")
  fi

  if [[ "${PROFILE_BUILD:-0}" == "1" ]]; then
    cmake_args+=(-DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" -DCMAKE_C_FLAGS="-fno-omit-frame-pointer")
  fi

  run_container cmake "${cmake_args[@]}" 2>&1 | sed -u "s#/workspace#${project_dir}#g"
  fixup_compile_commands
}

build() {
  run_container \
    cmake \
    --build "${build_dir}" \
    -j20 \
    2>&1 | sed -u "s#/workspace#${project_dir}#g"
  fixup_compile_commands
}

fixup_compile_commands() {
  local cc="${project_dir}/${build_dir}/compile_commands.json"
  [[ -f "${cc}" ]] || return 0

  sed -i \
    -e "s#/workspace#${project_dir}#g" \
    -e "s#/cpm-cache#${cpm_cache_dir}#g" \
    "${cc}"

  ln -sf "${build_dir}/compile_commands.json" "${project_dir}/compile_commands.json"
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

profile() {
  if [[ "${target}" != "linux-native" ]]; then
    echo "--profile requires TARGET=linux-native (perf/valgrind need to match the host)" >&2
    exit 1
  fi

  local exe="${project_dir}/${build_dir}/bin/${executable_name}"
  if [[ ! -x "${exe}" ]]; then
    echo "Executable not found: ${exe} (build it first: TARGET=linux-native ./build.sh --build)" >&2
    exit 1
  fi

  case "${profiler}" in
  perf)
    perf stat -d -- "${exe}" "$@"
    ;;
  perf-record)
    local out="${project_dir}/${build_dir}/perf.data"
    local graph_mode="dwarf"
    [[ "${PROFILE_BUILD:-0}" == "1" ]] && graph_mode="fp"
    perf record -g --call-graph "${graph_mode}" -o "${out}" -- "${exe}" "$@"
    echo "Report with: perf report -i ${out}"
    ;;
  callgrind)
    local out="${project_dir}/${build_dir}/callgrind.out.%p"
    valgrind --tool=callgrind --callgrind-out-file="${out}" -- "${exe}" "$@"
    echo "Open the resulting callgrind.out.<pid> file with kcachegrind"
    ;;
  *)
    echo "Unknown PROFILER: ${profiler} (expected perf, perf-record, or callgrind)" >&2
    exit 1
    ;;
  esac
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi

  local cmd="$1"
  shift

  case "${cmd}" in
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
  --profile)
    profile "$@"
    ;;
  --help | -h)
    usage
    ;;
  *)
    echo "Unknown option: ${cmd}" >&2
    usage >&2
    exit 1
    ;;
  esac
}

main "$@"
