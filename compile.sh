#!/usr/bin/env bash
set -euo pipefail

readonly image="cross-build:latest"
readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly build_type="${CMAKE_BUILD_TYPE:-Debug}"
readonly renderdoc_include_path="${RENDERDOC_INCLUDE_PATH:-}"
readonly cpm_cache_dir="${HOME}/.cache/CPM"

# The image has no non-root user, so without --user every file a rootful
# daemon's container writes into a bind mount (build/, ~/.cache/CPM) ends up
# root-owned on the host. Running as the invoking host user instead avoids
# that; HOME is repointed at a writable, persistent dir since root's HOME
# (/root) isn't accessible to this UID and isn't a real passwd entry either.
#
# A rootless daemon needs the opposite: it already maps container root (uid
# 0) back to the host user that started it, so bind-mounted writes as root
# come out correctly owned without any --user. Passing --user "<uid>:<gid>"
# there instead asks for a *different* container uid, which the daemon maps
# to some unrelated, unmapped subordinate host uid -- one that doesn't own
# the bind-mounted directories -- so every write fails with EACCES. See
# add_docker_user_args below.
readonly container_home="${HOME}/.cache/cross-build-container-home"

# windows-mingw: cross-compile to Windows via mingw-w64.
# linux-native:  build natively inside the container with its gcc/g++.
readonly target="${TARGET:-windows-mingw}"

# default: use the compiler/toolchain default linker.
# mold:    use mold for native Linux builds.
readonly linker="${LINKER:-default}"

# perf:        perf stat
# perf-record: perf record
# callgrind:   valgrind --tool=callgrind
#
# Profiling always runs on the host.
readonly profiler="${PROFILER:-perf}"

# Binary name/path relative to ${build_dir}/bin.
readonly executable_name="${EXECUTABLE_NAME:-mingw-vulkan}"

toolchain_file=""
build_dir=""

case "${target}" in
windows-mingw)
  toolchain_file="${project_dir}/cmake/toolchains/windows-mingw-x64.cmake"
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

case "${linker}" in
default)
  ;;

mold)
  if [[ "${target}" != "linux-native" ]]; then
    echo "LINKER=mold requires TARGET=linux-native" >&2
    exit 1
  fi
  ;;

*)
  echo "Unknown LINKER: ${linker} (expected default or mold)" >&2
  exit 1
  ;;
esac

usage() {
  cat <<EOF
Usage:
  TARGET=windows-mingw ./compile.sh --configure
  TARGET=linux-native  ./compile.sh --configure

  ./compile.sh --build
  ./compile.sh --rebuild
  ./compile.sh --clean
  ./compile.sh --shell

  TARGET=linux-native ./compile.sh --test
  TARGET=linux-native ./compile.sh --profile [-- extra args to the binary]

Options:
  --configure
      Configure the selected build with CMake.

  --build
      Build the existing configuration.

  --rebuild
      Remove the selected build directory, configure, and build.

  --clean
      Remove the selected target's build directory.

  --shell
      Open an interactive shell inside the build container.

  --test
      Run CTest for a linux-native build.

  --profile
      Run the linux-native binary under a profiler on the host.

      PROFILER=perf
          Uses perf stat.

      PROFILER=perf-record
          Uses perf record.

      PROFILER=callgrind
          Uses Valgrind/Callgrind.

TARGET selects the build:
  windows-mingw
      Cross-compile to Windows using mingw-w64 and the toolchain file
      under cmake/toolchains/.

  linux-native
      Build natively for Linux inside the container.

LINKER selects the linker:
  default
      Use the compiler/toolchain default linker.

  mold
      Use mold via CMAKE_LINKER_TYPE=MOLD.
      Requires TARGET=linux-native and mold installed in the container.

CMAKE_BUILD_TYPE selects the configuration:
  Debug
  Release
  RelWithDebInfo
  MinSizeRel

RENDERDOC_INCLUDE_PATH optionally points to the host directory containing:

  renderdoc_app.h

For example:

  /home/edwin/git/renderdoc/renderdoc/api/app

When set, that directory is mounted read-only into the build container at
the same absolute path and passed to CMake as RENDERDOC_INCLUDE_PATH.

PROFILE_BUILD=1 adds -fno-omit-frame-pointer to a linux-native configure
for cheaper:

  perf record --call-graph fp

unwinding instead of DWARF.

SANITIZE=1 configures a linux-native build with AddressSanitizer and
UndefinedBehaviorSanitizer enabled.

WERROR=1 treats warnings as errors on this project's own targets
(mingw-vulkan-core, mingw-vulkan, mingw-vulkan-tests) -- not on vendored
dependencies.

Examples:

  TARGET=linux-native \
    CMAKE_BUILD_TYPE=Debug \
    ./compile.sh --rebuild

  TARGET=linux-native \
    CMAKE_BUILD_TYPE=RelWithDebInfo \
    LINKER=mold \
    ./compile.sh --rebuild

  TARGET=linux-native \
    CMAKE_BUILD_TYPE=RelWithDebInfo \
    LINKER=mold \
    RENDERDOC_INCLUDE_PATH=/home/edwin/git/renderdoc/renderdoc/api/app \
    ./compile.sh --rebuild

  TARGET=linux-native \
    CMAKE_BUILD_TYPE=RelWithDebInfo \
    LINKER=mold \
    PROFILE_BUILD=1 \
    ./compile.sh --rebuild

  TARGET=windows-mingw \
    CMAKE_BUILD_TYPE=Release \
    ./compile.sh --rebuild

  TARGET=linux-native \
    CMAKE_BUILD_TYPE=Debug \
    SANITIZE=1 \
    ./compile.sh --rebuild
EOF
}

validate_renderdoc_path() {
  [[ -n "${renderdoc_include_path}" ]] || return 0

  if [[ ! -d "${renderdoc_include_path}" ]]; then
    echo "RENDERDOC_INCLUDE_PATH is not a directory:" >&2
    echo "  ${renderdoc_include_path}" >&2
    exit 1
  fi

  if [[ ! -f "${renderdoc_include_path}/renderdoc_app.h" ]]; then
    echo "renderdoc_app.h was not found:" >&2
    echo "  ${renderdoc_include_path}/renderdoc_app.h" >&2
    exit 1
  fi
}

# Cached in _rootless_docker so `docker info` only runs once per invocation.
_rootless_docker=""

is_rootless_docker() {
  if [[ -z "${_rootless_docker}" ]]; then
    if docker info --format '{{.SecurityOptions}}' 2>/dev/null | grep -q 'name=rootless'; then
      _rootless_docker=1
    else
      _rootless_docker=0
    fi
  fi

  [[ "${_rootless_docker}" == 1 ]]
}

# Appends --user to the named array (a nameref) only for a rootful daemon --
# see container_home's doc comment above for why a rootless one must not get
# it.
add_docker_user_args() {
  local -n out_args="$1"

  is_rootless_docker && return 0

  out_args+=(--user "$(id -u):$(id -g)")
}

run_container() {
  mkdir -p "${cpm_cache_dir}"
  mkdir -p "${container_home}"

  validate_renderdoc_path

  local docker_args=(
    run
    --rm
    --mount "type=bind,source=${project_dir},target=${project_dir}"
    --mount "type=bind,source=${cpm_cache_dir},target=${cpm_cache_dir}"
    --mount "type=bind,source=${container_home},target=${container_home}"
    --env "CPM_SOURCE_CACHE=${cpm_cache_dir}"
    --env "HOME=${container_home}"
    --workdir "${project_dir}"
  )

  add_docker_user_args docker_args

  if [[ -n "${renderdoc_include_path}" ]]; then
    docker_args+=(
      --mount "type=bind,source=${renderdoc_include_path},target=${renderdoc_include_path},readonly"
    )
  fi

  docker \
    "${docker_args[@]}" \
    "${image}" \
    "$@"
}

update_compile_commands_link() {
  local compile_commands="${project_dir}/${build_dir}/compile_commands.json"

  [[ -f "${compile_commands}" ]] || return 0

  ln -sfn \
    "${build_dir}/compile_commands.json" \
    "${project_dir}/compile_commands.json"
}

configure() {
  mkdir -p "${cpm_cache_dir}"

  validate_renderdoc_path

  local cmake_args=(
    -S "${project_dir}"
    -B "${project_dir}/${build_dir}"
    -G Ninja
    -DCMAKE_BUILD_TYPE="${build_type}"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  )

  if [[ -n "${toolchain_file}" ]]; then
    cmake_args+=(
      --toolchain "${toolchain_file}"
    )
  fi

  if [[ "${linker}" == "mold" ]]; then
    if ! run_container sh -c 'command -v mold >/dev/null 2>&1'; then
      echo "LINKER=mold requested, but mold is not installed in ${image}" >&2
      exit 1
    fi

    cmake_args+=(
      -DCMAKE_LINKER_TYPE=MOLD
    )
  fi

  if [[ -n "${renderdoc_include_path}" ]]; then
    cmake_args+=(
      "-DRENDERDOC_INCLUDE_PATH=${renderdoc_include_path}"
    )
  fi

  if [[ "${PROFILE_BUILD:-0}" == "1" ]]; then
    if [[ "${target}" != "linux-native" ]]; then
      echo "PROFILE_BUILD=1 requires TARGET=linux-native" >&2
      exit 1
    fi

    cmake_args+=(
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"
      -DCMAKE_C_FLAGS="-fno-omit-frame-pointer"
    )
  fi

  if [[ "${SANITIZE:-0}" == "1" ]]; then
    if [[ "${target}" != "linux-native" ]]; then
      echo "SANITIZE=1 requires TARGET=linux-native" >&2
      exit 1
    fi

    cmake_args+=(
      -DMINGW_VULKAN_SANITIZE=ON
    )
  fi

  if [[ "${WERROR:-0}" == "1" ]]; then
    cmake_args+=(
      -DMINGW_VULKAN_WERROR=ON
    )
  fi

  run_container cmake "${cmake_args[@]}"
  update_compile_commands_link
}

build() {
  run_container \
    cmake \
    --build "${project_dir}/${build_dir}" \
    -j20

  update_compile_commands_link
}

clean() {
  rm -rf "${project_dir:?}/${build_dir}"

  if [[ -L "${project_dir}/compile_commands.json" ]]; then
    local current_target
    current_target="$(readlink "${project_dir}/compile_commands.json")"

    if [[ "${current_target}" == "${build_dir}/compile_commands.json" ]]; then
      rm "${project_dir}/compile_commands.json"
    fi
  fi
}

shell() {
  mkdir -p "${cpm_cache_dir}"

  validate_renderdoc_path

  mkdir -p "${container_home}"

  local docker_args=(
    run
    --rm
    -it
    --mount "type=bind,source=${project_dir},target=${project_dir}"
    --mount "type=bind,source=${cpm_cache_dir},target=${cpm_cache_dir}"
    --mount "type=bind,source=${container_home},target=${container_home}"
    --env "CPM_SOURCE_CACHE=${cpm_cache_dir}"
    --env "HOME=${container_home}"
    --workdir "${project_dir}"
  )

  add_docker_user_args docker_args

  if [[ -n "${renderdoc_include_path}" ]]; then
    docker_args+=(
      --mount "type=bind,source=${renderdoc_include_path},target=${renderdoc_include_path},readonly"
    )
  fi

  docker \
    "${docker_args[@]}" \
    "${image}" \
    bash
}

run_test() {
  if [[ "${target}" != "linux-native" ]]; then
    echo "--test requires TARGET=linux-native" >&2
    exit 1
  fi

  run_container \
    ctest \
    --test-dir "${project_dir}/${build_dir}" \
    --output-on-failure \
    "$@"
}

profile() {
  if [[ "${target}" != "linux-native" ]]; then
    echo "--profile requires TARGET=linux-native (perf/valgrind run on the host)" >&2
    exit 1
  fi

  local exe="${project_dir}/${build_dir}/bin/${executable_name}"

  if [[ ! -x "${exe}" ]]; then
    echo "Executable not found: ${exe}" >&2
    echo "Build it first with:" >&2
    echo "  TARGET=linux-native CMAKE_BUILD_TYPE=${build_type} ./compile.sh --build" >&2
    exit 1
  fi

  case "${profiler}" in
  perf)
    perf stat \
      -d \
      -- "${exe}" "$@"
    ;;

  perf-record)
    local out="${project_dir}/${build_dir}/perf.data"
    local graph_mode="dwarf"

    if [[ "${PROFILE_BUILD:-0}" == "1" ]]; then
      graph_mode="fp"
    fi

    perf record \
      -g \
      --call-graph "${graph_mode}" \
      -o "${out}" \
      -- "${exe}" "$@"

    echo "Report with:"
    echo "  perf report -i ${out}"
    ;;

  callgrind)
    local out="${project_dir}/${build_dir}/callgrind.out.%p"

    valgrind \
      --tool=callgrind \
      --callgrind-out-file="${out}" \
      -- "${exe}" "$@"

    echo "Open the resulting callgrind.out.<pid> file with kcachegrind"
    ;;

  *)
    echo "Unknown PROFILER: ${profiler}" >&2
    echo "Expected: perf, perf-record, or callgrind" >&2
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

  --test)
    run_test "$@"
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
