#!/usr/bin/env bash
# Build caches for CI jobs on the self-hosted runner, kept in a Docker volume mounted at /ci-cache (see the
# `container:` blocks in .github/workflows/). They persist on the runner instead of round-tripping through the
# repository's 10 GB actions/cache quota.
#
#   tools/ci_cache.sh env                     append the cache locations to $GITHUB_ENV
#   tools/ci_cache.sh cargo-target-dir <rev>  print the tools/shader_reflect cargo target directory for <rev>
#                                             (nothing if <rev> has no tools/shader_reflect)
#
# CPM's source cache, ccache and cargo all lock what they share, so concurrent jobs can use the same volume.
set -euo pipefail

readonly cache_root="${CI_CACHE_ROOT:-/ci-cache}"
readonly cargo_target_root="${cache_root}/cargo-target"

# Target directories unused for this many days are removed; each build of the same sources touches its own.
readonly cargo_target_max_age_days=14

write_env() {
  : "${GITHUB_ENV:?not running under GitHub Actions}"

  {
    echo "CPM_SOURCE_CACHE=${cache_root}/cpm"
    echo "CCACHE_DIR=${cache_root}/ccache"
    echo "CLANG_TIDY_CACHE_DIR=${cache_root}/clang-tidy"
  } >> "${GITHUB_ENV}"

  mkdir -p "${cargo_target_root}"
  find "${cargo_target_root}" -mindepth 1 -maxdepth 1 -type d -mtime "+${cargo_target_max_age_days}" \
    -exec rm -rf {} +
}

# Keyed by the git tree hash of tools/shader_reflect: builds of differing sources never share (and overwrite) a
# binary, while every build of the same sources reuses the compiled crates.
print_cargo_target_dir() {
  local revision="$1"
  local tree

  if ! tree="$(git rev-parse --verify --quiet "${revision}:tools/shader_reflect")"; then
    return 0
  fi

  local target_dir="${cargo_target_root}/${tree}"
  mkdir -p "${target_dir}"
  touch "${target_dir}"

  echo "${target_dir}"
}

case "${1:-}" in
  env)
    write_env
    ;;
  cargo-target-dir)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 cargo-target-dir <rev>" >&2
      exit 2
    fi
    print_cargo_target_dir "$2"
    ;;
  *)
    echo "usage: $0 env | cargo-target-dir <rev>" >&2
    exit 2
    ;;
esac
