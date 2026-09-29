#!/usr/bin/env bash
# Runs clang-tidy, configured by .clang-tidy, over the project's own sources using <build_dir>'s compile database.
#
#   tools/run_clang_tidy.sh <build_dir> [extra run-clang-tidy args...]
#
# e.g.  tools/run_clang_tidy.sh build/linux-native-debug
#       tools/run_clang_tidy.sh build/linux-native-debug -fix
#
# The build must have run first: some sources include headers generated at build time. Exits nonzero on any
# finding, since .clang-tidy treats every warning as an error.
#
# Files that already passed with identical input are skipped (see tools/clang_tidy_cached.py). The cache lives in
# CLANG_TIDY_CACHE_DIR (default ${XDG_CACHE_HOME:-~/.cache}/clang-tidy); CLANG_TIDY_CACHE_DIR=off disables it.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <build_dir> [extra run-clang-tidy args...]" >&2
  exit 2
fi

readonly build_dir="$1"
shift

readonly project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${build_dir}/compile_commands.json" ]]; then
  echo "No compile_commands.json in ${build_dir}; configure and build it first." >&2
  exit 1
fi

# Distributions ship these with a -<major> suffix, unversioned, or both; prefer the newest versioned one.
# CLANG_TIDY / RUN_CLANG_TIDY / CLANG_APPLY_REPLACEMENTS override the lookup.
find_tool() {
  local name="$1"
  local candidate

  candidate="$(compgen -c "${name}-" | grep -E "^${name}-[0-9]+$" | sort -V -r | head -n 1 || true)"

  if [[ -z "${candidate}" ]] && command -v "${name}" >/dev/null 2>&1; then
    candidate="${name}"
  fi

  if [[ -z "${candidate}" ]]; then
    echo "${name} not found" >&2
    return 1
  fi

  command -v "${candidate}"
}

run_clang_tidy="${RUN_CLANG_TIDY:-$(find_tool run-clang-tidy)}"
clang_tidy="${CLANG_TIDY:-$(find_tool clang-tidy)}"

cache_dir="${CLANG_TIDY_CACHE_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/clang-tidy}"
tidy_binary="${clang_tidy}"

if [[ "${cache_dir}" != "off" ]]; then
  mkdir -p "${cache_dir}"
  # Hits refresh their entry's mtime, so this only drops keys no file has produced for a month.
  find "${cache_dir}" -type f -mtime +30 -delete

  export CLANG_TIDY_CACHE_DIR="${cache_dir}"
  export CLANG_TIDY_REAL="${clang_tidy}"
  tidy_binary="${project_dir}/tools/clang_tidy_cached.py"
fi

args=(
  -p "${build_dir}"
  -clang-tidy-binary "${tidy_binary}"
  -quiet
  -j "$(nproc)"
)

if clang_apply_replacements="${CLANG_APPLY_REPLACEMENTS:-$(find_tool clang-apply-replacements 2>/dev/null)}"; then
  args+=(-clang-apply-replacements-binary "${clang_apply_replacements}")
fi

"${clang_tidy}" --version | head -n 2

exec "${run_clang_tidy}" "${args[@]}" "$@" "^${project_dir}/(src|game|test)/.*\.cxx$"
