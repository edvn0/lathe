#!/usr/bin/env python3
"""Stands in for clang-tidy under run-clang-tidy, skipping files that already passed with identical input.

tools/run_clang_tidy.sh passes this as -clang-tidy-binary, with the real binary in CLANG_TIDY_REAL and the cache in
CLANG_TIDY_CACHE_DIR. A file's key hashes everything that decides its result:

- its preprocessed source, comments kept (-E -C), so every included header and every NOLINT counts;
- its compile command from the -p build directory's compile_commands.json;
- every .clang-tidy from its directory up to the root, clang-tidy's --version, this script, and the arguments
  run-clang-tidy passed.

A clean run (exit 0) records the key; a later run with the same key exits 0 without running clang-tidy. Failures are
never recorded, and anything this can't key (a failed preprocess, a file missing from the database, run-clang-tidy's
-list-checks probe) runs the real clang-tidy unchanged.
"""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

# Compile-command arguments that take a value and only affect outputs, never the preprocessed source.
output_flags_with_value = {"-o", "-MF", "-MT", "-MQ"}
output_flags = {"-c", "-MD", "-MMD"}


def run_real(real_binary: str, args: list[str]) -> int:
    return subprocess.call([real_binary, *args])


def find_compile_entry(build_dir: Path, source: Path) -> dict | None:
    try:
        entries = json.loads((build_dir / "compile_commands.json").read_text())
    except (OSError, ValueError):
        return None

    for entry in entries:
        entry_file = Path(entry["directory"], entry["file"]).resolve()
        if entry_file == source:
            return entry

    return None


def compile_arguments(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def preprocess_arguments(arguments: list[str]) -> list[str]:
    result = []
    skip_value = False

    for argument in arguments:
        if skip_value:
            skip_value = False
            continue
        if argument in output_flags_with_value:
            skip_value = True
            continue
        if argument in output_flags:
            continue
        result.append(argument)

    return [*result, "-E", "-C"]


def config_files(source: Path) -> list[Path]:
    return [parent / ".clang-tidy" for parent in source.parents if (parent / ".clang-tidy").is_file()]


def cache_key(real_binary: str, args: list[str], source: Path, entry: dict) -> str | None:
    arguments = compile_arguments(entry)

    preprocessed = subprocess.run(preprocess_arguments(arguments), cwd=entry["directory"], stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, check=False)
    if preprocessed.returncode != 0:
        return None

    version = subprocess.run([real_binary, "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=False)

    digest = hashlib.sha256()

    def add(data: bytes) -> None:
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)

    add(Path(__file__).read_bytes())
    add(version.stdout)
    add("\0".join(args[:-1]).encode())
    add("\0".join(arguments).encode())

    for config in config_files(source):
        add(str(config).encode())
        add(config.read_bytes())

    add(preprocessed.stdout)

    return digest.hexdigest()


def main() -> int:
    real_binary = os.environ["CLANG_TIDY_REAL"]
    cache_dir = os.environ.get("CLANG_TIDY_CACHE_DIR", "")
    args = sys.argv[1:]

    build_dir = next((arg.removeprefix("-p=") for arg in args if arg.startswith("-p=")), None)

    if not cache_dir or not args or args[-1].startswith("-") or "-list-checks" in args or build_dir is None:
        return run_real(real_binary, args)

    source = Path(args[-1]).resolve()
    entry = find_compile_entry(Path(build_dir), source)

    if entry is None:
        return run_real(real_binary, args)

    key = cache_key(real_binary, args, source, entry)

    if key is None:
        return run_real(real_binary, args)

    marker = Path(cache_dir, key[:2], key)

    if marker.is_file():
        # Refreshed so run_clang_tidy.sh's age-based pruning keeps entries that are still hit.
        marker.touch()
        return 0

    result = run_real(real_binary, args)

    if result == 0:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    return result


if __name__ == "__main__":
    sys.exit(main())
