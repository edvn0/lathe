#!/usr/bin/env python3
"""Fails if source code reaches the filesystem around core/paths.hxx.

Files are located through Paths (DataPath, CachePath, AssetPath, ...), never through the working directory or a
cwd-relative literal. Run by `cargo xtask tidy`.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCAN = ["src", "game", "include"]
ALLOWED_FILES = {"src/core/paths.cxx", "include/core/paths.hxx"}

CURRENT_PATH = re.compile(r"\bcurrent_path\s*\(")
RELATIVE_LITERAL = re.compile(r'"(?:assets|cache|screenshots)(?:/|")')
SANCTIONED = re.compile(r"\b(?:data_path|cache_path|state_path|screenshot_path)\s*\(")


def main() -> int:
    failures = []

    for directory in SCAN:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.suffix not in {".cxx", ".hxx"}:
                continue

            relative = path.relative_to(ROOT).as_posix()

            if relative in ALLOWED_FILES:
                continue

            for number, line in enumerate(path.read_text().splitlines(), 1):
                code = line.split("//", 1)[0]

                if code.lstrip().startswith("#include"):
                    continue

                if CURRENT_PATH.search(code):
                    failures.append(f"{relative}:{number}: current_path() -- resolve through Paths instead")
                elif RELATIVE_LITERAL.search(code) and not SANCTIONED.search(code):
                    failures.append(f"{relative}:{number}: cwd-relative path literal -- use data_path()/cache_path()")

    for failure in failures:
        print(failure)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
