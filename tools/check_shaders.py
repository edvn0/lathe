#!/usr/bin/env python3
"""Runs every shader entry point through the pipeline the engine runs at startup, and fails if any step breaks.

The runtime (SlangCompiler::compile) compiles each entry point with Slang at maximal optimization (-O3) and then runs
the SPIR-V optimizer's performance passes over the result. CI otherwise only compiles a handful of shaders with slangc
for push-constant reflection, so a shader that crashed the optimizer was first seen as a SIGSEGV at startup. This
script does the same two steps, plus a validation of the optimized module, for every entry point it finds:

    slangc  ->  spirv-opt -O  ->  spirv-val

Entry points are discovered from the [shader("<stage>")] attributes in the shader directory, so a new shader is
covered without listing it anywhere. Each entry point is built twice, with Slang debug info off and standard, since
the runtime's debug-info level is a request option (ShaderCompileRequest::generate_debug_info) and changes the SPIR-V
the optimizer sees.

Usage:
    tools/check_shaders.py [--slangc slangc] [--spirv-opt spirv-opt] [--spirv-val spirv-val]
                           [--shader-dir assets/shaders] [--jobs N] [--list]

Exit status: 0 when every entry point passes, 1 when any fails (a crash is reported by signal name), 2 on bad usage.
The tools' flags mirror src/assets/slang_compiler.cxx; change them together.
"""

import argparse
import concurrent.futures
import dataclasses
import os
import pathlib
import re
import signal
import subprocess
import sys
import tempfile

# [shader("compute")], then any further attributes ([numthreads(8, 8, 1)], [outputtopology("triangle")], ...), the
# return type, and the function name.
ENTRY_POINT = re.compile(
    r'\[shader\("(?P<stage>\w+)"\)\]\s*(?:\[[^\]]*\]\s*)*(?:[A-Za-z_][\w:<>,]*\s+)+?(?P<name>\w+)\s*\('
)

# ShaderCompileRequest::generate_debug_info false / true (SLANG_DEBUG_INFO_LEVEL_NONE / STANDARD).
DEBUG_LEVELS = ("0", "2")

# Slang command lines are short; the optimizer on the biggest module takes seconds. A hung tool is a failure.
TOOL_TIMEOUT_SECONDS = 300


@dataclasses.dataclass(frozen=True)
class EntryPoint:
    file: str
    name: str
    stage: str


@dataclasses.dataclass(frozen=True)
class Job:
    entry: EntryPoint
    debug_level: str

    def label(self) -> str:
        return f"{self.entry.file} {self.entry.name} ({self.entry.stage}, -g{self.debug_level})"


@dataclasses.dataclass
class Result:
    job: Job
    failed_step: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed_step


def discover(shader_dir: pathlib.Path) -> list[EntryPoint]:
    entries = []
    for path in sorted(shader_dir.glob("*.slang")):
        text = path.read_text(encoding="utf-8")
        for match in ENTRY_POINT.finditer(text):
            entries.append(EntryPoint(path.name, match.group("name"), match.group("stage")))
    return entries


def describe_exit(code: int) -> str:
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit status {code}"


def run_tool(command: list[str]) -> tuple[bool, str]:
    """Runs one tool, returning (succeeded, its combined output or a description of how it failed)."""
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TOOL_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {TOOL_TIMEOUT_SECONDS} s"
    except OSError as error:
        return False, f"could not run {command[0]}: {error}"

    output = completed.stdout.strip()
    if completed.returncode != 0:
        return False, f"{describe_exit(completed.returncode)}\n{output}".strip()
    return True, output


def check(job: Job, args: argparse.Namespace, work_dir: pathlib.Path) -> Result:
    stem = f"{pathlib.Path(job.entry.file).stem}.{job.entry.name}.g{job.debug_level}"
    compiled = work_dir / f"{stem}.spv"
    optimized = work_dir / f"{stem}.opt.spv"

    # Mirrors SlangCompiler::compile: SPIR-V 1.6, direct emit with entry-point names, column-major matrices, scalar
    # buffer layout, maximal optimization, and warning 41012 off.
    slang = [
        args.slangc,
        str(args.shader_dir / job.entry.file),
        "-entry", job.entry.name,
        "-stage", job.entry.stage,
        "-target", "spirv",
        "-profile", "spirv_1_6",
        "-emit-spirv-directly",
        "-fvk-use-entrypoint-name",
        "-matrix-layout-column-major",
        "-force-glsl-scalar-layout",
        "-O3",
        f"-g{job.debug_level}",
        "-warnings-disable", "41012",
        "-I", str(args.shader_dir),
        "-o", str(compiled),
    ]
    succeeded, detail = run_tool(slang)
    if not succeeded:
        return Result(job, "slangc", detail)
    if not compiled.exists() or compiled.stat().st_size == 0:
        return Result(job, "slangc", "wrote no SPIR-V")

    # RegisterPerformancePasses() is what `-O` registers; the engine's optimizer validator is off, so is this one.
    succeeded, detail = run_tool(
        [args.spirv_opt, "--target-env=vulkan1.3", "-O", "--skip-validation", str(compiled), "-o", str(optimized)]
    )
    if not succeeded:
        return Result(job, "spirv-opt", detail)

    # The pipeline needs the optimized module to be valid; scalar layout matches -force-glsl-scalar-layout.
    succeeded, detail = run_tool(
        [args.spirv_val, "--target-env", "vulkan1.3", "--scalar-block-layout", str(optimized)]
    )
    if not succeeded:
        return Result(job, "spirv-val", detail)

    return Result(job)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slangc", default="slangc", help="path of slangc (default: from PATH)")
    parser.add_argument("--spirv-opt", default="spirv-opt", help="path of spirv-opt (default: from PATH)")
    parser.add_argument("--spirv-val", default="spirv-val", help="path of spirv-val (default: from PATH)")
    parser.add_argument(
        "--shader-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent / "assets" / "shaders",
        help="directory of .slang files (default: assets/shaders)",
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel jobs (default: CPU count)")
    parser.add_argument("--list", action="store_true", help="print the discovered entry points and exit")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)

    if not args.shader_dir.is_dir():
        print(f"error: {args.shader_dir} is not a directory", file=sys.stderr)
        return 2

    entries = discover(args.shader_dir)

    if args.list:
        for entry in entries:
            print(f"{entry.file} {entry.name} {entry.stage}")
        return 0

    # A broken regex or a wrong directory would otherwise "pass" by checking nothing.
    if not entries:
        print(f"error: no [shader(\"...\")] entry points found in {args.shader_dir}", file=sys.stderr)
        return 1

    jobs = [Job(entry, level) for entry in entries for level in DEBUG_LEVELS]
    print(f"checking {len(entries)} entry points x {len(DEBUG_LEVELS)} debug levels: slangc -O3 -> spirv-opt -O -> spirv-val")
    sys.stdout.flush()

    results: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="check_shaders_") as work:
        work_dir = pathlib.Path(work)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            for result in pool.map(lambda job: check(job, args, work_dir), jobs):
                results.append(result)
                print(f"  {'ok  ' if result.ok else 'FAIL'} {result.job.label()}" + ("" if result.ok else f"  [{result.failed_step}]"))
                sys.stdout.flush()

    failures = [result for result in results if not result.ok]
    if not failures:
        print(f"all {len(results)} builds passed")
        return 0

    print(f"\n{len(failures)} of {len(results)} builds failed:", file=sys.stderr)
    for result in failures:
        print(f"\n--- {result.job.label()}: {result.failed_step} ---", file=sys.stderr)
        lines = result.detail.splitlines()
        print("\n".join(lines[:40]), file=sys.stderr)
        if len(lines) > 40:
            print(f"... ({len(lines) - 40} more lines)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
