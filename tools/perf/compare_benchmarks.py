#!/usr/bin/env python3
"""Compare two --benchmark result files (see include/app/benchmark.hxx).

    compare_benchmarks.py [--base base.json] --head head.json
                          [--threshold 10] [--fail-threshold 20]
                          [--markdown out.md]

Prints a Markdown table of per-stage median/p95 GPU times, base vs head.
Without --base (or when the file doesn't exist -- e.g. the base branch
predates benchmark mode) it reports head alone.

A stage whose median moves by more than --threshold percent is flagged.
Exits 1 when the full-frame median regresses by more than --fail-threshold
percent, so CI can gate on it; per-stage numbers are informational, since
small stages are noisy on a software rasterizer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FULL_FRAME = "full_frame"

# Stages under this many milliseconds are reported but never flagged: a
# 0.1 ms pass moving 30% is noise, not a regression.
MIN_FLAGGED_MS = 0.5


def load(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def stages_by_id(result: dict) -> dict[str, dict]:
    return {stage["id"]: stage for stage in result["stages"]}


def percent_change(base: float, head: float) -> float | None:
    if base <= 0.0:
        return None
    return (head - base) / base * 100.0


def describe_run(label: str, result: dict) -> str:
    width, height = result["render_extent"]
    settled = "" if result.get("streaming_settled", True) else ", **streaming never settled**"
    return (
        f"- **{label}**: {result['device']}, {width}x{height}, seed {result['seed']}, "
        f"{result['frames']} frames over {result['keyframes']} keyframes "
        f"(after {result['warmup_frames']} warmup){settled}"
    )


def mismatches(base: dict, head: dict) -> list[str]:
    notes = []
    for key in ("device", "render_extent", "seed", "keyframes", "frames"):
        if base.get(key) != head.get(key):
            notes.append(f"`{key}` differs ({base.get(key)} vs {head.get(key)}) -- numbers may not be comparable.")
    return notes


def head_only_report(head: dict) -> str:
    lines = [
        "### Benchmark (head only)",
        "",
        "No base result to compare against (the base branch has no benchmark mode, or its run failed).",
        "",
        describe_run("head", head),
        "",
        "| Stage | Median (ms) | p95 (ms) | Mean (ms) |",
        "|---|---:|---:|---:|",
    ]
    for stage in head["stages"]:
        lines.append(
            f"| {stage['name']} | {stage['median_ms']:.2f} | {stage['p95_ms']:.2f} | {stage['mean_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def comparison_report(base: dict, head: dict, threshold: float) -> tuple[str, float | None]:
    base_stages = stages_by_id(base)
    head_stages = stages_by_id(head)

    lines = [
        "### Benchmark: base vs head",
        "",
        describe_run("base", base),
        describe_run("head", head),
    ]

    for note in mismatches(base, head):
        lines.append(f"- :warning: {note}")

    lines += [
        "",
        "| Stage | Base median (ms) | Head median (ms) | Change | Base p95 (ms) | Head p95 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    full_frame_change = None

    for stage_id, head_stage in head_stages.items():
        base_stage = base_stages.get(stage_id)
        if base_stage is None:
            lines.append(f"| {head_stage['name']} | -- | {head_stage['median_ms']:.2f} | new | -- | "
                         f"{head_stage['p95_ms']:.2f} |")
            continue

        change = percent_change(base_stage["median_ms"], head_stage["median_ms"])
        if stage_id == FULL_FRAME:
            full_frame_change = change

        if change is None:
            change_text = "n/a"
        else:
            marker = ""
            significant = max(base_stage["median_ms"], head_stage["median_ms"]) >= MIN_FLAGGED_MS
            if significant and change > threshold:
                marker = " :red_circle:"
            elif significant and change < -threshold:
                marker = " :green_circle:"
            change_text = f"{change:+.1f}%{marker}"

        name = f"**{head_stage['name']}**" if stage_id == FULL_FRAME else head_stage["name"]
        lines.append(
            f"| {name} | {base_stage['median_ms']:.2f} | {head_stage['median_ms']:.2f} | {change_text} | "
            f"{base_stage['p95_ms']:.2f} | {head_stage['p95_ms']:.2f} |"
        )

    footer = f"Flagged: median moved more than {threshold:.0f}% (stages under {MIN_FLAGGED_MS} ms never are)."
    if "llvmpipe" in head["device"]:
        footer += (" Timings come from a software rasterizer (lavapipe): good at catching large regressions, "
                   "not a stand-in for a real GPU.")

    lines += ["", footer]
    return "\n".join(lines) + "\n", full_frame_change


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, help="base build's benchmark JSON (optional)")
    parser.add_argument("--head", type=Path, required=True, help="head build's benchmark JSON")
    parser.add_argument("--threshold", type=float, default=10.0, help="percent change to flag a stage")
    parser.add_argument("--fail-threshold", type=float, default=20.0,
                        help="full-frame median regression (percent) that fails the run")
    parser.add_argument("--markdown", type=Path, help="also write the report here")
    args = parser.parse_args()

    head = load(args.head)
    if head is None:
        print(f"error: no head result at {args.head}", file=sys.stderr)
        return 2

    base = load(args.base)
    exit_code = 0

    if base is None:
        report = head_only_report(head)
    else:
        report, full_frame_change = comparison_report(base, head, args.threshold)
        if full_frame_change is not None and full_frame_change > args.fail_threshold:
            report += (
                f"\n:x: Full-frame median regressed {full_frame_change:+.1f}% "
                f"(fails above {args.fail_threshold:.0f}%).\n"
            )
            exit_code = 1

    print(report)
    if args.markdown is not None:
        args.markdown.write_text(report, encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
