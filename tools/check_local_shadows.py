#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Checks that the local light shadow optimisations do not change the picture.

Renders the `shadows` test scene (opaque, double-sided, alpha-masked and skinned casters under two spot lights and a
point light) headless and takes a screenshot, once per combination of GPU culling and caching. The scene is driven by
frame count, so every run draws the same frame. The run with both optimisations off is the reference; every other run
must match it pixel for pixel.

Each combination is rendered in three phases, so both sides of the cache are covered:
  * the orbiting cube moving (its lights redraw every frame),
  * the cube stopped and the skinned characters hidden (every light is served from the cache),
  * the cube stopped with the skinned characters shown (skinned casters never cache, so their lights redraw).

Needs a GPU and a built engine. Usage:
    tools/check_local_shadows.py [--binary build/debug/bin/lathe] [--tolerance 0] [--frames 90]
Exit status 0 when every run matches the reference, 1 otherwise.
"""

import argparse
import itertools
import pathlib
import subprocess
import sys

import numpy as np
from PIL import Image

KEY_F12 = 301
KEY_C = 67  # toggles GPU culling
KEY_V = 86  # toggles caching
KEY_O = 79  # stops or starts the orbiter
KEY_K = 75  # shows or hides the skinned characters

SETUP_FRAME = 5
STOP_ORBITER_FRAME = 20
SCREENSHOT_FRAME = 70


Stats = dict[str, int]


def last_stats(output: str) -> Stats:
    """The most recent `stats lights=.. cached=.. tiles=.. culled=..` line the scene logged."""
    found: Stats = {}
    for line in output.splitlines():
        if "[ShadowCaster] stats" in line:
            found = {key: int(value) for key, value in (part.split("=") for part in line.split("stats ")[1].split())}
    return found


def run_once(binary: pathlib.Path, root: pathlib.Path, culling: bool, caching: bool, orbiting: bool, humans: bool,
             frames: int) -> tuple[np.ndarray, Stats]:
    keys = []
    if not culling:
        keys.append((SETUP_FRAME, KEY_C))
    if not caching:
        keys.append((SETUP_FRAME + 1, KEY_V))
    if not orbiting:
        keys.append((STOP_ORBITER_FRAME, KEY_O))
    if not humans:
        keys.append((STOP_ORBITER_FRAME + 1, KEY_K))
    keys.append((SCREENSHOT_FRAME, KEY_F12))

    screenshots = root / "screenshots"
    screenshots.mkdir(exist_ok=True)
    before = {p.name for p in screenshots.glob("*.png")}

    command = [
        str(binary),
        "--game=shadows",
        "--player",
        "--screen-type=headless",
        f"--inject-keys={','.join(f'{frame}:{key}' for frame, key in keys)}",
        f"--exit-after-frames={frames}",
    ]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-2000:] + result.stderr[-2000:])
        raise SystemExit(f"engine exited with {result.returncode}: {' '.join(command)}")

    problems = [line for line in (result.stdout + result.stderr).splitlines() if "VUID" in line or "[E]" in line]
    if problems:
        raise SystemExit("engine reported errors:\n" + "\n".join(problems[:10]))

    made = sorted(p for p in screenshots.glob("*.png") if p.name not in before)
    if len(made) != 1:
        raise SystemExit(f"expected one screenshot, found {len(made)}")

    image = np.asarray(Image.open(made[0]).convert("RGB")).astype(np.int16)
    made[0].unlink()
    return image, last_stats(result.stdout + result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", default="build/debug/bin/lathe")
    parser.add_argument("--tolerance", type=int, default=0, help="largest allowed per-channel difference")
    parser.add_argument("--frames", type=int, default=90)
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    binary = (root / args.binary).resolve()
    if not binary.exists():
        raise SystemExit(f"{binary} does not exist; build the engine first")

    failures = 0
    references: dict[str, np.ndarray] = {}

    # name, orbiter moving, skinned characters shown, whether every light should come from the cache
    phases = [
        ("orbiter moving", True, True, False),
        ("orbiter stopped, no skinned (cached)", False, False, True),
        ("orbiter stopped, skinned shown", False, True, False),
    ]

    for phase, orbiting, humans, all_cached in phases:
        reference, reference_stats = run_once(binary, root, culling=False, caching=False, orbiting=orbiting,
                                              humans=humans, frames=args.frames)
        references[phase] = reference

        if reference_stats.get("lights", 0) < 3 or reference_stats.get("cached") != 0 or reference_stats.get("culled"):
            print(f"[{phase}] reference run is not the plain path: {reference_stats}")
            return 1

        # An empty or flat picture would make every comparison pass for the wrong reason.
        if reference.std() < 5.0:
            print(f"[{phase}] reference image is nearly flat (std {reference.std():.1f}); the scene did not render")
            return 1

        for culling, caching in itertools.product((False, True), repeat=2):
            if not culling and not caching:
                continue

            image, stats = run_once(binary, root, culling=culling, caching=caching, orbiting=orbiting,
                                    humans=humans, frames=args.frames)
            difference = np.abs(image - reference).max(axis=2)
            worst = int(difference.max())
            wrong = int((difference > args.tolerance).sum())

            # The optimisation under test must actually have run, or matching the reference means nothing.
            # With every light cached there is nothing to cull, so no cull pass runs.
            engaged = bool(stats.get("culled")) == (culling and stats.get("tiles", 0) > 0)
            if caching and all_cached:
                engaged = engaged and stats.get("cached") == stats.get("lights") and stats.get("tiles") == 0
            if caching and not all_cached:
                engaged = engaged and stats.get("tiles", 0) > 0
            if not caching:
                engaged = engaged and stats.get("cached") == 0
            if not engaged:
                print(f"[{phase}] culling={'on' if culling else 'off'} caching={'on' if caching else 'off'}: "
                      f"optimisation did not engage as expected: {stats}")
                failures += 1
                continue

            ok = worst <= args.tolerance
            failures += 0 if ok else 1
            print(f"[{phase}] culling={'on' if culling else 'off'} caching={'on' if caching else 'off'}: "
                  f"{'ok' if ok else 'DIFFERS'} (max channel diff {worst}, {wrong} pixels over {args.tolerance})")

    # The comparison must be able to fail: the moving and the stopped orbiter really are different pictures.
    names = [phase for phase, *_ in phases]
    for first, second in itertools.combinations(names, 2):
        sensitivity = int(np.abs(references[first] - references[second]).max())
        if sensitivity <= args.tolerance:
            print(f"the '{first}' and '{second}' references are identical: the check cannot detect changes")
            return 1
    print("sensitivity check: the three phases render different pictures")

    print("all matching" if failures == 0 else f"{failures} run(s) differ from the reference or did not engage")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
