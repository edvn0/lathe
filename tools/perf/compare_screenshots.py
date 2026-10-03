#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow", "numpy"]
# ///
"""Bit-identical comparison of two directories of benchmark keyframe screenshots.

    tools/perf/compare_screenshots.py --base perf/baseline/occ_off --head /tmp/head

Screenshots are matched in sorted file-name order (the app names them by timestamp, so that is keyframe order). Exits
0 only if both directories hold the same number of images and every pair has identical pixels; otherwise it reports the
first differing pixel of each mismatching pair and exits 1.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    args = parser.parse_args()

    base = sorted(args.base.glob("*.png"))
    head = sorted(args.head.glob("*.png"))
    if not base or len(base) != len(head):
        print(f"error: {len(base)} base vs {len(head)} head images", file=sys.stderr)
        return 1

    failures = 0
    for index, (a, b) in enumerate(zip(base, head)):
        left, right = load(a), load(b)
        if left.shape != right.shape:
            print(f"keyframe {index}: size {left.shape[:2]} vs {right.shape[:2]}")
            failures += 1
            continue
        differing = np.any(left != right, axis=2)
        if not differing.any():
            continue
        failures += 1
        y, x = np.argwhere(differing)[0]
        print(f"keyframe {index}: {differing.mean() * 100:.3f}% of pixels differ; first at x={x} y={y}: "
              f"{left[y, x].tolist()} vs {right[y, x].tolist()}  ({a.name} / {b.name})")

    print(f"{len(base) - failures}/{len(base)} keyframes identical")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
