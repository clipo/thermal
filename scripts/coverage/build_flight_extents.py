#!/usr/bin/env python3
"""Build per-flight ground-footprint KMLs from the combined MAX/IRX directories.

For each `data/<slug>_combined/` directory (created by the SGD batch driver),
projects every frame's thermal FOV onto the ground via GPS+altitude+heading,
unions the per-frame rectangles to a single coverage polygon, and writes:

    sgd_output/<slug>_extents/<slug>_extent_merged.kml   <- one polygon per flight
    sgd_output/<slug>_extents/<slug>_extent_frames.kml   <- per-frame outlines

Reuses the existing ThermalFrameMapper class from
sgd_toolkit/georeferencing/footprint_generator.py — no new geometry code.

Usage:
    python scripts/build_flight_extents.py --combined-dirs data/*_combined
    python scripts/build_flight_extents.py --combined-dirs data/flight4_*_combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper


def slug_from_combined_dir(combined: Path) -> str:
    name = combined.name
    if name.endswith("_combined"):
        name = name[: -len("_combined")]
    return name


def build_extents_for(combined_dir: Path, output_root: Path, frame_skip: int = 1) -> dict:
    if not combined_dir.is_dir():
        return {"slug": combined_dir.name, "ok": False, "reason": "missing_dir"}

    slug = slug_from_combined_dir(combined_dir)
    out_dir = output_root / f"{slug}_extents"
    out_dir.mkdir(parents=True, exist_ok=True)

    mapper = ThermalFrameMapper(
        data_dir=str(combined_dir),
        output_base=f"{slug}_extent",  # basename only — ThermalFrameMapper prepends its own dir
        frame_skip=frame_skip,
        verbose=False,
    )
    mapper.output_dir = out_dir
    mapper.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        ok = mapper.run()
    except Exception as e:
        return {"slug": slug, "ok": False, "reason": f"crash: {e}"}

    n_frames = len(getattr(mapper, "frame_footprints", []))
    return {
        "slug": slug,
        "ok": ok,
        "n_frames": n_frames,
        "merged_kml": str(out_dir / f"{slug}_extent_merged.kml"),
        "frames_kml": str(out_dir / f"{slug}_extent_frames.kml"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--combined-dirs",
        nargs="+",
        required=True,
        help="One or more combined directories (e.g. data/*_combined)",
    )
    ap.add_argument(
        "--output-root",
        default="sgd_output",
        help="Root directory under which per-flight extents folders will be written",
    )
    ap.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1=all). 5 makes extents fast on huge surveys.",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root)
    results = []
    for d in args.combined_dirs:
        combined = Path(d)
        print(f"\n=== {combined.name} ===")
        r = build_extents_for(combined, output_root, frame_skip=args.frame_skip)
        if r.get("ok"):
            print(f"  ✓ {r['n_frames']} frames → {r['merged_kml']}")
        else:
            print(f"  ✗ failed: {r.get('reason', 'unknown')}")
        results.append(r)

    print("\n--- summary ---")
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        n = r.get("n_frames", 0)
        print(f"  {status:>4}  {r['slug']:<40}  {n:>5} frames")
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\n{n_ok}/{len(results)} flights succeeded")


if __name__ == "__main__":
    main()
