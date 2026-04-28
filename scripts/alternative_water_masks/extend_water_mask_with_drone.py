#!/usr/bin/env python3
"""Extend each per-flight water mask by ALSO including cells that the
drone's per-frame ocean segmenter consistently observed as ocean
(obs_count >= --min-obs).

Rationale: the satellite HSV classifier rejects shallow tropical bay
water (e.g., Hanga Nui at Ahu Tongariki) because the sandy bottom
makes the colour green-ish rather than deep blue. The drone's RGB
segmenter — operating on close-range imagery — correctly identifies
these cells as ocean. We trust the drone classification for cells
with obs_count >= 5 (i.e., persistently classified as ocean across
many overlapping frames).

The 3 °C anomaly cap (applied downstream by all consumers) still
filters out the most extreme cliff-shadow projection-bug values, so
this extension is safe even at cliff-coast flights — the worst-case
cliff projections are caught by the anomaly cap, not the water mask.

Updates each <slug>_water_mask.npz in place, adding 'is_water_extended'
alongside 'is_water', and bumping a `method` field. Downstream scripts
default to 'is_water_extended' if present.

Usage:
    python scripts/extend_water_mask_with_drone.py --all
    python scripts/extend_water_mask_with_drone.py --slug june2023_23_june_23_tongariki_flights
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def extend_for_slug(slug: str, *, min_obs: int = 5) -> dict:
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    wm_path = spread / f"{slug}_water_mask.npz"
    if not npz_path.exists() or not wm_path.exists():
        return {"slug": slug, "error": "missing npz files"}

    raster = np.load(npz_path)
    wm = np.load(wm_path)
    sat_water = wm["is_water"]
    obs = raster["observations"]
    if sat_water.shape != obs.shape:
        return {"slug": slug, "error": f"shape mismatch sat={sat_water.shape} obs={obs.shape}"}

    drone_water = (obs >= min_obs)
    extended = sat_water | drone_water

    n_added = int(((~sat_water) & drone_water).sum())
    n_total_before = int(sat_water.sum())
    n_total_after = int(extended.sum())

    # Save with all original keys preserved + new keys
    out = {k: wm[k] for k in wm.files}
    out["is_water"] = extended  # consumers read this; keep it as the unified field
    out["is_water_satellite_only"] = sat_water
    out["is_water_drone_only"] = drone_water
    out["min_obs_for_drone_water"] = min_obs
    out["method"] = "satellite_or_drone"
    np.savez_compressed(wm_path, **out)

    return {
        "slug": slug,
        "n_total_before": n_total_before,
        "n_total_after": n_total_after,
        "n_added_by_drone": n_added,
        "frac_before": n_total_before / extended.size,
        "frac_after": n_total_after / extended.size,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--min-obs", type=int, default=5)
    args = ap.parse_args()

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for sd in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = sd.name[: -len("_spread")]
            if (sd / f"{slug}_anomaly.npz").exists() and (sd / f"{slug}_water_mask.npz").exists():
                slugs.append(slug)

    print(f"Extending {len(slugs)} water mask(s) with drone obs (min_obs={args.min_obs})…")
    for slug in slugs:
        r = extend_for_slug(slug, min_obs=args.min_obs)
        if "error" in r:
            print(f"  ✗ {slug}: {r['error']}")
        else:
            print(f"  ✓ {slug}: {100*r['frac_before']:5.1f}% → {100*r['frac_after']:5.1f}% water "
                  f"(+{r['n_added_by_drone']:,} cells from drone)")


if __name__ == "__main__":
    main()
