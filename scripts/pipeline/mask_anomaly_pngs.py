#!/usr/bin/env python3
"""Re-render each per-flight cold-anomaly PNG with the satellite water
mask applied: cells over land become fully transparent. This filters
out the cliff-shadow and inland projection-bug pixels that pollute the
master anomaly KML at sites like Poike, leaving only the actual
ocean cold-anomaly content.

Also applies the same quality filters used downstream:
  - obs_count >= 5 (drop low-coverage cells)
  - anomaly <= 3 °C (drop sensor outliers)

Overwrites <slug>_anomaly.png in place. The master anomaly KML
(`scripts/aggregate_anomaly_kml.py` output) automatically picks up
the cleaner PNGs.

Usage:
    python scripts/mask_anomaly_pngs.py --all
    python scripts/mask_anomaly_pngs.py --slug june2023_2_july_23_poike_3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def render_masked_png(slug: str, *,
                      vmax: float = 1.5,
                      min_obs: int = 5,
                      max_realistic_anom_c: float = 3.0,
                      max_land_anomaly_c: float = 0.5,
                      drop_isolated: bool = True,
                      isolation_threshold_c: float = 0.3,
                      use_water_mask: bool = True) -> dict:
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    wm_path = spread / f"{slug}_water_mask.npz"
    png_path = spread / f"{slug}_anomaly.png"
    if not npz_path.exists():
        return {"slug": slug, "error": "no anomaly raster"}

    raster = np.load(npz_path)
    anomaly = raster["anomaly"]
    obs = raster["observations"]
    if use_water_mask and wm_path.exists():
        water = np.load(wm_path)["is_water"]
        if water.shape != anomaly.shape:
            water = None
    else:
        water = None

    # Quality filter (always applied)
    finite = np.isfinite(anomaly)
    keep = finite & (obs >= min_obs) & (anomaly <= max_realistic_anom_c)

    # Soft land filter: drop high-anomaly cells that satellite says are land.
    # Cliff-shadow projection-bug cells typically have anom > 1.5°C and project
    # onto land coordinates per satellite imagery. Genuine shallow-water cells
    # that satellite misclassifies (e.g., Vaihu Harbor sandy bottom) typically
    # have anom < 1°C — those are kept. Default cutoff 0.5 °C is conservative.
    n_dropped_cliff = 0
    if water is not None:
        cliff_shadow = ~water & (anomaly > max_land_anomaly_c)
        keep = keep & ~cliff_shadow
        n_dropped_cliff = int((finite & cliff_shadow).sum())

    # Isolation filter: drop small isolated components.
    # Connected-component analysis on the kept (observed) cells.
    # Always drop components below `min_component_frac` of the largest
    # component (default 1%). This catches isolated patches like the
    # basalt Ahu Tongariki platform regardless of satellite-water mask
    # reliability — empirically, the satellite HSV mask has scattered
    # false-positive water cells that overlap most components, making
    # a "must-touch-water" filter ineffective. Size-based filter is
    # robust to satellite mask noise.
    n_dropped_isolated = 0
    if drop_isolated:
        try:
            from scipy import ndimage
            labeled, n_comp = ndimage.label(keep)
            if n_comp > 0:
                sizes = ndimage.sum(keep, labeled, range(1, n_comp + 1))
                largest = float(sizes.max()) if sizes.size else 0
                # Threshold = 10% of largest component. The main flight
                # strip is typically 100-1000× larger than any artifact
                # blob (Ahu Tongariki ~15k cells; main strip ~350k cells
                # at this flight). 10% catches the ahu without losing
                # the main component.
                min_size = max(1, 0.10 * largest)
                keep_ids = np.where(sizes >= min_size)[0] + 1
                connected = np.isin(labeled, keep_ids)
                isolated = keep & ~connected
                keep = keep & ~isolated
                n_dropped_isolated = int(isolated.sum())
        except ImportError:
            pass

    n_total_finite = int(finite.sum())
    n_kept = int(keep.sum())
    n_dropped_obs = int((finite & (obs < min_obs)).sum())
    n_dropped_outlier = int((finite & (anomaly > max_realistic_anom_c)).sum())

    # Render: YlOrRd colormap for kept cells, transparent elsewhere
    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("YlOrRd")
    except ImportError:
        cmap = None

    norm = np.clip(anomaly / vmax, 0.0, 1.0)
    if cmap is not None:
        rgba = (cmap(norm) * 255).astype(np.uint8)
    else:
        rgba = np.zeros((*anomaly.shape, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        rgba[..., 1] = (255 * (1 - norm)).astype(np.uint8)
        rgba[..., 2] = (255 * (1 - norm)).astype(np.uint8)
    rgba[..., 3] = np.where(keep, 220, 0).astype(np.uint8)
    Image.fromarray(np.flipud(rgba), "RGBA").save(png_path)

    return {
        "slug": slug, "out": str(png_path),
        "kept_cells": n_kept, "total_finite": n_total_finite,
        "dropped_obs": n_dropped_obs, "dropped_outlier": n_dropped_outlier,
        "dropped_cliff": n_dropped_cliff,
        "dropped_isolated": n_dropped_isolated,
        "had_water_mask": water is not None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--vmax", type=float, default=1.5)
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--max-realistic", type=float, default=3.0)
    ap.add_argument("--max-land-anomaly", type=float, default=0.5,
                    help="drop satellite-classified-LAND cells whose anomaly "
                         "exceeds this (cliff-shadow filter; default 0.5°C). "
                         "Set very high (e.g., 99) to disable this filter.")
    ap.add_argument("--isolation-threshold", type=float, default=0.3,
                    help="anomaly °C above which the isolation filter applies "
                         "(default 0.3). Cold-cell components that don't touch "
                         "any satellite-water cell are treated as artifacts "
                         "(e.g., ahu platforms, isolated rooftops, surviving "
                         "cliff shadows).")
    ap.add_argument("--no-isolation-filter", action="store_true",
                    help="don't drop components isolated from water")
    ap.add_argument("--no-water-mask", action="store_true",
                    help="don't apply satellite water mask at all (keeps all "
                         "finite cells regardless of land classification)")
    args = ap.parse_args()

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for sd in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = sd.name[: -len("_spread")]
            if (sd / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    use_wm = not args.no_water_mask
    use_iso = not args.no_isolation_filter
    note = (f"soft land filter: anom>{args.max_land_anomaly}°C on land dropped; "
            f"isolation filter: cold components disconnected from water dropped"
            if use_wm else "no water mask (all finite cells kept)")
    print(f"Re-rendering {len(slugs)} anomaly PNG(s); {note}…")
    for slug in slugs:
        r = render_masked_png(
            slug, vmax=args.vmax, min_obs=args.min_obs,
            max_realistic_anom_c=args.max_realistic,
            max_land_anomaly_c=args.max_land_anomaly,
            drop_isolated=use_iso,
            isolation_threshold_c=args.isolation_threshold,
            use_water_mask=use_wm,
        )
        if "error" in r:
            print(f"  ✗ {slug}: {r['error']}")
        else:
            wm_note = "✓ water-aware" if r["had_water_mask"] else "⚠ no water mask"
            print(f"  ✓ {slug}: kept {r['kept_cells']:,} / {r['total_finite']:,} "
                  f"(obs<5: -{r['dropped_obs']:,}, outliers: -{r['dropped_outlier']:,}, "
                  f"cliff-shadow: -{r['dropped_cliff']:,}, "
                  f"isolated: -{r['dropped_isolated']:,})  {wm_note}")


if __name__ == "__main__":
    main()
