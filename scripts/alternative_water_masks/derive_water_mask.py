#!/usr/bin/env python3
"""For each per-flight anomaly raster, derive a land/water mask from the
Esri World Imagery satellite tiles at the same lat/lon footprint.

Cells whose corresponding satellite pixel fails the "looks like water" test
are flagged as land. The water-mask npz can then be applied:
  - by build_site_closeup.py (visualization)
  - by recompute_polygon_intensity.py (Σ_anomaly within polygons)
  - by derive_polygons_from_raster.py (raster-thresholded polygon set)

This corrects two known bugs:
  (a) RGB ocean segmenter occasionally classifying dark cliff shadows
      as ocean (mostly fine, but not perfect at vertical cliffs).
  (b) The ground-projection model in footprint_generator.py assumes
      altitude=0 ground. At tall cliffs (Poike, ~300 m), pixels showing
      the cliff face get projected to offshore lat/lon, producing
      phantom cold-water signal "offshore" the cliff base.

Output: sgd_output/<slug>_spread/<slug>_water_mask.npz with:
    is_water: (gy, gx) bool — True where satellite RGB looks like ocean

Water classification heuristic on Esri RGB tiles:
    blue_score = blue / (red + green + blue + 1)
    is_water = (blue_score > 0.36) AND (V < 0.78) AND (S > 0.10)
              AND (blue >= max(red, green))
The thresholds are calibrated for tropical-water tones in WorldImagery
basemap rendering; tune via --blue-score etc.

Usage:
    python scripts/derive_water_mask.py --slug june2023_2_july_23_poike_3
    python scripts/derive_water_mask.py --all
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def lonlat_to_webmerc(lon: float, lat: float) -> tuple[float, float]:
    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def fetch_basemap_array(min_lon: float, min_lat: float,
                         max_lon: float, max_lat: float,
                         zoom: int = 17) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Fetch Esri WorldImagery tiles for the bbox; return (RGB array, web-merc extent)."""
    import contextily as ctx
    src = ctx.providers.Esri.WorldImagery
    img, ext = ctx.bounds2img(min_lon, min_lat, max_lon, max_lat,
                               zoom=zoom, source=src, ll=True)
    # img is (H, W, 4) RGBA in web mercator with extent ext = (xmin, xmax, ymin, ymax)
    return img, ext


def classify_water(rgb: np.ndarray,
                   blue_score_min: float = 0.36,
                   v_max: float = 0.78,
                   s_min: float = 0.10) -> np.ndarray:
    """Return boolean mask same shape as rgb.shape[:2]: True = water.

    Lenient defaults that preserve continuous Vaihu coastal signal at
    the cost of some borderline cells:
      - blue_score >= 0.36
      - V < 0.78
      - S > 0.10
      - blue >= max(red, green) - 5/255 tolerance
    """
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    rgbf = rgb.astype(np.float32) / 255.0
    r = rgbf[..., 0]; g = rgbf[..., 1]; b = rgbf[..., 2]
    blue_score = b / (r + g + b + 1e-6)
    v = rgbf.max(axis=-1)
    s = (v - rgbf.min(axis=-1)) / (v + 1e-6)
    is_water = (
        (blue_score > blue_score_min)
        & (v < v_max)
        & (s > s_min)
        & (b >= np.maximum(r, g) - 5.0 / 255.0)
    )
    return is_water


def build_for_slug(slug: str, *, zoom: int = 17, dilate_water: int = 2,
                   force: bool = False) -> dict:
    spread_dir = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread_dir / f"{slug}_anomaly.npz"
    out_path = spread_dir / f"{slug}_water_mask.npz"
    if out_path.exists() and not force:
        return {"slug": slug, "skipped": True, "out": str(out_path)}
    if not npz_path.exists():
        return {"slug": slug, "error": "no anomaly raster", "out": None}

    raster = np.load(npz_path)
    minlon = float(raster["bbox_min_lon"]); maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"]); maxlat = float(raster["bbox_max_lat"])
    grid_res = float(raster["grid_resolution_m"])
    gy, gx = raster["anomaly"].shape

    # Fetch satellite tiles
    img, ext = fetch_basemap_array(minlon, minlat, maxlon, maxlat, zoom=zoom)
    # ext is (xmin, xmax, ymin, ymax) in web mercator
    xmin_m, xmax_m, ymin_m, ymax_m = ext
    img_water = classify_water(img)

    # For each raster cell, project its center lat/lon to web mercator and
    # sample the corresponding pixel in the satellite image.
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))

    # Cell-center lat/lon arrays (gy x gx)
    col_idx = np.arange(gx) + 0.5
    row_idx = np.arange(gy) + 0.5
    lons = minlon + col_idx * grid_res / mpd_lon
    lats = minlat + row_idx * grid_res / mpd_lat

    # web mercator coords for each cell center
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    R = 6378137.0
    x_grid = R * np.radians(lon_grid)
    y_grid = R * np.log(np.tan(np.pi / 4 + np.radians(lat_grid) / 2))

    H, W = img_water.shape
    # ext convention: extent = (xmin, xmax, ymin, ymax). Image origin is top-left:
    # row 0 corresponds to ymax, last row to ymin. col 0 to xmin, last col to xmax.
    px_x = ((x_grid - xmin_m) / (xmax_m - xmin_m)) * W
    px_y = ((ymax_m - y_grid) / (ymax_m - ymin_m)) * H
    px_c = np.clip(px_x.astype(np.int64), 0, W - 1)
    px_r = np.clip(px_y.astype(np.int64), 0, H - 1)
    is_water = img_water[px_r, px_c]

    # Optionally dilate water by a few cells to be lenient on coastline
    if dilate_water > 0:
        try:
            from scipy import ndimage
            is_water = ndimage.binary_dilation(is_water, iterations=dilate_water)
        except ImportError:
            pass

    np.savez_compressed(
        out_path,
        is_water=is_water,
        bbox_min_lon=minlon, bbox_max_lon=maxlon,
        bbox_min_lat=minlat, bbox_max_lat=maxlat,
        grid_resolution_m=grid_res,
        zoom=zoom,
        dilate_water=dilate_water,
    )
    n_water = int(is_water.sum())
    n_total = is_water.size
    return {
        "slug": slug, "out": str(out_path),
        "water_cells": n_water, "total_cells": n_total,
        "water_frac": n_water / n_total,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--zoom", type=int, default=17)
    ap.add_argument("--dilate-water", type=int, default=2,
                    help="cells to dilate water mask by (lenient on coast; default 2)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild masks even if already present")
    args = ap.parse_args()

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = spread_dir.name[: -len("_spread")]
            if (spread_dir / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"Deriving water masks for {len(slugs)} flight(s) at zoom={args.zoom}…")
    for slug in slugs:
        try:
            res = build_for_slug(slug, zoom=args.zoom,
                                  dilate_water=args.dilate_water,
                                  force=args.force)
            if res.get("skipped"):
                print(f"  → {slug}: already present (use --force to rebuild)")
            elif "error" in res:
                print(f"  ✗ {slug}: {res['error']}")
            else:
                pct = 100 * res["water_frac"]
                print(f"  ✓ {slug}: {res['water_cells']:,} water cells "
                      f"({pct:.1f}% of grid)")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
