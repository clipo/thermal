#!/usr/bin/env python3
"""For each flight, derive a cliff-zone mask from SRTM 30m elevation data.

Cliff coasts on Rapa Nui (Poike, Rano Kau, Hekii cliffs) lack the
lava-tube conduit topology that produces SGD outlets. Plumes detected
near these coasts are very likely either projection-bug artifacts (the
flat-ground projection in `footprint_generator.py` mismaps cliff-face
thermal pixels to offshore lat/lon) or shadow misclassifications.

This script samples the SRTM DEM at each flight's grid, applies a max-
filter over a 100 m radius, and produces a boolean `is_cliff_zone`
raster: True where the maximum elevation within 100 m exceeds 50 m.
Coast-anchored plume detection skips seeds in cliff zones, so detected
plumes only come from low-elevation (bay/inlet) coastlines where SGD
is geologically plausible.

The 50 m / 100 m thresholds correspond to the user's Rapa Nui
geological intuition: SGD emerges from collapsed lava tubes at small
bays/inlets where the coastline is at low elevation; vertical cliffs
have no conduit topology and don't produce surface SGD even when the
underlying aquifer is the same.

Output: `<slug>_cliff_zone.npz` per flight.

Usage:
    python scripts/derive_cliff_zone.py --all
    python scripts/derive_cliff_zone.py --slug june2023_2_july_23_poike_3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"
DEM_FILE = THERMAL / "data" / "dem" / "S28W110.hgt"


def load_srtm_dem():
    """Load SRTM 30m tile S28W110 (covers Rapa Nui).
    Returns (elev, origin_lat, origin_lon, cell_size_deg).
      origin = top-left corner (lat=-27, lon=-110)
      cell_size = 1/3600 degree (1 arc-second)
    """
    if not DEM_FILE.exists():
        raise SystemExit(
            f"DEM not found at {DEM_FILE}\n"
            f"Download with: curl -sSL -o {DEM_FILE.parent}/S28W110.SRTMGL1.hgt.zip "
            f"https://step.esa.int/auxdata/dem/SRTMGL1/S28W110.SRTMGL1.hgt.zip "
            f"&& unzip -p {DEM_FILE.parent}/S28W110.SRTMGL1.hgt.zip "
            f"> {DEM_FILE}"
        )
    raw = DEM_FILE.read_bytes()
    elev = np.frombuffer(raw, dtype=">i2").reshape(3601, 3601).astype(np.float32)
    # SRTM convention: row 0 is top of tile = north edge
    # Tile S28W110 spans lat [-28..-27] and lon [-110..-109]
    # row 0 → lat -27 (north edge); row 3600 → lat -28
    # col 0 → lon -110; col 3600 → lon -109
    return {
        "elev": elev,
        "origin_lat": -27.0,   # row 0
        "origin_lon": -110.0,  # col 0
        "cell_deg": 1.0 / 3600.0,
    }


def sample_dem_to_grid(dem, target_lats, target_lons):
    """Bilinear sampling of SRTM DEM at target lat/lon arrays.
    target_lats, target_lons can be 2D meshgrids."""
    e = dem["elev"]
    cell = dem["cell_deg"]
    # Convert target lat/lon to fractional row/col in DEM
    # row increases southward; lat decreases southward
    rows_f = (dem["origin_lat"] - target_lats) / cell
    cols_f = (target_lons - dem["origin_lon"]) / cell
    H, W = e.shape

    r0 = np.clip(np.floor(rows_f).astype(np.int32), 0, H - 2)
    c0 = np.clip(np.floor(cols_f).astype(np.int32), 0, W - 2)
    dr = rows_f - r0
    dc = cols_f - c0

    z00 = e[r0, c0]
    z10 = e[r0 + 1, c0]
    z01 = e[r0, c0 + 1]
    z11 = e[r0 + 1, c0 + 1]

    z = (z00 * (1 - dr) * (1 - dc)
         + z10 * dr * (1 - dc)
         + z01 * (1 - dr) * dc
         + z11 * dr * dc)
    return z


def derive_for_slug(slug: str, *,
                     cliff_height_m: float = 50.0,
                     proximity_radius_m: float = 100.0,
                     dem=None) -> dict:
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    out_path = spread / f"{slug}_cliff_zone.npz"
    if not npz_path.exists():
        return {"slug": slug, "error": "no anomaly raster"}

    raster = np.load(npz_path)
    minlon = float(raster["bbox_min_lon"]); maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"]); maxlat = float(raster["bbox_max_lat"])
    grid_res = float(raster["grid_resolution_m"])
    gy, gx = raster["anomaly"].shape
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))

    if dem is None:
        dem = load_srtm_dem()

    # Sample elevation at each cell of the flight grid
    col_idx = np.arange(gx, dtype=np.float64) + 0.5
    row_idx = np.arange(gy, dtype=np.float64) + 0.5
    lons = minlon + col_idx * grid_res / mpd_lon
    lats = minlat + row_idx * grid_res / mpd_lat
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    elev_grid = sample_dem_to_grid(dem, lat_grid, lon_grid)

    # Max-filter to find peak elevation within proximity_radius_m
    from scipy import ndimage
    radius_cells = max(1, int(round(proximity_radius_m / grid_res)))
    max_elev_local = ndimage.maximum_filter(elev_grid, size=2 * radius_cells + 1)

    is_cliff_zone = max_elev_local > cliff_height_m

    np.savez_compressed(
        out_path,
        is_cliff_zone=is_cliff_zone,
        max_elev_local=max_elev_local.astype(np.float32),
        elev=elev_grid.astype(np.float32),
        cliff_height_m=cliff_height_m,
        proximity_radius_m=proximity_radius_m,
        bbox_min_lon=minlon, bbox_max_lon=maxlon,
        bbox_min_lat=minlat, bbox_max_lat=maxlat,
        grid_resolution_m=grid_res,
    )
    return {
        "slug": slug,
        "out": str(out_path),
        "cliff_cells": int(is_cliff_zone.sum()),
        "total_cells": int(is_cliff_zone.size),
        "cliff_frac": float(is_cliff_zone.sum()) / float(is_cliff_zone.size),
        "max_elev": float(elev_grid.max()),
        "median_elev": float(np.median(elev_grid)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--cliff-height-m", type=float, default=50.0,
                    help="elevation threshold for cliff classification (default 50 m)")
    ap.add_argument("--proximity-radius-m", type=float, default=100.0,
                    help="search radius for max elevation (default 100 m)")
    args = ap.parse_args()

    print("Loading SRTM DEM…")
    dem = load_srtm_dem()
    print(f"  elev shape {dem['elev'].shape}, range "
          f"{int(dem['elev'].min())}–{int(dem['elev'].max())} m")

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for sd in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = sd.name[: -len("_spread")]
            if (sd / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"\nDeriving cliff zones for {len(slugs)} flight(s)…")
    for slug in slugs:
        try:
            r = derive_for_slug(
                slug,
                cliff_height_m=args.cliff_height_m,
                proximity_radius_m=args.proximity_radius_m,
                dem=dem,
            )
            if "error" in r:
                print(f"  ✗ {slug}: {r['error']}")
            else:
                print(f"  ✓ {slug}: cliff zone covers {100*r['cliff_frac']:.1f}% "
                      f"(max elev {r['max_elev']:.0f} m, median {r['median_elev']:.0f} m)")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
