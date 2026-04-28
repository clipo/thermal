#!/usr/bin/env python3
"""Derive SGD polygons directly from the per-flight cold-anomaly raster.

For sites where the per-frame detector pipeline produced projection-bug
artifacts (e.g., Tongariki-Poike, Poike-3 — where cliff-face thermal pixels
got geocoded to ocean coordinates), the original polygons can fail the
water-fraction filter and disappear, leaving real surf-zone SGD signal
without polygon coverage.

This script provides an alternative polygon set derived directly from the
water-masked raster:

  1. Load anomaly + obs_count + water_mask
  2. Apply quality filters: obs >= --min-obs, anomaly <= --max-realistic,
     is_water = True
  3. Smooth with a 3x3 median filter (suppress single-pixel noise)
  4. Threshold at --threshold (default 0.3 °C)
  5. Connected components, filter by --min-area-m2 (default 50 m²)
  6. Extract contour polygons via skimage
  7. Compute sigma_anomaly_m2c, area_m2, etc. per polygon
  8. Write GeoJSON to <slug>_spread/<slug>_sgd_raster.geojson

These polygons will exactly match the visible cold zones in the
water-masked raster — by construction.

Usage:
    python scripts/derive_polygons_from_raster.py --slug june2023_1_july_23_tongariki_poike
    python scripts/derive_polygons_from_raster.py --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_raster(slug: str):
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    wm_path = spread / f"{slug}_water_mask.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    npz = np.load(npz_path)
    water = None
    if wm_path.exists():
        water = np.load(wm_path)["is_water"]
        if water.shape != npz["anomaly"].shape:
            print(f"  warn: water mask shape mismatch; ignoring")
            water = None
    return npz, water


def _simplify_ring(coords_xy: np.ndarray, tolerance_m: float,
                    grid_res_m: float) -> np.ndarray:
    """Douglas-Peucker simplification on (col, row) ring coords.
    tolerance is in meters; convert to cell-units."""
    try:
        from shapely.geometry import LinearRing
    except ImportError:
        return coords_xy
    tol_cells = tolerance_m / grid_res_m
    ring = LinearRing(coords_xy)
    simp = ring.simplify(tol_cells, preserve_topology=True)
    return np.array(simp.coords)


def adaptive_thresholds(field: np.ndarray, water_mask: np.ndarray | None,
                         floor_peak: float = 0.4,
                         floor_edge: float = 0.2) -> tuple[float, float]:
    """Adapt peak and edge thresholds to each flight's anomaly distribution.

    Uses the 95th percentile of WATER-cell anomaly as the peak threshold —
    this picks up genuinely strong signals relative to the flight's noise
    floor, regardless of whether the flight has subtle SGD (Vaihu, peaks
    around 0.5°C) or strong (Hekii, peaks to 1°C+). The edge threshold
    is set at half the peak.

    Floors prevent picking up noise at very-quiet flights.
    """
    valid = (field > 0)
    if water_mask is not None:
        valid = valid & water_mask
    vals = field[valid]
    if vals.size < 100:
        return floor_peak, floor_edge
    p95 = float(np.percentile(vals, 95))
    p70 = float(np.percentile(vals, 70))
    peak = max(floor_peak, p95)
    edge = max(floor_edge, p70, 0.5 * peak)
    return peak, edge


def derive_polygons(slug: str, *,
                    threshold: float | None = None,        # None = adaptive
                    peak_threshold: float | None = None,   # None = adaptive
                    min_obs: int = 5,
                    max_realistic_anom_c: float = 3.0,
                    max_land_anomaly_c: float = 0.5,
                    min_area_m2: float = 50.0,
                    max_area_m2: float = 10000.0,
                    peak_min_distance_m: float = 30.0,
                    simplify_tolerance_m: float = 1.5,
                    smooth: bool = True,
                    use_watershed: bool = True) -> dict:
    npz, water = load_raster(slug)
    anom = npz["anomaly"].astype(np.float32)
    obs = npz["observations"]
    grid_res = float(npz["grid_resolution_m"])
    cell_area_m2 = grid_res * grid_res
    minlon = float(npz["bbox_min_lon"])
    minlat = float(npz["bbox_min_lat"])
    maxlat = float(npz["bbox_max_lat"])
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))

    # Quality-filtered field
    finite = np.isfinite(anom)
    quality = (
        finite
        & (obs >= min_obs)
        & (anom <= max_realistic_anom_c)
    )
    if water is not None:
        quality = quality & water
    field = np.where(quality, anom, 0.0).astype(np.float32)

    # Optional 3x3 median smoothing to suppress single-pixel noise
    if smooth:
        try:
            from scipy import ndimage
            field = ndimage.median_filter(field, size=3)
        except ImportError:
            pass

    # Drop high-anomaly cells over satellite-classified land (cliff-shadow filter)
    if water is not None:
        cliff_shadow = ~water & (anom > max_land_anomaly_c)
        field[cliff_shadow] = 0.0

    # Adaptive thresholds: use this flight's own anomaly distribution
    # so subtle-SGD flights (Vaihu, peaks ~0.5°C) get sensitive thresholds
    # while strong-SGD flights (Hekii, peaks 1°C+) get strict ones.
    auto_peak, auto_edge = adaptive_thresholds(field, water)
    if peak_threshold is None:
        peak_threshold = auto_peak
    if threshold is None:
        threshold = auto_edge
    print(f"  thresholds: peak={peak_threshold:.2f}°C, edge={threshold:.2f}°C "
          f"(adaptive: peak_p95={auto_peak:.2f}, edge_p70={auto_edge:.2f})")

    # Threshold to binary mask
    mask = field >= threshold

    try:
        from scipy import ndimage
    except ImportError:
        raise SystemExit("scipy required")

    if use_watershed:
        try:
            from skimage.feature import peak_local_max
            from skimage.segmentation import watershed
        except ImportError:
            raise SystemExit("scikit-image required for watershed segmentation")

        # Find local cold peaks. Each peak seeds one plume polygon.
        peak_min_distance_cells = max(1, int(round(peak_min_distance_m / grid_res)))
        peaks_rc = peak_local_max(
            field, min_distance=peak_min_distance_cells,
            threshold_abs=peak_threshold,
            exclude_border=False,
        )
        n_peaks = peaks_rc.shape[0]

        markers = np.zeros(field.shape, dtype=np.int32)
        for i, (r, c) in enumerate(peaks_rc):
            markers[r, c] = i + 1

        # Watershed flows from each peak outward through the thresholded
        # cold zone. Negate the field so peaks (high anomaly) become basins.
        labeled = watershed(-field, markers, mask=mask)
        n_comp = int(labeled.max())
        print(f"  {n_peaks} cold peaks (>={peak_threshold}°C, min_dist {peak_min_distance_m}m) "
              f"→ {n_comp} watershed regions in {int(mask.sum())} cells (>={threshold}°C)")
    else:
        labeled, n_comp = ndimage.label(mask)

    # Component sizes
    sizes = ndimage.sum(mask, labeled, range(1, n_comp + 1)).astype(np.int64) if n_comp > 0 else np.array([], dtype=np.int64)
    min_cells = int(math.ceil(min_area_m2 / cell_area_m2))
    max_cells = int(math.ceil(max_area_m2 / cell_area_m2))
    keep_ids = [
        i + 1 for i, s in enumerate(sizes)
        if s >= min_cells and s <= max_cells
    ]
    n_too_small = sum(1 for s in sizes if s < min_cells)
    n_too_big = sum(1 for s in sizes if s > max_cells)
    print(f"  components: {len(keep_ids)} kept "
          f"({n_too_small} <{min_area_m2:.0f}m²; {n_too_big} >{max_area_m2:.0f}m²)")

    try:
        from skimage import measure
    except ImportError:
        raise SystemExit("scikit-image is required (used for contour extraction)")

    features = []
    for new_id, comp_id in enumerate(keep_ids):
        comp_mask = (labeled == comp_id)
        # Find a reasonable contour. We use level=0.5 on the binary mask.
        # find_contours returns (row, col) coords in skimage convention.
        contours = measure.find_contours(comp_mask.astype(np.float32), level=0.5)
        if not contours:
            continue
        # Use the longest contour (outer boundary)
        contour = max(contours, key=lambda c: c.shape[0])

        # Convert (row, col) → (lon, lat). Note: row 0 is bottom in our
        # raster (lat-up convention), so lat = minlat + row * grid_res / mpd_lat.
        lon_ring = (minlon + (contour[:, 1] + 0.5) * grid_res / mpd_lon).tolist()
        lat_ring = (minlat + (contour[:, 0] + 0.5) * grid_res / mpd_lat).tolist()
        # Close ring
        if (lon_ring[0], lat_ring[0]) != (lon_ring[-1], lat_ring[-1]):
            lon_ring.append(lon_ring[0]); lat_ring.append(lat_ring[0])
        ring = list(zip(lon_ring, lat_ring))

        # Per-component metrics
        comp_anom = anom[comp_mask & finite]
        if comp_anom.size == 0:
            continue
        sigma = float(comp_anom.sum() * cell_area_m2)
        area = float(comp_mask.sum() * cell_area_m2)
        mean_a = float(comp_anom.mean())
        peak_a = float(comp_anom.max())
        # Centroid in lat/lon
        rs, cs = np.where(comp_mask)
        centroid_lat = float(minlat + (rs.mean() + 0.5) * grid_res / mpd_lat)
        centroid_lon = float(minlon + (cs.mean() + 0.5) * grid_res / mpd_lon)

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [list(ring)]},
            "properties": {
                "id": new_id,
                "source": "raster_threshold",
                "threshold_c": threshold,
                "area_m2": area,
                "sigma_anomaly_m2c": sigma,
                "mean_anomaly_c": mean_a,
                "peak_anomaly_c": peak_a,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "n_cells": int(comp_mask.sum()),
            },
        })

    out = SGD_OUTPUT / f"{slug}_spread" / f"{slug}_sgd_raster.geojson"
    fc = {"type": "FeatureCollection", "features": features,
          "properties": {
              "source": f"derive_polygons_from_raster.py threshold={threshold}",
              "min_obs": min_obs, "max_realistic_anom_c": max_realistic_anom_c,
              "min_area_m2": min_area_m2,
          }}
    out.write_text(json.dumps(fc, indent=2))
    total_sigma = sum(f["properties"]["sigma_anomaly_m2c"] for f in features)
    return {"slug": slug, "n_polys": len(features),
            "total_sigma_m2c": total_sigma, "out": str(out)}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="anomaly °C threshold for plume edges. Default: "
                         "adaptive — max(0.2, p70 of water-cell anomaly).")
    ap.add_argument("--peak-threshold", type=float, default=None,
                    help="anomaly °C threshold for plume cores / peaks. "
                         "Default: adaptive — max(0.4, p95 of water-cell "
                         "anomaly). Lets subtle-SGD flights like Vaihu "
                         "(peak ~0.5°C) get sensitive thresholds while "
                         "Hekii etc. get strict ones.")
    ap.add_argument("--peak-min-distance-m", type=float, default=30.0,
                    help="min distance between detected cold peaks "
                         "(default 30m).")
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--max-realistic", type=float, default=3.0)
    ap.add_argument("--max-land-anomaly", type=float, default=0.5,
                    help="zero out cells over land with anomaly > this "
                         "(cliff-shadow filter; default 0.5°C)")
    ap.add_argument("--min-area-m2", type=float, default=50.0)
    ap.add_argument("--max-area-m2", type=float, default=10000.0,
                    help="drop polygons larger than this (default 10000m²; "
                         "real SGD plumes rarely exceed this — anything "
                         "bigger is diffuse coastal cooling, not a plume)")
    ap.add_argument("--no-watershed", action="store_true",
                    help="use connected-components instead of watershed; "
                         "produces giant blob polygons spanning whole bays")
    ap.add_argument("--no-smooth", action="store_true")
    args = ap.parse_args()

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = spread_dir.name[: -len("_spread")]
            if (spread_dir / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"Deriving raster polygons for {len(slugs)} flight(s)…")
    for slug in slugs:
        try:
            r = derive_polygons(
                slug,
                threshold=args.threshold,
                peak_threshold=args.peak_threshold,
                peak_min_distance_m=args.peak_min_distance_m,
                min_obs=args.min_obs,
                max_realistic_anom_c=args.max_realistic,
                max_land_anomaly_c=args.max_land_anomaly,
                min_area_m2=args.min_area_m2,
                max_area_m2=args.max_area_m2,
                use_watershed=not args.no_watershed,
                smooth=not args.no_smooth,
            )
            print(f"  ✓ {slug}: {r['n_polys']} polygons, "
                  f"Σ_anomaly = {r['total_sigma_m2c']:,.0f} m²·°C")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
