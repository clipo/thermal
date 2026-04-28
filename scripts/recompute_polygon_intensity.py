#!/usr/bin/env python3
"""Integrate the per-flight cold-anomaly raster within each SGD polygon
footprint to compute a *threshold-independent* intensity metric:

    Σ_anomaly = Σ_cells (anomaly_°C × cell_area_m²)

Units: m²·°C. Unlike the legacy `intensity_index = area × peak_anomaly_c`,
this integral does not depend on where we drew the threshold to declare a
polygon — it captures the entire cold-anomaly content within the polygon
footprint, including subtle edge cells that didn't survive thresholding.

For each flight with both `<slug>_anomaly.npz` and `<slug>_sgd.geojson`,
adds the following properties to every polygon:

  - sigma_anomaly_m2c          : the integral above
  - mean_anomaly_in_polygon_c  : NaN-mean of anomaly cells inside polygon
  - peak_anomaly_in_polygon_c  : maximum cell anomaly inside polygon
  - raster_coverage_frac       : fraction of polygon cells with any observation
  - raster_n_obs_median        : median observation count per cell

The geojson is rewritten in place (existing fields preserved). A companion
CSV `<slug>_sgd_intensity.csv` is also written for cross-flight comparison.

Usage:
    # one flight
    python scripts/recompute_polygon_intensity.py --slug flight4_vaihu_east_full

    # every flight with both raster + polygons present
    python scripts/recompute_polygon_intensity.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from matplotlib.path import Path as MplPath  # for point-in-polygon
except ImportError as e:  # pragma: no cover
    raise SystemExit("matplotlib is required (used for polygon rasterization)")


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def find_flight_pairs(specific_slug: str | None = None) -> list[tuple[str, Path, Path]]:
    """Return (slug, anomaly_npz, sgd_geojson) tuples for flights that have both."""
    out = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        if specific_slug and slug != specific_slug:
            continue
        npz = spread_dir / f"{slug}_anomaly.npz"
        gj = spread_dir / f"{slug}_sgd.geojson"
        if npz.exists() and gj.exists():
            out.append((slug, npz, gj))
    return out


def integrate_polygon(
    polygon_lonlat: list[list[float]],
    anomaly: np.ndarray,
    obs_count: np.ndarray,
    minlon: float,
    minlat: float,
    mpd_lon: float,
    mpd_lat: float,
    grid_resolution_m: float,
    water_mask: np.ndarray | None = None,
) -> dict:
    """Integrate the raster within a single polygon. Returns metrics dict.

    If water_mask is given (same shape as anomaly), only cells where
    water_mask is True are counted; cells over land are excluded from the
    Σ_anomaly. The polygon_water_fraction tells you how much of the
    polygon is over water — values <0.5 indicate the polygon is mostly a
    projection-bug artifact (e.g., cliff shadow projected to ocean coords).
    """
    lons = np.array([p[0] for p in polygon_lonlat], dtype=np.float64)
    lats = np.array([p[1] for p in polygon_lonlat], dtype=np.float64)
    cols_f = (lons - minlon) * mpd_lon / grid_resolution_m
    rows_f = (lats - minlat) * mpd_lat / grid_resolution_m

    gy, gx = anomaly.shape
    cmin = max(0, int(math.floor(cols_f.min())))
    cmax = min(gx, int(math.ceil(cols_f.max())) + 1)
    rmin = max(0, int(math.floor(rows_f.min())))
    rmax = min(gy, int(math.ceil(rows_f.max())) + 1)
    if cmax <= cmin or rmax <= rmin:
        return _empty_metrics()

    path = MplPath(np.column_stack([cols_f, rows_f]))
    cs = np.arange(cmin, cmax) + 0.5
    rs = np.arange(rmin, rmax) + 0.5
    CC, RR = np.meshgrid(cs, rs)
    pts = np.column_stack([CC.ravel(), RR.ravel()])
    inside = path.contains_points(pts).reshape(RR.shape)

    if not inside.any():
        return _empty_metrics()

    sub_anom = anomaly[rmin:rmax, cmin:cmax]
    sub_obs = obs_count[rmin:rmax, cmin:cmax]
    sub_water = water_mask[rmin:rmax, cmin:cmax] if water_mask is not None else None

    n_total = int(inside.sum())
    if sub_water is not None:
        n_water_in_poly = int((inside & sub_water).sum())
        water_fraction = n_water_in_poly / n_total if n_total else 0.0
        finite = np.isfinite(sub_anom) & inside & sub_water
    else:
        n_water_in_poly = n_total  # assume all water if no mask
        water_fraction = 1.0
        finite = np.isfinite(sub_anom) & inside

    n_obs = int(finite.sum())
    if n_obs == 0:
        return _empty_metrics(n_total=n_total, water_fraction=water_fraction)

    anom_vals = sub_anom[finite]
    obs_vals = sub_obs[finite]
    cell_area_m2 = grid_resolution_m * grid_resolution_m

    return {
        "sigma_anomaly_m2c": float(anom_vals.sum() * cell_area_m2),
        "mean_anomaly_in_polygon_c": float(anom_vals.mean()),
        "peak_anomaly_in_polygon_c": float(anom_vals.max()),
        "raster_coverage_frac": float(n_obs) / float(n_total) if n_total else 0.0,
        "raster_n_obs_median": float(np.median(obs_vals)),
        "raster_polygon_cells": n_total,
        "polygon_water_fraction": float(water_fraction),
    }


def _empty_metrics(n_total: int = 0, water_fraction: float = 1.0) -> dict:
    return {
        "sigma_anomaly_m2c": 0.0,
        "mean_anomaly_in_polygon_c": 0.0,
        "peak_anomaly_in_polygon_c": 0.0,
        "raster_coverage_frac": 0.0,
        "raster_n_obs_median": 0.0,
        "raster_polygon_cells": n_total,
        "polygon_water_fraction": float(water_fraction),
    }


def process_flight(slug: str, npz_path: Path, gj_path: Path) -> dict:
    raster = np.load(npz_path)
    anomaly = raster["anomaly"]
    obs_count = raster["observations"]
    minlon = float(raster["bbox_min_lon"])
    maxlat_arr = float(raster["bbox_max_lat"])
    minlat = float(raster["bbox_min_lat"])
    grid_resolution_m = float(raster["grid_resolution_m"])
    centerlat = 0.5 * (minlat + maxlat_arr)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))

    # Optional satellite-derived water mask
    water_mask_path = npz_path.with_name(f"{slug}_water_mask.npz")
    water_mask = None
    if water_mask_path.exists():
        try:
            water_mask = np.load(water_mask_path)["is_water"]
            if water_mask.shape != anomaly.shape:
                print(f"  warn: water mask shape {water_mask.shape} != raster {anomaly.shape}; ignoring")
                water_mask = None
        except Exception as e:
            print(f"  warn: failed to load water mask: {e}")

    fc = json.loads(gj_path.read_text())
    feats = fc.get("features", [])

    rows = []
    for feat in feats:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        m = integrate_polygon(
            ring, anomaly, obs_count,
            minlon=minlon, minlat=minlat,
            mpd_lon=mpd_lon, mpd_lat=mpd_lat,
            grid_resolution_m=grid_resolution_m,
            water_mask=water_mask,
        )
        feat["properties"].update(m)
        rows.append({
            "slug": slug,
            "polygon_id": feat["properties"].get("id"),
            "area_m2": feat["properties"].get("area_m2"),
            "centroid_lat": feat["properties"].get("centroid_lat"),
            "centroid_lon": feat["properties"].get("centroid_lon"),
            "intensity_index": feat["properties"].get("intensity_index"),  # legacy
            **m,
        })

    # Write geojson back in place
    gj_path.write_text(json.dumps(fc, indent=2))

    # Companion CSV for cross-flight comparison
    csv_path = gj_path.with_name(f"{slug}_sgd_intensity.csv")
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    sigma_total = sum(r["sigma_anomaly_m2c"] for r in rows)
    return {
        "slug": slug,
        "n_polygons": len(rows),
        "sigma_anomaly_total_m2c": sigma_total,
        "csv": str(csv_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", help="Process only this flight slug")
    ap.add_argument("--all", action="store_true", help="Process all flights with both raster + polygons")
    args = ap.parse_args()

    if not args.slug and not args.all:
        ap.error("specify --slug <name> or --all")

    pairs = find_flight_pairs(specific_slug=args.slug)
    if not pairs:
        raise SystemExit("No matching flight pairs found.")

    print(f"Processing {len(pairs)} flight(s)…")
    summary = []
    for slug, npz, gj in pairs:
        try:
            res = process_flight(slug, npz, gj)
            summary.append(res)
            print(f"  ✓ {slug}: {res['n_polygons']} polygons, "
                  f"Σ_anomaly = {res['sigma_anomaly_total_m2c']:.0f} m²·°C")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")

    # Write a master cross-flight summary
    if summary:
        master = SGD_OUTPUT / "polygon_intensity_summary.csv"
        with master.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["slug", "n_polygons", "sigma_anomaly_total_m2c"])
            w.writeheader()
            for s in summary:
                w.writerow({k: s[k] for k in ["slug", "n_polygons", "sigma_anomaly_total_m2c"]})
        print(f"\nWrote summary → {master}")


if __name__ == "__main__":
    main()
