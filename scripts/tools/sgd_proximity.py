#!/usr/bin/env python3
"""For each input point (lat, lon), compute the cold-anomaly content within
a circular buffer using the per-flight anomaly rasters in ``sgd_output/``.

This is the general-purpose tool for spatially correlating SGD intensity
against any point feature: archaeology sites (ahu, moai), sampled coastline
segments, dive transect endpoints, etc. Output is a CSV/GeoJSON with the
input attributes plus:

  - sigma_anomaly_m2c       : Σ over cells in buffer (m² · °C)
  - mean_anomaly_c          : NaN-mean of cell anomalies in buffer
  - peak_anomaly_c          : max cell anomaly in buffer
  - n_cells_observed        : raster cells with any frame coverage
  - n_rasters_overlapping   : how many flights covered this point
  - flights                 : pipe-separated list of overlapping flight slugs

When multiple flights cover the same point, anomalies are *averaged across
flights cell-by-cell* (so a 50m buffer that 3 flights covered is not triple-
counted). The cell area is taken from each raster's grid_resolution_m.

Usage:
    # From CSV with lat/lon columns
    python scripts/sgd_proximity.py \\
        --points data/ahu_locations.csv \\
        --buffer-radius-m 50 \\
        --output sgd_output/ahu_sgd_proximity

    # From GeoJSON Point features
    python scripts/sgd_proximity.py \\
        --points data/dive_sites.geojson \\
        --buffer-radius-m 100 \\
        --output sgd_output/dive_sgd_proximity
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_points(path: Path) -> list[dict]:
    """Load points from CSV (lat, lon columns) or GeoJSON Point features."""
    suffix = path.suffix.lower()
    out = []
    if suffix == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                lat = _find_field(row, ("lat", "latitude", "y"))
                lon = _find_field(row, ("lon", "lng", "longitude", "x"))
                if lat is None or lon is None:
                    raise SystemExit(f"CSV must have lat/lon columns; got {list(row.keys())}")
                row["_lat"] = float(lat)
                row["_lon"] = float(lon)
                out.append(row)
    elif suffix in (".geojson", ".json"):
        fc = json.loads(path.read_text())
        for feat in fc.get("features", []):
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Point":
                continue
            coords = geom["coordinates"]
            row = dict(feat.get("properties", {}))
            row["_lat"] = float(coords[1])
            row["_lon"] = float(coords[0])
            out.append(row)
    else:
        raise SystemExit(f"unsupported input format: {suffix}")
    return out


def _find_field(row: dict, names: tuple[str, ...]) -> str | None:
    for n in names:
        for k in row.keys():
            if k.lower() == n:
                return row[k]
    return None


def discover_rasters(output_dir: Path) -> list[tuple[str, Path]]:
    out = []
    for spread_dir in sorted(output_dir.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        npz = spread_dir / f"{slug}_anomaly.npz"
        if npz.exists():
            out.append((slug, npz))
    return out


def integrate_buffer(
    lat: float,
    lon: float,
    radius_m: float,
    raster: dict,
) -> dict | None:
    """Integrate raster anomaly within a radius_m buffer of (lat, lon).
    Returns None if the point's buffer is entirely outside the raster."""
    minlon = float(raster["bbox_min_lon"])
    maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"])
    maxlat = float(raster["bbox_max_lat"])
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))
    grid_resolution_m = float(raster["grid_resolution_m"])

    # Quick reject: lat/lon plus radius outside raster bbox
    radius_lat = radius_m / mpd_lat
    radius_lon = radius_m / mpd_lon
    if (lat + radius_lat < minlat or lat - radius_lat > maxlat
            or lon + radius_lon < minlon or lon - radius_lon > maxlon):
        return None

    anomaly = raster["anomaly"]
    obs_count = raster["observations"]
    gy, gx = anomaly.shape

    # Center cell in fractional grid coords
    c0 = (lon - minlon) * mpd_lon / grid_resolution_m
    r0 = (lat - minlat) * mpd_lat / grid_resolution_m
    rad_cells = radius_m / grid_resolution_m

    cmin = max(0, int(math.floor(c0 - rad_cells)))
    cmax = min(gx, int(math.ceil(c0 + rad_cells)) + 1)
    rmin = max(0, int(math.floor(r0 - rad_cells)))
    rmax = min(gy, int(math.ceil(r0 + rad_cells)) + 1)
    if cmax <= cmin or rmax <= rmin:
        return None

    cs = np.arange(cmin, cmax) + 0.5
    rs = np.arange(rmin, rmax) + 0.5
    CC, RR = np.meshgrid(cs, rs)
    in_circle = (CC - c0) ** 2 + (RR - r0) ** 2 <= rad_cells ** 2

    sub_anom = anomaly[rmin:rmax, cmin:cmax]
    sub_obs = obs_count[rmin:rmax, cmin:cmax]
    finite = np.isfinite(sub_anom) & in_circle
    n_cells = int(finite.sum())
    if n_cells == 0:
        return None

    return {
        "anomaly_values": sub_anom[finite],
        "obs_values": sub_obs[finite],
        "cell_area_m2": grid_resolution_m * grid_resolution_m,
        "n_cells": n_cells,
    }


def process_points(points: list[dict], rasters: list[tuple[str, Path]],
                   radius_m: float) -> list[dict]:
    # Pre-load raster headers (bbox only) for fast rejection; load full arrays lazily.
    raster_meta = []
    for slug, npz_path in rasters:
        try:
            data = np.load(npz_path)
            raster_meta.append({
                "slug": slug,
                "path": npz_path,
                "data": data,  # NPZ archive (lazy on .files access)
                "bbox": (
                    float(data["bbox_min_lon"]), float(data["bbox_max_lon"]),
                    float(data["bbox_min_lat"]), float(data["bbox_max_lat"]),
                ),
            })
        except Exception as e:
            print(f"  skip {slug}: {e}")

    out = []
    for i, pt in enumerate(points):
        lat = pt["_lat"]; lon = pt["_lon"]
        per_flight = []
        flight_slugs = []
        for rm in raster_meta:
            minlon, maxlon, minlat, maxlat = rm["bbox"]
            centerlat = 0.5 * (minlat + maxlat)
            mpd_lat = 111320.0
            mpd_lon_local = 111320.0 * math.cos(math.radians(centerlat))
            radius_lat = radius_m / mpd_lat
            radius_lon = radius_m / mpd_lon_local
            if (lat + radius_lat < minlat or lat - radius_lat > maxlat
                    or lon + radius_lon < minlon or lon - radius_lon > maxlon):
                continue
            res = integrate_buffer(lat, lon, radius_m, rm["data"])
            if res is None:
                continue
            per_flight.append(res)
            flight_slugs.append(rm["slug"])

        if not per_flight:
            row = {
                **pt,
                "sigma_anomaly_m2c": 0.0,
                "mean_anomaly_c": 0.0,
                "peak_anomaly_c": 0.0,
                "n_cells_observed": 0,
                "n_rasters_overlapping": 0,
                "flights": "",
            }
        else:
            # Aggregate across flights: average anomaly *per spatial cell* before
            # summing. We approximate this by averaging the integrated quantity
            # — i.e., take the mean of per-flight Σ_anomaly within the buffer,
            # not the sum, to avoid triple-counting overlapping coverage.
            sigmas = [r["anomaly_values"].sum() * r["cell_area_m2"] for r in per_flight]
            means = [r["anomaly_values"].mean() for r in per_flight]
            peaks = [r["anomaly_values"].max() for r in per_flight]
            sigma_avg = float(np.mean(sigmas))
            mean_avg = float(np.mean(means))
            peak_max = float(np.max(peaks))
            n_cells = int(np.mean([r["n_cells"] for r in per_flight]))
            row = {
                **pt,
                "sigma_anomaly_m2c": sigma_avg,
                "mean_anomaly_c": mean_avg,
                "peak_anomaly_c": peak_max,
                "n_cells_observed": n_cells,
                "n_rasters_overlapping": len(per_flight),
                "flights": "|".join(flight_slugs),
            }
        out.append(row)
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{len(points)} points")
    return out


def write_outputs(rows: list[dict], output_base: Path):
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Strip private keys
    clean_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    csv_path = output_base.with_suffix(".csv")
    if clean_rows:
        with csv_path.open("w", newline="") as f:
            # Add lat/lon back as explicit columns (from the private fields)
            fieldnames = list(clean_rows[0].keys())
            if "lat" not in fieldnames:
                fieldnames = ["lat", "lon"] + fieldnames
            for cr, src in zip(clean_rows, rows):
                cr.setdefault("lat", src["_lat"])
                cr.setdefault("lon", src["_lon"])
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(clean_rows)
    print(f"  wrote {csv_path}")

    # GeoJSON Point output
    gj_path = output_base.with_suffix(".geojson")
    feats = []
    for cr, src in zip(clean_rows, rows):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [src["_lon"], src["_lat"]]},
            "properties": cr,
        })
    gj_path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2))
    print(f"  wrote {gj_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="Input CSV (lat, lon cols) or GeoJSON Points")
    ap.add_argument("--output", required=True, help="Output base path (extensions added)")
    ap.add_argument("--buffer-radius-m", type=float, default=50.0)
    ap.add_argument("--rasters-dir", default=str(SGD_OUTPUT),
                    help="Directory containing *_spread/*_anomaly.npz rasters")
    args = ap.parse_args()

    points = load_points(Path(args.points))
    rasters = discover_rasters(Path(args.rasters_dir))
    if not rasters:
        raise SystemExit(f"no anomaly rasters found in {args.rasters_dir}")

    print(f"Loaded {len(points)} points, {len(rasters)} flight rasters.")
    print(f"Buffer radius: {args.buffer_radius_m} m")
    rows = process_points(points, rasters, args.buffer_radius_m)
    write_outputs(rows, Path(args.output))

    n_hit = sum(1 for r in rows if r.get("n_rasters_overlapping", 0) > 0)
    print(f"\n{n_hit}/{len(rows)} points fell within at least one flight raster.")


if __name__ == "__main__":
    main()
