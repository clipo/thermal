#!/usr/bin/env python3
"""Coast-anchored SGD plume detection.

Real SGD is freshwater emerging from a coastal source (typically a
collapsed lava-tube outlet at a bay or inlet) and flowing OUTWARD into
the ocean. The plume:

  1. Has a SOURCE at the coastline (where freshwater enters the sea)
  2. EXTENDS offshore from the source as freshwater mixes with seawater
  3. DECAYS with distance from source (mixing dilutes the anomaly)
  4. Is bounded — typically <200 m offshore, <100 m alongshore
  5. By definition cannot exist as an isolated blob deep in the ocean
     with no path back to a coastal source

The general-purpose watershed in `derive_polygons_from_raster.py` finds
cold peaks ANYWHERE and floods outward — which produces both legitimate
plumes AND offshore artifact blobs (sensor noise, cliff-shadow
projections, sun-glint accumulations). This script explicitly encodes
the SGD physics:

  Step 1. Define the coastline as the water-cell boundary.
  Step 2. Within a near-coast buffer (default 30 m offshore), find
          local anomaly peaks above the peak threshold — these are
          candidate SGD sources.
  Step 3. Watershed-flood from each source seed OUTWARD through cells
          above the edge threshold, but bounded by max offshore
          distance.
  Step 4. Reject any polygon that doesn't actually touch a coastline
          cell.

Adaptive thresholds (peak = max(0.4, p95 of water anomaly), edge =
max(0.2, p70)) are reused from `derive_polygons_from_raster.py` so each
flight's signal range is respected.

Output: `<slug>_sgd_coastal.geojson` per flight (left distinct from
`<slug>_sgd_raster.geojson` so the comparison can be made).

Master KML aggregator: pass the new files to
`scripts/aggregate_sigma_anomaly_kml.py` to build
`sgd_output/rapa_nui_all_sgd_coastal.kml`.

Usage:
    python scripts/derive_plumes_coast_anchored.py --slug vaihu_full
    python scripts/derive_plumes_coast_anchored.py --all
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
    cz_path = spread / f"{slug}_cliff_zone.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")
    npz = np.load(npz_path)
    water = None
    if wm_path.exists():
        water = np.load(wm_path)["is_water"]
        if water.shape != npz["anomaly"].shape:
            water = None
    cliff_zone = None
    if cz_path.exists():
        cliff_zone = np.load(cz_path)["is_cliff_zone"]
        if cliff_zone.shape != npz["anomaly"].shape:
            cliff_zone = None
    return npz, water, cliff_zone


def adaptive_thresholds(field: np.ndarray, water: np.ndarray | None,
                         floor_peak: float = 0.35,
                         floor_edge: float = 0.2,
                         ceiling_peak: float = 0.55,
                         ceiling_edge: float = 0.35) -> tuple[float, float]:
    """Adapt thresholds to flight's water-cell anomaly distribution,
    clamped to [floor, ceiling]. The ceiling prevents strong-signal
    flights (Hekii peaks 1°C+, cliff-shadow projections at Poike etc.)
    from setting global peak thresholds so high that other coastal
    zones in the same flight get shut out (e.g., Anakena Bay's 0.57°C
    peaks were filtered when flight10's p95 reached 0.61°C)."""
    valid = (field > 0)
    if water is not None:
        valid = valid & water
    vals = field[valid]
    if vals.size < 100:
        return floor_peak, floor_edge
    p95 = float(np.percentile(vals, 95))
    p70 = float(np.percentile(vals, 70))
    peak = float(np.clip(p95, floor_peak, ceiling_peak))
    edge = float(np.clip(max(p70, 0.5 * peak), floor_edge, ceiling_edge))
    return peak, edge


def derive_coastal_plumes(slug: str, *,
                           peak_threshold: float | None = None,
                           edge_threshold: float | None = None,
                           coast_buffer_m: float = 60.0,
                           max_offshore_m: float = 150.0,
                           max_centroid_offshore_m: float = 75.0,
                           peak_min_distance_m: float = 30.0,
                           min_obs: int = 5,
                           max_realistic_anom_c: float = 3.0,
                           min_area_m2: float = 50.0,
                           max_area_m2: float = 8000.0,
                           reject_cliff_zone: bool = True,
                           smooth: bool = True) -> dict:
    npz, water, cliff_zone = load_raster(slug)
    if water is None:
        return {"slug": slug, "error": "no water mask available"}

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

    try:
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from skimage import measure
    except ImportError as e:
        raise SystemExit(f"missing scientific lib: {e}")

    # Quality-filtered field, restricted to water
    finite = np.isfinite(anom)
    quality = finite & (obs >= min_obs) & (anom <= max_realistic_anom_c) & water

    # Cliff-projection filter: water cells within ~15 m of VERY-high-anomaly
    # LAND cells (>1.5°C — clear cliff-shadow signature, not legit
    # surf-zone signal). These water cells are highly likely cliff-shadow
    # pixels misprojected to ocean coordinates by the geometric bug.
    # Filter only if their own anomaly is > 1.0°C (so subtle shore-zone
    # signal stays).
    high_anom_land = ~water & (anom > 1.5) & finite
    if high_anom_land.any():
        cliff_buffer = ndimage.binary_dilation(
            high_anom_land,
            iterations=max(1, int(round(15.0 / grid_res))),
        )
        cliff_projection = water & cliff_buffer & (anom > 1.0)
        quality = quality & ~cliff_projection
        n_cliff_proj = int(cliff_projection.sum())
    else:
        n_cliff_proj = 0

    field = np.where(quality, anom, 0.0).astype(np.float32)

    if smooth:
        field = ndimage.median_filter(field, size=3)

    # Adaptive thresholds — reuse the same logic as the general watershed.
    auto_peak, auto_edge = adaptive_thresholds(field, water)
    if peak_threshold is None:
        peak_threshold = auto_peak
    if edge_threshold is None:
        edge_threshold = auto_edge

    # === Step 1: Coastline (water cells adjacent to land) ===
    land = ~water
    # Cells in water that border land along their 4-neighborhood
    coastline = water & ndimage.binary_dilation(land, iterations=1)
    # 1-cell-thick ring around the coast.

    # === Step 2: Distance from coast (within water) ===
    # `distance_transform_edt` measures distance to the nearest 0 in a
    # boolean mask. Pass `~coastline` so distance is from coast cells.
    coast_dist_cells = ndimage.distance_transform_edt(~coastline)
    coast_dist_m = coast_dist_cells * grid_res

    # === Step 3: Source candidates — cells within coast_buffer_m
    # offshore AND with anomaly >= peak_threshold ===
    near_coast_water = water & (coast_dist_m <= coast_buffer_m)

    # === Cliff-zone exclusion (per SRTM DEM) ===
    # Reject seeds where the local terrain is cliff coast (max elev > 50m
    # within 100m). SGD requires lava-tube conduit topology, which only
    # forms at low-elevation bays / inlets. Vertical cliffs lack the
    # geometry for freshwater to emerge at the sea surface.
    n_cliff_excluded = 0
    if reject_cliff_zone and cliff_zone is not None:
        cliff_excluded = near_coast_water & cliff_zone
        n_cliff_excluded = int(cliff_excluded.sum())
        near_coast_water = near_coast_water & ~cliff_zone

    near_coast_field = np.where(near_coast_water, field, 0.0).astype(np.float32)

    peak_min_distance_cells = max(1, int(round(peak_min_distance_m / grid_res)))
    peaks_rc = peak_local_max(
        near_coast_field,
        min_distance=peak_min_distance_cells,
        threshold_abs=peak_threshold,
        exclude_border=False,
    )
    n_peaks = peaks_rc.shape[0]

    if n_peaks == 0:
        print(f"  no coastal peaks above {peak_threshold:.2f}°C within "
              f"{coast_buffer_m:.0f}m of coast")
        out_path = SGD_OUTPUT / f"{slug}_spread" / f"{slug}_sgd_coastal.geojson"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {"type": "FeatureCollection", "features": []}, indent=2))
        return {"slug": slug, "n_polys": 0, "total_sigma_m2c": 0.0,
                "out": str(out_path)}

    markers = np.zeros(field.shape, dtype=np.int32)
    for i, (r, c) in enumerate(peaks_rc):
        markers[r, c] = i + 1

    # === Step 4: Watershed-flood OUTWARD from each coastal seed.
    # Mask: water cells with anomaly >= edge_threshold AND within
    # max_offshore_m of coast. Also exclude cliff zones — a plume
    # seeded at a non-cliff coastal source shouldn't propagate
    # through cliff-zone cells.
    flood_mask = water & (field >= edge_threshold) & (coast_dist_m <= max_offshore_m)
    if reject_cliff_zone and cliff_zone is not None:
        flood_mask = flood_mask & ~cliff_zone

    # Watershed flows from peaks (high anomaly = basins after negation)
    # down through flood_mask. Each peak owns its own catchment.
    labeled = watershed(-field, markers, mask=flood_mask)
    n_comp = int(labeled.max())

    print(f"  thresholds: peak={peak_threshold:.2f}°C, edge={edge_threshold:.2f}°C; "
          f"{n_peaks} coastal seeds → {n_comp} plume regions; "
          f"cliff-projection cells filtered: {n_cliff_proj:,}; "
          f"cliff-zone source cells excluded: {n_cliff_excluded:,}")

    # === Step 5: Component filter ===
    sizes = ndimage.sum(labeled > 0, labeled, range(1, n_comp + 1)).astype(np.int64)
    min_cells = int(math.ceil(min_area_m2 / cell_area_m2))
    max_cells = int(math.ceil(max_area_m2 / cell_area_m2))

    features = []
    new_id = 0
    n_dropped_centroid = 0
    for comp_id in range(1, n_comp + 1):
        comp_mask = (labeled == comp_id)
        n_cells = int(comp_mask.sum())
        if n_cells < min_cells or n_cells > max_cells:
            continue
        # Must touch the coastline by construction (seed is in
        # near-coast buffer), but verify in case watershed grew away
        if not (comp_mask & coastline).any():
            continue
        # Centroid must be within max_centroid_offshore_m of coast
        rs, cs = np.where(comp_mask)
        centroid_dist_m = float(coast_dist_m[int(rs.mean()), int(cs.mean())])
        if centroid_dist_m > max_centroid_offshore_m:
            n_dropped_centroid += 1
            continue

        # Source: the coastline cell within the polygon closest to the peak
        # (or just any coast cell in the polygon — we'll pick the one with
        # highest anomaly).
        coast_cells_in = comp_mask & coastline
        if coast_cells_in.any():
            cs_anom = np.where(coast_cells_in, field, -1)
            r_src, c_src = np.unravel_index(np.argmax(cs_anom), cs_anom.shape)
            source_lat = float(minlat + (r_src + 0.5) * grid_res / mpd_lat)
            source_lon = float(minlon + (c_src + 0.5) * grid_res / mpd_lon)
        else:
            source_lat = source_lon = float("nan")

        contours = measure.find_contours(comp_mask.astype(np.float32), level=0.5)
        if not contours:
            continue
        contour = max(contours, key=lambda c: c.shape[0])
        lon_ring = (minlon + (contour[:, 1] + 0.5) * grid_res / mpd_lon).tolist()
        lat_ring = (minlat + (contour[:, 0] + 0.5) * grid_res / mpd_lat).tolist()
        if (lon_ring[0], lat_ring[0]) != (lon_ring[-1], lat_ring[-1]):
            lon_ring.append(lon_ring[0]); lat_ring.append(lat_ring[0])
        ring = list(zip(lon_ring, lat_ring))

        comp_anom = anom[comp_mask & finite]
        if comp_anom.size == 0:
            continue
        sigma = float(comp_anom.sum() * cell_area_m2)
        area = float(n_cells * cell_area_m2)
        mean_a = float(comp_anom.mean())
        peak_a = float(comp_anom.max())
        centroid_lat = float(minlat + (rs.mean() + 0.5) * grid_res / mpd_lat)
        centroid_lon = float(minlon + (cs.mean() + 0.5) * grid_res / mpd_lon)
        # Offshore extent: max distance from coast within the polygon
        max_dist = float(coast_dist_m[comp_mask].max())

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [list(ring)]},
            "properties": {
                "id": new_id,
                "source": "coast_anchored_watershed",
                "peak_threshold_c": float(peak_threshold),
                "edge_threshold_c": float(edge_threshold),
                "area_m2": area,
                "sigma_anomaly_m2c": sigma,
                "mean_anomaly_c": mean_a,
                "peak_anomaly_c": peak_a,
                "centroid_lat": centroid_lat,
                "centroid_lon": centroid_lon,
                "source_lat": source_lat,
                "source_lon": source_lon,
                "max_offshore_dist_m": max_dist,
                "n_cells": n_cells,
            },
        })
        new_id += 1

    out_path = SGD_OUTPUT / f"{slug}_spread" / f"{slug}_sgd_coastal.geojson"
    fc = {"type": "FeatureCollection", "features": features,
          "properties": {
              "source": "derive_plumes_coast_anchored.py",
              "peak_threshold_c": float(peak_threshold),
              "edge_threshold_c": float(edge_threshold),
              "coast_buffer_m": coast_buffer_m,
              "max_offshore_m": max_offshore_m,
              "min_area_m2": min_area_m2,
              "max_area_m2": max_area_m2,
          }}
    out_path.write_text(json.dumps(fc, indent=2))
    total = sum(f["properties"]["sigma_anomaly_m2c"] for f in features)
    return {"slug": slug, "n_polys": len(features),
            "total_sigma_m2c": total, "out": str(out_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--peak-threshold", type=float, default=None,
                    help="override adaptive peak threshold (°C)")
    ap.add_argument("--edge-threshold", type=float, default=None,
                    help="override adaptive edge threshold (°C)")
    ap.add_argument("--coast-buffer-m", type=float, default=30.0,
                    help="how far offshore to look for source cells")
    ap.add_argument("--max-offshore-m", type=float, default=200.0,
                    help="maximum extent of plume offshore from coast")
    ap.add_argument("--peak-min-distance-m", type=float, default=30.0)
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--max-realistic", type=float, default=3.0)
    ap.add_argument("--min-area-m2", type=float, default=50.0)
    ap.add_argument("--max-area-m2", type=float, default=8000.0)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--allow-cliff-zone", action="store_true",
                    help="don't exclude cliff-zone seeds from the SRTM-derived "
                         "cliff-zone mask (default: exclude — geological intuition)")
    args = ap.parse_args()

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for sd in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = sd.name[: -len("_spread")]
            if (sd / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"Coast-anchored plume detection for {len(slugs)} flight(s)…")
    for slug in slugs:
        try:
            r = derive_coastal_plumes(
                slug,
                peak_threshold=args.peak_threshold,
                edge_threshold=args.edge_threshold,
                coast_buffer_m=args.coast_buffer_m,
                max_offshore_m=args.max_offshore_m,
                peak_min_distance_m=args.peak_min_distance_m,
                min_obs=args.min_obs,
                max_realistic_anom_c=args.max_realistic,
                min_area_m2=args.min_area_m2,
                max_area_m2=args.max_area_m2,
                reject_cliff_zone=not args.allow_cliff_zone,
                smooth=not args.no_smooth,
            )
            if "error" in r:
                print(f"  ✗ {slug}: {r['error']}")
            else:
                print(f"  ✓ {slug}: {r['n_polys']} plumes, "
                      f"Σ_anomaly = {r['total_sigma_m2c']:,.0f} m²·°C")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
