#!/usr/bin/env python3
"""
Detect SGD along a stretch of coast and produce merged, area-ranked polygons.

Runs `RedesignedSGDDetector` over a frame range, georeferences each plume's
thermal-pixel contour to a lat/lon polygon, merges overlapping polygons across
frames (Shapely unary_union), computes each merged polygon's area on a local
equal-area projection, and writes:

  * `<output>.kml`      — merged polygons, color-ramped by area tier
  * `<output>.geojson`  — same polygons as GeoJSON FeatureCollection
  * `<output>.csv`      — one row per merged polygon with area_m2, centroid
  * `<output>_summary.json` — run stats (frames, detections, total area, percentiles)

Usage:

    python scripts/run_coast_stretch.py \\
        --data "/Volumes/RapaNui/Rapa Nui Jan 2024/Autel/Flight 4 - Vaihu - East/100MEDIA" \\
        --start 1 --end 250 --step 1 \\
        --output sgd_output/vaihu_east/vaihu_east_sgd

Area tiers (KML colors):
  < 2 m²   : yellow       (small seeps)
  2-10 m²  : orange
  10-50 m² : red
  50-200 m²: magenta
  > 200 m² : purple       (large plumes)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon, shape as shp_shape
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from shapely.strtree import STRtree
from scipy import ndimage
from scipy.spatial import cKDTree

from sgd_toolkit.detectors import RedesignedSGDDetector, SpreadSGDDetector
from sgd_toolkit.georeferencing.polygon_georef import SGDPolygonGeoref


AREA_TIERS = [
    (2.0, "small", "ff00ffff"),        # yellow (KML AABBGGRR)
    (10.0, "modest", "ff0080ff"),      # orange
    (50.0, "moderate", "ff0000ff"),    # red
    (200.0, "large", "ffff00ff"),      # magenta
    (float("inf"), "very_large", "ffff00a0"),  # purple
]


def area_tier(area_m2: float) -> tuple[str, str]:
    for upper, name, color in AREA_TIERS:
        if area_m2 < upper:
            return name, color
    return AREA_TIERS[-1][1], AREA_TIERS[-1][2]


@dataclass
class DetectionRun:
    frames_attempted: int = 0
    frames_with_detections: int = 0
    frames_failed: int = 0
    raw_polygons: int = 0
    merged_polygons: int = 0
    total_area_m2: float = 0.0
    elapsed_s: float = 0.0
    failures: list[tuple[int, str]] = None

    def __post_init__(self):
        if self.failures is None:
            self.failures = []


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def cluster_polygons_by_density_grid(
    polys: list[Polygon],
    grid_resolution_m: float = 1.0,
    min_observations: int | None = None,
    max_raw_area_m2: float = 500.0,
    min_observations_fraction: float = 0.5,
    min_observations_floor: int = 3,
    max_site_diameter_m: float | None = None,
    local_adaptive_window_m: float | None = None,
    local_adaptive_fraction: float = 0.5,
    polygon_anomalies: list[float] | None = None,
) -> tuple[list[tuple[Polygon, int, dict]], dict]:
    """Alternative site identification by observation-density on a fixed spatial grid.

    Chains are mathematically impossible: we rasterize each valid raw polygon
    onto a common metric grid, count how many polygons cover each cell, threshold
    at `min_observations`, and treat each connected-component of passing cells
    as one SGD site.

    Auto-tuning: if `min_observations` is None, it is set to
    `max(min_observations_floor, round(min_observations_fraction * max_count))`
    where `max_count` is the peak per-cell observation count after rasterization.
    This adapts to flight length / drone speed without per-survey tuning. A
    250-frame flight with peak ~30 observations gets min_obs ~15; an 865-frame
    flight with peak ~80 gets min_obs ~40.

    Returns (sites, diagnostics) where `sites` is a list of
    (site_polygon, peak_observation_count_in_component).
    """
    from skimage import measure as sk_measure

    areas = np.array([compute_polygon_area_m2(p) for p in polys])
    # Keep only valid single Polygons with reasonable area. MultiPolygons can
    # arise from shapely.buffer(0) fixup on self-intersecting rings; we split
    # them into their constituent polygons instead of discarding.
    kept: list[Polygon] = []
    kept_anomalies: list[float] = []
    use_anomalies = polygon_anomalies is not None and len(polygon_anomalies) == len(polys)
    for i, p in enumerate(polys):
        if p.is_empty or areas[i] <= 0 or areas[i] > max_raw_area_m2:
            continue
        anom = float(polygon_anomalies[i]) if use_anomalies else 0.0
        if isinstance(p, MultiPolygon):
            for piece in p.geoms:
                kept.append(piece)
                kept_anomalies.append(anom)
        elif isinstance(p, Polygon):
            kept.append(p)
            kept_anomalies.append(anom)
    if not kept:
        return [], {"max_peak_obs": 0, "min_observations_used": min_observations or 0}

    centroids = np.array([[p.centroid.x, p.centroid.y] for p in kept], dtype=np.float64)
    lat0 = float(centroids[:, 1].mean())
    lon0 = float(centroids[:, 0].mean())
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * np.cos(np.radians(lat0))

    # Project every polygon to local UTM-like meters (equirectangular).
    def to_meters(p: Polygon) -> Polygon:
        ext = np.array(p.exterior.coords, dtype=np.float64)
        xm = (ext[:, 0] - lon0) * mpd_lon
        ym = (ext[:, 1] - lat0) * mpd_lat
        meter_coords = list(zip(xm, ym))
        return Polygon(meter_coords)

    meter_polys = [to_meters(p) for p in kept]

    # Grid extent from polygon bounds.
    all_bounds = np.array([p.bounds for p in meter_polys])
    minx = float(all_bounds[:, 0].min()) - 2.0
    miny = float(all_bounds[:, 1].min()) - 2.0
    maxx = float(all_bounds[:, 2].max()) + 2.0
    maxy = float(all_bounds[:, 3].max()) + 2.0

    gx = int(np.ceil((maxx - minx) / grid_resolution_m))
    gy = int(np.ceil((maxy - miny) / grid_resolution_m))
    if gx <= 0 or gy <= 0:
        return [], {"max_peak_obs": 0, "min_observations_used": min_observations or 0}

    counts = np.zeros((gy, gx), dtype=np.int32)
    # Sum of per-polygon mean-anomaly contributions per cell. Used to compute
    # mean_anomaly_c per merged region: anomaly_sum[mask] / counts[mask] gives
    # per-cell average; mean over the region is mean of that. Equivalent to a
    # frame-weighted mean anomaly for each merged polygon.
    anomaly_sum = np.zeros((gy, gx), dtype=np.float32) if use_anomalies else None

    # Rasterize each polygon onto the counts grid.
    from skimage.draw import polygon as sk_polygon

    for idx, mp in enumerate(meter_polys):
        ext = np.array(mp.exterior.coords, dtype=np.float64)
        col = (ext[:, 0] - minx) / grid_resolution_m
        row = (ext[:, 1] - miny) / grid_resolution_m
        rr, cc = sk_polygon(row, col, shape=(gy, gx))
        if rr.size:
            counts[rr, cc] += 1
            if anomaly_sum is not None:
                anomaly_sum[rr, cc] += kept_anomalies[idx]

    max_peak_obs = int(counts.max()) if counts.size else 0

    # Auto-tune min_observations if not explicitly set.
    if min_observations is None:
        auto_value = int(round(min_observations_fraction * max_peak_obs))
        min_observations = max(min_observations_floor, auto_value)
        print(
            f"  min_observations auto-tuned: max_peak={max_peak_obs}, "
            f"fraction={min_observations_fraction:.2f} → using {min_observations}"
        )

    diagnostics = {
        "max_peak_obs": max_peak_obs,
        "min_observations_used": int(min_observations),
        "min_observations_fraction": float(min_observations_fraction),
    }

    # Threshold to a dense mask. For long-coast surveys, the global max_peak
    # is dominated by one heavily-overlapped area (e.g., a bay where the drone
    # circled), and any single global threshold either under-detects in that
    # bay or over-detects everywhere else. We use a LOCALLY-ADAPTIVE threshold
    # in addition to the global one: a cell qualifies if its count is ≥ a
    # fraction (default 0.5) of the local maximum observation count in a
    # window around it. This way the eastern tip of a flight (low coverage)
    # is judged against its own peak, not the survey-wide peak.
    if local_adaptive_window_m is not None and local_adaptive_window_m > 0:
        win_cells = max(3, int(round(local_adaptive_window_m / grid_resolution_m)))
        local_peak = ndimage.maximum_filter(counts, size=win_cells, mode="constant", cval=0)
        # Locally-adaptive: each cell judged against its window's peak count.
        # Equivalent to "cell is in the top X% of frame coverage in its
        # neighborhood." Eastern tip with local_peak=35 gets threshold 17
        # while the harbor with local_peak=85 gets threshold 42 — both
        # capture their respective bay-scale features.
        local_threshold = (local_adaptive_fraction * local_peak).astype(np.int32)
        # Absolute floor prevents low-coverage open-ocean noise from passing
        # (where local_peak might be just 3-5 and any cell trivially meets
        # local_fraction). Floor scales with the survey's coverage scale.
        local_floor = max(
            min_observations_floor,
            int(round(local_adaptive_fraction * 0.25 * max_peak_obs)),
            8,  # hard minimum: a real persistent feature needs ≥8 frame observations
        )
        dense = (counts >= local_threshold) & (counts >= local_floor)
        diagnostics["local_adaptive_window_m"] = float(local_adaptive_window_m)
        diagnostics["local_adaptive_fraction"] = float(local_adaptive_fraction)
        diagnostics["local_floor"] = local_floor
        print(
            f"  local-adaptive threshold ON: window={local_adaptive_window_m} m, "
            f"local_fraction={local_adaptive_fraction}, absolute_floor={local_floor}"
        )
    else:
        dense = counts >= min_observations
    labels, n_comp = sk_measure.label(dense, connectivity=2, return_num=True)
    if n_comp == 0:
        return [], diagnostics

    # Splitting oversize components into bay-scale sub-regions via simple grid
    # tiling. A convex bay has exactly one distance-transform peak so watershed
    # alone can't split it. Instead, divide the bounding box of each oversize
    # component into a regular grid of `max_site_diameter_m`-sized tiles and
    # assign each component cell to its tile. Fast (O(component_size)) and
    # produces sub-regions of bounded diameter.
    if max_site_diameter_m is not None and max_site_diameter_m > 0:
        tile_size_cells = max(2, int(round(max_site_diameter_m / grid_resolution_m)))
        new_labels = np.zeros_like(labels)
        next_id = 1
        n_split_components = 0
        for lid in range(1, n_comp + 1):
            comp = labels == lid
            if not comp.any():
                continue
            ys, xs = np.where(comp)
            comp_diag_cells = np.sqrt(
                (ys.max() - ys.min()) ** 2 + (xs.max() - xs.min()) ** 2
            )
            comp_diag_m = comp_diag_cells * grid_resolution_m
            if comp_diag_m <= max_site_diameter_m:
                new_labels[comp] = next_id
                next_id += 1
                continue
            # Tile with origin at component min so tile (0,0) lines up with
            # the bounding box rather than the global grid.
            ymin, xmin = int(ys.min()), int(xs.min())
            tile_y = (ys - ymin) // tile_size_cells
            tile_x = (xs - xmin) // tile_size_cells
            n_tx = int(tile_x.max()) + 1
            tile_id = tile_y * n_tx + tile_x  # unique flat tile index
            unique_tiles = np.unique(tile_id)
            for tid in unique_tiles:
                mask = tile_id == tid
                if not mask.any():
                    continue
                new_labels[ys[mask], xs[mask]] = next_id
                next_id += 1
                n_split_components += 1
        labels = new_labels
        n_comp = int(labels.max())
        diagnostics["tile_split_components"] = n_split_components
        diagnostics["max_site_diameter_m"] = float(max_site_diameter_m)

    results = []
    for lid in range(1, n_comp + 1):
        mask = labels == lid
        if not mask.any():
            continue
        # Build a polygon around the component (outer contour of the mask)
        # in local meters, then project back to lon/lat.
        contours = sk_measure.find_contours(mask.astype(float), 0.5)
        if not contours:
            continue
        # Largest contour only (skip holes for now).
        contour = max(contours, key=len)
        # Contour is in (row, col) floats. Convert to meter coords.
        rows = contour[:, 0]
        cols = contour[:, 1]
        xm = minx + cols * grid_resolution_m
        ym = miny + rows * grid_resolution_m
        # Back to lon/lat
        lon = xm / mpd_lon + lon0
        lat = ym / mpd_lat + lat0
        poly_ll = Polygon(zip(lon, lat))
        if not poly_ll.is_valid:
            poly_ll = poly_ll.buffer(0)
        if poly_ll.is_empty:
            continue
        if isinstance(poly_ll, MultiPolygon):
            poly_ll = max(poly_ll.geoms, key=lambda g: g.area)
        # Aggregated stats per merged region
        peak_count = int(counts[mask].max())
        median_obs_count = float(np.median(counts[mask]))
        info = {
            "peak_obs": peak_count,
            "median_obs": median_obs_count,
        }
        if anomaly_sum is not None:
            # Per-cell mean anomaly = anomaly_sum / counts (skip count==0).
            with np.errstate(invalid="ignore", divide="ignore"):
                per_cell_anomaly = np.where(counts > 0, anomaly_sum / counts, np.nan)
            cell_vals = per_cell_anomaly[mask]
            cell_vals = cell_vals[np.isfinite(cell_vals)]
            if cell_vals.size:
                info["mean_anomaly_c"] = float(cell_vals.mean())
                info["min_anomaly_c"] = float(cell_vals.max())  # max anomaly = coldest
                info["p90_anomaly_c"] = float(np.percentile(cell_vals, 90))
            else:
                info["mean_anomaly_c"] = 0.0
                info["min_anomaly_c"] = 0.0
                info["p90_anomaly_c"] = 0.0
        results.append((poly_ll, peak_count, info))

    return results, diagnostics


def cluster_polygons_by_site(
    polys: list[Polygon],
    cluster_distance_m: float,
    max_raw_area_m2: float,
    use_iou_merge: bool = True,
    iou_threshold: float = 0.15,
) -> list[list[int]]:
    """Group polygon indices into same-discharge-site clusters.

    Criteria for two polygons to be in the same cluster:
      (a) their centroids are within `cluster_distance_m` on the ground, OR
      (b) they overlap with IoU >= 0.15 (tight geometric overlap = same plume seen twice).

    A polygon with area > `max_raw_area_m2` is considered pathological (single-
    frame over-detection) and dropped from clustering entirely. Without this
    drop, a 5000 m² false positive links every nearby plume into a ribbon.

    Union-find over the (centroid-distance OR IoU) edge set.
    """
    n = len(polys)
    centroids = np.array([[p.centroid.x, p.centroid.y] for p in polys], dtype=np.float64)
    areas = np.array([compute_polygon_area_m2(p) for p in polys])

    drop_mask = areas > max_raw_area_m2
    kept_idx = np.array([i for i in range(n) if not drop_mask[i]], dtype=np.int64)
    if kept_idx.size == 0:
        return []

    parent = np.arange(n, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Project lon/lat to local metric XY (equirectangular around mean lat)
    # so we can use a Euclidean KDTree and a distance radius in meters.
    kept_centroids = centroids[kept_idx]
    lat0 = float(kept_centroids[:, 1].mean())
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * np.cos(np.radians(lat0))
    xy = np.column_stack(
        [
            (kept_centroids[:, 0] - kept_centroids[:, 0].mean()) * mpd_lon,
            (kept_centroids[:, 1] - lat0) * mpd_lat,
        ]
    )
    tree = cKDTree(xy)
    # Vectorized neighbor pairs within cluster_distance_m.
    pairs = tree.query_pairs(r=cluster_distance_m, output_type="ndarray")
    for a, b in pairs:
        union(int(kept_idx[a]), int(kept_idx[b]))

    # Optional: geometric IoU merge. OFF by default because with dense coastal
    # detections (4k+ polygons) even a 0.15 threshold creates chains via
    # transitive overlap — site A overlaps site B 20%, B overlaps C 20%, so
    # A-B-C all cluster even though A and C are far apart and unrelated.
    # Centroid clustering alone is the cleaner per-discharge-site grouping.
    if use_iou_merge:
        kept_polys = [polys[i] for i in kept_idx]
        stree = STRtree(kept_polys)
        for local_i in range(len(kept_polys)):
            poly_i = kept_polys[local_i]
            cand_local = stree.query(poly_i)
            for local_j in cand_local:
                if local_j <= local_i:
                    continue
                i_global = int(kept_idx[local_i])
                j_global = int(kept_idx[local_j])
                if find(i_global) == find(j_global):
                    continue
                inter = poly_i.intersection(kept_polys[local_j])
                if inter.is_empty:
                    continue
                smaller = min(poly_i.area, kept_polys[local_j].area)
                if smaller > 0 and inter.area / smaller >= iou_threshold:
                    union(i_global, j_global)

    clusters: dict[int, list[int]] = {}
    for i in kept_idx:
        i = int(i)
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def compute_polygon_area_m2(geom) -> float:
    """Area of a lon/lat geometry in square meters via equirectangular projection
    around the geometry's centroid. Accepts Polygon or MultiPolygon. Accurate
    to better than 0.1% for plumes of tens-to-hundreds of meters at Rapa Nui
    latitude."""
    if geom is None or geom.is_empty:
        return 0.0
    if isinstance(geom, MultiPolygon):
        return float(sum(compute_polygon_area_m2(p) for p in geom.geoms))
    if not isinstance(geom, Polygon):
        return 0.0
    cx, cy = geom.centroid.x, geom.centroid.y
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * np.cos(np.radians(cy))
    ext = np.array(geom.exterior.coords, dtype=np.float64)
    x = (ext[:, 0] - cx) * mpd_lon
    y = (ext[:, 1] - cy) * mpd_lat
    area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
    for interior in geom.interiors:
        ir = np.array(interior.coords, dtype=np.float64)
        ix = (ir[:, 0] - cx) * mpd_lon
        iy = (ir[:, 1] - cy) * mpd_lat
        hole = 0.5 * abs(np.dot(ix[:-1], iy[1:]) - np.dot(ix[1:], iy[:-1]))
        area -= hole
    return float(area)


def iter_frames(data_dir: Path, start: int, end: int, step: int) -> list[int]:
    frames = []
    for n in range(start, end + 1, step):
        if (data_dir / f"MAX_{n:04d}.JPG").exists() and (data_dir / f"IRX_{n:04d}.irg").exists():
            frames.append(n)
    return frames


def load_raw_polys_from_geojson(path: Path) -> tuple[list[Polygon], list[dict]]:
    with open(path) as f:
        fc = json.load(f)
    polys, records = [], []
    for feat in fc["features"]:
        try:
            g = shp_shape(feat["geometry"])
        except Exception:
            continue
        if not isinstance(g, Polygon) or g.is_empty:
            continue
        polys.append(g)
        records.append(feat.get("properties", {}))
    return polys, records


def save_raw_polys_geojson(path: Path, polys: list[Polygon], records: list[dict]):
    features = []
    for p, rec in zip(polys, records):
        if not isinstance(p, Polygon) or p.is_empty:
            continue
        ext = [list(pt) for pt in p.exterior.coords]
        holes = [[list(pt) for pt in ring.coords] for ring in p.interiors]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ext] + holes},
                "properties": rec,
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"  wrote raw polygons → {path}")


def run(args) -> DetectionRun:
    data_dir = Path(args.data) if args.data else None
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    raw_geojson_path = out_base.with_name(out_base.name + "_raw.geojson")

    if args.merge_only:
        if not raw_geojson_path.exists():
            raise SystemExit(
                f"--merge-only requested but {raw_geojson_path} not found. "
                "Run once without --merge-only (with --save-raw) first."
            )
        print(f"Loading raw polygons from {raw_geojson_path}")
        raw_polys, raw_records = load_raw_polys_from_geojson(raw_geojson_path)
        run_stats = DetectionRun(raw_polygons=len(raw_polys))
        print(f"Loaded {len(raw_polys)} raw polygons — skipping detection")
        return _merge_and_write(raw_polys, raw_records, run_stats, args, out_base)

    assert data_dir is not None, "--data required unless --merge-only"

    if args.detector == "spread":
        detector = SpreadSGDDetector(
            base_path=str(data_dir),
            use_ml=False,
            delta_c=args.delta_c,
            min_area_px=args.min_area,
        )
    else:
        detector = RedesignedSGDDetector(
            base_path=str(data_dir),
            use_ml=False,
            temp_threshold=args.temp_threshold,
            min_area=args.min_area,
        )
    georef = SGDPolygonGeoref(base_path=str(data_dir))

    frames = iter_frames(data_dir, args.start, args.end, args.step)
    if not frames:
        raise SystemExit(f"No paired MAX/IRX frames in range [{args.start}..{args.end}]")

    print(f"Processing {len(frames)} frames from {data_dir}")

    run_stats = DetectionRun(frames_attempted=len(frames))
    raw_polys: list[Polygon] = []  # all polygons before merge
    raw_records: list[dict] = []   # one dict per raw polygon with provenance

    t0 = time.perf_counter()
    for i, fn in enumerate(frames, start=1):
        try:
            data = detector.load_frame_data(fn)
        except Exception as e:
            run_stats.frames_failed += 1
            run_stats.failures.append((fn, f"load: {e}"))
            continue

        masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
        sgd_mask, plume_info, chars = detector.detect_sgd_plumes(data["thermal"], masks)

        if not plume_info:
            if args.verbose:
                print(f"  frame {fn}: no detections")
            continue

        # Georeference each plume's thermal contour → lat/lon polygon
        try:
            georef_features = georef.process_frame_with_polygons(
                frame_number=fn, plume_info_list=plume_info, verbose=False
            )
        except Exception as e:
            run_stats.frames_failed += 1
            run_stats.failures.append((fn, f"georef: {e}"))
            continue

        # The georef strips temperature fields (it expects "mean_temp_diff" but
        # the detector saves "temperature_anomaly"). Pair georef polygons with
        # the corresponding plume_info entries by index so we can carry the
        # full temperature record into the raw geojson.
        frame_polys_n = 0
        baseline_c = float(chars.get("baseline_c", 0.0)) if isinstance(chars, dict) else 0.0
        for feat_i, feat in enumerate(georef_features):
            poly_coords = feat.get("polygon")
            if not poly_coords or len(poly_coords) < 3:
                continue
            # Match georef feature back to detector plume by index
            src = plume_info[feat_i] if feat_i < len(plume_info) else {}
            try:
                p = Polygon(poly_coords)
                if not p.is_valid:
                    p = p.buffer(0)
                if p.is_empty or p.area == 0:
                    continue
                raw_polys.append(p)
                raw_records.append(
                    {
                        "frame": fn,
                        "plume_id": src.get("id"),
                        "area_pixels": int(src.get("area_pixels", 0) or 0),
                        "mean_temp_c": float(src.get("mean_temp", 0.0) or 0.0),
                        "min_temp_c": float(src.get("min_temp", 0.0) or 0.0),
                        # Anomaly: how much COLDER than the flight's warm baseline.
                        # SpreadSGDDetector stores it as `temperature_anomaly` =
                        # mean_temp - baseline (negative). Flip sign here so
                        # downstream code treats positive numbers as "X °C below
                        # ambient" — more intuitive when ranking by intensity.
                        "mean_anomaly_c": -float(src.get("temperature_anomaly", 0.0) or 0.0),
                        "min_anomaly_c": float(baseline_c) - float(src.get("min_temp", 0.0) or 0.0),
                        "baseline_c": baseline_c,
                    }
                )
                frame_polys_n += 1
            except Exception as e:
                run_stats.failures.append((fn, f"polygon: {e}"))
                continue

        if frame_polys_n:
            run_stats.frames_with_detections += 1
            run_stats.raw_polygons += frame_polys_n
        if args.verbose:
            print(f"  frame {fn}: {frame_polys_n} polygons ({len(plume_info)} plumes)")
        elif i % 25 == 0:
            elapsed = time.perf_counter() - t0
            rate = i / elapsed if elapsed else 0
            eta = (len(frames) - i) / rate if rate else 0
            print(f"  {i}/{len(frames)} frames | {run_stats.raw_polygons} raw polys | {rate:.1f} fps | eta {eta:.0f}s")

    run_stats.elapsed_s = time.perf_counter() - t0

    if args.save_raw:
        save_raw_polys_geojson(raw_geojson_path, raw_polys, raw_records)

    return _merge_and_write(raw_polys, raw_records, run_stats, args, out_base)


def _merge_and_write(raw_polys, raw_records, run_stats, args, out_base):
    if not raw_polys:
        print("No polygons produced — nothing to merge.")
        return run_stats

    if args.cluster_method == "density_grid":
        explicit_min = args.min_observations if args.min_observations is not None else "auto"
        print(
            f"\nClustering {len(raw_polys)} raw polygons by density grid "
            f"(grid {args.grid_resolution_m} m, min_observations={explicit_min}, "
            f"raw-area cap {args.max_raw_area_m2} m²)…"
        )
        # Pull mean_anomaly_c from each raw record (saved by save_raw_polys_geojson)
        # so the cluster step can compute per-merged-polygon temperature stats.
        polygon_anomalies = [float(r.get("mean_anomaly_c", 0.0) or 0.0) for r in raw_records]
        density_sites, density_diag = cluster_polygons_by_density_grid(
            raw_polys,
            grid_resolution_m=args.grid_resolution_m,
            min_observations=args.min_observations,
            max_raw_area_m2=args.max_raw_area_m2,
            min_observations_fraction=args.min_observations_fraction,
            max_site_diameter_m=args.max_site_diameter_m if args.detector == "spread" else None,
            local_adaptive_window_m=args.local_adaptive_window_m,
            local_adaptive_fraction=args.local_adaptive_fraction,
            polygon_anomalies=polygon_anomalies,
        )
        args.min_observations = density_diag["min_observations_used"]
        run_stats.failures.append(("merge_diag", str(density_diag)))
        merged_info = []
        for idx, (poly, peak_count, info) in enumerate(density_sites):
            if not isinstance(poly, Polygon) or poly.is_empty:
                continue
            poly = orient(poly, sign=1.0)
            area_m2 = compute_polygon_area_m2(poly)
            if area_m2 < args.min_site_area_m2:
                continue
            cx, cy = poly.centroid.x, poly.centroid.y
            tier_name, _ = area_tier(area_m2)
            mean_anom = float(info.get("mean_anomaly_c", 0.0))
            min_anom = float(info.get("min_anomaly_c", 0.0))
            # intensity_index = area × strength.
            # Combines spatial scale and per-cell coldness deviation; the
            # primary value for cross-flight ranking. Higher = bigger and/or
            # colder relative to that flight's ambient ocean.
            intensity_index = area_m2 * mean_anom
            merged_info.append(
                {
                    "id": idx,
                    "area_m2": area_m2,
                    "centroid_lon": float(cx),
                    "centroid_lat": float(cy),
                    "tier": tier_name,
                    "polygon": poly,
                    "n_observations": peak_count,
                    "mean_anomaly_c": mean_anom,
                    "min_anomaly_c": min_anom,
                    "p90_anomaly_c": float(info.get("p90_anomaly_c", 0.0)),
                    "intensity_index": intensity_index,
                }
            )
        merged_info.sort(key=lambda r: r["area_m2"], reverse=True)
        for rank, r in enumerate(merged_info):
            r["id"] = rank
        print(f"  density_grid: {len(merged_info)} sites from {len(raw_polys)} raw polys")
        run_stats.merged_polygons = len(merged_info)
        run_stats.total_area_m2 = sum(r["area_m2"] for r in merged_info)

        header_label = f"SGD stretch — {out_base.name}"
        write_kml(out_base.with_suffix(".kml"), merged_info, header=header_label)
        write_geojson(out_base.with_suffix(".geojson"), merged_info)
        write_csv(out_base.with_suffix(".csv"), merged_info)
        write_summary(out_base.parent / (out_base.name + "_summary.json"), run_stats, merged_info)
        return run_stats

    print(
        f"\nClustering {len(raw_polys)} raw polygons by discharge site "
        f"(cluster radius {args.cluster_distance_m} m, raw-area cap "
        f"{args.max_raw_area_m2} m², iou_merge={args.use_iou_merge})…"
    )
    clusters = cluster_polygons_by_site(
        raw_polys,
        cluster_distance_m=args.cluster_distance_m,
        max_raw_area_m2=args.max_raw_area_m2,
        use_iou_merge=args.use_iou_merge,
    )

    merged_info = []
    dropped_raw_count = len(raw_polys) - sum(len(c) for c in clusters)
    dropped_oversize = 0
    for idx, member_indices in enumerate(clusters):
        members = [raw_polys[i] for i in member_indices]
        merged = unary_union(members)
        if merged.is_empty:
            continue
        if isinstance(merged, MultiPolygon):
            pieces = sorted(merged.geoms, key=lambda p: p.area, reverse=True)
            merged = pieces[0]
        if not isinstance(merged, Polygon):
            continue
        merged = orient(merged, sign=1.0)
        area_m2 = compute_polygon_area_m2(merged)
        if area_m2 < args.min_site_area_m2:
            continue
        if len(member_indices) < args.min_observations:
            continue
        # Bounding-box diagonal in meters — catches transitive-chain clusters
        # that grew via A-B, B-C, ... centroid links spanning tens of meters
        # even though no two distinct discharge sources should cluster together.
        bx0, by0, bx1, by1 = merged.bounds
        cy = 0.5 * (by0 + by1)
        mpd_lat = 111320.0
        mpd_lon = 111320.0 * np.cos(np.radians(cy))
        diag_m = float(
            np.sqrt(((bx1 - bx0) * mpd_lon) ** 2 + ((by1 - by0) * mpd_lat) ** 2)
        )
        if diag_m > args.max_site_diameter_m:
            dropped_oversize += 1
            continue
        cx, cy = merged.centroid.x, merged.centroid.y
        tier_name, _ = area_tier(area_m2)
        merged_info.append(
            {
                "id": idx,
                "area_m2": area_m2,
                "centroid_lon": float(cx),
                "centroid_lat": float(cy),
                "tier": tier_name,
                "polygon": merged,
                "n_observations": len(member_indices),
            }
        )
    merged_info.sort(key=lambda r: r["area_m2"], reverse=True)
    for rank, r in enumerate(merged_info):
        r["id"] = rank
    print(f"  dropped {dropped_raw_count} raw polygons over {args.max_raw_area_m2} m² cap")
    print(f"  dropped {dropped_oversize} merged clusters over {args.max_site_diameter_m} m diameter (chain artifacts)")

    run_stats.merged_polygons = len(merged_info)
    run_stats.total_area_m2 = sum(r["area_m2"] for r in merged_info)

    # --- Write outputs ---
    header_label = f"SGD stretch — {out_base.name}"
    write_kml(out_base.with_suffix(".kml"), merged_info, header=header_label)
    write_geojson(out_base.with_suffix(".geojson"), merged_info)
    write_csv(out_base.with_suffix(".csv"), merged_info)
    write_summary(out_base.parent / (out_base.name + "_summary.json"), run_stats, merged_info)

    return run_stats


def write_kml(path: Path, merged_info: list[dict], header: str):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f"<name>{header}</name>",
        f"<description>Merged SGD plumes across stretch — {len(merged_info)} polygons, "
        f"{sum(r['area_m2'] for r in merged_info):.1f} m² total</description>",
    ]

    # Styles per tier
    for _, tier_name, color in AREA_TIERS:
        lines.append(
            f"""
<Style id="sgd_{tier_name}">
  <LineStyle><color>ff000000</color><width>1.2</width></LineStyle>
  <PolyStyle><color>{color}</color><fill>1</fill><outline>1</outline></PolyStyle>
</Style>""".strip()
        )

    for r in merged_info:
        poly = r["polygon"]
        coords = " ".join(f"{x:.8f},{y:.8f},0" for x, y in poly.exterior.coords)
        inner_lines = []
        for interior in poly.interiors:
            hole_coords = " ".join(f"{x:.8f},{y:.8f},0" for x, y in interior.coords)
            inner_lines.append(
                f"<innerBoundaryIs><LinearRing><coordinates>{hole_coords}</coordinates></LinearRing></innerBoundaryIs>"
            )
        inner_xml = "\n          ".join(inner_lines)
        intensity = r.get("intensity_index", 0.0)
        mean_anom = r.get("mean_anomaly_c", 0.0)
        min_anom = r.get("min_anomaly_c", 0.0)
        n_obs = r.get("n_observations", 1)
        lines.append(
            f"""
<Placemark>
  <name>SGD #{r['id']} — {r['area_m2']:.1f} m² · ΔT {mean_anom:.2f}°C</name>
  <description><![CDATA[
    <b>Area:</b> {r['area_m2']:.2f} m²<br/>
    <b>Mean anomaly:</b> {mean_anom:.3f} °C below ambient<br/>
    <b>Coldest cell:</b> {min_anom:.3f} °C below ambient<br/>
    <b>Intensity index (area × ΔT):</b> {intensity:.1f}<br/>
    <b>Observation count:</b> {n_obs} frames<br/>
    <b>Area tier:</b> {r['tier']}<br/>
    <b>Centroid:</b> {r['centroid_lat']:.6f}, {r['centroid_lon']:.6f}
  ]]></description>
  <styleUrl>#sgd_{r['tier']}</styleUrl>
  <Polygon>
    <outerBoundaryIs>
      <LinearRing>
        <coordinates>{coords}</coordinates>
      </LinearRing>
    </outerBoundaryIs>
    {inner_xml}
  </Polygon>
</Placemark>""".strip()
        )
    lines.append("</Document></kml>")
    path.write_text("\n".join(lines))
    print(f"  wrote {path}")


def write_geojson(path: Path, merged_info: list[dict]):
    features = []
    for r in merged_info:
        poly = r["polygon"]
        ext = [list(pt) for pt in poly.exterior.coords]
        holes = [[list(pt) for pt in ring.coords] for ring in poly.interiors]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ext] + holes},
                "properties": {
                    "id": r["id"],
                    "area_m2": r["area_m2"],
                    "tier": r["tier"],
                    "centroid_lon": r["centroid_lon"],
                    "centroid_lat": r["centroid_lat"],
                    "n_observations": r.get("n_observations", 1),
                    "mean_anomaly_c": r.get("mean_anomaly_c", 0.0),
                    "min_anomaly_c": r.get("min_anomaly_c", 0.0),
                    "p90_anomaly_c": r.get("p90_anomaly_c", 0.0),
                    "intensity_index": r.get("intensity_index", 0.0),
                },
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, indent=2))
    print(f"  wrote {path}")


def write_csv(path: Path, merged_info: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "area_m2", "tier", "centroid_lat", "centroid_lon", "n_observations",
             "mean_anomaly_c", "min_anomaly_c", "p90_anomaly_c", "intensity_index"]
        )
        for r in merged_info:
            w.writerow(
                [
                    r["id"],
                    f"{r['area_m2']:.3f}",
                    r["tier"],
                    f"{r['centroid_lat']:.7f}",
                    f"{r['centroid_lon']:.7f}",
                    r.get("n_observations", 1),
                    f"{r.get('mean_anomaly_c', 0.0):.4f}",
                    f"{r.get('min_anomaly_c', 0.0):.4f}",
                    f"{r.get('p90_anomaly_c', 0.0):.4f}",
                    f"{r.get('intensity_index', 0.0):.2f}",
                ]
            )
    print(f"  wrote {path}")


def write_summary(path: Path, run_stats: DetectionRun, merged_info: list[dict]):
    areas = np.array([r["area_m2"] for r in merged_info], dtype=np.float64) if merged_info else np.array([0.0])
    tier_counts: dict[str, int] = {}
    tier_area: dict[str, float] = {}
    for r in merged_info:
        tier_counts[r["tier"]] = tier_counts.get(r["tier"], 0) + 1
        tier_area[r["tier"]] = tier_area.get(r["tier"], 0.0) + r["area_m2"]
    summary = {
        "frames_attempted": run_stats.frames_attempted,
        "frames_with_detections": run_stats.frames_with_detections,
        "frames_failed": run_stats.frames_failed,
        "raw_polygons": run_stats.raw_polygons,
        "merged_polygons": run_stats.merged_polygons,
        "total_area_m2": run_stats.total_area_m2,
        "elapsed_s": run_stats.elapsed_s,
        "area_percentiles": {
            "p10": float(np.percentile(areas, 10)),
            "p50": float(np.percentile(areas, 50)),
            "p90": float(np.percentile(areas, 90)),
            "p99": float(np.percentile(areas, 99)),
            "max": float(areas.max()),
        },
        "tier_counts": tier_counts,
        "tier_area_m2": tier_area,
        "failures": run_stats.failures[:20],
    }
    path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="Directory with MAX_*.JPG / IRX_*.irg pairs (omit with --merge-only)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--step", type=int, default=1, help="Frame stride (1 = every frame)")
    ap.add_argument("--output", required=True, help="Output base path (extensions added)")
    ap.add_argument("--save-raw", action="store_true", help="Save raw per-frame polygons to <output>_raw.geojson so --merge-only can reuse them later.")
    ap.add_argument("--merge-only", action="store_true", help="Skip detection; load raw polygons from <output>_raw.geojson and re-cluster with current args.")
    ap.add_argument("--use-iou-merge", action="store_true", help="Also merge raw polygons with IoU>=0.15 (off by default — tends to chain dense detections).")
    ap.add_argument(
        "--detector",
        choices=["spread", "redesigned"],
        default="spread",
        help="spread (default): subtle continuous cold-lens detector for bay/lagoon SGD. "
        "redesigned: coast-emerging point-source detector for discrete plumes on rocky coast.",
    )
    ap.add_argument("--temp-threshold", type=float, default=0.5, help="Redesigned detector only")
    ap.add_argument("--delta-c", type=float, default=0.25, help="Spread detector: °C below warm baseline")
    ap.add_argument("--min-area", type=int, default=400, help="Minimum plume size in PIXELS")
    ap.add_argument(
        "--cluster-distance-m",
        type=float,
        default=8.0,
        help="Merge raw polygon observations whose centroids are within this many meters (same discharge site).",
    )
    ap.add_argument(
        "--max-raw-area-m2",
        type=float,
        default=None,
        help="Drop raw single-frame polygons larger than this (m²). Default depends "
        "on detector: 500 for 'redesigned' (kills baseline-collapse FPs), 100000 "
        "for 'spread' (large polygons ARE the signal). Pass a value to override.",
    )
    ap.add_argument(
        "--min-site-area-m2",
        type=float,
        default=0.5,
        help="Drop final merged sites smaller than this (noise).",
    )
    ap.add_argument(
        "--min-observations",
        type=int,
        default=None,
        help="Minimum frame observations per grid cell for density_grid merge. "
        "If unset on --detector spread, AUTO-TUNES to ~50%% of the peak per-cell "
        "observation count (controlled by --min-observations-fraction). "
        "On --detector redesigned, defaults to 3.",
    )
    ap.add_argument(
        "--min-observations-fraction",
        type=float,
        default=0.5,
        help="When --min-observations is auto-tuned (spread detector default), "
        "set min_observations to fraction × peak per-cell observation count. "
        "Default 0.5 — moderately permissive, includes nearshore features at "
        "the eastern/western tips of long-coast surveys where coverage is "
        "thinner. Raise to 0.65-0.75 for stricter 'only the obvious bays'.",
    )
    ap.add_argument(
        "--local-adaptive-window-m",
        type=float,
        default=None,
        help="If set, threshold each grid cell against the local max observation "
        "count in a window of this many meters (instead of the global max). "
        "Solves the 'long-coast survey' problem: eastern tip of a flight with "
        "10 overlaps gets judged against ITS peak, not the harbor's peak of 85. "
        "Try 100-200 m for typical drone surveys. Default OFF (uses global).",
    )
    ap.add_argument(
        "--local-adaptive-fraction",
        type=float,
        default=0.9,
        help="When --local-adaptive-window-m is set, keep cells whose count is ≥ "
        "this fraction of the local-window peak. 0.5 keeps cells observed at "
        "least half as much as the local hot spot.",
    )
    ap.add_argument(
        "--max-site-diameter-m",
        type=float,
        default=None,
        help="Cap merged sites at this size (bounding-box diagonal). Default depends "
        "on detector: 15 m for 'redesigned' (drops chains in centroid mode); 80 m "
        "for 'spread' (watershed-splits chained bay regions into separate per-bay "
        "polygons). Real single bay spreads rarely exceed 80 m; longer regions "
        "are almost always multi-bay chains.",
    )
    ap.add_argument(
        "--cluster-method",
        choices=["density_grid", "centroid"],
        default="density_grid",
        help="density_grid (default): rasterize raw polygons to a fixed metric "
        "grid and threshold by observation count; chains are mathematically "
        "impossible. centroid: the older union-find centroid clustering.",
    )
    ap.add_argument(
        "--grid-resolution-m",
        type=float,
        default=1.0,
        help="Grid cell size for density_grid clustering in meters. 1 m is "
        "a reasonable compromise between GPS precision and plume resolution.",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    # Detector-appropriate defaults for the merge stage.
    if args.max_raw_area_m2 is None:
        args.max_raw_area_m2 = 100000.0 if args.detector == "spread" else 500.0
    # spread detector auto-tunes min_observations from peak count in the density
    # grid (set inside cluster_polygons_by_density_grid). Only seed an explicit
    # default for the redesigned detector, where peak counts are typically tiny.
    if args.min_observations is None and args.detector == "redesigned":
        args.min_observations = 3
    # Default min_site_area floor for spread — drops grid-cell noise at the edges
    # of merged regions without needing a separate flag.
    if args.detector == "spread" and args.min_site_area_m2 == 0.5:
        args.min_site_area_m2 = 20.0
    # Default max-site-diameter: large for spread (don't tile-split bay-scale
    # continuous spreads into many fragments — that creates the "everything is
    # SGD" tiled visualization). 200m means natural bays stay as single polygons.
    if args.max_site_diameter_m is None:
        args.max_site_diameter_m = 200.0 if args.detector == "spread" else 15.0
    # Default local-adaptive window for spread detector — 200 m. Scales the
    # threshold with local frame coverage so eastern/western tips of a long
    # survey aren't penalized for having less overlap than the focal bay.
    if args.local_adaptive_window_m is None and args.detector == "spread":
        args.local_adaptive_window_m = 200.0
    run_stats = run(args)
    print("\n=== Summary ===")
    print(f"  frames attempted:        {run_stats.frames_attempted}")
    print(f"  frames with detections:  {run_stats.frames_with_detections}")
    print(f"  frames failed:           {run_stats.frames_failed}")
    print(f"  raw polygons:            {run_stats.raw_polygons}")
    print(f"  merged polygons:         {run_stats.merged_polygons}")
    print(f"  total merged area:       {run_stats.total_area_m2:.1f} m²")
    print(f"  elapsed:                 {run_stats.elapsed_s:.1f} s")


if __name__ == "__main__":
    main()
