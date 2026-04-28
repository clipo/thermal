#!/usr/bin/env python3
"""Authoritative water mask from OpenStreetMap coastline data.

Replaces the HSV-based satellite classifier (`derive_water_mask.py`)
with hand-mapped OSM coastline. OSM is:
  - Tide-independent (coastline = mean high water, not when satellite
    happened to capture the tile)
  - Hand-validated by the OSM community for well-mapped islands like
    Rapa Nui
  - Free of the HSV-classifier failure modes (sandy bays
    misclassified as land, gray rocks classified as water)
  - Reusable: fetch once, rasterize per flight

Workflow:
  1. Fetch Rapa Nui coastline ways from OSM via the Overpass API
     (one-time, cached at sgd_output/osm_coastline.json)
  2. Build a Shapely Polygon representing the island (land)
  3. For each flight, rasterize that polygon to the flight's grid
     (water = NOT inside the land polygon)
  4. Save to <slug>_water_mask.npz with `method='osm'`

Usage:
    python scripts/derive_water_mask_osm.py --all
    python scripts/derive_water_mask_osm.py --slug june2023_2_july_23_poike_3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"
COASTLINE_CACHE = SGD_OUTPUT / "osm_coastline.json"


# Rapa Nui bounding box (slightly padded)
RAPA_NUI_BBOX = {
    "south": -27.22,
    "west": -109.50,
    "north": -27.04,
    "east": -109.20,
}


def fetch_osm_coastline() -> list[list[tuple[float, float]]]:
    """Fetch coastline ways from OpenStreetMap Overpass API.
    Returns list of polylines, each a list of (lon, lat) points."""
    if COASTLINE_CACHE.exists():
        try:
            data = json.loads(COASTLINE_CACHE.read_text())
            print(f"  loaded cached OSM coastline: {len(data)} ways "
                  f"({COASTLINE_CACHE})")
            return [[(p[0], p[1]) for p in line] for line in data]
        except Exception:
            pass

    import urllib.request
    print(f"  fetching OSM coastline from Overpass API…")
    bbox = RAPA_NUI_BBOX
    query = (
        '[out:json][timeout:60];'
        f'way["natural"="coastline"]'
        f'({bbox["south"]},{bbox["west"]},{bbox["north"]},{bbox["east"]});'
        '(._;>;);'
        'out body;'
    )
    url = "https://overpass-api.de/api/interpreter"
    req = urllib.request.Request(
        url, data=query.encode("utf-8"),
        headers={"User-Agent": "thermal-sgd-mapper/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    nodes = {n["id"]: (n["lon"], n["lat"])
             for n in result["elements"] if n["type"] == "node"}
    ways = []
    for el in result["elements"]:
        if el["type"] != "way":
            continue
        line = [nodes[nid] for nid in el["nodes"] if nid in nodes]
        if len(line) >= 2:
            ways.append(line)

    print(f"  fetched {len(ways)} coastline ways "
          f"({sum(len(w) for w in ways)} total nodes)")
    COASTLINE_CACHE.write_text(json.dumps(ways))
    return ways


def build_land_polygon(ways: list[list[tuple[float, float]]]):
    """Stitch coastline ways into closed polygons representing land."""
    from shapely.geometry import LineString, Polygon, MultiPolygon
    from shapely.ops import unary_union, polygonize

    lines = [LineString(w) for w in ways if len(w) >= 2]
    merged = unary_union(lines)
    polys = list(polygonize(merged))
    if not polys:
        # Fallback: stitch into closed rings manually
        # OSM's coastline convention: land is to the RIGHT of the way
        # direction. Closed ways are full islands. For Rapa Nui which
        # is one mostly-closed coastline, attempt to close gaps.
        from shapely.geometry import Polygon as Poly
        rings = []
        for w in ways:
            if len(w) >= 4:
                if w[0] != w[-1]:
                    w = w + [w[0]]
                rings.append(Poly(w))
        polys = rings
    return MultiPolygon([p for p in polys if p.is_valid and p.area > 0])


def rasterize_land_for_flight(slug: str, land_geom) -> dict:
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    out_path = spread / f"{slug}_water_mask.npz"
    if not npz_path.exists():
        return {"slug": slug, "error": "no anomaly raster"}

    raster = np.load(npz_path)
    minlon = float(raster["bbox_min_lon"])
    maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"])
    maxlat = float(raster["bbox_max_lat"])
    grid_res = float(raster["grid_resolution_m"])
    gy, gx = raster["anomaly"].shape
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))

    from shapely.geometry import box, MultiPolygon, Polygon
    from matplotlib.path import Path as MplPath

    flight_bbox = box(minlon, minlat, maxlon, maxlat)
    land_in_bbox = land_geom.intersection(flight_bbox)

    # Cell-center lat/lon grid (vectorized)
    col_idx = np.arange(gx, dtype=np.float64) + 0.5
    row_idx = np.arange(gy, dtype=np.float64) + 0.5
    lon_grid, lat_grid = np.meshgrid(
        minlon + col_idx * grid_res / mpd_lon,
        minlat + row_idx * grid_res / mpd_lat,
    )
    pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])

    is_land = np.zeros((gy, gx), dtype=bool)

    if not land_in_bbox.is_empty:
        polys = []
        if isinstance(land_in_bbox, (MultiPolygon,)):
            polys = list(land_in_bbox.geoms)
        elif isinstance(land_in_bbox, Polygon):
            polys = [land_in_bbox]
        else:
            try:
                polys = list(land_in_bbox.geoms)
            except AttributeError:
                polys = [land_in_bbox]

        for poly in polys:
            # Outer ring containment (positive)
            ext = list(poly.exterior.coords)
            inside = MplPath(ext).contains_points(pts).reshape(gy, gx)
            # Subtract holes
            for hole in poly.interiors:
                hole_inside = MplPath(list(hole.coords)).contains_points(pts).reshape(gy, gx)
                inside = inside & ~hole_inside
            is_land |= inside

    is_water = ~is_land

    np.savez_compressed(
        out_path,
        is_water=is_water,
        bbox_min_lon=minlon, bbox_max_lon=maxlon,
        bbox_min_lat=minlat, bbox_max_lat=maxlat,
        grid_resolution_m=grid_res,
        method="osm_coastline",
    )
    return {
        "slug": slug, "out": str(out_path),
        "water_cells": int(is_water.sum()),
        "total_cells": int(is_water.size),
        "water_frac": float(is_water.sum()) / float(is_water.size),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()

    print("Fetching OSM coastline…")
    ways = fetch_osm_coastline()
    print("Building land polygon…")
    land_geom = build_land_polygon(ways)
    print(f"  land area: {land_geom.area * (111000**2):.0f} m² (rough)")

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for sd in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = sd.name[: -len("_spread")]
            if (sd / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"\nRasterizing OSM-derived water mask for {len(slugs)} flight(s)…")
    for slug in slugs:
        try:
            r = rasterize_land_for_flight(slug, land_geom)
            if "error" in r:
                print(f"  ✗ {slug}: {r['error']}")
            else:
                print(f"  ✓ {slug}: {100*r['water_frac']:.1f}% water "
                      f"({r['water_cells']:,} cells)")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
