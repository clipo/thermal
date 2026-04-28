#!/usr/bin/env python3
"""Resample a coastline polyline at fixed metric intervals (e.g. every 10 m
along-shore) and emit the sampled points as CSV or GeoJSON Points. The output
is the input format for ``sgd_proximity.py`` to compute Σ_anomaly per
along-shore segment.

Input:
  - GeoJSON LineString or MultiLineString
  - KML LineString (basic — extracts <coordinates> blocks under <Placemark>)
  - CSV with lat/lon columns describing the polyline in order

Sampling interval is in meters. The first vertex is always emitted; subsequent
points are placed every `--interval-m` meters along the cumulative arc length.

Usage:
    python scripts/sample_coastline.py \\
        --coastline data/rapa_nui_coast.geojson \\
        --interval-m 10 \\
        --output data/rapa_nui_coast_10m_points.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


def load_polylines(path: Path) -> list[list[tuple[float, float]]]:
    """Return list of polylines; each polyline is a list of (lon, lat)."""
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        fc = json.loads(path.read_text())
        out = []
        if fc.get("type") == "FeatureCollection":
            iterator = (f.get("geometry") or {} for f in fc["features"])
        elif fc.get("type") == "Feature":
            iterator = [fc.get("geometry") or {}]
        else:
            iterator = [fc]
        for geom in iterator:
            t = geom.get("type")
            if t == "LineString":
                out.append([(c[0], c[1]) for c in geom["coordinates"]])
            elif t == "MultiLineString":
                for line in geom["coordinates"]:
                    out.append([(c[0], c[1]) for c in line])
        return out
    if suffix == ".kml":
        text = path.read_text()
        # Crude extraction; matches all <coordinates>…</coordinates> blocks
        coords_blocks = re.findall(r"<coordinates>(.*?)</coordinates>", text, re.DOTALL)
        out = []
        for block in coords_blocks:
            pts = []
            for token in block.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        pts.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
            if len(pts) >= 2:
                out.append(pts)
        return out
    if suffix == ".csv":
        out = []
        with path.open() as f:
            reader = csv.DictReader(f)
            line = []
            for row in reader:
                lat = _find(row, ("lat", "latitude", "y"))
                lon = _find(row, ("lon", "lng", "longitude", "x"))
                if lat is None or lon is None:
                    raise SystemExit(f"CSV needs lat/lon: got {list(row.keys())}")
                line.append((float(lon), float(lat)))
            out.append(line)
        return out
    raise SystemExit(f"unsupported format: {suffix}")


def _find(row: dict, names: tuple[str, ...]) -> str | None:
    for n in names:
        for k in row.keys():
            if k.lower() == n:
                return row[k]
    return None


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371008.8
    p1 = math.radians(lat1); p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def sample_polyline(line: list[tuple[float, float]], interval_m: float,
                    line_id: int) -> list[dict]:
    """Resample at fixed metric interval. Returns dicts with seg_id, line_id, lat, lon, distance_m."""
    if len(line) < 2:
        return []
    out = []
    cumulative = 0.0
    target = 0.0
    seg_id = 0
    # Always emit the first vertex
    out.append({
        "line_id": line_id, "seg_id": seg_id,
        "lon": line[0][0], "lat": line[0][1],
        "distance_m": 0.0,
    })
    seg_id += 1
    target += interval_m
    for i in range(len(line) - 1):
        lon1, lat1 = line[i]
        lon2, lat2 = line[i + 1]
        d = haversine_m(lat1, lon1, lat2, lon2)
        if d == 0:
            continue
        next_cum = cumulative + d
        # Place sample points at every interval crossing within this edge
        while target <= next_cum:
            t = (target - cumulative) / d
            lon = lon1 + t * (lon2 - lon1)
            lat = lat1 + t * (lat2 - lat1)
            out.append({
                "line_id": line_id, "seg_id": seg_id,
                "lon": lon, "lat": lat,
                "distance_m": target,
            })
            seg_id += 1
            target += interval_m
        cumulative = next_cum
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coastline", required=True)
    ap.add_argument("--interval-m", type=float, default=10.0)
    ap.add_argument("--output", required=True, help="output .csv or .geojson")
    args = ap.parse_args()

    lines = load_polylines(Path(args.coastline))
    print(f"Loaded {len(lines)} polylines, total {sum(len(l) for l in lines)} vertices.")

    rows = []
    for i, line in enumerate(lines):
        rows.extend(sample_polyline(line, args.interval_m, line_id=i))
    print(f"Resampled to {len(rows)} points at {args.interval_m} m intervals.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["line_id", "seg_id", "lat", "lon", "distance_m"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
    elif out.suffix.lower() in (".geojson", ".json"):
        feats = [{
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {k: r[k] for k in ("line_id", "seg_id", "distance_m")},
        } for r in rows]
        out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=2))
    else:
        raise SystemExit(f"unsupported output format: {out.suffix}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
