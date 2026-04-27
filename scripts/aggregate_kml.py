#!/usr/bin/env python3
"""Combine all per-flight `*_sgd.kml` files into one master KML.

Each input flight's polygons go into their own toggleable Folder so you can
turn flights on/off in Google Earth's left panel. Styles (area tiers) are
defined once at the top and reused across all polygons.

Usage:
    python scripts/aggregate_kml.py \\
        --output sgd_output/rapa_nui_all_flights_sgd.kml \\
        sgd_output/flight*_spread/flight*_sgd.kml \\
        sgd_output/vaihu_*_spread/vaihu_*_sgd.kml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Same tier breakpoints as run_coast_stretch.py
TIERS = [
    (2.0, "small", "ff00ffff"),       # yellow
    (10.0, "modest", "ff0080ff"),      # orange
    (50.0, "moderate", "ff0000ff"),    # red
    (200.0, "large", "ffff00ff"),      # magenta
    (float("inf"), "very_large", "ffff00a0"),  # purple
]


def style_xml() -> str:
    out = []
    for _, tier_name, color in TIERS:
        out.append(
            f"""<Style id="sgd_{tier_name}">
  <LineStyle><color>ff000000</color><width>1.2</width></LineStyle>
  <PolyStyle><color>{color}</color><fill>1</fill><outline>1</outline></PolyStyle>
</Style>"""
        )
    return "\n".join(out)


def extract_placemarks(kml_path: Path) -> tuple[str, list[str], dict]:
    """Return (flight_label, list of <Placemark>...</Placemark> strings, tier_counts)."""
    text = kml_path.read_text()
    placemarks = re.findall(r"<Placemark>.*?</Placemark>", text, flags=re.DOTALL)
    # Use the corresponding GeoJSON to grab area + tier counts (more reliable)
    geojson_path = kml_path.with_suffix(".geojson")
    tier_counts: dict[str, int] = {t: 0 for _, t, _ in TIERS}
    total_area = 0.0
    if geojson_path.exists():
        try:
            fc = json.loads(geojson_path.read_text())
            for feat in fc["features"]:
                tier = feat["properties"].get("tier", "small")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                total_area += float(feat["properties"].get("area_m2", 0))
        except Exception:
            pass
    flight_label = kml_path.stem
    return flight_label, placemarks, {"counts": tier_counts, "total_area_m2": total_area}


def aggregate(kml_paths: list[Path], output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>')
    parts.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    parts.append("<Document>")
    parts.append("<name>Rapa Nui SGD — all flights</name>")
    parts.append(style_xml())

    grand_sites = 0
    grand_area = 0.0
    summary_rows = []

    for kml_path in kml_paths:
        if not kml_path.exists():
            print(f"  skip (missing): {kml_path}")
            continue
        flight_label, placemarks, meta = extract_placemarks(kml_path)
        n = len(placemarks)
        if n == 0:
            print(f"  skip (no placemarks): {kml_path}")
            continue
        area = meta["total_area_m2"]
        grand_sites += n
        grand_area += area
        summary_rows.append((flight_label, n, area))
        parts.append(f"<Folder><name>{flight_label} ({n} sites, {area:.0f} m²)</name>")
        parts.extend(placemarks)
        parts.append("</Folder>")
        print(f"  added {flight_label}: {n} sites, {area:.0f} m²")

    parts.append("<Folder><name>Summary</name><description><![CDATA[")
    parts.append(f"<b>{grand_sites} SGD sites</b> across {len(summary_rows)} flights<br/>")
    parts.append(f"<b>{grand_area:.0f} m² (≈{grand_area/10000:.1f} ha)</b> total extent<br/><br/>")
    for label, n, area in summary_rows:
        parts.append(f"{label}: {n} sites, {area:.0f} m²<br/>")
    parts.append("]]></description></Folder>")
    parts.append("</Document></kml>")

    output.write_text("\n".join(parts))
    print(f"\nWrote {output}")
    print(f"  {grand_sites} sites, {grand_area:.0f} m² total")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="Path to write the combined KML")
    ap.add_argument("kmls", nargs="+", help="Input per-flight KML paths (or globs already expanded by shell)")
    args = ap.parse_args()
    paths = [Path(p) for p in args.kmls]
    aggregate(paths, Path(args.output))


if __name__ == "__main__":
    main()
