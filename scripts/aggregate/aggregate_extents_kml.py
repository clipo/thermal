#!/usr/bin/env python3
"""Combine all per-flight `*_extent_merged.kml` files into one master extents KML.

Input: list of merged-extent KML files (one polygon per flight).
Output: a single KML with one Folder per flight, color-coded by season group.
Open in Google Earth to see where every flight flew, season by season.

Usage:
    python scripts/aggregate_extents_kml.py \\
        --output sgd_output/rapa_nui_all_flight_extents.kml \\
        sgd_output/*_extents/*_extent_merged.kml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# KML colors (AABBGGRR)
SEASON_STYLES = {
    "jan2024": ("ff8033ff", "Jan 2024 (blue)"),       # blue
    "june2023": ("ff66ff66", "June 2023 (green)"),    # green
    "other": ("ff00ffff", "Other (yellow)"),           # yellow
}


def style_xml() -> str:
    out = []
    for season, (color, _) in SEASON_STYLES.items():
        out.append(
            f"""<Style id="ext_{season}">
  <LineStyle><color>ff000000</color><width>2.0</width></LineStyle>
  <PolyStyle><color>{color}</color><fill>1</fill><outline>1</outline></PolyStyle>
</Style>"""
        )
    return "\n".join(out)


def detect_season(slug: str) -> str:
    s = slug.lower()
    if "june2023" in s:
        return "june2023"
    if s.startswith("flight"):
        return "jan2024"
    return "other"


def extract_polygon_coords(kml_text: str) -> list[str]:
    """Pull <coordinates>...</coordinates> blocks from a KML file."""
    return re.findall(r"<coordinates>([^<]+)</coordinates>", kml_text, flags=re.DOTALL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("kmls", nargs="+", help="Input *_extent_merged.kml files")
    args = ap.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        '<Document>',
        '<name>Rapa Nui — flight ground extents (all seasons)</name>',
        style_xml(),
    ]

    by_season: dict[str, list[tuple[str, str, list[str]]]] = {k: [] for k in SEASON_STYLES}

    for kml_path in args.kmls:
        p = Path(kml_path)
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        text = p.read_text()
        # slug = parent dir minus _extents suffix
        slug = p.parent.name
        if slug.endswith("_extents"):
            slug = slug[: -len("_extents")]
        season = detect_season(slug)
        coords_blocks = extract_polygon_coords(text)
        if not coords_blocks:
            print(f"  skip (no polygon): {p}")
            continue
        by_season[season].append((slug, str(p), coords_blocks))

    grand_total = 0
    for season, entries in by_season.items():
        if not entries:
            continue
        _, label = SEASON_STYLES[season]
        parts.append(f'<Folder><name>{label} — {len(entries)} flights</name>')
        for slug, _, coords_blocks in entries:
            parts.append(f'<Folder><name>{slug}</name>')
            for coords in coords_blocks:
                coords_clean = " ".join(coords.split())
                parts.append(f"""<Placemark>
  <name>{slug}</name>
  <styleUrl>#ext_{season}</styleUrl>
  <Polygon>
    <outerBoundaryIs><LinearRing>
      <coordinates>{coords_clean}</coordinates>
    </LinearRing></outerBoundaryIs>
  </Polygon>
</Placemark>""")
            parts.append('</Folder>')
            grand_total += 1
        parts.append('</Folder>')

    parts.append('</Document></kml>')
    output.write_text("\n".join(parts))
    print(f"\nWrote {output}  ({grand_total} flight extent polygons)")
    for season, entries in by_season.items():
        if entries:
            print(f"  {season}: {len(entries)} flights")


if __name__ == "__main__":
    main()
