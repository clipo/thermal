#!/usr/bin/env python3
"""Combine all per-flight `*_sgd.geojson` files into a master KML colored by
**intensity** (area × mean_anomaly) rather than area alone.

Use this for cross-flight / cross-season comparisons where the goal is to
rank polygons by SGD strength regardless of which flight they came from.
Each polygon's popup includes:

  - Area (m²)
  - Mean anomaly (°C below ambient)
  - Coldest cell anomaly
  - Intensity index (m² · °C)
  - Observation count
  - Source flight slug

Color tiers are based on intensity_index quantiles computed across the
ENTIRE input set so the legend is consistent across flights.

Usage:
    python scripts/aggregate_intensity_kml.py \\
        --output sgd_output/rapa_nui_all_sgd_intensity.kml \\
        sgd_output/flight*_spread/*_sgd.geojson \\
        sgd_output/june2023_*_spread/*_sgd.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Six-tier intensity ramp from cool blue (low) to hot red (high). KML uses
# AABBGGRR, so e.g. ff0000ff = opaque red (red=ff, green=00, blue=00).
INTENSITY_COLORS = [
    ("very_low",  "ffffe080"),  # pale cyan
    ("low",       "ffff8040"),  # cyan-blue
    ("moderate",  "ff80ff00"),  # green
    ("high",      "ff00ffff"),  # yellow
    ("very_high", "ff0080ff"),  # orange
    ("extreme",   "ff0000ff"),  # red
]


def style_xml() -> str:
    out = []
    for name, color in INTENSITY_COLORS:
        out.append(
            f"""<Style id="int_{name}">
  <LineStyle><color>ff000000</color><width>1.0</width></LineStyle>
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


def slug_from_path(p: Path) -> str:
    name = p.name
    if name.endswith("_sgd.geojson"):
        return name[: -len("_sgd.geojson")]
    return p.stem


def load_features(geojson_paths: list[Path]) -> list[dict]:
    out = []
    for p in geojson_paths:
        if not p.exists():
            continue
        slug = slug_from_path(p)
        season = detect_season(slug)
        try:
            fc = json.loads(p.read_text())
        except Exception as e:
            print(f"  skip ({e}): {p}")
            continue
        for feat in fc.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry") or {}
            if geom.get("type") != "Polygon":
                continue
            out.append(
                {
                    "slug": slug,
                    "season": season,
                    "area_m2": float(props.get("area_m2", 0.0)),
                    "mean_anomaly_c": float(props.get("mean_anomaly_c", 0.0)),
                    "min_anomaly_c": float(props.get("min_anomaly_c", 0.0)),
                    "p90_anomaly_c": float(props.get("p90_anomaly_c", 0.0)),
                    "intensity_index": float(props.get("intensity_index", 0.0)),
                    "n_observations": int(props.get("n_observations", 0)),
                    "centroid_lat": float(props.get("centroid_lat", 0.0)),
                    "centroid_lon": float(props.get("centroid_lon", 0.0)),
                    "geometry": geom,
                }
            )
    return out


def assign_tier(intensity: float, breaks: list[float]) -> str:
    for i, b in enumerate(breaks):
        if intensity < b:
            return INTENSITY_COLORS[i][0]
    return INTENSITY_COLORS[-1][0]


def coords_to_kml(coords: list[list[float]]) -> str:
    return " ".join(f"{c[0]:.8f},{c[1]:.8f},0" for c in coords)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("inputs", nargs="+", help="Input *_sgd.geojson files")
    ap.add_argument(
        "--breaks",
        nargs=5,
        type=float,
        default=None,
        help="Custom intensity tier boundaries (5 numbers; produces 6 tiers). "
        "If unset, derive from the 17/33/50/67/83 percentiles of intensity.",
    )
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    feats = load_features(paths)
    if not feats:
        raise SystemExit("No features loaded.")

    intensities = np.array([f["intensity_index"] for f in feats], dtype=np.float64)
    if args.breaks is None:
        breaks = [float(np.percentile(intensities, q)) for q in (17, 33, 50, 67, 83)]
    else:
        breaks = list(args.breaks)
    print(f"Intensity tier breakpoints: {[f'{b:.1f}' for b in breaks]}")

    # Build sorted output: highest-intensity first within each flight folder
    by_slug: dict[str, list[dict]] = {}
    by_season_count: dict[str, int] = {"jan2024": 0, "june2023": 0, "other": 0}
    for f in feats:
        f["tier"] = assign_tier(f["intensity_index"], breaks)
        by_slug.setdefault(f["slug"], []).append(f)
        by_season_count[f["season"]] = by_season_count.get(f["season"], 0) + 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Rapa Nui SGD — intensity-ranked (area × ΔT)</name>",
        f"<description><![CDATA["
        f"<b>{len(feats)} polygons</b> across {len(by_slug)} flights<br/>"
        f"Jan 2024: {by_season_count.get('jan2024', 0)} polygons<br/>"
        f"June 2023: {by_season_count.get('june2023', 0)} polygons<br/><br/>"
        f"<b>Intensity = area_m² × mean_anomaly_°C</b><br/>"
        f"Tier breakpoints (m²·°C): {', '.join(f'{b:.0f}' for b in breaks)}<br/>"
        f"Six tiers: very_low (cyan) → extreme (red)"
        f"]]></description>",
        style_xml(),
    ]

    # Group by season for top-level folders, then per-flight subfolders
    for season in ("jan2024", "june2023", "other"):
        season_slugs = sorted([s for s in by_slug if detect_season(s) == season])
        if not season_slugs:
            continue
        season_label = {"jan2024": "Jan 2024", "june2023": "June 2023", "other": "Other"}[season]
        parts.append(
            f"<Folder><name>{season_label} — {sum(len(by_slug[s]) for s in season_slugs)} polygons across {len(season_slugs)} flights</name>"
        )
        for slug in season_slugs:
            slug_feats = sorted(by_slug[slug], key=lambda f: -f["intensity_index"])
            total_area = sum(f["area_m2"] for f in slug_feats)
            total_intensity = sum(f["intensity_index"] for f in slug_feats)
            parts.append(
                f'<Folder><name>{slug} — {len(slug_feats)} polygons, '
                f'{total_area:.0f} m², Σ intensity {total_intensity:.0f}</name>'
            )
            for f in slug_feats:
                geom = f["geometry"]
                coords = geom["coordinates"]
                ext = coords[0]
                holes = coords[1:]
                ext_kml = coords_to_kml(ext)
                inner = "\n".join(
                    f"<innerBoundaryIs><LinearRing><coordinates>{coords_to_kml(h)}</coordinates></LinearRing></innerBoundaryIs>"
                    for h in holes
                )
                parts.append(
                    f"""<Placemark>
  <name>{f['area_m2']:.0f} m² · ΔT {f['mean_anomaly_c']:.2f}°C · I={f['intensity_index']:.0f}</name>
  <description><![CDATA[
    <b>Source flight:</b> {slug}<br/>
    <b>Area:</b> {f['area_m2']:.2f} m²<br/>
    <b>Mean anomaly:</b> {f['mean_anomaly_c']:.3f} °C below ambient<br/>
    <b>Coldest cell:</b> {f['min_anomaly_c']:.3f} °C below ambient<br/>
    <b>P90 anomaly:</b> {f['p90_anomaly_c']:.3f} °C<br/>
    <b>Intensity index:</b> {f['intensity_index']:.1f} m²·°C<br/>
    <b>Observation count:</b> {f['n_observations']} frames<br/>
    <b>Tier:</b> {f['tier']}<br/>
    <b>Centroid:</b> {f['centroid_lat']:.6f}, {f['centroid_lon']:.6f}
  ]]></description>
  <styleUrl>#int_{f['tier']}</styleUrl>
  <Polygon><outerBoundaryIs><LinearRing><coordinates>{ext_kml}</coordinates></LinearRing></outerBoundaryIs>
  {inner}
  </Polygon>
</Placemark>"""
                )
            parts.append("</Folder>")
        parts.append("</Folder>")

    parts.append("</Document></kml>")
    output.write_text("\n".join(parts))
    print(f"Wrote {output}  ({len(feats)} polygons, {len(by_slug)} flights)")


if __name__ == "__main__":
    main()
