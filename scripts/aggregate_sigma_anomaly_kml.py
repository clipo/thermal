#!/usr/bin/env python3
"""Cross-season master KML keyed on the threshold-independent
``sigma_anomaly_m2c`` integral (m²·°C) — see scripts/recompute_polygon_intensity.py.

Unlike aggregate_intensity_kml.py (which uses the legacy intensity_index =
area × peak_anomaly), this aggregator uses Σ_anomaly integrated within each
polygon's footprint over the per-flight anomaly raster. That metric is
robust to the SGD detection threshold and is the appropriate basis for
cross-flight / cross-season comparison.

Polygons missing a sigma_anomaly_m2c property (i.e., flights that haven't
been recomputed yet) are skipped with a warning, so this script is safe to
run mid-batch and re-run as more rasters complete.

Usage:
    python scripts/aggregate_sigma_anomaly_kml.py \\
        --output sgd_output/rapa_nui_all_sgd_sigma.kml \\
        sgd_output/flight*_spread/*_sgd.geojson \\
        sgd_output/june2023_*_spread/*_sgd.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Six-tier ramp; same KML AABBGGRR convention as aggregate_intensity_kml.py.
SIGMA_COLORS = [
    ("very_low",  "ffffe080"),
    ("low",       "ffff8040"),
    ("moderate",  "ff80ff00"),
    ("high",      "ff00ffff"),
    ("very_high", "ff0080ff"),
    ("extreme",   "ff0000ff"),
]


def style_xml() -> str:
    out = []
    for name, color in SIGMA_COLORS:
        out.append(
            f"""<Style id="sig_{name}">
  <LineStyle><color>ff000000</color><width>1.0</width></LineStyle>
  <PolyStyle><color>{color}</color><fill>1</fill><outline>1</outline></PolyStyle>
</Style>"""
        )
    return "\n".join(out)


def detect_season(slug: str) -> str:
    s = slug.lower()
    if "june2023" in s:
        return "june2023"
    if s.startswith("flight") or s.startswith("vaihu"):
        return "jan2024"
    return "other"


def slug_from_path(p: Path) -> str:
    name = p.name
    if name.endswith("_sgd.geojson"):
        return name[: -len("_sgd.geojson")]
    return p.stem


def load_features(geojson_paths: list[Path],
                  min_water_fraction: float = 0.0,
                  min_sigma: float = 1.0) -> tuple[list[dict], int, int]:
    """Returns (features_with_sigma, n_skipped_missing, n_skipped_artifact).

    Default filter is sigma_anomaly_m2c >= min_sigma (drops polygons that
    were entirely over land — sigma=0 after water-mask integration).
    Polygons with partial water coverage are KEPT (their sigma reflects
    only the water portion). The min_water_fraction filter was previously
    set to 0.5 but was too aggressive at sites like Vaihu Harbor where
    satellite imagery shows surf/sand near the shore that doesn't pass
    the strict HSV water test, even though SGD ground truth is solid.
    """
    out = []
    skipped_missing = 0
    skipped_artifact = 0
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
            if "sigma_anomaly_m2c" not in props:
                skipped_missing += 1
                continue
            sigma = float(props["sigma_anomaly_m2c"])
            water_frac = float(props.get("polygon_water_fraction", 1.0))
            if water_frac < min_water_fraction or sigma < min_sigma:
                skipped_artifact += 1
                continue
            out.append({
                "slug": slug,
                "season": season,
                "area_m2": float(props.get("area_m2", 0.0)),
                "sigma_anomaly_m2c": float(props["sigma_anomaly_m2c"]),
                "mean_anomaly_in_polygon_c": float(props.get("mean_anomaly_in_polygon_c", 0.0)),
                "peak_anomaly_in_polygon_c": float(props.get("peak_anomaly_in_polygon_c", 0.0)),
                "raster_coverage_frac": float(props.get("raster_coverage_frac", 0.0)),
                "raster_n_obs_median": float(props.get("raster_n_obs_median", 0.0)),
                "centroid_lat": float(props.get("centroid_lat", 0.0)),
                "centroid_lon": float(props.get("centroid_lon", 0.0)),
                "n_observations": int(props.get("n_observations", 0)),
                "geometry": geom,
            })
    return out, skipped_missing, skipped_artifact


def assign_tier(value: float, breaks: list[float]) -> str:
    for i, b in enumerate(breaks):
        if value < b:
            return SIGMA_COLORS[i][0]
    return SIGMA_COLORS[-1][0]


def coords_to_kml(coords: list[list[float]]) -> str:
    return " ".join(f"{c[0]:.8f},{c[1]:.8f},0" for c in coords)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("inputs", nargs="+", help="*_sgd.geojson files")
    ap.add_argument("--breaks", nargs=5, type=float, default=None,
                    help="Custom Σ_anomaly tier boundaries (5 numbers; produces 6 tiers).")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs]
    feats, n_skipped_missing, n_skipped_artifact = load_features(paths)
    if not feats:
        raise SystemExit(
            "No features with sigma_anomaly_m2c found. Run recompute_polygon_intensity.py first."
        )
    if n_skipped_missing:
        print(f"WARN: {n_skipped_missing} polygons skipped "
              f"(missing sigma_anomaly_m2c — recompute first).")
    if n_skipped_artifact:
        print(f"INFO: {n_skipped_artifact} polygons filtered as projection artifacts "
              f"(<50% water OR Σ_anomaly < 1).")

    sigmas = np.array([f["sigma_anomaly_m2c"] for f in feats], dtype=np.float64)
    if args.breaks is None:
        breaks = [float(np.percentile(sigmas, q)) for q in (17, 33, 50, 67, 83)]
    else:
        breaks = list(args.breaks)
    print(f"Σ_anomaly tier breakpoints (m²·°C): {[f'{b:.0f}' for b in breaks]}")

    by_slug: dict[str, list[dict]] = {}
    by_season_count: dict[str, int] = {"jan2024": 0, "june2023": 0, "other": 0}
    for f in feats:
        f["tier"] = assign_tier(f["sigma_anomaly_m2c"], breaks)
        by_slug.setdefault(f["slug"], []).append(f)
        by_season_count[f["season"]] = by_season_count.get(f["season"], 0) + 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Rapa Nui SGD — Σ_anomaly (threshold-independent)</name>",
        f"<description><![CDATA["
        f"<b>{len(feats)} polygons</b> across {len(by_slug)} flights<br/>"
        f"Jan 2024: {by_season_count.get('jan2024', 0)} polygons<br/>"
        f"June 2023: {by_season_count.get('june2023', 0)} polygons<br/><br/>"
        f"<b>Σ_anomaly = ∫(T_baseline − T) dA</b> over polygon footprint<br/>"
        f"Tier breakpoints (m²·°C): {', '.join(f'{b:.0f}' for b in breaks)}<br/>"
        f"Six tiers: very_low (cyan) → extreme (red)"
        f"]]></description>",
        style_xml(),
    ]

    for season in ("jan2024", "june2023", "other"):
        season_slugs = sorted([s for s in by_slug if detect_season(s) == season])
        if not season_slugs:
            continue
        season_label = {"jan2024": "Jan 2024", "june2023": "June 2023", "other": "Other"}[season]
        parts.append(
            f"<Folder><name>{season_label} — "
            f"{sum(len(by_slug[s]) for s in season_slugs)} polygons across "
            f"{len(season_slugs)} flights</name>"
        )
        for slug in season_slugs:
            slug_feats = sorted(by_slug[slug], key=lambda f: -f["sigma_anomaly_m2c"])
            total_sigma = sum(f["sigma_anomaly_m2c"] for f in slug_feats)
            total_area = sum(f["area_m2"] for f in slug_feats)
            parts.append(
                f'<Folder><name>{slug} — {len(slug_feats)} polygons, '
                f'{total_area:.0f} m², Σ {total_sigma:.0f} m²·°C</name>'
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
  <name>Σ={f['sigma_anomaly_m2c']:.0f} m²·°C · {f['area_m2']:.0f} m² · {f['mean_anomaly_in_polygon_c']:.2f}°C</name>
  <description><![CDATA[
    <b>Source flight:</b> {f['slug']}<br/>
    <b>Σ_anomaly:</b> {f['sigma_anomaly_m2c']:.1f} m²·°C<br/>
    <b>Area:</b> {f['area_m2']:.2f} m²<br/>
    <b>Mean anomaly in polygon:</b> {f['mean_anomaly_in_polygon_c']:.3f} °C below ambient<br/>
    <b>Peak anomaly in polygon:</b> {f['peak_anomaly_in_polygon_c']:.3f} °C<br/>
    <b>Raster coverage:</b> {f['raster_coverage_frac']*100:.1f}% of polygon cells observed<br/>
    <b>Median frame coverage:</b> {f['raster_n_obs_median']:.0f} frames per cell<br/>
    <b>Detection observations:</b> {f['n_observations']} frames<br/>
    <b>Tier:</b> {f['tier']}<br/>
    <b>Centroid:</b> {f['centroid_lat']:.6f}, {f['centroid_lon']:.6f}
  ]]></description>
  <styleUrl>#sig_{f['tier']}</styleUrl>
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
