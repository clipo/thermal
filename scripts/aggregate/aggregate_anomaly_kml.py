#!/usr/bin/env python3
"""Combine every per-flight cold-anomaly GroundOverlay into one master KML
for the whole island. Each flight is its own toggleable folder so you can
turn them on/off in Google Earth to compare coverage zones.

Each per-flight `<slug>_anomaly.kml` is a single GroundOverlay pointing at
`<slug>_anomaly.png` with the flight's bbox. We aggregate them into one
Document, grouped by season (Jan 2024 / June 2023), and rewrite the
`<href>` paths to be relative to the master KML location so the
GroundOverlays still resolve when opened.

Default output: sgd_output/rapa_nui_all_anomaly.kml

Open in Google Earth on top of its satellite imagery to get exactly the
view you liked from individual flights, but for the whole coast.

Usage:
    python scripts/aggregate_anomaly_kml.py
    python scripts/aggregate_anomaly_kml.py --opacity 0.7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def detect_season(slug: str) -> str:
    s = slug.lower()
    if "june2023" in s:
        return "june2023"
    if s.startswith("flight") or s.startswith("vaihu"):
        return "jan2024"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default=str(SGD_OUTPUT / "rapa_nui_all_anomaly.kml"))
    ap.add_argument("--opacity", type=float, default=0.85,
                    help="GroundOverlay alpha 0–1 (default 0.85)")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_dir = out_path.parent

    flights = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        npz = spread_dir / f"{slug}_anomaly.npz"
        png = spread_dir / f"{slug}_anomaly.png"
        if not (npz.exists() and png.exists()):
            continue
        try:
            d = np.load(npz)
            flights.append({
                "slug": slug,
                "season": detect_season(slug),
                "png_rel": str(png.relative_to(out_dir)),
                "north": float(d["bbox_max_lat"]),
                "south": float(d["bbox_min_lat"]),
                "east": float(d["bbox_max_lon"]),
                "west": float(d["bbox_min_lon"]),
                "n_frames": int(d["n_frames_used"]),
                "baseline_c": float(d["baseline_median_c"]),
            })
        except Exception as e:
            print(f"  skip {slug}: {e}")

    if not flights:
        raise SystemExit("No flights with both .npz and .png anomaly files found.")

    flights.sort(key=lambda f: (f["season"], f["slug"]))
    by_season = {"jan2024": [], "june2023": [], "other": []}
    for f in flights:
        by_season[f["season"]].append(f)

    # Color = AABBGGRR; alpha controls opacity
    alpha_byte = max(0, min(255, int(round(args.opacity * 255))))
    overlay_color = f"{alpha_byte:02x}ffffff"  # white tint with alpha

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        "<name>Rapa Nui SGD — cold anomaly rasters (all flights)</name>",
        f"<description><![CDATA["
        f"<b>{len(flights)} flights</b><br/>"
        f"Jan 2024: {len(by_season['jan2024'])} flights<br/>"
        f"June 2023: {len(by_season['june2023'])} flights<br/><br/>"
        f"Each GroundOverlay shows the per-flight integrated cold anomaly "
        f"(°C below ambient baseline) on top of Google Earth's satellite "
        f"imagery. Brighter red = colder. Toggle individual flights from "
        f"the sidebar.<br/>"
        f"]]></description>",
    ]

    season_labels = {"jan2024": "Jan 2024 (targeted SGD surveys)",
                      "june2023": "June 2023 (broad coastal surveys)",
                      "other": "Other"}

    for season in ("june2023", "jan2024", "other"):
        items = by_season[season]
        if not items:
            continue
        parts.append(
            f"<Folder><name>{season_labels[season]} — {len(items)} flights</name>"
            "<open>1</open>"
        )
        for f in items:
            parts.append(
                f"""<GroundOverlay>
  <name>{f['slug']}</name>
  <description><![CDATA[
    <b>Flight:</b> {f['slug']}<br/>
    <b>Frames used:</b> {f['n_frames']}<br/>
    <b>Per-frame baseline (median):</b> {f['baseline_c']:.2f} °C<br/>
    <b>BBox:</b> ({f['south']:.4f}, {f['west']:.4f}) → ({f['north']:.4f}, {f['east']:.4f})
  ]]></description>
  <color>{overlay_color}</color>
  <Icon><href>{f['png_rel']}</href></Icon>
  <LatLonBox>
    <north>{f['north']:.7f}</north>
    <south>{f['south']:.7f}</south>
    <east>{f['east']:.7f}</east>
    <west>{f['west']:.7f}</west>
  </LatLonBox>
</GroundOverlay>"""
            )
        parts.append("</Folder>")

    parts.append("</Document></kml>")
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path}  ({len(flights)} flights across {sum(1 for v in by_season.values() if v)} seasons)")


if __name__ == "__main__":
    main()
