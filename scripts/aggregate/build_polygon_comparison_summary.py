#!/usr/bin/env python3
"""Cross-flight summary comparing detector vs raster polygons.

For each flight, reports:
  - Detector polygons: count, Σ_anomaly_m2c (water-mask corrected),
                       n_artifact_polys (water_frac < 0.05)
  - Raster polygons:   count, Σ_anomaly_m2c

Output: sgd_output/polygon_comparison_summary.csv (sorted by raster Σ desc)
        + sgd_output/figures/polygon_comparison.png (bar chart)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_per_flight() -> list[dict]:
    rows = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        det = spread_dir / f"{slug}_sgd.geojson"
        rast = spread_dir / f"{slug}_sgd_raster.geojson"
        if not det.exists() and not rast.exists():
            continue

        det_n = det_sigma = det_artifacts = 0
        if det.exists():
            try:
                fc = json.loads(det.read_text())
                for f in fc["features"]:
                    p = f["properties"]
                    if "sigma_anomaly_m2c" not in p:
                        continue
                    s = float(p["sigma_anomaly_m2c"])
                    wf = float(p.get("polygon_water_fraction", 1.0))
                    if wf <= 0.05 or s < 1:
                        det_artifacts += 1
                    else:
                        det_n += 1
                        det_sigma += s
            except Exception:
                pass

        rast_n = rast_sigma = 0
        if rast.exists():
            try:
                fc = json.loads(rast.read_text())
                for f in fc["features"]:
                    p = f["properties"]
                    s = float(p.get("sigma_anomaly_m2c", 0.0))
                    rast_n += 1
                    rast_sigma += s
            except Exception:
                pass

        rows.append({
            "slug": slug,
            "det_polygons": det_n,
            "det_sigma_m2c": det_sigma,
            "det_artifact_polys": det_artifacts,
            "raster_polygons": rast_n,
            "raster_sigma_m2c": rast_sigma,
        })
    rows.sort(key=lambda r: -r["raster_sigma_m2c"])
    return rows


def write_csv(rows: list[dict], out: Path):
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


def short_label(slug: str) -> str:
    s = slug.replace("flight", "F").replace("june2023_", "")
    s = s.replace("_july_23", "").replace("_june_23", "")
    s = s.replace("_combined", "").replace("__", "_").strip("_")
    if len(s) > 32:
        s = s[:31] + "…"
    return s


def write_figure(rows: list[dict], out: Path):
    rows = sorted(rows, key=lambda r: r["raster_sigma_m2c"])
    n = len(rows)
    labels = [short_label(r["slug"]) for r in rows]
    det = np.array([r["det_sigma_m2c"] for r in rows])
    rast = np.array([r["raster_sigma_m2c"] for r in rows])
    artifacts = np.array([r["det_artifact_polys"] for r in rows])

    fig, ax = plt.subplots(figsize=(11, 12), constrained_layout=True)
    y = np.arange(n)
    h = 0.4
    ax.barh(y - h/2, rast, h, color="#1f77b4", label="Raster polygons (continuous)",
            edgecolor="black", lw=0.4)
    ax.barh(y + h/2, det, h, color="#d62728", label="Detector polygons (water-masked)",
            edgecolor="black", lw=0.4)
    for i, (rs, ds, a) in enumerate(zip(rast, det, artifacts)):
        ax.text(max(rs, ds) + max(rast) * 0.005, i,
                f"  R:{rs:,.0f}  D:{ds:,.0f} (-{a} arts)",
                va="center", fontsize=7, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Σ_anomaly  (m²·°C)")
    ax.set_title("Per-flight cold-anomaly content: raster polygons vs water-masked detector polygons\n"
                  "Raster polygons capture full plume halo; detector captures discrete coherent cores. "
                  "(-N arts) = projection-bug polygons dropped.",
                  fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load_per_flight()
    if not rows:
        raise SystemExit("No flights with polygons found.")
    out_csv = SGD_OUTPUT / "polygon_comparison_summary.csv"
    out_png = SGD_OUTPUT / "figures" / "polygon_comparison.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    write_figure(rows, out_png)

    print()
    print("=== Top 10 by raster Σ_anomaly ===")
    for r in rows[:10]:
        print(f"  {r['slug'][:40]:40s}  raster={r['raster_sigma_m2c']:>9,.0f}  "
              f"detector={r['det_sigma_m2c']:>8,.0f}  "
              f"({r['det_artifact_polys']} artifact polys)")


if __name__ == "__main__":
    main()
