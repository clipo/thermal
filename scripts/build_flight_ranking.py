#!/usr/bin/env python3
"""Cross-flight ranking bar chart by Σ_anomaly_m2c.

Two panels:
  Top:    horizontal bar chart of total Σ_anomaly per flight, colored by season
  Bottom: scatter of n_polygons vs total Σ_anomaly per flight (annotated)

Output: sgd_output/figures/rapa_nui_flight_ranking.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_summary(source: str = "detector") -> list[dict]:
    """source: 'detector' uses polygon_intensity_summary.csv (water-mask
    corrected detector polygons). 'raster' uses polygon_comparison_summary.csv
    raster column."""
    if source == "raster":
        p = SGD_OUTPUT / "polygon_comparison_summary.csv"
        if not p.exists():
            raise SystemExit("Run scripts/build_polygon_comparison_summary.py first")
        out = []
        with p.open() as f:
            for r in csv.DictReader(f):
                out.append({
                    "slug": r["slug"],
                    "n_polys": int(r["raster_polygons"]),
                    "sigma": float(r["raster_sigma_m2c"]),
                })
        return out
    p = SGD_OUTPUT / "polygon_intensity_summary.csv"
    out = []
    with p.open() as f:
        for r in csv.DictReader(f):
            out.append({
                "slug": r["slug"],
                "n_polys": int(r["n_polygons"]),
                "sigma": float(r["sigma_anomaly_total_m2c"]),
            })
    return out


def season_of(slug: str) -> str:
    s = slug.lower()
    if "june2023" in s:
        return "June 2023"
    return "Jan 2024"


def short_label(slug: str) -> str:
    """Compact, paper-friendly label for x-axis."""
    s = slug.replace("_combined", "")
    s = s.replace("flight", "F")
    s = s.replace("june2023_", "")
    s = s.replace("_july_23", "")
    s = s.replace("_june_23", "")
    s = s.replace("_flights", "")
    s = s.replace("sortphotos", "")
    s = s.replace("_full", "")
    s = s.replace("__", "_").strip("_")
    if len(s) > 35:
        s = s[:34] + "…"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default=str(SGD_OUTPUT / "figures"))
    ap.add_argument("--polygon-source", choices=("detector", "raster"),
                    default="detector")
    args = ap.parse_args()

    rows = sorted(load_summary(args.polygon_source), key=lambda r: r["sigma"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seasons = [season_of(r["slug"]) for r in rows]
    palette = {"Jan 2024": "#1f77b4", "June 2023": "#d62728"}
    colors = [palette[s] for s in seasons]
    labels = [short_label(r["slug"]) for r in rows]
    sigmas = [r["sigma"] for r in rows]
    polys = [r["n_polys"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13),
                                    gridspec_kw={"height_ratios": [3, 1.4]},
                                    constrained_layout=True)

    y = np.arange(len(rows))
    ax1.barh(y, sigmas, color=colors, edgecolor="black", linewidth=0.4)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Σ_anomaly  (m² · °C)")
    ax1.set_title("Per-flight cold-anomaly content (Σ_anomaly_m2c)\n"
                  f"{sum(polys):,} polygons across {len(rows)} flights",
                  fontsize=11)
    ax1.grid(True, axis="x", alpha=0.25, linewidth=0.4)
    # Annotate each bar with sigma + polycount
    for i, (s, n) in enumerate(zip(sigmas, polys)):
        ax1.text(s + max(sigmas) * 0.005, i,
                 f"  {s:,.0f}  ({n} polys)",
                 va="center", fontsize=7, color="#333")

    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=palette["Jan 2024"], label="Jan 2024"),
                        Patch(color=palette["June 2023"], label="June 2023")],
               loc="lower right", fontsize=9, framealpha=0.9)

    # Bottom panel: scatter of n_polys vs sigma
    sigma_arr = np.array(sigmas)
    poly_arr = np.array(polys)
    season_arr = np.array(seasons)
    for s in ("Jan 2024", "June 2023"):
        m = season_arr == s
        ax2.scatter(poly_arr[m], sigma_arr[m], color=palette[s], label=s,
                    edgecolor="black", linewidth=0.4, s=70, alpha=0.85)
    ax2.set_xlabel("Number of merged SGD polygons per flight")
    ax2.set_ylabel("Σ_anomaly  (m²·°C)")
    ax2.set_title("Polygon count vs total intensity per flight", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.25, linewidth=0.4)
    ax2.legend(fontsize=9)

    suffix = "_raster" if args.polygon_source == "raster" else ""
    out_png = output_dir / f"rapa_nui_flight_ranking{suffix}.png"
    out_pdf = output_dir / f"rapa_nui_flight_ranking{suffix}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
