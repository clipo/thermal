#!/usr/bin/env python3
"""Island-wide SGD overview figure for publication.

Aggregates every flight's polygon centroids and plots them on a map of
Rapa Nui, sized & colored by sigma_anomaly_m2c. Survey footprints from
all flights are shaded behind the points so the reader sees what was
covered. Top-N sites are labeled by location name (auto-derived from
the flight slug).

Outputs: sgd_output/figures/rapa_nui_overview.png + .pdf
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_all_polygons(min_sigma: float = 1.0,
                       source: str = "detector") -> list[dict]:
    """source: 'detector' / 'raster' / 'coastal'."""
    out = []
    n_artifacts = 0
    if source == "raster":
        suffix = "_sgd_raster.geojson"
    elif source == "coastal":
        suffix = "_sgd_coastal.geojson"
    else:
        suffix = "_sgd.geojson"
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        gj = spread_dir / f"{slug}{suffix}"
        if not gj.exists():
            continue
        try:
            fc = json.loads(gj.read_text())
        except Exception:
            continue
        for feat in fc.get("features", []):
            props = feat.get("properties", {})
            if "sigma_anomaly_m2c" not in props:
                continue
            if float(props["sigma_anomaly_m2c"]) < min_sigma:
                n_artifacts += 1
                continue
            out.append({
                "slug": slug,
                "lon": float(props.get("centroid_lon", 0.0)),
                "lat": float(props.get("centroid_lat", 0.0)),
                "sigma": float(props["sigma_anomaly_m2c"]),
                "area": float(props.get("area_m2", 0.0)),
                "mean_anom": float(props.get("mean_anomaly_in_polygon_c", 0.0)),
            })
    return out


def load_all_extents() -> list[tuple[float, float, float, float]]:
    """Load each flight's anomaly raster bbox for shading."""
    out = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        npz = spread_dir / f"{slug}_anomaly.npz"
        if not npz.exists():
            continue
        try:
            d = np.load(npz)
            out.append((float(d["bbox_min_lon"]), float(d["bbox_min_lat"]),
                        float(d["bbox_max_lon"]), float(d["bbox_max_lat"])))
        except Exception:
            continue
    return out


def deduce_site_name(slug: str) -> str:
    """Best-effort human-readable name from a flight slug."""
    s = slug.lower()
    if "vaihu_full" in s or "vaihu_east" in s or "vaihu_west" in s:
        return "Vaihu"
    if "hekii_east" in s or "hekii_west" in s:
        return "Hekii"
    if "hivahiva" in s or "hiva_hiva" in s:
        return "Hiva-Hiva"
    if "tongariki" in s and "poike" in s:
        return "Tongariki/Poike"
    if "tongariki" in s:
        return "Tongariki"
    if "poike" in s:
        return "Poike"
    if "anakena" in s:
        return "Anakena"
    if "tetenga" in s:
        return "Tetenga"
    if "rano_kau" in s and "hanga_roa" in s:
        return "Rano Kau / Hanga Roa"
    if "hanga_roa" in s:
        return "Hanga Roa"
    if "rano_kau" in s:
        return "Rano Kau"
    if "te_peu" in s:
        return "Te Peu"
    if "ahu_o_huari" in s:
        return "Ahu O Huari"
    if "vai_takari" in s or "hanga_oteo" in s:
        return "Vai Takari Ua / Hanga Oteo"
    if "vai_mata" in s:
        return "Vai Mata"
    if "one_makihi" in s:
        return "One Makihi"
    if "kikirahamea" in s:
        return "Kikirahamea"
    if s == "june2023_25_june_23" or "25_june" in s:
        return "Southwest survey (25 Jun)"
    if "24_june" in s:
        return "Northeast survey (24 Jun)"
    return slug


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(SGD_OUTPUT / "figures"))
    ap.add_argument("--top-n-labels", type=int, default=12)
    ap.add_argument("--polygon-source", choices=("detector", "raster", "coastal"),
                    default="detector")
    args = ap.parse_args()

    polys = load_all_polygons(source=args.polygon_source)
    extents = load_all_extents()
    if not polys:
        raise SystemExit("No polygons found — run recompute_polygon_intensity.py first.")

    print(f"Loaded {len(polys)} real-signal polygons across {len(set(p['slug'] for p in polys))} flights "
          f"and {len(extents)} flight extents (projection-bug polygons over land filtered).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sigmas = np.array([p["sigma"] for p in polys])
    lons = np.array([p["lon"] for p in polys])
    lats = np.array([p["lat"] for p in polys])

    fig, ax = plt.subplots(figsize=(12, 11), constrained_layout=True)

    # Shade each flight's coverage extent (light gray rectangle outline)
    for (mnl, mns, mxl, mxs) in extents:
        ax.add_patch(plt.Rectangle((mnl, mns), mxl - mnl, mxs - mns,
                                    facecolor="#88aacc", edgecolor="none",
                                    alpha=0.10, zorder=1))

    # Color/size by sigma (log scale because the dynamic range is 4+ decades)
    sig_clip = np.clip(sigmas, 1.0, None)
    sizes = 6 + 60 * np.log1p(sig_clip) / np.log1p(sigmas.max())
    sc = ax.scatter(lons, lats, c=sig_clip, s=sizes,
                    cmap="YlOrRd", norm=LogNorm(vmin=10, vmax=max(15000, sigmas.max())),
                    edgecolor="black", linewidth=0.3, zorder=3, alpha=0.92)

    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02,
                      label="Σ_anomaly per polygon  (m²·°C, log scale)")

    # Label top-N polygons by sigma with their site name
    topn = sorted(polys, key=lambda p: -p["sigma"])[: args.top_n_labels]
    seen_names = set()
    for p in topn:
        name = deduce_site_name(p["slug"])
        if name in seen_names:
            continue
        seen_names.add(name)
        ax.annotate(name, (p["lon"], p["lat"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="#222",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#666", lw=0.5, alpha=0.85),
                    zorder=4)

    # Aspect: lat:lon should match cos(centerlat) so the island isn't squashed
    centerlat = float(np.mean(lats))
    ax.set_aspect(1.0 / math.cos(math.radians(centerlat)))

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    n_polys = len(polys)
    n_flights = len(set(p["slug"] for p in polys))
    total_sigma = float(sigmas.sum())
    ax.set_title(
        f"Rapa Nui submarine groundwater discharge (SGD) — "
        f"{n_polys:,} cold-anomaly polygons across {n_flights} thermal-drone flights\n"
        f"Σ_anomaly grand total = {total_sigma:,.0f} m²·°C   "
        f"(point size & color: per-polygon Σ_anomaly, "
        f"shaded blocks: flight footprints)",
        fontsize=11, pad=12,
    )
    ax.grid(True, alpha=0.25, linewidth=0.4)

    suffix = ""
    if args.polygon_source == "raster":
        suffix = "_raster"
    elif args.polygon_source == "coastal":
        suffix = "_coastal"
    out_png = output_dir / f"rapa_nui_overview{suffix}.png"
    out_pdf = output_dir / f"rapa_nui_overview{suffix}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
