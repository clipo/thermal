#!/usr/bin/env python3
"""Render a publication-quality validation figure for a single flight site:

    Panel 1: cold-anomaly raster (color = °C below ambient)
    Panel 2: same raster with detected SGD polygons outlined
    Panel 3: observation count (how many frames covered each cell)

A title bar shows flight slug, polygon count, total Σ_anomaly_m2c, frame
count, and baseline temperature. A scale bar (100 m) and north arrow are
overlaid in the bottom-right.

Outputs `<slug>_validation.png` (single combined figure) into the flight's
spread directory.

Usage:
    # one site
    python scripts/build_validation_figure.py --slug flight4_vaihu_east_full

    # the four reference sites
    python scripts/build_validation_figure.py --reference

    # all flights with both raster + polygons
    python scripts/build_validation_figure.py --all
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"

REFERENCE_SITES = [
    "flight4_vaihu_east_full",   # Vaihu Harbor (validated reference)
    "flight7_hekii_east",        # Hekii East
    "flight8_hekii_west",        # Hekii West
    "june2023_2_july_23_anakena",  # Anakena
    "june2023_1_july_23_tongariki_poike",  # Tongariki-Poike
]


def find_pairs(slug_filter: list[str] | None = None,
                source: str = "detector") -> list[tuple[str, Path, Path]]:
    """source: 'detector' uses <slug>_sgd.geojson, 'raster' uses
    <slug>_sgd_raster.geojson."""
    suffix = "_sgd_raster.geojson" if source == "raster" else "_sgd.geojson"
    out = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        if slug_filter and slug not in slug_filter:
            continue
        npz = spread_dir / f"{slug}_anomaly.npz"
        gj = spread_dir / f"{slug}{suffix}"
        if npz.exists() and gj.exists():
            out.append((slug, npz, gj))
    return out


def lonlat_to_grid(lon: float, lat: float, raster) -> tuple[float, float]:
    minlon = float(raster["bbox_min_lon"])
    minlat = float(raster["bbox_min_lat"])
    maxlat = float(raster["bbox_max_lat"])
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))
    res = float(raster["grid_resolution_m"])
    c = (lon - minlon) * mpd_lon / res
    r = (lat - minlat) * mpd_lat / res
    return c, r


def add_scale_bar(ax, length_m: float, grid_res_m: float, label: str | None = None):
    """Add a length_m horizontal scale bar in bottom-right of axes (in pixel coords)."""
    px_len = length_m / grid_res_m
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_anchor = xlim[1] - 0.05 * (xlim[1] - xlim[0])
    y_anchor = ylim[0] + 0.06 * (ylim[1] - ylim[0])
    ax.plot(
        [x_anchor - px_len, x_anchor],
        [y_anchor, y_anchor],
        color="black", linewidth=2.5, solid_capstyle="butt",
    )
    ax.text(
        x_anchor - px_len / 2, y_anchor + 0.015 * (ylim[1] - ylim[0]),
        label or f"{int(length_m)} m",
        ha="center", va="bottom", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )


def add_north_arrow(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + 0.04 * (xlim[1] - xlim[0])
    y = ylim[1] - 0.08 * (ylim[1] - ylim[0])
    arrow_len = 0.05 * (ylim[1] - ylim[0])
    ax.annotate(
        "", xy=(x, y), xytext=(x, y - arrow_len),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
    )
    ax.text(x, y + 0.005 * (ylim[1] - ylim[0]), "N",
            ha="center", va="bottom", fontsize=10, fontweight="bold")


def render_site(slug: str, npz_path: Path, gj_path: Path, output_path: Path,
                vmax: float = 1.5):
    raster = np.load(npz_path)
    anomaly = raster["anomaly"]
    obs_count = raster["observations"]
    grid_res = float(raster["grid_resolution_m"])
    baseline_c = float(raster["baseline_median_c"])
    n_frames = int(raster["n_frames_used"])

    fc = json.loads(gj_path.read_text())
    poly_patches = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        pts = [lonlat_to_grid(p[0], p[1], raster) for p in ring]
        poly_patches.append(MplPoly(pts, closed=True))

    # Polygon-derived stats from the geojson properties (use sigma_anomaly_m2c if present)
    sigmas = [f["properties"].get("sigma_anomaly_m2c", 0.0) for f in fc["features"]]
    total_sigma = float(sum(sigmas))
    total_area = float(sum(f["properties"].get("area_m2", 0.0) for f in fc["features"]))
    n_polys = len(fc["features"])

    # Aspect-aware layout: long-thin strips look better stacked vertically
    gy, gx = anomaly.shape
    data_aspect = gx / max(gy, 1)
    stacked = data_aspect > 2.5  # wide and thin → stack
    if stacked:
        fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)

    # Origin lower so lat increases upward (matches bilinear projection)
    cmap = "YlOrRd"
    finite = np.isfinite(anomaly)
    masked = np.where(finite, anomaly, np.nan)

    im0 = axes[0].imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                         interpolation="nearest")
    axes[0].set_title("Cold anomaly (°C below baseline)", fontsize=11)
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.8, label="°C")

    im1 = axes[1].imshow(masked, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                         interpolation="nearest", alpha=0.85)
    if poly_patches:
        pc = PatchCollection(poly_patches, facecolor="none",
                             edgecolor="black", linewidth=0.7)
        axes[1].add_collection(pc)
    axes[1].set_title(f"With detected SGD polygons (n={n_polys})", fontsize=11)
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="°C")

    obs_disp = np.where(obs_count > 0, obs_count, np.nan)
    im2 = axes[2].imshow(obs_disp, origin="lower", cmap="viridis",
                         interpolation="nearest")
    axes[2].set_title("Observations per cell (frame coverage)", fontsize=11)
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="frames")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    add_scale_bar(axes[0], length_m=100, grid_res_m=grid_res)
    add_north_arrow(axes[0])

    title = (f"{slug}   |   {n_frames} frames   |   baseline {baseline_c:.2f}°C   "
             f"|   {n_polys} SGD polys, "
             f"total area {total_area:.0f} m², Σ={total_sigma:.0f} m²·°C")
    fig.suptitle(title, fontsize=12, y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "slug": slug, "n_polys": n_polys, "total_sigma_m2c": total_sigma,
        "total_area_m2": total_area, "baseline_c": baseline_c,
        "n_frames": n_frames, "out": str(output_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", help="single flight slug")
    g.add_argument("--reference", action="store_true",
                   help=f"render the reference set: {REFERENCE_SITES}")
    g.add_argument("--all", action="store_true", help="render all available")
    ap.add_argument("--vmax", type=float, default=1.5,
                    help="color scale top in °C below baseline (default 1.5)")
    ap.add_argument("--polygon-source", choices=("detector", "raster"),
                    default="detector",
                    help="which polygon set to overlay on the raster panel. "
                         "'raster' (raster-thresholded) matches the visible "
                         "cold zones exactly; 'detector' is more conservative.")
    args = ap.parse_args()

    if args.slug:
        pairs = find_pairs([args.slug], source=args.polygon_source)
    elif args.reference:
        pairs = find_pairs(REFERENCE_SITES, source=args.polygon_source)
    else:
        pairs = find_pairs(source=args.polygon_source)

    if not pairs:
        raise SystemExit("No matching flights found (need both *_anomaly.npz and *_sgd.geojson).")

    suffix_out = "_validation_raster" if args.polygon_source == "raster" else "_validation"
    print(f"Rendering {len(pairs)} validation figures (polygons: {args.polygon_source})…")
    for slug, npz, gj in pairs:
        out = SGD_OUTPUT / f"{slug}_spread" / f"{slug}{suffix_out}.png"
        try:
            res = render_site(slug, npz, gj, out, vmax=args.vmax)
            print(f"  ✓ {res['slug']}: {res['n_polys']} polys, "
                  f"Σ={res['total_sigma_m2c']:.0f} m²·°C → {res['out']}")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
