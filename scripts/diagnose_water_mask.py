#!/usr/bin/env python3
"""Render a side-by-side debug image showing what the satellite water
mask classified as water vs land for a given flight, against the
satellite tile and the anomaly raster. Use this to identify where
the HSV classifier is letting land through (rocky shore mistaken for
water) vs killing real water (shallow sandy bays).

Usage:
    python scripts/diagnose_water_mask.py --slug vaihu_full
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import contextily as ctx

    spread = SGD_OUTPUT / f"{args.slug}_spread"
    npz = np.load(spread / f"{args.slug}_anomaly.npz")
    wm = np.load(spread / f"{args.slug}_water_mask.npz")
    minlon = float(npz["bbox_min_lon"]); maxlon = float(npz["bbox_max_lon"])
    minlat = float(npz["bbox_min_lat"]); maxlat = float(npz["bbox_max_lat"])

    print(f"Fetching satellite tile for {args.slug}…")
    img, ext = ctx.bounds2img(minlon, minlat, maxlon, maxlat,
                               zoom=args.zoom, source=ctx.providers.Esri.WorldImagery,
                               ll=True)
    if img.shape[-1] == 4:
        img = img[..., :3]
    H, W = img.shape[:2]
    print(f"  satellite: {W}×{H} px")

    # Project water mask onto the satellite tile coords for visualization.
    # We have water mask in raster grid, but the satellite is in web mercator.
    # Easier: render water mask in raster coords and reproject.
    is_water = wm["is_water"]
    print(f"  water mask shape: {is_water.shape}, water frac: {100*is_water.mean():.1f}%")

    # Build an overlay: red where land (not water), blue where water
    # First, project the satellite to raster grid coords.
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))
    grid_res = float(npz["grid_resolution_m"])
    gy, gx = is_water.shape

    # Project satellite px → raster grid cell
    R = 6378137.0
    xmin_m, xmax_m, ymin_m, ymax_m = ext
    xs = np.linspace(xmin_m, xmax_m, W)
    ys = np.linspace(ymax_m, ymin_m, H)  # row 0 = north
    X_g, Y_g = np.meshgrid(xs, ys)
    lon_g = np.degrees(X_g / R)
    lat_g = np.degrees(2 * np.arctan(np.exp(Y_g / R)) - np.pi / 2)
    col_g = ((lon_g - minlon) * mpd_lon / grid_res).astype(np.int64)
    row_g = ((lat_g - minlat) * mpd_lat / grid_res).astype(np.int64)
    valid = (col_g >= 0) & (col_g < gx) & (row_g >= 0) & (row_g < gy)
    sat_water_mask = np.zeros((H, W), dtype=bool)
    sat_water_mask[valid] = is_water[row_g[valid], col_g[valid]]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)

    axes[0].imshow(img); axes[0].set_title("Satellite tile (Esri WorldImagery)")
    axes[0].axis("off")

    overlay = img.copy()
    # Color water cells blue (50% opacity)
    overlay_water = np.array([0, 80, 200], dtype=np.uint8)
    overlay[sat_water_mask] = (overlay[sat_water_mask] * 0.5 + overlay_water * 0.5).astype(np.uint8)
    axes[1].imshow(overlay); axes[1].set_title("Cells classified as WATER (blue)")
    axes[1].axis("off")

    # Highlight cells that are water but look like land (low blue_score)
    overlay2 = img.copy()
    rgbf = img.astype(np.float32) / 255.0
    blue_score = rgbf[..., 2] / (rgbf.sum(axis=-1) + 1e-6)
    suspect = sat_water_mask & (blue_score < 0.36)  # classified water but not blue-dominant
    suspect_color = np.array([255, 50, 50], dtype=np.uint8)
    overlay2[suspect] = (overlay2[suspect] * 0.4 + suspect_color * 0.6).astype(np.uint8)
    axes[2].imshow(overlay2); axes[2].set_title(f"SUSPECT cells: classified water but blue_score<0.36 ({suspect.sum():,} px)")
    axes[2].axis("off")

    out = args.out or f"/tmp/water_mask_diag_{args.slug}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  total water-classified px: {sat_water_mask.sum():,}")
    print(f"  suspect water px (blue_score<0.36): {suspect.sum():,}  ({100*suspect.sum()/(sat_water_mask.sum()+1):.1f}%)")


if __name__ == "__main__":
    main()
