#!/usr/bin/env python3
"""Render the SAM2 segmentation result for a given flight so we can see
why classify_mask_water voted what it did. Saves a multi-panel diagnostic
PNG showing:
  - Original satellite tile
  - All SAM2 masks (random colors)
  - Masks classified as water (highlighted blue)
  - Final water mask projected to raster grid

Usage:
    python scripts/debug_sam2_water_mask.py --slug flight1_west_of_tetenga
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def get_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True)
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import contextily as ctx
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    npz_path = SGD_OUTPUT / f"{args.slug}_spread" / f"{args.slug}_anomaly.npz"
    raster = np.load(npz_path)
    minlon = float(raster["bbox_min_lon"]); maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"]); maxlat = float(raster["bbox_max_lat"])
    print(f"bbox lat=[{minlat:.4f},{maxlat:.4f}] lon=[{minlon:.4f},{maxlon:.4f}]")
    print(f"  span: {(maxlat-minlat)*111000:.0f}m × {(maxlon-minlon)*111000*math.cos(math.radians(0.5*(minlat+maxlat))):.0f}m")

    img, ext = ctx.bounds2img(minlon, minlat, maxlon, maxlat,
                               zoom=args.zoom, source=ctx.providers.Esri.WorldImagery,
                               ll=True)
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.astype(np.uint8)
    H, W = img.shape[:2]
    print(f"satellite tile: {W}×{H} px, ext (web merc) x=[{ext[0]:.0f},{ext[1]:.0f}] y=[{ext[2]:.0f},{ext[3]:.0f}]")

    print("Loading SAM2…")
    mg = SAM2AutomaticMaskGenerator.from_pretrained(
        "facebook/sam2.1-hiera-tiny",
        device=get_device(),
        points_per_side=16,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=400,
        multimask_output=False,
    )
    masks = mg.generate(img)
    print(f"Generated {len(masks)} SAM2 masks")

    rgbf = img.astype(np.float32) / 255.0
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)

    axes[0, 0].imshow(img); axes[0, 0].set_title("Satellite tile (Esri)")
    axes[0, 0].axis("off")

    # All masks colored
    overlay_all = img.copy()
    rng = np.random.default_rng(42)
    for m in masks:
        seg = m["segmentation"]
        color = rng.integers(50, 255, size=3, dtype=np.uint8)
        overlay_all[seg] = (overlay_all[seg] * 0.4 + color * 0.6).astype(np.uint8)
    axes[0, 1].imshow(overlay_all); axes[0, 1].set_title(f"All {len(masks)} SAM2 masks (random colors)")
    axes[0, 1].axis("off")

    # Mask-by-mask classification
    overlay_water = img.copy()
    n_water = 0
    print("\nMask classification:")
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        if not seg.any(): continue
        pixels = rgbf[seg]
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        mr, mg_, mb = float(r.mean()), float(g.mean()), float(b.mean())
        blue_score = mb / (mr + mg_ + mb + 1e-6)
        v = float(np.maximum.reduce([r, g, b]).mean())
        is_water = (blue_score > 0.34 and v < 0.85
                     and mb >= max(mr, mg_) - 0.02)
        flag = "WATER" if is_water else "land "
        print(f"  mask {i:3d}: area={int(seg.sum()):>8d}  RGB=({mr:.2f},{mg_:.2f},{mb:.2f})  "
              f"blue_score={blue_score:.3f}  V={v:.3f}  → {flag}")
        if is_water:
            overlay_water[seg] = (overlay_water[seg] * 0.4 + np.array([0, 100, 255]) * 0.6).astype(np.uint8)
            n_water += 1
    axes[1, 0].imshow(overlay_water)
    axes[1, 0].set_title(f"Masks classified as water ({n_water}/{len(masks)}, blue overlay)")
    axes[1, 0].axis("off")

    # Final mask projected to raster grid
    grid_res = float(raster["grid_resolution_m"])
    gy, gx = raster["anomaly"].shape
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))
    col_idx = np.arange(gx) + 0.5
    row_idx = np.arange(gy) + 0.5
    lons = minlon + col_idx * grid_res / mpd_lon
    lats = minlat + row_idx * grid_res / mpd_lat
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    R = 6378137.0
    x_grid = R * np.radians(lon_grid)
    y_grid = R * np.log(np.tan(np.pi / 4 + np.radians(lat_grid) / 2))
    xmin_m, xmax_m, ymin_m, ymax_m = ext
    px_x = ((x_grid - xmin_m) / (xmax_m - xmin_m)) * W
    px_y = ((ymax_m - y_grid) / (ymax_m - ymin_m)) * H
    px_c = np.clip(px_x.astype(np.int64), 0, W - 1)
    px_r = np.clip(px_y.astype(np.int64), 0, H - 1)

    water_pix = np.zeros((H, W), dtype=bool)
    for m in masks:
        seg = m["segmentation"]
        pixels = rgbf[seg] if seg.any() else None
        if pixels is None: continue
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
        mr, mg_, mb = float(r.mean()), float(g.mean()), float(b.mean())
        blue_score = mb / (mr + mg_ + mb + 1e-6)
        v = float(np.maximum.reduce([r, g, b]).mean())
        if (blue_score > 0.34 and v < 0.85 and mb >= max(mr, mg_) - 0.02):
            water_pix |= seg
    is_water_grid = water_pix[px_r, px_c]
    print(f"\nProjected water cells in raster grid: {is_water_grid.sum():,} of {is_water_grid.size:,} ({100*is_water_grid.sum()/is_water_grid.size:.1f}%)")

    axes[1, 1].imshow(is_water_grid, cmap="Blues", origin="lower",
                      extent=(minlon, maxlon, minlat, maxlat), aspect="auto")
    axes[1, 1].set_title(f"Final water mask projected to raster grid "
                          f"({100*is_water_grid.sum()/is_water_grid.size:.1f}%)")

    out = args.out or f"/tmp/sam2_debug_{args.slug}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
