#!/usr/bin/env python3
"""SAM2-based water mask derivation. Replaces the HSV thresholding in
derive_water_mask.py with SAM2 automatic mask generation, which is much
more robust at sites with shallow tropical water, surf foam, sand bars,
and rocky shoreline (e.g., Vaihu Harbor) where simple color thresholds
fail.

Workflow per flight:
  1. Fetch Esri WorldImagery tiles for the raster bbox (contextily)
  2. Run SAM2AutomaticMaskGenerator to produce instance masks
  3. Classify each mask as water vs land using mean HSV statistics
     within the mask (much more reliable when applied to coherent
     SAM2 segments than to individual pixels)
  4. Project from web mercator → raster grid; save as
     <slug>_water_mask.npz (compatible with downstream scripts)

Runs on Apple Silicon GPU via PyTorch MPS backend automatically.

Setup:
    pip install sam2 contextily
    # SAM2-tiny weights (~150 MB) auto-download on first use, or:
    huggingface-cli download facebook/sam2-hiera-tiny

Usage:
    python scripts/derive_water_mask_sam2.py --slug vaihu_full
    python scripts/derive_water_mask_sam2.py --all --force
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_sam2(model_size: str = "tiny"):
    """Load SAM2AutomaticMaskGenerator from HuggingFace on the best available device."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = get_device()
    print(f"  SAM2 device: {device}")

    hf_id = {
        "tiny":  "facebook/sam2.1-hiera-tiny",
        "small": "facebook/sam2.1-hiera-small",
        "base":  "facebook/sam2.1-hiera-base-plus",
        "large": "facebook/sam2.1-hiera-large",
    }[model_size]

    mask_generator = SAM2AutomaticMaskGenerator.from_pretrained(
        hf_id,
        device=device,
        points_per_side=16,          # was 32 — 4× speedup
        pred_iou_thresh=0.75,
        stability_score_thresh=0.88,
        crop_n_layers=0,
        min_mask_region_area=400,    # px — drop noise
        multimask_output=False,      # 1 mask per prompt instead of 3
    )
    return mask_generator


def fetch_satellite_rgb(min_lon: float, min_lat: float,
                         max_lon: float, max_lat: float,
                         zoom: int = 16):
    import contextily as ctx
    src = ctx.providers.Esri.WorldImagery
    img, ext = ctx.bounds2img(min_lon, min_lat, max_lon, max_lat,
                               zoom=zoom, source=src, ll=True)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img.astype(np.uint8), ext


def classify_mask_water(rgb: np.ndarray, seg: np.ndarray,
                         min_water_pixel_frac: float = 0.55) -> tuple[bool, dict]:
    """Decide if a SAM2 mask region is water using per-pixel HSV votes.

    For each pixel in the segment, run a strict per-pixel water test
    (same as derive_water_mask.py HSV classifier). Classify the whole
    segment as water if the fraction of water pixels exceeds
    min_water_pixel_frac (default 55%).

    This is more robust than mean-color classification because:
      - Mean color of mixed surf/sand/water may pass; per-pixel won't
      - Gray cliff face has uniform color; mean might pass blue_score
        but per-pixel fails saturation test
    """
    if not seg.any():
        return False, {"n_px": 0}
    rgbf = (rgb.astype(np.float32) / 255.0)[seg]  # (N, 3)
    r = rgbf[:, 0]; g = rgbf[:, 1]; b = rgbf[:, 2]
    blue_score = b / (r + g + b + 1e-6)
    v = np.maximum.reduce([r, g, b])
    s = (v - np.minimum.reduce([r, g, b])) / (v + 1e-6)
    is_water_px = (
        (blue_score > 0.36)
        & (v < 0.78)
        & (s > 0.10)
        & (b >= np.maximum(r, g) - 5.0 / 255.0)
    )
    water_frac = float(is_water_px.mean())
    is_water_seg = water_frac >= min_water_pixel_frac
    return is_water_seg, {
        "water_frac": water_frac,
        "n_px": int(seg.sum()),
    }


def build_for_slug(slug: str, mask_generator, *, zoom: int = 16,
                   dilate_water: int = 3, force: bool = False) -> dict:
    spread = SGD_OUTPUT / f"{slug}_spread"
    npz_path = spread / f"{slug}_anomaly.npz"
    out_path = spread / f"{slug}_water_mask.npz"
    if out_path.exists() and not force:
        return {"slug": slug, "skipped": True}
    if not npz_path.exists():
        return {"slug": slug, "error": "no anomaly raster"}

    raster = np.load(npz_path)
    minlon = float(raster["bbox_min_lon"]); maxlon = float(raster["bbox_max_lon"])
    minlat = float(raster["bbox_min_lat"]); maxlat = float(raster["bbox_max_lat"])
    grid_res = float(raster["grid_resolution_m"])
    gy, gx = raster["anomaly"].shape

    img, ext = fetch_satellite_rgb(minlon, minlat, maxlon, maxlat, zoom=zoom)
    H, W = img.shape[:2]
    print(f"  satellite tile: {W}×{H} px, generating SAM2 masks…")

    masks = mask_generator.generate(img)
    print(f"  {len(masks)} SAM2 masks; classifying as water/land…")

    water_pix = np.zeros((H, W), dtype=bool)
    n_water_masks = 0
    for m in masks:
        seg = m["segmentation"]
        is_water, _stats = classify_mask_water(img, seg)
        if is_water:
            water_pix |= seg
            n_water_masks += 1
    print(f"  {n_water_masks} masks classified as water "
          f"({100*water_pix.sum()/water_pix.size:.1f}% of tile area)")

    # Project from satellite (web mercator) → raster grid (lat/lon)
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
    is_water = water_pix[px_r, px_c]

    if dilate_water > 0:
        try:
            from scipy import ndimage
            is_water = ndimage.binary_dilation(is_water, iterations=dilate_water)
        except ImportError:
            pass

    np.savez_compressed(
        out_path,
        is_water=is_water,
        bbox_min_lon=minlon, bbox_max_lon=maxlon,
        bbox_min_lat=minlat, bbox_max_lat=maxlat,
        grid_resolution_m=grid_res,
        zoom=zoom, dilate_water=dilate_water,
        method="sam2",
    )
    return {
        "slug": slug, "out": str(out_path),
        "water_cells": int(is_water.sum()),
        "total_cells": int(is_water.size),
        "water_frac": float(is_water.sum()) / float(is_water.size),
        "n_sam_masks": len(masks),
        "n_water_masks": n_water_masks,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--zoom", type=int, default=16)
    ap.add_argument("--dilate-water", type=int, default=3)
    ap.add_argument("--model-size", choices=("tiny", "small", "base", "large"),
                    default="tiny")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print("Loading SAM2…")
    mask_generator = load_sam2(args.model_size)

    if args.slug:
        slugs = [args.slug]
    else:
        slugs = []
        for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
            slug = spread_dir.name[: -len("_spread")]
            if (spread_dir / f"{slug}_anomaly.npz").exists():
                slugs.append(slug)

    print(f"Building SAM2 water masks for {len(slugs)} flight(s)…")
    for slug in slugs:
        try:
            r = build_for_slug(slug, mask_generator, zoom=args.zoom,
                                dilate_water=args.dilate_water,
                                force=args.force)
            if r.get("skipped"):
                print(f"  → {slug}: already present (--force to rebuild)")
            elif "error" in r:
                print(f"  ✗ {slug}: {r['error']}")
            else:
                print(f"  ✓ {slug}: {100*r['water_frac']:.1f}% water "
                      f"(from {r['n_water_masks']}/{r['n_sam_masks']} SAM masks)")
        except Exception as e:
            print(f"  ✗ {slug}: {e}")


if __name__ == "__main__":
    main()
