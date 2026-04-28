#!/usr/bin/env python3
"""Diagnostic: render RGB + thermal + ocean-mask overlay for one frame so we
can SEE whether cliff shadows are being misclassified as ocean.

Usage:
    python scripts/diagnose_ocean_mask.py \\
        --data data/june2023_2_july_23_poike_3_combined --frame 600
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sgd_toolkit.detectors import IntegratedSGDDetector
from sgd_toolkit.segmentation.thermal_refine import refine_ocean_with_thermal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    det = IntegratedSGDDetector(base_path=args.data, use_ml=False)
    data = det.load_frame_data(args.frame)
    rgb = data["rgb_aligned"]
    thermal = data["thermal"].astype(np.float32)
    masks_raw = det.segment_ocean_land_waves(rgb)
    try:
        rr = refine_ocean_with_thermal(masks_raw, rgb, thermal)
        masks_ref = rr.masks
    except Exception:
        masks_ref = masks_raw

    ocean_raw = masks_raw.get("ocean", np.zeros(thermal.shape, dtype=bool))
    ocean_ref = masks_ref.get("ocean", np.zeros(thermal.shape, dtype=bool))

    # Compute anomaly using refined ocean (matches build_anomaly_raster.py)
    if ocean_ref.any():
        ocean_t = thermal[ocean_ref]
        ocean_t = ocean_t[np.isfinite(ocean_t)]
        baseline = float(np.percentile(ocean_t, 75)) if ocean_t.size > 50 else float("nan")
    else:
        baseline = float("nan")

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB"); axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(thermal, cmap="RdYlBu_r")
    axes[0, 1].set_title(f"Thermal (°C)  baseline={baseline:.2f}°C")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.7)

    overlay_raw = rgb.copy()
    overlay_raw[ocean_raw] = (overlay_raw[ocean_raw] * 0.4 +
                              np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    axes[0, 2].imshow(overlay_raw)
    axes[0, 2].set_title(f"RGB ocean mask (raw seg) — {ocean_raw.sum()} px")
    axes[0, 2].axis("off")

    overlay_ref = rgb.copy()
    overlay_ref[ocean_ref] = (overlay_ref[ocean_ref] * 0.4 +
                              np.array([0, 0, 200]) * 0.6).astype(np.uint8)
    axes[1, 0].imshow(overlay_ref)
    axes[1, 0].set_title(f"RGB + thermal-refined ocean mask — {ocean_ref.sum()} px")
    axes[1, 0].axis("off")

    # Show the per-frame anomaly that this generates
    if np.isfinite(baseline):
        anomaly = np.where(ocean_ref, np.maximum(0, baseline - thermal), np.nan)
    else:
        anomaly = np.full_like(thermal, np.nan)
    im2 = axes[1, 1].imshow(anomaly, cmap="YlOrRd", vmin=0, vmax=2.0)
    axes[1, 1].set_title("Per-frame anomaly (°C below baseline)\n"
                          "in ocean-mask cells only — this is what's averaged")
    axes[1, 1].axis("off")
    fig.colorbar(im2, ax=axes[1, 1], shrink=0.7)

    # Highlight cells of refined-ocean that the raw ocean did NOT include
    # (these are pixels added by thermal_refine — could be surf zone OR cliff)
    added = ocean_ref & ~ocean_raw
    overlay_added = rgb.copy()
    overlay_added[added] = (overlay_added[added] * 0.4 +
                            np.array([200, 0, 0]) * 0.6).astype(np.uint8)
    axes[1, 2].imshow(overlay_added)
    axes[1, 2].set_title(f"Cells thermal_refine ADDED to ocean — {added.sum()} px\n"
                          "Should be surf zone, not cliff shadow")
    axes[1, 2].axis("off")

    out = args.out or f"/tmp/ocean_mask_diag_frame{args.frame}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")
    print(f"  ocean_raw = {ocean_raw.sum()} px ({100*ocean_raw.sum()/ocean_raw.size:.1f}%)")
    print(f"  ocean_ref = {ocean_ref.sum()} px ({100*ocean_ref.sum()/ocean_ref.size:.1f}%)")
    print(f"  added by thermal_refine = {added.sum()} px")
    print(f"  baseline_75pct (used for anomaly) = {baseline:.2f}°C")


if __name__ == "__main__":
    main()
