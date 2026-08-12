#!/usr/bin/env python3
"""Paired within-ground-cell test for image-radius-dependent bias.

The question
------------
Does a pixel read colder because of WHERE IT SITS IN THE FRAME, independently
of what is on the ground underneath it?

Why this design
---------------
Comparing frame centre against frame edge across many frames is confounded.
Ocean pixels are not uniformly distributed in the image: the coastline is
framed systematically and nearshore water is genuinely colder, so "centre is
colder" has an innocent explanation that a pooled comparison cannot exclude.

Frame overlap removes the confound entirely. The same patch of ocean is imaged
near the frame centre in one frame and near the edge in another. Comparing a
ground cell against ITSELF across those views holds the ground truth fixed:
coastal geometry, nearshore cooling, and real discharge plumes are properties
of the cell and cancel in the within-cell contrast. What survives is variation
that tracks image position alone.

For each ocean pixel we record

    residual = T - P75(ocean temperatures in that frame)     [removes per-frame
                                                              scalar offset]
    ground cell                                              [where it is]
    image radius bin                                         [where in the frame]

then average the residual per (cell, radius bin), subtract each cell's own mean
across the radius bins it was seen in, and average what is left over cells. A
flat profile means image position carries no information once the ground is
held fixed. A monotone profile is positional bias, in degC, directly comparable
to the delta_c = 0.25 degC detection threshold.

What the result includes
------------------------
This measures the total image-position-dependent effect, not the sensor
vignette alone. Off-nadir viewing at the frame edge lowers water emissivity and
also biases apparent temperature, in the opposite direction to a centre-cold
vignette. That is a feature rather than a limitation: what matters for
detection is the combined dependence on image position, whatever its cause.

Uncertainty
-----------
Neighbouring ground cells are not independent. Confidence intervals come from a
block bootstrap that resamples coarse spatial super-blocks (--boot-block-m),
not individual cells.

Usage
-----
    python scripts/diagnostics/radial_paired_test.py \\
        --data data/flight4_vaihu_east_full_combined \\
        --label flight4_vaihu_east_full --n-frames 120 \\
        --output sgd_output/diagnostics/radial_paired_flight4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

THERMAL = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THERMAL))

from sgd_toolkit.detectors import spread as spread_mod  # noqa: E402
from sgd_toolkit.detectors.spread import SpreadSGDDetector  # noqa: E402
from sgd_toolkit.georeferencing.polygon_georef import SGDPolygonGeoref  # noqa: E402
from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper  # noqa: E402


def elliptical_radius(shape):
    h, w = shape
    v = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    u = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    uu, vv = np.meshgrid(u, v)
    return np.sqrt(uu * uu + vv * vv)


def sample_blocks(nums: list[int], n_blocks: int, block_len: int) -> list[int]:
    """Contiguous runs of frames, evenly spaced across the flight.

    The paired design needs consecutive-frame OVERLAP so a ground cell is seen
    at several image radii, which a uniform stride across the flight destroys.
    A single contiguous block keeps the overlap but confines the sample to one
    leg and one time window, so heading, sun geometry and sea state are all
    near-constant. Several blocks spread across the flight give both.
    """
    total = len(nums)
    if n_blocks * block_len >= total:
        return list(nums)
    starts = np.linspace(0, total - block_len, n_blocks).round().astype(int)
    out: list[int] = []
    for s in sorted(set(starts.tolist())):
        out.extend(nums[s:s + block_len])
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(
        description="Within-ground-cell test for bias that depends on image position.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-frames", type=int, default=120,
                    help="How many frames to use. The paired design needs far fewer "
                         "than a pooled comparison; 100-150 is usually plenty.")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--step", type=int, default=1,
                    help="Stride, used only when --n-frames covers the whole flight. "
                         "Otherwise frames are sampled uniformly across the flight.")
    ap.add_argument("--cell-m", type=float, default=2.0, help="Ground cell size (m)")
    ap.add_argument("--r-bins", type=int, default=6)
    ap.add_argument("--min-obs-per-cell-bin", type=int, default=3,
                    help="Pixels needed before a (cell, radius bin) mean is used")
    ap.add_argument("--min-bins-per-cell", type=int, default=3,
                    help="Distinct radius bins a cell must be seen in to contribute. "
                         "This is what makes the comparison paired.")
    ap.add_argument("--boot-block-m", type=float, default=50.0,
                    help="Super-block size for the spatial block bootstrap")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-c", type=float, default=0.25)
    ap.add_argument("--baseline-pct", type=float, default=75.0)
    ap.add_argument("--n-blocks", type=int, default=6,
                    help="Contiguous frame blocks spread across the flight")
    ap.add_argument("--block-len", type=int, default=25,
                    help="Consecutive frames per block. Must stay large enough that frames within a block overlap on the ground.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data)
    label = args.label or data_dir.name
    nums = sorted(
        n for n in (int(p.name[4:8]) for p in data_dir.glob("MAX_*.JPG") if p.name[4:8].isdigit())
        if (data_dir / f"IRX_{n:04d}.irg").exists()
    )
    if not nums:
        raise SystemExit(f"No paired frames in {data_dir}")
    if args.start is not None:
        nums = [n for n in nums if n >= args.start]
    frames = sample_blocks(nums, args.n_blocks, args.block_len)

    probe = data_dir / f"MAX_{frames[0]:04d}.JPG"
    if not probe.exists() or probe.stat().st_size == 0:
        raise SystemExit(f"{probe} does not resolve. External volume not mounted?")

    print(f"{label}: paired radial test on {len(frames)} frames "
          f"[{frames[0]}..{frames[-1]}], {args.cell_m} m cells")

    detector = SpreadSGDDetector(base_path=str(data_dir), use_ml=False,
                                 baseline_percentile_ocean=args.baseline_pct)
    georef = SGDPolygonGeoref(base_path=str(data_dir))
    mapper = ThermalFrameMapper(data_dir=str(data_dir), output_base="_tmp", verbose=False)

    captured = {}
    orig_refine = spread_mod.refine_ocean_with_thermal

    def capturing_refine(masks, rgb, thermal, **kw):
        rr = orig_refine(masks, rgb, thermal, **kw)
        captured["ocean"] = rr.masks.get("ocean")
        return rr

    spread_mod.refine_ocean_with_thermal = capturing_refine

    # Pass 1: footprints, to fix the ground grid.
    recs = []
    for fn in frames:
        try:
            g = georef.extract_gps(str(data_dir / f"MAX_{fn:04d}.JPG"))
            if not g or "lat" not in g:
                continue
            corners = mapper.calculate_footprint_corners(
                lat=float(g["lat"]), lon=float(g["lon"]),
                altitude=float(g.get("altitude", 350)),
                heading=float(g.get("heading") or 0.0),
            )
            recs.append({"frame": fn, "corners": corners})
        except Exception:
            continue
    if len(recs) < 20:
        raise SystemExit(f"Only {len(recs)} frames with usable GPS; need 20+.")

    lons = np.array([c[0] for r in recs for c in r["corners"][:4]])
    lats = np.array([c[1] for r in recs for c in r["corners"][:4]])
    minlon, maxlon = float(lons.min()), float(lons.max())
    minlat, maxlat = float(lats.min()), float(lats.max())
    clat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(clat))
    gx = int(math.ceil((maxlon - minlon) * mpd_lon / args.cell_m)) + 1
    gy = int(math.ceil((maxlat - minlat) * mpd_lat / args.cell_m)) + 1
    n_cells = gx * gy
    print(f"  ground grid {gy}x{gx} = {n_cells} cells of {args.cell_m} m")

    nb = args.r_bins
    r_edges = np.linspace(0.0, 1.0, nb + 1)
    res_sum = np.zeros(n_cells * nb, dtype=np.float64)
    res_cnt = np.zeros(n_cells * nb, dtype=np.int32)

    radius = None
    r_bin_flat = None
    n_used = 0
    consecutive_fail = 0

    try:
        for i, rec in enumerate(recs, start=1):
            fn = rec["frame"]
            try:
                data = detector.load_frame_data(fn)
                consecutive_fail = 0
            except Exception as e:
                consecutive_fail += 1
                if consecutive_fail >= 15:
                    raise SystemExit(
                        f"ABORT: 15 consecutive load failures at frame {fn} "
                        f"({type(e).__name__}: {e}). Volume unmounted? Nothing written."
                    )
                continue

            thermal = data["thermal"].astype(np.float32)
            if radius is None:
                radius = elliptical_radius(thermal.shape)
                rb = np.digitize(radius, r_edges) - 1
                r_bin_flat = np.where(radius <= 1.0, rb, -1).ravel()

            captured.pop("ocean", None)
            masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
            _, _, chars = detector.detect_sgd_plumes(thermal, masks)
            ocean = captured.get("ocean")
            if ocean is None:
                ocean = masks.get("ocean")
            if ocean is None or not ocean.any():
                continue
            ocean = np.asarray(ocean, dtype=bool)

            baseline = chars.get("baseline_c") if isinstance(chars, dict) else None
            if baseline is None or not np.isfinite(baseline):
                continue
            residual = (thermal.astype(np.float64) - float(baseline)).ravel()

            H, W = thermal.shape
            BL, BR, TR, TL = rec["corners"][:4]
            u = np.linspace(0.0, 1.0, W)
            v = np.linspace(0.0, 1.0, H)
            uu, vv = np.meshgrid(u, v)
            plon = ((1 - uu) * (1 - vv) * TL[0] + uu * (1 - vv) * TR[0]
                    + uu * vv * BR[0] + (1 - uu) * vv * BL[0]).ravel()
            plat = ((1 - uu) * (1 - vv) * TL[1] + uu * (1 - vv) * TR[1]
                    + uu * vv * BR[1] + (1 - uu) * vv * BL[1]).ravel()

            col = ((plon - minlon) * mpd_lon / args.cell_m).astype(np.int64)
            row = ((plat - minlat) * mpd_lat / args.cell_m).astype(np.int64)

            ok = (ocean.ravel() & np.isfinite(residual) & (r_bin_flat >= 0)
                  & (col >= 0) & (col < gx) & (row >= 0) & (row < gy))
            if ok.sum() < 500:
                continue

            idx = (row[ok] * gx + col[ok]) * nb + r_bin_flat[ok]
            np.add.at(res_sum, idx, residual[ok])
            np.add.at(res_cnt, idx, 1)
            n_used += 1
            if i % 25 == 0:
                print(f"  {i}/{len(recs)} frames, {n_used} used")
    finally:
        spread_mod.refine_ocean_with_thermal = orig_refine

    if n_used < 20:
        raise SystemExit(f"Only {n_used} usable frames.")

    res_sum = res_sum.reshape(n_cells, nb)
    res_cnt = res_cnt.reshape(n_cells, nb)

    valid = res_cnt >= args.min_obs_per_cell_bin
    with np.errstate(invalid="ignore", divide="ignore"):
        cell_bin_mean = np.where(valid, res_sum / np.maximum(res_cnt, 1), np.nan)

    n_bins_seen = valid.sum(axis=1)
    keep = n_bins_seen >= args.min_bins_per_cell
    n_keep = int(keep.sum())
    print(f"  frames used {n_used}; ground cells contributing {n_keep} "
          f"(seen in >= {args.min_bins_per_cell} radius bins)")
    if n_keep < 200:
        raise SystemExit(
            f"Only {n_keep} paired cells. Increase --n-frames, enlarge --cell-m, "
            f"or lower --min-bins-per-cell."
        )

    cb = cell_bin_mean[keep]
    # Subtract each cell's own mean across the radius bins it was seen in.
    # This is the pairing: everything constant within a ground cell drops out.
    cell_mean = np.nanmean(cb, axis=1, keepdims=True)
    dev = cb - cell_mean

    profile = np.nanmean(dev, axis=0)
    per_bin_n = np.sum(np.isfinite(dev), axis=0)
    centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    # Spatial block bootstrap: resample coarse super-blocks of ground cells.
    cell_ids = np.flatnonzero(keep)
    crow, ccol = cell_ids // gx, cell_ids % gx
    bf = max(1, int(round(args.boot_block_m / args.cell_m)))
    bgx = gx // bf + 1
    block_id = (crow // bf) * bgx + (ccol // bf)
    ublocks, binv = np.unique(block_id, return_inverse=True)
    order = np.argsort(binv, kind="stable")
    starts = np.searchsorted(binv[order], np.arange(len(ublocks) + 1))
    members = [order[starts[i]:starts[i + 1]] for i in range(len(ublocks))]
    print(f"  block bootstrap: {len(ublocks)} blocks of {args.boot_block_m} m")

    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.bootstrap, nb))
    contrast = np.empty(args.bootstrap)
    inner, outer = centers < 0.4, centers > 0.7
    for b in range(args.bootstrap):
        pick = rng.integers(0, len(members), size=len(members))
        sel = np.concatenate([members[k] for k in pick])
        d = dev[sel]
        p = np.nanmean(d, axis=0)
        boot[b] = p
        contrast[b] = np.nanmean(p[inner]) - np.nanmean(p[outer])

    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    obs_contrast = float(np.nanmean(profile[inner]) - np.nanmean(profile[outer]))
    clo, chi = [float(x) for x in np.nanpercentile(contrast, [2.5, 97.5])]
    span = float(np.nanmax(profile) - np.nanmin(profile))

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "label": label,
        "data_dir": str(data_dir),
        "frames_used": n_used,
        "paired_ground_cells": n_keep,
        "cell_m": args.cell_m,
        "r_centers": centers.tolist(),
        "profile_c": profile.tolist(),
        "profile_ci_lo_c": lo.tolist(),
        "profile_ci_hi_c": hi.tolist(),
        "cells_per_bin": per_bin_n.tolist(),
        "inner_minus_outer_c": obs_contrast,
        "inner_minus_outer_ci_c": [clo, chi],
        "profile_span_c": span,
        "delta_c": args.delta_c,
        "bootstrap_blocks": int(len(ublocks)),
        "boot_block_m": args.boot_block_m,
    }
    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))

    print()
    print(f"=== {label}: residual vs image radius, ground held fixed ===")
    print(f"  {'radius':>7}  {'bias °C':>9}  {'95% CI':>20}   cells")
    for i in range(nb):
        print(f"  {centers[i]:>7.2f}  {profile[i]:>+9.3f}  "
              f"[{lo[i]:+.3f}, {hi[i]:+.3f}]   {per_bin_n[i]:>8d}")
    print(f"\n  centre − edge: {obs_contrast:+.3f} °C (95% CI {clo:+.3f}, {chi:+.3f})")
    print(f"  full profile span: {span:.3f} °C")
    print(f"  detection threshold delta_c = {args.delta_c:g} °C")

    # Report which criterion actually decided the verdict. A wide interval that
    # spans zero is NOT the same finding as a point estimate near zero: the
    # first is an underpowered sample, the second is evidence of no effect.
    ci_spans_zero = clo < 0 < chi
    ceiling = max(abs(clo), abs(chi))
    if ci_spans_zero:
        print(f"\n  Not resolved at this sample size. Point estimate "
              f"{obs_contrast:+.3f} °C ({abs(obs_contrast)/args.delta_c:.0%} of the "
              f"{args.delta_c:g} °C threshold), but the 95% CI [{clo:+.3f}, {chi:+.3f}] "
              f"includes zero.")
        print(f"  This bounds the effect rather than excluding it: the data are "
              f"consistent with anything up to {ceiling:.3f} °C "
              f"({ceiling/args.delta_c:.0%} of the threshold). Raise --n-frames "
              f"to tighten it.")
    else:
        print(f"\n  Image-position bias IS present: {abs(obs_contrast):.3f} °C is "
              f"{abs(obs_contrast)/args.delta_c:.0%} of the {args.delta_c:g} °C threshold "
              f"(95% CI {clo:+.3f}, {chi:+.3f}, excludes zero).")
    result["ci_spans_zero"] = bool(ci_spans_zero)
    result["effect_ceiling_c"] = float(ceiling)
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    main()
