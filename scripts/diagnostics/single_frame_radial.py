#!/usr/bin/env python3
"""Is the radial pattern visible in a single raw frame, before any projection?

Why this is worth doing separately
----------------------------------
`radial_paired_test.py` measures the radial bias by projecting every pixel to a
ground cell and comparing a cell against itself across views. That design
removes scene structure, but it BUYS that with a geometric assumption: pixel
ground positions come from bilinear interpolation of four footprint corners on
a flat sea surface, with no lens-distortion model. The thermal lens has barrel
distortion, so edge pixels are assigned to ground cells they did not actually
come from. Where the water has a real temperature gradient, systematic
mis-assignment at large image radius could manufacture a radius-dependent
signal that is not sensor bias at all.

This script has no projection, no ground grid, and no pairing. It looks at raw
frames in image coordinates only:

    residual = T(x) - median(T over ocean in that frame)

then averages azimuthally to get a radial profile, restricted to ocean pixels.
If the paired result is real, the same centre-cold ramp must appear here. If it
appears only after projection, the projection is the more likely source.

Scene structure is the cost of dropping the pairing, so it is controlled two
ways. Frames are ranked by ocean fraction and only the most ocean-dominated are
used, which minimises the coast gradient. And profiles are combined across many
frames with varied headings, so any residual scene gradient averages toward
zero while an image-fixed pattern accumulates.

The single most ocean-dominated frame is reported on its own, which answers the
question directly: is it visible by eye in ONE image, or only in a stack?

On the camera's internal correction
-----------------------------------
The Autel 640T runs shutter-based non-uniformity correction. That removes
fixed-pattern noise and offset drift referenced to a uniform shutter, and it is
why raw frames look flat rather than obviously vignetted. What it does not fully
remove is the term that drifts BETWEEN shutter events: detector self-heating
and the narcissus effect, where the lens reflects the cold detector back onto
itself, both of which are radially structured and scene-temperature dependent.
So "the camera already corrects this" and "a residual radial ramp remains" are
not in conflict, and only measurement settles which regime these data are in.

Usage
-----
    python scripts/diagnostics/single_frame_radial.py \\
        --data data/flight4_vaihu_east_full_combined \\
        --label flight4_vaihu_east_full --n-candidates 200 --n-use 60 \\
        --output sgd_output/diagnostics/single_frame_radial_flight4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THERMAL = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THERMAL))

from sgd_toolkit.detectors import spread as spread_mod  # noqa: E402
from sgd_toolkit.detectors.spread import SpreadSGDDetector  # noqa: E402


def sample_blocks(nums, n_blocks, block_len):
    total = len(nums)
    if n_blocks * block_len >= total:
        return list(nums)
    starts = np.linspace(0, total - block_len, n_blocks).round().astype(int)
    out = []
    for s in sorted(set(starts.tolist())):
        out.extend(nums[s:s + block_len])
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(
        description="Radial profile of raw frames in image space, no projection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-candidates", type=int, default=200,
                    help="Frames to scan for ocean fraction")
    ap.add_argument("--n-use", type=int, default=60,
                    help="Most ocean-dominated frames to keep for the stack")
    ap.add_argument("--min-ocean-frac", type=float, default=0.55)
    ap.add_argument("--r-bins", type=int, default=8)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta-c", type=float, default=0.25)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data)
    label = args.label or data_dir.name
    nums = sorted(
        n for n in (int(p.name[4:8]) for p in data_dir.glob("MAX_*.JPG") if p.name[4:8].isdigit())
        if (data_dir / f"IRX_{n:04d}.irg").exists()
    )
    if not nums:
        raise SystemExit(f"No paired frames in {data_dir}. Volume mounted?")
    cand = sample_blocks(nums, 10, max(1, args.n_candidates // 10))

    detector = SpreadSGDDetector(base_path=str(data_dir), use_ml=False)
    captured = {}
    orig = spread_mod.refine_ocean_with_thermal

    def cap(masks, rgb, thermal, **kw):
        rr = orig(masks, rgb, thermal, **kw)
        captured["ocean"] = rr.masks.get("ocean")
        return rr

    spread_mod.refine_ocean_with_thermal = cap

    print(f"{label}: scanning {len(cand)} frames for ocean-dominated views")

    recs = []
    geom = {}
    consecutive = 0
    try:
        for i, fn in enumerate(cand, start=1):
            try:
                data = detector.load_frame_data(fn)
                consecutive = 0
            except Exception as e:
                consecutive += 1
                if consecutive >= 15:
                    raise SystemExit(f"ABORT: 15 consecutive load failures at {fn} "
                                     f"({type(e).__name__}: {e}). Volume dropped. "
                                     f"Nothing written.")
                continue
            thermal = data["thermal"].astype(np.float32)
            if not geom:
                h, w = thermal.shape
                yy = (np.arange(h) - (h - 1) / 2.0) / ((h - 1) / 2.0)
                xx = (np.arange(w) - (w - 1) / 2.0) / ((w - 1) / 2.0)
                X, Y = np.meshgrid(xx, yy)
                R = np.sqrt(X * X + Y * Y).ravel()
                edges = np.linspace(0.0, 1.0, args.r_bins + 1)
                rb = np.digitize(R, edges) - 1
                geom = {"rb": np.where(R <= 1.0, rb, -1), "edges": edges,
                        "centers": 0.5 * (edges[:-1] + edges[1:])}

            captured.pop("ocean", None)
            masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
            detector.detect_sgd_plumes(thermal, masks)
            ocean = captured.get("ocean")
            if ocean is None:
                ocean = masks.get("ocean")
            if ocean is None or not ocean.any():
                continue
            ocean = np.asarray(ocean, dtype=bool)
            frac = float(ocean.mean())
            if frac < args.min_ocean_frac:
                continue

            t = thermal.ravel().astype(np.float64)
            om = ocean.ravel() & np.isfinite(t) & (geom["rb"] >= 0)
            if om.sum() < 5000:
                continue
            res = t - float(np.median(t[om]))
            prof = np.full(args.r_bins, np.nan)
            rb = geom["rb"][om]
            rv = res[om]
            for b in range(args.r_bins):
                sel = rb == b
                if sel.sum() >= 200:
                    prof[b] = float(np.median(rv[sel]))
            if np.isfinite(prof).sum() < args.r_bins - 1:
                continue
            recs.append({"frame": fn, "ocean_frac": frac, "prof": prof})
            if i % 50 == 0:
                print(f"  {i}/{len(cand)} scanned, {len(recs)} usable")
    finally:
        spread_mod.refine_ocean_with_thermal = orig

    if len(recs) < 15:
        raise SystemExit(f"Only {len(recs)} ocean-dominated frames "
                         f"(>= {args.min_ocean_frac:.0%} ocean). Lower --min-ocean-frac.")

    recs.sort(key=lambda r: -r["ocean_frac"])
    use = recs[: args.n_use]
    P = np.array([r["prof"] for r in use])
    centers = geom["centers"]

    stack = np.nanmedian(P, axis=0)
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.bootstrap, args.r_bins))
    for b in range(args.bootstrap):
        s = rng.integers(0, len(use), size=len(use))
        boot[b] = np.nanmedian(P[s], axis=0)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

    inner = centers < 0.4
    outer = centers > 0.7
    contrast = float(np.nanmean(stack[inner]) - np.nanmean(stack[outer]))
    bc = np.nanmean(boot[:, inner], axis=1) - np.nanmean(boot[:, outer], axis=1)
    clo, chi = [float(x) for x in np.nanpercentile(bc, [2.5, 97.5])]

    best = use[0]
    bp = best["prof"]
    best_contrast = float(np.nanmean(bp[inner]) - np.nanmean(bp[outer]))

    # How consistent is the sign frame to frame? A sensor pattern should be
    # present in nearly every frame, not just on average.
    per_frame = np.nanmean(P[:, inner], axis=1) - np.nanmean(P[:, outer], axis=1)
    frac_neg = float(np.mean(per_frame < 0))

    print(f"\n=== {label}: raw-frame radial profile, no projection ===")
    print(f"  frames used {len(use)} of {len(recs)} ocean-dominated "
          f"(ocean fraction {use[-1]['ocean_frac']:.2f}-{use[0]['ocean_frac']:.2f})")
    print(f"\n  {'radius':>7}  {'stacked °C':>11}  {'95% CI':>20}  {'single frame':>13}")
    for i in range(args.r_bins):
        print(f"  {centers[i]:>7.2f}  {stack[i]:>+11.3f}  [{lo[i]:+.3f}, {hi[i]:+.3f}]"
              f"  {bp[i]:>+13.3f}")
    print(f"\n  STACKED centre − edge: {contrast:+.3f} °C (95% CI {clo:+.3f}, {chi:+.3f})")
    print(f"  SINGLE best frame ({best['frame']}, {best['ocean_frac']:.0%} ocean): "
          f"{best_contrast:+.3f} °C")
    print(f"  frames with negative (centre-cold) contrast: {frac_neg:.0%} of {len(use)}")
    print(f"  detection threshold delta_c = {args.delta_c:g} °C")

    if abs(best_contrast) > 0.5 * args.delta_c:
        print(f"\n  Visible in a SINGLE frame: the camera's internal NUC is not "
              f"removing this.")
    elif not (clo < 0 < chi):
        print(f"\n  Not reliable in one frame, but resolved in the stack. Consistent "
              f"with a real but small image-fixed term that per-frame scene noise "
              f"hides individually.")
    else:
        print(f"\n  No image-fixed radial term detectable without projection. If the "
              f"paired test still shows one, suspect the footprint projection "
              f"(lens distortion at large radius) rather than the sensor.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    jp = Path(args.output).with_suffix(".json")
    jp.write_text(json.dumps({
        "label": label, "frames_used": len(use),
        "ocean_fraction_range": [use[-1]["ocean_frac"], use[0]["ocean_frac"]],
        "r_centers": centers.tolist(),
        "stacked_profile_c": stack.tolist(),
        "stacked_ci_lo_c": lo.tolist(), "stacked_ci_hi_c": hi.tolist(),
        "stacked_contrast_c": contrast, "stacked_contrast_ci_c": [clo, chi],
        "best_frame": int(best["frame"]),
        "best_frame_ocean_frac": best["ocean_frac"],
        "best_frame_profile_c": bp.tolist(),
        "best_frame_contrast_c": best_contrast,
        "fraction_frames_centre_cold": frac_neg,
        "delta_c": args.delta_c,
    }, indent=2))
    print(f"  wrote {jp}")


if __name__ == "__main__":
    main()
