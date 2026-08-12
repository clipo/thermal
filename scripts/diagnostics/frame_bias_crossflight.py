#!/usr/bin/env python3
"""Test whether the image-fixed bias reproduces across independent flights.

`frame_position_bias.py` splits each flight's residual map into an image-fixed
component (candidate sensor bias) and a ground-fixed component (coast
structure). That decomposition rests on an assumption: that a radial vignette
is invariant under a 180-degree image rotation while a coast gradient reverses.
This script tests the conclusion a different way, without relying on that
assumption at all.

The same camera flew every survey. So:

  - A real sensor bias is a property of the detector. Its map must reproduce
    across flights over completely different coastline, in different sea
    states, on different days.
  - Coast structure is a property of each site. Its map must NOT reproduce.

Pairwise correlations of the sensor maps, set against the same correlations of
the scene maps, therefore separate the two directly. High sensor correlation
with low scene correlation confirms an instrumental bias and gives its pooled
magnitude. Sensor correlations near zero mean the per-flight decomposition was
fitting noise, and the vignette concern is not supported by these data.

Usage
-----
    python scripts/diagnostics/frame_bias_crossflight.py \\
        sgd_output/diagnostics/frame_position_bias_*.json \\
        --delta-c 0.25 \\
        --output sgd_output/diagnostics/frame_bias_crossflight
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


def as_map(nested) -> np.ndarray:
    """Nested lists with nulls -> float array with NaN."""
    return np.array(
        [[np.nan if v is None else float(v) for v in row] for row in nested],
        dtype=np.float64,
    )


def nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def amplitude(m: np.ndarray) -> dict:
    f = m[np.isfinite(m)]
    if f.size < 4:
        return {"std_c": float("nan"), "p5_p95_range_c": float("nan"), "max_abs_c": float("nan")}
    lo, hi = np.percentile(f, [5, 95])
    return {
        "std_c": float(f.std()),
        "p5_p95_range_c": float(hi - lo),
        "max_abs_c": float(np.abs(f).max()),
    }


def radial_profile(m: np.ndarray, n_bins: int = 8):
    """Collapse a block map to a normalised-radius profile."""
    by, bx = m.shape
    v = (np.arange(by) - (by - 1) / 2.0) / ((by - 1) / 2.0)
    u = (np.arange(bx) - (bx - 1) / 2.0) / ((bx - 1) / 2.0)
    uu, vv = np.meshgrid(u, v)
    r = np.sqrt(uu * uu + vv * vv)
    edges = np.linspace(0.0, r.max() + 1e-9, n_bins + 1)
    centers, vals = [], []
    for i in range(n_bins):
        sel = (r >= edges[i]) & (r < edges[i + 1]) & np.isfinite(m)
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        vals.append(float(np.median(m[sel])) if sel.sum() >= 3 else float("nan"))
    return np.array(centers), np.array(vals)


def main():
    ap = argparse.ArgumentParser(
        description="Correlate per-flight sensor and scene maps across flights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("json", nargs="+", help="frame_position_bias_*.json files")
    ap.add_argument("--delta-c", type=float, default=0.25,
                    help="Detection threshold to compare the pooled amplitude against")
    ap.add_argument("--output", default=None, help="Output base path for the JSON summary")
    args = ap.parse_args()

    flights = []
    for p in args.json:
        d = json.loads(Path(p).read_text())
        hd = d.get("heading_decomposition")
        if not hd:
            print(f"  skip {d.get('label', p)}: no heading decomposition "
                  f"(one leg only, or headings unavailable)")
            continue
        flights.append({
            "label": d["label"],
            "frames_used": d["frames_used"],
            "sensor": as_map(hd["sensor_map_c"]),
            "scene": as_map(hd["scene_map_c"]),
            "corr_a_b": hd["corr_a_b"],
            "corr_a_rot180_b": hd["corr_a_rot180_b"],
            "sensor_amplitude": hd["sensor_amplitude"],
        })

    if len(flights) < 2:
        raise SystemExit(
            f"Need at least 2 flights with a heading decomposition; got {len(flights)}."
        )

    shapes = {f["sensor"].shape for f in flights}
    if len(shapes) != 1:
        raise SystemExit(f"Block-map shapes differ across flights: {shapes}. "
                         f"Re-run with the same --block.")

    print(f"\n{len(flights)} flights: " + ", ".join(f['label'] for f in flights))
    print("\nPer-flight, within-flight heading evidence:")
    print(f"  {'flight':<34} {'n':>5}  corr(A,B)  corr(A,rot180 B)  sensor p5-p95")
    for f in flights:
        print(f"  {f['label']:<34} {f['frames_used']:>5}  "
              f"{f['corr_a_b']:>+9.3f}  {f['corr_a_rot180_b']:>+16.3f}  "
              f"{f['sensor_amplitude']['p5_p95_range_c']:>12.3f} °C")

    sensor_pairs, scene_pairs = [], []
    print("\nCross-flight reproducibility (the decisive comparison):")
    print(f"  {'pair':<48} {'sensor r':>9} {'scene r':>9}")
    for a, b in combinations(flights, 2):
        rs = nan_corr(a["sensor"], b["sensor"])
        rg = nan_corr(a["scene"], b["scene"])
        sensor_pairs.append(rs)
        scene_pairs.append(rg)
        print(f"  {a['label'][:22]:<23}/{b['label'][:22]:<24} {rs:>+9.3f} {rg:>+9.3f}")

    sp = np.array(sensor_pairs, dtype=float)
    gp = np.array(scene_pairs, dtype=float)

    stack = np.stack([f["sensor"] for f in flights])
    pooled = np.nanmedian(stack, axis=0)
    pooled -= np.nanmedian(pooled)
    amp = amplitude(pooled)
    centers, prof = radial_profile(pooled)

    print(f"\n  mean sensor-map correlation: {np.nanmean(sp):+.3f} "
          f"(range {np.nanmin(sp):+.3f} to {np.nanmax(sp):+.3f})")
    print(f"  mean scene-map  correlation: {np.nanmean(gp):+.3f} "
          f"(range {np.nanmin(gp):+.3f} to {np.nanmax(gp):+.3f})")

    print(f"\nPooled image-fixed map across flights:")
    print(f"  p5-p95 range {amp['p5_p95_range_c']:.3f} °C, "
          f"max |bias| {amp['max_abs_c']:.3f} °C, std {amp['std_c']:.3f} °C")
    print(f"  detection threshold delta_c = {args.delta_c:g} °C  "
          f"-> pooled amplitude is {amp['p5_p95_range_c']/args.delta_c:.2f}x the threshold")
    print(f"  radial profile (centre -> corner), °C:")
    for c, v in zip(centers, prof):
        bar = "" if not np.isfinite(v) else ("#" * max(1, int(abs(v) / 0.02)))
        print(f"    r={c:.2f}  {v:+.3f}  {bar}")

    verdict = []
    if np.nanmean(sp) > 0.5 and np.nanmean(sp) > np.nanmean(gp) + 0.3:
        verdict.append("Image-fixed bias REPRODUCES across flights: instrumental, not scene.")
    elif np.nanmean(sp) < 0.2:
        verdict.append("Image-fixed component does NOT reproduce across flights. "
                       "The per-flight decomposition is consistent with noise; these data "
                       "do not support a vignette large enough to matter.")
    else:
        verdict.append("Ambiguous: sensor correlation is neither clearly high nor near zero. "
                       "More flights or more frames per flight are needed.")
    if amp["p5_p95_range_c"] < 0.5 * args.delta_c:
        verdict.append(f"Pooled amplitude ({amp['p5_p95_range_c']:.3f} °C) is under half the "
                       f"{args.delta_c:g} °C threshold.")
    print("\n" + "\n".join("  " + v for v in verdict))

    if args.output:
        out = Path(args.output).with_suffix(".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "flights": [f["label"] for f in flights],
            "frames_used": [f["frames_used"] for f in flights],
            "per_flight_corr_a_b": [f["corr_a_b"] for f in flights],
            "per_flight_corr_a_rot180_b": [f["corr_a_rot180_b"] for f in flights],
            "sensor_pair_correlations": sp.tolist(),
            "scene_pair_correlations": gp.tolist(),
            "mean_sensor_correlation": float(np.nanmean(sp)),
            "mean_scene_correlation": float(np.nanmean(gp)),
            "pooled_sensor_amplitude": amp,
            "pooled_radial_centers": centers.tolist(),
            "pooled_radial_profile_c": prof.tolist(),
            "delta_c": args.delta_c,
            "verdict": verdict,
        }, indent=2))
        print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
