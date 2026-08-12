#!/usr/bin/env python3
"""Separate sensor bias from scene structure in the thermal frame.

The question
------------
An uncooled microbolometer has an additive bias that is fixed in IMAGE
coordinates (the sensor centre self-heats and reads colder than the edges).
A scalar per-frame offset cancels exactly in this pipeline, because
SpreadSGDDetector thresholds against that frame's own 75th-percentile ocean
temperature. An intra-frame bias does not cancel: it adds directly to
(T - baseline) and competes with the delta_c = 0.25 degC threshold.

The difficulty is that "colder toward frame centre" has an innocent
explanation. Ocean pixels are not uniformly distributed in the image: the
coastline is framed systematically, and nearshore water is genuinely colder.
Simply comparing centre against edge cannot tell the two apart.

The discriminator: 180-degree legs
----------------------------------
Sensor bias is fixed in image coordinates. Ground structure is fixed in ground
coordinates, so when the aircraft flies the reciprocal leg of a transect its
image-space appearance rotates by 180 degrees.

Write the per-frame residual map for the two heading groups as

    A = S + G
    B = S + rot180(G)

with S the image-fixed sensor bias and G the ground pattern in group-A
orientation. A radial vignette is (near) invariant under rot180, while a coast
gradient across the frame roughly reverses sign. Under those conditions

    S ~ symmetric part of  (A + B) / 2      <- the sensor bias, in degC
    G ~ antisymmetric part of (A - B) / 2   <- the coast gradient, in degC

reported side by side against the 0.25 degC threshold. Two correlations,
corr(A, B) and corr(A, rot180(B)), say which model the data actually support
before the decomposition is read.

Independent cross-check: the same camera flew every survey, so a real S must
reproduce across flights over different coastline. Run several flights and
compare the saved `sensor_map_c` with scripts/diagnostics/frame_bias_crossflight.py.

What is measured, and at which stage
------------------------------------
Positional statistics use PLUME CENTROIDS from `plume_info`, i.e. the connected
components that survive `min_area` and are handed to georeferencing. That is
the stage where image position can still influence the published product.

The raw per-frame threshold mask is deliberately NOT the headline statistic. It
flags 35-78% of eligible ocean pixels, which is arithmetic rather than
malfunction: the baseline is the 75th percentile of ocean, so most of the ocean
sits below `P75 - delta_c` by construction. Those pixels are reduced to 415
island-wide plumes only by the downstream density-grid clustering. The radial
detection-rate profile is still reported, marked as the pre-clustering stage.

Frame-level resampling
----------------------
Pixels within a frame are strongly correlated, so all confidence intervals come
from a bootstrap that resamples FRAMES.

Usage
-----
    python scripts/diagnostics/frame_position_bias.py \\
        --data data/flight8_hekii_west_combined \\
        --label flight8_hekii_west --step 2 \\
        --output sgd_output/diagnostics/frame_position_bias_flight8_hekii_west
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


# --------------------------------------------------------------------------
# Frames and geometry
# --------------------------------------------------------------------------

def iter_frames(data_dir: Path, start: int, end: int, step: int) -> list[int]:
    return [
        n for n in range(start, end + 1, step)
        if (data_dir / f"MAX_{n:04d}.JPG").exists() and (data_dir / f"IRX_{n:04d}.irg").exists()
    ]


def frame_bounds(data_dir: Path) -> tuple[int, int]:
    nums = sorted(int(p.name[4:8]) for p in data_dir.glob("MAX_*.JPG") if p.name[4:8].isdigit())
    if not nums:
        raise SystemExit(f"No MAX_*.JPG frames in {data_dir}")
    return nums[0], nums[-1]


def elliptical_radius(shape: tuple[int, int]) -> np.ndarray:
    """0 at frame centre, 1.0 at the edge midpoints, sqrt(2) at the corners.

    Normalising each axis by its own half-extent keeps radial bins comparable
    in a non-square frame.
    """
    h, w = shape
    v = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    u = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    uu, vv = np.meshgrid(u, v)
    return np.sqrt(uu * uu + vv * vv)


def json_map(a: np.ndarray) -> list:
    """2D array -> nested lists with NaN rendered as null."""
    return [[(float(v) if np.isfinite(v) else None) for v in row] for row in a]


def rot180(a: np.ndarray) -> np.ndarray:
    """Image-space rotation by 180 degrees: what a yaw reversal does to a scene."""
    return a[::-1, ::-1]


def nan_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return float("nan")
    x, y = a[m], b[m]
    x = x - x.mean()
    y = y - y.mean()
    d = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _grouped_median(groups, values, n_groups, min_count=1):
    """Median of `values` per group id; NaN where a group has < min_count members."""
    out = np.full(n_groups, np.nan, dtype=np.float64)
    if groups.size == 0:
        return out
    order = np.argsort(groups, kind="stable")
    g, v = groups[order], values[order]
    edges = np.searchsorted(g, np.arange(n_groups + 1))
    for i in range(n_groups):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo >= min_count:
            out[i] = np.median(v[lo:hi])
    return out


# --------------------------------------------------------------------------
# Accumulation
# --------------------------------------------------------------------------

class Accumulator:
    def __init__(self, shape, n_rbins, block, min_block_px):
        self.shape = shape
        self.n_rbins = n_rbins
        self.block = block
        self.min_block_px = min_block_px

        self.radius = elliptical_radius(shape)
        self.r_edges = np.linspace(0.0, 1.0, n_rbins + 1)
        r_bin = np.digitize(self.radius, self.r_edges) - 1
        self.r_bin_flat = np.where(self.radius <= 1.0, r_bin, -1).ravel()

        h, w = shape
        self.by = int(np.ceil(h / block))
        self.bx = int(np.ceil(w / block))
        self.n_blocks = self.by * self.bx
        rows = (np.arange(h) // block)[:, None]
        cols = (np.arange(w) // block)[None, :]
        self.block_flat = (rows * self.bx + cols).astype(np.int32).ravel()

        self.frames: list[int] = []
        self.res_block: list[np.ndarray] = []
        self.elig_bin: list[np.ndarray] = []
        self.det_bin: list[np.ndarray] = []
        self.heading: list[float] = []
        self.ocean_frac: list[float] = []
        self.baseline_c: list[float] = []
        # Plume centroids in image coords, with area, per frame.
        self.cent_r: list[np.ndarray] = []
        self.cent_area: list[np.ndarray] = []

        self.leak_px = 0        # detections outside the ocean mask
        self.corner_ocean = 0
        self.total_ocean = 0

    def add(self, frame_no, ocean, sgd_mask, residual, centroids, areas, heading, baseline):
        ocean_f = ocean.ravel()
        res_f = residual.ravel()
        det_f = (sgd_mask & ocean).ravel()

        self.leak_px += int((sgd_mask.ravel() & ~ocean_f).sum())
        self.total_ocean += int(ocean_f.sum())
        self.corner_ocean += int((ocean_f & (self.r_bin_flat < 0)).sum())

        keep = ocean_f & np.isfinite(res_f)
        if keep.sum() < 500:
            return False
        keep_r = keep & (self.r_bin_flat >= 0)

        rb = self.r_bin_flat[keep_r]
        self.elig_bin.append(np.bincount(rb, minlength=self.n_rbins).astype(np.int64))
        self.det_bin.append(
            np.bincount(rb[det_f[keep_r]], minlength=self.n_rbins).astype(np.int64)
        )
        self.res_block.append(
            _grouped_median(self.block_flat[keep], res_f[keep], self.n_blocks, self.min_block_px)
        )

        self.frames.append(int(frame_no))
        self.heading.append(float(heading) if heading is not None else float("nan"))
        self.ocean_frac.append(float(ocean_f.mean()))
        self.baseline_c.append(float(baseline))
        self.cent_r.append(np.asarray(centroids, dtype=np.float64))
        self.cent_area.append(np.asarray(areas, dtype=np.float64))
        return True


# --------------------------------------------------------------------------
# Heading split
# --------------------------------------------------------------------------

def split_by_heading(headings: np.ndarray, halfwidth_deg: float):
    """Group frames into the two lobes of the dominant transect axis.

    The axis is the circular mean of the DOUBLED angles, which is insensitive
    to the 180-degree ambiguity of a back-and-forth transect. Frames within
    `halfwidth_deg` of each lobe form groups A and B; the rest (turns, cross
    legs) are unassigned.
    """
    ok = np.isfinite(headings)
    if ok.sum() < 20:
        return None, None, float("nan")
    th = np.deg2rad(headings[ok] % 360.0)
    axis = 0.5 * math.atan2(float(np.sin(2 * th).mean()), float(np.cos(2 * th).mean()))
    axis_deg = math.degrees(axis) % 360.0

    def near(target):
        d = np.abs((headings - target + 180.0) % 360.0 - 180.0)
        return np.isfinite(headings) & (d <= halfwidth_deg)

    a = np.flatnonzero(near(axis_deg))
    b = np.flatnonzero(near((axis_deg + 180.0) % 360.0))
    return a, b, axis_deg


def decompose(map_a: np.ndarray, map_b: np.ndarray):
    """Split two opposing-leg residual maps into sensor and scene components."""
    s0 = 0.5 * (map_a + map_b)
    d0 = 0.5 * (map_a - map_b)
    sensor = 0.5 * (s0 + rot180(s0))       # image-fixed, rot180-symmetric
    scene = 0.5 * (d0 - rot180(d0))        # ground-fixed, rot180-antisymmetric
    return sensor, scene


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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Separate image-fixed sensor bias from ground-fixed scene structure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--delta-c", type=float, default=0.25, help="Production spread threshold")
    ap.add_argument("--min-area", type=int, default=400, help="Production min plume area (px)")
    ap.add_argument("--baseline-pct", type=float, default=75.0)
    ap.add_argument("--r-bins", type=int, default=10)
    ap.add_argument("--block", type=int, default=32, help="Block size (px) for residual maps")
    ap.add_argument("--min-block-px", type=int, default=20,
                    help="Ocean pixels a block needs before its median is used")
    ap.add_argument("--heading-halfwidth-deg", type=float, default=60.0,
                    help="Half-width of each heading lobe. These surveys follow the "
                         "coast rather than a strict lawnmower grid (axial concentration "
                         "0.35-0.57), so 60 deg is needed to assign most frames; 45 deg "
                         "leaves too few on the return leg of flight 8.")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-consecutive-load-failures", type=int, default=15,
                    help="Abort if this many frames in a row fail to load. Catches the "
                         "external volume unmounting mid-run, which otherwise looks "
                         "identical to frames that simply have no ocean.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")
    label = args.label or data_dir.name

    first, last = frame_bounds(data_dir)
    frames = iter_frames(data_dir, args.start or first, args.end or last, args.step)
    if not frames:
        raise SystemExit(f"No paired frames under {data_dir}")

    # Fail loudly and immediately if the volume is not actually readable.
    probe = data_dir / f"MAX_{frames[0]:04d}.JPG"
    if not probe.exists() or probe.stat().st_size == 0:
        raise SystemExit(
            f"{probe} does not resolve. If this is a symlink farm, the external "
            f"volume is probably not mounted."
        )

    print(f"{label}: {len(frames)} frames (stride {args.step}) from [{frames[0]}..{frames[-1]}]")

    detector = SpreadSGDDetector(
        base_path=str(data_dir), use_ml=False,
        delta_c=args.delta_c, min_area_px=args.min_area,
        baseline_percentile_ocean=args.baseline_pct,
    )
    georef = SGDPolygonGeoref(base_path=str(data_dir))

    # The refined ocean mask is built inside detect_sgd_plumes and never
    # returned; wrapping the module-level name is the only way to observe
    # exactly the mask production used.
    captured = {}
    orig_refine = spread_mod.refine_ocean_with_thermal

    def capturing_refine(masks, rgb, thermal, **kw):
        rr = orig_refine(masks, rgb, thermal, **kw)
        captured["ocean"] = rr.masks.get("ocean")
        return rr

    spread_mod.refine_ocean_with_thermal = capturing_refine

    acc = None
    skips = {"load_fail": 0, "no_ocean": 0, "too_little_ocean": 0}
    consecutive_fail = 0
    n_ok = 0

    try:
        for i, fn in enumerate(frames, start=1):
            try:
                data = detector.load_frame_data(fn)
                consecutive_fail = 0
            except Exception as e:
                skips["load_fail"] += 1
                consecutive_fail += 1
                if consecutive_fail >= args.max_consecutive_load_failures:
                    raise SystemExit(
                        f"\nABORT: {consecutive_fail} consecutive load failures at frame {fn}.\n"
                        f"  last error: {type(e).__name__}: {e}\n"
                        f"  The external volume has most likely unmounted. Results so far "
                        f"would be a truncated, spatially biased subset of the flight, so "
                        f"nothing has been written. Remount and re-run."
                    )
                continue

            thermal = data["thermal"].astype(np.float32)
            if acc is None:
                acc = Accumulator(thermal.shape, args.r_bins, args.block, args.min_block_px)
                print(f"  thermal {thermal.shape[0]}x{thermal.shape[1]}, "
                      f"{acc.by}x{acc.bx} blocks of {args.block} px")

            captured.pop("ocean", None)
            masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
            sgd_mask, plume_info, chars = detector.detect_sgd_plumes(thermal, masks)

            ocean = captured.get("ocean")
            if ocean is None:
                ocean = masks.get("ocean")
            if ocean is None or not ocean.any():
                skips["no_ocean"] += 1
                continue
            ocean = ocean.astype(bool)

            baseline = chars.get("baseline_c") if isinstance(chars, dict) else None
            if baseline is None or not np.isfinite(baseline):
                ot = thermal[ocean]
                ot = ot[np.isfinite(ot)]
                if ot.size < 100:
                    skips["too_little_ocean"] += 1
                    continue
                baseline = float(np.percentile(ot, args.baseline_pct))

            residual = thermal.astype(np.float64) - float(baseline)

            # Plume centroids: the components handed to georeferencing.
            cents, areas = [], []
            h, w = thermal.shape
            for p in plume_info:
                c = p.get("centroid")
                if not c or len(c) < 2:
                    continue
                cy = (float(c[0]) - (h - 1) / 2.0) / ((h - 1) / 2.0)
                cx = (float(c[1]) - (w - 1) / 2.0) / ((w - 1) / 2.0)
                cents.append(math.sqrt(cx * cx + cy * cy))
                areas.append(float(p.get("area_pixels", 0)))

            heading = None
            try:
                g = georef.extract_gps(str(data_dir / f"MAX_{fn:04d}.JPG"))
                if g and g.get("heading") is not None:
                    heading = float(g["heading"])
            except Exception:
                pass

            if acc.add(fn, ocean, sgd_mask.astype(bool), residual, cents, areas, heading, baseline):
                n_ok += 1
            else:
                skips["too_little_ocean"] += 1

            if i % 50 == 0:
                print(f"  {i}/{len(frames)}  used {n_ok}  skipped {sum(skips.values())} {skips}")
    finally:
        spread_mod.refine_ocean_with_thermal = orig_refine

    if acc is None or n_ok < 40:
        raise SystemExit(f"Only {n_ok} usable frames; need at least 40. skips={skips}")

    used_frac = n_ok / len(frames)
    print(f"  usable {n_ok}/{len(frames)} ({used_frac:.0%})  skips={skips}")
    print(f"  detections leaking outside ocean mask: {acc.leak_px} px "
          f"({100.0*acc.leak_px/max(acc.total_ocean,1):.3f}% of ocean) "
          f"— binary_closing in spread.py dilates past the mask")

    rng = np.random.default_rng(args.seed)
    res_block = np.array(acc.res_block, dtype=np.float64)     # (frames, blocks)
    headings = np.array(acc.heading, dtype=np.float64)

    overall_map = np.nanmedian(res_block, axis=0).reshape(acc.by, acc.bx)
    overall_map -= np.nanmedian(overall_map)

    # ---- heading decomposition ------------------------------------------
    ga, gb, axis_deg = split_by_heading(headings, args.heading_halfwidth_deg)
    if ga is None or gb is None:
        ga = gb = np.array([], dtype=int)
    heading_result = None
    if len(ga) >= 20 and len(gb) >= 20:
        map_a = np.nanmedian(res_block[ga], axis=0).reshape(acc.by, acc.bx)
        map_b = np.nanmedian(res_block[gb], axis=0).reshape(acc.by, acc.bx)
        map_a -= np.nanmedian(map_a)
        map_b -= np.nanmedian(map_b)
        sensor, scene = decompose(map_a, map_b)

        boot_sensor, boot_scene = [], []
        for _ in range(args.bootstrap):
            sa = rng.choice(ga, size=len(ga), replace=True)
            sb = rng.choice(gb, size=len(gb), replace=True)
            ma = np.nanmedian(res_block[sa], axis=0).reshape(acc.by, acc.bx)
            mb = np.nanmedian(res_block[sb], axis=0).reshape(acc.by, acc.bx)
            ma -= np.nanmedian(ma)
            mb -= np.nanmedian(mb)
            s, g = decompose(ma, mb)
            boot_sensor.append(amplitude(s)["p5_p95_range_c"])
            boot_scene.append(amplitude(g)["p5_p95_range_c"])

        heading_result = {
            "axis_deg": axis_deg,
            "n_group_a": int(len(ga)),
            "n_group_b": int(len(gb)),
            "n_unassigned": int(n_ok - len(ga) - len(gb)),
            "corr_a_b": nan_corr(map_a, map_b),
            "corr_a_rot180_b": nan_corr(map_a, rot180(map_b)),
            "sensor_amplitude": amplitude(sensor),
            "scene_amplitude": amplitude(scene),
            "sensor_p5_p95_ci_c": [float(x) for x in np.nanpercentile(boot_sensor, [2.5, 97.5])],
            "scene_p5_p95_ci_c": [float(x) for x in np.nanpercentile(boot_scene, [2.5, 97.5])],
            "sensor_map_c": json_map(sensor),
            "scene_map_c": json_map(scene),
            "map_a_c": json_map(map_a),
            "map_b_c": json_map(map_b),
        }
    else:
        na, nb = len(ga), len(gb)
        print(f"  heading split unusable (group sizes {na}/{nb}); "
              f"sensor/scene separation skipped")

    # ---- plume centroid radial distribution (post-min_area stage) --------
    all_cent = np.concatenate([c for c in acc.cent_r if c.size]) if any(c.size for c in acc.cent_r) else np.array([])
    all_area = np.concatenate([a for a in acc.cent_area if a.size]) if any(a.size for a in acc.cent_area) else np.array([])
    elig = np.array(acc.elig_bin, dtype=np.float64).sum(0)
    det = np.array(acc.det_bin, dtype=np.float64).sum(0)
    centers = 0.5 * (acc.r_edges[:-1] + acc.r_edges[1:])

    cent_hist = np.histogram(all_cent, bins=acc.r_edges)[0].astype(float) if all_cent.size else np.zeros(acc.n_rbins)
    cent_area_hist = (
        np.histogram(all_cent, bins=acc.r_edges, weights=all_area)[0].astype(float)
        if all_cent.size else np.zeros(acc.n_rbins)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        cent_per_elig = np.where(elig > 0, cent_hist / elig, np.nan)
        rate = np.where(elig > 0, det / elig, np.nan)

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "label": label,
        "data_dir": str(data_dir),
        "frames_attempted": len(frames),
        "frames_used": n_ok,
        "frames_used_fraction": used_frac,
        "skips": skips,
        "frame_stride": args.step,
        "frame_shape": list(acc.shape),
        "block_px": args.block,
        "block_grid": [acc.by, acc.bx],
        "detector": {"name": "spread", "delta_c": args.delta_c,
                     "min_area_px": args.min_area,
                     "baseline_percentile_ocean": args.baseline_pct},
        "baseline_c_median": float(np.median(acc.baseline_c)),
        "baseline_c_std_across_frames": float(np.std(acc.baseline_c)),
        "detections_outside_ocean_px": acc.leak_px,
        "detections_outside_ocean_pct_of_ocean": 100.0 * acc.leak_px / max(acc.total_ocean, 1),
        "corner_ocean_pct_excluded": 100.0 * acc.corner_ocean / max(acc.total_ocean, 1),
        "headings_deg": [None if not np.isfinite(h) else float(h) for h in headings],
        "ocean_fraction_percentiles": [float(x) for x in np.percentile(acc.ocean_frac, [5, 25, 50, 75, 95])],
        "overall_residual_map_c": json_map(overall_map),
        "heading_decomposition": heading_result,
        "radial": {
            "r_centers": centers.tolist(),
            "eligible_px": elig.tolist(),
            "plume_centroids": cent_hist.tolist(),
            "plume_centroid_area_px": cent_area_hist.tolist(),
            "centroids_per_eligible_px": cent_per_elig.tolist(),
            "pre_clustering_pixel_detection_rate": rate.tolist(),
        },
        "n_plume_centroids": int(all_cent.size),
    }

    json_path = out_base.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  wrote {json_path}")

    print()
    print(f"=== {label} ===")
    print(f"per-frame P75 ocean baseline: median {result['baseline_c_median']:.2f} °C, "
          f"std across frames {result['baseline_c_std_across_frames']:.2f} °C "
          f"(a scalar offset of this size cancels exactly)")
    if heading_result:
        hr = heading_result
        print(f"heading axis {hr['axis_deg']:.0f}°, groups {hr['n_group_a']}/{hr['n_group_b']}, "
              f"{hr['n_unassigned']} unassigned")
        print(f"  corr(A, B)         = {hr['corr_a_b']:+.3f}   <- image-fixed (sensor)")
        print(f"  corr(A, rot180 B)  = {hr['corr_a_rot180_b']:+.3f}   <- ground-fixed (scene)")
        s, g = hr["sensor_amplitude"], hr["scene_amplitude"]
        slo, shi = hr["sensor_p5_p95_ci_c"]
        glo, ghi = hr["scene_p5_p95_ci_c"]
        print(f"  SENSOR bias p5-p95 = {s['p5_p95_range_c']:.3f} °C "
              f"(95% CI {slo:.3f}, {shi:.3f}), max |.| {s['max_abs_c']:.3f} °C")
        print(f"  SCENE  grad  p5-p95 = {g['p5_p95_range_c']:.3f} °C "
              f"(95% CI {glo:.3f}, {ghi:.3f}), max |.| {g['max_abs_c']:.3f} °C")
        print(f"  detection threshold delta_c = {args.delta_c:g} °C")
    print(f"plume centroids: {result['n_plume_centroids']} across {n_ok} frames")


if __name__ == "__main__":
    main()
