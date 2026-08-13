#!/usr/bin/env python3
"""Is the image-position bias fixed to the SENSOR or to the SUN?

Why this matters
----------------
`radial_paired_test.py` established that pixels near the frame centre read
colder than pixels near the edge, with the ground held fixed, in every flight
tested. Two very different mechanisms produce that signature, and they call for
opposite responses.

  Sensor vignette. The detector array self-heats at the centre. The pattern is
  fixed in IMAGE coordinates, is radially symmetric, and is a property of the
  camera. The fix is a flat-field correction.

  Sun glint. Specular reflection of sky and sun off the water is strongest at
  off-nadir view angles, which makes frame EDGES read warmer. This is fixed
  relative to the SOLAR AZIMUTH, is asymmetric rather than radially symmetric,
  and varies with time of day and sea state. The fix is glint masking, not
  calibration. Applying a flat field here would bake a sun-geometry artifact
  into something labelled a calibration and then apply it to frames whose sun
  geometry differs.

One observation motivated the test: the sign is edge-warm, which off-nadir
emissivity alone cannot produce, because falling emissivity at high incidence
makes edges read COLDER. Something else has to be responsible, and glint is
the obvious candidate since it peaks at off-nadir angles.

(An earlier version of this note also cited an apparent spread of -0.116 to
-0.634 degC across flights as evidence against fixed hardware. That spread was
a sampling artefact: three of the four flights were measured from a single
contiguous run of frames covering one transect leg. Measured with block
sampling across the whole flight, flights 4 and 11 give -0.242 and -0.281 degC
and their radial profiles correlate at r = 0.9948. The magnitude is stable.)

The discriminator
-----------------
Both effects are measured in the same paired, within-ground-cell way, but
binned by angle in two different coordinate frames:

  image-fixed   angle of the pixel measured clockwise from image "up"
  sun-relative  the same angle minus the sun's bearing within the frame

A sensor pattern is stationary in image coordinates, so it retains its
amplitude in the image-fixed binning and smears out in the sun-relative one.
Glint does the reverse. Whichever frame retains more angular amplitude is the
frame the effect lives in. This works only because heading varies within these
surveys, which decouples the two coordinate systems.

A purely radial vignette is flat in BOTH angular binnings, which is itself a
diagnostic result: it would mean the effect is radially symmetric and
sun-independent, i.e. the sensor explanation.

Result on flight 4, 250 block-sampled frames over 98,738 paired ground cells:
image-fixed angular amplitude 0.301 degC [0.257, 0.353] against sun-relative
0.119 degC [0.094, 0.199]. The intervals do not overlap, so the pattern lives
in sensor coordinates and a flat field is the appropriate remedy if one is ever
needed. The sun-relative amplitude is not zero, so a smaller
illumination-dependent component may sit on top of the instrumental one.

The test needs heading diversity to work at all, because solar azimuth moves
only about 1 degree across a 12-minute flight. Sample with --n-blocks spread
across the flight: 60 consecutive frames gave a relative-sun-bearing
concentration of 0.62 and an inconclusive answer, while 250 block-sampled
frames gave 0.96 and separated the two frames cleanly.

Usage
-----
    python scripts/diagnostics/sun_asymmetry_test.py \\
        --data data/flight4_vaihu_east_full_combined \\
        --label flight4_vaihu_east_full --n-frames 150 \\
        --output sgd_output/diagnostics/sun_asymmetry_flight4_vaihu_east_full
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

THERMAL = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THERMAL))

from sgd_toolkit.detectors import spread as spread_mod  # noqa: E402
from sgd_toolkit.detectors.spread import SpreadSGDDetector  # noqa: E402
from sgd_toolkit.georeferencing.polygon_georef import SGDPolygonGeoref  # noqa: E402
from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper  # noqa: E402


def solar_position(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """Solar elevation and azimuth (deg, 0=N 90=E 180=S 270=W).

    Same simplified formulation as analyze_sun_position.py. Accuracy of a few
    degrees is ample here: the angular bins are 60 degrees wide.
    """
    lat_rad = math.radians(lat)
    doy = dt.timetuple().tm_yday
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    solar_time = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_time += (lon + 109.0) * 4 / 60
    ha = math.radians(15 * (solar_time - 12))
    sin_el = (math.sin(lat_rad) * math.sin(decl)
              + math.cos(lat_rad) * math.cos(decl) * math.cos(ha))
    el = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
    cos_az = ((math.sin(decl) - math.sin(lat_rad) * sin_el)
              / (math.cos(lat_rad) * math.cos(math.radians(el)) + 1e-12))
    az = math.degrees(math.acos(max(-1.0, min(1.0, cos_az))))
    if solar_time < 12:
        az = 360 - az
    return el, az


def exif_datetime(path: Path):
    try:
        img = Image.open(path)
        ex = img._getexif()
        if not ex:
            return None
        tags = {TAGS.get(k, k): v for k, v in ex.items()}
        s = tags.get("DateTimeOriginal") or tags.get("DateTime")
        return datetime.strptime(s, "%Y:%m:%d %H:%M:%S") if s else None
    except Exception:
        return None


def _grouped_mean(groups, values, n_groups):
    s = np.bincount(groups, weights=values, minlength=n_groups)
    c = np.bincount(groups, minlength=n_groups)
    return s, c


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
        description="Discriminate a sensor vignette from sun glint, ground held fixed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--cell-m", type=float, default=2.0)
    ap.add_argument("--r-bins", type=int, default=4)
    ap.add_argument("--a-bins", type=int, default=6, help="Angular bins over 360 deg")
    ap.add_argument("--min-obs-per-cell-bin", type=int, default=3)
    ap.add_argument("--min-bins-per-cell", type=int, default=4)
    ap.add_argument("--boot-block-m", type=float, default=50.0)
    ap.add_argument("--bootstrap", type=int, default=400)
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
    if args.start is not None:
        nums = [n for n in nums if n >= args.start]
    frames = sample_blocks(nums, args.n_blocks, args.block_len)
    if not frames:
        raise SystemExit(f"No paired frames in {data_dir}")
    probe = data_dir / f"MAX_{frames[0]:04d}.JPG"
    if not probe.exists() or probe.stat().st_size == 0:
        raise SystemExit(f"{probe} does not resolve. External volume not mounted?")

    print(f"{label}: sun-asymmetry test on {len(frames)} frames")

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

    # Pass 1: geometry, sun, and the ground grid.
    recs = []
    for fn in frames:
        p = data_dir / f"MAX_{fn:04d}.JPG"
        try:
            g = georef.extract_gps(str(p))
            if not g or "lat" not in g or g.get("heading") is None:
                continue
            dt = exif_datetime(p)
            if dt is None:
                continue
            el, az = solar_position(float(g["lat"]), float(g["lon"]), dt)
            if el < 5.0:
                continue  # sun too low for meaningful glint geometry
            corners = mapper.calculate_footprint_corners(
                lat=float(g["lat"]), lon=float(g["lon"]),
                altitude=float(g.get("altitude", 350)),
                heading=float(g["heading"]),
            )
            recs.append({"frame": fn, "corners": corners,
                         "heading": float(g["heading"]), "sun_az": az, "sun_el": el})
        except Exception:
            continue
    if len(recs) < 30:
        raise SystemExit(f"Only {len(recs)} frames with GPS+heading+time; need 30+.")

    heads = np.array([r["heading"] for r in recs])
    sunaz = np.array([r["sun_az"] for r in recs])
    rel = (sunaz - heads) % 360.0
    print(f"  sun elevation {np.min([r['sun_el'] for r in recs]):.0f}-"
          f"{np.max([r['sun_el'] for r in recs]):.0f}°, azimuth "
          f"{sunaz.min():.0f}-{sunaz.max():.0f}°")
    print(f"  sun bearing within frame spans {rel.min():.0f}-{rel.max():.0f}° "
          f"(spread {np.std(np.exp(1j*np.deg2rad(rel))):.2f}); the two coordinate "
          f"frames are only separable if this varies")

    lons = np.array([c[0] for r in recs for c in r["corners"][:4]])
    lats = np.array([c[1] for r in recs for c in r["corners"][:4]])
    minlon, maxlon = float(lons.min()), float(lons.max())
    minlat, maxlat = float(lats.min()), float(lats.max())
    clat = 0.5 * (minlat + maxlat)
    mpd_lat, mpd_lon = 111320.0, 111320.0 * math.cos(math.radians(clat))
    gx = int(math.ceil((maxlon - minlon) * mpd_lon / args.cell_m)) + 1
    gy = int(math.ceil((maxlat - minlat) * mpd_lat / args.cell_m)) + 1
    n_cells = gx * gy
    nr, na = args.r_bins, args.a_bins
    nb = nr * na
    print(f"  ground grid {gy}x{gx} = {n_cells} cells; {nr} radial x {na} angular bins")

    # Two schemes: image-fixed angle, and sun-relative angle.
    sum_img = np.zeros(n_cells * nb); cnt_img = np.zeros(n_cells * nb, dtype=np.int32)
    sum_sun = np.zeros(n_cells * nb); cnt_sun = np.zeros(n_cells * nb, dtype=np.int32)

    geom = {}
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
                    raise SystemExit(f"ABORT: 15 consecutive load failures at {fn} "
                                     f"({type(e).__name__}: {e}). Volume unmounted? "
                                     f"Nothing written.")
                continue

            thermal = data["thermal"].astype(np.float32)
            if not geom:
                h, w = thermal.shape
                yy = (np.arange(h) - (h - 1) / 2.0) / ((h - 1) / 2.0)
                xx = (np.arange(w) - (w - 1) / 2.0) / ((w - 1) / 2.0)
                X, Y = np.meshgrid(xx, yy)
                R = np.sqrt(X * X + Y * Y)
                # Image azimuth: clockwise from image "up" (row 0 is the top).
                TH = (np.degrees(np.arctan2(X, -Y))) % 360.0
                geom = {"R": R.ravel(), "TH": TH.ravel(),
                        "rbin": np.clip((R.ravel() / 1.0 * nr).astype(np.int64), 0, nr - 1),
                        "in": (R.ravel() <= 1.0)}

            captured.pop("ocean", None)
            masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
            _, _, chars = detector.detect_sgd_plumes(thermal, masks)
            ocean = captured.get("ocean")
            if ocean is None:
                ocean = masks.get("ocean")
            if ocean is None or not ocean.any():
                continue
            baseline = chars.get("baseline_c") if isinstance(chars, dict) else None
            if baseline is None or not np.isfinite(baseline):
                continue
            residual = (thermal.astype(np.float64) - float(baseline)).ravel()

            H, W = thermal.shape
            BL, BR, TR, TL = rec["corners"][:4]
            u = np.linspace(0.0, 1.0, W); v = np.linspace(0.0, 1.0, H)
            uu, vv = np.meshgrid(u, v)
            plon = ((1-uu)*(1-vv)*TL[0] + uu*(1-vv)*TR[0] + uu*vv*BR[0] + (1-uu)*vv*BL[0]).ravel()
            plat = ((1-uu)*(1-vv)*TL[1] + uu*(1-vv)*TR[1] + uu*vv*BR[1] + (1-uu)*vv*BL[1]).ravel()
            col = ((plon - minlon) * mpd_lon / args.cell_m).astype(np.int64)
            row = ((plat - minlat) * mpd_lat / args.cell_m).astype(np.int64)

            ok = (ocean.ravel() & np.isfinite(residual) & geom["in"]
                  & (col >= 0) & (col < gx) & (row >= 0) & (row < gy))
            if ok.sum() < 500:
                continue

            cell = row[ok] * gx + col[ok]
            rb = geom["rbin"][ok]
            th = geom["TH"][ok]
            res = residual[ok]

            # Sun bearing inside this frame's image.
            sun_in_img = (rec["sun_az"] - rec["heading"]) % 360.0

            ab_img = np.clip((th / (360.0 / na)).astype(np.int64), 0, na - 1)
            ab_sun = np.clip((((th - sun_in_img) % 360.0) / (360.0 / na)).astype(np.int64),
                             0, na - 1)

            idx_i = cell * nb + rb * na + ab_img
            idx_s = cell * nb + rb * na + ab_sun
            np.add.at(sum_img, idx_i, res); np.add.at(cnt_img, idx_i, 1)
            np.add.at(sum_sun, idx_s, res); np.add.at(cnt_sun, idx_s, 1)
            n_used += 1
            if i % 25 == 0:
                print(f"  {i}/{len(recs)} frames, {n_used} used")
    finally:
        spread_mod.refine_ocean_with_thermal = orig_refine

    if n_used < 30:
        raise SystemExit(f"Only {n_used} usable frames.")

    rng = np.random.default_rng(args.seed)
    out = {"label": label, "frames_used": n_used, "n_radial": nr, "n_angular": na,
           "delta_c": args.delta_c,
           "sun_bearing_in_frame_deg": [float(rel.min()), float(rel.max())],
           "sun_elevation_deg": [float(min(r["sun_el"] for r in recs)),
                                 float(max(r["sun_el"] for r in recs))]}

    def analyse(sum_a, cnt_a, name):
        s = sum_a.reshape(n_cells, nb); c = cnt_a.reshape(n_cells, nb)
        valid = c >= args.min_obs_per_cell_bin
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.where(valid, s / np.maximum(c, 1), np.nan)
        keep = valid.sum(axis=1) >= args.min_bins_per_cell
        n_keep = int(keep.sum())
        if n_keep < 200:
            print(f"  [{name}] only {n_keep} paired cells; not enough")
            return None
        cb = m[keep]
        dev = cb - np.nanmean(cb, axis=1, keepdims=True)

        cell_ids = np.flatnonzero(keep)
        bf = max(1, int(round(args.boot_block_m / args.cell_m)))
        bgx = gx // bf + 1
        blk = (cell_ids // gx // bf) * bgx + (cell_ids % gx // bf)
        ub, inv = np.unique(blk, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        st = np.searchsorted(inv[order], np.arange(len(ub) + 1))
        members = [order[st[i]:st[i+1]] for i in range(len(ub))]

        def ang_profile(d):
            g = np.nanmean(d, axis=0).reshape(nr, na)
            return np.nanmean(g, axis=0)          # marginal over radius

        def rad_profile(d):
            g = np.nanmean(d, axis=0).reshape(nr, na)
            return np.nanmean(g, axis=1)          # marginal over angle

        ang = ang_profile(dev); rad = rad_profile(dev)
        ang_amp = float(np.nanmax(ang) - np.nanmin(ang))

        boot_amp = np.empty(args.bootstrap)
        for b in range(args.bootstrap):
            pick = rng.integers(0, len(members), size=len(members))
            sel = np.concatenate([members[k] for k in pick])
            a = ang_profile(dev[sel])
            boot_amp[b] = np.nanmax(a) - np.nanmin(a)
        lo, hi = [float(x) for x in np.nanpercentile(boot_amp, [2.5, 97.5])]

        print(f"\n  [{name}] paired cells {n_keep}, blocks {len(ub)}")
        print(f"    angular profile (°C): " + "  ".join(f"{v:+.3f}" for v in ang))
        print(f"    angular amplitude   : {ang_amp:.3f} °C (95% CI {lo:.3f}, {hi:.3f})")
        print(f"    radial profile (°C) : " + "  ".join(f"{v:+.3f}" for v in rad))
        return {"paired_cells": n_keep, "blocks": int(len(ub)),
                "angular_profile_c": ang.tolist(), "radial_profile_c": rad.tolist(),
                "angular_amplitude_c": ang_amp, "angular_amplitude_ci_c": [lo, hi]}

    out["image_fixed"] = analyse(sum_img, cnt_img, "image-fixed angle")
    out["sun_relative"] = analyse(sum_sun, cnt_sun, "sun-relative angle")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    jp = Path(args.output).with_suffix(".json")

    if out["image_fixed"] and out["sun_relative"]:
        ai = out["image_fixed"]["angular_amplitude_c"]
        as_ = out["sun_relative"]["angular_amplitude_c"]
        ilo, ihi = out["image_fixed"]["angular_amplitude_ci_c"]
        slo, shi = out["sun_relative"]["angular_amplitude_ci_c"]
        print(f"\n=== {label}: which coordinate frame holds the asymmetry? ===")
        print(f"  image-fixed  angular amplitude {ai:.3f} °C  [{ilo:.3f}, {ihi:.3f}]")
        print(f"  sun-relative angular amplitude {as_:.3f} °C  [{slo:.3f}, {shi:.3f}]")
        if as_ > ihi:
            verdict = ("Asymmetry is SUN-fixed: consistent with glint, not a sensor "
                       "vignette. A flat field is the wrong correction.")
        elif ai > shi:
            verdict = ("Asymmetry is IMAGE-fixed: consistent with a sensor property. "
                       "A flat-field correction is appropriate.")
        elif max(ai, as_) < 0.2 * args.delta_c:
            verdict = ("No angular asymmetry in either frame. The effect is radially "
                       "symmetric and sun-independent, which points to the sensor "
                       "rather than glint.")
        else:
            verdict = ("Inconclusive: the two amplitudes are not separable at this "
                       "sample size. Raise --n-frames.")
        print(f"  -> {verdict}")
        out["verdict"] = verdict

    jp.write_text(json.dumps(out, indent=2))
    print(f"  wrote {jp}")


if __name__ == "__main__":
    main()
