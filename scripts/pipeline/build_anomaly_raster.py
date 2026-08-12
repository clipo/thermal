#!/usr/bin/env python3
"""Build a per-flight cold-anomaly raster from raw thermal frames.

For each ocean pixel in each frame, computes the temperature anomaly relative
to that frame's warm-water baseline (75th percentile of in-frame ocean
temperatures). Projects every pixel onto a fixed lat/lon grid and aggregates
across overlapping frames.

Aggregation (`--aggregate`) is a real choice, not a formality:

  mean (default, and what every published raster to date used)
      Cheap and streaming. Because the per-frame anomaly is rectified at zero
      (`max(0, baseline - T)`), symmetric per-frame noise does NOT average
      away: it accumulates as a small positive background floor. The floor is
      roughly uniform across a flight, so it inflates absolute Σ_anomaly
      slightly while leaving site-to-site comparisons largely intact. It gives
      no protection against sun glint, cloud shadow, or frame-edge artifacts
      in any individual frame.

  median
      Robust to those per-frame outliers and to the rectification floor, at
      the cost of a per-cell histogram (see `--median-bin-c`) and a slower
      pass. Prefer this when a flight has visible glint or broken cloud.

Switching aggregation changes every Σ_anomaly value, so the default is left at
`mean` to keep existing outputs reproducible. Change it deliberately, and
re-run the whole flight set if you do.

Output: a GeoTIFF (or NumPy NPZ if rasterio unavailable) at fixed
`grid_resolution_m` with units of °C below ambient. This is the
*threshold-independent* substrate for all quantitative SGD analyses:

  - Σ_anomaly over a region = total cold-anomaly content in m²·°C
  - Recompute polygon intensity by summing the raster within polygon
  - Spatially join against archaeology features and integrate

Usage:
    python scripts/build_anomaly_raster.py \\
        --data data/flight4_vaihu_east_full_combined \\
        --output sgd_output/flight4_vaihu_east_full_spread/flight4_vaihu_east_full_anomaly \\
        --grid-resolution-m 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from sgd_toolkit.detectors import IntegratedSGDDetector
from sgd_toolkit.georeferencing.footprint_generator import ThermalFrameMapper
from sgd_toolkit.georeferencing.polygon_georef import SGDPolygonGeoref


def load_paired_frames(data_dir: Path) -> list[int]:
    out = []
    for p in sorted(data_dir.glob("MAX_*.JPG")):
        try:
            n = int(p.stem.split("_")[1])
        except Exception:
            continue
        if (data_dir / f"IRX_{n:04d}.irg").exists():
            out.append(n)
    return out


def compute_per_frame_anomaly(thermal_c: np.ndarray, ocean_mask: np.ndarray, baseline_pct: float = 75.0) -> tuple[np.ndarray, float]:
    """Return (anomaly_array, baseline) where anomaly = max(0, baseline - T) in °C.
    Non-ocean cells get NaN so they don't pollute the aggregation grid.
    """
    a = np.full_like(thermal_c, np.nan, dtype=np.float32)
    if ocean_mask.any():
        ocean_t = thermal_c[ocean_mask]
        ocean_t = ocean_t[np.isfinite(ocean_t)]
        if ocean_t.size > 50:
            baseline = float(np.percentile(ocean_t, baseline_pct))
            a[ocean_mask] = np.maximum(0.0, baseline - thermal_c[ocean_mask]).astype(np.float32)
            return a, baseline
    return a, float("nan")


def histogram_median(
    hist: np.ndarray,
    counts: np.ndarray,
    bin_c: float,
    chunk_cells: int = 200_000,
) -> np.ndarray:
    """Per-cell median recovered from a fixed-width histogram.

    `hist` is (n_cells, n_bins) of observation counts and `counts` is the
    per-cell total. Cells with no observations return NaN.

    Locates the two central order statistics and averages them, which is
    exactly how `np.median` is defined, then reports each as its bin centre.
    The error is therefore bounded by `bin_c / 2` regardless of how the values
    are distributed. Interpolating within the median's bin instead would give
    finer resolution on smooth data but is unreliable here: a cell's values are
    often bimodal (background near 0 °C, plume near 1 °C), the median falls in
    the sparse gap between the modes, and within-bin interpolation then misses
    the true median by far more than a bin width.
    """
    n_cells = hist.shape[0]
    out = np.full(n_cells, np.nan, dtype=np.float32)

    for lo in range(0, n_cells, chunk_cells):
        hi = min(lo + chunk_cells, n_cells)
        n = counts[lo:hi].astype(np.int64)
        active = n > 0
        if not active.any():
            continue

        block = hist[lo:hi][active]
        n_act = n[active]
        cum = np.cumsum(block, axis=1, dtype=np.int64)

        # 1-indexed ranks of the two central order statistics. They coincide
        # when n is odd.
        k_lo = (n_act + 1) // 2
        k_hi = n_act // 2 + 1

        idx_lo = np.argmax(cum >= k_lo[:, None], axis=1)
        idx_hi = np.argmax(cum >= k_hi[:, None], axis=1)

        # Bin centres, averaged.
        vals = (0.5 * (idx_lo + idx_hi).astype(np.float64) + 0.5) * bin_c

        chunk_out = np.full(hi - lo, np.nan, dtype=np.float32)
        chunk_out[active] = vals.astype(np.float32)
        out[lo:hi] = chunk_out

    return out


def build_anomaly_raster(
    data_dir: Path,
    output_base: Path,
    grid_resolution_m: float = 1.0,
    use_thermal_refine: bool = True,
    flat_field_path: str | None = None,
    progress_every: int = 25,
    aggregate: str = "mean",
    median_bin_c: float = 0.02,
    median_max_c: float = 5.0,
    median_max_bytes: float = 4e9,
):
    output_base.parent.mkdir(parents=True, exist_ok=True)

    frames = load_paired_frames(data_dir)
    if not frames:
        print(f"  no paired frames in {data_dir}")
        return

    detector = IntegratedSGDDetector(
        base_path=str(data_dir), use_ml=False, flat_field_path=flat_field_path
    )
    georef = SGDPolygonGeoref(base_path=str(data_dir))

    # Use ThermalFrameMapper just for the corner-projection helper.
    mapper = ThermalFrameMapper(data_dir=str(data_dir), output_base="_tmp", verbose=False)

    print(f"Pass 1: collect GPS for bbox ({len(frames)} frames)…")
    gps_records = []
    for i, fn in enumerate(frames):
        try:
            gps = georef.extract_gps(str(data_dir / f"MAX_{fn:04d}.JPG"))
            if not gps or "lat" not in gps:
                continue
            corners = mapper.calculate_footprint_corners(
                lat=float(gps["lat"]),
                lon=float(gps["lon"]),
                altitude=float(gps.get("altitude", 350)),
                heading=float(gps.get("heading") or 0.0),
            )
            gps_records.append({"frame": fn, "corners": corners})
        except Exception:
            continue
    if not gps_records:
        print("  no usable GPS")
        return

    all_lons = np.array([c[0] for d in gps_records for c in d["corners"]])
    all_lats = np.array([c[1] for d in gps_records for c in d["corners"]])
    minlon, maxlon = float(all_lons.min()), float(all_lons.max())
    minlat, maxlat = float(all_lats.min()), float(all_lats.max())
    centerlat = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat))
    grid_w_m = (maxlon - minlon) * mpd_lon
    grid_h_m = (maxlat - minlat) * mpd_lat
    gx = int(math.ceil(grid_w_m / grid_resolution_m))
    gy = int(math.ceil(grid_h_m / grid_resolution_m))
    print(f"  bbox lat [{minlat:.5f}..{maxlat:.5f}] lon [{minlon:.5f}..{maxlon:.5f}]")
    print(f"  grid {gy}×{gx} cells = {grid_h_m:.0f}m × {grid_w_m:.0f}m at {grid_resolution_m}m resolution")

    if aggregate not in ("mean", "median"):
        raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate!r}")

    # Mean path: a running sum plus an observation count, no per-cell storage.
    anom_sum = np.zeros((gy, gx), dtype=np.float64)
    obs_count = np.zeros((gy, gx), dtype=np.int32)

    # Median path: a fixed-width histogram per cell. An exact per-cell value list
    # would need tens of GB at 1 m over a full flight; a histogram costs
    # n_cells × n_bins × 4 bytes and resolves the median to well inside the
    # 0.20-0.25 °C detection thresholds this raster feeds.
    hist = None
    n_bins = 0
    if aggregate == "median":
        n_bins = int(math.ceil(median_max_c / median_bin_c)) + 1  # last bin is the overflow
        need = float(gy) * float(gx) * n_bins * 4
        if need > median_max_bytes:
            raise SystemExit(
                f"median aggregation needs {need/1e9:.1f} GB for a {gy}×{gx} grid with "
                f"{n_bins} bins (limit {median_max_bytes/1e9:.1f} GB). Coarsen "
                f"--grid-resolution-m, widen --median-bin-c, or raise --median-max-bytes."
            )
        print(f"  median histogram: {n_bins} bins of {median_bin_c} °C → {need/1e9:.2f} GB")
        hist = np.zeros((gy * gx, n_bins), dtype=np.uint32)

    # Optionally use thermal_refine to extend ocean mask into surf zone
    if use_thermal_refine:
        from sgd_toolkit.segmentation.thermal_refine import refine_ocean_with_thermal

    print(f"Pass 2: project + accumulate…")
    n_used = 0
    baselines = []
    for i, rec in enumerate(gps_records):
        fn = rec["frame"]
        try:
            data = detector.load_frame_data(fn)
        except Exception:
            continue
        thermal = data["thermal"].astype(np.float32)
        rgb = data["rgb_aligned"]
        masks = detector.segment_ocean_land_waves(rgb)
        if use_thermal_refine:
            try:
                rr = refine_ocean_with_thermal(masks, rgb, thermal)
                masks = rr.masks
            except Exception:
                pass
        ocean = masks.get("ocean", np.zeros(thermal.shape, dtype=bool))
        if not ocean.any():
            continue

        anomaly, baseline = compute_per_frame_anomaly(thermal, ocean)
        if not np.isfinite(baseline):
            continue
        baselines.append(baseline)

        H, W = thermal.shape
        # Bilinear-interp the four ground corners to get every pixel's lat/lon.
        # ThermalFrameMapper.calculate_footprint_corners returns 4 corners
        # in order BL, BR, TR, TL (per its corners_m list).
        corners = rec["corners"]
        BL, BR, TR, TL = corners[:4]
        u = np.linspace(0.0, 1.0, W, dtype=np.float32)
        v = np.linspace(0.0, 1.0, H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        # In image coords, row 0 = top of image. The frame mapper defines TL
        # as the back-left of the camera, which corresponds to image top-left
        # for a north-up image. We bilinear-interp accordingly:
        # (u=0,v=0) → TL, (u=1,v=0) → TR, (u=1,v=1) → BR, (u=0,v=1) → BL.
        lons = (
            (1 - uu) * (1 - vv) * TL[0]
            + uu * (1 - vv) * TR[0]
            + uu * vv * BR[0]
            + (1 - uu) * vv * BL[0]
        )
        lats = (
            (1 - uu) * (1 - vv) * TL[1]
            + uu * (1 - vv) * TR[1]
            + uu * vv * BR[1]
            + (1 - uu) * vv * BL[1]
        )

        col = ((lons - minlon) * mpd_lon / grid_resolution_m).astype(np.int32).ravel()
        row = ((lats - minlat) * mpd_lat / grid_resolution_m).astype(np.int32).ravel()
        a = anomaly.ravel()

        valid = (col >= 0) & (col < gx) & (row >= 0) & (row < gy) & np.isfinite(a)
        col = col[valid]
        row = row[valid]
        a = a[valid]
        # Accumulate (note: row 0 is bottom of grid in our lat-up convention)
        np.add.at(anom_sum, (row, col), a)
        np.add.at(obs_count, (row, col), 1)
        if hist is not None:
            # Anomalies are non-negative by construction; the top bin absorbs
            # anything at or beyond median_max_c.
            b = np.clip((a / median_bin_c).astype(np.int64), 0, n_bins - 1)
            np.add.at(hist, (row.astype(np.int64) * gx + col.astype(np.int64), b), 1)
        n_used += 1

        if (i + 1) % progress_every == 0:
            print(f"  {i+1}/{len(gps_records)} frames…")

    print(f"  used {n_used} frames, baselines: median={float(np.median(baselines)):.2f}°C, std={float(np.std(baselines)):.2f}°C")

    with np.errstate(invalid="ignore", divide="ignore"):
        anom_mean = np.where(obs_count > 0, anom_sum / obs_count, np.nan).astype(np.float32)

    if hist is not None:
        anom_agg = histogram_median(hist, obs_count.ravel(), median_bin_c).reshape(gy, gx)
        drift = float(np.nanmean(anom_mean - anom_agg))
        print(f"  median vs mean: mean is {drift:+.4f} °C higher on average "
              f"(rectification floor + per-frame outliers)")
    else:
        anom_agg = anom_mean

    # Save NPZ (fast) and a PNG preview
    out_npz = output_base.with_suffix(".npz")
    np.savez_compressed(
        out_npz,
        anomaly=anom_agg,
        anomaly_mean=anom_mean,
        observations=obs_count,
        bbox_min_lon=minlon,
        bbox_max_lon=maxlon,
        bbox_min_lat=minlat,
        bbox_max_lat=maxlat,
        grid_resolution_m=grid_resolution_m,
        baseline_median_c=float(np.median(baselines)) if baselines else float("nan"),
        baseline_std_c=float(np.std(baselines)) if baselines else float("nan"),
        n_frames_used=n_used,
        aggregate=aggregate,
        median_bin_c=median_bin_c if aggregate == "median" else float("nan"),
        flat_field=str(flat_field_path) if flat_field_path else "",
    )
    print(f"  wrote raster → {out_npz}")

    # PNG preview with a fixed colormap
    png_path = output_base.with_suffix(".png")
    write_anomaly_png(anom_agg, png_path)
    print(f"  wrote preview → {png_path}")

    # KML GroundOverlay so you can drop the PNG on Google Earth
    kml_path = output_base.with_suffix(".kml")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{output_base.stem} anomaly raster</name>
  <GroundOverlay>
    <name>Cold anomaly (°C below ambient)</name>
    <description><![CDATA[
      {aggregate.capitalize()} of (T_baseline − T) per cell across {n_used} frames.<br/>
      Baseline = 75th-percentile ocean temperature per frame.<br/>
      Resolution: {grid_resolution_m} m. Brighter = colder = stronger SGD signal.
    ]]></description>
    <Icon><href>{png_path.name}</href></Icon>
    <LatLonBox>
      <north>{maxlat:.7f}</north>
      <south>{minlat:.7f}</south>
      <east>{maxlon:.7f}</east>
      <west>{minlon:.7f}</west>
    </LatLonBox>
  </GroundOverlay>
</Document>
</kml>
"""
    kml_path.write_text(kml)
    print(f"  wrote KML overlay → {kml_path}")


def write_anomaly_png(anom: np.ndarray, path: Path, vmax: float = 1.5):
    """Render the anomaly array to an RGBA PNG.
    Color scheme: transparent where NaN; cold-blue → white → red ramp.
    Anomaly is 0 (no anomaly) → vmax (very cold). vmax=1.5°C captures most SGD
    signal while preserving contrast.
    """
    try:
        import matplotlib.cm as cm
    except ImportError:
        # Fallback: monochrome blue ramp
        norm = np.clip(anom / vmax, 0, 1)
        rgb = np.zeros((*anom.shape, 4), dtype=np.uint8)
        rgb[..., 0] = (255 * (1 - norm)).astype(np.uint8)  # red drops
        rgb[..., 1] = (255 * (1 - norm)).astype(np.uint8)  # green drops
        rgb[..., 2] = 255  # blue stays high
        rgb[..., 3] = np.where(np.isnan(anom), 0, 220).astype(np.uint8)
        Image.fromarray(np.flipud(rgb), "RGBA").save(path)
        return
    norm = np.clip(anom / vmax, 0.0, 1.0)
    cmap = cm.get_cmap("YlOrRd")  # 0 = pale yellow (no anomaly) → red (strong)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isnan(anom), 0, 220).astype(np.uint8)
    Image.fromarray(np.flipud(rgba), "RGBA").save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True, help="Output base path (extensions added)")
    ap.add_argument("--grid-resolution-m", type=float, default=1.0)
    ap.add_argument("--no-thermal-refine", action="store_true")
    ap.add_argument(
        "--flat-field",
        default=None,
        help="Optional flat-field .npz whose per-pixel bias is subtracted from "
        "every frame before the anomaly is computed. OFF by default, matching "
        "every published raster. Used to test how far Sigma_anomaly moves under "
        "an injected image-fixed bias (see scripts/diagnostics/).",
    )
    ap.add_argument(
        "--aggregate",
        choices=["mean", "median"],
        default="mean",
        help="How to combine overlapping frames per cell. 'mean' (default) matches "
        "every raster published to date. 'median' is robust to glint, cloud shadow "
        "and the positive floor left by rectifying the per-frame anomaly at zero, "
        "but costs a per-cell histogram and changes all Sigma_anomaly values.",
    )
    ap.add_argument(
        "--median-bin-c",
        type=float,
        default=0.02,
        help="Histogram bin width (°C) for --aggregate median. Bounds the "
        "median's error at half this value, i.e. 0.01 °C by default, well "
        "below the 0.20-0.25 °C detection thresholds this raster feeds.",
    )
    ap.add_argument(
        "--median-max-c",
        type=float,
        default=5.0,
        help="Top of the median histogram (°C). Anomalies at or above this land "
        "in the overflow bin; 5 °C is far beyond any observed SGD signal.",
    )
    ap.add_argument(
        "--median-max-bytes",
        type=float,
        default=4e9,
        help="Refuse to allocate a median histogram larger than this.",
    )
    args = ap.parse_args()

    build_anomaly_raster(
        data_dir=Path(args.data),
        output_base=Path(args.output),
        grid_resolution_m=args.grid_resolution_m,
        use_thermal_refine=not args.no_thermal_refine,
        flat_field_path=args.flat_field,
        aggregate=args.aggregate,
        median_bin_c=args.median_bin_c,
        median_max_c=args.median_max_c,
        median_max_bytes=args.median_max_bytes,
    )


if __name__ == "__main__":
    main()
