"""
Distance-from-shore spatial baseline and local anomaly z-score.

Replaces the single global ocean-median baseline used in IntegratedSGDDetector /
ImprovedSGDDetector with a spatially-varying background T_baseline(d) and noise
scale sigma(d), where d = distance-from-shoreline in pixels. A pixel's anomaly
is its z-score against the local background:

    z(x, y) = (T(x, y) - T_baseline(d(x, y))) / sigma(d(x, y))

Plume candidates are pixels with z below a negative threshold (e.g., -2.5). This
picks up small (<1 K) real anomalies against a near-uniform ocean while rejecting
the same delta where the ocean itself is noisy or where nearshore waters are
systematically warmer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator


@dataclass
class SpatialBaseline:
    """Result of fitting a distance-from-shore baseline and noise scale.

    Attributes:
        distance_from_shore: float array, pixel distance from shoreline.
            Non-ocean pixels are set to +inf.
        baseline_map: float array, per-pixel expected ocean temperature.
        scale_map: float array, per-pixel robust scale (MAD-equivalent sigma).
        z_map: float array, per-pixel anomaly z-score. NaN outside ocean.
        bin_centers: 1D array of bin centers used (pixels).
        bin_baseline: 1D array of baseline T per bin (°C).
        bin_scale: 1D array of scale per bin (°C).
        bin_count: 1D array of sample counts per bin.
    """

    distance_from_shore: np.ndarray
    baseline_map: np.ndarray
    scale_map: np.ndarray
    z_map: np.ndarray
    bin_centers: np.ndarray
    bin_baseline: np.ndarray
    bin_scale: np.ndarray
    bin_count: np.ndarray


def compute_distance_from_shore(
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
) -> np.ndarray:
    """Return per-pixel Euclidean distance (in pixels) to the ocean-land boundary.

    Non-ocean pixels are set to +inf so callers can mask them out uniformly.
    The boundary is one-pixel-wide: the set of ocean pixels 4-adjacent to land.
    """
    # Shoreline = ocean pixels touching land — a thin 1-pixel strand.
    # This differs from base.py's 2-px-dilation shoreline; we want distance from
    # the true boundary, not a fattened version.
    shifted_up = np.roll(land_mask, -1, axis=0)
    shifted_down = np.roll(land_mask, 1, axis=0)
    shifted_left = np.roll(land_mask, -1, axis=1)
    shifted_right = np.roll(land_mask, 1, axis=1)
    land_neighbor = shifted_up | shifted_down | shifted_left | shifted_right
    shoreline = ocean_mask & land_neighbor

    if not shoreline.any():
        # Drone fully over ocean (no coast in frame) — distance is undefined.
        # Return +inf everywhere so callers treat every pixel as "far from shore."
        return np.full(ocean_mask.shape, np.inf, dtype=np.float32)

    d = ndimage.distance_transform_edt(~shoreline).astype(np.float32)
    d[~ocean_mask] = np.inf
    return d


def _robust_bin_stats(
    values: np.ndarray,
    baseline_percentile: float = 75.0,
    scale_trim_low_pct: float = 25.0,
    scale_trim_high_pct: float = 5.0,
    min_samples: int = 30,
) -> tuple[float, float, int]:
    """Per-bin robust baseline + MAD-equivalent scale.

    Baseline is the `baseline_percentile`-th percentile of the bin. Using an
    upper percentile instead of the median is essential near the coast, where
    SGD plumes can occupy 30-50% of each nearshore bin — a median gets dragged
    cold by the plume itself and the plume stops being an anomaly against the
    (now biased) baseline. The 75th percentile is robust up to the plumes
    taking ~half the bin.

    Scale is MAD computed on values between `scale_trim_low_pct` and
    `100 - scale_trim_high_pct` percentile — the quiet "ambient ocean" part
    of the bin. This avoids plumes inflating the noise estimate.
    """
    values = values[~np.isnan(values)]
    n = values.size
    if n < min_samples:
        return np.nan, np.nan, n
    baseline = float(np.percentile(values, baseline_percentile))
    lo = np.percentile(values, scale_trim_low_pct)
    hi = np.percentile(values, 100.0 - scale_trim_high_pct)
    scale_values = values[(values >= lo) & (values <= hi)]
    if scale_values.size < min_samples:
        scale_values = values
    mad_center = float(np.median(scale_values))
    mad = float(np.median(np.abs(scale_values - mad_center)))
    scale = 1.4826 * mad
    return baseline, scale, n


def fit_spatial_baseline(
    thermal: np.ndarray,
    ocean_mask: np.ndarray,
    land_mask: np.ndarray,
    *,
    bin_width_px: float = 10.0,
    max_distance_px: Optional[float] = None,
    exclude_mask: Optional[np.ndarray] = None,
    min_scale: float = 0.05,
    baseline_percentile: float = 75.0,
    scale_trim_low_pct: float = 25.0,
    scale_trim_high_pct: float = 5.0,
) -> SpatialBaseline:
    """Fit T_baseline(d) and sigma(d) by distance-from-shore and return per-pixel
    z-score map.

    Args:
        thermal: 2D float array, temperature in °C. NaNs allowed.
        ocean_mask: 2D bool array, True where ocean.
        land_mask: 2D bool array, True where land.
        bin_width_px: distance bin width in pixels. 10 px ≈ 1 m at this sensor's
            typical GSD of ~10 cm/pixel.
        max_distance_px: cap distance at this value. None = use max ocean distance.
        exclude_mask: additional pixels to drop from baseline fitting (glint, waves,
            shadow, wake foam). Do NOT drop these from the z-map — callers still
            want to evaluate them.
        min_scale: floor on sigma to avoid divide-by-zero in near-uniform bins.
        baseline_percentile: per-bin baseline is this percentile of bin values.
            Default 75 — robust to cold plumes occupying up to ~50% of a bin.
        scale_trim_low_pct / scale_trim_high_pct: percentiles used to define
            the "ambient ocean" middle of each bin from which MAD-scale is
            computed. Excludes plumes (low tail) and glint (high tail).

    Returns:
        SpatialBaseline with per-pixel baseline_map, scale_map, z_map.
    """
    d = compute_distance_from_shore(ocean_mask, land_mask)

    fit_mask = ocean_mask & np.isfinite(d)
    if exclude_mask is not None:
        fit_mask = fit_mask & ~exclude_mask
    fit_mask = fit_mask & ~np.isnan(thermal)

    if not fit_mask.any():
        nan_map = np.full_like(thermal, np.nan, dtype=np.float32)
        return SpatialBaseline(
            distance_from_shore=d,
            baseline_map=nan_map,
            scale_map=nan_map.copy(),
            z_map=nan_map.copy(),
            bin_centers=np.array([]),
            bin_baseline=np.array([]),
            bin_scale=np.array([]),
            bin_count=np.array([]),
        )

    fit_d = d[fit_mask]
    fit_t = thermal[fit_mask]

    d_max = float(np.nanmax(fit_d))
    if max_distance_px is not None:
        d_max = min(d_max, max_distance_px)
    d_max = max(d_max, bin_width_px)

    # Bin edges; pixels beyond max_distance go into the farthest bin.
    edges = np.arange(0.0, d_max + bin_width_px, bin_width_px)
    if edges.size < 2:
        edges = np.array([0.0, bin_width_px])
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_idx = np.clip(np.digitize(fit_d, edges) - 1, 0, len(centers) - 1)

    bin_baseline = np.full(centers.size, np.nan, dtype=np.float64)
    bin_scale = np.full(centers.size, np.nan, dtype=np.float64)
    bin_count = np.zeros(centers.size, dtype=np.int64)
    for i in range(centers.size):
        mask_i = bin_idx == i
        if not mask_i.any():
            continue
        b, s, n = _robust_bin_stats(
            fit_t[mask_i],
            baseline_percentile=baseline_percentile,
            scale_trim_low_pct=scale_trim_low_pct,
            scale_trim_high_pct=scale_trim_high_pct,
        )
        bin_baseline[i] = b
        bin_scale[i] = s
        bin_count[i] = n

    # Fill sparse / empty bins from neighbors so interpolation has full support.
    bin_baseline = _fill_gaps(bin_baseline)
    bin_scale = _fill_gaps(bin_scale)
    # Scale can be tiny in uniform bins — enforce floor before interpolating.
    bin_scale = np.fmax(bin_scale, min_scale)

    valid = np.isfinite(bin_baseline)
    if valid.sum() < 2:
        # Fallback: global robust baseline — still better than raw median.
        g_b, g_s, _ = _robust_bin_stats(
            fit_t,
            baseline_percentile=baseline_percentile,
            scale_trim_low_pct=scale_trim_low_pct,
            scale_trim_high_pct=scale_trim_high_pct,
        )
        g_s = max(g_s, min_scale)
        baseline_map = np.full_like(thermal, g_b, dtype=np.float32)
        scale_map = np.full_like(thermal, g_s, dtype=np.float32)
    else:
        base_interp = PchipInterpolator(
            centers[valid], bin_baseline[valid], extrapolate=True
        )
        scale_interp = PchipInterpolator(
            centers[valid], bin_scale[valid], extrapolate=True
        )
        d_eval = d.copy()
        d_eval[~np.isfinite(d_eval)] = centers[-1]  # clamp; unused outside ocean
        baseline_map = base_interp(d_eval).astype(np.float32)
        scale_map = np.fmax(scale_interp(d_eval).astype(np.float32), min_scale)

    z_map = np.full_like(thermal, np.nan, dtype=np.float32)
    ocean_eval = ocean_mask & np.isfinite(d) & ~np.isnan(thermal)
    z_map[ocean_eval] = (
        thermal[ocean_eval] - baseline_map[ocean_eval]
    ) / scale_map[ocean_eval]

    return SpatialBaseline(
        distance_from_shore=d,
        baseline_map=baseline_map,
        scale_map=scale_map,
        z_map=z_map,
        bin_centers=centers,
        bin_baseline=bin_baseline,
        bin_scale=bin_scale,
        bin_count=bin_count,
    )


def _fill_gaps(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs with linear interpolation from neighbors (edges carry nearest)."""
    out = arr.astype(np.float64).copy()
    idx = np.arange(out.size)
    valid = np.isfinite(out)
    if not valid.any():
        return out
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out
