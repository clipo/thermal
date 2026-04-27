"""
Vignette / flat-field estimation and application for thermal imagery.

Uncooled microbolometer cameras (including the Autel 640T) have a radial
temperature bias: the sensor center heats up from self-emission and reads
colder than the edges by ~0.5-1.5 °C, even when the scene is uniform. This
bias is approximately additive, approximately radially symmetric, and stable
within one flight. It systematically creates the appearance of cold plumes
toward frame centers and masks real cold anomalies toward the margins.

This module estimates the bias *empirically* from the flight itself:

    For each of N sampled frames with an ocean mask m_i:
        residual_i[y, x] = thermal_i[y, x] - robust_baseline(thermal_i[m_i])
    Accumulate residual_i[y, x] across i, keeping only ocean pixels.
    Median-combine per (y, x) to get the per-pixel bias → the vignette.
    Optionally smooth with a radial polynomial or gaussian low-pass.

Real scene content averages out across many frames with different content;
what remains is the systematic sensor pattern — exactly the vignette.

Apply at detection time by subtracting the vignette from each loaded frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy import ndimage


@dataclass
class Vignette:
    """Per-pixel additive bias to SUBTRACT from each frame before analysis.

    Attributes:
        bias: HxW float32 array. thermal_corrected = thermal_raw - bias.
        observation_count: HxW int32 array, number of ocean observations that
            contributed to each pixel's estimate. Useful for diagnostics.
        source_frames: list of frame numbers used to build the vignette.
        shape: (H, W) tuple.
        metadata: free-form dict of provenance info (data_dir, thermal FOV ratio, etc.).
    """

    bias: np.ndarray
    observation_count: np.ndarray
    source_frames: list[int]
    shape: tuple[int, int]
    metadata: dict


def _robust_baseline(values: np.ndarray, percentile: float = 75.0) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def estimate_vignette(
    frame_iter: Iterable[tuple[int, np.ndarray, np.ndarray]],
    *,
    baseline_percentile: float = 75.0,
    min_observations_per_pixel: int = 5,
    smooth_sigma_px: float = 8.0,
    radial_polynomial_order: int = 0,
    metadata: Optional[dict] = None,
) -> Vignette:
    """Build an additive per-pixel vignette from many ocean-only frame residuals.

    Args:
        frame_iter: iterable yielding (frame_number, thermal °C, ocean_mask_bool).
            Thermal and ocean_mask must be same HxW for every yield.
        baseline_percentile: per-frame baseline for residual computation. 75th
            percentile is robust to cold plumes filling up to half the ocean.
        min_observations_per_pixel: pixels observed fewer than this many times
            are treated as unobserved and filled from neighbors during smoothing.
        smooth_sigma_px: gaussian blur σ applied to the raw per-pixel median
            residual, to remove shot noise and leave the low-spatial-frequency
            vignette. 0 disables. Typical: 6-12 px for 640x512 thermal.
        radial_polynomial_order: if > 0, after smoothing, fit an r^0 + r^2 + r^4...
            polynomial to the smoothed bias to enforce strict radial symmetry.
            0 = skip (trust the smoothed field as-is). Use 4 or 6 for very
            radial cameras.
        metadata: provenance kept with the vignette for reproducibility.

    Returns:
        Vignette with bias centered on zero mean over observed pixels (so that
        subtracting it does not shift the overall temperature scale).
    """
    residual_sum: Optional[np.ndarray] = None
    residual_sq_sum: Optional[np.ndarray] = None
    residual_stack: list[np.ndarray] = []  # memory-efficient alternative; see below
    obs_count: Optional[np.ndarray] = None
    source_frames: list[int] = []
    shape: Optional[tuple[int, int]] = None

    for frame_number, thermal, ocean_mask in frame_iter:
        if thermal.ndim != 2 or ocean_mask.ndim != 2:
            raise ValueError("thermal and ocean_mask must be 2D")
        if thermal.shape != ocean_mask.shape:
            raise ValueError(
                f"thermal {thermal.shape} != ocean_mask {ocean_mask.shape}"
            )
        if shape is None:
            shape = thermal.shape
            residual_sum = np.zeros(shape, dtype=np.float64)
            residual_sq_sum = np.zeros(shape, dtype=np.float64)
            obs_count = np.zeros(shape, dtype=np.int32)
        elif thermal.shape != shape:
            raise ValueError(f"inconsistent shape {thermal.shape} vs expected {shape}")

        ocean_temps = thermal[ocean_mask]
        if ocean_temps.size < 50:
            continue  # frame has almost no ocean — skip
        baseline = _robust_baseline(ocean_temps, baseline_percentile)
        residual = np.where(ocean_mask & ~np.isnan(thermal), thermal - baseline, np.nan)
        residual_stack.append(residual.astype(np.float32))
        # Also keep running sums to allow a streaming mean variant if memory matters.
        valid = ocean_mask & ~np.isnan(thermal)
        residual_sum[valid] += residual[valid]
        residual_sq_sum[valid] += residual[valid] ** 2
        obs_count[valid] += 1
        source_frames.append(frame_number)

    if shape is None or not source_frames:
        raise RuntimeError("No frames contributed to vignette estimation")

    # Median-combine across frames per pixel (robust to plumes, clouds, wakes).
    # Stack is (N, H, W). Memory: for 640x512x50 frames = ~65MB at float32 — fine.
    stack = np.stack(residual_stack, axis=0)
    with np.errstate(invalid="ignore"):
        bias_raw = np.nanmedian(stack, axis=0).astype(np.float32)

    # Mark under-observed pixels as NaN so smoothing can hole-fill them.
    bias_raw[obs_count < min_observations_per_pixel] = np.nan

    bias = _smooth_nan_aware(bias_raw, smooth_sigma_px) if smooth_sigma_px > 0 else bias_raw
    # Hole-fill anything still NaN by nearest-neighbor.
    bias = _fill_nan_nearest(bias)

    if radial_polynomial_order > 0:
        bias = _radial_polynomial_fit(bias, order=radial_polynomial_order)

    # Zero-mean the bias over observed pixels so applying it doesn't shift the
    # temperature scale. The *shape* is what matters for detection.
    observed = obs_count >= min_observations_per_pixel
    if observed.any():
        bias -= float(np.mean(bias[observed]))

    return Vignette(
        bias=bias.astype(np.float32),
        observation_count=obs_count,
        source_frames=source_frames,
        shape=shape,
        metadata={
            "n_frames": len(source_frames),
            "baseline_percentile": baseline_percentile,
            "smooth_sigma_px": smooth_sigma_px,
            "radial_polynomial_order": radial_polynomial_order,
            **(metadata or {}),
        },
    )


def _smooth_nan_aware(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that correctly handles NaNs by renormalizing."""
    valid = np.isfinite(arr).astype(np.float32)
    filled = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    num = ndimage.gaussian_filter(filled, sigma=sigma)
    den = ndimage.gaussian_filter(valid, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return out


def _fill_nan_nearest(arr: np.ndarray) -> np.ndarray:
    if not np.isnan(arr).any():
        return arr
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.zeros_like(arr)
    nearest_idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return arr[tuple(nearest_idx)]


def _radial_polynomial_fit(bias: np.ndarray, order: int) -> np.ndarray:
    """Fit b(r) = c0 + c2 r^2 + c4 r^4 + ... around image center."""
    H, W = bias.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float64)
    r_max = r.max() + 1e-6
    r_norm = (r / r_max).ravel()
    b = bias.astype(np.float64).ravel()
    valid = np.isfinite(b)
    if valid.sum() < order + 2:
        return bias
    # Even-powers-only design matrix: 1, r^2, r^4, ...
    features = [np.ones_like(r_norm)]
    for k in range(1, order // 2 + 1):
        features.append(r_norm ** (2 * k))
    A = np.stack(features, axis=1)
    coefs, *_ = np.linalg.lstsq(A[valid], b[valid], rcond=None)
    fitted = (A @ coefs).reshape(bias.shape).astype(np.float32)
    return fitted


def save_vignette(vignette: Vignette, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        bias=vignette.bias,
        observation_count=vignette.observation_count,
        source_frames=np.asarray(vignette.source_frames, dtype=np.int32),
        shape=np.asarray(vignette.shape, dtype=np.int32),
        metadata_json=np.asarray(_as_json(vignette.metadata)),
    )


def load_vignette(path: str | Path) -> Vignette:
    data = np.load(path, allow_pickle=False)
    import json

    meta = json.loads(str(data["metadata_json"]))
    return Vignette(
        bias=data["bias"].astype(np.float32),
        observation_count=data["observation_count"].astype(np.int32),
        source_frames=data["source_frames"].tolist(),
        shape=tuple(int(v) for v in data["shape"]),
        metadata=meta,
    )


def apply_vignette(thermal: np.ndarray, vignette: Vignette | np.ndarray) -> np.ndarray:
    """Return thermal with per-pixel bias subtracted. Safe for any thermal shape
    as long as the bias shape matches.

    Accepts either a Vignette dataclass or a bare bias ndarray.
    """
    bias = vignette.bias if isinstance(vignette, Vignette) else vignette
    if bias.shape != thermal.shape:
        raise ValueError(
            f"vignette bias shape {bias.shape} != thermal shape {thermal.shape}"
        )
    return thermal - bias


def _as_json(obj: dict) -> str:
    import json

    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    return json.dumps(obj, default=default)
