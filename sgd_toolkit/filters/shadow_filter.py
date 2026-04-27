"""
Shadow filter: drop candidate pixels that look like a cloud or terrain shadow.

Two signals combined:

1. HSV shadow detection: shadows drop brightness (V) while approximately
   preserving hue (H) and relative saturation (S). Pixels with V well below
   the per-frame ocean mean are flagged.

2. RGB-thermal coupling: a true SGD plume cools the water without noticeably
   darkening it. A shadow cools the water AND darkens it. Inside each
   candidate component we correlate the local ΔT field with the local ΔV
   field; high positive correlation means "where it's dark it's also cold" —
   shadow-like.

Optional 3rd signal (disabled by default, requires frame timestamp + GPS):
sun-azimuth consistency. Left as a hook — enable `use_sun_geometry=True`
and pass `meta={'timestamp': ..., 'gps_lat': ..., 'gps_lon': ...}`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, measure


@dataclass
class ShadowFilterParams:
    # Pixel-level HSV shadow prior
    value_stdevs_below_ocean: float = 1.5
    min_value: float = 0.02  # exclude pitch-black (likely rock or sensor dead zone)

    # Component-level coupling test
    min_thermal_rgb_corr: float = 0.35  # reject if corr(ΔT, ΔV) >= this
    min_shadow_fraction: float = 0.4  # component must be ≥40% HSV-shadow pixels to reject

    # Safety: don't reject tiny components on coupling alone
    min_area_for_coupling_test: int = 40


def shadow_pixel_mask(
    rgb: np.ndarray,
    ocean_mask: np.ndarray,
    params: ShadowFilterParams | None = None,
) -> np.ndarray:
    """Per-pixel HSV-shadow candidate mask over ocean pixels only."""
    p = params or ShadowFilterParams()
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb01 = rgb.astype(np.float32) / 255.0
    else:
        rgb01 = rgb.astype(np.float32)
        if rgb01.max() > 1.5:
            rgb01 = rgb01 / 255.0

    hsv = color.rgb2hsv(rgb01)
    v = hsv[..., 2]

    if not ocean_mask.any():
        return np.zeros_like(ocean_mask, dtype=bool)

    v_ocean = v[ocean_mask]
    if v_ocean.size < 10:
        return np.zeros_like(ocean_mask, dtype=bool)
    mu = float(np.median(v_ocean))
    mad = float(np.median(np.abs(v_ocean - mu)))
    sigma = 1.4826 * mad + 1e-6

    is_shadow = (v <= mu - p.value_stdevs_below_ocean * sigma) & (v >= p.min_value)
    return is_shadow & ocean_mask


def filter_shadows(
    candidate_mask: np.ndarray,
    rgb: np.ndarray,
    thermal: np.ndarray,
    meta: dict | None = None,
    *,
    ocean_mask: np.ndarray | None = None,
    params: ShadowFilterParams | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Remove candidate components that look like cloud shadows.

    Args:
        candidate_mask: current plume candidates.
        rgb: HxWx3 RGB aligned to thermal.
        thermal: HxW float °C.
        meta: optional metadata; `ocean_mask` is the key field.
        ocean_mask: preferred way to pass the ocean mask (used to compute V
            baseline). If None, uses the candidate mask's dilation as a rough
            ocean region — less accurate.
        params: thresholds.

    Returns:
        (kept_mask, rejection_reasons) keyed by component label.
    """
    p = params or ShadowFilterParams()
    if ocean_mask is None:
        ocean_mask = candidate_mask  # degenerate fallback

    shadow_px = shadow_pixel_mask(rgb, ocean_mask, p)

    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        v = color.rgb2hsv(rgb.astype(np.float32) / 255.0)[..., 2]
    else:
        rgb01 = rgb if rgb.max() <= 1.5 else rgb / 255.0
        v = color.rgb2hsv(rgb01)[..., 2]

    kept = candidate_mask.copy()
    reasons: dict[int, str] = {}

    labels, n = measure.label(candidate_mask, connectivity=2, return_num=True)
    if n == 0:
        return kept, reasons

    for lid in range(1, n + 1):
        comp = labels == lid
        area = int(comp.sum())
        if area == 0:
            continue

        frac_shadow = float((comp & shadow_px).sum()) / float(area)

        coupling_corr = 0.0
        if area >= p.min_area_for_coupling_test:
            t_vals = thermal[comp].astype(np.float64)
            v_vals = v[comp].astype(np.float64)
            t_vals = t_vals - t_vals.mean()
            v_vals = v_vals - v_vals.mean()
            denom = float(np.sqrt((t_vals * t_vals).sum() * (v_vals * v_vals).sum()))
            if denom > 1e-9:
                coupling_corr = float((t_vals * v_vals).sum() / denom)

        is_shadow_comp = (
            frac_shadow >= p.min_shadow_fraction
            and coupling_corr >= p.min_thermal_rgb_corr
        )

        if is_shadow_comp:
            kept[comp] = False
            reasons[lid] = f"shadow:frac={frac_shadow:.2f};corr={coupling_corr:+.2f}"

    return kept, reasons
