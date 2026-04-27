"""
Rock filter: drop candidate pixels that look like submerged volcanic rock.

Two signals combined:

1. RGB colour signature. Submerged rock on Rapa Nui is dark, desaturated,
   and low in blue dominance relative to open ocean. We flag pixels whose
   RGB/HSV features match a rock prior derived from a rough prior — no
   training is required, but a trained classifier can be swapped in later
   via the `classifier` parameter.

2. Shape / compactness. Rocks tend to be compact (high solidity) with low
   eccentricity. Candidate connected components with those properties AND
   a high rock-RGB-pixel fraction are rejected.

Signals are combined at the connected-component level, not per pixel, so a
real plume that happens to contain a few dark-water pixels is not shredded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import color, measure


@dataclass
class RockFilterParams:
    # Prior-based rock signature. Tune against the validation set once labels exist.
    max_value: float = 0.35  # V in HSV [0..1]; rocks are darker than ocean glare
    max_saturation: float = 0.35  # rocks desaturated
    max_blue_ratio: float = 1.15  # B / max(R, G); rocks < ~1.15, ocean > 1.2
    min_lightness: float = 0.05  # exclude true shadow-black

    # Component-level thresholds
    rock_fraction_reject: float = 0.6  # >60% rock-RGB pixels ⇒ reject
    min_eccentricity_for_plume: float = 0.35  # plume must be elongated enough
    max_solidity_for_plume: float = 0.92  # plume must fan (not a filled blob)


def rock_pixel_mask(rgb: np.ndarray, params: RockFilterParams | None = None) -> np.ndarray:
    """Per-pixel rock-looking mask from RGB.

    Args:
        rgb: HxWx3 uint8 or float array (values 0..255 or 0..1).
        params: tunable thresholds; defaults work as a starting prior for Rapa Nui basalt.

    Returns:
        Boolean mask, True where the pixel's RGB looks like rock.
    """
    p = params or RockFilterParams()
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb01 = rgb.astype(np.float32) / 255.0
    else:
        rgb01 = rgb.astype(np.float32)
        if rgb01.max() > 1.5:
            rgb01 = rgb01 / 255.0

    hsv = color.rgb2hsv(rgb01)
    v = hsv[..., 2]
    s = hsv[..., 1]

    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    rg_max = np.maximum(r, g)
    with np.errstate(divide="ignore", invalid="ignore"):
        blue_ratio = np.where(rg_max > 1e-6, b / rg_max, 0.0)

    is_dark_desat = (v <= p.max_value) & (s <= p.max_saturation) & (v >= p.min_lightness)
    not_blue_dominant = blue_ratio <= p.max_blue_ratio
    return is_dark_desat & not_blue_dominant


def filter_rocks(
    candidate_mask: np.ndarray,
    rgb: np.ndarray,
    thermal: np.ndarray | None = None,
    meta: dict | None = None,
    *,
    params: RockFilterParams | None = None,
    classifier=None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Remove candidate connected components dominated by rock-looking pixels.

    Args:
        candidate_mask: bool array, current per-pixel plume candidates.
        rgb: HxWx3 image aligned to the candidate mask.
        thermal: unused by this filter (kept for interface uniformity).
        meta: unused (kept for interface uniformity).
        params: tuning thresholds.
        classifier: optional callable `rgb -> bool[H, W]` that overrides the
            prior-based rock_pixel_mask (e.g., a trained logistic regression).

    Returns:
        (kept_mask, rejection_reasons) — reasons keyed by component label with
        strings like "rock:frac=0.74;ecc=0.21;sol=0.97".
    """
    p = params or RockFilterParams()
    if classifier is not None:
        rock_px = classifier(rgb).astype(bool)
    else:
        rock_px = rock_pixel_mask(rgb, p)

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

        frac_rock = float((comp & rock_px).sum()) / float(area)

        # Shape: encourage fan-shaped plumes, reject blob-shaped candidates when
        # they're also rock-coloured.
        try:
            props = measure.regionprops(comp.astype(np.int32))[0]
            ecc = float(props.eccentricity)
            sol = float(props.solidity)
        except Exception:
            ecc = 0.0
            sol = 0.0

        is_blob_shape = (ecc < p.min_eccentricity_for_plume) and (sol > p.max_solidity_for_plume)
        is_rock_coloured = frac_rock >= p.rock_fraction_reject

        # Combined rule: reject if rock-coloured AND blob-shaped, OR if rock
        # pixels dominate (very high rock fraction regardless of shape).
        if (is_rock_coloured and is_blob_shape) or frac_rock >= 0.85:
            kept[comp] = False
            reasons[lid] = f"rock:frac={frac_rock:.2f};ecc={ecc:.2f};sol={sol:.2f}"

    return kept, reasons
