"""
Thermal-assisted refinement of RGB ocean/land segmentation.

The rule-based RGB segmenter calls a pixel "ocean" based on blue-dominance /
saturation thresholds. That fails in shallow water (harbor bays, reef flats,
tide pools) where the sandy or volcanic bottom shows through and the water
is no longer distinctly blue. Cold harbor water ends up in the "land" mask,
so detection never sees it.

This module reclaims those pixels using thermal as a second signal:

1. Estimate a per-frame water-temperature range from the current (possibly
   shrunken) ocean mask.
2. Find pixels currently labeled as land whose thermal temperature falls in
   that range AND that are NOT obvious vegetation / dry rock in RGB.
3. Keep only those that are connected to the existing ocean mask via a
   path of water-temperature pixels — we don't want to flip a cool rock
   sitting in the middle of dry land.

This is a conservative, data-driven fix that runs after `segment_ocean_land_waves`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import color


@dataclass
class ThermalRefineResult:
    masks: dict
    reclaimed_pixels: int
    water_temp_upper_c: float
    water_temp_lower_c: float


def refine_ocean_with_thermal(
    masks: dict,
    rgb: np.ndarray,
    thermal: np.ndarray,
    *,
    water_temp_margin_c: float = 2.5,
    min_temperature_c: float = 15.0,
    max_value_hsv: float = 0.92,
    min_blueness_ratio: float = 0.55,
    max_iterations: int = 25,
) -> ThermalRefineResult:
    """Expand the ocean mask into adjacent land pixels that have water-like
    thermal temperatures (and non-terrestrial RGB).

    Args:
        masks: dict with 'ocean', 'land', optionally 'waves'. Modified COPY is
            returned — the input dict is not mutated.
        rgb: HxWx3 uint8 RGB aligned to thermal.
        thermal: HxW thermal °C.
        water_temp_margin_c: a pixel is a water-temperature candidate if its
            temperature is within this many °C of the current ocean 90th
            percentile (i.e. within the "slightly warmer than the coldest
            ocean, but much colder than land" band).
        min_temperature_c: hard floor to exclude sensor-edge NaN/dropouts.
        max_value_hsv: drop candidates with HSV V above this (reflective bright
            sand/cloud shadows — too optically bright to be water).
        min_blueness_ratio: drop candidates with clearly green/yellow RGB
            dominance (vegetation). Measured as B / max(R, G) >= this. Default
            0.55 is permissive — allows brownish-tinted shallow water over
            volcanic rock or sand to be reclaimed as ocean. The flood-fill
            connectivity requirement still prevents random brown-rock pixels
            on dry land from being misclassified as ocean.
        max_iterations: how many flood-fill passes to grow ocean into
            newly-reclaimed pixels (handles wide bay interiors).

    Reclaims pixels currently classified as either LAND or WAVE — wave-
    classified pixels at the surf zone are often cold water rather than
    warm foam, and without including them the SGD polygons get cut off
    at the coast/water interface with suspicious flat edges instead of
    extending naturally to the shoreline.

    Returns:
        ThermalRefineResult with updated masks dict + diagnostics.
    """
    ocean = masks["ocean"].astype(bool).copy()
    land = masks["land"].astype(bool).copy()
    waves = masks.get("waves")
    if waves is not None:
        waves = waves.astype(bool).copy()

    if not ocean.any():
        # Nothing to grow from — return as-is.
        return ThermalRefineResult(
            masks={"ocean": ocean, "land": land, **({"waves": waves} if waves is not None else {})},
            reclaimed_pixels=0,
            water_temp_upper_c=float("nan"),
            water_temp_lower_c=float("nan"),
        )

    ocean_temps = thermal[ocean]
    ocean_temps = ocean_temps[np.isfinite(ocean_temps)]
    if ocean_temps.size < 50:
        return ThermalRefineResult(
            masks={"ocean": ocean, "land": land, **({"waves": waves} if waves is not None else {})},
            reclaimed_pixels=0,
            water_temp_upper_c=float("nan"),
            water_temp_lower_c=float("nan"),
        )

    # Water-temperature band: the top of the existing ocean distribution plus
    # a margin on the WARM side, and go as cold as the existing ocean gets on
    # the cold side (no reason to exclude colder — those are plumes we want).
    t_upper = float(np.nanpercentile(ocean_temps, 90)) + water_temp_margin_c
    t_lower = float(np.nanmin(ocean_temps)) - 0.5  # small slack

    # RGB candidate: not obviously vegetation or dry sand.
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb01 = rgb.astype(np.float32) / 255.0
    else:
        rgb01 = rgb if rgb.max() <= 1.5 else (rgb / 255.0)
    hsv = color.rgb2hsv(rgb01)
    v = hsv[..., 2]
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    rg_max = np.maximum(r, g)
    with np.errstate(invalid="ignore", divide="ignore"):
        blueness = np.where(rg_max > 1e-6, b / rg_max, 0.0)

    temp_ok = np.isfinite(thermal) & (thermal >= min_temperature_c) & (thermal <= t_upper) & (thermal >= t_lower)
    rgb_ok = (v <= max_value_hsv) & (blueness >= min_blueness_ratio)

    # A pixel is a candidate for reclamation as ocean if it's currently
    # classified as land OR wave AND has water-temperature AND non-vegetation
    # RGB. Wave pixels right at the surf zone often look like cold water
    # rather than warm white foam — without including them we cut SGD
    # polygons off at the coast/water interface and the polygons end up
    # set back from the actual shoreline.
    nonocean_candidate = land.copy()
    if waves is not None:
        nonocean_candidate = nonocean_candidate | waves
    candidate = nonocean_candidate & temp_ok & rgb_ok
    if not candidate.any():
        return ThermalRefineResult(
            masks={"ocean": ocean, "land": land, **({"waves": waves} if waves is not None else {})},
            reclaimed_pixels=0,
            water_temp_upper_c=t_upper,
            water_temp_lower_c=t_lower,
        )

    # Flood-fill from existing ocean into candidate pixels. We grow only into
    # pixels that are candidate AND adjacent to the current ocean, iterating
    # until no more pixels flip. This way a cool rock embedded in dry land
    # never joins the ocean — only shallow-water pixels connected through
    # other shallow-water pixels back to the open ocean.
    kernel = np.ones((3, 3), dtype=bool)
    grown = ocean.copy()
    reclaimed_total = 0
    for _ in range(max_iterations):
        # Dilate current ocean by 1 pixel and intersect with candidate pool.
        fringe = ndimage.binary_dilation(grown, structure=kernel) & candidate & ~grown
        if not fringe.any():
            break
        grown |= fringe
        reclaimed_total += int(fringe.sum())

    if reclaimed_total == 0:
        return ThermalRefineResult(
            masks={"ocean": ocean, "land": land, **({"waves": waves} if waves is not None else {})},
            reclaimed_pixels=0,
            water_temp_upper_c=t_upper,
            water_temp_lower_c=t_lower,
        )

    new_ocean = grown
    # Land pixels reclaimed as ocean leave the land mask; wave pixels likewise.
    new_land = land & ~new_ocean
    new_masks = {"ocean": new_ocean, "land": new_land}
    if waves is not None:
        new_masks["waves"] = waves & ~new_ocean
    # Diagnostics breakdown
    new_masks_meta_land_reclaimed = int((land & new_ocean & ~ocean).sum())
    new_masks_meta_wave_reclaimed = int((waves & new_ocean & ~ocean).sum()) if waves is not None else 0

    return ThermalRefineResult(
        masks=new_masks,
        reclaimed_pixels=reclaimed_total,
        water_temp_upper_c=t_upper,
        water_temp_lower_c=t_lower,
    )
