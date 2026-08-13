"""
SpreadSGDDetector — detect subtle, continuous cold freshwater lenses.

Designed for sites where SGD manifests as a diffuse spread of cold low-salinity
water floating on top of the saline ocean — e.g., Vaihu harbor on Rapa Nui.
The signal is:

  * Subtle (~0.2-0.5 °C below the warm ambient)
  * Continuous over a large area (often most of a shallow bay)
  * Not confined to a point discharge; shore-anchoring does not apply
  * Fades toward the open ocean rather than forming a compact fan

This differs from classic point-source SGD (small cold fans rooted at the
shoreline) that `RedesignedSGDDetector` was built for. The two detectors are
complementary — use Spread for bay/lagoon/shallow-water surveys, Redesigned
for rocky coast surveys with discrete discharges.

Algorithm:

  1. Load frame + RGB-based ocean segmentation (with thermal_refine so shallow
     turquoise water is NOT misclassified as land).
  2. Compute a single warm-ambient baseline across the whole ocean mask:
     the `baseline_percentile` (default 75th) of ocean temperatures.
  3. Threshold: pixels colder than `baseline - delta_c` are spread candidates.
  4. Morphological open then close to suppress speckle and fill small gaps
     without fragmenting the lens.
  5. Keep connected components >= `min_area_px`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage
from skimage import measure

from sgd_toolkit.detectors.improved import ImprovedSGDDetector
from sgd_toolkit.segmentation.thermal_refine import refine_ocean_with_thermal


class SpreadSGDDetector(ImprovedSGDDetector):
    """Detector for subtle continuous-area cold freshwater spreads.

    Extra tuning knobs beyond the parent:

        baseline_percentile_ocean: percentile of whole-ocean temperature used
            as the "warm ambient" baseline. 75th is robust when the plume
            occupies up to ~half of the ocean pixels in-frame.
        delta_c: pixel is a candidate if its temperature is below
            `baseline - delta_c`. 0.2-0.3 °C is typical for this signal.
        min_area_px: minimum contiguous region size after morphology.
        smooth_iterations: morphological open/close iterations for cleanup.
        apply_thermal_refine: run `refine_ocean_with_thermal` before detection
            so shallow turquoise water is included in the ocean mask.
    """

    def __init__(
        self,
        *args,
        baseline_percentile_ocean: float = 75.0,
        delta_c: float = 0.25,
        min_area_px: int = 400,
        smooth_iterations: int = 2,
        apply_thermal_refine: bool = True,
        reclip_to_ocean: bool = False,
        refine_in_segment: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.baseline_percentile_ocean = baseline_percentile_ocean
        self.delta_c = delta_c
        self.min_area_px = min_area_px
        self.smooth_iterations = smooth_iterations
        self.apply_thermal_refine = apply_thermal_refine
        # Both default to the legacy behaviour, so existing outputs reproduce
        # exactly. See the two comments below for what each corrects.
        self.reclip_to_ocean = reclip_to_ocean
        self.refine_in_segment = refine_in_segment

    def load_frame_data(self, frame_number):
        data = super().load_frame_data(frame_number)
        self._last_rgb_aligned = data.get("rgb_aligned")
        return data

    def segment_ocean_land_waves(self, rgb_aligned):
        masks = super().segment_ocean_land_waves(rgb_aligned)
        # `_last_thermal` is set inside detect_sgd_plumes, which the pipeline
        # calls AFTER this method. So from the second frame onward this refines
        # using the PREVIOUS frame's thermal data, and detect_sgd_plumes then
        # refines the result again with the current frame's. Refinement only
        # ever expands the ocean mask, so the mask ends up over-grown from stale
        # data. Setting refine_in_segment=False skips this pass and leaves the
        # single correct refinement in detect_sgd_plumes.
        if (self.refine_in_segment and self.apply_thermal_refine
                and getattr(self, "_last_thermal", None) is not None):
            rr = refine_ocean_with_thermal(masks, rgb_aligned, self._last_thermal)
            masks = rr.masks
        return masks

    def detect_sgd_plumes(self, thermal: np.ndarray, masks: dict):
        # Stash so `segment_ocean_land_waves` (called earlier in the parent
        # pipeline) can use thermal for refinement on re-invocation.
        self._last_thermal = thermal

        ocean = masks.get("ocean")
        if ocean is None or not ocean.any():
            empty = np.zeros_like(thermal, dtype=bool)
            return empty, [], {"reason": "no_ocean"}

        # If refinement wasn't applied by segment_ocean_land_waves (e.g., caller
        # passed in masks directly), apply it here too.
        rgb = getattr(self, "_last_rgb_aligned", None)
        if self.apply_thermal_refine and rgb is not None:
            rr = refine_ocean_with_thermal(masks, rgb, thermal)
            ocean = rr.masks["ocean"]
            masks = rr.masks

        ocean_t = thermal[ocean]
        ocean_t = ocean_t[np.isfinite(ocean_t)]
        if ocean_t.size < 100:
            return np.zeros_like(thermal, dtype=bool), [], {"reason": "too_little_ocean"}

        baseline = float(np.percentile(ocean_t, self.baseline_percentile_ocean))
        threshold = baseline - self.delta_c

        cold = ocean & (thermal < threshold)

        if self.smooth_iterations > 0:
            cold = ndimage.binary_opening(cold, iterations=self.smooth_iterations)
            cold = ndimage.binary_closing(cold, iterations=self.smooth_iterations)
            # Closing dilates before it erodes, so it can push the cold mask past
            # the ocean boundary and put detected pixels on land. Measured at
            # 0.1-0.2% of detected pixels across flights. Re-mask to keep the
            # smoothing while restoring the ocean constraint that the threshold
            # step applied.
            if self.reclip_to_ocean:
                cold &= ocean

        labels, n_comp = measure.label(cold, connectivity=2, return_num=True)
        sgd_mask = np.zeros_like(cold)
        plume_info = []
        for lid in range(1, n_comp + 1):
            comp = labels == lid
            area = int(comp.sum())
            if area < self.min_area_px:
                continue
            sgd_mask |= comp
            try:
                props = measure.regionprops(comp.astype(np.int32))[0]
                ecc = float(props.eccentricity)
                sol = float(props.solidity)
                centroid = tuple(props.centroid)
                bbox = tuple(props.bbox)
            except Exception:
                ecc = 0.0
                sol = 0.0
                centroid = (0, 0)
                bbox = None

            from skimage import measure as sk_measure
            contours = sk_measure.find_contours(comp.astype(float), 0.5)
            contour = max(contours, key=len).tolist() if contours else []

            comp_temps = thermal[comp]
            plume_info.append(
                {
                    "id": lid,
                    "area_pixels": area,
                    "min_shore_distance": 0,  # not meaningful for spread detector
                    "centroid": centroid,
                    "bbox": bbox,
                    "eccentricity": ecc,
                    "solidity": sol,
                    "contour": contour,
                    "mask": comp,
                    "mean_temp": float(np.mean(comp_temps)),
                    "min_temp": float(np.min(comp_temps)),
                    "temperature_anomaly": float(np.mean(comp_temps) - baseline),
                }
            )

        characteristics = {
            "baseline_method": f"ocean_p{self.baseline_percentile_ocean:.0f}",
            "baseline_c": baseline,
            "threshold_c": threshold,
            "delta_c": self.delta_c,
            "num_plumes": len(plume_info),
        }
        if sgd_mask.any():
            sgd_t = thermal[sgd_mask]
            characteristics.update(
                {
                    "mean_temp": float(np.mean(sgd_t)),
                    "min_temp": float(np.min(sgd_t)),
                    "max_temp": float(np.max(sgd_t)),
                    "temp_anomaly": float(np.mean(sgd_t) - baseline),
                    "area_pixels": int(sgd_mask.sum()),
                    "area_m2": float(sgd_mask.sum() * 0.01),  # nominal 10cm/px
                }
            )
        else:
            characteristics.update(
                {"mean_temp": 0.0, "min_temp": 0.0, "max_temp": 0.0, "temp_anomaly": 0.0, "area_pixels": 0, "area_m2": 0.0}
            )
        return sgd_mask, plume_info, characteristics
