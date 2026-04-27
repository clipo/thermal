"""
RedesignedSGDDetector: integrates the pipeline redesign into a drop-in detector.

Keeps the public interface of IntegratedSGDDetector (process_frame, load_frame_data,
segment_ocean_land_waves) so downstream consumers — scripts/sgd_autodetect.py,
scripts/sgd_viewer.py, scripts/sgd_wizard.py — can switch detectors by swapping a
class name.

Pipeline per frame:

    load_frame_data
        -> segment_ocean_land_waves (inherited: RF / SAM / rule-based)
        -> exclude glint / wave-foam pixels from baseline fit
        -> fit_spatial_baseline(thermal, ocean, land)             # Stage 2
        -> grow_plumes_from_shore(thermal, z_map, ...)            # Stage 3
        -> filter_rocks(candidate, rgb)                           # Stage 5a
        -> filter_shadows(candidate, rgb, thermal, ocean_mask)    # Stage 5c

Returns the same dict shape as IntegratedSGDDetector.process_frame:
    {'data', 'masks', 'sgd_mask', 'plume_info', 'characteristics'}
plus an extra 'redesign' key carrying per-stage diagnostics for the viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from sgd_toolkit.detectors.improved import ImprovedSGDDetector
from sgd_toolkit.detectors.spatial_baseline import fit_spatial_baseline
from sgd_toolkit.detectors.coast_emerging import grow_plumes_from_shore
from sgd_toolkit.filters import filter_rocks, filter_shadows


class RedesignedSGDDetector(ImprovedSGDDetector):
    """SGD detector with spatial baseline + coast-emerging region grower + FP filters.

    Tunable knobs for the pipeline (all safe defaults tuned against the synthetic
    test set; per-site calibration expected once ground-truth labels exist):

        bin_width_px: distance-bin width for spatial baseline (pixels).
        seed_z_threshold: shoreline-seed cold-threshold z-score.
        grow_z_threshold: region-growing cold-threshold z-score.
        max_plume_distance_px: cap plume extent from its shoreline seed.
        monotonicity_threshold: min Spearman rho(d, z) along plume.
        apply_rock_filter / apply_shadow_filter: enable/disable each FP filter.

    Construction args not listed here are forwarded to ImprovedSGDDetector.
    """

    def __init__(
        self,
        *args,
        bin_width_px: float = 10.0,
        seed_z_threshold: float = -2.0,
        grow_z_threshold: float = -1.2,
        max_plume_distance_px: float = 300.0,
        monotonicity_threshold: float = 0.1,
        min_shore_touch_pixels: int = 3,
        apply_rock_filter: bool = True,
        apply_shadow_filter: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bin_width_px = bin_width_px
        self.seed_z_threshold = seed_z_threshold
        self.grow_z_threshold = grow_z_threshold
        self.max_plume_distance_px = max_plume_distance_px
        self.monotonicity_threshold = monotonicity_threshold
        self.min_shore_touch_pixels = min_shore_touch_pixels
        self.apply_rock_filter = apply_rock_filter
        self.apply_shadow_filter = apply_shadow_filter

    def detect_sgd_plumes(self, thermal: np.ndarray, masks: dict):
        """Replace the parent's detection with spatial-baseline + coast-emerging grower + filters.

        Returns the same (sgd_mask, plume_info, characteristics) tuple so callers
        don't need to change. plume_info entries include a superset of the fields
        the old detector produced.
        """
        ocean = masks.get("ocean")
        land = masks.get("land")
        waves = masks.get("waves")

        if ocean is None or land is None or not ocean.any():
            empty = np.zeros_like(thermal, dtype=bool)
            return empty, [], {"reason": "no_ocean_or_land_mask"}

        # Exclude glint + wave-foam from the baseline fit (they would drag baseline warm / bias scale).
        exclude = None
        if waves is not None:
            exclude = waves.astype(bool).copy()

        sb = fit_spatial_baseline(
            thermal=thermal,
            ocean_mask=ocean,
            land_mask=land,
            bin_width_px=self.bin_width_px,
            exclude_mask=exclude,
        )

        grow_res = grow_plumes_from_shore(
            thermal=thermal,
            z_map=sb.z_map,
            ocean_mask=ocean,
            land_mask=land,
            distance_from_shore=sb.distance_from_shore,
            seed_z_threshold=self.seed_z_threshold,
            grow_z_threshold=self.grow_z_threshold,
            max_distance_px=self.max_plume_distance_px,
            min_area=self.min_area,
            monotonicity_threshold=self.monotonicity_threshold,
            min_shore_touch_pixels=self.min_shore_touch_pixels,
        )

        sgd_mask = grow_res.sgd_mask.copy()
        rejection_log: dict[str, dict] = {
            "monotonicity": {p.id: p.rejected_reason for p in grow_res.rejected},
            "rock": {},
            "shadow": {},
        }

        # RGB needed by filters; reuse the aligned RGB saved by load_frame_data.
        rgb = getattr(self, "_last_rgb_aligned", None)

        if rgb is not None and self.apply_rock_filter and sgd_mask.any():
            sgd_mask, rock_reasons = filter_rocks(
                candidate_mask=sgd_mask, rgb=rgb, thermal=thermal
            )
            rejection_log["rock"] = rock_reasons

        if rgb is not None and self.apply_shadow_filter and sgd_mask.any():
            sgd_mask, shadow_reasons = filter_shadows(
                candidate_mask=sgd_mask,
                rgb=rgb,
                thermal=thermal,
                ocean_mask=ocean,
            )
            rejection_log["shadow"] = shadow_reasons

        # Build plume_info in the same schema as ImprovedSGDDetector so callers work unchanged.
        plume_info = []
        for p in grow_res.plumes:
            if not (p.mask & sgd_mask).any():
                # Was accepted by the grower but killed by a downstream filter.
                continue
            # Re-derive contour from the possibly-filtered mask.
            try:
                from skimage import measure as sk_measure

                contours = sk_measure.find_contours((p.mask & sgd_mask).astype(float), 0.5)
                contour = contours[0].tolist() if contours else []
                props_list = sk_measure.regionprops((p.mask & sgd_mask).astype(np.int32))
                props = props_list[0] if props_list else None
            except Exception:
                contour = []
                props = None

            entry = {
                "id": p.id,
                "area_pixels": int((p.mask & sgd_mask).sum()),
                "min_shore_distance": 0,  # plume touches shoreline by construction
                "centroid": tuple(props.centroid) if props is not None else p.seed_yx,
                "bbox": tuple(props.bbox) if props is not None else None,
                "eccentricity": p.eccentricity,
                "solidity": p.solidity,
                "contour": contour,
                "mask": p.mask & sgd_mask,
                "mean_temp": p.mean_temp_c,
                "min_temp": p.min_temp_c,
                "temperature_anomaly": p.mean_z,  # now in sigma units, not °C
                "mean_z": p.mean_z,
                "min_z": p.min_z,
                "monotonicity": p.monotonicity,
                "shore_touch_pixels": p.shore_touch_pixels,
                "max_distance_px": p.max_distance_px,
            }
            plume_info.append(entry)

        characteristics = {
            "baseline_method": "spatial_distance_binned",
            "num_plumes": len(plume_info),
            "num_rejected": len(grow_res.rejected),
            "rejection_log": rejection_log,
        }
        if sgd_mask.any():
            sgd_temps = thermal[sgd_mask]
            characteristics.update(
                {
                    "mean_temp": float(np.mean(sgd_temps)),
                    "min_temp": float(np.min(sgd_temps)),
                    "max_temp": float(np.max(sgd_temps)),
                    "area_pixels": int(sgd_mask.sum()),
                    "area_m2": float(sgd_mask.sum() * 0.01),
                }
            )
        else:
            characteristics.update(
                {
                    "mean_temp": 0.0,
                    "min_temp": 0.0,
                    "max_temp": 0.0,
                    "area_pixels": 0,
                    "area_m2": 0.0,
                }
            )

        # Expose diagnostics for the viewer / evaluate script.
        characteristics["redesign"] = {
            "bin_centers_px": sb.bin_centers.tolist(),
            "bin_baseline_c": sb.bin_baseline.tolist(),
            "bin_scale_c": sb.bin_scale.tolist(),
            "bin_count": sb.bin_count.tolist(),
            "z_min": float(np.nanmin(sb.z_map)) if np.isfinite(sb.z_map).any() else float("nan"),
            "z_median": float(np.nanmedian(sb.z_map)) if np.isfinite(sb.z_map).any() else float("nan"),
        }

        return sgd_mask, plume_info, characteristics

    def load_frame_data(self, frame_number):
        """Override to stash RGB for downstream filters."""
        data = super().load_frame_data(frame_number)
        self._last_rgb_aligned = data.get("rgb_aligned")
        return data
