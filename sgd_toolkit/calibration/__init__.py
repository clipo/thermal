"""Thermal-camera calibration utilities: flat-field / vignette correction."""

from sgd_toolkit.calibration.vignette import (
    estimate_vignette,
    load_vignette,
    save_vignette,
    apply_vignette,
)

__all__ = ["estimate_vignette", "load_vignette", "save_vignette", "apply_vignette"]
