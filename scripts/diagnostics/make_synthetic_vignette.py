#!/usr/bin/env python3
"""Generate a synthetic radial bias field for sensitivity testing.

Purpose
-------
We measured an image-fixed radial bias in the survey frames (see README,
"Per-frame thermal bias"). The question that actually decides whether anything
must change is narrower: does the published pipeline CARE?

Rather than build a real flat field and diff against it, this supports a
sensitivity test. Inject a known radial ramp of controlled magnitude, re-run the
pipeline unchanged, and measure how far the plume inventory and Sigma_anomaly
move. Bracketing at one and two times the largest measured value, in both signs,
turns "the results look unaffected" into a quantitative robustness statement:
the inventory is insensitive to an image-fixed radial bias of +/- X degC.

This route deliberately avoids building a real flat field. The cause of the
measured bias is not yet settled between a sensor vignette and sun glint, and a
flat field estimated from the flight itself would bake whatever sun geometry
prevailed that day into something labelled a calibration. A synthetic ramp of
known shape has no such ambiguity, and sensitivity to it is what the decision
needs regardless of the true cause.

Mechanism
---------
No new code is needed in the detector. `IntegratedSGDDetector.load_frame_data`
already applies `temp_celsius = temp_celsius - vignette.bias` whenever a flat
field is configured, and `run_coast_stretch.py --flat-field` passes one through.
So writing `bias = -ramp` injects `+ramp` into every frame.

Sign convention matches `radial_paired_test.py`: `--contrast-c` is centre minus
edge, so a NEGATIVE value makes the frame centre read colder, reproducing the
sign observed in all four flights.

Usage
-----
    # Inject the flight-4 magnitude
    python scripts/diagnostics/make_synthetic_vignette.py \\
        --contrast-c -0.24 --output models/flat_fields/synthetic_m0.24.npz

    # Then run the pipeline with it
    python scripts/pipeline/run_coast_stretch.py --detector spread \\
        --data data/staged/flight4_vaihu_east_full \\
        --flat-field models/flat_fields/synthetic_m0.24.npz ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THERMAL = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(THERMAL))

from sgd_toolkit.calibration.vignette import Vignette, save_vignette  # noqa: E402


def radial_ramp(shape: tuple[int, int], contrast_c: float, profile: str = "linear") -> np.ndarray:
    """Ramp in degC that is `contrast_c` warmer at the centre than at the edge.

    Radius is normalised per-axis so the field is comparable across a non-square
    frame, matching the convention in radial_paired_test.py. The field is
    zero-meaned over the in-frame disc, so it adds structure without shifting
    the overall temperature scale. That matters because the detector's P75
    baseline would absorb a constant anyway, and we want to test the SHAPE.
    """
    h, w = shape
    v = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    u = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    uu, vv = np.meshgrid(u, v)
    r = np.sqrt(uu * uu + vv * vv)

    if profile == "linear":
        shape_fn = -r
    elif profile == "quadratic":
        # Closer to a true lens/self-heating vignette, which falls off as r^2.
        shape_fn = -(r * r)
    else:
        raise ValueError(f"unknown profile {profile!r}")

    # Scale so the ramp delivers exactly `contrast_c` under the SAME inner/outer
    # zone definition radial_paired_test.py measures with. Without this the
    # realised contrast is only ~0.58x the requested value, because the zones
    # are centred near r=0.27 and r=0.85 rather than at r=0 and r=1, and the
    # injected magnitude would not match the measurement it is meant to mirror.
    disc = r <= 1.0
    inner = r < 0.4
    outer = (r > 0.7) & disc
    unit_contrast = float(shape_fn[inner].mean() - shape_fn[outer].mean())
    if abs(unit_contrast) < 1e-9:
        raise ValueError("degenerate ramp shape")

    field = shape_fn * (contrast_c / unit_contrast)
    field = field - float(field[disc].mean())
    return field.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Write a synthetic radial bias field as a flat-field .npz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--contrast-c", type=float, required=True,
                    help="Centre minus edge, in degC. Negative = centre reads colder, "
                         "which is the sign measured in every flight.")
    ap.add_argument("--profile", choices=["linear", "quadratic"], default="linear",
                    help="linear matches the measured profile shape; quadratic is "
                         "closer to a physical lens vignette.")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    shape = (args.height, args.width)
    ramp = radial_ramp(shape, args.contrast_c, args.profile)

    # base.py subtracts bias, so store the negated ramp to ADD the ramp.
    bias = (-ramp).astype(np.float32)

    vig = Vignette(
        bias=bias,
        observation_count=np.full(shape, 9999, dtype=np.int32),
        source_frames=[],
        shape=shape,
        metadata={
            "synthetic": True,
            "purpose": "sensitivity test, not a calibration",
            "contrast_c_centre_minus_edge": args.contrast_c,
            "profile": args.profile,
            "note": "bias is the NEGATED ramp so that base.py's subtraction injects it",
        },
    )
    save_vignette(vig, Path(args.output))

    # Report the ramp as the detector will experience it.
    h, w = shape
    vv = (np.arange(h) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    uu = (np.arange(w) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    U, V = np.meshgrid(uu, vv)
    r = np.sqrt(U * U + V * V)
    inner = (r < 0.4)
    outer = (r > 0.7) & (r <= 1.0)
    got = float(ramp[inner].mean() - ramp[outer].mean())

    print(f"wrote {args.output}")
    print(f"  profile        : {args.profile}")
    print(f"  requested contrast (centre − edge): {args.contrast_c:+.3f} °C")
    print(f"  realised  contrast (centre − edge): {got:+.3f} °C")
    print(f"  injected ramp range: [{float(ramp.min()):+.3f}, {float(ramp.max()):+.3f}] °C")
    print(f"  stored bias range  : [{float(bias.min()):+.3f}, {float(bias.max()):+.3f}] °C "
          f"(negated; base.py subtracts it)")


if __name__ == "__main__":
    main()
