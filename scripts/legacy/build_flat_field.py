#!/usr/bin/env python3
"""
Build an empirical flat-field (vignette) for a thermal survey.

One-time calibration per flight. Samples N frames broadly across the flight,
computes each frame's residual against a robust ocean baseline, then median-
combines the residuals per (y, x) to recover the sensor's radial bias. Save
as .npz; pass --flat-field <path> to sgd_autodetect.py (or to the detector's
constructor) to apply at load time.

Typical usage:

    python scripts/build_flat_field.py \\
        --data data/100MEDIA \\
        --output models/flat_fields/vaihu_east.npz \\
        --samples 50

    python scripts/build_flat_field.py \\
        --data "/Volumes/RapaNui/Rapa Nui Jan 2024/Autel/Flight 4 - Vaihu - East/100MEDIA" \\
        --output models/flat_fields/flight4_vaihu_east.npz \\
        --samples 60 --visualize

`--visualize` saves a PNG of the estimated bias map next to the .npz.

The sampler walks uniformly across the available frames so real scene content
(plumes, rocks, land) averages out and only the sensor bias remains.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sgd_toolkit.calibration.vignette import estimate_vignette, save_vignette
from sgd_toolkit.detectors import IntegratedSGDDetector


def discover_frames(data_dir: Path) -> list[int]:
    frames = []
    for p in sorted(data_dir.glob("MAX_*.JPG")):
        try:
            n = int(p.stem.split("_")[1])
        except ValueError:
            continue
        if (data_dir / f"IRX_{n:04d}.irg").exists():
            frames.append(n)
    return frames


def sample_indices(total: int, n: int) -> list[int]:
    if n >= total:
        return list(range(total))
    # Uniform sampling (not random) so calibration is reproducible per directory.
    idx = np.linspace(0, total - 1, n).round().astype(int)
    return sorted(set(idx.tolist()))


def iter_frames(detector: IntegratedSGDDetector, frame_numbers: list[int]):
    for fn in frame_numbers:
        try:
            data = detector.load_frame_data(fn)
        except FileNotFoundError:
            continue
        rgb = data["rgb_aligned"]
        thermal = data["thermal"]
        masks = detector.segment_ocean_land_waves(rgb)
        ocean = masks["ocean"].astype(bool)
        if ocean.sum() < 500:
            # Skip frames where the drone is mostly over land — little info for flat-field.
            continue
        yield fn, thermal, ocean


def maybe_visualize(vignette, output_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping visualization")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(vignette.bias, cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    axes[0].set_title(
        f"Vignette bias (°C)  range [{float(vignette.bias.min()):+.2f}, {float(vignette.bias.max()):+.2f}]"
    )
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(vignette.observation_count, cmap="viridis")
    axes[1].set_title(f"Observation count (max {int(vignette.observation_count.max())})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.suptitle(
        f"Flat-field from {vignette.metadata.get('n_frames')} frames — "
        f"{vignette.metadata.get('data_dir', '')}"
    )
    fig.tight_layout()
    png_path = output_path.with_suffix(".png")
    fig.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote visualization → {png_path}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Directory with MAX_*.JPG + IRX_*.irg pairs")
    ap.add_argument("--output", required=True, help="Where to save the .npz flat-field file")
    ap.add_argument("--samples", type=int, default=50, help="Number of frames to sample")
    ap.add_argument("--baseline-percentile", type=float, default=75.0)
    ap.add_argument("--smooth-sigma", type=float, default=8.0, help="Gaussian σ in pixels (0 to disable)")
    ap.add_argument(
        "--radial-order",
        type=int,
        default=0,
        help="Even-power radial polynomial order (0=disabled, 4-6 for very radial cameras)",
    )
    ap.add_argument("--visualize", action="store_true", help="Save a PNG preview of the flat field")
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    output = Path(args.output)
    if not data.exists():
        raise SystemExit(f"Data directory not found: {data}")

    detector = IntegratedSGDDetector(base_path=str(data), use_ml=False)
    all_frames = discover_frames(data)
    if not all_frames:
        raise SystemExit(f"No paired MAX/IRX frames found in {data}")

    chosen = [all_frames[i] for i in sample_indices(len(all_frames), args.samples)]
    print(f"Sampling {len(chosen)} of {len(all_frames)} frames from {data}")

    vignette = estimate_vignette(
        iter_frames(detector, chosen),
        baseline_percentile=args.baseline_percentile,
        smooth_sigma_px=args.smooth_sigma,
        radial_polynomial_order=args.radial_order,
        metadata={"data_dir": str(data.resolve())},
    )

    save_vignette(vignette, output)
    print(
        f"Wrote flat field → {output}\n"
        f"  shape: {vignette.shape}\n"
        f"  contributing frames: {len(vignette.source_frames)}\n"
        f"  bias range: [{float(vignette.bias.min()):+.3f}, {float(vignette.bias.max()):+.3f}] °C\n"
        f"  bias std:   {float(vignette.bias.std()):.3f} °C"
    )

    if args.visualize:
        maybe_visualize(vignette, output)


if __name__ == "__main__":
    main()
