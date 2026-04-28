#!/usr/bin/env python3
"""
Evaluation harness for SGD detectors.

Runs a detector over a set of labeled frames and reports precision, recall,
IoU, and false-positive breakdown by confounder class (rock / wake / shadow).

Label format: see scripts/annotate.py. One JSON file per frame under a
`labels/` directory alongside the data, with polygons in thermal-aligned
pixel coordinates and classes in {sgd, rock, wake, shadow}.

Usage:

    # Compare baseline detector vs. the redesigned one on the same labels.
    python scripts/evaluate.py \\
        --data data/100MEDIA \\
        --labels-dir data/100MEDIA/labels \\
        --detector redesigned \\
        --report reports/eval_redesigned.md

    python scripts/evaluate.py --data data/100MEDIA --detector improved \\
        --report reports/eval_improved.md

Metrics reported per-frame and in aggregate:
  - SGD precision, recall, IoU against union of labeled SGD polygons.
  - FP pixels broken down by which confounder polygon they overlap
    (rock / wake / shadow / other) — tells you which filter is doing work.
  - Wall-clock time per frame.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage.draw import polygon as sk_polygon


CONFOUNDER_CLASSES = ("rock", "wake", "shadow")


def polygon_to_mask(points: list[list[float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a polygon (list of [x, y]) into a boolean mask."""
    if len(points) < 3:
        return np.zeros(shape, dtype=bool)
    xs = np.array([p[0] for p in points], dtype=np.float32)
    ys = np.array([p[1] for p in points], dtype=np.float32)
    rr, cc = sk_polygon(ys, xs, shape=shape)
    m = np.zeros(shape, dtype=bool)
    m[rr, cc] = True
    return m


def load_labels(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def labels_to_masks(record: dict) -> dict[str, np.ndarray]:
    H, W = record["image_shape"]
    shape = (H, W)
    masks = {c: np.zeros(shape, dtype=bool) for c in ("sgd",) + CONFOUNDER_CLASSES}
    for poly in record.get("polygons", []):
        c = poly["class"]
        if c not in masks:
            continue
        masks[c] |= polygon_to_mask(poly["points"], shape)
    return masks


@dataclass
class FrameMetrics:
    frame: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou: float = 0.0
    fp_rock: int = 0
    fp_wake: int = 0
    fp_shadow: int = 0
    fp_other: int = 0
    gt_pixels: int = 0
    pred_pixels: int = 0
    elapsed_s: float = 0.0
    num_pred_components: int = 0
    num_gt_components: int = 0
    error: str | None = None

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")


def score_frame(gt_sgd: np.ndarray, pred_sgd: np.ndarray, confounder_masks: dict) -> dict:
    tp = int((gt_sgd & pred_sgd).sum())
    fp = int((~gt_sgd & pred_sgd).sum())
    fn = int((gt_sgd & ~pred_sgd).sum())
    union = int((gt_sgd | pred_sgd).sum())
    iou = (tp / union) if union else float("nan")
    fp_mask = pred_sgd & ~gt_sgd
    counts = {
        f"fp_{c}": int((fp_mask & confounder_masks[c]).sum()) for c in CONFOUNDER_CLASSES
    }
    counted = sum(counts.values())
    counts["fp_other"] = max(fp - counted, 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": iou,
        **counts,
    }


def build_detector(kind: str, data_dir: Path):
    kind = kind.lower()
    if kind == "redesigned":
        from sgd_toolkit.detectors import RedesignedSGDDetector

        return RedesignedSGDDetector(
            base_path=str(data_dir),
            use_ml=False,  # rule-based by default for eval; flip on once ML model is verified
        )
    if kind == "improved":
        from sgd_toolkit.detectors import ImprovedSGDDetector

        return ImprovedSGDDetector(base_path=str(data_dir), use_ml=False)
    if kind == "integrated":
        from sgd_toolkit.detectors import IntegratedSGDDetector

        return IntegratedSGDDetector(base_path=str(data_dir), use_ml=False)
    raise ValueError(f"Unknown detector: {kind}")


def run_detector_on_frame(detector, frame: int) -> tuple[np.ndarray, dict]:
    """Run detector on frame and return (sgd_mask, characteristics)."""
    data = detector.load_frame_data(frame)
    masks = detector.segment_ocean_land_waves(data["rgb_aligned"])
    sgd_mask, plume_info, characteristics = detector.detect_sgd_plumes(
        data["thermal"], masks
    )
    return sgd_mask.astype(bool), {
        "num_plumes": len(plume_info),
        "characteristics": characteristics,
    }


def evaluate(
    data_dir: Path,
    labels_dir: Path,
    detector_kind: str,
    frames: list[int] | None = None,
) -> list[FrameMetrics]:
    detector = build_detector(detector_kind, data_dir)

    label_files = sorted(labels_dir.glob("*.labels.json"))
    if frames is not None:
        label_files = [p for p in label_files if int(p.name.split(".")[0]) in frames]
    if not label_files:
        raise SystemExit(f"No label files found under {labels_dir}")

    results: list[FrameMetrics] = []
    for lp in label_files:
        frame_id = int(lp.name.split(".")[0])
        record = load_labels(lp)
        gt_masks = labels_to_masks(record)
        gt_sgd = gt_masks["sgd"]
        confounders = {c: gt_masks[c] for c in CONFOUNDER_CLASSES}

        fm = FrameMetrics(frame=f"{frame_id:04d}")
        t0 = time.perf_counter()
        try:
            pred_sgd, diag = run_detector_on_frame(detector, frame_id)
            # If prediction shape differs from label shape, skip — label was saved for a different alignment.
            if pred_sgd.shape != gt_sgd.shape:
                fm.error = f"shape_mismatch pred={pred_sgd.shape} gt={gt_sgd.shape}"
                fm.elapsed_s = time.perf_counter() - t0
                results.append(fm)
                continue
            scores = score_frame(gt_sgd, pred_sgd, confounders)
            fm.tp = scores["tp"]
            fm.fp = scores["fp"]
            fm.fn = scores["fn"]
            fm.iou = scores["iou"]
            fm.fp_rock = scores["fp_rock"]
            fm.fp_wake = scores["fp_wake"]
            fm.fp_shadow = scores["fp_shadow"]
            fm.fp_other = scores["fp_other"]
            fm.gt_pixels = int(gt_sgd.sum())
            fm.pred_pixels = int(pred_sgd.sum())
            fm.num_pred_components = int(diag["num_plumes"])
            # Count GT components
            from skimage import measure as sk_measure

            _, n_gt = sk_measure.label(gt_sgd, connectivity=2, return_num=True)
            fm.num_gt_components = int(n_gt)
        except Exception as e:
            fm.error = repr(e)
        finally:
            fm.elapsed_s = time.perf_counter() - t0
        results.append(fm)
    return results


def render_report(results: list[FrameMetrics], detector_kind: str) -> str:
    ok = [r for r in results if r.error is None]
    n_frames = len(ok)
    tp = sum(r.tp for r in ok)
    fp = sum(r.fp for r in ok)
    fn = sum(r.fn for r in ok)
    pr = tp / (tp + fp) if (tp + fp) else float("nan")
    rc = tp / (tp + fn) if (tp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")

    fp_rock = sum(r.fp_rock for r in ok)
    fp_wake = sum(r.fp_wake for r in ok)
    fp_shadow = sum(r.fp_shadow for r in ok)
    fp_other = sum(r.fp_other for r in ok)

    header = f"# Evaluation: {detector_kind}\n\n"
    summary = (
        f"**Frames evaluated:** {n_frames}  (errors: {len(results) - n_frames})\n\n"
        f"## Aggregate\n\n"
        f"| metric | value |\n|---|---|\n"
        f"| precision | {pr:.3f} |\n"
        f"| recall | {rc:.3f} |\n"
        f"| IoU | {iou:.3f} |\n"
        f"| TP pixels | {tp} |\n"
        f"| FP pixels | {fp} |\n"
        f"| FN pixels | {fn} |\n"
        f"| FP rock | {fp_rock} |\n"
        f"| FP wake | {fp_wake} |\n"
        f"| FP shadow | {fp_shadow} |\n"
        f"| FP other | {fp_other} |\n\n"
    )
    rows = ["## Per-frame\n\n| frame | precision | recall | IoU | TP | FP | FN | FP_rock | FP_wake | FP_shadow | elapsed_s | error |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(
            f"| {r.frame} | {r.precision:.3f} | {r.recall:.3f} | {r.iou:.3f} | "
            f"{r.tp} | {r.fp} | {r.fn} | {r.fp_rock} | {r.fp_wake} | {r.fp_shadow} | "
            f"{r.elapsed_s:.2f} | {r.error or ''} |"
        )
    return header + summary + "\n".join(rows) + "\n"


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate an SGD detector against labeled frames")
    ap.add_argument("--data", required=True, help="Path to data directory")
    ap.add_argument("--labels-dir", default=None, help="Labels directory (default: <data>/labels)")
    ap.add_argument(
        "--detector",
        choices=("redesigned", "improved", "integrated"),
        default="redesigned",
        help="Which detector to evaluate",
    )
    ap.add_argument("--frames", nargs="*", type=int, default=None, help="Subset of frames")
    ap.add_argument("--report", default=None, help="Write markdown report here")
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    labels = Path(args.labels_dir) if args.labels_dir else data / "labels"
    results = evaluate(data, labels, args.detector, args.frames)
    report = render_report(results, args.detector)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"Wrote {out}")
    print(report)


if __name__ == "__main__":
    main()
