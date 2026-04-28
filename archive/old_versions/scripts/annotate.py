#!/usr/bin/env python3
"""
Polygon-annotation tool for building SGD validation labels.

Usage:

    python scripts/annotate.py --data data/100MEDIA --frame 228
    python scripts/annotate.py --data data/100MEDIA --start 200 --end 260 --step 5

Click points to draw a polygon, press the class key to commit it with that
class (s=sgd, r=rock, w=wake, h=shadow), 'u' to undo the last polygon,
'n'/'p' to move to next/previous frame, 'q' to quit. Saves each frame's
labels to `<data_dir>/labels/<frame>.labels.json` alongside the thermal file.

Label schema (intentionally simple — not formal GeoJSON — because labels are
in pixel space of the thermal image and need to be trivially round-trippable
with numpy masks):

    {
      "frame": "0228",
      "image_shape": [height, width],
      "polygons": [
        {"class": "sgd",    "points": [[x, y], ...]},
        {"class": "rock",   "points": [...]},
        {"class": "wake",   "points": [...]},
        {"class": "shadow", "points": [...]}
      ]
    }

The image displayed is the RGB frame aligned to the thermal FOV, because the
detector operates in that coordinate system; polygons drawn here align
directly with detector output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from sgd_toolkit.detectors import IntegratedSGDDetector


CLASS_KEYS = {"s": "sgd", "r": "rock", "w": "wake", "h": "shadow"}
CLASS_COLORS = {
    "sgd": "#00d4ff",
    "rock": "#9a5a2a",
    "wake": "#e8e8e8",
    "shadow": "#7a4bc7",
}


class AnnotationSession:
    def __init__(self, data_dir: Path, labels_dir: Path, frames: list[int]):
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.frames = frames
        self.idx = 0
        self.detector = IntegratedSGDDetector(
            base_path=str(data_dir), use_ml=False
        )

        self.fig, (self.ax_rgb, self.ax_thermal) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.current_points: list[tuple[float, float]] = []
        self.current_line = None
        self.polygons: list[dict] = []  # list of {class, points, patch}

        self._load_frame()
        self._redraw()

    def _labels_path(self, frame: int) -> Path:
        return self.labels_dir / f"{frame:04d}.labels.json"

    def _load_frame(self):
        frame = self.frames[self.idx]
        data = self.detector.load_frame_data(frame)
        self.rgb = data["rgb_aligned"]
        self.thermal = data["thermal"]
        self.current_points = []
        self.polygons = []

        lp = self._labels_path(frame)
        if lp.exists():
            with open(lp) as f:
                saved = json.load(f)
            for poly in saved.get("polygons", []):
                self.polygons.append(
                    {
                        "class": poly["class"],
                        "points": [tuple(p) for p in poly["points"]],
                        "patch": None,
                    }
                )
            print(f"[frame {frame}] loaded {len(self.polygons)} existing polygons")
        else:
            print(f"[frame {frame}] no existing labels")

    def _save(self):
        frame = self.frames[self.idx]
        record = {
            "frame": f"{frame:04d}",
            "image_shape": list(self.thermal.shape),
            "polygons": [
                {"class": p["class"], "points": [list(pt) for pt in p["points"]]}
                for p in self.polygons
            ],
        }
        with open(self._labels_path(frame), "w") as f:
            json.dump(record, f, indent=2)
        print(f"[frame {frame}] saved {len(self.polygons)} polygons → {self._labels_path(frame)}")

    def _redraw(self):
        frame = self.frames[self.idx]
        self.ax_rgb.clear()
        self.ax_thermal.clear()
        self.ax_rgb.imshow(self.rgb)
        self.ax_rgb.set_title(f"Frame {frame}  RGB (click to add polygon points)")
        self.ax_thermal.imshow(self.thermal, cmap="inferno")
        self.ax_thermal.set_title("Thermal (°C) — reference only")

        for poly in self.polygons:
            if len(poly["points"]) < 3:
                continue
            patch = MplPolygon(
                poly["points"],
                closed=True,
                facecolor=CLASS_COLORS[poly["class"]],
                edgecolor="black",
                alpha=0.35,
                linewidth=1.5,
            )
            self.ax_rgb.add_patch(patch)

        if self.current_points:
            xs = [p[0] for p in self.current_points]
            ys = [p[1] for p in self.current_points]
            self.ax_rgb.plot(xs, ys, "o-", color="yellow", markersize=4)

        counts = {c: 0 for c in CLASS_COLORS}
        for p in self.polygons:
            counts[p["class"]] = counts.get(p["class"], 0) + 1
        counts_str = "  ".join(f"{c}:{n}" for c, n in counts.items())
        self.fig.suptitle(
            f"[s]gd  [r]ock  [w]ake  s[h]adow   |   [u]ndo   [n]ext   [p]rev   [q]uit     {counts_str}"
        )
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes is not self.ax_rgb:
            return
        if event.button != 1:
            return
        self.current_points.append((event.xdata, event.ydata))
        self._redraw()

    def _commit_polygon(self, cls: str):
        if len(self.current_points) < 3:
            print("Need at least 3 points before committing a polygon.")
            return
        self.polygons.append({"class": cls, "points": list(self.current_points), "patch": None})
        self.current_points = []
        self._redraw()
        self._save()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in CLASS_KEYS:
            self._commit_polygon(CLASS_KEYS[key])
        elif key == "u":
            if self.current_points:
                self.current_points.pop()
            elif self.polygons:
                self.polygons.pop()
                self._save()
            self._redraw()
        elif key == "n":
            self._save()
            self.idx = (self.idx + 1) % len(self.frames)
            self._load_frame()
            self._redraw()
        elif key == "p":
            self._save()
            self.idx = (self.idx - 1) % len(self.frames)
            self._load_frame()
            self._redraw()
        elif key == "q":
            self._save()
            plt.close(self.fig)


def parse_args():
    ap = argparse.ArgumentParser(description="SGD validation-label annotation tool")
    ap.add_argument("--data", required=True, help="Path to data directory with MAX_*.JPG / IRX_*.irg")
    ap.add_argument("--labels-dir", default=None, help="Where to save labels (default: <data>/labels)")
    ap.add_argument("--frame", type=int, default=None, help="Annotate a single frame")
    ap.add_argument("--start", type=int, default=None, help="First frame of a range")
    ap.add_argument("--end", type=int, default=None, help="Last frame of a range (inclusive)")
    ap.add_argument("--step", type=int, default=1, help="Frame stride")
    return ap.parse_args()


def main():
    args = parse_args()
    data = Path(args.data)
    labels = Path(args.labels_dir) if args.labels_dir else data / "labels"

    if args.frame is not None:
        frames = [args.frame]
    elif args.start is not None and args.end is not None:
        frames = list(range(args.start, args.end + 1, args.step))
    else:
        raise SystemExit("Provide --frame or --start/--end")

    session = AnnotationSession(data, labels, frames)
    plt.show()


if __name__ == "__main__":
    main()
