#!/usr/bin/env python3
"""Render a single island-wide image showing all 29 per-flight cold-anomaly
rasters placed at their actual lat/lon footprints, on a satellite basemap.
Same view as the master `rapa_nui_all_anomaly.kml` GroundOverlay layer in
Google Earth, but as a static figure for publication.

Each flight's anomaly raster is plotted with its bbox extent on a
matplotlib axis with Esri WorldImagery basemap underneath. Filter rules
(obs >= 5, anomaly <= 3, soft-land filter) match `mask_anomaly_pngs.py`
so the visible cold anomaly is the cleaned-up version, not raw.

Output: sgd_output/figures/rapa_nui_all_anomaly_mosaic.{png,pdf}
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


THERMAL = Path(__file__).resolve().parent.parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(SGD_OUTPUT / "figures"))
    ap.add_argument("--vmax", type=float, default=1.5,
                    help="color scale top in °C below baseline (default 1.5)")
    ap.add_argument("--min-obs", type=int, default=5)
    ap.add_argument("--max-realistic", type=float, default=3.0)
    ap.add_argument("--max-land-anomaly", type=float, default=0.5,
                    help="drop satellite-classified-LAND cells with anomaly > this")
    ap.add_argument("--alpha", type=float, default=0.85,
                    help="overlay opacity (default 0.85)")
    ap.add_argument("--no-basemap", action="store_true",
                    help="skip Esri satellite basemap (faster, no internet needed)")
    args = ap.parse_args()

    flights = []
    for spread_dir in sorted(SGD_OUTPUT.glob("*_spread")):
        slug = spread_dir.name[: -len("_spread")]
        npz = spread_dir / f"{slug}_anomaly.npz"
        if not npz.exists():
            continue
        wm = spread_dir / f"{slug}_water_mask.npz"
        try:
            d = np.load(npz)
            water = np.load(wm)["is_water"] if wm.exists() else None
        except Exception as e:
            print(f"  skip {slug}: {e}")
            continue
        anomaly = d["anomaly"]
        obs = d["observations"]
        finite = np.isfinite(anomaly)
        keep = finite & (obs >= args.min_obs) & (anomaly <= args.max_realistic)
        if water is not None and water.shape == anomaly.shape:
            cliff_shadow = ~water & (anomaly > args.max_land_anomaly)
            keep = keep & ~cliff_shadow
            # Isolation filter: size-based. Drop components < 1% of
            # the largest component. Catches isolated patches like the
            # basalt Ahu Tongariki platform.
            try:
                from scipy import ndimage
                labeled, n_comp = ndimage.label(keep)
                if n_comp > 0:
                    sizes = ndimage.sum(keep, labeled, range(1, n_comp + 1))
                    if sizes.size:
                        min_size = max(1, 0.10 * float(sizes.max()))
                        keep_ids = np.where(sizes >= min_size)[0] + 1
                        keep = keep & np.isin(labeled, keep_ids)
            except ImportError:
                pass
        display = np.where(keep, anomaly, np.nan)
        flights.append({
            "slug": slug,
            "display": display,
            "minlon": float(d["bbox_min_lon"]), "maxlon": float(d["bbox_max_lon"]),
            "minlat": float(d["bbox_min_lat"]), "maxlat": float(d["bbox_max_lat"]),
        })

    if not flights:
        raise SystemExit("No anomaly rasters found.")
    print(f"Loaded {len(flights)} flights for mosaic.")

    minlon = min(f["minlon"] for f in flights)
    maxlon = max(f["maxlon"] for f in flights)
    minlat = min(f["minlat"] for f in flights)
    maxlat = max(f["maxlat"] for f in flights)
    centerlat = 0.5 * (minlat + maxlat)

    fig, ax = plt.subplots(figsize=(14, 13), constrained_layout=True)
    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_aspect(1.0 / math.cos(math.radians(centerlat)))

    # Basemap
    if not args.no_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, crs="EPSG:4326",
                            source=ctx.providers.Esri.WorldImagery,
                            attribution_size=7, zorder=1)
            print("  satellite basemap added")
        except Exception as e:
            print(f"  basemap fetch failed: {e}")

    # Layer all anomaly rasters
    norm = Normalize(vmin=0, vmax=args.vmax)
    for f in flights:
        ax.imshow(
            f["display"], origin="lower", cmap="YlOrRd",
            norm=norm, extent=(f["minlon"], f["maxlon"], f["minlat"], f["maxlat"]),
            alpha=args.alpha, zorder=3, interpolation="bilinear",
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap="YlOrRd")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02,
                      label="Cold anomaly  (°C below ambient baseline)")

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}"))
    ax.tick_params(labelsize=8)

    ax.set_title(
        f"Rapa Nui — submarine groundwater discharge cold-anomaly raster mosaic\n"
        f"{len(flights)} thermal-drone flights, integrated cold-anomaly content "
        f"per cell after water-aware quality filtering",
        fontsize=11, pad=10,
    )

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "rapa_nui_all_anomaly_mosaic.png"
    out_pdf = output_dir / "rapa_nui_all_anomaly_mosaic.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
