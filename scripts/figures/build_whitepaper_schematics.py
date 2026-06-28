#!/usr/bin/env python3
"""Generate the two schematic figures used only by the whitepaper:

  1. Cliff-zone exclusion schematic (lava-tube conduit at a low bay vs a
     vertical cliff face) -> docs/images/whitepaper/cliff_exclusion_schematic.png
  2. Sigma_anomaly integration schematic (per-cell mean anomaly summed
     over a polygon footprint) -> docs/images/whitepaper/sigma_anomaly_schematic.png

Style follows the shared research conventions: sans-serif, colorblind-friendly
Okabe-Ito palette, no figure titles (descriptive text goes in captions),
PNG at 300 dpi, 7 in wide.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle

THERMAL = Path(__file__).resolve().parent.parent.parent
OUT_DIR = THERMAL / "docs" / "images" / "whitepaper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colorblind-friendly palette
OI = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.8,
})


def cliff_exclusion():
    """Cross-section: SGD emerges through a collapsed lava tube at a low bay,
    but a vertical cliff face has no conduit at sea level so no SGD forms."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    sea_level = 1.6

    # Ocean
    ax.add_patch(Rectangle((0, 0), 10, sea_level, facecolor=OI["sky"],
                           edgecolor="none", alpha=0.55, zorder=0))

    # Land profile: low bay on the left, rising to a high cliff on the right.
    lx = np.linspace(0, 10, 400)
    ground = np.piecewise(
        lx,
        [lx < 3.0, (lx >= 3.0) & (lx < 6.2), lx >= 6.2],
        [
            lambda x: sea_level + 0.15 * (3.0 - x),          # gentle bay shore
            lambda x: sea_level + 0.9 * (x - 3.0),            # rising slope
            lambda x: sea_level + 0.9 * 3.2,                  # plateau top
        ],
    )
    # Vertical cliff face at x ~= 7.4 (truncate the plateau to the sea).
    cliff_x = 7.4
    # Build the land polygon explicitly: bay shore, rising slope, plateau,
    # then a vertical cliff dropping to the sea.
    land_pts = [(0.0, 0.0)]
    for x, y in zip(lx, ground):
        if x <= cliff_x:
            land_pts.append((x, y))
    land_pts.append((cliff_x, sea_level))   # cliff drops vertically to the sea
    land_pts.append((cliff_x, 0))
    land_pts.append((0, 0))
    ax.add_patch(MplPolygon(land_pts, closed=True, facecolor="#d9cdbf",
                            edgecolor=OI["black"], linewidth=1.0, zorder=2))

    # Aquifer / water table (dashed) inside the land mass.
    wt_x = lx[lx <= cliff_x]
    wt_y = np.minimum(ground[lx <= cliff_x] - 0.35, sea_level + 0.55)
    wt_y = np.maximum(wt_y, sea_level + 0.05)
    ax.plot(wt_x, wt_y, color=OI["blue"], lw=1.2, ls=(0, (4, 2)), zorder=3)
    ax.text(4.6, sea_level + 0.72, "water table", color=OI["blue"], fontsize=8,
            rotation=14, ha="center")

    # Lava-tube conduit at the low bay: a tube from the aquifer to the sea.
    tube_x = [2.55, 2.1, 1.55, 1.05]
    tube_y = [sea_level + 0.5, sea_level + 0.35, sea_level + 0.2, sea_level + 0.02]
    ax.plot(tube_x, tube_y, color=OI["vermillion"], lw=3.2, solid_capstyle="round",
            zorder=4)
    ax.text(1.9, sea_level + 0.62, "collapsed\nlava-tube conduit",
            color=OI["vermillion"], fontsize=8, ha="center")

    # SGD plume emerging at the bay shoreline (cool freshwater spreading offshore).
    plume = MplPolygon([(1.05, sea_level), (0.2, sea_level - 0.05),
                        (0.2, sea_level - 0.7), (1.7, sea_level - 0.55),
                        (1.2, sea_level - 0.2)],
                       closed=True, facecolor=OI["green"], edgecolor="none",
                       alpha=0.55, zorder=3)
    ax.add_patch(plume)
    ax.annotate("SGD plume\n(cool, detected)", xy=(0.8, sea_level - 0.4),
                xytext=(0.3, 0.55), fontsize=8, color=OI["green"],
                ha="left",
                arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.0))

    # Cliff face: aquifer truncated, no conduit at sea level -> no SGD.
    ax.annotate("vertical cliff face:\nno conduit at sea level,\nno SGD",
                xy=(cliff_x, sea_level + 0.5), xytext=(8.0, 3.4),
                fontsize=8, color=OI["black"], ha="center",
                arrowprops=dict(arrowstyle="->", color=OI["black"], lw=1.0))

    # Elevation reference + 80 m SRTM exclusion threshold marker.
    ax.annotate("", xy=(9.6, sea_level), xytext=(9.6, ground[-1]),
                arrowprops=dict(arrowstyle="<->", color=OI["black"], lw=0.9))
    ax.text(9.72, (sea_level + ground[-1]) / 2, "cliff > 80 m\n(SRTM filter\nexcludes zone)",
            rotation=90, va="center", ha="left", fontsize=7.5)

    # Sea-level label
    ax.plot([0, 10], [sea_level, sea_level], color=OI["blue"], lw=0.7, alpha=0.6,
            zorder=1)
    ax.text(0.1, sea_level + 0.08, "sea level", fontsize=7, color=OI["blue"])

    # Zone brackets along the bottom.
    ax.text(1.4, 0.12, "low bay: SGD-plausible", fontsize=8, ha="center",
            color=OI["green"], weight="bold")
    ax.text(8.4, 0.12, "cliff coast: excluded", fontsize=8, ha="center",
            color=OI["vermillion"], weight="bold")

    fig.tight_layout(pad=0.4)
    out = OUT_DIR / "cliff_exclusion_schematic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def sigma_anomaly():
    """Show a 1 m grid of per-cell mean cold-anomaly values, a plume polygon
    drawn over them, and the integral that defines Sigma_anomaly (m^2 C)."""
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                  gridspec_kw={"width_ratios": [1.25, 1]})

    # Synthetic but realistic anomaly field: a plume centered off-center.
    n = 9
    yy, xx = np.mgrid[0:n, 0:n]
    cx, cy = 3.4, 4.2
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    anom = 1.15 * np.exp(-r2 / 7.0)
    anom[anom < 0.05] = 0.0

    im = ax.imshow(anom, origin="lower", cmap="viridis", vmin=0, vmax=1.2)
    # Annotate each cell with its mean anomaly.
    for i in range(n):
        for j in range(n):
            v = anom[i, j]
            if v >= 0.05:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if v > 0.6 else "black")

    # Plume polygon (cells above edge threshold) drawn as an outline.
    poly = MplPolygon([(0.5, 1.5), (1.4, 4.6), (3.5, 6.4), (5.6, 5.2),
                       (6.2, 2.8), (4.4, 0.7), (2.0, 0.6)],
                      closed=True, fill=False, edgecolor=OI["vermillion"],
                      lw=2.0)
    ax.add_patch(poly)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("1 m x 1 m grid cells", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean cold anomaly (°C)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.text(0.02, 1.04, "plume polygon", transform=ax.transAxes,
            color=OI["vermillion"], fontsize=8, weight="bold")

    # Right panel: the definition, as math + plain words.
    axt.axis("off")
    axt.text(0.0, 0.93, r"$\Sigma_{\mathrm{anomaly}}$", fontsize=20,
             color=OI["blue"], va="top")
    axt.text(0.0, 0.72,
             r"$= \int_{\mathrm{polygon}} \overline{(T_{base}-T)}\; dA$",
             fontsize=13, va="top")
    axt.text(0.0, 0.55,
             r"$\approx \sum_{\mathrm{cells}} \overline{\mathrm{anom}}_c \cdot A_{cell}$",
             fontsize=13, va="top")
    axt.text(0.0, 0.36,
             "Sum each cell's mean cold\nanomaly (°C) times its area\n(1 m²), over the polygon.",
             fontsize=9, va="top")
    axt.text(0.0, 0.13,
             "Units: m²·°C.\nIndependent of the detection\nthreshold and of absolute\ntemperature between flights.",
             fontsize=9, va="top", color=OI["black"])

    fig.tight_layout(pad=0.5)
    out = OUT_DIR / "sigma_anomaly_schematic.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    cliff_exclusion()
    sigma_anomaly()
