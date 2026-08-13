#!/usr/bin/env python3
"""Figures for the per-frame thermal bias investigation (README section).

Figure 1  The measured image-position bias.
          (a) radial profile from two independent methods on the same flight,
              one that projects pixels to ground cells and one that does not,
              so agreement rules out the projection as the source;
          (b) centre-minus-edge contrast per flight, with intervals.

Figure 2  How far the products move when a bias of that size is injected.
          (a) per-site Sigma_anomaly, injected against baseline, on log axes
              with the 1:1 line, which is where rank preservation is visible;
          (b) percentage change by metric, showing which products are
              sensitive and which are not.

Style follows the project convention: no titles, Arial, Okabe-Ito palette,
300 dpi, 7 in wide.

Usage:
    python scripts/figures/build_frame_bias_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THERMAL = Path(__file__).resolve().parent.parent.parent
# Read from the small tracked result files, not from sgd_output/, which is
# gitignored. This keeps the figures rebuildable on a fresh clone without
# re-running the whole analysis.
RESULTS = THERMAL / "docs" / "results" / "frame_bias"
DIAG = RESULTS
SENS = RESULTS
OUT = THERMAL / "docs" / "images" / "frame_bias"

# Okabe-Ito
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Block-sampled measurements first (the trustworthy ones), then the
# consecutive-frame measurements, which describe one leg of each transect.
FLIGHTS = [
    ("flight4_vaihu_east_full", "Flight 4 Vaihu East"),
    ("flight11_blocksampled", "Flight 11 Hivahiva"),
    ("flight10_anakena_to_west", "Flight 10 Anakena (1 leg)"),
    ("flight8_hekii_west", "Flight 8 Hekii West (1 leg)"),
]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def figure1(out_png: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)

    # (a) two independent methods, flight 4
    ax = axes[0]
    paired = load(DIAG / "radial_paired_flight4_vaihu_east_full.json")
    single = load(DIAG / "single_frame_radial_flight4_vaihu_east_full.json")

    if paired:
        c = np.array(paired["r_centers"])
        y = np.array(paired["profile_c"])
        ax.fill_between(c, paired["profile_ci_lo_c"], paired["profile_ci_hi_c"],
                        color=BLUE, alpha=0.20, linewidth=0)
        ax.plot(c, y, color=BLUE, lw=1.4, marker="o", ms=3,
                label="flight 4, paired, ground held fixed")
    if single:
        c2 = np.array(single["r_centers"])
        y2 = np.array(single["stacked_profile_c"])
        ax.fill_between(c2, single["stacked_ci_lo_c"], single["stacked_ci_hi_c"],
                        color=ORANGE, alpha=0.20, linewidth=0)
        ax.plot(c2, y2, color=ORANGE, lw=1.4, marker="s", ms=3,
                label="flight 4, raw frames, no projection")
    # A second flight, measured the same way. Three near-identical curves show
    # both that the projection is not responsible and that the pattern is a
    # stable property of the camera rather than of either survey.
    f11 = load(DIAG / "radial_paired_flight11_blocksampled.json")
    if f11:
        c3 = np.array(f11["r_centers"])
        ax.plot(c3, np.array(f11["profile_c"]), color=GREEN, lw=1.4, marker="^",
                ms=3, label="flight 11, paired")

    ax.axhline(0, color=GREY, lw=0.5, ls=":")
    ax.set_xlabel("Normalised radius from frame centre")
    ax.set_ylabel("Residual (°C)")
    ax.legend(fontsize=5.8, loc="upper left")
    ax.text(-0.20, 1.02, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    # (b) per-flight contrast
    ax = axes[1]
    labels, vals, los, his = [], [], [], []
    for slug, nice in FLIGHTS:
        d = load(DIAG / f"radial_paired_{slug}.json")
        if not d:
            continue
        labels.append(nice)
        vals.append(d["inner_minus_outer_c"])
        lo, hi = d["inner_minus_outer_ci_c"]
        los.append(lo); his.append(hi)

    ypos = np.arange(len(labels))
    vals = np.array(vals); los = np.array(los); his = np.array(his)
    ax.errorbar(vals, ypos, xerr=[vals - los, his - vals], fmt="o", ms=4,
                color=BLUE, ecolor=BLUE, elinewidth=1.2, capsize=2.5)
    ax.axvline(0, color=GREY, lw=0.5, ls=":")
    ax.axvline(-0.25, color=VERM, lw=0.9, ls="--")
    ax.text(-0.25, -0.62, "detection threshold", color=VERM, fontsize=6,
            va="bottom", ha="center")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(len(labels) - 0.5, -0.9)
    ax.set_xlabel("Centre − edge (°C)")
    ax.text(-0.55, 1.02, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def figure2(out_png: Path):
    sig = load(SENS / "f4_sigma_comparison.json")
    poly = load(SENS / "f4_comparison.json")
    if not sig:
        raise SystemExit("missing f4_sigma_comparison.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=300)

    # (a) per-site Sigma_anomaly, injected vs baseline
    ax = axes[0]
    b = np.array([np.nan if v is None else v for v in sig["baseline_per_site_sigma"]])
    styles = {"m024": (BLUE, "o", "−0.24 °C (measured)"),
              "m048": (VERM, "^", "−0.48 °C (2×)"),
              "p024": (GREEN, "s", "+0.24 °C (reversed)")}
    for row in sig["arms"]:
        name = row["arm"]
        if name not in styles:
            continue
        col, mk, lab = styles[name]
        s = np.array([np.nan if v is None else v for v in row["per_site_sigma"]])
        ok = np.isfinite(b) & np.isfinite(s) & (b > 0) & (s > 0)
        ax.scatter(b[ok], s[ok], s=11, color=col, marker=mk, alpha=0.75,
                   edgecolors="none",
                   label=f"{lab}  ρ={row['spearman_rho']:.4f}")
    lim_lo = np.nanmin(b[b > 0]) * 0.6
    lim_hi = np.nanmax(b) * 1.8
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=GREY, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Baseline Σ$_{anomaly}$ per site (m²·°C)")
    ax.set_ylabel("With injected bias (m²·°C)")
    ax.legend(fontsize=6, loc="upper left")
    ax.text(-0.22, 1.02, "a", transform=ax.transAxes, fontsize=10, fontweight="bold")

    # (b) change by metric
    ax = axes[1]
    def poly_row(name):
        for r in (poly or {}).get("arms", []):
            if r["arm"] == name:
                return r
        return {}
    def sig_row(name):
        for r in sig["arms"]:
            if r["arm"] == name:
                return r
        return {}

    metrics = ["plume\ncount", "polygon\narea", "global\nΣ", "median\nsite Σ"]
    arms = [("m024", BLUE, "−0.24 °C"), ("m048", VERM, "−0.48 °C"),
            ("p024", GREEN, "+0.24 °C")]
    x = np.arange(len(metrics))
    w = 0.26
    for k, (name, col, lab) in enumerate(arms):
        pr, sr = poly_row(name), sig_row(name)
        vals = [pr.get("count_change_pct", np.nan),
                pr.get("area_change_pct", np.nan),
                sr.get("global_change_pct", np.nan),
                sr.get("median_site_change_pct", np.nan)]
        ax.bar(x + (k - 1) * w, vals, width=w, color=col, label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=6.5)
    ax.set_ylabel("Change from baseline (%)")
    ax.set_ylim(-52, 88)          # headroom so the +60% bar clears the legend
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    ax.text(-0.20, 1.02, "b", transform=ax.transAxes, fontsize=10, fontweight="bold")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    figure1(OUT / "fig1_measured_radial_bias.png")
    figure2(OUT / "fig2_sensitivity.png")
