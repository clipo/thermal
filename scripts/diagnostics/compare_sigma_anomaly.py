#!/usr/bin/env python3
"""Compare Sigma_anomaly across pipeline arms that differ only by injected bias.

The question this sharpens
--------------------------
The polygon-area comparison showed the threshold-dependent product moves a lot
under an injected radial bias (about +32% total area at the measured magnitude).
Sigma_anomaly is the metric the paper actually compares across sites, and it is
built to be threshold-independent, so it needs testing on its own terms.

But "does Sigma_anomaly change?" is the wrong question. The paper's claims are
COMPARATIVE: which sites discharge more than which others. A bias that inflates
every site's Sigma_anomaly by the same factor leaves every comparative claim
intact. A bias that inflates some sites and deflates others invalidates them
even if the global total barely moves.

So three things are reported, in increasing order of importance:

  global       total Sigma_anomaly over the whole raster
  per-site     Sigma_anomaly summed inside each baseline polygon, and how far
               individual sites move
  rank         Spearman correlation of per-site Sigma_anomaly between baseline
               and each arm

The rank correlation is the one that decides whether the paper's conclusions
survive. Near 1.0 means the ordering of sites is preserved and the comparative
findings hold regardless of the absolute shift. A meaningful drop means the
bias reorders sites, and the published comparisons would need the correction
resolved first.

Usage
-----
    python scripts/diagnostics/compare_sigma_anomaly.py \\
        --baseline sgd_output/sensitivity/rasters/f4_baseline_anom.npz \\
        --arm m024:sgd_output/sensitivity/rasters/f4_m024_anom.npz \\
        --arm m048:sgd_output/sensitivity/rasters/f4_m048_anom.npz \\
        --arm p024:sgd_output/sensitivity/rasters/f4_p024_anom.npz \\
        --polygons sgd_output/sensitivity/f4_baseline.geojson
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_raster(path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "anomaly": d["anomaly"].astype(np.float64),
        "obs": d["observations"],
        "minlon": float(d["bbox_min_lon"]), "maxlon": float(d["bbox_max_lon"]),
        "minlat": float(d["bbox_min_lat"]), "maxlat": float(d["bbox_max_lat"]),
        "res_m": float(d["grid_resolution_m"]),
        "flat_field": str(d["flat_field"]) if "flat_field" in d else "",
    }


def sigma_total(r: dict) -> float:
    """Sigma_anomaly in m^2 degC over the whole raster."""
    a = r["anomaly"]
    return float(np.nansum(a) * r["res_m"] ** 2)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def polygon_cell_masks(polys: list, r: dict):
    """Yield (index, flat cell indices) for each polygon, on the raster grid."""
    from matplotlib.path import Path as MplPath

    gy, gx = r["anomaly"].shape
    mpd_lat = 111320.0
    clat = 0.5 * (r["minlat"] + r["maxlat"])
    mpd_lon = 111320.0 * math.cos(math.radians(clat))
    res = r["res_m"]

    for i, ring in enumerate(polys):
        lons = np.array([c[0] for c in ring])
        lats = np.array([c[1] for c in ring])
        c0 = int(np.floor((lons.min() - r["minlon"]) * mpd_lon / res))
        c1 = int(np.ceil((lons.max() - r["minlon"]) * mpd_lon / res))
        r0 = int(np.floor((lats.min() - r["minlat"]) * mpd_lat / res))
        r1 = int(np.ceil((lats.max() - r["minlat"]) * mpd_lat / res))
        c0, c1 = max(c0, 0), min(c1 + 1, gx)
        r0, r1 = max(r0, 0), min(r1 + 1, gy)
        if c1 <= c0 or r1 <= r0:
            yield i, None
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        cell_lon = r["minlon"] + (cc + 0.5) * res / mpd_lon
        cell_lat = r["minlat"] + (rr + 0.5) * res / mpd_lat
        pts = np.column_stack([cell_lon.ravel(), cell_lat.ravel()])
        inside = MplPath(np.column_stack([lons, lats])).contains_points(pts)
        if not inside.any():
            yield i, None
            continue
        yield i, (rr.ravel()[inside], cc.ravel()[inside])


def main():
    ap = argparse.ArgumentParser(
        description="Compare Sigma_anomaly globally, per site, and by site ranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arm", action="append", default=[], help="name:path.npz")
    ap.add_argument("--polygons", required=True,
                    help="Baseline geojson; its polygons define the sites compared")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = load_raster(Path(args.baseline))
    arms = {}
    for spec in args.arm:
        name, path = spec.split(":", 1)
        arms[name] = load_raster(Path(path))

    for name, r in arms.items():
        if r["anomaly"].shape != base["anomaly"].shape:
            raise SystemExit(
                f"arm {name} grid {r['anomaly'].shape} != baseline "
                f"{base['anomaly'].shape}; rasters must share a grid to compare."
            )

    fc = json.loads(Path(args.polygons).read_text())
    rings = [f["geometry"]["coordinates"][0] for f in fc.get("features", [])
             if (f.get("geometry") or {}).get("type") == "Polygon"]
    print(f"grid {base['anomaly'].shape}, {base['res_m']} m cells, {len(rings)} baseline sites")

    b_tot = sigma_total(base)
    print(f"\n--- global Sigma_anomaly (m²·°C) ---")
    print(f"  {'baseline':<10} {b_tot:>14,.0f}")
    rows = []
    for name, r in arms.items():
        t = sigma_total(r)
        print(f"  {name:<10} {t:>14,.0f}   {100*(t-b_tot)/b_tot:>+7.1f}%")
        rows.append({"arm": name, "global_sigma": t,
                     "global_change_pct": 100 * (t - b_tot) / b_tot})

    # Per-site sums
    idx_cache = []
    for i, sel in polygon_cell_masks(rings, base):
        idx_cache.append(sel)

    def per_site(r):
        out = np.full(len(rings), np.nan)
        a = r["anomaly"]
        for i, sel in enumerate(idx_cache):
            if sel is None:
                continue
            out[i] = float(np.nansum(a[sel[0], sel[1]]) * r["res_m"] ** 2)
        return out

    b_site = per_site(base)
    valid = np.isfinite(b_site) & (b_site > 0)
    print(f"\n--- per-site Sigma_anomaly over {int(valid.sum())} sites ---")
    print(f"  {'arm':<10} {'median Δ%':>10} {'p10 Δ%':>9} {'p90 Δ%':>9} "
          f"{'Spearman ρ':>11} {'sites >±25%':>12}")

    for row in rows:
        r = arms[row["arm"]]
        s = per_site(r)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100.0 * (s - b_site) / b_site
        pv = pct[valid]
        rho = spearman(b_site[valid], s[valid])
        big = float(np.mean(np.abs(pv) > 25.0) * 100)
        row["per_site_sigma"] = [None if not np.isfinite(v) else float(v) for v in s]
        row.update({
            "median_site_change_pct": float(np.nanmedian(pv)),
            "p10_site_change_pct": float(np.nanpercentile(pv, 10)),
            "p90_site_change_pct": float(np.nanpercentile(pv, 90)),
            "spearman_rho": rho,
            "pct_sites_over_25pct_change": big,
        })
        print(f"  {row['arm']:<10} {row['median_site_change_pct']:>+10.1f} "
              f"{row['p10_site_change_pct']:>+9.1f} {row['p90_site_change_pct']:>+9.1f} "
              f"{rho:>11.4f} {big:>11.0f}%")

    worst_rho = min((r["spearman_rho"] for r in rows), default=1.0)
    worst_glob = max((abs(r["global_change_pct"]) for r in rows), default=0.0)
    print()
    if worst_rho > 0.95:
        print(f"  SITE RANKING PRESERVED (worst Spearman ρ = {worst_rho:.4f}).")
        print(f"  Absolute Sigma_anomaly shifts by up to {worst_glob:.0f}%, but the "
              f"ordering of sites is essentially unchanged, so comparative claims "
              f"across sites are robust to this bias.")
    elif worst_rho > 0.85:
        print(f"  SITE RANKING MOSTLY PRESERVED (worst ρ = {worst_rho:.4f}), but some "
              f"reordering occurs. Check whether any site the paper singles out "
              f"moves.")
    else:
        print(f"  SITE RANKING NOT PRESERVED (worst ρ = {worst_rho:.4f}). The bias "
              f"reorders sites; comparative claims are affected and the correction "
              f"must be resolved.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).with_suffix(".json").write_text(json.dumps({
            "baseline_global_sigma": b_tot, "n_sites": int(valid.sum()),
            "baseline_per_site_sigma": [None if not np.isfinite(v) else float(v) for v in b_site],
            "arms": rows,
        }, indent=2))
        print(f"  wrote {Path(args.output).with_suffix('.json')}")


if __name__ == "__main__":
    main()
