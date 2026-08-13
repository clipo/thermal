#!/usr/bin/env python3
"""Compare pipeline runs that differ only by an injected radial bias.

What this answers
-----------------
An image-fixed radial bias of roughly 0.2 degC is present in the raw frames
(README, "Per-frame thermal bias"). It is uncorrected in the pipeline. The
question that decides whether anything must change is whether it moves the
published product.

Each arm is the SAME pipeline over the SAME frames, differing only by a
synthetic radial ramp injected through --flat-field. Comparing the arms against
the uncorrected baseline gives the sensitivity directly.

What is compared
----------------
Counts and total area are the coarse view, and can hide compensating changes: a
site lost here and another gained there leaves the count unchanged. So polygons
are also matched spatially between arms by centroid distance, which separates

    retained  a baseline site with a counterpart in the injected arm
    lost      a baseline site with no counterpart
    gained    an injected-arm site with no baseline counterpart

and reports the area change over retained sites.

What this does NOT answer
-------------------------
Everything here is threshold-dependent: counts and areas tally pixels or
polygons on one side of a fixed cut, so a bias that shifts pixels across that
cut moves them by construction. They duly move, and on two flights the count
does not even keep its sign.

The paper compares Sigma_anomaly across sites, and what decides whether its
conclusions hold is whether the ORDERING of sites survives, not whether
absolute values shift. Use compare_sigma_anomaly.py for that. It reports
Spearman rho of 0.9968 on flight 4 and 0.9981 on flight 11, so the comparative
claims are robust even where the numbers here are not.

Usage
-----
    python scripts/diagnostics/compare_sensitivity_arms.py \\
        --baseline sgd_output/sensitivity/f4_baseline \\
        --arm m024:sgd_output/sensitivity/f4_m024 \\
        --arm m048:sgd_output/sensitivity/f4_m048 \\
        --arm p024:sgd_output/sensitivity/f4_p024 \\
        --match-distance-m 15
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_polys(base: str) -> list[dict]:
    """Read an arm's merged polygons as centroid + area records."""
    p = Path(base).with_suffix(".geojson")
    if not p.exists():
        raise SystemExit(f"missing {p} (did that arm finish?)")
    fc = json.loads(p.read_text())
    out = []
    for f in fc.get("features", []):
        geom = f.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        clat = sum(lats) / len(lats)
        clon = sum(lons) / len(lons)
        props = f.get("properties", {}) or {}
        area = props.get("area_m2")
        if area is None:
            # Shoelace on a local equirectangular projection.
            mpd_lat = 111320.0
            mpd_lon = 111320.0 * math.cos(math.radians(clat))
            xs = [(lo - clon) * mpd_lon for lo in lons]
            ys = [(la - clat) * mpd_lat for la in lats]
            s = 0.0
            for i in range(len(xs) - 1):
                s += xs[i] * ys[i + 1] - xs[i + 1] * ys[i]
            area = abs(s) / 2.0
        out.append({"lat": clat, "lon": clon, "area_m2": float(area)})
    return out


def match(a: list[dict], b: list[dict], max_m: float):
    """Greedy nearest-neighbour matching between two polygon sets."""
    if not a or not b:
        return [], list(range(len(a))), list(range(len(b)))
    mpd_lat = 111320.0
    pairs = []
    for i, pa in enumerate(a):
        mpd_lon = 111320.0 * math.cos(math.radians(pa["lat"]))
        for j, pb in enumerate(b):
            dx = (pb["lon"] - pa["lon"]) * mpd_lon
            dy = (pb["lat"] - pa["lat"]) * mpd_lat
            d = math.hypot(dx, dy)
            if d <= max_m:
                pairs.append((d, i, j))
    pairs.sort()
    used_a, used_b, matched = set(), set(), []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j); matched.append((i, j, d))
    lost = [i for i in range(len(a)) if i not in used_a]
    gained = [j for j in range(len(b)) if j not in used_b]
    return matched, lost, gained


def main():
    ap = argparse.ArgumentParser(
        description="Compare sensitivity arms against the uncorrected baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--baseline", required=True, help="Output base path of the baseline arm")
    ap.add_argument("--arm", action="append", default=[],
                    help="name:output_base, repeatable")
    ap.add_argument("--match-distance-m", type=float, default=15.0,
                    help="Centroid separation under which two polygons are the same site")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    base = load_polys(args.baseline)
    base_area = sum(p["area_m2"] for p in base)
    print(f"baseline: {len(base)} polygons, {base_area:,.0f} m² total")

    rows = []
    for spec in args.arm:
        if ":" not in spec:
            raise SystemExit(f"--arm expects name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        arm = load_polys(path)
        arm_area = sum(p["area_m2"] for p in arm)
        matched, lost, gained = match(base, arm, args.match_distance_m)

        ret_base = sum(base[i]["area_m2"] for i, _, _ in matched)
        ret_arm = sum(arm[j]["area_m2"] for _, j, _ in matched)
        d_count = 100.0 * (len(arm) - len(base)) / max(len(base), 1)
        d_area = 100.0 * (arm_area - base_area) / max(base_area, 1e-9)
        d_ret = 100.0 * (ret_arm - ret_base) / max(ret_base, 1e-9)

        rows.append({
            "arm": name, "n": len(arm), "area_m2": arm_area,
            "matched": len(matched), "lost": len(lost), "gained": len(gained),
            "count_change_pct": d_count, "area_change_pct": d_area,
            "retained_area_change_pct": d_ret,
            "retention_pct": 100.0 * len(matched) / max(len(base), 1),
        })

    print(f"\n{'arm':<8} {'n':>5} {'Δn%':>7} {'Δarea%':>8} {'kept':>5} {'lost':>5} "
          f"{'gain':>5} {'keep%':>6} {'Δarea(kept)%':>13}")
    for r in rows:
        print(f"{r['arm']:<8} {r['n']:>5} {r['count_change_pct']:>+7.1f} "
              f"{r['area_change_pct']:>+8.1f} {r['matched']:>5} {r['lost']:>5} "
              f"{r['gained']:>5} {r['retention_pct']:>6.1f} "
              f"{r['retained_area_change_pct']:>+13.1f}")

    worst = max((abs(r["count_change_pct"]) for r in rows), default=0.0)
    worst_area = max((abs(r["area_change_pct"]) for r in rows), default=0.0)
    worst_keep = min((r["retention_pct"] for r in rows), default=100.0)
    print()
    print(f"  Polygon-level response: up to {worst:.1f}% count change, "
          f"{worst_area:.1f}% area change, {worst_keep:.0f}% of baseline sites "
          f"matched within {args.match_distance_m:g} m.")
    print()
    print("  These are THRESHOLD-DEPENDENT products and are expected to move: both")
    print("  count pixels or polygons on one side of a fixed cut, so a bias that")
    print("  shifts pixels across that cut changes them directly. Measured on two")
    print("  flights, count is not even stable in sign (+7.8% on flight 4, -11.8%")
    print("  on flight 11), because colder frame centres merge neighbouring")
    print("  detections into fewer, larger polygons.")
    print()
    print("  This script does NOT answer whether the published results are")
    print("  affected. Sigma_anomaly is the metric the paper compares across")
    print("  sites, and what matters there is whether the ORDERING of sites")
    print("  survives, not whether absolute values shift. Run")
    print("  scripts/diagnostics/compare_sigma_anomaly.py for that. On flights 4")
    print("  and 11 it gives Spearman rho of 0.9968 and 0.9981, so comparative")
    print("  claims are robust even though the numbers above are not.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).with_suffix(".json").write_text(json.dumps({
            "baseline_n": len(base), "baseline_area_m2": base_area,
            "match_distance_m": args.match_distance_m, "arms": rows,
        }, indent=2))
        print(f"  wrote {Path(args.output).with_suffix('.json')}")


if __name__ == "__main__":
    main()
