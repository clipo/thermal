#!/usr/bin/env python3
"""Publication-quality close-up of a single SGD site.

Layered visualization on Esri World Imagery satellite basemap:

  Layer 1 (bottom): satellite tiles via contextily (Esri.WorldImagery)
  Layer 2: anomaly raster, semi-transparent, with quality filters:
           - cells with obs_count < --min-obs are masked (default 5)
           - cells with anomaly above --max-realistic are masked
             (default 3.0 °C — anything colder is a thermal-sensor
             outlier, since real SGD is 0.3–1.5 °C below ambient)
  Layer 3: 0.3 °C anomaly contour (red dashed) — the broader plume
           envelope that extends beyond the discrete detector polygons
  Layer 4 (top): detector SGD polygons (black, weight ∝ Σ_anomaly)

Polygon visibility filter: polygon bounding box must overlap the crop bbox
(fixes a bug in the prior version where polygons whose centroid was just
outside the crop got dropped, even though they overlapped it).

Includes locator inset, scale bar, north arrow, and a caption-ready title.

Outputs: sgd_output/figures/closeups/<slug>_<label>_closeup.{png,pdf}

Usage:
    # auto-find the strongest cluster
    python scripts/build_site_closeup.py --slug flight8_hekii_west

    # explicit center + custom box
    python scripts/build_site_closeup.py --slug vaihu_full \\
        --center -27.16830 -109.38510 --box-m 700 --label "Vaihu Harbor"

    # turn off basemap (e.g., offline)
    python scripts/build_site_closeup.py --slug ... --no-basemap
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection


THERMAL = Path(__file__).resolve().parent.parent
SGD_OUTPUT = THERMAL / "sgd_output"


def load_polys(slug: str, source: str = "detector") -> list[dict]:
    """source: 'detector' (default, from per-frame detection pipeline),
              'raster' (from derive_polygons_from_raster.py),
           or 'coastal' (from derive_plumes_coast_anchored.py)"""
    base = SGD_OUTPUT / f"{slug}_spread"
    if source == "raster":
        gj = base / f"{slug}_sgd_raster.geojson"
        if not gj.exists():
            raise SystemExit(
                f"missing raster polygons: {gj}\n"
                f"Run: python scripts/derive_polygons_from_raster.py --slug {slug}"
            )
    elif source == "coastal":
        gj = base / f"{slug}_sgd_coastal.geojson"
        if not gj.exists():
            raise SystemExit(
                f"missing coastal polygons: {gj}\n"
                f"Run: python scripts/derive_plumes_coast_anchored.py --slug {slug}"
            )
    else:
        gj = base / f"{slug}_sgd.geojson"
        if not gj.exists():
            raise SystemExit(f"missing geojson: {gj}")
    fc = json.loads(gj.read_text())
    out = []
    for feat in fc["features"]:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Polygon":
            continue
        ring = geom["coordinates"][0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        out.append({
            "ring": ring,
            "props": feat["properties"],
            "bbox": (min(lons), min(lats), max(lons), max(lats)),
        })
    return out


def auto_center(polys: list[dict]) -> tuple[float, float]:
    best = max(polys, key=lambda p: p["props"].get("sigma_anomaly_m2c", 0.0))
    return float(best["props"]["centroid_lat"]), float(best["props"]["centroid_lon"])


def polys_overlapping(polys: list[dict], minlon: float, minlat: float,
                      maxlon: float, maxlat: float) -> list[dict]:
    out = []
    for p in polys:
        bx_min_lon, bx_min_lat, bx_max_lon, bx_max_lat = p["bbox"]
        if bx_max_lon < minlon or bx_min_lon > maxlon:
            continue
        if bx_max_lat < minlat or bx_min_lat > maxlat:
            continue
        out.append(p)
    return out


def crop_raster(npz, center_lat: float, center_lon: float, box_m: float):
    minlon = float(npz["bbox_min_lon"])
    minlat = float(npz["bbox_min_lat"])
    maxlat = float(npz["bbox_max_lat"])
    grid_res = float(npz["grid_resolution_m"])
    centerlat_full = 0.5 * (minlat + maxlat)
    mpd_lat = 111320.0
    mpd_lon = 111320.0 * math.cos(math.radians(centerlat_full))

    half_lat = (box_m / 2) / mpd_lat
    half_lon = (box_m / 2) / mpd_lon
    crop_minlon = max(minlon, center_lon - half_lon)
    crop_maxlon = min(float(npz["bbox_max_lon"]), center_lon + half_lon)
    crop_minlat = max(minlat, center_lat - half_lat)
    crop_maxlat = min(maxlat, center_lat + half_lat)

    cmin = max(0, int(math.floor((crop_minlon - minlon) * mpd_lon / grid_res)))
    cmax = min(npz["anomaly"].shape[1],
               int(math.ceil((crop_maxlon - minlon) * mpd_lon / grid_res)))
    rmin = max(0, int(math.floor((crop_minlat - minlat) * mpd_lat / grid_res)))
    rmax = min(npz["anomaly"].shape[0],
               int(math.ceil((crop_maxlat - minlat) * mpd_lat / grid_res)))

    return {
        "anom": npz["anomaly"][rmin:rmax, cmin:cmax],
        "obs": npz["observations"][rmin:rmax, cmin:cmax],
        "minlon": minlon + cmin * grid_res / mpd_lon,
        "minlat": minlat + rmin * grid_res / mpd_lat,
        "maxlon": minlon + cmax * grid_res / mpd_lon,
        "maxlat": minlat + rmax * grid_res / mpd_lat,
        "grid_res": grid_res,
    }


def add_basemap(ax, source_name: str = "Esri.WorldImagery"):
    """Add a satellite basemap via contextily. Returns True on success."""
    try:
        import contextily as ctx
    except ImportError:
        print("  warn: contextily not installed; skipping basemap")
        return False
    providers = {
        "Esri.WorldImagery": ctx.providers.Esri.WorldImagery,
        "OpenStreetMap": ctx.providers.OpenStreetMap.Mapnik,
    }
    src = providers.get(source_name, ctx.providers.Esri.WorldImagery)
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", source=src,
                        attribution_size=7, zorder=1)
        return True
    except Exception as e:
        print(f"  warn: basemap fetch failed: {e}")
        return False


def add_scale_bar_geo(ax, length_m: float, center_lat: float):
    """Scale bar in lon/lat space. Length corresponds to length_m at center_lat."""
    mpd_lon = 111320.0 * math.cos(math.radians(center_lat))
    length_lon = length_m / mpd_lon
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_anchor = xlim[1] - 0.05 * (xlim[1] - xlim[0])
    y_anchor = ylim[0] + 0.06 * (ylim[1] - ylim[0])
    ax.plot([x_anchor - length_lon, x_anchor], [y_anchor, y_anchor],
            color="white", linewidth=4, solid_capstyle="butt", zorder=10)
    ax.plot([x_anchor - length_lon, x_anchor], [y_anchor, y_anchor],
            color="black", linewidth=2.2, solid_capstyle="butt", zorder=11)
    ax.text(x_anchor - length_lon / 2,
            y_anchor + 0.013 * (ylim[1] - ylim[0]),
            f"{int(length_m)} m", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white",
                      ec="none", alpha=0.9), zorder=12)


def add_north_arrow(ax):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = xlim[0] + 0.04 * (xlim[1] - xlim[0])
    y = ylim[1] - 0.08 * (ylim[1] - ylim[0])
    arrow_len = 0.06 * (ylim[1] - ylim[0])
    ax.annotate("", xy=(x, y), xytext=(x, y - arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2,
                                shrinkA=0, shrinkB=0), zorder=11)
    ax.text(x, y + 0.005 * (ylim[1] - ylim[0]), "N",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="none", alpha=0.85), zorder=11)


def add_locator_inset(fig, center_lon: float, center_lat: float):
    inset = fig.add_axes([0.745, 0.74, 0.18, 0.18])
    all_lats, all_lons = [], []
    for spread_dir in SGD_OUTPUT.glob("*_spread"):
        slug = spread_dir.name[: -len("_spread")]
        gj = spread_dir / f"{slug}_sgd.geojson"
        if not gj.exists():
            continue
        try:
            fc = json.loads(gj.read_text())
        except Exception:
            continue
        for feat in fc["features"]:
            p = feat.get("properties", {})
            if "centroid_lat" in p:
                all_lats.append(float(p["centroid_lat"]))
                all_lons.append(float(p["centroid_lon"]))
    inset.scatter(all_lons, all_lats, s=0.6, color="#aac", alpha=0.65)
    inset.scatter([center_lon], [center_lat], s=110, color="red",
                  marker="*", edgecolor="black", linewidth=0.6, zorder=5)
    inset.set_xticks([]); inset.set_yticks([])
    centerlat = float(np.mean(all_lats)) if all_lats else center_lat
    inset.set_aspect(1.0 / math.cos(math.radians(centerlat)))
    inset.set_title("Rapa Nui", fontsize=8, pad=2)
    for spine in inset.spines.values():
        spine.set_edgecolor("#666")
        spine.set_linewidth(0.6)


def load_water_mask(slug: str) -> np.ndarray | None:
    """Return the satellite-derived water mask aligned with the anomaly grid,
    or None if not built yet (run scripts/derive_water_mask.py first)."""
    p = SGD_OUTPUT / f"{slug}_spread" / f"{slug}_water_mask.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["is_water"]


def render(slug: str, center_lat: float, center_lon: float, box_m: float,
           output_dir: Path, *,
           vmax_override: float | None = None,
           contour_level: float = 0.3,
           min_obs: int = 5,
           max_realistic_anom_c: float = 3.0,
           use_basemap: bool = True,
           use_water_mask: bool = True,
           polygon_source: str = "detector",
           site_label: str | None = None):
    npz = np.load(SGD_OUTPUT / f"{slug}_spread" / f"{slug}_anomaly.npz")
    polys = load_polys(slug, source=polygon_source)
    crop = crop_raster(npz, center_lat, center_lon, box_m)

    # Quality-filter the raster: drop low-coverage cells AND outlier anomalies
    raw = crop["anom"]
    obs = crop["obs"]
    quality = (
        np.isfinite(raw)
        & (obs >= min_obs)
        & (raw <= max_realistic_anom_c)
    )

    # Apply satellite-derived water mask if available (kills land-projected cells)
    n_dropped_landmask = 0
    water_mask_used = False
    if use_water_mask:
        full_water = load_water_mask(slug)
        if full_water is not None:
            # Crop the water mask to match the raster crop
            grid_res = crop["grid_res"]
            mpd_lat = 111320.0
            centerlat_full = 0.5 * (float(npz["bbox_min_lat"]) + float(npz["bbox_max_lat"]))
            mpd_lon = 111320.0 * math.cos(math.radians(centerlat_full))
            cmin = int(round((crop["minlon"] - float(npz["bbox_min_lon"]))
                             * mpd_lon / grid_res))
            rmin = int(round((crop["minlat"] - float(npz["bbox_min_lat"]))
                             * mpd_lat / grid_res))
            wm = full_water[rmin:rmin + raw.shape[0], cmin:cmin + raw.shape[1]]
            if wm.shape == raw.shape:
                pre = quality.sum()
                quality = quality & wm
                n_dropped_landmask = int(pre - quality.sum())
                water_mask_used = True

    display = np.where(quality, raw, np.nan)
    n_dropped_obs = int(((obs < min_obs) & np.isfinite(raw)).sum())
    n_dropped_outlier = int((np.isfinite(raw) & (raw > max_realistic_anom_c)).sum())

    # Auto vmax based on filtered data, clamped to publication-friendly range
    if vmax_override is not None:
        vmax = vmax_override
    elif np.any(quality):
        vmax = float(np.nanpercentile(display, 99))
        vmax = float(np.clip(vmax, 0.4, 2.0))
    else:
        vmax = 1.0

    # Filter polygons by bbox overlap with the crop (fixes the centroid-only bug)
    visible_polys = polys_overlapping(
        polys, crop["minlon"], crop["minlat"], crop["maxlon"], crop["maxlat"],
    )

    # If a water mask is available, drop polygons that are mostly on land
    # (these are projection-bug artifacts: detector polygons formed from
    # cliff-shadow pixels misprojected to ocean coordinates).
    # Skip this filter for raster polygons — they're derived FROM the
    # water mask and are guaranteed to be over water by construction.
    n_polys_dropped_land = 0
    if (water_mask_used and polygon_source not in ("raster", "coastal")
            and len(visible_polys) > 0):
        full_water = load_water_mask(slug)
        if full_water is not None:
            try:
                from matplotlib.path import Path as MplPath
            except ImportError:
                MplPath = None
            if MplPath is not None:
                grid_res = float(npz["grid_resolution_m"])
                rmin_full = 0
                cmin_full = 0
                # Use full-raster lat/lon → cell index conversion
                full_minlon = float(npz["bbox_min_lon"])
                full_minlat = float(npz["bbox_min_lat"])
                full_maxlat = float(npz["bbox_max_lat"])
                centerlat_full = 0.5 * (full_minlat + full_maxlat)
                mpd_lat_f = 111320.0
                mpd_lon_f = 111320.0 * math.cos(math.radians(centerlat_full))

                kept = []
                for p in visible_polys:
                    ring = p["ring"]
                    lons = np.array([c[0] for c in ring], dtype=np.float64)
                    lats = np.array([c[1] for c in ring], dtype=np.float64)
                    cs = (lons - full_minlon) * mpd_lon_f / grid_res
                    rs = (lats - full_minlat) * mpd_lat_f / grid_res
                    cmin = max(0, int(math.floor(cs.min())))
                    cmax = min(full_water.shape[1], int(math.ceil(cs.max())) + 1)
                    rmin = max(0, int(math.floor(rs.min())))
                    rmax = min(full_water.shape[0], int(math.ceil(rs.max())) + 1)
                    if cmax <= cmin or rmax <= rmin:
                        kept.append(p); continue
                    path = MplPath(np.column_stack([cs, rs]))
                    cc, rr = np.meshgrid(np.arange(cmin, cmax) + 0.5,
                                         np.arange(rmin, rmax) + 0.5)
                    inside = path.contains_points(
                        np.column_stack([cc.ravel(), rr.ravel()])
                    ).reshape(rr.shape)
                    if not inside.any():
                        kept.append(p); continue
                    sub_water = full_water[rmin:rmax, cmin:cmax]
                    n_inside = int(inside.sum())
                    n_water_inside = int((inside & sub_water).sum())
                    water_frac = n_water_inside / n_inside if n_inside else 0.0
                    # Drop only polygons that are FULLY over land (water_frac=0)
                    # Polygons with partial water coverage are kept; the
                    # underlying Σ_anomaly already reflects only water cells.
                    # The 0.5 threshold previously used dropped legitimate
                    # SGD polygons at sites like Vaihu Harbor (shallow water
                    # near shore that the satellite HSV classifier doesn't
                    # reliably detect).
                    if water_frac <= 0.05:
                        n_polys_dropped_land += 1
                    else:
                        kept.append(p)
                visible_polys = kept

    visible_polys.sort(key=lambda p: p["props"].get("sigma_anomaly_m2c", 0.0))

    # === Restrict raster display to inside polygons only ===
    # The integrated cold-anomaly raster covers the entire flight strip,
    # but for paper figures we want the colored shading to represent
    # exactly what's inside each detected plume — not extend across
    # the bay or onto adjacent inlets. Display only cells inside polygon
    # rings (zero buffer).
    if visible_polys:
        from matplotlib.path import Path as MplPath
        crop_gy, crop_gx = display.shape
        grid_res = crop["grid_res"]
        poly_mask = np.zeros((crop_gy, crop_gx), dtype=bool)
        mpd_lat_loc = 111320.0
        _crop_centerlat = 0.5 * (crop["minlat"] + crop["maxlat"])
        mpd_lon_loc = 111320.0 * math.cos(math.radians(_crop_centerlat))
        for p in visible_polys:
            ring = p["ring"]
            lons_r = np.array([c[0] for c in ring], dtype=np.float64)
            lats_r = np.array([c[1] for c in ring], dtype=np.float64)
            cs_r = (lons_r - crop["minlon"]) * mpd_lon_loc / grid_res
            rs_r = (lats_r - crop["minlat"]) * mpd_lat_loc / grid_res
            pts_r = np.column_stack([cs_r, rs_r])
            cmin_r = max(0, int(math.floor(cs_r.min())))
            cmax_r = min(crop_gx, int(math.ceil(cs_r.max())) + 1)
            rmin_r = max(0, int(math.floor(rs_r.min())))
            rmax_r = min(crop_gy, int(math.ceil(rs_r.max())) + 1)
            if cmax_r <= cmin_r or rmax_r <= rmin_r:
                continue
            path_r = MplPath(pts_r)
            cc_r, rr_r = np.meshgrid(np.arange(cmin_r, cmax_r) + 0.5,
                                      np.arange(rmin_r, rmax_r) + 0.5)
            inside_r = path_r.contains_points(
                np.column_stack([cc_r.ravel(), rr_r.ravel()])
            ).reshape(rr_r.shape)
            poly_mask[rmin_r:rmax_r, cmin_r:cmax_r] |= inside_r
        display = np.where(poly_mask, display, np.nan)

    # ---------------- figure ----------------
    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=False)

    extent = (crop["minlon"], crop["maxlon"], crop["minlat"], crop["maxlat"])
    ax.set_xlim(crop["minlon"], crop["maxlon"])
    ax.set_ylim(crop["minlat"], crop["maxlat"])
    centerlat_for_aspect = 0.5 * (crop["minlat"] + crop["maxlat"])
    ax.set_aspect(1.0 / math.cos(math.radians(centerlat_for_aspect)))

    # 1) basemap
    have_basemap = False
    if use_basemap:
        have_basemap = add_basemap(ax)

    # 2) anomaly raster overlay
    im = ax.imshow(
        display, origin="lower", cmap="YlOrRd",
        vmin=0, vmax=vmax, interpolation="bilinear",
        extent=extent, alpha=0.72, zorder=2,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02,
                      label="Cold anomaly  (°C below ambient baseline)")

    # 3) 0.3 °C contour outlining the plume envelope (only if requested)
    if contour_level > 0 and np.any(quality):
        gy, gx = display.shape
        xs = np.linspace(crop["minlon"], crop["maxlon"], gx)
        ys = np.linspace(crop["minlat"], crop["maxlat"], gy)
        cs = ax.contour(xs, ys, np.nan_to_num(display, nan=0.0),
                         levels=[contour_level],
                         colors="#cc0033", linewidths=1.0, linestyles="--",
                         zorder=3)

    # 4) detector polygons (black, weight ∝ sigma)
    if visible_polys:
        sigmas = [p["props"].get("sigma_anomaly_m2c", 0.0) for p in visible_polys]
        smax = max(sigmas) or 1.0
        patches = [MplPoly(p["ring"], closed=True) for p in visible_polys]
        widths = [0.7 + 1.6 * (s / smax) for s in sigmas]
        pc = PatchCollection(patches, facecolor="none",
                             edgecolor="black", linewidths=widths, zorder=4)
        ax.add_collection(pc)

    add_scale_bar_geo(ax, length_m=100, center_lat=centerlat_for_aspect)
    add_north_arrow(ax)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.tick_params(labelsize=8)
    # Force fixed-decimal axis labels (no offset like -1.09e2)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.4f}"))

    # Caption
    name = site_label or slug
    total_sigma = sum(p["props"].get("sigma_anomaly_m2c", 0.0) for p in visible_polys)
    total_area = sum(p["props"].get("area_m2", 0.0) for p in visible_polys)
    fig.suptitle(
        f"{name}  —  {len(visible_polys)} SGD polygons in {int(box_m)} m × {int(box_m)} m frame  "
        f"(Σ_anomaly = {total_sigma:,.0f} m²·°C, area {total_area:,.0f} m²)",
        fontsize=12, y=0.985,
    )
    subtitle = (
        f"flight: {slug}  ·  "
        f"baseline: {float(npz['baseline_median_c']):.2f} °C  ·  "
        f"{int(npz['n_frames_used'])} frames"
    )
    if n_dropped_obs or n_dropped_outlier or n_dropped_landmask:
        parts = [f"obs≥{min_obs}: -{n_dropped_obs:,}",
                 f"anomaly≤{max_realistic_anom_c}°C: -{n_dropped_outlier:,}"]
        if water_mask_used:
            parts.append(f"land-mask: -{n_dropped_landmask:,}")
        subtitle += "  ·  raster filtered: " + " ".join(parts)
    if n_polys_dropped_land > 0:
        subtitle += f"  ·  polygons dropped (>50% on land): {n_polys_dropped_land}"
    poly_label = {
        "raster": "raster-derived SGD polygons",
        "coastal": "coast-anchored SGD plumes",
        "detector": "detector SGD polygons",
    }.get(polygon_source, "SGD polygons")
    legend_line = (
        f"black: {poly_label}  ·  red dashed: {contour_level}°C anomaly contour  ·  "
        f"shaded: integrated cold anomaly  ·  basemap: "
        + ("Esri World Imagery" if have_basemap else "(no basemap)")
    )
    fig.text(0.5, 0.948, subtitle, ha="center", fontsize=9, color="#444")
    fig.text(0.5, 0.928, legend_line, ha="center", fontsize=8, color="#444",
             style="italic")

    add_locator_inset(fig, center_lon, center_lat)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe = (site_label or slug).replace("/", "_").replace(" ", "_")
    out_png = output_dir / f"{slug}_{safe}_closeup.png"
    out_pdf = output_dir / f"{slug}_{safe}_closeup.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {out_png}")
    return {
        "png": str(out_png),
        "n_polys": len(visible_polys),
        "total_sigma": total_sigma,
        "n_dropped_obs": n_dropped_obs,
        "n_dropped_outlier": n_dropped_outlier,
        "vmax_used": vmax,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--box-m", type=float, default=600.0)
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--contour-level", type=float, default=0.3,
                    help="°C anomaly contour overlaid on raster (default 0.3)")
    ap.add_argument("--min-obs", type=int, default=5,
                    help="mask raster cells with fewer observations (default 5)")
    ap.add_argument("--max-realistic", type=float, default=3.0,
                    help="mask raster cells with anomaly above this (sensor outliers; default 3.0)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-basemap", action="store_true")
    ap.add_argument("--no-water-mask", action="store_true",
                    help="don't apply satellite-derived water mask "
                         "(useful for debugging or if mask not built yet)")
    ap.add_argument("--polygon-source", choices=("detector", "raster", "coastal"),
                    default="detector",
                    help="which polygon set to draw. 'raster' = watershed-"
                         "from-anywhere; 'coastal' = coast-anchored "
                         "watershed (each polygon has a documented coastal "
                         "source); 'detector' = original per-frame detection.")
    ap.add_argument("--output-dir",
                    default=str(SGD_OUTPUT / "figures" / "closeups"))
    args = ap.parse_args()

    polys = load_polys(args.slug, source=args.polygon_source)
    if args.center:
        lat, lon = args.center
    else:
        lat, lon = auto_center(polys)
    render(args.slug, lat, lon, args.box_m, Path(args.output_dir),
           vmax_override=args.vmax, contour_level=args.contour_level,
           min_obs=args.min_obs, max_realistic_anom_c=args.max_realistic,
           use_basemap=not args.no_basemap,
           use_water_mask=not args.no_water_mask,
           polygon_source=args.polygon_source,
           site_label=args.label)


if __name__ == "__main__":
    main()
