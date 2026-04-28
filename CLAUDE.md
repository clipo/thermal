# CLAUDE.md — context for future Claude sessions on this repo

This file is loaded automatically by Claude Code at the start of each
session. It tells Claude where things are, what the user is working
on, and what NOT to break.

## Project at a glance

Quantitative SGD (submarine groundwater discharge) detection from
thermal-drone surveys of Rapa Nui (Easter Island). The user is
preparing an academic paper that correlates SGD locations with
archaeology features (ahu, moai). The detection pipeline produces
per-flight cold-anomaly rasters and three parallel polygon products
(detector / raster watershed / coast-anchored).

## What the user cares about most

- **Coast-anchored plumes** are the canonical product for the paper.
  Every polygon has a documented coastal source (`source_lat` /
  `source_lon`). File: `sgd_output/rapa_nui_all_sgd_coastal.kml`.
- **Σ_anomaly_m2c** (m² · °C) is the threshold-independent metric.
  This is what the paper compares across sites.
- **Geological consistency**: Rapa Nui SGD emerges through collapsed
  lava-tube outlets at low-elevation bays. Cliff coasts (Poike, Rano
  Kau, parts of Maunga Terevaka) should NOT have SGD because they
  lack conduit topology. The pipeline filters cliff zones via SRTM DEM.

## What's done and validated (2026-04-28)

- **OSM coastline mask** is the production water/land boundary (replaced
  HSV satellite classifier — user said "the OSM map is a huge help").
- **SRTM 30 m DEM cliff-zone mask** wired into the coast-anchored
  detector. Default thresholds: max-elev > 80 m within 100 m radius.
- **Adaptive thresholds**: peak = clip(p95-of-water, 0.35, 0.55) °C,
  edge = clip(max(p70, 0.5×peak), 0.20, 0.35) °C. Per-flight.
- **Three master products** in `sgd_output/`:
  - `rapa_nui_all_sgd_coastal.kml` — 415 plumes (canonical)
  - `rapa_nui_all_sgd_sigma_raster.kml` — 1,066 plumes (full halo)
  - `rapa_nui_all_sgd_sigma.kml` — 1,789 polygons (detector)
- **Paper Methods .docx**: `docs/PAPER_METHODS.docx` at project root. Has 7
  embedded figures (overview + 6 site closeups). Source is
  `docs/PAPER_METHODS.md`. Compiled by `scripts/figures/build_methods_docx.py`.
- **Reproducibility**: see `docs/REPRODUCE.md` at project root for the full
  end-to-end re-run sequence and where everything lives.

## Directory structure

Scripts are organised into subdirectories under `scripts/`:

```
scripts/
├── README.md           categorised script index
├── pipeline/           core SGD detection (run in order)
├── aggregate/          cross-flight master KMLs
├── figures/            publication figures
├── coverage/           flight footprint maps
├── tools/              downstream analysis (proximity, coastline sampling)
├── diagnostics/        QA / debug
├── alternative_water_masks/   legacy / exploratory water-mask methods
├── batch/              shell drivers for batch runs
├── legacy/             older scripts (still referenced by docs)
└── (root)              user-facing tools (sgd_wizard, sgd_autodetect)
```

When adding new scripts, place them in the appropriate subdirectory
(don't dump in `scripts/` root). When invoking, use the full path:
`python scripts/pipeline/X.py`.

## Critical things NOT to break

1. **Coordinate accuracy.** Vaihu (Ahu Vaihu) is at (-27.167,
   -109.365), NOT (-27.168, -109.385) which I had wrong earlier in
   one session. Other key sites:
   - Hanga Nui at Ahu Tongariki: (-27.1275, -109.2775)
   - Anakena Bay: (-27.0712, -109.318)
2. **The OSM water mask** in `<slug>_water_mask.npz` is downstream of
   `derive_water_mask_osm.py`. The HSV-based `derive_water_mask.py`
   and the drone-extension hack are LEGACY (`scripts/alternative_water_masks/`).
   Don't accidentally re-run those and overwrite the OSM masks.
3. **Cliff-zone exclusion** is part of the canonical pipeline. The
   user explicitly wants cliff coasts excluded as SGD-implausible.
4. **Adaptive threshold ceiling at 0.55 °C** prevents strong-signal
   flights from setting thresholds so high that subtle bays in the
   same flight (Anakena, parts of Vaihu) are filtered out.
5. **The paper Methods doc** (`docs/PAPER_METHODS.docx`) regenerates from
   `docs/PAPER_METHODS.md` and the closeup PNGs. To update text or figures,
   edit those, then run `python scripts/figures/build_methods_docx.py`.

## What's still to address (in priority order)

1. Validate cliff-zone identification visually (task #54).
2. Phase 2 (deferred until external volume is mounted): SAM2 per-frame
   ocean segmentation + p75 → p90 baseline. Both require rebuilding
   all 30,000+ frames' anomaly rasters. ~1–2 hr GPU compute.
3. Phase 3 (long-term): DEM-aware footprint projection — the proper
   fix for the cliff-shadow projection bug. Major code change in
   `sgd_toolkit/georeferencing/footprint_generator.py`.

## Key reference files

| File | Purpose |
|---|---|
| `docs/METHODS.md` | Living methods doc + decisions log (append-only) |
| `docs/PAPER_METHODS.md` | Academic Methods section source |
| `docs/PAPER_METHODS.docx` | Compiled .docx paper deliverable |
| `docs/REPRODUCE.md` | How to validate and re-run everything |
| `scripts/README.md` | Categorised script index |
| `data/dem/S28W110.hgt` | NASA SRTM 30 m DEM tile (Rapa Nui) |
| `sgd_output/osm_coastline.json` | Cached OSM Rapa Nui coastline |

## When the user asks to re-run

Open `docs/REPRODUCE.md`. The end-to-end recipe is documented step-by-step.
For a clean re-run from raw frames you also need the external volume
mounted at `/Volumes/RapaNui`.

## Conventions

- Scripts use `THERMAL = Path(__file__).resolve().parent.parent.parent`
  (three levels up — script in `scripts/<subdir>/X.py`).
- All Python scripts have argparse + `--help` and don't require
  positional arguments unless explicitly meant for one site.
- Per-flight outputs live under `sgd_output/<slug>_spread/`.
- Master / cross-flight outputs live at `sgd_output/` root.
- Figures live under `sgd_output/figures/`.
