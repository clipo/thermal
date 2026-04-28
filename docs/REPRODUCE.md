# Reproducing the quantitative SGD pipeline

This document tells you exactly where everything lives and how to
re-run the analysis end-to-end. The currently-good outputs were
produced on **2026-04-28** with the OSM coastline + SRTM cliff-zone
+ coast-anchored detector pipeline. If you want to validate or
regenerate, follow the steps below.

---

## Where everything lives

```
thermal/
├── docs/METHODS.md              ← Living methods doc (current pipeline + decisions log)
├── docs/PAPER_METHODS.md        ← Academic-paper-ready Methods section (source)
├── docs/PAPER_METHODS.docx      ← ★ Compiled Methods .docx with 7 figures (paper deliverable)
├── docs/REPRODUCE.md            ← (this file)
├── README.md               ← Toolkit user guide (existing detector pipeline)
│
├── data/                   ← (gitignored) symlinks to raw drone frames
│   └── <flight_slug>_combined/   per-flight MAX_*.JPG + IRX_*.irg pairs
│   └── dem/S28W110.hgt           NASA SRTM 30m DEM tile (Rapa Nui)
│
├── sgd_toolkit/            ← The detection library code
│
├── scripts/                ← Pipeline + tools (organised by purpose)
│   ├── README.md           Categorised script index
│   ├── pipeline/           Core SGD detection (run in order; see below)
│   ├── aggregate/          Cross-flight master KML + comparison CSVs
│   ├── figures/            Publication figures
│   ├── coverage/           Flight footprint / extent maps
│   ├── tools/              Downstream analysis (sgd_proximity, sample_coastline)
│   ├── diagnostics/        QA + debug
│   ├── alternative_water_masks/   Legacy/exploratory water mask methods
│   ├── batch/              Shell drivers for batch runs
│   ├── legacy/             Older scripts kept for reference
│   └── (root)              User-facing tools (sgd_wizard, sgd_autodetect, etc.)
│
└── sgd_output/             ← (gitignored) all generated outputs
    ├── osm_coastline.json                Cached OSM Rapa Nui coastline (one-time fetch)
    ├── <slug>_spread/                    per-flight outputs:
    │   ├── <slug>_anomaly.{npz,png,kml}      cold-anomaly raster + GroundOverlay
    │   ├── <slug>_water_mask.npz             OSM-derived land/water mask
    │   ├── <slug>_cliff_zone.npz             SRTM-derived cliff-zone mask
    │   ├── <slug>_sgd.geojson                detector polygons (run_coast_stretch)
    │   ├── <slug>_sgd_raster.geojson         raster watershed polygons
    │   ├── <slug>_sgd_coastal.geojson        ★ coast-anchored plumes (canonical)
    │   ├── <slug>_sgd_intensity.csv          per-polygon Σ_anomaly table
    │   └── <slug>_validation.png             3-panel diagnostic
    │
    ├── rapa_nui_all_sgd_coastal.kml          ★ MASTER KML — 415 plumes / 26 flights
    ├── rapa_nui_all_sgd_sigma_raster.kml     master KML — raster polygons (full halo)
    ├── rapa_nui_all_sgd_sigma.kml            master KML — detector polygons
    ├── rapa_nui_all_anomaly.kml              GroundOverlay master across all flights
    ├── polygon_intensity_summary.csv         per-flight Σ_anomaly table
    ├── polygon_comparison_summary.csv        detector vs raster comparison
    │
    └── figures/
        ├── rapa_nui_overview_coastal.{png,pdf}    ★ Figure 1 of paper
        ├── rapa_nui_overview_raster.{png,pdf}     raster polygons overview
        ├── rapa_nui_overview.{png,pdf}            detector polygons overview
        ├── rapa_nui_flight_ranking_raster.png     per-flight bar chart
        ├── rapa_nui_flight_ranking.png            per-flight bar chart (detector)
        ├── rapa_nui_all_anomaly_mosaic.{png,pdf}  whole-island anomaly raster
        ├── polygon_comparison.png                 detector vs raster comparison
        └── closeups/<slug>_<label>_closeup.{png,pdf}   site closeups (paper Figs 2–7)
```

---

## How to validate the current outputs

Just open these in Google Earth or a browser:

| Check | Open |
|---|---|
| Master plume map (canonical product) | `sgd_output/rapa_nui_all_sgd_coastal.kml` |
| Whole-island anomaly raster overlay | `sgd_output/rapa_nui_all_anomaly.kml` |
| Per-flight diagnostic | `sgd_output/<slug>_spread/<slug>_validation.png` |
| Per-site hero figure | `sgd_output/figures/closeups/*_closeup.png` |
| Compiled paper Methods | `docs/PAPER_METHODS.docx` |
| Per-flight Σ_anomaly table | `sgd_output/polygon_intensity_summary.csv` |

The KMLs all reference image PNGs by relative path, so they only
work if you keep the directory structure intact.

---

## How to re-run end-to-end

If you change anything in the pipeline (e.g., switch to SAM2 per-frame
segmentation in Phase 2), run these in order from the project root.
Most steps are idempotent (they overwrite the per-flight files).

```bash
cd /path/to/thermal

# 1. (Only if anomaly rasters need rebuilding — needs raw drone frames)
bash scripts/pipeline/build_all_anomaly_rasters.sh

# 2. OSM coastline → per-flight water mask
python scripts/pipeline/derive_water_mask_osm.py --all

# 3. SRTM DEM → per-flight cliff-zone mask
#    Requires data/dem/S28W110.hgt (download once via:
#    curl -sSL -o data/dem/S28W110.SRTMGL1.hgt.zip \
#      https://step.esa.int/auxdata/dem/SRTMGL1/S28W110.SRTMGL1.hgt.zip
#    && unzip -p data/dem/S28W110.SRTMGL1.hgt.zip > data/dem/S28W110.hgt)
python scripts/pipeline/derive_cliff_zone.py --all

# 4. Recompute polygon Σ_anomaly using the corrected raster + masks
python scripts/pipeline/recompute_polygon_intensity.py --all

# 5. Watershed-from-anywhere raster polygons
python scripts/pipeline/derive_polygons_from_raster.py --all

# 6. Coast-anchored plumes (the canonical product)
python scripts/pipeline/derive_plumes_coast_anchored.py --all

# 7. Re-render anomaly PNGs (for the master GroundOverlay KML)
python scripts/pipeline/mask_anomaly_pngs.py --all

# 8. Aggregate master KMLs
python scripts/aggregate/aggregate_sigma_anomaly_kml.py \
  --output sgd_output/rapa_nui_all_sgd_coastal.kml \
  sgd_output/*_spread/*_sgd_coastal.geojson

python scripts/aggregate/aggregate_sigma_anomaly_kml.py \
  --output sgd_output/rapa_nui_all_sgd_sigma_raster.kml \
  sgd_output/*_spread/*_sgd_raster.geojson

python scripts/aggregate/aggregate_sigma_anomaly_kml.py \
  --output sgd_output/rapa_nui_all_sgd_sigma.kml \
  sgd_output/*_spread/*_sgd.geojson

python scripts/aggregate/aggregate_anomaly_kml.py
# → sgd_output/rapa_nui_all_anomaly.kml

# 9. Cross-flight comparison summary
python scripts/aggregate/build_polygon_comparison_summary.py

# 10. Refresh figures
python scripts/figures/build_island_overview.py --polygon-source coastal
python scripts/figures/build_island_overview.py --polygon-source raster
python scripts/figures/build_island_overview.py
python scripts/figures/build_flight_ranking.py
python scripts/figures/build_flight_ranking.py --polygon-source raster
python scripts/figures/build_island_anomaly_mosaic.py
python scripts/figures/build_validation_figure.py --all

# 11. Per-site closeups (paper Figs 2–7)
python scripts/figures/build_site_closeup.py --slug vaihu_full \
    --polygon-source coastal --center -27.16703 -109.36471 \
    --box-m 700 --label "Vaihu (Ahu Vaihu)" --contour-level 0
python scripts/figures/build_site_closeup.py \
    --slug june2023_23_june_23_tongariki_flights \
    --polygon-source coastal --center -27.1275 -109.2775 --box-m 700 \
    --label "Hanga Nui (Ahu Tongariki)" --contour-level 0
python scripts/figures/build_site_closeup.py --slug flight8_hekii_west \
    --polygon-source coastal --center -27.0858 -109.2999 --box-m 700 \
    --label "Hekii West" --contour-level 0
python scripts/figures/build_site_closeup.py --slug flight10_anakena_to_west \
    --polygon-source coastal --center -27.0658 -109.3358 --box-m 2200 \
    --label "Anakena Bay" --contour-level 0
python scripts/figures/build_site_closeup.py \
    --slug flight11_hivahiva_to_hangapiko --polygon-source coastal \
    --box-m 700 --label "Hivahiva-Hangapiko" --contour-level 0
python scripts/figures/build_site_closeup.py --slug june2023_2_july_23_poike_3 \
    --polygon-source coastal --box-m 600 \
    --label "Poike (cliff coast control)" --contour-level 0

# 12. Compile paper Methods .docx with embedded figures
python scripts/figures/build_methods_docx.py
```

After step 12, `docs/PAPER_METHODS.docx` at the project root has the
current results embedded.

---

## Headline numbers (current good run, 2026-04-28)

| Product | Polygons | Σ_anomaly grand total | File |
|---|---|---|---|
| Coast-anchored plumes (canonical) | 415 across 26 flights | 442,941 m²·°C | `rapa_nui_all_sgd_coastal.kml` |
| Raster watershed polygons | 1,066 across 29 flights | 8,538,560 m²·°C | `rapa_nui_all_sgd_sigma_raster.kml` |
| Detector polygons (water-mask filtered) | 1,789 across 29 flights | 1,147,187 m²·°C | `rapa_nui_all_sgd_sigma.kml` |

**Top sites by coast-anchored Σ_anomaly:**
- 28-June Rano Kau region: 76 plumes, 143k m²·°C
- 25-June southwest survey: 102 plumes, 102k
- 24-June northeast survey: 106 plumes, 100k
- 1-July Hanga Roa-Rano Kau: 35 plumes, 52k
- 27-June Hanga Roa-Hivahiva: 62 plumes, 53k
- Anakena West: 25 plumes, 42k
- 29-June Vai Takari Ua: 32 plumes, 39k
- Hekii West: 14 plumes, 34k
- Kikirahamea-Hivahiva: 38 plumes, 33k
- **Vaihu Full: 33 plumes, 35k** (textbook ground truth ✓)
- **Hanga Nui (23-June Tongariki): 16 plumes, 23k** (Ahu Tongariki ground truth ✓)
- Poike sites: 1–13 plumes each (geologically minimal — cliff coasts)

---

## Phases that need raw drone frames (deferred)

The external volume (`/Volumes/RapaNui`) was not always mounted. These
are deferred until it is and the time/compute is available:

- **Phase 2 — SAM2 per-frame ocean segmenter + p90 baseline.** Replaces
  the rule-based HSV segmenter inside `build_anomaly_raster.py` and
  raises the per-frame baseline percentile from 75 to 90 (more robust
  to large plumes). Then re-run steps 1–12 above. Needs ~1–2 hours of
  Apple Silicon GPU time for the SAM2 step.

- **Phase 3 — DEM-aware footprint projection.** The proper geometric
  fix for the cliff-shadow projection bug (the OSM water mask + SRTM
  cliff-zone filter are post-hoc workarounds). Needs modification of
  `sgd_toolkit/georeferencing/footprint_generator.py` to ray-march
  pixels against terrain instead of assuming flat ground.

Both are detailed in `docs/METHODS.md` under "What's still to address".
