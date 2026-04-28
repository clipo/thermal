# scripts/

Operational scripts for the SGD detection pipeline. Run from the
project root: `python scripts/<script>.py [args]`.

This index groups scripts by purpose. The actual files are kept
flat in this directory (rather than subdirectories) because shell
scripts and documentation reference them as `scripts/X.py` — moving
files would break those references.

For the full methodology, see [`../docs/METHODS.md`](../docs/METHODS.md).

---

## 1. Core pipeline (run in this order)

End-to-end SGD detection from raw thermal/RGB drone frames.

| Script | Purpose |
|---|---|
| `build_anomaly_raster.py` | Build per-flight cold-anomaly raster from raw paired frames. Per-pixel `max(0, T_baseline - T)` accumulated on a 1m grid. |
| `build_all_anomaly_rasters.sh` | Batch driver — runs `build_anomaly_raster.py` for every flight. |
| `derive_water_mask_osm.py` | **Production water mask.** Fetches Rapa Nui coastline from OpenStreetMap via Overpass API and rasterizes per-flight. Replaces HSV satellite classifier. |
| `derive_cliff_zone.py` | Per-flight cliff-zone mask from SRTM 30m DEM (cells where max nearby elevation > 80m). Excludes geologically infeasible SGD locations. |
| `recompute_polygon_intensity.py` | Integrate the anomaly raster within each polygon to compute `sigma_anomaly_m2c` (threshold-independent metric). Updates GeoJSON in place. |
| `derive_polygons_from_raster.py` | Watershed segmentation of the anomaly raster — one polygon per local cold peak (with adaptive thresholds). |
| `derive_plumes_coast_anchored.py` | **Canonical plume detector.** Coast-anchored watershed: every plume has a documented coastal source per OSM coastline + cliff-zone exclusion. |
| `mask_anomaly_pngs.py` | Re-render per-flight anomaly PNGs with quality + water-mask + isolation filters. |

## 2. Aggregation (cross-flight)

| Script | Purpose |
|---|---|
| `aggregate_sigma_anomaly_kml.py` | Master KML across flights, polygons coloured by `Σ_anomaly_m2c` tier. Default for both detector and raster polygon products. |
| `aggregate_anomaly_kml.py` | Master KML overlaying every flight's anomaly GroundOverlay PNG on satellite imagery. |
| `aggregate_intensity_kml.py` | Legacy: master KML keyed on `intensity_index = area × peak_anomaly`. Replaced by `aggregate_sigma_anomaly_kml.py`. |
| `aggregate_extents_kml.py` | Master KML showing flight footprints. |
| `aggregate_kml.py` | Earlier generic aggregator — superseded. |
| `build_polygon_comparison_summary.py` | Cross-flight CSV + bar chart comparing detector vs raster polygon products. |

## 3. Publication figures

| Script | Purpose |
|---|---|
| `build_island_overview.py` | Island-wide map of polygons (`--polygon-source detector|raster|coastal`). |
| `build_island_anomaly_mosaic.py` | Single-image equivalent of the master anomaly KML — all 29 anomaly rasters on satellite basemap. |
| `build_flight_ranking.py` | Per-flight horizontal bar chart of Σ_anomaly. |
| `build_site_closeup.py` | Hero figure for a single site with satellite basemap + overlays (`--polygon-source detector|raster|coastal`). |
| `build_validation_figure.py` | Per-flight 3-panel diagnostic: anomaly raster, polygon overlay, observation count. |
| `build_methods_docx.py` | Convert `docs/PAPER_METHODS.md` to `docs/PAPER_METHODS.docx` with embedded figures for paper submission. |

## 4. Detector + per-frame processing

| Script | Purpose |
|---|---|
| `run_coast_stretch.py` | Per-frame thermal SGD detector + density-grid clustering → `<slug>_sgd.geojson`. |
| `run_all_flights.sh` | Batch driver for Jan 2024 flights. |
| `run_june2023_flights.sh` | Batch driver for June 2023 flights. |
| `run_frame_coverage.sh` | Compute per-flight frame coverage stats. |
| `run_sgd_improved.sh` | Earlier driver — superseded by per-flight runners above. |
| `sgd_wizard.py` | Interactive setup helper. |
| `sgd_autodetect.py` | Auto-detection of suitable parameters. |
| `sam_segmenter.py` | SAM-based per-frame ocean segmenter (alternative to rule-based HSV). |
| `setup_sam.sh` | Install SAM dependencies. |
| `train_segmentation.py` | Train RF segmentation model. |
| `train_segmentation_supervised.py` | Supervised segmentation training. |
| `auto_label.py` | Auto-labelling helper. |

## 5. Coverage and extents

| Script | Purpose |
|---|---|
| `generate_coverage_map.py` | Per-flight ground-footprint coverage KML. |
| `build_flight_extents.py` | Build per-flight footprint polygons. |
| `build_june2023_extents.py` | Footprints for June 2023 flights. |

## 6. Utilities for downstream analysis

| Script | Purpose |
|---|---|
| `sgd_proximity.py` | Score arbitrary point features (e.g., archaeology sites) by integrated cold-anomaly content within a radius — primary archaeology-correlation tool. |
| `sample_coastline.py` | Sample a coastline polyline at metric intervals (input to `sgd_proximity.py` for per-segment shoreline metrics). |

## 7. Diagnostics

| Script | Purpose |
|---|---|
| `diagnose_water_mask.py` | Render satellite tile + water-mask overlay for visual QA of the OSM/HSV mask. |
| `diagnose_ocean_mask.py` | Render per-frame RGB + thermal + ocean mask for visual QA of the per-frame segmenter. |
| `diagnose_setup.py` | Sanity-check the project setup. |
| `debug_sam2_water_mask.py` | Visualize SAM2-based water mask classification. |
| `analyze_frame_overlap.py` | Frame-to-frame spatial overlap stats. |
| `analyze_thresholds.py` | Sweep detection threshold; report polygon yield. |
| `check_altitude_consistency.py` | Sanity-check drone altitude consistency across a flight. |

## 8. Alternative water-mask methods (not currently in production)

| Script | Status |
|---|---|
| `derive_water_mask.py` | HSV satellite classifier. **Superseded by OSM (1).** Kept for reproducing earlier results. |
| `derive_water_mask_sam2.py` | SAM2-based satellite tile classification. Exploratory; the hybrid SAM2+HSV approach didn't outperform pure HSV reliably. |
| `extend_water_mask_with_drone.py` | Augment a HSV mask using cells the drone observed as ocean. Was used as a targeted hack for shallow-bay sites (Vaihu, Hanga Nui) before OSM replaced both. |

## 9. Misc

| Script | Purpose |
|---|---|
| `build_flat_field.py` | Per-flight flat-field correction (vignette). Currently disabled — empirical evidence showed it regressed at Vaihu (see docs/METHODS.md decisions log). |
| `compare_segmentation.py` | Compare segmenter outputs side by side. |
| `sam_prompt_creator.py` | Hand-craft prompts for SAM segmentation. |
