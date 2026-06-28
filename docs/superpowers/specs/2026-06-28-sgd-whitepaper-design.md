# SGD Thermal-Detection Whitepaper — Design

Date: 2026-06-28
Status: Approved (design phase)

## Goal

Produce a standalone, current whitepaper describing the SGD detection
pipeline as it actually works today (OSM water mask, SRTM cliff-zone
exclusion, coast-anchored watershed plumes, adaptive thresholds,
threshold-independent Σ_anomaly metric, three master products). The
existing `docs/TECHNICAL_PAPER.md` is legacy and describes a superseded
methodology (HSV classifier, "90+ locations"); it is left untouched.

## Audience and voice

Method + practitioners: remote-sensing scientists, coastal hydrologists,
drone-survey teams who might adopt or adapt the method. Technical but
standalone. Follows shared writing conventions (no em dashes, narrative
prose not bullet lists in body, colorblind-friendly figures, direct on
limitations).

## Deliverables

- `docs/WHITEPAPER.md` — editable Markdown source of truth.
- `docs/WHITEPAPER.docx` — compiled Word doc with embedded figures.
- `scripts/figures/build_whitepaper_docx.py` — compiler, modeled on
  `scripts/figures/build_methods_docx.py`.

## Section outline

1. Summary (abstract): problem, approach, headline result (415
   coast-anchored plumes island-wide; Σ_anomaly metric).
2. The problem: why SGD matters; thermal cold signature (1–3 °C);
   limits of manual / satellite / manned methods.
3. Survey and data: Autel 640T, paired thermal–RGB, 29 flights,
   ~30,000 frames, June 2023 + Jan 2024, 1 m resolution.
4. Pipeline overview: end-to-end numbered narrative + diagram.
5. Per-frame anomaly: ocean segmentation, per-frame 75th-pct baseline,
   `max(0, baseline − T)`.
6. Spatial integration: projection to 1 m lat/lon grid, multi-frame
   averaging, transient-artifact suppression.
7. Authoritative masks: OSM coastline water mask + SRTM cliff-zone
   exclusion; geological rationale (lava-tube conduit topology; cliffs
   lack conduit geometry).
8. Coast-anchored detection: watershed from coastline seeds, adaptive
   peak/edge thresholds, polygon filters, documented `source_lat/lon`.
9. Σ_anomaly metric: threshold-independent, m²·°C, cross-site metric.
10. Results: island-wide distribution; three master products
    (415 coastal / 1,066 raster / 1,789 detector); site exemplars
    (Vaihu, Hanga Nui/Tongariki, Hekii, Anakena).
11. Validation and limitations: cliff-coast control (Poike); three
    known limitations (HSV per-frame segmentation, flat-ground
    projection, baseline bias for large plumes) and planned fixes.
    Honest treatment (confirmed with user).
12. Reproducibility: software stack, data sources, repo pointers.

## Figure plan (reuse + targeted new — confirmed)

Reuse (git-tracked under `docs/images/`):
- `sgd_pipeline/island_overview.png`
- `sgd_pipeline/vaihu_closeup.png`, `hanga_nui_closeup.png`,
  `hekii_west_closeup.png`, `anakena_closeup.png`
- `sgd_pipeline/polygon_comparison.png`
- `thermal_rgb_pair.png`, `thermal_alignment.png`,
  `thermal_fov_coverage.png`, `detection_pipeline.png`

Regenerate if existing scripts run cleanly (current data):
- island overview via `scripts/figures/build_island_overview.py`
- site closeups via `scripts/figures/build_site_closeup.py`
  (regeneration is best-effort; fall back to tracked PNGs if a run
  needs the external `/Volumes/RapaNui` volume or per-flight artifacts
  not present)

New schematic figures (2):
- Cliff-zone exclusion schematic: lava-tube conduit at a low bay vs a
  vertical cliff face; why bays produce SGD and cliffs do not.
- Σ_anomaly schematic: integrating per-cell mean anomaly over a polygon
  footprint to yield m²·°C.

## Out of scope / do not break

- Do not modify `docs/TECHNICAL_PAPER.md`.
- Do not re-run the detection pipeline or overwrite OSM/SRTM masks
  (`*_water_mask.npz`, cliff-zone masks).
- Do not invoke the legacy HSV water-mask scripts.

## Verification

- `docs/WHITEPAPER.md` renders; all referenced figure paths exist.
- `build_whitepaper_docx.py` produces `docs/WHITEPAPER.docx` with every
  figure embedded (no missing-image placeholders).
- Numbers in prose (plume counts, thresholds, metric definition) match
  `docs/PAPER_METHODS.md` and `CLAUDE.md`.
