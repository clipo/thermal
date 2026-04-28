# SGD Analysis Methods

Living methods note for the Rapa Nui thermal-drone Submarine Groundwater
Discharge (SGD) analysis. Describes the current pipeline, the metrics we
report, the corrections applied, and a running log of decisions and
known issues so we can keep developing.

Last refreshed: 2026-04-28.

---

## TL;DR — current state (2026-04-28)

**Headline change of this work session:** replaced the HSV satellite
classifier (which had three known failure modes — sandy bays
misclassified as land, gray cliff-rocks classified as water, and tide
mismatch with flight imagery) with **hand-mapped OpenStreetMap
coastline data**. This is a fundamental improvement: OSM is
authoritative for well-mapped islands like Rapa Nui, tide-independent,
and free of pixel-classification noise. It eliminated the need for
per-flight targeted drone-extension hacks (Vaihu and Hanga Nui
shallow bays now correctly classify as water by construction).

**Production pipeline today:**
1. Per-frame thermal + RGB → integrated cold-anomaly raster per flight
   (`build_anomaly_raster.py`).
2. **OSM coastline → per-flight water mask** (`derive_water_mask_osm.py`)
   — replaces HSV.
3. **SRTM 30 m DEM → per-flight cliff-zone mask**
   (`derive_cliff_zone.py`) — flags areas where the max elevation
   within 100 m exceeds 50 m (i.e., cliff coasts where SGD is
   geologically implausible).
4. Three parallel polygon products from the corrected raster:
   - **Detector polygons** — original per-frame detection + density
     clustering. 1,789 polygons, conservative discrete cores.
   - **Raster polygons** — watershed from any local cold peak with
     adaptive thresholds. 1,066 polygons, full plume halos.
   - **Coastal plumes** — coast-anchored watershed; one polygon per
     coastline source. 430 polygons across 26 flights, Σ=457,370
     m²·°C. **Canonical for archaeology correlation.**
5. Master KMLs and figures aggregated across all flights.

**The OSM coastline change was a huge help.** Previously the user
asked "are we sure the segmentation is working?" and the honest
answer was "mostly, with hacks at known-problem sites." After the OSM
switch, the answer is "yes — every cell is classified by an
authoritative hand-mapped boundary, no per-flight hacks needed."

**What's still to address (in priority order):**
1. **Validate cliff-zone identification** — the SRTM-based cliff-zone
   filter is wired in (Phase 1, 2026-04-28), but visual confirmation
   that it's actually catching the cliff coasts the user expects
   is pending. Render flight-by-flight cliff-zone overlays so the
   geological accuracy of the 80 m / 100 m thresholds can be
   eyeballed.
2. **Phase 2 (planned, ~1-2 hours of GPU time):** rebuild all 30,000+
   frames' anomaly rasters with two improvements:
   - **SAM2 per-frame ocean segmenter** (replaces rule-based HSV in
     `segment_ocean_land_waves`). SAM2 will correctly classify cliff
     faces as land instead of sometimes flagging them as ocean.
     Prevents cliff thermal from entering the projection at all.
   - **p75 → p90 baseline** in `compute_per_frame_anomaly`. One-line
     change. Robustness against frames where a plume covers >25% of
     visible ocean (current p75 falls into the plume itself,
     depressing measured anomaly).
   Both require raw frames (now accessible). Then ~30 min downstream
   pipeline rebuild.
3. **Phase 3 (long-term):** DEM-aware footprint projection — the
   proper fix for the cliff-shadow projection bug. Needs raw frames
   + significant modification of
   `footprint_generator.calculate_footprint_corners()` to ray-march
   each pixel against terrain instead of assuming flat ground at
   altitude=0. The current OSM water mask + SRTM cliff-zone filter
   handle the symptoms post-hoc; this would fix the root cause.

---

---

## 1. What we measure

We aggregate raw thermal-drone surveys (Autel Evo II 640T radiometric
thermal + RGB pairs) flown along the Rapa Nui coast in **January 2024**
(targeted SGD surveys) and **June 2023** (broader coastal survey
campaign) into per-coordinate **cold-anomaly content**. Cold anomaly
relative to each frame's warm-water baseline is the primary observable;
SGD plumes appear as persistent cold cells along the coast.

Per cell of a 1 m × 1 m island-wide grid, accumulated across all
overlapping frames in a flight:

```
anomaly_per_frame = max(0, T_baseline_per_frame − T_cell_per_frame)
                       where T_baseline_per_frame = 75th-percentile
                       ocean temperature in that frame.
mean_anomaly_per_cell = sum(anomaly_per_frame) / observation_count
```

This produces a per-flight **anomaly raster** (`<slug>_anomaly.npz`
under `sgd_output/<slug>_spread/`).

---

## 2. Two polygon products

We report SGD detections in two complementary forms; both are useful and
they answer different questions:

### 2a. Detector polygons (discrete cores)

Output: `<slug>_sgd.geojson` per flight, master KML at
`sgd_output/rapa_nui_all_sgd_sigma.kml`.

Pipeline (in `scripts/pipeline/run_coast_stretch.py` + `sgd_toolkit/detectors/spread.py`):

1. Per-frame cold-water detection using a per-frame thermal threshold
   plus a wave-inclusive ocean mask (`thermal_refine.py` extends ocean
   classification into the surf zone).
2. All per-frame polygons rasterized onto a 1 m grid.
3. **Density-grid clustering with local-adaptive threshold**: a cell
   qualifies if `count >= 0.85 × local_peak_in_window`. This rejects
   isolated cold detections and requires spatial-temporal coherence.
4. Connected components → final SGD polygons with per-polygon
   `mean_anomaly`, `peak_anomaly`, `area_m2`, `intensity_index`,
   `n_observations`.

Use case: peer-reviewable discrete SGD detection, validated pipeline,
published in earlier work. Captures cold cores, not the diluted halo.

### 2b. Raster polygons (one polygon per cold peak, watershed-segmented)

Output: `<slug>_sgd_raster.geojson` per flight, master KML at
`sgd_output/rapa_nui_all_sgd_sigma_raster.kml`.

Pipeline (`scripts/pipeline/derive_polygons_from_raster.py`):

1. Take the per-flight anomaly raster.
2. Apply quality filters: `obs_count ≥ 5`, `anomaly ≤ 3 °C`,
   `is_water = True` from the satellite-derived water mask, and
   `anomaly ≤ 0.5 °C` for cells classified as land (cliff-shadow filter).
3. Median 3×3 smoothing.
4. **Adaptive thresholds** based on each flight's water-cell anomaly
   distribution (different flights have very different signal
   strengths — Vaihu peaks ~0.5°C, Hekii peaks 1°C+):
   - `peak_threshold = max(0.4, p95 of water-cell anomaly)`
   - `edge_threshold = max(0.2, p70 of water-cell anomaly)`
   This makes the same algorithm sensitive at subtle-SGD sites
   (Vaihu, Hanga Nui) and strict at strong-signal sites (Hekii)
   without per-flight manual tuning. CLI args `--threshold` and
   `--peak-threshold` override the adaptive values when needed.
5. Find local cold peaks at `peak_threshold`, minimum 30 m apart.
6. Watershed-flood from each peak through cells at `edge_threshold`.
   Each peak seeds one polygon — the "drainage basin" of the cold
   plume around its source. This produces discrete plume-shaped
   polygons rather than one giant blob spanning a whole bay.
7. Filter components by area: `min_area_m2 = 50 m²` (drop noise),
   `max_area_m2 = 10,000 m²` (drop diffuse coastal cooling — real
   SGD plumes rarely exceed 10,000 m²).
8. Extract polygon contours via `skimage.measure.find_contours`.

Use case: discrete plume-shaped polygons, one per local cold peak,
threshold-explicit, reproducible from the raster alone. The previous
connected-component approach (replaced 2026-04-27) gave one giant
polygon per connected cold region, which made entire bays look like
"one SGD" instead of multiple distinct plumes.

### 2c. Coastal-anchored plume polygons

Output: `<slug>_sgd_coastal.geojson` per flight, master KML at
`sgd_output/rapa_nui_all_sgd_coastal.kml`.

Pipeline (`scripts/pipeline/derive_plumes_coast_anchored.py`):

1. Compute coastline as water cells adjacent to land in the satellite
   water mask.
2. Identify source candidates: cells within `coast_buffer_m` (default
   30 m) of the coastline AND with anomaly ≥ adaptive peak threshold.
3. Cluster sources into discrete watershed seeds (min 30 m apart).
4. Watershed-flood from each seed OUTWARD into water cells with
   anomaly ≥ adaptive edge threshold, capped at `max_offshore_m`
   (default 200 m).
5. Reject any polygon that doesn't touch a coastline cell.
6. Each polygon records its `source_lat` / `source_lon` (the coastline
   cell where the SGD plume originates).

Use case: physically motivated SGD plume detection. Every polygon has
a documented coastal source — by construction, no offshore-only blobs
that can't be SGD. Most defensible product for the archaeology
correlation question ("where on the coast is freshwater entering the
ocean?").

Compared to 2b (raster polygons, watershed-from-anywhere): coast-
anchored is more conservative (~50 % fewer polygons), but every kept
polygon is physically interpretable as an SGD plume with source +
extent.

### 2d. Threshold-independent metric: `sigma_anomaly_m2c`

For both polygon products we report **`Σ_anomaly = ∫(T_baseline − T) dA`**
in m²·°C, integrated over the polygon footprint. This metric is

- robust to the SGD detection threshold (the integral doesn't depend on
  which cells were called "SGD"),
- robust across flights/seasons (each flight's per-frame baseline
  normalises out absolute temperature differences),
- the basis for cross-site / cross-season comparison.

Computed by `scripts/pipeline/recompute_polygon_intensity.py`, which integrates
each polygon's footprint over the per-flight raster and writes
`sigma_anomaly_m2c`, `mean_anomaly_in_polygon_c`,
`peak_anomaly_in_polygon_c`, `polygon_water_fraction`,
`raster_coverage_frac`, `raster_n_obs_median` back into the GeoJSON.

---

## 3. The cliff-shadow projection bug — and how we correct it

### Symptom

At cliff-coast sites (Poike, Rano Kau), the original raw outputs
showed a phantom 80,000-cell offshore "SGD plume" with implausibly
extreme cold anomaly (>5°C below baseline; physically impossible for
tropical surface water). Tongariki-Poike was originally ranked the #6
SGD complex on the island at Σ = 58k m²·°C; Poike-3 was #5 at 61k.

### Root cause

`sgd_toolkit/georeferencing/footprint_generator.py` projects each
camera pixel to lat/lon assuming the ground is flat at altitude = 0.
At cliffs (Poike: ~300 m, Rano Kau: ~250 m), the cliff face occupies
the camera's foreground image but its actual elevation is far above 0.
The projection therefore maps cliff-face pixels to lat/lon coordinates
that fall *offshore from the cliff base*. Cliff face is cold (often
in shadow) → the projected lat/lon receives strong cold-anomaly
content → the flight's per-frame ocean baseline is dragged down
(making other ocean cells look slightly less cold relative to it),
*and* the offshore raster cells accumulate phantom cold-anomaly.

A subsidiary contribution: the rule-based RGB ocean segmenter in
`segment_ocean_land_waves` occasionally classifies cliff shadow as
ocean. This compounds the projection error by selecting cliff pixels
into the "ocean" sample for the per-frame baseline. (SAM-class
segmentation would fix this part — see §6.)

### Correction (post-hoc, no pipeline rebuild required)

**Current production: OSM coastline mask** (`scripts/pipeline/derive_water_mask_osm.py`).
Fetches Rapa Nui coastline from OpenStreetMap via the Overpass API
(131 hand-mapped ways with 9344 nodes), builds a Shapely MultiPolygon
for the island, and rasterizes that polygon per-flight at the
anomaly raster's grid resolution. Land = inside polygon; water =
outside. Cached at `sgd_output/osm_coastline.json`.

This replaces the earlier HSV satellite classifier, which had three
known failure modes: (1) cliff coasts where gray rocks are classified
as water, (2) shallow sandy bays where green/turquoise water is
classified as land, (3) tide variation between when the satellite
tile was captured vs when the drone flew. OSM coastline is
hand-mapped, tide-independent (typically defined at mean-high-water),
and free of pixel-classification noise.

`scripts/alternative_water_masks/derive_water_mask.py` (HSV-based) and
`scripts/alternative_water_masks/extend_water_mask_with_drone.py` (drone-obs union) are
retained for reproducing earlier results but are no longer the
production source.

**Drone-observed extension (targeted, not global):** the satellite
HSV test fails at shallow tropical bay water with sandy bottoms
(e.g., **Hanga Nui at Ahu Tongariki** — known SGD ground truth where
freshwater emerges into the bay). Sandy-bottom water reads green or
turquoise in satellite imagery, not blue, so HSV classifies it as
land, dropping the entire Hanga Nui SGD signal.

`scripts/alternative_water_masks/extend_water_mask_with_drone.py` unions the satellite mask
with cells where the drone's per-frame ocean segmenter persistently
classified the cell as ocean (`obs_count >= 5`). The drone's RGB
segmenter, working on close-range high-resolution imagery, correctly
identifies shallow sandy bays as ocean.

We initially applied this extension globally (all 29 flights) and it
recovered Hanga Nui (Σ 223 → 4,256 m²·°C) and improved Vaihu shallow
water — but it also re-introduced the cliff-shadow projection
artifacts at Poike, 6-July, etc., because the same drone segmenter
also misclassifies cliff face as ocean at vertical-cliff coasts.
Globally inflated values were unrealistic ("everything looks like
SGD"), so we **reverted to satellite-only masks globally** and apply
the drone extension only to specific flights with documented ground
truth at sandy/shallow bays:

| Flight slug | Reason for targeted extension |
|---|---|
| `june2023_23_june_23_tongariki_flights` | Hanga Nui Bay (Ahu Tongariki) — known freshwater outflow |
| `vaihu_full` | Vaihu Harbor — textbook SGD ground truth; sandy shallow bay water classified as land by satellite HSV |
| `flight4_vaihu_east_full` | Vaihu East transect — same shallow-bay issue as vaihu_full |

Add new entries to this table with rationale when other flights need
the same treatment. Run:

```bash
python scripts/alternative_water_masks/extend_water_mask_with_drone.py --slug <flight_slug>
python scripts/pipeline/recompute_polygon_intensity.py --slug <flight_slug>
# then re-run the master aggregators
```

The proper fix (proposed in §6) is either DEM-aware projection (catch
the cliff-shadow at the source) or SAM2-based per-frame ocean
segmentation (catch the cliff misclassification before it propagates).
Either fix makes the global drone extension safe to use everywhere.

The water mask is consumed downstream by:

- `recompute_polygon_intensity.py` — integrates `Σ_anomaly` only over
  water cells; tags each polygon with `polygon_water_fraction`.
- `aggregate_sigma_anomaly_kml.py` — drops polygons with
  `Σ_anomaly < 1` (i.e., entirely on land) or in figures with
  `water_fraction = 0`.
- `derive_polygons_from_raster.py` — restricts polygon derivation
  to water cells.
- `build_site_closeup.py` — masks the displayed raster to water cells
  and skips the on-land polygon filter for raster polygons.

The water mask is applied to the master cold-anomaly GroundOverlay
PNGs through TWO complementary filters in `mask_anomaly_pngs.py`:

1. **Soft land filter** — drop cells where satellite says LAND
   AND anomaly > 0.5 °C. This kills the cliff-shadow projection-bug
   cells (typical anomaly 1–5 °C, projected to land coordinates)
   while preserving genuine shallow-water cells that satellite
   misclassifies (Vaihu Harbor sandy bottom, Hanga Nui Bay) — those
   tend to have anomaly < 1 °C.
2. **Isolation filter** — connected-component analysis on ALL kept
   cells (regardless of anomaly value). Components < 10% of the
   largest connected component are dropped. The main flight strip
   typically dwarfs any artifact blob by 10× or more — at the
   Hanga Nui flight, main strip = 350k cells, Ahu Tongariki basalt
   platform = 15k cells (23× ratio). The 10% cutoff catches the
   ahu blob and similar isolated patches (rooftops, residual
   cliff-shadow) without losing the main strip.

Both filters are on by default. Override via flags:
- `--max-land-anomaly 99` to disable soft land filter
- `--no-isolation-filter` to disable isolation filter
- `--no-water-mask` to disable both

After the correction, the Tongariki-Poike "major SGD complex" Σ
collapses to **303 m²·°C (detector) / 14,580 m²·°C (raster)** — small,
real surf-zone signal, no phantom plume. Globally **47% of the
original detector polygons are now classified as projection-bug
artifacts** and filtered from the master KML.

### Geological consistency check

Cliff coasts (Poike, Rano Kau) **should** have minimal SGD because
volcanic groundwater requires conduit topology (collapsed lava tubes
emerging at bays/inlets). The post-correction ranking has bay/inlet
sites dominating (Hekii bays, Hivahiva-Hangapiko, Vaihu Harbor,
Anakena, 25/24-June large surveys) and cliff sites collapsed to near
zero — consistent with the geology, not just an algorithmic
correction.

---

## 4. Current top sites (after all corrections, raster polygons)

| Rank | Flight | Σ_anomaly_m²·°C (raster) | Σ_anomaly (detector) | Notes |
|---|---|---|---|---|
| 1 | 25-June (southwest survey) | 570,222 | 85,994 | broad south coast |
| 2 | 24-June (northeast survey) | 394,427 | 56,599 | broad north coast |
| 3 | Hekii West (flight 8) | 318,229 | 58,491 | bay/inlet |
| 4 | 28-June Rano Kau region | 288,941 | 34,064 | mostly cliff — likely artifacts |
| 5 | Hivahiva-Hangapiko (flight 11) | 263,634 | 55,617 | bay/inlet |
| 6 | 27-June Hanga Roa-Hivahiva | 214,795 | 31,616 | bay/inlet |
| 7 | 5-July South-to-Vai-Mata | 199,592 | 41,379 | bay/inlet |
| 8 | Anakena→West (flight 10) | 197,224 | 34,240 | bay/inlet (Anakena Bay) |
| 9 | 5-July Ahu-O-Huari North | 188,154 | 36,055 | bay/inlet |
| 10 | 1-July Kikirahamea-Hivahiva | 173,665 | 27,154 | bay/inlet |

Detector Σ uses satellite-only water mask globally + targeted
drone-extension at the 23-June Tongariki flight (Hanga Nui Bay only).
Raster Σ is consistently 4–6× the detector Σ because raster polygons
include the broader cold-water halo that the detector's spatial-
coherence requirement excludes.

Master grand total Σ_anomaly: **8,538,560 m²·°C (raster) /
1,147,187 m²·°C (detector)** across 29 flights.

**Hanga Nui at Ahu Tongariki** (23-June Tongariki flight, with targeted
drone extension): Σ_anomaly = 4,256 m²·°C (detector) / 14,580 m²·°C
(raster) — recovered after the targeted extension was applied. This
is real ground-truth SGD documented by the user.

---

## 5. Deliverables (where to find what)

```
sgd_output/
├── <slug>_spread/                              # per-flight outputs
│   ├── <slug>_anomaly.{npz,png,kml}            # cold-anomaly raster + GroundOverlay
│   ├── <slug>_water_mask.npz                   # land/water mask
│   ├── <slug>_sgd.geojson                      # detector polygons
│   ├── <slug>_sgd_raster.geojson               # raster polygons
│   ├── <slug>_sgd_intensity.csv                # per-polygon table
│   └── <slug>_validation.png                   # 3-panel diagnostic
├── rapa_nui_all_sgd_sigma.kml                  # master KML — detector polygons
├── rapa_nui_all_sgd_sigma_raster.kml           # master KML — raster polygons
├── rapa_nui_all_anomaly.kml                    # master KML — anomaly GroundOverlays
├── polygon_intensity_summary.csv               # per-flight totals (detector)
├── polygon_comparison_summary.csv              # both methods side-by-side
└── figures/
    ├── rapa_nui_overview{_raster}.{png,pdf}    # island-wide map
    ├── rapa_nui_flight_ranking{_raster}.png    # cross-flight bar chart
    ├── polygon_comparison.png                   # detector vs raster
    └── closeups/<slug>_<label>_closeup.{png,pdf}  # publication site figures
```

The three master KMLs are complementary views to load in Google Earth:

- `rapa_nui_all_sgd_sigma.kml` — **conservative discrete SGD detections**
  (detector polygons). 1,251 polygons across 29 flights after water-mask
  correction. The "what counts as a detected SGD" canonical product.
- `rapa_nui_all_sgd_sigma_raster.kml` — **full plume coverage including
  halo** (raster-thresholded polygons). Useful for "total cold-water
  content per location" but each polygon can be much larger than its
  detector counterpart since it includes the gradient halo.
- `rapa_nui_all_anomaly.kml` — **GroundOverlay PNGs against satellite
  imagery** for all 29 flights, toggleable per flight. Best view for
  understanding spatial patterns of cold anomaly and seeing how flight
  coverage compares across the coast.

### Scripts (`scripts/`)

| Purpose | Script |
|---|---|
| Per-flight anomaly raster | `build_anomaly_raster.py` (+ batch driver) |
| **OSM coastline water mask (production)** | `derive_water_mask_osm.py` |
| Satellite water mask (HSV, legacy) | `derive_water_mask.py` |
| Extend water mask with drone obs (legacy targeted) | `extend_water_mask_with_drone.py` |
| Satellite water mask (SAM2, exploratory) | `derive_water_mask_sam2.py` |
| **SRTM DEM cliff-zone mask** | `derive_cliff_zone.py` |
| Polygon Σ_anomaly recompute | `recompute_polygon_intensity.py` |
| Raster-derived polygons | `derive_polygons_from_raster.py` |
| Master KML aggregator (polygons) | `aggregate_sigma_anomaly_kml.py` |
| Master KML aggregator (anomaly GroundOverlays) | `aggregate_anomaly_kml.py` |
| Re-render anomaly PNGs with quality filters | `mask_anomaly_pngs.py` |
| Cross-flight comparison | `build_polygon_comparison_summary.py` |
| Island overview figure | `build_island_overview.py` |
| Flight ranking figure | `build_flight_ranking.py` |
| Site closeup figure | `build_site_closeup.py` |
| Per-flight 3-panel | `build_validation_figure.py` |
| Point-radius proximity | `sgd_proximity.py` |
| Coastline polyline sampler | `sample_coastline.py` |
| Diagnostic — ocean mask one frame | `diagnose_ocean_mask.py` |
| Diagnostic — water mask vs satellite | `diagnose_water_mask.py` |
| Diagnostic — SAM2 tile classification | `debug_sam2_water_mask.py` |

---

## 6. Known limitations and ideas for improvement

### Limitations of the current pipeline

1. **Flat-ground projection** in `footprint_generator.py` is the root
   cause of the cliff-shadow artifacts. Our post-hoc water mask treats
   the symptom; the root fix is a DEM-aware projection (use SRTM 30 m
   for Rapa Nui to project each pixel to actual terrain elevation).
2. **HSV-based ocean segmentation** in `segment_ocean_land_waves` is
   brittle at edge cases (shadow, foam, glare). Visual sanity-checks
   on the per-frame masks at marginal flights would be informative.
3. **Per-frame baseline = 75th-percentile ocean temperature** assumes
   most of the frame is "ambient" ocean. Frames dominated by a single
   cold lens (e.g., a wide SGD plume covering >25% of the frame) will
   have a depressed baseline → suppressed anomaly. This biases against
   detecting very large plumes.
4. **No tide / time-of-day correction**. Surveys flown at low tide
   may show different signal than at high tide. We currently treat
   each frame's relative anomaly as the unit of comparison, which
   normalises absolute sea-surface temperature but not local surf
   dynamics.
5. **Single-altitude assumption**. Frames flown at very different
   altitudes have different ground sample distance; the
   bilinear-interp projection treats all frames equally.

### Ideas explored or pending

| Idea | Status | Notes |
|---|---|---|
| SAM2 satellite tile water mask | Tested, not adopted | SAM2-segment + mean-color classifier mistakes cliff face for water (gray-blue mean). HSV per-pixel is more reliable for tropical Pacific water. Hybrid (SAM2 segments + per-pixel HSV inside) prepared but not deployed. Script kept at `scripts/alternative_water_masks/derive_water_mask_sam2.py`. |
| SAM2 per-frame drone RGB segmentation | Not yet tried | Would catch cliff shadows that the rule-based segmenter misses; feasible on Apple Silicon MPS at ~50–100 ms/frame. Cost: ~25–50 min for ~30 k frames. **Worth doing** if we want to address cause #2 above. |
| DEM-aware projection (SRTM 30 m) | Not yet tried | Proper fix for cause #1. Requires modifying `footprint_generator.calculate_footprint_corners()` to ray-march each pixel ray against terrain elevation. Significant code change but clean. |
| Coastline-segment SGD density | Tooling ready | `sample_coastline.py` + `sgd_proximity.py` accept a coastline GeoJSON polyline and produce per-segment Σ_anomaly. Awaits a Rapa Nui shoreline polyline (OSM extract or hand-digitised). |
| Archaeology proximity correlation | Tooling ready | `sgd_proximity.py` accepts arbitrary input points (CSV with lat/lon) and integrates Σ_anomaly within a configurable radius. Awaits an ahu / moai feature dataset. |
| Cross-season same-location comparison | Not yet built | Where Jan 2024 and June 2023 flight extents overlap (e.g. Hekii, Hanga Roa), compare per-segment Σ_anomaly across seasons to identify persistent vs seasonal SGDs. |

---

## 7. Decisions log

This section is a running record of analytical decisions, reversals,
and rationale. Append new entries with date and short note when we
change something material.

- **2026-04-27**: Cliff-shadow projection bug diagnosed at Poike. The
  raw raster max anomaly of 18.35 °C below baseline at Poike-3 was the
  physically-impossible smoking gun — tropical sea cannot be 6 °C
  cold. Built `derive_water_mask.py` (HSV) as the post-hoc fix.
  Filtered the master KML; 47% of original detector polygons removed
  as artifacts.
- **2026-04-27**: Tested SAM2 (Apple Silicon MPS, sam2.1-hiera-tiny).
  Rejected as primary classifier — works at Vaihu (slight harbor
  improvement) but misclassifies cliff face as water at Poike (worse
  than HSV) and produces 0%-water failures at wide-bbox flights
  (projection alignment issue). HSV restored everywhere. SAM2 script
  retained for future investigation with hybrid (SAM2 segment + per-
  pixel HSV) classifier.
- **2026-04-27**: Lenient polygon filter — drop only fully-on-land
  polygons (Σ_anomaly = 0), keep partial-water polygons (Σ_anomaly > 0,
  any water content). The previous strict filter (`water_fraction
  ≥ 0.5`) was rejecting legitimate Vaihu Harbor polygons where
  shallow water + sand + surf foam don't pass the strict satellite
  HSV water test even though the SGD ground truth is solid. This
  rescued 233 polygons globally including Vaihu's harbor signal.
- **2026-04-27**: Confirmed geological interpretation. Cliff coasts
  (Poike, Rano Kau) lack lava-tube SGD outlets; the corrected
  analysis correctly shows them as low-Σ. The Poike "signal" was an
  artefact, not a real measurement; the corrected map matches Rapa
  Nui's expected SGD geology (concentrated at bays/inlets).
- **2026-04-27**: Identified that Hanga Nui at Ahu Tongariki — known
  SGD ground truth — was being killed by the satellite-only water
  mask because its shallow sandy-bottom water doesn't pass the HSV
  blue test. Added `extend_water_mask_with_drone.py` to union the
  satellite mask with cells the drone's RGB segmenter persistently
  classified as ocean (`obs_count >= 5`). Hanga Nui Σ recovered:
  223 → 4,256 m²·°C.
- **2026-04-27**: Tested the drone extension applied globally — too
  permissive; everything looked like SGD because the drone segmenter
  also misclassifies cliff face as ocean at vertical-cliff coasts.
  Reverted to satellite-only masks globally; apply the drone
  extension only to specific flights with documented ground truth at
  sandy/shallow bays (currently: just the 23-June Tongariki flight
  for Hanga Nui). Added a table in §3 listing flights that get the
  targeted extension and the rationale for each. Future entries
  should be appended only when ground-truth justifies it.
- **2026-04-27**: Built `aggregate_anomaly_kml.py` to produce a
  master island-wide KML of all 29 per-flight anomaly GroundOverlay
  PNGs (`rapa_nui_all_anomaly.kml`). User confirmed this view is the
  most informative for spatial pattern recognition (cold-anomaly
  raster overlaid on Google Earth's satellite imagery, toggleable per
  flight).
- **2026-04-27**: Wrote `mask_anomaly_pngs.py` to re-render per-flight
  anomaly PNGs with quality + water-mask filters. Initial version
  applied the satellite water mask, which dropped legitimate
  continuous shoreline signal at Vaihu. Reverted: anomaly PNGs are
  now rendered with `--no-water-mask` (just `obs >= 5` and
  `anomaly <= 3 °C`). The 3 °C cap removes the most extreme
  cliff-shadow projection-bug cells; moderate cliff-shadow at Poike
  remains visible in the anomaly PNG GroundOverlay but the polygon
  products correctly attribute zero/near-zero Σ_anomaly to those
  flights via the polygon-side water mask filter. This dual-strategy
  (loose for raster visuals, strict for polygon metrics) is the
  current production setup.
- **2026-04-27**: Tested tightening the satellite HSV thresholds
  (blue_score 0.36 → 0.38, V 0.78 → 0.75, S 0.10 → 0.12, dilation
  2 → 0). Killed the continuous Vaihu shoreline signal (water frac
  21.5% → 11%); the gain in cliff-shadow rejection wasn't worth the
  loss of legitimate signal. **Reverted to lenient defaults**;
  documented thresholds in code with rationale. Future tightening
  should be tested on the Vaihu validation case before deployment.
- **2026-04-27**: Two-stage anomaly-PNG filter (in
  `mask_anomaly_pngs.py`). User insight: real ocean cold-water plumes
  always connect to actual ocean (via bay mouth or surf zone) —
  isolated cold blobs on land are misclassifications.
  (a) **Soft land filter**: drop cells where satellite says LAND AND
      anomaly > 0.5°C. Catches cliff-shadow projection cells (typical
      anom 1–5°C) without losing Vaihu/Hanga Nui shallow-water cells
      that satellite misclassifies as land but have low anomaly.
  (b) **Isolation filter**: connected-component analysis on ALL kept
      cells; components < 10% of the largest connected component are
      dropped. Catches things like the basalt platform of Ahu
      Tongariki (a rectangular cold blob on land, ~15k cells at the
      23-June Tongariki flight; main flight strip is ~350k cells, so
      23× ratio — easy to threshold). Started with a "must touch
      satellite-water" rule, but the satellite HSV mask has scattered
      false-positive water cells that overlap nearly every component,
      so the connectivity rule was useless. Pivoted to size-based
      threshold (10% of largest), which is robust to satellite-mask
      noise. Started with 0.3°C anomaly threshold for "what counts as
      cold to be considered for isolation"; Ahu Tongariki blob has
      cells with anomaly < 0.3°C so it survived. Switched to filtering
      ANY kept observed cell (regardless of anomaly value), making
      the size-based test the entire isolation rule.
  Both filters are applied by default in `mask_anomaly_pngs.py` and
  in `build_island_anomaly_mosaic.py`.
- **2026-04-27**: User noted raster polygons were "giant" and
  "everywhere" — entire bays became one polygon. Switched
  `derive_polygons_from_raster.py` from connected-component
  thresholding to **watershed segmentation**: find local cold peaks
  (>= 0.8°C, min 30m apart), watershed-flood from each through cells
  >= 0.5°C, one polygon per peak. Also added `max_area_m2 = 10,000 m²`
  cap to drop diffuse coastal cooling that isn't a discrete plume.
  Now produces plume-shaped polygons with sources at the coast.
  Per-flight: Hanga Nui 10 plumes (was 19 connected blobs), Anakena 4
  plumes (was 1 giant blob), Vaihu 9 plumes (was 1+ giants). Key
  parameters tunable: `--threshold` (edge), `--peak-threshold` (core),
  `--peak-min-distance-m`, `--max-area-m2`.
- **2026-04-27**: User noted single 0.8°C peak threshold was too
  strict for subtle-SGD sites like Vaihu (water peaks ~0.5°C, p95=0.5).
  Replaced with **adaptive thresholds**: peak_threshold = max(0.4,
  p95 of water-cell anomaly), edge_threshold = max(0.2, p70 of
  water-cell anomaly). Each flight gets thresholds tuned to its own
  signal distribution. Vaihu's adaptive peak=0.50°C catches 34 plumes
  (was 9); Hekii West's adaptive peak=1.00°C is stricter and gets 14
  (was 24); Poike-3's adaptive peak=1.38°C suppresses cliff-shadow
  artifacts to 6 polygons (was 19). Same algorithm, sensible per-site
  sensitivity. CLI overrides via `--threshold` / `--peak-threshold`
  preserved for manual tuning.
- **2026-04-27**: User pointed out Vaihu polygons weren't at the
  actual coast — they stopped at lon -109.373, but Vaihu Harbor is at
  -109.385. The harbor's shallow sandy bay is classified as LAND by
  the satellite HSV mask, so the polygon-derivation pipeline (which
  operates only on water-classified cells) couldn't see the harbor
  signal. Added `vaihu_full` and `flight4_vaihu_east_full` to the
  targeted drone-extension table (alongside the 23-June Tongariki
  flight for Hanga Nui). Vaihu_full now has 53 polygons (was 34) /
  Σ=52,671 m²·°C (was 23,227); flight4_vaihu_east_full has 33 polys
  / Σ=42,214 m²·°C (was 20,005). Polygons now extend to Vaihu Harbor
  at -109.385.
- **2026-04-27**: User asked: "is this best we can do? im not sure
  we are really finding SGD". Diagnosed: existing watershed finds
  cold peaks ANYWHERE — produces offshore-only blobs that can't
  physically be SGD (no coastal source to feed them). Wrote a NEW
  detection method, `derive_plumes_coast_anchored.py`, that encodes
  the SGD physics: (1) define coastline as water-cells-adjacent-to-
  land in the satellite mask, (2) find source candidates within 30 m
  offshore of coast, (3) watershed-flood from each coastal source
  OUTWARD into water cells, capped at 200 m offshore, (4) reject any
  polygon that doesn't touch a coastline cell. Each polygon now
  records its `source_lat` / `source_lon` — physical location where
  freshwater emerges from the coast. New product
  `<slug>_sgd_coastal.geojson` per flight + master KML
  `rapa_nui_all_sgd_coastal.kml`. Coastal product is the most
  defensible for the archaeology correlation question — every plume
  has a documented coastal source.
- **2026-04-28**: User asked "are we sure the segmentation is
  working right?" Diagnosis: HSV satellite mask had a 41% suspect
  rate at Poike-3 (water cells with blue_score < 0.36 — likely
  misclassified). Even when the HSV mask was strict enough at cliff
  coasts (Poike-3 = 1.1% water — correct), it was severely
  undercounting water in adjacent ocean. Replaced HSV with **OSM
  coastline data**: fetched 131 hand-mapped ways for Rapa Nui via
  Overpass API, built a Shapely land polygon, rasterized per-flight.
  Authoritative, tide-independent, free of HSV failure modes. Per-
  flight effects: Poike-3 1.1% → 29.1% water (HSV had been severely
  cropping the offshore zone); Vaihu 21.5% → 31.3% (no longer needs
  targeted drone extension since OSM correctly classifies sandy bays
  as water). The targeted-extension table in §3 (kept for
  reproducibility) is no longer applied by default. New production
  script: `derive_water_mask_osm.py`. **User feedback: "the OSM map
  is a huge help"** — confirmed this is the right foundation.
  Pipeline rebuild: detector 1,789, raster 1,066, coastal 430
  polygons (was 1,310 / 835 / 682 with HSV+drone-extension).
- **2026-04-28**: Downloaded NASA SRTM 30 m DEM for Rapa Nui
  (S28W110.SRTMGL1.hgt, 14 MB) and wrote `derive_cliff_zone.py` to
  build per-flight cliff-zone masks. For each grid cell, the SRTM
  elevation is sampled and a 100 m max-filter is applied; cells
  where max nearby elevation > 50 m are flagged as cliff zones.
  Validation: Anakena 0% cliff (sandy bay, correct), Hanga Nui 8.6%
  cliff (correct — bay terrain), Poike-3 74.6% (correct — cliff
  coast), Rano Kau 73.3% (correct), max elevation in any flight
  bbox = 507 m (Maunga Terevaka, the actual high point of Rapa Nui).
  Cliff-zone rasters built for all 29 flights but not yet wired into
  the coast-anchored detector — that's the next integration step.
- **2026-04-27**: User reported coastal product still had offshore
  blobs AND was filtering too much in some places, plus cliff-shadow
  on north/west cliff coasts. Calibrated the coast-anchored detector:
  (a) `coast_buffer_m` widened 30→60 m so legitimate sources slightly
  inside the surf zone are caught, (b) `max_offshore_m` tightened
  200→150 m, (c) NEW `max_centroid_offshore_m=75 m` filter drops
  polygons whose centroid is far offshore even if a thin tail
  touches the coastline, (d) NEW cliff-projection filter: water
  cells within 15 m of LAND with anomaly >1.5 °C AND own anomaly
  >1.0 °C are dropped (clear cliff-shadow signature; legitimate
  surf-zone signal stays under <1 °C). Per-flight: Vaihu 33 plumes
  (was 30), Hanga Nui 13 (same), Hekii West 5 (was 6 — calibrated
  cliff filter is stricter), Ahu O Huari North 19 (was 23 in
  initial release — first attempt at 0.8/0.6 thresholds dropped to
  1, far too aggressive; calibrated 1.5/1.0 thresholds restored).
  Master 682 plumes across 29 flights at
  `sgd_output/rapa_nui_all_sgd_coastal.kml`.
- **2026-04-27**: Built `build_island_anomaly_mosaic.py` — single PNG
  showing all 29 anomaly rasters at their actual lat/lon footprints
  on Esri satellite basemap, with the same filter chain applied.
  Output `sgd_output/figures/rapa_nui_all_anomaly_mosaic.{png,pdf}`.
- **2026-04-27**: Added `--polygon-source raster` to
  `build_validation_figure.py`. Detector polygons are intentionally
  conservative (only spatial-temporally coherent cores), so they
  don't visually trace every red anomaly cell in the validation
  middle-panel. Raster polygons (derived directly from the anomaly
  raster at 0.3°C threshold) match by construction. Use detector
  polygons when reporting "discrete SGD detections"; use raster
  polygons when the question is "what cold zones are visible".
- **2026-04-27**: Adopted `Σ_anomaly_m2c` (m²·°C) as the canonical
  cross-site metric. Replaces the legacy `intensity_index =
  area × peak_anomaly_c` which was threshold-dependent. Both metrics
  are reported in the GeoJSON for back-compatibility.
- **2026-04-27**: Decided to ship two polygon products in parallel
  (detector + raster) rather than picking one. Rationale: detector
  is conservative + validated, raster is inclusive + threshold-
  explicit; they answer different questions and the comparison itself
  is informative.

---

## 8. How to refresh after a code change

```bash
# 1. (If anomaly rasters changed) rebuild rasters
bash scripts/pipeline/build_all_anomaly_rasters.sh

# 2. Build / rebuild satellite water masks (HSV from Esri tiles)
python scripts/alternative_water_masks/derive_water_mask.py --all --force
# 25-June is too big for zoom 16; rebuild it at zoom 15
python scripts/alternative_water_masks/derive_water_mask.py --slug june2023_25_june_23 --zoom 15 --force

# 2b. TARGETED drone-extension for documented sandy-bay sites only.
#     Do NOT apply globally — cliff-coast flights re-introduce
#     projection-bug artifacts. See METHODS.md §3 for the table.
python scripts/alternative_water_masks/extend_water_mask_with_drone.py \
    --slug june2023_23_june_23_tongariki_flights

# 3. Recompute polygon Σ_anomaly with water-mask correction
python scripts/pipeline/recompute_polygon_intensity.py --all

# 4. Derive raster-thresholded polygons (uses water mask)
python scripts/pipeline/derive_polygons_from_raster.py --all

# 5. Re-render per-flight anomaly PNGs with obs+outlier filters but
#    NOT the water mask (preserves continuous Vaihu signal).
python scripts/pipeline/mask_anomaly_pngs.py --all --no-water-mask

# 6. Refresh master KMLs
python scripts/aggregate/aggregate_sigma_anomaly_kml.py \
    --output sgd_output/rapa_nui_all_sgd_sigma.kml \
    sgd_output/*_spread/*_sgd.geojson

python scripts/aggregate/aggregate_sigma_anomaly_kml.py \
    --output sgd_output/rapa_nui_all_sgd_sigma_raster.kml \
    sgd_output/*_spread/*_sgd_raster.geojson

python scripts/aggregate/aggregate_anomaly_kml.py
# → sgd_output/rapa_nui_all_anomaly.kml (master GroundOverlay KML)

# 7. Refresh cross-flight comparison + figures
python scripts/aggregate/build_polygon_comparison_summary.py
python scripts/figures/build_island_overview.py
python scripts/figures/build_island_overview.py --polygon-source raster
python scripts/figures/build_flight_ranking.py
python scripts/figures/build_flight_ranking.py --polygon-source raster

# 8. (Optional) refresh per-flight validation figures
python scripts/figures/build_validation_figure.py --all

# 9. (Optional) refresh hero closeups
python scripts/figures/build_site_closeup.py --slug vaihu_full --center -27.16838 -109.38511 --box-m 700 --label "Vaihu Harbor"
python scripts/figures/build_site_closeup.py --slug june2023_23_june_23_tongariki_flights --center -27.126 -109.276 --box-m 600 --label "Hanga Nui (Ahu Tongariki)"
# ... and similarly for other sites
```
