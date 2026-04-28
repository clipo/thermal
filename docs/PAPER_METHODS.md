# Methods

## Study area and data acquisition

We surveyed the entire coast of Rapa Nui (Easter Island; 27.1°S, 109.4°W) using paired thermal–RGB drone flights between June 2023 and January 2024. A total of 29 flights covered the bays, inlets, and cliff coasts of the island, producing approximately 30,000 paired frames at 1 m ground resolution. Each frame consisted of a radiometric thermal image (Autel 640T payload, 640 × 512 pixels, calibrated to °C) co-registered with a high-resolution RGB frame (4096 × 3072 pixels). All flights flew at 80–150 m altitude with camera nadir-locked. Each frame's GPS position, camera heading, and altitude were recorded in EXIF metadata.

## Per-frame thermal anomaly

For each frame we segmented ocean from land and wave classes using a rule-based HSV classifier on the RGB frame, then computed per-pixel cold anomaly relative to the frame's own warm-water baseline. The baseline was the 75th percentile of ocean-classified pixel temperatures within the frame, and the anomaly at each pixel was `max(0, baseline − T_pixel)` in °C. The per-frame baseline normalises out diurnal and atmospheric drift between flights and seasons, leaving the relative cold-anomaly signature of submarine groundwater discharge (SGD) plumes against the local ambient ocean.

## Spatial integration into anomaly rasters

We projected each ocean-classified pixel from image space to geographic coordinates using the camera's GPS position, altitude, heading, and known field of view, accumulating cells onto a fixed 1 m × 1 m latitude/longitude grid spanning each flight's footprint. Per cell, the mean cold anomaly was computed across all overlapping frame contributions: the sum of `max(0, baseline − T)` over all frames that observed the cell, divided by the cell's observation count. This produced a per-flight integrated cold-anomaly raster, robust against transient artifacts (sun glint, single-frame sensor noise) because such effects do not persist at the same lat/lon across overlapping frames.

## Authoritative water mask: OpenStreetMap coastline

To distinguish ocean cells from cells inadvertently mapped to land (for example, cliff-face pixels misprojected to offshore coordinates by the flat-ground assumption inherent in the camera projection), we built a per-flight water mask from OpenStreetMap (OSM) coastline data. We fetched 131 hand-mapped coastline ways for Rapa Nui (9344 nodes) through the Overpass API, stitched them into a Shapely MultiPolygon representing the island's emergent land surface, and rasterized at each flight's grid using vectorised point-in-polygon containment. The OSM coastline is hand-mapped, tide-independent (defined at mean high water), and free of the noise that affects pixel-level classification of satellite imagery. Cells inside the polygon were labeled land; cells outside were labeled water.

## Cliff-zone exclusion using SRTM elevation

The flat-ground projection in the camera footprint model can mismap pixels of vertical cliff faces (Poike, ~370 m; Rano Kau, ~250 m) to ocean coordinates near the cliff base, producing phantom cold-anomaly content offshore. Even where the OSM water mask correctly identifies those cells as water, they can still be polluted by misprojected cliff thermal. We mitigated this by deriving a per-flight cliff-zone mask from the NASA SRTM 30 m global digital elevation model. For each cell of the flight grid we sampled the bilinearly-interpolated SRTM elevation, applied a 100 m maximum-elevation filter, and flagged cells where the local maximum elevation exceeded 80 m as cliff-zone cells. This filter is grounded in Rapa Nui's volcanic groundwater geology: SGD requires lava-tube conduit topology, which forms only at low-elevation bays and inlets where collapsed tubes intersect the sea surface; vertical cliffs lack the geometry for freshwater to emerge at the sea surface even if the underlying aquifer is contiguous. Cliff-zone cells were excluded from both plume seed candidates and watershed flood propagation.

## Coast-anchored plume detection

We extracted SGD plume polygons from the integrated anomaly raster using a coast-anchored watershed segmentation. The coastline was defined as cells classified as water in the OSM mask that are immediately adjacent to land cells; this produces a one-cell-wide ring tracing the actual island shoreline. Source candidates were water cells within 60 m offshore of the coastline (excluding cliff-zone cells) with cold anomaly above an adaptive peak threshold. Candidate sources were clustered using local-maximum detection with a minimum separation of 30 m between seeds. From each seed, the watershed algorithm propagated outward through water cells with anomaly above the adaptive edge threshold, bounded at 150 m offshore from the coastline. We then applied polygon filters: minimum area 50 m² (drop noise), maximum area 8000 m² (drop diffuse coastal cooling), and centroid within 75 m of the coastline (drop polygons with thin tails reaching back to shore but bodies far offshore). Each surviving polygon represents one discrete SGD plume with a documented coastal source, recorded as `source_lat` and `source_lon` properties.

### Adaptive thresholds

The adaptive peak and edge thresholds are derived per-flight from the cold-anomaly distribution of water cells. The peak threshold is the 95th percentile of cold-anomaly values among water-classified cells in the flight, clipped to the range 0.35 to 0.55 °C. The edge threshold is the larger of the 70th percentile and one-half the peak threshold, clipped to the range 0.20 to 0.35 °C. The floor (0.35 / 0.20 °C) ensures detection at subtle-SGD sites such as Vaihu Harbor, where peak anomalies are typically 0.4 to 0.6 °C; the ceiling (0.55 / 0.35 °C) prevents strong-signal flights or flights with cliff-shadow contamination from setting thresholds so high that other coastal zones in the same flight are filtered out.

## Σ_anomaly: threshold-independent intensity metric

Because the 0.3 °C threshold is somewhat arbitrary, we report each polygon's integrated cold-anomaly content as a threshold-independent metric, Σ_anomaly, defined as the integral of mean cell anomaly over the polygon footprint, in units of m² · °C. In practice it is computed as the sum, over all cells inside a polygon, of the mean per-cell anomaly multiplied by the cell area. Σ_anomaly is robust to the SGD detection threshold (the integral does not depend on which cells were called "SGD" or where the polygon boundary was drawn), robust across flights and seasons (each flight's per-frame baseline normalises out absolute temperature differences), and is the appropriate metric for cross-site quantitative comparison.

## Quality filters and projection-bug mitigation

Three filters are applied to suppress the known projection-bug artifacts at cliff coasts. First, cells classified as land by the OSM mask with cold anomaly exceeding 1.5 °C and adjacent (within 15 m) water cells with anomaly exceeding 1.0 °C are flagged as misprojected cliff thermal and dropped from the watershed input. Second, cells with cold anomaly exceeding 3.0 °C are dropped (real SGD plumes rarely exceed 1.5 °C below ambient; values above 3 °C are sensor or atmospheric outliers). Third, cells observed by fewer than five overlapping frames are dropped (insufficient temporal averaging).

## Cross-flight aggregation

Per-flight polygons were aggregated into a master KML by season, with polygons coloured by Σ_anomaly tier (six logarithmic tiers from "very low" to "extreme"). Master grand-total Σ_anomaly is reported across all flights. The system also generates per-flight validation figures (a three-panel diagnostic showing the anomaly raster, polygon overlay, and observation count) and site closeups that overlay detected plumes on Esri World Imagery satellite basemaps.

## Limitations

The current pipeline has three known limitations. First, per-frame ocean segmentation uses a rule-based HSV classifier that can occasionally misclassify shaded cliff face as ocean. The downstream OSM water mask and SRTM cliff-zone filter mitigate this, but a SAM2-based per-frame segmenter (planned future work) would prevent cliff thermal from entering the projection at all. Second, the camera footprint projection assumes a flat ground at sea level, mismapping high-elevation pixels to offshore coordinates; a DEM-aware projection (planned long-term) using SRTM terrain to ray-march each pixel would fix this at the source. Third, the per-frame baseline (75th percentile of ocean cells) can be biased low when a plume covers more than 25 percent of the frame; switching to the 90th percentile (planned alongside SAM2) would improve robustness for very large plumes. These limitations are explicitly logged in the project's living methods document and prioritised for future iterations.

## Software and reproducibility

All processing is implemented in Python with scientific-Python ecosystem libraries (numpy, scipy, scikit-image, shapely, matplotlib). Geospatial coastline data is fetched from OpenStreetMap via the Overpass API; elevation data is NASA SRTM 30 m via ESA Sentinel mirror. Source code, processing scripts, and a living methods document are available at the project repository. Per-flight raw inputs (paired thermal and RGB frames) and per-flight outputs (anomaly rasters, water masks, cliff-zone masks, polygon GeoJSONs, validation figures) are organised by flight slug under a per-flight directory for traceability.

## Figure captions

**Figure 1.** Island-wide SGD distribution. Coast-anchored SGD plumes detected across 29 thermal-drone flights of Rapa Nui (June 2023 and January 2024). Each point represents one plume polygon; size and colour scale with the polygon's Σ_anomaly (m² · °C of integrated cold-anomaly content). Background shaded blocks show flight footprints. The full-resolution KML version allows toggling individual flights and inspecting polygon attributes.

**Figure 2.** Vaihu (Ahu Vaihu, textbook SGD reference site on the south coast). Coast-anchored plumes overlaid on Esri WorldImagery satellite tile. Seven discrete plumes traced along the rocky shore and surf zone, with sources where freshwater is known to emerge through collapsed lava-tube outlets. Coloured shading shows mean cold-anomaly inside each plume polygon.

**Figure 3.** Hanga Nui at Ahu Tongariki. Discrete plumes detected in the bay where the famous moai platform is sited. Bay geometry typical of Rapa Nui's SGD-active sites: low-elevation beach and rocks, sheltered from the open Pacific, with Poike's volcanic slopes inland.

**Figure 4.** Hekii West. Discrete plumes along a known SGD hot zone on the east coast. Strong cold-anomaly signal (peak around 1 °C below ambient) typical of strong-flow lava-tube outlets.

**Figure 5.** Anakena and Ovahe (north coast). Four coast-anchored plumes across this 2.2 km stretch — three concentrated at the rocky Ovahe headland on the west side of the frame, plus one weaker plume offshore of Anakena's sandy beach itself. The modest Σ_anomaly (~1,800 m²·°C) is consistent with this coast's small upstream drainage basin and modest groundwater throughput.

**Figure 6.** Hivahiva-Hangapiko (south coast). Plumes along the south coast bay system with strong cold signal at the rocky inshore zone.

**Figure 7.** Poike (cliff-coast control). The cliff-zone filter based on SRTM elevation correctly suppresses the spurious cliff-shadow polygons that would otherwise dominate this flight. Surviving plumes are at the surf zone immediately offshore of the cliff base, where some real cold-water signal may exist (or may still represent residual misprojection — flagged as a known limitation).
