# Archive Directory

This directory contains old code, test scripts, and development artifacts that have been superseded by newer versions or are no longer needed for production use.

## Cleanup Date

November 18, 2025

## Directory Structure

### old_versions/
Contains older versions of production scripts that have been superseded by newer implementations:
- `multi_threshold_analysis.py` - Superseded by multi_threshold_analysis_v2.py

### tests/
Test scripts moved from the root directory:
- `test_area_model_selection.py`
- `test_baseline_methods.py`
- `test_frame_rotation.py`
- `test_frame_sampling.py`
- `test_kml_paths.py`
- `test_ocean_cleanup.py`
- `test_ocean_filtering_improved.py`
- `test_polygon_merge.py`
- `test_segmentation_upgrade.py`

These were development test scripts used during feature development and debugging.

### utilities/
One-off utility scripts and development artifacts:

**Fix Scripts** (one-time fixes for specific issues):
- `fix_figure9_correctly.py`
- `fix_figure9_final.py`
- `fix_paper_figures.py`
- `fix_segmentation_figure_properly.py`

**Figure Generation Scripts** (documentation/paper figure creation):
- `create_accurate_figures.py`
- `generate_paper_figures.py`
- `generate_plume_detail.py`
- `generate_quick_figures.py`
- `generate_single_figure.py`
- `generate_thermal_comparison.py`
- `generate_vaihu_figures.py`

**Training Improvement Scripts** (development artifacts):
- `enhanced_segmentation.py`
- `improve_training_interface.py`
- `improve_training_sampling.py`

### test_outputs/
Old test output files from development and testing:
- Test CSV, GeoJSON, and KML files (test_*.csv, test_*.geojson, test_*.kml)
- Old test run outputs from September 6, 2025 (sgd_frame_3_20250906_*, sgd_all_20250906_*)

## Deleted Files

The following test output directories and files in `sgd_output/` were permanently deleted as they were development artifacts taking up significant disk space (~200-400 MB):

- `test_aggregated_debug_individual/`
- `test_aggregated_fixed_individual/`
- `test_aggregated_individual/`
- `test_clean_individual/`
- `test_complete_individual/`
- `test_final_complete_individual/`
- `test_final_individual/`
- `test_final_v2_individual/`
- `test_final_v3_individual/`
- `test_search_multi_individual/`
- `test_success_individual/`
- `test_trace_individual/`
- `test_working_individual/`
- All associated test_*.kml, test_*.json, and test_*.geojson files

## Files Moved to docs/images/

The following PNG files were moved from the root directory to `docs/images/`:
- `irx_variable_regions.png`
- `test_orientation_effect.png`
- `thermal_fov_coverage.png`

## Production Files Retained

All production scripts, real survey outputs, and active models were retained in the main project directory. The cleanup focused solely on development artifacts and superseded code.

## Recovery

If any archived file is needed, it can be moved back to the appropriate location in the project. The files are preserved here for reference and potential future use.
