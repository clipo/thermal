# Configuration Templates

This directory contains example configuration files for the SGD Analysis Wizard.

## Available Templates

### example_config.json
**Recommended starting point** - Balanced settings that work well for most surveys
- Temperature threshold: 0.5°C
- Improved detector with upper quartile baseline
- ML segmentation enabled
- Sun glint detection enabled
- All output formats enabled

### quick_scan.json
**Fast preliminary analysis** - For quick checks or testing
- Higher temperature threshold (1.0°C) = fewer false positives
- Larger minimum area (100 pixels) = only significant SGDs
- Basic integrated detector (faster)
- No ML segmentation = faster processing
- KML output only

### detailed_analysis.json
**Maximum sensitivity** - For comprehensive scientific analysis
- Lower temperature threshold (0.3°C) = detect subtle SGDs
- Smaller minimum area (25 pixels) = detect small features
- Improved detector with all enhancements
- ML segmentation enabled
- All output formats for thorough documentation

## Using Templates

### Copy and modify for your survey:
```bash
# Copy a template
cp configs/example_config.json configs/my_survey.json

# Edit the data_dir and output_name
nano configs/my_survey.json

# Run with your config
python scripts/sgd_wizard.py --config configs/my_survey.json
```

### Create from scratch (interactive):
```bash
# Interactive wizard creates a custom config
python scripts/sgd_wizard.py --save-only --output configs/my_custom.json
```

### Reuse on different datasets:
```bash
# Use same settings on different data directory
python scripts/sgd_wizard.py --config configs/my_survey.json --data data/101MEDIA
```

## Configuration Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_dir` | string | Path to directory with thermal/RGB images |
| `output_name` | string | Base name for output files |
| `output_dir` | string | Directory where results will be saved |
| `temp_threshold` | number | Temperature difference (°C) to detect as SGD |
| `min_area` | integer | Minimum plume area in pixels |
| `detector` | string | Detector type: `integrated`, `improved`, `temporal`, `edge_aware` |
| `baseline_method` | string | For improved detector: `upper_quartile`, `median`, `trimmed_mean`, `modal_peak` |
| `use_ml` | boolean | Enable ML-based ocean/land segmentation |
| `ml_model` | string | Path to ML model file (if use_ml is true) |
| `detect_glint` | boolean | Enable sun glint detection and filtering |
| `min_shore_distance` | integer | Minimum distance from shore (pixels) |
| `max_shore_distance` | integer | Maximum distance from shore (pixels) |
| `export_kml` | boolean | Export Google Earth KML files |
| `export_geojson` | boolean | Export GeoJSON files |
| `export_csv` | boolean | Export CSV spreadsheets |

## Tips for Parameter Tuning

### Temperature Threshold
- **0.3-0.5°C**: Sensitive, may include noise
- **0.5-1.0°C**: Balanced (recommended)
- **1.0-2.0°C**: Conservative, only strong SGDs

### Minimum Area
- **25-50 pixels**: Detect small features
- **50-100 pixels**: Balanced (recommended)
- **100-200 pixels**: Only significant plumes

### Detector Type
- **integrated**: Fast, good for most cases
- **improved**: Better baseline calculation, sun glint filtering (recommended)
- **temporal**: Use for video/sequential frames
- **edge_aware**: Use for handling overlapping frames

### Baseline Method (for improved detector)
- **upper_quartile**: Robust when cold plumes dominate (recommended)
- **median**: Traditional approach
- **trimmed_mean**: Good for varied conditions
- **modal_peak**: Best for uniform ocean temperatures
