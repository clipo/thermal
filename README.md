# Thermal Image Analysis for Submarine Groundwater Discharge (SGD) Detection

This repository contains Python tools for analyzing thermal drone imagery from an Autel 640T UAV to detect and map submarine groundwater discharge (SGD) along coastlines.

## Overview

The Autel 640T drone captures both RGB (4096×3072) and thermal (640×512) imagery. This toolkit processes these paired images to:
- Segment ocean from land using RGB color information
- Detect cold freshwater plumes indicating SGD
- Map and quantify SGD locations for GIS integration

## Key Features

- **Automatic RGB-thermal alignment** - Handles different fields of view
- **Advanced ocean/land segmentation** - Uses RGB color space analysis
- **SGD plume detection** - Identifies cold anomalies near shoreline
- **Wave zone exclusion** - Removes noisy surf zone from analysis  
- **GIS export** - Outputs GeoJSON for mapping applications
- **Interactive visualization** - Real-time parameter tuning

## Installation

```bash
# Clone repository
git clone https://github.com/clipo/thermal.git
cd thermal

# Install requirements
pip install numpy matplotlib pillow scipy scikit-image
```

## Data Structure

Place your drone data in the `data/100MEDIA/` directory:
```
data/
└── 100MEDIA/
    ├── IRX_0001.irg  # Thermal raw data
    ├── IRX_0001.jpg  # Thermal preview
    ├── IRX_0001.TIFF # Thermal TIFF
    └── MAX_0001.JPG  # RGB image
```

## Main Scripts

### 1. SGD Detection (Integrated)
```bash
python sgd_detector_integrated.py
```
Options:
- **Option 1**: Single frame analysis
- **Option 2**: Batch processing with GIS export
- **Option 3**: Interactive viewer with navigation controls

### 2. Basic Thermal Analysis
```bash
python thermal_viewer.py          # Interactive thermal data viewer
python thermal_analysis_report.py # Generate analysis report
python verify_units_nogui.py      # Verify temperature units
```

### 3. Ocean Segmentation Tools
```bash
python ocean_thermal_analyzer.py  # Ocean/land segmentation
python rgb_ocean_segmenter.py     # RGB-based segmentation
python batch_ocean_processor.py   # Batch ocean processing
```

### 4. Image Alignment
```bash
python image_aligner.py           # Calibrate RGB-thermal alignment
```

## Quick Start

1. **Analyze a single frame for SGD:**
```bash
python sgd_detector_integrated.py
# Choose option 1
# Enter frame number (e.g., 248)
```

2. **Process multiple frames for GIS:**
```bash
python sgd_detector_integrated.py
# Choose option 2
# Results saved to sgd_output/
```

3. **Interactive exploration:**
```bash
python sgd_detector_integrated.py
# Choose option 3
# Use arrow keys to navigate frames
# Adjust temperature threshold and minimum area sliders
```

## Interactive Viewer Controls

- **Navigation**: Arrow keys (←→) or Previous/Next buttons
- **Jump**: Home/End keys or First/Last buttons
- **Save**: S key or Save Fig button
- **Parameters**: Adjust temperature threshold and minimum plume area with sliders

## Temperature Data

- Raw thermal values are in **deciKelvin** (Kelvin × 10)
- Conversion formula: `Temperature(°C) = Raw/10 - 273.15`
- Typical ocean temperatures: 15-25°C
- SGD typically 1-3°C cooler than ambient ocean

## SGD Detection Parameters

- **Temperature Threshold**: Degrees below ocean median to flag as potential SGD
- **Minimum Area**: Minimum plume size in pixels to exclude noise
- **Shore Distance**: Maximum distance from shoreline for valid SGD

## Output Files

- `sgd_output/sgd_frame_XXXX.png` - Visualization for each frame
- `sgd_output/sgd_summary.json` - Detection statistics
- `sgd_output/sgd_detections.geojson` - GIS-ready SGD locations

## Citation

If you use this code in your research, please cite:
```
[Your citation information here]
```

## License

[Specify your license]

## Contact

[Your contact information]