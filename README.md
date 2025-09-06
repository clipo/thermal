# Submarine Groundwater Discharge (SGD) Detection Toolkit

A Python toolkit for detecting submarine groundwater discharge (cold freshwater seeps) in coastal waters using thermal and RGB imagery from Autel 640T UAV.

## Overview

Submarine Groundwater Discharge (SGD) occurs when freshwater from underground aquifers seeps into the ocean along the coastline. This freshwater is typically cooler than seawater and creates detectable thermal anomalies. This toolkit automatically identifies these cold plumes in thermal drone imagery.

This toolkit processes paired thermal (640×512) and RGB (4096×3072) images from an Autel 640T drone to identify areas where cold groundwater emerges at the shoreline. The thermal camera has a narrower field of view (~70% of RGB FOV), which is properly handled for accurate alignment and georeferencing.

## Key Features

- **Thermal Analysis**: Process Autel 640T thermal images (deciKelvin format)
- **Ocean Segmentation**: ML-based segmentation to isolate ocean from land and waves
- **SGD Detection**: Identify cold freshwater plumes near shorelines
- **Georeferencing**: Extract GPS coordinates for detected SGD locations
- **Aggregate Mapping**: Handle overlapping survey frames with deduplication
- **Interactive Viewers**: Real-time parameter tuning and frame navigation

## Primary Scripts

### 1. `sgd_viewer.py` - Main Production Tool
The primary application for SGD detection and mapping across multiple frames.

```bash
python sgd_viewer.py
```

**Features:**
- Interactive viewer with navigation controls
- Real-time parameter adjustment (temperature threshold, minimum area)
- Aggregate mapping with deduplication (handles 90% frame overlap)
- GPS extraction and georeferencing
- Export SGD locations to GeoJSON for GIS applications
- Mark and save confirmed SGD locations

**Controls:**
- Arrow keys or buttons: Navigate frames
- Sliders: Adjust detection parameters
- Mark SGD: Confirm current detections
- Export Map: Save aggregate results to GeoJSON

### 2. `sgd_detector_integrated.py` - Core Detection System
Standalone detector with multiple operation modes.

```bash
python sgd_detector_integrated.py
```

**Options:**
1. Single frame analysis - Process one frame
2. Batch process - Process multiple frames
3. Interactive parameter tuning - Test parameters

**Features:**
- ML-based ocean/land/wave segmentation
- Thermal-RGB alignment (accounts for 70% FOV)
- Cold plume detection algorithms
- Shoreline proximity analysis

### 3. `segmentation_trainer.py` - ML Training Tool
Interactive tool for creating training data and training the segmentation model.

```bash
python segmentation_trainer.py
```

**Usage:**
1. Click on image to label pixels:
   - Left click: Ocean (blue)
   - Right click: Land (green)
   - Middle click: Rock (gray)
   - Shift+click: Wave (white)
2. Press 't' to train model
3. Press 's' to save model
4. Press space for next image

Creates `segmentation_model.pkl` used by other scripts.

### 4. `test_segmentation.py` - Parameter Testing
Test and visualize segmentation parameters on different images.

```bash
python test_segmentation.py
```

**Features:**
- Interactive sliders for HSV thresholds
- Real-time segmentation preview
- Frame navigation for testing on multiple images
- Side-by-side comparison of original and segmented

## Installation

### Requirements
```bash
pip install numpy matplotlib pillow scikit-image scipy scikit-learn
```

### Directory Structure
```
thermal/
├── data/
│   └── 100MEDIA/
│       ├── MAX_XXXX.JPG    # RGB images
│       └── IRX_XXXX.irg    # Thermal data
├── segmentation_model.pkl   # Trained ML model
└── sgd_aggregate.json      # Persistent SGD locations
```

## Quick Start

1. **Prepare Data**: Place Autel 640T imagery in `data/100MEDIA/`

2. **Train Segmentation Model** (if needed):
   ```bash
   python segmentation_trainer.py
   ```

3. **Run SGD Detection**:
   ```bash
   python sgd_viewer.py
   ```

4. **Export Results**: Click "Export Map" to save GeoJSON

## Technical Details

### Image Alignment
- Thermal FOV is ~70% of RGB FOV (centered)
- Automatic extraction of matching RGB region
- Proper scaling for pixel-perfect alignment

### Temperature Processing
- Raw thermal values in deciKelvin
- Conversion: °C = Raw/10 - 273.15
- Typical ocean: 24-26°C
- SGD plumes: 1-3°C cooler

### ML Segmentation
- Random Forest classifier
- Features: RGB, HSV, LAB color spaces
- Fast inference using vectorized operations
- Handles complex rocky shores and wave foam

### SGD Detection Algorithm
1. Segment ocean from land/rocks
2. Extract ocean temperatures
3. Find cold anomalies near shore
4. Filter by size and temperature threshold
5. Georeference using EXIF GPS data

## Output Formats

### GeoJSON Export
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Point",
      "coordinates": [longitude, latitude]
    },
    "properties": {
      "temperature_anomaly": -2.1,
      "area_m2": 15.3,
      "confidence": 0.85,
      "frame": 248
    }
  }]
}
```

## Tips for Best Results

1. **Segmentation Quality**: Train model on representative images with varied conditions
2. **Temperature Threshold**: Start with 1.0°C, adjust based on conditions
3. **Minimum Area**: 50 pixels works well, increase to reduce false positives
4. **Flight Planning**: Maintain consistent altitude for accurate area calculations
5. **Survey Overlap**: 90% overlap ensures complete coverage

## Troubleshooting

### No Controls Visible
- Ensure matplotlib backend supports interactive widgets
- Try: `export MPLBACKEND=TkAgg` before running

### Segmentation Issues
- Retrain model with more labeled examples
- Adjust HSV thresholds in test_segmentation.py
- Check if segmentation_model.pkl exists

### GPS Not Found
- Verify EXIF data in images
- Check GPS was enabled during flight

## Citation

If you use this toolkit in your research, please cite:
```
SGD Detection Toolkit
https://github.com/clipo/thermal
```

## License

MIT License - See LICENSE file for details

## Contact

For issues and questions, please open an issue on GitHub:
https://github.com/clipo/thermal/issues