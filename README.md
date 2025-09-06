# 🌊 Thermal Image Analysis for Submarine Groundwater Discharge (SGD) Detection

A comprehensive Python toolkit for detecting and mapping submarine groundwater discharge using thermal drone imagery from the Autel 640T UAV.

## 📋 Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Technical Details](#technical-details)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

Submarine Groundwater Discharge (SGD) occurs when freshwater from underground aquifers seeps into the ocean along the coastline. This freshwater is typically cooler than seawater and creates detectable thermal anomalies. This toolkit automatically identifies these cold plumes in thermal drone imagery.

### Key Capabilities
- 🔍 **Detects cold freshwater plumes** emerging at the shoreline
- 🗺️ **Maps SGD locations** for GIS integration
- 📊 **Quantifies discharge areas** and temperature anomalies
- 🌊 **Excludes wave zones** to reduce false positives
- 📸 **Handles dual camera systems** (RGB + Thermal)

## 🔬 How It Works

### 1. **Data Acquisition**
The Autel 640T drone captures synchronized image pairs:
- **RGB Camera**: 4096×3072 pixels (wide field of view)
- **Thermal Camera**: 640×512 pixels (narrower FOV, centered)

### 2. **Image Alignment**
The thermal camera's field of view is a subset of the RGB image. The toolkit automatically:
- Extracts the RGB region corresponding to thermal coverage
- Resizes to match thermal dimensions (640×512)
- Ensures perfect pixel alignment

### 3. **Ocean/Land Segmentation**
Using the RGB image's color information:
- **Ocean**: Blue hues (HSV: 180-250°)
- **Land**: Green/brown colors (HSV: 40-150°)
- **Waves**: White foam (high brightness, low saturation)

### 4. **SGD Detection Algorithm**
The system identifies SGD through multiple criteria:
1. **Temperature Anomaly**: Areas 1-3°C cooler than ocean median
2. **Proximity to Shore**: Must be within 5 pixels of shoreline
3. **Plume Characteristics**: Minimum area, elongated shape
4. **Persistence**: Consistent across multiple frames

### 5. **Validation & Export**
- Confidence scoring based on temperature differential and distance from shore
- GeoJSON export for GIS mapping
- Statistical analysis and visualization

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/clipo/thermal.git
cd thermal

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.7+
- NumPy, Matplotlib, Pillow, SciPy, scikit-image

## 🚀 Quick Start

### 1. Prepare Your Data
Organize drone data in this structure:
```
thermal/
├── data/
│   └── 100MEDIA/
│       ├── IRX_0001.irg   # Thermal raw (deciKelvin)
│       ├── IRX_0001.jpg   # Thermal preview
│       ├── IRX_0001.TIFF  # Thermal TIFF
│       └── MAX_0001.JPG   # RGB image
```

### 2. Run SGD Detection
```bash
python sgd_detector_integrated.py
```

Choose from three modes:
- **1** - Analyze single frame
- **2** - Batch process (creates GIS output)
- **3** - Interactive viewer

### 3. Interactive Analysis (Recommended)
```bash
python sgd_detector_integrated.py
# Choose option 3
```

Controls:
- **Arrow Keys**: Navigate frames (← →)
- **Sliders**: Adjust detection parameters
- **S Key**: Save current visualization

## 📖 Detailed Usage

### Main Analysis Pipeline

#### **sgd_detector_integrated.py** - Primary SGD Detection Tool
The all-in-one solution with built-in alignment:
```python
from sgd_detector_integrated import IntegratedSGDDetector

detector = IntegratedSGDDetector(
    temp_threshold=1.0,  # °C below ocean median
    min_area=50          # Minimum plume size (pixels)
)

result = detector.process_frame(248, visualize=True)
```

### Supporting Tools

#### **thermal_viewer.py** - Explore Thermal Data
Interactive viewer for understanding thermal patterns:
```bash
python thermal_viewer.py
```
- Navigate frames with slider
- Compare raw values vs visualization
- Analyze temperature distributions

#### **ocean_thermal_analyzer.py** - Ocean Segmentation
Specialized ocean/land separation:
```bash
python ocean_thermal_analyzer.py
```
- Multiple segmentation methods
- Variance-based detection
- Thermal gradient analysis

#### **rgb_ocean_segmenter.py** - RGB-Based Segmentation
Advanced color-space segmentation:
```bash
python rgb_ocean_segmenter.py
```
- HSV and LAB color space analysis
- Wave zone detection
- Confidence mapping

#### **verify_units_nogui.py** - Temperature Verification
Confirms temperature units and conversions:
```bash
python verify_units_nogui.py
```
Output confirms:
- Raw values in deciKelvin (K × 10)
- Proper hot/cold mapping

## 🔧 Technical Details

### Temperature Data Format
- **Raw Values**: DeciKelvin (Kelvin × 10)
- **Conversion**: `T(°C) = Raw/10 - 273.15`
- **Example**: Raw value 2931 = 293.1K = 20.0°C

### Detection Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| temp_threshold | 1.0°C | 0.5-3.0 | Temperature below ocean median |
| min_area | 50 px | 10-200 | Minimum plume size |
| shore_distance | 5 px | 1-10 | Max distance from shoreline |

### Color Segmentation Ranges (HSV)

| Feature | Hue | Saturation | Value |
|---------|-----|------------|-------|
| Ocean | 180-250° | 20-255 | 20-200 |
| Land | 40-150° | 15-255 | 10-255 |
| Waves | Any | 0-30 | 180-255 |

### Alignment Parameters
For Autel 640T (can be adjusted in code):
- **Scale**: 6.4× (horizontal), 6.0× (vertical)
- **Offset**: Centered (0, 0)

## 📁 Output Files

### Batch Processing Outputs
```
sgd_output/
├── sgd_frame_0001.png      # Visualization for each frame
├── sgd_frame_0002.png
├── sgd_summary.json        # Detection statistics
└── sgd_detections.geojson # GIS-ready locations
```

### JSON Summary Format
```json
{
  "frames_processed": 50,
  "frames_with_sgd": 12,
  "total_plumes": 23,
  "frame_details": [
    {
      "frame": 1,
      "num_plumes": 2,
      "characteristics": {
        "temp_anomaly": -2.3,
        "area_m2": 15.4
      }
    }
  ]
}
```

### GeoJSON Format
Compatible with QGIS, ArcGIS, and web mapping:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "frame": 248,
        "area_m2": 15.4,
        "temp_anomaly": -2.3,
        "confidence": 0.85
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      }
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

**1. No SGD detected:**
- Adjust temperature threshold (try 0.5-2.0°C)
- Reduce minimum area requirement
- Check if scene has actual SGD

**2. Too many false positives:**
- Increase temperature threshold
- Increase minimum area
- Ensure wave zones are excluded

**3. Poor ocean/land segmentation:**
- Check RGB image quality
- Adjust color thresholds in code
- Try different segmentation methods

**4. Alignment issues:**
- Run `python image_aligner.py` to calibrate
- Verify thermal FOV is centered in RGB

### Data Requirements
- Paired RGB-thermal images with matching frame numbers
- Consistent file naming (IRX_XXXX, MAX_XXXX)
- Good contrast between ocean and land in RGB

## 📊 Scientific Background

SGD characteristics in thermal imagery:
- **Temperature**: 1-5°C cooler than ambient seawater
- **Shape**: Elongated plumes extending from shore
- **Location**: Near-shore, often at specific geological features
- **Temporal**: May vary with tides and groundwater levels

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

MIT License - See LICENSE file for details

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [https://github.com/clipo/thermal/issues](https://github.com/clipo/thermal/issues)

## 🙏 Acknowledgments

Developed with assistance from Claude AI for thermal image analysis and SGD detection algorithms.

---

*For detailed API documentation and advanced usage, see the docstrings in individual Python files.*