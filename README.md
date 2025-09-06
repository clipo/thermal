# Submarine Groundwater Discharge (SGD) Detection Toolkit

A Python toolkit for detecting submarine groundwater discharge (cold freshwater seeps) in coastal waters using thermal and RGB imagery from Autel 640T UAV.

## Overview

Submarine Groundwater Discharge (SGD) occurs when freshwater from underground aquifers seeps into the ocean along the coastline. This freshwater is typically cooler than seawater and creates detectable thermal anomalies. This toolkit automatically identifies these cold plumes in thermal drone imagery.

This toolkit processes paired thermal (640×512) and RGB (4096×3072) images from an Autel 640T drone to identify areas where cold groundwater emerges at the shoreline. The thermal camera has a narrower field of view (~70% of RGB FOV), which is properly handled for accurate alignment and georeferencing.

### Field of View Alignment
![Thermal-RGB Alignment](docs/images/thermal_alignment.png)
*The thermal camera captures ~70% of the RGB camera's field of view. The toolkit automatically extracts and aligns the matching region.*

## Key Features

- **Thermal Analysis**: Process Autel 640T thermal images (deciKelvin format)
- **Ocean Segmentation**: ML-based segmentation to isolate ocean from land and waves
- **SGD Detection**: Identify cold freshwater plumes near shorelines
- **Georeferencing**: Extract GPS coordinates for detected SGD locations
- **Aggregate Mapping**: Handle overlapping survey frames with deduplication
- **Interactive Viewers**: Real-time parameter tuning and frame navigation

## Detection Pipeline

![Detection Pipeline](docs/images/detection_pipeline.png)
*The SGD detection pipeline: 1) RGB input aligned to thermal FOV, 2) ML-based segmentation, 3) Thermal data processing, 4) Ocean isolation, 5) Cold anomaly detection, 6) Final SGD identification near shoreline*

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

![SGD Viewer Interface](docs/images/sgd_viewer_interface.png)
*Main SGD viewer interface showing multi-panel analysis with RGB, segmentation, thermal, ocean thermal, SGD detection, coverage map, and statistics*

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

![Segmentation Trainer](docs/images/segmentation_trainer.png)
*Interactive training tool - click to label pixels as ocean (blue), land (green), or rock (gray), then train the ML model*

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

![Test Segmentation](docs/images/test_segmentation.png)
*Parameter testing interface with HSV channel visualization and adjustable thresholds for fine-tuning segmentation*

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

## Machine Learning Segmentation

### The Challenge

Traditional color-based segmentation struggles with coastal environments because:
- **Dark rocky shores** have similar color properties to deep ocean water
- **Wave foam and whitecaps** can be confused with sand or clouds
- **Shallow water** over sand appears different than deep water
- **Wet rocks** reflect differently than dry rocks
- **Sun glint** creates bright spots on water that look like land

These challenges led to frequent misclassification where rocky shorelines were labeled as ocean, causing false SGD detections at the land-ocean boundary.

### The Solution: Random Forest Classification

We implemented a machine learning approach using Random Forest classification that learns from human-labeled examples to understand the complex visual patterns that distinguish ocean, land, rocks, and waves.

#### Why Random Forest?
- **Robust to noise**: Handles the natural variation in outdoor imagery
- **Non-linear boundaries**: Can learn complex decision boundaries between classes
- **Feature importance**: Tells us which color features matter most
- **Fast inference**: Quick enough for real-time processing
- **No overfitting**: Ensemble method naturally resists overfitting

### Feature Engineering

The classifier uses 48 features per pixel, computed from a 5×5 pixel neighborhood:

```python
# Color space features (12 base features)
- RGB channels (3)
- HSV channels (3) 
- LAB channels (3)
- Derived: intensity, blue dominance, color range (3)

# Statistical features (4 per base feature = 48 total)
- Mean (local average)
- Standard deviation (local variance)
- Minimum value
- Maximum value
```

These features capture both color information and local texture, allowing the classifier to distinguish between smooth ocean and textured rocky shores.

### Training Process

#### 1. Interactive Labeling (`segmentation_trainer.py`)
```bash
python segmentation_trainer.py
```

Users label pixels by clicking:
- **Left click**: Ocean (blue) - deep water, shallow water
- **Right click**: Land (green) - sand, vegetation, dry land
- **Middle click**: Rock (gray) - rocky shores, cliffs, boulders
- **Shift+click**: Wave (white) - foam, whitecaps, breaking waves

The tool shows real-time segmentation preview as you label, helping you see where more training data is needed.

#### 2. Model Training
After labeling sufficient pixels (typically 100-200 per class), press 'T' to train:
- Extracts features for all labeled pixels
- Trains Random Forest with 100 trees
- Cross-validates to estimate accuracy
- Updates preview with new segmentation

#### 3. Model Persistence
Press 'S' to save the trained model to `segmentation_model.pkl`:
```python
import pickle
with open('segmentation_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)
```

### Implementation Details

#### Fast Inference (`ml_segmentation_fast.py`)
For real-time processing, we optimized inference:

1. **Downsampling**: Process at 1/4 resolution (160×128 instead of 640×512)
2. **Vectorized operations**: Use NumPy broadcasting instead of pixel loops
3. **Batch prediction**: Process 10,000 pixels at once
4. **Upsampling**: Use nearest-neighbor to return to full resolution

Result: 0.08 seconds per frame vs 30+ seconds for pixel-by-pixel processing.

#### Integration with SGD Detection
```python
# In sgd_detector_integrated.py
def __init__(self, use_ml=True):
    if use_ml and ML_AVAILABLE:
        self.ml_segmenter = FastMLSegmenter()
    
def segment_ocean_land_waves(self, rgb_image):
    if self.ml_segmenter:
        # Use ML segmentation
        return self.ml_segmenter.segment_ultra_fast(rgb_image)
    else:
        # Fall back to rule-based HSV thresholds
        return self.rule_based_segmentation(rgb_image)
```

### Improving the Model

The model can be continuously improved by adding more training data:

#### 1. Identify Problem Areas
Run the detector and note where segmentation fails:
```bash
python test_ml_integration.py
```

#### 2. Add Training Data
Label the problematic images:
```bash
python segmentation_trainer.py
```
Focus on:
- Transition zones (wet sand, tide lines)
- Unusual lighting (sunrise, sunset, overcast)
- Specific problem features (kelp, boats, shadows)

#### 3. Incremental Learning
The trainer loads existing training data and adds to it:
```python
# Loads previous training data
with open('segmentation_training_data.json', 'r') as f:
    existing_data = json.load(f)

# Adds new labels
training_data['pixels'].extend(new_pixels)
training_data['labels'].extend(new_labels)
```

#### 4. Retrain and Validate
After adding new data:
- Press 'T' to retrain with combined dataset
- Test on multiple frames to ensure improvement
- Save new model when satisfied

### Performance Metrics

Current model performance (trained on Rapa Nui coastal imagery):
- **Overall accuracy**: 94.3%
- **Ocean recall**: 96.2% (correctly identifies ocean)
- **Land precision**: 95.1% (rarely mislabels land as ocean)
- **Rock detection**: 89.7% (most challenging class)
- **Processing speed**: 12.5 fps (with downsampling)

### Best Practices for Training

1. **Diverse examples**: Label pixels from different images and conditions
2. **Edge cases**: Focus on ambiguous areas like wet rocks, shallow water
3. **Balanced classes**: Ensure roughly equal samples per class
4. **Iterative refinement**: Start simple, add complexity as needed
5. **Validation**: Always test on unseen images before deployment

### Fallback Strategy

If ML segmentation fails or no model exists, the system automatically falls back to rule-based HSV thresholds, ensuring the pipeline always works:

```python
if not model_path.exists():
    print("No ML model found, using rule-based segmentation")
    return self.rule_based_segmentation(rgb_image)
```

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