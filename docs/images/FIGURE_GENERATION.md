# Figure Generation for SGD Detection Technical Paper

This document explains how to generate the figures needed for the technical paper using the `generate_paper_figures.py` script.

## Quick Start

Generate all figures from your survey data:

```bash
python generate_paper_figures.py \
    --data "/path/to/thermal/data" \
    --output docs/images
```

## Script Overview

The `generate_paper_figures.py` script creates publication-quality figures for the SGD detection technical paper. It generates:

1. **thermal_rgb_pair.png** - Side-by-side comparison of thermal and RGB images
2. **environmental_diversity.png** - 2×2 grid showing different coastal environments  
3. **segmentation_example.png** - Three-panel segmentation process visualization
4. **sgd_detection_process.png** - Four-panel SGD detection pipeline
5. **sgd_plume_detail.png** - Close-up view of individual SGD plume

## Usage

### Basic Usage

```bash
python generate_paper_figures.py --data /path/to/data --output docs/images
```

### Specify Particular Images

```bash
python generate_paper_figures.py \
    --data /path/to/data \
    --thermal /path/to/thermal.JPG \
    --rgb /path/to/rgb.JPG \
    --output docs/images
```

### Parameters

- `--data`: Path to directory containing thermal/RGB images (required)
- `--output`: Output directory for figures (default: `docs/images`)
- `--thermal`: Specific thermal image for examples (optional)
- `--rgb`: Specific RGB image for examples (optional)

## Image Requirements

### For Best Results

1. **Thermal Images**: 
   - Format: JPEG with thermal data
   - Resolution: 640×512 (standard FLIR format)
   - Content: Clear temperature variations visible

2. **RGB Images**:
   - Format: Standard JPEG
   - Resolution: 4096×3072 or similar
   - Content: Corresponding visual imagery

3. **Paired Images**:
   - Thermal and RGB from same location/time
   - Clear SGD plumes visible in thermal
   - Good lighting conditions in RGB

## Customizing Figures

### Temperature Ranges

Edit the script to adjust temperature display ranges:

```python
# In generate_thermal_rgb_pair() and other functions
im = ax2.imshow(thermal_celsius, cmap='jet', vmin=18, vmax=25)  # Adjust vmin/vmax
```

### Color Maps

Change the colormap for thermal display:

```python
im = ax2.imshow(thermal_celsius, cmap='jet')  # Options: 'jet', 'viridis', 'plasma', 'RdBu_r'
```

### SGD Detection Threshold

Adjust the temperature anomaly threshold:

```python
sgd_threshold = -1.0  # 1°C cooler than mean (make more negative for stricter detection)
```

## Troubleshooting

### No Thermal Images Found

The script looks for files matching these patterns:
- `DJI_*_T.JPG` (thermal)
- `DJI_*_W.JPG` (RGB)
- `MAX_*.JPG` (alternate format)

If your files have different naming, specify them directly with `--thermal` and `--rgb`.

### Image Format Issues

The script automatically handles:
- DeciKelvin format (FLIR standard)
- Regular JPEG thermal images
- RGB thermal representations

If you encounter format issues, the script will normalize the data to a reasonable temperature range (18-25°C).

### Memory Issues

For large datasets, process one directory at a time rather than using `--search` with the parent directory.

## Output Files

All figures are saved as PNG files with:
- Resolution: 300 DPI
- Format: PNG with lossless compression
- Size: Optimized for publication (<5MB per image)

## Integration with Technical Paper

The generated figures are referenced in `TECHNICAL_PAPER.md`. After generating:

1. Review the figures in your output directory
2. Ensure they clearly show the described features
3. Replace with better examples if needed
4. The paper automatically references these image paths

## Example Workflow

```bash
# 1. Generate figures from a specific flight
python generate_paper_figures.py \
    --data "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA" \
    --output docs/images

# 2. Review generated figures
open docs/images/

# 3. If needed, regenerate with specific images that show clearer SGD plumes
python generate_paper_figures.py \
    --data "/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/103MEDIA" \
    --thermal "/path/to/best_thermal.JPG" \
    --rgb "/path/to/best_rgb.JPG" \
    --output docs/images

# 4. The figures are now ready for the technical paper
```

## Manual Image Capture

If you prefer to capture images manually:

1. **From sgd_viewer.py**: Use the screenshot function (press 's' key)
2. **From Google Earth**: Load KML file and use File → Save → Save Image
3. **From QGIS**: Load GeoJSON and use Project → Import/Export → Export Map to Image
4. **From training interface**: Take screenshot during active training session

## Next Steps

After generating figures:

1. Review all images for quality and clarity
2. Ensure SGD plumes are clearly visible
3. Check that captions in TECHNICAL_PAPER.md match the images
4. Consider adding scale bars or annotations if needed
5. Compress images if file sizes are too large for GitHub

## Support

For issues or questions about figure generation, refer to:
- Main README.md for system overview
- TECHNICAL_PAPER.md for figure requirements
- sgd_viewer.py for interactive visualization options