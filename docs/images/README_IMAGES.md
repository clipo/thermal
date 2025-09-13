# Required Images for Technical Paper

This directory should contain the following images referenced in the TECHNICAL_PAPER.md document. These images should be actual screenshots and photographs from the Rapa Nui SGD surveys.

## Image List and Descriptions

### 1. thermal_rgb_pair.png
**Description**: Side-by-side comparison of RGB and thermal images from the same frame
- Left panel: Original RGB image (4096×3072) from drone
- Right panel: Corresponding thermal image (640×512) 
- Should clearly show the FOV difference and visible SGD plumes

### 2. environmental_diversity.png
**Description**: 2×2 grid showing different coastal environments
- Panel A: Rocky volcanic shoreline
- Panel B: Sandy beach area
- Panel C: Boulder field
- Panel D: Active surf zone with waves

### 3. segmentation_example.png
**Description**: Three-panel image showing segmentation process
- Left: Original RGB coastal image
- Center: Color-coded segmentation (blue=ocean, brown=land, gray=rock, white=wave)
- Right: Final binary ocean mask

### 4. sgd_detection_process.png
**Description**: Four-panel image showing SGD detection steps
- Panel A: Raw thermal image of ocean
- Panel B: Ocean-masked thermal data
- Panel C: Temperature anomaly map
- Panel D: Detected SGD plumes with polygon overlays

### 5. google_earth_sgd_polygons.png
**Description**: Screenshot from Google Earth
- Shows KML file loaded with SGD polygons
- Should include Rapa Nui coastline
- Red/blue polygons indicating SGD locations
- Include scale and north arrow if possible

### 6. sgd_plume_detail.png
**Description**: Close-up view of individual SGD plume
- Left: Thermal image with temperature scale
- Right: RGB image with polygon overlay
- Should show clear temperature anomaly (-2 to -3°C)
- Include scale bar (e.g., 10m)

### 7. training_interface.png
**Description**: Screenshot of the segmentation training GUI
- Show the matplotlib window with image
- Visible labeled points in different colors
- Training data panel on the right
- "Train" and "Save & Continue" buttons visible

### 8. complete_workflow_results.png
**Description**: Four-panel overview of complete workflow
- Panel A: Folder/file browser showing input images
- Panel B: Terminal/console showing processing progress
- Panel C: Output files (KML, JSON, GeoJSON)
- Panel D: Final result in QGIS or Google Earth

## Image Specifications

- **Format**: PNG preferred for quality
- **Resolution**: At least 1920×1080 for full screenshots
- **Color**: RGB color, 8-bit depth minimum
- **Compression**: Lossless PNG compression
- **File size**: Keep under 5MB per image for GitHub

## How to Capture These Images

1. **Thermal-RGB pairs**: Export directly from sgd_viewer.py using the screenshot function
2. **Segmentation examples**: Use the interactive viewer with segmentation overlay enabled
3. **Google Earth**: Load KML output and use Google Earth's save image function
4. **Training interface**: Screenshot during actual training session
5. **Processing terminal**: Capture during actual batch processing run

## Alternative: Placeholder Generation

If actual images are not yet available, consider creating placeholder images with the correct dimensions and labels to indicate what content should go where. This helps with document layout and planning.

## Copyright and Attribution

Ensure all images are either:
- Original work from the SGD detection project
- Properly attributed if from other sources
- Have appropriate permissions for publication

Add watermarks if necessary for protection of unpublished research data.