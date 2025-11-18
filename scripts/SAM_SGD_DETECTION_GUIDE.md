# SAM-based SGD Detection Test Tool

## Overview

Interactive tool to evaluate whether SAM (Segment Anything Model) can improve SGD detection compared to the current threshold-based method.

## What It Does

Provides a 4-panel interactive interface:

1. **Top Left**: Thermal deviations from baseline (coolwarm colormap)
   - Shows temperature differences from ocean median
   - Blue = colder (potential SGD)
   - Red = warmer
   - Click here to mark potential SGD locations

2. **Bottom Left**: Absolute ocean temperatures (plasma colormap)
   - Shows actual temperature values
   - Only ocean areas visible (land masked out)

3. **Top Right**: SAM Detection Results
   - Shows areas detected by SAM based on your clicks
   - Red overlay indicates detected SGD regions

4. **Bottom Right**: Threshold Detection Results (baseline)
   - Current threshold-based method (0.5°C below median)
   - Green overlay shows detected areas
   - For comparison with SAM method

## How to Use

### Quick Start

```bash
python scripts/test_sam_sgd_detection.py \
    --rgb data/100MEDIA/MAX_0001.JPG \
    --thermal data/100MEDIA/IRX_0001.irg
```

### With Existing Ocean Mask

If you've already run ocean segmentation:

```bash
python scripts/test_sam_sgd_detection.py \
    --rgb data/100MEDIA/MAX_0001.JPG \
    --thermal data/100MEDIA/IRX_0001.irg \
    --ocean-mask sgd_output/ocean_masks/MAX_0001_ocean.npy
```

### Workflow

1. **Launch the tool** - Wait for SAM to load (~5 seconds)

2. **Examine thermal deviations** (top left panel)
   - Look for blue areas (colder than baseline)
   - These are potential SGD locations

3. **Click on cold spots**
   - Left-click on suspicious cold areas
   - Yellow points mark your selections
   - Temperature shown in terminal

4. **Run SAM segmentation**
   - Press 'S' to segment with SAM
   - Top right shows SAM detection
   - Bottom right shows threshold method for comparison

5. **Compare methods**
   - Statistics printed in terminal
   - Pixel counts for each method
   - Temperature ranges
   - Overlap/agreement (IoU)

6. **Iterate**
   - Press 'C' to clear points and try again
   - Press 'Q' to quit

## Controls

| Key | Action |
|-----|--------|
| **Left Click** | Mark potential SGD location (on top-left panel only) |
| **S** | Segment with SAM |
| **C** | Clear all points |
| **Q** | Quit |

## Technical Details

### Dimension Handling

The script handles different resolutions seamlessly:
- RGB images: 4096x3072 pixels (full resolution)
- Thermal images: 640x512 pixels (IRG data)
- Ocean masks: Automatically resized to match thermal
- SAM operates on RGB, results scaled to thermal for analysis
- Visualizations scaled back to RGB for display

### SAM Integration

- SAM processes the RGB image
- Your clicks (in thermal coordinates) are scaled to RGB coordinates
- SAM segments features in RGB space
- Results are scaled back to thermal resolution for temperature analysis
- Only ocean areas are considered (land automatically excluded)
- Temperature filtering applied (must be cooler than ocean mean)

### What It Evaluates

**Threshold Method (Current)**:
- Finds areas cooler than median by fixed threshold (0.5°C)
- Fast and deterministic
- May miss features or include false positives

**SAM Method (Experimental)**:
- Uses AI to segment features you identify
- Can capture complex shapes and boundaries
- Requires user input (clicks)
- May be more accurate for irregular SGD patterns

### Example Output

```
📊 DETECTION COMPARISON:
  SAM Detection:
    Pixels: 1234
    Mean temp: 21.3°C
    Min temp: 20.8°C
  Threshold Detection:
    Pixels: 2345
    Mean temp: 21.5°C
    Min temp: 20.9°C
  Overlap: 987 pixels
  IoU (agreement): 42.3%
```

## Evaluation Questions

Use this tool to answer:

1. **Does SAM find features the threshold method misses?**
   - Look for areas in red (SAM) but not green (threshold)

2. **Does SAM reduce false positives?**
   - Check if SAM excludes areas incorrectly flagged by threshold

3. **Are the detected areas thermally consistent?**
   - Do SAM regions have colder mean/min temps?

4. **Is the manual effort worth it?**
   - How many clicks needed for good results?
   - Is it practical for 1000+ images?

## Next Steps

Based on your evaluation:

- **If SAM is better**: Consider training an automatic SGD detector using SAM-labeled data
- **If threshold is sufficient**: Stick with current fast method
- **If mixed results**: Maybe combine both (SAM for edge cases, threshold for bulk)

## Troubleshooting

**Issue**: GUI doesn't appear
- Check if matplotlib backend is configured
- Try: `export MPLBACKEND=TkAgg` before running

**Issue**: SAM takes too long to load
- Normal on first run (~5-10 seconds)
- Uses Apple Silicon GPU (MPS) if available

**Issue**: Clicks don't register
- Make sure you're clicking on the top-left panel (thermal deviations)
- Clicks on land are automatically rejected

**Issue**: No ocean mask
- Script creates a basic mask using blue channel threshold
- For better results, provide --ocean-mask from existing segmentation
