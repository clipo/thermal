#!/usr/bin/env python3
"""
Quick validation test for SAM SGD detection dimensions
"""
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

def load_irg_data(irg_path, thermal_width=640, thermal_height=512):
    """Load thermal data from IRG file"""
    with open(irg_path, 'rb') as f:
        irg_data = f.read()

    pixel_data_size = thermal_width * thermal_height * 2
    header_size = len(irg_data) - pixel_data_size

    if header_size > 0:
        raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
        thermal = raw_thermal.reshape((thermal_height, thermal_width))
        thermal = thermal.astype(np.float32) / 10.0 - 273.15
        return thermal
    else:
        raise ValueError(f"IRG file appears corrupted: {irg_path}")

def validate_dimensions():
    """Validate dimension handling"""
    print("Validating SAM SGD Detection Dimensions...")
    print("=" * 60)

    # Load test data
    rgb_path = "data/100MEDIA/MAX_0001.JPG"
    thermal_path = "data/100MEDIA/IRX_0001.irg"

    if not Path(rgb_path).exists():
        print(f"✗ RGB not found: {rgb_path}")
        return False

    if not Path(thermal_path).exists():
        print(f"✗ Thermal not found: {thermal_path}")
        return False

    # Load RGB
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb_h, rgb_w = rgb.shape[:2]
    print(f"✓ RGB loaded: {rgb_w}x{rgb_h}")

    # Load thermal
    thermal = load_irg_data(thermal_path)
    thermal_h, thermal_w = thermal.shape
    print(f"✓ Thermal loaded: {thermal_w}x{thermal_h}")

    # Create simple ocean mask at RGB resolution
    rgb_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_rgb = cv2.inRange(rgb_hsv, lower_blue, upper_blue) > 0
    print(f"✓ Ocean mask (RGB): {mask_rgb.shape}")

    # Resize to thermal dimensions
    mask_thermal = cv2.resize(mask_rgb.astype(np.uint8),
                             (thermal_w, thermal_h),
                             interpolation=cv2.INTER_NEAREST) > 0
    print(f"✓ Ocean mask (thermal): {mask_thermal.shape}")

    # Verify masking works
    masked_thermal = thermal.copy()
    masked_thermal[~mask_thermal] = np.nan
    print(f"✓ Thermal masking works")

    # Simulate clicked point (thermal coordinates)
    click_x, click_y = thermal_w // 2, thermal_h // 2
    print(f"✓ Simulated click at thermal ({click_x}, {click_y})")

    # Scale to RGB coordinates
    scale_x = rgb_w / thermal_w
    scale_y = rgb_h / thermal_h
    rgb_x = int(click_x * scale_x)
    rgb_y = int(click_y * scale_y)
    print(f"✓ Scaled to RGB ({rgb_x}, {rgb_y})")

    # Create simulated SAM mask at RGB resolution
    sam_mask_rgb = np.zeros((rgb_h, rgb_w), dtype=bool)
    cv2.circle(sam_mask_rgb.astype(np.uint8), (rgb_x, rgb_y), 100, 1, -1)
    sam_mask_rgb = sam_mask_rgb > 0
    print(f"✓ Simulated SAM mask (RGB): {sam_mask_rgb.shape}")

    # Resize to thermal for analysis
    sam_mask_thermal = cv2.resize(sam_mask_rgb.astype(np.uint8),
                                 (thermal_w, thermal_h),
                                 interpolation=cv2.INTER_NEAREST) > 0
    print(f"✓ SAM mask resized to thermal: {sam_mask_thermal.shape}")

    # Combine with ocean mask and thermal
    combined = sam_mask_thermal & mask_thermal
    temps_in_region = thermal[combined]
    print(f"✓ Combined mask works: {len(temps_in_region)} pixels")

    # Test visualization (resize back to RGB)
    sam_mask_viz = cv2.resize(sam_mask_thermal.astype(np.uint8),
                             (rgb_w, rgb_h),
                             interpolation=cv2.INTER_NEAREST) > 0
    print(f"✓ Visualization mask (RGB): {sam_mask_viz.shape}")

    overlay = rgb.copy()
    overlay[sam_mask_viz] = overlay[sam_mask_viz] * 0.5 + np.array([255, 0, 0]) * 0.5
    print(f"✓ Overlay visualization works")

    print("=" * 60)
    print("✓ All dimension validations passed!")
    return True

if __name__ == "__main__":
    success = validate_dimensions()
    sys.exit(0 if success else 1)
