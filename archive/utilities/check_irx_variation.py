#!/usr/bin/env python3
"""Check variation in IRX image regions"""

import numpy as np
from pathlib import Path
from PIL import Image

frame_num = 248
base_path = Path("data/100MEDIA")

# Load IRX image
irx_path = base_path / f"IRX_{frame_num:04d}.jpg"
irx_image = np.array(Image.open(irx_path))

print(f"IRX image shape: {irx_image.shape}")
print(f"IRX dtype: {irx_image.dtype}")

# Check if color or grayscale
if len(irx_image.shape) == 3:
    print("IRX is color (RGB)")
    # Convert to grayscale for analysis
    irx_gray = np.mean(irx_image, axis=2).astype(np.uint8)
else:
    print("IRX is grayscale")
    irx_gray = irx_image

# Check variation in different regions
regions = [
    ("Top-left (0:100, 0:100)", irx_gray[0:100, 0:100]),
    ("Ocean (200:280, 100:180)", irx_gray[200:280, 100:180]),
    ("Land (100:180, 450:530)", irx_gray[100:180, 450:530]),
    ("Center (256:336, 320:400)", irx_gray[256:336, 320:400])
]

print("\nRegion statistics:")
print("-" * 50)
for name, region in regions:
    print(f"{name:30s}: min={region.min():3d}, max={region.max():3d}, "
          f"mean={region.mean():6.1f}, std={region.std():5.1f}")

# Show overall statistics
print("\nOverall IRX statistics:")
print(f"  Min: {irx_gray.min()}")
print(f"  Max: {irx_gray.max()}")
print(f"  Mean: {irx_gray.mean():.1f}")
print(f"  Std: {irx_gray.std():.1f}")

# Check if regions are uniform (low std means uniform)
for name, region in regions:
    if region.std() < 5:
        print(f"\nWARNING: {name} has very low variation (std={region.std():.1f})")
        print(f"  This region appears nearly uniform")