#!/usr/bin/env python3
"""Find regions with high variation in IRX image"""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

frame_num = 248
base_path = Path("data/100MEDIA")

# Load IRX image
irx_path = base_path / f"IRX_{frame_num:04d}.jpg"
irx_image = np.array(Image.open(irx_path))

print(f"IRX image shape: {irx_image.shape}")

# Convert to grayscale for analysis
if len(irx_image.shape) == 3:
    irx_gray = np.mean(irx_image, axis=2).astype(np.uint8)
else:
    irx_gray = irx_image

# Find regions with high standard deviation
window_size = 80
stride = 20

regions_info = []

for y in range(0, irx_gray.shape[0] - window_size, stride):
    for x in range(0, irx_gray.shape[1] - window_size, stride):
        region = irx_gray[y:y+window_size, x:x+window_size]
        std = region.std()
        mean = region.mean()
        regions_info.append({
            'x': x, 'y': y,
            'std': std,
            'mean': mean,
            'min': region.min(),
            'max': region.max()
        })

# Sort by standard deviation (high variation)
regions_sorted = sorted(regions_info, key=lambda r: r['std'], reverse=True)

print("\nTop 10 regions with highest variation:")
print("-" * 60)
print(f"{'Rank':<5} {'Position':<15} {'Std':<8} {'Mean':<8} {'Min':<5} {'Max':<5}")
print("-" * 60)

for i, region in enumerate(regions_sorted[:10], 1):
    print(f"{i:<5} ({region['x']:3}, {region['y']:3})      "
          f"{region['std']:6.1f}  {region['mean']:6.1f}  "
          f"{region['min']:3}   {region['max']:3}")

# Visualize top regions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Top 10 Variable Regions in IRX Image')

for i, (ax, region) in enumerate(zip(axes.flat, regions_sorted[:10])):
    x, y = region['x'], region['y']
    region_img = irx_image[y:y+window_size, x:x+window_size]
    ax.imshow(region_img)
    ax.set_title(f"#{i+1} ({x},{y})\nStd: {region['std']:.1f}")
    ax.axis('off')

plt.tight_layout()
plt.savefig('irx_variable_regions.png')
print("\nSaved visualization to irx_variable_regions.png")

# Suggest good region pairs
print("\nSuggested region pairs for comparison:")
print("-" * 60)

# Find regions with similar means but different locations
for i in range(min(5, len(regions_sorted))):
    region1 = regions_sorted[i]
    # Find another region with high variation but different location
    for region2 in regions_sorted[i+1:]:
        # Ensure regions are not overlapping
        if abs(region2['x'] - region1['x']) > window_size or \
           abs(region2['y'] - region1['y']) > window_size:
            print(f"Region 1: ({region1['x']:3}, {region1['y']:3}) - "
                  f"Std: {region1['std']:5.1f}, Mean: {region1['mean']:6.1f}")
            print(f"Region 2: ({region2['x']:3}, {region2['y']:3}) - "
                  f"Std: {region2['std']:5.1f}, Mean: {region2['mean']:6.1f}")
            print()
            break

# plt.show()