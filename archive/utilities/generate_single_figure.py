#!/usr/bin/env python3
"""
Quick script to generate a single thermal-RGB comparison figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import sys

def generate_thermal_rgb_pair(thermal_path: Path, rgb_path: Path, output_path: Path):
    """Generate side-by-side thermal and RGB comparison."""
    
    # Set publication quality
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Load and display RGB
    if rgb_path.exists():
        try:
            rgb_img = np.array(Image.open(rgb_path))
            ax1.imshow(rgb_img)
            ax1.set_title(f'RGB Image ({rgb_img.shape[1]}×{rgb_img.shape[0]})')
            ax1.axis('off')
            
            # Add FOV rectangle showing thermal coverage
            h, w = rgb_img.shape[:2]
            thermal_fov_rect = patches.Rectangle((w*0.3, h*0.3), w*0.4, h*0.4,
                                                linewidth=2, edgecolor='yellow',
                                                facecolor='none', linestyle='--')
            ax1.add_patch(thermal_fov_rect)
            ax1.text(w*0.5, h*0.25, 'Thermal FOV', color='yellow', 
                    ha='center', fontsize=10, weight='bold')
        except Exception as e:
            print(f"Error loading RGB: {e}")
            ax1.text(0.5, 0.5, 'RGB Image\n(Error loading)', 
                    ha='center', va='center')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
    
    # Load and display thermal
    if thermal_path.exists():
        try:
            thermal_img = np.array(Image.open(thermal_path))
            
            # Handle RGB thermal images
            if len(thermal_img.shape) == 3:
                thermal_img = np.mean(thermal_img, axis=2)
            
            # Simple normalization for display
            thermal_display = thermal_img.astype(np.float32)
            thermal_display = (thermal_display - thermal_display.min()) / (thermal_display.max() - thermal_display.min())
            
            im = ax2.imshow(thermal_display, cmap='jet')
            ax2.set_title(f'Thermal Image (640×512)')
            ax2.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('Relative Temperature', rotation=270, labelpad=20)
        except Exception as e:
            print(f"Error loading thermal: {e}")
            ax2.text(0.5, 0.5, 'Thermal Image\n(Error loading)', 
                    ha='center', va='center')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
    
    plt.suptitle('Thermal-RGB Image Pair from UAV Survey (Nadir View)', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_single_figure.py <thermal_path> <rgb_path> <output_path>")
        sys.exit(1)
    
    thermal = Path(sys.argv[1])
    rgb = Path(sys.argv[2])
    output = Path(sys.argv[3])
    
    generate_thermal_rgb_pair(thermal, rgb, output)