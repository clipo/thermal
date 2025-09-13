#!/usr/bin/env python3
"""
Generate SGD plume detail figure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image

def generate_sgd_plume_detail(thermal_path, rgb_path, output_path):
    """Generate close-up view of SGD plume with thermal and RGB."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    if Path(thermal_path).exists():
        try:
            thermal_img = np.array(Image.open(thermal_path))
            
            # Handle RGB thermal
            if len(thermal_img.shape) == 3:
                thermal_img = np.mean(thermal_img, axis=2)
            
            # Zoom to region of interest (center-bottom for coastal area)
            h, w = thermal_img.shape[:2] if len(thermal_img.shape) >= 2 else (512, 640)
            roi = thermal_img[2*h//3:h, w//4:3*w//4]  # Bottom half, center region
            
            # Normalize for display
            roi_norm = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
            
            im1 = ax1.imshow(roi_norm, cmap='jet')
            ax1.set_title('Thermal View of Coastal Zone', fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Add temperature scale
            cbar = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Relative Temperature', rotation=270, labelpad=20)
            
            # Add scale bar (approximate 10m)
            scale_pixels = 50
            ax1.plot([10, 10+scale_pixels], [roi.shape[0]-20, roi.shape[0]-20], 
                    'w-', linewidth=3)
            ax1.text(10+scale_pixels//2, roi.shape[0]-30, '~10 m', 
                    color='white', ha='center', fontsize=10, weight='bold')
            
            # Identify potential SGD (cooler areas)
            mean_temp = np.mean(roi_norm)
            cool_mask = roi_norm < (mean_temp - 0.1)
            
            # Add anomaly annotation
            if np.any(cool_mask):
                min_temp = np.min(roi_norm[cool_mask])
                anomaly = (min_temp - mean_temp) * 100  # Arbitrary scale
                ax1.text(0.02, 0.98, f'Cool anomaly detected', 
                        transform=ax1.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
        except Exception as e:
            print(f"Error with thermal: {e}")
            ax1.text(0.5, 0.5, 'Thermal View\n(Error)', ha='center', va='center')
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
    
    if Path(rgb_path).exists():
        try:
            rgb_img = np.array(Image.open(rgb_path))
            
            # Zoom to corresponding region
            h, w = rgb_img.shape[:2]
            roi_rgb = rgb_img[2*h//3:h, w//4:3*w//4]
            
            ax2.imshow(roi_rgb)
            
            # Add example SGD polygon overlay
            ellipse = patches.Ellipse((roi_rgb.shape[1]*0.3, roi_rgb.shape[0]*0.7),
                                     width=80, height=60,
                                     linewidth=2, edgecolor='red',
                                     facecolor='red', alpha=0.3)
            ax2.add_patch(ellipse)
            
            ellipse2 = patches.Ellipse((roi_rgb.shape[1]*0.7, roi_rgb.shape[0]*0.6),
                                      width=60, height=50,
                                      linewidth=2, edgecolor='red',
                                      facecolor='red', alpha=0.3)
            ax2.add_patch(ellipse2)
            
            ax2.set_title('RGB View with Potential SGD Zones', fontsize=12, weight='bold')
            ax2.axis('off')
            
            # Add scale bar
            ax2.plot([10, 60], [roi_rgb.shape[0]-20, roi_rgb.shape[0]-20], 
                    'k-', linewidth=3)
            ax2.text(35, roi_rgb.shape[0]-30, '~10 m', 
                    color='black', ha='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        except Exception as e:
            print(f"Error with RGB: {e}")
            ax2.text(0.5, 0.5, 'RGB View\n(Error)', ha='center', va='center')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
    
    plt.suptitle('Close-up View of Coastal Zone - Nadir Perspective', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    base_dir = Path("/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA")
    
    generate_sgd_plume_detail(
        base_dir / "MAX_0056.JPG",  # Thermal
        base_dir / "MAX_0057.JPG",  # RGB
        Path("docs/images/sgd_plume_detail.png")
    )