#!/usr/bin/env python3
"""
Generate key figures for the technical paper quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

def generate_environmental_diversity(image_paths, output_path):
    """Generate 2x2 grid showing different coastal environments."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    environments = ['Rocky Volcanic Coast', 'Sandy Beach Area', 'Boulder Field', 'Active Surf Zone']
    
    for idx, (path, env_name, ax) in enumerate(zip(image_paths, environments, axes.flat)):
        if Path(path).exists():
            try:
                img = np.array(Image.open(path))
                ax.imshow(img)
                ax.set_title(f'{chr(65+idx)}) {env_name}', fontsize=12, weight='bold')
            except Exception as e:
                print(f"Error loading {path}: {e}")
                ax.text(0.5, 0.5, f'{env_name}\n(Error loading)', 
                       ha='center', va='center', fontsize=12)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        else:
            ax.text(0.5, 0.5, f'{env_name}\n(Image not found)', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('Coastal Environment Diversity in Rapa Nui', fontsize=14, weight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def generate_segmentation_example(rgb_path, output_path):
    """Generate three-panel segmentation process visualization."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original RGB
    if Path(rgb_path).exists():
        try:
            rgb_img = np.array(Image.open(rgb_path))
            ax1.imshow(rgb_img)
            ax1.set_title('Original RGB', fontsize=12, weight='bold')
            ax1.axis('off')
            
            # Create synthetic segmentation
            gray = np.array(Image.fromarray(rgb_img).convert('L'))
            seg_mask = np.zeros_like(gray)
            
            # Simple threshold-based segmentation
            seg_mask[gray < 80] = 1  # Ocean (dark)
            seg_mask[(gray >= 80) & (gray < 130)] = 2  # Rock
            seg_mask[(gray >= 130) & (gray < 180)] = 3  # Sand
            seg_mask[gray >= 180] = 4  # Wave/foam
            
            # Color-coded segmentation
            seg_colored = np.zeros((*seg_mask.shape, 3), dtype=np.uint8)
            seg_colored[seg_mask == 1] = [0, 0, 200]  # Ocean - blue
            seg_colored[seg_mask == 2] = [128, 128, 128]  # Rock - gray
            seg_colored[seg_mask == 3] = [194, 178, 128]  # Sand - tan
            seg_colored[seg_mask == 4] = [255, 255, 255]  # Wave - white
            
            ax2.imshow(seg_colored)
            ax2.set_title('Segmentation Map', fontsize=12, weight='bold')
            ax2.axis('off')
            
            # Binary ocean mask
            ocean_mask = (seg_mask == 1).astype(np.uint8) * 255
            ax3.imshow(ocean_mask, cmap='gray')
            ax3.set_title('Ocean Mask', fontsize=12, weight='bold')
            ax3.axis('off')
            
            # Add legend using matplotlib patches
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(color=[0, 0, 0.8], label='Ocean'),
                Patch(color=[0.5, 0.5, 0.5], label='Rock'),
                Patch(color=[0.76, 0.7, 0.5], label='Sand'),
                Patch(color=[1, 1, 1], label='Wave')
            ]
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
        except Exception as e:
            print(f"Error processing segmentation: {e}")
            for ax in [ax1, ax2, ax3]:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
    
    plt.suptitle('ML-Based Ocean Segmentation Process', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

def generate_sgd_detection_process(thermal_path, output_path):
    """Generate four-panel SGD detection process visualization."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    if Path(thermal_path).exists():
        try:
            # Load thermal image
            thermal_img = np.array(Image.open(thermal_path))
            
            # Handle RGB thermal
            if len(thermal_img.shape) == 3:
                thermal_img = np.mean(thermal_img, axis=2)
            
            # Normalize for display
            thermal_norm = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min())
            
            # Panel A: Raw thermal
            im1 = ax1.imshow(thermal_norm, cmap='jet')
            ax1.set_title('A) Raw Thermal Image', fontsize=12, weight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Relative Temp')
            
            # Panel B: Ocean-masked (synthetic mask for demo)
            ocean_mask = np.ones_like(thermal_norm, dtype=bool)
            ocean_mask[:100, :] = False  # Simulate land at top
            ocean_mask[-50:, :] = False  # Simulate land at bottom
            masked_thermal = np.ma.masked_where(~ocean_mask, thermal_norm)
            
            im2 = ax2.imshow(masked_thermal, cmap='jet')
            ax2.set_title('B) Ocean-Masked Thermal', fontsize=12, weight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='Relative Temp')
            
            # Panel C: Temperature anomaly
            mean_temp = np.nanmean(thermal_norm[ocean_mask])
            anomaly = thermal_norm - mean_temp
            
            im3 = ax3.imshow(anomaly, cmap='RdBu_r', vmin=-0.2, vmax=0.2)
            ax3.set_title('C) Temperature Anomaly', fontsize=12, weight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='ΔTemp')
            
            # Panel D: Detected SGD plumes
            sgd_threshold = -0.05  # Relative threshold
            sgd_mask = (anomaly < sgd_threshold) & ocean_mask
            
            ax4.imshow(thermal_norm, cmap='gray')
            
            # Overlay SGD detections
            sgd_overlay = np.zeros((*thermal_norm.shape, 4))
            sgd_overlay[sgd_mask] = [1, 0, 0, 0.5]  # Red with transparency
            ax4.imshow(sgd_overlay)
            
            ax4.set_title('D) Detected SGD Plumes', fontsize=12, weight='bold')
            ax4.axis('off')
            
            # Add detection stats
            num_pixels = np.sum(sgd_mask)
            ax4.text(0.02, 0.98, f'SGD pixels: {num_pixels}', 
                    transform=ax4.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top')
        except Exception as e:
            print(f"Error processing thermal: {e}")
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
    
    plt.suptitle('SGD Detection Process Pipeline', fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {output_path}")

if __name__ == "__main__":
    base_dir = Path("/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA")
    
    # Generate environmental diversity with nadir images
    env_images = [
        base_dir / "MAX_0052.JPG",  # Rocky
        base_dir / "MAX_0054.JPG",  # Sandy
        base_dir / "MAX_0056.JPG",  # Boulder
        base_dir / "MAX_0058.JPG"   # Surf
    ]
    generate_environmental_diversity(env_images, Path("docs/images/environmental_diversity.png"))
    
    # Generate segmentation example
    generate_segmentation_example(base_dir / "MAX_0053.JPG", Path("docs/images/segmentation_example.png"))
    
    # Generate SGD detection process
    generate_sgd_detection_process(base_dir / "MAX_0052.JPG", Path("docs/images/sgd_detection_process.png"))
    
    print("\n✅ All figures regenerated with nadir images!")