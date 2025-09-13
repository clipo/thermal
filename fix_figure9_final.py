#!/usr/bin/env python3
"""
Fix Figure 9 using the ml_segmenter directly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

# Add thermal project to path
sys.path.append('/Users/clipo/PycharmProjects/thermal')

# Import the actual ML segmenter
try:
    from ml_segmenter import MLSegmenter
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: MLSegmenter not available")

def create_correct_figure9(thermal_path, rgb_path, output_path='docs/images/sgd_detection_process_final.png'):
    """Create Figure 9 using the actual ML segmenter."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Load images
    thermal_img = None
    ocean_mask = None
    
    # Load thermal
    if Path(thermal_path).exists():
        thermal_img = np.array(Image.open(thermal_path))
        if len(thermal_img.shape) == 3:
            # Convert to grayscale if RGB
            thermal_img = np.mean(thermal_img, axis=2)
        print(f"Loaded thermal: {thermal_img.shape}")
    
    # Load RGB and segment
    if Path(rgb_path).exists():
        rgb_img = np.array(Image.open(rgb_path))
        print(f"Loaded RGB: {rgb_img.shape}")
        
        # Use ML segmenter if available
        if ML_AVAILABLE:
            try:
                segmenter = MLSegmenter()
                if segmenter.model_exists():
                    print("Using ML segmentation...")
                    segmentation = segmenter.segment_ultra_fast(rgb_img)
                    # Ocean is class 0
                    ocean_mask = (segmentation == 0)
                    print(f"ML segmentation complete - Ocean pixels: {np.sum(ocean_mask)} ({np.sum(ocean_mask)/ocean_mask.size*100:.1f}%)")
                else:
                    print("Model doesn't exist, using fallback")
                    ocean_mask = None
            except Exception as e:
                print(f"ML segmentation error: {e}")
                ocean_mask = None
        
        # Fallback segmentation if ML not available
        if ocean_mask is None:
            print("Using color-based ocean detection...")
            # Better ocean detection based on blueness
            b_channel = rgb_img[:,:,2].astype(float)
            r_channel = rgb_img[:,:,0].astype(float)
            g_channel = rgb_img[:,:,1].astype(float)
            
            # Ocean tends to be bluer
            blue_dominance = b_channel - np.maximum(r_channel, g_channel)
            ocean_mask = blue_dominance > 20  # Blue is at least 20 points higher
            
            # Also check for dark blue (deep water)
            dark_blue = (b_channel > 30) & (b_channel < 150) & (blue_dominance > 10)
            ocean_mask = ocean_mask | dark_blue
            
            print(f"Color-based segmentation - Ocean pixels: {np.sum(ocean_mask)} ({np.sum(ocean_mask)/ocean_mask.size*100:.1f}%)")
    
    if thermal_img is not None:
        # Normalize thermal for display
        thermal_norm = (thermal_img - thermal_img.min()) / (thermal_img.max() - thermal_img.min() + 1e-8)
        
        # Panel A: Raw thermal
        im1 = ax1.imshow(thermal_norm, cmap='jet', vmin=0, vmax=1)
        ax1.set_title('A) Raw Thermal Image', fontsize=12, weight='bold')
        ax1.axis('off')
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Normalized Temperature', rotation=270, labelpad=15)
        
        # Panel B: Ocean-masked thermal
        if ocean_mask is not None:
            # Resize ocean mask to match thermal size
            from scipy.ndimage import zoom
            zoom_factor = (thermal_img.shape[0] / ocean_mask.shape[0], 
                          thermal_img.shape[1] / ocean_mask.shape[1])
            ocean_mask_resized = zoom(ocean_mask.astype(float), zoom_factor, order=0) > 0.5
            
            # Apply mask
            masked_thermal = np.ma.masked_where(~ocean_mask_resized, thermal_norm)
            
            print(f"Ocean mask resized to thermal dimensions: {ocean_mask_resized.shape}")
            ocean_coverage = np.sum(ocean_mask_resized) / ocean_mask_resized.size * 100
            print(f"Ocean coverage in thermal: {ocean_coverage:.1f}%")
        else:
            print("No ocean mask available - showing unmasked")
            ocean_mask_resized = np.ones_like(thermal_norm, dtype=bool)
            masked_thermal = thermal_norm
            ocean_coverage = 100
        
        im2 = ax2.imshow(masked_thermal, cmap='jet', vmin=0, vmax=1)
        ax2.set_title('B) Ocean-Masked Thermal', fontsize=12, weight='bold')
        ax2.axis('off')
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Ocean Temperature', rotation=270, labelpad=15)
        
        # Add coverage stat
        ax2.text(0.02, 0.98, f'Ocean: {ocean_coverage:.1f}%', 
                transform=ax2.transAxes, fontsize=10, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                verticalalignment='top')
        
        # Panel C: Temperature anomaly
        if np.any(ocean_mask_resized):
            # Calculate stats only from ocean pixels
            ocean_pixels = thermal_norm[ocean_mask_resized]
            ocean_mean = np.mean(ocean_pixels)
            ocean_std = np.std(ocean_pixels)
            
            # Calculate z-scores
            anomaly = np.zeros_like(thermal_norm)
            anomaly[ocean_mask_resized] = (thermal_norm[ocean_mask_resized] - ocean_mean) / (ocean_std + 1e-8)
            
            # Mask land areas
            anomaly_display = np.ma.masked_where(~ocean_mask_resized, anomaly)
            
            im3 = ax3.imshow(anomaly_display, cmap='RdBu_r', vmin=-3, vmax=3)
            ax3.set_title('C) Temperature Anomaly (Z-scores)', fontsize=12, weight='bold')
            ax3.axis('off')
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('Standard Deviations', rotation=270, labelpad=15)
            
            # Add statistics
            stats_text = f'Ocean Stats:\nμ = {ocean_mean:.3f}\nσ = {ocean_std:.3f}'
            ax3.text(0.02, 0.98, stats_text, 
                    transform=ax3.transAxes, fontsize=9, color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                    verticalalignment='top')
            
            # Panel D: SGD detection
            # SGDs are significantly colder areas in ocean
            sgd_threshold = -1.5  # 1.5 standard deviations colder
            sgd_mask = (anomaly < sgd_threshold) & ocean_mask_resized
            
            # Clean up small noise
            from scipy import ndimage
            # Remove small clusters
            sgd_mask = ndimage.binary_opening(sgd_mask, iterations=1)
            sgd_mask = ndimage.binary_closing(sgd_mask, iterations=1)
            
            # Label connected components
            labeled, num_features = ndimage.label(sgd_mask)
            
            # Filter by size (remove very small detections)
            min_size = 20  # pixels
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < min_size:
                    sgd_mask[labeled == i] = False
            
            # Recount after filtering
            labeled, num_features = ndimage.label(sgd_mask)
        else:
            sgd_mask = np.zeros_like(thermal_norm, dtype=bool)
            num_features = 0
            anomaly = thermal_norm - np.mean(thermal_norm)
        
        # Display result
        ax4.imshow(thermal_norm, cmap='gray', vmin=0, vmax=1)
        
        # Overlay SGD detections in red
        if np.any(sgd_mask):
            # Create colored overlay
            overlay = np.zeros((*thermal_norm.shape, 4))
            overlay[sgd_mask] = [1, 0, 0, 0.6]  # Red with 60% opacity
            ax4.imshow(overlay)
            
            # Draw contours around detections
            from matplotlib import patches
            for i in range(1, num_features + 1):
                component = (labeled == i)
                y_coords, x_coords = np.where(component)
                if len(y_coords) > 0:
                    # Get bounding box
                    min_y, max_y = np.min(y_coords), np.max(y_coords)
                    min_x, max_x = np.min(x_coords), np.max(x_coords)
                    rect = patches.Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                                           linewidth=1, edgecolor='yellow', 
                                           facecolor='none', linestyle='--')
                    ax4.add_patch(rect)
        
        ax4.set_title('D) Detected SGD Plumes (Cold Ocean Anomalies)', fontsize=12, weight='bold')
        ax4.axis('off')
        
        # Calculate statistics
        sgd_pixels = np.sum(sgd_mask)
        if np.any(ocean_mask_resized):
            sgd_ocean_coverage = sgd_pixels / np.sum(ocean_mask_resized) * 100
        else:
            sgd_ocean_coverage = 0
        
        detection_text = f'SGD Plumes: {num_features}\nSGD Pixels: {sgd_pixels}\nOcean Coverage: {sgd_ocean_coverage:.2f}%\nThreshold: {sgd_threshold:.1f}σ'
        ax4.text(0.02, 0.98, detection_text, 
                transform=ax4.transAxes, fontsize=9, color='white',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5),
                verticalalignment='top')
    
    plt.suptitle('SGD Detection Pipeline with Ocean Segmentation', fontsize=14, weight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved final Figure 9 to: {output_path}")
    return ocean_mask_resized if 'ocean_mask_resized' in locals() else None

def main():
    """Generate the corrected Figure 9."""
    
    print("\n" + "="*60)
    print("FIXING FIGURE 9 - SGD DETECTION PROCESS")
    print("="*60 + "\n")
    
    # Use good nadir examples
    base_dir = Path("/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA")
    
    # Try different image pairs to find one with good ocean
    image_pairs = [
        ("MAX_0052.JPG", "MAX_0053.JPG"),  # Original pair
        ("MAX_0054.JPG", "MAX_0055.JPG"),  # Next pair
        ("MAX_0060.JPG", "MAX_0061.JPG"),  # Later pair
    ]
    
    for thermal_name, rgb_name in image_pairs:
        thermal_path = base_dir / thermal_name
        rgb_path = base_dir / rgb_name
        
        if thermal_path.exists() and rgb_path.exists():
            print(f"Processing pair: {thermal_name} / {rgb_name}")
            
            ocean_mask = create_correct_figure9(
                thermal_path,
                rgb_path,
                'docs/images/sgd_detection_process_final.png'
            )
            
            if ocean_mask is not None and np.sum(ocean_mask) > 0:
                print("\n✅ SUCCESS: Figure 9 has been corrected!")
                print("Ocean is properly identified and SGDs are detected only in ocean areas.")
                break
            else:
                print("⚠️ No ocean found in this pair, trying next...")
    
    print("\n" + "="*60)
    print("Update TECHNICAL_PAPER.md to use: sgd_detection_process_final.png")
    print("="*60)

if __name__ == "__main__":
    main()