#!/usr/bin/env python3
"""
Fix Figure 7A using actual working segmentation from the project.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys
import pickle

# Add thermal project to path
sys.path.append('/Users/clipo/PycharmProjects/thermal')

def create_working_segmentation_figure():
    """Create segmentation figure using known working example."""
    
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    # Try to use the ML segmenter directly
    try:
        from ml_segmenter import MLSegmenter
        segmenter = MLSegmenter()
        ML_AVAILABLE = segmenter.model_exists()
        print(f"ML Segmenter available: {ML_AVAILABLE}")
    except:
        ML_AVAILABLE = False
        segmenter = None
        print("ML Segmenter not available")
    
    # Use a known good example - try multiple locations
    test_locations = [
        # Hanga Roa images
        Path('/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/1 July 23/Hanga Roa - Rano Kau/100MEDIA/MAX_0020.JPG'),
        Path('/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/1 July 23/Hanga Roa - Rano Kau/101MEDIA/MAX_0010.JPG'),
        # 24 June images  
        Path('/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/102MEDIA/MAX_0060.JPG'),
        Path('/Volumes/RapaNui/Rapa Nui June 2023/Thermal Flights/24 June 23/103MEDIA/MAX_0050.JPG'),
    ]
    
    rgb_path = None
    for path in test_locations:
        if path.exists():
            rgb_path = path
            print(f"Using image: {rgb_path}")
            break
    
    if not rgb_path:
        print("No suitable image found")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Load RGB
    rgb_img = np.array(Image.open(rgb_path))
    h, w = rgb_img.shape[:2]
    
    # Display original
    ax1.imshow(rgb_img)
    ax1.set_title('Original RGB Image', fontsize=12, weight='bold')
    ax1.axis('off')
    
    segmentation = None
    
    # Try ML segmentation first
    if ML_AVAILABLE and segmenter:
        try:
            print("Applying ML segmentation...")
            segmentation = segmenter.segment_ultra_fast(rgb_img)
            print(f"ML segmentation shape: {segmentation.shape}")
            print(f"Unique classes: {np.unique(segmentation)}")
        except Exception as e:
            print(f"ML segmentation failed: {e}")
    
    # Fallback to manual/color-based segmentation
    if segmentation is None:
        print("Using manual color-based segmentation")
        
        # Convert to different color spaces for better segmentation
        hsv = np.array(Image.fromarray(rgb_img).convert('HSV'))
        gray = np.array(Image.fromarray(rgb_img).convert('L'))
        
        # Extract channels
        h_channel = hsv[:,:,0]  # Hue
        s_channel = hsv[:,:,1]  # Saturation
        v_channel = hsv[:,:,2]  # Value
        
        r = rgb_img[:,:,0].astype(float)
        g = rgb_img[:,:,1].astype(float)
        b = rgb_img[:,:,2].astype(float)
        
        # Initialize segmentation
        segmentation = np.ones((rgb_img.shape[0], rgb_img.shape[1]), dtype=int) * 2  # Default to land
        
        # Ocean detection - multiple criteria
        # 1. Blue dominance
        blue_dominance = (b > r + 30) & (b > g + 20)
        
        # 2. HSV criteria for water (blue-cyan hues)
        water_hue = ((h_channel > 90) & (h_channel < 150)) | ((h_channel > 170) & (h_channel < 210))
        water_sat = s_channel > 30  # Some saturation
        water_val = (v_channel > 30) & (v_channel < 200)  # Not too dark or bright
        
        # 3. Dark blue/green water
        dark_water = (b > 40) & (b < 120) & (g > 30) & (g < 100) & (r < 80)
        
        # Combine ocean criteria
        ocean_mask = (blue_dominance | (water_hue & water_sat & water_val) | dark_water)
        segmentation[ocean_mask] = 0  # Ocean
        
        # Wave/foam detection - very bright areas
        bright = (gray > 200) | ((r > 180) & (g > 180) & (b > 180))
        segmentation[bright] = 3  # Waves
        
        # Rock detection - dark, low saturation
        dark = (gray < 60) & (s_channel < 50)
        segmentation[dark & ~ocean_mask] = 1  # Rocks
        
        print(f"Manual segmentation complete")
        print(f"Ocean pixels: {np.sum(segmentation == 0)} ({np.sum(segmentation == 0)/segmentation.size*100:.1f}%)")
        print(f"Rock pixels: {np.sum(segmentation == 1)} ({np.sum(segmentation == 1)/segmentation.size*100:.1f}%)")
        print(f"Land pixels: {np.sum(segmentation == 2)} ({np.sum(segmentation == 2)/segmentation.size*100:.1f}%)")
        print(f"Wave pixels: {np.sum(segmentation == 3)} ({np.sum(segmentation == 3)/segmentation.size*100:.1f}%)")
    
    # Create color-coded visualization
    seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Use distinct colors
    seg_colored[segmentation == 0] = [30, 100, 255]   # Ocean - bright blue
    seg_colored[segmentation == 1] = [100, 100, 100]  # Rock - gray
    seg_colored[segmentation == 2] = [160, 130, 90]   # Land - tan/brown
    seg_colored[segmentation == 3] = [255, 255, 255]  # Wave - white
    
    ax2.imshow(seg_colored)
    ax2.set_title('Segmentation Map', fontsize=12, weight='bold')
    ax2.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(color=[30/255, 100/255, 255/255], label='Ocean'),
        Patch(color=[100/255, 100/255, 100/255], label='Rock'),
        Patch(color=[160/255, 130/255, 90/255], label='Land'),
        Patch(color=[1, 1, 1], label='Wave/Foam')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Create ocean mask - white for ocean, black for everything else
    ocean_binary = np.zeros((h, w), dtype=np.uint8)
    ocean_binary[segmentation == 0] = 255  # White for ocean
    
    ax3.imshow(ocean_binary, cmap='gray', vmin=0, vmax=255)
    ax3.set_title('Ocean Mask', fontsize=12, weight='bold')
    ax3.axis('off')
    
    # Add statistics
    ocean_pct = np.sum(segmentation == 0) / segmentation.size * 100
    land_pct = np.sum(segmentation == 2) / segmentation.size * 100
    rock_pct = np.sum(segmentation == 1) / segmentation.size * 100
    wave_pct = np.sum(segmentation == 3) / segmentation.size * 100
    
    stats_text = f'Ocean: {ocean_pct:.1f}%\nLand: {land_pct:.1f}%\nRock: {rock_pct:.1f}%\nWave: {wave_pct:.1f}%'
    ax3.text(0.02, 0.98, stats_text, 
            transform=ax3.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    # Add note about what white means
    ax3.text(0.5, 0.02, 'White = Ocean areas for SGD detection', 
            transform=ax3.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            ha='center', verticalalignment='bottom')
    
    plt.suptitle('Ocean Segmentation for SGD Detection', fontsize=14, weight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path('docs/images/segmentation_working.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved working segmentation figure to: {output_path}")
    
    # Also try to copy the detection pipeline image if it's better
    pipeline_path = Path('/Users/clipo/PycharmProjects/thermal/docs/images/detection_pipeline.png')
    if pipeline_path.exists():
        print(f"Note: You can also use the existing detection_pipeline.png from README")

def main():
    print("\n" + "="*60)
    print("CREATING WORKING SEGMENTATION FIGURE")
    print("="*60 + "\n")
    
    create_working_segmentation_figure()
    
    print("\n" + "="*60)
    print("Update TECHNICAL_PAPER.md to use: segmentation_working.png")
    print("Or use the existing detection_pipeline.png from README")
    print("="*60)

if __name__ == "__main__":
    main()