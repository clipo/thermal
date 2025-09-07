#!/usr/bin/env python3
"""
Aligned RGB-Based Ocean Segmentation
Uses proper alignment between RGB and thermal images
"""

import numpy as np
from PIL import Image
from pathlib import Path
from skimage import color
from image_aligner import ThermalRGBAligner
import json


class AlignedRGBSegmenter:
    """RGB segmenter that properly aligns with thermal FOV"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        self.aligner = ThermalRGBAligner(base_path)
        
        # Color thresholds for segmentation
        self.ocean_hsv_range = {
            'h_min': 180, 'h_max': 250,  
            's_min': 20, 's_max': 255,   
            'v_min': 20, 'v_max': 200    
        }
        
        self.land_hsv_range = {
            'h_min': 40, 'h_max': 150,   
            's_min': 15, 's_max': 255,   
            'v_min': 10, 'v_max': 255    
        }
        
        self.wave_hsv_range = {
            'h_min': 0, 'h_max': 360,     
            's_min': 0, 's_max': 30,      
            'v_min': 180, 'v_max': 255   
        }
    
    def load_aligned_pair(self, frame_number):
        """Load properly aligned RGB-thermal pair"""
        
        # Use aligner to get properly cropped RGB
        aligned_data = self.aligner.apply_alignment_to_data(frame_number)
        
        if aligned_data is None:
            raise FileNotFoundError(f"Could not load frame {frame_number}")
        
        return {
            'rgb_full': aligned_data['rgb_full'],
            'rgb_aligned': aligned_data['rgb_aligned'],  # This matches thermal FOV
            'thermal': aligned_data['thermal'],
            'frame_number': frame_number
        }
    
    def segment_ocean_land_waves(self, rgb_aligned):
        """
        Segment the aligned RGB image into ocean, land, and waves
        rgb_aligned: RGB image that matches thermal dimensions exactly
        """
        
        # Convert to HSV
        hsv = color.rgb2hsv(rgb_aligned)
        h = hsv[:, :, 0] * 360
        s = hsv[:, :, 1] * 255
        v = hsv[:, :, 2] * 255
        
        # Detect ocean (blue colors)
        ocean_mask = (
            (h >= self.ocean_hsv_range['h_min']) & 
            (h <= self.ocean_hsv_range['h_max']) &
            (s >= self.ocean_hsv_range['s_min']) & 
            (v >= self.ocean_hsv_range['v_min']) & 
            (v <= self.ocean_hsv_range['v_max'])
        )
        
        # Detect land (green/brown colors)
        land_mask = (
            (h >= self.land_hsv_range['h_min']) & 
            (h <= self.land_hsv_range['h_max']) &
            (s >= self.land_hsv_range['s_min'])
        )
        
        # Also include very dark areas as land
        dark_mask = (v < 30)
        land_mask = land_mask | dark_mask
        
        # Detect waves/foam (white/bright areas)
        wave_mask = (
            (s <= self.wave_hsv_range['s_max']) &
            (v >= self.wave_hsv_range['v_min'])
        )
        
        # Clean up conflicts
        land_mask = land_mask & ~wave_mask
        ocean_mask = ocean_mask & ~wave_mask & ~land_mask
        
        # Fill undefined areas based on color
        undefined = ~(ocean_mask | land_mask | wave_mask)
        
        # Simple classification for undefined pixels
        # If more blue than green -> ocean
        # If more green than blue -> land
        if undefined.sum() > 0:
            r, g, b = rgb_aligned[:,:,0], rgb_aligned[:,:,1], rgb_aligned[:,:,2]
            
            # Where undefined and blue > green
            undefined_ocean = undefined & (b > g * 1.1)
            ocean_mask = ocean_mask | undefined_ocean
            
            # Where undefined and green > blue  
            undefined_land = undefined & (g > b * 1.1)
            land_mask = land_mask | undefined_land
        
        return {
            'ocean': ocean_mask,
            'land': land_mask,
            'waves': wave_mask
        }
    
    def process_frame(self, frame_number):
        """Process a frame with proper alignment"""
        
        # Load aligned data
        data = self.load_aligned_pair(frame_number)
        
        # Segment the aligned RGB (same size as thermal)
        masks = self.segment_ocean_land_waves(data['rgb_aligned'])
        
        # Now masks are perfectly aligned with thermal data
        return {
            'frame_number': frame_number,
            'rgb_full': data['rgb_full'],
            'rgb_aligned': data['rgb_aligned'],
            'thermal': data['thermal'],
            'masks': masks
        }


def test_aligned_segmentation(frame_number=248):
    """Test the aligned segmentation"""
    
    import matplotlib.pyplot as plt
    
    segmenter = AlignedRGBSegmenter()
    
    # Process frame
    result = segmenter.process_frame(frame_number)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1
    axes[0,0].imshow(result['rgb_full'])
    axes[0,0].set_title(f'Full RGB (4096x3072)')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(result['rgb_aligned'])
    axes[0,1].set_title(f'Aligned RGB Region (640x512)')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(result['thermal'], cmap='hot')
    axes[0,2].set_title(f'Thermal (640x512)')
    axes[0,2].axis('off')
    
    # Row 2 - Segmentation results
    mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
    mask_display[result['masks']['ocean']] = [0, 0.3, 1]  # Blue for ocean
    mask_display[result['masks']['land']] = [0, 0.5, 0]   # Green for land
    mask_display[result['masks']['waves']] = [1, 1, 1]     # White for waves
    
    axes[1,0].imshow(mask_display)
    axes[1,0].set_title('Segmentation Masks')
    axes[1,0].axis('off')
    
    # Apply masks to thermal
    ocean_thermal = result['thermal'].copy()
    ocean_thermal[~result['masks']['ocean']] = np.nan
    
    axes[1,1].imshow(ocean_thermal, cmap='viridis')
    axes[1,1].set_title('Ocean Thermal Only')
    axes[1,1].axis('off')
    
    # Coverage stats
    total_pixels = result['masks']['ocean'].size
    stats_text = f"Coverage Statistics:\n\n"
    stats_text += f"Ocean: {100*result['masks']['ocean'].sum()/total_pixels:.1f}%\n"
    stats_text += f"Land: {100*result['masks']['land'].sum()/total_pixels:.1f}%\n"
    stats_text += f"Waves: {100*result['masks']['waves'].sum()/total_pixels:.1f}%\n\n"
    
    ocean_temps = result['thermal'][result['masks']['ocean']]
    if len(ocean_temps) > 0:
        stats_text += f"Ocean Temperature:\n"
        stats_text += f"Mean: {ocean_temps.mean():.1f}°C\n"
        stats_text += f"Std: {ocean_temps.std():.2f}°C\n"
        stats_text += f"Range: [{ocean_temps.min():.1f}, {ocean_temps.max():.1f}]°C"
    
    axes[1,2].text(0.1, 0.5, stats_text, transform=axes[1,2].transAxes,
                   fontsize=11, verticalalignment='center')
    axes[1,2].axis('off')
    
    plt.suptitle(f'Aligned RGB-Thermal Segmentation - Frame {frame_number}')
    plt.tight_layout()
    plt.show()
    
    return result


if __name__ == "__main__":
    print("Testing aligned RGB-thermal segmentation...")
    
    # First run the aligner to set up proper alignment
    print("\nStep 1: Set up alignment")
    print("Run: python image_aligner.py")
    print("Choose option 1 to calibrate alignment")
    
    # Then test segmentation
    frame = int(input("\nEnter frame to test (default 248): ") or "248")
    result = test_aligned_segmentation(frame)