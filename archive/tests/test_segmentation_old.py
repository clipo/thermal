#!/usr/bin/env python3
"""
Test and tune segmentation parameters for rocky shores and waves.
"""

import matplotlib
matplotlib.use('TkAgg')  # Force interactive backend
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path
from PIL import Image as PILImage
from skimage import color, morphology
import matplotlib.patches as mpatches

class SegmentationTuner:
    """Interactive tuner for ocean/land/wave segmentation"""
    
    def __init__(self, frame_number=248):
        self.frame_number = frame_number
        self.base_path = Path("data/100MEDIA")
        
        # Load RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        self.rgb_full = np.array(PILImage.open(rgb_path))
        
        # Extract thermal FOV region (center 70%)
        self.thermal_fov_ratio = 0.7
        h, w = self.rgb_full.shape[:2]
        crop_h = int(h * self.thermal_fov_ratio)
        crop_w = int(w * self.thermal_fov_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        self.rgb_aligned = self.rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize to thermal dimensions for display
        img_pil = PILImage.fromarray(self.rgb_aligned)
        self.rgb_display = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
        
        # Convert to HSV and LAB
        self.hsv = color.rgb2hsv(self.rgb_display)
        self.lab = color.rgb2lab(self.rgb_display)
        
        # Initial parameters
        self.params = {
            'ocean_h_min': 180,
            'ocean_h_max': 250,
            'ocean_s_min': 20,
            'ocean_v_min': 30,
            'ocean_v_max': 200,
            
            'rock_v_max': 60,  # Dark threshold
            'rock_variance': 40,  # Color uniformity
            
            'wave_s_max': 25,  # Low saturation
            'wave_v_min': 150,  # High brightness
            
            'blue_ratio': 1.2,  # How much more blue for ocean
        }
        
        self.setup_gui()
        self.update_segmentation()
    
    def segment_advanced(self):
        """Advanced segmentation with better rock/wave handling"""
        h = self.hsv[:, :, 0] * 360
        s = self.hsv[:, :, 1] * 255
        v = self.hsv[:, :, 2] * 255
        
        r = self.rgb_display[:,:,0].astype(float)
        g = self.rgb_display[:,:,1].astype(float)
        b = self.rgb_display[:,:,2].astype(float)
        
        # Calculate color characteristics
        max_channel = np.maximum(r, np.maximum(g, b))
        min_channel = np.minimum(r, np.minimum(g, b))
        color_range = max_channel - min_channel
        
        # Blue dominance ratio
        blue_dominance = np.zeros_like(b)
        mask = (r > 0) & (g > 0)
        blue_dominance[mask] = b[mask] / np.maximum(r[mask], g[mask])
        
        # 1. WHITE WAVES/FOAM (high brightness, low saturation)
        wave_mask = (
            (s < self.params['wave_s_max']) & 
            (v > self.params['wave_v_min'])
        )
        
        # 2. DARK ROCKY SHORES (dark, neutral colors)
        # Rocks are dark but NOT blue-tinted
        dark_mask = (v < self.params['rock_v_max'])
        neutral_mask = color_range < self.params['rock_variance']
        not_blue = blue_dominance < self.params['blue_ratio']
        rock_mask = dark_mask & (neutral_mask | not_blue)
        
        # 3. OCEAN (blue hues, not too dark, not waves)
        ocean_mask = (
            (h >= self.params['ocean_h_min']) & 
            (h <= self.params['ocean_h_max']) &
            (s >= self.params['ocean_s_min']) &
            (v >= self.params['ocean_v_min']) &
            (v <= self.params['ocean_v_max'])
        ) | (
            # Also include dark blue areas
            (blue_dominance > self.params['blue_ratio']) &
            (b > 20) &  # Not completely black
            ~wave_mask
        )
        
        # 4. LAND (everything else, including vegetation)
        land_mask = ~(ocean_mask | wave_mask | rock_mask)
        
        # Combine rocks with land
        land_mask = land_mask | rock_mask
        
        # Ensure no overlaps
        ocean_mask = ocean_mask & ~wave_mask & ~land_mask
        
        # Morphological cleanup
        ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)
        land_mask = morphology.remove_small_objects(land_mask, min_size=100)
        wave_mask = morphology.remove_small_objects(wave_mask, min_size=20)
        
        # Fill holes
        ocean_mask = morphology.remove_small_holes(ocean_mask, area_threshold=100)
        land_mask = morphology.remove_small_holes(land_mask, area_threshold=100)
        
        return ocean_mask, land_mask, wave_mask
    
    def update_segmentation(self):
        """Update segmentation with current parameters"""
        ocean, land, waves = self.segment_advanced()
        
        # Create color overlay
        overlay = np.zeros((*ocean.shape, 3))
        overlay[ocean] = [0, 0.3, 1]  # Blue for ocean
        overlay[land] = [0, 0.5, 0]   # Green for land
        overlay[waves] = [1, 1, 0.5]   # Yellow for waves
        
        # Update displays
        self.ax1.clear()
        self.ax1.imshow(self.rgb_display)
        self.ax1.set_title('Original RGB')
        self.ax1.axis('off')
        
        self.ax2.clear()
        self.ax2.imshow(overlay)
        self.ax2.set_title('Segmentation')
        self.ax2.axis('off')
        
        # Add legend below the image
        ocean_patch = mpatches.Patch(color=[0, 0.3, 1], label='Ocean')
        land_patch = mpatches.Patch(color=[0, 0.5, 0], label='Land/Rocks')
        wave_patch = mpatches.Patch(color=[1, 1, 0.5], label='Waves')
        self.ax2.legend(handles=[ocean_patch, land_patch, wave_patch], 
                       loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                       ncol=3, frameon=False)
        
        self.ax3.clear()
        blend = self.rgb_display.copy() / 255.0
        mask = ocean | waves
        blend[mask] = blend[mask] * 0.5 + overlay[mask] * 0.5
        blend[land] = blend[land] * 0.7 + overlay[land] * 0.3
        self.ax3.imshow(blend)
        self.ax3.set_title('Overlay')
        self.ax3.axis('off')
        
        # Statistics
        total = ocean.size
        self.ax4.clear()
        self.ax4.axis('off')
        stats = f"Coverage Statistics:\n"
        stats += f"Ocean: {100*ocean.sum()/total:.1f}%\n"
        stats += f"Land:  {100*land.sum()/total:.1f}%\n"
        stats += f"Waves: {100*waves.sum()/total:.1f}%\n"
        self.ax4.text(0.1, 0.5, stats, fontsize=12, family='monospace')
        
        plt.draw()
    
    def setup_gui(self):
        """Setup the interactive GUI"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Display panels
        self.ax1 = plt.subplot(2, 2, 1)
        self.ax2 = plt.subplot(2, 2, 2)
        self.ax3 = plt.subplot(2, 2, 3)
        self.ax4 = plt.subplot(2, 2, 4)
        
        # Sliders
        plt.subplots_adjust(bottom=0.35)
        
        # Ocean parameters
        ax_oh_min = plt.axes([0.1, 0.25, 0.35, 0.02])
        ax_oh_max = plt.axes([0.1, 0.22, 0.35, 0.02])
        ax_os_min = plt.axes([0.1, 0.19, 0.35, 0.02])
        ax_ov_min = plt.axes([0.1, 0.16, 0.35, 0.02])
        ax_ov_max = plt.axes([0.1, 0.13, 0.35, 0.02])
        
        # Rock parameters
        ax_rock_v = plt.axes([0.55, 0.25, 0.35, 0.02])
        ax_rock_var = plt.axes([0.55, 0.22, 0.35, 0.02])
        
        # Wave parameters
        ax_wave_s = plt.axes([0.55, 0.16, 0.35, 0.02])
        ax_wave_v = plt.axes([0.55, 0.13, 0.35, 0.02])
        
        # Blue ratio
        ax_blue = plt.axes([0.1, 0.08, 0.35, 0.02])
        
        self.sliders = {
            'ocean_h_min': Slider(ax_oh_min, 'Ocean H min', 0, 360, valinit=self.params['ocean_h_min']),
            'ocean_h_max': Slider(ax_oh_max, 'Ocean H max', 0, 360, valinit=self.params['ocean_h_max']),
            'ocean_s_min': Slider(ax_os_min, 'Ocean S min', 0, 255, valinit=self.params['ocean_s_min']),
            'ocean_v_min': Slider(ax_ov_min, 'Ocean V min', 0, 255, valinit=self.params['ocean_v_min']),
            'ocean_v_max': Slider(ax_ov_max, 'Ocean V max', 0, 255, valinit=self.params['ocean_v_max']),
            
            'rock_v_max': Slider(ax_rock_v, 'Rock V max', 0, 100, valinit=self.params['rock_v_max']),
            'rock_variance': Slider(ax_rock_var, 'Rock color var', 0, 100, valinit=self.params['rock_variance']),
            
            'wave_s_max': Slider(ax_wave_s, 'Wave S max', 0, 50, valinit=self.params['wave_s_max']),
            'wave_v_min': Slider(ax_wave_v, 'Wave V min', 100, 255, valinit=self.params['wave_v_min']),
            
            'blue_ratio': Slider(ax_blue, 'Blue ratio', 1.0, 2.0, valinit=self.params['blue_ratio']),
        }
        
        # Connect sliders
        def make_update_func(k):
            return lambda val: self.update_param(k, val)
        
        for key, slider in self.sliders.items():
            slider.on_changed(make_update_func(key))
        
        # Print button
        ax_print = plt.axes([0.55, 0.08, 0.1, 0.03])
        from matplotlib.widgets import Button
        self.btn_print = Button(ax_print, 'Print Params')
        self.btn_print.on_clicked(self.print_params)
    
    def update_param(self, key, val):
        """Update parameter and refresh display"""
        self.params[key] = val
        print(f"Updated {key} to {val:.1f}")
        self.update_segmentation()
    
    def print_params(self, event):
        """Print current parameters for copying into code"""
        print("\n# Optimized segmentation parameters:")
        print("self.ocean_hsv = {")
        print(f"    'h': ({self.params['ocean_h_min']:.0f}, {self.params['ocean_h_max']:.0f}),")
        print(f"    's': ({self.params['ocean_s_min']:.0f}, 255),")
        print(f"    'v': ({self.params['ocean_v_min']:.0f}, {self.params['ocean_v_max']:.0f})")
        print("}")
        print(f"self.rock_v_threshold = {self.params['rock_v_max']:.0f}")
        print(f"self.rock_variance_threshold = {self.params['rock_variance']:.0f}")
        print(f"self.wave_s_threshold = {self.params['wave_s_max']:.0f}")
        print(f"self.wave_v_threshold = {self.params['wave_v_min']:.0f}")
        print(f"self.blue_ratio_threshold = {self.params['blue_ratio']:.2f}")
    
    def run(self):
        """Run the interactive tuner"""
        print("Segmentation Tuner")
        print("=" * 50)
        print("Adjust sliders to optimize ocean/land/wave detection")
        print("Dark rocky shores should appear GREEN (land)")
        print("White waves should appear YELLOW")
        print("Only actual water should appear BLUE")
        plt.show()

if __name__ == "__main__":
    tuner = SegmentationTuner(frame_number=248)
    tuner.run()