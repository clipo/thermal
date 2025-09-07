#!/usr/bin/env python3
"""
Test and tune segmentation parameters across multiple frames.
Allows navigation between frames to ensure parameters work well across different conditions.
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

class MultiFrameSegmentationTuner:
    """Interactive tuner for ocean/land/wave segmentation across multiple frames"""
    
    def __init__(self):
        self.base_path = Path("data/100MEDIA")
        
        # Find available frames
        self.frames = []
        for f in sorted(self.base_path.glob("MAX_*.JPG"))[:50]:
            num = int(f.stem.split('_')[1])
            if (self.base_path / f"IRX_{num:04d}.irg").exists():
                self.frames.append(num)
        
        if not self.frames:
            raise FileNotFoundError("No paired frames found!")
        
        print(f"Found {len(self.frames)} frames: {self.frames[0]} to {self.frames[-1]}")
        
        self.current_frame_idx = 0
        self.current_frame = self.frames[0]
        
        # Your optimized parameters as defaults
        self.params = {
            'ocean_h_min': 180,
            'ocean_h_max': 250,
            'ocean_s_min': 20,
            'ocean_v_min': 30,
            'ocean_v_max': 200,
            
            'rock_v_max': 82,  # Your optimized value
            'rock_variance': 40,  # Your optimized value
            
            'wave_s_max': 25,  # Your optimized value
            'wave_v_min': 150,  # Your optimized value
            
            'blue_ratio': 1.20,  # Your optimized value
        }
        
        self.load_frame(self.current_frame)
        self.setup_gui()
        self.update_segmentation()
    
    def load_frame(self, frame_number):
        """Load a specific frame"""
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
        
        self.current_frame = frame_number
    
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
        self.ax1.set_title(f'Frame {self.current_frame}: RGB')
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
        stats = f"Frame {self.current_frame} ({self.current_frame_idx+1}/{len(self.frames)})\n\n"
        stats += f"Coverage Statistics:\n"
        stats += f"Ocean: {100*ocean.sum()/total:.1f}%\n"
        stats += f"Land:  {100*land.sum()/total:.1f}%\n"
        stats += f"Waves: {100*waves.sum()/total:.1f}%\n\n"
        stats += f"Frame range: {self.frames[0]}-{self.frames[-1]}"
        self.ax4.text(0.1, 0.5, stats, fontsize=12, family='monospace')
        
        plt.draw()
    
    def next_frame(self, event):
        """Go to next frame"""
        if self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.load_frame(self.frames[self.current_frame_idx])
            self.update_segmentation()
            print(f"Switched to frame {self.current_frame}")
    
    def prev_frame(self, event):
        """Go to previous frame"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.load_frame(self.frames[self.current_frame_idx])
            self.update_segmentation()
            print(f"Switched to frame {self.current_frame}")
    
    def setup_gui(self):
        """Setup the interactive GUI"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Display panels
        self.ax1 = plt.subplot(2, 2, 1)
        self.ax2 = plt.subplot(2, 2, 2)
        self.ax3 = plt.subplot(2, 2, 3)
        self.ax4 = plt.subplot(2, 2, 4)
        
        # Sliders
        plt.subplots_adjust(bottom=0.40)
        
        # Ocean parameters
        ax_oh_min = plt.axes([0.1, 0.30, 0.35, 0.02])
        ax_oh_max = plt.axes([0.1, 0.27, 0.35, 0.02])
        ax_os_min = plt.axes([0.1, 0.24, 0.35, 0.02])
        ax_ov_min = plt.axes([0.1, 0.21, 0.35, 0.02])
        ax_ov_max = plt.axes([0.1, 0.18, 0.35, 0.02])
        
        # Rock parameters
        ax_rock_v = plt.axes([0.55, 0.30, 0.35, 0.02])
        ax_rock_var = plt.axes([0.55, 0.27, 0.35, 0.02])
        
        # Wave parameters
        ax_wave_s = plt.axes([0.55, 0.21, 0.35, 0.02])
        ax_wave_v = plt.axes([0.55, 0.18, 0.35, 0.02])
        
        # Blue ratio
        ax_blue = plt.axes([0.1, 0.13, 0.35, 0.02])
        
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
        
        # Navigation buttons
        ax_prev = plt.axes([0.35, 0.08, 0.08, 0.03])
        ax_next = plt.axes([0.44, 0.08, 0.08, 0.03])
        ax_print = plt.axes([0.55, 0.08, 0.10, 0.03])
        ax_apply = plt.axes([0.66, 0.08, 0.10, 0.03])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_print = Button(ax_print, 'Print Params')
        self.btn_apply = Button(ax_apply, 'Apply to Code')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_print.on_clicked(self.print_params)
        self.btn_apply.on_clicked(self.apply_to_detector)
        
        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'left':
            self.prev_frame(None)
        elif event.key == 'right':
            self.next_frame(None)
    
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
    
    def apply_to_detector(self, event):
        """Apply parameters to the main detector file"""
        print("\nApplying parameters to sgd_detector_integrated.py...")
        
        # Read the current detector file
        detector_path = Path("sgd_detector_integrated.py")
        if not detector_path.exists():
            print("Error: sgd_detector_integrated.py not found!")
            return
        
        with open(detector_path, 'r') as f:
            lines = f.readlines()
        
        # Find and replace the parameter section
        # This is a simplified version - you might need to adjust based on exact file structure
        print(f"Updated parameters:")
        print(f"  Ocean H: {self.params['ocean_h_min']:.0f}-{self.params['ocean_h_max']:.0f}")
        print(f"  Ocean V: {self.params['ocean_v_min']:.0f}-{self.params['ocean_v_max']:.0f}")
        print(f"  Rock V threshold: {self.params['rock_v_max']:.0f}")
        print(f"  Wave thresholds: S<{self.params['wave_s_max']:.0f}, V>{self.params['wave_v_min']:.0f}")
        print(f"  Blue ratio: {self.params['blue_ratio']:.2f}")
        print("\nParameters ready to apply. Update sgd_detector_integrated.py manually with printed values.")
    
    def run(self):
        """Run the interactive tuner"""
        print("\nMulti-Frame Segmentation Tuner")
        print("=" * 50)
        print("Controls:")
        print("  - Use sliders to adjust segmentation parameters")
        print("  - Previous/Next buttons or Left/Right arrows to change frames")
        print("  - Print Params: Display parameters for copying")
        print("  - Apply to Code: Instructions for updating detector")
        print("\nStarting with your optimized parameters as defaults")
        print("Test across different frames to ensure robustness!")
        plt.show()

if __name__ == "__main__":
    tuner = MultiFrameSegmentationTuner()
    tuner.run()