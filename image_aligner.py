#!/usr/bin/env python3
"""
RGB-Thermal Image Alignment Tool
Properly aligns thermal images with RGB images accounting for different FOVs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from matplotlib.patches import Rectangle
from PIL import Image
from pathlib import Path
import json
import cv2


class ThermalRGBAligner:
    """Align thermal images with RGB images"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Default alignment parameters (to be calibrated)
        self.alignment = {
            'scale_x': 6.4,  # RGB/thermal width ratio
            'scale_y': 6.0,  # RGB/thermal height ratio
            'offset_x': 0,   # Horizontal offset in RGB pixels
            'offset_y': 0,   # Vertical offset in RGB pixels
            'rotation': 0    # Rotation in degrees
        }
        
        # Try to load saved alignment
        self.load_alignment()
    
    def load_alignment(self, filepath="thermal_alignment.json"):
        """Load saved alignment parameters"""
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                self.alignment = json.load(f)
                print(f"Loaded alignment from {filepath}")
                print(f"Alignment: {self.alignment}")
    
    def save_alignment(self, filepath="thermal_alignment.json"):
        """Save alignment parameters"""
        with open(filepath, 'w') as f:
            json.dump(self.alignment, f, indent=2)
        print(f"Saved alignment to {filepath}")
    
    def estimate_initial_alignment(self, frame_number):
        """
        Estimate initial alignment by finding thermal FOV in RGB
        Assumes thermal is roughly centered in RGB
        """
        # Load images
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        thermal_data = self.load_thermal(frame_number)
        
        if not rgb_path.exists():
            return None
        
        rgb_img = np.array(Image.open(rgb_path))
        
        # Estimate scale based on typical drone camera setups
        # Thermal cameras typically have narrower FOV than RGB
        # Common ratios: thermal FOV = 40-50°, RGB FOV = 80-90°
        
        # Start with centered assumption
        rgb_center_x = self.rgb_width // 2
        rgb_center_y = self.rgb_height // 2
        
        # Estimate thermal footprint in RGB image
        thermal_width_in_rgb = self.thermal_width * self.alignment['scale_x']
        thermal_height_in_rgb = self.thermal_height * self.alignment['scale_y']
        
        # Calculate offset to center thermal in RGB
        self.alignment['offset_x'] = rgb_center_x - thermal_width_in_rgb / 2
        self.alignment['offset_y'] = rgb_center_y - thermal_height_in_rgb / 2
        
        print(f"Initial alignment estimate:")
        print(f"  Thermal FOV in RGB: {thermal_width_in_rgb:.0f} x {thermal_height_in_rgb:.0f}")
        print(f"  Offset: ({self.alignment['offset_x']:.0f}, {self.alignment['offset_y']:.0f})")
        
        return self.alignment
    
    def load_thermal(self, frame_number):
        """Load thermal data"""
        irg_path = self.base_path / f"IRX_{frame_number:04d}.irg"
        
        if irg_path.exists():
            with open(irg_path, 'rb') as f:
                irg_data = f.read()
            
            # Parse thermal data
            expected_pixels = self.thermal_width * self.thermal_height
            pixel_data_size = expected_pixels * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((self.thermal_height, self.thermal_width))
            
            # Convert to Celsius
            temp_celsius = (raw_thermal / 10.0) - 273.15
            
            return temp_celsius
        
        return None
    
    def thermal_to_rgb_coords(self, thermal_x, thermal_y):
        """Convert thermal pixel coordinates to RGB pixel coordinates"""
        # Apply scale and offset
        rgb_x = thermal_x * self.alignment['scale_x'] + self.alignment['offset_x']
        rgb_y = thermal_y * self.alignment['scale_y'] + self.alignment['offset_y']
        
        # Apply rotation if needed
        if self.alignment['rotation'] != 0:
            # Rotation around center
            center_x = self.alignment['offset_x'] + self.thermal_width * self.alignment['scale_x'] / 2
            center_y = self.alignment['offset_y'] + self.thermal_height * self.alignment['scale_y'] / 2
            
            angle = np.radians(self.alignment['rotation'])
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            
            dx = rgb_x - center_x
            dy = rgb_y - center_y
            
            rgb_x = center_x + dx * cos_a - dy * sin_a
            rgb_y = center_y + dx * sin_a + dy * cos_a
        
        return rgb_x, rgb_y
    
    def get_thermal_bbox_in_rgb(self):
        """Get bounding box of thermal image in RGB coordinates"""
        # Four corners of thermal image
        corners = [
            (0, 0),
            (self.thermal_width, 0),
            (self.thermal_width, self.thermal_height),
            (0, self.thermal_height)
        ]
        
        # Convert to RGB coordinates
        rgb_corners = []
        for tx, ty in corners:
            rx, ry = self.thermal_to_rgb_coords(tx, ty)
            rgb_corners.append((rx, ry))
        
        return rgb_corners
    
    def extract_rgb_region_for_thermal(self, rgb_img):
        """Extract the RGB region that corresponds to thermal FOV"""
        # Get thermal bbox in RGB
        x_min = int(self.alignment['offset_x'])
        y_min = int(self.alignment['offset_y'])
        x_max = int(x_min + self.thermal_width * self.alignment['scale_x'])
        y_max = int(y_min + self.thermal_height * self.alignment['scale_y'])
        
        # Ensure within bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(self.rgb_width, x_max)
        y_max = min(self.rgb_height, y_max)
        
        # Extract region
        rgb_region = rgb_img[y_min:y_max, x_min:x_max]
        
        # Resize to match thermal dimensions
        rgb_resized = cv2.resize(rgb_region, (self.thermal_width, self.thermal_height))
        
        return rgb_resized
    
    def calibration_viewer(self):
        """Interactive viewer to calibrate alignment"""
        
        # Find a frame with both RGB and thermal
        frame_numbers = []
        for rgb_file in sorted(self.base_path.glob("MAX_*.JPG"))[:50]:
            frame_num = int(rgb_file.stem.split('_')[1])
            if (self.base_path / f"IRX_{frame_num:04d}.irg").exists():
                frame_numbers.append(frame_num)
        
        if not frame_numbers:
            print("No matching RGB-thermal pairs found!")
            return
        
        current_frame = frame_numbers[0]
        
        # Load images
        rgb_img = np.array(Image.open(self.base_path / f"MAX_{current_frame:04d}.JPG"))
        thermal = self.load_thermal(current_frame)
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Main display
        ax_main = plt.subplot(2, 3, (1, 5))
        ax_thermal = plt.subplot(2, 3, 3)
        ax_rgb_crop = plt.subplot(2, 3, 6)
        
        def update_display():
            ax_main.clear()
            ax_thermal.clear()
            ax_rgb_crop.clear()
            
            # Show RGB with thermal overlay box
            ax_main.imshow(rgb_img)
            
            # Draw thermal FOV rectangle
            rect = Rectangle(
                (self.alignment['offset_x'], self.alignment['offset_y']),
                self.thermal_width * self.alignment['scale_x'],
                self.thermal_height * self.alignment['scale_y'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax_main.add_patch(rect)
            
            ax_main.set_title(f'RGB with Thermal FOV (Frame {current_frame})')
            ax_main.axis('off')
            
            # Show thermal
            im = ax_thermal.imshow(thermal, cmap='hot')
            ax_thermal.set_title('Thermal Image')
            ax_thermal.axis('off')
            
            # Show cropped RGB region
            rgb_crop = self.extract_rgb_region_for_thermal(rgb_img)
            ax_rgb_crop.imshow(rgb_crop)
            ax_rgb_crop.set_title('RGB Region (resized to thermal)')
            ax_rgb_crop.axis('off')
            
            # Update title
            fig.suptitle(
                f'Thermal-RGB Alignment Calibration\n'
                f'Scale: ({self.alignment["scale_x"]:.2f}, {self.alignment["scale_y"]:.2f}), '
                f'Offset: ({self.alignment["offset_x"]:.0f}, {self.alignment["offset_y"]:.0f})',
                fontsize=14
            )
            
            fig.canvas.draw_idle()
        
        # Add sliders
        ax_scale_x = plt.axes([0.1, 0.15, 0.35, 0.02])
        slider_scale_x = Slider(ax_scale_x, 'Scale X', 4.0, 8.0, 
                               valinit=self.alignment['scale_x'], valstep=0.1)
        
        ax_scale_y = plt.axes([0.1, 0.12, 0.35, 0.02])
        slider_scale_y = Slider(ax_scale_y, 'Scale Y', 4.0, 8.0,
                               valinit=self.alignment['scale_y'], valstep=0.1)
        
        ax_offset_x = plt.axes([0.55, 0.15, 0.35, 0.02])
        slider_offset_x = Slider(ax_offset_x, 'Offset X', 0, self.rgb_width,
                                valinit=self.alignment['offset_x'], valstep=10)
        
        ax_offset_y = plt.axes([0.55, 0.12, 0.35, 0.02])
        slider_offset_y = Slider(ax_offset_y, 'Offset Y', 0, self.rgb_height,
                                valinit=self.alignment['offset_y'], valstep=10)
        
        ax_rotation = plt.axes([0.1, 0.09, 0.35, 0.02])
        slider_rotation = Slider(ax_rotation, 'Rotation', -10, 10,
                                valinit=self.alignment['rotation'], valstep=0.5)
        
        def update_alignment(val):
            self.alignment['scale_x'] = slider_scale_x.val
            self.alignment['scale_y'] = slider_scale_y.val
            self.alignment['offset_x'] = slider_offset_x.val
            self.alignment['offset_y'] = slider_offset_y.val
            self.alignment['rotation'] = slider_rotation.val
            update_display()
        
        slider_scale_x.on_changed(update_alignment)
        slider_scale_y.on_changed(update_alignment)
        slider_offset_x.on_changed(update_alignment)
        slider_offset_y.on_changed(update_alignment)
        slider_rotation.on_changed(update_alignment)
        
        # Add buttons
        ax_save = plt.axes([0.7, 0.05, 0.1, 0.03])
        btn_save = Button(ax_save, 'Save')
        
        ax_auto = plt.axes([0.82, 0.05, 0.1, 0.03])
        btn_auto = Button(ax_auto, 'Auto')
        
        def save_alignment(event):
            self.save_alignment()
        
        def auto_align(event):
            self.estimate_initial_alignment(current_frame)
            slider_offset_x.set_val(self.alignment['offset_x'])
            slider_offset_y.set_val(self.alignment['offset_y'])
            update_display()
        
        btn_save.on_clicked(save_alignment)
        btn_auto.on_clicked(auto_align)
        
        # Frame selector
        ax_frame = plt.axes([0.1, 0.05, 0.35, 0.02])
        slider_frame = Slider(ax_frame, 'Frame', 0, len(frame_numbers)-1,
                            valinit=0, valstep=1)
        
        def update_frame(val):
            nonlocal current_frame, rgb_img, thermal
            idx = int(slider_frame.val)
            current_frame = frame_numbers[idx]
            
            rgb_img = np.array(Image.open(self.base_path / f"MAX_{current_frame:04d}.JPG"))
            thermal = self.load_thermal(current_frame)
            update_display()
        
        slider_frame.on_changed(update_frame)
        
        # Initial display
        update_display()
        
        plt.show()
    
    def apply_alignment_to_data(self, frame_number):
        """
        Apply alignment to get properly cropped RGB for thermal
        """
        # Load RGB
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        if not rgb_path.exists():
            return None
        
        rgb_img = np.array(Image.open(rgb_path))
        
        # Extract aligned region
        rgb_aligned = self.extract_rgb_region_for_thermal(rgb_img)
        
        # Load thermal
        thermal = self.load_thermal(frame_number)
        
        return {
            'rgb_full': rgb_img,
            'rgb_aligned': rgb_aligned,
            'thermal': thermal,
            'alignment': self.alignment.copy()
        }


def test_alignment(frame_number=248):
    """Test alignment on a specific frame"""
    
    aligner = ThermalRGBAligner()
    
    # Apply alignment
    result = aligner.apply_alignment_to_data(frame_number)
    
    if result is None:
        print("Failed to load data")
        return
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Full RGB with thermal FOV box
    axes[0].imshow(result['rgb_full'])
    
    # Draw thermal FOV
    rect = Rectangle(
        (aligner.alignment['offset_x'], aligner.alignment['offset_y']),
        aligner.thermal_width * aligner.alignment['scale_x'],
        aligner.thermal_height * aligner.alignment['scale_y'],
        linewidth=2, edgecolor='red', facecolor='none'
    )
    axes[0].add_patch(rect)
    axes[0].set_title(f'Full RGB ({aligner.rgb_width}x{aligner.rgb_height})')
    axes[0].axis('off')
    
    # Aligned RGB region
    axes[1].imshow(result['rgb_aligned'])
    axes[1].set_title(f'Aligned RGB Region ({aligner.thermal_width}x{aligner.thermal_height})')
    axes[1].axis('off')
    
    # Thermal
    im = axes[2].imshow(result['thermal'], cmap='hot')
    axes[2].set_title(f'Thermal ({aligner.thermal_width}x{aligner.thermal_height})')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.suptitle(f'Frame {frame_number} - Thermal/RGB Alignment Test')
    plt.tight_layout()
    plt.show()
    
    print(f"\nAlignment parameters:")
    for key, val in aligner.alignment.items():
        print(f"  {key}: {val}")


def main():
    """Main entry point"""
    print("Thermal-RGB Image Alignment Tool")
    print("=" * 40)
    
    print("\nRGB dimensions: 4096 x 3072")
    print("Thermal dimensions: 640 x 512")
    print("Scale ratio: ~6.4x")
    
    print("\nOptions:")
    print("1. Calibration viewer (interactive alignment)")
    print("2. Test current alignment")
    print("3. Auto-estimate alignment")
    
    choice = input("\nChoice (1-3): ").strip()
    
    aligner = ThermalRGBAligner()
    
    if choice == '1':
        print("\nLaunching calibration viewer...")
        print("Adjust sliders to align thermal FOV with RGB")
        print("Click 'Save' to store alignment")
        aligner.calibration_viewer()
        
    elif choice == '2':
        frame = int(input("Enter frame number to test (default 248): ") or "248")
        test_alignment(frame)
        
    else:
        frame = int(input("Enter frame number for auto-alignment (default 248): ") or "248")
        aligner.estimate_initial_alignment(frame)
        aligner.save_alignment()
        test_alignment(frame)


if __name__ == "__main__":
    main()