#!/usr/bin/env python3
"""
Interactive SGD Detection Viewer with Georeferencing Export

This is the main interactive tool for detecting submarine groundwater discharge
in thermal drone imagery and exporting georeferenced results.
"""

import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sgd_detector_integrated import IntegratedSGDDetector
from sgd_georef import SGDGeoref
from pathlib import Path
import numpy as np
from datetime import datetime

class SGDViewer:
    """Interactive viewer for SGD detection with export capabilities"""
    
    def __init__(self):
        """Initialize the viewer"""
        print("SGD Detection Viewer with Georeferencing")
        print("=" * 50)
        
        # Initialize detector and georeferencer
        self.detector = IntegratedSGDDetector()
        self.georef = SGDGeoref(thermal_fov_ratio=0.7)
        
        # Find available frames
        self.frames = []
        for f in sorted(self.detector.base_path.glob("MAX_*.JPG"))[:50]:
            num = int(f.stem.split('_')[1])
            if (self.detector.base_path / f"IRX_{num:04d}.irg").exists():
                self.frames.append(num)
        
        if not self.frames:
            raise FileNotFoundError("No paired RGB-thermal frames found!")
        
        print(f"Found {len(self.frames)} frames: {self.frames[0]} to {self.frames[-1]}")
        
        # State variables
        self.current_idx = 0
        self.temp_threshold = 1.0
        self.min_area = 50
        self.current_result = None
        self.all_detections = {}  # Store all processed frames
        
        # Create figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('SGD Detection Viewer', fontsize=16)
        self.axes = self.axes.flatten()
        
        # Setup controls
        self.setup_controls()
        
        # Initial display
        self.update_display()
        
    def setup_controls(self):
        """Setup GUI controls"""
        # Navigation buttons
        ax_prev = plt.axes([0.15, 0.02, 0.08, 0.04])
        ax_next = plt.axes([0.24, 0.02, 0.08, 0.04])
        ax_export = plt.axes([0.35, 0.02, 0.12, 0.04])
        ax_export_all = plt.axes([0.48, 0.02, 0.12, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_export = Button(ax_export, 'Export Frame')
        self.btn_export_all = Button(ax_export_all, 'Export All')
        
        # Parameter sliders
        ax_temp = plt.axes([0.15, 0.10, 0.30, 0.03])
        ax_area = plt.axes([0.15, 0.06, 0.30, 0.03])
        
        self.slider_temp = Slider(ax_temp, 'Temp Threshold (°C)', 0.5, 3.0, valinit=1.0)
        self.slider_area = Slider(ax_area, 'Min Area (px)', 10, 200, valinit=50, valstep=10)
        
        # Info text area
        ax_info = plt.axes([0.65, 0.02, 0.33, 0.12])
        ax_info.axis('off')
        self.info_text = ax_info.text(0, 0.5, '', fontsize=10, family='monospace')
        
        # Connect callbacks
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_export.on_clicked(self.export_current)
        self.btn_export_all.on_clicked(self.export_all)
        self.slider_temp.on_changed(self.update_temp)
        self.slider_area.on_changed(self.update_area)
        
        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
    
    def update_display(self):
        """Update the display with current frame and parameters"""
        frame_num = self.frames[self.current_idx]
        
        # Update detector parameters
        self.detector.temp_threshold = self.temp_threshold
        self.detector.min_area = self.min_area
        
        # Process frame
        print(f"\nProcessing frame {frame_num}...")
        self.current_result = self.detector.process_frame(frame_num, visualize=False)
        
        # Store detection for export
        if self.current_result['plume_info']:
            self.all_detections[frame_num] = self.current_result['plume_info']
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Display results
        result = self.current_result
        
        # 1. RGB aligned
        self.axes[0].imshow(result['data']['rgb_aligned'])
        self.axes[0].set_title(f'Frame {frame_num}: RGB')
        self.axes[0].axis('off')
        
        # 2. Segmentation
        mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
        mask_display[result['masks']['ocean']] = [0, 0.3, 1]
        mask_display[result['masks']['land']] = [0, 0.5, 0]
        mask_display[result['masks']['waves']] = [1, 1, 1]
        self.axes[1].imshow(mask_display)
        self.axes[1].set_title('Segmentation')
        self.axes[1].axis('off')
        
        # 3. Thermal
        thermal = result['data']['thermal']
        self.axes[2].imshow(thermal, cmap='RdYlBu_r')
        self.axes[2].set_title(f'Thermal ({thermal.min():.1f}-{thermal.max():.1f}°C)')
        self.axes[2].axis('off')
        
        # 4. Ocean thermal
        ocean_thermal = thermal.copy()
        ocean_thermal[~result['masks']['ocean']] = np.nan
        self.axes[3].imshow(ocean_thermal, cmap='viridis')
        self.axes[3].set_title(f'Ocean (median: {np.nanmedian(ocean_thermal):.1f}°C)')
        self.axes[3].axis('off')
        
        # 5. SGD detections
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        if result['sgd_mask'].any():
            sgd_display[result['sgd_mask']] = [0, 1, 1]
        shoreline, _ = self.detector.detect_shoreline(result['masks'])
        sgd_display[shoreline] = [1, 1, 0]
        self.axes[4].imshow(sgd_display)
        self.axes[4].set_title(f'SGD: {len(result["plume_info"])} plumes')
        self.axes[4].axis('off')
        
        # 6. Overlay
        overlay = result['data']['rgb_aligned'].copy() / 255.0
        if result['sgd_mask'].any():
            overlay[result['sgd_mask']] = [0, 1, 0]
        self.axes[5].imshow(overlay)
        self.axes[5].set_title('RGB + SGD')
        self.axes[5].axis('off')
        
        # Update info
        self.update_info()
        
        plt.draw()
    
    def update_info(self):
        """Update info text panel"""
        frame_num = self.frames[self.current_idx]
        info = f"Frame: {frame_num} ({self.current_idx + 1}/{len(self.frames)})\n"
        info += f"Parameters:\n"
        info += f"  Temp threshold: {self.temp_threshold:.1f}°C\n"
        info += f"  Min area: {self.min_area} px\n\n"
        
        if self.current_result['plume_info']:
            info += f"Detections: {len(self.current_result['plume_info'])} plumes\n"
            total_area = sum(p['area_pixels'] for p in self.current_result['plume_info'])
            info += f"Total area: {total_area} pixels\n"
            
            # Show first 3 plumes
            for i, plume in enumerate(self.current_result['plume_info'][:3], 1):
                info += f"  {i}. {plume['area_pixels']} px"
                if 'mean_temp_diff' in plume:
                    info += f", {plume['mean_temp_diff']:.1f}°C"
                info += "\n"
        else:
            info += "No SGD detected\n"
        
        info += f"\nTotal frames with SGD: {len(self.all_detections)}"
        
        self.info_text.set_text(info)
    
    def export_current(self, event):
        """Export current frame's detections"""
        frame_num = self.frames[self.current_idx]
        
        if not self.current_result['plume_info']:
            print(f"No SGD detected in frame {frame_num} to export")
            return
        
        # Clear previous georef data
        self.georef.sgd_locations = []
        
        # Process current frame
        locations = self.georef.process_frame(frame_num, self.current_result['plume_info'])
        
        if locations:
            # Generate filenames with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"sgd_frame_{frame_num}_{timestamp}"
            
            # Export all formats
            geojson_file = self.georef.export_geojson(f"{base_name}.geojson")
            csv_file = self.georef.export_csv(f"{base_name}.csv")
            kml_file = self.georef.export_kml(f"{base_name}.kml")
            
            print(f"\nExported frame {frame_num} with {len(locations)} SGD locations:")
            print(f"  - {geojson_file}")
            print(f"  - {csv_file}")
            print(f"  - {kml_file}")
    
    def export_all(self, event):
        """Export all detected SGD across all processed frames"""
        if not self.all_detections:
            print("No SGD detections to export")
            return
        
        # Clear and reprocess all
        self.georef.sgd_locations = []
        
        total_plumes = 0
        for frame_num, plume_list in self.all_detections.items():
            locations = self.georef.process_frame(frame_num, plume_list)
            total_plumes += len(locations)
        
        if self.georef.sgd_locations:
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"sgd_all_{timestamp}"
            
            # Export all formats
            geojson_file = self.georef.export_geojson(f"{base_name}.geojson")
            csv_file = self.georef.export_csv(f"{base_name}.csv")
            kml_file = self.georef.export_kml(f"{base_name}.kml")
            
            print(f"\nExported {total_plumes} SGD locations from {len(self.all_detections)} frames:")
            print(f"  - {geojson_file}")
            print(f"  - {csv_file}")
            print(f"  - {kml_file}")
    
    def next_frame(self, event):
        """Go to next frame"""
        if self.current_idx < len(self.frames) - 1:
            self.current_idx += 1
            self.update_display()
    
    def prev_frame(self, event):
        """Go to previous frame"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
    
    def update_temp(self, val):
        """Update temperature threshold"""
        self.temp_threshold = val
        self.update_display()
    
    def update_area(self, val):
        """Update minimum area"""
        self.min_area = int(val)
        self.update_display()
    
    def on_key(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right':
            self.next_frame(None)
        elif event.key == 'left':
            self.prev_frame(None)
        elif event.key == 'e':
            self.export_current(None)
        elif event.key == 'a':
            self.export_all(None)
    
    def run(self):
        """Run the interactive viewer"""
        print("\n" + "="*50)
        print("CONTROLS:")
        print("  Mouse:")
        print("    - Click buttons to navigate and export")
        print("    - Adjust sliders to tune detection")
        print("  Keyboard:")
        print("    - Left/Right arrows: Navigate frames")
        print("    - E: Export current frame")
        print("    - A: Export all detected SGD")
        print("="*50)
        print("\nClose window to exit.")
        
        plt.show()

def main():
    """Main entry point"""
    try:
        viewer = SGDViewer()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()