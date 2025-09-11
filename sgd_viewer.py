#!/usr/bin/env python3
"""
SGD Detection Viewer with Aggregate Mapping

Handles overlapping frames and builds a comprehensive map of unique SGD locations.
Accounts for 90% overlap between consecutive drone images.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle
from sgd_detector_integrated import IntegratedSGDDetector
try:
    from sgd_georef_polygons import SGDPolygonGeoref
    POLYGON_SUPPORT = True
except ImportError:
    from sgd_georef import SGDGeoref
    POLYGON_SUPPORT = False
from pathlib import Path
import numpy as np
from datetime import datetime
import json
import os

class SGDAggregateViewer:
    """SGD viewer with deduplication and aggregate mapping"""
    
    def __init__(self, data_dir="data/100MEDIA",
                 aggregate_file="sgd_aggregate.json", 
                 distance_threshold=10.0,
                 ml_model_path="segmentation_model.pkl"):
        """
        Initialize viewer with aggregate tracking.
        
        Args:
            data_dir: Path to directory containing image files
            aggregate_file: Path to persistent aggregate data file
            distance_threshold: Meters - SGD within this distance are considered same location
            ml_model_path: Path to ML segmentation model (None to use rule-based)
        """
        print("SGD Aggregate Mapping Viewer")
        print("=" * 50)
        
        # Initialize components with specified ML model and data directory
        self.detector = IntegratedSGDDetector(
            base_path=data_dir,
            use_ml=ml_model_path is not None,
            ml_model_path=ml_model_path
        )
        # Use polygon georeferencing if available
        if POLYGON_SUPPORT:
            self.georef = SGDPolygonGeoref(base_path=data_dir)
            print("✓ Using enhanced polygon georeferencing")
        else:
            self.georef = SGDGeoref(base_path=data_dir, thermal_fov_ratio=0.7)
            print("○ Using point georeferencing (polygon module not found)")
        
        # Aggregate tracking
        self.aggregate_file = aggregate_file
        self.distance_threshold = distance_threshold  # meters
        self.unique_sgd_locations = []  # List of unique SGD locations
        self.frame_sgd_map = {}  # Map frame -> SGD indices
        
        # Load existing aggregate data if exists
        self.load_aggregate()
        
        # Find available frames
        self.frames = []
        for f in sorted(self.detector.base_path.glob("MAX_*.JPG"))[:250]:
            num = int(f.stem.split('_')[1])
            if (self.detector.base_path / f"IRX_{num:04d}.irg").exists():
                self.frames.append(num)
        
        if not self.frames:
            raise FileNotFoundError("No paired RGB-thermal frames found!")
        
        print(f"Found {len(self.frames)} frames: {self.frames[0]} to {self.frames[-1]}")
        print(f"Existing unique SGD locations: {len(self.unique_sgd_locations)}")
        
        # State variables
        self.current_idx = 0
        self.temp_threshold = 1.0
        self.min_area = 50
        self.include_waves = False  # Toggle for including wave areas in SGD search
        self.current_result = None
        self.current_new_sgd = []  # New SGD in current frame
        self.current_existing_sgd = []  # Previously detected SGD
        
        # Statistics
        self.frames_processed = set()
        self.total_detections = 0
        
        # Create figure
        self.fig = plt.figure(figsize=(18, 10))
        self.setup_layout()
        
        # Initial display
        self.update_display()
    
    def setup_layout(self):
        """Setup the display layout"""
        # Main display grid - increase bottom margin for controls
        gs = self.fig.add_gridspec(2, 4, left=0.05, right=0.95, top=0.92, bottom=0.20,
                                   hspace=0.3, wspace=0.3)
        
        self.axes = []
        self.axes.append(self.fig.add_subplot(gs[0, 0]))  # RGB
        self.axes.append(self.fig.add_subplot(gs[0, 1]))  # Segmentation
        self.axes.append(self.fig.add_subplot(gs[0, 2]))  # Thermal
        self.axes.append(self.fig.add_subplot(gs[0, 3]))  # Ocean thermal
        self.axes.append(self.fig.add_subplot(gs[1, 0]))  # SGD detection
        self.axes.append(self.fig.add_subplot(gs[1, 1]))  # New vs Existing
        self.axes.append(self.fig.add_subplot(gs[1, 2]))  # Coverage map
        self.axes.append(self.fig.add_subplot(gs[1, 3]))  # Statistics
        
        # Control buttons - enhanced navigation stacked at bottom
        btn_height = 0.03
        btn_width = 0.06
        bottom_margin = 0.02
        
        # Row 1 (upper) - Fine navigation and main actions
        row1_y = bottom_margin + 2*btn_height + 0.01
        
        ax_prev = plt.axes([0.08, row1_y, btn_width, btn_height])
        ax_next = plt.axes([0.15, row1_y, btn_width, btn_height])
        ax_minus10 = plt.axes([0.23, row1_y, btn_width, btn_height])
        ax_plus10 = plt.axes([0.30, row1_y, btn_width, btn_height])
        ax_first = plt.axes([0.38, row1_y, btn_width, btn_height])
        ax_last = plt.axes([0.45, row1_y, btn_width, btn_height])
        
        # Action buttons on the right
        ax_mark = plt.axes([0.50, row1_y, 0.07, btn_height])
        ax_waves = plt.axes([0.58, row1_y, 0.06, btn_height])
        ax_save = plt.axes([0.65, row1_y, 0.06, btn_height])
        ax_export = plt.axes([0.72, row1_y, 0.06, btn_height])
        ax_new = plt.axes([0.79, row1_y, 0.07, btn_height])
        
        # Row 2 (middle) - Medium and large jumps
        row2_y = bottom_margin + btn_height + 0.005
        
        ax_minus5 = plt.axes([0.08, row2_y, btn_width, btn_height])
        ax_plus5 = plt.axes([0.15, row2_y, btn_width, btn_height])
        ax_minus25 = plt.axes([0.23, row2_y, btn_width, btn_height])
        ax_plus25 = plt.axes([0.30, row2_y, btn_width, btn_height])
        
        # Frame counter (adjusted position)
        ax_info = plt.axes([0.47, row1_y, 0.08, btn_height])
        ax_info.axis('off')
        self.frame_info = ax_info.text(0.5, 0.5, f'Frame 1/{len(self.frames)}',
                                       ha='center', va='center', fontsize=9)
        
        # Create buttons
        self.btn_prev = Button(ax_prev, '← Prev')
        self.btn_next = Button(ax_next, 'Next →')
        self.btn_minus5 = Button(ax_minus5, '← -5')
        self.btn_plus5 = Button(ax_plus5, '+5 →')
        self.btn_minus10 = Button(ax_minus10, '← -10')
        self.btn_plus10 = Button(ax_plus10, '+10 →')
        self.btn_minus25 = Button(ax_minus25, '← -25')
        self.btn_plus25 = Button(ax_plus25, '+25 →')
        self.btn_first = Button(ax_first, 'First')
        self.btn_last = Button(ax_last, 'Last')
        
        self.btn_mark = Button(ax_mark, 'Mark SGD')
        self.btn_waves = Button(ax_waves, 'Waves')
        self.btn_save = Button(ax_save, 'Save')
        self.btn_export = Button(ax_export, 'Export')
        self.btn_new = Button(ax_new, 'New Agg')
        
        # Row 3 (bottom) - Parameter sliders
        row3_y = bottom_margin
        
        ax_temp = plt.axes([0.08, row3_y, 0.20, btn_height])
        ax_area = plt.axes([0.38, row3_y, 0.20, btn_height])
        ax_dist = plt.axes([0.68, row3_y, 0.20, btn_height])
        
        self.slider_temp = Slider(ax_temp, 'Temp (°C)', 0.5, 3.0, valinit=self.temp_threshold)
        self.slider_area = Slider(ax_area, 'Min Area', 10, 200, valinit=self.min_area, valstep=10)
        self.slider_dist = Slider(ax_dist, 'Merge Dist (m)', 5, 50, valinit=self.distance_threshold)
        
        # Connect callbacks
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        self.btn_minus5.on_clicked(lambda e: self.navigate(-5))
        self.btn_plus5.on_clicked(lambda e: self.navigate(5))
        self.btn_minus10.on_clicked(lambda e: self.navigate(-10))
        self.btn_plus10.on_clicked(lambda e: self.navigate(10))
        self.btn_minus25.on_clicked(lambda e: self.navigate(-25))
        self.btn_plus25.on_clicked(lambda e: self.navigate(25))
        self.btn_first.on_clicked(lambda e: self.first_frame())
        self.btn_last.on_clicked(lambda e: self.last_frame())
        self.btn_mark.on_clicked(self.mark_sgd)
        self.btn_waves.on_clicked(self.toggle_waves)
        self.btn_save.on_clicked(self.save_aggregate)
        self.btn_export.on_clicked(self.export_aggregate_map)
        self.btn_new.on_clicked(self.new_aggregate)
        
        self.slider_temp.on_changed(self.update_temp)
        self.slider_area.on_changed(self.update_area)
        self.slider_dist.on_changed(self.update_distance)
        
        # Keyboard
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Add labels for control sections
        self.fig.text(0.02, row1_y + 0.01, 'Nav:', fontsize=8, fontweight='bold')
        self.fig.text(0.02, row3_y + 0.01, 'Params:', fontsize=8, fontweight='bold')
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in meters"""
        # Simple equirectangular approximation
        R = 6371000  # Earth radius in meters
        lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        x = delta_lon * np.cos((lat1_rad + lat2_rad) / 2)
        y = delta_lat
        
        return R * np.sqrt(x*x + y*y)
    
    def find_nearby_sgd(self, lat, lon):
        """Find if this SGD location already exists nearby"""
        for i, existing in enumerate(self.unique_sgd_locations):
            dist = self.calculate_distance(
                lat, lon, 
                existing['latitude'], existing['longitude']
            )
            if dist < self.distance_threshold:
                return i
        return None
    
    def update_display(self):
        """Update display with current frame"""
        frame_num = self.frames[self.current_idx]
        
        # Update detector parameters
        self.detector.temp_threshold = self.temp_threshold
        self.detector.min_area = self.min_area
        
        # Process frame with wave inclusion setting
        print(f"\nProcessing frame {frame_num}...")
        self.current_result = self.detector.process_frame(
            frame_num, 
            visualize=False,
            include_waves=self.include_waves
        )
        
        # Check for new vs existing SGD
        self.current_new_sgd = []
        self.current_existing_sgd = []
        
        if self.current_result['plume_info']:
            # Get georeferenced locations with polygons if available
            if POLYGON_SUPPORT:
                temp_georef = SGDPolygonGeoref(base_path=self.detector.base_path)
                locations = temp_georef.process_frame_with_polygons(frame_num, self.current_result['plume_info'])
            else:
                temp_georef = SGDGeoref(base_path=self.detector.base_path, thermal_fov_ratio=0.7)
                locations = temp_georef.process_frame(frame_num, self.current_result['plume_info'])
            
            for loc in locations:
                # Check if near existing SGD
                # Handle both point and polygon formats
                if 'centroid' in loc:
                    lat = loc['centroid']['latitude']
                    lon = loc['centroid']['longitude']
                else:
                    lat = loc['latitude']
                    lon = loc['longitude']
                
                nearby_idx = self.find_nearby_sgd(lat, lon)
                
                # Update location with lat/lon for consistency
                loc['latitude'] = lat
                loc['longitude'] = lon
                
                if nearby_idx is not None:
                    self.current_existing_sgd.append((loc, nearby_idx))
                    # Update existing with better data if needed
                    if loc['area_m2'] > self.unique_sgd_locations[nearby_idx]['area_m2']:
                        self.unique_sgd_locations[nearby_idx]['area_m2'] = loc['area_m2']
                else:
                    self.current_new_sgd.append(loc)
        
        # Clear axes
        for ax in self.axes:
            ax.clear()
        
        # Display panels
        self.display_panels(frame_num)
        
        # Update title
        new_count = len(self.current_new_sgd)
        existing_count = len(self.current_existing_sgd)
        total_unique = len(self.unique_sgd_locations)
        
        title = f"Frame {frame_num} ({self.current_idx+1}/{len(self.frames)}) | "
        title += f"NEW: {new_count} | Existing: {existing_count} | "
        title += f"Total Unique SGD: {total_unique}"
        self.fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.draw()
    
    def display_panels(self, frame_num):
        """Display all panels"""
        result = self.current_result
        
        # 1. RGB
        self.axes[0].imshow(result['data']['rgb_aligned'])
        self.axes[0].set_title('RGB Image')
        self.axes[0].axis('off')
        
        # 2. Segmentation
        mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
        mask_display[result['masks']['ocean']] = [0, 0.3, 1]
        mask_display[result['masks']['land']] = [0, 0.5, 0]
        mask_display[result['masks']['waves']] = [1, 1, 1]
        self.axes[1].imshow(mask_display)
        self.axes[1].set_title('Ocean/Land/Waves')
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
        
        # 5. SGD detection
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        if result['sgd_mask'].any():
            sgd_display[result['sgd_mask']] = [0, 1, 1]
        shoreline, _ = self.detector.detect_shoreline(result['masks'])
        sgd_display[shoreline] = [1, 1, 0]
        self.axes[4].imshow(sgd_display)
        self.axes[4].set_title(f'Detected: {len(result["plume_info"])} plumes')
        self.axes[4].axis('off')
        
        # 6. New vs Existing overlay
        overlay = result['data']['rgb_aligned'].copy() / 255.0
        if result['sgd_mask'].any() and result['plume_info']:
            # Color code: Green = new, Yellow = existing
            for plume in result['plume_info']:
                # Check if this plume is new or existing
                cy, cx = plume['centroid']
                is_new = any(abs(n['frame'] - frame_num) < 1 for n in self.current_new_sgd)
                
                # Create a small circle around centroid
                y, x = np.ogrid[:overlay.shape[0], :overlay.shape[1]]
                mask = (x - cx)**2 + (y - cy)**2 <= 25
                
                if is_new:
                    overlay[mask] = [0, 1, 0]  # Green for new
                else:
                    overlay[mask] = [1, 1, 0]  # Yellow for existing
        
        self.axes[5].imshow(overlay)
        self.axes[5].set_title(f'New (green) vs Existing (yellow)')
        self.axes[5].axis('off')
        
        # 7. Coverage map
        self.axes[6].set_title('Coverage Map')
        if self.unique_sgd_locations:
            lats = [s['latitude'] for s in self.unique_sgd_locations]
            lons = [s['longitude'] for s in self.unique_sgd_locations]
            sizes = [s['area_m2'] for s in self.unique_sgd_locations]
            
            # Normalize sizes for display
            sizes_norm = np.array(sizes)
            sizes_norm = 20 + (sizes_norm / max(sizes_norm)) * 200 if max(sizes_norm) > 0 else [20]
            
            self.axes[6].scatter(lons, lats, s=sizes_norm, c=sizes, 
                                cmap='YlOrRd', alpha=0.6, edgecolors='black')
            self.axes[6].set_xlabel('Longitude')
            self.axes[6].set_ylabel('Latitude')
            self.axes[6].grid(True, alpha=0.3)
        else:
            self.axes[6].text(0.5, 0.5, 'No SGD mapped yet', 
                             ha='center', va='center', fontsize=12)
            self.axes[6].set_xlim(0, 1)
            self.axes[6].set_ylim(0, 1)
        
        # 8. Statistics
        self.axes[7].axis('off')
        stats = f"STATISTICS\n" + "="*20 + "\n"
        stats += f"Frames processed: {len(self.frames_processed)}\n"
        stats += f"Total detections: {self.total_detections}\n"
        stats += f"Unique SGD sites: {len(self.unique_sgd_locations)}\n"
        stats += f"Merge distance: {self.distance_threshold:.0f} m\n\n"
        
        if self.unique_sgd_locations:
            areas = [s['area_m2'] for s in self.unique_sgd_locations]
            stats += f"Area range: {min(areas):.1f} - {max(areas):.1f} m²\n"
            stats += f"Mean area: {np.mean(areas):.1f} m²\n"
            stats += f"Total area: {sum(areas):.1f} m²\n"
        
        stats += f"\nCurrent frame:\n"
        stats += f"  New SGD: {len(self.current_new_sgd)}\n"
        stats += f"  Existing: {len(self.current_existing_sgd)}"
        
        self.axes[7].text(0.05, 0.95, stats, transform=self.axes[7].transAxes,
                         fontsize=10, family='monospace', va='top')
    
    def toggle_waves(self, event):
        """Toggle inclusion of wave areas in SGD search"""
        self.include_waves = not self.include_waves
        
        # Update button appearance
        if self.include_waves:
            self.btn_waves.label.set_text('Waves ✓')
            self.btn_waves.color = 'lightblue'
            print("Including wave areas in SGD search")
        else:
            self.btn_waves.label.set_text('Waves')
            self.btn_waves.color = '0.85'
            print("Excluding wave areas from SGD search")
        
        # Reprocess current frame
        self.update_display()
    
    def clear_frame_sgd(self, event):
        """Clear all SGDs from the current frame"""
        frame_num = self.frames[self.current_idx]
        
        # Find and remove all SGDs from this frame
        original_count = len(self.unique_sgd_locations)
        self.unique_sgd_locations = [
            sgd for sgd in self.unique_sgd_locations 
            if sgd.get('frame') != frame_num and sgd.get('first_frame') != frame_num
        ]
        removed_count = original_count - len(self.unique_sgd_locations)
        
        if removed_count > 0:
            # Remove frame from processed list
            self.frames_processed.discard(frame_num)
            
            print(f"Cleared {removed_count} SGD(s) from frame {frame_num}")
            print(f"Total unique SGD: {len(self.unique_sgd_locations)}")
            
            # Save and update display
            self.save_aggregate(None)
            self.update_display()
        else:
            print(f"No SGDs to clear from frame {frame_num}")
    
    def mark_sgd(self, event):
        """Mark current frame's new SGD as confirmed"""
        frame_num = self.frames[self.current_idx]
        
        # If no new SGDs but there are existing ones from this frame, allow re-marking
        if not self.current_new_sgd:
            if self.current_existing_sgd:
                print(f"Frame {frame_num} already processed - {len(self.current_existing_sgd)} SGDs previously marked")
                print("To update, delete old entries or use 'New Agg' to start fresh")
            else:
                print("No SGD detections to mark in this frame")
            return
        
        # Add new SGD to unique locations
        for sgd in self.current_new_sgd:
            sgd['first_frame'] = frame_num
            sgd['confirmed'] = True
            self.unique_sgd_locations.append(sgd)
        
        # Track this frame as processed
        self.frames_processed.add(frame_num)
        self.total_detections += len(self.current_new_sgd)
        
        print(f"Marked {len(self.current_new_sgd)} new SGD locations")
        print(f"Total unique SGD: {len(self.unique_sgd_locations)}")
        
        # Clear current new SGD
        self.current_new_sgd = []
        
        # Save automatically
        self.save_aggregate(None)
        
        # Update display
        self.update_display()
    
    def save_aggregate(self, event):
        """Save aggregate data to file"""
        data = {
            'metadata': {
                'last_updated': datetime.now().isoformat(),
                'total_frames': len(self.frames),
                'frames_processed': list(self.frames_processed),
                'distance_threshold': self.distance_threshold
            },
            'sgd_locations': self.unique_sgd_locations,
            'statistics': {
                'total_unique': len(self.unique_sgd_locations),
                'total_detections': self.total_detections,
                'total_area_m2': sum(s['area_m2'] for s in self.unique_sgd_locations)
            }
        }
        
        with open(self.aggregate_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved aggregate data to {self.aggregate_file}")
    
    def load_aggregate(self):
        """Load existing aggregate data"""
        if os.path.exists(self.aggregate_file):
            with open(self.aggregate_file, 'r') as f:
                data = json.load(f)
            
            self.unique_sgd_locations = data.get('sgd_locations', [])
            self.frames_processed = set(data.get('metadata', {}).get('frames_processed', []))
            self.total_detections = data.get('statistics', {}).get('total_detections', 0)
            
            print(f"Loaded {len(self.unique_sgd_locations)} existing SGD locations")
    
    def export_aggregate_map(self, event):
        """Export aggregate SGD map to multiple formats"""
        if not self.unique_sgd_locations:
            print("No SGD locations to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"sgd_map_aggregate_{timestamp}"
        
        # Prepare georef with all locations
        if POLYGON_SUPPORT:
            # Use polygon export with KML support
            self.georef.sgd_polygons = self.unique_sgd_locations
            geojson_file = self.georef.export_geojson_polygons(f"{base_name}_polygons.geojson")
            csv_file = self.georef.export_csv_with_areas(f"{base_name}_areas.csv")
            kml_file = self.georef.export_kml_polygons(f"{base_name}_polygons.kml")
        else:
            self.georef.sgd_locations = self.unique_sgd_locations
            geojson_file = self.georef.export_geojson(f"{base_name}.geojson")
            csv_file = self.georef.export_csv(f"{base_name}.csv")
            kml_file = self.georef.export_kml(f"{base_name}.kml")
        
        print(f"\nExported aggregate map with {len(self.unique_sgd_locations)} unique SGD:")
        print(f"  - {geojson_file}")
        print(f"  - {csv_file}")
        print(f"  - {kml_file}")
    
    def new_aggregate(self, event):
        """Start a new aggregate file, clearing all existing data"""
        from datetime import datetime
        
        # Ask for confirmation (using simple print since we don't have GUI dialog)
        print("\n" + "="*50)
        print("NEW AGGREGATE FILE")
        print("="*50)
        print("This will clear all existing SGD locations and start fresh.")
        print("The current data will be saved with a timestamp.")
        
        # Save current data with timestamp if there's any
        if self.unique_sgd_locations:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"sgd_aggregate_backup_{timestamp}.json"
            
            # Save backup
            data = {
                'metadata': {
                    'last_updated': datetime.now().isoformat(),
                    'total_frames': len(self.frames),
                    'frames_processed': list(self.frames_processed),
                    'distance_threshold': self.distance_threshold
                },
                'sgd_locations': self.unique_sgd_locations,
                'statistics': {
                    'total_unique': len(self.unique_sgd_locations),
                    'total_detections': self.total_detections,
                    'total_area_m2': sum(s['area_m2'] for s in self.unique_sgd_locations)
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Current data backed up to: {backup_file}")
        
        # Clear all aggregate data
        self.unique_sgd_locations = []
        self.frames_processed = set()
        self.total_detections = 0
        self.frame_sgd_map = {}
        self.current_new_sgd = []
        self.current_existing_sgd = []
        
        # Save empty aggregate file
        self.save_aggregate(None)
        
        # Reset georeferencer
        if POLYGON_SUPPORT:
            self.georef.sgd_polygons = []
        else:
            self.georef.sgd_locations = []
        
        print(f"✓ Started new aggregate file: {self.aggregate_file}")
        print(f"✓ All SGD locations cleared")
        print(f"✓ Ready to mark new SGD locations")
        print("="*50)
        
        # Update display
        self.update_display()
    
    def navigate(self, step):
        """Navigate by step frames"""
        new_idx = self.current_idx + step
        if 0 <= new_idx < len(self.frames):
            self.current_idx = new_idx
            self.frame_info.set_text(f'Frame {self.current_idx+1}/{len(self.frames)}')
            self.update_display()
    
    def next_frame(self, event):
        self.navigate(1)
    
    def prev_frame(self, event):
        self.navigate(-1)
    
    def first_frame(self):
        if self.current_idx != 0:
            self.current_idx = 0
            self.frame_info.set_text(f'Frame {self.current_idx+1}/{len(self.frames)}')
            self.update_display()
    
    def last_frame(self):
        last_idx = len(self.frames) - 1
        if self.current_idx != last_idx:
            self.current_idx = last_idx
            self.frame_info.set_text(f'Frame {self.current_idx+1}/{len(self.frames)}')
            self.update_display()
    
    def update_temp(self, val):
        self.temp_threshold = val
        self.update_display()
    
    def update_area(self, val):
        self.min_area = int(val)
        self.update_display()
    
    def update_distance(self, val):
        self.distance_threshold = val
        # Reprocess current frame with new threshold
        self.update_display()
    
    def on_key(self, event):
        if event.key == 'right':
            self.navigate(1)
        elif event.key == 'left':
            self.navigate(-1)
        elif event.key == 'home':
            self.first_frame()
        elif event.key == 'end':
            self.last_frame()
        elif event.key == 'm':
            self.mark_sgd(None)
        elif event.key == 'w':
            self.toggle_waves(None)
        elif event.key == 'c':
            self.clear_frame_sgd(None)
        elif event.key == 's':
            self.save_aggregate(None)
        elif event.key == 'e':
            self.export_aggregate_map(None)
        elif event.key == 'n':
            self.new_aggregate(None)
    
    def run(self):
        """Run the viewer"""
        print("\n" + "="*60)
        print("AGGREGATE SGD MAPPING CONTROLS:")
        print("  Navigation:")
        print("    - Prev/Next (±1), ±5, ±10, ±25, First/Last buttons")
        print("    - Keyboard: ← → arrows, Home/End keys")
        print("  Mark & Save:")
        print("    - 'Mark SGD' button or 'M' key: Confirm new SGD")
        print("    - 'Waves' button or 'W' key: Toggle wave area inclusion")
        print("    - 'C' key: Clear SGDs from current frame (for re-processing)")
        print("    - 'Save' button or 'S' key: Save progress")
        print("    - 'Export' button or 'E' key: Export aggregate map")
        print("    - 'New Agg' button or 'N' key: Start new aggregate file")
        print("  Visual Indicators:")
        print("    - GREEN highlights: New SGD locations")
        print("    - YELLOW highlights: Previously detected SGD")
        print("  Parameter Sliders:")
        print("    - Temp (°C): Temperature threshold for SGD")
        print("    - Min Area: Minimum plume size")
        print("    - Merge Dist: Distance to merge SGD locations")
        print("="*60)
        
        plt.show()

def main():
    """Main entry point with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SGD Detection Viewer with Aggregate Mapping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default settings
  python sgd_viewer.py
  
  # Use different data directory
  python sgd_viewer.py --data /path/to/survey/images
  
  # Use custom ML model for different conditions
  python sgd_viewer.py --model rocky_shore_model.pkl
  
  # Create new aggregate file for different survey
  python sgd_viewer.py --data data/survey2 --aggregate survey2_aggregate.json
  
  # Disable ML segmentation (use rule-based)
  python sgd_viewer.py --no-ml
  
  # Combine options for complete survey setup
  python sgd_viewer.py --data /drone/flight3 --model sunrise_model.pkl --aggregate flight3.json
        """
    )
    
    parser.add_argument('--data', type=str, default='data/100MEDIA',
                       help='Path to data directory containing MAX_*.JPG and IRX_*.irg files (default: data/100MEDIA)')
    parser.add_argument('--model', type=str, default='segmentation_model.pkl',
                       help='Path to ML segmentation model (default: segmentation_model.pkl)')
    parser.add_argument('--no-ml', action='store_true',
                       help='Disable ML segmentation, use rule-based instead')
    parser.add_argument('--aggregate', type=str, default='sgd_aggregate.json',
                       help='Path to aggregate data file (default: sgd_aggregate.json)')
    parser.add_argument('--distance', type=float, default=10.0,
                       help='Merge distance in meters for duplicate SGD (default: 10.0)')
    
    args = parser.parse_args()
    
    # Determine ML model path
    ml_model_path = None if args.no_ml else args.model
    
    if ml_model_path:
        print(f"Using ML model: {ml_model_path}")
    else:
        print("Using rule-based segmentation (ML disabled)")
    
    print(f"Data directory: {args.data}")
    print(f"Aggregate file: {args.aggregate}")
    print(f"Merge distance: {args.distance} meters")
    print()
    
    try:
        viewer = SGDAggregateViewer(
            data_dir=args.data,
            aggregate_file=args.aggregate,
            distance_threshold=args.distance,
            ml_model_path=ml_model_path
        )
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()