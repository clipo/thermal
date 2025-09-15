#!/usr/bin/env python3
"""
SGD Automated Detection Script

Batch processes thermal/RGB image pairs to automatically detect and map
submarine groundwater discharge (SGD) locations. Outputs georeferenced
polygons in KML format for visualization in Google Earth.

Usage:
    python sgd_autodetect.py --data data/survey --output survey_sgd.kml
    
Author: SGD Detection Toolkit
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import time

# Import core detection modules
from sgd_detector_integrated import IntegratedSGDDetector
from sgd_detector_improved import ImprovedSGDDetector
try:
    from sgd_georef_polygons import SGDPolygonGeoref
    POLYGON_SUPPORT = True
except ImportError:
    print("Warning: Polygon support not available, using point-based georeferencing")
    from sgd_georef import SGDGeoref
    POLYGON_SUPPORT = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and datetime"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class SGDAutoDetector:
    """Automated SGD detection with batch processing"""
    
    def __init__(self, data_dir, output_file,
                 temp_threshold=1.0,
                 distance_threshold=10.0,
                 frame_skip=1,
                 min_area=50,
                 include_waves=False,
                 baseline_method='median',
                 percentile_value=75,
                 verbose=True):
        """
        Initialize automated SGD detector
        
        Args:
            data_dir: Directory containing MAX_*.JPG and IRX_*.irg files
            output_file: Output KML filename
            temp_threshold: Temperature difference threshold (°C)
            distance_threshold: Minimum distance between unique SGDs (meters)
            frame_skip: Process every Nth frame (1=all, 5=every 5th, etc.)
            min_area: Minimum plume area in pixels
            include_waves: Include wave areas in ocean mask
            baseline_method: Method for calculating ocean baseline ('median', 'upper_quartile', etc.)
            percentile_value: Percentile value if using custom percentile baseline
            verbose: Show detailed progress
        """
        self.data_dir = Path(data_dir)
        
        # Ensure output goes to sgd_output directory
        output_dir = Path("sgd_output")
        output_dir.mkdir(exist_ok=True)
        
        # If output_file doesn't have a directory, put it in sgd_output
        output_path = Path(output_file)
        if not output_path.parent.name or output_path.parent == Path("."):
            self.output_file = str(output_dir / output_path.name)
        else:
            self.output_file = output_file
            
        self.temp_threshold = temp_threshold
        self.distance_threshold = distance_threshold
        self.frame_skip = frame_skip
        self.min_area = min_area
        self.include_waves = include_waves
        self.baseline_method = baseline_method
        self.percentile_value = percentile_value
        self.verbose = verbose

        # Parse baseline method parameters
        baseline_params = {}
        if baseline_method == 'upper_quartile':
            baseline_params = {'baseline_method': 'upper_quartile'}
        elif baseline_method == 'percentile_80':
            baseline_params = {'baseline_method': 'upper_percentile', 'percentile_value': 80}
        elif baseline_method == 'percentile_90':
            baseline_params = {'baseline_method': 'upper_percentile', 'percentile_value': 90}
        elif baseline_method == 'trimmed_mean':
            baseline_params = {'baseline_method': 'trimmed_mean', 'trim_percentage': 25}
        else:  # median
            baseline_params = {'baseline_method': 'median'}

        # Initialize improved detector with baseline method
        # Get the model path for ML segmentation
        model_path = self.select_area_model(self.data_dir)

        self.detector = ImprovedSGDDetector(
            base_path=str(self.data_dir),
            temp_threshold=self.temp_threshold,
            min_area=self.min_area,
            use_ml=True,
            ml_model_path=model_path,
            **baseline_params
        )

        # Note: ImprovedSGDDetector inherits from IntegratedSGDDetector
        # so it should have ML support if needed

        # Load available frames
        self.frames = []
        self.rgb_files = []
        self.thermal_files = []
        
        for f in sorted(self.data_dir.glob("MAX_*.JPG"))[:500]:  # Limit to 500 frames
            num = int(f.stem.split('_')[1])
            thermal_file = self.data_dir / f"IRX_{num:04d}.irg"
            if thermal_file.exists():
                self.frames.append(f"Frame_{num:04d}")
                self.rgb_files.append(f)
                self.thermal_files.append(thermal_file)
        
        if not self.frames:
            raise FileNotFoundError(f"No paired RGB-thermal frames found in {self.data_dir}")
        
        # Set frames in detector for compatibility
        self.detector.frames = self.frames
        self.detector.rgb_files = self.rgb_files
        self.detector.thermal_files = self.thermal_files
        
        # Initialize georeferencer
        if POLYGON_SUPPORT:
            self.georef = SGDPolygonGeoref(base_path=str(self.data_dir))
        else:
            self.georef = SGDGeoref(base_path=str(self.data_dir))
        
        # Storage for all detected SGDs
        self.all_sgd_locations = []
        self.unique_sgd_locations = []
        self.frames_processed = []
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_processed': 0,
            'frames_skipped': 0,
            'total_detections': 0,
            'unique_locations': 0,
            'total_area_m2': 0,
            'processing_time': 0,
            'start_time': None,
            'end_time': None
        }
    
    def select_area_model(self, data_path):
        """
        Select the appropriate segmentation model based on the survey area.
        
        Args:
            data_path: Path to the data directory
            
        Returns:
            Path to the most appropriate model file
        """
        from improve_training_sampling import get_area_name
        
        # Get area name from directory structure
        area_name = get_area_name(data_path)
        
        # Check for area-specific model
        models_dir = Path('models')
        area_model = models_dir / f"{area_name}_segmentation.pkl"
        
        if area_model.exists():
            if self.verbose:
                print(f"✓ Using area-specific model: {area_model.name}")
            return str(area_model)
        
        # Check for parent area model if in a XXXMEDIA subdirectory
        if data_path.name.endswith('MEDIA'):
            parent_name = get_area_name(data_path.parent)
            parent_model = models_dir / f"{parent_name}_segmentation.pkl"
            if parent_model.exists():
                if self.verbose:
                    print(f"✓ Using parent area model: {parent_model.name}")
                return str(parent_model)
        
        # Check environment variable
        env_model = os.environ.get('SGD_MODEL_PATH')
        if env_model and Path(env_model).exists():
            if self.verbose:
                print(f"✓ Using model from environment: {env_model}")
            return env_model
        
        # Fall back to default model
        default_model = Path('segmentation_model.pkl')
        if default_model.exists():
            if self.verbose:
                print(f"✓ Using default model: {default_model.name}")
            return str(default_model)
        
        # No model found - will need to train
        if self.verbose:
            print("⚠ No segmentation model found - training will be required")
        return 'segmentation_model.pkl'  # Return default name for new training
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        delta_phi = np.radians(lat2 - lat1)
        delta_lambda = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_phi/2)**2 + 
             np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def is_duplicate_location(self, new_lat, new_lon):
        """Check if location is too close to existing SGDs"""
        # If distance_threshold is negative, disable deduplication
        if self.distance_threshold < 0:
            return False, None
            
        for existing in self.unique_sgd_locations:
            distance = self.haversine_distance(
                new_lat, new_lon,
                existing['centroid_lat'], 
                existing['centroid_lon']
            )
            if distance < self.distance_threshold:
                return True, existing
        return False, None
    
    def process_frame(self, frame_index):
        """Process a single frame and detect SGDs"""
        # Get actual frame number from filename
        rgb_file = self.rgb_files[frame_index]
        frame_number = int(rgb_file.stem.split('_')[1])
        
        if self.verbose:
            print(f"\n  Analyzing frame {frame_number} (index {frame_index})...", end="")
        
        # Process frame with detector using actual frame number
        result = self.detector.process_frame(
            frame_number, 
            visualize=False,
            include_waves=self.include_waves
        )
        
        if result is None or result.get('plume_info') is None or len(result['plume_info']) == 0:
            if self.verbose:
                print(" no SGDs detected")
            return []
        
        # Extract plumes from result
        plumes = result['plume_info']
        
        # Get frame metadata
        frame_name = self.frames[frame_index]
        rgb_path = self.rgb_files[frame_index]
        
        # Georeference each plume
        georeferenced_sgds = []
        
        # Get overall characteristics for temperature anomaly
        characteristics = result.get('characteristics', {})
        temp_anomaly = characteristics.get('temp_anomaly', -1.0)
        
        for i, plume in enumerate(plumes):
            # Extract centroid from plume data
            centroid = plume.get('centroid', (256, 320))
            centroid_y = int(centroid[0])
            centroid_x = int(centroid[1])
            
            # Get area information
            area_pixels = plume.get('area_pixels', 0)
            area_m2 = area_pixels * 0.01  # Assuming 10cm resolution
            
            # Georeference the centroid
            try:
                if POLYGON_SUPPORT:
                    # Extract GPS data from image
                    gps_data = self.georef.extract_gps(str(rgb_path))
                    if gps_data is None or 'lat' not in gps_data or 'lon' not in gps_data:
                        if self.verbose:
                            print(f"\n    Warning: No GPS data in {rgb_path.name}")
                        continue
                    
                    rgb_center_lat = gps_data['lat']
                    rgb_center_lon = gps_data['lon']
                    altitude = gps_data.get('altitude', 100)  # Default 100m if missing
                    heading = gps_data.get('heading', 0)
                    
                    # Use polygon georeferencer with all required parameters
                    lat, lon = self.georef.thermal_to_latlon(
                        centroid_x, centroid_y,
                        rgb_center_lat, rgb_center_lon,
                        altitude, heading
                    )
                    
                    # Create SGD location record
                    sgd_loc = {
                        'frame': frame_name,
                        'frame_number': frame_number,
                        'centroid_lat': lat,
                        'centroid_lon': lon,
                        'temperature_anomaly': temp_anomaly,
                        'area_m2': area_m2,
                        'area_pixels': area_pixels,
                        'polygon': None
                    }
                    
                    # Extract polygon outline if available
                    if 'contour' in plume and plume['contour']:
                        contour = plume['contour']
                        if len(contour) > 3:  # Need at least 3 points for polygon
                            # Georeference each polygon point
                            georef_polygon = []
                            for y, x in contour[::2]:  # Sample every 2nd point for efficiency
                                pt_lat, pt_lon = self.georef.thermal_to_latlon(
                                    int(x), int(y),
                                    rgb_center_lat, rgb_center_lon,
                                    altitude, heading
                                )
                                georef_polygon.append([pt_lon, pt_lat])
                            
                            # Close the polygon
                            if georef_polygon and georef_polygon[0] != georef_polygon[-1]:
                                georef_polygon.append(georef_polygon[0])
                            
                            sgd_loc['polygon'] = georef_polygon
                else:
                    # Use basic georeferencer (fallback)
                    lat, lon, heading = self.georef.georeference_point(
                        str(rgb_path), centroid_x, centroid_y
                    )
                    
                    sgd_loc = {
                        'frame': frame_name,
                        'frame_number': frame_number,
                        'centroid_lat': lat,
                        'centroid_lon': lon,
                        'temperature_anomaly': temp_anomaly,
                        'area_m2': area_m2,
                        'area_pixels': area_pixels,
                        'polygon': None
                    }
                
                georeferenced_sgds.append(sgd_loc)
                
            except Exception as e:
                if self.verbose:
                    print(f"\n    Warning: Could not georeference plume {i}: {e}")
                continue
        
        if self.verbose and georeferenced_sgds:
            print(f" found {len(georeferenced_sgds)} SGDs")
        
        return georeferenced_sgds
    
    def deduplicate_sgds(self, new_sgds, frame_number):
        """Deduplicate SGDs based on distance threshold"""
        added_count = 0
        
        for sgd in new_sgds:
            is_dup, existing = self.is_duplicate_location(
                sgd['centroid_lat'], 
                sgd['centroid_lon']
            )
            
            if not is_dup:
                # New unique location
                sgd['first_frame'] = frame_number
                sgd['last_frame'] = frame_number
                sgd['detection_count'] = 1
                self.unique_sgd_locations.append(sgd)
                added_count += 1
            else:
                # Update existing location
                existing['last_frame'] = frame_number
                existing['detection_count'] = existing.get('detection_count', 1) + 1
                # Update area if larger
                if sgd['area_m2'] > existing.get('area_m2', 0):
                    existing['area_m2'] = sgd['area_m2']
                    existing['area_pixels'] = sgd['area_pixels']
                    if 'polygon' in sgd and sgd['polygon']:
                        existing['polygon'] = sgd['polygon']
        
        return added_count
    
    def export_kml(self):
        """Export results to KML format"""
        print(f"\nExporting results to {self.output_file}...")
        
        kml_content = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml_content.append('<Document>')
        kml_content.append(f'<name>SGD Detection Results - {datetime.now().strftime("%Y-%m-%d %H:%M")}</name>')
        kml_content.append('<description>Automated submarine groundwater discharge detection</description>')
        
        # Style for polygons
        kml_content.append('''
        <Style id="sgdPolygon">
            <LineStyle>
                <color>ff0000ff</color>
                <width>2</width>
            </LineStyle>
            <PolyStyle>
                <color>7f0000ff</color>
            </PolyStyle>
        </Style>
        <Style id="sgdPoint">
            <IconStyle>
                <Icon>
                    <href>http://maps.google.com/mapfiles/kml/shapes/water.png</href>
                </Icon>
            </IconStyle>
        </Style>
        ''')
        
        # Add summary statistics folder
        kml_content.append('<Folder>')
        kml_content.append('<name>Summary Statistics</name>')
        kml_content.append('<description>')
        kml_content.append(f'Total frames analyzed: {self.stats["frames_processed"]}\n')
        kml_content.append(f'Total SGDs detected: {self.stats["total_detections"]}\n')
        kml_content.append(f'Unique locations: {self.stats["unique_locations"]}\n')
        kml_content.append(f'Total area: {self.stats["total_area_m2"]:.1f} m²\n')
        kml_content.append(f'Processing time: {self.stats["processing_time"]:.1f} seconds\n')
        kml_content.append(f'Temperature threshold: {self.temp_threshold}°C\n')
        kml_content.append(f'Distance threshold: {self.distance_threshold}m\n')
        kml_content.append(f'Minimum area: {self.min_area} pixels')
        kml_content.append('</description>')
        kml_content.append('</Folder>')
        
        # Add SGD locations folder
        kml_content.append('<Folder>')
        kml_content.append('<name>SGD Locations</name>')
        
        for i, sgd in enumerate(self.unique_sgd_locations):
            kml_content.append('<Placemark>')
            kml_content.append(f'<name>SGD {i+1}</name>')
            
            # Description with metadata
            desc = []
            desc.append(f'Temperature anomaly: {sgd.get("temperature_anomaly", 0):.1f}°C')
            desc.append(f'Area: {sgd.get("area_m2", 0):.1f} m²')
            desc.append(f'First detected: Frame {sgd.get("first_frame", "unknown")}')
            desc.append(f'Last detected: Frame {sgd.get("last_frame", "unknown")}')
            desc.append(f'Detection count: {sgd.get("detection_count", 1)}')
            kml_content.append(f'<description>{chr(10).join(desc)}</description>')
            
            # Add geometry
            if sgd.get('polygon') and POLYGON_SUPPORT:
                # Polygon geometry
                kml_content.append('<styleUrl>#sgdPolygon</styleUrl>')
                kml_content.append('<Polygon>')
                kml_content.append('<outerBoundaryIs>')
                kml_content.append('<LinearRing>')
                kml_content.append('<coordinates>')
                
                # Add polygon coordinates
                for coord in sgd['polygon']:
                    kml_content.append(f'{coord[0]},{coord[1]},0')
                
                # Close the polygon
                if sgd['polygon']:
                    kml_content.append(f'{sgd["polygon"][0][0]},{sgd["polygon"][0][1]},0')
                
                kml_content.append('</coordinates>')
                kml_content.append('</LinearRing>')
                kml_content.append('</outerBoundaryIs>')
                kml_content.append('</Polygon>')
            else:
                # Point geometry (fallback)
                kml_content.append('<styleUrl>#sgdPoint</styleUrl>')
                kml_content.append('<Point>')
                kml_content.append(f'<coordinates>{sgd["centroid_lon"]},{sgd["centroid_lat"]},0</coordinates>')
                kml_content.append('</Point>')
            
            kml_content.append('</Placemark>')
        
        kml_content.append('</Folder>')
        kml_content.append('</Document>')
        kml_content.append('</kml>')
        
        # Write KML file
        with open(self.output_file, 'w') as f:
            f.write('\n'.join(kml_content))
        
        print(f"✓ KML file saved: {self.output_file}")
        
        # Create merged polygon KML if we have polygons
        if POLYGON_SUPPORT and any(sgd.get('polygon') for sgd in self.unique_sgd_locations):
            try:
                from merge_sgd_polygons import create_merged_kml
                
                # Create merged KML filename
                merged_kml = self.output_file.replace('.kml', '_merged.kml')
                
                # Create merged KML
                result = create_merged_kml(
                    self.unique_sgd_locations,
                    merged_kml,
                    use_shapely=True
                )
                
                if result:
                    print(f"✓ Merged KML saved: {merged_kml}")
            except ImportError:
                pass  # Merging not available
            except Exception as e:
                print(f"Warning: Could not create merged KML: {e}")
    
    def export_summary_json(self):
        """Export detailed summary in JSON format"""
        json_file = self.output_file.replace('.kml', '_summary.json')
        
        # Convert datetime objects to strings in stats
        stats_copy = self.stats.copy()
        if 'start_time' in stats_copy and isinstance(stats_copy['start_time'], datetime):
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
        if 'end_time' in stats_copy and isinstance(stats_copy['end_time'], datetime):
            stats_copy['end_time'] = stats_copy['end_time'].isoformat()
        
        summary = {
            'metadata': {
                'processing_date': datetime.now().isoformat(),
                'data_directory': str(self.data_dir),
                'parameters': {
                    'temperature_threshold': self.temp_threshold,
                    'distance_threshold': self.distance_threshold,
                    'frame_skip': self.frame_skip,
                    'minimum_area': self.min_area,
                    'include_waves': self.include_waves
                }
            },
            'statistics': stats_copy,
            'sgd_locations': self.unique_sgd_locations
        }
        
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Summary JSON saved: {json_file}")
    
    def run(self):
        """Run the automated detection process"""
        print("="*60)
        print("SGD AUTOMATED DETECTION")
        print("="*60)
        print(f"Data directory: {self.data_dir}")
        print(f"Output file: {self.output_file}")
        print(f"Temperature threshold: {self.temp_threshold}°C")
        if self.distance_threshold < 0:
            print(f"Distance threshold: DISABLED (keeping all detections)")
        else:
            print(f"Distance threshold: {self.distance_threshold}m")
        print(f"Frame skip: {self.frame_skip} (process every {self.frame_skip} frame(s))")
        print(f"Minimum area: {self.min_area} pixels")
        print(f"Include waves: {self.include_waves}")
        print("-"*60)
        
        # Start timer
        self.stats['start_time'] = datetime.now()
        start_time = time.time()
        
        # Get list of frames
        frames = self.detector.frames
        self.stats['total_frames'] = len(frames)
        
        # Select frames to process based on skip parameter
        frames_to_process = list(range(0, len(frames), self.frame_skip))
        print(f"\nProcessing {len(frames_to_process)} of {len(frames)} frames")
        print("-"*60)
        
        # Process frames with progress bar
        with tqdm(frames_to_process, desc="Processing frames", unit="frame") as pbar:
            for frame_idx in pbar:
                # Update progress bar description
                pbar.set_description(f"Frame {frame_idx}: {frames[frame_idx]}")
                
                # Process frame
                sgds = self.process_frame(frame_idx)
                
                # Add to all detections
                self.all_sgd_locations.extend(sgds)
                self.stats['total_detections'] += len(sgds)
                
                # Deduplicate
                new_unique = self.deduplicate_sgds(sgds, frame_idx)
                
                # Update statistics
                self.stats['frames_processed'] += 1
                self.frames_processed.append(frame_idx)
                
                # Update progress bar postfix
                pbar.set_postfix({
                    'SGDs': self.stats['total_detections'],
                    'Unique': len(self.unique_sgd_locations)
                })
        
        # Calculate final statistics
        self.stats['end_time'] = datetime.now()
        self.stats['processing_time'] = time.time() - start_time
        self.stats['unique_locations'] = len(self.unique_sgd_locations)
        self.stats['total_area_m2'] = sum(s.get('area_m2', 0) for s in self.unique_sgd_locations)
        self.stats['frames_skipped'] = self.stats['total_frames'] - self.stats['frames_processed']
        
        # Print summary
        print("\n" + "="*60)
        print("DETECTION COMPLETE")
        print("="*60)
        print(f"Frames processed: {self.stats['frames_processed']}/{self.stats['total_frames']}")
        print(f"Total SGDs detected: {self.stats['total_detections']}")
        print(f"Unique SGD locations: {self.stats['unique_locations']}")
        print(f"Total SGD area: {self.stats['total_area_m2']:.1f} m²")
        print(f"Processing time: {self.stats['processing_time']:.1f} seconds")
        print(f"Average time per frame: {self.stats['processing_time']/self.stats['frames_processed']:.2f} seconds")
        
        # Export results
        if self.unique_sgd_locations:
            # Populate georef.sgd_polygons for aggregation
            if POLYGON_SUPPORT and hasattr(self.georef, 'sgd_polygons'):
                self.georef.sgd_polygons = self.unique_sgd_locations.copy()
            
            self.export_kml()
            self.export_summary_json()
            
            # Also export GeoJSON if polygon support available
            if POLYGON_SUPPORT and self.georef.sgd_polygons:
                geojson_file = self.output_file.replace('.kml', '.geojson')
                self.georef.export_geojson_polygons(geojson_file)
                print(f"✓ GeoJSON file saved: {geojson_file}")
        else:
            print("\n⚠ No SGDs detected with current parameters")
            print("Consider adjusting:")
            print("  - Lowering temperature threshold")
            print("  - Including wave areas (--waves)")
            print("  - Reducing minimum area")
        
        print("\n" + "="*60)
        print("Done!")
        return self.stats


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Automated SGD detection from thermal/RGB drone imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python sgd_autodetect.py --data data/survey --output survey_sgd.kml
  
  # Interactive training (manual labeling) then detection
  python sgd_autodetect.py --data data/survey --output sgd.kml --train
  
  # Automatic training (no labeling) then detection  
  python sgd_autodetect.py --data data/survey --output sgd.kml --train-auto
  
  # Use existing custom model from models directory
  python sgd_autodetect.py --data data/survey --output sgd.kml --model models/custom_model.pkl
  
  # Process every 5th frame with custom thresholds
  python sgd_autodetect.py --data data/survey --output sgd.kml --skip 5 --temp 0.5
  
  # Quick processing with automatic training
  python sgd_autodetect.py --data data/survey --output test.kml --train-auto --skip 10
  
  # Process multiple XXXMEDIA subdirectories in one flight
  python sgd_autodetect.py --data "/path/to/flight" --output flight.kml --search
  # Finds and processes: 100MEDIA/, 101MEDIA/, 102MEDIA/, etc.
        """
    )
    
    # Required arguments
    parser.add_argument('--data', required=True,
                       help='Directory containing MAX_*.JPG and IRX_*.irg files')
    parser.add_argument('--output', required=True,
                       help='Output KML filename (e.g., survey_sgd.kml)')
    
    # Detection parameters
    parser.add_argument('--temp', type=float, default=1.0,
                       help='Temperature difference threshold in °C (default: 1.0)')
    parser.add_argument('--distance', type=float, default=10.0,
                       help='Minimum distance between unique SGDs in meters (default: 10.0, use -1 to disable deduplication)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 5=every 5th, etc.) (default: 1)')
    parser.add_argument('--area', type=int, default=50,
                       help='Minimum SGD area in pixels (default: 50)')
    parser.add_argument('--waves', action='store_true',
                       help='Include wave/foam areas in ocean mask')

    # Baseline temperature method
    parser.add_argument('--baseline', type=str, default='median',
                       choices=['median', 'upper_quartile', 'percentile_80', 'percentile_90', 'trimmed_mean'],
                       help='Method for calculating ocean baseline temperature (default: median)')
    parser.add_argument('--percentile', type=float, default=75,
                       help='Custom percentile value if using percentile baseline (default: 75)')

    # Multi-threshold analysis options
    parser.add_argument('--interval-step', type=float, default=None,
                       help='Temperature interval for multi-threshold analysis (e.g., 0.5)')
    parser.add_argument('--interval-step-number', type=int, default=4,
                       help='Number of threshold steps to analyze (default: 4)')
    
    # Model options
    parser.add_argument('--model', type=str, default='segmentation_model.pkl',
                       help='Segmentation model to use (default: segmentation_model.pkl)')
    parser.add_argument('--train', action='store_true',
                       help='Launch interactive segmentation trainer before detection')
    parser.add_argument('--train-auto', action='store_true',
                       help='Automatically train segmentation model (no manual labeling)')
    parser.add_argument('--train-samples', type=int, default=10,
                       help='Number of frames to sample for auto training (default: 10)')
    
    # Training sampling options
    parser.add_argument('--train-sampling', type=str, default='distributed',
                       choices=['distributed', 'increment', 'random'],
                       help='Frame sampling method for training (default: distributed)')
    parser.add_argument('--train-increment', type=int, default=25,
                       help='Frame skip interval for increment sampling (default: 25)')
    parser.add_argument('--train-max-frames', type=int, default=20,
                       help='Maximum frames to use for training (default: 20)')
    
    # Output options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, show only progress bar')
    
    # Search options
    parser.add_argument('--search', action='store_true',
                       help='Search for XXXMEDIA subdirectories (100MEDIA, 101MEDIA, etc.) in the data path')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    # Handle search mode - find all XXXMEDIA subdirectories
    data_directories = []
    if args.search:
        import re
        base_path = Path(args.data)
        media_pattern = re.compile(r'^\d{3}MEDIA$')
        
        # Find all matching subdirectories
        for item in sorted(base_path.iterdir()):
            if item.is_dir() and media_pattern.match(item.name):
                data_directories.append(str(item))
        
        if not data_directories:
            print(f"Error: No XXXMEDIA subdirectories found in {args.data}")
            print("Looking for directories like: 100MEDIA, 101MEDIA, 102MEDIA, etc.")
            sys.exit(1)
        
        print(f"\n✓ Found {len(data_directories)} media directories to process:")
        for dir_path in data_directories:
            dir_name = Path(dir_path).name
            # Quick check for paired files
            jpg_count = len(list(Path(dir_path).glob("MAX_*.JPG")))
            irg_count = len(list(Path(dir_path).glob("IRX_*.irg")))
            print(f"  - {dir_name}: {jpg_count} RGB, {irg_count} thermal images")
    else:
        # Single directory mode
        data_directories = [args.data]
    
    # Ensure output has .kml extension
    if not args.output.endswith('.kml'):
        args.output += '.kml'
    
    # Handle model path and training
    model_path = args.model
    
    # Check for conflicting training options
    if args.train and args.train_auto:
        print("Error: Cannot use both --train and --train-auto. Choose one training mode.")
        sys.exit(1)
    
    # Create model filename based on output name
    output_name = Path(args.output).stem
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # If manual training requested, launch interactive trainer
    if args.train:
        # Import enhanced training utilities
        from improve_training_sampling import get_area_name, create_model_paths
        
        # If using search mode, use the first subdirectory for training
        training_dir = args.data
        if args.search and data_directories:
            training_dir = data_directories[0]
            print(f"\n📍 Using {Path(training_dir).name} for training (first directory)")
        
        # Get area-based model naming
        area_name = get_area_name(training_dir)
        model_path, training_path = create_model_paths(area_name)
        
        print("\n" + "="*60)
        print("INTERACTIVE SEGMENTATION TRAINING")
        print("="*60)
        print(f"Survey Area: {area_name}")
        print(f"Data directory: {training_dir}")
        print(f"Model will be saved as: {model_path}")
        print(f"Training data: {training_path}")
        print("-"*60)
        print(f"Sampling Configuration:")
        print(f"  Method: {args.train_sampling}")
        if args.train_sampling == 'increment':
            print(f"  Increment: every {args.train_increment}th frame")
        elif args.train_sampling == 'random':
            print(f"  Random selection from available frames")
        else:
            print(f"  Evenly distributed across all frames")
        print(f"  Max frames: {args.train_max_frames}")
        print("-"*60)
        print("\nInstructions:")
        print("1. Click on regions to label them (need 100+ samples per class):")
        print("   - Ocean (blue water)")
        print("   - Land (vegetation, sand)")
        print("   - Rock (gray rocky areas)")
        print("   - Wave (white foam)")
        print("2. Press 'Train' button to train the model")
        print("3. Press 'Save & Continue' button to proceed to detection")
        print("   (The window will close automatically)")
        print("-"*60)
        
        # Launch the interactive trainer with better frame sampling
        from segmentation_trainer import SegmentationTrainer
        
        try:
            # Use command-line specified sampling parameters
            trainer = SegmentationTrainer(
                base_path=training_dir,
                model_file=str(model_path),
                training_file=str(training_path),
                sampling=args.train_sampling,
                frame_increment=args.train_increment,
                max_frames=args.train_max_frames
            )
            trainer.run()
            
            # Check if model was actually saved (model_path is a Path object from create_model_paths)
            if not model_path.exists():
                print(f"\n⚠ Warning: Model file not found at {model_path}")
                print("Training may have been cancelled. Exiting.")
                sys.exit(1)
            
            print(f"\n✓ Interactive training complete!")
            print(f"✓ Model saved to: {model_path}")
            print("\n" + "="*60)
            print("PROCEEDING TO SGD DETECTION")
            print("="*60)
            print(f"Using newly trained model: {model_path}")
            print()
            
            # Convert Path to string for later use
            model_path = str(model_path)
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            sys.exit(1)
    
    # If automatic training requested
    elif args.train_auto:
        from auto_train_segmentation import AutoSegmentationTrainer
        
        model_path = str(model_dir / f"{output_name}_model.pkl")
        training_path = str(model_dir / f"{output_name}_training.json")
        
        # If using search mode, use the first subdirectory for training
        training_dir = args.data
        if args.search and data_directories:
            training_dir = data_directories[0]
            print(f"\n📍 Using {Path(training_dir).name} for training (first directory)")
        
        print("\n" + "="*60)
        print("AUTOMATED SEGMENTATION TRAINING")
        print("="*60)
        print(f"Training new model from: {training_dir}")
        print(f"Model will be saved as: {model_path}")
        print(f"Sampling {args.train_samples} frames...")
        print("-"*60)
        
        trainer = AutoSegmentationTrainer(
            data_dir=training_dir,
            model_file=model_path,
            training_file=training_path,
            sample_frames=args.train_samples,
            verbose=not args.quiet
        )
        
        try:
            stats = trainer.train()
            print(f"\n✓ Training complete! Accuracy: {stats['test_accuracy']:.2%}")
            print("="*60)
            print()
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            sys.exit(1)
    else:
        # If no training was done, try to find or use specified model
        
        # If model not specified, try to find matching model based on output name
        if model_path == 'segmentation_model.pkl':  # Default value
            output_based_model = Path("models") / f"{output_name}_model.pkl"
            if output_based_model.exists():
                model_path = str(output_based_model)
                print(f"✓ Found matching model: {model_path}")
        
        # Check if model exists in models directory if not absolute path
        if not Path(model_path).is_absolute():
            # First check models directory
            models_dir_path = Path("models") / model_path
            if models_dir_path.exists():
                model_path = str(models_dir_path)
            # Otherwise use as-is (will look in current directory)
    
    # Set the model for the detector (ensure it's a string)
    os.environ['SGD_MODEL_PATH'] = str(model_path)
    
    # Process each directory
    all_stats = []
    all_sgd_polygons = []  # Collect all SGD polygons for aggregation
    total_sgds = 0
    total_unique = 0
    
    # For multi-threshold analysis with --search, collect results from all directories
    all_multi_threshold_results = {}  # {threshold: [polygons from all directories]}
    
    # Create subdirectory for individual outputs if in search mode
    individual_output_dir = None
    if args.search and len(data_directories) > 1:
        base_output = Path(args.output).stem
        individual_output_dir = Path("sgd_output") / f"{base_output}_individual"
        individual_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ Individual outputs will be saved to: {individual_output_dir}/")
    
    for idx, data_dir in enumerate(data_directories):
        # Modify output filename for multiple directories
        if args.search and len(data_directories) > 1:
            dir_name = Path(data_dir).name
            base_output = Path(args.output).stem
            output_ext = Path(args.output).suffix
            
            # Save individual outputs to subdirectory
            current_output = str(individual_output_dir / f"{base_output}_{dir_name}{output_ext}")
            
            print(f"\n{'='*60}")
            print(f"Processing directory {idx+1}/{len(data_directories)}: {dir_name}")
            print(f"Output: {Path(current_output).name}")
            print('='*60)
        else:
            current_output = args.output
        
        # Check if multi-threshold analysis is requested
        if args.interval_step is not None:
            # Use enhanced multi-threshold analyzer
            from multi_threshold_analysis_v2 import EnhancedMultiThresholdAnalyzer
            
            print(f"\n{'='*60}")
            print(f"MULTI-THRESHOLD ANALYSIS")
            print(f"Base threshold: {args.temp}°C")
            print(f"Interval step: {args.interval_step}°C")
            print(f"Number of thresholds: {args.interval_step_number}")
            print('='*60)
            
            analyzer = EnhancedMultiThresholdAnalyzer(
                data_dir=data_dir,
                base_output=current_output,
                base_threshold=args.temp,
                interval_step=args.interval_step,
                num_steps=args.interval_step_number,
                distance_threshold=args.distance,
                frame_skip=args.skip,
                min_area=args.area,
                include_waves=args.waves,
                verbose=not args.quiet
            )
            
            try:
                # Run enhanced multi-threshold analysis
                all_results = analyzer.run_multi_threshold_analysis()
                
                # Verify nesting property
                analyzer.verify_nesting(all_results)
                
                # Create combined KMLs (merged and unmerged)
                combined_kmls = analyzer.create_combined_kml(all_results)
                for kml_file in combined_kmls:
                    print(f"✓ Created: {kml_file}")
                
                # Get aggregated stats for summary
                stats = analyzer.get_aggregated_stats(all_results)
                all_stats.append(stats)
                
                # Accumulate totals
                if stats:
                    total_sgds += stats.get('total_detections', 0)
                    total_unique += stats.get('unique_locations', 0)
                    
                    # Collect SGD polygons for aggregation
                    if args.search and len(data_directories) > 1 and POLYGON_SUPPORT:
                        if all_results:
                            # Collect polygons from all threshold levels for aggregation
                            for threshold, result in all_results.items():
                                if 'sgd_polygons' in result and result['sgd_polygons']:
                                    # Initialize threshold list if needed
                                    if threshold not in all_multi_threshold_results:
                                        all_multi_threshold_results[threshold] = []
                                    # Add this directory's polygons for this threshold
                                    all_multi_threshold_results[threshold].extend(result['sgd_polygons'])
                                    # Also add to general polygon list (for backwards compatibility)
                                    all_sgd_polygons.extend(result['sgd_polygons'])
                    
            except Exception as e:
                import traceback
                print(f"\nError during multi-threshold analysis: {e}")
                traceback.print_exc()
                if not args.search:
                    sys.exit(1)
                else:
                    print(f"Skipping {data_dir} due to error")
                    continue
        else:
            # Standard single-threshold detection
            detector = SGDAutoDetector(
                data_dir=data_dir,
                output_file=current_output,
                temp_threshold=args.temp,
                distance_threshold=args.distance,
                frame_skip=args.skip,
                min_area=args.area,
                include_waves=args.waves,
                baseline_method=args.baseline,
                percentile_value=args.percentile,
                verbose=not args.quiet
            )
            
            try:
                stats = detector.run()
                all_stats.append(stats)
                
                # Accumulate totals and polygons
                if stats:
                    total_sgds += stats.get('total_detections', 0)
                    total_unique += stats.get('unique_locations', 0)
                    
                    # Collect SGD polygons for aggregation
                    if args.search and len(data_directories) > 1 and POLYGON_SUPPORT:
                        if hasattr(detector, 'georef') and hasattr(detector.georef, 'sgd_polygons'):
                            all_sgd_polygons.extend(detector.georef.sgd_polygons)
            
            except KeyboardInterrupt:
                print("\n\nDetection interrupted by user")
                sys.exit(1)
            except Exception as e:
                import traceback
                print(f"\nError during detection in {data_dir}: {e}")
                print(f"Error type: {type(e).__name__}")
                traceback.print_exc()
                if not args.search:
                    sys.exit(1)
                else:
                    print(f"Skipping {dir_name} due to error")
                    continue
    
    # Print summary and create aggregated outputs for search mode
    if args.search and len(data_directories) > 1:
        print("\n" + "="*60)
        print("CREATING AGGREGATED OUTPUTS")
        print("="*60)
        
        # Deduplicate SGD polygons across all directories
        if all_sgd_polygons and POLYGON_SUPPORT:
            from sgd_georef_polygons import SGDPolygonGeoref
            import math
            
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate distance between two points in meters"""
                R = 6371000  # Earth's radius in meters
                phi1 = math.radians(lat1)
                phi2 = math.radians(lat2)
                dphi = math.radians(lat2 - lat1)
                dlambda = math.radians(lon2 - lon1)
                
                a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                return R * c
            
            # Deduplicate SGDs based on location proximity
            unique_sgds = []
            for sgd in all_sgd_polygons:
                # Check if this SGD is close to any existing unique SGD
                is_duplicate = False
                for unique in unique_sgds:
                    # Calculate distance between centroids
                    dist = haversine_distance(
                        sgd['centroid_lat'], sgd['centroid_lon'],
                        unique['centroid_lat'], unique['centroid_lon']
                    )
                    
                    # Check if deduplication is disabled (negative distance)
                    if args.distance >= 0 and dist < args.distance:  # Within threshold distance
                        is_duplicate = True
                        # Update the unique SGD if this one has larger area
                        if sgd.get('area_m2', 0) > unique.get('area_m2', 0):
                            unique['area_m2'] = sgd['area_m2']
                            unique['area_pixels'] = sgd['area_pixels']
                            if 'polygon' in sgd and sgd['polygon']:
                                unique['polygon'] = sgd['polygon']
                        break
                
                if not is_duplicate:
                    unique_sgds.append(sgd)
            
            print(f"Total SGDs before deduplication: {len(all_sgd_polygons)}")
            print(f"Unique SGDs after deduplication: {len(unique_sgds)}")
            
            # Create aggregated KML using deduplicated SGDs
            if unique_sgds:
                temp_georef = SGDPolygonGeoref()
                temp_georef.sgd_polygons = unique_sgds
                
                # Ensure the output has .kml extension but not double
                output_name = args.output
                if not output_name.endswith('.kml'):
                    output_name += '.kml'
                aggregated_kml_path = Path("sgd_output") / output_name
                temp_georef.export_kml_polygons(str(aggregated_kml_path))
                print(f"✓ Aggregated KML saved: {aggregated_kml_path}")
                
                # Create merged KML for aggregated results
                try:
                    from merge_sgd_polygons import create_merged_kml
                    
                    merged_aggregated = str(aggregated_kml_path).replace('.kml', '_merged.kml')
                    result = create_merged_kml(
                        unique_sgds,
                        merged_aggregated,
                        use_shapely=True
                    )
                    if result:
                        print(f"✓ Aggregated merged KML saved: {merged_aggregated}")
                except:
                    pass  # Merging not available
            
            # Update totals
            total_unique = len(unique_sgds)
        
        # Create aggregated multi-threshold KML if interval-step was used
        if args.interval_step is not None and all_multi_threshold_results:
            print("\n" + "="*60)
            print("CREATING AGGREGATED MULTI-THRESHOLD KML")
            print("="*60)
            
            from multi_threshold_analysis_v2 import EnhancedMultiThresholdAnalyzer
            
            # Create a temporary analyzer just for the aggregated KML creation
            aggregated_analyzer = EnhancedMultiThresholdAnalyzer(
                data_dir="",  # Not used for aggregation
                base_output=args.output,
                base_threshold=args.temp,
                interval_step=args.interval_step,
                num_steps=args.interval_step_number,
                distance_threshold=args.distance,
                frame_skip=args.skip,
                min_area=args.area,
                include_waves=args.waves,
                verbose=False
            )
            
            # Format the aggregated results for the analyzer
            aggregated_results = {}
            for threshold, polygons in all_multi_threshold_results.items():
                aggregated_results[threshold] = {
                    'threshold': threshold,
                    'sgd_polygons': polygons,
                    'stats': {
                        'total_detections': len(polygons),
                        'unique_locations': len(polygons)  # Will be deduplicated in KML
                    }
                }
            
            # Create aggregated combined KMLs
            aggregated_kmls = aggregated_analyzer.create_combined_kml(aggregated_results)
            print(f"✓ Created aggregated multi-threshold KMLs:")
            for kml_file in aggregated_kmls:
                print(f"   - {kml_file}")
        
        print("\n" + "="*60)
        print("COMBINED SUMMARY")
        print("="*60)
        print(f"Directories processed: {len(all_stats)}/{len(data_directories)}")
        print(f"Total SGDs detected: {total_sgds}")
        print(f"Unique locations (after deduplication): {total_unique}")
        
        # Create combined summary file
        combined_summary = {
            'directories_processed': len(all_stats),
            'total_directories': len(data_directories),
            'total_sgds_detected': total_sgds,
            'unique_sgds_after_deduplication': total_unique,
            'distance_threshold_meters': args.distance,
            'temperature_threshold_celsius': args.temp,
            'individual_results': all_stats,
            'data_directories': data_directories,
            'individual_outputs_directory': str(individual_output_dir),
            'aggregated_kml': str(Path("sgd_output") / args.output)
        }
        
        summary_file = Path(args.output).stem + "_summary.json"
        summary_path = Path("sgd_output") / summary_file
        
        with open(summary_path, 'w') as f:
            json.dump(combined_summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"✓ Aggregated summary saved: {summary_path}")
        print("\n" + "="*60)
        print("OUTPUT FILES:")
        print("="*60)
        print(f"📁 Individual outputs: {individual_output_dir}/")
        print(f"🗺️  Aggregated KML: sgd_output/{args.output}")
        print(f"📊 Aggregated summary: {summary_path}")
        print("="*60)
    
    sys.exit(0)


if __name__ == "__main__":
    main()