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
try:
    from sgd_georef_polygons import SGDPolygonGeoref
    POLYGON_SUPPORT = True
except ImportError:
    print("Warning: Polygon support not available, using point-based georeferencing")
    from sgd_georef import SGDGeoref
    POLYGON_SUPPORT = False


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class SGDAutoDetector:
    """Automated SGD detection with batch processing"""
    
    def __init__(self, data_dir, output_file, 
                 temp_threshold=1.0, 
                 distance_threshold=10.0,
                 frame_skip=1,
                 min_area=50,
                 include_waves=False,
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
            verbose: Show detailed progress
        """
        self.data_dir = Path(data_dir)
        self.output_file = output_file
        self.temp_threshold = temp_threshold
        self.distance_threshold = distance_threshold
        self.frame_skip = frame_skip
        self.min_area = min_area
        self.include_waves = include_waves
        self.verbose = verbose
        
        # Initialize detector
        self.detector = IntegratedSGDDetector(
            temp_threshold=self.temp_threshold,
            min_area=self.min_area,
            base_path=str(self.data_dir)
        )
        
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
                    # Use polygon georeferencer (no verbose parameter)
                    lat, lon = self.georef.thermal_to_latlon(
                        centroid_x, centroid_y,
                        str(rgb_path)
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
                    
                    # Add polygon if available (skip for speed in batch mode)
                    # Polygon georeferencing is expensive, so only do centroids
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
    
    def export_summary_json(self):
        """Export detailed summary in JSON format"""
        json_file = self.output_file.replace('.kml', '_summary.json')
        
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
            'statistics': self.stats,
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
            self.export_kml()
            self.export_summary_json()
            
            # Also export GeoJSON if polygon support available
            if POLYGON_SUPPORT and self.georef.sgd_polygons:
                geojson_file = self.output_file.replace('.kml', '.geojson')
                self.georef.export_geojson(geojson_file)
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
  
  # Process every 5th frame with custom thresholds
  python sgd_autodetect.py --data data/survey --output sgd.kml --skip 5 --temp 0.5
  
  # Include wave areas and use strict deduplication
  python sgd_autodetect.py --data data/survey --output sgd.kml --waves --distance 20
  
  # Quick processing of subset
  python sgd_autodetect.py --data data/survey --output test.kml --skip 10 --quiet
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
                       help='Minimum distance between unique SGDs in meters (default: 10.0)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 5=every 5th, etc.) (default: 1)')
    parser.add_argument('--area', type=int, default=50,
                       help='Minimum SGD area in pixels (default: 50)')
    parser.add_argument('--waves', action='store_true',
                       help='Include wave/foam areas in ocean mask')
    
    # Output options
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output, show only progress bar')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data):
        print(f"Error: Data directory not found: {args.data}")
        sys.exit(1)
    
    # Ensure output has .kml extension
    if not args.output.endswith('.kml'):
        args.output += '.kml'
    
    # Run detection
    detector = SGDAutoDetector(
        data_dir=args.data,
        output_file=args.output,
        temp_threshold=args.temp,
        distance_threshold=args.distance,
        frame_skip=args.skip,
        min_area=args.area,
        include_waves=args.waves,
        verbose=not args.quiet
    )
    
    try:
        stats = detector.run()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nDetection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()