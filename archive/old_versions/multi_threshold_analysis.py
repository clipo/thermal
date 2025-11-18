#!/usr/bin/env python3
"""
Multi-threshold SGD analysis for temperature interval detection.

Analyzes frames at multiple temperature thresholds to create a gradient
map of SGD intensity based on temperature anomalies.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import copy

# Try to import shapely for polygon operations
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not installed. Polygon merging will be simplified.")


class MultiThresholdAnalyzer:
    """Analyze SGDs at multiple temperature thresholds."""
    
    # Color scheme for different temperature thresholds
    THRESHOLD_COLORS = {
        0.5: {'name': 'yellow', 'kml': '7f00ffff'},      # Yellow (weak SGD)
        1.0: {'name': 'green', 'kml': '7f00ff00'},       # Green
        1.5: {'name': 'orange', 'kml': '7f0080ff'},      # Orange
        2.0: {'name': 'red', 'kml': '7f0000ff'},         # Red
        2.5: {'name': 'purple', 'kml': '7fff00ff'},      # Purple
        3.0: {'name': 'darkred', 'kml': '7f000080'},     # Dark red (strong SGD)
        3.5: {'name': 'brown', 'kml': '7f004080'},       # Brown
        4.0: {'name': 'black', 'kml': '7f000000'},       # Black
    }
    
    def __init__(self, data_dir, base_output, base_threshold=0.5, interval_step=0.5, 
                 num_steps=4, distance_threshold=50, frame_skip=10, min_area=20,
                 include_waves=False, verbose=True):
        """
        Initialize multi-threshold analyzer.
        
        Args:
            data_dir: Directory containing thermal images
            base_output: Base output filename
            base_threshold: Starting temperature threshold
            interval_step: Temperature increment between thresholds
            num_steps: Number of threshold steps to analyze
            distance_threshold: Distance for deduplication (meters)
            frame_skip: Number of frames to skip
            min_area: Minimum SGD area (pixels)
            include_waves: Include wave patterns
            verbose: Show detailed output
        """
        self.data_dir = data_dir
        self.base_output = base_output
        self.base_threshold = base_threshold
        self.interval_step = interval_step
        self.num_steps = num_steps
        self.distance_threshold = distance_threshold
        self.frame_skip = frame_skip
        self.min_area = min_area
        self.include_waves = include_waves
        self.verbose = verbose
        
        # Calculate all thresholds
        self.thresholds = [
            base_threshold + (i * interval_step) 
            for i in range(num_steps)
        ]
        
        # Storage for detections at each threshold
        self.detections_by_threshold = defaultdict(list)
        
        print(f"Multi-threshold analysis configured:")
        print(f"  Base threshold: {base_threshold}°C")
        print(f"  Interval: {interval_step}°C")
        print(f"  Steps: {num_steps}")
        print(f"  Thresholds: {self.thresholds}")
    
    def get_color_for_threshold(self, threshold):
        """Get KML color code for a threshold value."""
        # Find closest defined threshold
        defined_thresholds = sorted(self.THRESHOLD_COLORS.keys())
        
        # Find the closest match
        closest = min(defined_thresholds, key=lambda x: abs(x - threshold))
        
        # If exact match or very close, use that color
        if abs(closest - threshold) < 0.1:
            return self.THRESHOLD_COLORS[closest]
        
        # Otherwise, interpolate or use a default
        if threshold < 0.5:
            return {'name': 'lightyellow', 'kml': '7f80ffff'}
        elif threshold > 4.0:
            return {'name': 'darkblack', 'kml': '7f000000'}
        else:
            # Use the closest color
            return self.THRESHOLD_COLORS[closest]
    
    def process_frame_multi_threshold(self, detector, frame_index):
        """
        Process a single frame at multiple temperature thresholds.
        
        Args:
            detector: SGD detector instance
            frame_index: Index of frame to process
            
        Returns:
            Dictionary of detections by threshold
        """
        frame_detections = {}
        
        # Store original threshold
        original_threshold = detector.temp_threshold
        
        # Process at each threshold
        for threshold in self.thresholds:
            # Update detector threshold
            detector.temp_threshold = threshold
            
            # Process frame
            try:
                sgds = detector.process_frame(frame_index)
                
                if sgds:
                    frame_detections[threshold] = sgds
                    
                    # Store in overall detections
                    for sgd in sgds:
                        sgd['threshold'] = threshold
                        self.detections_by_threshold[threshold].append(sgd)
                        
            except Exception as e:
                print(f"Error processing at threshold {threshold}: {e}")
                frame_detections[threshold] = []
        
        # Restore original threshold
        detector.temp_threshold = original_threshold
        
        return frame_detections
    
    def merge_overlapping_by_threshold(self):
        """
        Merge overlapping polygons within each threshold group.
        
        Returns:
            Dictionary of merged polygons by threshold
        """
        if not SHAPELY_AVAILABLE:
            print("Shapely not available - returning unmerged polygons")
            return self.detections_by_threshold
        
        merged_by_threshold = {}
        
        for threshold, detections in self.detections_by_threshold.items():
            if not detections:
                continue
            
            # Extract polygons
            polygons = []
            for sgd in detections:
                if 'polygon' in sgd and sgd['polygon']:
                    poly_coords = []
                    for point in sgd['polygon']:
                        if isinstance(point, dict):
                            poly_coords.append((point['lon'], point['lat']))
                        elif isinstance(point, (list, tuple)) and len(point) >= 2:
                            poly_coords.append((point[0], point[1]))
                    
                    if len(poly_coords) >= 3:
                        try:
                            poly = Polygon(poly_coords)
                            if poly.is_valid:
                                polygons.append(poly)
                        except:
                            continue
            
            if polygons:
                # Merge overlapping polygons
                try:
                    merged = unary_union(polygons)
                    
                    # Convert back to coordinate lists
                    if isinstance(merged, Polygon):
                        merged_polys = [merged]
                    elif isinstance(merged, MultiPolygon):
                        merged_polys = list(merged.geoms)
                    else:
                        merged_polys = []
                    
                    # Store merged polygons
                    merged_by_threshold[threshold] = []
                    for poly in merged_polys:
                        coords = list(poly.exterior.coords[:-1])  # Remove duplicate closing point
                        merged_by_threshold[threshold].append({
                            'polygon': coords,
                            'threshold': threshold,
                            'area_m2': poly.area * (111000 ** 2),  # Rough conversion
                            'color': self.get_color_for_threshold(threshold)
                        })
                    
                    print(f"  Threshold {threshold}°C: {len(polygons)} polygons → {len(merged_polys)} merged")
                    
                except Exception as e:
                    print(f"Error merging at threshold {threshold}: {e}")
                    merged_by_threshold[threshold] = detections
            else:
                merged_by_threshold[threshold] = detections
        
        return merged_by_threshold
    
    def create_threshold_kml(self, threshold, detections, output_file):
        """
        Create KML for a single threshold level.
        
        Args:
            threshold: Temperature threshold
            detections: List of SGD detections
            output_file: Output KML path
        """
        color_info = self.get_color_for_threshold(threshold)
        
        kml_content = []
        kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml_content.append('<Document>')
        kml_content.append(f'<name>SGD Detection - {threshold}°C Threshold</name>')
        kml_content.append(f'<description>SGDs detected with temperature anomaly ≥ {threshold}°C</description>')
        
        # Style for this threshold
        style_id = f'sgd_{threshold}'.replace('.', '_')
        kml_content.append(f'<Style id="{style_id}">')
        kml_content.append('  <LineStyle>')
        kml_content.append(f'    <color>{color_info["kml"]}</color>')
        kml_content.append('    <width>2</width>')
        kml_content.append('  </LineStyle>')
        kml_content.append('  <PolyStyle>')
        kml_content.append(f'    <color>{color_info["kml"]}</color>')
        kml_content.append('  </PolyStyle>')
        kml_content.append('</Style>')
        
        # Add folder
        kml_content.append('<Folder>')
        kml_content.append(f'<name>{threshold}°C Threshold SGDs</name>')
        kml_content.append('<open>1</open>')
        
        # Add each detection
        for i, sgd in enumerate(detections, 1):
            kml_content.append('<Placemark>')
            kml_content.append(f'  <name>SGD {i} ({threshold}°C)</name>')
            
            if 'area_m2' in sgd:
                kml_content.append(f'  <description>Area: {sgd["area_m2"]:.1f} m²</description>')
            
            kml_content.append(f'  <styleUrl>#{style_id}</styleUrl>')
            
            # Add polygon if available
            if 'polygon' in sgd and sgd['polygon']:
                kml_content.append('  <Polygon>')
                kml_content.append('    <outerBoundaryIs>')
                kml_content.append('      <LinearRing>')
                kml_content.append('        <coordinates>')
                
                # Add coordinates
                for coord in sgd['polygon']:
                    if isinstance(coord, dict):
                        kml_content.append(f'          {coord["lon"]},{coord["lat"]},0')
                    else:
                        kml_content.append(f'          {coord[0]},{coord[1]},0')
                
                # Close polygon
                first_coord = sgd['polygon'][0]
                if isinstance(first_coord, dict):
                    kml_content.append(f'          {first_coord["lon"]},{first_coord["lat"]},0')
                else:
                    kml_content.append(f'          {first_coord[0]},{first_coord[1]},0')
                
                kml_content.append('        </coordinates>')
                kml_content.append('      </LinearRing>')
                kml_content.append('    </outerBoundaryIs>')
                kml_content.append('  </Polygon>')
            
            kml_content.append('</Placemark>')
        
        kml_content.append('</Folder>')
        kml_content.append('</Document>')
        kml_content.append('</kml>')
        
        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(kml_content))
        
        print(f"✓ Created {output_file}")
    
    def create_combined_kml(self, merged_by_threshold, output_file):
        """
        Create combined KML with all thresholds in different colors.
        
        Args:
            merged_by_threshold: Dictionary of merged polygons by threshold
            output_file: Output KML path
        """
        kml_content = []
        kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml_content.append('<Document>')
        kml_content.append(f'<name>Multi-Threshold SGD Analysis</name>')
        kml_content.append('<description>')
        kml_content.append(f'Temperature thresholds: {", ".join(f"{t}°C" for t in self.thresholds)}\n')
        kml_content.append('Color coding by temperature anomaly strength')
        kml_content.append('</description>')
        
        # Add styles for each threshold
        for threshold in self.thresholds:
            color_info = self.get_color_for_threshold(threshold)
            style_id = f'sgd_{threshold}'.replace('.', '_')
            
            kml_content.append(f'<Style id="{style_id}">')
            kml_content.append('  <LineStyle>')
            kml_content.append(f'    <color>{color_info["kml"]}</color>')
            kml_content.append('    <width>2</width>')
            kml_content.append('  </LineStyle>')
            kml_content.append('  <PolyStyle>')
            kml_content.append(f'    <color>{color_info["kml"]}</color>')
            kml_content.append('  </PolyStyle>')
            kml_content.append('</Style>')
        
        # Add folders for each threshold (reverse order so strongest on top)
        for threshold in reversed(self.thresholds):
            if threshold not in merged_by_threshold:
                continue
            
            detections = merged_by_threshold[threshold]
            if not detections:
                continue
            
            color_info = self.get_color_for_threshold(threshold)
            style_id = f'sgd_{threshold}'.replace('.', '_')
            
            kml_content.append('<Folder>')
            kml_content.append(f'<name>{threshold}°C Threshold ({color_info["name"]})</name>')
            kml_content.append(f'<open>1</open>')
            
            # Add placemarks
            for i, sgd in enumerate(detections, 1):
                kml_content.append('<Placemark>')
                kml_content.append(f'  <name>SGD {threshold}°C-{i}</name>')
                
                if 'area_m2' in sgd:
                    kml_content.append(f'  <description>Threshold: {threshold}°C\nArea: {sgd["area_m2"]:.1f} m²</description>')
                else:
                    kml_content.append(f'  <description>Threshold: {threshold}°C</description>')
                
                kml_content.append(f'  <styleUrl>#{style_id}</styleUrl>')
                
                # Add polygon
                if 'polygon' in sgd and sgd['polygon']:
                    kml_content.append('  <Polygon>')
                    kml_content.append('    <outerBoundaryIs>')
                    kml_content.append('      <LinearRing>')
                    kml_content.append('        <coordinates>')
                    
                    for coord in sgd['polygon']:
                        if isinstance(coord, dict):
                            kml_content.append(f'          {coord["lon"]},{coord["lat"]},0')
                        else:
                            kml_content.append(f'          {coord[0]},{coord[1]},0')
                    
                    # Close polygon
                    first_coord = sgd['polygon'][0]
                    if isinstance(first_coord, dict):
                        kml_content.append(f'          {first_coord["lon"]},{first_coord["lat"]},0')
                    else:
                        kml_content.append(f'          {first_coord[0]},{first_coord[1]},0')
                    
                    kml_content.append('        </coordinates>')
                    kml_content.append('      </LinearRing>')
                    kml_content.append('    </outerBoundaryIs>')
                    kml_content.append('  </Polygon>')
                
                kml_content.append('</Placemark>')
            
            kml_content.append('</Folder>')
        
        kml_content.append('</Document>')
        kml_content.append('</kml>')
        
        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(kml_content))
        
        print(f"✓ Created combined multi-threshold KML: {output_file}")
    
    def run_multi_threshold(self):
        """
        Run SGD detection at multiple temperature thresholds.
        
        Returns:
            Dictionary of results for each threshold
        """
        from sgd_autodetect import SGDAutoDetector
        
        all_results = {}
        
        # Process each threshold
        for threshold in self.thresholds:
            print(f"\n{'='*60}")
            print(f"Processing threshold: {threshold}°C")
            print('='*60)
            
            # Create output filename for this threshold
            base_name = Path(self.base_output).stem
            base_ext = Path(self.base_output).suffix
            threshold_output = f"{base_name}_threshold_{threshold}{base_ext}"
            
            # Create detector with this threshold
            detector = SGDAutoDetector(
                data_dir=self.data_dir,
                output_file=threshold_output,
                temp_threshold=threshold,
                distance_threshold=self.distance_threshold,
                frame_skip=self.frame_skip,
                min_area=self.min_area,
                include_waves=self.include_waves,
                verbose=self.verbose
            )
            
            try:
                # Run detection
                stats = detector.run()
                
                # Store results
                result = {
                    'threshold': threshold,
                    'stats': stats,
                    'output_file': threshold_output
                }
                
                # Get SGD polygons if available
                if hasattr(detector, 'georef') and hasattr(detector.georef, 'sgd_polygons'):
                    result['sgd_polygons'] = detector.georef.sgd_polygons
                    
                    # Store in our internal structure
                    self.detections_by_threshold[threshold] = detector.georef.sgd_polygons
                
                all_results[threshold] = result
                
                print(f"✓ Threshold {threshold}°C: {stats.get('total_detections', 0)} SGDs detected")
                
            except Exception as e:
                print(f"Error processing threshold {threshold}: {e}")
                all_results[threshold] = {
                    'threshold': threshold,
                    'error': str(e),
                    'stats': {'total_detections': 0, 'unique_locations': 0}
                }
        
        return all_results
    
    def create_all_threshold_kml(self, all_results):
        """
        Create a combined KML with all threshold results.
        
        Args:
            all_results: Dictionary of results from run_multi_threshold
            
        Returns:
            Path to combined KML file
        """
        # Prepare merged detections by threshold
        merged_by_threshold = {}
        
        for threshold, result in all_results.items():
            if 'sgd_polygons' in result and result['sgd_polygons']:
                merged_by_threshold[threshold] = result['sgd_polygons']
        
        # Create combined output filename
        base_name = Path(self.base_output).stem
        combined_output = f"{base_name}_combined_thresholds.kml"
        
        # Use the create_combined_kml method to create multi-colored KML
        self.create_combined_kml(merged_by_threshold, combined_output)
        
        return combined_output
    
    def get_aggregated_stats(self, all_results):
        """
        Get aggregated statistics from all threshold runs.
        
        Args:
            all_results: Dictionary of results from run_multi_threshold
            
        Returns:
            Aggregated statistics dictionary
        """
        total_detections = 0
        unique_locations = 0
        threshold_summary = {}
        
        for threshold, result in all_results.items():
            if 'stats' in result and result['stats']:
                stats = result['stats']
                detections = stats.get('total_detections', 0)
                unique = stats.get('unique_locations', 0)
                
                total_detections += detections
                threshold_summary[f"threshold_{threshold}"] = {
                    'detections': detections,
                    'unique': unique
                }
                
                # Use the max unique locations (since higher thresholds are subsets)
                if unique > unique_locations:
                    unique_locations = unique
        
        return {
            'multi_threshold_analysis': True,
            'thresholds_analyzed': list(all_results.keys()),
            'total_detections': total_detections,
            'unique_locations': unique_locations,
            'threshold_summary': threshold_summary,
            'base_threshold': self.base_threshold,
            'interval_step': self.interval_step,
            'num_steps': self.num_steps
        }
    
    def export_all_kmls(self, base_output_path):
        """
        Export all KML files for multi-threshold analysis.
        
        Args:
            base_output_path: Base path for output files
            
        Returns:
            List of created KML files
        """
        base_path = Path(base_output_path)
        base_name = base_path.stem
        output_dir = base_path.parent
        
        created_files = []
        
        # Merge overlapping polygons within each threshold
        print("\nMerging overlapping polygons by threshold...")
        merged_by_threshold = self.merge_overlapping_by_threshold()
        
        # Create individual KML for each threshold
        print("\nCreating individual threshold KMLs...")
        for threshold in self.thresholds:
            if threshold in merged_by_threshold:
                threshold_file = output_dir / f"{base_name}_{threshold}C.kml"
                self.create_threshold_kml(
                    threshold, 
                    merged_by_threshold[threshold],
                    threshold_file
                )
                created_files.append(threshold_file)
                
                # Also create merged version
                if SHAPELY_AVAILABLE:
                    from merge_sgd_polygons import create_merged_kml
                    merged_file = output_dir / f"{base_name}_{threshold}C_merged.kml"
                    try:
                        create_merged_kml(
                            merged_by_threshold[threshold],
                            merged_file,
                            use_shapely=True
                        )
                        created_files.append(merged_file)
                    except:
                        pass
        
        # Create combined multi-threshold KML
        print("\nCreating combined multi-threshold KML...")
        combined_file = output_dir / f"{base_name}_multi_threshold.kml"
        self.create_combined_kml(merged_by_threshold, combined_file)
        created_files.append(combined_file)
        
        # Create summary statistics
        summary = {
            'thresholds': self.thresholds,
            'detection_counts': {
                str(t): len(self.detections_by_threshold[t]) 
                for t in self.thresholds
            },
            'merged_counts': {
                str(t): len(merged_by_threshold.get(t, []))
                for t in self.thresholds
            },
            'files_created': [str(f) for f in created_files]
        }
        
        summary_file = output_dir / f"{base_name}_multi_threshold_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Multi-threshold analysis complete!")
        print(f"  Created {len(created_files)} KML files")
        print(f"  Summary saved to: {summary_file}")
        
        return created_files