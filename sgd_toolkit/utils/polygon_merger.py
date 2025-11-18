#!/usr/bin/env python3
"""
Merge overlapping SGD polygons into unified shapes for better visualization.

This module takes detected SGD polygons and merges overlapping ones into single
shapes, making it easier to see the overall distribution of SGD areas.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Try to import shapely for polygon operations
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.geometry.polygon import orient
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not installed. Install with: pip install shapely")
    print("Polygon merging will use simplified bounding box approach.")


def merge_polygons_shapely(polygons):
    """
    Merge overlapping polygons using Shapely library.
    
    Args:
        polygons: List of polygon coordinate lists [[(lon, lat), ...], ...]
        
    Returns:
        List of merged polygon coordinate lists
    """
    if not polygons:
        return []
    
    # Convert to Shapely polygons
    shapely_polygons = []
    for poly_coords in polygons:
        if len(poly_coords) >= 3:  # Valid polygon needs at least 3 points
            try:
                # Ensure polygon is closed
                if poly_coords[0] != poly_coords[-1]:
                    poly_coords = poly_coords + [poly_coords[0]]
                
                # Create Shapely polygon
                poly = Polygon(poly_coords)
                
                # Fix any self-intersections
                if not poly.is_valid:
                    poly = poly.buffer(0)
                
                if poly.is_valid and not poly.is_empty:
                    shapely_polygons.append(poly)
            except Exception as e:
                print(f"Warning: Could not create polygon: {e}")
                continue
    
    if not shapely_polygons:
        return []
    
    # Merge all overlapping polygons
    merged = unary_union(shapely_polygons)
    
    # Convert back to coordinate lists
    merged_polygons = []
    
    # Handle both single polygon and multipolygon results
    if isinstance(merged, Polygon):
        merged = [merged]
    elif isinstance(merged, MultiPolygon):
        merged = list(merged.geoms)
    else:
        return []
    
    for poly in merged:
        # Ensure exterior ring is counter-clockwise (KML convention)
        poly = orient(poly, sign=1.0)
        
        # Extract coordinates
        coords = list(poly.exterior.coords)
        
        # Remove duplicate closing point for our format
        if coords[0] == coords[-1]:
            coords = coords[:-1]
        
        merged_polygons.append(coords)
    
    return merged_polygons


def merge_polygons_simple(polygons):
    """
    Simple polygon merging using bounding boxes (fallback when Shapely not available).
    
    Args:
        polygons: List of polygon coordinate lists [[(lon, lat), ...], ...]
        
    Returns:
        List of merged bounding boxes as polygons
    """
    if not polygons:
        return []
    
    # Convert to bounding boxes
    bboxes = []
    for poly_coords in polygons:
        if len(poly_coords) >= 3:
            lons = [c[0] for c in poly_coords]
            lats = [c[1] for c in poly_coords]
            bboxes.append({
                'min_lon': min(lons),
                'max_lon': max(lons),
                'min_lat': min(lats),
                'max_lat': max(lats),
                'merged': False
            })
    
    # Merge overlapping bounding boxes
    merged_bboxes = []
    
    for i, bbox1 in enumerate(bboxes):
        if bbox1['merged']:
            continue
            
        # Start with current bbox
        merged_bbox = bbox1.copy()
        merged_bbox['merged'] = True
        
        # Check for overlaps with remaining bboxes
        changed = True
        while changed:
            changed = False
            for j, bbox2 in enumerate(bboxes):
                if i == j or bbox2['merged']:
                    continue
                
                # Check if bboxes overlap
                if (merged_bbox['min_lon'] <= bbox2['max_lon'] and 
                    merged_bbox['max_lon'] >= bbox2['min_lon'] and
                    merged_bbox['min_lat'] <= bbox2['max_lat'] and 
                    merged_bbox['max_lat'] >= bbox2['min_lat']):
                    
                    # Merge bboxes
                    merged_bbox['min_lon'] = min(merged_bbox['min_lon'], bbox2['min_lon'])
                    merged_bbox['max_lon'] = max(merged_bbox['max_lon'], bbox2['max_lon'])
                    merged_bbox['min_lat'] = min(merged_bbox['min_lat'], bbox2['min_lat'])
                    merged_bbox['max_lat'] = max(merged_bbox['max_lat'], bbox2['max_lat'])
                    bbox2['merged'] = True
                    changed = True
        
        merged_bboxes.append(merged_bbox)
    
    # Convert bounding boxes back to polygons
    merged_polygons = []
    for bbox in merged_bboxes:
        poly = [
            (bbox['min_lon'], bbox['min_lat']),
            (bbox['max_lon'], bbox['min_lat']),
            (bbox['max_lon'], bbox['max_lat']),
            (bbox['min_lon'], bbox['max_lat'])
        ]
        merged_polygons.append(poly)
    
    return merged_polygons


def create_merged_kml(sgd_locations, output_file, use_shapely=True):
    """
    Create a KML file with merged SGD polygons.
    
    Args:
        sgd_locations: List of SGD detection dictionaries with polygon data
        output_file: Output KML filename
        use_shapely: Whether to use Shapely for accurate merging
        
    Returns:
        Path to created KML file
    """
    # Extract all polygons
    all_polygons = []
    
    for sgd in sgd_locations:
        if 'polygon' in sgd and sgd['polygon']:
            # Convert polygon format if needed
            polygon_coords = []
            for point in sgd['polygon']:
                if isinstance(point, dict):
                    polygon_coords.append((point['lon'], point['lat']))
                elif isinstance(point, (list, tuple)) and len(point) >= 2:
                    polygon_coords.append((point[0], point[1]))
            
            if polygon_coords:
                all_polygons.append(polygon_coords)
    
    if not all_polygons:
        print("No polygons found to merge")
        return None
    
    print(f"Merging {len(all_polygons)} SGD polygons...")
    
    # Merge polygons
    if use_shapely and SHAPELY_AVAILABLE:
        merged_polygons = merge_polygons_shapely(all_polygons)
        merge_method = "Shapely (accurate)"
    else:
        merged_polygons = merge_polygons_simple(all_polygons)
        merge_method = "Bounding box (approximate)"
    
    print(f"Merged into {len(merged_polygons)} unified shapes using {merge_method}")
    
    # Calculate total area
    total_area = 0
    if SHAPELY_AVAILABLE:
        for poly_coords in merged_polygons:
            try:
                # Create polygon and calculate area (in square degrees, approximate)
                poly = Polygon(poly_coords)
                # Rough conversion: 1 degree ≈ 111km at equator
                # For more accurate area, would need proper projection
                area_deg = poly.area
                area_m2 = area_deg * (111000 ** 2)  # Very rough approximation
                total_area += area_m2
            except:
                pass
    
    # Create KML content
    kml_content = []
    kml_content.append('<?xml version="1.0" encoding="UTF-8"?>')
    kml_content.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    kml_content.append('<Document>')
    kml_content.append(f'<name>Merged SGD Areas - {datetime.now().strftime("%Y-%m-%d %H:%M")}</name>')
    kml_content.append('<description>')
    kml_content.append(f'Merged overlapping SGD polygons into {len(merged_polygons)} unified shapes\n')
    kml_content.append(f'Original detections: {len(all_polygons)}\n')
    kml_content.append(f'Merge method: {merge_method}\n')
    if SHAPELY_AVAILABLE and total_area > 0:
        kml_content.append(f'Approximate total area: {total_area:.1f} m²\n')
    kml_content.append('</description>')
    
    # Add style for merged polygons
    kml_content.append('<Style id="mergedSGD">')
    kml_content.append('  <LineStyle>')
    kml_content.append('    <color>ff0000ff</color>')  # Red outline
    kml_content.append('    <width>2</width>')
    kml_content.append('  </LineStyle>')
    kml_content.append('  <PolyStyle>')
    kml_content.append('    <color>7f0000ff</color>')  # Semi-transparent red fill
    kml_content.append('  </PolyStyle>')
    kml_content.append('</Style>')
    
    # Add folder for merged polygons
    kml_content.append('<Folder>')
    kml_content.append('<name>Merged SGD Areas</name>')
    kml_content.append('<open>1</open>')
    
    # Add each merged polygon
    for i, poly_coords in enumerate(merged_polygons, 1):
        kml_content.append('<Placemark>')
        kml_content.append(f'  <name>Merged Area {i}</name>')
        
        # Calculate area if possible
        if SHAPELY_AVAILABLE:
            try:
                poly = Polygon(poly_coords)
                area_deg = poly.area
                area_m2 = area_deg * (111000 ** 2)
                kml_content.append(f'  <description>Approximate area: {area_m2:.1f} m²</description>')
            except:
                kml_content.append('  <description>Merged SGD area</description>')
        else:
            kml_content.append('  <description>Merged SGD area (bounding box)</description>')
        
        kml_content.append('  <styleUrl>#mergedSGD</styleUrl>')
        kml_content.append('  <Polygon>')
        kml_content.append('    <outerBoundaryIs>')
        kml_content.append('      <LinearRing>')
        kml_content.append('        <coordinates>')
        
        # Add coordinates (ensure polygon is closed)
        for lon, lat in poly_coords:
            kml_content.append(f'          {lon},{lat},0')
        # Close the polygon
        if poly_coords[0] != poly_coords[-1]:
            kml_content.append(f'          {poly_coords[0][0]},{poly_coords[0][1]},0')
        
        kml_content.append('        </coordinates>')
        kml_content.append('      </LinearRing>')
        kml_content.append('    </outerBoundaryIs>')
        kml_content.append('  </Polygon>')
        kml_content.append('</Placemark>')
    
    kml_content.append('</Folder>')
    kml_content.append('</Document>')
    kml_content.append('</kml>')
    
    # Write KML file
    with open(output_file, 'w') as f:
        f.write('\n'.join(kml_content))
    
    print(f"✓ Merged KML saved to: {output_file}")
    
    return output_file


def merge_sgd_outputs(kml_file=None, json_file=None, output_suffix="_merged"):
    """
    Convenience function to merge SGD polygons from existing output files.
    
    Args:
        kml_file: Path to existing KML file (will derive JSON path)
        json_file: Path to JSON file with SGD data
        output_suffix: Suffix to add to output filename
        
    Returns:
        Path to merged KML file
    """
    # Determine input files
    if kml_file:
        kml_path = Path(kml_file)
        json_path = kml_path.parent / kml_path.name.replace('.kml', '_polygons.json')
        output_path = kml_path.parent / kml_path.name.replace('.kml', f'{output_suffix}.kml')
    elif json_file:
        json_path = Path(json_file)
        kml_path = json_path.parent / json_path.name.replace('_polygons.json', '.kml')
        output_path = json_path.parent / json_path.name.replace('_polygons.json', f'{output_suffix}.kml')
    else:
        raise ValueError("Must provide either kml_file or json_file")
    
    # Load SGD data
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        sgd_data = json.load(f)
    
    if 'sgd_locations' in sgd_data:
        sgd_locations = sgd_data['sgd_locations']
    else:
        sgd_locations = sgd_data  # Assume it's just a list
    
    print(f"Loaded {len(sgd_locations)} SGD detections from {json_path.name}")
    
    # Create merged KML
    return create_merged_kml(sgd_locations, output_path, use_shapely=True)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
        if input_file.endswith('.kml'):
            output = merge_sgd_outputs(kml_file=input_file)
        elif input_file.endswith('.json'):
            output = merge_sgd_outputs(json_file=input_file)
        else:
            print("Usage: python merge_sgd_polygons.py <file.kml or file.json>")
            sys.exit(1)
            
        if output:
            print(f"\nMerged polygons saved to: {output}")
    else:
        print("Usage: python merge_sgd_polygons.py <file.kml or file.json>")
        print("\nThis will create a _merged.kml file with overlapping SGD polygons combined.")
        print("\nInstall Shapely for accurate polygon merging:")
        print("  pip install shapely")