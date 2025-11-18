#!/usr/bin/env python3
"""
Test the polygon merging functionality.
"""

from pathlib import Path
import json
from merge_sgd_polygons import merge_polygons_shapely, create_merged_kml

def test_merge():
    """Test polygon merging with sample data."""
    
    print("="*60)
    print("TESTING POLYGON MERGE FUNCTIONALITY")
    print("="*60)
    
    # Create sample overlapping polygons (as would come from SGD detections)
    # These are lon/lat coordinates representing overlapping detection areas
    test_polygons = [
        # First SGD polygon
        [
            (-109.4401, -27.1501),
            (-109.4400, -27.1501),
            (-109.4400, -27.1500),
            (-109.4401, -27.1500)
        ],
        # Second SGD polygon (overlaps with first)
        [
            (-109.4400, -27.1501),
            (-109.4399, -27.1501),
            (-109.4399, -27.1500),
            (-109.4400, -27.1500)
        ],
        # Third SGD polygon (separate)
        [
            (-109.4395, -27.1502),
            (-109.4394, -27.1502),
            (-109.4394, -27.1501),
            (-109.4395, -27.1501)
        ],
        # Fourth SGD polygon (overlaps with third)
        [
            (-109.4395, -27.1502),
            (-109.4394, -27.1502),
            (-109.4394, -27.1503),
            (-109.4395, -27.1503)
        ]
    ]
    
    print(f"\nInput: {len(test_polygons)} overlapping SGD polygons")
    
    # Test merging with Shapely
    try:
        merged = merge_polygons_shapely(test_polygons)
        print(f"Merged: {len(merged)} unified shapes (using Shapely)")
        
        for i, poly in enumerate(merged, 1):
            print(f"  Shape {i}: {len(poly)} vertices")
    except Exception as e:
        print(f"Error during merging: {e}")
        return
    
    # Create sample SGD data structure
    test_sgds = []
    for i, poly in enumerate(test_polygons):
        sgd = {
            'centroid_lat': sum(p[1] for p in poly) / len(poly),
            'centroid_lon': sum(p[0] for p in poly) / len(poly),
            'polygon': poly,
            'area_m2': 50 + i * 10,
            'temperature_anomaly': 1.5 + i * 0.1,
            'frame': f"Frame_{i:04d}"
        }
        test_sgds.append(sgd)
    
    # Test KML creation
    output_file = Path("test_merged.kml")
    
    print(f"\nCreating merged KML: {output_file}")
    result = create_merged_kml(test_sgds, output_file, use_shapely=True)
    
    if result:
        print(f"✓ Success! Merged KML created: {result}")
        print(f"  File size: {Path(result).stat().st_size} bytes")
        
        # Read and display part of the KML
        with open(result, 'r') as f:
            lines = f.readlines()
            print(f"  KML lines: {len(lines)}")
            
            # Find and count placemarks
            placemarks = [l for l in lines if '<Placemark>' in l]
            print(f"  Placemarks: {len(placemarks)}")
    else:
        print("✗ Failed to create merged KML")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    
    return result

def test_with_real_data():
    """Test with actual SGD output if available."""
    
    print("\nLooking for real SGD output files...")
    
    sgd_output = Path("sgd_output")
    if not sgd_output.exists():
        print("No sgd_output directory found")
        return
    
    # Find JSON files with polygon data
    json_files = list(sgd_output.glob("*_polygons.json"))
    
    if not json_files:
        print("No polygon JSON files found in sgd_output/")
        return
    
    print(f"Found {len(json_files)} polygon files:")
    for jf in json_files[:3]:  # Show first 3
        print(f"  - {jf.name}")
    
    # Test with first file
    test_file = json_files[0]
    print(f"\nTesting with: {test_file.name}")
    
    from merge_sgd_polygons import merge_sgd_outputs
    
    result = merge_sgd_outputs(json_file=str(test_file))
    
    if result:
        print(f"✓ Created merged KML: {result}")
    else:
        print("✗ Failed to create merged KML")

if __name__ == "__main__":
    # Run tests
    test_merge()
    test_with_real_data()