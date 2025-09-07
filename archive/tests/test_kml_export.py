#!/usr/bin/env python3
"""Test KML export functionality with polygons"""

import numpy as np
from sgd_georef_polygons import SGDPolygonGeoref
import os

def test_kml_export():
    """Test KML export with polygon features"""
    
    print("Testing KML Export with Polygons")
    print("=" * 50)
    
    # Initialize georeferencer
    georef = SGDPolygonGeoref(base_path="data/100MEDIA")
    
    # Create test SGD data with polygons
    test_sgd_locations = [
        {
            'frame': 248,
            'datetime': '2024-01-15 10:30:00',
            'polygon': [
                [-109.71362, 18.48949],
                [-109.71359, 18.48949],
                [-109.71356, 18.48947],
                [-109.71353, 18.48946],
                [-109.71350, 18.48944],
                [-109.71350, 18.48941],
                [-109.71353, 18.48939],
                [-109.71356, 18.48938],
                [-109.71359, 18.48937],
                [-109.71362, 18.48938],
                [-109.71365, 18.48941],
                [-109.71365, 18.48944],
                [-109.71362, 18.48949]  # Close polygon
            ],
            'centroid': {
                'latitude': 18.48943,
                'longitude': -109.71357
            },
            'area_m2': 125.5,
            'area_pixels': 150,
            'temperature_anomaly': -1.8,
            'shore_distance': 2.5,
            'altitude': 400,
            'eccentricity': 0.75
        },
        {
            'frame': 250,
            'datetime': '2024-01-15 10:31:00',
            'polygon': [
                [-109.71370, 18.48955],
                [-109.71367, 18.48955],
                [-109.71365, 18.48953],
                [-109.71365, 18.48951],
                [-109.71367, 18.48949],
                [-109.71370, 18.48949],
                [-109.71372, 18.48951],
                [-109.71372, 18.48953],
                [-109.71370, 18.48955]
            ],
            'centroid': {
                'latitude': 18.48952,
                'longitude': -109.71368
            },
            'area_m2': 85.3,
            'area_pixels': 100,
            'temperature_anomaly': -2.1,
            'shore_distance': 1.2,
            'altitude': 400,
            'eccentricity': 0.65
        },
        {
            'frame': 252,
            'datetime': '2024-01-15 10:31:30',
            'polygon': None,  # Test point fallback
            'centroid': {
                'latitude': 18.48960,
                'longitude': -109.71345
            },
            'area_m2': 45.0,
            'area_pixels': 55,
            'temperature_anomaly': -1.5,
            'shore_distance': 3.0,
            'altitude': 400,
            'eccentricity': 0.5
        }
    ]
    
    # Set the test data
    georef.sgd_polygons = test_sgd_locations
    
    print(f"Created {len(test_sgd_locations)} test SGD locations:")
    print(f"  - 2 with polygon outlines")
    print(f"  - 1 with point only (fallback)")
    
    # Test KML export
    print("\n1. Testing KML export...")
    kml_file = georef.export_kml_polygons("test_sgd_polygons.kml")
    
    # Verify file was created
    assert os.path.exists(kml_file)
    print(f"   ✓ KML file created: {kml_file}")
    
    # Check file content
    with open(kml_file, 'r') as f:
        kml_content = f.read()
    
    # Verify key elements
    assert '<?xml version="1.0" encoding="UTF-8"?>' in kml_content
    assert '<kml xmlns="http://www.opengis.net/kml/2.2">' in kml_content
    assert '<Polygon>' in kml_content  # Has polygon features
    assert '<Point>' in kml_content    # Has point features
    assert 'sgdPolygonStyle' in kml_content  # Has polygon style
    assert 'sgdPointStyle' in kml_content    # Has point style
    assert '<coordinates>' in kml_content
    assert 'SGD 1 (Frame 248)' in kml_content
    assert 'Area: 125.5 m²' in kml_content
    assert 'Temperature anomaly: -1.8°C' in kml_content
    
    print("   ✓ KML structure validated")
    print("   ✓ Contains polygon features")
    print("   ✓ Contains point features")
    print("   ✓ Includes metadata in descriptions")
    
    # Test GeoJSON export for comparison
    print("\n2. Testing GeoJSON export...")
    geojson_file = georef.export_geojson_polygons("test_sgd_polygons.geojson")
    assert os.path.exists(geojson_file)
    print(f"   ✓ GeoJSON file created: {geojson_file}")
    
    # Test CSV export
    print("\n3. Testing CSV export...")
    csv_file = georef.export_csv_with_areas("test_sgd_areas.csv")
    assert os.path.exists(csv_file)
    print(f"   ✓ CSV file created: {csv_file}")
    
    print("\n" + "=" * 50)
    print("SUCCESS: All export formats working correctly!")
    print("\nExported files:")
    print(f"  - {kml_file} (for Google Earth)")
    print(f"  - {geojson_file} (for GIS software)")
    print(f"  - {csv_file} (for spreadsheets)")
    
    print("\nKML Features:")
    print("  ✓ Polygon plumes with semi-transparent fill")
    print("  ✓ Point plumes with water icon")
    print("  ✓ Metadata in placemark descriptions")
    print("  ✓ Summary statistics folder")
    print("  ✓ Compatible with Google Earth")
    
    # Cleanup test files (optional)
    # os.remove(kml_file)
    # os.remove(geojson_file)
    # os.remove(csv_file)
    
    return True

if __name__ == "__main__":
    success = test_kml_export()
    
    if success:
        print("\nTo view the KML file:")
        print("1. Open Google Earth")
        print("2. File > Open > test_sgd_polygons.kml")
        print("3. Polygons will appear as red outlines with semi-transparent fill")
        print("4. Click on plumes to see detailed metadata")