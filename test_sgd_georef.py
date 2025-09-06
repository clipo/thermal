#!/usr/bin/env python3
"""
Test script for SGD detection with georeferencing
"""

from sgd_detector_integrated import IntegratedSGDDetector
from sgd_georef_thermal_subset import ThermalSubsetGeoref as SimpleGeoref
import numpy as np

def test_georeferencing():
    """Test the complete pipeline"""
    
    print("Testing SGD Detection with Georeferencing")
    print("=" * 50)
    
    # 1. Test GPS extraction
    print("\n1. Testing GPS extraction...")
    georef = SimpleGeoref(thermal_fov_ratio=0.7)  # Assuming thermal sees ~70% of RGB FOV
    gps_info = georef.extract_gps_simple("data/100MEDIA/MAX_0248.JPG")
    
    if gps_info and gps_info['lat']:
        print(f"✅ GPS found: {gps_info['lat']:.6f}, {gps_info['lon']:.6f}")
        print(f"   Location: Easter Island (Rapa Nui)")
        print(f"   Altitude: {gps_info.get('altitude', 0):.1f} m")
    else:
        print("❌ No GPS data found")
        return False
    
    # 2. Test SGD detection on single frame
    print("\n2. Testing SGD detection...")
    detector = IntegratedSGDDetector()
    
    try:
        result = detector.process_frame(248, visualize=False)
        print(f"✅ Found {len(result['plume_info'])} SGD plumes")
        
        if result['plume_info']:
            print("   Plume details:")
            for i, plume in enumerate(result['plume_info'][:3], 1):  # Show first 3
                print(f"   - Plume {i}: {plume['area_pixels']} pixels, "
                      f"shore dist: {plume['min_shore_distance']:.1f}")
    except Exception as e:
        print(f"❌ Error in detection: {e}")
        return False
    
    # 3. Test georeferencing
    print("\n3. Testing georeferencing...")
    
    if result['plume_info']:
        locations = georef.process_sgd_frame(248, result['plume_info'])
        
        if locations:
            print(f"✅ Georeferenced {len(locations)} SGD locations:")
            for i, loc in enumerate(locations[:3], 1):  # Show first 3
                print(f"   - SGD {i}: {loc['latitude']:.6f}, {loc['longitude']:.6f}")
                print(f"     Area: {loc['area_m2']:.1f} m²")
            
            # Export test results
            georef.export_to_geojson("test_integrated_sgd.geojson")
            georef.export_to_csv("test_integrated_sgd.csv")
            
            print("\n✅ Created test files:")
            print("   - test_integrated_sgd.geojson")
            print("   - test_integrated_sgd.csv")
            
            return True
    
    return False

def quick_stats():
    """Quick statistics on available data"""
    
    from pathlib import Path
    base_path = Path("data/100MEDIA")
    
    rgb_files = list(base_path.glob("MAX_*.JPG"))
    thermal_files = list(base_path.glob("IRX_*.irg"))
    
    print("\nData Statistics:")
    print(f"  RGB images: {len(rgb_files)}")
    print(f"  Thermal images: {len(thermal_files)}")
    
    if rgb_files:
        # Check how many have GPS
        georef = SimpleGeoref(thermal_fov_ratio=0.7)
        gps_count = 0
        
        for rgb_file in rgb_files[:10]:  # Check first 10
            gps = georef.extract_gps_simple(str(rgb_file))
            if gps and gps.get('lat'):
                gps_count += 1
        
        print(f"  GPS data: {gps_count}/10 checked have GPS")

if __name__ == "__main__":
    # Run tests
    if test_georeferencing():
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("The system can successfully:")
        print("  1. Extract GPS coordinates from drone images")
        print("  2. Detect SGD plumes in thermal data")
        print("  3. Georeference plumes to real-world coordinates")
        print("  4. Export to GIS-compatible formats")
        
        quick_stats()
    else:
        print("\n❌ Tests failed - check error messages above")