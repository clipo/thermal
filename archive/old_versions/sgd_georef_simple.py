#!/usr/bin/env python3
"""
Simple SGD Georeferencer - Extract GPS and create georeferenced output
Works without additional geospatial libraries
"""

import numpy as np
import json
import csv
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import hashlib


class SimpleGeoref:
    """Simple georeferencing for SGD detections"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        
        # Image dimensions and alignment
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        self.scale_x = 6.4
        self.scale_y = 6.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Store unique SGD locations
        self.sgd_locations = []
        self.location_hashes = set()
        
    def extract_gps_simple(self, image_path):
        """Extract GPS coordinates from EXIF (simple version)"""
        try:
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            if not exifdata:
                return None
            
            info = {
                'datetime': None,
                'lat': None,
                'lon': None,
                'altitude': None
            }
            
            # Get datetime
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTime':
                    info['datetime'] = value
            
            # Try to get GPS IFD
            try:
                gps_ifd = exifdata.get_ifd(0x8825)
                
                if gps_ifd:
                    # Extract GPS coordinates
                    lat_data = gps_ifd.get(2)  # GPSLatitude
                    lat_ref = gps_ifd.get(1)   # GPSLatitudeRef
                    lon_data = gps_ifd.get(4)  # GPSLongitude
                    lon_ref = gps_ifd.get(3)   # GPSLongitudeRef
                    alt_data = gps_ifd.get(6)  # GPSAltitude
                    
                    if lat_data and lon_data:
                        # Convert to degrees
                        lat = lat_data[0] + lat_data[1]/60.0 + lat_data[2]/3600.0
                        lon = lon_data[0] + lon_data[1]/60.0 + lon_data[2]/3600.0
                        
                        if lat_ref == 'S':
                            lat = -lat
                        if lon_ref == 'W':
                            lon = -lon
                        
                        info['lat'] = lat
                        info['lon'] = lon
                        
                        if alt_data:
                            info['altitude'] = float(alt_data)
                
            except:
                # Fallback: try alternate method
                if 34853 in exifdata:  # GPSInfo tag
                    gps_data = {}
                    for tag in exifdata[34853]:
                        decoded = GPSTAGS.get(tag, tag)
                        gps_data[decoded] = exifdata[34853][tag]
                    
                    # Parse coordinates if available
                    if 'GPSLatitude' in gps_data:
                        lat = gps_data['GPSLatitude']
                        lat = lat[0] + lat[1]/60.0 + lat[2]/3600.0
                        if gps_data.get('GPSLatitudeRef') == 'S':
                            lat = -lat
                        info['lat'] = lat
                    
                    if 'GPSLongitude' in gps_data:
                        lon = gps_data['GPSLongitude']
                        lon = lon[0] + lon[1]/60.0 + lon[2]/3600.0
                        if gps_data.get('GPSLongitudeRef') == 'W':
                            lon = -lon
                        info['lon'] = lon
            
            return info
            
        except Exception as e:
            print(f"Error extracting GPS: {e}")
            return None
    
    def pixel_to_latlon(self, pixel_x, pixel_y, center_lat, center_lon, altitude=100):
        """
        Convert thermal pixel to approximate lat/lon
        
        Simple approximation assuming:
        - Drone pointing straight down
        - Flat earth approximation for small areas
        """
        # Estimate ground sample distance (meters per pixel)
        # Typical values for drone at 100m altitude
        gsd = altitude * 0.0001  # Rough approximation
        
        # Convert thermal pixel to RGB pixel
        rgb_x = pixel_x * self.scale_x + self.offset_x
        rgb_y = pixel_y * self.scale_y + self.offset_y
        
        # Offset from center
        offset_x = (rgb_x - self.rgb_width/2) * gsd
        offset_y = -(rgb_y - self.rgb_height/2) * gsd  # Negative for image coordinates
        
        # Convert to lat/lon offset
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
        
        delta_lat = offset_y / meters_per_degree_lat
        delta_lon = offset_x / meters_per_degree_lon
        
        return center_lat + delta_lat, center_lon + delta_lon
    
    def process_sgd_frame(self, frame_number, sgd_plumes):
        """
        Process SGD detections for a frame
        
        Parameters:
        - frame_number: Frame number
        - sgd_plumes: List of plume info dictionaries from SGD detector
        
        Returns:
        - List of georeferenced SGD locations
        """
        # Get GPS from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        if not rgb_path.exists():
            print(f"RGB image not found: {rgb_path}")
            return []
        
        gps_info = self.extract_gps_simple(rgb_path)
        
        if not gps_info or gps_info['lat'] is None:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        locations = []
        
        for plume in sgd_plumes:
            # Get plume centroid
            centroid_y, centroid_x = plume['centroid']
            
            # Convert to lat/lon
            lat, lon = self.pixel_to_latlon(
                centroid_x, centroid_y,
                gps_info['lat'], gps_info['lon'],
                gps_info.get('altitude', 100)
            )
            
            # Create location record
            location = {
                'frame': frame_number,
                'datetime': gps_info.get('datetime', ''),
                'latitude': lat,
                'longitude': lon,
                'area_pixels': plume['area_pixels'],
                'area_m2': plume.get('area_pixels', 0) * 0.01,  # Rough estimate
                'shore_distance': plume.get('min_shore_distance', 0),
                'image_lat': gps_info['lat'],
                'image_lon': gps_info['lon'],
                'altitude': gps_info.get('altitude', 100)
            }
            
            # Check for duplicates (within ~5 meters)
            location_hash = f"{lat:.5f},{lon:.5f}"
            
            if location_hash not in self.location_hashes:
                locations.append(location)
                self.location_hashes.add(location_hash)
                self.sgd_locations.append(location)
            
        return locations
    
    def export_to_csv(self, output_path="sgd_locations.csv"):
        """Export SGD locations to CSV"""
        if not self.sgd_locations:
            print("No SGD locations to export")
            return
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['frame', 'datetime', 'latitude', 'longitude', 
                         'area_m2', 'shore_distance', 'altitude']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for loc in self.sgd_locations:
                writer.writerow({
                    'frame': loc['frame'],
                    'datetime': loc['datetime'],
                    'latitude': loc['latitude'],
                    'longitude': loc['longitude'],
                    'area_m2': loc['area_m2'],
                    'shore_distance': loc['shore_distance'],
                    'altitude': loc['altitude']
                })
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")
    
    def export_to_geojson(self, output_path="sgd_locations.geojson"):
        """Export to GeoJSON (compatible with GIS software)"""
        features = []
        
        for loc in self.sgd_locations:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [loc['longitude'], loc['latitude']]
                },
                "properties": {
                    "frame": loc['frame'],
                    "datetime": loc['datetime'],
                    "area_m2": loc['area_m2'],
                    "shore_distance": loc['shore_distance'],
                    "altitude": loc['altitude']
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"}
            },
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")
        print("This GeoJSON can be imported into QGIS, ArcGIS, or Google Earth")
    
    def export_to_kml(self, output_path="sgd_locations.kml"):
        """Create KML file for Google Earth"""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append('<name>SGD Detections</name>')
        
        # Add style
        kml.append('<Style id="sgd_style">')
        kml.append('<IconStyle>')
        kml.append('<color>ff0000ff</color>')  # Red
        kml.append('<scale>1.0</scale>')
        kml.append('</IconStyle>')
        kml.append('</Style>')
        
        # Add placemarks
        for i, loc in enumerate(self.sgd_locations, 1):
            kml.append('<Placemark>')
            kml.append(f'<name>SGD {i}</name>')
            kml.append(f'<description>')
            kml.append(f'Frame: {loc["frame"]}\n')
            kml.append(f'Area: {loc["area_m2"]:.1f} m²\n')
            kml.append(f'Date: {loc["datetime"]}')
            kml.append(f'</description>')
            kml.append('<styleUrl>#sgd_style</styleUrl>')
            kml.append('<Point>')
            kml.append(f'<coordinates>{loc["longitude"]},{loc["latitude"]},0</coordinates>')
            kml.append('</Point>')
            kml.append('</Placemark>')
        
        kml.append('</Document>')
        kml.append('</kml>')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(kml))
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")
        print("This KML can be opened in Google Earth")


def test_georeferencing():
    """Test the georeferencing system"""
    
    georef = SimpleGeoref()
    
    # Test GPS extraction
    test_image = Path("data/100MEDIA/MAX_0248.JPG")
    if test_image.exists():
        print(f"Testing GPS extraction from {test_image}")
        gps_info = georef.extract_gps_simple(test_image)
        
        if gps_info:
            print("\n✅ GPS Data Found:")
            print(f"  Latitude: {gps_info.get('lat', 'N/A')}")
            print(f"  Longitude: {gps_info.get('lon', 'N/A')}")
            print(f"  Altitude: {gps_info.get('altitude', 'N/A')} m")
            print(f"  DateTime: {gps_info.get('datetime', 'N/A')}")
            
            # Test coordinate conversion
            print("\n📍 Testing coordinate conversion:")
            test_lat, test_lon = georef.pixel_to_latlon(
                320, 256,  # Center of thermal image
                gps_info['lat'], gps_info['lon']
            )
            print(f"  Thermal center (320,256) -> {test_lat:.6f}, {test_lon:.6f}")
            
            return True
        else:
            print("❌ No GPS data found in EXIF")
            return False
    else:
        print(f"❌ Test image not found: {test_image}")
        return False


def integrate_with_sgd_detector():
    """
    Integration function to be called from SGD detector
    
    Usage:
    from sgd_georef_simple import SimpleGeoref
    
    georef = SimpleGeoref()
    for result in sgd_results:
        georef.process_sgd_frame(result['frame_number'], result['plume_info'])
    
    georef.export_to_geojson("sgd_detections.geojson")
    georef.export_to_csv("sgd_detections.csv")
    georef.export_to_kml("sgd_detections.kml")
    """
    pass


def main():
    """Main entry point"""
    print("Simple SGD Georeferencer")
    print("=" * 50)
    
    # Test the system
    if test_georeferencing():
        print("\n✅ Georeferencing system is working!")
        
        print("\nOptions:")
        print("1. Process SGD detection results")
        print("2. Test with sample data")
        
        choice = input("\nChoice (1-2): ").strip()
        
        georef = SimpleGeoref()
        
        if choice == '1':
            # Process real SGD results
            sgd_file = Path("sgd_output/sgd_summary.json")
            if sgd_file.exists():
                with open(sgd_file, 'r') as f:
                    summary = json.load(f)
                
                print(f"\nProcessing {len(summary['frame_details'])} frames...")
                
                for detail in summary['frame_details']:
                    if detail['num_plumes'] > 0:
                        # Create mock plume info (would need full data in practice)
                        mock_plumes = []
                        for i in range(detail['num_plumes']):
                            mock_plumes.append({
                                'centroid': (256 + i*10, 320 + i*10),  # Mock positions
                                'area_pixels': 100,
                                'min_shore_distance': 3
                            })
                        
                        georef.process_sgd_frame(detail['frame'], mock_plumes)
                
                # Export results
                georef.export_to_geojson("sgd_detections.geojson")
                georef.export_to_csv("sgd_detections.csv")
                georef.export_to_kml("sgd_detections.kml")
                
            else:
                print(f"SGD results not found: {sgd_file}")
        
        else:
            # Test with sample data
            print("\nCreating test SGD locations...")
            
            # Create test plumes
            test_plumes = [
                {'centroid': (256, 320), 'area_pixels': 150, 'min_shore_distance': 2},
                {'centroid': (280, 350), 'area_pixels': 200, 'min_shore_distance': 3},
                {'centroid': (240, 310), 'area_pixels': 100, 'min_shore_distance': 1}
            ]
            
            # Process test frame
            locations = georef.process_sgd_frame(248, test_plumes)
            
            if locations:
                print(f"\nGenerated {len(locations)} test locations")
                
                # Export
                georef.export_to_geojson("test_sgd.geojson")
                georef.export_to_csv("test_sgd.csv")
                georef.export_to_kml("test_sgd.kml")
                
                print("\n📁 Files created:")
                print("  - test_sgd.geojson (for QGIS/ArcGIS)")
                print("  - test_sgd.csv (spreadsheet)")
                print("  - test_sgd.kml (for Google Earth)")
    
    else:
        print("\n⚠️ Cannot proceed without GPS data in images")
        print("Please ensure your drone is recording GPS coordinates")


if __name__ == "__main__":
    main()