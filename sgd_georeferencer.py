#!/usr/bin/env python3
"""
SGD Georeferencer - Extract GPS coordinates and create georeferenced shapefiles
Maps detected SGD plumes to real-world coordinates using drone GPS data
"""

import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
import hashlib
from shapely.geometry import Polygon, Point, mapping
from shapely.ops import unary_union
import fiona
from fiona.crs import from_epsg
from pyproj import Transformer
from collections import defaultdict


class SGDGeoreferencer:
    """Georeference SGD detections using GPS EXIF data"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Alignment parameters (thermal FOV within RGB)
        self.scale_x = 6.4
        self.scale_y = 6.0
        self.offset_x = 0  # Centered
        self.offset_y = 0
        
        # Coordinate transformer (WGS84 to UTM will be set based on location)
        self.transformer = None
        self.utm_zone = None
        
        # Store processed SGD features to avoid duplicates
        self.sgd_features = []
        self.sgd_hashes = set()  # Track unique features
        
    def extract_gps_from_exif(self, image_path):
        """Extract GPS coordinates from image EXIF data"""
        try:
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            if not exifdata:
                return None
            
            # Get basic info
            info = {
                'datetime': None,
                'altitude': None,
                'lat': None,
                'lon': None,
                'heading': None
            }
            
            # Get datetime
            if 306 in exifdata:  # DateTime tag
                info['datetime'] = exifdata[306]
            
            # Get GPS IFD
            gps_ifd = exifdata.get_ifd(0x8825)  # GPS IFD pointer
            
            if gps_ifd:
                # Extract GPS data
                gps_data = {}
                for tag, value in gps_ifd.items():
                    decoded = GPSTAGS.get(tag, tag)
                    gps_data[decoded] = value
                
                # Parse latitude
                if 'GPSLatitude' in gps_data and 'GPSLatitudeRef' in gps_data:
                    lat = self.convert_to_degrees(gps_data['GPSLatitude'])
                    if gps_data['GPSLatitudeRef'] == 'S':
                        lat = -lat
                    info['lat'] = lat
                
                # Parse longitude
                if 'GPSLongitude' in gps_data and 'GPSLongitudeRef' in gps_data:
                    lon = self.convert_to_degrees(gps_data['GPSLongitude'])
                    if gps_data['GPSLongitudeRef'] == 'W':
                        lon = -lon
                    info['lon'] = lon
                
                # Parse altitude
                if 'GPSAltitude' in gps_data:
                    info['altitude'] = float(gps_data['GPSAltitude'])
                
                # Parse heading/bearing
                if 'GPSImgDirection' in gps_data:
                    info['heading'] = float(gps_data['GPSImgDirection'])
                elif 'GPSTrack' in gps_data:
                    info['heading'] = float(gps_data['GPSTrack'])
            
            return info
            
        except Exception as e:
            print(f"Error extracting GPS from {image_path}: {e}")
            return None
    
    def convert_to_degrees(self, value):
        """Convert GPS coordinates to degrees"""
        # GPS coordinates are in (degrees, minutes, seconds) format
        d = float(value[0])
        m = float(value[1])
        s = float(value[2]) if len(value) > 2 else 0.0
        
        return d + (m / 60.0) + (s / 3600.0)
    
    def estimate_gsd(self, altitude, focal_length=10.0):
        """
        Estimate Ground Sample Distance (GSD) - meters per pixel
        
        Parameters:
        - altitude: Flight altitude in meters
        - focal_length: Camera focal length in mm (estimate)
        
        Returns:
        - gsd_x, gsd_y: Meters per pixel in x and y directions
        """
        # Typical sensor size for consumer drones (estimate)
        sensor_width_mm = 13.2  # Typical 1" sensor
        sensor_height_mm = 8.8
        
        # Calculate GSD
        gsd_x = (altitude * sensor_width_mm) / (focal_length * self.rgb_width)
        gsd_y = (altitude * sensor_height_mm) / (focal_length * self.rgb_height)
        
        return gsd_x, gsd_y
    
    def pixel_to_geographic(self, pixel_x, pixel_y, center_lat, center_lon, 
                           altitude, heading=0):
        """
        Convert pixel coordinates to geographic coordinates
        
        Parameters:
        - pixel_x, pixel_y: Pixel coordinates in thermal image
        - center_lat, center_lon: GPS coordinates of image center
        - altitude: Flight altitude in meters
        - heading: Camera heading in degrees
        
        Returns:
        - lat, lon: Geographic coordinates
        """
        # Estimate GSD (ground sample distance)
        gsd_x, gsd_y = self.estimate_gsd(altitude)
        
        # Calculate offset from center in thermal pixels
        # First convert thermal pixel to RGB pixel space
        rgb_pixel_x = pixel_x * self.scale_x + self.offset_x
        rgb_pixel_y = pixel_y * self.scale_y + self.offset_y
        
        # Offset from RGB image center
        offset_x_pixels = rgb_pixel_x - (self.rgb_width / 2)
        offset_y_pixels = rgb_pixel_y - (self.rgb_height / 2)
        
        # Convert to meters
        offset_x_meters = offset_x_pixels * gsd_x
        offset_y_meters = -offset_y_pixels * gsd_y  # Negative because image y is inverted
        
        # Apply rotation based on heading
        heading_rad = np.radians(heading)
        rotated_x = offset_x_meters * np.cos(heading_rad) - offset_y_meters * np.sin(heading_rad)
        rotated_y = offset_x_meters * np.sin(heading_rad) + offset_y_meters * np.cos(heading_rad)
        
        # Convert offset to lat/lon
        # Approximate conversion (accurate for small distances)
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
        
        delta_lat = rotated_y / meters_per_degree_lat
        delta_lon = rotated_x / meters_per_degree_lon
        
        lat = center_lat + delta_lat
        lon = center_lon + delta_lon
        
        return lat, lon
    
    def sgd_plume_to_polygon(self, plume_mask, center_lat, center_lon, 
                           altitude, heading=0):
        """
        Convert SGD plume mask to geographic polygon
        
        Parameters:
        - plume_mask: Boolean mask of SGD plume in thermal image
        - center_lat, center_lon: GPS coordinates of image center
        - altitude: Flight altitude
        - heading: Camera heading
        
        Returns:
        - Shapely Polygon object with geographic coordinates
        """
        # Find plume boundary
        from skimage import measure
        contours = measure.find_contours(plume_mask.astype(float), 0.5)
        
        if not contours:
            return None
        
        # Use largest contour
        contour = max(contours, key=len)
        
        # Simplify contour (reduce points)
        # Take every nth point to reduce complexity
        step = max(1, len(contour) // 20)  # Max 20 points
        contour = contour[::step]
        
        # Convert pixel coordinates to geographic
        coords = []
        for point in contour:
            y, x = point  # Contour returns (row, col)
            lat, lon = self.pixel_to_geographic(
                x, y, center_lat, center_lon, altitude, heading
            )
            coords.append((lon, lat))  # Shapely uses (lon, lat) order
        
        # Close polygon
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        
        # Create polygon
        if len(coords) >= 4:  # Minimum for valid polygon
            return Polygon(coords)
        
        return None
    
    def process_sgd_detection(self, sgd_result, frame_number):
        """
        Process SGD detection result and create georeferenced features
        
        Parameters:
        - sgd_result: Result from SGD detector
        - frame_number: Frame number for GPS lookup
        
        Returns:
        - List of georeferenced features
        """
        # Get GPS data from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        if not rgb_path.exists():
            print(f"RGB image not found: {rgb_path}")
            return []
        
        gps_info = self.extract_gps_from_exif(rgb_path)
        
        if not gps_info or gps_info['lat'] is None or gps_info['lon'] is None:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        # Default altitude if not in EXIF (typical drone height)
        altitude = gps_info.get('altitude', 100.0)  # Default 100m
        heading = gps_info.get('heading', 0.0)
        
        features = []
        
        # Process each plume
        for plume_info in sgd_result.get('plume_info', []):
            # Create plume mask for this specific plume
            plume_id = plume_info['id']
            
            # Get bounding box
            min_row, min_col, max_row, max_col = plume_info['bbox']
            
            # Extract plume region from full mask
            if 'sgd_mask' in sgd_result:
                # Create individual plume mask
                from skimage import measure
                labeled = measure.label(sgd_result['sgd_mask'])
                plume_mask = (labeled == plume_id)
                
                # Convert to polygon
                polygon = self.sgd_plume_to_polygon(
                    plume_mask, 
                    gps_info['lat'], 
                    gps_info['lon'],
                    altitude, 
                    heading
                )
                
                if polygon:
                    # Create feature
                    feature = {
                        'geometry': polygon,
                        'properties': {
                            'frame': frame_number,
                            'datetime': gps_info.get('datetime', ''),
                            'plume_id': plume_id,
                            'area_pixels': plume_info['area_pixels'],
                            'area_m2': plume_info.get('area_m2', plume_info['area_pixels'] * 0.01),
                            'confidence': plume_info.get('confidence', 0.5),
                            'temp_anomaly': sgd_result.get('characteristics', {}).get('temp_anomaly', 0),
                            'center_lat': gps_info['lat'],
                            'center_lon': gps_info['lon'],
                            'altitude': altitude,
                            'centroid_x': plume_info['centroid'][1],
                            'centroid_y': plume_info['centroid'][0]
                        }
                    }
                    
                    # Calculate hash for duplicate detection
                    # Based on approximate location (rounded to ~1m precision)
                    centroid = polygon.centroid
                    location_hash = hashlib.md5(
                        f"{centroid.x:.6f},{centroid.y:.6f}".encode()
                    ).hexdigest()
                    
                    feature['properties']['location_hash'] = location_hash
                    features.append(feature)
        
        return features
    
    def aggregate_sgd_features(self, new_features, distance_threshold=5.0):
        """
        Aggregate SGD features, avoiding duplicates
        
        Parameters:
        - new_features: New features to add
        - distance_threshold: Minimum distance (meters) to consider as separate SGD
        
        Returns:
        - Updated feature list without duplicates
        """
        for new_feature in new_features:
            # Check if this location already exists
            new_hash = new_feature['properties']['location_hash']
            
            if new_hash not in self.sgd_hashes:
                # Check spatial proximity to existing features
                new_geom = new_feature['geometry']
                is_duplicate = False
                
                for existing_feature in self.sgd_features:
                    existing_geom = existing_feature['geometry']
                    
                    # Check distance between centroids
                    distance = new_geom.centroid.distance(existing_geom.centroid)
                    
                    # Convert to meters (approximate)
                    lat = new_feature['properties']['center_lat']
                    meters_per_degree = 111320.0 * np.cos(np.radians(lat))
                    distance_meters = distance * meters_per_degree
                    
                    if distance_meters < distance_threshold:
                        # Merge with existing feature (update properties)
                        is_duplicate = True
                        
                        # Update existing feature with better confidence or larger area
                        if new_feature['properties']['area_m2'] > existing_feature['properties']['area_m2']:
                            existing_feature['geometry'] = new_geom
                            existing_feature['properties'].update({
                                'frame': new_feature['properties']['frame'],
                                'area_m2': new_feature['properties']['area_m2'],
                                'updated': datetime.now().isoformat()
                            })
                        break
                
                if not is_duplicate:
                    # Add as new feature
                    self.sgd_features.append(new_feature)
                    self.sgd_hashes.add(new_hash)
        
        return self.sgd_features
    
    def export_to_shapefile(self, output_path="sgd_detections.shp"):
        """
        Export SGD detections to shapefile
        
        Parameters:
        - output_path: Output shapefile path
        """
        if not self.sgd_features:
            print("No SGD features to export")
            return
        
        # Define schema
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'frame': 'int',
                'datetime': 'str',
                'area_m2': 'float',
                'temp_anom': 'float',
                'confidence': 'float',
                'lat': 'float',
                'lon': 'float',
                'altitude': 'float'
            }
        }
        
        # Write shapefile
        with fiona.open(
            output_path, 
            'w',
            driver='ESRI Shapefile',
            crs=from_epsg(4326),  # WGS84
            schema=schema
        ) as shp:
            for feature in self.sgd_features:
                # Prepare record
                record = {
                    'geometry': mapping(feature['geometry']),
                    'properties': {
                        'frame': feature['properties']['frame'],
                        'datetime': feature['properties'].get('datetime', '')[:19],  # Truncate for shapefile
                        'area_m2': feature['properties']['area_m2'],
                        'temp_anom': feature['properties'].get('temp_anomaly', 0),
                        'confidence': feature['properties'].get('confidence', 0.5),
                        'lat': feature['properties']['center_lat'],
                        'lon': feature['properties']['center_lon'],
                        'altitude': feature['properties']['altitude']
                    }
                }
                shp.write(record)
        
        print(f"Exported {len(self.sgd_features)} SGD features to {output_path}")
        
        # Also export as GeoJSON for better compatibility
        geojson_path = output_path.replace('.shp', '.geojson')
        self.export_to_geojson(geojson_path)
    
    def export_to_geojson(self, output_path="sgd_detections.geojson"):
        """Export to GeoJSON format"""
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "EPSG:4326"
                }
            },
            "features": []
        }
        
        for feature in self.sgd_features:
            geojson_feature = {
                "type": "Feature",
                "geometry": mapping(feature['geometry']),
                "properties": feature['properties']
            }
            geojson["features"].append(geojson_feature)
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(self.sgd_features)} SGD features to {output_path}")
    
    def process_batch(self, sgd_results_dir="sgd_output", output_dir="gis_output"):
        """
        Process batch of SGD detection results
        
        Parameters:
        - sgd_results_dir: Directory with SGD detection JSON files
        - output_dir: Output directory for shapefiles
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_path = Path(sgd_results_dir)
        
        # Load SGD detection results
        if (results_path / "sgd_summary.json").exists():
            with open(results_path / "sgd_summary.json", 'r') as f:
                summary = json.load(f)
            
            print(f"Processing {len(summary['frame_details'])} frames...")
            
            for frame_detail in summary['frame_details']:
                if frame_detail['num_plumes'] > 0:
                    # Reconstruct minimal result for georeferencing
                    sgd_result = {
                        'frame_number': frame_detail['frame'],
                        'plume_info': [],  # Would need full detection data
                        'characteristics': frame_detail.get('characteristics', {})
                    }
                    
                    # Process frame
                    features = self.process_sgd_detection(
                        sgd_result, 
                        frame_detail['frame']
                    )
                    
                    # Aggregate features
                    self.aggregate_sgd_features(features)
        
        # Export to shapefile
        shapefile_path = output_path / "sgd_detections.shp"
        self.export_to_shapefile(str(shapefile_path))
        
        # Create summary report
        self.create_summary_report(output_path / "sgd_summary.txt")
    
    def create_summary_report(self, output_path):
        """Create text summary of georeferenced SGD detections"""
        with open(output_path, 'w') as f:
            f.write("SGD Detection Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total unique SGD locations: {len(self.sgd_features)}\n")
            
            if self.sgd_features:
                total_area = sum(f['properties']['area_m2'] for f in self.sgd_features)
                avg_anomaly = np.mean([f['properties'].get('temp_anomaly', 0) 
                                      for f in self.sgd_features])
                
                f.write(f"Total SGD area: {total_area:.1f} m²\n")
                f.write(f"Average temperature anomaly: {avg_anomaly:.1f}°C\n\n")
                
                f.write("Individual SGD Locations:\n")
                f.write("-" * 30 + "\n")
                
                for i, feature in enumerate(self.sgd_features, 1):
                    props = feature['properties']
                    centroid = feature['geometry'].centroid
                    
                    f.write(f"\nSGD #{i}:\n")
                    f.write(f"  Frame: {props['frame']}\n")
                    f.write(f"  Location: {centroid.y:.6f}°N, {centroid.x:.6f}°E\n")
                    f.write(f"  Area: {props['area_m2']:.1f} m²\n")
                    f.write(f"  Temperature anomaly: {props.get('temp_anomaly', 0):.1f}°C\n")
                    f.write(f"  Confidence: {props.get('confidence', 0.5):.2f}\n")
        
        print(f"Summary report saved to {output_path}")


def test_gps_extraction():
    """Test GPS extraction from sample image"""
    georef = SGDGeoreferencer()
    
    # Test on frame 248
    test_image = Path("data/100MEDIA/MAX_0248.JPG")
    if test_image.exists():
        gps_info = georef.extract_gps_from_exif(test_image)
        
        if gps_info:
            print("GPS Information extracted:")
            print(f"  Latitude: {gps_info.get('lat', 'N/A')}")
            print(f"  Longitude: {gps_info.get('lon', 'N/A')}")
            print(f"  Altitude: {gps_info.get('altitude', 'N/A')} m")
            print(f"  Heading: {gps_info.get('heading', 'N/A')}°")
            print(f"  DateTime: {gps_info.get('datetime', 'N/A')}")
            return gps_info
    
    return None


def main():
    """Main entry point for georeferencing"""
    print("SGD Georeferencer")
    print("=" * 50)
    
    # Test GPS extraction first
    print("\nTesting GPS extraction...")
    gps_info = test_gps_extraction()
    
    if not gps_info or gps_info.get('lat') is None:
        print("\n⚠️ Warning: No GPS data found in images!")
        print("Georeferencing requires GPS coordinates in EXIF data.")
        print("Please ensure your drone is recording GPS data.")
        return
    
    print("\n✅ GPS data found! Can proceed with georeferencing.")
    
    # Create georeferencer
    georef = SGDGeoreferencer()
    
    print("\nOptions:")
    print("1. Process existing SGD detections")
    print("2. Test single frame georeferencing")
    
    choice = input("\nChoice (1-2): ").strip()
    
    if choice == '1':
        # Process batch
        sgd_dir = input("SGD results directory (default: sgd_output): ") or "sgd_output"
        output_dir = input("Output directory (default: gis_output): ") or "gis_output"
        
        georef.process_batch(sgd_dir, output_dir)
        
    else:
        # Test single frame
        frame = int(input("Enter frame number (default 248): ") or "248")
        
        # Mock SGD result for testing
        test_result = {
            'frame_number': frame,
            'plume_info': [
                {
                    'id': 1,
                    'area_pixels': 100,
                    'area_m2': 10.0,
                    'centroid': (256, 320),
                    'bbox': (250, 310, 262, 330),
                    'confidence': 0.8
                }
            ],
            'characteristics': {
                'temp_anomaly': -2.5
            },
            'sgd_mask': np.zeros((512, 640), dtype=bool)
        }
        
        # Create test mask
        test_result['sgd_mask'][250:262, 310:330] = True
        
        features = georef.process_sgd_detection(test_result, frame)
        
        if features:
            print(f"\nGeoreferenced {len(features)} SGD plumes")
            for feature in features:
                centroid = feature['geometry'].centroid
                print(f"  Location: {centroid.y:.6f}°N, {centroid.x:.6f}°E")
            
            # Export
            georef.sgd_features = features
            georef.export_to_shapefile("test_sgd.shp")


if __name__ == "__main__":
    main()