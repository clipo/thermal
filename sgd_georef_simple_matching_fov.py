#!/usr/bin/env python3
"""
Simplified georeferencing for SGD detection with MATCHING thermal/RGB FOV.

Based on the confirmed alignment from sgd_detector_integrated.py:
- Thermal and RGB cameras have the SAME field of view
- They see the same area, just at different resolutions
- Thermal: 640x512, RGB: 4096x3072
- Scale factor: 6.4x horizontally, 6.0x vertically
"""

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import numpy as np
from datetime import datetime

class SimpleGeorefMatchingFOV:
    """Georeferencing for thermal detections with matching FOV"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # IMPORTANT: Thermal and RGB have MATCHING field of view
        # Scale factors for pixel mapping
        self.scale_x = self.rgb_width / self.thermal_width  # 6.4
        self.scale_y = self.rgb_height / self.thermal_height  # 6.0
        
        print(f"Thermal and RGB have matching FOV")
        print(f"Scale: {self.scale_x:.1f}x{self.scale_y:.1f} (RGB pixels per thermal pixel)")
        
        # Store SGD locations
        self.sgd_locations = []
    
    def extract_gps_simple(self, image_path):
        """Extract GPS coordinates from image EXIF"""
        try:
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            if not exifdata:
                return None
            
            # Get GPS IFD
            gps_ifd = exifdata.get_ifd(0x8825)
            if not gps_ifd:
                return None
            
            # Extract GPS data
            gps_info = {}
            
            # Latitude
            if 2 in gps_ifd:  # GPSLatitude
                lat_data = gps_ifd[2]
                lat = lat_data[0] + lat_data[1]/60.0 + lat_data[2]/3600.0
                if 1 in gps_ifd and gps_ifd[1] == 'S':  # GPSLatitudeRef
                    lat = -lat
                gps_info['lat'] = lat
            
            # Longitude  
            if 4 in gps_ifd:  # GPSLongitude
                lon_data = gps_ifd[4]
                lon = lon_data[0] + lon_data[1]/60.0 + lon_data[2]/3600.0
                if 3 in gps_ifd and gps_ifd[3] == 'W':  # GPSLongitudeRef
                    lon = -lon
                gps_info['lon'] = lon
            
            # Altitude
            if 6 in gps_ifd:  # GPSAltitude
                gps_info['altitude'] = float(gps_ifd[6])
            
            # Heading
            if 17 in gps_ifd:  # GPSImgDirection
                gps_info['heading'] = float(gps_ifd[17])
            
            # DateTime
            datetime_str = exifdata.get(306) or exifdata.get(36867)
            if datetime_str:
                gps_info['datetime'] = datetime_str
            
            return gps_info if 'lat' in gps_info else None
            
        except Exception as e:
            print(f"Error extracting GPS: {e}")
            return None
    
    def thermal_pixel_to_latlon(self, thermal_x, thermal_y, 
                                rgb_center_lat, rgb_center_lon, 
                                altitude, heading=None):
        """
        Convert thermal pixel to lat/lon coordinates.
        
        Since thermal and RGB have matching FOV, we just scale the coordinates.
        """
        # Map thermal pixel to RGB pixel (simple scaling)
        rgb_x = thermal_x * self.scale_x
        rgb_y = thermal_y * self.scale_y
        
        # Calculate offset from center in RGB pixels
        rgb_center_x = self.rgb_width / 2
        rgb_center_y = self.rgb_height / 2
        
        offset_x_pixels = rgb_x - rgb_center_x
        offset_y_pixels = rgb_y - rgb_center_y
        
        # Calculate ground sample distance (meters per pixel)
        # Using standard drone camera FOV approximation
        # FOV for DJI-style camera: ~75° horizontal
        fov_horizontal_deg = 75
        fov_horizontal_rad = np.radians(fov_horizontal_deg)
        
        # Ground width covered by RGB image
        ground_width = 2 * altitude * np.tan(fov_horizontal_rad / 2)
        
        # Meters per RGB pixel
        gsd = ground_width / self.rgb_width
        
        # Convert pixel offset to meters
        offset_x_meters = offset_x_pixels * gsd
        offset_y_meters = -offset_y_pixels * gsd  # Negative for image Y axis
        
        # Apply rotation if heading provided
        if heading is not None and heading != 0:
            heading_rad = np.radians(heading)
            rotated_x = offset_x_meters * np.cos(heading_rad) - offset_y_meters * np.sin(heading_rad)
            rotated_y = offset_x_meters * np.sin(heading_rad) + offset_y_meters * np.cos(heading_rad)
            offset_x_meters = rotated_x
            offset_y_meters = rotated_y
        
        # Convert meters to degrees
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(rgb_center_lat))
        
        delta_lat = offset_y_meters / meters_per_degree_lat
        delta_lon = offset_x_meters / meters_per_degree_lon
        
        return rgb_center_lat + delta_lat, rgb_center_lon + delta_lon
    
    def process_sgd_frame(self, frame_number, plume_info_list):
        """Process detected SGD plumes and georeference them"""
        # Get GPS from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        gps_info = self.extract_gps_simple(str(rgb_path))
        
        if not gps_info or 'lat' not in gps_info:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        # Process each plume
        georeferenced = []
        for plume in plume_info_list:
            # Get plume centroid in thermal coordinates
            thermal_x = plume['centroid'][0]
            thermal_y = plume['centroid'][1]
            
            # Convert to lat/lon
            lat, lon = self.thermal_pixel_to_latlon(
                thermal_x, thermal_y,
                gps_info['lat'], gps_info['lon'],
                gps_info.get('altitude', 400),
                gps_info.get('heading')
            )
            
            # Create georeferenced record
            # Convert area from pixels to m²
            gsd = gps_info.get('altitude', 400) * np.tan(np.radians(75/2)) * 2 / self.rgb_width
            pixel_area_m2 = (gsd * self.scale_x) ** 2  # Area of one thermal pixel in m²
            area_m2 = plume['area_pixels'] * pixel_area_m2
            
            location = {
                'frame': frame_number,
                'datetime': gps_info.get('datetime', ''),
                'latitude': lat,
                'longitude': lon,
                'area_m2': area_m2,
                'shore_distance': plume['min_shore_distance'],
                'altitude': gps_info.get('altitude', 400)
            }
            
            georeferenced.append(location)
            self.sgd_locations.append(location)
        
        return georeferenced
    
    def export_to_geojson(self, output_path):
        """Export SGD locations to GeoJSON"""
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
        
        print(f"Exported {len(features)} locations to {output_path}")
    
    def export_to_csv(self, output_path):
        """Export to CSV format"""
        with open(output_path, 'w') as f:
            f.write("frame,datetime,latitude,longitude,area_m2,shore_distance,altitude\n")
            for loc in self.sgd_locations:
                f.write(f"{loc['frame']},{loc['datetime']},{loc['latitude']},{loc['longitude']},"
                       f"{loc['area_m2']},{loc['shore_distance']},{loc['altitude']}\n")
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")

if __name__ == "__main__":
    # Test the corrected georeferencing
    print("Testing Georeferencing with Matching FOV")
    print("=" * 50)
    
    georef = SimpleGeorefMatchingFOV()
    
    # Test GPS extraction
    gps = georef.extract_gps_simple("data/100MEDIA/MAX_0248.JPG")
    if gps:
        print(f"GPS: {gps['lat']:.6f}, {gps['lon']:.6f}")
        print(f"Altitude: {gps.get('altitude', 0):.1f} m")
    
    # Test coordinate conversion
    print("\nTesting coordinate conversion:")
    print("Thermal pixel (320, 256) - center of thermal image")
    lat, lon = georef.thermal_pixel_to_latlon(
        320, 256,  # Center of thermal
        gps['lat'], gps['lon'], 
        gps.get('altitude', 400)
    )
    print(f"  -> GPS: {lat:.6f}, {lon:.6f}")
    print(f"  Should be close to RGB center: {gps['lat']:.6f}, {gps['lon']:.6f}")
    print(f"  Difference: {abs(lat-gps['lat'])*111320:.2f}m N-S, {abs(lon-gps['lon'])*111320*0.953:.2f}m E-W")