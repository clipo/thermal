#!/usr/bin/env python3
"""
Main georeferencing module for SGD detection.

This module properly accounts for the thermal camera having a narrower
field of view than the RGB camera (thermal sees ~70% of RGB FOV).
"""

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import numpy as np
from datetime import datetime

class SGDGeoref:
    """Main georeferencing class for SGD detection"""
    
    def __init__(self, base_path="data/100MEDIA", thermal_fov_ratio=0.7):
        """
        Initialize georeferencing with proper FOV handling.
        
        Args:
            base_path: Path to data directory
            thermal_fov_ratio: Fraction of RGB FOV that thermal sees (default 0.7)
        """
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Thermal FOV is a subset of RGB FOV
        # Thermal typically sees ~70% of RGB field of view
        self.thermal_fov_ratio = thermal_fov_ratio
        
        # Calculate how many RGB pixels the thermal FOV covers
        self.thermal_coverage_x = self.rgb_width * thermal_fov_ratio
        self.thermal_coverage_y = self.rgb_height * thermal_fov_ratio
        
        # Scale factors: RGB pixels per thermal pixel within thermal's FOV
        self.scale_x = self.thermal_coverage_x / self.thermal_width
        self.scale_y = self.thermal_coverage_y / self.thermal_height
        
        # Offset to center thermal FOV in RGB image
        self.offset_x = (self.rgb_width - self.thermal_coverage_x) / 2
        self.offset_y = (self.rgb_height - self.thermal_coverage_y) / 2
        
        print(f"Georeferencing initialized:")
        print(f"  Thermal FOV: {thermal_fov_ratio*100:.0f}% of RGB")
        print(f"  Coverage: {self.thermal_coverage_x:.0f}x{self.thermal_coverage_y:.0f} RGB pixels")
        print(f"  Scale: {self.scale_x:.2f}x{self.scale_y:.2f}")
        
        # Store SGD locations
        self.sgd_locations = []
    
    def extract_gps(self, image_path):
        """Extract GPS coordinates from image EXIF data"""
        try:
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            if not exifdata:
                return None
            
            # Get GPS IFD
            gps_ifd = exifdata.get_ifd(0x8825)
            if not gps_ifd:
                return None
            
            gps_info = {}
            
            # Latitude
            if 2 in gps_ifd:  # GPSLatitude
                lat_data = gps_ifd[2]
                lat = lat_data[0] + lat_data[1]/60.0 + lat_data[2]/3600.0
                if 1 in gps_ifd and gps_ifd[1] == 'S':
                    lat = -lat
                gps_info['lat'] = lat
            
            # Longitude  
            if 4 in gps_ifd:  # GPSLongitude
                lon_data = gps_ifd[4]
                lon = lon_data[0] + lon_data[1]/60.0 + lon_data[2]/3600.0
                if 3 in gps_ifd and gps_ifd[3] == 'W':
                    lon = -lon
                gps_info['lon'] = lon
            
            # Altitude
            if 6 in gps_ifd:
                gps_info['altitude'] = float(gps_ifd[6])
            
            # Heading
            if 17 in gps_ifd:
                gps_info['heading'] = float(gps_ifd[17])
            
            # DateTime
            datetime_str = exifdata.get(306) or exifdata.get(36867)
            if datetime_str:
                gps_info['datetime'] = datetime_str
            
            return gps_info if 'lat' in gps_info else None
            
        except Exception as e:
            print(f"Error extracting GPS: {e}")
            return None
    
    def thermal_to_latlon(self, thermal_x, thermal_y, 
                          rgb_center_lat, rgb_center_lon, 
                          altitude, heading=None):
        """
        Convert thermal pixel coordinates to lat/lon.
        
        Accounts for thermal FOV being a subset of RGB FOV.
        """
        # Map thermal pixel to RGB coordinates
        rgb_x = self.offset_x + thermal_x * self.scale_x
        rgb_y = self.offset_y + thermal_y * self.scale_y
        
        # Calculate offset from RGB center
        rgb_center_x = self.rgb_width / 2
        rgb_center_y = self.rgb_height / 2
        
        offset_x_pixels = rgb_x - rgb_center_x
        offset_y_pixels = rgb_y - rgb_center_y
        
        # Ground sample distance for RGB camera
        # Typical RGB drone camera: ~80° FOV
        rgb_fov_deg = 80
        rgb_fov_rad = np.radians(rgb_fov_deg)
        
        # Ground width covered by RGB
        ground_width = 2 * altitude * np.tan(rgb_fov_rad / 2)
        
        # Meters per RGB pixel
        gsd = ground_width / self.rgb_width
        
        # Convert to meters
        offset_x_meters = offset_x_pixels * gsd
        offset_y_meters = -offset_y_pixels * gsd
        
        # Apply heading rotation if provided
        if heading is not None and heading != 0:
            heading_rad = np.radians(heading)
            cos_h = np.cos(heading_rad)
            sin_h = np.sin(heading_rad)
            rotated_x = offset_x_meters * cos_h - offset_y_meters * sin_h
            rotated_y = offset_x_meters * sin_h + offset_y_meters * cos_h
            offset_x_meters = rotated_x
            offset_y_meters = rotated_y
        
        # Convert to degrees
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(rgb_center_lat))
        
        delta_lat = offset_y_meters / meters_per_degree_lat
        delta_lon = offset_x_meters / meters_per_degree_lon
        
        return rgb_center_lat + delta_lat, rgb_center_lon + delta_lon
    
    def process_frame(self, frame_number, plume_info_list):
        """
        Georeference detected SGD plumes from a frame.
        
        Args:
            frame_number: Frame number
            plume_info_list: List of plume dictionaries from detector
        
        Returns:
            List of georeferenced location dictionaries
        """
        # Get GPS from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        gps_info = self.extract_gps(str(rgb_path))
        
        if not gps_info or 'lat' not in gps_info:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        georeferenced = []
        altitude = gps_info.get('altitude', 400)
        
        # Calculate area per thermal pixel in m²
        thermal_fov_deg = 45  # Typical thermal camera FOV
        thermal_fov_rad = np.radians(thermal_fov_deg)
        ground_width_thermal = 2 * altitude * np.tan(thermal_fov_rad / 2)
        gsd_thermal = ground_width_thermal / self.thermal_width
        pixel_area_m2 = gsd_thermal ** 2
        
        for plume in plume_info_list:
            # Get centroid in thermal coordinates
            thermal_x = plume['centroid'][0]
            thermal_y = plume['centroid'][1]
            
            # Convert to lat/lon
            lat, lon = self.thermal_to_latlon(
                thermal_x, thermal_y,
                gps_info['lat'], gps_info['lon'],
                altitude,
                gps_info.get('heading')
            )
            
            # Calculate area in m²
            area_m2 = plume['area_pixels'] * pixel_area_m2
            
            # Create georeferenced record
            location = {
                'frame': frame_number,
                'datetime': gps_info.get('datetime', ''),
                'latitude': lat,
                'longitude': lon,
                'area_m2': area_m2,
                'temperature_anomaly': plume.get('mean_temp_diff', 0),
                'shore_distance': plume['min_shore_distance'],
                'altitude': altitude
            }
            
            georeferenced.append(location)
            self.sgd_locations.append(location)
        
        return georeferenced
    
    def export_geojson(self, output_path="sgd_locations.geojson"):
        """Export all SGD locations to GeoJSON format"""
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
                    "temperature_anomaly": loc.get('temperature_anomaly', 0),
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
        
        print(f"Exported {len(features)} SGD locations to {output_path}")
        return output_path
    
    def export_csv(self, output_path="sgd_locations.csv"):
        """Export to CSV format for analysis"""
        with open(output_path, 'w') as f:
            f.write("frame,datetime,latitude,longitude,area_m2,temperature_anomaly,shore_distance,altitude\n")
            for loc in self.sgd_locations:
                f.write(f"{loc['frame']},{loc['datetime']},{loc['latitude']},{loc['longitude']},"
                       f"{loc['area_m2']},{loc.get('temperature_anomaly', 0)},"
                       f"{loc['shore_distance']},{loc['altitude']}\n")
        
        print(f"Exported {len(self.sgd_locations)} SGD locations to {output_path}")
        return output_path
    
    def export_kml(self, output_path="sgd_locations.kml"):
        """Export to KML format for Google Earth"""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append('<name>SGD Locations</name>')
        
        # Add style for placemarks
        kml.append('<Style id="sgdStyle">')
        kml.append('  <IconStyle>')
        kml.append('    <color>ff0000ff</color>')  # Red color
        kml.append('    <scale>1.0</scale>')
        kml.append('  </IconStyle>')
        kml.append('</Style>')
        
        for i, loc in enumerate(self.sgd_locations):
            kml.append('<Placemark>')
            kml.append(f'  <name>SGD {i+1} (Frame {loc["frame"]})</name>')
            kml.append(f'  <description>')
            kml.append(f'    Area: {loc["area_m2"]:.1f} m²\n')
            kml.append(f'    Temperature anomaly: {loc.get("temperature_anomaly", 0):.1f}°C\n')
            kml.append(f'    Shore distance: {loc["shore_distance"]:.1f} pixels\n')
            kml.append(f'    Altitude: {loc["altitude"]:.1f} m')
            kml.append(f'  </description>')
            kml.append('  <styleUrl>#sgdStyle</styleUrl>')
            kml.append('  <Point>')
            kml.append(f'    <coordinates>{loc["longitude"]},{loc["latitude"]},0</coordinates>')
            kml.append('  </Point>')
            kml.append('</Placemark>')
        
        kml.append('</Document>')
        kml.append('</kml>')
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(kml))
        
        print(f"Exported {len(self.sgd_locations)} SGD locations to {output_path}")
        return output_path

if __name__ == "__main__":
    # Test the georeferencing
    print("SGD Georeferencing Module")
    print("=" * 50)
    
    georef = SGDGeoref(thermal_fov_ratio=0.7)
    
    # Test GPS extraction
    gps = georef.extract_gps("data/100MEDIA/MAX_0248.JPG")
    if gps:
        print(f"\nGPS extracted: {gps['lat']:.6f}, {gps['lon']:.6f}")
        print(f"Altitude: {gps.get('altitude', 0):.1f} m")
    
    print("\nReady to process SGD detections!")