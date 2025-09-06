#!/usr/bin/env python3
"""
Georeferencing for SGD with thermal as subset of RGB FOV.

Based on typical drone camera specifications:
- RGB camera: ~80° FOV (wide angle)
- Thermal camera: ~45° FOV (narrower, centered)
- Thermal sees approximately 56% of what RGB sees
"""

from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import json
import numpy as np
from datetime import datetime

class ThermalSubsetGeoref:
    """Georeferencing with thermal FOV as subset of RGB FOV"""
    
    def __init__(self, base_path="data/100MEDIA", thermal_fov_ratio=0.56):
        """
        Initialize with FOV ratio.
        
        thermal_fov_ratio: fraction of RGB FOV that thermal sees (default 0.56)
        """
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Thermal FOV is a subset of RGB FOV
        # Default: thermal sees about 56% of RGB field of view
        self.thermal_fov_ratio = thermal_fov_ratio
        
        # Calculate how many RGB pixels the thermal FOV covers
        self.thermal_coverage_x = self.rgb_width * thermal_fov_ratio
        self.thermal_coverage_y = self.rgb_height * thermal_fov_ratio
        
        # Scale factors: how many RGB pixels per thermal pixel
        # within the thermal's actual FOV
        self.scale_x = self.thermal_coverage_x / self.thermal_width
        self.scale_y = self.thermal_coverage_y / self.thermal_height
        
        # Offset to center thermal FOV in RGB image
        self.offset_x = (self.rgb_width - self.thermal_coverage_x) / 2
        self.offset_y = (self.rgb_height - self.thermal_coverage_y) / 2
        
        print(f"Thermal FOV Configuration:")
        print(f"  Thermal sees {thermal_fov_ratio*100:.0f}% of RGB field of view")
        print(f"  Thermal covers {self.thermal_coverage_x:.0f}x{self.thermal_coverage_y:.0f} RGB pixels")
        print(f"  Centered at offset ({self.offset_x:.0f}, {self.offset_y:.0f})")
        print(f"  Scale: {self.scale_x:.2f}x{self.scale_y:.2f} RGB pixels per thermal pixel")
        
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
        Convert thermal pixel to lat/lon, accounting for thermal FOV subset.
        """
        # Map thermal pixel to its position in RGB image
        # Thermal FOV is centered in RGB, so:
        rgb_x = self.offset_x + thermal_x * self.scale_x
        rgb_y = self.offset_y + thermal_y * self.scale_y
        
        # Calculate offset from RGB center in pixels
        rgb_center_x = self.rgb_width / 2
        rgb_center_y = self.rgb_height / 2
        
        offset_x_pixels = rgb_x - rgb_center_x
        offset_y_pixels = rgb_y - rgb_center_y
        
        # Calculate ground sample distance for RGB camera
        # Using typical RGB camera FOV of ~80°
        rgb_fov_deg = 80
        rgb_fov_rad = np.radians(rgb_fov_deg)
        
        # Ground width covered by RGB image
        ground_width = 2 * altitude * np.tan(rgb_fov_rad / 2)
        
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
            
            # Calculate area in m²
            # GSD for thermal pixels (accounting for FOV subset)
            thermal_fov_deg = 45  # Typical thermal FOV
            thermal_fov_rad = np.radians(thermal_fov_deg)
            ground_width_thermal = 2 * gps_info.get('altitude', 400) * np.tan(thermal_fov_rad / 2)
            gsd_thermal = ground_width_thermal / self.thermal_width
            pixel_area_m2 = gsd_thermal ** 2
            area_m2 = plume['area_pixels'] * pixel_area_m2
            
            # Create georeferenced record
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
    # Test with different FOV ratios
    print("Testing Thermal Subset Georeferencing")
    print("=" * 50)
    
    # Try different FOV ratios to see which gives reasonable results
    for fov_ratio in [0.56, 0.7, 0.85]:
        print(f"\nTesting with FOV ratio: {fov_ratio}")
        print("-" * 30)
        
        georef = ThermalSubsetGeoref(thermal_fov_ratio=fov_ratio)
        
        # Test GPS extraction
        gps = georef.extract_gps_simple("data/100MEDIA/MAX_0248.JPG")
        if gps:
            # Test coordinate conversion for center pixel
            lat, lon = georef.thermal_pixel_to_latlon(
                320, 256,  # Center of thermal
                gps['lat'], gps['lon'], 
                gps.get('altitude', 400)
            )
            print(f"  Center pixel -> GPS: {lat:.6f}, {lon:.6f}")
            print(f"  RGB center GPS: {gps['lat']:.6f}, {gps['lon']:.6f}")
            
            # For correct FOV ratio, these should be very close
            diff_m = abs(lat - gps['lat']) * 111320
            print(f"  Difference: {diff_m:.2f} meters")