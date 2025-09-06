#!/usr/bin/env python3
"""
Corrected SGD Georeferencer - Properly handles thermal FOV subset
Accounts for the fact that thermal image is a centered subset of RGB image
"""

import numpy as np
import json
import csv
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime


class CorrectedGeoref:
    """Corrected georeferencing that properly handles thermal FOV subset"""
    
    def __init__(self, base_path="data/100MEDIA"):
        self.base_path = Path(base_path)
        
        # Image dimensions
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # CRITICAL: Thermal FOV is a subset of RGB FOV
        # The thermal camera sees approximately 1/6.4 of the RGB width
        # and 1/6.0 of the RGB height
        self.thermal_fov_ratio_x = self.thermal_width / self.rgb_width  # ~0.156
        self.thermal_fov_ratio_y = self.thermal_height / self.rgb_height  # ~0.167
        
        # The thermal image is centered in the RGB image
        # So the thermal image starts at these RGB pixel coordinates:
        self.thermal_start_x = (self.rgb_width - self.thermal_width * 6.4) / 2
        self.thermal_start_y = (self.rgb_height - self.thermal_height * 6.0) / 2
        
        # For centered thermal FOV:
        self.thermal_center_offset_x = 0  # Centered horizontally
        self.thermal_center_offset_y = 0  # Centered vertically
        
        print(f"Thermal FOV covers {self.thermal_fov_ratio_x*100:.1f}% of RGB width")
        print(f"Thermal FOV covers {self.thermal_fov_ratio_y*100:.1f}% of RGB height")
        
        # Store unique SGD locations
        self.sgd_locations = []
        self.location_hashes = set()
    
    def extract_gps_simple(self, image_path):
        """Extract GPS coordinates from EXIF"""
        try:
            img = Image.open(image_path)
            exifdata = img.getexif()
            
            if not exifdata:
                return None
            
            info = {
                'datetime': None,
                'lat': None,
                'lon': None,
                'altitude': None,
                'heading': None
            }
            
            # Get datetime
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTime':
                    info['datetime'] = value
            
            # Get GPS IFD
            try:
                gps_ifd = exifdata.get_ifd(0x8825)
                
                if gps_ifd:
                    # GPS coordinates
                    lat_data = gps_ifd.get(2)  # GPSLatitude
                    lat_ref = gps_ifd.get(1)   # GPSLatitudeRef
                    lon_data = gps_ifd.get(4)  # GPSLongitude
                    lon_ref = gps_ifd.get(3)   # GPSLongitudeRef
                    alt_data = gps_ifd.get(6)  # GPSAltitude
                    heading_data = gps_ifd.get(17)  # GPSImgDirection
                    
                    if lat_data and lon_data:
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
                        
                        if heading_data:
                            info['heading'] = float(heading_data)
            except:
                pass
            
            return info
            
        except Exception as e:
            print(f"Error extracting GPS: {e}")
            return None
    
    def estimate_gsd(self, altitude, sensor_width_mm=13.2, focal_length_mm=10.0):
        """
        Estimate Ground Sample Distance (meters per pixel)
        
        For the RGB camera (full frame):
        - sensor_width_mm: Physical sensor width (typical 1" sensor = 13.2mm)
        - focal_length_mm: Lens focal length (typical wide angle = 10mm)
        - altitude: Flight altitude in meters
        
        Returns GSD for RGB image pixels
        """
        # GSD for RGB image
        gsd_rgb_x = (altitude * sensor_width_mm) / (focal_length_mm * self.rgb_width)
        gsd_rgb_y = (altitude * sensor_width_mm * 0.75) / (focal_length_mm * self.rgb_height)
        
        return gsd_rgb_x, gsd_rgb_y
    
    def thermal_pixel_to_latlon(self, thermal_x, thermal_y, rgb_center_lat, rgb_center_lon, 
                               altitude=100, heading=0):
        """
        Convert thermal pixel coordinates to lat/lon
        
        CRITICAL: The thermal image is a subset of the RGB image
        We need to:
        1. Understand where the thermal pixel is within the RGB frame
        2. Calculate its offset from the RGB center
        3. Convert to geographic coordinates
        
        Parameters:
        - thermal_x, thermal_y: Pixel coordinates in thermal image (0-640, 0-512)
        - rgb_center_lat, rgb_center_lon: GPS coords of RGB image center
        - altitude: Flight altitude in meters
        - heading: Camera heading in degrees
        """
        
        # Get ground sample distance for RGB image
        gsd_x, gsd_y = self.estimate_gsd(altitude)
        
        # CRITICAL CALCULATION:
        # The thermal FOV is only a portion of the RGB FOV
        # We need to find where this thermal pixel maps in the RGB image
        
        # Since thermal is centered in RGB, the mapping is:
        # RGB pixel = thermal pixel * scale + offset
        # where scale = RGB_size / (thermal_size * zoom_factor)
        
        # For centered thermal with 6.4x zoom:
        # The thermal image covers the center portion of RGB
        # From RGB pixel ~728 to ~3368 horizontally (640*6.4 = 4096 pixels worth)
        # From RGB pixel ~0 to ~3072 vertically (512*6.0 = 3072 pixels worth)
        
        # Calculate which RGB pixel this thermal pixel corresponds to
        # Thermal is effectively "zoomed in" on the center
        rgb_pixel_x = (self.rgb_width / 2) + (thermal_x - self.thermal_width / 2) * self.thermal_fov_ratio_x * self.rgb_width / self.thermal_width
        rgb_pixel_y = (self.rgb_height / 2) + (thermal_y - self.thermal_height / 2) * self.thermal_fov_ratio_y * self.rgb_height / self.thermal_height
        
        # Simplified: Since thermal is centered and covers a fraction of RGB
        # The thermal pixel position needs to be scaled to RGB coordinates
        # thermal pixel 0,0 -> RGB pixel at start of thermal FOV
        # thermal pixel 640,512 -> RGB pixel at end of thermal FOV
        
        # More accurate calculation:
        # Thermal covers the center 15.6% horizontally and 16.7% vertically
        thermal_fov_start_x = self.rgb_width * (1 - self.thermal_fov_ratio_x) / 2
        thermal_fov_start_y = self.rgb_height * (1 - self.thermal_fov_ratio_y) / 2
        
        # Map thermal pixel to RGB pixel
        rgb_pixel_x = thermal_fov_start_x + (thermal_x / self.thermal_width) * (self.rgb_width * self.thermal_fov_ratio_x)
        rgb_pixel_y = thermal_fov_start_y + (thermal_y / self.thermal_height) * (self.rgb_height * self.thermal_fov_ratio_y)
        
        # Now calculate offset from RGB center in pixels
        offset_x_pixels = rgb_pixel_x - (self.rgb_width / 2)
        offset_y_pixels = rgb_pixel_y - (self.rgb_height / 2)
        
        # Convert pixel offset to meters on ground
        offset_x_meters = offset_x_pixels * gsd_x
        offset_y_meters = -offset_y_pixels * gsd_y  # Negative because image Y is inverted
        
        # Apply rotation if drone is not pointing north
        if heading is not None and heading != 0:
            heading_rad = np.radians(heading)
            rotated_x = offset_x_meters * np.cos(heading_rad) - offset_y_meters * np.sin(heading_rad)
            rotated_y = offset_x_meters * np.sin(heading_rad) + offset_y_meters * np.cos(heading_rad)
            offset_x_meters = rotated_x
            offset_y_meters = rotated_y
        
        # Convert meters to degrees
        # At the equator: 1 degree latitude = 111,320 meters
        # Longitude varies with latitude
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(rgb_center_lat))
        
        delta_lat = offset_y_meters / meters_per_degree_lat
        delta_lon = offset_x_meters / meters_per_degree_lon
        
        # Final coordinates
        lat = rgb_center_lat + delta_lat
        lon = rgb_center_lon + delta_lon
        
        return lat, lon
    
    def process_sgd_frame(self, frame_number, sgd_plumes):
        """
        Process SGD detections for a frame with corrected georeferencing
        
        Parameters:
        - frame_number: Frame number
        - sgd_plumes: List of plume info from SGD detector (in thermal pixel coords)
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
        
        print(f"\nProcessing frame {frame_number}:")
        print(f"  RGB center GPS: {gps_info['lat']:.6f}, {gps_info['lon']:.6f}")
        print(f"  Altitude: {gps_info.get('altitude', 100):.1f}m")
        print(f"  Thermal FOV: {self.thermal_fov_ratio_x*100:.1f}% x {self.thermal_fov_ratio_y*100:.1f}% of RGB")
        
        for i, plume in enumerate(sgd_plumes):
            # Get plume centroid in thermal pixel coordinates
            centroid_y, centroid_x = plume['centroid']
            
            # Convert to lat/lon using corrected calculation
            lat, lon = self.thermal_pixel_to_latlon(
                centroid_x, centroid_y,
                gps_info['lat'], gps_info['lon'],
                gps_info.get('altitude', 100),
                gps_info.get('heading', 0)
            )
            
            # Debug info for first plume
            if i == 0:
                print(f"  Example plume at thermal pixel ({centroid_x:.0f}, {centroid_y:.0f})")
                print(f"    -> GPS: {lat:.6f}, {lon:.6f}")
            
            # Create location record
            location = {
                'frame': frame_number,
                'datetime': gps_info.get('datetime', ''),
                'latitude': lat,
                'longitude': lon,
                'thermal_x': centroid_x,
                'thermal_y': centroid_y,
                'area_pixels': plume['area_pixels'],
                'area_m2': plume.get('area_pixels', 0) * 0.01,  # Rough estimate
                'shore_distance': plume.get('min_shore_distance', 0),
                'rgb_center_lat': gps_info['lat'],
                'rgb_center_lon': gps_info['lon'],
                'altitude': gps_info.get('altitude', 100)
            }
            
            # Check for duplicates (within ~5 meters)
            location_hash = f"{lat:.5f},{lon:.5f}"
            
            if location_hash not in self.location_hashes:
                locations.append(location)
                self.location_hashes.add(location_hash)
                self.sgd_locations.append(location)
        
        return locations
    
    def export_to_csv(self, output_path="sgd_locations_corrected.csv"):
        """Export SGD locations to CSV with detailed info"""
        if not self.sgd_locations:
            print("No SGD locations to export")
            return
        
        with open(output_path, 'w', newline='') as f:
            fieldnames = ['frame', 'datetime', 'latitude', 'longitude', 
                         'thermal_x', 'thermal_y', 'area_m2', 'shore_distance', 
                         'altitude', 'rgb_center_lat', 'rgb_center_lon']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for loc in self.sgd_locations:
                writer.writerow({
                    'frame': loc['frame'],
                    'datetime': loc['datetime'],
                    'latitude': f"{loc['latitude']:.7f}",
                    'longitude': f"{loc['longitude']:.7f}",
                    'thermal_x': f"{loc['thermal_x']:.1f}",
                    'thermal_y': f"{loc['thermal_y']:.1f}",
                    'area_m2': f"{loc['area_m2']:.1f}",
                    'shore_distance': f"{loc['shore_distance']:.1f}",
                    'altitude': f"{loc['altitude']:.1f}",
                    'rgb_center_lat': f"{loc['rgb_center_lat']:.7f}",
                    'rgb_center_lon': f"{loc['rgb_center_lon']:.7f}"
                })
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")
    
    def export_to_geojson(self, output_path="sgd_locations_corrected.geojson"):
        """Export to GeoJSON with metadata"""
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
                    "thermal_pixel": f"({loc['thermal_x']:.0f}, {loc['thermal_y']:.0f})",
                    "area_m2": loc['area_m2'],
                    "shore_distance": loc['shore_distance'],
                    "altitude": loc['altitude'],
                    "rgb_center": f"{loc['rgb_center_lat']:.6f}, {loc['rgb_center_lon']:.6f}"
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "EPSG:4326"}
            },
            "metadata": {
                "description": "SGD detections with corrected thermal FOV georeferencing",
                "thermal_fov": f"{self.thermal_fov_ratio_x*100:.1f}% x {self.thermal_fov_ratio_y*100:.1f}% of RGB",
                "thermal_resolution": f"{self.thermal_width}x{self.thermal_height}",
                "rgb_resolution": f"{self.rgb_width}x{self.rgb_height}"
            },
            "features": features
        }
        
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Exported {len(self.sgd_locations)} locations to {output_path}")
        print(f"Thermal FOV: {self.thermal_fov_ratio_x*100:.1f}% x {self.thermal_fov_ratio_y*100:.1f}% of RGB coverage")


def visualize_fov_coverage():
    """Visualize how thermal FOV maps to RGB FOV"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # RGB image boundary
    rgb_rect = patches.Rectangle((0, 0), 4096, 3072, 
                                linewidth=2, edgecolor='blue', 
                                facecolor='lightblue', alpha=0.3)
    ax.add_patch(rgb_rect)
    
    # Thermal FOV (centered, covering ~15.6% x 16.7%)
    thermal_width_in_rgb = 4096 * 0.156  # ~640 pixels worth
    thermal_height_in_rgb = 3072 * 0.167  # ~512 pixels worth
    
    thermal_x = (4096 - thermal_width_in_rgb) / 2
    thermal_y = (3072 - thermal_height_in_rgb) / 2
    
    thermal_rect = patches.Rectangle((thermal_x, thermal_y), 
                                    thermal_width_in_rgb, thermal_height_in_rgb,
                                    linewidth=2, edgecolor='red', 
                                    facecolor='pink', alpha=0.5)
    ax.add_patch(thermal_rect)
    
    # Add labels
    ax.text(2048, 1536, 'RGB Image\n4096×3072', 
           ha='center', va='center', fontsize=12, color='blue')
    ax.text(2048, thermal_y + thermal_height_in_rgb/2, 
           f'Thermal FOV\n{thermal_width_in_rgb:.0f}×{thermal_height_in_rgb:.0f}\n(~15.6% × 16.7%)', 
           ha='center', va='center', fontsize=10, color='red')
    
    # Add center point
    ax.plot(2048, 1536, 'ko', markersize=8)
    ax.text(2048, 1400, 'GPS Location\n(Image Center)', 
           ha='center', va='top', fontsize=9)
    
    # Set limits and labels
    ax.set_xlim(-100, 4200)
    ax.set_ylim(-100, 3200)
    ax.set_xlabel('Pixels (horizontal)')
    ax.set_ylabel('Pixels (vertical)')
    ax.set_title('Thermal FOV Coverage within RGB Image\n(Thermal is zoomed in on center ~15% of RGB view)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('thermal_fov_coverage.png', dpi=150)
    plt.show()
    
    print("\nFOV Coverage Analysis:")
    print(f"RGB Image: 4096 × 3072 pixels (full wide-angle view)")
    print(f"Thermal effective coverage: {thermal_width_in_rgb:.0f} × {thermal_height_in_rgb:.0f} pixels")
    print(f"Thermal sees only the center {0.156*100:.1f}% × {0.167*100:.1f}% of RGB view")
    print(f"This means thermal has ~6.4× zoom compared to RGB")


def test_corrected_georeferencing():
    """Test the corrected georeferencing"""
    
    print("Testing Corrected Georeferencing")
    print("=" * 50)
    
    georef = CorrectedGeoref()
    
    # Test GPS extraction
    test_image = Path("data/100MEDIA/MAX_0248.JPG")
    if test_image.exists():
        gps_info = georef.extract_gps_simple(test_image)
        
        if gps_info and gps_info['lat']:
            print(f"\n✅ GPS Data:")
            print(f"  RGB center: {gps_info['lat']:.6f}, {gps_info['lon']:.6f}")
            print(f"  Altitude: {gps_info.get('altitude', 100):.1f}m")
            
            # Test coordinate conversion for different thermal pixels
            print(f"\n📍 Testing coordinate conversion:")
            
            test_points = [
                (320, 256, "Thermal center"),
                (0, 0, "Thermal top-left"),
                (639, 511, "Thermal bottom-right"),
                (100, 100, "Thermal offset point")
            ]
            
            for x, y, desc in test_points:
                lat, lon = georef.thermal_pixel_to_latlon(
                    x, y, gps_info['lat'], gps_info['lon'], 
                    gps_info.get('altitude', 100)
                )
                print(f"  {desc} ({x},{y}) -> {lat:.6f}, {lon:.6f}")
            
            # Visualize FOV coverage
            visualize_fov_coverage()
            
            return True
    
    return False


def main():
    """Main entry point"""
    print("Corrected SGD Georeferencer")
    print("=" * 50)
    print("\nThis version correctly handles the thermal FOV being a subset of RGB FOV")
    
    if test_corrected_georeferencing():
        print("\n✅ Corrected georeferencing is working!")
        
        # Test with sample SGD data
        georef = CorrectedGeoref()
        
        test_plumes = [
            {'centroid': (256, 320), 'area_pixels': 150, 'min_shore_distance': 2},
            {'centroid': (100, 100), 'area_pixels': 200, 'min_shore_distance': 3},
            {'centroid': (500, 400), 'area_pixels': 100, 'min_shore_distance': 1}
        ]
        
        locations = georef.process_sgd_frame(248, test_plumes)
        
        if locations:
            georef.export_to_geojson("sgd_corrected.geojson")
            georef.export_to_csv("sgd_corrected.csv")
            
            print("\n📁 Created corrected output files:")
            print("  - sgd_corrected.geojson")
            print("  - sgd_corrected.csv")


if __name__ == "__main__":
    main()