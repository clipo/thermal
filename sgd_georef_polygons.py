#!/usr/bin/env python3
"""
Enhanced SGD Georeferencing with Polygon Support
Exports plume outlines as georeferenced polygons with accurate area calculations
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from datetime import datetime
from skimage import measure


class SGDPolygonGeoref:
    """Enhanced georeferencing with polygon/outline support"""
    
    def __init__(self, base_path="data/100MEDIA"):
        """Initialize georeferencing system"""
        self.base_path = Path(base_path)
        
        # Camera parameters (Autel 640T)
        self.rgb_width = 4096
        self.rgb_height = 3072
        self.thermal_width = 640
        self.thermal_height = 512
        
        # Thermal FOV is ~70% of RGB FOV
        self.thermal_fov_ratio = 0.7
        
        # Calculate mapping parameters
        thermal_width_in_rgb = self.rgb_width * self.thermal_fov_ratio
        thermal_height_in_rgb = self.rgb_height * self.thermal_fov_ratio
        
        self.scale_x = thermal_width_in_rgb / self.thermal_width
        self.scale_y = thermal_height_in_rgb / self.thermal_height
        
        self.offset_x = (self.rgb_width - thermal_width_in_rgb) / 2
        self.offset_y = (self.rgb_height - thermal_height_in_rgb) / 2
        
        # Storage for georeferenced locations
        self.sgd_polygons = []
    
    def extract_gps(self, image_path):
        """Extract GPS metadata from image"""
        try:
            img = Image.open(image_path)
            exifdata = img._getexif()
            
            if not exifdata:
                return None
            
            gps_info = {}
            
            # Process GPS tags
            for tag, value in exifdata.items():
                tag_name = TAGS.get(tag, tag)
                
                if tag_name == 'GPSInfo':
                    for gps_tag, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        
                        if gps_tag_name == 'GPSLatitude':
                            lat = gps_value[0] + gps_value[1]/60 + gps_value[2]/3600
                            gps_info['lat'] = lat
                        elif gps_tag_name == 'GPSLongitude':
                            lon = gps_value[0] + gps_value[1]/60 + gps_value[2]/3600
                            gps_info['lon'] = -lon  # West is negative
                        elif gps_tag_name == 'GPSAltitude':
                            gps_info['altitude'] = float(gps_value)
                        elif gps_tag_name == 'GPSImgDirection':
                            gps_info['heading'] = float(gps_value)
            
            # Get timestamp
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
        """Convert thermal pixel coordinates to lat/lon"""
        # Map thermal pixel to RGB coordinates
        rgb_x = self.offset_x + thermal_x * self.scale_x
        rgb_y = self.offset_y + thermal_y * self.scale_y
        
        # Calculate offset from RGB center
        rgb_center_x = self.rgb_width / 2
        rgb_center_y = self.rgb_height / 2
        
        offset_x_pixels = rgb_x - rgb_center_x
        offset_y_pixels = rgb_y - rgb_center_y
        
        # Ground sample distance for RGB camera
        rgb_fov_deg = 80  # Typical RGB drone camera FOV
        rgb_fov_rad = np.radians(rgb_fov_deg)
        
        # Ground width covered by RGB
        ground_width = 2 * altitude * np.tan(rgb_fov_rad / 2)
        
        # Meters per RGB pixel
        gsd = ground_width / self.rgb_width
        
        # Convert to meters
        offset_x_meters = offset_x_pixels * gsd
        offset_y_meters = -offset_y_pixels * gsd  # Y is inverted
        
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
    
    def contour_to_polygon(self, contour, rgb_center_lat, rgb_center_lon, 
                          altitude, heading=None):
        """Convert thermal contour to georeferenced polygon"""
        polygon_coords = []
        
        for point in contour:
            # Note: contour points are in (row, col) format, need to swap for (x, y)
            thermal_y, thermal_x = point
            lat, lon = self.thermal_to_latlon(
                thermal_x, thermal_y,
                rgb_center_lat, rgb_center_lon,
                altitude, heading
            )
            polygon_coords.append([lon, lat])  # GeoJSON uses [lon, lat]
        
        # Close the polygon
        if len(polygon_coords) > 0 and polygon_coords[0] != polygon_coords[-1]:
            polygon_coords.append(polygon_coords[0])
        
        return polygon_coords
    
    def calculate_polygon_area(self, polygon_coords):
        """Calculate area of polygon in square meters using shoelace formula"""
        if len(polygon_coords) < 3:
            return 0.0
        
        # Extract lat/lon arrays
        lons = np.array([coord[0] for coord in polygon_coords])
        lats = np.array([coord[1] for coord in polygon_coords])
        
        # Convert to meters using equirectangular projection
        # Use centroid as reference point
        lat_center = np.mean(lats)
        lon_center = np.mean(lons)
        
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat_center))
        
        x = (lons - lon_center) * meters_per_degree_lon
        y = (lats - lat_center) * meters_per_degree_lat
        
        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        
        return area
    
    def process_frame_with_polygons(self, frame_number, plume_info_list):
        """
        Georeference detected SGD plumes with polygon outlines.
        
        Args:
            frame_number: Frame number
            plume_info_list: List of plume dictionaries from detector (with contours)
        
        Returns:
            List of georeferenced polygon features
        """
        # Get GPS from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        gps_info = self.extract_gps(str(rgb_path))
        
        if not gps_info or 'lat' not in gps_info:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        georeferenced = []
        altitude = gps_info.get('altitude', 400)
        heading = gps_info.get('heading')
        
        for plume in plume_info_list:
            # Get contour if available
            contour = plume.get('contour', [])
            
            if len(contour) > 2:  # Need at least 3 points for a polygon
                # Convert contour to georeferenced polygon
                polygon_coords = self.contour_to_polygon(
                    contour,
                    gps_info['lat'], gps_info['lon'],
                    altitude, heading
                )
                
                # Calculate accurate area from polygon
                area_m2 = self.calculate_polygon_area(polygon_coords)
            else:
                # Fallback to point with estimated area
                thermal_y, thermal_x = plume['centroid']
                lat, lon = self.thermal_to_latlon(
                    thermal_x, thermal_y,
                    gps_info['lat'], gps_info['lon'],
                    altitude, heading
                )
                polygon_coords = None
                
                # Estimate area from pixel count
                thermal_fov_deg = 45
                thermal_fov_rad = np.radians(thermal_fov_deg)
                ground_width_thermal = 2 * altitude * np.tan(thermal_fov_rad / 2)
                gsd_thermal = ground_width_thermal / self.thermal_width
                area_m2 = plume['area_pixels'] * (gsd_thermal ** 2)
            
            # Create georeferenced record
            sgd_feature = {
                'frame': frame_number,
                'datetime': gps_info.get('datetime', ''),
                'polygon': polygon_coords,  # Full outline
                'centroid': {
                    'latitude': lat if polygon_coords is None else np.mean([c[1] for c in polygon_coords[:-1]]),
                    'longitude': lon if polygon_coords is None else np.mean([c[0] for c in polygon_coords[:-1]])
                },
                'area_m2': area_m2,
                'area_pixels': plume['area_pixels'],
                'temperature_anomaly': plume.get('mean_temp_diff', 0),
                'shore_distance': plume['min_shore_distance'],
                'altitude': altitude,
                'eccentricity': plume.get('eccentricity', 0)
            }
            
            georeferenced.append(sgd_feature)
            self.sgd_polygons.append(sgd_feature)
        
        return georeferenced
    
    def export_geojson_polygons(self, output_path="sgd_polygons.geojson"):
        """Export SGD locations as polygon features in GeoJSON"""
        features = []
        
        for loc in self.sgd_polygons:
            if loc['polygon']:
                # Create polygon feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [loc['polygon']]  # Outer ring
                    },
                    "properties": {
                        "frame": loc['frame'],
                        "datetime": loc['datetime'],
                        "area_m2": round(loc['area_m2'], 2),
                        "area_pixels": loc['area_pixels'],
                        "temperature_anomaly": round(loc.get('temperature_anomaly', 0), 2),
                        "shore_distance": round(loc['shore_distance'], 1),
                        "altitude": round(loc['altitude'], 1),
                        "eccentricity": round(loc['eccentricity'], 3)
                    }
                }
            else:
                # Fallback to point if no polygon
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [loc['centroid']['longitude'], loc['centroid']['latitude']]
                    },
                    "properties": {
                        "frame": loc['frame'],
                        "datetime": loc['datetime'],
                        "area_m2": round(loc['area_m2'], 2),
                        "area_pixels": loc['area_pixels'],
                        "temperature_anomaly": round(loc.get('temperature_anomaly', 0), 2),
                        "shore_distance": round(loc['shore_distance'], 1),
                        "altitude": round(loc['altitude'], 1)
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
        
        polygon_count = sum(1 for loc in self.sgd_polygons if loc['polygon'])
        point_count = len(self.sgd_polygons) - polygon_count
        
        print(f"Exported {len(features)} SGD features to {output_path}")
        print(f"  - {polygon_count} polygons with accurate areas")
        print(f"  - {point_count} points (fallback)")
        
        # Calculate total area
        total_area = sum(loc['area_m2'] for loc in self.sgd_polygons)
        print(f"  - Total SGD area: {total_area:.1f} m²")
        
        return output_path
    
    def export_csv_with_areas(self, output_path="sgd_areas.csv"):
        """Export to CSV with detailed area information"""
        with open(output_path, 'w') as f:
            f.write("frame,datetime,centroid_lat,centroid_lon,area_m2,area_pixels,"
                   "temperature_anomaly,shore_distance,altitude,eccentricity,has_polygon\n")
            
            for loc in self.sgd_polygons:
                has_polygon = "yes" if loc['polygon'] else "no"
                f.write(f"{loc['frame']},{loc['datetime']},"
                       f"{loc['centroid']['latitude']},{loc['centroid']['longitude']},"
                       f"{loc['area_m2']:.2f},{loc['area_pixels']},"
                       f"{loc.get('temperature_anomaly', 0):.2f},"
                       f"{loc['shore_distance']:.1f},{loc['altitude']:.1f},"
                       f"{loc.get('eccentricity', 0):.3f},{has_polygon}\n")
        
        print(f"Exported {len(self.sgd_polygons)} SGD areas to {output_path}")
        return output_path


def main():
    """Test polygon georeferencing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SGD Polygon Georeferencing')
    parser.add_argument('--data', type=str, default='data/100MEDIA',
                       help='Path to data directory')
    parser.add_argument('--frame', type=int, default=248,
                       help='Frame number to test')
    
    args = parser.parse_args()
    
    # Initialize
    georef = SGDPolygonGeoref(args.data)
    
    print(f"Testing polygon georeferencing on frame {args.frame}")
    print("This requires running the detector first to get plume contours")
    
    # Note: This is a test - actual integration would get plume_info from detector
    print("\nTo use this:")
    print("1. Run sgd_detector_integrated.py to detect plumes")
    print("2. Pass the plume_info list to process_frame_with_polygons()")
    print("3. Export results as polygon GeoJSON")


if __name__ == "__main__":
    main()