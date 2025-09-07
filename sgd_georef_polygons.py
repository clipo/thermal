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
    
    def extract_gps(self, image_path, verbose=False):
        """Extract GPS and orientation metadata from image (including XMP)"""
        try:
            img = Image.open(image_path)
            exifdata = img._getexif()
            
            if not exifdata:
                return None
            
            gps_info = {}
            
            # Process all EXIF tags first to get orientation
            for tag, value in exifdata.items():
                tag_name = TAGS.get(tag, tag)
                
                # Standard orientation tag (1-8 values)
                if tag_name == 'Orientation':
                    gps_info['exif_orientation'] = value
                    if verbose:
                        print(f"  EXIF Orientation tag: {value}")
                
                # Process GPS tags
                if tag_name == 'GPSInfo':
                    for gps_tag, gps_value in value.items():
                        gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                        
                        if gps_tag_name == 'GPSLatitude':
                            lat = float(gps_value[0]) + float(gps_value[1])/60 + float(gps_value[2])/3600
                            gps_info['lat'] = lat
                        elif gps_tag_name == 'GPSLongitude':
                            lon = float(gps_value[0]) + float(gps_value[1])/60 + float(gps_value[2])/3600
                            gps_info['lon'] = -lon  # West is negative
                        elif gps_tag_name == 'GPSAltitude':
                            gps_info['altitude'] = float(gps_value)
                        elif gps_tag_name == 'GPSImgDirection':
                            # This is the compass heading when image was taken
                            gps_info['heading'] = float(gps_value)
                            if verbose:
                                print(f"  GPS Image Direction (heading): {gps_value}°")
                        elif gps_tag_name == 'GPSImgDirectionRef':
                            # T = True North, M = Magnetic North
                            gps_info['heading_ref'] = gps_value
                            if verbose:
                                print(f"  Heading Reference: {gps_value}")
                        elif gps_tag_name == 'GPSTrack':
                            # Direction of movement
                            gps_info['track'] = float(gps_value)
                            if verbose:
                                print(f"  GPS Track (movement direction): {gps_value}°")
            
            # Get timestamp
            datetime_str = exifdata.get(306) or exifdata.get(36867)
            if datetime_str:
                gps_info['datetime'] = datetime_str
            
            # Check XMP metadata for Camera:Yaw if no GPSImgDirection found
            if 'heading' not in gps_info and hasattr(img, 'info'):
                # Look for XMP data
                xmp_data = img.info.get('xmp')
                if xmp_data:
                    # Convert bytes to string if needed
                    if isinstance(xmp_data, bytes):
                        xmp_str = xmp_data.decode('utf-8', errors='ignore')
                    else:
                        xmp_str = str(xmp_data)
                    
                    # Search for Camera:Yaw in XMP
                    import re
                    yaw_match = re.search(r'Camera:Yaw="?([\-\d\.]+)"?', xmp_str)
                    if yaw_match:
                        yaw_value = float(yaw_match.group(1))
                        # Convert yaw to compass heading (0-360)
                        # Yaw is typically -180 to 180, with 0 being north
                        # Negative values are west, positive are east
                        if yaw_value < 0:
                            heading = 360 + yaw_value
                        else:
                            heading = yaw_value
                        gps_info['heading'] = heading
                        gps_info['heading_source'] = 'XMP:Camera:Yaw'
                        if verbose:
                            print(f"  XMP Camera:Yaw: {yaw_value}° → Heading: {heading}°")
            
            # Log what we found
            if verbose and 'lat' in gps_info:
                print(f"  Location: ({gps_info['lat']:.6f}, {gps_info['lon']:.6f})")
                print(f"  Altitude: {gps_info.get('altitude', 'N/A')} m")
                if 'heading' in gps_info:
                    source = gps_info.get('heading_source', 'EXIF:GPSImgDirection')
                    print(f"  ✓ Heading: {gps_info['heading']:.1f}° (from {source})")
                else:
                    print("  ⚠️ No heading data found - georeferencing may be less accurate")
            
            return gps_info if 'lat' in gps_info else None
            
        except Exception as e:
            print(f"Error extracting GPS: {e}")
            return None
    
    def thermal_to_latlon(self, thermal_x, thermal_y, 
                          rgb_center_lat, rgb_center_lon, 
                          altitude, heading=None):
        """Convert thermal pixel coordinates to lat/lon"""
        # Ensure coordinates are float (not Fraction)
        rgb_center_lat = float(rgb_center_lat)
        rgb_center_lon = float(rgb_center_lon)
        altitude = float(altitude)
        
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
        # Heading is the direction the camera was pointing (0° = North, 90° = East, etc.)
        if heading is not None:
            # Convert heading to radians
            # Note: We rotate by negative heading because we're converting from 
            # camera coordinates to world coordinates
            heading_rad = np.radians(-heading)
            cos_h = np.cos(heading_rad)
            sin_h = np.sin(heading_rad)
            
            # Rotate offsets from camera frame to world frame
            rotated_x = offset_x_meters * cos_h - offset_y_meters * sin_h
            rotated_y = offset_x_meters * sin_h + offset_y_meters * cos_h
            offset_x_meters = rotated_x
            offset_y_meters = rotated_y
        # If no heading, assume north-facing (0°)
        
        # Convert to degrees
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(rgb_center_lat))
        
        delta_lat = offset_y_meters / meters_per_degree_lat
        delta_lon = offset_x_meters / meters_per_degree_lon
        
        return rgb_center_lat + delta_lat, rgb_center_lon + delta_lon
    
    def contour_to_polygon(self, contour, rgb_center_lat, rgb_center_lon, 
                          altitude, heading=None):
        """Convert thermal contour to georeferenced polygon"""
        # Ensure coordinates are float (not Fraction)
        rgb_center_lat = float(rgb_center_lat)
        rgb_center_lon = float(rgb_center_lon)
        altitude = float(altitude)
        
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
    
    def process_frame_with_polygons(self, frame_number, plume_info_list, verbose=False):
        """
        Georeference detected SGD plumes with polygon outlines.
        
        Args:
            frame_number: Frame number
            plume_info_list: List of plume dictionaries from detector (with contours)
            verbose: Print orientation information
        
        Returns:
            List of georeferenced polygon features
        """
        # Get GPS from RGB image
        rgb_path = self.base_path / f"MAX_{frame_number:04d}.JPG"
        
        if verbose:
            print(f"\nProcessing frame {frame_number}:")
        
        gps_info = self.extract_gps(str(rgb_path), verbose=verbose)
        
        if not gps_info or 'lat' not in gps_info:
            print(f"No GPS data for frame {frame_number}")
            return []
        
        georeferenced = []
        altitude = gps_info.get('altitude', 400)
        heading = gps_info.get('heading')
        
        if verbose:
            if heading is not None:
                print(f"  ✓ Applying rotation correction: {heading:.1f}°")
            else:
                print("  ⚠️ No heading data - assuming north-facing (0°)")
        
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
    
    def export_kml_polygons(self, output_path="sgd_polygons.kml"):
        """Export SGD locations as KML with polygon support for Google Earth"""
        kml = ['<?xml version="1.0" encoding="UTF-8"?>']
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append('<name>SGD Detection Polygons</name>')
        kml.append('<description>Submarine Groundwater Discharge plume outlines</description>')
        
        # Define styles for polygons and points
        # Style for polygon SGD
        kml.append('<Style id="sgdPolygonStyle">')
        kml.append('  <LineStyle>')
        kml.append('    <color>ff0000ff</color>')  # Red outline
        kml.append('    <width>2</width>')
        kml.append('  </LineStyle>')
        kml.append('  <PolyStyle>')
        kml.append('    <color>7f0000ff</color>')  # Semi-transparent red fill
        kml.append('  </PolyStyle>')
        kml.append('</Style>')
        
        # Style for point SGD (fallback)
        kml.append('<Style id="sgdPointStyle">')
        kml.append('  <IconStyle>')
        kml.append('    <color>ff0000ff</color>')
        kml.append('    <scale>1.2</scale>')
        kml.append('    <Icon>')
        kml.append('      <href>http://maps.google.com/mapfiles/kml/shapes/water.png</href>')
        kml.append('    </Icon>')
        kml.append('  </IconStyle>')
        kml.append('</Style>')
        
        # Add each SGD location
        for i, loc in enumerate(self.sgd_polygons):
            kml.append('<Placemark>')
            kml.append(f'  <name>SGD {i+1} (Frame {loc["frame"]})</name>')
            
            # Create description with all metadata
            desc = []
            desc.append(f'Frame: {loc["frame"]}')
            desc.append(f'Date/Time: {loc["datetime"]}')
            desc.append(f'Area: {loc["area_m2"]:.1f} m²')
            desc.append(f'Area (pixels): {loc["area_pixels"]}')
            desc.append(f'Temperature anomaly: {loc.get("temperature_anomaly", 0):.1f}°C')
            desc.append(f'Shore distance: {loc["shore_distance"]:.1f} m')
            desc.append(f'Altitude: {loc["altitude"]:.1f} m')
            desc.append(f'Eccentricity: {loc.get("eccentricity", 0):.3f}')
            
            kml.append(f'  <description><![CDATA[{chr(10).join(desc)}]]></description>')
            
            if loc['polygon'] and len(loc['polygon']) > 2:
                # Add as polygon
                kml.append('  <styleUrl>#sgdPolygonStyle</styleUrl>')
                kml.append('  <Polygon>')
                kml.append('    <extrude>0</extrude>')
                kml.append('    <altitudeMode>clampToGround</altitudeMode>')
                kml.append('    <outerBoundaryIs>')
                kml.append('      <LinearRing>')
                kml.append('        <coordinates>')
                
                # Add polygon coordinates (KML uses lon,lat,altitude format)
                coord_strings = []
                for coord in loc['polygon']:
                    coord_strings.append(f'          {coord[0]},{coord[1]},0')
                kml.append('\n'.join(coord_strings))
                
                kml.append('        </coordinates>')
                kml.append('      </LinearRing>')
                kml.append('    </outerBoundaryIs>')
                kml.append('  </Polygon>')
            else:
                # Add as point (fallback)
                kml.append('  <styleUrl>#sgdPointStyle</styleUrl>')
                kml.append('  <Point>')
                kml.append('    <coordinates>')
                kml.append(f'      {loc["centroid"]["longitude"]},{loc["centroid"]["latitude"]},0')
                kml.append('    </coordinates>')
                kml.append('  </Point>')
            
            kml.append('</Placemark>')
        
        # Add a folder with summary statistics
        kml.append('<Folder>')
        kml.append('  <name>Summary Statistics</name>')
        kml.append('  <description>')
        total_area = sum(loc['area_m2'] for loc in self.sgd_polygons)
        polygon_count = sum(1 for loc in self.sgd_polygons if loc['polygon'])
        point_count = len(self.sgd_polygons) - polygon_count
        kml.append(f'    Total SGD locations: {len(self.sgd_polygons)}')
        kml.append(f'    Polygons: {polygon_count}')
        kml.append(f'    Points: {point_count}')
        kml.append(f'    Total area: {total_area:.1f} m²')
        kml.append('  </description>')
        kml.append('</Folder>')
        
        kml.append('</Document>')
        kml.append('</kml>')
        
        # Write KML file
        with open(output_path, 'w') as f:
            f.write('\n'.join(kml))
        
        print(f"Exported {len(self.sgd_polygons)} SGD locations to {output_path}")
        polygon_count = sum(1 for loc in self.sgd_polygons if loc['polygon'])
        point_count = len(self.sgd_polygons) - polygon_count
        print(f"  - {polygon_count} polygons with outlines")
        print(f"  - {point_count} points (fallback)")
        print(f"  - Total area: {sum(loc['area_m2'] for loc in self.sgd_polygons):.1f} m²")
        print(f"  - Open in Google Earth to visualize")
        
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