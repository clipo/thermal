#!/usr/bin/env python3
"""
Generate KML files showing thermal image footprints/coverage along the coast.

Creates two outputs:
1. Individual frame outlines - shows each thermal image as a rectangular polygon
2. Merged coverage - combines all frames into a single polygon showing total survey area

Usage:
    python generate_frame_footprints.py --data "/path/to/survey/data"
    python generate_frame_footprints.py --data "/path/to/survey" --output custom_name --skip 5
"""

import argparse
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
import os
from xml.etree import ElementTree as ET
import xml.dom.minidom as minidom

# Try to import piexif, fall back to exifread if not available
try:
    import piexif
    from piexif import GPSIFD
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False
    try:
        import exifread
        EXIFREAD_AVAILABLE = True
    except ImportError:
        EXIFREAD_AVAILABLE = False
        print("Warning: Neither piexif nor exifread installed. GPS extraction limited.")
        print("Install with: pip install piexif   OR   pip install exifread")

# Try to import Shapely for polygon merging
try:
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: Shapely not installed. Merged coverage polygon will use bounding box approximation.")
    print("Install with: pip install shapely")


class ThermalFrameMapper:
    """Generate KML files showing thermal image footprints"""

    def __init__(self, data_dir, output_base="thermal_coverage", frame_skip=1, verbose=True):
        """
        Initialize the frame mapper.

        Args:
            data_dir: Directory containing MAX_*.JPG and IRX_*.irg files
            output_base: Base name for output files (will create _frames.kml and _merged.kml)
            frame_skip: Process every Nth frame (1=all frames)
            verbose: Print progress information
        """
        self.data_dir = Path(data_dir)
        self.output_base = output_base
        self.frame_skip = frame_skip
        self.verbose = verbose

        # Thermal camera specifications
        self.thermal_width = 640
        self.thermal_height = 512
        self.thermal_fov_deg = 50.0  # Field of view in degrees

        # Storage for frame data
        self.frame_footprints = []

        # Ensure output directory exists
        self.output_dir = Path("sgd_output")
        self.output_dir.mkdir(exist_ok=True)

    def extract_gps_info(self, image_path):
        """
        Extract GPS information from image EXIF data.

        Returns:
            Dictionary with lat, lon, altitude, heading, datetime
        """
        if PIEXIF_AVAILABLE:
            return self._extract_gps_piexif(image_path)
        elif EXIFREAD_AVAILABLE:
            return self._extract_gps_exifread(image_path)
        else:
            # Fallback: Try using the same method as sgd_georef_polygons.py
            return self._extract_gps_fallback(image_path)

    def _extract_gps_fallback(self, image_path):
        """Fallback GPS extraction using the method from sgd_georef_polygons.py"""
        try:
            # Import here to avoid dependency if not needed
            from sgd_georef_polygons import SGDPolygonGeoref
            georef = SGDPolygonGeoref(base_path=image_path.parent)
            gps_info = georef.extract_gps(str(image_path), verbose=False)

            if gps_info and 'lat' in gps_info:
                return {
                    'lat': gps_info['lat'],
                    'lon': gps_info['lon'],
                    'altitude': gps_info.get('altitude', 100.0),
                    'heading': gps_info.get('heading', 0.0),
                    'datetime': gps_info.get('datetime', '')
                }
            return None
        except Exception as e:
            if self.verbose:
                print(f"  Fallback GPS extraction failed: {e}")
            return None

    def _extract_gps_piexif(self, image_path):
        """Extract GPS using piexif library with XMP heading support"""
        try:
            from PIL import Image
            import re

            # First get basic EXIF with piexif
            exif_dict = piexif.load(str(image_path))

            # Extract GPS data
            gps_data = exif_dict.get('GPS', {})

            if not gps_data:
                return None

            # Get latitude
            lat_ref = gps_data.get(GPSIFD.GPSLatitudeRef, b'N').decode('utf-8')
            lat_tuple = gps_data.get(GPSIFD.GPSLatitude)
            if lat_tuple:
                lat = self.convert_to_degrees(lat_tuple)
                if lat_ref == 'S':
                    lat = -lat
            else:
                return None

            # Get longitude
            lon_ref = gps_data.get(GPSIFD.GPSLongitudeRef, b'E').decode('utf-8')
            lon_tuple = gps_data.get(GPSIFD.GPSLongitude)
            if lon_tuple:
                lon = self.convert_to_degrees(lon_tuple)
                if lon_ref == 'W':
                    lon = -lon
            else:
                return None

            # Get altitude
            altitude = 100.0  # Default altitude
            alt_tuple = gps_data.get(GPSIFD.GPSAltitude)
            if alt_tuple:
                altitude = float(alt_tuple[0]) / float(alt_tuple[1]) if alt_tuple[1] != 0 else 100.0

            # Get heading/direction from EXIF
            heading = 0.0
            heading_source = None
            heading_tuple = gps_data.get(GPSIFD.GPSImgDirection)
            if heading_tuple:
                heading = float(heading_tuple[0]) / float(heading_tuple[1]) if heading_tuple[1] != 0 else 0.0
                heading_source = 'EXIF:GPSImgDirection'

            # If no EXIF heading, check XMP for Camera:Yaw (Autel drones)
            if heading == 0.0:
                try:
                    img = Image.open(image_path)
                    if hasattr(img, 'info'):
                        xmp_data = img.info.get('xmp')
                        if xmp_data:
                            # Convert bytes to string if needed
                            if isinstance(xmp_data, bytes):
                                xmp_str = xmp_data.decode('utf-8', errors='ignore')
                            else:
                                xmp_str = str(xmp_data)

                            # Search for Camera:Yaw in XMP
                            yaw_match = re.search(r'Camera:Yaw="?([\-\d\.]+)"?', xmp_str)
                            if yaw_match:
                                yaw_value = float(yaw_match.group(1))
                                # Convert yaw to compass heading (0-360)
                                if yaw_value < 0:
                                    heading = 360 + yaw_value
                                else:
                                    heading = yaw_value
                                heading_source = 'XMP:Camera:Yaw'
                                if self.verbose:
                                    print(f"    Found heading from XMP: {heading:.1f}°")
                except Exception as e:
                    if self.verbose:
                        print(f"    Could not extract XMP heading: {e}")

            # Get timestamp
            datetime_str = ""
            if '0th' in exif_dict:
                datetime_bytes = exif_dict['0th'].get(0x0132)  # DateTime tag
                if datetime_bytes:
                    datetime_str = datetime_bytes.decode('utf-8')

            return {
                'lat': lat,
                'lon': lon,
                'altitude': altitude,
                'heading': heading,
                'datetime': datetime_str
            }

        except Exception as e:
            if self.verbose:
                print(f"  Warning: Could not extract GPS from {image_path}: {e}")
            return None

    def convert_to_degrees(self, value):
        """Convert GPS coordinates to decimal degrees"""
        d = float(value[0][0]) / float(value[0][1])
        m = float(value[1][0]) / float(value[1][1])
        s = float(value[2][0]) / float(value[2][1])
        return d + (m / 60.0) + (s / 3600.0)

    def calculate_footprint_corners(self, lat, lon, altitude, heading):
        """
        Calculate the four corners of the thermal image footprint on the ground.

        Args:
            lat: Center latitude
            lon: Center longitude
            altitude: Altitude in meters
            heading: Camera heading in degrees

        Returns:
            List of (lon, lat) tuples for the four corners
        """
        # Calculate ground coverage
        fov_rad = np.radians(self.thermal_fov_deg)
        ground_width = 2 * altitude * np.tan(fov_rad / 2)

        # Aspect ratio
        aspect_ratio = self.thermal_height / self.thermal_width
        ground_height = ground_width * aspect_ratio

        # Half dimensions
        half_width = ground_width / 2
        half_height = ground_height / 2

        # Convert to degrees (approximate)
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat))

        # Corner offsets in meters (before rotation)
        corners_m = [
            (-half_width, -half_height),  # Bottom-left
            (half_width, -half_height),   # Bottom-right
            (half_width, half_height),    # Top-right
            (-half_width, half_height),   # Top-left
        ]

        # Apply rotation based on heading
        heading_rad = np.radians(heading)
        cos_h = np.cos(heading_rad)
        sin_h = np.sin(heading_rad)

        corners_deg = []
        for x_m, y_m in corners_m:
            # Rotate
            x_rot = x_m * cos_h - y_m * sin_h
            y_rot = x_m * sin_h + y_m * cos_h

            # Convert to degrees
            lon_offset = x_rot / meters_per_degree_lon
            lat_offset = y_rot / meters_per_degree_lat

            corner_lon = lon + lon_offset
            corner_lat = lat + lat_offset
            corners_deg.append((corner_lon, corner_lat))

        # Close the polygon by adding the first point again
        corners_deg.append(corners_deg[0])

        return corners_deg

    def process_frames(self):
        """Process all frames and calculate footprints"""
        # Find all RGB files
        rgb_files = sorted(self.data_dir.glob("MAX_*.JPG"))

        if not rgb_files:
            # Try searching in subdirectories (100MEDIA, 101MEDIA, etc.)
            for subdir in sorted(self.data_dir.glob("*MEDIA")):
                rgb_files.extend(sorted(subdir.glob("MAX_*.JPG")))

        if not rgb_files:
            print(f"No MAX_*.JPG files found in {self.data_dir}")
            return False

        print(f"Found {len(rgb_files)} RGB files")

        # Process frames with skipping
        processed = 0
        skipped = 0

        for i, rgb_file in enumerate(rgb_files):
            # Skip frames based on frame_skip parameter
            if i % self.frame_skip != 0:
                skipped += 1
                continue

            # Extract frame number
            frame_num = int(rgb_file.stem.split('_')[1])

            # Check if thermal file exists
            thermal_file = rgb_file.parent / f"IRX_{frame_num:04d}.irg"
            if not thermal_file.exists():
                if self.verbose:
                    print(f"  Skipping frame {frame_num}: No thermal file")
                continue

            # Extract GPS info
            gps_info = self.extract_gps_info(rgb_file)
            if not gps_info:
                if self.verbose:
                    print(f"  Skipping frame {frame_num}: No GPS data")
                continue

            # Report heading status on first frame and periodically
            if self.verbose and (processed == 0 or processed % 20 == 0):
                if gps_info['heading'] != 0.0:
                    print(f"  Frame {frame_num}: Heading {gps_info['heading']:.1f}° (rotation applied)")
                else:
                    print(f"  Frame {frame_num}: No heading data (no rotation)")

            # Calculate footprint corners
            corners = self.calculate_footprint_corners(
                gps_info['lat'],
                gps_info['lon'],
                gps_info['altitude'],
                gps_info['heading']
            )

            # Store frame data
            self.frame_footprints.append({
                'frame': frame_num,
                'rgb_path': str(rgb_file),
                'thermal_path': str(thermal_file),
                'lat': gps_info['lat'],
                'lon': gps_info['lon'],
                'altitude': gps_info['altitude'],
                'heading': gps_info['heading'],
                'datetime': gps_info['datetime'],
                'corners': corners
            })

            processed += 1

            if self.verbose and processed % 50 == 0:
                print(f"  Processed {processed} frames...")

        print(f"\nProcessed {processed} frames, skipped {skipped}")
        return processed > 0

    def create_individual_frames_kml(self):
        """Create KML with individual frame outlines"""
        output_file = self.output_dir / f"{self.output_base}_frames.kml"

        kml = []
        kml.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append(f'<name>Thermal Frame Coverage - Individual Frames</name>')
        kml.append(f'<description>Thermal image footprints from {self.data_dir}</description>')

        # Define styles
        kml.append('<Style id="frameStyle">')
        kml.append('  <LineStyle>')
        kml.append('    <color>ff00ffff</color>')  # Yellow outline
        kml.append('    <width>2</width>')
        kml.append('  </LineStyle>')
        kml.append('  <PolyStyle>')
        kml.append('    <color>3300ffff</color>')  # Semi-transparent yellow fill
        kml.append('  </PolyStyle>')
        kml.append('</Style>')

        # Add each frame as a placemark
        for frame_data in self.frame_footprints:
            kml.append('<Placemark>')
            kml.append(f'  <name>Frame {frame_data["frame"]}</name>')

            # Description with metadata
            desc = []
            desc.append(f'Frame: {frame_data["frame"]}')
            desc.append(f'DateTime: {frame_data["datetime"]}')
            desc.append(f'Center: {frame_data["lat"]:.6f}, {frame_data["lon"]:.6f}')
            desc.append(f'Altitude: {frame_data["altitude"]:.1f} m')
            desc.append(f'Heading: {frame_data["heading"]:.1f}°')
            desc.append('---')
            desc.append(f'RGB: {frame_data["rgb_path"]}')
            desc.append(f'Thermal: {frame_data["thermal_path"]}')

            kml.append(f'  <description><![CDATA[{chr(10).join(desc)}]]></description>')
            kml.append('  <styleUrl>#frameStyle</styleUrl>')

            # Add polygon
            kml.append('  <Polygon>')
            kml.append('    <extrude>0</extrude>')
            kml.append('    <altitudeMode>clampToGround</altitudeMode>')
            kml.append('    <outerBoundaryIs>')
            kml.append('      <LinearRing>')
            kml.append('        <coordinates>')

            # Add corners
            for lon, lat in frame_data['corners']:
                kml.append(f'          {lon},{lat},0')

            kml.append('        </coordinates>')
            kml.append('      </LinearRing>')
            kml.append('    </outerBoundaryIs>')
            kml.append('  </Polygon>')
            kml.append('</Placemark>')

        kml.append('</Document>')
        kml.append('</kml>')

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(kml))

        print(f"Created individual frames KML: {output_file}")
        return output_file

    def create_merged_coverage_kml(self):
        """Create KML with merged coverage polygon"""
        output_file = self.output_dir / f"{self.output_base}_merged.kml"

        # Create merged polygon
        if SHAPELY_AVAILABLE:
            # Use Shapely to create accurate union
            polygons = []
            for frame_data in self.frame_footprints:
                # Remove the duplicate closing point for Shapely
                corners = frame_data['corners'][:-1]
                poly = Polygon(corners)
                if poly.is_valid:
                    polygons.append(poly)

            if polygons:
                # Create union of all polygons
                merged = unary_union(polygons)

                # Extract coordinates
                if isinstance(merged, Polygon):
                    merged_coords = list(merged.exterior.coords)
                elif isinstance(merged, MultiPolygon):
                    # If multiple disconnected areas, use the largest
                    largest = max(merged.geoms, key=lambda p: p.area)
                    merged_coords = list(largest.exterior.coords)
                else:
                    merged_coords = None
            else:
                merged_coords = None
        else:
            # Simple bounding box approximation
            all_lats = []
            all_lons = []
            for frame_data in self.frame_footprints:
                for lon, lat in frame_data['corners'][:-1]:
                    all_lats.append(lat)
                    all_lons.append(lon)

            if all_lats and all_lons:
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)

                # Create bounding box
                merged_coords = [
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                    (min_lon, min_lat)  # Close the polygon
                ]
            else:
                merged_coords = None

        if not merged_coords:
            print("Could not create merged coverage polygon")
            return None

        # Create KML
        kml = []
        kml.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append(f'<name>Thermal Frame Coverage - Merged</name>')
        kml.append(f'<description>Combined coverage area from all thermal frames</description>')

        # Define style
        kml.append('<Style id="mergedStyle">')
        kml.append('  <LineStyle>')
        kml.append('    <color>ff0000ff</color>')  # Red outline
        kml.append('    <width>3</width>')
        kml.append('  </LineStyle>')
        kml.append('  <PolyStyle>')
        kml.append('    <color>330000ff</color>')  # Semi-transparent red fill
        kml.append('  </PolyStyle>')
        kml.append('</Style>')

        # Add merged polygon
        kml.append('<Placemark>')
        kml.append(f'  <name>Total Survey Coverage</name>')

        # Calculate statistics
        total_frames = len(self.frame_footprints)
        if self.frame_footprints:
            start_time = self.frame_footprints[0]['datetime']
            end_time = self.frame_footprints[-1]['datetime']
            avg_altitude = np.mean([f['altitude'] for f in self.frame_footprints])
        else:
            start_time = end_time = "N/A"
            avg_altitude = 0

        desc = []
        desc.append(f'Total frames: {total_frames}')
        desc.append(f'Frame skip: Every {self.frame_skip} frames')
        desc.append(f'Start: {start_time}')
        desc.append(f'End: {end_time}')
        desc.append(f'Average altitude: {avg_altitude:.1f} m')
        desc.append(f'Data directory: {self.data_dir}')

        kml.append(f'  <description><![CDATA[{chr(10).join(desc)}]]></description>')
        kml.append('  <styleUrl>#mergedStyle</styleUrl>')

        # Add polygon
        kml.append('  <Polygon>')
        kml.append('    <extrude>0</extrude>')
        kml.append('    <altitudeMode>clampToGround</altitudeMode>')
        kml.append('    <outerBoundaryIs>')
        kml.append('      <LinearRing>')
        kml.append('        <coordinates>')

        # Add merged coordinates
        for coord in merged_coords:
            if isinstance(coord, tuple) and len(coord) == 2:
                lon, lat = coord
            else:
                lon, lat = coord[0], coord[1]
            kml.append(f'          {lon},{lat},0')

        kml.append('        </coordinates>')
        kml.append('      </LinearRing>')
        kml.append('    </outerBoundaryIs>')
        kml.append('  </Polygon>')
        kml.append('</Placemark>')

        kml.append('</Document>')
        kml.append('</kml>')

        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(kml))

        print(f"Created merged coverage KML: {output_file}")
        return output_file

    def run(self):
        """Process frames and generate both KML files"""
        print(f"\nProcessing thermal frames from: {self.data_dir}")
        print("="*60)

        # Process frames
        if not self.process_frames():
            print("No frames to process!")
            return False

        # Generate KML files
        individual_kml = self.create_individual_frames_kml()
        merged_kml = self.create_merged_coverage_kml()

        print("\n" + "="*60)
        print("COMPLETED!")
        print("="*60)
        print(f"Individual frames: {individual_kml}")
        print(f"Merged coverage: {merged_kml}")
        print(f"\nTotal frames mapped: {len(self.frame_footprints)}")

        return True


def main():
    """Main function to run frame footprint generation"""
    parser = argparse.ArgumentParser(
        description='Generate KML files showing thermal image footprints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all frames in a directory
  python generate_frame_footprints.py --data "/path/to/survey/data"

  # Process every 5th frame for faster processing
  python generate_frame_footprints.py --data "/path/to/survey" --skip 5

  # Custom output name
  python generate_frame_footprints.py --data "/path/to/survey" --output vaihu_coverage

Output:
  Creates two KML files in sgd_output/:
  - *_frames.kml: Individual frame outlines
  - *_merged.kml: Combined coverage area
        """
    )

    parser.add_argument('--data', required=True,
                       help='Directory containing MAX_*.JPG and IRX_*.irg files')
    parser.add_argument('--output', default='thermal_coverage',
                       help='Base name for output files (default: thermal_coverage)')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame (1=all, 5=every 5th, etc.)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    args = parser.parse_args()

    # Create mapper
    mapper = ThermalFrameMapper(
        data_dir=args.data,
        output_base=args.output,
        frame_skip=args.skip,
        verbose=not args.quiet
    )

    # Run processing
    success = mapper.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()