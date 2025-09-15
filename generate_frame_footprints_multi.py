#!/usr/bin/env python3
"""
Generate KML files showing thermal image footprints across multiple directories.

Searches for and processes all XXXMEDIA subdirectories, creating combined coverage maps.

Usage:
    python generate_frame_footprints_multi.py --data "/path/to/survey"
    python generate_frame_footprints_multi.py --data "/path/to/survey" --search
"""

import argparse
import sys
from pathlib import Path
from generate_frame_footprints import ThermalFrameMapper

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class MultiDirectoryFrameMapper:
    """Process multiple directories and create combined coverage maps"""

    def __init__(self, base_dir, output_base="thermal_coverage", frame_skip=1, verbose=True):
        """
        Initialize multi-directory mapper.

        Args:
            base_dir: Base directory to search for XXXMEDIA subdirectories
            output_base: Base name for output files
            frame_skip: Process every Nth frame
            verbose: Print progress
        """
        self.base_dir = Path(base_dir)
        self.output_base = output_base
        self.frame_skip = frame_skip
        self.verbose = verbose

        # Storage for all frame data
        self.all_frame_footprints = []
        self.directory_results = {}

        # Output directory
        self.output_dir = Path("sgd_output")
        self.output_dir.mkdir(exist_ok=True)

    def find_data_directories(self):
        """Find all directories containing thermal data"""
        data_dirs = []

        # Check if base_dir itself contains data
        if list(self.base_dir.glob("MAX_*.JPG")):
            data_dirs.append(self.base_dir)
            if self.verbose:
                print(f"Found data in: {self.base_dir}")

        # Search for XXXMEDIA subdirectories
        for subdir in sorted(self.base_dir.glob("*MEDIA")):
            if subdir.is_dir() and list(subdir.glob("MAX_*.JPG")):
                data_dirs.append(subdir)
                if self.verbose:
                    print(f"Found data in: {subdir}")

        return data_dirs

    def process_all_directories(self):
        """Process all found directories"""
        data_dirs = self.find_data_directories()

        if not data_dirs:
            print(f"No data directories found in {self.base_dir}")
            return False

        print(f"\nProcessing {len(data_dirs)} directories...")
        print("="*60)

        for data_dir in data_dirs:
            print(f"\nProcessing: {data_dir.name}")

            # Create mapper for this directory
            mapper = ThermalFrameMapper(
                data_dir=data_dir,
                output_base=f"{self.output_base}_{data_dir.name}",
                frame_skip=self.frame_skip,
                verbose=False  # Less verbose for individual directories
            )

            # Process frames
            if mapper.process_frames():
                # Store results
                self.directory_results[str(data_dir)] = {
                    'frames': len(mapper.frame_footprints),
                    'footprints': mapper.frame_footprints
                }

                # Add to combined list
                self.all_frame_footprints.extend(mapper.frame_footprints)

                print(f"  ✓ Processed {len(mapper.frame_footprints)} frames")
            else:
                print(f"  ✗ No valid frames found")

        return len(self.all_frame_footprints) > 0

    def create_combined_frames_kml(self):
        """Create KML with all individual frames from all directories"""
        output_file = self.output_dir / f"{self.output_base}_all_frames.kml"

        kml = []
        kml.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append(f'<name>Combined Thermal Coverage - All Frames</name>')
        kml.append(f'<description>All thermal frames from {len(self.directory_results)} directories</description>')

        # Define styles for different directories
        colors = ['ff0000ff', 'ff00ff00', 'ffff0000', 'ff00ffff', 'ffff00ff', 'ffffff00']

        for i, (dir_path, dir_data) in enumerate(self.directory_results.items()):
            dir_name = Path(dir_path).name
            color = colors[i % len(colors)]

            kml.append(f'<Style id="dir_{i}_style">')
            kml.append('  <LineStyle>')
            kml.append(f'    <color>{color}</color>')
            kml.append('    <width>2</width>')
            kml.append('  </LineStyle>')
            kml.append('  <PolyStyle>')
            kml.append(f'    <color>33{color[2:]}</color>')  # Semi-transparent
            kml.append('  </PolyStyle>')
            kml.append('</Style>')

        # Group frames by directory
        for i, (dir_path, dir_data) in enumerate(self.directory_results.items()):
            dir_name = Path(dir_path).name

            kml.append('<Folder>')
            kml.append(f'  <name>{dir_name} ({dir_data["frames"]} frames)</name>')

            for frame_data in dir_data['footprints']:
                kml.append('  <Placemark>')
                kml.append(f'    <name>Frame {frame_data["frame"]}</name>')

                desc = []
                desc.append(f'Directory: {dir_name}')
                desc.append(f'Frame: {frame_data["frame"]}')
                desc.append(f'DateTime: {frame_data["datetime"]}')
                desc.append(f'Altitude: {frame_data["altitude"]:.1f} m')

                kml.append(f'    <description><![CDATA[{chr(10).join(desc)}]]></description>')
                kml.append(f'    <styleUrl>#dir_{i}_style</styleUrl>')

                kml.append('    <Polygon>')
                kml.append('      <extrude>0</extrude>')
                kml.append('      <altitudeMode>clampToGround</altitudeMode>')
                kml.append('      <outerBoundaryIs>')
                kml.append('        <LinearRing>')
                kml.append('          <coordinates>')

                for lon, lat in frame_data['corners']:
                    kml.append(f'            {lon},{lat},0')

                kml.append('          </coordinates>')
                kml.append('        </LinearRing>')
                kml.append('      </outerBoundaryIs>')
                kml.append('    </Polygon>')
                kml.append('  </Placemark>')

            kml.append('</Folder>')

        kml.append('</Document>')
        kml.append('</kml>')

        with open(output_file, 'w') as f:
            f.write('\n'.join(kml))

        print(f"Created combined frames KML: {output_file}")
        return output_file

    def create_combined_merged_kml(self):
        """Create KML with merged coverage from all directories"""
        output_file = self.output_dir / f"{self.output_base}_all_merged.kml"

        # Create merged polygon from all frames
        if SHAPELY_AVAILABLE and self.all_frame_footprints:
            polygons = []
            for frame_data in self.all_frame_footprints:
                corners = frame_data['corners'][:-1]
                poly = Polygon(corners)
                if poly.is_valid:
                    polygons.append(poly)

            if polygons:
                merged = unary_union(polygons)
                if hasattr(merged, 'exterior'):
                    merged_coords = list(merged.exterior.coords)
                else:
                    # MultiPolygon - take the largest
                    largest = max(merged.geoms, key=lambda p: p.area)
                    merged_coords = list(largest.exterior.coords)
            else:
                merged_coords = None
        else:
            # Bounding box fallback
            all_lats = []
            all_lons = []
            for frame_data in self.all_frame_footprints:
                for lon, lat in frame_data['corners'][:-1]:
                    all_lats.append(lat)
                    all_lons.append(lon)

            if all_lats:
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                merged_coords = [
                    (min_lon, min_lat),
                    (max_lon, min_lat),
                    (max_lon, max_lat),
                    (min_lon, max_lat),
                    (min_lon, min_lat)
                ]
            else:
                merged_coords = None

        if not merged_coords:
            print("Could not create merged coverage")
            return None

        # Create KML
        kml = []
        kml.append('<?xml version="1.0" encoding="UTF-8"?>')
        kml.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
        kml.append('<Document>')
        kml.append(f'<name>Combined Thermal Coverage - Merged</name>')
        kml.append(f'<description>Total coverage from all directories</description>')

        kml.append('<Style id="totalCoverage">')
        kml.append('  <LineStyle>')
        kml.append('    <color>ff0000ff</color>')  # Red
        kml.append('    <width>3</width>')
        kml.append('  </LineStyle>')
        kml.append('  <PolyStyle>')
        kml.append('    <color>330000ff</color>')  # Semi-transparent red
        kml.append('  </PolyStyle>')
        kml.append('</Style>')

        kml.append('<Placemark>')
        kml.append(f'  <name>Total Survey Coverage</name>')

        desc = []
        desc.append(f'Directories processed: {len(self.directory_results)}')
        desc.append(f'Total frames: {len(self.all_frame_footprints)}')
        desc.append('---')
        for dir_path, dir_data in self.directory_results.items():
            desc.append(f'{Path(dir_path).name}: {dir_data["frames"]} frames')

        kml.append(f'  <description><![CDATA[{chr(10).join(desc)}]]></description>')
        kml.append('  <styleUrl>#totalCoverage</styleUrl>')

        kml.append('  <Polygon>')
        kml.append('    <extrude>0</extrude>')
        kml.append('    <altitudeMode>clampToGround</altitudeMode>')
        kml.append('    <outerBoundaryIs>')
        kml.append('      <LinearRing>')
        kml.append('        <coordinates>')

        for coord in merged_coords:
            if isinstance(coord, tuple):
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

        with open(output_file, 'w') as f:
            f.write('\n'.join(kml))

        print(f"Created combined merged KML: {output_file}")
        return output_file

    def run(self):
        """Process all directories and create combined KMLs"""
        print(f"\nSearching for thermal data in: {self.base_dir}")
        print("="*60)

        if not self.process_all_directories():
            print("No frames found to process!")
            return False

        # Generate combined KML files
        print("\nGenerating combined KML files...")
        frames_kml = self.create_combined_frames_kml()
        merged_kml = self.create_combined_merged_kml()

        print("\n" + "="*60)
        print("COMPLETED!")
        print("="*60)
        print(f"All frames: {frames_kml}")
        print(f"Merged coverage: {merged_kml}")
        print(f"\nTotal directories: {len(self.directory_results)}")
        print(f"Total frames: {len(self.all_frame_footprints)}")

        return True


def main():
    """Main function for multi-directory processing"""
    parser = argparse.ArgumentParser(
        description='Generate thermal coverage KML from multiple directories',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all XXXMEDIA subdirectories
  python generate_frame_footprints_multi.py --data "/path/to/survey" --search

  # Process with frame skipping
  python generate_frame_footprints_multi.py --data "/path/to/survey" --search --skip 10

Output:
  Creates combined KML files in sgd_output/:
  - *_all_frames.kml: All individual frames, grouped by directory
  - *_all_merged.kml: Total combined coverage area
        """
    )

    parser.add_argument('--data', required=True,
                       help='Base directory to search for data')
    parser.add_argument('--search', action='store_true',
                       help='Search for and process all XXXMEDIA subdirectories')
    parser.add_argument('--output', default='thermal_coverage',
                       help='Base name for output files')
    parser.add_argument('--skip', type=int, default=1,
                       help='Process every Nth frame')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')

    args = parser.parse_args()

    if args.search:
        # Multi-directory processing
        mapper = MultiDirectoryFrameMapper(
            base_dir=args.data,
            output_base=args.output,
            frame_skip=args.skip,
            verbose=not args.quiet
        )
        success = mapper.run()
    else:
        # Single directory (use base mapper)
        from generate_frame_footprints import ThermalFrameMapper
        mapper = ThermalFrameMapper(
            data_dir=args.data,
            output_base=args.output,
            frame_skip=args.skip,
            verbose=not args.quiet
        )
        success = mapper.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()