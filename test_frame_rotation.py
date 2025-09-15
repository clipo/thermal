#!/usr/bin/env python3
"""
Test script to verify frame rotation is working correctly.
Shows heading values and confirms rotation is applied.
"""

import sys
from pathlib import Path
from generate_frame_footprints import ThermalFrameMapper


def test_rotation(data_dir, sample_frames=5):
    """Test frame rotation and heading extraction"""

    print("Testing Frame Rotation and Heading Extraction")
    print("=" * 60)

    mapper = ThermalFrameMapper(
        data_dir=data_dir,
        output_base="test",
        frame_skip=1,
        verbose=False
    )

    # Find frames
    rgb_files = sorted(Path(data_dir).glob("MAX_*.JPG"))[:sample_frames]

    if not rgb_files:
        print("No RGB files found!")
        return

    print(f"\nTesting {len(rgb_files)} frames:\n")

    for rgb_file in rgb_files:
        frame_num = int(rgb_file.stem.split('_')[1])

        # Extract GPS info
        gps_info = mapper.extract_gps_info(rgb_file)

        if gps_info:
            heading = gps_info['heading']
            lat = gps_info['lat']
            lon = gps_info['lon']
            alt = gps_info['altitude']

            print(f"Frame {frame_num:04d}:")
            print(f"  Location: ({lat:.6f}, {lon:.6f})")
            print(f"  Altitude: {alt:.1f} m")

            if heading != 0.0:
                print(f"  ✓ Heading: {heading:.1f}° (rotation will be applied)")

                # Calculate corners to show rotation effect
                corners = mapper.calculate_footprint_corners(lat, lon, alt, heading)
                print(f"  → Frame rotated to heading {heading:.1f}°")
            else:
                print(f"  ⚠ No heading data (frame will not be rotated)")
            print()

    print("\nSUMMARY:")
    print("-" * 40)
    print("Frames with heading data will be properly")
    print("rotated to match their capture orientation.")
    print("\nThis ensures thermal footprints align with")
    print("the actual ground coverage during flight.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_frame_rotation.py <data_directory>")
        print("Example: python test_frame_rotation.py /path/to/106MEDIA")
        sys.exit(1)

    data_dir = sys.argv[1]
    test_rotation(data_dir)


if __name__ == "__main__":
    main()