#!/usr/bin/env python3
"""
Check altitude consistency between SGD detection and frame footprint generation.
Ensures both systems use the same altitude for ground coverage calculations.
"""

import sys
from pathlib import Path
import numpy as np
from generate_frame_footprints import ThermalFrameMapper
from sgd_georef_polygons import SGDPolygonGeoref


def check_altitude_consistency(data_dir, sample_frames=30):
    """
    Compare altitude extraction between frame footprints and SGD georeferencing.

    Args:
        data_dir: Directory with thermal/RGB data
        sample_frames: Number of frames to check
    """
    print("Altitude Consistency Check")
    print("=" * 60)
    print(f"Data directory: {data_dir}\n")

    # Initialize both systems
    frame_mapper = ThermalFrameMapper(data_dir, verbose=False)
    sgd_georef = SGDPolygonGeoref(base_path=str(data_dir))

    # Find RGB files
    rgb_files = sorted(Path(data_dir).glob("MAX_*.JPG"))[:sample_frames]

    if not rgb_files:
        print("No RGB files found!")
        return

    altitudes_frame = []
    altitudes_sgd = []
    mismatches = []

    print(f"Checking {len(rgb_files)} frames:\n")
    print(f"{'Frame':<15} {'Frame Mapper':<15} {'SGD Georef':<15} {'Match':<10}")
    print("-" * 55)

    for rgb_file in rgb_files:
        frame_num = int(rgb_file.stem.split('_')[1])

        # Get altitude from frame mapper
        gps_frame = frame_mapper.extract_gps_info(rgb_file)
        alt_frame = gps_frame['altitude'] if gps_frame else None

        # Get altitude from SGD georeferencer
        gps_sgd = sgd_georef.extract_gps(str(rgb_file), verbose=False)
        alt_sgd = gps_sgd.get('altitude', 400) if gps_sgd else None

        if alt_frame and alt_sgd:
            altitudes_frame.append(alt_frame)
            altitudes_sgd.append(alt_sgd)

            match = "✓" if abs(alt_frame - alt_sgd) < 1.0 else "✗"
            if match == "✗":
                mismatches.append(frame_num)

            print(f"Frame_{frame_num:04d}     {alt_frame:>10.1f} m    {alt_sgd:>10.1f} m    {match}")

    print("\n" + "=" * 60)
    print("SUMMARY:")

    if altitudes_frame and altitudes_sgd:
        avg_frame = np.mean(altitudes_frame)
        avg_sgd = np.mean(altitudes_sgd)
        std_frame = np.std(altitudes_frame)
        std_sgd = np.std(altitudes_sgd)

        print(f"\nFrame Mapper Statistics:")
        print(f"  Average altitude: {avg_frame:.1f} m")
        print(f"  Std deviation: {std_frame:.1f} m")
        print(f"  Range: {min(altitudes_frame):.1f} - {max(altitudes_frame):.1f} m")

        print(f"\nSGD Georef Statistics:")
        print(f"  Average altitude: {avg_sgd:.1f} m")
        print(f"  Std deviation: {std_sgd:.1f} m")
        print(f"  Range: {min(altitudes_sgd):.1f} - {max(altitudes_sgd):.1f} m")

        # Calculate ground coverage at different altitudes
        thermal_fov_deg = 45.0
        thermal_width_pixels = 640

        print(f"\nGround Coverage Impact (45° FOV, 640 pixels wide):")
        for alt in [300, 350, 400, 450, 500]:
            fov_rad = np.radians(thermal_fov_deg)
            ground_width = 2 * alt * np.tan(fov_rad / 2)
            resolution = ground_width / thermal_width_pixels
            print(f"  At {alt}m: {ground_width:.1f}m wide, {resolution:.2f}m/pixel")

        if mismatches:
            print(f"\n⚠️ WARNING: {len(mismatches)} frames have altitude mismatches!")
            print(f"   Frames with issues: {mismatches[:5]}...")
        else:
            print(f"\n✅ All altitudes match between systems!")

        # Check if altitudes are reasonable for survey
        if avg_frame < 200 or avg_frame > 600:
            print(f"\n⚠️ WARNING: Average altitude {avg_frame:.1f}m is unusual for thermal surveys")
            print("   Typical survey altitude is 300-500m")

    else:
        print("No valid altitude data found!")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_altitude_consistency.py <data_directory>")
        print("Example: python check_altitude_consistency.py /path/to/106MEDIA")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not Path(data_dir).exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    check_altitude_consistency(data_dir)


if __name__ == "__main__":
    main()