#!/usr/bin/env python3
"""
Verify that SGD detections fall within thermal frame boundaries.
Checks alignment between thermal footprints and SGD locations.
"""

import sys
import json
import numpy as np
from pathlib import Path
from shapely.geometry import Point, Polygon
import xml.etree.ElementTree as ET


def parse_kml_polygons(kml_file):
    """Extract polygon coordinates from KML file"""
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # KML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    polygons = []

    # Find all Placemarks with Polygons
    for placemark in root.findall('.//kml:Placemark', ns):
        name_elem = placemark.find('.//kml:name', ns)
        name = name_elem.text if name_elem is not None else "Unknown"

        # Look for polygon coordinates
        coords_elem = placemark.find('.//kml:Polygon//kml:coordinates', ns)
        if coords_elem is None:
            # Try without namespace
            coords_elem = placemark.find('.//Polygon//coordinates')

        if coords_elem is not None and coords_elem.text:
            coords_text = coords_elem.text.strip()
            coords = []

            for line in coords_text.split('\n'):
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords.append((lon, lat))

            if coords:
                polygons.append({
                    'name': name,
                    'coordinates': coords,
                    'polygon': Polygon(coords) if len(coords) > 2 else None
                })

    return polygons


def check_sgd_in_frames(sgd_kml, frames_kml):
    """Check if SGD detections fall within thermal frame boundaries"""

    print("Verifying SGD-Frame Alignment")
    print("=" * 60)

    # Parse KML files
    print(f"\nReading SGD detections from: {sgd_kml}")
    sgd_polygons = parse_kml_polygons(sgd_kml)
    print(f"  Found {len(sgd_polygons)} SGD detections")

    print(f"\nReading thermal frames from: {frames_kml}")
    frame_polygons = parse_kml_polygons(frames_kml)
    print(f"  Found {len(frame_polygons)} thermal frames")

    if not sgd_polygons or not frame_polygons:
        print("\nError: No polygons found in one or both KML files")
        return

    # Check each SGD
    print("\n" + "-" * 60)
    print("Checking SGD locations against frame boundaries:\n")

    sgds_in_frames = 0
    sgds_outside = 0
    problem_sgds = []

    for sgd in sgd_polygons:
        if not sgd['polygon']:
            continue

        # Get SGD centroid
        sgd_centroid = sgd['polygon'].centroid
        sgd_point = Point(sgd_centroid.x, sgd_centroid.y)

        # Check against all frames
        found_in_frame = False
        for frame in frame_polygons:
            if frame['polygon'] and frame['polygon'].contains(sgd_point):
                found_in_frame = True
                break

        if found_in_frame:
            sgds_in_frames += 1
            status = "✓ INSIDE frame"
        else:
            sgds_outside += 1
            status = "✗ OUTSIDE all frames"
            problem_sgds.append({
                'name': sgd['name'],
                'centroid': (sgd_centroid.y, sgd_centroid.x)  # lat, lon
            })

        if sgds_outside <= 5 or not found_in_frame:  # Show first 5 and all problems
            print(f"  {sgd['name']}: {status}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  SGDs inside frames: {sgds_in_frames} ({100*sgds_in_frames/len(sgd_polygons):.1f}%)")
    print(f"  SGDs outside frames: {sgds_outside} ({100*sgds_outside/len(sgd_polygons):.1f}%)")

    if sgds_outside > 0:
        print("\n⚠️ WARNING: Some SGDs detected outside thermal frame boundaries!")
        print("This could indicate:")
        print("  1. Misalignment between thermal and RGB coordinate systems")
        print("  2. Incorrect FOV or rotation calculations")
        print("  3. GPS/heading synchronization issues")

        if problem_sgds:
            print(f"\nProblem SGDs (first 10):")
            for sgd in problem_sgds[:10]:
                print(f"  - {sgd['name']} at ({sgd['centroid'][0]:.6f}, {sgd['centroid'][1]:.6f})")
    else:
        print("\n✅ All SGDs are within thermal frame boundaries!")

    return sgds_in_frames, sgds_outside


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_sgd_frame_alignment.py <sgd_kml> <frames_kml>")
        print("\nExample:")
        print("  python verify_sgd_frame_alignment.py \\")
        print("    sgd_output/survey.kml \\")
        print("    sgd_output/thermal_frames.kml")
        sys.exit(1)

    sgd_kml = sys.argv[1]
    frames_kml = sys.argv[2]

    if not Path(sgd_kml).exists():
        print(f"Error: SGD KML not found: {sgd_kml}")
        sys.exit(1)

    if not Path(frames_kml).exists():
        print(f"Error: Frames KML not found: {frames_kml}")
        sys.exit(1)

    check_sgd_in_frames(sgd_kml, frames_kml)


if __name__ == "__main__":
    main()