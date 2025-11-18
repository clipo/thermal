#!/usr/bin/env python3
"""
Test script to verify that KML output includes frame paths and file information.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


def test_kml_file_paths(kml_file):
    """
    Test that a KML file contains the expected file path information.

    Args:
        kml_file: Path to the KML file to test
    """
    kml_path = Path(kml_file)

    if not kml_path.exists():
        print(f"❌ KML file not found: {kml_file}")
        return False

    print(f"Testing KML file: {kml_file}")
    print("-" * 60)

    # Parse the KML file
    tree = ET.parse(kml_file)
    root = tree.getroot()

    # Define KML namespace
    ns = {'kml': 'http://www.opengis.net/kml/2.2'}

    # Find all Placemarks
    placemarks = root.findall('.//kml:Placemark', ns)

    if not placemarks:
        # Try without namespace
        placemarks = root.findall('.//Placemark')

    print(f"Found {len(placemarks)} SGD placemarks")

    # Check each placemark for file path information
    has_paths = 0
    missing_paths = 0

    for i, placemark in enumerate(placemarks):
        # Get name
        name_elem = placemark.find('.//kml:name', ns) or placemark.find('.//name')
        name = name_elem.text if name_elem is not None else f"Placemark {i+1}"

        # Get description
        desc_elem = placemark.find('.//kml:description', ns) or placemark.find('.//description')

        if desc_elem is not None and desc_elem.text:
            description = desc_elem.text

            # Check for file path information
            has_rgb = 'RGB:' in description or 'rgb_path' in description
            has_thermal = 'Thermal:' in description or 'thermal_path' in description
            has_folder = 'Folder:' in description or 'data_folder' in description
            has_frame = 'Frame:' in description
            has_datetime = 'Date/Time:' in description

            if has_rgb and has_thermal:
                has_paths += 1
                print(f"✓ {name}: Contains file paths")

                # Extract and display the paths
                for line in description.split('\n'):
                    if 'RGB:' in line:
                        print(f"    {line.strip()}")
                    elif 'Thermal:' in line:
                        print(f"    {line.strip()}")
                    elif 'Folder:' in line:
                        print(f"    {line.strip()}")
            else:
                missing_paths += 1
                print(f"❌ {name}: Missing file paths")
                if not has_rgb:
                    print("    - Missing RGB path")
                if not has_thermal:
                    print("    - Missing thermal path")

    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Total SGDs: {len(placemarks)}")
    print(f"  With file paths: {has_paths}")
    print(f"  Missing paths: {missing_paths}")

    if has_paths == len(placemarks):
        print("✅ All SGDs have file path information!")
        return True
    elif has_paths > 0:
        print("⚠️  Some SGDs have file paths, but not all")
        return False
    else:
        print("❌ No SGDs have file path information")
        return False


def main():
    """
    Main function to test KML files.
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_kml_paths.py <kml_file>")
        print("Example: python test_kml_paths.py sgd_output/survey_sgd.kml")
        sys.exit(1)

    kml_file = sys.argv[1]
    success = test_kml_file_paths(kml_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()