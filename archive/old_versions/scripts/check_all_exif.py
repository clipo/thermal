#!/usr/bin/env python3
"""Deep dive into EXIF data to find all GPS and orientation information"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from pathlib import Path
import json

def get_all_exif_data(image_path):
    """Extract ALL EXIF data from image"""
    print(f"\nAnalyzing: {image_path}")
    print("=" * 60)
    
    try:
        img = Image.open(image_path)
        exifdata = img._getexif()
        
        if not exifdata:
            print("No EXIF data found")
            return None
        
        all_data = {}
        gps_data = {}
        
        # Process all EXIF tags
        for tag_id, value in exifdata.items():
            tag_name = TAGS.get(tag_id, f"Unknown_{tag_id}")
            
            # Handle GPS data specially
            if tag_name == 'GPSInfo':
                print("\nGPS Information Found:")
                print("-" * 40)
                for gps_tag_id, gps_value in value.items():
                    gps_tag_name = GPSTAGS.get(gps_tag_id, f"Unknown_GPS_{gps_tag_id}")
                    gps_data[gps_tag_name] = gps_value
                    
                    # Pretty print GPS data
                    if isinstance(gps_value, bytes):
                        print(f"  {gps_tag_name}: {gps_value.decode('utf-8', errors='ignore')}")
                    elif isinstance(gps_value, tuple) and len(gps_value) == 3:
                        # Likely coordinates
                        decimal = gps_value[0] + gps_value[1]/60 + gps_value[2]/3600
                        print(f"  {gps_tag_name}: {gps_value} = {decimal:.6f}°")
                    else:
                        print(f"  {gps_tag_name}: {gps_value}")
            else:
                all_data[tag_name] = value
                
                # Print interesting orientation-related tags
                if any(term in tag_name.lower() for term in ['orient', 'rotat', 'direct', 'head', 'bearing', 'azimuth', 'yaw']):
                    print(f"Orientation-related: {tag_name} = {value}")
        
        # Look for XMP data (sometimes heading is there)
        if hasattr(img, 'info'):
            for key, value in img.info.items():
                if any(term in str(key).lower() for term in ['xmp', 'xml', 'head', 'yaw', 'orient']):
                    print(f"Image info: {key} = {str(value)[:100]}...")
        
        # Check for specific Autel/DJI tags
        print("\n" + "-" * 40)
        print("Checking for drone-specific tags:")
        
        # Common drone EXIF tags
        drone_tags = {
            0x0001: "InteropIndex",
            0x0002: "InteropVersion", 
            0x0011: "ProcessingSoftware",
            0x0012: "Software",
            0x010F: "Make",
            0x0110: "Model",
            0x0112: "Orientation",
            0x0131: "Software",
            0x0132: "DateTime",
            0x829A: "ExposureTime",
            0x829D: "FNumber",
            0x8827: "ISOSpeedRatings",
            0x9000: "ExifVersion",
            0x9003: "DateTimeOriginal",
            0x9004: "DateTimeDigitized",
            0xA002: "PixelXDimension",
            0xA003: "PixelYDimension",
            0xA430: "CameraOwnerName",
            0xA431: "BodySerialNumber",
            0xA432: "LensSpecification",
            0xA433: "LensMake",
            0xA434: "LensModel",
            0xA435: "LensSerialNumber"
        }
        
        for tag_id, tag_desc in drone_tags.items():
            if tag_id in exifdata:
                value = exifdata[tag_id]
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                print(f"  {tag_desc} ({hex(tag_id)}): {value}")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(f"  Total EXIF tags: {len(all_data)}")
        print(f"  GPS tags found: {len(gps_data)}")
        
        # Check what's missing
        important_gps = ['GPSImgDirection', 'GPSImgDirectionRef', 'GPSTrack', 'GPSDestBearing']
        found = [tag for tag in important_gps if tag in gps_data]
        missing = [tag for tag in important_gps if tag not in gps_data]
        
        if found:
            print(f"  ✓ Found orientation tags: {', '.join(found)}")
        if missing:
            print(f"  ✗ Missing orientation tags: {', '.join(missing)}")
        
        return all_data, gps_data
        
    except Exception as e:
        print(f"Error reading EXIF: {e}")
        return None, None

def check_irg_metadata(irg_path):
    """Check if IRG thermal files have embedded metadata"""
    print(f"\nChecking IRG file: {irg_path}")
    print("-" * 40)
    
    try:
        with open(irg_path, 'rb') as f:
            data = f.read(1000)  # Read first 1KB
            
        # Look for text strings that might be metadata
        printable = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data)
        
        # Look for GPS patterns
        if 'GPS' in printable or 'gps' in printable:
            print("Found GPS reference in IRG header")
            
        # Look for heading/yaw/bearing
        if any(term in printable.lower() for term in ['head', 'yaw', 'bearing', 'azimuth']):
            print("Found orientation reference in IRG header")
            
        # Show readable portion
        print("First 200 bytes (readable):")
        print(printable[:200])
        
    except Exception as e:
        print(f"Error reading IRG: {e}")

def main():
    print("COMPREHENSIVE EXIF ANALYSIS FOR AUTEL 640T IMAGES")
    print("=" * 60)
    
    # Check RGB image
    rgb_path = Path("data/100MEDIA/MAX_0248.JPG")
    if rgb_path.exists():
        all_exif, gps_exif = get_all_exif_data(rgb_path)
        
        # Save to JSON for analysis
        if all_exif:
            output = {
                "file": str(rgb_path),
                "exif": {k: str(v)[:100] for k, v in all_exif.items()},
                "gps": {k: str(v) for k, v in gps_exif.items()}
            }
            
            with open("exif_analysis.json", "w") as f:
                json.dump(output, f, indent=2)
            print("\nFull EXIF data saved to exif_analysis.json")
    else:
        print(f"RGB image not found: {rgb_path}")
    
    # Check thermal IRG
    irg_path = Path("data/100MEDIA/IRX_0248.irg")
    if irg_path.exists():
        check_irg_metadata(irg_path)
    
    # Check IRX JPEG (processed thermal)
    irx_jpg_path = Path("data/100MEDIA/IRX_0248.jpg")
    if irx_jpg_path.exists():
        print("\n" + "=" * 60)
        print("Checking IRX JPEG (processed thermal):")
        all_exif, gps_exif = get_all_exif_data(irx_jpg_path)
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 40)
    print("If heading data is missing:")
    print("1. Check drone GPS settings - enable 'Record Image Direction'")
    print("2. Update drone firmware for better EXIF support")
    print("3. Some Autel models store heading in flight logs, not EXIF")
    print("4. Consider using flight planning software that records heading")
    print("\nNote: Even without heading, the system will work but may be less accurate")
    print("when the drone is not facing north.")

if __name__ == "__main__":
    main()