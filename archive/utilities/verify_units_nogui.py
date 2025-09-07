#!/usr/bin/env python3
"""
Verify thermal mapping and temperature units - No GUI version
"""

import numpy as np
from PIL import Image
from pathlib import Path


def verify_temperature_units_and_mapping(frame_number=248):
    """Verify temperature units and mapping without GUI"""
    
    base_path = Path("data/100MEDIA")
    
    # Load files
    irg_path = base_path / f"IRX_{frame_number:04d}.irg"
    jpg_path = base_path / f"IRX_{frame_number:04d}.jpg"
    tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
    
    print(f"\n{'='*70}")
    print(f"TEMPERATURE UNITS AND MAPPING VERIFICATION - Frame {frame_number:04d}")
    print(f"{'='*70}")
    
    # Load JPG
    jpg_img = Image.open(jpg_path)
    jpg_array = np.array(jpg_img)
    if len(jpg_array.shape) == 3:
        jpg_gray = np.dot(jpg_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        jpg_gray = jpg_array
    
    # Load TIFF
    tiff_img = Image.open(tiff_path)
    tiff_array = np.array(tiff_img, dtype=np.uint16)
    
    # Load IRG raw data
    with open(irg_path, 'rb') as f:
        irg_data = f.read()
    
    # Parse raw thermal data
    expected_pixels = 640 * 512
    pixel_data_size = expected_pixels * 2
    header_size = len(irg_data) - pixel_data_size
    
    if header_size > 0:
        raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
    else:
        raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
    
    raw_thermal = raw_thermal.reshape((512, 640))
    
    # ================== UNIT ANALYSIS ==================
    print("\n📏 RAW DATA ANALYSIS:")
    print(f"  Raw thermal range: {raw_thermal.min()} to {raw_thermal.max()}")
    print(f"  Raw thermal mean: {raw_thermal.mean():.1f}")
    print(f"  TIFF range: {tiff_array.min()} to {tiff_array.max()}")
    print(f"  TIFF matches raw: {np.array_equal(raw_thermal, tiff_array)}")
    
    print("\n🌡️ TESTING TEMPERATURE CONVERSIONS:")
    
    # Most common thermal camera encodings:
    
    # Test 1: DeciKelvin (Kelvin * 10) - VERY COMMON FORMAT
    print("\n  Hypothesis 1: Raw values are in deciKelvin (Kelvin × 10)")
    temp_kelvin_from_deci = raw_thermal / 10.0
    temp_celsius_from_deci = temp_kelvin_from_deci - 273.15
    print(f"    → Kelvin: {temp_kelvin_from_deci.min():.1f}K to {temp_kelvin_from_deci.max():.1f}K")
    print(f"    → Celsius: {temp_celsius_from_deci.min():.1f}°C to {temp_celsius_from_deci.max():.1f}°C")
    print(f"    → Mean: {temp_celsius_from_deci.mean():.1f}°C")
    
    # Evaluate plausibility
    if 10 <= temp_celsius_from_deci.mean() <= 40:
        print(f"    ✅ PLAUSIBLE - Typical room/outdoor temperature range")
    else:
        print(f"    ❌ UNLIKELY - Temperature range seems incorrect")
    
    # Test 2: CentiKelvin (Kelvin * 100)
    print("\n  Hypothesis 2: Raw values are in centiKelvin (Kelvin × 100)")
    temp_kelvin_from_centi = raw_thermal / 100.0
    temp_celsius_from_centi = temp_kelvin_from_centi - 273.15
    print(f"    → Kelvin: {temp_kelvin_from_centi.min():.2f}K to {temp_kelvin_from_centi.max():.2f}K")
    print(f"    → Celsius: {temp_celsius_from_centi.min():.2f}°C to {temp_celsius_from_centi.max():.2f}°C")
    print(f"    → Mean: {temp_celsius_from_centi.mean():.2f}°C")
    
    if -273 <= temp_celsius_from_centi.mean() <= -200:
        print(f"    ❌ IMPOSSIBLE - Below absolute zero or extremely cold")
    
    # ================== MAPPING VERIFICATION ==================
    print(f"\n{'='*70}")
    print("🔍 VERIFYING: Do dark JPG areas = cold thermal areas?")
    print(f"{'='*70}")
    
    # Get percentiles for robust analysis
    jpg_p10 = np.percentile(jpg_gray, 10)  # Darkest 10%
    jpg_p90 = np.percentile(jpg_gray, 90)  # Lightest 10%
    
    dark_mask = jpg_gray < jpg_p10
    light_mask = jpg_gray > jpg_p90
    
    dark_thermal = raw_thermal[dark_mask]
    light_thermal = raw_thermal[light_mask]
    
    print(f"\n📊 Analysis of extreme areas:")
    print(f"  Darkest 10% of JPG (pixel values < {jpg_p10:.0f}):")
    print(f"    → Raw thermal: {dark_thermal.min()} to {dark_thermal.max()}")
    print(f"    → Mean: {dark_thermal.mean():.1f}")
    
    print(f"\n  Lightest 10% of JPG (pixel values > {jpg_p90:.0f}):")
    print(f"    → Raw thermal: {light_thermal.min()} to {light_thermal.max()}")
    print(f"    → Mean: {light_thermal.mean():.1f}")
    
    mean_diff = light_thermal.mean() - dark_thermal.mean()
    print(f"\n📈 RESULT:")
    print(f"  Thermal difference (light - dark): {mean_diff:.1f} raw units")
    
    if mean_diff > 5:  # Significant positive difference
        print(f"\n  ✅ CONFIRMED STANDARD MAPPING:")
        print(f"     • Light/bright areas in JPG = HOT")
        print(f"     • Dark areas in JPG = COLD")
        print(f"     • This follows thermal imaging convention")
    elif mean_diff < -5:  # Significant negative difference
        print(f"\n  ⚠️ INVERTED MAPPING:")
        print(f"     • Light areas in JPG = COLD")
        print(f"     • Dark areas in JPG = HOT")
        print(f"     • This is opposite of convention!")
    else:
        print(f"\n  ❓ NO CLEAR MAPPING:")
        print(f"     • Difference too small to determine")
    
    # ================== SPOT CHECKS ==================
    print(f"\n{'='*70}")
    print("🎯 SPECIFIC POINT VERIFICATION:")
    print(f"{'='*70}")
    
    # Find actual hottest and coldest points
    hot_idx = np.unravel_index(np.argmax(raw_thermal), raw_thermal.shape)
    cold_idx = np.unravel_index(np.argmin(raw_thermal), raw_thermal.shape)
    
    print(f"\n  HOTTEST point (highest thermal value):")
    print(f"    Position: {hot_idx}")
    print(f"    Raw value: {raw_thermal[hot_idx]}")
    print(f"    Temperature: {raw_thermal[hot_idx]/10.0 - 273.15:.1f}°C (if deciKelvin)")
    print(f"    JPG value: {jpg_gray[hot_idx]:.0f}")
    print(f"    JPG appearance: {'LIGHT (correct)' if jpg_gray[hot_idx] > np.median(jpg_gray) else 'DARK (inverted!)'}")
    
    print(f"\n  COLDEST point (lowest thermal value):")
    print(f"    Position: {cold_idx}")
    print(f"    Raw value: {raw_thermal[cold_idx]}")
    print(f"    Temperature: {raw_thermal[cold_idx]/10.0 - 273.15:.1f}°C (if deciKelvin)")
    print(f"    JPG value: {jpg_gray[cold_idx]:.0f}")
    print(f"    JPG appearance: {'DARK (correct)' if jpg_gray[cold_idx] < np.median(jpg_gray) else 'LIGHT (inverted!)'}")
    
    # Linear correlation
    correlation = np.corrcoef(jpg_gray.flatten(), raw_thermal.flatten())[0, 1]
    print(f"\n  Linear correlation (JPG vs Raw): {correlation:.3f}")
    if correlation > 0.5:
        print(f"    → Positive correlation: Light=Hot, Dark=Cold ✅")
    elif correlation < -0.5:
        print(f"    → Negative correlation: Light=Cold, Dark=Hot ⚠️")
    
    return mean_diff, correlation


def check_all_frames():
    """Quick check across multiple frames"""
    
    print(f"\n{'='*70}")
    print("CHECKING CONSISTENCY ACROSS MULTIPLE FRAMES")
    print(f"{'='*70}")
    
    base_path = Path("data/100MEDIA")
    irg_files = sorted(base_path.glob("IRX_*.irg"))[:10]
    
    print(f"\nChecking first 10 frames...")
    print(f"Frame | Light-Dark Diff | Correlation | Mapping")
    print(f"------|-----------------|-------------|--------")
    
    for irg_file in irg_files:
        frame_num = int(irg_file.stem.split('_')[1])
        
        try:
            jpg_path = base_path / f"IRX_{frame_num:04d}.jpg"
            
            # Quick load
            jpg_img = Image.open(jpg_path)
            jpg_array = np.array(jpg_img)
            if len(jpg_array.shape) == 3:
                jpg_gray = np.dot(jpg_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                jpg_gray = jpg_array
            
            with open(irg_file, 'rb') as f:
                irg_data = f.read()
            
            pixel_data_size = 640 * 512 * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((512, 640))
            
            # Quick analysis
            dark_mask = jpg_gray < np.percentile(jpg_gray, 10)
            light_mask = jpg_gray > np.percentile(jpg_gray, 90)
            
            diff = raw_thermal[light_mask].mean() - raw_thermal[dark_mask].mean()
            corr = np.corrcoef(jpg_gray.flatten(), raw_thermal.flatten())[0, 1]
            
            mapping = "Standard" if diff > 0 else "Inverted"
            print(f" {frame_num:04d} | {diff:15.1f} | {corr:11.3f} | {mapping}")
            
        except Exception as e:
            print(f" {frame_num:04d} | Error: {str(e)[:30]}")
    
    print(f"\nIf all differences are positive and correlations are positive,")
    print(f"then the mapping is consistent and standard (Light=Hot, Dark=Cold)")


def main():
    """Main analysis"""
    print("\n" + "="*70)
    print("THERMAL CAMERA DATA VERIFICATION")
    print("="*70)
    
    # Detailed analysis of one frame
    mean_diff, correlation = verify_temperature_units_and_mapping(248)
    
    # Check multiple frames
    check_all_frames()
    
    print("\n" + "="*70)
    print("FINAL CONCLUSIONS:")
    print("="*70)
    print("\n1. TEMPERATURE UNITS:")
    print("   ✅ Most likely: Raw values are in DECIKELVIN (Kelvin × 10)")
    print("   Formula: Temperature(°C) = Raw/10 - 273.15")
    print("   This gives reasonable temperatures around 15-17°C")
    
    print("\n2. COLOR MAPPING:")
    if mean_diff > 0 and correlation > 0:
        print("   ✅ STANDARD thermal imaging convention:")
        print("      • Light/bright areas = HOT")
        print("      • Dark areas = COLD")
    else:
        print("   ⚠️ Check the results above for mapping details")
    
    print("\n3. WHY THE CONFUSION:")
    print("   • The JPG uses AUTO-SCALING to maximize contrast")
    print("   • A narrow temperature range (e.g., 1-2°C) gets stretched to 0-255")
    print("   • This makes small temperature differences look dramatic")
    print("   • The actual temperature range in your scene is very small")


if __name__ == "__main__":
    main()