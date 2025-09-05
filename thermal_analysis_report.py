#!/usr/bin/env python3
"""
Thermal Data Analysis Report - Non-GUI version
Analyzes why JPG appearance doesn't match thermal values
"""

import numpy as np
from PIL import Image
from pathlib import Path


def analyze_thermal_discrepancy(frame_number=248):
    """Analyze why JPG visualization differs from raw thermal values"""
    
    base_path = Path("data/100MEDIA")
    
    # Load files
    irg_path = base_path / f"IRX_{frame_number:04d}.irg"
    jpg_path = base_path / f"IRX_{frame_number:04d}.jpg"
    tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
    
    print(f"\n{'='*60}")
    print(f"THERMAL DATA DISCREPANCY ANALYSIS - Frame {frame_number:04d}")
    print(f"{'='*60}")
    
    # Load JPG
    jpg_img = Image.open(jpg_path)
    jpg_array = np.array(jpg_img)
    if len(jpg_array.shape) == 3:
        jpg_gray = np.dot(jpg_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        jpg_gray = jpg_array
    
    # Load TIFF
    tiff_img = Image.open(tiff_path)
    tiff_array = np.array(tiff_img)
    
    # Load IRG raw data
    with open(irg_path, 'rb') as f:
        irg_data = f.read()
    
    # Parse raw thermal data (640x512 uint16)
    expected_pixels = 640 * 512
    pixel_data_size = expected_pixels * 2
    header_size = len(irg_data) - pixel_data_size
    
    if header_size > 0:
        raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
    else:
        raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
    
    raw_thermal = raw_thermal.reshape((512, 640))
    
    print("\n📊 DATA CHARACTERISTICS:")
    print(f"  • JPG range: {jpg_gray.min():.0f} - {jpg_gray.max():.0f} (8-bit)")
    print(f"  • Raw thermal range: {raw_thermal.min()} - {raw_thermal.max()} (16-bit)")
    print(f"  • TIFF range: {tiff_array.min()} - {tiff_array.max()}")
    
    # Analyze the relationship
    jpg_norm = (jpg_gray - jpg_gray.min()) / (jpg_gray.max() - jpg_gray.min())
    raw_norm = (raw_thermal - raw_thermal.min()) / (raw_thermal.max() - raw_thermal.min())
    
    correlation = np.corrcoef(jpg_norm.flatten(), raw_norm.flatten())[0, 1]
    
    print(f"\n🔍 CORRELATION ANALYSIS:")
    print(f"  • Direct correlation: {correlation:.4f}")
    
    # Check for inverted relationship
    correlation_inv = np.corrcoef(jpg_norm.flatten(), 1-raw_norm.flatten())[0, 1]
    print(f"  • Inverted correlation: {correlation_inv:.4f}")
    
    # Analyze specific regions
    print("\n🎯 REGIONAL ANALYSIS:")
    
    # Find hottest and coldest spots in raw data
    hot_idx = np.unravel_index(np.argmax(raw_thermal), raw_thermal.shape)
    cold_idx = np.unravel_index(np.argmin(raw_thermal), raw_thermal.shape)
    
    print(f"\n  Hottest spot (raw): Position {hot_idx}")
    print(f"    • Raw value: {raw_thermal[hot_idx]}")
    print(f"    • JPG value: {jpg_gray[hot_idx]:.0f}")
    print(f"    • Expected: Light/White in JPG")
    print(f"    • Actual: {'Light' if jpg_gray[hot_idx] > 128 else 'Dark'}")
    
    print(f"\n  Coldest spot (raw): Position {cold_idx}")
    print(f"    • Raw value: {raw_thermal[cold_idx]}")
    print(f"    • JPG value: {jpg_gray[cold_idx]:.0f}")
    print(f"    • Expected: Dark/Black in JPG")
    print(f"    • Actual: {'Light' if jpg_gray[cold_idx] > 128 else 'Dark'}")
    
    # Check dynamic range compression
    print("\n📈 DYNAMIC RANGE ANALYSIS:")
    raw_dynamic_range = raw_thermal.max() - raw_thermal.min()
    jpg_dynamic_range = jpg_gray.max() - jpg_gray.min()
    
    print(f"  • Raw dynamic range: {raw_dynamic_range} levels")
    print(f"  • JPG dynamic range: {jpg_dynamic_range:.0f} levels")
    print(f"  • Compression ratio: {raw_dynamic_range/jpg_dynamic_range:.1f}:1")
    
    # Histogram analysis
    print("\n📊 HISTOGRAM CHARACTERISTICS:")
    jpg_hist, _ = np.histogram(jpg_gray.flatten(), bins=256)
    raw_hist, _ = np.histogram(raw_thermal.flatten(), bins=256)
    
    # Check for histogram equalization
    jpg_hist_std = np.std(jpg_hist)
    ideal_flat_std = np.std(np.ones(256) * len(jpg_gray.flatten())/256)
    
    print(f"  • JPG histogram uniformity: {ideal_flat_std/jpg_hist_std:.3f}")
    print(f"    (1.0 = perfectly equalized, <0.5 = not equalized)")
    
    # Temperature estimation
    print("\n🌡️ TEMPERATURE ESTIMATION:")
    # Simple linear scaling (actual formula depends on camera calibration)
    temp_celsius = (raw_thermal.astype(float) - 27315) / 100.0
    
    print(f"  • Estimated temperature range: {temp_celsius.min():.1f}°C to {temp_celsius.max():.1f}°C")
    print(f"  • Mean temperature: {temp_celsius.mean():.1f}°C")
    print(f"  • Std deviation: {temp_celsius.std():.1f}°C")
    
    # KEY FINDINGS
    print(f"\n{'='*60}")
    print("🔑 KEY FINDINGS:")
    print(f"{'='*60}")
    
    reasons = []
    
    if correlation < 0.7 and correlation_inv > 0.7:
        reasons.append("✓ The JPG appears to have INVERTED values (hot=dark, cold=light)")
        reasons.append("  This is opposite of typical thermal imagery convention")
    elif correlation > 0.7:
        reasons.append("✓ The JPG follows standard thermal convention (hot=light, cold=dark)")
    else:
        reasons.append("✓ The JPG uses NON-LINEAR mapping from raw values")
    
    if jpg_hist_std < 1000:
        reasons.append("✓ Histogram equalization detected - enhances contrast but distorts values")
    
    if raw_dynamic_range/jpg_dynamic_range > 100:
        reasons.append(f"✓ Extreme dynamic range compression ({raw_dynamic_range/jpg_dynamic_range:.0f}:1)")
        reasons.append("  16-bit raw compressed to 8-bit JPG loses precision")
    
    reasons.append("✓ JPG is optimized for visualization, not measurement")
    reasons.append("✓ Automatic gain/level adjustment likely applied")
    
    for reason in reasons:
        print(f"  {reason}")
    
    print(f"\n{'='*60}")
    print("💡 RECOMMENDATIONS:")
    print(f"{'='*60}")
    print("  1. Use TIFF or IRG files for accurate temperature measurements")
    print("  2. JPG files should only be used for quick visual reference")
    print("  3. Develop custom visualization from raw data for analysis")
    print("  4. Apply known calibration coefficients to raw values for absolute temps")
    
    return {
        'correlation': correlation,
        'correlation_inverted': correlation_inv,
        'raw_range': (raw_thermal.min(), raw_thermal.max()),
        'jpg_range': (jpg_gray.min(), jpg_gray.max()),
        'temp_range': (temp_celsius.min(), temp_celsius.max())
    }


def check_multiple_frames():
    """Check pattern across multiple frames"""
    
    print(f"\n{'='*60}")
    print("MULTI-FRAME PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    base_path = Path("data/100MEDIA")
    
    # Find available frames
    irg_files = list(base_path.glob("IRX_*.irg"))
    frame_numbers = sorted([int(f.stem.split('_')[1]) for f in irg_files])[:5]  # Check first 5
    
    correlations = []
    correlations_inv = []
    
    for frame_num in frame_numbers:
        try:
            result = analyze_thermal_discrepancy(frame_num)
            correlations.append(result['correlation'])
            correlations_inv.append(result['correlation_inverted'])
        except Exception as e:
            print(f"Error analyzing frame {frame_num}: {e}")
    
    print(f"\n📊 PATTERN SUMMARY ACROSS FRAMES:")
    print(f"  • Average correlation (direct): {np.mean(correlations):.3f}")
    print(f"  • Average correlation (inverted): {np.mean(correlations_inv):.3f}")
    
    if np.mean(correlations_inv) > np.mean(correlations):
        print("\n  ⚠️ CONSISTENT PATTERN: JPG values appear INVERTED across all frames!")
        print("     Dark areas in JPG = Hot in thermal")
        print("     Light areas in JPG = Cold in thermal")


def main():
    """Main analysis"""
    print("\n" + "="*60)
    print("AUTEL 640T THERMAL DATA DISCREPANCY ANALYSIS")
    print("="*60)
    
    # Analyze single frame in detail
    analyze_thermal_discrepancy(248)
    
    # Check pattern across frames
    check_multiple_frames()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nThe viewer scripts (thermal_viewer.py) can show interactive visualizations")
    print("Use: python thermal_viewer.py")


if __name__ == "__main__":
    main()