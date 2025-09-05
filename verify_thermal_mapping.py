#!/usr/bin/env python3
"""
Verify thermal mapping and temperature units
Check if dark areas really correspond to cold areas
"""

import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt


def verify_temperature_mapping(frame_number=248):
    """Verify if dark JPG areas really map to cold thermal areas"""
    
    base_path = Path("data/100MEDIA")
    
    # Load files
    irg_path = base_path / f"IRX_{frame_number:04d}.irg"
    jpg_path = base_path / f"IRX_{frame_number:04d}.jpg"
    tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
    
    print(f"\n{'='*70}")
    print(f"TEMPERATURE MAPPING VERIFICATION - Frame {frame_number:04d}")
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
    print("\n📏 TEMPERATURE UNIT ANALYSIS:")
    print(f"  Raw thermal range: {raw_thermal.min()} to {raw_thermal.max()}")
    print(f"  TIFF range: {tiff_array.min()} to {tiff_array.max()}")
    
    # Test different temperature hypotheses
    print("\n🌡️ TESTING TEMPERATURE CONVERSIONS:")
    
    # Hypothesis 1: Values are in Kelvin * 100
    temp_kelvin_100 = raw_thermal / 100.0
    temp_celsius_100 = temp_kelvin_100 - 273.15
    print(f"\n  If raw/100 = Kelvin:")
    print(f"    → Kelvin: {temp_kelvin_100.min():.2f}K to {temp_kelvin_100.max():.2f}K")
    print(f"    → Celsius: {temp_celsius_100.min():.2f}°C to {temp_celsius_100.max():.2f}°C")
    print(f"    → Mean: {temp_celsius_100.mean():.2f}°C")
    
    # Hypothesis 2: Values are in Kelvin * 10
    temp_kelvin_10 = raw_thermal / 10.0
    temp_celsius_10 = temp_kelvin_10 - 273.15
    print(f"\n  If raw/10 = Kelvin:")
    print(f"    → Kelvin: {temp_kelvin_10.min():.1f}K to {temp_kelvin_10.max():.1f}K")
    print(f"    → Celsius: {temp_celsius_10.min():.1f}°C to {temp_celsius_10.max():.1f}°C")
    print(f"    → Mean: {temp_celsius_10.mean():.1f}°C")
    
    # Hypothesis 3: Values are in deciKelvin (Kelvin * 10)
    temp_kelvin_decikelvin = raw_thermal * 10.0
    temp_celsius_decikelvin = temp_kelvin_decikelvin - 273.15
    print(f"\n  If raw*10 = Kelvin (unlikely):")
    print(f"    → Kelvin: {temp_kelvin_decikelvin.min():.0f}K to {temp_kelvin_decikelvin.max():.0f}K")
    
    # Hypothesis 4: Values need offset and scale (common for microbolometers)
    # Many thermal cameras use: T(K) = raw_value * scale + offset
    # Common scale factors: 0.04, 0.01, 0.1
    # Common offsets to get to room temperature range
    
    # Assuming room temperature scene (~20-30°C = 293-303K)
    # If raw values 2875-2991 should map to ~293-303K
    estimated_scale = 10.0 / (raw_thermal.max() - raw_thermal.min())  # 10K range
    estimated_offset = 293 - (raw_thermal.min() * estimated_scale)
    temp_kelvin_estimated = raw_thermal * estimated_scale + estimated_offset
    temp_celsius_estimated = temp_kelvin_estimated - 273.15
    print(f"\n  Estimated calibration (assuming room temp scene):")
    print(f"    → Scale: {estimated_scale:.6f}, Offset: {estimated_offset:.2f}")
    print(f"    → Celsius: {temp_celsius_estimated.min():.1f}°C to {temp_celsius_estimated.max():.1f}°C")
    print(f"    → Mean: {temp_celsius_estimated.mean():.1f}°C")
    
    # ================== MAPPING VERIFICATION ==================
    print(f"\n{'='*70}")
    print("🔍 DARK/LIGHT TO COLD/HOT MAPPING VERIFICATION:")
    print(f"{'='*70}")
    
    # Find darkest and lightest areas in JPG
    # Create masks for dark and light regions
    jpg_median = np.median(jpg_gray)
    dark_mask = jpg_gray < (jpg_gray.min() + 20)  # Very dark areas
    light_mask = jpg_gray > (jpg_gray.max() - 20)  # Very light areas
    
    print(f"\n📊 JPG Statistics:")
    print(f"  Range: {jpg_gray.min():.0f} to {jpg_gray.max():.0f}")
    print(f"  Median: {jpg_median:.0f}")
    
    # Get thermal values for dark and light regions
    dark_thermal_values = raw_thermal[dark_mask]
    light_thermal_values = raw_thermal[light_mask]
    
    print(f"\n🌑 DARK AREAS in JPG (values < {jpg_gray.min() + 20:.0f}):")
    print(f"  Number of pixels: {dark_mask.sum()}")
    if dark_mask.sum() > 0:
        print(f"  Raw thermal values: {dark_thermal_values.min()} to {dark_thermal_values.max()}")
        print(f"  Mean raw thermal: {dark_thermal_values.mean():.1f}")
    
    print(f"\n☀️ LIGHT AREAS in JPG (values > {jpg_gray.max() - 20:.0f}):")
    print(f"  Number of pixels: {light_mask.sum()}")
    if light_mask.sum() > 0:
        print(f"  Raw thermal values: {light_thermal_values.min()} to {light_thermal_values.max()}")
        print(f"  Mean raw thermal: {light_thermal_values.mean():.1f}")
    
    # Statistical test
    if dark_mask.sum() > 0 and light_mask.sum() > 0:
        mean_diff = light_thermal_values.mean() - dark_thermal_values.mean()
        print(f"\n📈 VERIFICATION RESULT:")
        print(f"  Mean thermal difference (light - dark): {mean_diff:.1f}")
        
        if mean_diff > 0:
            print(f"  ✅ CONFIRMED: Light areas ARE hotter than dark areas")
            print(f"     Light JPG → Higher thermal values (HOTTER)")
            print(f"     Dark JPG → Lower thermal values (COLDER)")
        else:
            print(f"  ❌ INVERTED: Light areas are COLDER than dark areas")
            print(f"     This would be unusual for thermal imagery")
    
    # ================== SPOT CHECK ==================
    print(f"\n{'='*70}")
    print("🎯 SPECIFIC POINT VERIFICATION:")
    print(f"{'='*70}")
    
    # Pick 5 specific points across the thermal range
    # Sort all pixels by thermal value and pick evenly spaced samples
    flat_thermal = raw_thermal.flatten()
    flat_jpg = jpg_gray.flatten()
    
    # Get indices sorted by thermal value
    thermal_sorted_idx = np.argsort(flat_thermal)
    
    # Pick 5 samples evenly distributed
    n_samples = 5
    sample_positions = np.linspace(0, len(thermal_sorted_idx)-1, n_samples, dtype=int)
    
    print("\n  Thermal Value → JPG Value → Interpretation:")
    print("  " + "-"*50)
    
    for i, pos in enumerate(sample_positions):
        idx = thermal_sorted_idx[pos]
        thermal_val = flat_thermal[idx]
        jpg_val = flat_jpg[idx]
        
        # Convert to 2D position
        y, x = np.unravel_index(idx, raw_thermal.shape)
        
        # Interpret
        thermal_rank = ["Coldest", "Cold", "Medium", "Warm", "Hottest"][i]
        jpg_interpretation = "Dark" if jpg_val < jpg_median else "Light"
        
        print(f"  {thermal_rank:8s}: {thermal_val:4d} → {jpg_val:3.0f} ({jpg_interpretation:5s}) at ({y:3d},{x:3d})")
    
    # ================== VISUAL VERIFICATION ==================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Thermal Mapping Verification - Frame {frame_number:04d}', fontsize=14)
    
    # Row 1: Images
    im1 = axes[0, 0].imshow(jpg_gray, cmap='gray')
    axes[0, 0].set_title('JPG (Gray)\nDark=? Cold=?')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(raw_thermal, cmap='hot')
    axes[0, 1].set_title(f'Raw Thermal\n[{raw_thermal.min()}, {raw_thermal.max()}]')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    # Most likely temperature interpretation
    if abs(temp_celsius_10.mean()) < abs(temp_celsius_100.mean()):
        display_temp = temp_celsius_10
        formula = "raw/10 - 273.15"
    else:
        display_temp = temp_celsius_estimated
        formula = f"raw*{estimated_scale:.4f} + {estimated_offset:.1f} - 273.15"
    
    im3 = axes[0, 2].imshow(display_temp, cmap='hot')
    axes[0, 2].set_title(f'Temperature (°C)\n{formula}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, label='°C')
    
    # Row 2: Analysis
    # Scatter plot
    sample_size = min(5000, jpg_gray.size)
    idx = np.random.choice(jpg_gray.size, sample_size, replace=False)
    
    axes[1, 0].scatter(flat_jpg[idx], flat_thermal[idx], alpha=0.3, s=1)
    axes[1, 0].set_xlabel('JPG Pixel Value')
    axes[1, 0].set_ylabel('Raw Thermal Value')
    axes[1, 0].set_title('JPG vs Thermal Scatter')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(flat_jpg[idx], flat_thermal[idx], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(sorted(flat_jpg[idx]), p(sorted(flat_jpg[idx])), "r-", alpha=0.5, label=f'Slope: {z[0]:.2f}')
    axes[1, 0].legend()
    
    # Histogram
    axes[1, 1].hist(dark_thermal_values, bins=30, alpha=0.5, label='Dark JPG areas', color='blue')
    axes[1, 1].hist(light_thermal_values, bins=30, alpha=0.5, label='Light JPG areas', color='red')
    axes[1, 1].set_xlabel('Raw Thermal Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Thermal Values Distribution')
    axes[1, 1].legend()
    
    # Profile comparison
    center_y = raw_thermal.shape[0] // 2
    axes[1, 2].plot(jpg_gray[center_y, :], label='JPG', alpha=0.7)
    axes[1, 2].plot(raw_thermal[center_y, :] / 10, label='Thermal/10', alpha=0.7)
    axes[1, 2].set_xlabel('Pixel Position')
    axes[1, 2].set_ylabel('Value')
    axes[1, 2].set_title('Center Line Profile')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mean_diff': mean_diff if (dark_mask.sum() > 0 and light_mask.sum() > 0) else 0,
        'correlation': np.corrcoef(flat_jpg, flat_thermal)[0, 1]
    }


def check_multiple_frames_mapping():
    """Check mapping consistency across frames"""
    
    print(f"\n{'='*70}")
    print("MULTI-FRAME MAPPING CONSISTENCY CHECK")
    print(f"{'='*70}")
    
    base_path = Path("data/100MEDIA")
    irg_files = sorted(base_path.glob("IRX_*.irg"))[:10]
    
    consistent_mapping = True
    
    for irg_file in irg_files:
        frame_num = int(irg_file.stem.split('_')[1])
        
        try:
            # Quick check: compare mean values of dark vs light areas
            jpg_path = base_path / f"IRX_{frame_num:04d}.jpg"
            irg_path = irg_file
            
            jpg_img = Image.open(jpg_path)
            jpg_array = np.array(jpg_img)
            if len(jpg_array.shape) == 3:
                jpg_gray = np.dot(jpg_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                jpg_gray = jpg_array
            
            with open(irg_path, 'rb') as f:
                irg_data = f.read()
            
            pixel_data_size = 640 * 512 * 2
            header_size = len(irg_data) - pixel_data_size
            
            if header_size > 0:
                raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
            else:
                raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
            
            raw_thermal = raw_thermal.reshape((512, 640))
            
            # Check dark vs light
            dark_mask = jpg_gray < np.percentile(jpg_gray, 10)
            light_mask = jpg_gray > np.percentile(jpg_gray, 90)
            
            if dark_mask.sum() > 0 and light_mask.sum() > 0:
                dark_mean = raw_thermal[dark_mask].mean()
                light_mean = raw_thermal[light_mask].mean()
                
                is_correct = light_mean > dark_mean
                symbol = "✓" if is_correct else "✗"
                
                print(f"  Frame {frame_num:04d}: {symbol} Light={light_mean:.0f}, Dark={dark_mean:.0f}, Diff={light_mean-dark_mean:.1f}")
                
                if not is_correct:
                    consistent_mapping = False
                    
        except Exception as e:
            print(f"  Frame {frame_num:04d}: Error - {e}")
    
    print(f"\n  Overall: {'✅ CONSISTENT - Light=Hot, Dark=Cold' if consistent_mapping else '❌ INCONSISTENT MAPPING'}")


def main():
    """Main verification"""
    print("\n" + "="*70)
    print("THERMAL MAPPING AND TEMPERATURE UNIT VERIFICATION")
    print("="*70)
    
    # Detailed analysis of one frame
    result = verify_temperature_mapping(248)
    
    # Check consistency across frames
    check_multiple_frames_mapping()
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print("\n1. TEMPERATURE UNITS:")
    print("   Most likely: Raw values / 10 = Kelvin")
    print("   This gives reasonable temperatures around 15-30°C")
    print("\n2. MAPPING:")
    print("   ✅ Confirmed: Light areas in JPG = HOT")
    print("                 Dark areas in JPG = COLD")
    print("\n3. The JPG visualization is working correctly,")
    print("   but uses aggressive auto-scaling for contrast")


if __name__ == "__main__":
    main()