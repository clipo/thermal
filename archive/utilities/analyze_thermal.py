#!/usr/bin/env python3
"""
Thermal Data Analysis Script
Analyzes discrepancies between JPG visualization and raw thermal values
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import struct


def analyze_single_frame(frame_number=248):
    """Analyze a single frame to understand thermal data mapping"""
    
    base_path = Path("data/100MEDIA")
    
    # Load all files for this frame
    irg_path = base_path / f"IRX_{frame_number:04d}.irg"
    jpg_path = base_path / f"IRX_{frame_number:04d}.jpg"
    tiff_path = base_path / f"IRX_{frame_number:04d}.TIFF"
    
    print(f"\nAnalyzing Frame {frame_number:04d}")
    print("=" * 50)
    
    # 1. Load and analyze JPG
    print("\n1. JPG Analysis:")
    jpg_img = Image.open(jpg_path)
    jpg_array = np.array(jpg_img)
    print(f"   - Shape: {jpg_array.shape}")
    print(f"   - Data type: {jpg_array.dtype}")
    print(f"   - Value range: [{jpg_array.min()}, {jpg_array.max()}]")
    
    # Convert to grayscale if RGB
    if len(jpg_array.shape) == 3:
        jpg_gray = np.dot(jpg_array[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        jpg_gray = jpg_array
    
    # 2. Load and analyze TIFF
    print("\n2. TIFF Analysis:")
    tiff_img = Image.open(tiff_path)
    tiff_array = np.array(tiff_img)
    print(f"   - Mode: {tiff_img.mode}")
    print(f"   - Shape: {tiff_array.shape}")
    print(f"   - Data type: {tiff_array.dtype}")
    print(f"   - Value range: [{tiff_array.min()}, {tiff_array.max()}]")
    
    # 3. Load and analyze IRG (raw data)
    print("\n3. IRG Raw Data Analysis:")
    with open(irg_path, 'rb') as f:
        irg_data = f.read()
    
    file_size = len(irg_data)
    print(f"   - File size: {file_size} bytes")
    
    # Expected pixels for 640x512 resolution
    expected_pixels = 640 * 512  # 327,680 pixels
    pixel_data_size = expected_pixels * 2  # 655,360 bytes for uint16
    
    # Calculate header size
    header_size = file_size - pixel_data_size
    print(f"   - Expected pixel data: {pixel_data_size} bytes")
    print(f"   - Calculated header size: {header_size} bytes")
    
    # Extract raw thermal data (assuming uint16 format)
    if header_size > 0:
        raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
    else:
        raw_thermal = np.frombuffer(irg_data[:pixel_data_size], dtype=np.uint16)
    
    raw_thermal = raw_thermal.reshape((512, 640))
    print(f"   - Raw shape: {raw_thermal.shape}")
    print(f"   - Raw value range: [{raw_thermal.min()}, {raw_thermal.max()}]")
    
    # 4. Temperature conversion (simplified)
    print("\n4. Temperature Conversion:")
    # Simple linear conversion (this would need proper calibration constants)
    # Typical range for uncooled microbolometer
    temp_celsius = (raw_thermal.astype(float) - 27315) / 100.0  # Rough conversion
    print(f"   - Temperature range: [{temp_celsius.min():.1f}°C, {temp_celsius.max():.1f}°C]")
    
    # 5. Analyze the mapping between JPG and raw values
    print("\n5. JPG vs Raw Value Mapping:")
    
    # Normalize both to 0-1 range
    jpg_norm = (jpg_gray - jpg_gray.min()) / (jpg_gray.max() - jpg_gray.min())
    raw_norm = (raw_thermal - raw_thermal.min()) / (raw_thermal.max() - raw_thermal.min())
    
    # Calculate correlation
    correlation = np.corrcoef(jpg_norm.flatten(), raw_norm.flatten())[0, 1]
    print(f"   - Correlation coefficient: {correlation:.4f}")
    
    # Find the transformation
    # Sample some points to understand the mapping
    sample_points = 1000
    indices = np.random.choice(jpg_norm.size, sample_points, replace=False)
    jpg_samples = jpg_norm.flatten()[indices]
    raw_samples = raw_norm.flatten()[indices]
    
    # Fit a polynomial to understand the mapping
    coeffs = np.polyfit(raw_samples, jpg_samples, 2)
    print(f"   - Mapping polynomial coefficients: {coeffs}")
    
    # 6. Identify specific discrepancies
    print("\n6. Discrepancy Analysis:")
    
    # Find areas with largest differences
    difference = np.abs(jpg_norm - raw_norm)
    max_diff_idx = np.unravel_index(np.argmax(difference), difference.shape)
    print(f"   - Maximum difference location: {max_diff_idx}")
    print(f"   - Maximum difference value: {difference[max_diff_idx]:.4f}")
    
    # Check if JPG uses histogram equalization or other enhancement
    jpg_hist, _ = np.histogram(jpg_gray.flatten(), bins=256)
    raw_hist, _ = np.histogram(raw_norm.flatten() * 255, bins=256)
    
    # Calculate histogram similarity
    hist_correlation = np.corrcoef(jpg_hist, raw_hist)[0, 1]
    print(f"   - Histogram correlation: {hist_correlation:.4f}")
    
    # 7. Possible reasons for discrepancies
    print("\n7. Possible Reasons for Discrepancies:")
    print("   a) JPG may use automatic gain/level adjustment")
    print("   b) JPG might apply histogram equalization for better contrast")
    print("   c) JPG could use a non-linear mapping (gamma correction)")
    print("   d) Color palette/LUT might be applied before JPG conversion")
    print("   e) Dynamic range compression for 8-bit JPG from 16-bit raw")
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Thermal Data Analysis - Frame {frame_number:04d}', fontsize=14)
    
    # Row 1: Original data
    im1 = axes[0, 0].imshow(jpg_gray, cmap='gray')
    axes[0, 0].set_title('JPG (Grayscale)')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
    
    im2 = axes[0, 1].imshow(raw_thermal, cmap='hot')
    axes[0, 1].set_title(f'Raw Thermal\n[{raw_thermal.min()}, {raw_thermal.max()}]')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
    
    im3 = axes[0, 2].imshow(tiff_array, cmap='hot')
    axes[0, 2].set_title(f'TIFF\n[{tiff_array.min()}, {tiff_array.max()}]')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)
    
    im4 = axes[0, 3].imshow(temp_celsius, cmap='hot')
    axes[0, 3].set_title(f'Temperature (°C)\n[{temp_celsius.min():.1f}, {temp_celsius.max():.1f}]')
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Analysis
    axes[1, 0].hist(jpg_gray.flatten(), bins=50, alpha=0.7, label='JPG', color='blue')
    axes[1, 0].hist(raw_norm.flatten() * 255, bins=50, alpha=0.7, label='Raw (normalized)', color='red')
    axes[1, 0].set_title('Histogram Comparison')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(raw_samples, jpg_samples, alpha=0.5, s=1)
    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='1:1 line')
    axes[1, 1].set_xlabel('Normalized Raw')
    axes[1, 1].set_ylabel('Normalized JPG')
    axes[1, 1].set_title(f'Mapping (r={correlation:.3f})')
    axes[1, 1].legend()
    
    im5 = axes[1, 2].imshow(difference, cmap='viridis')
    axes[1, 2].set_title('Absolute Difference')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    
    # Line profiles
    center_y = raw_thermal.shape[0] // 2
    axes[1, 3].plot(jpg_norm[center_y, :], label='JPG (normalized)', alpha=0.7)
    axes[1, 3].plot(raw_norm[center_y, :], label='Raw (normalized)', alpha=0.7)
    axes[1, 3].set_title('Center Line Profile')
    axes[1, 3].set_xlabel('Pixel')
    axes[1, 3].set_ylabel('Normalized Value')
    axes[1, 3].legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'jpg': jpg_array,
        'raw': raw_thermal,
        'tiff': tiff_array,
        'temp': temp_celsius,
        'correlation': correlation
    }


def main():
    """Main analysis"""
    print("Thermal Data Discrepancy Analysis")
    print("==================================")
    
    # Analyze a sample frame
    results = analyze_single_frame(248)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print("\nThe discrepancy between JPG and raw thermal values likely occurs because:")
    print("1. The JPG uses automatic gain/contrast adjustment for visualization")
    print("2. Dynamic range compression from 16-bit raw to 8-bit JPG")
    print("3. Possible histogram equalization or gamma correction")
    print("4. The JPG is optimized for human viewing, not absolute temperature values")
    print("\nFor accurate temperature measurement, use the TIFF or IRG raw data,")
    print("not the JPG visualization.")


if __name__ == "__main__":
    main()