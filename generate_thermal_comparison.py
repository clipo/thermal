#!/usr/bin/env python3
"""
Generate comparison images showing why IRX processed thermal images
are not suitable for SGD detection due to local contrast enhancement.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import struct

def load_irg_data(irg_path):
    """Load raw thermal data from IRG file"""
    with open(irg_path, 'rb') as f:
        irg_data = f.read()
    
    # Skip header and extract raw thermal values
    offset = 0x280
    width, height = 640, 512
    
    thermal_raw = []
    for i in range(height):
        for j in range(width):
            idx = offset + (i * width + j) * 2
            if idx + 1 < len(irg_data):
                value = struct.unpack('<H', irg_data[idx:idx+2])[0]
                thermal_raw.append(value)
    
    thermal_raw = np.array(thermal_raw).reshape((height, width))
    
    # Convert from deciKelvin to Celsius
    thermal_celsius = thermal_raw / 10.0 - 273.15
    
    return thermal_celsius

def create_comparison_figure():
    """Create figure comparing IRX processed vs raw thermal"""
    print("Creating thermal comparison visualization...")
    
    frame_num = 248
    base_path = Path("data/100MEDIA")
    
    # Load IRX processed image
    irx_path = base_path / f"IRX_{frame_num:04d}.jpg"
    irx_image = np.array(Image.open(irx_path))
    
    # Load raw thermal data
    irg_path = base_path / f"IRX_{frame_num:04d}.irg"
    thermal_raw = load_irg_data(irg_path)
    
    # Load RGB for context
    rgb_path = base_path / f"MAX_{frame_num:04d}.JPG"
    rgb_full = np.array(Image.open(rgb_path))
    
    # Extract RGB region matching thermal FOV
    h, w = rgb_full.shape[:2]
    thermal_w = int(w * 0.7)
    thermal_h = int(h * 0.7)
    x_offset = (w - thermal_w) // 2
    y_offset = (h - thermal_h) // 2
    rgb_region = rgb_full[y_offset:y_offset+thermal_h, x_offset:x_offset+thermal_w]
    
    # Resize RGB to match thermal
    rgb_aligned = np.array(Image.fromarray(rgb_region).resize((640, 512), Image.Resampling.BILINEAR))
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Why IRX Processed Images Cannot Be Used for SGD Detection', 
                 fontsize=16, fontweight='bold')
    
    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Full comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Display IRX processed
    ax1.imshow(irx_image, cmap='gray')
    ax1.set_title('IRX Processed Image\n(Local Contrast Enhanced)', fontweight='bold')
    ax1.axis('off')
    
    # Display raw thermal
    im2 = ax2.imshow(thermal_raw, cmap='RdYlBu_r', vmin=23, vmax=26)
    ax2.set_title('Raw Thermal Data\n(Absolute Temperature °C)', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='°C', fraction=0.046)
    
    # Display RGB for reference
    ax3.imshow(rgb_aligned)
    ax3.set_title('RGB Reference\n(Aligned to Thermal FOV)', fontweight='bold')
    ax3.axis('off')
    
    # Row 2: Zoomed regions showing the problem
    # Select two regions with similar appearance in IRX but different temps
    region1_coords = (200, 250, 100, 100)  # x, y, w, h - ocean area
    region2_coords = (400, 300, 100, 100)  # different ocean area
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Extract regions
    x1, y1, w1, h1 = region1_coords
    x2, y2, w2, h2 = region2_coords
    
    # Region 1 - IRX
    region1_irx = irx_image[y1:y1+h1, x1:x1+w1]
    ax4.imshow(region1_irx, cmap='gray', vmin=0, vmax=255)
    ax4.set_title(f'IRX Region 1\nMean pixel value: {region1_irx.mean():.1f}')
    ax4.axis('off')
    
    # Region 1 - Raw thermal
    region1_thermal = thermal_raw[y1:y1+h1, x1:x1+w1]
    im5 = ax5.imshow(region1_thermal, cmap='RdYlBu_r', vmin=23, vmax=26)
    ax5.set_title(f'Raw Thermal Region 1\nMean temp: {np.nanmean(region1_thermal):.2f}°C')
    ax5.axis('off')
    
    # Region 2 comparison
    region2_irx = irx_image[y2:y2+h2, x2:x2+w2]
    region2_thermal = thermal_raw[y2:y2+h2, x2:x2+w2]
    
    # Create comparison plot
    ax6.bar(['Region 1\nIRX', 'Region 2\nIRX'], 
            [region1_irx.mean(), region2_irx.mean()],
            color=['gray', 'gray'], alpha=0.7, label='IRX Pixel Values')
    ax6_twin = ax6.twinx()
    ax6_twin.bar(['Region 1\nTemp', 'Region 2\nTemp'], 
                 [np.nanmean(region1_thermal), np.nanmean(region2_thermal)],
                 color=['red', 'blue'], alpha=0.7, label='Actual Temperature')
    
    ax6.set_ylabel('IRX Pixel Value', color='gray')
    ax6_twin.set_ylabel('Temperature (°C)', color='red')
    ax6.set_title('Similar IRX Values ≠ Similar Temperatures')
    ax6.tick_params(axis='y', labelcolor='gray')
    ax6_twin.tick_params(axis='y', labelcolor='red')
    
    # Row 3: Histograms and analysis
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])
    ax9 = fig.add_subplot(gs[2, 2])
    
    # IRX histogram
    ax7.hist(irx_image.flatten(), bins=50, color='gray', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Pixel Value')
    ax7.set_ylabel('Frequency')
    ax7.set_title('IRX Histogram\n(Equalized Distribution)')
    ax7.axvline(irx_image.mean(), color='red', linestyle='--', label=f'Mean: {irx_image.mean():.1f}')
    ax7.legend()
    
    # Raw thermal histogram
    valid_temps = thermal_raw[~np.isnan(thermal_raw)]
    ax8.hist(valid_temps.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Temperature (°C)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Raw Thermal Histogram\n(Natural Distribution)')
    ax8.axvline(np.nanmean(thermal_raw), color='red', linestyle='--', 
                label=f'Mean: {np.nanmean(thermal_raw):.2f}°C')
    ax8.legend()
    
    # Add text explanation
    explanation = """
    Problems with IRX Processed Images:
    
    1. LOCAL CONTRAST ENHANCEMENT
       • Dark pixels enhanced differently in each region
       • Same gray value ≠ same temperature
       • Enhancement removes absolute temperature info
    
    2. HISTOGRAM EQUALIZATION
       • Spreads values across full 0-255 range
       • Destroys natural temperature distribution
       • Makes subtle differences appear dramatic
    
    3. LOSS OF QUANTITATIVE DATA
       • Cannot measure actual temperature differences
       • Cannot detect small (1-2°C) SGD anomalies
       • Visual appearance misleading for analysis
    
    Solution: Use raw thermal data (.irg files)
    • Preserves absolute temperature values
    • Allows detection of subtle anomalies
    • Enables quantitative analysis
    """
    
    ax9.text(0.05, 0.95, explanation, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax9.axis('off')
    
    # Add region indicators on main images
    for ax in [ax1, ax2]:
        rect1 = plt.Rectangle((x1, y1), w1, h1, linewidth=2, 
                              edgecolor='yellow', facecolor='none')
        rect2 = plt.Rectangle((x2, y2), w2, h2, linewidth=2, 
                              edgecolor='cyan', facecolor='none')
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.text(x1+w1/2, y1-5, 'Region 1', color='yellow', 
                ha='center', fontweight='bold')
        ax.text(x2+w2/2, y2-5, 'Region 2', color='cyan', 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/images/irx_vs_raw_thermal.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/irx_vs_raw_thermal.png")

def create_sgd_detection_comparison():
    """Create figure showing why raw thermal is needed for SGD"""
    print("Creating SGD detection comparison...")
    
    frame_num = 248
    base_path = Path("data/100MEDIA")
    
    # Load data
    irx_image = np.array(Image.open(base_path / f"IRX_{frame_num:04d}.jpg"))
    thermal_raw = load_irg_data(base_path / f"IRX_{frame_num:04d}.irg")
    
    # Simulate ocean mask (simplified)
    ocean_mask = thermal_raw < 25.5  # Simple threshold for demo
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SGD Detection: IRX Processed vs Raw Thermal', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: IRX approach (fails)
    axes[0, 0].imshow(irx_image, cmap='gray')
    axes[0, 0].set_title('IRX Processed Image')
    axes[0, 0].axis('off')
    
    # Try to find cold spots in IRX (this will fail)
    # Convert to grayscale if needed
    if len(irx_image.shape) == 3:
        irx_gray = np.mean(irx_image, axis=2)
    else:
        irx_gray = irx_image
    
    irx_threshold = np.percentile(irx_gray, 20)  # Bottom 20% of values
    irx_cold = irx_gray < irx_threshold
    axes[0, 1].imshow(irx_cold, cmap='Blues')
    axes[0, 1].set_title('IRX "Cold" Areas\n(Unreliable - includes land/rocks)')
    axes[0, 1].axis('off')
    
    # Show why it fails
    false_positives = np.zeros((irx_cold.shape[0], irx_cold.shape[1], 3))
    false_positives[irx_cold & ~ocean_mask] = [1, 0, 0]  # Red for false positives
    false_positives[irx_cold & ocean_mask] = [0, 0, 1]   # Blue for ocean
    axes[0, 2].imshow(false_positives)
    axes[0, 2].set_title('IRX Result\n(Red = False Positives)')
    axes[0, 2].axis('off')
    
    # Row 2: Raw thermal approach (works)
    im = axes[1, 0].imshow(thermal_raw, cmap='RdYlBu_r', vmin=23, vmax=26)
    axes[1, 0].set_title('Raw Thermal Data (°C)')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], label='°C')
    
    # Ocean thermal only
    ocean_thermal = thermal_raw.copy()
    ocean_thermal[~ocean_mask] = np.nan
    median_temp = np.nanmedian(ocean_thermal)
    
    im2 = axes[1, 1].imshow(ocean_thermal, cmap='viridis', vmin=24, vmax=26)
    axes[1, 1].set_title(f'Ocean Only\n(Median: {median_temp:.2f}°C)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], label='°C')
    
    # SGD detection (cold anomalies in ocean)
    sgd_mask = (ocean_thermal < median_temp - 1.0) & ocean_mask
    sgd_display = np.zeros((*sgd_mask.shape, 3))
    sgd_display[sgd_mask] = [0, 1, 1]  # Cyan for SGD
    sgd_display[ocean_mask & ~sgd_mask] = [0, 0, 0.3]  # Dark blue for ocean
    
    axes[1, 2].imshow(sgd_display)
    axes[1, 2].set_title(f'SGD Detection\n(< {median_temp-1:.1f}°C in ocean)')
    axes[1, 2].axis('off')
    
    # Add text annotations
    fig.text(0.5, 0.52, '✗ IRX FAILS: Cannot distinguish temperature from contrast enhancement', 
             ha='center', fontsize=11, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.text(0.5, 0.02, '✓ RAW THERMAL WORKS: Preserves absolute temperatures for quantitative analysis', 
             ha='center', fontsize=11, color='green', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, top=0.92)
    plt.savefig('docs/images/sgd_detection_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/sgd_detection_comparison.png")

def main():
    """Generate all comparison images"""
    print("\nGenerating thermal comparison images...")
    print("=" * 50)
    
    try:
        create_comparison_figure()
        create_sgd_detection_comparison()
        
        print("\n" + "=" * 50)
        print("Comparison images generated successfully!")
        print("Images saved in: docs/images/")
        
    except Exception as e:
        print(f"\nError generating images: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())