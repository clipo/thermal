#!/usr/bin/env python3
"""
Generate screenshots for documentation
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image
import sys

# Import our modules
from sgd_detector_integrated import IntegratedSGDDetector
from ml_segmentation_fast import FastMLSegmenter

def generate_sgd_viewer_screenshot():
    """Generate main SGD viewer interface screenshot"""
    print("Generating SGD viewer screenshot...")
    
    # Create detector
    detector = IntegratedSGDDetector(use_ml=True)
    
    # Process a frame
    frame_num = 248
    result = detector.process_frame(frame_num, visualize=False)
    
    # Create figure similar to sgd_viewer layout
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'SGD Detection Viewer - Frame {frame_num} | NEW: 2 | Existing: 0 | Total Unique SGD: 2', 
                 fontsize=14, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(2, 4, left=0.05, right=0.95, top=0.92, bottom=0.20,
                          hspace=0.3, wspace=0.3)
    
    axes = []
    for i in range(2):
        for j in range(4):
            axes.append(fig.add_subplot(gs[i, j]))
    
    # Panel 1: RGB
    axes[0].imshow(result['data']['rgb_aligned'])
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Panel 2: Segmentation
    mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
    mask_display[result['masks']['ocean']] = [0, 0.3, 1]
    mask_display[result['masks']['land']] = [0, 0.5, 0]
    mask_display[result['masks']['waves']] = [1, 1, 1]
    axes[1].imshow(mask_display)
    axes[1].set_title('Ocean/Land/Waves')
    axes[1].axis('off')
    
    # Panel 3: Thermal
    thermal = result['data']['thermal']
    im = axes[2].imshow(thermal, cmap='RdYlBu_r')
    axes[2].set_title(f'Thermal ({thermal.min():.1f}-{thermal.max():.1f}°C)')
    axes[2].axis('off')
    
    # Panel 4: Ocean thermal
    ocean_thermal = thermal.copy()
    ocean_thermal[~result['masks']['ocean']] = np.nan
    axes[3].imshow(ocean_thermal, cmap='viridis')
    axes[3].set_title(f'Ocean (median: {np.nanmedian(ocean_thermal):.1f}°C)')
    axes[3].axis('off')
    
    # Panel 5: SGD detection
    sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
    if result['sgd_mask'].any():
        sgd_display[result['sgd_mask']] = [0, 1, 1]
    shoreline, _ = detector.detect_shoreline(result['masks'])
    sgd_display[shoreline] = [1, 1, 0]
    axes[4].imshow(sgd_display)
    axes[4].set_title(f'SGD Detection: {len(result["plume_info"])} plumes')
    axes[4].axis('off')
    
    # Panel 6: New vs Existing
    status_display = np.zeros((*result['sgd_mask'].shape, 3))
    if result['sgd_mask'].any():
        status_display[result['sgd_mask']] = [0, 1, 0]  # Green for new
    status_display[shoreline] = [1, 1, 0]
    axes[5].imshow(status_display)
    axes[5].set_title('New SGD (Green) vs Existing (Blue)')
    axes[5].axis('off')
    
    # Panel 7: Coverage map placeholder
    axes[6].set_xlim(-109.5, -109.4)
    axes[6].set_ylim(-27.2, -27.1)
    axes[6].scatter([-109.445], [-27.163], c='red', s=100, marker='*')
    axes[6].set_title('Geographic Coverage')
    axes[6].set_xlabel('Longitude')
    axes[6].set_ylabel('Latitude')
    axes[6].grid(True, alpha=0.3)
    
    # Panel 8: Statistics
    stats_text = "Statistics:\n\n"
    stats_text += f"Frames processed: 1\n"
    stats_text += f"Total detections: {len(result['plume_info'])}\n"
    stats_text += f"Unique locations: 2\n\n"
    stats_text += "Parameters:\n"
    stats_text += f"Temp threshold: 1.0°C\n"
    stats_text += f"Min area: 50 px\n"
    stats_text += f"Merge distance: 10 m\n\n"
    if result['plume_info']:
        stats_text += f"Largest plume: {result['plume_info'][0]['area_pixels']} px\n"
        stats_text += f"Temp anomaly: {result['characteristics']['temp_anomaly']:.1f}°C"
    
    axes[7].text(0.1, 0.5, stats_text, transform=axes[7].transAxes,
                fontsize=10, verticalalignment='center')
    axes[7].axis('off')
    
    # Add control labels (text only, not actual controls)
    fig.text(0.05, 0.14, 'Navigation:', fontsize=9, fontweight='bold')
    fig.text(0.10, 0.10, '[Previous] [Next] [Mark SGD] [Save] [Export]', fontsize=8)
    fig.text(0.05, 0.08, 'Parameters:', fontsize=9, fontweight='bold')
    fig.text(0.10, 0.04, 'Temperature: [====|====] 1.0°C    Min Area: [====|====] 50px    Merge: [====|====] 10m', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('docs/images/sgd_viewer_interface.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/sgd_viewer_interface.png")

def generate_detection_process():
    """Generate detection process visualization"""
    print("Generating detection process visualization...")
    
    detector = IntegratedSGDDetector(use_ml=True)
    frame_num = 248
    result = detector.process_frame(frame_num, visualize=False)
    
    # Create figure showing the detection pipeline
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SGD Detection Pipeline', fontsize=16, fontweight='bold')
    
    # Step 1: Original RGB
    axes[0, 0].imshow(result['data']['rgb_aligned'])
    axes[0, 0].set_title('Step 1: RGB Input\n(Aligned to Thermal FOV)')
    axes[0, 0].axis('off')
    
    # Step 2: Segmentation
    mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
    mask_display[result['masks']['ocean']] = [0, 0.3, 1]
    mask_display[result['masks']['land']] = [0, 0.5, 0]
    mask_display[result['masks']['waves']] = [1, 1, 1]
    axes[0, 1].imshow(mask_display)
    axes[0, 1].set_title('Step 2: ML Segmentation\n(Ocean/Land/Waves)')
    axes[0, 1].axis('off')
    
    # Step 3: Thermal
    thermal = result['data']['thermal']
    axes[0, 2].imshow(thermal, cmap='RdYlBu_r', vmin=23, vmax=26)
    axes[0, 2].set_title('Step 3: Thermal Data\n(Temperature in °C)')
    axes[0, 2].axis('off')
    
    # Step 4: Ocean thermal only
    ocean_thermal = thermal.copy()
    ocean_thermal[~result['masks']['ocean']] = np.nan
    axes[1, 0].imshow(ocean_thermal, cmap='viridis')
    axes[1, 0].set_title('Step 4: Ocean Thermal\n(Land masked out)')
    axes[1, 0].axis('off')
    
    # Step 5: Cold anomalies
    # Calculate cold anomalies from thermal data
    ocean_thermal = thermal.copy()
    ocean_thermal[~result['masks']['ocean']] = np.nan
    median_temp = np.nanmedian(ocean_thermal)
    cold_mask = (ocean_thermal < median_temp - 1.0) & result['masks']['ocean']
    
    anomaly_display = np.zeros((*cold_mask.shape, 3))
    anomaly_display[cold_mask] = [0, 1, 1]
    axes[1, 1].imshow(anomaly_display)
    axes[1, 1].set_title('Step 5: Cold Anomalies\n(< median - 1.0°C)')
    axes[1, 1].axis('off')
    
    # Step 6: Final SGD
    sgd_display = result['data']['rgb_aligned'].copy() / 255.0
    if result['sgd_mask'].any():
        # Highlight SGD in green
        sgd_display[result['sgd_mask']] = [0, 1, 0]
    # Add shoreline in yellow
    shoreline, _ = detector.detect_shoreline(result['masks'])
    sgd_display[shoreline] = [1, 1, 0]
    axes[1, 2].imshow(sgd_display)
    axes[1, 2].set_title(f'Step 6: SGD Detection\n({len(result["plume_info"])} plumes near shore)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/images/detection_pipeline.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/detection_pipeline.png")

def generate_segmentation_trainer():
    """Generate segmentation trainer interface mockup"""
    print("Generating segmentation trainer screenshot...")
    
    # Load an RGB image
    rgb_path = Path("data/100MEDIA/MAX_0248.JPG")
    rgb_full = np.array(Image.open(rgb_path))
    
    # Crop to match thermal FOV
    h, w = rgb_full.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    rgb_cropped = rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize
    img_pil = Image.fromarray(rgb_cropped)
    rgb_display = np.array(img_pil.resize((800, 640), Image.Resampling.BILINEAR))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Segmentation Training Tool - Click to Label Pixels', fontsize=14, fontweight='bold')
    
    # Left: Image with some labeled points
    axes[0].imshow(rgb_display)
    axes[0].set_title('Click to Label:\nLeft=Ocean (blue) | Right=Land (green) | Middle=Rock (gray)')
    
    # Add some example labeled points
    np.random.seed(42)
    # Ocean points (blue)
    ocean_x = np.random.randint(50, 300, 20)
    ocean_y = np.random.randint(100, 500, 20)
    axes[0].scatter(ocean_x, ocean_y, c='blue', s=20, alpha=0.6, label='Ocean')
    
    # Land points (green)
    land_x = np.random.randint(500, 750, 15)
    land_y = np.random.randint(50, 400, 15)
    axes[0].scatter(land_x, land_y, c='green', s=20, alpha=0.6, label='Land')
    
    # Rock points (gray)
    rock_x = np.random.randint(400, 600, 10)
    rock_y = np.random.randint(100, 300, 10)
    axes[0].scatter(rock_x, rock_y, c='gray', s=20, alpha=0.6, label='Rock')
    
    axes[0].legend(loc='upper right')
    axes[0].axis('off')
    
    # Right: Show predicted segmentation
    segmentation = np.zeros((640, 800, 3))
    # Create rough segmentation regions
    segmentation[:400, :350] = [0, 0.3, 1]  # Ocean
    segmentation[300:, 400:] = [0, 0.5, 0]  # Land
    segmentation[200:350, 350:450] = [0.5, 0.5, 0.5]  # Rock
    
    axes[1].imshow(segmentation)
    axes[1].set_title('Live Segmentation Preview\n(Press T to train, S to save)')
    axes[1].axis('off')
    
    # Add instructions
    instruction_text = """
    Controls:
    • Left Click: Label as Ocean
    • Right Click: Label as Land  
    • Middle Click: Label as Rock
    • Shift+Click: Label as Wave
    • T: Train model with current labels
    • S: Save trained model
    • C: Clear all labels
    • Space: Next image
    • U: Undo last label
    
    Labels: 45 points | Model: Not trained
    """
    fig.text(0.02, 0.5, instruction_text, fontsize=9, 
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('docs/images/segmentation_trainer.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/segmentation_trainer.png")

def generate_test_segmentation():
    """Generate test segmentation interface"""
    print("Generating test segmentation screenshot...")
    
    detector = IntegratedSGDDetector(use_ml=False)
    frame_num = 248
    result = detector.process_frame(frame_num, visualize=False)
    
    # Create figure
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Segmentation Parameter Testing - Frame 248', fontsize=14, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.95, top=0.90, bottom=0.25,
                          hspace=0.3, wspace=0.3)
    
    # Original RGB
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(result['data']['rgb_aligned'])
    ax1.set_title('Original RGB')
    ax1.axis('off')
    
    # Current segmentation
    ax2 = fig.add_subplot(gs[0, 1])
    mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
    mask_display[result['masks']['ocean']] = [0, 0.3, 1]
    mask_display[result['masks']['land']] = [0, 0.5, 0]
    mask_display[result['masks']['waves']] = [1, 1, 1]
    ax2.imshow(mask_display)
    ax2.set_title('Current Segmentation')
    ax2.axis('off')
    
    # Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    overlay = result['data']['rgb_aligned'].copy() / 255.0
    overlay[result['masks']['ocean']] = overlay[result['masks']['ocean']] * 0.5 + np.array([0, 0, 0.5]) * 0.5
    ax3.imshow(overlay)
    ax3.set_title('Overlay Visualization')
    ax3.axis('off')
    
    # HSV components
    from skimage import color
    hsv = color.rgb2hsv(result['data']['rgb_aligned'])
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(hsv[:,:,0], cmap='hsv')
    ax4.set_title('Hue Channel')
    ax4.axis('off')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(hsv[:,:,1], cmap='gray')
    ax5.set_title('Saturation Channel')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(hsv[:,:,2], cmap='gray')
    ax6.set_title('Value (Brightness) Channel')
    ax6.axis('off')
    
    # Add parameter controls (visual representation)
    fig.text(0.10, 0.18, 'Ocean HSV Parameters:', fontsize=10, fontweight='bold')
    fig.text(0.10, 0.15, 'Hue:        [180 ========|======== 250]', fontsize=9, family='monospace')
    fig.text(0.10, 0.12, 'Saturation: [20  ====|============ 255]', fontsize=9, family='monospace')
    fig.text(0.10, 0.09, 'Value:      [30  =====|========== 200]', fontsize=9, family='monospace')
    
    fig.text(0.50, 0.18, 'Land HSV Parameters:', fontsize=10, fontweight='bold')
    fig.text(0.50, 0.15, 'Hue:        [40  ====|========== 150]', fontsize=9, family='monospace')
    fig.text(0.50, 0.12, 'Saturation: [15  ==|============= 255]', fontsize=9, family='monospace')
    fig.text(0.50, 0.09, 'Value:      [10  =|============== 255]', fontsize=9, family='monospace')
    
    fig.text(0.10, 0.05, 'Navigation: [< Prev Frame] [Next Frame >]     Frame: 248/500', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('docs/images/test_segmentation.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/test_segmentation.png")

def generate_thermal_alignment():
    """Generate thermal-RGB alignment visualization"""
    print("Generating thermal alignment visualization...")
    
    # Create figure showing FOV relationship
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Thermal-RGB Field of View Alignment', fontsize=14, fontweight='bold')
    
    # Panel 1: Full RGB with thermal FOV outlined
    rgb_path = Path("data/100MEDIA/MAX_0248.JPG")
    rgb_full = np.array(Image.open(rgb_path))
    
    axes[0].imshow(rgb_full)
    axes[0].set_title(f'Full RGB Image\n(4096 × 3072 pixels)')
    
    # Draw thermal FOV rectangle (70% of image, centered)
    h, w = rgb_full.shape[:2]
    thermal_w = w * 0.7
    thermal_h = h * 0.7
    x_offset = (w - thermal_w) / 2
    y_offset = (h - thermal_h) / 2
    
    from matplotlib.patches import Rectangle
    rect = Rectangle((x_offset, y_offset), thermal_w, thermal_h, 
                    linewidth=3, edgecolor='yellow', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].annotate('Thermal FOV\n(~70% of RGB)', 
                    xy=(w/2, h/2), xytext=(w*0.8, h*0.2),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=2),
                    fontsize=12, color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    axes[0].axis('off')
    
    # Panel 2: Extracted and aligned RGB
    detector = IntegratedSGDDetector()
    rgb_aligned = detector.extract_aligned_rgb(rgb_full)
    
    axes[1].imshow(rgb_aligned)
    axes[1].set_title(f'Extracted & Aligned RGB\n(640 × 512 pixels)')
    axes[1].axis('off')
    
    # Panel 3: Thermal image
    frame_num = 248
    result = detector.process_frame(frame_num, visualize=False)
    thermal = result['data']['thermal']
    
    im = axes[2].imshow(thermal, cmap='RdYlBu_r')
    axes[2].set_title(f'Thermal Image\n(640 × 512 pixels)')
    axes[2].axis('off')
    
    # Add colorbar for thermal
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig('docs/images/thermal_alignment.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  Saved: docs/images/thermal_alignment.png")

def main():
    """Generate all screenshots"""
    print("\nGenerating documentation screenshots...")
    print("=" * 50)
    
    try:
        generate_sgd_viewer_screenshot()
        generate_detection_process()
        generate_segmentation_trainer()
        generate_test_segmentation()
        generate_thermal_alignment()
        
        print("\n" + "=" * 50)
        print("All screenshots generated successfully!")
        print("Images saved in: docs/images/")
        
    except Exception as e:
        print(f"\nError generating screenshots: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())