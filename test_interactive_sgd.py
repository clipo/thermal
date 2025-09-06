#!/usr/bin/env python3
"""
Simple interactive SGD viewer for testing.
"""

import matplotlib
matplotlib.use('TkAgg')  # Ensure interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from sgd_detector_integrated import IntegratedSGDDetector
from pathlib import Path
import numpy as np

def test_interactive():
    """Test the interactive SGD viewer"""
    
    print("Interactive SGD Viewer Test")
    print("=" * 50)
    
    # Initialize detector
    detector = IntegratedSGDDetector()
    
    # Find available frames
    frames = []
    for f in sorted(detector.base_path.glob("MAX_*.JPG"))[:20]:
        num = int(f.stem.split('_')[1])
        if (detector.base_path / f"IRX_{num:04d}.irg").exists():
            frames.append(num)
    
    if not frames:
        print("No frames found!")
        return
    
    print(f"Found {len(frames)} frames: {frames[0]} to {frames[-1]}")
    
    # State variables
    current_idx = [0]  # Use list to allow modification in nested functions
    temp_threshold = [1.0]
    min_area = [50]
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SGD Detection Interactive Viewer', fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    def update_display():
        """Update the display with current frame and parameters"""
        frame_num = frames[current_idx[0]]
        
        # Update detector parameters
        detector.temp_threshold = temp_threshold[0]
        detector.min_area = min_area[0]
        
        # Process frame
        print(f"Processing frame {frame_num} (threshold: {temp_threshold[0]:.1f}°C, min area: {min_area[0]} px)")
        result = detector.process_frame(frame_num, visualize=False)
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Display results
        # 1. RGB aligned
        axes[0].imshow(result['data']['rgb_aligned'])
        axes[0].set_title(f'Frame {frame_num}: RGB Aligned')
        axes[0].axis('off')
        
        # 2. Segmentation
        mask_display = np.zeros((*result['masks']['ocean'].shape, 3))
        mask_display[result['masks']['ocean']] = [0, 0.3, 1]  # Blue for ocean
        mask_display[result['masks']['land']] = [0, 0.5, 0]   # Green for land
        mask_display[result['masks']['waves']] = [1, 1, 1]    # White for waves
        axes[1].imshow(mask_display)
        axes[1].set_title('Ocean/Land/Waves')
        axes[1].axis('off')
        
        # 3. Thermal
        thermal_display = result['data']['thermal']
        im = axes[2].imshow(thermal_display, cmap='RdYlBu_r')
        axes[2].set_title(f'Thermal ({thermal_display.min():.1f} - {thermal_display.max():.1f}°C)')
        axes[2].axis('off')
        
        # 4. Ocean thermal only
        ocean_thermal = thermal_display.copy()
        ocean_thermal[~result['masks']['ocean']] = np.nan
        axes[3].imshow(ocean_thermal, cmap='viridis')
        axes[3].set_title(f'Ocean Thermal (median: {np.nanmedian(ocean_thermal):.1f}°C)')
        axes[3].axis('off')
        
        # 5. SGD detections
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        if result['sgd_mask'].any():
            sgd_display[result['sgd_mask']] = [0, 1, 1]  # Cyan for SGD
        
        # Add shoreline
        shoreline, _ = detector.detect_shoreline(result['masks'])
        sgd_display[shoreline] = [1, 1, 0]  # Yellow for shoreline
        
        axes[4].imshow(sgd_display)
        axes[4].set_title(f'SGD Detection: {len(result["plume_info"])} plumes')
        axes[4].axis('off')
        
        # 6. Overlay
        overlay = result['data']['rgb_aligned'].copy() / 255.0
        if result['sgd_mask'].any():
            # Make SGD areas bright green
            overlay[result['sgd_mask']] = [0, 1, 0]
        axes[5].imshow(overlay)
        axes[5].set_title('RGB + SGD Overlay')
        axes[5].axis('off')
        
        # Update info text
        fig.suptitle(f'Frame {frame_num} - Found {len(result["plume_info"])} SGD plumes | ' +
                    f'Temp threshold: {temp_threshold[0]:.1f}°C | Min area: {min_area[0]} px', 
                    fontsize=14)
        
        # Print plume info
        if result['plume_info']:
            print(f"  Found {len(result['plume_info'])} plumes:")
            for i, plume in enumerate(result['plume_info'][:3], 1):
                print(f"    Plume {i}: {plume['area_pixels']} pixels, "
                      f"temp diff: {plume.get('mean_temp_diff', 0):.1f}°C")
        
        plt.draw()
    
    # Navigation functions
    def next_frame(event):
        if current_idx[0] < len(frames) - 1:
            current_idx[0] += 1
            update_display()
    
    def prev_frame(event):
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_display()
    
    def update_temp(val):
        temp_threshold[0] = val
        update_display()
    
    def update_area(val):
        min_area[0] = int(val)
        update_display()
    
    # Add control widgets
    ax_prev = plt.axes([0.2, 0.02, 0.1, 0.04])
    ax_next = plt.axes([0.31, 0.02, 0.1, 0.04])
    ax_temp = plt.axes([0.55, 0.07, 0.3, 0.03])
    ax_area = plt.axes([0.55, 0.02, 0.3, 0.03])
    
    btn_prev = Button(ax_prev, 'Previous')
    btn_next = Button(ax_next, 'Next')
    slider_temp = Slider(ax_temp, 'Temp Threshold (°C)', 0.5, 3.0, valinit=1.0)
    slider_area = Slider(ax_area, 'Min Area (px)', 10, 200, valinit=50, valstep=10)
    
    # Connect callbacks
    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)
    slider_temp.on_changed(update_temp)
    slider_area.on_changed(update_area)
    
    # Keyboard shortcuts
    def on_key(event):
        if event.key == 'right':
            next_frame(None)
        elif event.key == 'left':
            prev_frame(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\nControls:")
    print("  Buttons: Previous/Next to navigate frames")
    print("  Keyboard: Left/Right arrow keys")
    print("  Sliders: Adjust detection parameters")
    print("\nClose window to exit.")
    
    # Initial display
    update_display()
    
    # Show the interactive window
    plt.show()

if __name__ == "__main__":
    test_interactive()