#!/usr/bin/env python3
"""
Test SAM for SGD Detection - Interactive Experiment

Compares SAM-based SGD detection with threshold-based detection.

Usage:
    python scripts/test_sam_sgd_detection.py --rgb data/100MEDIA/MAX_0001.JPG --thermal data/100MEDIA/IRX_0001.irg
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
import argparse
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("✗ SAM not installed! Run: bash scripts/setup_sam.sh")
    sys.exit(1)

# IRG loading function (from base detector)
def load_irg_data(irg_path, thermal_width=640, thermal_height=512):
    """Load thermal data from IRG file"""
    with open(irg_path, 'rb') as f:
        irg_data = f.read()

    # Parse thermal data (skip header)
    pixel_data_size = thermal_width * thermal_height * 2  # 2 bytes per pixel
    header_size = len(irg_data) - pixel_data_size

    if header_size > 0:
        raw_thermal = np.frombuffer(irg_data[header_size:], dtype=np.uint16)
        thermal = raw_thermal.reshape((thermal_height, thermal_width))

        # Convert to Celsius (same conversion as base detector)
        thermal = thermal.astype(np.float32) / 10.0 - 273.15

        return thermal
    else:
        raise ValueError(f"IRG file appears corrupted: {irg_path}")


class SAMSGDDetector:
    """Interactive SAM-based SGD detection tool"""

    def __init__(self, rgb_path, thermal_path, ocean_mask=None):
        self.rgb_path = Path(rgb_path)
        self.thermal_path = Path(thermal_path)

        # Load data
        print("Loading RGB image...")
        self.rgb = cv2.imread(str(self.rgb_path))
        self.rgb = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2RGB)

        print("Loading thermal data...")
        self.thermal = load_irg_data(str(self.thermal_path))
        self.h, self.w = self.thermal.shape

        # Load or create ocean mask
        if ocean_mask is not None:
            print("Using provided ocean mask")
            self.ocean_mask = ocean_mask
        else:
            print("Creating basic ocean mask (rule-based)...")
            self.ocean_mask = self.create_basic_ocean_mask()

        # Mask thermal to show only ocean
        self.masked_thermal = self.thermal.copy()
        self.masked_thermal[~self.ocean_mask] = np.nan  # NaN for land areas

        # Calculate ocean statistics
        ocean_temps = self.thermal[self.ocean_mask]
        self.ocean_mean = np.mean(ocean_temps)
        self.ocean_std = np.std(ocean_temps)
        self.ocean_median = np.median(ocean_temps)

        print(f"Ocean temperature stats:")
        print(f"  Mean: {self.ocean_mean:.2f}°C")
        print(f"  Std: {self.ocean_std:.2f}°C")
        print(f"  Median: {self.ocean_median:.2f}°C")

        # SAM setup
        device = self.get_device()
        print(f"Loading SAM (using {device})...")
        sam = sam_model_registry["vit_b"](checkpoint="models/sam/sam_vit_b_01ec64.pth")
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

        # SAM works on RGB, but we'll use thermal for detection
        self.sam_predictor.set_image(self.rgb)
        print("✓ SAM ready\n")

        # Track clicked SGD points
        self.sgd_points = []
        self.sam_masks = []

        # Track colorbars to avoid duplicates
        self.colorbars = {}

    def get_device(self):
        """Get best available device"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
        except:
            pass
        return "cpu"

    def create_basic_ocean_mask(self):
        """Create a simple ocean mask using blue channel threshold"""
        # This is a fallback - in real use, you'd use SAM segmentation
        rgb_hsv = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2HSV)

        # Ocean is typically blue/cyan
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(rgb_hsv, lower_blue, upper_blue)
        mask = mask > 0

        # Clean up
        from skimage import morphology
        mask = morphology.remove_small_objects(mask, min_size=1000)
        mask = morphology.remove_small_holes(mask, area_threshold=1000)

        # Resize to match thermal dimensions (512x640)
        mask_resized = cv2.resize(mask.astype(np.uint8),
                                 (self.w, self.h),
                                 interpolation=cv2.INTER_NEAREST)

        return mask_resized > 0

    def resize_mask_to_rgb(self, mask):
        """Resize a thermal-resolution mask to RGB dimensions for visualization"""
        rgb_h, rgb_w = self.rgb.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.uint8),
                                 (rgb_w, rgb_h),
                                 interpolation=cv2.INTER_NEAREST)
        return mask_resized > 0

    def detect_threshold_sgd(self, threshold=0.5):
        """Current threshold-based SGD detection"""
        # Find areas cooler than median by threshold
        sgd_mask = (self.thermal < (self.ocean_median - threshold)) & self.ocean_mask

        # Remove small objects
        from skimage import morphology
        sgd_mask = morphology.remove_small_objects(sgd_mask, min_size=50)

        return sgd_mask

    def detect_sam_sgd(self):
        """SAM-based SGD detection using clicked points"""
        if len(self.sgd_points) == 0:
            return None

        # Scale clicked points from thermal to RGB coordinates
        rgb_h, rgb_w = self.rgb.shape[:2]
        scale_x = rgb_w / self.w
        scale_y = rgb_h / self.h

        points_thermal = np.array(self.sgd_points)
        points_rgb = points_thermal * [scale_x, scale_y]
        labels_array = np.ones(len(self.sgd_points))  # All foreground

        # SAM operates on RGB image
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points_rgb,
            point_labels=labels_array,
            multimask_output=True,
        )

        # Use best mask (at RGB resolution)
        best_idx = np.argmax(scores)
        sam_mask_rgb = masks[best_idx]

        # Resize to thermal resolution for combining with ocean/thermal data
        sam_mask = cv2.resize(sam_mask_rgb.astype(np.uint8),
                             (self.w, self.h),
                             interpolation=cv2.INTER_NEAREST) > 0

        # Combine with ocean mask (only keep areas in ocean)
        sam_mask = sam_mask & self.ocean_mask

        # Also filter by temperature (should be cooler than mean)
        sam_mask = sam_mask & (self.thermal < self.ocean_mean)

        return sam_mask

    def run(self):
        """Run interactive SGD detection comparison"""
        print("="*70)
        print("SAM-BASED SGD DETECTION - INTERACTIVE TEST")
        print("="*70)
        print("\nWORKFLOW:")
        print("  1. Look at masked thermal visualization (left)")
        print("  2. Click on potential SGD cold spots (blue points)")
        print("  3. See SAM detection results (top right)")
        print("  4. Compare with threshold detection (bottom right)")
        print("\nCONTROLS:")
        print("  Left Click:  Mark potential SGD location")
        print("  Press 's':   Segment with SAM")
        print("  Press 'c':   Clear points")
        print("  Press 'q':   Quit")
        print("="*70 + "\n")

        # Create figure with 3 panels
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()
        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.axes[0, 0]:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Only allow clicks in ocean
        if not self.ocean_mask[y, x]:
            print(f"⚠️  Point ({x}, {y}) is on land, ignored")
            return

        self.sgd_points.append([x, y])
        temp = self.thermal[y, x]
        print(f"Added SGD point at ({x}, {y}) - Temp: {temp:.2f}°C (baseline: {self.ocean_median:.2f}°C)")

        self.update_display()

    def on_key(self, event):
        """Handle keyboard"""
        if event.key == 'c':
            self.sgd_points = []
            self.sam_masks = []
            print("Cleared all points")
            self.update_display()

        elif event.key == 's':
            self.segment_and_compare()

        elif event.key == 'q':
            plt.close()

    def update_display(self):
        """Update the display"""
        # Remove old colorbars to prevent accumulation
        for key in list(self.colorbars.keys()):
            if self.colorbars[key] is not None:
                self.colorbars[key].remove()
        self.colorbars.clear()

        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Top left: Masked thermal with temperature deviations
        deviation = self.thermal - self.ocean_median
        deviation_masked = deviation.copy()
        deviation_masked[~self.ocean_mask] = np.nan

        im1 = self.axes[0, 0].imshow(deviation_masked, cmap='coolwarm',
                                     vmin=-2, vmax=2, interpolation='nearest')
        self.axes[0, 0].set_title(f'Thermal Deviations from Baseline ({self.ocean_median:.1f}°C)\nClick on cold spots (blue)',
                                 fontweight='bold')
        self.colorbars['deviation'] = plt.colorbar(im1, ax=self.axes[0, 0], label='ΔT (°C)')

        # Draw SGD points
        if self.sgd_points:
            points = np.array(self.sgd_points)
            self.axes[0, 0].scatter(points[:, 0], points[:, 1],
                                   c='yellow', s=100, marker='o',
                                   edgecolors='black', linewidths=2,
                                   label=f'SGD Points ({len(self.sgd_points)})')
            self.axes[0, 0].legend()

        # Bottom left: Absolute temperature (masked)
        im2 = self.axes[1, 0].imshow(self.masked_thermal, cmap='plasma',
                                     interpolation='nearest')
        self.axes[1, 0].set_title('Absolute Ocean Temperature', fontweight='bold')
        self.colorbars['absolute'] = plt.colorbar(im2, ax=self.axes[1, 0], label='Temperature (°C)')

        # Top right: SAM detection results
        if len(self.sgd_points) > 0 and len(self.sam_masks) > 0:
            sam_mask = self.sam_masks[-1]
            # Resize mask to RGB dimensions for visualization
            sam_mask_rgb = self.resize_mask_to_rgb(sam_mask)
            overlay = self.rgb.copy()
            overlay[sam_mask_rgb] = overlay[sam_mask_rgb] * 0.5 + np.array([255, 0, 0]) * 0.5

            self.axes[0, 1].imshow(overlay)
            sgd_count = np.sum(sam_mask)
            self.axes[0, 1].set_title(f'SAM Detection\n{sgd_count} pixels detected',
                                     fontweight='bold', color='red')
        else:
            self.axes[0, 1].text(0.5, 0.5,
                                'Click SGD points\nthen press S',
                                ha='center', va='center',
                                transform=self.axes[0, 1].transAxes,
                                fontsize=14)
            self.axes[0, 1].set_title('SAM Detection (press S)', fontweight='bold')

        self.axes[0, 1].axis('off')

        # Bottom right: Threshold detection
        threshold_mask = self.detect_threshold_sgd(threshold=0.5)
        # Resize mask to RGB dimensions for visualization
        threshold_mask_rgb = self.resize_mask_to_rgb(threshold_mask)
        overlay2 = self.rgb.copy()
        overlay2[threshold_mask_rgb] = overlay2[threshold_mask_rgb] * 0.5 + np.array([0, 255, 0]) * 0.5

        self.axes[1, 1].imshow(overlay2)
        threshold_count = np.sum(threshold_mask)
        self.axes[1, 1].set_title(f'Threshold Detection (0.5°C)\n{threshold_count} pixels detected',
                                 fontweight='bold', color='green')
        self.axes[1, 1].axis('off')

        plt.draw()

    def segment_and_compare(self):
        """Run SAM segmentation and compare with threshold"""
        if len(self.sgd_points) == 0:
            print("⚠️  No SGD points marked! Click on cold spots first.")
            return

        print("\nRunning SAM segmentation...")
        sam_mask = self.detect_sam_sgd()

        if sam_mask is not None:
            self.sam_masks.append(sam_mask)

            # Statistics
            sam_count = np.sum(sam_mask)
            sam_temps = self.thermal[sam_mask]

            threshold_mask = self.detect_threshold_sgd(threshold=0.5)
            threshold_count = np.sum(threshold_mask)
            threshold_temps = self.thermal[threshold_mask]

            print(f"\n📊 DETECTION COMPARISON:")
            print(f"  SAM Detection:")
            print(f"    Pixels: {sam_count}")
            if sam_count > 0:
                print(f"    Mean temp: {np.mean(sam_temps):.2f}°C")
                print(f"    Min temp: {np.min(sam_temps):.2f}°C")

            print(f"  Threshold Detection:")
            print(f"    Pixels: {threshold_count}")
            if threshold_count > 0:
                print(f"    Mean temp: {np.mean(threshold_temps):.2f}°C")
                print(f"    Min temp: {np.min(threshold_temps):.2f}°C")

            # Overlap
            overlap = np.sum(sam_mask & threshold_mask)
            print(f"  Overlap: {overlap} pixels")

            if sam_count > 0 and threshold_count > 0:
                iou = overlap / np.sum(sam_mask | threshold_mask)
                print(f"  IoU (agreement): {iou:.2%}")

            self.update_display()


def main():
    parser = argparse.ArgumentParser(description='Test SAM for SGD detection')
    parser.add_argument('--rgb', required=True, help='Path to RGB image (MAX_*.JPG)')
    parser.add_argument('--thermal', required=True, help='Path to thermal image (IRX_*.irg)')
    parser.add_argument('--ocean-mask', help='Path to ocean mask (.npy file)')
    args = parser.parse_args()

    if not Path(args.rgb).exists():
        print(f"✗ RGB image not found: {args.rgb}")
        return 1

    if not Path(args.thermal).exists():
        print(f"✗ Thermal image not found: {args.thermal}")
        return 1

    # Load ocean mask if provided
    ocean_mask = None
    if args.ocean_mask and Path(args.ocean_mask).exists():
        ocean_mask = np.load(args.ocean_mask)

    detector = SAMSGDDetector(args.rgb, args.thermal, ocean_mask)
    detector.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
