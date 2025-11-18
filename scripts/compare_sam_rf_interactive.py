#!/usr/bin/env python3
"""
Interactive comparison: Click points to guide SAM, compare with Random Forest

Usage:
    python scripts/compare_sam_rf_interactive.py --image data/100MEDIA/MAX_0001.JPG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# Try to import Random Forest segmenter
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sgd_toolkit.segmentation.ml_segmenter import FastMLSegmenter
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

class InteractiveComparator:
    """Interactive SAM vs Random Forest comparison"""

    def __init__(self, image_path, rf_model_path=None):
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.image_rgb.shape[:2]

        # SAM setup
        if SAM_AVAILABLE:
            device = self.get_device()
            print(f"Loading SAM (using {device})...")
            sam = sam_model_registry["vit_b"](checkpoint="models/sam/sam_vit_b_01ec64.pth")
            sam.to(device=device)
            self.sam_predictor = SamPredictor(sam)
            self.sam_predictor.set_image(self.image_rgb)
            print("✓ SAM ready")
        else:
            self.sam_predictor = None

        # Random Forest setup
        if RF_AVAILABLE and rf_model_path and Path(rf_model_path).exists():
            print(f"Loading Random Forest model...")
            self.rf_segmenter = FastMLSegmenter(model_path=rf_model_path)
            print("✓ Random Forest ready")
        else:
            self.rf_segmenter = None

        # Point tracking
        self.ocean_points = []
        self.land_points = []

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

    def run_random_forest(self):
        """Segment with Random Forest"""
        if not self.rf_segmenter:
            return None

        print("Running Random Forest segmentation...")
        result = self.rf_segmenter.segment_ultra_fast(self.image_rgb)
        # Extract ocean mask from result dict
        if isinstance(result, dict):
            return result.get('ocean', None)
        return result

    def run_sam(self):
        """Segment with SAM using current points"""
        if not self.sam_predictor or (len(self.ocean_points) == 0 and len(self.land_points) == 0):
            return None

        print("Running SAM segmentation...")

        # Combine points
        all_points = self.ocean_points + self.land_points
        all_labels = [1] * len(self.ocean_points) + [0] * len(self.land_points)

        if len(all_points) == 0:
            return None

        points_array = np.array(all_points)
        labels_array = np.array(all_labels)

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=True,
        )

        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx]

    def run(self):
        """Run interactive comparison"""
        print("\n" + "="*70)
        print("INTERACTIVE SAM vs RANDOM FOREST COMPARISON")
        print("="*70)
        print("\nInstructions:")
        print("  Left Click:  Add OCEAN point (blue)")
        print("  Right Click: Add LAND point (red) to exclude")
        print("  Press 'c':   Clear all points")
        print("  Press ENTER: Segment with current points")
        print("  Press 'q':   Quit")
        print("\nTip: Add 3-5 ocean points in different areas, then press ENTER")
        print("="*70 + "\n")

        # Create figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Disable default 's' (save) keybinding
        try:
            self.fig.canvas.manager.key_press_handler_id = None
        except:
            pass

        self.update_display()
        plt.show()

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.axes[0, 0]:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left click - ocean
            self.ocean_points.append([x, y])
            print(f"Added ocean point at ({x}, {y})")
        elif event.button == 3:  # Right click - land
            self.land_points.append([x, y])
            print(f"Added land point at ({x}, {y})")

        self.update_display()

    def on_key(self, event):
        """Handle keyboard"""
        if event.key == 'c':
            self.ocean_points = []
            self.land_points = []
            print("Cleared all points")
            self.update_display()

        elif event.key == 'enter':
            self.segment_and_compare()

        elif event.key == 'q':
            plt.close()

    def update_display(self):
        """Update the display"""
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()

        # Top left: Interactive image with points
        self.axes[0, 0].imshow(self.image_rgb)
        self.axes[0, 0].set_title('Click to add points\n(Left=Ocean, Right=Land)', fontweight='bold')

        # Draw points
        if self.ocean_points:
            points = np.array(self.ocean_points)
            self.axes[0, 0].scatter(points[:, 0], points[:, 1],
                                   c='blue', s=200, marker='o',
                                   edgecolors='white', linewidths=3,
                                   label='Ocean')

        if self.land_points:
            points = np.array(self.land_points)
            self.axes[0, 0].scatter(points[:, 0], points[:, 1],
                                   c='red', s=200, marker='x',
                                   linewidths=3,
                                   label='Land (exclude)')

        if self.ocean_points or self.land_points:
            self.axes[0, 0].legend()

        self.axes[0, 0].axis('off')

        # Instructions
        self.axes[0, 1].text(0.1, 0.5, f"""
POINTS ADDED:
  Ocean (blue): {len(self.ocean_points)}
  Land (red): {len(self.land_points)}

KEYBOARD:
  ENTER - Segment
  C - Clear points
  Q - Quit

STATUS:
  SAM: {'Ready' if SAM_AVAILABLE else 'Not available'}
  Random Forest: {'Ready' if self.rf_segmenter else 'Not available'}

Add 3-5 ocean points,
then press ENTER to segment!
        """, transform=self.axes[0, 1].transAxes,
                              fontsize=11, family='monospace', va='center')
        self.axes[0, 1].axis('off')

        # Bottom two will show results after segmentation
        self.axes[1, 0].text(0.5, 0.5, 'Random Forest Result\n(press ENTER to segment)',
                            ha='center', va='center',
                            transform=self.axes[1, 0].transAxes,
                            fontsize=12)
        self.axes[1, 0].axis('off')

        self.axes[1, 1].text(0.5, 0.5, 'SAM Result\n(press ENTER to segment)',
                            ha='center', va='center',
                            transform=self.axes[1, 1].transAxes,
                            fontsize=12)
        self.axes[1, 1].axis('off')

        plt.draw()

    def segment_and_compare(self):
        """Run segmentation and show comparison"""
        print("\nSegmenting...")

        # Run Random Forest
        rf_mask = self.run_random_forest()

        # Run SAM
        sam_mask = self.run_sam()

        # Display results
        self.axes[1, 0].clear()
        self.axes[1, 1].clear()

        if rf_mask is not None:
            # Show RF result
            colored_rf = self.image_rgb.copy()
            colored_rf[rf_mask == 0] = colored_rf[rf_mask == 0] * 0.3  # Darken non-ocean

            self.axes[1, 0].imshow(colored_rf)
            ocean_pct = np.sum(rf_mask > 0) / rf_mask.size * 100
            self.axes[1, 0].set_title(f'Random Forest\n{ocean_pct:.1f}% ocean', fontweight='bold')
            self.axes[1, 0].axis('off')
            print(f"✓ Random Forest: {ocean_pct:.1f}% ocean")
        else:
            self.axes[1, 0].text(0.5, 0.5, 'Random Forest\nNot Available',
                                ha='center', va='center',
                                transform=self.axes[1, 0].transAxes)
            self.axes[1, 0].axis('off')

        if sam_mask is not None:
            # Show SAM result
            colored_sam = self.image_rgb.copy()
            colored_sam[~sam_mask] = colored_sam[~sam_mask] * 0.3  # Darken non-ocean

            self.axes[1, 1].imshow(colored_sam)
            ocean_pct = np.sum(sam_mask) / sam_mask.size * 100
            self.axes[1, 1].set_title(f'SAM\n{ocean_pct:.1f}% ocean', fontweight='bold')
            self.axes[1, 1].axis('off')
            print(f"✓ SAM: {ocean_pct:.1f}% ocean")
        else:
            self.axes[1, 1].text(0.5, 0.5, 'SAM\nAdd points and press S',
                                ha='center', va='center',
                                transform=self.axes[1, 1].transAxes)
            self.axes[1, 1].axis('off')

        plt.draw()

def main():
    parser = argparse.ArgumentParser(description='Interactive SAM vs Random Forest comparison')
    parser.add_argument('--image', required=True, help='Path to test image')
    parser.add_argument('--rf-model', default='models/segmentation_model.pkl',
                       help='Path to Random Forest model')
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"✗ Image not found: {args.image}")
        return 1

    if not SAM_AVAILABLE:
        print("✗ SAM not installed!")
        return 1

    comparator = InteractiveComparator(args.image, args.rf_model)
    comparator.run()

    return 0

if __name__ == "__main__":
    sys.exit(main())
