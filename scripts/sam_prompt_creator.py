#!/usr/bin/env python3
"""
Simple SAM Prompt Creator - Click points, see results, save when happy

Usage:
    python scripts/sam_prompt_creator.py --image data/100MEDIA/MAX_0001.JPG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
import sys
import json
from datetime import datetime

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("✗ SAM not installed! Run: bash scripts/setup_sam.sh")
    sys.exit(1)


class SAMPromptCreator:
    """Simple interactive SAM prompt creator"""

    def __init__(self, image_path):
        self.image_path = Path(image_path)

        # Find all images in same directory for testing
        self.all_images = sorted(self.image_path.parent.glob("MAX_*.JPG"))
        self.current_index = self.all_images.index(self.image_path) if self.image_path in self.all_images else 0

        # SAM setup
        device = self.get_device()
        print(f"Loading SAM (using {device})...")
        sam = sam_model_registry["vit_b"](checkpoint="models/sam/sam_vit_b_01ec64.pth")
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        print("✓ SAM ready\n")

        # Point tracking
        self.ocean_points = []
        self.land_points = []
        self.current_mask = None
        self.saved = False
        self.saved_filename = None

        # Load first image
        self.load_image(self.image_path)

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

    def load_image(self, image_path):
        """Load a new image and apply current prompts"""
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.h, self.w = self.image_rgb.shape[:2]

        # Set image for SAM
        self.sam_predictor.set_image(self.image_rgb)

        # Re-segment with existing prompts if any
        if self.ocean_points or self.land_points:
            self.current_mask = self.run_sam()
        else:
            self.current_mask = None

        print(f"\nLoaded: {self.image_path.name} ({self.current_index + 1}/{len(self.all_images)})")

    def load_next_image(self):
        """Load next image in directory"""
        if self.current_index < len(self.all_images) - 1:
            self.current_index += 1
            self.load_image(self.all_images[self.current_index])
            return True
        else:
            print("Already at last image")
            return False

    def load_prev_image(self):
        """Load previous image in directory"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image(self.all_images[self.current_index])
            return True
        else:
            print("Already at first image")
            return False

    def run_sam(self):
        """Segment with SAM using current points"""
        if len(self.ocean_points) == 0 and len(self.land_points) == 0:
            return None

        # Combine points
        all_points = self.ocean_points + self.land_points
        all_labels = [1] * len(self.ocean_points) + [0] * len(self.land_points)

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
        """Run interactive prompt creator"""
        print("="*70)
        print("SAM PROMPT CREATOR & TESTER")
        print("="*70)
        print("\nWORKFLOW:")
        print("  1. Click ocean/land points on first image")
        print("  2. Press W to save prompts")
        print("  3. Press → to test on next images")
        print("  4. If segmentation looks good on all → Done!")
        print("  5. If not → adjust points or create new prompts")
        print("\nCONTROLS:")
        print("  Left Click:  Add OCEAN point (blue)")
        print("  Right Click: Add LAND point (red) to exclude")
        print("  Press '→':   Next image (test prompts)")
        print("  Press '←':   Previous image")
        print("  Press 'w':   Save prompts")
        print("  Press 'c':   Clear all points")
        print("  Press 'q':   Quit")
        print(f"\nFound {len(self.all_images)} images in directory")
        print("="*70 + "\n")

        # Create figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()
        plt.tight_layout()
        plt.show()

    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.axes[0]:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left click - ocean
            self.ocean_points.append([x, y])
            print(f"Added ocean point at ({x}, {y})")
            self.saved = False  # Mark as unsaved
        elif event.button == 3:  # Right click - land
            self.land_points.append([x, y])
            print(f"Added land point at ({x}, {y})")
            self.saved = False  # Mark as unsaved

        # Auto-segment after adding point
        self.segment()
        self.update_display()

    def on_key(self, event):
        """Handle keyboard"""
        if event.key == 'c':
            self.ocean_points = []
            self.land_points = []
            self.current_mask = None
            self.saved = False
            print("Cleared all points")
            self.update_display()

        elif event.key == 'enter':
            self.segment()

        elif event.key == 'w':
            self.save_prompts()

        elif event.key == 'right':
            # Next image - test prompts
            if self.load_next_image():
                self.update_display()

        elif event.key == 'left':
            # Previous image
            if self.load_prev_image():
                self.update_display()

        elif event.key == 'q':
            plt.close()

    def segment(self):
        """Run segmentation and update"""
        if len(self.ocean_points) > 0 or len(self.land_points) > 0:
            print("Segmenting...")
            self.current_mask = self.run_sam()
            self.update_display()

    def save_prompts(self):
        """Save SAM prompts to JSON file"""
        if not self.ocean_points and not self.land_points:
            print("⚠️  No points to save! Add some points first.")
            self.update_display()  # Refresh to show warning
            return

        # Create prompts directory
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)

        # Generate filename based on image name
        image_stem = self.image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = prompts_dir / f"sam_{image_stem}_{timestamp}.json"

        # Format prompts
        prompts = {}

        if self.ocean_points:
            prompts['ocean'] = {
                'points': self.ocean_points,
                'labels': [1] * len(self.ocean_points)
            }

        if self.land_points:
            prompts['land'] = {
                'points': self.land_points,
                'labels': [1] * len(self.land_points)
            }

        # Save
        with open(filename, 'w') as f:
            json.dump(prompts, f, indent=2)

        self.saved = True
        self.saved_filename = filename

        print(f"\n✓ Saved prompts to: {filename}")
        print(f"  Ocean points: {len(self.ocean_points)}")
        print(f"  Land points: {len(self.land_points)}")

        self.update_display()  # Refresh to show saved status

    def update_display(self):
        """Update the display"""
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Left: Interactive image with points
        self.axes[0].imshow(self.image_rgb)
        title = f'{self.image_path.name}\nImage {self.current_index + 1}/{len(self.all_images)}'
        self.axes[0].set_title(title, fontsize=12, fontweight='bold')

        # Draw points
        if self.ocean_points:
            points = np.array(self.ocean_points)
            self.axes[0].scatter(points[:, 0], points[:, 1],
                               c='blue', s=200, marker='o',
                               edgecolors='white', linewidths=3,
                               label=f'Ocean ({len(self.ocean_points)})')

        if self.land_points:
            points = np.array(self.land_points)
            self.axes[0].scatter(points[:, 0], points[:, 1],
                               c='red', s=200, marker='x',
                               linewidths=3,
                               label=f'Land ({len(self.land_points)})')

        if self.ocean_points or self.land_points:
            self.axes[0].legend(loc='upper left', fontsize=12)

        self.axes[0].axis('off')

        # Right: Segmentation result
        if self.current_mask is not None:
            # Show result
            colored = self.image_rgb.copy()
            colored[~self.current_mask] = colored[~self.current_mask] * 0.3

            self.axes[1].imshow(colored)
            ocean_pct = np.sum(self.current_mask) / self.current_mask.size * 100

            # Show title with testing status
            if self.saved and self.current_index > 0:
                title = f'TESTING: {ocean_pct:.1f}% ocean\nPress → for next image'
            elif self.saved:
                title = f'SAVED: {ocean_pct:.1f}% ocean\nPress → to test on next images'
            else:
                title = f'SAM Result: {ocean_pct:.1f}% ocean\nPress W to save'

            self.axes[1].set_title(title, fontsize=12, fontweight='bold')

            # Add save/testing status indicators
            if self.saved and self.current_index > 0:
                # Blue border for testing mode
                self.axes[1].add_patch(Rectangle((0, 0), self.w, self.h,
                                                fill=False, edgecolor='cyan',
                                                linewidth=6))
                # Testing indicator
                self.axes[1].text(self.w/2, 50, '🧪 TESTING PROMPTS',
                                ha='center', va='top',
                                fontsize=18, fontweight='bold',
                                color='blue',
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='white', alpha=0.9))
                # Show navigation
                self.axes[1].text(self.w/2, 100,
                                f'Image {self.current_index + 1}/{len(self.all_images)} | ← → to navigate',
                                ha='center', va='top',
                                fontsize=11, color='blue',
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', alpha=0.8))
            elif self.saved:
                # Green border for saved
                self.axes[1].add_patch(Rectangle((0, 0), self.w, self.h,
                                                fill=False, edgecolor='green',
                                                linewidth=6))
                # Save confirmation text
                self.axes[1].text(self.w/2, 50, '✓ SAVED',
                                ha='center', va='top',
                                fontsize=20, fontweight='bold',
                                color='green',
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='white', alpha=0.8))
                if self.saved_filename:
                    self.axes[1].text(self.w/2, 100,
                                    f'{self.saved_filename.name}',
                                    ha='center', va='top',
                                    fontsize=10, color='green',
                                    bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='white', alpha=0.8))
                # Testing prompt
                self.axes[1].text(self.w/2, 150,
                                'Press → to test on other images',
                                ha='center', va='top',
                                fontsize=12, color='green',
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='white', alpha=0.8))
            elif self.ocean_points or self.land_points:
                # Yellow border for unsaved changes
                self.axes[1].add_patch(Rectangle((0, 0), self.w, self.h,
                                                fill=False, edgecolor='yellow',
                                                linewidth=6))
                self.axes[1].text(self.w/2, 50, 'Press W to Save',
                                ha='center', va='top',
                                fontsize=16,
                                color='orange',
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='white', alpha=0.8))

        else:
            # No segmentation yet
            self.axes[1].text(0.5, 0.5,
                            'Click ocean points\n(left click)\n\nSegmentation updates automatically',
                            ha='center', va='center',
                            transform=self.axes[1].transAxes,
                            fontsize=14, style='italic')
            self.axes[1].set_xlim(0, self.w)
            self.axes[1].set_ylim(self.h, 0)

        self.axes[1].axis('off')

        # Update window title with save status
        status = "✓ SAVED" if self.saved else "⚠ UNSAVED" if self.ocean_points else ""
        self.fig.canvas.manager.set_window_title(f"SAM Prompt Creator - {self.image_path.name} {status}")

        plt.draw()


def main():
    parser = argparse.ArgumentParser(description='SAM prompt creator')
    parser.add_argument('--image', required=True, help='Path to image')
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"✗ Image not found: {args.image}")
        return 1

    creator = SAMPromptCreator(args.image)
    creator.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
