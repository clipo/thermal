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
        print("SAM PROMPT CREATOR & BATCH PROCESSOR")
        print("="*70)
        print("\nWORKFLOW:")
        print("  1. Click ocean/land points on first image")
        print("  2. Press W to save prompts")
        print("  3. Press P to process ALL images → DONE!")
        print("  (Or press → to test on a few more images first)")
        print("\nCONTROLS:")
        print("  Left Click:  Add OCEAN point (blue)")
        print("  Right Click: Add LAND point (red) to exclude")
        print("  Press 'w':   Save prompts")
        print("  Press 'p':   Process ALL images with saved prompts")
        print("  Press '→':   Next image (test prompts)")
        print("  Press '←':   Previous image")
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

        elif event.key == 'p':
            # Batch process all images
            if self.saved:
                plt.close()  # Close GUI
                self.batch_process_all()
            else:
                print("⚠️  Save prompts first (press W)")

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
        print(f"\nFound {len(self.all_images)} images in this directory")
        print(f"You can now:")
        print(f"  1. Press 'p' to Process ALL images with these prompts")
        print(f"  2. Press → to test on a few more images first")
        print(f"  3. Press 'q' to quit and run detection manually")

        self.update_display()  # Refresh to show saved status

    def batch_process_all(self):
        """Process all images with saved prompts"""
        if not self.saved:
            print("⚠️  Save prompts first (press W)")
            return

        print("\n" + "="*70)
        print(f"BATCH PROCESSING: {len(self.all_images)} images")
        print("="*70)
        print(f"Using prompts: {self.saved_filename}")
        print(f"Ocean points: {len(self.ocean_points)}")
        print(f"Land points: {len(self.land_points)}")
        print()

        # Ask for confirmation
        response = input(f"Process all {len(self.all_images)} images? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Cancelled batch processing")
            return

        # Ask for output directory
        output_dir = input("Output directory [sgd_output]: ").strip() or "sgd_output"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process all images
        print(f"\nProcessing {len(self.all_images)} images...")
        print("This may take a while...\n")

        results = []
        for i, image_path in enumerate(self.all_images, 1):
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"⚠️  [{i}/{len(self.all_images)}] Failed to load: {image_path.name}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Set image for SAM
            self.sam_predictor.set_image(img_rgb)

            # Segment with current prompts
            all_points = self.ocean_points + self.land_points
            all_labels = [1] * len(self.ocean_points) + [0] * len(self.land_points)

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array(all_points),
                point_labels=np.array(all_labels),
                multimask_output=True,
            )

            # Use best mask
            best_idx = np.argmax(scores)
            ocean_mask = masks[best_idx]

            # Calculate ocean percentage
            ocean_pct = np.sum(ocean_mask) / ocean_mask.size * 100

            # Save mask
            mask_filename = output_path / f"mask_{image_path.stem}.npy"
            np.save(mask_filename, ocean_mask)

            results.append({
                'image': image_path.name,
                'ocean_percent': ocean_pct,
                'mask_file': str(mask_filename)
            })

            # Progress
            if i % 10 == 0 or i == len(self.all_images):
                print(f"[{i}/{len(self.all_images)}] {image_path.name}: {ocean_pct:.1f}% ocean")

        # Save results summary
        summary_file = output_path / "segmentation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'prompts_file': str(self.saved_filename),
                'total_images': len(self.all_images),
                'processed': len(results),
                'results': results
            }, f, indent=2)

        print("\n" + "="*70)
        print("✓ BATCH PROCESSING COMPLETE!")
        print("="*70)
        print(f"Processed: {len(results)}/{len(self.all_images)} images")
        print(f"Masks saved to: {output_path}/")
        print(f"Summary: {summary_file}")
        print()
        print("Next step: Run SGD detection with these masks")
        print("="*70 + "\n")

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
                title = f'TESTING: {ocean_pct:.1f}% ocean\n⚡ Press P to process ALL | → for next image'
            elif self.saved:
                title = f'SAVED: {ocean_pct:.1f}% ocean\n⚡ Press P to process ALL | → to test first'
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
                # BIG CALL TO ACTION - Process All
                self.axes[1].text(self.w/2, 160,
                                f'⚡ Press P to Process ALL {len(self.all_images)} Images ⚡',
                                ha='center', va='top',
                                fontsize=15, fontweight='bold',
                                color='white',
                                bbox=dict(boxstyle='round,pad=0.7',
                                        facecolor='darkgreen', alpha=0.95, edgecolor='yellow', linewidth=3))
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
                # Main action prompt - process all
                self.axes[1].text(self.w/2, 150,
                                f'Press P to Process ALL {len(self.all_images)} images',
                                ha='center', va='top',
                                fontsize=14, fontweight='bold', color='green',
                                bbox=dict(boxstyle='round,pad=0.5',
                                        facecolor='lightgreen', alpha=0.9))
                # Secondary option - test first
                self.axes[1].text(self.w/2, 200,
                                'Or press → to test on a few more images first',
                                ha='center', va='top',
                                fontsize=11, color='darkgreen',
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
