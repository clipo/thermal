#!/usr/bin/env python3
"""
Compare Random Forest vs SAM Segmentation

Tests both segmentation approaches on a single image and displays results side-by-side.

Usage:
    # Compare on single image
    python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG

    # Use specific models
    python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG \
        --rf-model models/segmentation_model.pkl \
        --sam-prompts prompts/sam_prompts_20250118_120000.json

    # Interactive mode - create SAM prompts on the fly
    python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG \
        --interactive
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import json

# Import existing Random Forest segmenter
try:
    from sgd_toolkit.segmentation.ml_segmenter import FastMLSegmenter
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False
    print("⚠️  Random Forest segmenter not found")

# Import SAM segmenter
try:
    from sam_segmenter import SAMSegmenter, InteractivePrompter
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not installed. Run: bash scripts/setup_sam.sh")


class SegmentationComparator:
    """Compare Random Forest and SAM segmentation approaches"""

    def __init__(self, image_path, rf_model_path=None, sam_prompts_path=None):
        self.image_path = Path(image_path)
        self.rf_model_path = rf_model_path
        self.sam_prompts_path = sam_prompts_path

        # Load image
        self.image = cv2.imread(str(self.image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        print(f"Loaded image: {self.image_path}")
        print(f"Image size: {self.image.shape[1]}x{self.image.shape[0]}")

        # Initialize segmenters
        self.rf_segmenter = None
        self.sam_segmenter = None

        if RF_AVAILABLE and rf_model_path:
            self.rf_segmenter = self._init_rf_segmenter(rf_model_path)

        if SAM_AVAILABLE:
            self.sam_segmenter = self._init_sam_segmenter()

    def _init_rf_segmenter(self, model_path):
        """Initialize Random Forest segmenter"""
        try:
            segmenter = FastMLSegmenter()
            segmenter.load_model(model_path)
            print(f"✓ Loaded Random Forest model: {model_path}")
            return segmenter
        except Exception as e:
            print(f"✗ Failed to load Random Forest model: {e}")
            return None

    def _init_sam_segmenter(self):
        """Initialize SAM segmenter"""
        try:
            segmenter = SAMSegmenter(model_type='vit_h')
            segmenter.load_image(self.image_path)
            print(f"✓ Initialized SAM segmenter")
            return segmenter
        except Exception as e:
            print(f"✗ Failed to initialize SAM: {e}")
            return None

    def segment_with_rf(self):
        """Segment using Random Forest"""
        if not self.rf_segmenter:
            print("✗ Random Forest segmenter not available")
            return None

        print("\nRunning Random Forest segmentation...")
        try:
            # Segment image
            mask = self.rf_segmenter.segment_frame(self.image)

            # Convert to semantic mask (0=ocean, 1=land, 2=rock, 3=wave)
            # Assuming RF returns binary mask where 1=ocean
            semantic_mask = np.zeros(mask.shape, dtype=np.uint8)
            semantic_mask[mask > 0] = 0  # Ocean

            print("✓ Random Forest segmentation complete")
            return semantic_mask

        except Exception as e:
            print(f"✗ Random Forest segmentation failed: {e}")
            return None

    def segment_with_sam(self, prompts=None, interactive=False):
        """Segment using SAM"""
        if not self.sam_segmenter:
            print("✗ SAM segmenter not available")
            return None

        # Interactive mode - create prompts
        if interactive:
            print("\nLaunching interactive prompt creator...")
            prompter = InteractivePrompter(self.sam_segmenter, self.image_path)
            prompter.run()
            prompts = prompter.prompts

        # Load prompts from file
        elif self.sam_prompts_path:
            try:
                with open(self.sam_prompts_path, 'r') as f:
                    prompts = json.load(f)
                print(f"✓ Loaded SAM prompts: {self.sam_prompts_path}")
            except Exception as e:
                print(f"✗ Failed to load prompts: {e}")
                return None

        # Need prompts
        if not prompts:
            print("✗ No SAM prompts provided. Use --interactive or --sam-prompts")
            return None

        print("\nRunning SAM segmentation...")
        try:
            # Convert prompts to SAM format
            class_prompts = {}
            for class_name, prompt in prompts.items():
                if prompt['points']:
                    class_prompts[class_name] = prompt

            # Segment
            semantic_mask, confidence = self.sam_segmenter.create_semantic_mask(class_prompts)

            print("✓ SAM segmentation complete")
            return semantic_mask

        except Exception as e:
            print(f"✗ SAM segmentation failed: {e}")
            return None

    def visualize_comparison(self, rf_mask, sam_mask):
        """Display side-by-side comparison"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        # Random Forest result
        if rf_mask is not None:
            rf_overlay = self._create_overlay(self.image_rgb, rf_mask)
            axes[0, 1].imshow(rf_overlay)
            axes[0, 1].set_title('Random Forest Segmentation', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')

            # RF mask only
            rf_colored = self._colorize_mask(rf_mask)
            axes[1, 1].imshow(rf_colored)
            axes[1, 1].set_title('Random Forest Mask', fontsize=12)
            axes[1, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'RF Not Available', ha='center', va='center',
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].axis('off')
            axes[1, 1].axis('off')

        # SAM result
        if sam_mask is not None:
            sam_overlay = self._create_overlay(self.image_rgb, sam_mask)
            axes[0, 2].imshow(sam_overlay)
            axes[0, 2].set_title('SAM Segmentation', fontsize=14, fontweight='bold')
            axes[0, 2].axis('off')

            # SAM mask only
            sam_colored = self._colorize_mask(sam_mask)
            axes[1, 2].imshow(sam_colored)
            axes[1, 2].set_title('SAM Mask', fontsize=12)
            axes[1, 2].axis('off')
        else:
            axes[0, 2].text(0.5, 0.5, 'SAM Not Available', ha='center', va='center',
                           transform=axes[0, 2].transAxes, fontsize=14)
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')

        # Statistics comparison
        self._plot_statistics(axes[1, 0], rf_mask, sam_mask)

        plt.tight_layout()
        plt.show()

    def _create_overlay(self, image, mask, alpha=0.4):
        """Create overlay of mask on image"""
        colored_mask = self._colorize_mask(mask)
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        return overlay

    def _colorize_mask(self, mask):
        """Convert semantic mask to colored image"""
        colors = {
            0: [0, 0, 255],      # Ocean - Blue
            1: [0, 255, 0],      # Land - Green
            2: [128, 128, 128],  # Rock - Gray
            3: [255, 255, 255]   # Wave - White
        }

        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, color in colors.items():
            colored[mask == class_idx] = color

        return colored

    def _plot_statistics(self, ax, rf_mask, sam_mask):
        """Plot statistics comparison"""
        ax.axis('off')

        stats_text = "SEGMENTATION STATISTICS\n" + "="*40 + "\n\n"

        class_names = ['Ocean', 'Land', 'Rock', 'Wave']

        # Random Forest stats
        if rf_mask is not None:
            stats_text += "Random Forest:\n"
            unique, counts = np.unique(rf_mask, return_counts=True)
            total = rf_mask.size
            for class_idx, count in zip(unique, counts):
                pct = count / total * 100
                stats_text += f"  {class_names[class_idx]:8s}: {pct:5.1f}%\n"
            stats_text += "\n"
        else:
            stats_text += "Random Forest: Not available\n\n"

        # SAM stats
        if sam_mask is not None:
            stats_text += "SAM:\n"
            unique, counts = np.unique(sam_mask, return_counts=True)
            total = sam_mask.size
            for class_idx, count in zip(unique, counts):
                pct = count / total * 100
                stats_text += f"  {class_names[class_idx]:8s}: {pct:5.1f}%\n"
            stats_text += "\n"
        else:
            stats_text += "SAM: Not available\n\n"

        # Difference
        if rf_mask is not None and sam_mask is not None:
            stats_text += "\nDifference:\n"
            rf_unique, rf_counts = np.unique(rf_mask, return_counts=True)
            sam_unique, sam_counts = np.unique(sam_mask, return_counts=True)

            rf_dict = dict(zip(rf_unique, rf_counts / rf_mask.size * 100))
            sam_dict = dict(zip(sam_unique, sam_counts / sam_mask.size * 100))

            for class_idx in range(4):
                rf_pct = rf_dict.get(class_idx, 0)
                sam_pct = sam_dict.get(class_idx, 0)
                diff = sam_pct - rf_pct
                sign = "+" if diff > 0 else ""
                stats_text += f"  {class_names[class_idx]:8s}: {sign}{diff:5.1f}%\n"

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=10, family='monospace', va='top')

    def compare(self, interactive=False):
        """Run full comparison"""
        print("\n" + "="*60)
        print("SEGMENTATION COMPARISON: Random Forest vs SAM")
        print("="*60)

        # Run Random Forest
        rf_mask = self.segment_with_rf() if self.rf_segmenter else None

        # Run SAM
        sam_mask = self.segment_with_sam(interactive=interactive)

        # Visualize
        if rf_mask is not None or sam_mask is not None:
            self.visualize_comparison(rf_mask, sam_mask)
        else:
            print("\n✗ No segmentation results available")

        print("\n" + "="*60)
        print("Comparison complete")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Compare Random Forest vs SAM segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison (needs pre-trained RF model and SAM prompts)
  python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG \\
      --rf-model models/segmentation_model.pkl \\
      --sam-prompts prompts/sam_prompts.json

  # Interactive mode - create SAM prompts on the fly
  python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG \\
      --rf-model models/segmentation_model.pkl \\
      --interactive

  # Just test SAM (no RF comparison)
  python scripts/compare_segmentation.py --image data/100MEDIA/MAX_0001.JPG \\
      --interactive
        """
    )

    parser.add_argument('--image', required=True,
                       help='Path to RGB image to segment')
    parser.add_argument('--rf-model',
                       default='models/segmentation_model.pkl',
                       help='Path to Random Forest model')
    parser.add_argument('--sam-prompts',
                       help='Path to SAM prompts JSON file')
    parser.add_argument('--interactive', action='store_true',
                       help='Create SAM prompts interactively')

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"✗ Image not found: {args.image}")
        return 1

    # Check if RF model exists
    rf_model_path = None
    if RF_AVAILABLE:
        if Path(args.rf_model).exists():
            rf_model_path = args.rf_model
        else:
            print(f"⚠️  Random Forest model not found: {args.rf_model}")
            print("    Comparison will only show SAM results")

    # Create comparator
    comparator = SegmentationComparator(
        image_path=args.image,
        rf_model_path=rf_model_path,
        sam_prompts_path=args.sam_prompts
    )

    # Run comparison
    comparator.compare(interactive=args.interactive)

    return 0


if __name__ == '__main__':
    sys.exit(main())
