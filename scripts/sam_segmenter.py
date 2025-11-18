#!/usr/bin/env python3
"""
SAM-based Ocean/Land Segmentation for Thermal Imagery

Uses Meta's Segment Anything Model (SAM) to segment ocean, land, rock, and waves
from RGB imagery, then transfers masks to aligned thermal images.

Usage:
    # Interactive mode - create prompts
    python sam_segmenter.py --interactive --data data/100MEDIA

    # Batch mode - use saved prompts
    python sam_segmenter.py --data data/100MEDIA --prompts prompts/coastal_rocky.json

    # Test on single image
    python sam_segmenter.py --test --image data/100MEDIA/MAX_0001.JPG
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime
import sys

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  SAM not installed. Run: bash scripts/setup_sam.sh")


class SAMSegmenter:
    """Segment Anything Model for ocean/land segmentation"""

    def __init__(self, model_type='vit_h', checkpoint_path=None):
        """
        Initialize SAM segmenter

        Args:
            model_type: 'vit_h', 'vit_l', or 'vit_b'
            checkpoint_path: Path to SAM checkpoint file
        """
        if not SAM_AVAILABLE:
            raise ImportError("SAM not installed. Run: bash scripts/setup_sam.sh")

        # Auto-detect checkpoint if not provided
        if checkpoint_path is None:
            checkpoint_path = self.find_checkpoint(model_type)

        print(f"Loading SAM model: {model_type}")
        print(f"Checkpoint: {checkpoint_path}")

        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

        # Move to GPU if available
        device = "cuda" if self._has_cuda() else "cpu"
        sam.to(device=device)
        print(f"Using device: {device}")

        self.predictor = SamPredictor(sam)
        self.current_image = None
        self.classes = ['ocean', 'land', 'rock', 'wave']
        self.class_colors = {
            'ocean': [0, 0, 255],      # Blue
            'land': [0, 255, 0],       # Green
            'rock': [128, 128, 128],   # Gray
            'wave': [255, 255, 255]    # White
        }

    def _has_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def find_checkpoint(self, model_type):
        """Auto-detect SAM checkpoint file"""
        models_dir = Path("models/sam")

        checkpoint_names = {
            'vit_h': 'sam_vit_h_4b8939.pth',
            'vit_l': 'sam_vit_l_0b3195.pth',
            'vit_b': 'sam_vit_b_01ec64.pth'
        }

        checkpoint = models_dir / checkpoint_names[model_type]

        if not checkpoint.exists():
            # Try to find any .pth file
            pth_files = list(models_dir.glob("*.pth"))
            if pth_files:
                print(f"⚠️  Using checkpoint: {pth_files[0]}")
                return str(pth_files[0])

            raise FileNotFoundError(
                f"SAM checkpoint not found at {checkpoint}\n"
                f"Run: bash scripts/setup_sam.sh"
            )

        return str(checkpoint)

    def load_image(self, image_path):
        """Load and prepare image for SAM"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image for SAM
        self.predictor.set_image(image)
        self.current_image = image

        return image

    def segment_with_points(self, point_coords, point_labels):
        """
        Segment image with point prompts

        Args:
            point_coords: Nx2 array of (x, y) coordinates
            point_labels: N array of labels (1=foreground, 0=background)

        Returns:
            masks: Segmentation masks
            scores: Confidence scores
        """
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Return best mask (highest score)
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_with_box(self, box):
        """
        Segment image with box prompt

        Args:
            box: [x1, y1, x2, y2] bounding box

        Returns:
            mask: Segmentation mask
            score: Confidence score
        """
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False,
        )

        return masks[0], scores[0]

    def create_semantic_mask(self, class_prompts):
        """
        Create semantic segmentation from class prompts

        Args:
            class_prompts: Dict mapping class names to point/box prompts
                Example: {
                    'ocean': {'points': [[100, 200], [150, 250]], 'labels': [1, 1]},
                    'land': {'points': [[400, 300]], 'labels': [1]}
                }

        Returns:
            semantic_mask: H x W mask with class indices
            confidence_map: H x W confidence scores
        """
        h, w = self.current_image.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)

        # Segment each class
        for class_idx, class_name in enumerate(self.classes):
            if class_name not in class_prompts:
                continue

            prompt = class_prompts[class_name]

            # Get mask for this class
            if 'points' in prompt:
                points = np.array(prompt['points'])
                labels = np.array(prompt['labels'])
                mask, score = self.segment_with_points(points, labels)
            elif 'box' in prompt:
                mask, score = self.segment_with_box(prompt['box'])
            else:
                continue

            # Update semantic mask (only where confidence is higher)
            update_mask = mask & (score > confidence_map)
            semantic_mask[update_mask] = class_idx
            confidence_map[update_mask] = score

        return semantic_mask, confidence_map

    def visualize_segmentation(self, semantic_mask, alpha=0.5):
        """Visualize segmentation overlay on image"""
        h, w = semantic_mask.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        for class_idx, class_name in enumerate(self.classes):
            mask = semantic_mask == class_idx
            overlay[mask] = self.class_colors[class_name]

        # Blend with original image
        result = cv2.addWeighted(self.current_image, 1-alpha, overlay, alpha, 0)

        return result

    def save_mask(self, semantic_mask, output_path):
        """Save segmentation mask"""
        # Save as numpy array
        np.save(output_path, semantic_mask)

        # Also save as colored PNG for visualization
        h, w = semantic_mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_idx, class_name in enumerate(self.classes):
            mask = semantic_mask == class_idx
            colored[mask] = self.class_colors[class_name]

        png_path = Path(output_path).with_suffix('.png')
        cv2.imwrite(str(png_path), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

        print(f"✓ Saved mask: {output_path}")
        print(f"✓ Saved visualization: {png_path}")


class InteractivePrompter:
    """Interactive tool to create prompts for SAM"""

    def __init__(self, segmenter, image_path):
        self.segmenter = segmenter
        self.image = segmenter.load_image(image_path)
        self.prompts = {class_name: {'points': [], 'labels': []}
                       for class_name in segmenter.classes}
        self.current_class = 'ocean'

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        self.ax_image, self.ax_points, self.ax_result = self.axes

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_display()

    def on_click(self, event):
        """Handle mouse clicks to add points"""
        if event.inaxes != self.ax_image:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Left click = foreground (1), Right click = background (0)
        label = 1 if event.button == 1 else 0

        self.prompts[self.current_class]['points'].append([x, y])
        self.prompts[self.current_class]['labels'].append(label)

        print(f"Added {'foreground' if label == 1 else 'background'} point for {self.current_class} at ({x}, {y})")

        self.update_display()

    def on_key(self, event):
        """Handle keyboard shortcuts"""
        key = event.key

        # Class selection
        if key == '1':
            self.current_class = 'ocean'
        elif key == '2':
            self.current_class = 'land'
        elif key == '3':
            self.current_class = 'rock'
        elif key == '4':
            self.current_class = 'wave'

        # Actions
        elif key == 's':
            self.segment_and_show()
        elif key == 'c':
            self.prompts[self.current_class] = {'points': [], 'labels': []}
            print(f"Cleared prompts for {self.current_class}")
        elif key == 'w':
            self.save_prompts()
        elif key == 'q':
            plt.close()
            return

        print(f"Current class: {self.current_class}")
        self.update_display()

    def update_display(self):
        """Update the display"""
        # Clear axes
        for ax in self.axes:
            ax.clear()

        # Show original image
        self.ax_image.imshow(self.image)
        self.ax_image.set_title(f'Click to add {self.current_class} points\n(Left=foreground, Right=background)')
        self.ax_image.axis('off')

        # Show points
        self.ax_points.imshow(self.image)
        for class_name, color in self.segmenter.class_colors.items():
            points = self.prompts[class_name]['points']
            labels = self.prompts[class_name]['labels']
            if points:
                points = np.array(points)
                fg_points = points[np.array(labels) == 1]
                bg_points = points[np.array(labels) == 0]

                if len(fg_points) > 0:
                    self.ax_points.scatter(fg_points[:, 0], fg_points[:, 1],
                                          c=[color], s=200, marker='o',
                                          edgecolors='white', linewidths=2,
                                          label=f'{class_name} (fg)')
                if len(bg_points) > 0:
                    self.ax_points.scatter(bg_points[:, 0], bg_points[:, 1],
                                          c=[color], s=200, marker='x',
                                          linewidths=3,
                                          label=f'{class_name} (bg)')

        self.ax_points.set_title('Prompts')
        self.ax_points.axis('off')
        self.ax_points.legend(loc='upper right', fontsize=8)

        # Instructions
        self.ax_result.text(0.1, 0.9, """
KEYBOARD SHORTCUTS:
  1       Ocean mode
  2       Land mode
  3       Rock mode
  4       Wave mode

  S       Segment and show
  C       Clear current class
  W       Save prompts
  Q       Quit

MOUSE:
  Left    Add foreground point
  Right   Add background point
        """, transform=self.ax_result.transAxes,
        fontsize=10, family='monospace', va='top')
        self.ax_result.axis('off')

        plt.draw()

    def segment_and_show(self):
        """Segment with current prompts and show result"""
        print("Segmenting...")

        # Convert prompts format
        class_prompts = {}
        for class_name, prompt in self.prompts.items():
            if prompt['points']:
                class_prompts[class_name] = prompt

        if not class_prompts:
            print("No prompts defined. Add points first!")
            return

        # Segment
        semantic_mask, confidence = self.segmenter.create_semantic_mask(class_prompts)

        # Show result
        result = self.segmenter.visualize_segmentation(semantic_mask, alpha=0.4)
        self.ax_result.clear()
        self.ax_result.imshow(result)
        self.ax_result.set_title('Segmentation Result')
        self.ax_result.axis('off')

        # Print statistics
        unique, counts = np.unique(semantic_mask, return_counts=True)
        total = semantic_mask.size
        print("\nSegmentation Statistics:")
        for class_idx, count in zip(unique, counts):
            class_name = self.segmenter.classes[class_idx]
            pct = count / total * 100
            print(f"  {class_name:8s}: {pct:5.1f}%")

        plt.draw()

    def save_prompts(self):
        """Save prompts to file"""
        # Create prompts directory
        prompts_dir = Path("prompts")
        prompts_dir.mkdir(exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = prompts_dir / f"sam_prompts_{timestamp}.json"

        # Save
        with open(filename, 'w') as f:
            json.dump(self.prompts, f, indent=2)

        print(f"✓ Saved prompts to: {filename}")

    def run(self):
        """Run the interactive prompter"""
        print("\nInteractive SAM Prompter")
        print("=" * 50)
        print("Click on the image to add foreground/background points")
        print("Press S to segment, W to save prompts, Q to quit")
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='SAM-based segmentation for thermal imagery',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', help='Data directory with images')
    parser.add_argument('--image', help='Single image to process')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode to create prompts')
    parser.add_argument('--test', action='store_true',
                       help='Test SAM installation')
    parser.add_argument('--prompts', help='JSON file with saved prompts')
    parser.add_argument('--model', default='vit_h',
                       choices=['vit_h', 'vit_l', 'vit_b'],
                       help='SAM model size')
    parser.add_argument('--checkpoint', help='Path to SAM checkpoint')

    args = parser.parse_args()

    if not SAM_AVAILABLE:
        print("❌ SAM not installed!")
        print("\nInstall with:")
        print("  bash scripts/setup_sam.sh")
        return 1

    # Initialize SAM
    segmenter = SAMSegmenter(
        model_type=args.model,
        checkpoint_path=args.checkpoint
    )

    if args.test:
        print("\n✓ SAM is installed and working!")
        print(f"✓ Using device: {'CUDA' if segmenter._has_cuda() else 'CPU'}")
        if segmenter._has_cuda():
            import torch
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        return 0

    if args.interactive:
        # Interactive mode
        if args.image:
            image_path = args.image
        elif args.data:
            # Find first RGB image
            data_dir = Path(args.data)
            rgb_images = list(data_dir.glob("MAX_*.JPG"))
            if not rgb_images:
                print(f"❌ No MAX_*.JPG images found in {data_dir}")
                return 1
            image_path = rgb_images[0]
        else:
            print("❌ Specify --image or --data for interactive mode")
            return 1

        print(f"Loading image: {image_path}")
        prompter = InteractivePrompter(segmenter, image_path)
        prompter.run()

    else:
        print("Usage:")
        print("  Interactive: python sam_segmenter.py --interactive --data data/100MEDIA")
        print("  Test:        python sam_segmenter.py --test")

    return 0


if __name__ == '__main__':
    sys.exit(main())
