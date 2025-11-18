#!/usr/bin/env python3
"""
Simple SAM Test for M3 Mac
Tests SAM on a single thermal image
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    sys.exit(1)

def get_device():
    """Get best available device (MPS for M3 Mac, CUDA for NVIDIA, or CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def test_sam_on_image(image_path, checkpoint_path="models/sam/sam_vit_b_01ec64.pth"):
    """Test SAM segmentation on a single image"""

    print("="*60)
    print("SAM Simple Test")
    print("="*60)

    # Check device
    device = get_device()
    print(f"\nUsing device: {device}")
    if device == "mps":
        print("✓ Using Apple Silicon GPU acceleration (Metal)")
    elif device == "cuda":
        print("✓ Using NVIDIA GPU acceleration")
    else:
        print("⚠️  Using CPU (will be slower)")

    # Load SAM
    print(f"\nLoading SAM model from: {checkpoint_path}")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("✓ SAM loaded successfully")

    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"✗ Could not load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    print(f"✓ Image loaded: {w}x{h}")

    # Set image for SAM
    print("\nProcessing image with SAM...")
    predictor.set_image(image_rgb)
    print("✓ Image encoded")

    # Create some example points for ocean segmentation
    # For a coastal image, let's try to segment the ocean
    # We'll use points in the middle/lower part of the image (likely ocean)
    print("\nSegmenting ocean with example points...")

    # Ocean points (foreground)
    ocean_points = np.array([
        [w//4, h*3//4],      # Lower left quadrant
        [w//2, h*3//4],      # Lower center
        [w*3//4, h*3//4],    # Lower right quadrant
    ])

    # Point labels (1 = foreground/ocean)
    point_labels = np.array([1, 1, 1])

    # Segment
    masks, scores, logits = predictor.predict(
        point_coords=ocean_points,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Get best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    print(f"✓ Segmentation complete")
    print(f"  Confidence: {best_score:.3f}")
    print(f"  Segmented area: {np.sum(best_mask) / best_mask.size * 100:.1f}% of image")

    # Visualize
    print("\nCreating visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Original with prompt points
    axes[1].imshow(image_rgb)
    axes[1].scatter(ocean_points[:, 0], ocean_points[:, 1],
                   c='blue', s=200, marker='o',
                   edgecolors='white', linewidths=3)
    axes[1].set_title('Prompt Points (Ocean)')
    axes[1].axis('off')

    # Segmentation result
    axes[2].imshow(image_rgb)
    # Create colored overlay
    colored_mask = np.zeros((h, w, 4))
    colored_mask[best_mask] = [0, 0, 1, 0.4]  # Blue with transparency
    axes[2].imshow(colored_mask)
    axes[2].set_title(f'SAM Segmentation (conf: {best_score:.3f})')
    axes[2].axis('off')

    plt.tight_layout()

    # Save result
    output_path = Path("sam_test_result.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")

    # Show
    print("\nShowing result... (close window to continue)")
    plt.show()

    print("\n" + "="*60)
    print("✓ SAM test complete!")
    print("="*60)

def main():
    # Find a test image
    data_dirs = [
        "data/100MEDIA",
        "data/101MEDIA",
        "data/102MEDIA",
    ]

    test_image = None
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if data_path.exists():
            # Find first MAX (RGB) image
            images = list(data_path.glob("MAX_*.JPG"))
            if images:
                test_image = images[0]
                break

    if test_image is None:
        print("✗ No test images found!")
        print(f"Looked in: {', '.join(data_dirs)}")
        print("\nUsage: python scripts/test_sam_simple.py [path/to/image.jpg]")
        return 1

    # Allow command-line override
    if len(sys.argv) > 1:
        test_image = Path(sys.argv[1])

    # Run test
    test_sam_on_image(test_image)

    return 0

if __name__ == "__main__":
    sys.exit(main())
