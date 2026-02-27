#!/usr/bin/env python3
"""
Generate documentation images showing the SAM prompt creator workflow.
Creates annotated screenshots for the README.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
from PIL import Image as PILImage
from pathlib import Path
import cv2

DATA_DIR = Path("results/overlap_calibrated")
OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_aligned_rgb(frame_num):
    """Load and align RGB to thermal FOV."""
    rgb_path = DATA_DIR / f"MAX_{frame_num:04d}.JPG"
    if not rgb_path.exists():
        return None
    rgb_full = np.array(PILImage.open(rgb_path))
    h, w = rgb_full.shape[:2]
    fov = 0.7
    ch, cw = int(h * fov), int(w * fov)
    sh, sw = (h - ch) // 2, (w - cw) // 2
    cropped = rgb_full[sh:sh+ch, sw:sw+cw]
    return np.array(PILImage.fromarray(cropped).resize((640, 512), PILImage.Resampling.BILINEAR))


def generate_workflow_overview():
    """Generate a 3-panel workflow overview image."""
    print("Generating workflow overview...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Click points on image
    rgb = load_aligned_rgb(100)
    if rgb is None:
        print("  Skipped (no frame 100)")
        return
    axes[0].imshow(rgb)

    # Simulate ocean clicks (blue dots)
    ocean_pts = [(400, 350), (500, 200), (350, 100), (550, 400)]
    for x, y in ocean_pts:
        axes[0].plot(x, y, 'o', color='#00BFFF', markersize=14, markeredgecolor='white', markeredgewidth=2.5)

    # Simulate land clicks (red X)
    land_pts = [(100, 300), (50, 150), (200, 450)]
    for x, y in land_pts:
        axes[0].plot(x, y, 'x', color='red', markersize=14, markeredgewidth=3)

    axes[0].set_title('Step 1: Click Ocean & Land Points', fontsize=13, fontweight='bold', pad=10)
    axes[0].text(320, 490, 'Left-click = Ocean (blue)    Right-click = Land (red)',
                ha='center', fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    axes[0].axis('off')

    # Panel 2: SAM segments the ocean
    # Create a fake but realistic ocean mask
    mask = np.zeros((512, 640), dtype=bool)
    # Ocean is roughly the right half + bottom
    for y in range(512):
        for x in range(640):
            # Simple diagonal boundary
            if x > 180 + y * 0.1 and not (y > 400 and x < 300):
                r, g, b = rgb[y, x]
                brightness = (int(r) + int(g) + int(b)) / 3
                blue_dom = float(b) / (max(float(r), float(g)) + 1)
                if blue_dom > 1.0 or brightness > 170:
                    mask[y, x] = True

    colored = rgb.copy().astype(float) / 255
    colored[~mask] = colored[~mask] * 0.3  # Dim land
    # Tint ocean slightly blue
    colored[mask] = colored[mask] * 0.7 + np.array([0, 0.15, 0.4]) * 0.3

    axes[1].imshow(colored)
    ocean_pct = mask.sum() / mask.size * 100
    axes[1].set_title(f'Step 2: SAM Segments Ocean ({ocean_pct:.0f}%)', fontsize=13, fontweight='bold', pad=10)
    axes[1].text(320, 490, 'SAM automatically finds ocean boundaries',
                ha='center', fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))
    axes[1].axis('off')

    # Panel 3: Apply to all frames
    # Create a grid of mini thumbnails
    frames = [1, 50, 100, 150, 200, 248]
    inner_ax = axes[2]
    inner_ax.axis('off')
    inner_ax.set_title('Step 3: Batch Process All Frames', fontsize=13, fontweight='bold', pad=10)

    # Create a composite thumbnail grid
    grid_h, grid_w = 2, 3
    tw, th = 160, 128  # thumb width, height
    pad = 8
    composite = np.ones(((th + pad) * grid_h + pad, (tw + pad) * grid_w + pad, 3), dtype=np.uint8) * 40

    for idx, fnum in enumerate(frames):
        row = idx // grid_w
        col = idx % grid_w
        thumb_rgb = load_aligned_rgb(fnum)
        if thumb_rgb is not None:
            thumb = np.array(PILImage.fromarray(thumb_rgb).resize((tw, th), PILImage.Resampling.BILINEAR))
            y_off = row * (th + pad) + pad
            x_off = col * (tw + pad) + pad
            composite[y_off:y_off+th, x_off:x_off+tw] = thumb

    inner_ax.imshow(composite)
    inner_ax.text(composite.shape[1]//2, composite.shape[0] - 5,
                 'Same prompts applied to entire survey',
                 ha='center', va='bottom', fontsize=9, color='white',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='black', alpha=0.7))

    # Add arrows between panels
    for i in range(2):
        fig.patches.append(FancyArrowPatch(
            (0.34 + i * 0.33, 0.5), (0.36 + i * 0.33, 0.5),
            transform=fig.transFigure,
            arrowstyle='->', mutation_scale=30,
            color='#333333', linewidth=3
        ))

    plt.tight_layout(pad=2)
    path = OUTPUT_DIR / "sam_workflow_overview.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def generate_prompt_creator_screenshot():
    """Generate a simulated screenshot of the SAM prompt creator interface."""
    print("Generating prompt creator screenshot...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    rgb = load_aligned_rgb(100)
    if rgb is None:
        return

    # Left panel: Image with click points
    axes[0].imshow(rgb)

    # Ocean points (blue circles)
    ocean_pts = [(420, 300), (500, 180), (380, 80), (560, 420), (480, 250)]
    for x, y in ocean_pts:
        axes[0].plot(x, y, 'o', color='#1E90FF', markersize=16,
                    markeredgecolor='white', markeredgewidth=3, zorder=5)

    # Land points (red X marks)
    land_pts = [(80, 280), (40, 150), (150, 420), (120, 50)]
    for x, y in land_pts:
        axes[0].plot(x, y, 'x', color='#FF4444', markersize=16,
                    markeredgewidth=4, zorder=5)

    axes[0].legend(
        [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E90FF', markersize=12, markeredgecolor='white', markeredgewidth=2),
         plt.Line2D([0], [0], marker='x', color='#FF4444', markersize=12, markeredgewidth=3)],
        [f'Ocean ({len(ocean_pts)})', f'Land ({len(land_pts)})'],
        loc='upper left', fontsize=12, framealpha=0.9
    )
    axes[0].set_title(f'MAX_0100.JPG\nImage 100/250', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # Right panel: Segmented result
    mask = np.zeros((512, 640), dtype=bool)
    for y in range(512):
        for x in range(640):
            if x > 170 + y * 0.15:
                r, g, b = rgb[y, x]
                blue_dom = float(b) / (max(float(r), float(g)) + 1)
                if blue_dom > 0.95 or (int(r) + int(g) + int(b)) / 3 > 170:
                    mask[y, x] = True

    colored = rgb.copy().astype(float) / 255
    colored[~mask] = colored[~mask] * 0.3

    axes[1].imshow(colored)
    ocean_pct = mask.sum() / mask.size * 100

    # Green border (saved state)
    axes[1].add_patch(Rectangle((0, 0), 639, 511, fill=False, edgecolor='green', linewidth=5))
    axes[1].text(320, 40, 'SAVED', ha='center', va='top', fontsize=22, fontweight='bold',
                color='green', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85))
    axes[1].text(320, 85, 'sam_MAX_0100_prompts.json', ha='center', va='top', fontsize=10,
                color='green', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    axes[1].text(320, 130, 'Press P to Process ALL 250 images',
                ha='center', va='top', fontsize=13, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#90EE90', alpha=0.9))
    axes[1].text(320, 175, 'Or press \u2192 to test on more images first',
                ha='center', va='top', fontsize=10, color='darkgreen',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[1].set_title(f'SAM Result: {ocean_pct:.1f}% ocean\nPress W to save', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle('SAM Prompt Creator - Interactive Ocean Segmentation', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(pad=1.5)
    path = OUTPUT_DIR / "sam_prompt_creator.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def generate_controls_reference():
    """Generate a controls reference card."""
    print("Generating controls reference...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'SAM Prompt Creator Controls', ha='center', va='top',
            fontsize=18, fontweight='bold', transform=ax.transAxes)

    controls = [
        ('Left Click', 'Add OCEAN point (blue circle)', '#1E90FF'),
        ('Right Click', 'Add LAND point (red X) to exclude', '#FF4444'),
        ('W', 'Save prompts to JSON file', '#228B22'),
        ('P', 'Batch process ALL images with saved prompts', '#228B22'),
        ('\u2192 / \u2190', 'Test prompts on next/previous image', '#333'),
        ('C', 'Clear all points and start over', '#FF8C00'),
        ('Q', 'Quit', '#666'),
    ]

    y_start = 0.82
    for i, (key, desc, color) in enumerate(controls):
        y = y_start - i * 0.10
        # Key box
        ax.text(0.18, y, key, ha='center', va='center', fontsize=13, fontweight='bold',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#999', linewidth=1.5))
        # Description
        ax.text(0.28, y, desc, ha='left', va='center', fontsize=12,
                color=color, transform=ax.transAxes)

    plt.tight_layout()
    path = OUTPUT_DIR / "sam_controls.png"
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def generate_comparison_image():
    """Generate a before/after comparison: RF vs SAM segmentation."""
    print("Generating RF vs SAM comparison...")

    # Use existing comparison images if available
    comp_dir = Path("results/sam_comparison")
    best_frame = comp_dir / "compare_frame_0248.png"
    good_frame = comp_dir / "compare_frame_0200.png"

    if best_frame.exists() and good_frame.exists():
        # Combine the two best comparison frames
        img1 = np.array(PILImage.open(good_frame))
        img2 = np.array(PILImage.open(best_frame))

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))
        axes[0].imshow(img1)
        axes[0].set_title('Frame 200 - Mixed Coastal Scene', fontsize=13, fontweight='bold')
        axes[0].axis('off')
        axes[1].imshow(img2)
        axes[1].set_title('Frame 248 - Clear Ocean/Land Boundary', fontsize=13, fontweight='bold')
        axes[1].axis('off')

        plt.suptitle('Segmentation Comparison: Random Forest vs SAM', fontsize=16, fontweight='bold')
        plt.tight_layout()
        path = OUTPUT_DIR / "sam_vs_rf_comparison.png"
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {path}")
    else:
        print("  Skipped (no comparison images found)")


if __name__ == "__main__":
    print("=" * 50)
    print("Generating SAM Documentation Images")
    print("=" * 50)

    generate_workflow_overview()
    generate_prompt_creator_screenshot()
    generate_controls_reference()
    generate_comparison_image()

    print("\nDone! Images saved to docs/images/")
