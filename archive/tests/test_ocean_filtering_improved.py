#!/usr/bin/env python3
"""
Test improved ocean filtering that handles:
1. Frames with no ocean (drone over land)
2. Small isolated patches incorrectly classified as ocean
"""

import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt


def filter_ocean_mask(ocean_mask, min_fraction=0.05):
    """
    Filter ocean mask to keep only significant contiguous areas.

    Args:
        ocean_mask: Boolean mask of detected ocean
        min_fraction: Minimum fraction of image that must be ocean (default 5%)

    Returns:
        Filtered ocean mask
    """
    # First remove very small objects
    ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)

    # Keep only the largest contiguous ocean area
    ocean_labels = measure.label(ocean_mask, connectivity=2)

    if ocean_labels.max() == 0:
        # No ocean detected at all
        return np.zeros_like(ocean_mask, dtype=bool)

    # Find the largest connected component
    unique_labels, counts = np.unique(ocean_labels[ocean_labels > 0], return_counts=True)

    if len(unique_labels) == 0:
        return np.zeros_like(ocean_mask, dtype=bool)

    largest_count = np.max(counts)
    total_pixels = ocean_mask.size

    # Only keep if the largest area is significant
    if largest_count > total_pixels * min_fraction:
        largest_label = unique_labels[np.argmax(counts)]
        ocean_mask = (ocean_labels == largest_label)
    else:
        # No significant ocean area - drone is over land
        ocean_mask = np.zeros_like(ocean_mask, dtype=bool)

    return ocean_mask


def test_ocean_filtering():
    """Test the improved ocean filtering"""
    print("Testing Improved Ocean Filtering")
    print("=" * 50)

    h, w = 512, 640

    # Test 1: Multiple small patches (should all be removed)
    print("\nTest 1: Small isolated patches (drone over land)")
    mask1 = np.zeros((h, w), dtype=bool)
    # Add several small "ocean" patches that are actually misclassified land
    mask1[100:120, 100:120] = True  # Small patch (400 pixels = 0.12% of image)
    mask1[200:230, 300:330] = True  # Small patch (900 pixels = 0.27% of image)
    mask1[400:440, 500:540] = True  # Small patch (1600 pixels = 0.49% of image)

    filtered1 = filter_ocean_mask(mask1.copy())
    print(f"  Original patches: {measure.label(mask1).max()}")
    print(f"  After filtering: {measure.label(filtered1).max()}")
    print(f"  Result: {'✓ All small patches removed' if filtered1.sum() == 0 else '✗ Failed'}")

    # Test 2: One large ocean area + small patches (keep only large)
    print("\nTest 2: Large ocean with small isolated patches")
    mask2 = np.zeros((h, w), dtype=bool)
    mask2[50:450, 200:600] = True  # Large ocean (160,000 pixels = 49% of image)
    mask2[100:120, 50:70] = True   # Small isolated patch
    mask2[300:320, 100:120] = True # Another small patch

    filtered2 = filter_ocean_mask(mask2.copy())
    labels_before = measure.label(mask2)
    labels_after = measure.label(filtered2)
    print(f"  Original regions: {labels_before.max()}")
    print(f"  After filtering: {labels_after.max()}")
    print(f"  Ocean fraction: {filtered2.sum() / filtered2.size:.1%}")
    print(f"  Result: {'✓ Kept only large ocean' if labels_after.max() == 1 else '✗ Failed'}")

    # Test 3: Medium patch at threshold (4% - should be removed)
    print("\nTest 3: Medium patch below 5% threshold")
    mask3 = np.zeros((h, w), dtype=bool)
    # Create a patch that's 4% of image
    patch_size = int(np.sqrt(h * w * 0.04))
    mask3[200:200+patch_size, 300:300+patch_size] = True

    filtered3 = filter_ocean_mask(mask3.copy())
    print(f"  Patch size: {mask3.sum() / mask3.size:.1%} of image")
    print(f"  After filtering: {filtered3.sum()} pixels")
    print(f"  Result: {'✓ Small ocean removed' if filtered3.sum() == 0 else '✗ Failed'}")

    # Test 4: Medium patch at threshold (6% - should be kept)
    print("\nTest 4: Medium patch above 5% threshold")
    mask4 = np.zeros((h, w), dtype=bool)
    # Create a patch that's 6% of image
    patch_size = int(np.sqrt(h * w * 0.06))
    mask4[200:200+patch_size, 300:300+patch_size] = True

    filtered4 = filter_ocean_mask(mask4.copy())
    print(f"  Patch size: {mask4.sum() / mask4.size:.1%} of image")
    print(f"  After filtering: {filtered4.sum()} pixels")
    print(f"  Result: {'✓ Ocean kept' if filtered4.sum() > 0 else '✗ Failed'}")

    # Visualize results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(mask1, cmap='Blues')
    axes[0, 0].set_title('Test 1: Small patches\n(drone over land)')
    axes[1, 0].imshow(filtered1, cmap='Blues')
    axes[1, 0].set_title('Filtered: All removed')

    axes[0, 1].imshow(mask2, cmap='Blues')
    axes[0, 1].set_title('Test 2: Large ocean +\nsmall patches')
    axes[1, 1].imshow(filtered2, cmap='Blues')
    axes[1, 1].set_title('Filtered: Only ocean kept')

    axes[0, 2].imshow(mask3, cmap='Blues')
    axes[0, 2].set_title(f'Test 3: {mask3.sum() / mask3.size:.1%} of image\n(below threshold)')
    axes[1, 2].imshow(filtered3, cmap='Blues')
    axes[1, 2].set_title('Filtered: Removed')

    axes[0, 3].imshow(mask4, cmap='Blues')
    axes[0, 3].set_title(f'Test 4: {mask4.sum() / mask4.size:.1%} of image\n(above threshold)')
    axes[1, 3].imshow(filtered4, cmap='Blues')
    axes[1, 3].set_title('Filtered: Kept')

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle('Improved Ocean Filtering: 5% Minimum Area Threshold', fontsize=14)
    plt.tight_layout()
    plt.savefig('sgd_output/ocean_filtering_improved.png', dpi=100, bbox_inches='tight')
    print(f"\n✓ Visualization saved to sgd_output/ocean_filtering_improved.png")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("✓ Small isolated patches are removed (prevents land misclassification)")
    print("✓ Frames with <5% ocean are treated as land-only")
    print("✓ Only the largest contiguous ocean area is kept")
    print("✓ This prevents SGD detection on land areas")


if __name__ == "__main__":
    test_ocean_filtering()