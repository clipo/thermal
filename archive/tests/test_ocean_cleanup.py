#!/usr/bin/env python3
"""
Test script to verify that only the largest contiguous ocean area is kept,
preventing small landlocked areas from being misclassified as ocean.
"""

import numpy as np
from skimage import morphology, measure
import matplotlib.pyplot as plt

def create_test_mask():
    """Create a test mask with multiple disconnected 'ocean' regions"""
    mask = np.zeros((512, 640), dtype=bool)

    # Main ocean area (large, on the right side)
    mask[50:450, 300:600] = True

    # Small disconnected "ocean" area that should be removed (landlocked)
    mask[100:150, 50:100] = True

    # Another small area (landlocked)
    mask[300:320, 150:180] = True

    # Small isolated patch (should be removed)
    mask[400:410, 100:110] = True

    return mask

def keep_largest_ocean_area(ocean_mask):
    """Keep only the largest contiguous ocean area"""
    # First clean up very small regions
    ocean_mask = morphology.remove_small_objects(ocean_mask, min_size=100)

    # Keep only the largest contiguous ocean area
    ocean_labels = measure.label(ocean_mask, connectivity=2)
    if ocean_labels.max() > 0:
        # Find the largest connected component
        unique_labels, counts = np.unique(ocean_labels[ocean_labels > 0], return_counts=True)
        if len(unique_labels) > 0:
            largest_label = unique_labels[np.argmax(counts)]
            ocean_mask = (ocean_labels == largest_label)
            print(f"Found {len(unique_labels)} ocean regions")
            print(f"Kept largest region with {np.max(counts)} pixels")
            print(f"Removed {len(unique_labels)-1} smaller regions")

    return ocean_mask

def test_ocean_cleanup():
    """Test the ocean cleanup logic"""
    print("Testing Ocean Area Cleanup")
    print("="*50)

    # Create test mask
    original_mask = create_test_mask()

    # Apply cleanup
    cleaned_mask = keep_largest_ocean_area(original_mask.copy())

    # Count regions before and after
    labels_before = measure.label(original_mask, connectivity=2)
    labels_after = measure.label(cleaned_mask, connectivity=2)

    print(f"\nBefore cleanup: {labels_before.max()} separate ocean regions")
    print(f"After cleanup: {labels_after.max()} ocean region(s)")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_mask, cmap='Blues')
    axes[0].set_title(f'Before: {labels_before.max()} separate regions')
    axes[0].axis('off')

    axes[1].imshow(cleaned_mask, cmap='Blues')
    axes[1].set_title(f'After: {labels_after.max()} region (largest only)')
    axes[1].axis('off')

    plt.suptitle('Ocean Mask Cleanup: Keeping Only Largest Contiguous Area')
    plt.tight_layout()
    plt.savefig('sgd_output/ocean_cleanup_test.png', dpi=100, bbox_inches='tight')
    print(f"\nVisualization saved to sgd_output/ocean_cleanup_test.png")

    # Verify it works as expected
    assert labels_after.max() == 1, "Should have exactly 1 ocean region after cleanup"
    print("\n✓ Test passed: Only the largest ocean area was kept")

if __name__ == "__main__":
    test_ocean_cleanup()