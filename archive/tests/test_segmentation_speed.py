#!/usr/bin/env python3
"""
Test segmentation speed: ML vs Rule-based
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from sgd_detector_integrated import IntegratedSGDDetector
from ml_segmentation import MLSegmenter

def test_speed():
    """Compare segmentation speeds"""
    print("Segmentation Speed Test")
    print("=" * 50)
    
    # Load a test image
    frame = 248
    rgb_path = Path(f"data/100MEDIA/MAX_{frame:04d}.JPG")
    rgb_full = np.array(PILImage.open(rgb_path))
    
    # Crop to thermal FOV
    h, w = rgb_full.shape[:2]
    crop_h = int(h * 0.7)
    crop_w = int(w * 0.7)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    rgb_cropped = rgb_full[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    # Resize to thermal size
    img_pil = PILImage.fromarray(rgb_cropped)
    rgb_test = np.array(img_pil.resize((640, 512), PILImage.Resampling.BILINEAR))
    
    print(f"Image size: {rgb_test.shape}")
    print()
    
    # Test ML segmentation at different strides
    ml_segmenter = MLSegmenter()
    
    if ml_segmenter.classifier:
        print("ML Segmentation Speeds:")
        print("-" * 30)
        
        for stride in [1, 2, 3, 4, 5]:
            start = time.time()
            masks = ml_segmenter.segment_fast(rgb_test, stride=stride)
            elapsed = time.time() - start
            print(f"  Stride {stride}: {elapsed:.2f} seconds")
        
        # Test full resolution (slow)
        print("\n  Full resolution (no stride):")
        start = time.time()
        masks = ml_segmenter.segment(rgb_test, chunk_size=20)
        elapsed = time.time() - start
        print(f"    Time: {elapsed:.2f} seconds")
    
    # Test rule-based
    print("\nRule-based Segmentation:")
    print("-" * 30)
    detector = IntegratedSGDDetector(use_ml=False)
    
    start = time.time()
    masks = detector.segment_ocean_land_waves(rgb_test)
    elapsed = time.time() - start
    print(f"  Time: {elapsed:.3f} seconds")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS:")
    print("  - Stride 3-4 offers good speed/quality balance")
    print("  - Rule-based is fastest but less accurate")
    print("  - Full ML is most accurate but slow")

if __name__ == "__main__":
    test_speed()