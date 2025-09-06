#!/usr/bin/env python3
"""
Test ML segmentation integration in SGD detector.
Compare ML-based vs rule-based segmentation results.
"""

import numpy as np
import matplotlib.pyplot as plt
from sgd_detector_integrated import IntegratedSGDDetector
from pathlib import Path

def test_ml_integration(frame_number=248):
    """Test and compare ML vs rule-based segmentation"""
    
    print("Testing ML Segmentation Integration")
    print("=" * 50)
    
    # Test with ML segmentation
    print("\n1. Testing WITH ML segmentation:")
    print("-" * 30)
    detector_ml = IntegratedSGDDetector(use_ml=True)
    result_ml = detector_ml.process_frame(frame_number, visualize=False)
    
    # Test without ML segmentation
    print("\n2. Testing WITHOUT ML segmentation (rule-based):")
    print("-" * 30)
    detector_rule = IntegratedSGDDetector(use_ml=False)
    result_rule = detector_rule.process_frame(frame_number, visualize=False)
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON:")
    print(f"ML-based:    Found {len(result_ml['plume_info'])} SGD plumes")
    print(f"Rule-based:  Found {len(result_rule['plume_info'])} SGD plumes")
    
    # Visualize comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Frame {frame_number}: ML vs Rule-based Segmentation Comparison', fontsize=14)
    
    # Row 1: ML-based
    axes[0, 0].imshow(result_ml['data']['rgb_aligned'])
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # ML Segmentation
    seg_ml = np.zeros((*result_ml['masks']['ocean'].shape, 3))
    seg_ml[result_ml['masks']['ocean']] = [0, 0.3, 1]
    seg_ml[result_ml['masks']['land']] = [0, 0.7, 0]
    seg_ml[result_ml['masks']['waves']] = [1, 1, 0.5]
    axes[0, 1].imshow(seg_ml)
    axes[0, 1].set_title('ML Segmentation')
    axes[0, 1].axis('off')
    
    # ML SGD detection
    sgd_ml = np.zeros((*result_ml['sgd_mask'].shape, 3))
    if result_ml['sgd_mask'].any():
        sgd_ml[result_ml['sgd_mask']] = [0, 1, 1]
    # Add shoreline
    shoreline_ml, _ = detector_ml.detect_shoreline(result_ml['masks'])
    sgd_ml[shoreline_ml] = [1, 1, 0]
    axes[0, 2].imshow(sgd_ml)
    axes[0, 2].set_title(f'ML SGD: {len(result_ml["plume_info"])} plumes')
    axes[0, 2].axis('off')
    
    # ML overlay
    overlay_ml = result_ml['data']['rgb_aligned'].copy() / 255.0
    if result_ml['sgd_mask'].any():
        overlay_ml[result_ml['sgd_mask']] = [0, 1, 0]
    axes[0, 3].imshow(overlay_ml)
    axes[0, 3].set_title('ML SGD Overlay')
    axes[0, 3].axis('off')
    
    # Row 2: Rule-based
    axes[1, 0].imshow(result_rule['data']['rgb_aligned'])
    axes[1, 0].set_title('RGB Image')
    axes[1, 0].axis('off')
    
    # Rule-based Segmentation
    seg_rule = np.zeros((*result_rule['masks']['ocean'].shape, 3))
    seg_rule[result_rule['masks']['ocean']] = [0, 0.3, 1]
    seg_rule[result_rule['masks']['land']] = [0, 0.7, 0]
    seg_rule[result_rule['masks']['waves']] = [1, 1, 0.5]
    axes[1, 1].imshow(seg_rule)
    axes[1, 1].set_title('Rule-based Segmentation')
    axes[1, 1].axis('off')
    
    # Rule-based SGD detection
    sgd_rule = np.zeros((*result_rule['sgd_mask'].shape, 3))
    if result_rule['sgd_mask'].any():
        sgd_rule[result_rule['sgd_mask']] = [0, 1, 1]
    # Add shoreline
    shoreline_rule, _ = detector_rule.detect_shoreline(result_rule['masks'])
    sgd_rule[shoreline_rule] = [1, 1, 0]
    axes[1, 2].imshow(sgd_rule)
    axes[1, 2].set_title(f'Rule SGD: {len(result_rule["plume_info"])} plumes')
    axes[1, 2].axis('off')
    
    # Rule-based overlay
    overlay_rule = result_rule['data']['rgb_aligned'].copy() / 255.0
    if result_rule['sgd_mask'].any():
        overlay_rule[result_rule['sgd_mask']] = [0, 1, 0]
    axes[1, 3].imshow(overlay_rule)
    axes[1, 3].set_title('Rule SGD Overlay')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    # Print detailed statistics
    print("\nSegmentation Statistics:")
    print("-" * 30)
    
    # ML stats
    total = result_ml['masks']['ocean'].size
    print("ML-based:")
    print(f"  Ocean: {100*result_ml['masks']['ocean'].sum()/total:.1f}%")
    print(f"  Land:  {100*result_ml['masks']['land'].sum()/total:.1f}%")
    print(f"  Waves: {100*result_ml['masks']['waves'].sum()/total:.1f}%")
    
    # Rule stats
    print("\nRule-based:")
    print(f"  Ocean: {100*result_rule['masks']['ocean'].sum()/total:.1f}%")
    print(f"  Land:  {100*result_rule['masks']['land'].sum()/total:.1f}%")
    print(f"  Waves: {100*result_rule['masks']['waves'].sum()/total:.1f}%")
    
    # Difference in ocean mask (most important for SGD)
    ocean_diff = np.abs(result_ml['masks']['ocean'].astype(int) - 
                       result_rule['masks']['ocean'].astype(int))
    print(f"\nOcean mask difference: {100*ocean_diff.sum()/total:.1f}% of pixels")
    
    # SGD plume details
    if result_ml['plume_info']:
        print("\nML SGD plumes:")
        for i, plume in enumerate(result_ml['plume_info'][:3], 1):
            print(f"  {i}. Area: {plume['area_pixels']} px, "
                  f"Shore dist: {plume['min_shore_distance']:.1f}")
    
    if result_rule['plume_info']:
        print("\nRule-based SGD plumes:")
        for i, plume in enumerate(result_rule['plume_info'][:3], 1):
            print(f"  {i}. Area: {plume['area_pixels']} px, "
                  f"Shore dist: {plume['min_shore_distance']:.1f}")
    
    plt.show()
    
    return result_ml, result_rule

def test_multiple_frames():
    """Test on multiple frames to see consistency"""
    frames_to_test = [248, 249, 250, 251, 252]
    
    ml_counts = []
    rule_counts = []
    
    for frame in frames_to_test:
        try:
            print(f"\nTesting frame {frame}...")
            detector_ml = IntegratedSGDDetector(use_ml=True)
            detector_rule = IntegratedSGDDetector(use_ml=False)
            
            result_ml = detector_ml.process_frame(frame, visualize=False)
            result_rule = detector_rule.process_frame(frame, visualize=False)
            
            ml_counts.append(len(result_ml['plume_info']))
            rule_counts.append(len(result_rule['plume_info']))
            
            print(f"  ML: {len(result_ml['plume_info'])} plumes")
            print(f"  Rule: {len(result_rule['plume_info'])} plumes")
        except Exception as e:
            print(f"  Error: {e}")
    
    if ml_counts and rule_counts:
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print(f"Average ML detections: {np.mean(ml_counts):.1f}")
        print(f"Average Rule detections: {np.mean(rule_counts):.1f}")
        print(f"ML more consistent: std={np.std(ml_counts):.2f} vs {np.std(rule_counts):.2f}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Single frame comparison (visual)")
    print("2. Multiple frames (statistics)")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "2":
        test_multiple_frames()
    else:
        test_ml_integration(frame_number=248)