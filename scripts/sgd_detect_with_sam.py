#!/usr/bin/env python3
"""
SGD Detection with SAM Segmentation

Example script showing how to use SAM instead of Random Forest for segmentation.

Usage:
    # First, create prompts interactively
    python scripts/sam_segmenter.py --interactive --data data/100MEDIA

    # Then run SGD detection with those prompts
    python scripts/sgd_detect_with_sam.py \\
        --data data/100MEDIA \\
        --prompts prompts/sam_prompts_TIMESTAMP.json \\
        --output sgd_output/sam_results.kml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgd_toolkit.detectors import IntegratedSGDDetector


def main():
    parser = argparse.ArgumentParser(
        description='SGD Detection with SAM Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data', required=True,
                       help='Path to data directory with thermal images')
    parser.add_argument('--prompts', required=True,
                       help='Path to SAM prompts JSON file')
    parser.add_argument('--output', required=True,
                       help='Output KML file path')
    parser.add_argument('--temp-threshold', type=float, default=0.5,
                       help='Temperature threshold (°C) below ocean mean')
    parser.add_argument('--min-area', type=int, default=50,
                       help='Minimum SGD plume area (pixels)')

    args = parser.parse_args()

    # Validate inputs
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"✗ Data directory not found: {args.data}")
        return 1

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"✗ Prompts file not found: {args.prompts}")
        print("\nCreate prompts first with:")
        print(f"  python scripts/sam_segmenter.py --interactive --data {args.data}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SGD DETECTION WITH SAM SEGMENTATION")
    print("="*70)
    print(f"\nData: {args.data}")
    print(f"Prompts: {args.prompts}")
    print(f"Output: {args.output}")
    print(f"Threshold: {args.temp_threshold}°C")
    print(f"Min Area: {args.min_area} pixels")
    print()

    # Initialize detector with SAM
    detector = IntegratedSGDDetector(
        temp_threshold=args.temp_threshold,
        min_area=args.min_area,
        base_path=args.data,
        use_sam=True,
        sam_prompts_path=args.prompts
    )

    # Find all image pairs
    irg_files = sorted(data_path.glob("IRX_*.irg"))

    if not irg_files:
        print(f"✗ No thermal images (IRX_*.irg) found in {args.data}")
        return 1

    print(f"Found {len(irg_files)} thermal images\n")

    # Process each image
    all_detections = []
    for i, irg_file in enumerate(irg_files, 1):
        frame_num = int(irg_file.stem.split('_')[1])

        print(f"[{i}/{len(irg_files)}] Processing frame {frame_num}...")

        try:
            # Detect SGD in this frame
            detections = detector.process_frame_with_alignment(frame_num)

            if detections and len(detections) > 0:
                all_detections.extend(detections)
                print(f"  ✓ Found {len(detections)} SGD detection(s)")
            else:
                print(f"  • No SGD detected")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Save results
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total SGD detections: {len(all_detections)}")

    if all_detections:
        # Save to KML
        detector.save_detections_kml(all_detections, str(output_path))
        print(f"✓ Saved to: {output_path}")

        # Print summary
        total_area = sum(d.get('area_pixels', 0) for d in all_detections)
        avg_temp_diff = sum(d.get('temp_diff', 0) for d in all_detections) / len(all_detections)

        print(f"\nSummary:")
        print(f"  Total area: {total_area} pixels")
        print(f"  Average temp difference: {avg_temp_diff:.2f}°C")
    else:
        print("\nNo SGD detections found.")
        print("\nTroubleshooting:")
        print("  1. Check if SAM prompts are accurate for your images")
        print("  2. Try lower temperature threshold (--temp-threshold 0.3)")
        print("  3. Verify thermal images have ocean areas")

    return 0


if __name__ == '__main__':
    sys.exit(main())
