#!/usr/bin/env python3
"""
Analyze frame overlap and SGD continuity between consecutive frames.
Helps diagnose why SGDs at frame edges don't continue in next frame.
"""

import numpy as np
from pathlib import Path
import sys
from sgd_georef_polygons import SGDPolygonGeoref
from sgd_detector_integrated import IntegratedSGDDetector
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def calculate_frame_overlap(gps1, gps2, altitude, fov_deg=45, width_pixels=640):
    """
    Calculate overlap between two thermal frames based on GPS positions.

    Args:
        gps1, gps2: GPS info dicts with 'lat', 'lon', 'heading'
        altitude: Flight altitude in meters
        fov_deg: Field of view in degrees
        width_pixels: Thermal image width

    Returns:
        Overlap percentage and pixel overlap
    """
    # Calculate ground coverage
    fov_rad = np.radians(fov_deg)
    ground_width = 2 * altitude * np.tan(fov_rad / 2)
    meters_per_pixel = ground_width / width_pixels

    # Calculate distance between frame centers
    from math import radians, cos, sin, sqrt, atan2
    R = 6371000  # Earth radius in meters

    lat1, lon1 = radians(gps1['lat']), radians(gps1['lon'])
    lat2, lon2 = radians(gps2['lat']), radians(gps2['lon'])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c

    # Calculate overlap
    if distance >= ground_width:
        overlap_m = 0
        overlap_pct = 0
    else:
        overlap_m = ground_width - distance
        overlap_pct = (overlap_m / ground_width) * 100

    overlap_pixels = int(overlap_m / meters_per_pixel)

    return {
        'distance_m': distance,
        'overlap_m': overlap_m,
        'overlap_pct': overlap_pct,
        'overlap_pixels': overlap_pixels,
        'ground_width': ground_width,
        'meters_per_pixel': meters_per_pixel
    }


def analyze_sgd_continuity(data_dir, frame1_num, frame2_num, verbose=True):
    """
    Analyze SGD continuity between two specific frames.

    Args:
        data_dir: Directory with thermal/RGB data
        frame1_num, frame2_num: Frame numbers to compare
        verbose: Print detailed analysis

    Returns:
        Analysis results dictionary
    """
    data_path = Path(data_dir)

    # Initialize detector and georeferencer
    detector = IntegratedSGDDetector(base_path=str(data_path))
    georef = SGDPolygonGeoref(base_path=str(data_path))

    # Process both frames
    results = {}
    for frame_num in [frame1_num, frame2_num]:
        if verbose:
            print(f"\nProcessing frame {frame_num}...")

        result = detector.process_frame(frame_num, visualize=False)

        # Get GPS info
        rgb_path = data_path / f"MAX_{frame_num:04d}.JPG"
        gps_info = georef.extract_gps(str(rgb_path), verbose=False)

        results[frame_num] = {
            'sgd_mask': result.get('sgd_plumes', np.zeros((512, 640))),
            'plume_info': result.get('plume_info', []),
            'gps': gps_info,
            'thermal': result.get('thermal')
        }

    # Calculate frame overlap
    if results[frame1_num]['gps'] and results[frame2_num]['gps']:
        altitude = results[frame1_num]['gps'].get('altitude', 400)
        overlap = calculate_frame_overlap(
            results[frame1_num]['gps'],
            results[frame2_num]['gps'],
            altitude
        )

        if verbose:
            print(f"\n{'='*60}")
            print("FRAME OVERLAP ANALYSIS:")
            print(f"  Distance between centers: {overlap['distance_m']:.1f}m")
            print(f"  Ground coverage per frame: {overlap['ground_width']:.1f}m")
            print(f"  Overlap: {overlap['overlap_m']:.1f}m ({overlap['overlap_pct']:.1f}%)")
            print(f"  Overlap in pixels: {overlap['overlap_pixels']} pixels")
            print(f"  Resolution: {overlap['meters_per_pixel']:.2f}m/pixel")

    # Analyze edge SGDs in frame 1
    edge_buffer = 50  # pixels from edge
    height, width = 512, 640

    edge_sgds = []
    for plume in results[frame1_num]['plume_info']:
        cy, cx = plume['centroid']

        # Check if near any edge
        edges = []
        if cx < edge_buffer:
            edges.append('left')
        if cx > width - edge_buffer:
            edges.append('right')
        if cy < edge_buffer:
            edges.append('top')
        if cy > height - edge_buffer:
            edges.append('bottom')

        if edges:
            plume['edges'] = edges
            edge_sgds.append(plume)

    if verbose and edge_sgds:
        print(f"\n{'='*60}")
        print(f"EDGE SGDs IN FRAME {frame1_num}:")
        for plume in edge_sgds:
            print(f"  SGD at ({plume['centroid'][1]:.0f}, {plume['centroid'][0]:.0f})")
            print(f"    - Area: {plume['area_pixels']} pixels")
            print(f"    - Temperature anomaly: {plume.get('temp_anomaly', 0):.2f}°C")
            print(f"    - Near edges: {', '.join(plume['edges'])}")

    # Check for continuation in frame 2
    # This would require pixel-level comparison in overlapping regions
    # For now, just check if similar temperature anomalies exist

    if verbose:
        print(f"\n{'='*60}")
        print(f"SGDs IN FRAME {frame2_num}:")
        for plume in results[frame2_num]['plume_info']:
            print(f"  SGD at ({plume['centroid'][1]:.0f}, {plume['centroid'][0]:.0f})")
            print(f"    - Area: {plume['area_pixels']} pixels")
            print(f"    - Temperature anomaly: {plume.get('temp_anomaly', 0):.2f}°C")

    return {
        'frame1': results[frame1_num],
        'frame2': results[frame2_num],
        'overlap': overlap if 'overlap' in locals() else None,
        'edge_sgds': edge_sgds
    }


def visualize_overlap(results, save_path=None):
    """
    Visualize the overlap and SGD positions between frames.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Frame 1
    ax = axes[0]
    sgd_mask1 = results['frame1']['sgd_mask']
    ax.imshow(sgd_mask1, cmap='hot')
    ax.set_title(f"Frame 1 SGDs")
    ax.axis('off')

    # Add edge zone
    rect = patches.Rectangle((0, 0), 640, 512, linewidth=2,
                            edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    rect = patches.Rectangle((590, 0), 50, 512, linewidth=1,
                            edgecolor='yellow', facecolor='yellow', alpha=0.2)
    ax.add_patch(rect)

    # Frame 2
    ax = axes[1]
    sgd_mask2 = results['frame2']['sgd_mask']
    ax.imshow(sgd_mask2, cmap='hot')
    ax.set_title(f"Frame 2 SGDs")
    ax.axis('off')

    # Add edge zone
    rect = patches.Rectangle((0, 0), 50, 512, linewidth=1,
                            edgecolor='yellow', facecolor='yellow', alpha=0.2)
    ax.add_patch(rect)

    # Temperature difference
    ax = axes[2]
    if results['frame1']['thermal'] is not None and results['frame2']['thermal'] is not None:
        temp_diff = results['frame2']['thermal'] - results['frame1']['thermal']
        im = ax.imshow(temp_diff, cmap='RdBu_r', vmin=-2, vmax=2)
        ax.set_title("Temperature Difference (F2-F1)")
        plt.colorbar(im, ax=ax, label='°C')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_frame_overlap.py <data_dir> [frame1] [frame2]")
        print("Example: python analyze_frame_overlap.py /path/to/106MEDIA 1501 1502")
        sys.exit(1)

    data_dir = sys.argv[1]

    # Default to consecutive frames if not specified
    if len(sys.argv) >= 4:
        frame1 = int(sys.argv[2])
        frame2 = int(sys.argv[3])
    else:
        # Find first two frames
        frames = sorted(Path(data_dir).glob("MAX_*.JPG"))[:2]
        if len(frames) < 2:
            print("Need at least 2 frames in directory")
            sys.exit(1)
        frame1 = int(frames[0].stem.split('_')[1])
        frame2 = int(frames[1].stem.split('_')[1])

    print(f"Analyzing continuity between frames {frame1} and {frame2}")
    print("=" * 60)

    results = analyze_sgd_continuity(data_dir, frame1, frame2)

    # Visualize if matplotlib is available
    try:
        visualize_overlap(results, f"overlap_{frame1}_{frame2}.png")
    except Exception as e:
        print(f"Could not create visualization: {e}")


if __name__ == "__main__":
    main()