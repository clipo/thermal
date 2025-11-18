#!/usr/bin/env python3
"""
Test script to compare different ocean baseline temperature methods.

This script demonstrates how different baseline calculation methods affect
SGD detection, especially in frames where cold plumes dominate.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sgd_detector_improved import ImprovedSGDDetector
import sys


def analyze_baseline_methods(base_path, frame_numbers=None, output_dir="baseline_comparison"):
    """
    Analyze how different baseline methods affect SGD detection across multiple frames.

    Args:
        base_path: Path to data directory
        frame_numbers: List of frame numbers to analyze (default: first 5)
        output_dir: Directory to save comparison results
    """
    base_path = Path(base_path)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Default to first 5 frames if not specified
    if frame_numbers is None:
        rgb_files = sorted(base_path.glob("MAX_*.JPG"))[:5]
        frame_numbers = []
        for rgb_file in rgb_files:
            num = int(rgb_file.stem.split('_')[1])
            if (base_path / f"IRX_{num:04d}.irg").exists():
                frame_numbers.append(num)

    if not frame_numbers:
        print("No valid frames found!")
        return

    # Methods to compare
    methods = {
        'median': {'baseline_method': 'median'},
        'upper_quartile': {'baseline_method': 'upper_quartile'},
        '80th_percentile': {'baseline_method': 'upper_percentile', 'percentile_value': 80},
        '90th_percentile': {'baseline_method': 'upper_percentile', 'percentile_value': 90},
        'trimmed_mean_25': {'baseline_method': 'trimmed_mean', 'trim_percentage': 25}
    }

    # Store results for all frames
    all_results = {}

    for frame_num in frame_numbers:
        print(f"\\nAnalyzing frame {frame_num}...")
        frame_results = {}

        for method_name, method_params in methods.items():
            try:
                # Create detector with specific method
                detector = ImprovedSGDDetector(base_path, **method_params)

                # Process frame
                result = detector.process_frame(frame_num, visualize=False)

                # Store key metrics
                frame_results[method_name] = {
                    'num_plumes': len(result['plume_info']),
                    'total_area': result['sgd_mask'].sum(),
                    'characteristics': result['characteristics'],
                    'sgd_mask': result['sgd_mask'],
                    'thermal': result['data']['thermal'],
                    'ocean_mask': result['masks']['ocean']
                }

                # Extract ocean statistics
                if 'ocean_stats' in result['characteristics']:
                    stats = result['characteristics']['ocean_stats']
                    frame_results[method_name]['baseline'] = stats.get('baseline', np.nan)
                    frame_results[method_name]['ocean_median'] = stats.get('median', np.nan)
                    frame_results[method_name]['ocean_q75'] = stats.get('q75', np.nan)

            except Exception as e:
                print(f"  Error with {method_name}: {e}")
                frame_results[method_name] = None

        all_results[frame_num] = frame_results

        # Create comparison visualization for this frame
        create_frame_comparison(frame_results, frame_num, output_path)

    # Create summary statistics
    create_summary_report(all_results, output_path)

    return all_results


def create_frame_comparison(frame_results, frame_num, output_path):
    """
    Create visual comparison of different methods for a single frame.
    """
    # Filter out failed methods
    valid_results = {k: v for k, v in frame_results.items() if v is not None}

    if not valid_results:
        print(f"  No valid results for frame {frame_num}")
        return

    # Create figure
    n_methods = len(valid_results)
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(3, n_methods, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.2)

    # Get thermal data (same for all methods)
    first_result = next(iter(valid_results.values()))
    thermal = first_result['thermal']
    ocean_mask = first_result['ocean_mask']

    # Normalize thermal for display
    vmin, vmax = np.nanpercentile(thermal[ocean_mask], [1, 99])

    for idx, (method_name, result) in enumerate(valid_results.items()):
        # Top row: Thermal with SGD overlay
        ax1 = fig.add_subplot(gs[0, idx])
        thermal_display = thermal.copy()
        thermal_display[~ocean_mask] = np.nan

        im = ax1.imshow(thermal_display, cmap='RdYlBu_r', vmin=vmin, vmax=vmax)

        # Overlay SGD detections
        sgd_overlay = np.ma.masked_where(~result['sgd_mask'], np.ones_like(thermal))
        ax1.imshow(sgd_overlay, cmap='spring', alpha=0.5, vmin=0, vmax=1)

        ax1.set_title(f'{method_name}\\n{result["num_plumes"]} plumes')
        ax1.axis('off')

        if idx == n_methods - 1:
            plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # Middle row: SGD mask only
        ax2 = fig.add_subplot(gs[1, idx])
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        sgd_display[result['sgd_mask']] = [0, 1, 1]  # Cyan
        ax2.imshow(sgd_display)
        ax2.set_title(f'Area: {result["total_area"]} px')
        ax2.axis('off')

        # Bottom row: Statistics
        ax3 = fig.add_subplot(gs[2, idx])
        stats_text = f"Baseline: {result.get('baseline', 0):.2f}°C\\n"
        stats_text += f"Median: {result.get('ocean_median', 0):.2f}°C\\n"
        stats_text += f"Q75: {result.get('ocean_q75', 0):.2f}°C"

        ax3.text(0.5, 0.5, stats_text, ha='center', va='center',
                transform=ax3.transAxes, fontsize=10)
        ax3.axis('off')

    plt.suptitle(f'Baseline Method Comparison - Frame {frame_num}', fontsize=14)

    # Save figure
    output_file = output_path / f'baseline_comparison_frame_{frame_num:04d}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison to {output_file}")


def create_summary_report(all_results, output_path):
    """
    Create a summary report comparing methods across all frames.
    """
    # Prepare data for summary
    methods = list(next(iter(all_results.values())).keys())
    frames = sorted(all_results.keys())

    # Create summary tables
    summary_data = {method: {'plumes': [], 'areas': []} for method in methods}

    for frame in frames:
        for method in methods:
            if all_results[frame][method] is not None:
                summary_data[method]['plumes'].append(all_results[frame][method]['num_plumes'])
                summary_data[method]['areas'].append(all_results[frame][method]['total_area'])
            else:
                summary_data[method]['plumes'].append(0)
                summary_data[method]['areas'].append(0)

    # Create summary figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Number of plumes
    x = np.arange(len(frames))
    width = 0.15

    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width
        ax1.bar(x + offset, summary_data[method]['plumes'],
               width, label=method.replace('_', ' '))

    ax1.set_xlabel('Frame Number')
    ax1.set_ylabel('Number of SGD Plumes')
    ax1.set_title('SGD Plume Count by Baseline Method')
    ax1.set_xticks(x)
    ax1.set_xticklabels(frames)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total area
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2) * width
        ax2.bar(x + offset, summary_data[method]['areas'],
               width, label=method.replace('_', ' '))

    ax2.set_xlabel('Frame Number')
    ax2.set_ylabel('Total SGD Area (pixels)')
    ax2.set_title('SGD Area Coverage by Baseline Method')
    ax2.set_xticks(x)
    ax2.set_xticklabels(frames)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Summary: Baseline Method Performance Across Frames', fontsize=14)
    plt.tight_layout()

    # Save summary
    summary_file = output_path / 'baseline_method_summary.png'
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\\nSaved summary to {summary_file}")

    # Print text summary
    print("\\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for method in methods:
        plumes = summary_data[method]['plumes']
        areas = summary_data[method]['areas']

        print(f"\\n{method.replace('_', ' ').upper()}:")
        print(f"  Average plumes per frame: {np.mean(plumes):.1f}")
        print(f"  Total plumes detected: {sum(plumes)}")
        print(f"  Average area per frame: {np.mean(areas):.0f} pixels")
        print(f"  Max area in a frame: {max(areas)} pixels")


def main():
    """
    Main function to run baseline comparison analysis.
    """
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_baseline_methods.py <data_directory> [frame_numbers]")
        print("Example: python test_baseline_methods.py /path/to/data 1,2,3,4,5")
        sys.exit(1)

    base_path = sys.argv[1]

    # Parse frame numbers if provided
    frame_numbers = None
    if len(sys.argv) > 2:
        try:
            frame_numbers = [int(x) for x in sys.argv[2].split(',')]
        except ValueError:
            print("Invalid frame numbers. Use comma-separated integers.")
            sys.exit(1)

    # Run analysis
    print(f"Analyzing baseline methods for data in: {base_path}")
    results = analyze_baseline_methods(base_path, frame_numbers)

    print("\\nAnalysis complete!")


if __name__ == "__main__":
    main()