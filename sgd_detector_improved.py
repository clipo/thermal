#!/usr/bin/env python3
"""
Improved SGD Detector with configurable ocean baseline temperature calculation.

This module extends the SGD detection capability with more robust ocean temperature
baseline calculations to handle cases where cold plumes dominate the frame.
"""

import numpy as np
from pathlib import Path
from sgd_detector_integrated import IntegratedSGDDetector
from scipy import ndimage
from skimage import measure
from sgd_glint_detector import SunGlintDetector


class ImprovedSGDDetector(IntegratedSGDDetector):
    """
    Enhanced SGD detector with configurable ocean temperature baseline methods.

    Baseline methods:
    - 'median': Traditional median temperature (default for compatibility)
    - 'upper_quartile': 75th percentile - more robust when cold plumes dominate
    - 'upper_percentile': Custom percentile (e.g., 80th, 90th)
    - 'trimmed_mean': Mean after excluding lowest X% of temperatures
    - 'modal_peak': Temperature mode estimated from histogram peak
    """

    def __init__(self, base_path="data/100MEDIA",
                 temp_threshold=0.5,
                 min_area=50,
                 baseline_method='upper_quartile',
                 percentile_value=75,
                 trim_percentage=25,
                 use_ml=True,
                 ml_model_path="segmentation_model.pkl",
                 detect_glint=True,
                 glint_area_threshold=0.15):
        """
        Initialize improved SGD detector.

        Args:
            base_path: Path to data directory
            temp_threshold: Temperature difference threshold for SGD detection
            min_area: Minimum plume area in pixels
            baseline_method: Method for calculating ocean baseline temperature
                - 'median': Use median temperature
                - 'upper_quartile': Use 75th percentile
                - 'upper_percentile': Use custom percentile
                - 'trimmed_mean': Mean after trimming lower values
                - 'modal_peak': Use histogram peak
            percentile_value: Percentile to use if baseline_method is 'upper_percentile'
            trim_percentage: Percentage to trim if baseline_method is 'trimmed_mean'
            use_ml: Whether to use ML for segmentation
            ml_model_path: Path to ML model
        """
        super().__init__(temp_threshold=temp_threshold,
                        min_area=min_area,
                        base_path=base_path,
                        use_ml=use_ml,
                        ml_model_path=ml_model_path)

        self.baseline_method = baseline_method
        self.percentile_value = percentile_value
        self.trim_percentage = trim_percentage

        # Sun glint detection
        self.detect_glint = detect_glint
        self.glint_area_threshold = glint_area_threshold
        if self.detect_glint:
            self.glint_detector = SunGlintDetector(area_threshold=glint_area_threshold)

    def calculate_ocean_baseline(self, ocean_temps):
        """
        Calculate ocean baseline temperature using specified method.

        Args:
            ocean_temps: Array of ocean temperatures

        Returns:
            Baseline temperature and statistics dict
        """
        if len(ocean_temps) == 0:
            return np.nan, {}

        # Remove NaN values
        valid_temps = ocean_temps[~np.isnan(ocean_temps)]
        if len(valid_temps) == 0:
            return np.nan, {}

        # Calculate various statistics
        stats = {
            'median': np.median(valid_temps),
            'mean': np.mean(valid_temps),
            'std': np.std(valid_temps),
            'q25': np.percentile(valid_temps, 25),
            'q75': np.percentile(valid_temps, 75),
            'q90': np.percentile(valid_temps, 90),
            'min': np.min(valid_temps),
            'max': np.max(valid_temps),
            'count': len(valid_temps)
        }
        stats['iqr'] = stats['q75'] - stats['q25']

        # Calculate baseline based on method
        if self.baseline_method == 'median':
            baseline = stats['median']
            stats['baseline_method'] = 'median'

        elif self.baseline_method == 'upper_quartile':
            baseline = stats['q75']
            stats['baseline_method'] = 'upper_quartile (75th percentile)'

        elif self.baseline_method == 'upper_percentile':
            baseline = np.percentile(valid_temps, self.percentile_value)
            stats['baseline_method'] = f'{self.percentile_value}th percentile'
            stats[f'p{self.percentile_value}'] = baseline

        elif self.baseline_method == 'trimmed_mean':
            # Remove lowest X% of temperatures
            threshold = np.percentile(valid_temps, self.trim_percentage)
            trimmed_temps = valid_temps[valid_temps >= threshold]
            baseline = np.mean(trimmed_temps) if len(trimmed_temps) > 0 else stats['mean']
            stats['baseline_method'] = f'trimmed_mean (exclude lowest {self.trim_percentage}%)'
            stats['trimmed_mean'] = baseline
            stats['trimmed_count'] = len(trimmed_temps)

        elif self.baseline_method == 'modal_peak':
            # Estimate mode from histogram
            hist, bin_edges = np.histogram(valid_temps, bins=50)
            peak_idx = np.argmax(hist)
            baseline = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
            stats['baseline_method'] = 'modal_peak (histogram peak)'
            stats['modal_peak'] = baseline

        else:
            # Default to median
            baseline = stats['median']
            stats['baseline_method'] = 'median (default)'

        stats['baseline'] = baseline
        return baseline, stats

    def detect_sgd_plumes(self, thermal, masks):
        """
        Detect SGD plumes with improved baseline calculation.

        Args:
            thermal: Thermal data array
            masks: Dictionary with 'ocean', 'land', 'waves' masks

        Returns:
            sgd_mask: Boolean mask of detected SGD areas
            plume_info: List of plume information dictionaries
            characteristics: Dictionary of SGD characteristics
        """
        # Get ocean temperatures
        ocean_temps = thermal[masks['ocean']]

        if len(ocean_temps) == 0:
            return np.zeros_like(thermal, dtype=bool), [], {}

        # Calculate baseline with improved method
        ocean_baseline, ocean_stats = self.calculate_ocean_baseline(ocean_temps)

        if np.isnan(ocean_baseline):
            return np.zeros_like(thermal, dtype=bool), [], {}

        # SGD threshold
        threshold = ocean_baseline - self.temp_threshold

        # Detect cold anomalies
        cold_mask = (thermal < threshold) & masks['ocean']

        # Get shoreline
        shoreline, shoreline_thin = self.detect_shoreline(masks)

        # Distance from shore
        distance_from_shore = ndimage.distance_transform_edt(~shoreline)
        distance_from_shore[~masks['ocean']] = np.inf

        # Find connected cold regions
        labeled_cold, num_features = measure.label(cold_mask, return_num=True)

        sgd_mask = np.zeros_like(cold_mask, dtype=bool)
        plume_info = []

        for i in range(1, num_features + 1):
            plume = labeled_cold == i

            # Check if near shore
            plume_distances = distance_from_shore[plume]
            min_shore_distance = np.nanmin(plume_distances)

            # Criteria for SGD
            if min_shore_distance < 5 and plume.sum() >= self.min_area:
                sgd_mask = sgd_mask | plume

                props = measure.regionprops(plume.astype(int))[0]

                # Extract contour
                contours = measure.find_contours(plume.astype(float), 0.5)
                plume_contour = contours[0] if contours else []

                # Calculate plume statistics
                plume_temps = thermal[plume]

                plume_info.append({
                    'id': i,
                    'area_pixels': plume.sum(),
                    'min_shore_distance': min_shore_distance,
                    'centroid': props.centroid,
                    'bbox': props.bbox,
                    'eccentricity': props.eccentricity,
                    'contour': plume_contour.tolist() if len(plume_contour) > 0 else [],
                    'mask': plume,
                    'mean_temp': float(np.mean(plume_temps)),
                    'min_temp': float(np.min(plume_temps)),
                    'temperature_anomaly': float(np.mean(plume_temps) - ocean_baseline)  # Use full name for compatibility
                })

        # Calculate overall characteristics
        characteristics = {
            'ocean_stats': ocean_stats,
            'detection_threshold': float(threshold),
            'num_plumes': len(plume_info)
        }

        if sgd_mask.any():
            sgd_temps = thermal[sgd_mask]
            characteristics.update({
                'mean_temp': float(np.mean(sgd_temps)),
                'min_temp': float(np.min(sgd_temps)),
                'max_temp': float(np.max(sgd_temps)),
                'temp_anomaly': float(np.mean(sgd_temps) - ocean_baseline),
                'area_pixels': int(sgd_mask.sum()),
                'area_m2': float(sgd_mask.sum() * 0.01)  # Assuming 10cm resolution
            })
        else:
            # Provide default values when no SGD is detected
            characteristics.update({
                'mean_temp': 0.0,
                'min_temp': 0.0,
                'max_temp': 0.0,
                'temp_anomaly': 0.0,
                'area_pixels': 0,
                'area_m2': 0.0
            })

        return sgd_mask, plume_info, characteristics

    def process_frame(self, frame_number, visualize=False, include_waves=False):
        """
        Process a single frame with sun glint detection.

        Args:
            frame_number: Frame number to process
            visualize: Whether to visualize results
            include_waves: Whether to include wave areas in ocean mask

        Returns:
            dict: Frame processing results with glint detection
        """
        # Call parent process_frame
        result = super().process_frame(frame_number, visualize=False, include_waves=include_waves)

        if result is None:
            return None

        # Add sun glint detection if enabled
        if self.detect_glint and result.get('data') and result.get('masks'):
            rgb = result['data'].get('rgb_aligned')
            thermal = result['data'].get('thermal')
            ocean_mask = result['masks'].get('ocean')
            sgd_mask = result.get('sgd_mask')

            if rgb is not None and thermal is not None and ocean_mask is not None:
                # Check for sun glint
                glint_analysis = self.glint_detector.detect_turn_glint(
                    rgb, thermal, ocean_mask, sgd_mask,
                    check_continuity=True
                )

                result['glint_analysis'] = glint_analysis

                # If glint detected with high confidence, flag the frame
                if glint_analysis['has_glint']:
                    print(f"  ⚠ Sun glint detected in frame {frame_number}: {glint_analysis['summary']}")

                    # Optionally remove SGD detections from glint frames
                    if glint_analysis['confidence'] > 0.7:  # High confidence glint
                        result['plume_info'] = []  # Clear SGD detections
                        result['sgd_mask'] = np.zeros_like(sgd_mask, dtype=bool) if sgd_mask is not None else None
                        result['characteristics'] = {}
                        result['glint_filtered'] = True

        if visualize:
            self.visualize_detection(result)

        return result

    def compare_baseline_methods(self, frame_number):
        """
        Process a frame with different baseline methods for comparison.

        Args:
            frame_number: Frame number to process

        Returns:
            Dictionary with results for each method
        """
        # Get the base data
        rgb_full, rgb_aligned, thermal = self.load_and_align(frame_number)
        masks = self.segment_rgb(rgb_aligned)

        # Test different methods
        methods = ['median', 'upper_quartile', 'upper_percentile', 'trimmed_mean']
        results = {}

        for method in methods:
            # Temporarily change method
            original_method = self.baseline_method
            self.baseline_method = method

            # Detect SGD
            sgd_mask, plume_info, characteristics = self.detect_sgd_plumes(thermal, masks)

            results[method] = {
                'sgd_mask': sgd_mask,
                'plume_info': plume_info,
                'characteristics': characteristics,
                'num_plumes': len(plume_info),
                'total_area': sgd_mask.sum() if sgd_mask.any() else 0
            }

            # Restore original method
            self.baseline_method = original_method

        return results


def demonstrate_baseline_comparison(base_path, frame_number=1):
    """
    Demonstrate the difference between baseline methods.

    Args:
        base_path: Path to data directory
        frame_number: Frame to analyze
    """
    import matplotlib.pyplot as plt

    # Create detector
    detector = ImprovedSGDDetector(base_path)

    # Compare methods
    print(f"\\nComparing baseline methods for frame {frame_number}:")
    print("-" * 60)

    results = detector.compare_baseline_methods(frame_number)

    # Print comparison
    for method, result in results.items():
        chars = result['characteristics']
        ocean_stats = chars.get('ocean_stats', {})

        print(f"\\nMethod: {method}")
        print(f"  Baseline temp: {ocean_stats.get('baseline', 'N/A'):.2f}°C")
        print(f"  Detection threshold: {chars.get('detection_threshold', 'N/A'):.2f}°C")
        print(f"  Plumes detected: {result['num_plumes']}")
        print(f"  Total SGD area: {result['total_area']} pixels")

        if result['num_plumes'] > 0:
            print(f"  Mean anomaly: {chars.get('temp_anomaly', 0):.2f}°C")

    # Visualize differences
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (method, result) in enumerate(results.items()):
        ax = axes[idx]

        # Create visualization
        sgd_display = np.zeros((*result['sgd_mask'].shape, 3))
        sgd_display[result['sgd_mask']] = [0, 1, 1]  # Cyan for SGD

        ax.imshow(sgd_display)
        ax.set_title(f'{method}\\n({result["num_plumes"]} plumes, {result["total_area"]} px)')
        ax.axis('off')

    plt.suptitle(f'Baseline Method Comparison - Frame {frame_number}')
    plt.tight_layout()

    return fig, results


if __name__ == "__main__":
    # Example usage
    base_path = Path("/path/to/your/data")

    # Create detector with upper quartile baseline
    detector = ImprovedSGDDetector(
        base_path,
        baseline_method='upper_quartile',  # More robust for frames with large cold areas
        temp_threshold=0.5
    )

    # Process a frame
    result = detector.process_frame(1, visualize=True)

    # Or compare different methods
    # fig, comparison = demonstrate_baseline_comparison(base_path, frame_number=1)
    # plt.show()