#!/usr/bin/env python3
"""
SGD Detector with Moving Average Ocean Baseline
Provides more stable detection across frames by averaging ocean temperatures
"""

import numpy as np
from collections import deque
from pathlib import Path
from sgd_detector_improved import ImprovedSGDDetector
from scipy import ndimage
from skimage import measure


class MovingAverageSGDDetector(ImprovedSGDDetector):
    """
    SGD detector that uses a moving average of ocean temperatures
    across multiple frames to provide more stable baseline temperatures.

    This helps prevent dramatic shifts in SGD detection when the UAV
    turns and captures different ocean areas.
    """

    def __init__(self, data_dir, model_name='segmentation_model.pkl',
                 temp_threshold=0.5, min_area=10, max_area=None,
                 window_size=5, baseline_method='median', **kwargs):
        """
        Initialize detector with moving average capability.

        Args:
            data_dir: Directory containing thermal/RGB data
            model_name: Segmentation model filename
            temp_threshold: Temperature threshold below baseline (°C)
            min_area: Minimum plume size in pixels
            max_area: Maximum plume size in pixels
            window_size: Number of frames for moving average (default 5)
            baseline_method: Method for baseline calculation within window
        """
        # Store data_dir before calling super().__init__
        self.data_dir = Path(data_dir)

        # Initialize parent class
        super().__init__(
            base_path=str(data_dir),
            temp_threshold=temp_threshold,
            min_area=min_area,
            use_ml=True,
            ml_model_path=model_name,
            baseline_method=baseline_method,
            **kwargs
        )

        self.window_size = window_size
        self.ocean_temp_buffer = deque(maxlen=window_size)
        self.frame_metadata_buffer = deque(maxlen=window_size)
        self.max_area = max_area

        # Statistics tracking
        self.baseline_history = []

    def detect_sgd(self, thermal, masks):
        """
        Override parent's detect_sgd to use buffered ocean temperatures.

        This method is called by process_frame after loading and segmenting.
        """
        # Extract ocean temperatures from current frame
        ocean_mask = masks['ocean']
        ocean_temps = thermal[ocean_mask]

        # Get frame number from the thermal data (this is a bit of a hack)
        # In real usage, frame number should be passed through
        frame_number = len(self.ocean_temp_buffer)  # Use buffer length as proxy

        # Update buffer with current frame's ocean temps
        self.update_ocean_buffer(ocean_temps, frame_number)

        # Calculate baseline from buffer
        baseline_stats = self.calculate_buffered_baseline()

        if baseline_stats is None:
            # No valid ocean temperatures
            return {
                'sgd_plumes': np.zeros_like(ocean_mask),
                'shoreline': np.zeros_like(ocean_mask),
                'shoreline_thin': np.zeros_like(ocean_mask),
                'ocean_stats': {},
                'plume_info': []
            }

        baseline_temp = baseline_stats['baseline']
        threshold = baseline_temp - self.temp_threshold

        # Detect cold anomalies using buffered baseline
        cold_mask = (thermal < threshold) & ocean_mask

        # Get shoreline
        shoreline, shoreline_thin = self.detect_shoreline(masks)

        # Distance from shore
        distance_from_shore = ndimage.distance_transform_edt(~shoreline)
        distance_from_shore[~ocean_mask] = np.inf

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
                # Apply max area filter if specified
                if self.max_area is not None and plume.sum() > self.max_area:
                    continue

                sgd_mask = sgd_mask | plume

                props = measure.regionprops(plume.astype(int))[0]

                # Extract contour of the plume
                contours = measure.find_contours(plume.astype(float), 0.5)
                plume_contour = contours[0] if contours else []

                plume_temps = thermal[plume]

                plume_info.append({
                    'id': i,
                    'area_pixels': plume.sum(),
                    'min_shore_distance': min_shore_distance,
                    'centroid': props.centroid,
                    'min_temp': np.nanmin(plume_temps),
                    'mean_temp': np.nanmean(plume_temps),
                    'max_temp': np.nanmax(plume_temps),
                    'contour': plume_contour.tolist() if len(plume_contour) > 0 else [],
                    'baseline_temp': baseline_temp,
                    'temp_anomaly': baseline_temp - np.nanmean(plume_temps)
                })

        # Track baseline for analysis
        self.baseline_history.append({
            'frames_used': baseline_stats['frames_used'].copy() if 'frames_used' in baseline_stats else [],
            'baseline': baseline_temp,
            'std': baseline_stats.get('std', 0)
        })

        return {
            'sgd_plumes': sgd_mask,
            'shoreline': shoreline,
            'shoreline_thin': shoreline_thin,
            'ocean_stats': baseline_stats,
            'plume_info': plume_info,
            'baseline_method': 'moving_average',
            'window_size': self.window_size
        }

    def update_ocean_buffer(self, ocean_temps, frame_number):
        """
        Update the moving window buffer with ocean temperatures from current frame.

        Args:
            ocean_temps: Ocean temperatures from current frame
            frame_number: Current frame number for tracking
        """
        # Filter out invalid temperatures
        valid_temps = ocean_temps[~np.isnan(ocean_temps)]

        if len(valid_temps) > 0:
            # Add current frame's ocean temps to buffer
            self.ocean_temp_buffer.append(valid_temps)
            self.frame_metadata_buffer.append({
                'frame': frame_number,
                'median': np.median(valid_temps),
                'mean': np.mean(valid_temps),
                'std': np.std(valid_temps),
                'count': len(valid_temps)
            })

    def calculate_buffered_baseline(self):
        """
        Calculate ocean baseline using buffered temperatures from multiple frames.

        Returns:
            dict: Baseline statistics including temperature and metadata
        """
        if not self.ocean_temp_buffer:
            return None

        # Combine all ocean temperatures from buffer
        all_ocean_temps = np.concatenate(list(self.ocean_temp_buffer))

        # Remove NaN values
        valid_temps = all_ocean_temps[~np.isnan(all_ocean_temps)]

        if len(valid_temps) == 0:
            return None

        # Calculate baseline using selected method
        baseline = self.calculate_ocean_baseline(valid_temps)

        # Calculate additional statistics
        stats = {
            'baseline': baseline,
            'method': self.baseline_method,
            'window_size': len(self.ocean_temp_buffer),
            'total_pixels': len(valid_temps),
            'frames_used': [m['frame'] for m in self.frame_metadata_buffer],
            'median': np.median(valid_temps),
            'mean': np.mean(valid_temps),
            'std': np.std(valid_temps),
            'q25': np.percentile(valid_temps, 25),
            'q75': np.percentile(valid_temps, 75)
        }

        return stats

    def get_baseline_statistics(self):
        """
        Get statistics about baseline variation across processing.

        Returns:
            dict: Statistics about baseline stability
        """
        if not self.baseline_history:
            return {}

        baselines = [h['baseline'] for h in self.baseline_history]

        return {
            'num_windows': len(self.baseline_history),
            'baseline_mean': np.mean(baselines),
            'baseline_std': np.std(baselines),
            'baseline_min': np.min(baselines),
            'baseline_max': np.max(baselines),
            'baseline_range': np.max(baselines) - np.min(baselines),
            'history': self.baseline_history
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python sgd_detector_moving_avg.py <data_directory> [window_size]")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f"Testing moving average detector on: {data_dir}")
    print(f"Window size: {window_size} frames")
    print("=" * 60)

    # Initialize detector
    detector = MovingAverageSGDDetector(
        data_dir,
        window_size=window_size,
        baseline_method='median'
    )

    # Test on first 20 frames
    test_frames = []
    for f in sorted(data_dir.glob("MAX_*.JPG"))[:20]:
        frame_num = int(f.stem.split('_')[1])
        test_frames.append(frame_num)

    print(f"\nProcessing {len(test_frames)} frames...")

    for frame_num in test_frames:
        result = detector.process_frame(frame_num, visualize=False)

        if 'ocean_stats' in result:
            stats = result['ocean_stats']
            print(f"  Frame {frame_num}: {len(result.get('plume_info', []))} SGDs detected")
            if stats:
                print(f"    Baseline: {stats.get('baseline', 0):.2f}°C from {stats.get('window_size', 0)} frames")

    # Print overall statistics
    print("\n" + "=" * 60)
    print("BASELINE STABILITY ANALYSIS:")
    stats = detector.get_baseline_statistics()
    if stats:
        print(f"  Mean baseline: {stats['baseline_mean']:.2f}°C")
        print(f"  Std deviation: {stats['baseline_std']:.3f}°C")
        print(f"  Range: {stats['baseline_range']:.3f}°C")
        print(f"  Min: {stats['baseline_min']:.2f}°C")
        print(f"  Max: {stats['baseline_max']:.2f}°C")