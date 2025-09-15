#!/usr/bin/env python3
"""
Edge-Aware SGD Detector
Handles SGD detection at frame boundaries to ensure continuity across overlapping frames
"""

import numpy as np
from sgd_detector_moving_avg import MovingAverageSGDDetector
from scipy import ndimage
from skimage import measure


class EdgeAwareSGDDetector(MovingAverageSGDDetector):
    """
    SGD detector that handles edge cases for continuous detection across frames.

    Key features:
    - Relaxes shore distance requirements near frame edges
    - Tracks SGD positions between frames
    - Handles partial SGDs at boundaries
    """

    def __init__(self, data_dir, model_name='segmentation_model.pkl',
                 temp_threshold=0.5, min_area=10, max_area=None,
                 window_size=5, baseline_method='median',
                 edge_buffer=50, edge_shore_distance=20, **kwargs):
        """
        Initialize edge-aware detector.

        Args:
            edge_buffer: Pixel distance from edge to apply relaxed criteria
            edge_shore_distance: Max shore distance for edge SGDs (vs 5 for center)
        """
        super().__init__(
            data_dir=data_dir,
            model_name=model_name,
            temp_threshold=temp_threshold,
            min_area=min_area,
            max_area=max_area,
            window_size=window_size,
            baseline_method=baseline_method,
            **kwargs
        )

        self.edge_buffer = edge_buffer
        self.edge_shore_distance = edge_shore_distance

        # Track SGD positions across frames for continuity
        self.previous_frame_sgds = []
        self.frame_overlap_history = []

    def detect_sgd(self, thermal, masks):
        """
        Override to handle edge cases in SGD detection.
        """
        # Extract ocean temperatures from current frame
        ocean_mask = masks['ocean']
        ocean_temps = thermal[ocean_mask]

        # Get frame number
        frame_number = len(self.ocean_temp_buffer)

        # Update buffer with current frame's ocean temps
        self.update_ocean_buffer(ocean_temps, frame_number)

        # Calculate baseline from buffer
        baseline_stats = self.calculate_buffered_baseline()

        if baseline_stats is None:
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

        # Create edge proximity mask
        edge_mask = self.create_edge_mask(thermal.shape)

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

            # Check if plume is near frame edge
            is_edge_plume = np.any(plume & edge_mask)

            # Relaxed criteria for edge plumes
            if is_edge_plume:
                # For edge plumes, use relaxed shore distance
                shore_distance_threshold = self.edge_shore_distance
                # Also reduce minimum area requirement for partial plumes
                min_area_threshold = max(1, self.min_area // 2)
            else:
                # Standard criteria for center plumes
                shore_distance_threshold = 5
                min_area_threshold = self.min_area

            # Apply criteria
            if min_shore_distance < shore_distance_threshold and plume.sum() >= min_area_threshold:
                # Apply max area filter if specified (but not for edge plumes)
                if not is_edge_plume and self.max_area is not None and plume.sum() > self.max_area:
                    continue

                sgd_mask = sgd_mask | plume

                props = measure.regionprops(plume.astype(int))[0]

                # Extract contour of the plume
                contours = measure.find_contours(plume.astype(float), 0.5)
                plume_contour = contours[0] if contours else []

                plume_temps = thermal[plume]

                # Determine which edges this plume touches
                edges_touched = self.get_touched_edges(plume, thermal.shape)

                plume_data = {
                    'id': i,
                    'area_pixels': plume.sum(),
                    'min_shore_distance': min_shore_distance,
                    'centroid': props.centroid,
                    'min_temp': np.nanmin(plume_temps),
                    'mean_temp': np.nanmean(plume_temps),
                    'max_temp': np.nanmax(plume_temps),
                    'contour': plume_contour.tolist() if len(plume_contour) > 0 else [],
                    'baseline_temp': baseline_temp,
                    'temp_anomaly': baseline_temp - np.nanmean(plume_temps),
                    'is_edge_plume': is_edge_plume,
                    'edges_touched': edges_touched
                }

                plume_info.append(plume_data)

        # Track SGDs for frame-to-frame continuity
        self.previous_frame_sgds = plume_info.copy()

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
            'baseline_method': 'moving_average_edge_aware',
            'window_size': self.window_size,
            'edge_buffer': self.edge_buffer
        }

    def create_edge_mask(self, shape):
        """
        Create a mask indicating pixels near frame edges.

        Args:
            shape: Shape of the thermal image (height, width)

        Returns:
            Boolean mask where True indicates edge proximity
        """
        height, width = shape
        edge_mask = np.zeros(shape, dtype=bool)

        # Mark pixels within edge_buffer distance of frame boundaries
        edge_mask[:self.edge_buffer, :] = True  # Top edge
        edge_mask[-self.edge_buffer:, :] = True  # Bottom edge
        edge_mask[:, :self.edge_buffer] = True  # Left edge
        edge_mask[:, -self.edge_buffer:] = True  # Right edge

        return edge_mask

    def get_touched_edges(self, plume_mask, shape):
        """
        Determine which frame edges a plume touches.

        Args:
            plume_mask: Boolean mask of the plume
            shape: Shape of the thermal image

        Returns:
            List of edge names: ['top', 'bottom', 'left', 'right']
        """
        height, width = shape
        edges = []

        # Check each edge (within buffer zone)
        if np.any(plume_mask[:self.edge_buffer, :]):
            edges.append('top')
        if np.any(plume_mask[-self.edge_buffer:, :]):
            edges.append('bottom')
        if np.any(plume_mask[:, :self.edge_buffer]):
            edges.append('left')
        if np.any(plume_mask[:, -self.edge_buffer:]):
            edges.append('right')

        return edges

    def analyze_frame_continuity(self):
        """
        Analyze SGD continuity across frames.

        Returns:
            Statistics about edge SGDs and potential continuity issues
        """
        if not self.baseline_history:
            return {}

        total_sgds = sum(len(h.get('plume_info', [])) for h in self.baseline_history)
        edge_sgds = 0

        # Count edge SGDs
        for h in self.baseline_history:
            for plume in h.get('plume_info', []):
                if plume.get('is_edge_plume', False):
                    edge_sgds += 1

        return {
            'total_sgds': total_sgds,
            'edge_sgds': edge_sgds,
            'edge_percentage': (edge_sgds / total_sgds * 100) if total_sgds > 0 else 0,
            'frames_analyzed': len(self.baseline_history)
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python sgd_detector_edge_aware.py <data_directory>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    print(f"Testing edge-aware detector on: {data_dir}")
    print("=" * 60)

    # Initialize detector with edge awareness
    detector = EdgeAwareSGDDetector(
        data_dir,
        window_size=5,
        baseline_method='median',
        edge_buffer=50,  # 50 pixels from edge
        edge_shore_distance=20  # Relaxed shore distance for edges
    )

    # Test on first 20 frames
    test_frames = []
    for f in sorted(data_dir.glob("MAX_*.JPG"))[:20]:
        frame_num = int(f.stem.split('_')[1])
        test_frames.append(frame_num)

    print(f"\nProcessing {len(test_frames)} frames...")

    for frame_num in test_frames:
        result = detector.process_frame(frame_num, visualize=False)

        if 'plume_info' in result:
            plumes = result['plume_info']
            edge_plumes = [p for p in plumes if p.get('is_edge_plume', False)]

            print(f"  Frame {frame_num}: {len(plumes)} SGDs ({len(edge_plumes)} at edges)")

            for plume in edge_plumes:
                edges = plume.get('edges_touched', [])
                if edges:
                    print(f"    - Edge plume touching: {', '.join(edges)}")

    # Print continuity analysis
    print("\n" + "=" * 60)
    print("FRAME CONTINUITY ANALYSIS:")
    continuity = detector.analyze_frame_continuity()
    if continuity:
        print(f"  Total SGDs detected: {continuity['total_sgds']}")
        print(f"  Edge SGDs: {continuity['edge_sgds']} ({continuity['edge_percentage']:.1f}%)")
        print(f"  Frames analyzed: {continuity['frames_analyzed']}")