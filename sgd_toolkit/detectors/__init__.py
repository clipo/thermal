"""
SGD Detection Algorithms

This module contains the detector class hierarchy for submarine groundwater discharge detection:

- IntegratedSGDDetector: Base detector with RGB-thermal alignment
- ImprovedSGDDetector: Enhanced baseline methods and sun glint filtering
- MovingAverageSGDDetector: Temporal smoothing across frame sequences
- EdgeAwareSGDDetector: Frame boundary handling for overlap scenarios
"""

from sgd_toolkit.detectors.base import IntegratedSGDDetector
from sgd_toolkit.detectors.improved import ImprovedSGDDetector
from sgd_toolkit.detectors.temporal import MovingAverageSGDDetector
from sgd_toolkit.detectors.edge_aware import EdgeAwareSGDDetector

__all__ = [
    'IntegratedSGDDetector',
    'ImprovedSGDDetector',
    'MovingAverageSGDDetector',
    'EdgeAwareSGDDetector',
]
